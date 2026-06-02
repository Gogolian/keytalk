"""LLM backends used by the host to answer prompts.

A backend turns a prompt string into an async stream of response text pieces
(tokens).  The :class:`OllamaBackend` talks to a local Ollama server; the other
backends are deterministic and dependency-free so the test-suite can drive the
whole pipeline without a model.
"""

from __future__ import annotations

import abc
import asyncio
import json
import urllib.error
import urllib.request
from typing import AsyncIterator, Optional

__all__ = [
    "LLMBackend",
    "EchoBackend",
    "StaticBackend",
    "OllamaBackend",
    "OllamaError",
    "parse_ollama_line",
]


class LLMBackend(abc.ABC):
    """Produces a streamed text response for a prompt."""

    @abc.abstractmethod
    def generate(self, prompt: str) -> AsyncIterator[str]:
        """Yield response text pieces for ``prompt``.

        Implementations are async generators.  Raising inside the generator is
        how a backend reports failure; the host turns that into an ERROR
        message for the consumer.
        """
        raise NotImplementedError


class EchoBackend(LLMBackend):
    """Test backend that streams the prompt back one word at a time."""

    def __init__(self, prefix: str = "echo: ") -> None:
        self._prefix = prefix

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        yield self._prefix
        for word in prompt.split():
            # A tiny sleep lets concurrent requests interleave in tests.
            await asyncio.sleep(0)
            yield word + " "


class StaticBackend(LLMBackend):
    """Test backend that streams a fixed response in fixed-size pieces."""

    def __init__(self, response: str, piece_size: int = 4) -> None:
        if piece_size <= 0:
            raise ValueError("piece_size must be positive")
        self._response = response
        self._piece_size = piece_size

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        text = self._response
        for i in range(0, len(text), self._piece_size):
            await asyncio.sleep(0)
            yield text[i : i + self._piece_size]


class OllamaError(Exception):
    """Raised when the Ollama HTTP API cannot be reached or errors out."""


def parse_ollama_line(line: bytes) -> Optional[str]:
    """Extract the text fragment from one line of an Ollama stream.

    Ollama's ``/api/generate`` streaming endpoint emits one JSON object per
    line.  Each object has a ``response`` field with the next text fragment and
    a ``done`` boolean.  Blank lines are ignored (return ``None``).  An object
    carrying an ``error`` field raises :class:`OllamaError`.
    """

    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise OllamaError(f"invalid JSON from Ollama: {line!r}") from exc
    if "error" in obj:
        raise OllamaError(str(obj["error"]))
    fragment = obj.get("response")
    if fragment:
        return fragment
    return None


class OllamaBackend(LLMBackend):
    """Stream completions from a local Ollama server.

    The blocking HTTP request runs in a worker thread; decoded lines are pushed
    onto an :class:`asyncio.Queue` and yielded as they arrive so the host can
    forward tokens to the consumer without waiting for the full completion.
    """

    _DONE = object()

    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        *,
        timeout: float = 300.0,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: "asyncio.Queue[object]" = asyncio.Queue()

        def worker() -> None:
            url = f"{self._host}/api/generate"
            body = json.dumps(
                {"model": self._model, "prompt": prompt, "stream": True}
            ).encode("utf-8")
            request = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self._timeout
                ) as response:
                    for raw_line in response:
                        loop.call_soon_threadsafe(queue.put_nowait, raw_line)
            except urllib.error.URLError as exc:
                loop.call_soon_threadsafe(
                    queue.put_nowait, OllamaError(f"cannot reach Ollama: {exc}")
                )
            except Exception as exc:  # pragma: no cover - defensive
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, self._DONE)

        worker_future = loop.run_in_executor(None, worker)
        try:
            while True:
                item = await queue.get()
                if item is self._DONE:
                    break
                if isinstance(item, Exception):
                    raise item
                assert isinstance(item, (bytes, bytearray))
                fragment = parse_ollama_line(bytes(item))
                if fragment:
                    yield fragment
        finally:
            await worker_future
