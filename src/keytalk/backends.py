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
import logging
import os
import ssl
import urllib.error
import urllib.request
from typing import AsyncIterator, Optional


def _ssl_context() -> ssl.SSLContext:
    """Return an SSL context, loading system CA certs when Python's default bundle is absent."""
    ctx = ssl.create_default_context()
    if not ctx.cert_store_stats()["x509"]:
        for cafile in ("/etc/ssl/cert.pem", "/etc/ssl/certs/ca-certificates.crt"):
            if os.path.isfile(cafile):
                ctx.load_verify_locations(cafile)
                break
    return ctx

__all__ = [
    "LLMBackend",
    "EchoBackend",
    "StaticBackend",
    "OllamaBackend",
    "OllamaError",
    "parse_ollama_line",
    "LMStudioBackend",
    "LMStudioError",
    "OpenRouterBackend",
    "OpenRouterError",
]

logger = logging.getLogger("keytalk.backends")


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

    async def list_models(self) -> list[str]:
        """Return the names of the models this backend can serve.

        The default returns an empty list (the consumer then falls back to the
        statically configured model name).  Backends that talk to a real server
        override this to report the models that server actually has loaded.
        """

        return []


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
        num_ctx: Optional[int] = 32768,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._num_ctx = num_ctx

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        logger.info("Starting Ollama generation for model=%r, prompt=%r", self._model, prompt[:100])
        loop = asyncio.get_running_loop()
        queue: "asyncio.Queue[object]" = asyncio.Queue()

        def worker() -> None:
            url = f"{self._host}/api/generate"
            request_body: dict[str, object] = {
                "model": self._model,
                "prompt": prompt,
                "stream": True,
            }
            if self._num_ctx is not None:
                # Load the model with a larger context window so big agent
                # prompts don't overflow Ollama's default (which can be as
                # small as 4096 tokens and yields an n_keep >= n_ctx error).
                request_body["options"] = {"num_ctx": self._num_ctx}
            body = json.dumps(request_body).encode("utf-8")
            request = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}
            )
            try:
                logger.debug("Sending request to Ollama at %s", url)
                with urllib.request.urlopen(
                    request, timeout=self._timeout
                ) as response:
                    logger.info("Connected to Ollama, streaming response")
                    for raw_line in response:
                        loop.call_soon_threadsafe(queue.put_nowait, raw_line)
            except urllib.error.URLError as exc:
                logger.error("Failed to reach Ollama: %s", exc)
                loop.call_soon_threadsafe(
                    queue.put_nowait, OllamaError(f"cannot reach Ollama: {exc}")
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Unexpected error in Ollama worker: %s", exc)
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                logger.debug("Ollama worker finished")
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

    async def list_models(self) -> list[str]:
        """Return the model names reported by Ollama's ``/api/tags`` endpoint."""

        loop = asyncio.get_running_loop()

        def worker() -> object:
            url = f"{self._host}/api/tags"
            request = urllib.request.Request(url, method="GET")
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.URLError as exc:
                return OllamaError(f"cannot reach Ollama: {exc}")
            except Exception as exc:  # pragma: no cover - defensive
                return exc

        result = await loop.run_in_executor(None, worker)
        if isinstance(result, Exception):
            raise result
        models = result.get("models", []) if isinstance(result, dict) else []
        names: list[str] = []
        for entry in models:
            if isinstance(entry, dict):
                name = entry.get("name") or entry.get("model")
                if name:
                    names.append(str(name))
        return names


class LMStudioError(Exception):
    """Raised when the LM Studio HTTP API cannot be reached or errors out."""


class LMStudioBackend(LLMBackend):
    """Stream completions from a local LM Studio server using OpenAI-compatible API.

    LM Studio provides an OpenAI-compatible endpoint at /v1/chat/completions.
    The blocking HTTP request runs in a worker thread; decoded lines are pushed
    onto an :class:`asyncio.Queue` and yielded as they arrive so the host can
    forward tokens to the consumer without waiting for the full completion.
    """

    _DONE = object()

    def __init__(
        self,
        model: str = "gemma-4-31b-it",
        host: str = "http://localhost:1234",
        *,
        timeout: float = 300.0,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        logger.info("Starting LM Studio generation for model=%r, prompt=%r", self._model, prompt[:100])
        loop = asyncio.get_running_loop()
        queue: "asyncio.Queue[object]" = asyncio.Queue()

        def worker() -> None:
            url = f"{self._host}/v1/chat/completions"
            body = json.dumps({
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "temperature": 0.7,
            }).encode("utf-8")
            request = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}
            )
            try:
                logger.debug("Sending request to LM Studio at %s", url)
                with urllib.request.urlopen(
                    request, timeout=self._timeout
                ) as response:
                    logger.info("Connected to LM Studio, streaming response")
                    for raw_line in response:
                        loop.call_soon_threadsafe(queue.put_nowait, raw_line)
            except urllib.error.URLError as exc:
                logger.error("Failed to reach LM Studio: %s", exc)
                loop.call_soon_threadsafe(
                    queue.put_nowait, LMStudioError(f"cannot reach LM Studio: {exc}")
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Unexpected error in LM Studio worker: %s", exc)
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                logger.debug("LM Studio worker finished")
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
                fragment = self._parse_openai_sse_line(bytes(item))
                if fragment:
                    yield fragment
        finally:
            await worker_future

    def _parse_openai_sse_line(self, line: bytes) -> Optional[str]:
        """Extract text fragment from OpenAI-compatible SSE stream.

        LM Studio uses Server-Sent Events format:
        data: {"choices":[{"delta":{"content":"text"}}]}
        """
        line = line.strip()
        if not line or line == b"data: [DONE]":
            return None
        
        # SSE lines start with "data: "
        if line.startswith(b"data: "):
            line = line[6:]  # Remove "data: " prefix
        
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # Skip malformed lines (empty data, etc.)
            return None
        
        if "error" in obj:
            raise LMStudioError(str(obj["error"]))
        
        # Extract content from OpenAI-style streaming response
        choices = obj.get("choices", [])
        if choices and len(choices) > 0:
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if content:
                return content
        
        return None

    async def list_models(self) -> list[str]:
        """Return the model ids reported by LM Studio's ``/v1/models`` endpoint."""

        loop = asyncio.get_running_loop()

        def worker() -> object:
            url = f"{self._host}/v1/models"
            request = urllib.request.Request(url, method="GET")
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.URLError as exc:
                return LMStudioError(f"cannot reach LM Studio: {exc}")
            except Exception as exc:  # pragma: no cover - defensive
                return exc

        result = await loop.run_in_executor(None, worker)
        if isinstance(result, Exception):
            raise result
        data = result.get("data", []) if isinstance(result, dict) else []
        names: list[str] = []
        for entry in data:
            if isinstance(entry, dict):
                name = entry.get("id")
                if name:
                    names.append(str(name))
        return names


class OpenRouterError(Exception):
    """Raised when the OpenRouter HTTP API cannot be reached or errors out."""


class OpenRouterBackend(LLMBackend):
    """Stream completions from OpenRouter using its OpenAI-compatible API.

    OpenRouter is a hosted gateway to many models (OpenAI, Anthropic, Google,
    Meta, …).  Requires an API key passed via ``api_key`` or the
    ``OPENROUTER_API_KEY`` environment variable.  The default model can be
    overridden per-request via ``model``.
    """

    _DONE = object()
    _HOST = "https://openrouter.ai"

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        api_key: str = "",
        *,
        timeout: float = 300.0,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise OpenRouterError(
                "OpenRouter API key not set; pass --openrouter-key or set OPENROUTER_API_KEY"
            )
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        logger.info("Starting OpenRouter generation for model=%r, prompt=%r", self._model, prompt[:100])
        loop = asyncio.get_running_loop()
        queue: "asyncio.Queue[object]" = asyncio.Queue()

        def worker() -> None:
            url = f"{self._HOST}/api/v1/chat/completions"
            body = json.dumps({
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            }).encode("utf-8")
            try:
                headers = self._headers()
            except OpenRouterError as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
                loop.call_soon_threadsafe(queue.put_nowait, self._DONE)
                return
            request = urllib.request.Request(url, data=body, headers=headers)
            try:
                logger.debug("Sending request to OpenRouter at %s", url)
                with urllib.request.urlopen(request, timeout=self._timeout, context=_ssl_context()) as response:
                    logger.info("Connected to OpenRouter, streaming response")
                    for raw_line in response:
                        loop.call_soon_threadsafe(queue.put_nowait, raw_line)
            except urllib.error.URLError as exc:
                logger.error("Failed to reach OpenRouter: %s", exc)
                loop.call_soon_threadsafe(
                    queue.put_nowait, OpenRouterError(f"cannot reach OpenRouter: {exc}")
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Unexpected error in OpenRouter worker: %s", exc)
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                logger.debug("OpenRouter worker finished")
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
                fragment = self._parse_sse_line(bytes(item))
                if fragment:
                    yield fragment
        finally:
            await worker_future

    def _parse_sse_line(self, line: bytes) -> Optional[str]:
        """Extract text fragment from an OpenAI-compatible SSE line."""
        line = line.strip()
        if not line or line == b"data: [DONE]":
            return None
        if line.startswith(b"data: "):
            line = line[6:]
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None
        if "error" in obj:
            raise OpenRouterError(str(obj["error"]))
        choices = obj.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if content:
                return content
        return None

    async def list_models(self) -> list[str]:
        """Return model ids from OpenRouter's ``/api/v1/models`` endpoint."""
        if not self._api_key:
            return []
        loop = asyncio.get_running_loop()

        def worker() -> object:
            url = f"{self._HOST}/api/v1/models"
            request = urllib.request.Request(
                url,
                method="GET",
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            try:
                with urllib.request.urlopen(request, timeout=self._timeout, context=_ssl_context()) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.URLError as exc:
                return OpenRouterError(f"cannot reach OpenRouter: {exc}")
            except Exception as exc:  # pragma: no cover - defensive
                return exc

        result = await loop.run_in_executor(None, worker)
        if isinstance(result, Exception):
            raise result
        data = result.get("data", []) if isinstance(result, dict) else []
        names: list[str] = []
        for entry in data:
            if isinstance(entry, dict):
                name = entry.get("id")
                if name:
                    names.append(str(name))
        return sorted(names)
