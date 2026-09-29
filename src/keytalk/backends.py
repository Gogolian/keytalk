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
    "DummyFileBackend",
    "OllamaBackend",
    "OllamaError",
    "parse_ollama_line",
    "LMStudioBackend",
    "LMStudioError",
    "OpenRouterBackend",
    "OpenRouterError",
    "LlamaCppBackend",
    "LlamaCppError",
    "TokenStream",
    "messages_to_prompt",
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


class DummyFileBackend(LLMBackend):
    """Speed-testing backend that streams the contents of a file, ignoring the prompt.

    Reads *response_file* once at construction time so disk I/O never touches
    the hot path.  Streams the text in ``piece_size``-character chunks with
    no artificial delay, making it the fastest possible stand-in for a real
    model when benchmarking transport throughput.
    """

    def __init__(self, response_file: str = "dummy_response.txt", piece_size: int = 4) -> None:
        if piece_size <= 0:
            raise ValueError("piece_size must be positive")
        with open(response_file, "r", encoding="utf-8") as fh:
            self._response = fh.read()
        self._piece_size = piece_size
        logger.info("DummyFileBackend loaded %d chars from %r", len(self._response), response_file)

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

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class TokenStream:
    """Async text stream that captures server metadata when the stream ends.

    ``llama.cpp`` attaches a ``timings`` object (including MTP
    ``draft_n``/``draft_n_accepted`` acceptance stats) to the final streaming
    chunk.  A plain async generator cannot carry that out-of-band, so backends
    that have metadata return this wrapper instead: iterate it like any async
    iterator, then read :attr:`timings`.
    """

    def __init__(self, gen: AsyncIterator[str], meta: dict) -> None:
        self._gen = gen
        self._meta = meta

    def __aiter__(self) -> "TokenStream":
        return self

    async def __anext__(self) -> str:
        return await self._gen.__anext__()

    @property
    def timings(self) -> Optional[dict]:
        """Generation timings reported by the server, once available."""

        value = self._meta.get("timings")
        return value if isinstance(value, dict) else None


def messages_to_prompt(messages: list) -> str:
    """Render chat messages into a plain transcript for prompt-only backends.

    Backends with native chat support (e.g. :class:`LlamaCppBackend`) pass the
    structured messages straight through so the model's own chat template is
    used; this is only the fallback for backends that accept a single prompt.
    """

    systems: list = []
    turns: list = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user")).strip().lower()
        content = message.get("content", "")
        if content is None:
            content = ""
        content = str(content)
        if role == "system":
            if content:
                systems.append(content)
        elif role == "assistant":
            turns.append(f"Assistant: {content}")
        else:  # treat anything else (user/tool/...) as a user turn
            turns.append(f"User: {content}")
    parts: list = []
    if systems:
        parts.append("\n".join(systems))
    parts.extend(turns)
    body = "\n".join(parts)
    if body:
        return f"{body}\nAssistant:"
    return "Assistant:"


class LlamaCppError(Exception):
    """Raised when the llama.cpp server cannot be reached or errors out."""


class LlamaCppBackend(LLMBackend):
    """Stream completions from a ``llama-server`` (llama.cpp) instance.

    Wire formats are taken from llama.cpp's ``tools/server`` (server-task.cpp):

    * ``POST /completions`` (native): SSE ``data: {json}`` lines with top-level
      ``content`` and ``stop`` fields; the final chunk carries ``timings``.
    * ``POST /v1/chat/completions`` (OpenAI-style): SSE chunks with
      ``choices[0].delta.content``; the final chunk carries ``timings`` and the
      stream ends with ``data: [DONE]``.  The server renders the model's own
      chat template (``--jinja``), so structured messages are passed through
      untouched.
    * ``GET /v1/models``: ``{"models": [...], "object": "list", "data": [...]}``.

    Both streaming endpoints frame every JSON object as ``data: ...\\n\\n``.
    ``reasoning_content`` deltas (deepseek thinking format) are skipped; the
    content stream carries assistant output only.
    """

    def __init__(
        self,
        model: str = "",
        host: str = "http://127.0.0.1:8080",
        *,
        timeout: float = 300.0,
        n_predict: int = 0,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._n_predict = n_predict

    # -- public API -----------------------------------------------------------

    def generate(self, prompt: str) -> TokenStream:
        """Stream a plain-text completion from the native ``/completions``."""

        body: dict = {"prompt": prompt, "stream": True}
        if self._n_predict:
            body["n_predict"] = self._n_predict
        meta: dict = {}
        return TokenStream(self._run(body, "/completions", meta, chat=False), meta)

    def generate_messages(self, messages: list, **params) -> TokenStream:
        """Stream a chat completion from ``/v1/chat/completions``.

        ``messages`` are passed through as-is so llama.cpp's Jinja chat
        template (including ``preserve_thinking`` etc.) does the rendering.
        Extra keyword arguments (``temperature``, ``max_tokens``, ...) are
        merged into the request body.
        """

        body: dict = {"messages": list(messages), "stream": True}
        if self._model:
            body["model"] = self._model
        body.update(params)
        meta: dict = {}
        return TokenStream(self._run(body, "/v1/chat/completions", meta, chat=True), meta)

    async def list_models(self) -> list[str]:
        loop = asyncio.get_running_loop()

        def worker() -> object:
            url = f"{self._host}/v1/models"
            request = urllib.request.Request(url, method="GET")
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                return LlamaCppError(_http_error_text(exc))
            except urllib.error.URLError as exc:
                return LlamaCppError(f"cannot reach llama-server: {exc}")
            except Exception as exc:  # noqa: BLE001 - defensive
                return exc

        result = await loop.run_in_executor(None, worker)
        if isinstance(result, Exception):
            raise result
        # The endpoint reports both an OAI-style "data" and an Ollama-style
        # "models" array; prefer "data", fall back to "models".
        names = [
            str(entry.get("id"))
            for entry in result.get("data", [])
            if isinstance(entry, dict) and entry.get("id")
        ]
        if not names:
            names = [
                str(entry.get("name"))
                for entry in result.get("models", [])
                if isinstance(entry, dict) and entry.get("name")
            ]
        return names

    # -- internals ------------------------------------------------------------

    async def _run(self, body: dict, path: str, meta: dict, *, chat: bool) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: "asyncio.Queue[object]" = asyncio.Queue()
        done = object()
        url = f"{self._host}{path}"
        payload = json.dumps(body).encode("utf-8")

        def worker() -> None:
            request = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    for raw_line in response:
                        loop.call_soon_threadsafe(queue.put_nowait, raw_line)
            except urllib.error.HTTPError as exc:
                loop.call_soon_threadsafe(
                    queue.put_nowait, LlamaCppError(_http_error_text(exc))
                )
            except urllib.error.URLError as exc:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    LlamaCppError(f"cannot reach llama-server: {exc}"),
                )
            except Exception as exc:  # noqa: BLE001 - defensive
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, done)

        worker_future = loop.run_in_executor(None, worker)
        try:
            while True:
                item = await queue.get()
                if item is done:
                    break
                if isinstance(item, Exception):
                    raise item
                chunk = _parse_sse_json(bytes(item))
                if chunk is None:
                    continue
                if isinstance(chunk.get("error"), (dict, str)):
                    raise LlamaCppError(str(chunk["error"]))
                timings = chunk.get("timings")
                if isinstance(timings, dict):
                    meta["timings"] = timings
                text = _chunk_content(chunk, chat=chat)
                if text:
                    yield text
        finally:
            await worker_future


def _http_error_text(exc: "urllib.error.HTTPError") -> str:
    try:
        body = exc.read().decode("utf-8", "replace")
        obj = json.loads(body)
        if isinstance(obj, dict) and "error" in obj:
            return str(obj["error"])
        return body or str(exc)
    except Exception:  # noqa: BLE001 - fall back to the raw error
        return str(exc)


def _parse_sse_json(line: bytes) -> Optional[dict]:
    """Parse one SSE ``data: {json}`` line (llama.cpp frames everything as SSE)."""

    line = line.strip()
    if not line or line == b"data: [DONE]" or line.startswith(b":"):
        return None  # [DONE] terminator and SSE ping lines
    if line.startswith(b"data: "):
        line = line[6:]
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _chunk_content(chunk: dict, *, chat: bool) -> Optional[str]:
    """Extract streamed text from a native or OpenAI-style chunk."""

    if chat:
        choices = chunk.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return None
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
    else:
        content = chunk.get("content")
    return content if isinstance(content, str) and content else None
