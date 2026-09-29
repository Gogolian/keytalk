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
from typing import AsyncIterator, Callable, Dict, List, Optional

from .toolcalls import (
    render_tool_result,
    tool_calls_to_prompt_lines,
    tools_to_prompt_section,
)


def _ssl_context() -> ssl.SSLContext:
    """Return an SSL context, loading system CA certs when Python's default bundle is absent."""
    ctx = ssl.create_default_context()
    if not ctx.cert_store_stats()["x509"]:
        for cafile in ("/etc/ssl/cert.pem", "/etc/ssl/certs/ca-certificates.crt"):
            if os.path.isfile(cafile):
                ctx.load_verify_locations(cafile)
                break
    return ctx


# ---------------------------------------------------------------------------
# Shared HTTP plumbing
#
# Every backend below is the same shape: POST a JSON body to a local (or
# remote) inference server, read a line-at-a-time text stream, and pull the
# text fragment out of each line.  ``urllib`` is blocking, so the request
# runs in a worker thread and each line is handed back to the event loop
# through a queue.  Only the URL, the body and the per-line parser differ
# between backends, so those are the only things callers supply.
# ---------------------------------------------------------------------------


def _error_factory(cls: type) -> Callable[..., Exception]:
    """Build an error factory raising ``cls``, tolerating an HTTP status code.

    ``LlamaCppError`` takes a ``code``; the others do not, so the code is
    dropped rather than every backend having to special-case it.
    """

    def make(message: str, code: Optional[int] = None) -> Exception:
        if code is None:
            return cls(message)
        return cls(message, code=code)  # type: ignore[call-arg]

    return make


async def _stream_lines(
    url: str,
    body: Optional[bytes],
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: float,
    error: Callable[..., Exception],
    context: Optional[ssl.SSLContext] = None,
) -> AsyncIterator[bytes]:
    """Yield raw response lines from a streaming HTTP endpoint.

    Raises ``error(...)`` if the endpoint is unreachable or returns an HTTP
    error status.  The connection is always closed and the worker thread is
    always awaited, even when the consumer stops early.
    """

    loop = asyncio.get_running_loop()
    queue: "asyncio.Queue[object]" = asyncio.Queue()
    done = object()

    def worker() -> None:
        request = urllib.request.Request(
            url,
            data=body,
            headers=headers or {"Content-Type": "application/json"},
            method="POST" if body is not None else "GET",
        )
        try:
            kwargs = {"timeout": timeout}
            if context is not None:
                kwargs["context"] = context
            with urllib.request.urlopen(request, **kwargs) as response:
                for raw_line in response:
                    loop.call_soon_threadsafe(queue.put_nowait, raw_line)
        except urllib.error.HTTPError as exc:
            loop.call_soon_threadsafe(queue.put_nowait, error(_http_error_text(exc), exc.code))
        except urllib.error.URLError as exc:
            loop.call_soon_threadsafe(queue.put_nowait, error(f"cannot reach {url}: {exc}"))
        except Exception as exc:  # pragma: no cover - defensive
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, done)

    worker_future = loop.run_in_executor(None, worker)
    try:
        while True:
            item = await queue.get()
            if item is done:
                return
            if isinstance(item, Exception):
                raise item
            yield bytes(item)
    finally:
        await worker_future


async def _json_get(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: float,
    error: Callable[..., Exception],
    context: Optional[ssl.SSLContext] = None,
) -> object:
    """GET a JSON document, raising ``error(...)`` on any failure.

    The result is returned untyped so each backend can pick its own shape
    (``/v1/models``, Ollama's ``/api/tags``, ...).
    """

    loop = asyncio.get_running_loop()

    def worker() -> object:
        request = urllib.request.Request(url, method="GET", headers=headers or {})
        try:
            kwargs = {"timeout": timeout}
            if context is not None:
                kwargs["context"] = context
            with urllib.request.urlopen(request, **kwargs) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return error(_http_error_text(exc), exc.code)
        except urllib.error.URLError as exc:
            return error(f"cannot reach {url}: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            return exc

    result = await loop.run_in_executor(None, worker)
    if isinstance(result, Exception):
        raise result
    return result


def _names_from(payload: object, *keys: str) -> List[str]:
    """Pull model names out of a ``{"models": ...}``/``{"data": ...}`` listing.

    Servers disagree on the field name (``id``, ``name`` or ``model``) and on
    the envelope key, so both are tried before giving up.
    """

    if not isinstance(payload, dict):
        return []
    for key in keys:
        entries = payload.get(key)
        if not isinstance(entries, list):
            continue
        names: List[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            value = entry.get("id") or entry.get("name") or entry.get("model")
            if value:
                names.append(str(value))
        if names:
            return names
    return []


def _parse_openai_sse_line(line: bytes, error: type) -> Optional[str]:
    """Extract the text fragment from an OpenAI-compatible SSE line.

    Shared by the LM Studio and OpenRouter backends: both frame every chunk as
    ``data: {json}`` with the assistant text in ``choices[0].delta.content``.
    Comment/keep-alive lines and the ``[DONE]`` terminator yield ``None``; an
    ``error`` member raises the backend's own error type.
    """

    line = line.strip()
    if not line or line == b"data: [DONE]" or line.startswith(b":"):
        return None
    if line.startswith(b"data: "):
        line = line[6:]
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if "error" in obj:
        raise error(str(obj["error"]))
    choices = obj.get("choices") or []
    if choices and isinstance(choices[0], dict):
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if content:
            return content
    return None


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

    Ollama's ``/api/generate`` emits one JSON object per line (not SSE), so the
    line parser is :func:`parse_ollama_line`; the streaming plumbing is shared
    with the other HTTP backends.
    """

    _ERROR = staticmethod(_error_factory(OllamaError))

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
        async for line in _stream_lines(
            f"{self._host}/api/generate",
            json.dumps(request_body).encode("utf-8"),
            timeout=self._timeout,
            error=self._ERROR,
        ):
            fragment = parse_ollama_line(line)
            if fragment:
                yield fragment

    async def list_models(self) -> list[str]:
        """Return the model names reported by Ollama's ``/api/tags`` endpoint."""

        payload = await _json_get(
            f"{self._host}/api/tags", timeout=self._timeout, error=self._ERROR
        )
        return _names_from(payload, "models")


class LMStudioError(Exception):
    """Raised when the LM Studio HTTP API cannot be reached or errors out."""


class LMStudioBackend(LLMBackend):
    """Stream completions from a local LM Studio server using its OpenAI-compatible API.

    LM Studio provides an OpenAI-compatible endpoint at ``/v1/chat/completions``
    that frames every chunk as SSE (``data: {json}``), with the text in
    ``choices[0].delta.content``.
    """

    _ERROR = staticmethod(_error_factory(LMStudioError))

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
        body = json.dumps({
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": 0.7,
        }).encode("utf-8")
        async for line in _stream_lines(
            f"{self._host}/v1/chat/completions",
            body,
            timeout=self._timeout,
            error=self._ERROR,
        ):
            fragment = _parse_openai_sse_line(line, LMStudioError)
            if fragment:
                yield fragment

    async def list_models(self) -> list[str]:
        """Return the model ids reported by LM Studio's ``/v1/models`` endpoint."""

        payload = await _json_get(
            f"{self._host}/v1/models", timeout=self._timeout, error=self._ERROR
        )
        return _names_from(payload, "data")


class OpenRouterError(Exception):
    """Raised when the OpenRouter HTTP API cannot be reached or errors out."""


class OpenRouterBackend(LLMBackend):
    """Stream completions from OpenRouter using its OpenAI-compatible API.

    OpenRouter is a hosted gateway to many models (OpenAI, Anthropic, Google,
    Meta, ...).  Requires an API key passed via ``api_key`` or the
    ``OPENROUTER_API_KEY`` environment variable.  The default model can be
    overridden per-request via ``model``.
    """

    _ERROR = staticmethod(_error_factory(OpenRouterError))
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

    def _headers(self) -> Dict[str, str]:
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
        # Fail fast, before spending a connection, when no key is configured.
        headers = self._headers()
        body = json.dumps({
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }).encode("utf-8")
        async for line in _stream_lines(
            f"{self._HOST}/api/v1/chat/completions",
            body,
            headers=headers,
            timeout=self._timeout,
            error=self._ERROR,
            context=_ssl_context(),
        ):
            fragment = _parse_openai_sse_line(line, OpenRouterError)
            if fragment:
                yield fragment

    async def list_models(self) -> list[str]:
        """Return model ids from OpenRouter's ``/api/v1/models`` endpoint."""
        if not self._api_key:
            return []
        payload = await _json_get(
            f"{self._HOST}/api/v1/models",
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self._timeout,
            error=self._ERROR,
            context=_ssl_context(),
        )
        return sorted(_names_from(payload, "data"))


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

    @property
    def tool_calls(self) -> Optional[list]:
        """Structured tool calls reported by the server, once available."""

        value = self._meta.get("tool_calls")
        return value if isinstance(value, list) and value else None

    @property
    def finish_reason(self) -> Optional[str]:
        """Why generation stopped ("stop", "tool_calls", ...), once available."""

        value = self._meta.get("finish_reason")
        return value if isinstance(value, str) else None


def messages_to_prompt(messages: list, tools: Optional[list] = None) -> str:
    """Render chat messages into a plain transcript for prompt-only backends.

    Backends with native chat support (e.g. :class:`LlamaCppBackend`) pass the
    structured messages straight through so the model's own chat template is
    used; this is only the fallback for backends that accept a single prompt.

    Tool-calling turns survive the flattening: assistant ``tool_calls`` and
    ``role: "tool"`` results are rendered as transcript lines, and ``tools``
    (when given) adds an instruction block teaching the plaintext call syntax
    :func:`keytalk.toolcalls.parse_text_tool_calls` can read back.
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
            calls = tool_calls_to_prompt_lines(message.get("tool_calls") or [])
            if content or not calls:
                turns.append(f"Assistant: {content}")
            for line in calls:
                turns.append(f"Assistant: {line}")
        elif role == "tool":
            turns.append(render_tool_result(message))
        else:  # treat anything else (user/...) as a user turn
            turns.append(f"User: {content}")
    parts: list = []
    if systems:
        parts.append("\n".join(systems))
    if tools:
        parts.append(tools_to_prompt_section(tools))
    parts.extend(turns)
    body = "\n".join(parts)
    if body:
        return f"{body}\nAssistant:"
    return "Assistant:"


class LlamaCppError(Exception):
    """Raised when the llama.cpp server cannot be reached or errors out.

    ``code`` carries the HTTP status when the failure came from an HTTP
    error response (``None`` for connection problems and in-stream errors).
    """

    def __init__(self, message: str, *, code: Optional[int] = None) -> None:
        super().__init__(message)
        self.code = code


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

    _ERROR = staticmethod(_error_factory(LlamaCppError))

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
        return TokenStream(self._generate_with_fallback(body, prompt, meta), meta)

    async def _generate_with_fallback(
        self, body: dict, prompt: str, meta: dict
    ) -> AsyncIterator[str]:
        """Stream a completion, falling back to the chat endpoint on 404.

        OpenAI-only servers that speak ``llama-server``'s API subset but not
        its native route (e.g. ``mlx_vlm.server``) answer ``404`` on
        ``/completions``; wrap the prompt as a single user message and retry
        via ``/v1/chat/completions`` instead of failing the request.
        """

        try:
            async for text in self._run(body, "/completions", meta, chat=False):
                yield text
            return
        except LlamaCppError as exc:
            if exc.code != 404:
                raise
        chat_body: dict = {
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        if self._model:
            chat_body["model"] = self._model
        if self._n_predict:
            chat_body["max_tokens"] = self._n_predict
        async for text in self._run(chat_body, "/v1/chat/completions", meta, chat=True):
            yield text

    def generate_messages(self, messages: list, **params) -> TokenStream:
        """Stream a chat completion from ``/v1/chat/completions``.

        ``messages`` are passed through as-is so llama.cpp's Jinja chat
        template (including ``preserve_thinking`` etc.) does the rendering.
        Extra keyword arguments (``temperature``, ``max_tokens``, ``tools``,
        ``tool_choice``, ...) are merged into the request body untouched, so
        tool-calling reaches the model natively.  Any tool calls the model
        makes are accumulated on the stream's ``tool_calls`` metadata.
        """

        body: dict = {"messages": list(messages), "stream": True}
        if self._model:
            body["model"] = self._model
        body.update(params)
        meta: dict = {}
        return TokenStream(self._run(body, "/v1/chat/completions", meta, chat=True), meta)

    async def list_models(self) -> list[str]:
        # The endpoint reports both an OAI-style "data" and an Ollama-style
        # "models" array; prefer "data", fall back to "models".
        payload = await _json_get(
            f"{self._host}/v1/models", timeout=self._timeout, error=self._ERROR
        )
        return _names_from(payload, "data", "models")

    # -- internals ------------------------------------------------------------

    async def _run(self, body: dict, path: str, meta: dict, *, chat: bool) -> AsyncIterator[str]:
        async for line in _stream_lines(
            f"{self._host}{path}",
            json.dumps(body).encode("utf-8"),
            timeout=self._timeout,
            error=self._ERROR,
        ):
            chunk = _parse_sse_json(line)
            if chunk is None:
                continue
            if isinstance(chunk.get("error"), (dict, str)):
                raise LlamaCppError(str(chunk["error"]))
            timings = chunk.get("timings")
            if isinstance(timings, dict):
                meta["timings"] = timings
            _collect_tool_call(chunk, meta)
            text = _chunk_content(chunk, chat=chat)
            if text:
                yield text


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


def _collect_tool_call(chunk: dict, meta: dict) -> None:
    """Accumulate streamed tool-call deltas into ``meta["tool_calls"]``.

    OpenAI-style chunks carry ``choices[0].delta.tool_calls`` as fragments
    (``{"index": 0, "id": ..., "function": {"name": ..., "arguments": ...}}``)
    where later fragments append to the same call's ``arguments`` string;
    ``finish_reason: "tool_calls"`` marks a tool-calling turn.  Fragments are
    merged by ``index`` so the metadata ends up with one complete call per
    model invocation, ready to be forwarded as a message trailer.
    """

    choices = chunk.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return
    choice = choices[0]
    finish = choice.get("finish_reason")
    if isinstance(finish, str) and finish:
        meta["finish_reason"] = finish
    delta = choice.get("delta") or {}
    fragments = delta.get("tool_calls")
    if isinstance(fragments, dict):  # some servers send a single object
        fragments = [fragments]
    if not isinstance(fragments, list):
        return
    calls: list = meta.setdefault("tool_calls", [])
    for fragment in fragments:
        if not isinstance(fragment, dict):
            continue
        index = fragment.get("index")
        if not isinstance(index, int):
            index = len(calls) - 1 if calls else 0
        while len(calls) <= index:
            calls.append({})
        call = calls[index]
        if fragment.get("id"):
            call["id"] = str(fragment["id"])
        if fragment.get("type"):
            call["type"] = str(fragment["type"])
        function = fragment.get("function")
        if isinstance(function, dict):
            target = call.setdefault("function", {})
            if function.get("name"):
                target["name"] = str(function["name"])
            arguments = function.get("arguments")
            if isinstance(arguments, str) and arguments:
                target["arguments"] = target.get("arguments", "") + arguments
