"""An Ollama-compatible HTTP endpoint that bridges to a remote BLE host.

Many tools (editors, IDE extensions, chat front-ends) already know how to talk
to a local `Ollama <https://ollama.com>`_ server over HTTP.  This module lets the
**consumer** machine expose exactly that HTTP surface while the model actually
runs on the *host* machine reached over Bluetooth LE - so a tool such as VS Code
only needs to point at this server's port instead of a real Ollama install.

The server is intentionally dependency-free: it speaks just enough of HTTP/1.1
(using :mod:`asyncio` stream servers) and of Ollama's JSON API to drive the
common ``/api/generate`` and ``/api/chat`` flows, plus the discovery endpoints
(``/``, ``/api/version``, ``/api/tags``, ``/api/show``) that clients probe first.

It is transport agnostic: it talks to anything implementing :class:`PromptStreamer`
(notably :class:`keytalk.consumer.ConsumerClient`), so it can be exercised
end-to-end over the in-memory loopback transport with no Bluetooth hardware.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import uuid
from typing import (
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

from .backends import messages_to_prompt
from .toolcalls import ToolCallExtractor, normalize_tool_calls, tool_calls_to_ollama

try:  # pragma: no cover - typing-only convenience
    from typing import Protocol
except ImportError:  # pragma: no cover - Python < 3.8 has no Protocol
    Protocol = object  # type: ignore[assignment]

__all__ = [
    "PromptStreamer",
    "OllamaBridgeServer",
    "build_prompt_from_messages",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_MODEL",
    "OLLAMA_VERSION",
]

logger = logging.getLogger("keytalk.server")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434  # the port a real Ollama server listens on
DEFAULT_MODEL = "keytalk"
#: Version string reported through ``/api/version``; some clients check it.
OLLAMA_VERSION = "0.6.4"
#: Context window advertised through ``/api/show`` (clients use this to size
#: prompts).  Copilot's prompt renderer needs a large budget or it fails to fit
#: the agent prompt ("No lowest priority node found"), so default high.
DEFAULT_CONTEXT_LENGTH = 128000

# Cap a single request body so a misbehaving client cannot exhaust memory.
MAX_BODY_BYTES = 16 * 1024 * 1024
# Cap request/response header lines for the same reason.
MAX_LINE_BYTES = 64 * 1024


class PromptStreamer(Protocol):
    """Minimal interface the bridge needs from a consumer client.

    :class:`keytalk.consumer.ConsumerClient` satisfies this: it turns a prompt
    string into an async stream of response-text pieces.
    """

    def stream(self, prompt: str) -> AsyncIterator[str]:
        ...  # pragma: no cover - structural typing only

    def list_models(self) -> Awaitable[List[str]]:
        ...  # pragma: no cover - optional, structural typing only


def build_prompt_from_messages(
    messages: List[Dict[str, object]], tools: Optional[List[dict]] = None
) -> str:

    """Flatten Ollama ``/api/chat`` messages into a single prompt string.

    The remote host bridges to prompt-style completion endpoints, so chat-style
    message lists are rendered into a simple, readable transcript ending with
    an ``Assistant:`` cue (see :func:`keytalk.backends.messages_to_prompt`).
    ``tools`` - the request's tool definitions - add an instruction block
    teaching the plaintext call syntax :func:`keytalk.toolcalls.
    parse_text_tool_calls` reads back into structured ``tool_calls``, so
    tool-calling survives even prompt-only backends.
    """

    return messages_to_prompt(messages, tools)


def _now_iso() -> str:
    """Return an RFC3339/ISO-8601 UTC timestamp, as Ollama uses."""

    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _now_unix() -> int:
    """Return a Unix timestamp in seconds, as the OpenAI API uses."""

    return int(datetime.datetime.now(datetime.timezone.utc).timestamp())


async def _close_quietly(stream: object) -> None:
    """Close a response stream early.

    Closing the generator tells the consumer to send a CANCEL to the host, so
    abandoned requests stop consuming the remote LLM and the BLE link.
    """

    aclose = getattr(stream, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception:  # noqa: BLE001 - best effort on an already-failing path
        logger.debug("failed to close response stream", exc_info=True)


async def _drain(
    pieces: object,
    on_piece: Optional[Callable[[str], Awaitable[None]]] = None,
) -> Tuple[List[str], Optional[BaseException]]:
    """Run a response stream to exhaustion, collecting its text pieces.

    ``on_piece`` is awaited for each non-empty piece as it arrives, so a
    streaming handler can forward it without buffering the whole reply.
    Returns ``(pieces, failure)`` where ``failure`` is the exception the
    stream raised, or ``None`` when it completed normally - each endpoint
    reports a failure in its own wire format, so the caller decides.

    The stream is always closed, which is what makes an abandoned request tell
    the host to stop generating.
    """

    parts: List[str] = []
    failure: Optional[BaseException] = None
    try:
        async for piece in pieces:  # type: ignore[union-attr]
            if not piece:
                continue
            parts.append(piece)
            if on_piece is not None:
                await on_piece(piece)
    except asyncio.CancelledError:  # pragma: no cover - teardown path
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller
        failure = exc
    finally:
        await _close_quietly(pieces)
    return parts, failure


class _Request:
    """A parsed HTTP request."""

    __slots__ = ("method", "target", "path", "headers", "body", "keep_alive")

    def __init__(
        self,
        method: str,
        target: str,
        headers: Dict[str, str],
        body: bytes,
        keep_alive: bool,
    ) -> None:
        self.method = method
        self.target = target
        self.path = target.split("?", 1)[0]
        self.headers = headers
        self.body = body
        self.keep_alive = keep_alive

    def json(self) -> Dict[str, object]:
        """Parse the body as a JSON object (``{}`` when empty)."""

        if not self.body:
            return {}
        obj = json.loads(self.body.decode("utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("request body must be a JSON object")
        return obj


class _BadRequest(Exception):
    """Raised while parsing a malformed request line/headers."""


# Type of the per-connection writer helper used by route handlers.
Responder = Callable[..., Awaitable[None]]


class OllamaBridgeServer:
    """Serve the Ollama HTTP API, forwarding prompts to a :class:`PromptStreamer`.

    Construct it with a started consumer client, then :meth:`start` it (or use it
    as an async context manager).  Each HTTP request that needs a completion is
    forwarded to ``client.stream(prompt)`` and the resulting text pieces are
    streamed back in Ollama's newline-delimited JSON format.
    """

    def __init__(
        self,
        client: PromptStreamer,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self._client = client
        self._host = host
        self._port = port
        self._model = model
        self._server: Optional[asyncio.AbstractServer] = None
        self._connections: "set[asyncio.Task[None]]" = set()

    # -- lifecycle ------------------------------------------------------------

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        """The bound port (resolved if ``0`` was requested)."""

        return self._port

    @property
    def model(self) -> str:
        return self._model

    async def start(self) -> None:
        """Bind the listening socket and begin accepting connections."""

        self._server = await asyncio.start_server(
            self._handle_connection, self._host, self._port
        )
        # Resolve the actual port when the caller asked for an ephemeral one.
        sockets = self._server.sockets or ()
        if sockets:
            self._port = sockets[0].getsockname()[1]
        logger.info(
            "keytalk serving Ollama-compatible API on http://%s:%d (model %r)",
            self._host,
            self._port,
            self._model,
        )

    async def serve_forever(self) -> None:
        """Run until cancelled (convenient for the CLI)."""

        if self._server is None:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def close(self) -> None:
        """Stop accepting connections and cancel in-flight ones."""

        server = self._server
        self._server = None
        if server is not None:
            server.close()
            try:
                await server.wait_closed()
            except Exception:  # pragma: no cover - defensive on teardown
                logger.debug("error while waiting for server close", exc_info=True)
        for task in list(self._connections):
            task.cancel()
        if self._connections:
            await asyncio.gather(*self._connections, return_exceptions=True)
        self._connections.clear()

    async def __aenter__(self) -> "OllamaBridgeServer":
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # -- connection handling --------------------------------------------------

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._connections.add(task)
        try:
            await self._serve_connection(reader, writer)
        except asyncio.CancelledError:  # pragma: no cover - teardown path
            raise
        except (ConnectionResetError, BrokenPipeError):  # pragma: no cover
            logger.debug("client disconnected", exc_info=True)
        except Exception:  # noqa: BLE001 - never let one connection kill server
            logger.exception("unhandled error serving connection")
        finally:
            if task is not None:
                self._connections.discard(task)
            try:
                writer.close()
            except Exception:  # pragma: no cover - best effort
                pass

    async def _serve_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        while True:
            try:
                request = await self._read_request(reader)
            except _BadRequest as exc:
                await self._write_json(
                    writer, 400, {"error": str(exc)}, keep_alive=False
                )
                return
            if request is None:
                return  # clean EOF between requests
            try:
                await self._dispatch(request, writer)
            except (ConnectionResetError, BrokenPipeError):  # pragma: no cover
                return
            if not request.keep_alive:
                return

    async def _read_request(
        self, reader: asyncio.StreamReader
    ) -> Optional[_Request]:
        try:
            request_line = await reader.readline()
        except (ConnectionResetError, asyncio.IncompleteReadError):  # pragma: no cover
            return None
        if not request_line:
            return None  # connection closed cleanly
        if len(request_line) > MAX_LINE_BYTES:
            raise _BadRequest("request line too long")
        try:
            method, target, version = (
                request_line.decode("latin-1").rstrip("\r\n").split(" ")
            )
        except ValueError:
            raise _BadRequest("malformed request line")

        headers: Dict[str, str] = {}
        while True:
            line = await reader.readline()
            if not line:
                raise _BadRequest("unexpected EOF in headers")
            if line in (b"\r\n", b"\n"):
                break
            if len(line) > MAX_LINE_BYTES:
                raise _BadRequest("header line too long")
            text = line.decode("latin-1").rstrip("\r\n")
            if ":" not in text:
                raise _BadRequest("malformed header line")
            name, value = text.split(":", 1)
            headers[name.strip().lower()] = value.strip()

        body = b""
        length_header = headers.get("content-length")
        if length_header is not None:
            try:
                length = int(length_header)
            except ValueError:
                raise _BadRequest("invalid Content-Length")
            if length < 0 or length > MAX_BODY_BYTES:
                raise _BadRequest("invalid Content-Length")
            try:
                body = await reader.readexactly(length)
            except asyncio.IncompleteReadError:
                raise _BadRequest("unexpected EOF in body")

        keep_alive = self._wants_keep_alive(version, headers)
        return _Request(method, target, headers, body, keep_alive)

    @staticmethod
    def _wants_keep_alive(version: str, headers: Dict[str, str]) -> bool:
        connection = headers.get("connection", "").lower()
        if "close" in connection:
            return False
        if "keep-alive" in connection:
            return True
        # HTTP/1.1 defaults to keep-alive; HTTP/1.0 defaults to close.
        return version.strip().upper() == "HTTP/1.1"

    # -- routing --------------------------------------------------------------

    async def _dispatch(
        self, request: _Request, writer: asyncio.StreamWriter
    ) -> None:
        path = request.path.rstrip("/") or "/"
        method = request.method.upper()

        if path == "/" and method in ("GET", "HEAD"):
            await self._write_text(request, writer, 200, "Ollama is running")
            return
        if path == "/api/version" and method == "GET":
            await self._write_json(
                writer, 200, {"version": OLLAMA_VERSION},
                keep_alive=request.keep_alive,
            )
            return
        if path == "/api/tags" and method == "GET":
            await self._write_json(
                writer, 200, await self._tags_payload(),
                keep_alive=request.keep_alive,
            )
            return
        if path in ("/api/ps",) and method == "GET":
            await self._write_json(
                writer, 200, {"models": []}, keep_alive=request.keep_alive
            )
            return
        if path == "/api/show" and method == "POST":
            try:
                show_body = request.json()
            except (ValueError, json.JSONDecodeError):
                show_body = {}
            show_model = str(show_body.get("model") or show_body.get("name") or "") or None
            await self._write_json(
                writer, 200, self._show_payload(show_model),
                keep_alive=request.keep_alive,
            )
            return
        if path == "/api/generate" and method == "POST":
            await self._handle_generate(request, writer)
            return
        if path == "/api/chat" and method == "POST":
            await self._handle_chat(request, writer)
            return
        if path == "/v1/models" and method == "GET":
            await self._write_json(
                writer, 200, await self._openai_models_payload(),
                keep_alive=request.keep_alive,
            )
            return
        if path == "/v1/chat/completions" and method == "POST":
            await self._handle_openai_chat(request, writer)
            return

        await self._write_json(
            writer, 404, {"error": f"unknown endpoint {request.path}"},
            keep_alive=request.keep_alive,
        )

    async def _tags_payload(self) -> Dict[str, object]:
        names = await self._discover_models()
        if not names:
            return {"models": [self._model_entry()]}
        return {"models": [self._model_entry(name) for name in names]}

    async def _discover_models(self) -> List[str]:
        """Ask the remote host for its model list, if the client supports it.

        Falls back to an empty list (so the caller uses the configured model)
        when the client has no ``list_models`` capability or the host cannot be
        reached.
        """

        list_models = getattr(self._client, "list_models", None)
        if list_models is None:
            return []
        try:
            names = await list_models()
        except Exception:  # noqa: BLE001 - discovery is best-effort
            logger.warning("could not fetch model list from host", exc_info=True)
            return []
        return [str(name) for name in names if name]

    def _model_entry(self, model: Optional[str] = None) -> Dict[str, object]:
        raw = model if model is not None else self._model
        name = raw if ":" in raw else f"{raw}:latest"
        return {
            "name": name,
            "model": name,
            "modified_at": _now_iso(),
            "size": 0,
            "digest": "",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "keytalk",
                "families": ["keytalk"],
                "parameter_size": "",
                "quantization_level": "",
            },
        }

    def _show_payload(self, model: Optional[str] = None) -> Dict[str, object]:
        details = self._model_entry(model)["details"]
        return {
            "license": "",
            "modelfile": "",
            "parameters": "",
            "template": "",
            "details": details,
            # VS Code (and other clients) read model_info for the context
            # window; advertise a generous default so prompts are not clipped.
            "model_info": {
                "general.architecture": "keytalk",
                "keytalk.context_length": DEFAULT_CONTEXT_LENGTH,
            },
            # Clients filter models by capability: tool-calling clients such as
            # GitHub Copilot only register a model if it reports "tools".  The
            # remote host bridges to a real model, so advertise the common set.
            "capabilities": ["completion", "tools"],
        }

    # -- completion endpoints -------------------------------------------------

    async def _handle_generate(
        self, request: _Request, writer: asyncio.StreamWriter
    ) -> None:
        try:
            payload = request.json()
        except (ValueError, json.JSONDecodeError) as exc:
            await self._write_json(
                writer, 400, {"error": f"invalid request body: {exc}"},
                keep_alive=request.keep_alive,
            )
            return

        prompt = payload.get("prompt", "")
        prompt = "" if prompt is None else str(prompt)
        model = str(payload.get("model") or self._model)
        stream = payload.get("stream", True)

        def envelope(piece: str, done: bool) -> Dict[str, object]:
            obj: Dict[str, object] = {
                "model": model,
                "created_at": _now_iso(),
                "response": piece,
                "done": done,
            }
            if done:
                obj["done_reason"] = "stop"
            return obj

        await self._stream_completion(
            request, writer, self._request_stream(prompt=prompt), model,
            bool(stream), envelope,
        )

    async def _handle_chat(
        self, request: _Request, writer: asyncio.StreamWriter
    ) -> None:
        try:
            payload = request.json()
        except (ValueError, json.JSONDecodeError) as exc:
            await self._write_json(
                writer, 400, {"error": f"invalid request body: {exc}"},
                keep_alive=request.keep_alive,
            )
            return

        raw_messages = payload.get("messages")
        messages: List[Dict[str, object]] = (
            raw_messages if isinstance(raw_messages, list) else []
        )
        model = str(payload.get("model") or self._model)
        stream = payload.get("stream", True)
        params = self._tool_params(payload)
        pieces = ToolCallExtractor(
            self._request_stream(messages=messages, params=params),
            params.get("tools"),
        )

        def envelope(piece: str, done: bool) -> Dict[str, object]:
            obj: Dict[str, object] = {
                "model": model,
                "created_at": _now_iso(),
                "message": {"role": "assistant", "content": piece},
                "done": done,
            }
            if done:
                calls = self._collect_tool_calls(pieces, params.get("tools"))
                if calls:
                    # Ollama carries tool calls beside the content, never in it.
                    obj["message"]["tool_calls"] = tool_calls_to_ollama(calls)  # type: ignore[index]
                    obj["done_reason"] = "tool_calls"
                else:
                    obj["done_reason"] = "stop"
            return obj

        await self._stream_completion(
            request, writer, pieces, model,
            bool(stream), envelope,
        )

    async def _handle_openai_chat(
        self, request: _Request, writer: asyncio.StreamWriter
    ) -> None:
        """Serve the OpenAI-compatible ``POST /v1/chat/completions`` endpoint.

        VS Code Copilot's Ollama provider runs inference through the
        OpenAI-compatible endpoint (``${url}/v1/chat/completions``) rather than
        ``/api/chat``, so this bridges that request onto the same remote host
        prompt stream and re-frames the reply as OpenAI chunks.
        """

        try:
            payload = request.json()
        except (ValueError, json.JSONDecodeError) as exc:
            await self._write_json(
                writer, 400, {"error": {"message": f"invalid request body: {exc}"}},
                keep_alive=request.keep_alive,
            )
            return

        raw_messages = payload.get("messages")
        messages: List[Dict[str, object]] = (
            raw_messages if isinstance(raw_messages, list) else []
        )
        params = self._tool_params(payload)
        prompt = build_prompt_from_messages(messages, tools=params.get("tools"))  # type: ignore[arg-type]
        model = str(payload.get("model") or self._model)
        stream = payload.get("stream", True)
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = _now_unix()
        logger.info(
            "/v1/chat/completions: model=%s stream=%s messages=%d prompt=%d chars",
            model,
            bool(stream),
            len(messages),
            len(prompt),
        )
        pieces = ToolCallExtractor(
            self._request_stream(messages=messages, params=params),
            params.get("tools"),
        )
        tools = params.get("tools")
        if stream:
            await self._stream_openai(
                request, writer, pieces,
                model, completion_id, created, tools,
            )
        else:
            await self._aggregate_openai(
                request, writer, pieces,
                model, completion_id, created, tools,
            )

    @staticmethod
    def _openai_chunk(
        completion_id: str,
        created: int,
        model: str,
        delta: Dict[str, object],
        finish_reason: Optional[str],
    ) -> Dict[str, object]:
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }

    @staticmethod
    def _format_bridge_error(exc: BaseException) -> str:
        """Render a host/transport failure as a user-facing assistant message.

        Returning the failure as ordinary message content (rather than an SSE
        ``error`` object) means VS Code renders it as a normal reply instead of
        surfacing the opaque "Response contained no choices" error, and the
        bridge keeps running for the next request.
        """

        text = str(exc).strip() or exc.__class__.__name__
        return f"⚠️ keytalk bridge error: {text}"

    async def _stream_openai(
        self,
        request: _Request,
        writer: asyncio.StreamWriter,
        pieces,
        model: str,
        completion_id: str,
        created: int,
        tools: Optional[List[dict]] = None,
    ) -> None:
        await self._begin_sse(writer, request.keep_alive)
        await self._write_sse(
            writer,
            self._openai_chunk(
                completion_id, created, model, {"role": "assistant"}, None
            ),
        )
        count = 0
        chars = 0
        error_text: Optional[str] = None

        async def _forward(piece: str) -> None:
            nonlocal count, chars
            count += 1
            chars += len(piece)
            await self._write_sse(
                writer,
                self._openai_chunk(
                    completion_id, created, model, {"content": piece}, None
                ),
            )

        _, failure = await _drain(pieces, _forward)
        if failure is not None:
            logger.exception("error streaming OpenAI completion")
            error_text = self._format_bridge_error(failure)

        if error_text is None and count == 0:
            calls_probe = self._collect_tool_calls(pieces, tools)
            if not calls_probe:
                logger.warning(
                    "/v1/chat/completions produced no content from the host "
                    "(empty stream); returning a placeholder message"
                )
                error_text = self._format_bridge_error(
                    RuntimeError("the host returned no output for this prompt")
                )

        # Always emit a content chunk + a finish_reason so the response is a
        # valid OpenAI choice; on failure the error text rides along as the
        # message content instead of aborting the stream.  Tool calls ride as
        # a ``tool_calls`` delta with ``finish_reason: "tool_calls"``, exactly
        # like a local tool-aware server - never as assistant text.
        if error_text is not None:
            await self._write_sse(
                writer,
                self._openai_chunk(
                    completion_id, created, model, {"content": error_text}, None
                ),
            )
        calls = self._collect_tool_calls(pieces, tools)
        finish_reason = "stop"
        if calls and error_text is None:
            await self._write_sse(
                writer,
                self._openai_chunk(
                    completion_id,
                    created,
                    model,
                    {"tool_calls": self._tool_call_deltas(calls)},
                    None,
                ),
            )
            finish_reason = "tool_calls"
        await self._write_sse(
            writer,
            self._with_timings(
                self._openai_chunk(completion_id, created, model, {}, finish_reason),
                pieces,
            ),
        )
        if error_text is None:
            logger.info(
                "/v1/chat/completions streamed %d pieces (%d chars, %d tool calls)",
                count,
                chars,
                len(calls),
            )
        await self._write_sse_done(writer)
        await self._end_chunked(writer)

    async def _aggregate_openai(
        self,
        request: _Request,
        writer: asyncio.StreamWriter,
        pieces,
        model: str,
        completion_id: str,
        created: int,
        tools: Optional[List[dict]] = None,
    ) -> None:
        parts, failure = await _drain(pieces)
        error_text: Optional[str] = None
        if failure is not None:
            logger.exception("error generating OpenAI completion")
            error_text = self._format_bridge_error(failure)

        text = "".join(parts)
        calls = self._collect_tool_calls(pieces, tools) if error_text is None else []
        finish_reason = "stop"
        message: Dict[str, object] = {"role": "assistant", "content": text}
        if calls:
            message["tool_calls"] = calls
            finish_reason = "tool_calls"
        if error_text is not None:
            # Deliver the failure as assistant content (keeping the bridge alive
            # for the next request) rather than a 500 that aborts the client.
            message["content"] = error_text
            finish_reason = "error"
        obj: Dict[str, object] = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        self._with_timings(obj, pieces)
        await self._write_json(
            writer, 200, obj, keep_alive=request.keep_alive
        )

    async def _openai_models_payload(self) -> Dict[str, object]:
        names = await self._discover_models()
        if not names:
            names = [self._model]
        created = _now_unix()
        return {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "created": created,
                    "owned_by": "keytalk",
                }
                for name in names
            ],
        }

    def _request_stream(
        self,
        *,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, object]]] = None,
        params: Optional[Dict[str, object]] = None,
    ) -> "AsyncIterator[str]":
        """Open the response stream for a request, wrapped for the envelope.

        When the request offers ``tools`` the stream is wrapped in a
        :class:`~keytalk.toolcalls.ToolCallExtractor` so plaintext invocations
        can be lifted out of the prose; a request without tools is passed
        through untouched (filtering prose that cannot contain a call would
        only risk mangling it).  Either way the envelope reads this request's
        own tool calls and generation stats off the returned stream, never off
        shared client state.

        When the client supports structured chat (``chat_stream``), messages
        are passed through untouched (so the remote backend can render its
        model's own chat template) along with ``params`` - notably ``tools`` /
        ``tool_choice``, which reach the model as native tool definitions;
        otherwise they are flattened into a prompt that teaches the plaintext
        tool-call syntax the bridge parses back.
        """

        source: object
        if messages is not None:
            chat_stream = getattr(self._client, "chat_stream", None)
            if chat_stream is not None:
                source = chat_stream(messages, **dict(params or {}))
            else:
                prompt = build_prompt_from_messages(
                    messages, tools=(params or {}).get("tools")  # type: ignore[arg-type]
                )
                source = self._client.stream(prompt)
        else:
            source = self._client.stream(prompt or "")
        tools = (params or {}).get("tools")
        if not tools:
            return source  # type: ignore[return-value]
        return ToolCallExtractor(source, tools)  # type: ignore[arg-type]

    @staticmethod
    def _tool_params(payload: Dict[str, object]) -> Dict[str, object]:
        """Extract the tool-calling parameters to forward to the remote host."""

        params: Dict[str, object] = {}
        tools = payload.get("tools")
        if isinstance(tools, list) and tools:
            params["tools"] = tools
        tool_choice = payload.get("tool_choice")
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
        return params

    def _collect_tool_calls(
        self, pieces: object, tools: Optional[object] = None
    ) -> List[Dict[str, object]]:
        """Tool calls for *this* request, in strict OpenAI shape.

        Plaintext calls (``<function=...>`` markup) are extracted by the
        :class:`~keytalk.toolcalls.ToolCallExtractor` wrapping the stream;
        structured calls made through a backend's native tool channel arrive in
        the request's own meta trailer.  Both are read from the request's
        stream, never from shared client state, so concurrent requests cannot
        pick up each other's calls.  Only requests that actually offered tools
        can produce calls.
        """

        if not tools:
            return []
        return normalize_tool_calls(getattr(pieces, "tool_calls", None) or [])

    @staticmethod
    def _tool_call_deltas(calls: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """Render tool calls as OpenAI streaming ``delta.tool_calls`` fragments."""

        return [
            {
                "index": index,
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["function"]["name"],  # type: ignore[index]
                    "arguments": call["function"]["arguments"],  # type: ignore[index]
                },
            }
            for index, call in enumerate(calls)
        ]

    @staticmethod
    def _with_timings(obj: Dict[str, object], pieces: object) -> Dict[str, object]:
        """Attach *this request's* server-side generation stats to a final envelope."""

        timings = getattr(pieces, "timings", None)
        if timings:
            obj["timings"] = timings
        return obj

    async def _stream_completion(
        self,
        request: _Request,
        writer: asyncio.StreamWriter,
        pieces,
        model: str,
        stream: bool,
        envelope: Callable[[str, bool], Dict[str, object]],
    ) -> None:
        if stream:
            await self._stream_ndjson(request, writer, pieces, envelope)
        else:
            await self._aggregate_completion(
                request, writer, pieces, envelope
            )

    async def _stream_ndjson(
        self,
        request: _Request,
        writer: asyncio.StreamWriter,
        pieces,
        envelope: Callable[[str, bool], Dict[str, object]],
    ) -> None:
        # Headers are flushed before the model produces anything, so any error
        # must be reported in-band as an Ollama-style ``{"error": ...}`` line.
        await self._begin_chunked(writer, request.keep_alive)

        async def _forward(piece: str) -> None:
            await self._write_chunk_json(writer, envelope(piece, False))

        _, failure = await _drain(pieces, _forward)
        if failure is not None:
            logger.exception("error streaming completion")
            # Include done=true so clients using Symbol.asyncIterator don't
            # throw "Did not receive done or success response in stream".
            final = envelope("", True)
            final["error"] = str(failure)
            await self._write_chunk_json(writer, final)
        else:
            await self._write_chunk_json(
                writer, self._with_timings(envelope("", True), pieces)
            )
        await self._end_chunked(writer)

    async def _aggregate_completion(
        self,
        request: _Request,
        writer: asyncio.StreamWriter,
        pieces,
        envelope: Callable[[str, bool], Dict[str, object]],
    ) -> None:
        parts, failure = await _drain(pieces)
        if failure is not None:
            logger.exception("error generating completion")
            await self._write_json(
                writer, 500, {"error": str(failure)}, keep_alive=request.keep_alive
            )
            return

        text = "".join(parts)
        obj = self._with_timings(envelope(text, True), pieces)
        await self._write_json(
            writer, 200, obj, keep_alive=request.keep_alive
        )

    # -- low-level HTTP writing ----------------------------------------------

    @staticmethod
    def _status_line(status: int) -> bytes:
        reasons = {
            200: "OK",
            400: "Bad Request",
            404: "Not Found",
            500: "Internal Server Error",
        }
        reason = reasons.get(status, "OK")
        return f"HTTP/1.1 {status} {reason}\r\n".encode("latin-1")

    async def _write_text(
        self,
        request: _Request,
        writer: asyncio.StreamWriter,
        status: int,
        text: str,
    ) -> None:
        body = b"" if request.method.upper() == "HEAD" else text.encode("utf-8")
        head = self._status_line(status)
        head += b"Content-Type: text/plain; charset=utf-8\r\n"
        head += f"Content-Length: {len(text.encode('utf-8'))}\r\n".encode("latin-1")
        head += self._connection_header(request.keep_alive)
        head += b"\r\n"
        writer.write(head + body)
        await writer.drain()

    async def _write_json(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        obj: Dict[str, object],
        *,
        keep_alive: bool,
    ) -> None:
        body = json.dumps(obj).encode("utf-8")
        head = self._status_line(status)
        head += b"Content-Type: application/json; charset=utf-8\r\n"
        head += f"Content-Length: {len(body)}\r\n".encode("latin-1")
        head += self._connection_header(keep_alive)
        head += b"\r\n"
        writer.write(head + body)
        await writer.drain()

    async def _begin_chunked(
        self, writer: asyncio.StreamWriter, keep_alive: bool
    ) -> None:
        head = self._status_line(200)
        head += b"Content-Type: application/x-ndjson\r\n"
        head += b"Transfer-Encoding: chunked\r\n"
        head += self._connection_header(keep_alive)
        head += b"\r\n"
        writer.write(head)
        await writer.drain()

    async def _begin_sse(
        self, writer: asyncio.StreamWriter, keep_alive: bool
    ) -> None:
        head = self._status_line(200)
        head += b"Content-Type: text/event-stream; charset=utf-8\r\n"
        head += b"Cache-Control: no-cache\r\n"
        head += b"Transfer-Encoding: chunked\r\n"
        head += self._connection_header(keep_alive)
        head += b"\r\n"
        writer.write(head)
        await writer.drain()

    async def _write_chunk_json(
        self, writer: asyncio.StreamWriter, obj: Dict[str, object]
    ) -> None:
        payload = (json.dumps(obj) + "\n").encode("utf-8")
        await self._write_chunk(writer, payload)

    async def _write_sse(
        self, writer: asyncio.StreamWriter, obj: Dict[str, object]
    ) -> None:
        payload = ("data: " + json.dumps(obj) + "\n\n").encode("utf-8")
        await self._write_chunk(writer, payload)

    async def _write_sse_done(self, writer: asyncio.StreamWriter) -> None:
        await self._write_chunk(writer, b"data: [DONE]\n\n")

    @staticmethod
    async def _write_chunk(writer: asyncio.StreamWriter, data: bytes) -> None:
        if not data:
            return
        writer.write(f"{len(data):x}\r\n".encode("latin-1") + data + b"\r\n")
        await writer.drain()

    @staticmethod
    async def _end_chunked(writer: asyncio.StreamWriter) -> None:
        writer.write(b"0\r\n\r\n")
        await writer.drain()

    @staticmethod
    def _connection_header(keep_alive: bool) -> bytes:
        value = "keep-alive" if keep_alive else "close"
        return f"Connection: {value}\r\n".encode("latin-1")
