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


def build_prompt_from_messages(messages: List[Dict[str, object]]) -> str:
    """Flatten Ollama ``/api/chat`` messages into a single prompt string.

    The remote host bridges to Ollama's ``/api/generate`` (a plain-text prompt),
    so chat-style message lists are rendered into a simple, readable transcript
    ending with an ``Assistant:`` cue.  ``system`` messages are emitted first as
    context; ``user``/``assistant`` turns follow in order.
    """

    systems: List[str] = []
    turns: List[str] = []
    for message in messages:
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

    parts: List[str] = []
    if systems:
        parts.append("\n".join(systems))
    parts.extend(turns)
    body = "\n".join(parts)
    # Cue the model to produce the assistant's next turn.
    if body:
        return f"{body}\nAssistant:"
    return "Assistant:"


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
            request, writer, prompt, model, bool(stream), envelope, "response"
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
        prompt = build_prompt_from_messages(messages)
        model = str(payload.get("model") or self._model)
        stream = payload.get("stream", True)

        def envelope(piece: str, done: bool) -> Dict[str, object]:
            obj: Dict[str, object] = {
                "model": model,
                "created_at": _now_iso(),
                "message": {"role": "assistant", "content": piece},
                "done": done,
            }
            if done:
                obj["done_reason"] = "stop"
            return obj

        await self._stream_completion(
            request, writer, prompt, model, bool(stream), envelope, "message"
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
        prompt = build_prompt_from_messages(messages)
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

        if stream:
            await self._stream_openai(
                request, writer, prompt, model, completion_id, created
            )
        else:
            await self._aggregate_openai(
                request, writer, prompt, model, completion_id, created
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
        prompt: str,
        model: str,
        completion_id: str,
        created: int,
    ) -> None:
        await self._begin_sse(writer, request.keep_alive)
        await self._write_sse(
            writer,
            self._openai_chunk(
                completion_id, created, model, {"role": "assistant"}, None
            ),
        )
        pieces = 0
        chars = 0
        error_text: Optional[str] = None
        try:
            async for piece in self._client.stream(prompt):
                if not piece:
                    continue
                pieces += 1
                chars += len(piece)
                await self._write_sse(
                    writer,
                    self._openai_chunk(
                        completion_id, created, model, {"content": piece}, None
                    ),
                )
        except asyncio.CancelledError:  # pragma: no cover - teardown path
            raise
        except Exception as exc:  # noqa: BLE001 - surface any backend failure
            logger.exception("error streaming OpenAI completion")
            error_text = self._format_bridge_error(exc)

        if error_text is None and pieces == 0:
            logger.warning(
                "/v1/chat/completions produced no content from the host "
                "(empty stream); returning a placeholder message"
            )
            error_text = self._format_bridge_error(
                RuntimeError("the host returned no output for this prompt")
            )

        # Always emit a content chunk + a "stop" finish_reason so the response
        # is a valid OpenAI choice; on failure the error text rides along as the
        # message content instead of aborting the stream.
        if error_text is not None:
            await self._write_sse(
                writer,
                self._openai_chunk(
                    completion_id, created, model, {"content": error_text}, None
                ),
            )
        await self._write_sse(
            writer,
            self._openai_chunk(completion_id, created, model, {}, "stop"),
        )
        if error_text is None:
            logger.info(
                "/v1/chat/completions streamed %d pieces (%d chars)",
                pieces,
                chars,
            )
        await self._write_sse_done(writer)
        await self._end_chunked(writer)

    async def _aggregate_openai(
        self,
        request: _Request,
        writer: asyncio.StreamWriter,
        prompt: str,
        model: str,
        completion_id: str,
        created: int,
    ) -> None:
        parts: List[str] = []
        error_text: Optional[str] = None
        try:
            async for piece in self._client.stream(prompt):
                if piece:
                    parts.append(piece)
        except asyncio.CancelledError:  # pragma: no cover - teardown path
            raise
        except Exception as exc:  # noqa: BLE001 - surface any backend failure
            logger.exception("error generating OpenAI completion")
            error_text = self._format_bridge_error(exc)

        text = "".join(parts)
        finish_reason = "stop"
        if error_text is not None:
            # Deliver the failure as assistant content (keeping the bridge alive
            # for the next request) rather than a 500 that aborts the client.
            text = error_text
            finish_reason = "error"
        obj: Dict[str, object] = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
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

    async def _stream_completion(
        self,
        request: _Request,
        writer: asyncio.StreamWriter,
        prompt: str,
        model: str,
        stream: bool,
        envelope: Callable[[str, bool], Dict[str, object]],
        aggregate_field: str,
    ) -> None:
        if stream:
            await self._stream_ndjson(request, writer, prompt, envelope)
        else:
            await self._aggregate_completion(
                request, writer, prompt, envelope, aggregate_field
            )

    async def _stream_ndjson(
        self,
        request: _Request,
        writer: asyncio.StreamWriter,
        prompt: str,
        envelope: Callable[[str, bool], Dict[str, object]],
    ) -> None:
        # Headers are flushed before the model produces anything, so any error
        # must be reported in-band as an Ollama-style ``{"error": ...}`` line.
        await self._begin_chunked(writer, request.keep_alive)
        try:
            async for piece in self._client.stream(prompt):
                if not piece:
                    continue
                await self._write_chunk_json(writer, envelope(piece, False))
            await self._write_chunk_json(writer, envelope("", True))
        except asyncio.CancelledError:  # pragma: no cover - teardown path
            raise
        except Exception as exc:  # noqa: BLE001 - surface any backend failure
            logger.exception("error streaming completion")
            # Include done=true so clients using Symbol.asyncIterator don't
            # throw "Did not receive done or success response in stream".
            final = envelope("", True)
            final["error"] = str(exc)
            await self._write_chunk_json(writer, final)
        await self._end_chunked(writer)

    async def _aggregate_completion(
        self,
        request: _Request,
        writer: asyncio.StreamWriter,
        prompt: str,
        envelope: Callable[[str, bool], Dict[str, object]],
        aggregate_field: str,
    ) -> None:
        parts: List[str] = []
        try:
            async for piece in self._client.stream(prompt):
                if piece:
                    parts.append(piece)
        except asyncio.CancelledError:  # pragma: no cover - teardown path
            raise
        except Exception as exc:  # noqa: BLE001 - surface any backend failure
            logger.exception("error generating completion")
            await self._write_json(
                writer, 500, {"error": str(exc)}, keep_alive=request.keep_alive
            )
            return

        text = "".join(parts)
        obj = envelope(text, True)
        # For non-streaming chat the assistant content carries the whole text;
        # for generate the ``response`` field does.  ``envelope`` placed the
        # text via its first argument, so the full payload is already correct.
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
