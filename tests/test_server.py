"""Tests for the Ollama-compatible HTTP bridge (:mod:`keytalk.server`).

These exercise the server end-to-end over a real TCP socket bound to an
ephemeral port.  Completions are driven either by a tiny in-process fake
streamer or by a genuine :class:`~keytalk.consumer.ConsumerClient` wired to a
:class:`~keytalk.host.HostService` over the in-memory loopback transport - so
the whole prompt-over-BLE pipeline is covered with no Bluetooth hardware and no
real Ollama install.
"""

import asyncio
import json
import unittest
from typing import AsyncIterator, Dict, List, Optional, Tuple

from keytalk.backends import LLMBackend, StaticBackend
from keytalk.consumer import ConsumerClient
from keytalk.host import HostService
from keytalk.server import (
    OllamaBridgeServer,
    build_prompt_from_messages,
)
from keytalk.transport import create_loopback


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class FakeStreamer:
    """A minimal :class:`~keytalk.server.PromptStreamer`.

    Records the prompts it was asked to stream and replays a canned response in
    fixed-size pieces, so streaming behaviour is observable and deterministic.
    """

    def __init__(self, response: str = "hello world", piece_size: int = 3) -> None:
        self.response = response
        self.piece_size = piece_size
        self.prompts: List[str] = []

    def stream(self, prompt: str) -> AsyncIterator[str]:
        self.prompts.append(prompt)
        return self._gen()

    async def _gen(self) -> AsyncIterator[str]:
        text = self.response
        for i in range(0, len(text), self.piece_size):
            await asyncio.sleep(0)
            yield text[i : i + self.piece_size]


class FailingStreamer:
    """Yields a token, then raises - to test mid-stream error reporting."""

    def __init__(self, message: str = "boom") -> None:
        self.message = message

    def stream(self, prompt: str) -> AsyncIterator[str]:
        return self._gen()

    async def _gen(self) -> AsyncIterator[str]:
        yield "partial"
        raise RuntimeError(self.message)


# --------------------------------------------------------------------------- #
# A tiny async HTTP/1.1 client that understands chunked + content-length
# --------------------------------------------------------------------------- #
class HTTPResponse:
    def __init__(
        self, status: int, headers: Dict[str, str], body: bytes
    ) -> None:
        self.status = status
        self.headers = headers
        self.body = body

    def text(self) -> str:
        return self.body.decode("utf-8")

    def json(self) -> object:
        return json.loads(self.body.decode("utf-8"))

    def ndjson(self) -> List[dict]:
        objects = []
        for line in self.body.decode("utf-8").splitlines():
            line = line.strip()
            if line:
                objects.append(json.loads(line))
        return objects


async def _read_response(
    reader: asyncio.StreamReader, method: str = "GET"
) -> HTTPResponse:
    status_line = await reader.readline()
    parts = status_line.decode("latin-1").rstrip("\r\n").split(" ", 2)
    status = int(parts[1])

    headers: Dict[str, str] = {}
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        name, value = line.decode("latin-1").rstrip("\r\n").split(":", 1)
        headers[name.strip().lower()] = value.strip()

    body = b""
    if method.upper() == "HEAD":
        return HTTPResponse(status, headers, body)  # HEAD has no body
    if headers.get("transfer-encoding", "").lower() == "chunked":
        while True:
            size_line = await reader.readline()
            size = int(size_line.strip(), 16)
            if size == 0:
                await reader.readline()  # trailing CRLF after last chunk
                break
            chunk = await reader.readexactly(size)
            await reader.readexactly(2)  # CRLF
            body += chunk
    elif "content-length" in headers:
        length = int(headers["content-length"])
        if length:
            body = await reader.readexactly(length)
    return HTTPResponse(status, headers, body)


async def _request(
    host: str,
    port: int,
    method: str,
    path: str,
    *,
    body: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
    keep_alive: bool = False,
) -> HTTPResponse:
    reader, writer = await asyncio.open_connection(host, port)
    try:
        return await _send_on(
            reader, writer, method, path,
            body=body, headers=headers, keep_alive=keep_alive, host=host,
        )
    finally:
        writer.close()


async def _send_on(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    method: str,
    path: str,
    *,
    body: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
    keep_alive: bool = False,
    host: str = "127.0.0.1",
) -> HTTPResponse:
    lines = [f"{method} {path} HTTP/1.1", f"Host: {host}"]
    lines.append("Connection: keep-alive" if keep_alive else "Connection: close")
    all_headers = dict(headers or {})
    if body is not None:
        all_headers.setdefault("Content-Type", "application/json")
        all_headers["Content-Length"] = str(len(body))
    for name, value in all_headers.items():
        lines.append(f"{name}: {value}")
    raw = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")
    if body:
        raw += body
    writer.write(raw)
    await writer.drain()
    return await _read_response(reader, method)


# --------------------------------------------------------------------------- #
# Server test base
# --------------------------------------------------------------------------- #
class ServerTestBase(unittest.IsolatedAsyncioTestCase):
    async def _serve(self, client, *, model: str = "keytalk") -> OllamaBridgeServer:
        server = OllamaBridgeServer(client, host="127.0.0.1", port=0, model=model)
        await server.start()
        self.addAsyncCleanup(server.close)
        return server


# --------------------------------------------------------------------------- #
# build_prompt_from_messages
# --------------------------------------------------------------------------- #
class PromptBuildingTests(unittest.TestCase):
    def test_empty_messages(self):
        self.assertEqual(build_prompt_from_messages([]), "Assistant:")

    def test_single_user_message(self):
        out = build_prompt_from_messages([{"role": "user", "content": "hi"}])
        self.assertEqual(out, "User: hi\nAssistant:")

    def test_system_then_conversation(self):
        out = build_prompt_from_messages(
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "More?"},
            ]
        )
        self.assertEqual(
            out,
            "Be brief.\nUser: Hello\nAssistant: Hi!\nUser: More?\nAssistant:",
        )

    def test_multiple_system_messages_join(self):
        out = build_prompt_from_messages(
            [
                {"role": "system", "content": "A."},
                {"role": "system", "content": "B."},
                {"role": "user", "content": "go"},
            ]
        )
        self.assertEqual(out, "A.\nB.\nUser: go\nAssistant:")

    def test_unknown_role_treated_as_user(self):
        out = build_prompt_from_messages([{"role": "tool", "content": "x"}])
        self.assertEqual(out, "User: x\nAssistant:")

    def test_missing_and_none_content(self):
        out = build_prompt_from_messages(
            [{"role": "user"}, {"role": "assistant", "content": None}]
        )
        self.assertEqual(out, "User: \nAssistant: \nAssistant:")

    def test_non_dict_entries_skipped(self):
        out = build_prompt_from_messages(
            ["nope", 123, {"role": "user", "content": "ok"}]  # type: ignore[list-item]
        )
        self.assertEqual(out, "User: ok\nAssistant:")


# --------------------------------------------------------------------------- #
# Discovery endpoints
# --------------------------------------------------------------------------- #
class DiscoveryEndpointTests(ServerTestBase):
    async def test_root_reports_ollama_running(self):
        server = await self._serve(FakeStreamer())
        resp = await _request(server.host, server.port, "GET", "/")
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.text(), "Ollama is running")

    async def test_head_root_has_no_body(self):
        server = await self._serve(FakeStreamer())
        resp = await _request(server.host, server.port, "HEAD", "/")
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.body, b"")

    async def test_version(self):
        server = await self._serve(FakeStreamer())
        resp = await _request(server.host, server.port, "GET", "/api/version")
        self.assertEqual(resp.status, 200)
        self.assertIn("version", resp.json())

    async def test_tags_lists_configured_model(self):
        server = await self._serve(FakeStreamer(), model="my-model")
        resp = await _request(server.host, server.port, "GET", "/api/tags")
        self.assertEqual(resp.status, 200)
        data = resp.json()
        names = [m["name"] for m in data["models"]]
        self.assertEqual(names, ["my-model:latest"])

    async def test_tags_preserves_explicit_tag(self):
        server = await self._serve(FakeStreamer(), model="my-model:7b")
        resp = await _request(server.host, server.port, "GET", "/api/tags")
        names = [m["name"] for m in resp.json()["models"]]
        self.assertEqual(names, ["my-model:7b"])

    async def test_show_returns_object(self):
        server = await self._serve(FakeStreamer())
        resp = await _request(
            server.host, server.port, "POST", "/api/show",
            body=json.dumps({"name": "keytalk"}).encode(),
        )
        self.assertEqual(resp.status, 200)
        self.assertIn("details", resp.json())

    async def test_ps_endpoint(self):
        server = await self._serve(FakeStreamer())
        resp = await _request(server.host, server.port, "GET", "/api/ps")
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.json(), {"models": []})

    async def test_unknown_endpoint_404(self):
        server = await self._serve(FakeStreamer())
        resp = await _request(server.host, server.port, "GET", "/nope")
        self.assertEqual(resp.status, 404)
        self.assertIn("error", resp.json())


# --------------------------------------------------------------------------- #
# /api/generate
# --------------------------------------------------------------------------- #
class GenerateEndpointTests(ServerTestBase):
    async def test_streaming_generate(self):
        fake = FakeStreamer(response="hello world", piece_size=3)
        server = await self._serve(fake)
        resp = await _request(
            server.host, server.port, "POST", "/api/generate",
            body=json.dumps({"model": "m", "prompt": "hi"}).encode(),
        )
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.headers.get("transfer-encoding"), "chunked")
        objects = resp.ndjson()
        # All but the last carry response text; the last is the done marker.
        self.assertTrue(all(o["model"] == "m" for o in objects))
        self.assertEqual(objects[-1]["done"], True)
        self.assertEqual(objects[-1]["done_reason"], "stop")
        self.assertFalse(any(o["done"] for o in objects[:-1]))
        text = "".join(o["response"] for o in objects)
        self.assertEqual(text, "hello world")
        self.assertEqual(fake.prompts, ["hi"])

    async def test_streaming_is_incremental(self):
        fake = FakeStreamer(response="abcdef", piece_size=2)
        server = await self._serve(fake)
        resp = await _request(
            server.host, server.port, "POST", "/api/generate",
            body=json.dumps({"prompt": "go"}).encode(),
        )
        objects = resp.ndjson()
        data_chunks = [o for o in objects if not o["done"]]
        self.assertEqual(len(data_chunks), 3)  # ab cd ef

    async def test_non_streaming_generate(self):
        fake = FakeStreamer(response="all at once", piece_size=2)
        server = await self._serve(fake)
        resp = await _request(
            server.host, server.port, "POST", "/api/generate",
            body=json.dumps({"prompt": "hi", "stream": False}).encode(),
        )
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.headers.get("transfer-encoding"), None)
        data = resp.json()
        self.assertEqual(data["response"], "all at once")
        self.assertEqual(data["done"], True)

    async def test_default_model_used_when_absent(self):
        server = await self._serve(FakeStreamer(response="x"), model="defmod")
        resp = await _request(
            server.host, server.port, "POST", "/api/generate",
            body=json.dumps({"prompt": "hi"}).encode(),
        )
        self.assertEqual(resp.ndjson()[-1]["model"], "defmod")

    async def test_invalid_json_body(self):
        server = await self._serve(FakeStreamer())
        resp = await _request(
            server.host, server.port, "POST", "/api/generate",
            body=b"{not json",
        )
        self.assertEqual(resp.status, 400)
        self.assertIn("error", resp.json())

    async def test_streaming_error_reported_inband(self):
        server = await self._serve(FailingStreamer("kaput"))
        resp = await _request(
            server.host, server.port, "POST", "/api/generate",
            body=json.dumps({"prompt": "hi"}).encode(),
        )
        self.assertEqual(resp.status, 200)
        objects = resp.ndjson()
        self.assertEqual(objects[0]["response"], "partial")
        self.assertIn("error", objects[-1])
        self.assertIn("kaput", objects[-1]["error"])

    async def test_non_streaming_error_is_500(self):
        # FailingStreamer raises after one token; without anything flushed yet
        # (headers not sent) the aggregate path can return a 500.
        class ImmediateFail:
            def stream(self, prompt):
                async def gen():
                    if False:
                        yield ""
                    raise RuntimeError("nope")
                return gen()

        server = await self._serve(ImmediateFail())
        resp = await _request(
            server.host, server.port, "POST", "/api/generate",
            body=json.dumps({"prompt": "hi", "stream": False}).encode(),
        )
        self.assertEqual(resp.status, 500)
        self.assertIn("nope", resp.json()["error"])


# --------------------------------------------------------------------------- #
# /api/chat
# --------------------------------------------------------------------------- #
class ChatEndpointTests(ServerTestBase):
    async def test_streaming_chat(self):
        fake = FakeStreamer(response="Hi there", piece_size=2)
        server = await self._serve(fake)
        body = json.dumps(
            {
                "model": "m",
                "messages": [
                    {"role": "system", "content": "Be nice."},
                    {"role": "user", "content": "hello"},
                ],
            }
        ).encode()
        resp = await _request(
            server.host, server.port, "POST", "/api/chat", body=body
        )
        self.assertEqual(resp.status, 200)
        objects = resp.ndjson()
        self.assertEqual(objects[-1]["done"], True)
        content = "".join(o["message"]["content"] for o in objects)
        self.assertEqual(content, "Hi there")
        for o in objects:
            self.assertEqual(o["message"]["role"], "assistant")
        # the prompt forwarded reflects the chat transcript
        self.assertEqual(
            fake.prompts[0], "Be nice.\nUser: hello\nAssistant:"
        )

    async def test_non_streaming_chat(self):
        fake = FakeStreamer(response="Reply text", piece_size=3)
        server = await self._serve(fake)
        body = json.dumps(
            {"messages": [{"role": "user", "content": "q"}], "stream": False}
        ).encode()
        resp = await _request(
            server.host, server.port, "POST", "/api/chat", body=body
        )
        self.assertEqual(resp.status, 200)
        data = resp.json()
        self.assertEqual(data["message"]["content"], "Reply text")
        self.assertEqual(data["done"], True)

    async def test_chat_without_messages(self):
        fake = FakeStreamer(response="ok")
        server = await self._serve(fake)
        resp = await _request(
            server.host, server.port, "POST", "/api/chat",
            body=json.dumps({}).encode(),
        )
        self.assertEqual(resp.status, 200)
        self.assertEqual(fake.prompts[0], "Assistant:")

    async def test_chat_invalid_json(self):
        server = await self._serve(FakeStreamer())
        resp = await _request(
            server.host, server.port, "POST", "/api/chat", body=b"oops",
        )
        self.assertEqual(resp.status, 400)


# --------------------------------------------------------------------------- #
# Connection handling
# --------------------------------------------------------------------------- #
class ConnectionHandlingTests(ServerTestBase):
    async def test_keep_alive_serves_multiple_requests(self):
        server = await self._serve(FakeStreamer(response="pong"))
        reader, writer = await asyncio.open_connection(server.host, server.port)
        try:
            for _ in range(3):
                resp = await _send_on(
                    reader, writer, "GET", "/api/version", keep_alive=True
                )
                self.assertEqual(resp.status, 200)
                self.assertEqual(resp.headers.get("connection"), "keep-alive")
        finally:
            writer.close()

    async def test_connection_close_header(self):
        server = await self._serve(FakeStreamer())
        resp = await _request(server.host, server.port, "GET", "/api/version")
        self.assertEqual(resp.headers.get("connection"), "close")

    async def test_malformed_request_line_400(self):
        server = await self._serve(FakeStreamer())
        reader, writer = await asyncio.open_connection(server.host, server.port)
        try:
            writer.write(b"GARBAGE\r\n\r\n")
            await writer.drain()
            resp = await _read_response(reader)
            self.assertEqual(resp.status, 400)
        finally:
            writer.close()

    async def test_method_not_allowed_falls_through_to_404(self):
        server = await self._serve(FakeStreamer())
        resp = await _request(server.host, server.port, "DELETE", "/api/tags")
        self.assertEqual(resp.status, 404)

    async def test_oversized_content_length_rejected(self):
        server = await self._serve(FakeStreamer())
        reader, writer = await asyncio.open_connection(server.host, server.port)
        try:
            writer.write(
                b"POST /api/generate HTTP/1.1\r\n"
                b"Content-Length: 99999999999\r\n\r\n"
            )
            await writer.drain()
            resp = await _read_response(reader)
            self.assertEqual(resp.status, 400)
        finally:
            writer.close()


# --------------------------------------------------------------------------- #
# Full pipeline: server -> ConsumerClient -> loopback -> HostService -> backend
# --------------------------------------------------------------------------- #
class _SlowBackend(LLMBackend):
    def __init__(self, text: str) -> None:
        self._text = text

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        for ch in self._text:
            await asyncio.sleep(0.001)
            yield ch


class EndToEndPipelineTests(ServerTestBase):
    async def _make_full_stack(
        self, backend: LLMBackend
    ) -> Tuple[OllamaBridgeServer, HostService]:
        host_t, consumer_t = create_loopback()
        host = HostService(host_t, backend, max_payload_size=6)
        consumer = ConsumerClient(consumer_t, max_payload_size=6, timeout=5.0)
        await host.start()
        await consumer.start()
        self.addAsyncCleanup(host.close)
        self.addAsyncCleanup(consumer.close)
        server = await self._serve(consumer, model="bridged")
        return server, host

    async def test_generate_through_ble_pipeline(self):
        server, _ = await self._make_full_stack(
            StaticBackend("The quick brown fox", piece_size=2)
        )
        resp = await _request(
            server.host, server.port, "POST", "/api/generate",
            body=json.dumps({"prompt": "go", "stream": False}).encode(),
        )
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.json()["response"], "The quick brown fox")

    async def test_chat_streaming_through_ble_pipeline(self):
        server, _ = await self._make_full_stack(StaticBackend("héllo 世界", 2))
        body = json.dumps(
            {"messages": [{"role": "user", "content": "hi"}]}
        ).encode()
        resp = await _request(
            server.host, server.port, "POST", "/api/chat", body=body
        )
        objects = resp.ndjson()
        content = "".join(o.get("message", {}).get("content", "") for o in objects)
        self.assertEqual(content, "héllo 世界")
        self.assertTrue(objects[-1]["done"])

    async def test_concurrent_requests_isolated(self):
        server, _ = await self._make_full_stack(_SlowBackend("ANSWER"))
        results = await asyncio.gather(
            _request(
                server.host, server.port, "POST", "/api/generate",
                body=json.dumps({"prompt": "a", "stream": False}).encode(),
            ),
            _request(
                server.host, server.port, "POST", "/api/generate",
                body=json.dumps({"prompt": "b", "stream": False}).encode(),
            ),
            _request(
                server.host, server.port, "POST", "/api/generate",
                body=json.dumps({"prompt": "c", "stream": False}).encode(),
            ),
        )
        for resp in results:
            self.assertEqual(resp.json()["response"], "ANSWER")


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_manager_and_port_resolution(self):
        async with OllamaBridgeServer(
            FakeStreamer(), host="127.0.0.1", port=0
        ) as server:
            self.assertGreater(server.port, 0)
            resp = await _request(server.host, server.port, "GET", "/")
            self.assertEqual(resp.status, 200)

    async def test_close_is_idempotent(self):
        server = OllamaBridgeServer(FakeStreamer(), host="127.0.0.1", port=0)
        await server.start()
        await server.close()
        await server.close()  # second close must not raise


if __name__ == "__main__":
    unittest.main()
