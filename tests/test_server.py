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

from keytalk.backends import LLMBackend, StaticBackend, TokenStream
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


class ModelListingStreamer(FakeStreamer):
    """A streamer that also reports a model list, like a real consumer client."""

    def __init__(self, models: List[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self._models = models

    async def list_models(self) -> List[str]:
        return list(self._models)


class ModelListErrorStreamer(FakeStreamer):
    """A streamer whose model-list lookup fails, to test graceful fallback."""

    async def list_models(self) -> List[str]:
        raise RuntimeError("host unreachable")


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

    def sse(self) -> List[dict]:
        objects = []
        for line in self.body.decode("utf-8").splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                objects.append(json.loads(line[6:]))
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

    async def _make_full_stack(
        self, backend: LLMBackend
    ) -> Tuple[OllamaBridgeServer, HostService]:
        """A real host + consumer over the loopback transport, plus the bridge.

        Uses a deliberately tiny frame payload so responses span many frames
        and exercise the chunking path.
        """

        host_t, consumer_t = create_loopback()
        host = HostService(host_t, backend, max_payload_size=6)
        consumer = ConsumerClient(consumer_t, max_payload_size=6, timeout=5.0)
        await host.start()
        await consumer.start()
        self.addAsyncCleanup(host.close)
        self.addAsyncCleanup(consumer.close)
        return await self._serve(consumer, model="bridged"), host


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
        out = build_prompt_from_messages([{"role": "narrator", "content": "x"}])
        self.assertEqual(out, "User: x\nAssistant:")

    def test_tool_result_rendered_as_tool_line(self):
        out = build_prompt_from_messages(
            [{"role": "tool", "content": "found it", "tool_call_id": "call_0"}]
        )
        self.assertEqual(out, "Tool result (call_0): found it\nAssistant:")

    def test_assistant_tool_calls_rendered_as_calls(self):
        out = build_prompt_from_messages(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": '{"query": "x"}',
                            },
                        }
                    ],
                }
            ]
        )
        self.assertEqual(
            out,
            "Assistant: <function=search>{\"query\": \"x\"}</function>\nAssistant:",
        )

    def test_tools_add_instruction_section(self):
        out = build_prompt_from_messages(
            [{"role": "user", "content": "hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "search the web",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )
        self.assertIn("<function=tool_name>", out)
        self.assertIn("- search: search the web", out)
        self.assertTrue(out.endswith("User: hi\nAssistant:"))

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

    async def test_tags_lists_models_reported_by_host(self):
        streamer = ModelListingStreamer(["llama3:8b", "qwen2.5:latest"])
        server = await self._serve(streamer, model="keytalk")
        resp = await _request(server.host, server.port, "GET", "/api/tags")
        names = [m["name"] for m in resp.json()["models"]]
        self.assertEqual(names, ["llama3:8b", "qwen2.5:latest"])

    async def test_tags_falls_back_when_host_lists_nothing(self):
        streamer = ModelListingStreamer([])
        server = await self._serve(streamer, model="my-model")
        resp = await _request(server.host, server.port, "GET", "/api/tags")
        names = [m["name"] for m in resp.json()["models"]]
        self.assertEqual(names, ["my-model:latest"])

    async def test_tags_falls_back_when_host_errors(self):
        server = await self._serve(ModelListErrorStreamer(), model="my-model")
        resp = await _request(server.host, server.port, "GET", "/api/tags")
        names = [m["name"] for m in resp.json()["models"]]
        self.assertEqual(names, ["my-model:latest"])

    async def test_show_returns_object(self):
        server = await self._serve(FakeStreamer())
        resp = await _request(
            server.host, server.port, "POST", "/api/show",
            body=json.dumps({"name": "keytalk"}).encode(),
        )
        self.assertEqual(resp.status, 200)
        self.assertIn("details", resp.json())

    async def test_show_advertises_tool_capability(self):
        # Copilot only registers models whose /api/show reports "tools".
        server = await self._serve(FakeStreamer())
        resp = await _request(
            server.host, server.port, "POST", "/api/show",
            body=json.dumps({"model": "keytalk"}).encode(),
        )
        data = resp.json()
        self.assertIn("tools", data["capabilities"])
        self.assertIn("completion", data["capabilities"])

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
# OpenAI-compatible /v1/chat/completions (used by VS Code Copilot for inference)
# --------------------------------------------------------------------------- #
def _parse_sse(text: str) -> List[object]:
    """Parse a ``text/event-stream`` body into its ``data:`` payloads.

    Returns parsed JSON objects, with the literal ``[DONE]`` sentinel kept as a
    string so tests can assert on stream termination.
    """

    events: List[object] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block.startswith("data:"):
            continue
        payload = block[len("data:"):].strip()
        if payload == "[DONE]":
            events.append("[DONE]")
        elif payload:
            events.append(json.loads(payload))
    return events


class OpenAIEndpointTests(ServerTestBase):
    async def test_streaming_chat_completions(self):
        fake = FakeStreamer(response="Hi there", piece_size=2)
        server = await self._serve(fake, model="m")
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
            server.host, server.port, "POST", "/v1/chat/completions", body=body
        )
        self.assertEqual(resp.status, 200)
        self.assertIn("text/event-stream", resp.headers.get("content-type", ""))
        events = _parse_sse(resp.text())
        self.assertEqual(events[-1], "[DONE]")
        chunks = [e for e in events if isinstance(e, dict)]
        for chunk in chunks:
            self.assertEqual(chunk["object"], "chat.completion.chunk")
            self.assertEqual(chunk["model"], "m")
        # first chunk announces the assistant role
        self.assertEqual(chunks[0]["choices"][0]["delta"].get("role"), "assistant")
        # the final content chunk closes with finish_reason "stop"
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        content = "".join(
            c["choices"][0]["delta"].get("content", "") for c in chunks
        )
        self.assertEqual(content, "Hi there")
        self.assertEqual(fake.prompts[0], "Be nice.\nUser: hello\nAssistant:")

    async def test_non_streaming_chat_completions(self):
        fake = FakeStreamer(response="Reply text", piece_size=3)
        server = await self._serve(fake, model="m")
        body = json.dumps(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "q"}],
                "stream": False,
            }
        ).encode()
        resp = await _request(
            server.host, server.port, "POST", "/v1/chat/completions", body=body
        )
        self.assertEqual(resp.status, 200)
        data = resp.json()
        self.assertEqual(data["object"], "chat.completion")
        choice = data["choices"][0]
        self.assertEqual(choice["message"]["role"], "assistant")
        self.assertEqual(choice["message"]["content"], "Reply text")
        self.assertEqual(choice["finish_reason"], "stop")

    async def test_chat_completions_invalid_json(self):
        server = await self._serve(FakeStreamer())
        resp = await _request(
            server.host, server.port, "POST", "/v1/chat/completions",
            body=b"oops",
        )
        self.assertEqual(resp.status, 400)
        self.assertIn("message", resp.json()["error"])

    async def test_non_streaming_chat_completions_error_is_graceful(self):
        class ImmediateFail:
            def stream(self, prompt):
                async def gen():
                    if False:
                        yield ""
                    raise RuntimeError("nope")
                return gen()

        server = await self._serve(ImmediateFail())
        resp = await _request(
            server.host, server.port, "POST", "/v1/chat/completions",
            body=json.dumps(
                {"messages": [{"role": "user", "content": "hi"}], "stream": False}
            ).encode(),
        )
        # The bridge stays alive and surfaces the failure as assistant content
        # rather than an HTTP 500 that would abort the client.
        self.assertEqual(resp.status, 200)
        choice = resp.json()["choices"][0]
        self.assertEqual(choice["finish_reason"], "error")
        self.assertIn("nope", choice["message"]["content"])

    async def test_streaming_chat_completions_error_is_graceful(self):
        server = await self._serve(FailingStreamer("kaput"))
        resp = await _request(
            server.host, server.port, "POST", "/v1/chat/completions",
            body=json.dumps(
                {"messages": [{"role": "user", "content": "hi"}]}
            ).encode(),
        )
        self.assertEqual(resp.status, 200)
        events = _parse_sse(resp.text())
        self.assertEqual(events[-1], "[DONE]")
        chunks = [e for e in events if isinstance(e, dict)]
        # No raw SSE error objects: every chunk is a valid completion chunk.
        for chunk in chunks:
            self.assertEqual(chunk["object"], "chat.completion.chunk")
            self.assertNotIn("error", chunk)
        # The stream still closes with a "stop" finish_reason ...
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        # ... and the error text is delivered as assistant content.
        content = "".join(
            c["choices"][0]["delta"].get("content", "") for c in chunks
        )
        self.assertIn("kaput", content)

    async def test_v1_models_lists_host_models(self):
        streamer = ModelListingStreamer(models=["alpha", "beta"])
        server = await self._serve(streamer)
        resp = await _request(server.host, server.port, "GET", "/v1/models")
        self.assertEqual(resp.status, 200)
        data = resp.json()
        self.assertEqual(data["object"], "list")
        ids = [entry["id"] for entry in data["data"]]
        self.assertEqual(ids, ["alpha", "beta"])
        for entry in data["data"]:
            self.assertEqual(entry["object"], "model")

    async def test_v1_models_falls_back_to_configured_model(self):
        server = await self._serve(FakeStreamer(), model="solo")
        resp = await _request(server.host, server.port, "GET", "/v1/models")
        self.assertEqual(resp.status, 200)
        ids = [entry["id"] for entry in resp.json()["data"]]
        self.assertEqual(ids, ["solo"])



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


async def _async_list(items: List[str]) -> List[str]:
    return list(items)


class EndToEndPipelineTests(ServerTestBase):
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

    async def test_tags_lists_host_models_through_ble_pipeline(self):
        backend = StaticBackend("unused", piece_size=2)
        # The host's backend reports two models; the bridge should surface both.
        backend.list_models = lambda: _async_list(["llama3:8b", "phi3:mini"])  # type: ignore[method-assign]
        server, _ = await self._make_full_stack(backend)
        resp = await _request(server.host, server.port, "GET", "/api/tags")
        names = [m["name"] for m in resp.json()["models"]]
        self.assertEqual(names, ["llama3:8b", "phi3:mini"])


# --------------------------------------------------------------------------- #
# Tool calling over the bridge
# --------------------------------------------------------------------------- #
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "search the web",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
}

PLAINTEXT_CALL_REPLY = 'Let me check.\n<function=search>{"query": "ble"}</function>'
#: The exact shape reported in the wild: functionary-v1-style plaintext calls.
FUNCTIONARY_CALL_REPLY = 'function search(query="ble docs")</function>'


class _ToolBackend(LLMBackend):
    """Chat-capable backend that records the params it got and can report
    structured tool calls through its stream metadata."""

    def __init__(self, reply: str = "hi", meta: dict = None) -> None:
        self._reply = reply
        self._meta = meta or {}
        self.seen_params: Dict[str, object] = {}
        self.seen_messages: List[dict] = []

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        for i in range(0, len(self._reply), 3):
            yield self._reply[i : i + 3]

    def generate_messages(self, messages, **params):
        self.seen_messages = messages
        self.seen_params = params
        reply = self._reply

        async def gen():
            for i in range(0, len(reply), 3):
                yield reply[i : i + 3]

        return TokenStream(gen(), dict(self._meta))


class ToolCallBridgeTests(ServerTestBase):
    async def _make_full_stack(self, backend):
        host_t, consumer_t = create_loopback()
        host = HostService(host_t, backend, max_payload_size=6)
        consumer = ConsumerClient(consumer_t, max_payload_size=6, timeout=5.0)
        await host.start()
        await consumer.start()
        self.addAsyncCleanup(host.close)
        self.addAsyncCleanup(consumer.close)
        server = await self._serve(consumer, model="bridged")
        return server, host

    @staticmethod
    def _chat_body(**extra) -> bytes:
        body: Dict[str, object] = {
            "model": "bridged",
            "messages": [{"role": "user", "content": "find ble docs"}],
            "tools": [SEARCH_TOOL],
        }
        body.update(extra)
        return json.dumps(body).encode()

    # -- plaintext calls re-framed as tool_calls ------------------------------

    async def test_openai_streaming_re_frames_plaintext_calls(self):
        # A prompt-only model replying with <function=...> markup: the client
        # must see structured tool_calls, never the markup as content.
        server, _ = await self._make_full_stack(
            StaticBackend(PLAINTEXT_CALL_REPLY, piece_size=3)
        )
        resp = await _request(
            server.host, server.port, "POST", "/v1/chat/completions",
            body=self._chat_body(stream=True),
        )
        self.assertEqual(resp.status, 200)
        chunks = resp.sse()
        content = "".join(
            c["choices"][0]["delta"].get("content") or "" for c in chunks
        )
        self.assertNotIn("<function=", content)
        self.assertEqual(content.strip(), "Let me check.")
        deltas = [
            c["choices"][0]["delta"]["tool_calls"][0]
            for c in chunks
            if c["choices"][0]["delta"].get("tool_calls")
        ]
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["function"]["name"], "search")
        self.assertEqual(
            json.loads(deltas[0]["function"]["arguments"]), {"query": "ble"}
        )
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")

    async def test_openai_non_streaming_re_frames_plaintext_calls(self):
        server, _ = await self._make_full_stack(
            StaticBackend(PLAINTEXT_CALL_REPLY, piece_size=3)
        )
        resp = await _request(
            server.host, server.port, "POST", "/v1/chat/completions",
            body=self._chat_body(stream=False),
        )
        self.assertEqual(resp.status, 200)
        choice = resp.json()["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertNotIn("<function=", choice["message"]["content"])
        calls = choice["message"]["tool_calls"]
        self.assertEqual(calls[0]["function"]["name"], "search")

    async def test_ollama_chat_reports_tool_calls(self):
        server, _ = await self._make_full_stack(
            StaticBackend(PLAINTEXT_CALL_REPLY, piece_size=3)
        )
        resp = await _request(
            server.host, server.port, "POST", "/api/chat", body=self._chat_body()
        )
        objects = resp.ndjson()
        final = objects[-1]
        self.assertTrue(final["done"])
        self.assertEqual(final["done_reason"], "tool_calls")
        calls = final["message"]["tool_calls"]
        self.assertEqual(calls[0]["function"]["name"], "search")
        # Ollama models arguments as an object, not a JSON string
        self.assertEqual(calls[0]["function"]["arguments"], {"query": "ble"})
        content = "".join(o["message"]["content"] for o in objects)
        self.assertNotIn("<function=", content)

    async def test_functionary_plaintext_calls_re_framed(self):
        # ``function name(args)</function>`` (kwargs, not JSON) must arrive as
        # a structured call too - this is what models emit in the wild.
        server, _ = await self._make_full_stack(
            StaticBackend(FUNCTIONARY_CALL_REPLY, piece_size=2)
        )
        resp = await _request(
            server.host, server.port, "POST", "/v1/chat/completions",
            body=self._chat_body(stream=True),
        )
        chunks = resp.sse()
        content = "".join(
            c["choices"][0]["delta"].get("content") or "" for c in chunks
        )
        self.assertNotIn("function search", content)
        deltas = [
            c["choices"][0]["delta"]["tool_calls"][0]
            for c in chunks
            if c["choices"][0]["delta"].get("tool_calls")
        ]
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["function"]["name"], "search")
        self.assertEqual(
            json.loads(deltas[0]["function"]["arguments"]), {"query": "ble docs"}
        )
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")

    # -- structured calls reported by the remote backend ----------------------

    async def test_native_tool_calls_from_trailer_reach_client(self):
        meta = {
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"query": "x"}'},
                }
            ],
            "finish_reason": "tool_calls",
        }
        server, _ = await self._make_full_stack(_ToolBackend("Looking...", meta))
        resp = await _request(
            server.host, server.port, "POST", "/v1/chat/completions",
            body=self._chat_body(stream=True),
        )
        chunks = resp.sse()
        deltas = [
            c["choices"][0]["delta"]["tool_calls"][0]
            for c in chunks
            if c["choices"][0]["delta"].get("tool_calls")
        ]
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["id"], "call_abc")
        self.assertEqual(deltas[0]["function"]["name"], "search")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")

    async def test_tools_forwarded_to_remote_backend(self):
        backend = _ToolBackend("ok")
        server, _ = await self._make_full_stack(backend)
        await _request(
            server.host, server.port, "POST", "/v1/chat/completions",
            body=self._chat_body(stream=False),
        )
        # tool definitions crossed the link as native request params
        self.assertEqual(backend.seen_params.get("tools"), [SEARCH_TOOL])
        self.assertEqual(backend.seen_messages[0]["content"], "find ble docs")

    # -- no false positives ---------------------------------------------------

    async def test_plain_reply_has_no_tool_calls(self):
        server, _ = await self._make_full_stack(
            StaticBackend("Try {curly} braces.", piece_size=3)
        )
        resp = await _request(
            server.host, server.port, "POST", "/v1/chat/completions",
            body=self._chat_body(stream=False),
        )
        choice = resp.json()["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertNotIn("tool_calls", choice["message"])
        self.assertEqual(choice["message"]["content"], "Try {curly} braces.")


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


class ConcurrentMetadataTests(ServerTestBase):
    """Per-request trailer metadata must not leak between concurrent requests.

    ``/v1/chat/completions`` and ``/api/chat`` read tool calls and generation
    stats off the request's own response stream.  If those were read from
    shared client state instead, a request that finishes last would hand its
    metadata to whichever envelope happened to be built next.
    """

    class _PerPromptBackend(LLMBackend):
        """Reports a distinct tool call per prompt, with staggered timing.

        ``generate`` receives the *flattened* prompt (this backend has no
        ``generate_messages``), so the request is identified by its content.
        """

        async def _run(self, name: str, meta: dict):
            yield "hi "
            # "slow" deliberately finishes last, after "fast" is done.
            await asyncio.sleep(0.15 if name == "slow" else 0.02)
            yield "there"
            meta["tool_calls"] = [
                {
                    "id": f"call_{name}",
                    "type": "function",
                    "function": {
                        "name": f"tool_{name}",
                        "arguments": json.dumps({"n": name}),
                    },
                }
            ]
            meta["finish_reason"] = "tool_calls"
            meta["timings"] = {"which": name}

        def generate(self, prompt: str) -> TokenStream:
            name = "slow" if "User: slow" in prompt else "fast"
            meta: dict = {}
            return TokenStream(self._run(name, meta), meta)

    async def test_concurrent_tool_calls_stay_isolated(self) -> None:
        server, _ = await self._make_full_stack(self._PerPromptBackend())

        def _body(prompt: str) -> bytes:
            return json.dumps(
                {
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": f"tool_{prompt}",
                                "parameters": {},
                            },
                        }
                    ],
                }
            ).encode()

        # "fast" is started first but answered first; "slow" finishes last and
        # must not overwrite the metadata already reported for "fast".
        fast, slow = await asyncio.gather(
            _request(server.host, server.port, "POST", "/api/chat", body=_body("fast")),
            _request(server.host, server.port, "POST", "/api/chat", body=_body("slow")),
        )
        for name, resp in (("fast", fast), ("slow", slow)):
            message = resp.json()["message"]
            names = [c["function"]["name"] for c in message.get("tool_calls", [])]
            self.assertEqual(names, [f"tool_{name}"], f"{name} got {names}")
            self.assertEqual(resp.json().get("timings"), {"which": name})
            self.assertEqual(resp.json()["done_reason"], "tool_calls")

    async def test_concurrent_openai_tool_calls_stay_isolated(self) -> None:
        server, _ = await self._make_full_stack(self._PerPromptBackend())

        def _body(prompt: str) -> bytes:
            return json.dumps(
                {
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": f"tool_{prompt}",
                                "parameters": {},
                            },
                        }
                    ],
                }
            ).encode()

        fast, slow = await asyncio.gather(
            _request(
                server.host, server.port, "POST", "/v1/chat/completions",
                body=_body("fast"),
            ),
            _request(
                server.host, server.port, "POST", "/v1/chat/completions",
                body=_body("slow"),
            ),
        )
        for name, resp in (("fast", fast), ("slow", slow)):
            choice = resp.json()["choices"][0]
            message = choice["message"]
            names = [c["function"]["name"] for c in message.get("tool_calls", [])]
            self.assertEqual(names, [f"tool_{name}"], f"{name} got {names}")
            self.assertEqual(choice["finish_reason"], "tool_calls")
            self.assertEqual(resp.json().get("timings"), {"which": name})


if __name__ == "__main__":
    unittest.main()
