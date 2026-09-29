"""Tests for the llama.cpp backend against a real local HTTP server.

The canned responses replicate the exact wire shapes of this llama.cpp tree
(``tools/server/server-task.cpp``, verified against the source):

* ``POST /completions`` streams SSE ``data: {json}`` lines with top-level
  ``content``/``stop`` and a final ``timings`` object (incl. MTP
  ``draft_n``/``draft_n_accepted``);
* ``POST /v1/chat/completions`` streams OpenAI-style ``choices[0].delta``
  chunks, a final chunk carrying ``timings``, then ``data: [DONE]``;
* ``GET /v1/models`` returns ``{"models": [...], "object": "list", "data": [...]}``.

Everything runs over a real loopback TCP socket - no mocked urllib.
"""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from keytalk.backends import LlamaCppBackend, LlamaCppError, TokenStream


def _sse(obj) -> str:
    return f"data: {json.dumps(obj)}\n\n"


NATIVE_STREAM = (
    _sse({"index": 0, "content": "Hel", "tokens": [1], "stop": False,
          "id_slot": 0, "tokens_predicted": 1, "tokens_evaluated": 3})
    + ":\n\n"  # SSE ping line (llama.cpp sends these on slow streams)
    + _sse({"index": 0, "content": "lo", "tokens": [2], "stop": False,
            "id_slot": 0, "tokens_predicted": 2, "tokens_evaluated": 3})
    + _sse({"index": 0, "content": "", "tokens": [], "stop": True,
            "id_slot": 0, "tokens_predicted": 3, "tokens_evaluated": 3,
            "timings": {"prompt_n": 3, "predicted_n": 3,
                        "predicted_per_second": 50.0,
                        "draft_n": 4, "draft_n_accepted": 3}})
)

CHAT_STREAM = (
    _sse({"choices": [{"delta": {"role": "assistant", "content": None},
                       "finish_reason": None, "index": 0}],
          "created": 1, "id": "chatcmpl-x", "model": "m",
          "object": "chat.completion.chunk"})
    + _sse({"choices": [{"delta": {"content": "Wo"},
                         "finish_reason": None, "index": 0}],
            "created": 1, "id": "chatcmpl-x", "model": "m",
            "object": "chat.completion.chunk"})
    + _sse({"choices": [{"delta": {"reasoning_content": "thinking..."},
                         "finish_reason": None, "index": 0}],
            "created": 1, "id": "chatcmpl-x", "model": "m",
            "object": "chat.completion.chunk"})
    + _sse({"choices": [{"delta": {"content": "rld"},
                         "finish_reason": None, "index": 0}],
            "created": 1, "id": "chatcmpl-x", "model": "m",
            "object": "chat.completion.chunk"})
    + _sse({"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
            "created": 1, "id": "chatcmpl-x", "model": "m",
            "object": "chat.completion.chunk",
            "timings": {"prompt_n": 5, "predicted_n": 2,
                        "draft_n": 2, "draft_n_accepted": 1}})
    + "data: [DONE]\n\n"
)

MODELS_BODY = {
    "models": [{"name": "bench"}],
    "object": "list",
    "data": [{"id": "Qwen3.8-3.6-27B-blend-Q5_K_M", "object": "model",
              "owned_by": "llamacpp"}],
}


class _Handler(BaseHTTPRequestHandler):
    """Serves canned llama.cpp responses and records what was requested."""

    routes: dict = {}
    requests: list = []

    def log_message(self, *args):  # noqa: ANN001 - silence the stdlib logger
        pass

    def do_GET(self):  # noqa: N802 - stdlib naming
        self._respond()

    def do_POST(self):  # noqa: N802 - stdlib naming
        self._respond()

    def _respond(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        type(self).requests.append((self.command, self.path, body))
        status, ctype, payload = type(self).routes.get(
            self.path, (404, "application/json", json.dumps({"error": "no route"}))
        )
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(payload.encode("utf-8") if isinstance(payload, str) else payload)


class LlamaCppTestBase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _Handler.routes = {}
        _Handler.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def route(self, path, body, status=200, ctype="text/event-stream"):
        _Handler.routes[path] = (status, ctype, body)


class NativeCompletionsTests(LlamaCppTestBase):
    async def test_streams_content_and_timings(self):
        self.route("/completions", NATIVE_STREAM)
        backend = LlamaCppBackend(host=self.host)
        stream = backend.generate("hi")
        self.assertIsInstance(stream, TokenStream)
        pieces = [p async for p in stream]
        self.assertEqual("".join(pieces), "Hello")
        self.assertEqual(stream.timings["draft_n"], 4)
        self.assertEqual(stream.timings["draft_n_accepted"], 3)

        method, path, body = _Handler.requests[0]
        self.assertEqual((method, path), ("POST", "/completions"))
        request = json.loads(body)
        self.assertEqual(request["prompt"], "hi")
        self.assertIs(request["stream"], True)

    async def test_n_predict_passthrough(self):
        self.route("/completions", NATIVE_STREAM)
        backend = LlamaCppBackend(host=self.host, n_predict=384)
        [p async for p in backend.generate("hi")]
        request = json.loads(_Handler.requests[0][2])
        self.assertEqual(request["n_predict"], 384)

    async def test_no_timings_means_none(self):
        body = _sse({"index": 0, "content": "x", "stop": True,
                     "id_slot": 0, "tokens_predicted": 1, "tokens_evaluated": 1})
        self.route("/completions", body)
        backend = LlamaCppBackend(host=self.host)
        stream = backend.generate("hi")
        self.assertEqual([p async for p in stream], ["x"])
        self.assertIsNone(stream.timings)


class ChatCompletionsTests(LlamaCppTestBase):
    async def test_messages_pass_through_untouched(self):
        self.route("/v1/chat/completions", CHAT_STREAM)
        backend = LlamaCppBackend(model="m", host=self.host)
        messages = [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello",
             "reasoning_content": "thinking..."},
        ]
        stream = backend.generate_messages(messages, temperature=0.2)
        pieces = [p async for p in stream]
        # content deltas only: the reasoning delta must not leak into the text
        self.assertEqual("".join(pieces), "World")
        self.assertEqual(stream.timings["draft_n_accepted"], 1)

        method, path, body = _Handler.requests[0]
        self.assertEqual((method, path), ("POST", "/v1/chat/completions"))
        request = json.loads(body)
        # structured messages (incl. reasoning_content) survive verbatim
        self.assertEqual(request["messages"], messages)
        self.assertIs(request["stream"], True)
        self.assertEqual(request["model"], "m")
        self.assertEqual(request["temperature"], 0.2)

    async def test_http_error_surfaces(self):
        self.route("/v1/chat/completions", json.dumps({"error": {"message": "boom"}}),
                   status=500, ctype="application/json")
        backend = LlamaCppBackend(host=self.host)
        with self.assertRaises(LlamaCppError) as ctx:
            [p async for p in backend.generate_messages([{"role": "user", "content": "x"}])]
        self.assertIn("boom", str(ctx.exception))

    async def test_in_stream_error_surfaces(self):
        body = _sse({"choices": [{"delta": {"content": "a"}, "index": 0}]}) \
            + _sse({"error": {"message": "model exploded"}})
        self.route("/v1/chat/completions", body)
        backend = LlamaCppBackend(host=self.host)
        with self.assertRaises(LlamaCppError):
            [p async for p in backend.generate_messages([{"role": "user", "content": "x"}])]


class ListModelsTests(LlamaCppTestBase):
    async def test_reads_data_array_with_models_fallback(self):
        self.route("/v1/models", json.dumps(MODELS_BODY), ctype="application/json")
        backend = LlamaCppBackend(host=self.host)
        self.assertEqual(await backend.list_models(), ["Qwen3.8-3.6-27B-blend-Q5_K_M"])

        # llama.cpp's /v1/models also carries an Ollama-style "models" array;
        # fall back to it when "data" is empty.
        self.route("/v1/models",
                   json.dumps({"models": [{"name": "fallback-model"}], "data": []}),
                   ctype="application/json")
        self.assertEqual(await backend.list_models(), ["fallback-model"])


if __name__ == "__main__":
    unittest.main()
