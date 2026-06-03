"""Tests for the LLM backends (fakes + Ollama line parsing)."""

import io
import json
import unittest
from unittest import mock

from keytalk.backends import (
    EchoBackend,
    OllamaBackend,
    OllamaError,
    StaticBackend,
    parse_ollama_line,
)


async def _collect(backend, prompt):
    return [piece async for piece in backend.generate(prompt)]


class EchoBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_echo_streams_words(self):
        pieces = await _collect(EchoBackend(prefix="> "), "alpha beta")
        self.assertEqual("".join(pieces), "> alpha beta ")
        # streamed in multiple pieces (prefix + per word)
        self.assertGreater(len(pieces), 1)


class StaticBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_response_in_pieces(self):
        backend = StaticBackend("abcdefg", piece_size=3)
        pieces = await _collect(backend, "anything")
        self.assertEqual(pieces, ["abc", "def", "g"])

    async def test_rejects_bad_piece_size(self):
        with self.assertRaises(ValueError):
            StaticBackend("x", piece_size=0)


class OllamaLineParsingTests(unittest.TestCase):
    def test_extracts_response(self):
        self.assertEqual(
            parse_ollama_line(b'{"response": "Hel", "done": false}'), "Hel"
        )

    def test_blank_line_is_none(self):
        self.assertIsNone(parse_ollama_line(b"  \n"))

    def test_done_without_text_is_none(self):
        self.assertIsNone(parse_ollama_line(b'{"response": "", "done": true}'))

    def test_error_field_raises(self):
        with self.assertRaises(OllamaError):
            parse_ollama_line(b'{"error": "model not found"}')

    def test_invalid_json_raises(self):
        with self.assertRaises(OllamaError):
            parse_ollama_line(b"not json")


class OllamaRequestBodyTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _capture(captured):
        def fake_urlopen(request, timeout=None):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return io.BytesIO(b'{"response": "hi", "done": true}\n')

        return fake_urlopen

    async def test_num_ctx_sent_as_option(self):
        captured: dict = {}
        with mock.patch(
            "keytalk.backends.urllib.request.urlopen",
            self._capture(captured),
        ):
            backend = OllamaBackend(model="m", num_ctx=8192)
            await _collect(backend, "hello")
        self.assertEqual(captured["body"]["options"], {"num_ctx": 8192})

    async def test_num_ctx_omitted_when_none(self):
        captured: dict = {}
        with mock.patch(
            "keytalk.backends.urllib.request.urlopen",
            self._capture(captured),
        ):
            backend = OllamaBackend(model="m", num_ctx=None)
            await _collect(backend, "hello")
        self.assertNotIn("options", captured["body"])


if __name__ == "__main__":
    unittest.main()
