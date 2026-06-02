"""Tests for the LLM backends (fakes + Ollama line parsing)."""

import unittest

from keytalk.backends import (
    EchoBackend,
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


if __name__ == "__main__":
    unittest.main()
