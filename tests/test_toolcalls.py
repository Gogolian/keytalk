"""Unit tests for :mod:`keytalk.toolcalls`.

The bridge must turn the plaintext tool-call markup models emit when they have
no structured tool channel (``<function=search>{"query": "x"}</function>`` etc.)
into OpenAI-style ``tool_calls`` - and must never leave that markup in the
assistant content or mistake prose for a call.
"""

import json
import unittest

from keytalk.toolcalls import (
    ToolCallExtractor,
    ToolCallFilter,
    normalize_tool_calls,
    parse_text_tool_calls,
    tool_calls_to_ollama,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "search the web",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {"name": "read_file", "parameters": {"type": "object"}},
    },
]


class ParseTextToolCallsTests(unittest.TestCase):
    def test_function_tag_call(self):
        text = 'Searching...\n<function=search>{"query": "ble"}</function>'
        content, calls = parse_text_tool_calls(text)
        self.assertEqual(content, "Searching...")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "search")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]), {"query": "ble"}
        )
        self.assertTrue(calls[0]["id"])

    def test_function_tag_without_closing_tag(self):
        content, calls = parse_text_tool_calls(
            '<function=search>{"query": "ble"}</function>'
            '<function=read_file>{"path": "a.txt"}'
        )
        self.assertEqual(content, "")
        self.assertEqual([c["function"]["name"] for c in calls], ["search", "read_file"])

    def test_multi_line_arguments(self):
        text = '<function=search>{\n  "query": "ble",\n  "k": 3\n}</function>'
        content, calls = parse_text_tool_calls(text)
        self.assertEqual(content, "")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"])["k"], 3)

    def test_wrapped_arguments_unwrapped(self):
        text = '<function=search>{"call": {"query": "x"}}</function>'
        _, calls = parse_text_tool_calls(text)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"query": "x"})

    def test_qwen_to_syntax(self):
        text = 'to=search {"query": "x"}'
        content, calls = parse_text_tool_calls(text)
        self.assertEqual(content, "")
        self.assertEqual(calls[0]["function"]["name"], "search")

    def test_functionary_call_syntax(self):
        text = 'function search(query="ble docs")</function>'
        content, calls = parse_text_tool_calls(text)
        self.assertEqual(content, "")
        self.assertEqual(calls[0]["function"]["name"], "search")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]), {"query": "ble docs"}
        )

    def test_functionary_kwargs_and_parens_in_strings(self):
        text = 'Checking.\nfunction read_file(path="a (1).txt", k=3)'
        content, calls = parse_text_tool_calls(text)
        self.assertEqual(content, "Checking.")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"path": "a (1).txt", "k": 3},
        )

    def test_functionary_prose_is_not_a_call(self):
        text = "function call style is nice but not a call."
        content, calls = parse_text_tool_calls(text, TOOLS)
        self.assertEqual(calls, [])
        self.assertEqual(content, text)

    def test_bare_call_only_for_declared_tools(self):
        content, calls = parse_text_tool_calls('search{"query": "x"}', TOOLS)
        self.assertEqual(content, "")
        self.assertEqual(calls[0]["function"]["name"], "search")
        # ... but prose that merely looks like one stays prose without tools
        content, calls = parse_text_tool_calls('search{"query": "x"}')
        self.assertEqual(calls, [])
        self.assertEqual(content, 'search{"query": "x"}')

    def test_prose_never_mistaken_for_call(self):
        text = "Use search{...} carefully, or call <function> helpers."
        content, calls = parse_text_tool_calls(text, TOOLS)
        self.assertEqual(calls, [])
        self.assertEqual(content, text)

    def test_content_survives_beside_calls(self):
        text = 'Let me check.\n<function=search>{"query": "x"}</function>\nOne moment.'
        content, calls = parse_text_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(content, "Let me check.\nOne moment.")
        self.assertNotIn("<function=", content)

    def test_no_markup_is_untouched(self):
        text = "Just a normal reply.\nWith lines."
        self.assertEqual(parse_text_tool_calls(text), (text, []))


class NormalizeToolCallsTests(unittest.TestCase):
    def test_arguments_coerced_to_json_string(self):
        calls = normalize_tool_calls(
            [
                {"function": {"name": "search", "arguments": {"query": "x"}}},
                {"name": "read_file", "arguments": '{"path": "a"}'},
            ]
        )
        self.assertEqual(calls[0]["function"]["arguments"], '{"query":"x"}')
        self.assertEqual(calls[1]["id"], "call_1_read_file")
        self.assertEqual(calls[1]["type"], "function")

    def test_ollama_shape(self):
        ollama = tool_calls_to_ollama(
            [{"id": "c0", "function": {"name": "search", "arguments": '{"query": "x"}'}}]
        )
        self.assertEqual(
            ollama, [{"function": {"name": "search", "arguments": {"query": "x"}}}]
        )


class ToolCallFilterTests(unittest.TestCase):
    def _drain(self, filter_, text):
        return filter_.feed(text)

    def test_prose_passes_through(self):
        f = ToolCallFilter(TOOLS)
        self.assertEqual(self._drain(f, "hello world"), "hello world")
        self.assertEqual(f.finish(), "")
        self.assertEqual(f.calls, [])

    def test_call_withheld_then_extracted(self):
        f = ToolCallFilter(TOOLS)
        out = self._drain(f, 'Let me check.\n<function=se')
        self.assertEqual(out, "Let me check.\n")
        out += self._drain(f, 'arch>{"query": ')
        self.assertEqual(out, "Let me check.\n")
        out += self._drain(f, '"x"}</function>\nDone.')
        self.assertEqual(out, "Let me check.\nDone.")
        self.assertEqual(f.finish(), "")
        self.assertEqual(len(f.calls), 1)
        self.assertEqual(f.calls[0]["function"]["name"], "search")

    def test_tail_call_harvested_at_finish(self):
        f = ToolCallFilter()
        out = f.feed('Sure.\n<function=search>{"query": "x"}')
        self.assertEqual(out, "Sure.\n")
        self.assertEqual(f.finish(), "")
        self.assertEqual(len(f.calls), 1)

    def test_partial_tool_name_line_is_guarded(self):
        # "sea" could still grow into the tool name "search": never flush it.
        f = ToolCallFilter(TOOLS)
        out = f.feed("Looking.\nsea")
        self.assertEqual(out, "Looking.\n")
        out += f.feed("rch not needed after all.\n")
        self.assertEqual(out, "Looking.\nsearch not needed after all.\n")
        self.assertEqual(f.calls, [])


class ToolCallExtractorTests(unittest.IsolatedAsyncioTestCase):
    async def test_extractor_strips_calls_from_stream(self):
        async def source():
            for piece in ["Hello!\n<function=se", 'arch>{"query": "x"}</function>']:
                yield piece

        extractor = ToolCallExtractor(source(), TOOLS)
        pieces = [piece async for piece in extractor]
        text = "".join(pieces)
        self.assertEqual(text, "Hello!\n")
        self.assertEqual(len(extractor.tool_calls), 1)
        self.assertEqual(extractor.tool_calls[0]["function"]["name"], "search")

    async def test_extractor_merges_source_tool_calls(self):
        class Structured:
            tool_calls = [
                {"id": "x", "type": "function",
                 "function": {"name": "read_file", "arguments": '{"path": "a"}'}}
            ]

            def __aiter__(self):
                async def gen():
                    yield "text"

                return gen()

        extractor = ToolCallExtractor(Structured(), TOOLS)
        self.assertEqual("".join([p async for p in extractor]), "text")
        names = [c["function"]["name"] for c in extractor.tool_calls]
        self.assertEqual(names, ["read_file"])


if __name__ == "__main__":
    unittest.main()
