"""End-to-end tests for structured chat passthrough and the TIMINGS trailer.

A CHAT message carries ``{"messages": [...], "params": {...}}`` so a backend
with native chat support can render the model's own template server-side;
backends without it fall back to a flattened transcript.  Generation stats ride
back as plain-text TIMINGS frames at the tail of the response message (after
the compressed stream is cut), so they can never race request teardown.
"""

import asyncio
import json
import unittest

from keytalk.backends import LLMBackend, TokenStream
from keytalk.consumer import ConsumerClient, RemoteError, _PendingRequest
from keytalk.host import HostService
from keytalk.protocol import Flags, Frame, MessageType, chunk_message
from keytalk.transport import create_loopback

TINY = 6


class _ChatBackend(LLMBackend):
    """Captures the messages it receives and streams a canned reply."""

    def __init__(self, reply: str, timings: dict = None) -> None:
        self._reply = reply
        self._timings = timings
        self.seen_messages = None
        self.seen_params = None

    async def generate(self, prompt: str):
        yield f"flat:{prompt}"

    def generate_messages(self, messages, **params):
        self.seen_messages = messages
        self.seen_params = params
        meta = {"timings": self._timings} if self._timings else {}

        async def gen():
            for i in range(0, len(self._reply), 3):
                yield self._reply[i : i + 3]

        return TokenStream(gen(), meta)


class _ExplodingChatBackend(_ChatBackend):
    def generate_messages(self, messages, **params):
        async def gen():
            yield "partial "
            raise RuntimeError("chat exploded")

        return TokenStream(gen(), {})


class ChatTestBase(unittest.IsolatedAsyncioTestCase):
    async def _make(self, backend, **consumer_kw):
        host_t, consumer_t = create_loopback()
        host = HostService(host_t, backend, max_payload_size=TINY)
        consumer = ConsumerClient(
            consumer_t,
            max_payload_size=TINY,
            timeout=5.0,
            keepalive_interval=0,
            **consumer_kw,
        )
        await host.start()
        await consumer.start()
        self.addAsyncCleanup(consumer.close)
        self.addAsyncCleanup(host.close)
        return host, consumer


class ChatPassthroughTests(ChatTestBase):
    async def test_messages_reach_backend_structured(self):
        backend = _ChatBackend("hello world")
        _, consumer = await self._make(backend)
        messages = [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo", "reasoning_content": "t"},
        ]
        result = await consumer.chat(messages, temperature=0.7)
        self.assertEqual(result, "hello world")
        # structured messages arrive verbatim - no template flattening
        self.assertEqual(backend.seen_messages, messages)
        self.assertEqual(backend.seen_params, {"temperature": 0.7})

    async def test_chat_stream_matches_chat(self):
        backend = _ChatBackend("hello world")
        _, consumer = await self._make(backend)
        pieces = [p async for p in consumer.chat_stream([{"role": "user", "content": "x"}])]
        self.assertEqual("".join(pieces), "hello world")

    async def test_fallback_for_prompt_only_backends(self):
        from keytalk.backends import EchoBackend

        _, consumer = await self._make(EchoBackend(prefix="got: "))
        result = await consumer.chat(
            [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
        )
        # flattened transcript reaches the prompt-only backend
        self.assertIn("s", result)
        self.assertIn("User: hi", result)

    async def test_bad_payload_reports_remote_error(self):
        backend = _ChatBackend("unused")
        host, consumer = await self._make(backend)
        # Bypass the client API to send a malformed CHAT payload.
        pending = _PendingRequest(1)
        consumer._pending[1] = pending  # register first: the error can be fast
        frames = chunk_message(MessageType.CHAT, 1, b"not json", TINY)
        for frame in frames:
            await consumer._transport.send(frame.encode())
        with self.assertRaises(RemoteError) as ctx:
            await asyncio.wait_for(pending.__aiter__().__anext__(), timeout=2.0)
        self.assertIn("bad chat payload", str(ctx.exception))

    async def test_chat_error_becomes_remote_error(self):
        _, consumer = await self._make(_ExplodingChatBackend("x"))
        with self.assertRaises(RemoteError) as ctx:
            await consumer.chat([{"role": "user", "content": "x"}])
        self.assertIn("chat exploded", str(ctx.exception))


class TimingsTrailerTests(ChatTestBase):
    TIMINGS = {
        "prompt_n": 12,
        "predicted_n": 34,
        "predicted_per_second": 55.5,
        "draft_n": 10,
        "draft_n_accepted": 7,
    }

    async def test_timings_roundtrip_on_chat(self):
        backend = _ChatBackend("hello world", timings=self.TIMINGS)
        _, consumer = await self._make(backend)
        self.assertIsNone(consumer.last_timings)
        result = await consumer.chat([{"role": "user", "content": "x"}])
        self.assertEqual(result, "hello world")
        # trailer arrived (before END) and is captured on the client
        self.assertEqual(consumer.last_timings["draft_n_accepted"], 7)
        self.assertEqual(consumer.last_timings["predicted_per_second"], 55.5)

    async def test_timings_roundtrip_on_prompt(self):
        # A TokenStream-based prompt backend's stats ride the same trailer.
        from keytalk.backends import EchoBackend

        class _TimedEcho(EchoBackend):
            def generate(self, prompt: str):
                meta = {"timings": self.TIMINGS}

                async def gen():
                    async for piece in EchoBackend.generate(self, prompt):
                        yield piece

                return TokenStream(gen(), meta)

        backend = _TimedEcho()
        backend.TIMINGS = self.TIMINGS
        _, consumer = await self._make(backend)
        self.assertEqual(await consumer.generate("hi"), "echo: hi ")
        self.assertEqual(consumer.last_timings["draft_n"], 10)

    async def test_no_timings_reports_none(self):
        _, consumer = await self._make(_ChatBackend("hello"))
        await consumer.chat([{"role": "user", "content": "x"}])
        self.assertIsNone(consumer.last_timings)

    async def test_timings_survive_heavy_fragmentation(self):
        # Trailer spans many frames at TINY payload size and interleaves with
        # the compressed response stream without corrupting either.
        big = {"prompt_n": 1, "predicted_n": 2, "draft_n": 3,
               "draft_n_accepted": 4, "note": "x" * 200}
        backend = _ChatBackend("fragmented reply " * 5, timings=big)
        _, consumer = await self._make(backend)
        result = await consumer.chat([{"role": "user", "content": "x"}])
        self.assertEqual(result, "fragmented reply " * 5)
        self.assertEqual(consumer.last_timings["note"], "x" * 200)


class ToolCallTrailerTests(ChatTestBase):
    CALLS = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search", "arguments": '{"query": "x"}'},
        }
    ]

    @staticmethod
    def _backend(reply="on it", meta=None):
        class _CallingBackend(_ChatBackend):
            def generate_messages(self, messages, **params):
                self.seen_messages = messages
                self.seen_params = params
                text = reply

                async def gen():
                    yield text

                return TokenStream(gen(), dict(meta or {}))

        return _CallingBackend("unused")

    async def test_tool_calls_roundtrip_on_chat(self):
        backend = self._backend(
            meta={"tool_calls": self.CALLS, "finish_reason": "tool_calls"}
        )
        _, consumer = await self._make(backend)
        result = await consumer.chat([{"role": "user", "content": "x"}])
        self.assertEqual(result, "on it")
        # structured calls ride the trailer and land on the client
        self.assertEqual(consumer.last_tool_calls, self.CALLS)
        self.assertEqual(consumer.last_finish_reason, "tool_calls")

    async def test_tool_calls_roundtrip_on_prompt(self):
        from keytalk.backends import EchoBackend

        meta = {"tool_calls": self.CALLS, "finish_reason": "tool_calls"}

        class _CallingEcho(EchoBackend):
            def generate(self, prompt: str):
                async def gen():
                    async for piece in EchoBackend.generate(self, prompt):
                        yield piece

                return TokenStream(gen(), dict(meta))

        _, consumer = await self._make(_CallingEcho())
        await consumer.generate("hi")
        self.assertEqual(consumer.last_tool_calls, self.CALLS)

    async def test_no_tool_calls_reports_none(self):
        _, consumer = await self._make(self._backend())
        await consumer.chat([{"role": "user", "content": "x"}])
        self.assertIsNone(consumer.last_tool_calls)
        self.assertIsNone(consumer.last_finish_reason)

    async def test_tool_call_trailer_survives_heavy_fragmentation(self):
        # Calls span many trailer frames at TINY payload size and interleave
        # with the compressed response stream without corrupting either.
        calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": '{"query": "' + "x" * 200 + '"}',
                },
            }
        ]
        backend = self._backend(
            reply="fragmented reply " * 5,
            meta={"tool_calls": calls, "finish_reason": "tool_calls"},
        )
        _, consumer = await self._make(backend)
        result = await consumer.chat([{"role": "user", "content": "x"}])
        self.assertEqual(result, "fragmented reply " * 5)
        self.assertEqual(consumer.last_tool_calls, calls)


class PendingRequestTimingsUnitTests(unittest.IsolatedAsyncioTestCase):
    async def test_meta_frames_do_not_leak_into_text(self):
        # Drive the pending request exactly like the host emits a message with
        # a timings trailer: compressed data, plain trailer, END.
        import zlib

        pending = _PendingRequest(7)
        enc = zlib.compressobj()
        data = enc.compress(b"ab") + enc.flush(zlib.Z_SYNC_FLUSH)
        seq = 0
        pending.feed(
            Frame(MessageType.RESPONSE, 7, seq, data, Flags.START | Flags.COMPRESSED)
        )
        seq += 1
        payload = json.dumps({"timings": {"draft_n": 1}}).encode()
        pieces = [payload[i : i + TINY] for i in range(0, len(payload), TINY)]
        for i, piece in enumerate(pieces):
            flags = Flags.END if i == len(pieces) - 1 else Flags.NONE
            pending.feed(Frame(MessageType.TIMINGS, 7, seq, piece, flags))
            seq += 1
        text = ""
        async for chunk in pending:
            text += chunk
        self.assertEqual(text, "ab")  # trailer never leaks into the text stream
        self.assertEqual(pending.timings, {"draft_n": 1})


if __name__ == "__main__":
    unittest.main()
