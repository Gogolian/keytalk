"""End-to-end integration tests: HostService <-> ConsumerClient over loopback.

These drive the whole pipeline (chunking, transport, reassembly, streaming,
errors, concurrency) with a deliberately tiny payload size so that prompts and
responses are forced to span many frames - the regime where a real BLE link
operates and where framing bugs surface.
"""

import asyncio
import unittest
from typing import AsyncIterator

from keytalk.backends import EchoBackend, LLMBackend, StaticBackend
from keytalk.consumer import ConsumerClient, RemoteError
from keytalk.host import HostService
from keytalk.transport import create_loopback

# Tiny payload size to force heavy fragmentation (mirrors BLE's ~20 byte ATT).
TINY = 6


class _FailingBackend(LLMBackend):
    """Yields a couple of tokens then raises, to test error propagation."""

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        yield "partial "
        raise RuntimeError("model exploded")


class _EmptyBackend(LLMBackend):
    async def generate(self, prompt: str) -> AsyncIterator[str]:
        return
        yield  # pragma: no cover - makes this an async generator


class _SlowBackend(LLMBackend):
    """Emits the response slowly so concurrent requests interleave."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        for ch in self._text:
            await asyncio.sleep(0.001)
            yield ch


class IntegrationTestBase(unittest.IsolatedAsyncioTestCase):
    async def _make_pair(self, backend: LLMBackend, payload_size: int = TINY):
        host_t, consumer_t = create_loopback()
        host = HostService(host_t, backend, max_payload_size=payload_size)
        consumer = ConsumerClient(
            consumer_t, max_payload_size=payload_size, timeout=5.0
        )
        await host.start()
        await consumer.start()
        self.addAsyncCleanup(host.close)
        self.addAsyncCleanup(consumer.close)
        return host, consumer


class BasicRoundTripTests(IntegrationTestBase):
    async def test_simple_prompt_response(self):
        _, consumer = await self._make_pair(StaticBackend("Hello, world!", 3))
        result = await consumer.generate("hi")
        self.assertEqual(result, "Hello, world!")

    async def test_echo_backend(self):
        _, consumer = await self._make_pair(EchoBackend(prefix="echo: "))
        result = await consumer.generate("one two three")
        self.assertEqual(result, "echo: one two three ")

    async def test_large_prompt_and_response(self):
        # Response far larger than the payload size -> hundreds of frames.
        big = "X" * 5000
        _, consumer = await self._make_pair(StaticBackend(big, 7))
        result = await consumer.generate("please be verbose " * 50)
        self.assertEqual(result, big)

    async def test_unicode_roundtrip(self):
        text = "héllo 世界 🌍 café"
        _, consumer = await self._make_pair(StaticBackend(text, 2))
        # multibyte chars get split across frames at the byte level; the
        # consumer must reassemble bytes before decoding.
        result = await consumer.generate("greet me")
        self.assertEqual(result, text)

    async def test_empty_response(self):
        _, consumer = await self._make_pair(_EmptyBackend())
        result = await consumer.generate("nothing please")
        self.assertEqual(result, "")

    async def test_empty_prompt(self):
        _, consumer = await self._make_pair(StaticBackend("ok", 2))
        result = await consumer.generate("")
        self.assertEqual(result, "ok")


class StreamingTests(IntegrationTestBase):
    async def test_stream_yields_incrementally(self):
        _, consumer = await self._make_pair(StaticBackend("abcdefghij", 2))
        pieces = [p async for p in consumer.stream("go")]
        self.assertEqual("".join(pieces), "abcdefghij")
        # streamed in more than one piece
        self.assertGreater(len(pieces), 1)


class ErrorHandlingTests(IntegrationTestBase):
    async def test_backend_error_becomes_remote_error(self):
        _, consumer = await self._make_pair(_FailingBackend())
        with self.assertRaises(RemoteError) as ctx:
            await consumer.generate("trigger")
        self.assertIn("model exploded", str(ctx.exception))

    async def test_error_in_streaming(self):
        _, consumer = await self._make_pair(_FailingBackend())
        collected = []
        with self.assertRaises(RemoteError):
            async for piece in consumer.stream("trigger"):
                collected.append(piece)
        # Whatever tokens were flushed before the failure are a prefix of the
        # backend's pre-error output; the rest is dropped and an error is sent.
        partial = "".join(collected)
        self.assertTrue("partial ".startswith(partial))
        self.assertTrue(partial)  # at least one frame made it through


class ConcurrencyTests(IntegrationTestBase):
    async def test_concurrent_requests_are_isolated(self):
        _, consumer = await self._make_pair(_SlowBackend("RESPONSE-BODY"))
        results = await asyncio.gather(
            consumer.generate("a"),
            consumer.generate("b"),
            consumer.generate("c"),
        )
        self.assertEqual(results, ["RESPONSE-BODY"] * 3)

    async def test_many_sequential_requests_reuse_ids(self):
        _, consumer = await self._make_pair(StaticBackend("pong", 2))
        for _ in range(20):
            self.assertEqual(await consumer.generate("ping"), "pong")
        # all requests completed; nothing left pending
        self.assertEqual(len(consumer._pending), 0)


class TimeoutTests(IntegrationTestBase):
    async def test_timeout_when_host_silent(self):
        host_t, consumer_t = create_loopback()
        # No host attached to host_t, so prompts are received by nobody and the
        # consumer never gets a response.
        consumer = ConsumerClient(
            consumer_t, max_payload_size=TINY, timeout=0.1
        )
        await consumer.start()
        self.addAsyncCleanup(consumer.close)
        with self.assertRaises(asyncio.TimeoutError):
            await consumer.generate("hello?")


if __name__ == "__main__":
    unittest.main()
