"""Compression tests: prompt compression and streaming response compression.

Prompts are zlib-compressed as a whole before chunking (the host reassembles and
decompresses); responses are compressed incrementally with a zlib stream so the
consumer can decode tokens as frames arrive.  ERROR frames are plain text: the
host flushes the compressed stream before switching types.
"""

import unittest
import zlib

from keytalk.backends import LLMBackend, StaticBackend
from keytalk.consumer import ConsumerClient, RemoteError
from keytalk.host import HostService
from keytalk.protocol import (
    Flags,
    Frame,
    FrameStreamEncoder,
    MessageType,
    Reassembler,
    chunk_message,
)
from keytalk.transport import create_loopback

TINY = 6


class _FailingBackend(LLMBackend):
    """Yields some output then raises, like a model blowing its context."""

    async def generate(self, prompt: str):
        yield "partial "
        raise RuntimeError("model exploded")


class _CountingBackend(LLMBackend):
    def __init__(self, text: str, piece: int = 2) -> None:
        self._text = text
        self._piece = piece
        self.prompts = []

    async def generate(self, prompt: str):
        self.prompts.append(prompt)
        for i in range(0, len(self._text), self._piece):
            yield self._text[i : i + self._piece]


class PromptCompressionTests(unittest.IsolatedAsyncioTestCase):
    async def _roundtrip(self, prompt: str, *, compress: bool) -> str:
        backend = _CountingBackend("ok")
        host_t, consumer_t = create_loopback()
        host = HostService(host_t, backend, max_payload_size=TINY)
        consumer = ConsumerClient(
            consumer_t,
            max_payload_size=TINY,
            timeout=5.0,
            compress_prompts=compress,
            keepalive_interval=0,
        )
        await host.start()
        await consumer.start()
        try:
            result = await consumer.generate(prompt)
            self.assertEqual(result, "ok")
            # The prompt must arrive at the backend byte-for-byte intact.
            self.assertEqual(backend.prompts, [prompt])
        finally:
            await consumer.close()
            await host.close()
        return result

    async def test_compressed_prompt_roundtrip(self):
        await self._roundtrip("this is a test prompt. " * 50, compress=True)

    async def test_uncompressed_prompt_roundtrip(self):
        await self._roundtrip("this is a test prompt. " * 50, compress=False)

    async def test_compression_reduces_frame_count(self):
        payload = ("this is a test prompt. " * 100).encode("utf-8")
        max_payload = 13  # the 23-byte ATT MTU default
        plain = chunk_message(MessageType.PROMPT, 1, payload, max_payload)
        compressed = chunk_message(
            MessageType.PROMPT, 1, zlib.compress(payload), max_payload
        )
        self.assertLess(len(compressed), len(plain) // 4)

    async def test_small_prompt_is_not_compressed(self):
        # Compression must be skipped when it does not save space.
        backend = _CountingBackend("ok")
        host_t, consumer_t = create_loopback()
        host = HostService(host_t, backend, max_payload_size=64)
        consumer = ConsumerClient(
            consumer_t, max_payload_size=64, timeout=5.0, keepalive_interval=0
        )
        await host.start()
        await consumer.start()
        try:
            await consumer.generate("hi")
            sent = [Frame.decode(f) for f in consumer_t.sent]
            prompts = [f for f in sent if f.msg_type == MessageType.PROMPT]
            self.assertTrue(prompts)
            self.assertFalse(prompts[0].flags & Flags.COMPRESSED)
        finally:
            await consumer.close()
            await host.close()


class ResponseCompressionTests(unittest.IsolatedAsyncioTestCase):
    async def _make_pair(self, backend, payload_size=TINY, **host_kw):
        host_t, consumer_t = create_loopback()
        host = HostService(
            host_t, backend, max_payload_size=payload_size, **host_kw
        )
        consumer = ConsumerClient(
            consumer_t, max_payload_size=payload_size, timeout=5.0, keepalive_interval=0
        )
        await host.start()
        await consumer.start()
        self.addAsyncCleanup(consumer.close)
        self.addAsyncCleanup(host.close)
        return host, consumer

    async def test_compressed_response_roundtrip(self):
        text = "the quick brown fox jumps over the lazy dog " * 10
        _, consumer = await self._make_pair(StaticBackend(text, 3))
        self.assertEqual(await consumer.generate("go"), text)

    async def test_compressed_response_unicode(self):
        text = "héllo 世界 🌍 café " * 20
        _, consumer = await self._make_pair(StaticBackend(text, 2))
        self.assertEqual(await consumer.generate("go"), text)

    async def test_response_frames_are_actually_compressed(self):
        text = "aaaa bbbb cccc dddd " * 20
        host, consumer = await self._make_pair(StaticBackend(text, 2))
        await consumer.generate("go")
        frames = [Frame.decode(f) for f in host._transport.sent]
        responses = [f for f in frames if f.msg_type == MessageType.RESPONSE]
        self.assertTrue(responses[0].flags & Flags.COMPRESSED)
        wire_bytes = sum(len(f.payload) for f in responses)
        self.assertLess(wire_bytes, len(text.encode()) // 2)

    async def test_uncompressed_response_still_works(self):
        text = "plain responses keep working"
        _, consumer = await self._make_pair(
            StaticBackend(text, 2), compress_responses=False
        )
        self.assertEqual(await consumer.generate("go"), text)

    async def test_mid_stream_error_after_compressed_data(self):
        _, consumer = await self._make_pair(_FailingBackend())
        collected = []
        with self.assertRaises(RemoteError) as ctx:
            async for piece in consumer.stream("trigger"):
                collected.append(piece)
        self.assertIn("model exploded", str(ctx.exception))
        self.assertEqual("".join(collected), "partial ")


class EncoderCompressionUnitTests(unittest.TestCase):
    def test_encoder_roundtrip_via_reassembler(self):
        text = ("compress me, I am very repetitive text. " * 30).encode()
        enc = FrameStreamEncoder(MessageType.RESPONSE, 5, 13, compressed=True)
        frames = enc.push(text[:40]) + enc.push(text[40:]) + enc.finish()
        self.assertTrue(frames[0].flags & Flags.COMPRESSED)
        reassembler = Reassembler()
        message = None
        for frame in frames:
            message = reassembler.feed(frame) or message
        self.assertIsNotNone(message)
        self.assertEqual(message.payload, text)

    def test_flush_cut_is_clean(self):
        # flush() must emit every pending byte without ending the stream, so
        # nothing is lost when the sender switches to error frames.
        enc = FrameStreamEncoder(MessageType.RESPONSE, 5, 64, compressed=True)
        enc.push(b"data")
        flushed = enc.flush()
        enc.push(b"more")
        tail = enc.finish()
        dec = zlib.decompressobj()
        out = b"".join(dec.decompress(f.payload) for f in flushed)
        self.assertEqual(out, b"data")
        out2 = b"".join(dec.decompress(f.payload) for f in tail)
        out2 += dec.flush()
        self.assertEqual(out2, b"more")

    def test_finish_splits_oversized_tail(self):
        enc = FrameStreamEncoder(MessageType.RESPONSE, 5, 4, compressed=True)
        enc.push(b"some compressible text " * 3)
        frames = enc.finish()
        self.assertGreater(len(frames), 1)
        self.assertTrue(frames[-1].is_end)
        self.assertFalse(any(f.is_end for f in frames[:-1]))

    def test_flush_on_empty_encoder_emits_nothing(self):
        enc = FrameStreamEncoder(MessageType.RESPONSE, 5, 64, compressed=True)
        self.assertEqual(enc.flush(), [])
        self.assertFalse(enc.has_started)


if __name__ == "__main__":
    unittest.main()
