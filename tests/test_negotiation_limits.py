"""Regression tests for frame-size negotiation limits and buffered-mode parity.

These cover behaviour that previously went wrong and was easy to reintroduce:

* a consumer's ``max_mtu`` must be honoured on every negotiated transfer mode,
  not just on the plain BLE path;
* the host must clamp a peer's self-reported MTU instead of sizing frames to
  whatever the peer claims;
* a duplicated request must produce exactly one generation, in both the
  streaming and the buffered path;
* buffered mode must keep the same guarantees as streaming mode: frame
  retention for RESUME, the TIMINGS trailer, and backend error reporting.
"""

from __future__ import annotations

import asyncio
import unittest

from keytalk.backends import LLMBackend, StaticBackend, TokenStream
from keytalk.consumer import ConsumerClient
from keytalk.host import DEFAULT_MAX_MTU, HostService
from keytalk.modes import make_l2cap_coc_profile
from keytalk.protocol import (
    Flags,
    Frame,
    MessageType,
    chunk_message,
    encode_select_payload,
    max_payload_for_mtu,
)
from keytalk.transport import InMemoryTransport, create_loopback

LARGE_MTU = 4096


def _select(mode_id: int, mtu: int) -> bytes:
    return Frame(
        msg_type=MessageType.SELECT,
        message_id=0,
        seq=0,
        payload=encode_select_payload(mode_id, mtu),
        flags=Flags.START | Flags.END,
    ).encode()


class _WriteCounter(LLMBackend):
    """Backend that records how many times a prompt was actually generated."""

    def __init__(self, text: str = "answer", delay: float = 0.05) -> None:
        self.calls = 0
        self._text = text
        self._delay = delay

    async def generate(self, prompt: str):
        self.calls += 1
        await asyncio.sleep(self._delay)
        yield self._text


class _MetaBackend(LLMBackend):
    """Backend reporting timings/tool_calls the way llama.cpp does."""

    def __init__(self, meta: dict) -> None:
        self._meta = meta

    async def _run(self):
        for word in ("alpha ", "beta ", "gamma "):
            await asyncio.sleep(0)
            yield word

    def generate(self, prompt: str) -> TokenStream:
        return TokenStream(self._run(), self._meta)


class _BoomBackend(LLMBackend):
    async def generate(self, prompt: str):
        yield "partial"
        raise RuntimeError("backend exploded")


class _ReportedMTU(InMemoryTransport):
    """Consumer-side loopback that advertises a large link MTU."""

    @property
    def mtu_size(self) -> int:
        return LARGE_MTU


class ConsumerMTUCapTests(unittest.IsolatedAsyncioTestCase):
    """``max_mtu`` must cap the frame size on every negotiated mode."""

    async def _consumer(self, caps, *, max_mtu: int) -> ConsumerClient:
        host_t = InMemoryTransport("host")
        consumer_t = _ReportedMTU("consumer", caps=caps)
        host_t.link(consumer_t)
        consumer_t.link(host_t)
        client = ConsumerClient(consumer_t, max_mtu=max_mtu)
        await client.start()
        return client

    async def test_negotiated_mode_honours_max_mtu(self) -> None:
        client = await self._consumer(["legacy", "fast_gatt"], max_mtu=512)
        try:
            self.assertEqual(client._profile.mode.value, "fast_gatt")
            self.assertLessEqual(
                client._max_payload, max_payload_for_mtu(512)
            )
        finally:
            await client.close()

    async def test_explicit_payload_size_is_not_overridden(self) -> None:
        host_t = InMemoryTransport("host")
        consumer_t = _ReportedMTU("consumer", caps=["legacy", "fast_gatt"])
        host_t.link(consumer_t)
        consumer_t.link(host_t)
        client = ConsumerClient(consumer_t, max_payload_size=40)
        await client.start()
        try:
            self.assertEqual(client._max_payload, 40)
        finally:
            await client.close()


class HostMTUClampTests(unittest.IsolatedAsyncioTestCase):
    """The host must never size frames beyond its configured ceiling."""

    async def _host(self, **kwargs) -> HostService:
        host_t, consumer_t = create_loopback()
        host = HostService(host_t, StaticBackend("x"), **kwargs)
        await host.start()
        await consumer_t.start()
        return host, consumer_t

    async def test_select_mtu_is_clamped_to_max_mtu(self) -> None:
        host, consumer_t = await self._host()
        try:
            await consumer_t.send(_select(1, 65535))
            await asyncio.sleep(0.05)
            self.assertLessEqual(host._max_payload, max_payload_for_mtu(DEFAULT_MAX_MTU))
            self.assertLessEqual(
                host._max_payload_cap, max_payload_for_mtu(DEFAULT_MAX_MTU)
            )
        finally:
            await host.close()
            await consumer_t.close()

    async def test_explicit_max_mtu_is_respected(self) -> None:
        host, consumer_t = await self._host(max_mtu=300)
        try:
            self.assertEqual(host.max_mtu, 300)
            await consumer_t.send(_select(1, 65535))
            await asyncio.sleep(0.05)
            self.assertLessEqual(host._max_payload, max_payload_for_mtu(300))
        finally:
            await host.close()
            await consumer_t.close()

    async def test_hello_cannot_exceed_ceiling(self) -> None:
        host, consumer_t = await self._host()
        try:
            from keytalk.protocol import encode_max_payload

            for frame in chunk_message(
                MessageType.HELLO, 0, encode_max_payload(60000), 13
            ):
                await consumer_t.send(frame.encode())
            await asyncio.sleep(0.05)
            self.assertLessEqual(host._max_payload, max_payload_for_mtu(DEFAULT_MAX_MTU))
        finally:
            await host.close()
            await consumer_t.close()

    async def test_low_mtu_select_still_shrinks_frames(self) -> None:
        host, consumer_t = await self._host()
        try:
            await consumer_t.send(_select(1, 185))
            await asyncio.sleep(0.05)
            self.assertEqual(host._max_payload, max_payload_for_mtu(185))
        finally:
            await host.close()
            await consumer_t.close()


class DuplicateRequestTests(unittest.IsolatedAsyncioTestCase):
    """A retried write must not start a second generation."""

    async def _replay_prompt(self, host_t, consumer_t, message_id: int, times: int):
        for _ in range(times):
            for frame in chunk_message(
                MessageType.PROMPT, message_id, b"hello", 100
            ):
                await consumer_t.send(frame.encode())

    async def _run(self, *, buffered: bool) -> int:
        host_t, consumer_t = create_loopback()
        backend = _WriteCounter()
        profile = make_l2cap_coc_profile(1024) if buffered else None
        host = HostService(host_t, backend, profile=profile, buffer_response=buffered)
        await host.start()
        await consumer_t.start()
        try:
            await self._replay_prompt(host_t, consumer_t, 5, times=3)
            await asyncio.sleep(0.3)
            return backend.calls
        finally:
            await host.close()
            await consumer_t.close()

    async def test_streaming_path_ignores_duplicates(self) -> None:
        self.assertEqual(await self._run(buffered=False), 1)

    async def test_buffered_path_ignores_duplicates(self) -> None:
        self.assertEqual(await self._run(buffered=True), 1)


class BufferedModeParityTests(unittest.IsolatedAsyncioTestCase):
    """Buffered mode must keep every guarantee the streaming path has."""

    META = {
        "timings": {"draft_n": 4, "draft_n_accepted": 3},
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q": "x"}'},
            }
        ],
        "finish_reason": "tool_calls",
    }

    async def _pair(self, *, buffered: bool):
        host_t, consumer_t = create_loopback()
        profile = make_l2cap_coc_profile(1024) if buffered else None
        host = HostService(
            host_t, _MetaBackend(dict(self.META)),
            profile=profile, buffer_response=buffered,
        )
        await host.start()
        client = ConsumerClient(consumer_t)
        await client.start()
        return host, client

    async def _stream_once(self, *, buffered: bool):
        host, client = await self._pair(buffered=buffered)
        try:
            stream = client.stream("hi")
            text = "".join([piece async for piece in stream])
            return text, stream.timings, stream.tool_calls, stream.finish_reason
        finally:
            await client.close()
            await host.close()

    async def test_buffered_text_roundtrips_intact(self) -> None:
        text, _, _, _ = await self._stream_once(buffered=True)
        self.assertEqual(text, "alpha beta gamma ")

    async def test_buffered_path_delivers_trailer(self) -> None:
        _, timings, calls, reason = await self._stream_once(buffered=True)
        self.assertEqual(timings, self.META["timings"])
        self.assertEqual(calls, self.META["tool_calls"])
        self.assertEqual(reason, "tool_calls")

    async def test_streaming_and_buffered_agree(self) -> None:
        streamed = await self._stream_once(buffered=False)
        buffered = await self._stream_once(buffered=True)
        self.assertEqual(streamed, buffered)

    async def test_buffered_retains_frames_for_resume(self) -> None:
        host, client = await self._pair(buffered=True)
        try:
            stream = client.stream("hi")
            async for _ in stream:
                pass
            # The message is still tracked after completion, so a late RESUME
            # replays from the retained store instead of a restart.
            self.assertTrue(host._messages)
            state = next(iter(host._messages.values()))
            self.assertTrue(state.delivered)
        finally:
            await client.close()
            await host.close()

    async def test_buffered_backend_error_reaches_consumer(self) -> None:
        from keytalk.consumer import RemoteError

        host_t, consumer_t = create_loopback()
        host = HostService(host_t, _BoomBackend(), buffer_response=True)
        await host.start()
        client = ConsumerClient(consumer_t, timeout=2.0, retries=0)
        await client.start()
        try:
            with self.assertRaises(RemoteError) as ctx:
                await client.generate("hi")
            self.assertIn("backend exploded", str(ctx.exception))
        finally:
            await client.close()
            await host.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
