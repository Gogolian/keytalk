"""Phase-3 tests: L2CAP_COC mode — stream framing, end-to-end data, throughput.

All tests use :func:`create_l2cap_loopback` (a socketpair-backed stream pair)
so no Bluetooth hardware is required.  The Go-Back-N layer is absent: frames
are sent directly over the reliable stream.
"""

from __future__ import annotations

import asyncio
import struct
import time
import unittest

from keytalk.backends import StaticBackend
from keytalk.ble.l2cap import L2CAPLoopbackTransport, create_l2cap_loopback
from keytalk.ble.l2cap.channel import L2CAPStreamTransport, _LEN_SIZE, _LEN_STRUCT
from keytalk.consumer import ConsumerClient
from keytalk.host import HostService
from keytalk.modes import (
    L2CAP_COC_PROFILE,
    LEGACY_PROFILE,
    Mode,
    make_l2cap_coc_profile,
    profile_for_mode,
)
from keytalk.protocol import (
    DEFAULT_ATT_MTU,
    Flags,
    Frame,
    MessageType,
    max_payload_for_mtu,
)
from keytalk.transport import InMemoryTransport, create_loopback

# ── constants ─────────────────────────────────────────────────────────────────

_L2CAP_MTU = 1024  # default SDU payload bytes for these tests


# ── helpers ───────────────────────────────────────────────────────────────────

class _L2CAPPair:
    """Linked L2CAP loopback transports started together."""

    def __init__(
        self,
        host_t: L2CAPLoopbackTransport,
        consumer_t: L2CAPLoopbackTransport,
    ) -> None:
        self.host_t = host_t
        self.consumer_t = consumer_t


async def _make_pair(response: str) -> tuple[HostService, ConsumerClient, _L2CAPPair]:
    host_t, consumer_t = await create_l2cap_loopback()
    pair = _L2CAPPair(host_t, consumer_t)
    profile = make_l2cap_coc_profile(_L2CAP_MTU)
    host = HostService(host_t, StaticBackend(response), profile=profile)
    consumer = ConsumerClient(consumer_t, profile=profile, timeout=5.0)
    await host.start()
    await consumer.start()
    return host, consumer, pair


# ── framing unit tests ────────────────────────────────────────────────────────

class L2CAPFramingTests(unittest.IsolatedAsyncioTestCase):
    """Verify the 4-byte length-prefix framing layer in isolation."""

    async def test_single_frame_roundtrip(self):
        host_t, consumer_t = await create_l2cap_loopback()
        received: list[bytes] = []

        async def _on_recv(data: bytes) -> None:
            received.append(data)

        consumer_t.on_receive(_on_recv)
        await host_t.start()
        await consumer_t.start()

        payload = b"hello l2cap"
        await host_t.send(payload)
        await asyncio.sleep(0.05)
        self.assertEqual(received, [payload])

        await host_t.close()
        await consumer_t.close()

    async def test_multiple_frames_ordered(self):
        host_t, consumer_t = await create_l2cap_loopback()
        received: list[bytes] = []

        async def _on_recv(data: bytes) -> None:
            received.append(data)

        consumer_t.on_receive(_on_recv)
        await host_t.start()
        await consumer_t.start()

        payloads = [f"frame-{i}".encode() for i in range(10)]
        for p in payloads:
            await host_t.send(p)
        await asyncio.sleep(0.1)
        self.assertEqual(received, payloads)

        await host_t.close()
        await consumer_t.close()

    async def test_large_frame(self):
        host_t, consumer_t = await create_l2cap_loopback()
        received: list[bytes] = []

        async def _on_recv(data: bytes) -> None:
            received.append(data)

        consumer_t.on_receive(_on_recv)
        await host_t.start()
        await consumer_t.start()

        payload = b"x" * 65_000
        await host_t.send(payload)
        await asyncio.sleep(0.1)
        self.assertEqual(received, [payload])

        await host_t.close()
        await consumer_t.close()

    async def test_bidirectional(self):
        host_t, consumer_t = await create_l2cap_loopback()
        from_host: list[bytes] = []
        from_consumer: list[bytes] = []

        consumer_t.on_receive(lambda d: from_consumer.append(d) or asyncio.sleep(0))
        host_t.on_receive(lambda d: from_host.append(d) or asyncio.sleep(0))

        await host_t.start()
        await consumer_t.start()

        await host_t.send(b"host->consumer")
        await consumer_t.send(b"consumer->host")
        await asyncio.sleep(0.05)

        self.assertIn(b"host->consumer", from_consumer)
        self.assertIn(b"consumer->host", from_host)

        await host_t.close()
        await consumer_t.close()


# ── profile / modes tests ─────────────────────────────────────────────────────

class L2CAPCOCProfileTests(unittest.TestCase):
    def test_mode_value(self):
        self.assertEqual(L2CAP_COC_PROFILE.mode, Mode.L2CAP_COC)

    def test_no_write_with_response(self):
        self.assertFalse(L2CAP_COC_PROFILE.write_with_response)

    def test_flow_control_l2cap_credits(self):
        self.assertEqual(L2CAP_COC_PROFILE.flow_control, "l2cap_credits")

    def test_reliability_window_zero(self):
        self.assertEqual(L2CAP_COC_PROFILE.reliability_window, 0)

    def test_compression_codec_zlib(self):
        self.assertEqual(L2CAP_COC_PROFILE.compression_codec, "zlib")

    def test_make_l2cap_coc_profile_custom_mtu(self):
        p = make_l2cap_coc_profile(2048)
        self.assertEqual(p.mtu, 2048)
        self.assertEqual(p.max_payload_size, max_payload_for_mtu(2048))

    def test_profile_for_mode_returns_l2cap_coc(self):
        p = profile_for_mode("l2cap_coc")
        self.assertEqual(p.mode, Mode.L2CAP_COC)

    def test_negotiate_auto_picks_l2cap_coc(self):
        from keytalk.modes import negotiate_mode
        result = negotiate_mode(["legacy", "fast_gatt", "l2cap_coc"], "auto")
        # l2cap_coc is in _MODE_PRIORITY above fast_gatt
        self.assertEqual(result.mode, Mode.L2CAP_COC)

    def test_negotiate_explicit_l2cap_coc(self):
        from keytalk.modes import negotiate_mode
        result = negotiate_mode(["legacy", "l2cap_coc"], "l2cap_coc")
        self.assertEqual(result.mode, Mode.L2CAP_COC)


# ── end-to-end integration tests ──────────────────────────────────────────────

class L2CAPCOCIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def _make_pair(self, response: str):
        host, consumer, pair = await _make_pair(response)
        self.addAsyncCleanup(host.close)
        self.addAsyncCleanup(consumer.close)
        return host, consumer, pair

    async def test_simple_response(self):
        response = "hello from l2cap"
        _, consumer, _ = await self._make_pair(response)
        result = await consumer.generate("hi")
        self.assertEqual(result, response)

    async def test_large_response(self):
        response = "z" * 8192
        _, consumer, _ = await self._make_pair(response)
        result = await consumer.generate("big")
        self.assertEqual(result, response)

    async def test_host_profile_set_to_l2cap_coc(self):
        host, consumer, _ = await self._make_pair("ok")
        await consumer.generate("test")
        self.assertEqual(host._profile.mode, Mode.L2CAP_COC)

    async def test_consumer_profile_set_to_l2cap_coc(self):
        _, consumer, _ = await self._make_pair("ok")
        self.assertEqual(consumer._profile.mode, Mode.L2CAP_COC)

    async def test_response_correct_after_multiple_requests(self):
        responses = ["first", "second", "third"]
        host_t, consumer_t = await create_l2cap_loopback()
        profile = make_l2cap_coc_profile(_L2CAP_MTU)

        # Use a backend that cycles through responses
        from keytalk.backends import StaticBackend
        backend = StaticBackend("cycling")
        host = HostService(host_t, backend, profile=profile)
        consumer = ConsumerClient(consumer_t, profile=profile, timeout=5.0)
        await host.start()
        await consumer.start()
        self.addAsyncCleanup(host.close)
        self.addAsyncCleanup(consumer.close)

        for _ in range(3):
            result = await consumer.generate("prompt")
            self.assertEqual(result, "cycling")

    async def test_l2cap_coc_fewer_frames_than_legacy(self):
        """L2CAP_COC with a 1 KiB MTU sends fewer frames than LEGACY (23-byte MTU)."""
        response = "w" * 4096

        # Legacy pair (no CAPS → auto → legacy).
        host_t_l, consumer_t_l = create_loopback()
        host_l = HostService(host_t_l, StaticBackend(response))
        consumer_l = ConsumerClient(consumer_t_l, requested_mode="auto", timeout=5.0)
        await host_l.start()
        await consumer_l.start()
        await consumer_l.generate("prompt")
        legacy_frames = sum(
            1 for raw in host_t_l.sent
            if Frame.decode(raw).msg_type == MessageType.RESPONSE
        )
        await host_l.close()
        await consumer_l.close()

        # L2CAP_COC pair — count keytalk frames (the length-prefix wrapper is
        # transparent to the Frame layer).
        host_t, consumer_t = await create_l2cap_loopback()
        profile = make_l2cap_coc_profile(_L2CAP_MTU)

        # Capture raw stream bytes to count frames: read after the run.
        host = HostService(host_t, StaticBackend(response), profile=profile)
        consumer = ConsumerClient(consumer_t, profile=profile, timeout=5.0)
        await host.start()
        await consumer.start()

        # Count frames by hooking the host transport's send path.
        sent_frames: list[bytes] = []
        _orig_send = host_t.send

        async def _counting_send(frame: bytes) -> None:
            sent_frames.append(frame)
            await _orig_send(frame)

        host_t.send = _counting_send  # type: ignore[method-assign]
        await consumer.generate("prompt")
        l2cap_frames = sum(
            1 for raw in sent_frames
            if Frame.decode(raw).msg_type == MessageType.RESPONSE
        )
        await host.close()
        await consumer.close()

        self.assertLess(
            l2cap_frames, legacy_frames,
            f"L2CAP_COC ({l2cap_frames}) should use fewer frames than LEGACY ({legacy_frames})",
        )


# ── throughput benchmark ──────────────────────────────────────────────────────

class L2CAPCOCThroughputBenchmark(unittest.IsolatedAsyncioTestCase):
    """Compare LEGACY vs FAST_GATT vs L2CAP_COC throughput over in-memory transports."""

    _PAYLOAD_SIZE = 8192

    async def _run_legacy(self, repeats: int = 3) -> tuple[float, int]:
        response = "a" * self._PAYLOAD_SIZE
        total_bytes = 0
        total_frames = 0
        elapsed = 0.0
        for _ in range(repeats):
            host_t, consumer_t = create_loopback()
            host = HostService(host_t, StaticBackend(response))
            consumer = ConsumerClient(consumer_t, requested_mode="auto", timeout=10.0)
            await host.start()
            await consumer.start()
            t0 = time.perf_counter()
            result = await consumer.generate("bench")
            elapsed += time.perf_counter() - t0
            total_bytes += len(result.encode())
            total_frames += sum(
                1 for raw in host_t.sent
                if Frame.decode(raw).msg_type == MessageType.RESPONSE
            )
            await host.close()
            await consumer.close()
        bps = total_bytes / elapsed
        return bps, total_frames // repeats

    async def _run_l2cap(self, mtu: int, repeats: int = 3) -> tuple[float, int]:
        response = "a" * self._PAYLOAD_SIZE
        total_bytes = 0
        total_frames = 0
        elapsed = 0.0
        for _ in range(repeats):
            host_t, consumer_t = await create_l2cap_loopback()
            profile = make_l2cap_coc_profile(mtu)
            sent: list[bytes] = []
            _orig = host_t.send

            async def _counting_send(frame: bytes, _orig=_orig, _sent=sent) -> None:
                _sent.append(frame)
                await _orig(frame)

            host_t.send = _counting_send  # type: ignore[method-assign]
            host = HostService(host_t, StaticBackend(response), profile=profile)
            consumer = ConsumerClient(consumer_t, profile=profile, timeout=10.0)
            await host.start()
            await consumer.start()
            t0 = time.perf_counter()
            result = await consumer.generate("bench")
            elapsed += time.perf_counter() - t0
            total_bytes += len(result.encode())
            total_frames += sum(
                1 for raw in sent
                if Frame.decode(raw).msg_type == MessageType.RESPONSE
            )
            await host.close()
            await consumer.close()
        bps = total_bytes / elapsed
        return bps, total_frames // repeats

    async def test_benchmark(self):
        legacy_bps, legacy_frames = await self._run_legacy()
        l2cap_bps, l2cap_frames = await self._run_l2cap(_L2CAP_MTU)

        print(
            f"\n--- L2CAP_COC Throughput Benchmark ({self._PAYLOAD_SIZE} B payload) ---\n"
            f"  LEGACY   : {legacy_bps:>10,.0f} B/s  ({legacy_frames} frames/msg)\n"
            f"  L2CAP_COC: {l2cap_bps:>10,.0f} B/s  ({l2cap_frames} frames/msg)\n"
            f"  Frames saved: {legacy_frames - l2cap_frames} "
            f"({legacy_frames / max(l2cap_frames, 1):.1f}x fewer)\n"
        )

        # L2CAP_COC with a large MTU must use fewer frames than legacy.
        self.assertLess(l2cap_frames, legacy_frames)
        self.assertGreater(legacy_bps, 0)
        self.assertGreater(l2cap_bps, 0)
