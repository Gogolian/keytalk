"""Phase-4 tests: CLASSIC_RFCOMM mode — stream framing, end-to-end data, throughput.

All tests use :func:`create_rfcomm_loopback` (a socketpair-backed stream pair)
so no Bluetooth hardware is required.  The Go-Back-N layer is absent: frames
are sent directly over the reliable stream, identical to the L2CAP_COC path.
"""

from __future__ import annotations

import asyncio
import time
import unittest

from keytalk.backends import StaticBackend
from keytalk.classic import RFCOMMLoopbackTransport, create_rfcomm_loopback
from keytalk.classic.channel import RFCOMMStreamTransport
from keytalk.consumer import ConsumerClient
from keytalk.host import HostService
from keytalk.modes import (
    CLASSIC_RFCOMM_PROFILE,
    LEGACY_PROFILE,
    Mode,
    make_classic_rfcomm_profile,
    profile_for_mode,
)
from keytalk.protocol import (
    DEFAULT_ATT_MTU,
    Frame,
    MessageType,
    max_payload_for_mtu,
)
from keytalk.transport import create_loopback

# ── constants ─────────────────────────────────────────────────────────────────

_RFCOMM_MTU = 1024  # default SDU payload bytes for these tests


# ── helpers ───────────────────────────────────────────────────────────────────

async def _make_pair(response: str):
    host_t, consumer_t = await create_rfcomm_loopback()
    profile = make_classic_rfcomm_profile(_RFCOMM_MTU)
    host = HostService(host_t, StaticBackend(response), profile=profile)
    consumer = ConsumerClient(consumer_t, profile=profile, timeout=5.0)
    await host.start()
    await consumer.start()
    return host, consumer


# ── framing unit tests ────────────────────────────────────────────────────────

class RFCOMMFramingTests(unittest.IsolatedAsyncioTestCase):
    """Verify the 4-byte length-prefix framing layer in isolation."""

    async def test_single_frame_roundtrip(self):
        host_t, consumer_t = await create_rfcomm_loopback()
        received: list[bytes] = []

        async def _on_recv(data: bytes) -> None:
            received.append(data)

        consumer_t.on_receive(_on_recv)
        await host_t.start()
        await consumer_t.start()

        payload = b"hello rfcomm"
        await host_t.send(payload)
        await asyncio.sleep(0.05)
        self.assertEqual(received, [payload])

        await host_t.close()
        await consumer_t.close()

    async def test_multiple_frames_ordered(self):
        host_t, consumer_t = await create_rfcomm_loopback()
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
        host_t, consumer_t = await create_rfcomm_loopback()
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
        host_t, consumer_t = await create_rfcomm_loopback()
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

    async def test_empty_frame(self):
        host_t, consumer_t = await create_rfcomm_loopback()
        received: list[bytes] = []

        consumer_t.on_receive(lambda d: received.append(d) or asyncio.sleep(0))
        await host_t.start()
        await consumer_t.start()

        await host_t.send(b"")
        await asyncio.sleep(0.05)
        self.assertEqual(received, [b""])

        await host_t.close()
        await consumer_t.close()


# ── class hierarchy tests ─────────────────────────────────────────────────────

class RFCOMMClassHierarchyTests(unittest.TestCase):
    def test_loopback_is_rfcomm_stream_transport(self):
        # RFCOMMLoopbackTransport must be a subtype of RFCOMMStreamTransport
        self.assertTrue(issubclass(RFCOMMLoopbackTransport, RFCOMMStreamTransport))

    def test_spp_uuid_format(self):
        from keytalk.classic.channel import SPP_UUID
        # Standard SPP UUID: 8-4-4-4-12 hex groups
        parts = SPP_UUID.split("-")
        self.assertEqual(len(parts), 5)
        self.assertEqual(SPP_UUID.upper(), "00001101-0000-1000-8000-00805F9B34FB")


# ── profile / modes tests ─────────────────────────────────────────────────────

class ClassicRFCOMMProfileTests(unittest.TestCase):
    def test_mode_value(self):
        self.assertEqual(CLASSIC_RFCOMM_PROFILE.mode, Mode.CLASSIC_RFCOMM)

    def test_no_write_with_response(self):
        self.assertFalse(CLASSIC_RFCOMM_PROFILE.write_with_response)

    def test_flow_control_rfcomm_stream(self):
        self.assertEqual(CLASSIC_RFCOMM_PROFILE.flow_control, "rfcomm_stream")

    def test_reliability_window_zero(self):
        self.assertEqual(CLASSIC_RFCOMM_PROFILE.reliability_window, 0)

    def test_compression_codec_zlib(self):
        self.assertEqual(CLASSIC_RFCOMM_PROFILE.compression_codec, "zlib")

    def test_make_classic_rfcomm_profile_custom_mtu(self):
        p = make_classic_rfcomm_profile(2048)
        self.assertEqual(p.mtu, 2048)
        self.assertEqual(p.max_payload_size, max_payload_for_mtu(2048))

    def test_profile_for_mode_returns_rfcomm(self):
        p = profile_for_mode("rfcomm")
        self.assertEqual(p.mode, Mode.CLASSIC_RFCOMM)

    def test_negotiate_auto_picks_rfcomm_first(self):
        from keytalk.modes import negotiate_mode
        # rfcomm has the highest priority in _MODE_PRIORITY
        result = negotiate_mode(["legacy", "fast_gatt", "l2cap_coc", "rfcomm"], "auto")
        self.assertEqual(result.mode, Mode.CLASSIC_RFCOMM)

    def test_negotiate_explicit_rfcomm(self):
        from keytalk.modes import negotiate_mode
        result = negotiate_mode(["legacy", "rfcomm"], "rfcomm")
        self.assertEqual(result.mode, Mode.CLASSIC_RFCOMM)

    def test_mode_id_is_3(self):
        from keytalk.modes import mode_id_for
        self.assertEqual(mode_id_for(Mode.CLASSIC_RFCOMM), 3)

    def test_mode_for_id_3_is_rfcomm(self):
        from keytalk.modes import mode_for_id
        self.assertEqual(mode_for_id(3), Mode.CLASSIC_RFCOMM)


# ── end-to-end integration tests ──────────────────────────────────────────────

class ClassicRFCOMMIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def _make_pair(self, response: str):
        host, consumer = await _make_pair(response)
        self.addAsyncCleanup(host.close)
        self.addAsyncCleanup(consumer.close)
        return host, consumer

    async def test_simple_response(self):
        response = "hello from rfcomm"
        _, consumer = await self._make_pair(response)
        result = await consumer.generate("hi")
        self.assertEqual(result, response)

    async def test_large_response(self):
        response = "z" * 8192
        _, consumer = await self._make_pair(response)
        result = await consumer.generate("big")
        self.assertEqual(result, response)

    async def test_host_profile_set_to_rfcomm(self):
        host, consumer = await self._make_pair("ok")
        await consumer.generate("test")
        self.assertEqual(host._profile.mode, Mode.CLASSIC_RFCOMM)

    async def test_consumer_profile_set_to_rfcomm(self):
        _, consumer = await self._make_pair("ok")
        self.assertEqual(consumer._profile.mode, Mode.CLASSIC_RFCOMM)

    async def test_multiple_sequential_requests(self):
        host_t, consumer_t = await create_rfcomm_loopback()
        profile = make_classic_rfcomm_profile(_RFCOMM_MTU)
        host = HostService(host_t, StaticBackend("fixed"), profile=profile)
        consumer = ConsumerClient(consumer_t, profile=profile, timeout=5.0)
        await host.start()
        await consumer.start()
        self.addAsyncCleanup(host.close)
        self.addAsyncCleanup(consumer.close)

        for _ in range(3):
            result = await consumer.generate("prompt")
            self.assertEqual(result, "fixed")

    async def test_rfcomm_fewer_frames_than_legacy(self):
        """RFCOMM with a 1 KiB MTU sends fewer frames than LEGACY (23-byte MTU)."""
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

        # RFCOMM pair — count keytalk frames via a send hook.
        host_t, consumer_t = await create_rfcomm_loopback()
        profile = make_classic_rfcomm_profile(_RFCOMM_MTU)
        sent_frames: list[bytes] = []
        _orig_send = host_t.send

        async def _counting_send(frame: bytes) -> None:
            sent_frames.append(frame)
            await _orig_send(frame)

        host_t.send = _counting_send  # type: ignore[method-assign]
        host = HostService(host_t, StaticBackend(response), profile=profile)
        consumer = ConsumerClient(consumer_t, profile=profile, timeout=5.0)
        await host.start()
        await consumer.start()
        await consumer.generate("prompt")
        rfcomm_frames = sum(
            1 for raw in sent_frames
            if Frame.decode(raw).msg_type == MessageType.RESPONSE
        )
        await host.close()
        await consumer.close()

        self.assertLess(
            rfcomm_frames, legacy_frames,
            f"RFCOMM ({rfcomm_frames}) should use fewer frames than LEGACY ({legacy_frames})",
        )


# ── platform skeleton tests ───────────────────────────────────────────────────

class RFCOMMPlatformSkeletonTests(unittest.TestCase):
    """Verify that platform-specific classes raise NotImplementedError (not ImportError)."""

    def test_linux_host_raises_on_wrong_platform(self):
        import sys
        if sys.platform == "linux":
            from keytalk.classic.linux import LinuxRFCOMMHostTransport
            t = LinuxRFCOMMHostTransport.__new__(LinuxRFCOMMHostTransport)
            t._psm = 0  # avoid platform check in __init__
            # Just verifying the class exists; start() would require hardware
        else:
            with self.assertRaises(RuntimeError):
                from keytalk.classic.linux import LinuxRFCOMMHostTransport
                LinuxRFCOMMHostTransport()

    def test_macos_host_raises_on_wrong_platform(self):
        import sys
        if sys.platform == "darwin":
            from keytalk.classic.macos import MacOSRFCOMMHostTransport
            t = MacOSRFCOMMHostTransport.__new__(MacOSRFCOMMHostTransport)
        else:
            with self.assertRaises(RuntimeError):
                from keytalk.classic.macos import MacOSRFCOMMHostTransport
                MacOSRFCOMMHostTransport()

    def test_windows_host_raises_on_wrong_platform(self):
        import sys
        if sys.platform == "win32":
            from keytalk.classic.windows import WindowsRFCOMMHostTransport
            t = WindowsRFCOMMHostTransport.__new__(WindowsRFCOMMHostTransport)
        else:
            with self.assertRaises(RuntimeError):
                from keytalk.classic.windows import WindowsRFCOMMHostTransport
                WindowsRFCOMMHostTransport()


# ── throughput benchmark ──────────────────────────────────────────────────────

class ClassicRFCOMMThroughputBenchmark(unittest.IsolatedAsyncioTestCase):
    """Compare LEGACY vs CLASSIC_RFCOMM throughput over in-memory transports."""

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
        return total_bytes / elapsed, total_frames // repeats

    async def _run_rfcomm(self, mtu: int, repeats: int = 3) -> tuple[float, int]:
        response = "a" * self._PAYLOAD_SIZE
        total_bytes = 0
        total_frames = 0
        elapsed = 0.0
        for _ in range(repeats):
            host_t, consumer_t = await create_rfcomm_loopback()
            profile = make_classic_rfcomm_profile(mtu)
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
        return total_bytes / elapsed, total_frames // repeats

    async def test_benchmark(self):
        legacy_bps, legacy_frames = await self._run_legacy()
        rfcomm_bps, rfcomm_frames = await self._run_rfcomm(_RFCOMM_MTU)

        print(
            f"\n--- CLASSIC_RFCOMM Throughput Benchmark ({self._PAYLOAD_SIZE} B payload) ---\n"
            f"  LEGACY        : {legacy_bps:>10,.0f} B/s  ({legacy_frames} frames/msg)\n"
            f"  CLASSIC_RFCOMM: {rfcomm_bps:>10,.0f} B/s  ({rfcomm_frames} frames/msg)\n"
            f"  Frames saved  : {legacy_frames - rfcomm_frames} "
            f"({legacy_frames / max(rfcomm_frames, 1):.1f}x fewer)\n"
        )

        self.assertLess(rfcomm_frames, legacy_frames)
        self.assertGreater(legacy_bps, 0)
        self.assertGreater(rfcomm_bps, 0)
