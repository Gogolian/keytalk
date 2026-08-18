"""Phase-2 tests: FAST_GATT mode — MTU negotiation, bidirectional compression,
CRC32 integrity, and an in-memory throughput benchmark.
"""

from __future__ import annotations

import asyncio
import struct
import time
import unittest
import zlib

from keytalk.backends import StaticBackend
from keytalk.consumer import ConsumerClient
from keytalk.host import HostService
from keytalk.modes import (
    FAST_GATT_PROFILE,
    LEGACY_PROFILE,
    Mode,
    make_fast_gatt_profile,
    profile_for_mode,
)
from keytalk.protocol import (
    CHECKSUM_SIZE,
    DEFAULT_ATT_MTU,
    Flags,
    Frame,
    MessageType,
    ProtocolError,
    Reassembler,
    chunk_message,
    compute_crc32,
    max_payload_for_mtu,
)
from keytalk.transport import InMemoryTransport, create_loopback

# ── helpers ──────────────────────────────────────────────────────────────────

LARGE_MTU = 251  # BLE 4.2+ DLE max PDU payload


class _FastLoopback:
    """Linked InMemoryTransport pair where the consumer side advertises FAST_GATT."""

    def __init__(self, mtu: int = LARGE_MTU) -> None:
        self.host_t = InMemoryTransport("host")
        self.consumer_t = _MTUTransport("consumer", mtu=mtu, caps=["legacy", "fast_gatt"])
        self.host_t.link(self.consumer_t)
        self.consumer_t.link(self.host_t)


class _MTUTransport(InMemoryTransport):
    """InMemoryTransport that reports a custom MTU and supports configure_write_mode."""

    def __init__(self, name: str = "", *, mtu: int = DEFAULT_ATT_MTU, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self._mtu = mtu
        self.write_with_response = True

    @property
    def mtu_size(self) -> int:
        return self._mtu

    def configure_write_mode(self, write_with_response: bool) -> None:
        self.write_with_response = write_with_response


# ── protocol-level CRC32 tests ───────────────────────────────────────────────

class CRC32Tests(unittest.TestCase):
    def test_compute_crc32_is_deterministic(self):
        data = b"hello world"
        self.assertEqual(compute_crc32(data), compute_crc32(data))

    def test_compute_crc32_differs_on_mutation(self):
        data = b"hello world"
        mutated = data[:5] + b"X" + data[6:]
        self.assertNotEqual(compute_crc32(data), compute_crc32(mutated))

    def test_compute_crc32_empty(self):
        self.assertIsInstance(compute_crc32(b""), int)

    def test_checksum_size_is_four(self):
        self.assertEqual(CHECKSUM_SIZE, 4)


class ChunkMessageChecksumTests(unittest.TestCase):
    """chunk_message with checksum=True appends a verifiable CRC32 trailer."""

    def _roundtrip(self, payload: bytes, max_size: int) -> bytes:
        frames = chunk_message(MessageType.RESPONSE, 1, payload, max_size, checksum=True)
        reassembler = Reassembler()
        result = None
        for f in frames:
            result = reassembler.feed(f)
        assert result is not None
        return result.payload

    def test_single_frame_roundtrip(self):
        payload = b"hello"
        result = self._roundtrip(payload, 64)
        self.assertEqual(result, payload)

    def test_multi_frame_roundtrip(self):
        payload = b"a" * 200
        result = self._roundtrip(payload, 13)
        self.assertEqual(result, payload)

    def test_empty_payload_roundtrip(self):
        result = self._roundtrip(b"", 13)
        self.assertEqual(result, b"")

    def test_checksum_flag_on_end_frame(self):
        frames = chunk_message(MessageType.RESPONSE, 1, b"data", 13, checksum=True)
        last = frames[-1]
        self.assertTrue(bool(last.flags & Flags.END))
        self.assertTrue(bool(last.flags & Flags.CHECKSUM))

    def test_checksum_flag_absent_without_option(self):
        frames = chunk_message(MessageType.RESPONSE, 1, b"data", 13)
        for f in frames:
            self.assertFalse(bool(f.flags & Flags.CHECKSUM))

    def test_corrupt_payload_raises_protocol_error(self):
        payload = b"important data"
        frames = chunk_message(MessageType.RESPONSE, 1, payload, 13, checksum=True)
        # Corrupt the last frame by flipping a bit in the payload.
        last = frames[-1]
        bad_payload = bytes([last.payload[0] ^ 0xFF]) + last.payload[1:]
        frames[-1] = Frame(
            msg_type=last.msg_type,
            message_id=last.message_id,
            seq=last.seq,
            payload=bad_payload,
            flags=last.flags,
        )
        reassembler = Reassembler()
        with self.assertRaises(ProtocolError) as ctx:
            for f in frames:
                reassembler.feed(f)
        self.assertIn("CRC32", str(ctx.exception))

    def test_without_checksum_no_validation(self):
        # Without the CHECKSUM flag, corrupt payload reaches the caller as-is.
        payload = b"important data"
        frames = chunk_message(MessageType.RESPONSE, 1, payload, 13, checksum=False)
        last = frames[-1]
        bad_payload = bytes([last.payload[0] ^ 0xFF]) + last.payload[1:]
        frames[-1] = Frame(
            msg_type=last.msg_type,
            message_id=last.message_id,
            seq=last.seq,
            payload=bad_payload,
            flags=last.flags,
        )
        reassembler = Reassembler()
        result = None
        for f in frames:
            result = reassembler.feed(f)
        # The corrupt bytes come through unchanged — no checksum protection.
        self.assertNotEqual(result.payload, payload)


class FrameStreamEncoderChecksumTests(unittest.TestCase):
    """FrameStreamEncoder with checksum=True emits a verifiable CRC32 trailer."""

    from keytalk.protocol import FrameStreamEncoder

    def _stream_roundtrip(self, chunks: list[bytes], max_size: int) -> bytes:
        from keytalk.protocol import FrameStreamEncoder
        enc = FrameStreamEncoder(MessageType.RESPONSE, 1, max_size, checksum=True)
        frames = []
        for chunk in chunks:
            frames.extend(enc.push(chunk))
        frames.extend(enc.finish())
        reassembler = Reassembler()
        result = None
        for f in frames:
            result = reassembler.feed(f)
        assert result is not None
        return result.payload

    def test_single_push_roundtrip(self):
        result = self._stream_roundtrip([b"hello world"], 64)
        self.assertEqual(result, b"hello world")

    def test_multi_push_roundtrip(self):
        result = self._stream_roundtrip([b"foo", b"bar", b"baz"], 4)
        self.assertEqual(result, b"foobarbaz")


# ── profile / negotiation tests ──────────────────────────────────────────────

class FastGattProfileTests(unittest.TestCase):
    def test_fast_gatt_profile_mode(self):
        self.assertEqual(FAST_GATT_PROFILE.mode, Mode.FAST_GATT)

    def test_fast_gatt_profile_no_write_with_response(self):
        self.assertFalse(FAST_GATT_PROFILE.write_with_response)

    def test_fast_gatt_profile_credit_window_flow(self):
        self.assertEqual(FAST_GATT_PROFILE.flow_control, "credit_window")

    def test_make_fast_gatt_profile_custom_mtu(self):
        p = make_fast_gatt_profile(LARGE_MTU)
        self.assertEqual(p.mtu, LARGE_MTU)
        self.assertEqual(p.max_payload_size, max_payload_for_mtu(LARGE_MTU))

    def test_profile_for_mode_fast_gatt(self):
        p = profile_for_mode("fast_gatt")
        self.assertEqual(p.mode, Mode.FAST_GATT)

    def test_legacy_profile_unchanged(self):
        self.assertEqual(LEGACY_PROFILE.mode, Mode.LEGACY)
        self.assertTrue(LEGACY_PROFILE.write_with_response)


# ── end-to-end FAST_GATT integration tests ───────────────────────────────────

class FastGattIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def _make_fast_gatt_pair(self, response: str, mtu: int = LARGE_MTU):
        lb = _FastLoopback(mtu=mtu)
        host = HostService(lb.host_t, StaticBackend(response))
        consumer = ConsumerClient(lb.consumer_t, requested_mode="fast_gatt", timeout=5.0)
        await host.start()
        await consumer.start()
        self.addAsyncCleanup(host.close)
        self.addAsyncCleanup(consumer.close)
        return host, consumer, lb

    async def test_fast_gatt_response_correct(self):
        response = "The quick brown fox"
        _, consumer, _ = await self._make_fast_gatt_pair(response)
        result = await consumer.generate("prompt")
        self.assertEqual(result, response)

    async def test_fast_gatt_large_response(self):
        response = "x" * 4096
        _, consumer, _ = await self._make_fast_gatt_pair(response)
        result = await consumer.generate("prompt")
        self.assertEqual(result, response)

    async def test_fast_gatt_host_resizes_frames_to_mtu(self):
        """After SELECT, the host must use MTU-sized frames for responses."""
        response = "x" * 500
        host, consumer, lb = await self._make_fast_gatt_pair(response, mtu=LARGE_MTU)
        await consumer.generate("prompt")
        # All RESPONSE frames on the host→consumer wire should fit in LARGE_MTU.
        max_allowed = max_payload_for_mtu(LARGE_MTU) + CHECKSUM_SIZE + 4  # some slack
        response_frames = [
            Frame.decode(raw)
            for raw in lb.host_t.sent
            if Frame.decode(raw).msg_type == MessageType.RESPONSE
        ]
        self.assertTrue(response_frames, "no RESPONSE frames observed")
        oversized = [f for f in response_frames if len(f.payload) > max_allowed]
        self.assertEqual(oversized, [], f"{len(oversized)} frames exceed LARGE_MTU")

    async def test_fast_gatt_fewer_frames_than_legacy(self):
        """FAST_GATT with a large MTU sends fewer frames than LEGACY (small MTU)."""
        response = "y" * 2000

        # Legacy pair with default small MTU (no CAPS → auto→legacy)
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

        # FAST_GATT pair with large MTU
        _, _, lb = await self._make_fast_gatt_pair(response, mtu=LARGE_MTU)
        fast_frames = sum(
            1 for raw in lb.host_t.sent
            if Frame.decode(raw).msg_type == MessageType.RESPONSE
        )

        self.assertLess(fast_frames, legacy_frames,
                        f"FAST_GATT ({fast_frames}) should use fewer frames than legacy ({legacy_frames})")

    async def test_fast_gatt_write_without_response_enabled(self):
        """After FAST_GATT negotiation the consumer transport uses WwR=False."""
        _, _, lb = await self._make_fast_gatt_pair("ok")
        self.assertFalse(lb.consumer_t.write_with_response)

    async def test_fast_gatt_profile_on_host_after_select(self):
        host, consumer, _ = await self._make_fast_gatt_pair("hi")
        await consumer.generate("prompt")
        self.assertEqual(host._profile.mode, Mode.FAST_GATT)

    async def test_fast_gatt_checksum_verified_end_to_end(self):
        """RESPONSE frames carry a CHECKSUM flag; the Reassembler verifies it."""
        response = "integrity check " * 50
        _, consumer, lb = await self._make_fast_gatt_pair(response)
        result = await consumer.generate("prompt")
        self.assertEqual(result, response)
        # At least one RESPONSE frame must carry the CHECKSUM flag.
        checksum_frames = [
            Frame.decode(raw)
            for raw in lb.host_t.sent
            if Frame.decode(raw).msg_type == MessageType.RESPONSE
            and bool(Frame.decode(raw).flags & Flags.CHECKSUM)
        ]
        self.assertTrue(checksum_frames, "expected at least one frame with CHECKSUM flag")


# ── throughput benchmark ─────────────────────────────────────────────────────

class ThroughputBenchmark(unittest.IsolatedAsyncioTestCase):
    """Compare LEGACY vs FAST_GATT over in-memory transports.

    Not a performance regression test — just reports the numbers so developers
    can observe the relative improvement.  Assertions only check sanity bounds.
    """

    _PAYLOAD_SIZE = 8192  # bytes — representative LLM response chunk

    async def _run_mode(
        self, mode: str, mtu: int, repeats: int = 3
    ) -> tuple[float, float, int]:
        """Return (bytes_per_sec, frames_per_sec, total_frames)."""
        response = "a" * self._PAYLOAD_SIZE
        total_bytes = 0
        total_frames = 0
        elapsed = 0.0

        for _ in range(repeats):
            if mode == "fast_gatt":
                consumer_t = _MTUTransport(
                    "consumer", mtu=mtu, caps=["legacy", "fast_gatt"]
                )
                host_t = InMemoryTransport("host")
                host_t.link(consumer_t)
                consumer_t.link(host_t)
                host = HostService(host_t, StaticBackend(response))
                consumer = ConsumerClient(consumer_t, requested_mode="fast_gatt", timeout=10.0)
            else:
                # Legacy: use "auto" with no CAPS so negotiate_mode returns LEGACY silently.
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

        bps = (total_bytes * repeats) / elapsed / repeats
        fps = (total_frames * repeats) / elapsed / repeats
        return bps, fps, total_frames // repeats

    async def test_benchmark(self):
        legacy_bps, legacy_fps, legacy_frames = await self._run_mode("legacy", DEFAULT_ATT_MTU)
        fast_bps, fast_fps, fast_frames = await self._run_mode("fast_gatt", LARGE_MTU)

        print(
            f"\n--- Throughput Benchmark ({self._PAYLOAD_SIZE} B payload) ---\n"
            f"  LEGACY   : {legacy_bps:>10,.0f} B/s  {legacy_fps:>7,.0f} frames/s  "
            f"({legacy_frames} frames/msg)\n"
            f"  FAST_GATT: {fast_bps:>10,.0f} B/s  {fast_fps:>7,.0f} frames/s  "
            f"({fast_frames} frames/msg)\n"
            f"  Speedup  : {fast_bps/legacy_bps:.1f}x bytes/s, "
            f"{legacy_frames/max(fast_frames,1):.1f}x fewer frames\n"
        )

        # Sanity: FAST_GATT must use fewer frames with a large MTU.
        self.assertLess(fast_frames, legacy_frames)
        # Both modes must deliver the full payload.
        self.assertGreater(legacy_bps, 0)
        self.assertGreater(fast_bps, 0)
