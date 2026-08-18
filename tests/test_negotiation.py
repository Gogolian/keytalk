"""Tests for Phase-1 capability handshake and mode negotiation."""

from __future__ import annotations

import unittest

from keytalk.consumer import ConsumerClient
from keytalk.host import HostService
from keytalk.modes import (
    LEGACY_PROFILE,
    Mode,
    NegotiationError,
    negotiate_mode,
    mode_id_for,
    mode_for_id,
    profile_for_mode,
)
from keytalk.protocol import (
    Frame,
    Flags,
    MessageType,
    ProtocolError,
    encode_select_payload,
    decode_select_payload,
)
from keytalk.transport import InMemoryTransport, create_loopback


# ---------------------------------------------------------------------------
# negotiate_mode() unit tests
# ---------------------------------------------------------------------------

class NegotiateModeTests(unittest.TestCase):
    def test_auto_old_host_returns_legacy(self):
        """host_modes=None + auto → legacy, no exception."""
        self.assertEqual(negotiate_mode(None, "auto"), LEGACY_PROFILE)

    def test_explicit_mode_old_host_raises(self):
        with self.assertRaises(NegotiationError):
            negotiate_mode(None, "fast_gatt")

    def test_auto_picks_legacy_from_new_host(self):
        """auto with a host offering only legacy → legacy."""
        self.assertEqual(negotiate_mode(["legacy"], "auto"), LEGACY_PROFILE)

    def test_auto_picks_fast_gatt_when_implemented(self):
        """auto with host offering fast_gatt picks it now that Phase 2 is done."""
        from keytalk.modes import FAST_GATT_PROFILE
        result = negotiate_mode(["legacy", "fast_gatt"], "auto")
        self.assertEqual(result, FAST_GATT_PROFILE)

    def test_explicit_legacy_accepted(self):
        result = negotiate_mode(["legacy"], "legacy")
        self.assertEqual(result, LEGACY_PROFILE)

    def test_explicit_mode_not_offered_raises(self):
        with self.assertRaises(NegotiationError) as ctx:
            negotiate_mode(["legacy"], "fast_gatt")
        self.assertIn("fast_gatt", str(ctx.exception))

    def test_explicit_fast_gatt_accepted(self):
        """fast_gatt is now implemented; selecting it explicitly must succeed."""
        from keytalk.modes import FAST_GATT_PROFILE
        result = negotiate_mode(["legacy", "fast_gatt"], "fast_gatt")
        self.assertEqual(result, FAST_GATT_PROFILE)

    def test_unknown_requested_mode_raises(self):
        with self.assertRaises(ValueError):
            negotiate_mode(["legacy"], "teleport")


# ---------------------------------------------------------------------------
# Mode ID wire encoding
# ---------------------------------------------------------------------------

class ModeIdTests(unittest.TestCase):
    def test_roundtrip_all_defined_modes(self):
        for mode in Mode:
            self.assertEqual(mode_for_id(mode_id_for(mode)), mode)

    def test_unknown_id_raises(self):
        with self.assertRaises(ValueError):
            mode_for_id(255)


# ---------------------------------------------------------------------------
# SELECT payload encode / decode
# ---------------------------------------------------------------------------

class SelectPayloadTests(unittest.TestCase):
    def test_roundtrip(self):
        raw = encode_select_payload(0, 247)
        mode_id, mtu = decode_select_payload(raw)
        self.assertEqual(mode_id, 0)
        self.assertEqual(mtu, 247)

    def test_fits_in_legacy_frame(self):
        """3-byte SELECT payload must fit in the legacy 13-byte frame payload."""
        self.assertLessEqual(len(encode_select_payload(0, 23)), 13)

    def test_too_short_raises(self):
        with self.assertRaises(ProtocolError):
            decode_select_payload(b"\x00")


# ---------------------------------------------------------------------------
# Integration: consumer sends SELECT, host receives it
# ---------------------------------------------------------------------------

class SelectIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def _make_pair(self, consumer_caps=None):
        from keytalk.backends import StaticBackend
        host_t, consumer_t = create_loopback(consumer_caps=consumer_caps)
        host = HostService(host_t, StaticBackend("hi"))
        consumer = ConsumerClient(consumer_t, timeout=5.0)
        await host.start()
        await consumer.start()
        self.addAsyncCleanup(host.close)
        self.addAsyncCleanup(consumer.close)
        return host, consumer

    async def test_old_host_no_select_sent(self):
        """InMemoryTransport with caps=None → no SELECT frame sent."""
        host_t, consumer_t = create_loopback()  # no caps
        from keytalk.backends import StaticBackend
        host = HostService(host_t, StaticBackend("hi"))
        consumer = ConsumerClient(consumer_t, timeout=5.0)
        await host.start()
        await consumer.start()
        self.addAsyncCleanup(host.close)
        self.addAsyncCleanup(consumer.close)

        # consumer_t (host side) should have received zero frames — no SELECT.
        select_frames = [
            f for f in host_t.sent
            if len(f) >= 2 and f[1] == MessageType.SELECT
        ]
        self.assertEqual(select_frames, [])
        # Both sides use legacy profile.
        self.assertEqual(consumer._profile, LEGACY_PROFILE)
        self.assertEqual(host._profile, LEGACY_PROFILE)

    async def test_new_host_select_sent_and_received(self):
        """With caps=['legacy'] the consumer sends SELECT and host processes it."""
        host_t, consumer_t = create_loopback(consumer_caps=["legacy"])
        from keytalk.backends import StaticBackend
        host = HostService(host_t, StaticBackend("hi"))
        consumer = ConsumerClient(consumer_t, timeout=5.0)
        await host.start()
        await consumer.start()
        self.addAsyncCleanup(host.close)
        self.addAsyncCleanup(consumer.close)

        # Exactly one SELECT frame should have been sent by the consumer.
        import asyncio; await asyncio.sleep(0)  # let deliveries drain
        select_frames = [
            f for f in consumer_t.sent
            if len(f) >= 2 and f[1] == MessageType.SELECT
        ]
        self.assertEqual(len(select_frames), 1)

        # Host should have adopted the legacy profile (no change from default).
        self.assertEqual(host._profile, LEGACY_PROFILE)
        # Host stores the reported MTU.
        from keytalk.protocol import DEFAULT_ATT_MTU
        self.assertEqual(host._negotiated_mtu, DEFAULT_ATT_MTU)

    async def test_explicit_legacy_mode_negotiated(self):
        """--mode legacy with a host offering legacy succeeds."""
        host_t, consumer_t = create_loopback(consumer_caps=["legacy"])
        from keytalk.backends import StaticBackend
        host = HostService(host_t, StaticBackend("hi"))
        consumer = ConsumerClient(consumer_t, requested_mode="legacy", timeout=5.0)
        await host.start()
        await consumer.start()
        self.addAsyncCleanup(host.close)
        self.addAsyncCleanup(consumer.close)
        self.assertEqual(consumer._profile, LEGACY_PROFILE)

    async def test_explicit_mode_not_offered_raises(self):
        """--mode fast_gatt when host offers only legacy → NegotiationError."""
        host_t, consumer_t = create_loopback(consumer_caps=["legacy"])
        from keytalk.backends import StaticBackend
        host = HostService(host_t, StaticBackend("hi"))
        consumer = ConsumerClient(consumer_t, requested_mode="fast_gatt", timeout=5.0)
        await host.start()
        with self.assertRaises(NegotiationError):
            await consumer.start()
        self.addAsyncCleanup(host.close)

    async def test_explicit_mode_old_host_raises(self):
        """--mode fast_gatt when host has no CAPS → NegotiationError."""
        host_t, consumer_t = create_loopback()  # no caps → old host
        from keytalk.backends import StaticBackend
        host = HostService(host_t, StaticBackend("hi"))
        consumer = ConsumerClient(consumer_t, requested_mode="fast_gatt", timeout=5.0)
        await host.start()
        with self.assertRaises(NegotiationError):
            await consumer.start()
        self.addAsyncCleanup(host.close)

    async def test_data_transfer_after_negotiation(self):
        """Full round-trip still works after negotiation completes."""
        host_t, consumer_t = create_loopback(consumer_caps=["legacy"])
        from keytalk.backends import StaticBackend
        host = HostService(host_t, StaticBackend("Hello!"))
        consumer = ConsumerClient(consumer_t, timeout=5.0)
        await host.start()
        await consumer.start()
        self.addAsyncCleanup(host.close)
        self.addAsyncCleanup(consumer.close)
        result = await consumer.generate("ping")
        self.assertEqual(result, "Hello!")


if __name__ == "__main__":
    unittest.main()
