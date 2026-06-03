"""Tests for the BLE layer that do not require a Bluetooth radio.

The real adapters need optional ``bleak``/``bless`` dependencies and physical
hardware, so here we only assert on the parts that are safe to check offline:
the shared UUIDs and graceful errors when the optional deps are missing.
"""

import unittest

from keytalk.ble import (
    PROMPT_CHAR_UUID,
    RESPONSE_CHAR_UUID,
    SERVICE_NAME,
    SERVICE_UUID,
)


class UuidTests(unittest.TestCase):
    def test_uuids_are_distinct_and_well_formed(self):
        uuids = {SERVICE_UUID, PROMPT_CHAR_UUID, RESPONSE_CHAR_UUID}
        self.assertEqual(len(uuids), 3)
        for value in uuids:
            self.assertRegex(
                value,
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
                r"[0-9a-f]{4}-[0-9a-f]{12}$",
            )

    def test_service_name(self):
        self.assertEqual(SERVICE_NAME, "keytalk")


class OptionalDependencyTests(unittest.TestCase):
    def test_central_reports_missing_bleak(self):
        from keytalk.ble import central

        try:
            import bleak  # noqa: F401

            self.skipTest("bleak is installed; cannot test missing-dep path")
        except ImportError:
            pass
        with self.assertRaises(RuntimeError):
            central._import_bleak()

    def test_peripheral_reports_missing_bless(self):
        from keytalk.ble import peripheral

        try:
            import bless  # noqa: F401

            self.skipTest("bless is installed; cannot test missing-dep path")
        except ImportError:
            pass
        with self.assertRaises(RuntimeError):
            peripheral._import_bless()

    def test_transports_construct_without_radio(self):
        # Construction must not touch the radio or import optional deps.
        from keytalk.ble.central import BleakCentralTransport
        from keytalk.ble.peripheral import BlessPeripheralTransport

        BleakCentralTransport("AA:BB:CC:DD:EE:FF")
        BlessPeripheralTransport(name="keytalk-test")


class _FakeClient:
    """Minimal stand-in for a bleak client, no radio required."""

    def __init__(self, fail_writes: int = 0) -> None:
        self.is_connected = True
        self.writes: list = []
        self._fail_writes = fail_writes

    async def write_gatt_char(self, char, frame, response):  # noqa: ANN001
        if self._fail_writes > 0:
            self._fail_writes -= 1
            self.is_connected = False
            raise RuntimeError("disconnected")
        self.writes.append(frame)


class CentralReconnectTests(unittest.IsolatedAsyncioTestCase):
    def _transport(self, **kwargs):
        from keytalk.ble.central import BleakCentralTransport

        return BleakCentralTransport(
            "AA:BB:CC:DD:EE:FF", reconnect_delay=0, **kwargs
        )

    async def test_send_reconnects_when_link_dropped(self):
        t = self._transport()
        t._client = _FakeClient()
        t._prompt_char_obj = object()

        await t.send(b"hello")
        self.assertEqual(t._client.writes, [b"hello"])

        # Simulate a dropped link and a successful reconnect.
        t._client.is_connected = False
        connects = []

        async def fake_connect():
            connects.append(1)
            t._client = _FakeClient()
            t._prompt_char_obj = object()

        t._connect = fake_connect
        await t.send(b"again")

        self.assertEqual(len(connects), 1)
        self.assertEqual(t._client.writes, [b"again"])

    async def test_send_retries_write_on_mid_write_disconnect(self):
        # The first write raises "disconnected"; after a reconnect the retry
        # must succeed instead of bubbling the failure up.
        t = self._transport(reconnect_attempts=3)
        t._client = _FakeClient(fail_writes=1)
        t._prompt_char_obj = object()
        connects = []

        async def fake_connect():
            connects.append(1)
            t._client = _FakeClient()
            t._prompt_char_obj = object()

        t._connect = fake_connect
        await t.send(b"payload")

        self.assertEqual(len(connects), 1)
        self.assertEqual(t._client.writes, [b"payload"])

    async def test_send_raises_when_reconnect_exhausted(self):
        from keytalk.transport import TransportClosed

        t = self._transport(reconnect_attempts=2)
        t._client = None
        attempts = []

        async def failing_connect():
            attempts.append(1)
            raise RuntimeError("no radio")

        t._connect = failing_connect
        with self.assertRaises(TransportClosed):
            await t.send(b"x")
        self.assertEqual(len(attempts), 2)

    async def test_send_after_close_raises(self):
        from keytalk.transport import TransportClosed

        t = self._transport()
        await t.close()
        with self.assertRaises(TransportClosed):
            await t.send(b"x")


if __name__ == "__main__":
    unittest.main()
