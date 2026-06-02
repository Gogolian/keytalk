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


if __name__ == "__main__":
    unittest.main()
