"""RFCOMM stream transport base — 4-byte length-prefixed framing.

Bluetooth Classic RFCOMM provides a reliable ordered byte stream (analogous to
TCP over IP).  keytalk uses the same 4-byte big-endian length-prefixed framing
as L2CAP COC so both modes share the same reassembly and CRC32 checksum logic.
"""

from __future__ import annotations

from ..ble.l2cap.channel import L2CAPStreamTransport

__all__ = ["RFCOMMStreamTransport", "SPP_UUID"]

# Serial Port Profile UUID registered with the Bluetooth SIG.
SPP_UUID = "00001101-0000-1000-8000-00805F9B34FB"


class RFCOMMStreamTransport(L2CAPStreamTransport):
    """Transport backed by a Bluetooth Classic RFCOMM stream.

    Inherits the 4-byte length-prefix framing from
    :class:`~keytalk.ble.l2cap.channel.L2CAPStreamTransport`; subclasses only
    need to open the RFCOMM socket/channel and call ``_attach`` + ``_start_recv``.
    """
