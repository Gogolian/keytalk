"""Bluetooth Classic RFCOMM transports for keytalk.

Platform-specific implementations live in sub-modules; all share
:class:`~keytalk.classic.channel.RFCOMMStreamTransport` and its 4-byte
length-prefixed framing protocol (identical to the BLE L2CAP COC transport).

For tests and simulation, use :func:`create_rfcomm_loopback` to obtain a
connected ``(host, consumer)`` transport pair without Bluetooth hardware.
"""

from __future__ import annotations

from .channel import RFCOMMStreamTransport
from .loopback import RFCOMMLoopbackTransport, create_rfcomm_loopback

__all__ = [
    "RFCOMMStreamTransport",
    "RFCOMMLoopbackTransport",
    "create_rfcomm_loopback",
]
