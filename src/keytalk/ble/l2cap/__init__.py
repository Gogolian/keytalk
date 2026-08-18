"""BLE L2CAP LE Credit-based Connection-Oriented Channel transports.

Platform-specific implementations live in sub-modules; all share
:class:`~keytalk.ble.l2cap.channel.L2CAPStreamTransport` and its 4-byte
length-prefixed framing protocol.

For tests and simulation, use :func:`create_l2cap_loopback` to obtain a
connected ``(host, consumer)`` transport pair without Bluetooth hardware.
"""

from __future__ import annotations

from .channel import L2CAPStreamTransport
from .loopback import L2CAPLoopbackTransport, create_l2cap_loopback

__all__ = [
    "L2CAPStreamTransport",
    "L2CAPLoopbackTransport",
    "create_l2cap_loopback",
]
