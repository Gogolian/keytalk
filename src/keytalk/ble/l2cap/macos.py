"""macOS L2CAP LE CoC transport skeleton via CoreBluetooth / pyobjc.

The peripheral calls ``CBPeripheralManager.publishL2CAPChannel(withEncryption:)``
to obtain a dynamically-assigned PSM; the central calls
``CBPeripheral.openL2CAPChannel(withPSM:)``.  Full implementation is deferred
to hardware testing; this file provides the class interface so imports succeed.
"""

from __future__ import annotations

import sys
from typing import Any, Optional

from .channel import L2CAPStreamTransport

__all__ = ["MacOSL2CAPHostTransport", "MacOSL2CAPConsumerTransport"]


def _require_macos() -> None:
    if sys.platform != "darwin":
        raise RuntimeError(
            "MacOSL2CAPTransport is only supported on macOS; "
            "use LinuxL2CAPTransport on Linux or fall back to FAST_GATT"
        )


class MacOSL2CAPHostTransport(L2CAPStreamTransport):
    """Host-side L2CAP LE CoC channel via CBPeripheralManager."""

    def __init__(self) -> None:
        super().__init__()
        _require_macos()
        self._psm: int = 0

    @property
    def psm(self) -> int:
        return self._psm

    async def start(self) -> None:
        # TODO(phase3-macos): call CBPeripheralManager.publishL2CAPChannelWithEncryption
        # via pyobjc; await the didPublishL2CAPChannel delegate callback; store
        # the assigned PSM in self._psm.
        raise NotImplementedError(
            "macOS L2CAP host transport is not yet fully implemented"
        )


class MacOSL2CAPConsumerTransport(L2CAPStreamTransport):
    """Consumer-side L2CAP LE CoC channel via CBPeripheral."""

    def __init__(self, peripheral: Any, psm: int) -> None:
        super().__init__()
        _require_macos()
        self._peripheral = peripheral
        self._psm = psm

    async def start(self) -> None:
        # TODO(phase3-macos): call peripheral.openL2CAPChannel_(psm) via pyobjc;
        # await the didOpenL2CAPChannel delegate callback; wrap the resulting
        # CBL2CAPChannel inputStream/outputStream as asyncio streams.
        raise NotImplementedError(
            "macOS L2CAP consumer transport is not yet fully implemented"
        )
