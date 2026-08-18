"""macOS Bluetooth Classic RFCOMM transport skeleton via IOBluetooth/pyobjc.

The host registers an RFCOMM service record and opens a server channel;
the consumer calls ``IOBluetoothDevice.openRFCOMMChannelSync`` (or the async
variant).  Full implementation is deferred to hardware testing; this file
provides the class interface so imports succeed.
"""

from __future__ import annotations

import sys
from typing import Any

from .channel import RFCOMMStreamTransport

__all__ = ["MacOSRFCOMMHostTransport", "MacOSRFCOMMConsumerTransport"]


def _require_macos() -> None:
    if sys.platform != "darwin":
        raise RuntimeError(
            "MacOSRFCOMMTransport is only supported on macOS; "
            "use LinuxRFCOMMTransport on Linux or WindowsRFCOMMTransport on Windows"
        )


class MacOSRFCOMMHostTransport(RFCOMMStreamTransport):
    """Host-side RFCOMM channel via IOBluetoothRFCOMMChannel."""

    def __init__(self) -> None:
        super().__init__()
        _require_macos()
        self._channel_id: int = 0

    @property
    def channel_id(self) -> int:
        return self._channel_id

    async def start(self) -> None:
        # TODO(phase4-macos): register SPP service record via IOBluetooth,
        # obtain a server channel id, listen for incoming RFCOMM connections,
        # wrap the IOBluetoothRFCOMMChannel streams, _attach + _start_recv.
        raise NotImplementedError(
            "macOS RFCOMM host transport is not yet fully implemented"
        )


class MacOSRFCOMMConsumerTransport(RFCOMMStreamTransport):
    """Consumer-side RFCOMM channel via IOBluetoothDevice."""

    def __init__(self, device: Any, channel_id: int) -> None:
        super().__init__()
        _require_macos()
        self._device = device
        self._channel_id = channel_id

    async def start(self) -> None:
        # TODO(phase4-macos): call device.openRFCOMMChannelSync_(channel_id) via
        # pyobjc; wrap the resulting IOBluetoothRFCOMMChannel as asyncio streams,
        # _attach + _start_recv.
        raise NotImplementedError(
            "macOS RFCOMM consumer transport is not yet fully implemented"
        )
