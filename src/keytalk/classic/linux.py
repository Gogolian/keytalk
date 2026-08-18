"""Linux Bluetooth Classic RFCOMM transport skeleton.

Uses ``AF_BLUETOOTH``/``BTPROTO_RFCOMM`` sockets.  The remote device must be
paired (classic pairing) before connecting.  Full implementation is deferred to
hardware testing; this file provides the class interface so imports succeed.
"""

from __future__ import annotations

import sys

from .channel import RFCOMMStreamTransport

__all__ = ["LinuxRFCOMMHostTransport", "LinuxRFCOMMConsumerTransport"]

_DEFAULT_RFCOMM_CHANNEL = 1


def _require_linux() -> None:
    if sys.platform != "linux":
        raise RuntimeError(
            "LinuxRFCOMMTransport is only supported on Linux; "
            "use MacOSRFCOMMTransport on macOS or WindowsRFCOMMTransport on Windows"
        )


class LinuxRFCOMMHostTransport(RFCOMMStreamTransport):
    """Host-side RFCOMM listener on Linux (AF_BLUETOOTH/BTPROTO_RFCOMM)."""

    def __init__(self, channel: int = _DEFAULT_RFCOMM_CHANNEL) -> None:
        super().__init__()
        _require_linux()
        self._channel = channel

    @property
    def channel(self) -> int:
        return self._channel

    async def start(self) -> None:
        # TODO(phase4-linux): bind AF_BLUETOOTH/BTPROTO_RFCOMM socket on
        # self._channel, listen, accept via asyncio loop, call _attach + _start_recv.
        raise NotImplementedError(
            "Linux RFCOMM host transport is not yet fully implemented"
        )

    async def close(self) -> None:
        await super().close()


class LinuxRFCOMMConsumerTransport(RFCOMMStreamTransport):
    """Consumer-side RFCOMM connection on Linux."""

    def __init__(self, address: str, channel: int = _DEFAULT_RFCOMM_CHANNEL) -> None:
        super().__init__()
        _require_linux()
        self._address = address
        self._channel = channel

    async def start(self) -> None:
        # TODO(phase4-linux): connect AF_BLUETOOTH/BTPROTO_RFCOMM socket to
        # (self._address, self._channel), wrap as asyncio streams, _attach + _start_recv.
        raise NotImplementedError(
            "Linux RFCOMM consumer transport is not yet fully implemented"
        )
