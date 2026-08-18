"""Windows Bluetooth Classic RFCOMM transport skeleton.

Uses the WinSock RFCOMM sockets (``AF_BTH``/``BTHPROTO_RFCOMM``) available from
Windows 8+.  Full implementation is deferred to hardware testing; this file
provides the class interface so imports succeed.
"""

from __future__ import annotations

import sys

from .channel import RFCOMMStreamTransport

__all__ = ["WindowsRFCOMMHostTransport", "WindowsRFCOMMConsumerTransport"]

_DEFAULT_RFCOMM_PORT = 1


def _require_windows() -> None:
    if sys.platform != "win32":
        raise RuntimeError(
            "WindowsRFCOMMTransport is only supported on Windows; "
            "use LinuxRFCOMMTransport on Linux or MacOSRFCOMMTransport on macOS"
        )


class WindowsRFCOMMHostTransport(RFCOMMStreamTransport):
    """Host-side RFCOMM listener on Windows (AF_BTH/BTHPROTO_RFCOMM)."""

    def __init__(self, port: int = _DEFAULT_RFCOMM_PORT) -> None:
        super().__init__()
        _require_windows()
        self._port = port

    @property
    def port(self) -> int:
        return self._port

    async def start(self) -> None:
        # TODO(phase4-windows): bind AF_BTH/BTHPROTO_RFCOMM socket, register
        # SPP service record via WSASetService, listen, accept via asyncio loop,
        # _attach + _start_recv.
        raise NotImplementedError(
            "Windows RFCOMM host transport is not yet fully implemented"
        )

    async def close(self) -> None:
        await super().close()


class WindowsRFCOMMConsumerTransport(RFCOMMStreamTransport):
    """Consumer-side RFCOMM connection on Windows."""

    def __init__(self, address: str, port: int = _DEFAULT_RFCOMM_PORT) -> None:
        super().__init__()
        _require_windows()
        self._address = address
        self._port = port

    async def start(self) -> None:
        # TODO(phase4-windows): connect AF_BTH/BTHPROTO_RFCOMM socket to the
        # remote device, wrap as asyncio streams, _attach + _start_recv.
        raise NotImplementedError(
            "Windows RFCOMM consumer transport is not yet fully implemented"
        )
