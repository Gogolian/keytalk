"""Windows Bluetooth Classic RFCOMM transport.

Uses WinSock ``AF_BTH``/``BTHPROTO_RFCOMM`` sockets available from Windows 8+.
The remote device must be paired before connecting.
"""

from __future__ import annotations

import asyncio
import socket
import sys

from .channel import RFCOMMStreamTransport

__all__ = ["WindowsRFCOMMHostTransport", "WindowsRFCOMMConsumerTransport"]

_DEFAULT_RFCOMM_PORT = 1
# Numeric fallbacks; Python exposes AF_BTH and BTHPROTO_RFCOMM on Windows builds.
_AF_BTH: int = getattr(socket, "AF_BTH", 32)
_BTHPROTO_RFCOMM: int = getattr(socket, "BTHPROTO_RFCOMM", 3)
# Windows bindable "any" BT address (BDADDR_ANY).
_BDADDR_ANY = "00:00:00:00:00:00"


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
        self._srv_sock: socket.socket | None = None

    @property
    def port(self) -> int:
        return self._port

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        srv = socket.socket(_AF_BTH, socket.SOCK_STREAM, _BTHPROTO_RFCOMM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.setblocking(False)
        srv.bind((_BDADDR_ANY, self._port))
        srv.listen(1)
        self._srv_sock = srv
        conn, _addr = await loop.sock_accept(srv)
        srv.close()
        self._srv_sock = None
        conn.setblocking(False)
        reader, writer = await asyncio.open_connection(sock=conn)
        self._attach(reader, writer)
        self._start_recv()

    async def close(self) -> None:
        if self._srv_sock is not None:
            try:
                self._srv_sock.close()
            except OSError:
                pass
            self._srv_sock = None
        await super().close()


class WindowsRFCOMMConsumerTransport(RFCOMMStreamTransport):
    """Consumer-side RFCOMM connection on Windows."""

    def __init__(self, address: str, port: int = _DEFAULT_RFCOMM_PORT) -> None:
        super().__init__()
        _require_windows()
        self._address = address
        self._port = port

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        sock = socket.socket(_AF_BTH, socket.SOCK_STREAM, _BTHPROTO_RFCOMM)
        sock.setblocking(False)
        await loop.sock_connect(sock, (self._address, self._port))
        reader, writer = await asyncio.open_connection(sock=sock)
        self._attach(reader, writer)
        self._start_recv()
