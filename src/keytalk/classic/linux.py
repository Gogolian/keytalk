"""Linux Bluetooth Classic RFCOMM transport.

Uses ``AF_BLUETOOTH``/``BTPROTO_RFCOMM`` sockets available on Linux 2.6+.
The remote device must be paired (classic pairing) before connecting.
"""

from __future__ import annotations

import asyncio
import socket
import sys

from .channel import RFCOMMStreamTransport

__all__ = ["LinuxRFCOMMHostTransport", "LinuxRFCOMMConsumerTransport"]

_DEFAULT_RFCOMM_CHANNEL = 1
# Numeric fallbacks in case the constants aren't exposed on this Python build.
_AF_BLUETOOTH: int = getattr(socket, "AF_BLUETOOTH", 31)
_BTPROTO_RFCOMM: int = getattr(socket, "BTPROTO_RFCOMM", 3)
_BDADDR_ANY = "00:00:00:00:00:00"


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
        self._srv_sock: socket.socket | None = None

    @property
    def channel(self) -> int:
        return self._channel

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        srv = socket.socket(_AF_BLUETOOTH, socket.SOCK_STREAM, _BTPROTO_RFCOMM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.setblocking(False)
        srv.bind((_BDADDR_ANY, self._channel))
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


class LinuxRFCOMMConsumerTransport(RFCOMMStreamTransport):
    """Consumer-side RFCOMM connection on Linux."""

    def __init__(self, address: str, channel: int = _DEFAULT_RFCOMM_CHANNEL) -> None:
        super().__init__()
        _require_linux()
        self._address = address
        self._channel = channel

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        sock = socket.socket(_AF_BLUETOOTH, socket.SOCK_STREAM, _BTPROTO_RFCOMM)
        sock.setblocking(False)
        await loop.sock_connect(sock, (self._address, self._channel))
        reader, writer = await asyncio.open_connection(sock=sock)
        self._attach(reader, writer)
        self._start_recv()
