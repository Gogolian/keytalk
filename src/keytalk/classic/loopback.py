"""In-process RFCOMM loopback using a socketpair — for tests and simulation.

:func:`create_rfcomm_loopback` returns a connected ``(host, consumer)`` transport
pair backed by ``socket.socketpair()``.  No Bluetooth hardware required.
"""

from __future__ import annotations

import asyncio
import socket
from typing import Tuple

from .channel import RFCOMMStreamTransport

__all__ = ["RFCOMMLoopbackTransport", "create_rfcomm_loopback"]


class RFCOMMLoopbackTransport(RFCOMMStreamTransport):
    """RFCOMM stream transport pre-wired to a socket; ``start()`` begins the recv loop."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        super().__init__()
        self._attach(reader, writer)

    async def start(self) -> None:
        self._start_recv()


async def create_rfcomm_loopback() -> Tuple[RFCOMMLoopbackTransport, RFCOMMLoopbackTransport]:
    """Return a ``(host_transport, consumer_transport)`` pair backed by a socket pair."""
    sock_a, sock_b = socket.socketpair()
    reader_a, writer_a = await asyncio.open_connection(sock=sock_a)
    reader_b, writer_b = await asyncio.open_connection(sock=sock_b)
    return RFCOMMLoopbackTransport(reader_a, writer_a), RFCOMMLoopbackTransport(reader_b, writer_b)
