"""In-process L2CAP loopback using a socketpair — for tests and simulation.

:func:`create_l2cap_loopback` returns a connected ``(host, consumer)`` transport
pair backed by ``socket.socketpair()``.  No Bluetooth hardware required.
"""

from __future__ import annotations

import asyncio
import socket
from typing import Tuple

from .channel import L2CAPStreamTransport

__all__ = ["L2CAPLoopbackTransport", "create_l2cap_loopback"]


class L2CAPLoopbackTransport(L2CAPStreamTransport):
    """L2CAP stream transport pre-wired to a socket; ``start()`` begins the recv loop."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        super().__init__()
        self._attach(reader, writer)

    async def start(self) -> None:
        self._start_recv()


async def create_l2cap_loopback() -> Tuple[L2CAPLoopbackTransport, L2CAPLoopbackTransport]:
    """Return a ``(host_transport, consumer_transport)`` pair backed by a socket pair."""
    sock_a, sock_b = socket.socketpair()
    reader_a, writer_a = await asyncio.open_connection(sock=sock_a)
    reader_b, writer_b = await asyncio.open_connection(sock=sock_b)
    return L2CAPLoopbackTransport(reader_a, writer_a), L2CAPLoopbackTransport(reader_b, writer_b)
