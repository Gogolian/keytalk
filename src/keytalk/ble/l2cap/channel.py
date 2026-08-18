"""L2CAP Connection-Oriented Channel transport base — 4-byte length-prefixed stream.

On a reliable ordered stream (BLE LE CoC / Linux AF_BLUETOOTH socket / loopback),
every keytalk frame is prefixed by a 4-byte big-endian length field so the
receiver can reconstruct message boundaries without gap-filling.  The Go-Back-N
reliability layer used by GATT is not needed here.
"""

from __future__ import annotations

import asyncio
import struct
from typing import Optional

from ...transport import Transport, TransportClosed

__all__ = ["L2CAPStreamTransport"]

_LEN_STRUCT = struct.Struct(">I")
_LEN_SIZE = _LEN_STRUCT.size  # 4 bytes


class L2CAPStreamTransport(Transport):
    """Transport backed by an asyncio stream; adds 4-byte length-prefix framing.

    Subclasses call ``_attach(reader, writer)`` once the underlying stream is
    established, then ``_start_recv()`` to begin the read loop.
    """

    def __init__(self) -> None:
        super().__init__()
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._recv_task: Optional["asyncio.Task[None]"] = None
        self._closed = False

    def _attach(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._reader = reader
        self._writer = writer

    def _start_recv(self) -> None:
        self._recv_task = asyncio.ensure_future(self._recv_loop())

    async def _recv_loop(self) -> None:
        assert self._reader is not None
        try:
            while not self._closed:
                len_bytes = await self._reader.readexactly(_LEN_SIZE)
                (frame_len,) = _LEN_STRUCT.unpack(len_bytes)
                frame_bytes = await self._reader.readexactly(frame_len)
                await self._dispatch(frame_bytes)
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass

    async def send(self, frame: bytes) -> None:
        if self._writer is None or self._closed:
            raise TransportClosed("L2CAP stream is not connected")
        self._writer.write(_LEN_STRUCT.pack(len(frame)) + frame)
        await self._writer.drain()

    async def close(self) -> None:
        self._closed = True
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass
            self._recv_task = None
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
        self._reader = None
