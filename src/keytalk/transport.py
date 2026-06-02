"""Transport abstraction and an in-memory loopback implementation.

A :class:`Transport` is a bidirectional, frame-oriented byte pipe.  The host and
the consumer each own one transport.  Whatever one side ``send``s is delivered
to the other side's registered receive callback.

The real implementations live in :mod:`keytalk.ble`; the in-memory transport in
this module backs the test-suite and lets the host/consumer logic be exercised
end-to-end without any Bluetooth hardware.
"""

from __future__ import annotations

import abc
import asyncio
from typing import Awaitable, Callable, List, Optional, Tuple

__all__ = ["Transport", "TransportClosed", "InMemoryTransport", "create_loopback"]

ReceiveCallback = Callable[[bytes], Awaitable[None]]


class TransportClosed(Exception):
    """Raised when sending on a transport that has been closed."""


class Transport(abc.ABC):
    """Bidirectional frame transport.

    Subclasses move opaque ``bytes`` frames between two endpoints.  Framing and
    chunking are the caller's responsibility (see :mod:`keytalk.protocol`).
    """

    def __init__(self) -> None:
        self._receive_cb: Optional[ReceiveCallback] = None

    def on_receive(self, callback: ReceiveCallback) -> None:
        """Register the coroutine invoked for every received frame."""

        self._receive_cb = callback

    async def _dispatch(self, frame: bytes) -> None:
        """Deliver an inbound frame to the registered callback, if any."""

        if self._receive_cb is not None:
            await self._receive_cb(frame)

    @abc.abstractmethod
    async def start(self) -> None:
        """Bring the transport up (connect / advertise / subscribe)."""

    @abc.abstractmethod
    async def send(self, frame: bytes) -> None:
        """Send a single frame to the peer."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Tear the transport down."""

    async def __aenter__(self) -> "Transport":
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()


class InMemoryTransport(Transport):
    """A loopback transport linked to a peer in the same process.

    Frames sent here are delivered to the peer's receive callback on the running
    event loop.  Delivery is scheduled via ``call_soon`` so it behaves
    asynchronously, like a real link, rather than re-entrantly.
    """

    def __init__(self, name: str = "") -> None:
        super().__init__()
        self.name = name
        self._peer: Optional["InMemoryTransport"] = None
        self._closed = False
        #: Every frame this endpoint has sent, in order (handy for assertions).
        self.sent: List[bytes] = []
        self._inflight: "set[asyncio.Task[None]]" = set()

    def link(self, peer: "InMemoryTransport") -> None:
        self._peer = peer

    async def start(self) -> None:  # noqa: D401 - nothing to do for loopback
        self._closed = False

    async def send(self, frame: bytes) -> None:
        if self._closed:
            raise TransportClosed(f"transport {self.name!r} is closed")
        if self._peer is None:
            raise RuntimeError("transport is not linked to a peer")
        # Copy so the receiver cannot observe later mutation of the buffer.
        data = bytes(frame)
        self.sent.append(data)
        await self._peer._deliver(data)

    async def _deliver(self, frame: bytes) -> None:
        if self._closed:
            # Peer went away; silently drop, mirroring a lost BLE link.
            return
        # Schedule delivery on the loop so send() returns promptly and callbacks
        # run as independent tasks, like notifications from a real peripheral.
        task = asyncio.create_task(self._dispatch(frame))
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    async def close(self) -> None:
        self._closed = True
        # Let any already-scheduled deliveries finish so callers awaiting a
        # response are not stranded by an abrupt teardown.
        if self._inflight:
            await asyncio.gather(*self._inflight, return_exceptions=True)


def create_loopback(
    names: Tuple[str, str] = ("host", "consumer"),
) -> Tuple[InMemoryTransport, InMemoryTransport]:
    """Create a linked pair of in-memory transports ``(host, consumer)``."""

    a = InMemoryTransport(names[0])
    b = InMemoryTransport(names[1])
    a.link(b)
    b.link(a)
    return a, b
