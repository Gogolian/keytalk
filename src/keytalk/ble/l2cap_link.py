"""BLE L2CAP Connection-Oriented Channel (CoC) transport.

GATT notifications cap payloads at the ATT MTU (185 bytes on macOS) and are
fire-and-forget.  L2CAP CoC (Bluetooth 4.2+) is a reliable, credit-based
*stream* channel with negotiated MTUs up to 64 KiB - one or two orders of
magnitude more throughput than GATT, over the same radio and pairing model.

Layering:

    keytalk frames  ->  frame link (length-prefixed or datagram)  ->  L2CAP CoC

Two frame links are provided because the platform APIs differ:

* :class:`StreamLink` - length-prefixed frames over a *byte stream*.  This
  matches macOS ``CBL2CAPChannel`` (``NSInputStream``/``NSOutputStream``, which
  do not preserve message boundaries) and stream sockets.
* :class:`DatagramLink` - one keytalk frame per datagram on message-preserving
  sockets (Linux ``SOCK_SEQPACKET`` L2CAP).

The keytalk protocol layer (framing, ACK/retransmission, RESUME) is unchanged
and rides on top exactly as with GATT.

Platform status (verified against this machine's bindings, not assumed):

* **macOS**: ``CBPeripheralManager.publishL2CAPChannelWithEncryption_`` /
  ``unpublishL2CAPChannel_`` and ``CBL2CAPChannel`` (``PSM``,
  ``inputStream``, ``outputStream``) exist in the installed PyObjC
  CoreBluetooth bindings (asserted by the test-suite).  The end-to-end RF path
  must be validated with ``tools/l2cap_smoke.py`` on real hardware.
* **Linux**: BlueZ exposes L2CAP CoC as ``socket(AF_BLUETOOTH, SOCK_SEQPACKET,
  BTPROTO_L2CAP)`` sockets; :func:`open_l2cap_socket` wraps that.

The core links and :class:`LinkTransport` are fully unit-tested over real
socket pairs.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import struct
import threading
from typing import Callable, List, Optional

from ..transport import Transport, TransportClosed

__all__ = [
    "LinkClosed",
    "pack_frame",
    "FrameReader",
    "StreamLink",
    "DatagramLink",
    "LinkTransport",
    "open_l2cap_socket",
    "CBL2CAP_AVAILABLE",
    "MAC_PUBLISH_SELECTOR",
    "MAC_OPEN_SELECTOR",
]

logger = logging.getLogger("keytalk.ble.l2cap")

#: Verified PyObjC selector names (see tests/test_l2cap.py).
MAC_PUBLISH_SELECTOR = "publishL2CAPChannelWithEncryption_"
MAC_OPEN_SELECTOR = "openL2CAPChannel_"

MAX_FRAME = 0xFFFF


class LinkClosed(Exception):
    """Raised when the underlying channel is closed or hits EOF."""


# ---------------------------------------------------------------------------
# Frame links
# ---------------------------------------------------------------------------


def pack_frame(payload: bytes) -> bytes:
    """Length-prefix a frame (u16 big-endian) for transport over a byte stream."""

    if len(payload) > MAX_FRAME:
        raise ValueError(f"frame too large: {len(payload)} > {MAX_FRAME}")
    return struct.pack(">H", len(payload)) + payload


class FrameReader:
    """Incremental de-framer for length-prefixed frames.

    Feed raw stream bytes with :meth:`feed`; it returns every complete frame
    that became available.  Partial reads (frames split across ``recv`` calls,
    several frames in one ``recv``) are handled.
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> List[bytes]:
        self._buf += data
        frames: List[bytes] = []
        while len(self._buf) >= 2:
            (size,) = struct.unpack(">H", self._buf[:2])
            if len(self._buf) < 2 + size:
                break
            frames.append(bytes(self._buf[2 : 2 + size]))
            del self._buf[: 2 + size]
        return frames


class StreamLink:
    """Length-prefixed frame link over a blocking byte stream.

    ``read_exact(n)`` must return exactly ``n`` bytes or raise/return b"" on
    EOF; ``write_all(data)`` must write everything or raise.  This matches
    ``socket.recv``/``sendall`` wrappers and the macOS NSStream adapter.
    """

    def __init__(
        self,
        read_exact: Callable[[int], bytes],
        write_all: Callable[[bytes], None],
        close: Optional[Callable[[], None]] = None,
    ) -> None:
        self._read_exact = read_exact
        self._write_all = write_all
        self._close_fn = close
        self._send_lock = threading.Lock()

    def send_frame(self, payload: bytes) -> None:
        data = pack_frame(payload)
        with self._send_lock:
            try:
                self._write_all(data)
            except OSError as exc:
                raise LinkClosed(f"stream write failed: {exc}") from exc

    def recv_frame(self) -> bytes:
        header = self._read_exact(2)
        if len(header) < 2:
            raise LinkClosed("stream closed by peer")
        (size,) = struct.unpack(">H", header)
        payload = self._read_exact(size) if size else b""
        if len(payload) < size:
            raise LinkClosed("stream closed mid-frame")
        return payload

    def close(self) -> None:
        if self._close_fn is not None:
            try:
                self._close_fn()
            except OSError:  # pragma: no cover - best effort on teardown
                pass


def socket_read_exact(sock: socket.socket) -> Callable[[int], bytes]:
    """Return a blocking ``read_exact`` for a stream socket."""

    def read_exact(n: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < n:
            chunk = sock.recv(n - len(chunks))
            if not chunk:
                break
            chunks += chunk
        return bytes(chunks)

    return read_exact


def stream_link_from_socket(sock: socket.socket) -> StreamLink:
    """Wrap a connected *stream* socket into a :class:`StreamLink`."""

    return StreamLink(
        socket_read_exact(sock),
        sock.sendall,
        sock.close,
    )


class DatagramLink:
    """One frame per datagram on a message-preserving socket.

    L2CAP CoC on Linux is ``SOCK_SEQPACKET``: ``send``/``recv`` preserve
    message boundaries, so no length prefix is needed.  Frames larger than the
    negotiated L2CAP MTU (up to 64 KiB) are rejected by the socket itself -
    keytalk frames always fit within the 16-bit frame size.
    """

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock

    def send_frame(self, payload: bytes) -> None:
        try:
            self._sock.send(payload)
        except OSError as exc:
            raise LinkClosed(f"datagram send failed: {exc}") from exc

    def recv_frame(self) -> bytes:
        try:
            data = self._sock.recv(MAX_FRAME + 1)
        except OSError as exc:
            raise LinkClosed(f"datagram recv failed: {exc}") from exc
        if not data:
            raise LinkClosed("datagram link closed by peer")
        return data

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:  # pragma: no cover - best effort on teardown
            pass


# ---------------------------------------------------------------------------
# Transport over a frame link
# ---------------------------------------------------------------------------


class LinkTransport(Transport):
    """keytalk :class:`Transport` over a frame link (either flavour).

    A reader thread blocks on ``recv_frame`` and dispatches each frame onto the
    event loop in arrival order (via the transport's ordered dispatch queue).
    ``send`` runs the blocking link write in the default executor.
    """

    def __init__(self, link) -> None:  # noqa: ANN001 - link protocol above
        super().__init__()
        self._link = link
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._reader: Optional[threading.Thread] = None
        self._closed = threading.Event()

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._closed.clear()
        self._reader = threading.Thread(
            target=self._read_loop, name="keytalk-l2cap-rx", daemon=True
        )
        self._reader.start()

    async def send(self, frame: bytes) -> None:
        if self._closed.is_set():
            raise TransportClosed("L2CAP link is closed")
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, self._link.send_frame, bytes(frame)
            )
        except LinkClosed as exc:
            raise TransportClosed(str(exc)) from exc

    async def close(self) -> None:
        self._closed.set()
        self._link.close()
        reader = self._reader
        self._reader = None
        if reader is not None:
            await asyncio.get_running_loop().run_in_executor(None, reader.join, 1.0)
        await self._shutdown_dispatch()

    def _read_loop(self) -> None:
        assert self._loop is not None
        while not self._closed.is_set():
            try:
                frame = self._link.recv_frame()
            except LinkClosed:
                break
            except OSError:  # pragma: no cover - defensive
                break
            self._loop.call_soon_threadsafe(self._dispatch, frame)
        logger.debug("L2CAP reader thread exiting")


# ---------------------------------------------------------------------------
# Platform glue
# ---------------------------------------------------------------------------


def open_l2cap_socket(
    psm: int, address: str = "", *, server: bool = False
) -> socket.socket:
    """Open a Linux/BlueZ L2CAP CoC socket (``SOCK_SEQPACKET``).

    ``server=True`` binds and listens on ``psm`` (returns a listening socket to
    ``accept()`` on); otherwise connects to ``address``/``psm``.  Wrap the
    result in a :class:`DatagramLink` and pass it to :class:`LinkTransport`.
    """

    try:
        sock = socket.socket(
            socket.AF_BLUETOOTH, socket.SOCK_SEQPACKET, socket.BTPROTO_L2CAP
        )
    except (AttributeError, OSError) as exc:
        raise RuntimeError(
            "L2CAP sockets require BlueZ (Linux); this platform exposes L2CAP "
            "via CoreBluetooth instead - see tools/l2cap_smoke.py"
        ) from exc
    try:
        if server:
            sock.bind((address, psm))
            sock.listen(1)
        else:
            sock.connect((address, psm))
    except OSError:
        sock.close()
        raise
    return sock


# PyObjC CoreBluetooth L2CAP surface (macOS 10.13+).  The constants below are
# asserted against the real bindings in tests/test_l2cap.py so this glue can
# never silently rot against a PyObjC upgrade.
try:  # pragma: no cover - exercised on macOS only
    import CoreBluetooth  # noqa: F401 - loads the framework bundle
    import objc

    _CB_PERIPHERAL_MANAGER = objc.lookUpClass("CBPeripheralManager")
    _CB_L2CAP_CHANNEL = objc.lookUpClass("CBL2CAPChannel")
    CBL2CAP_AVAILABLE = (
        hasattr(_CB_PERIPHERAL_MANAGER, MAC_PUBLISH_SELECTOR)
        and hasattr(_CB_PERIPHERAL_MANAGER, "unpublishL2CAPChannel_")
        and hasattr(_CB_L2CAP_CHANNEL, "PSM")
        and hasattr(_CB_L2CAP_CHANNEL, "inputStream")
        and hasattr(_CB_L2CAP_CHANNEL, "outputStream")
    )
except Exception:  # pragma: no cover - non-macOS or no PyObjC
    CBL2CAP_AVAILABLE = False


class MacStreamPair:
    """Frame link adapter for ``CBL2CAPChannel``'s NSInputStream/NSOutputStream.

    NSStreams are byte streams without message boundaries, so frames are
    length-prefixed (:class:`StreamLink`).  This adapter polls the streams from
    a worker thread (``hasBytesAvailable``/``read_maxLength_``) which avoids
    run-loop delegate plumbing.

    .. warning::
       Validated against the PyObjC API surface, but the RF path needs a real
       second device - run ``tools/l2cap_smoke.py`` before relying on it.
    """

    def __init__(self, in_stream, out_stream, max_read: int = 65536) -> None:  # noqa: ANN001
        self._in = in_stream
        self._out = out_stream
        self._max_read = max_read

    def read_exact(self, n: int) -> bytes:
        import time

        chunks = bytearray()
        while len(chunks) < n:
            if self._in.hasBytesAvailable():
                chunk = bytes(self._in.read_maxLength_(min(n - len(chunks), self._max_read)))
                if not chunk:
                    break
                chunks += chunk
            else:
                time.sleep(0.001)
        return bytes(chunks)

    def write_all(self, data: bytes) -> None:
        view = memoryview(data)
        while view:
            written = self._out.write_maxLength_(bytes(view), len(view))
            if written <= 0:
                raise OSError("NSOutputStream refused the write")
            view = view[written:]

    def close(self) -> None:
        for stream in (self._in, self._out):
            try:
                stream.close()
            except Exception:  # noqa: BLE001 - best effort on teardown
                pass

    def as_link(self) -> StreamLink:
        return StreamLink(self.read_exact, self.write_all, self.close)
