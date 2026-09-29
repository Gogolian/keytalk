"""Wire protocol for the keytalk BLE bridge.

Bluetooth LE GATT characteristics can only carry small payloads (the default
ATT MTU is 23 bytes, i.e. 20 usable bytes, and even a negotiated MTU is at most
a few hundred bytes).  Prompts and LLM responses are far larger than that, so
every logical message has to be split into small *frames* that are sent one at
a time over a characteristic and then reassembled on the other side.

This module is completely transport agnostic: it knows nothing about Bluetooth.
That makes the chunking / framing / reassembly logic easy to test exhaustively
with plain bytes, which is exactly where subtle bugs tend to live.

Frame layout (all integers big-endian)::

    offset  size  field
    0       1     version
    1       1     message type
    2       1     flags
    3       2     message id
    5       2     sequence number
    7       N     payload

A *message* is an ordered run of frames that share the same ``message_id``.
The first frame has the ``START`` flag set, the last frame has the ``END`` flag
set (a one-frame message has both).  Sequence numbers start at 0 and increase by
exactly one per frame, which lets the receiver detect drops or reordering.
"""

from __future__ import annotations

import struct
import time
import zlib
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import Dict, List, Optional

__all__ = [
    "PROTOCOL_VERSION",
    "HEADER_SIZE",
    "DEFAULT_ATT_MTU",
    "CHECKSUM_SIZE",
    "MessageType",
    "Flags",
    "Frame",
    "CompleteMessage",
    "ProtocolError",
    "max_payload_for_mtu",
    "chunk_message",
    "Reassembler",
    "FrameStreamEncoder",
    "compute_crc32",
    "encode_select_payload",
    "decode_select_payload",
    "RESUME_UNKNOWN",
    "encode_max_payload",
    "decode_max_payload",
    "encode_resume",
    "decode_resume",
    "encode_cancel",
    "decode_cancel",
]

PROTOCOL_VERSION = 1

# struct format for the fixed-size header: version, type, flags, id, seq.
_HEADER_STRUCT = struct.Struct(">BBBHH")
HEADER_SIZE = _HEADER_STRUCT.size  # 7 bytes

# The default ATT MTU defined by the Bluetooth spec.  3 bytes are consumed by
# the ATT notification/write opcode + handle, leaving 20 usable bytes.
DEFAULT_ATT_MTU = 23
_ATT_OVERHEAD = 3

_MAX_UINT16 = 0xFFFF


class MessageType(IntEnum):
    """Logical kind of a message."""

    PROMPT = 1
    RESPONSE = 2
    ERROR = 3
    CANCEL = 4
    ACK = 5
    LIST_MODELS = 6
    # 7 = DELTA_PROMPT — reserved (unused; the value stays unallocated so it
    #     can never be reassigned to a different meaning on the wire).
    # Phase 1 — capability handshake
    HELLO = 8    # consumer → host: frame payload size it can accept
    CAPS = 9     # reserved: the host advertises modes on the CAPS GATT
                #     characteristic instead of over the frame protocol
    SELECT = 10  # consumer → host: select a transfer mode
    # 11 = NAK — reserved (failures ride on ERROR; see _send_nack)
    # Reliability / structured-request extensions (keytalk)
    PING = 12    # consumer → host: keepalive probe
    PONG = 13    # host → consumer: keepalive reply
    RESUME = 14  # consumer → host: retransmit message from a sequence number
    CHAT = 15    # consumer → host: structured chat messages (JSON)
    TIMINGS = 16  # generation-statistics trailer frames (host → consumer)


#: Error text used by the host when a RESUME references a message it no longer
#: knows about, so the consumer can tell "request is gone" (retry from scratch)
#: apart from ordinary backend errors.
RESUME_UNKNOWN = "keytalk:unknown-request"


class Flags(IntFlag):
    """Per-frame flags marking message boundaries."""

    NONE = 0
    START = 1
    END = 2
    COMPRESSED = 4  # Payload is zlib-compressed
    # 8 = DELTA — reserved (unused)
    CHECKSUM = 16   # END frame carries a 4-byte CRC32 trailer after the payload


class ProtocolError(Exception):
    """Raised when a frame or a sequence of frames violates the protocol."""


def max_payload_for_mtu(mtu: int = DEFAULT_ATT_MTU) -> int:
    """Return the largest frame payload that fits in a single GATT packet.

    ``mtu`` is the negotiated ATT MTU.  We subtract the ATT opcode/handle
    overhead and our own fixed header to find how many payload bytes remain.
    """

    usable = mtu - _ATT_OVERHEAD - HEADER_SIZE
    if usable <= 0:
        raise ValueError(
            f"MTU {mtu} is too small to carry any payload "
            f"(need > {_ATT_OVERHEAD + HEADER_SIZE})"
        )
    return usable


@dataclass(frozen=True)
class Frame:
    """A single framed packet ready to be written to a characteristic."""

    msg_type: MessageType
    message_id: int
    seq: int
    payload: bytes = b""
    flags: Flags = Flags.NONE
    version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if not 0 <= self.message_id <= _MAX_UINT16:
            raise ValueError(f"message_id out of range: {self.message_id}")
        if not 0 <= self.seq <= _MAX_UINT16:
            raise ValueError(f"seq out of range: {self.seq}")
        if not 0 <= self.version <= 0xFF:
            raise ValueError(f"version out of range: {self.version}")

    @property
    def is_start(self) -> bool:
        return bool(self.flags & Flags.START)

    @property
    def is_end(self) -> bool:
        return bool(self.flags & Flags.END)

    def encode(self) -> bytes:
        """Serialise the frame to bytes."""

        header = _HEADER_STRUCT.pack(
            self.version,
            int(self.msg_type),
            int(self.flags),
            self.message_id,
            self.seq,
        )
        return header + self.payload

    @classmethod
    def decode(cls, data: bytes) -> "Frame":
        """Parse a frame from bytes, validating its header."""

        if len(data) < HEADER_SIZE:
            raise ProtocolError(
                f"frame too short: {len(data)} bytes, need >= {HEADER_SIZE}"
            )
        version, raw_type, raw_flags, message_id, seq = _HEADER_STRUCT.unpack(
            data[:HEADER_SIZE]
        )
        if version != PROTOCOL_VERSION:
            raise ProtocolError(
                f"unsupported protocol version: {version} "
                f"(expected {PROTOCOL_VERSION})"
            )
        try:
            msg_type = MessageType(raw_type)
        except ValueError as exc:
            raise ProtocolError(f"unknown message type: {raw_type}") from exc
        # Flags is an IntFlag; reject bits we do not understand so that a
        # corrupted byte does not silently look like a valid boundary marker.
        known = int(Flags.START | Flags.END | Flags.COMPRESSED | Flags.CHECKSUM)
        if raw_flags & ~known:
            raise ProtocolError(f"unknown flag bits set: {raw_flags:#04x}")
        return cls(
            msg_type=msg_type,
            message_id=message_id,
            seq=seq,
            payload=bytes(data[HEADER_SIZE:]),
            flags=Flags(raw_flags),
            version=version,
        )


@dataclass(frozen=True)
class CompleteMessage:
    """A fully reassembled message."""

    msg_type: MessageType
    message_id: int
    payload: bytes

    def text(self, encoding: str = "utf-8") -> str:
        return self.payload.decode(encoding)


# CRC32 trailer appended to the END frame payload when Flags.CHECKSUM is set.
_CRC_STRUCT = struct.Struct(">I")
CHECKSUM_SIZE = _CRC_STRUCT.size  # 4 bytes


def compute_crc32(data: bytes) -> int:
    """Return the IEEE CRC32 of *data* as an unsigned 32-bit integer."""
    return zlib.crc32(data) & 0xFFFFFFFF


def chunk_message(
    msg_type: MessageType,
    message_id: int,
    payload: bytes,
    max_payload_size: int,
    *,
    checksum: bool = False,
    start_flags: Flags = Flags.NONE,
) -> List[Frame]:
    """Split ``payload`` into an ordered list of frames.

    The first frame carries ``START`` and the last carries ``END``.  An empty
    payload yields exactly one frame with both flags set.  When ``checksum``
    is ``True``, a 4-byte CRC32 trailer is appended to the last frame's payload
    and the ``CHECKSUM`` flag is set on that frame; the :class:`Reassembler`
    verifies the trailer and strips it before returning the message.
    ``start_flags`` are OR'd into the ``START`` frame (e.g. ``COMPRESSED``),
    which is how whole-message metadata is attached without post-processing
    the returned frames.
    """

    if max_payload_size <= 0:
        raise ValueError("max_payload_size must be positive")

    wire_payload = payload
    if checksum:
        wire_payload = payload + _CRC_STRUCT.pack(compute_crc32(payload))

    # Always emit at least one frame, even for an empty payload.
    pieces: List[bytes] = [
        wire_payload[i : i + max_payload_size]
        for i in range(0, len(wire_payload), max_payload_size)
    ] or [b""]

    last = len(pieces) - 1
    if last > _MAX_UINT16:
        raise ValueError(
            f"payload needs {len(pieces)} frames which overflows the 16-bit "
            "sequence space"
        )

    frames: List[Frame] = []
    for seq, piece in enumerate(pieces):
        flags = Flags.NONE
        if seq == 0:
            flags |= Flags.START | start_flags
        if seq == last:
            flags |= Flags.END
            if checksum:
                flags |= Flags.CHECKSUM
        frames.append(
            Frame(
                msg_type=msg_type,
                message_id=message_id,
                seq=seq,
                payload=piece,
                flags=flags,
            )
        )
    return frames


class _Buffer:
    """Accumulates frames belonging to a single in-flight message."""

    __slots__ = ("msg_type", "chunks", "next_seq", "compressed", "has_checksum", "at")

    def __init__(self, msg_type: MessageType, compressed: bool = False) -> None:
        self.msg_type = msg_type
        self.chunks: List[bytes] = []
        self.next_seq = 0
        self.compressed = compressed
        self.has_checksum = False
        self.at = time.monotonic()


class Reassembler:
    """Stateful reassembler that turns frames back into whole messages.

    Feed frames in arrival order with :meth:`feed`.  When a message completes
    (its ``END`` frame arrives) the assembled :class:`CompleteMessage` is
    returned; otherwise ``None`` is returned.  Multiple messages may be in
    flight concurrently as long as they use distinct ``message_id`` values.
    """

    def __init__(self) -> None:
        self._buffers: Dict[int, _Buffer] = {}
        self._feeds = 0

    def feed(self, frame: Frame) -> Optional[CompleteMessage]:
        self._feeds += 1
        if self._feeds % 64 == 0:
            self.purge_stale()
        buf = self._buffers.get(frame.message_id)

        if frame.is_start:
            # A new message always resets any stale partial buffer for this id.
            compressed = bool(frame.flags & Flags.COMPRESSED)
            buf = _Buffer(frame.msg_type, compressed)
            self._buffers[frame.message_id] = buf
        elif buf is None:
            raise ProtocolError(
                f"received non-start frame for unknown message "
                f"{frame.message_id} (seq={frame.seq})"
            )

        if frame.msg_type != buf.msg_type:
            raise ProtocolError(
                f"message {frame.message_id} changed type mid-stream: "
                f"{buf.msg_type.name} -> {frame.msg_type.name}"
            )
        if frame.seq != buf.next_seq:
            raise ProtocolError(
                f"out-of-order frame for message {frame.message_id}: "
                f"expected seq {buf.next_seq}, got {frame.seq}"
            )

        buf.chunks.append(frame.payload)
        buf.next_seq += 1
        buf.at = time.monotonic()

        if frame.is_end:
            if bool(frame.flags & Flags.CHECKSUM):
                buf.has_checksum = True
            del self._buffers[frame.message_id]
            payload = b"".join(buf.chunks)
            # Verify and strip the CRC32 trailer before decompression.
            if buf.has_checksum:
                if len(payload) < CHECKSUM_SIZE:
                    raise ProtocolError(
                        f"CHECKSUM flag set on message {frame.message_id} "
                        "but payload is too short to contain the CRC trailer"
                    )
                expected_crc, = _CRC_STRUCT.unpack(payload[-CHECKSUM_SIZE:])
                actual_data = payload[:-CHECKSUM_SIZE]
                computed_crc = compute_crc32(actual_data)
                if computed_crc != expected_crc:
                    raise ProtocolError(
                        f"CRC32 mismatch for message {frame.message_id}: "
                        f"expected {expected_crc:#010x}, got {computed_crc:#010x}"
                    )
                payload = actual_data
            # Decompress if the START frame had the COMPRESSED flag.
            if buf.compressed:
                try:
                    payload = zlib.decompress(payload)
                except zlib.error as exc:
                    raise ProtocolError(
                        f"failed to decompress message {frame.message_id}"
                    ) from exc
            return CompleteMessage(
                msg_type=buf.msg_type,
                message_id=frame.message_id,
                payload=payload,
            )
        return None

    def discard(self, message_id: int) -> None:
        """Drop any partial state for ``message_id`` (e.g. after a timeout)."""

        self._buffers.pop(message_id, None)

    def purge_stale(self, max_age: float = 300.0) -> int:
        """Drop partial buffers untouched for ``max_age`` seconds.

        A half-received message whose peer vanished would otherwise be buffered
        forever.  Returns the number of buffers dropped.
        """

        now = time.monotonic()
        stale = [
            message_id
            for message_id, buf in self._buffers.items()
            if now - buf.at > max_age
        ]
        for message_id in stale:
            del self._buffers[message_id]
        return len(stale)

    @property
    def pending(self) -> int:
        """Number of partially-received messages currently buffered."""

        return len(self._buffers)


class FrameStreamEncoder:
    """Incrementally encode a streamed message into frames.

    Unlike :func:`chunk_message`, the total payload is not known up front (LLM
    tokens arrive one at a time).  Call :meth:`push` with each piece of data to
    get back any full frames that can be emitted so far, then call
    :meth:`finish` exactly once to flush the remainder with the ``END`` flag.

    With ``compressed=True`` the frame payloads form a single zlib stream (the
    ``COMPRESSED`` flag is set on the START frame) and are compressed
    incrementally; call :meth:`flush` before switching to uncompressed frames
    (ERROR / TIMINGS trailers) so no compressed bytes are pending across the
    type change.  With ``checksum=True`` a CRC32 trailer is appended to the
    final frame (``CHECKSUM`` flag), verified by the :class:`Reassembler`.

    The invariant maintained is that ``finish`` always emits at least one frame
    with the ``END`` flag on the last one, even when the payload size is an
    exact multiple of ``max_payload_size``.
    """

    def __init__(
        self,
        msg_type: MessageType,
        message_id: int,
        max_payload_size: int,
        *,
        compressed: bool = False,
        level: int = 6,
        checksum: bool = False,
        start_flags: Flags = Flags.NONE,
    ) -> None:
        if max_payload_size <= 0:
            raise ValueError("max_payload_size must be positive")
        self._msg_type = msg_type
        self._message_id = message_id
        self._max = max_payload_size
        self._use_checksum = checksum
        self._buf = bytearray()
        self._seq = 0
        self._started = False
        self._finished = False
        self._running_crc = 0  # updated as each payload chunk is emitted
        self._start_flags = start_flags
        self._compressed = compressed
        self._comp = zlib.compressobj(level) if compressed else None

    @property
    def next_seq(self) -> int:
        """Sequence number the next emitted frame will carry."""

        return self._seq

    @property
    def has_started(self) -> bool:
        """Whether a START frame has already been emitted."""

        return self._started

    def _emit(self, payload: bytes, last: bool) -> Frame:
        flags = Flags.NONE
        if not self._started:
            flags |= Flags.START | self._start_flags
            if self._comp is not None:
                flags |= Flags.COMPRESSED
            self._started = True
        if last:
            flags |= Flags.END
        if self._seq > _MAX_UINT16:
            raise ProtocolError("stream exceeded the 16-bit sequence space")
        # Update running CRC with this chunk's data (before any trailer).
        self._running_crc = zlib.crc32(payload, self._running_crc) & 0xFFFFFFFF
        actual_payload = payload
        if last and self._use_checksum:
            flags |= Flags.CHECKSUM
            actual_payload = payload + _CRC_STRUCT.pack(self._running_crc)
        frame = Frame(
            msg_type=self._msg_type,
            message_id=self._message_id,
            seq=self._seq,
            payload=actual_payload,
            flags=flags,
        )
        self._seq += 1
        return frame

    def push(self, data: bytes) -> List[Frame]:
        """Append ``data`` and return any frames that are now full.

        We keep at most ``max_payload_size`` bytes buffered so that
        :meth:`finish` always has a final frame to emit.
        """

        if self._finished:
            raise RuntimeError("cannot push after finish()")
        if self._comp is not None:
            data = self._comp.compress(data)
        self._buf += data
        return self._drain()

    def _drain(self) -> List[Frame]:
        """Emit full frames from the buffer, keeping any partial remainder."""

        frames: List[Frame] = []
        while len(self._buf) > self._max:
            chunk = bytes(self._buf[: self._max])
            del self._buf[: self._max]
            frames.append(self._emit(chunk, last=False))
        return frames

    def flush(self) -> List[Frame]:
        """Force-emit the buffered bytes as frames *without* ending the message.

        Used at the data -> error/trailer transition, where all pending bytes
        (including pending compressor state) must be flushed out while frames
        are still data-typed.
        """

        if self._finished:
            raise RuntimeError("cannot flush after finish()")
        if not self._started and not self._buf:
            return []  # nothing was ever pushed: do not start the stream
        if self._comp is not None:
            # Z_SYNC_FLUSH emits every pending compressed byte without ending
            # the zlib stream.
            self._buf += self._comp.flush(zlib.Z_SYNC_FLUSH)
        frames: List[Frame] = []
        while self._buf:
            chunk = bytes(self._buf[: self._max])
            del self._buf[: self._max]
            frames.append(self._emit(chunk, last=False))
        return frames

    def finish(self) -> List[Frame]:
        """Flush the buffered remainder as final (``END``) frame(s)."""

        if self._finished:
            raise RuntimeError("finish() called twice")
        self._finished = True
        if self._comp is not None:
            self._buf += self._comp.flush()  # Z_FINISH: terminates the stream
        pieces = [
            bytes(self._buf[i : i + self._max])
            for i in range(0, len(self._buf), self._max)
        ] or [b""]
        self._buf.clear()
        last = len(pieces) - 1
        return [
            self._emit(piece, last=(index == last))
            for index, piece in enumerate(pieces)
        ]


# SELECT payload: 1-byte mode_id (uint8) + 2-byte MTU (uint16 big-endian).
# 3 bytes total — always fits in a single legacy frame.
_SELECT_STRUCT = struct.Struct(">BH")


def encode_select_payload(mode_id: int, mtu: int) -> bytes:
    """Encode a SELECT frame payload."""
    return _SELECT_STRUCT.pack(mode_id, mtu)


def decode_select_payload(data: bytes) -> tuple[int, int]:
    """Decode a SELECT frame payload → (mode_id, mtu)."""
    if len(data) < _SELECT_STRUCT.size:
        raise ProtocolError(
            f"SELECT payload too short: {len(data)} bytes, need {_SELECT_STRUCT.size}"
        )
    mode_id, mtu = _SELECT_STRUCT.unpack(data[: _SELECT_STRUCT.size])
    return mode_id, mtu


# ---------------------------------------------------------------------------
# Control-message payload helpers (keytalk reliability extensions)
#
# HELLO, RESUME and CANCEL are small enough to fit in a single frame even at
# the default 23-byte ATT MTU, so their payloads are packed binary rather than
# JSON.
# ---------------------------------------------------------------------------


def encode_max_payload(max_payload: int) -> bytes:
    """Encode the HELLO payload: the largest frame payload the peer accepts."""

    if not 0 < max_payload <= _MAX_UINT16:
        raise ValueError(f"max_payload out of range: {max_payload}")
    return struct.pack(">H", max_payload)


def decode_max_payload(payload: bytes) -> int:
    """Decode a HELLO payload into a maximum frame payload size."""

    if len(payload) < 2:
        raise ProtocolError("HELLO payload too short")
    return struct.unpack(">H", payload[:2])[0]


def encode_resume(target_id: int, ack_seq: int) -> bytes:
    """Encode a RESUME payload: retransmit message ``target_id`` from ``ack_seq``."""

    return struct.pack(">HH", target_id, ack_seq)


def decode_resume(payload: bytes) -> tuple[int, int]:
    """Decode a RESUME payload into ``(target_id, ack_seq)``."""

    if len(payload) < 4:
        raise ProtocolError("RESUME payload too short")
    return struct.unpack(">HH", payload[:4])


def encode_cancel(target_id: int) -> bytes:
    """Encode a CANCEL payload: abort message ``target_id``."""

    return struct.pack(">H", target_id)


def decode_cancel(payload: bytes) -> int:
    """Decode a CANCEL payload into the target message id."""

    if len(payload) < 2:
        raise ProtocolError("CANCEL payload too short")
    return struct.unpack(">H", payload[:2])[0]
