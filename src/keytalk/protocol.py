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

import hashlib
import struct
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
    "compute_message_checksum",
    "compute_crc32",
    "encode_delta_payload",
    "decode_delta_payload",
    "encode_select_payload",
    "decode_select_payload",
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
    DELTA_PROMPT = 7  # Incremental prompt with checksum reference
    # Phase 1 — capability handshake
    HELLO = 8    # consumer → host: initiate handshake (reserved, not yet sent)
    CAPS = 9     # host → consumer: capability advertisement
    SELECT = 10  # consumer → host: select a transfer mode
    NAK = 11     # either direction: reject / signal mismatch (Phase 2+)


class Flags(IntFlag):
    """Per-frame flags marking message boundaries."""

    NONE = 0
    START = 1
    END = 2
    COMPRESSED = 4  # Payload is zlib-compressed
    DELTA = 8       # Message is a delta (prefix + new content)
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
        known = int(Flags.START | Flags.END | Flags.COMPRESSED | Flags.DELTA | Flags.CHECKSUM)
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
) -> List[Frame]:
    """Split ``payload`` into an ordered list of frames.

    The first frame carries ``START`` and the last carries ``END``.  An empty
    payload yields exactly one frame with both flags set.  When ``checksum``
    is ``True``, a 4-byte CRC32 trailer is appended to the last frame's payload
    and the ``CHECKSUM`` flag is set on that frame; the :class:`Reassembler`
    verifies the trailer and strips it before returning the message.
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
            flags |= Flags.START
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

    __slots__ = ("msg_type", "chunks", "next_seq", "compressed", "has_checksum")

    def __init__(self, msg_type: MessageType, compressed: bool = False) -> None:
        self.msg_type = msg_type
        self.chunks: List[bytes] = []
        self.next_seq = 0
        self.compressed = compressed
        self.has_checksum = False


class Reassembler:
    """Stateful reassembler that turns frames back into whole messages.

    Feed frames in arrival order with :meth:`feed`.  When a message completes
    (its ``END`` frame arrives) the assembled :class:`CompleteMessage` is
    returned; otherwise ``None`` is returned.  Multiple messages may be in
    flight concurrently as long as they use distinct ``message_id`` values.
    """

    def __init__(self) -> None:
        self._buffers: Dict[int, _Buffer] = {}

    def feed(self, frame: Frame) -> Optional[CompleteMessage]:
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

    The invariant maintained is that ``finish`` always emits the final frame, so
    the ``END`` marker is guaranteed to land on the last frame even when the
    payload size is an exact multiple of ``max_payload_size``.
    """

    def __init__(
        self,
        msg_type: MessageType,
        message_id: int,
        max_payload_size: int,
        *,
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
        frames: List[Frame] = []
        self._buf += data
        while len(self._buf) > self._max:
            chunk = bytes(self._buf[: self._max])
            del self._buf[: self._max]
            frames.append(self._emit(chunk, last=False))
        return frames

    def finish(self) -> List[Frame]:
        """Flush the buffered remainder as the final (``END``) frame."""

        if self._finished:
            raise RuntimeError("finish() called twice")
        self._finished = True
        chunk = bytes(self._buf)
        self._buf.clear()
        return [self._emit(chunk, last=True)]


def compute_message_checksum(data: bytes) -> str:
    """Compute a SHA-256 checksum for message content.
    
    Returns the first 16 hex characters (64 bits) for bandwidth efficiency.
    """
    return hashlib.sha256(data).hexdigest()[:16]


def encode_delta_payload(checksum_prefix: str, delta_content: bytes) -> bytes:
    """Encode a delta message payload with checksum + new content.
    
    Format: <checksum_len><checksum><delta_content>
    - checksum_len: 1 byte indicating length of checksum string
    - checksum: UTF-8 encoded checksum string
    - delta_content: the new bytes to append
    """
    checksum_bytes = checksum_prefix.encode('utf-8')
    if len(checksum_bytes) > 255:
        raise ValueError("Checksum too long")
    return bytes([len(checksum_bytes)]) + checksum_bytes + delta_content


def decode_delta_payload(payload: bytes) -> tuple[str, bytes]:
    """Decode a delta message payload.
    
    Returns: (checksum_prefix, delta_content)
    """
    if len(payload) < 1:
        raise ProtocolError("Delta payload too short")
    checksum_len = payload[0]
    if len(payload) < 1 + checksum_len:
        raise ProtocolError("Delta payload truncated")
    checksum = payload[1:1 + checksum_len].decode('utf-8')
    delta_content = payload[1 + checksum_len:]
    return checksum, delta_content


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
