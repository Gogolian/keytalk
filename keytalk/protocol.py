"""Keyboard-safe framing protocol for message transport."""

from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import math
import uuid


PROTOCOL_PREFIX = "KT1"
DEFAULT_CHUNK_SIZE = 96


class ProtocolError(ValueError):
    """Raised when a frame or message cannot be decoded."""


@dataclass(frozen=True)
class KeytalkMessage:
    """A decoded logical message."""

    kind: str
    message_id: str
    payload: str


def _encode_payload(payload: str) -> str:
    raw = payload.encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_payload(payload: str) -> str:
    padding = "=" * ((4 - (len(payload) % 4)) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload + padding)
    except Exception as exc:  # pragma: no cover - exact binascii type is implementation detail
        raise ProtocolError("invalid frame payload") from exc
    return raw.decode("utf-8")


def _checksum(parts: list[str]) -> str:
    joined = "|".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def encode_text(
    payload: str,
    *,
    kind: str = "text",
    message_id: str | None = None,
    max_chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[str]:
    """Encode text into keyboard-safe frame lines."""

    if not kind or "|" in kind:
        raise ProtocolError("kind must be non-empty and must not contain '|'")
    if max_chunk_size <= 0:
        raise ProtocolError("max_chunk_size must be positive")

    message_id = message_id or uuid.uuid4().hex[:12]
    if "|" in message_id:
        raise ProtocolError("message_id must not contain '|'")

    encoded = _encode_payload(payload)
    total = max(1, math.ceil(len(encoded) / max_chunk_size))
    chunks = [
        encoded[index * max_chunk_size : (index + 1) * max_chunk_size]
        for index in range(total)
    ]
    if not chunks:
        chunks = [""]

    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        parts = [
            PROTOCOL_PREFIX,
            kind,
            message_id,
            str(index),
            str(total),
            chunk,
        ]
        digest = _checksum(parts)
        lines.append("|".join(parts + [digest]))
    return lines


def _parse_frame(line: str) -> tuple[str, str, int, int, str]:
    stripped = line.strip()
    parts = stripped.split("|")
    if len(parts) != 7 or parts[0] != PROTOCOL_PREFIX:
        raise ProtocolError(f"invalid frame: {line!r}")

    _, kind, message_id, index_text, total_text, payload, digest = parts
    expected = _checksum(parts[:-1])
    if digest != expected:
        raise ProtocolError("frame checksum mismatch")

    try:
        index = int(index_text)
        total = int(total_text)
    except ValueError as exc:
        raise ProtocolError("frame positions must be integers") from exc

    if index < 1 or total < 1 or index > total:
        raise ProtocolError("frame positions are out of range")

    return kind, message_id, index, total, payload


def decode_lines(lines: list[str]) -> list[KeytalkMessage]:
    """Decode one or more complete logical messages from frame lines."""

    grouped: dict[tuple[str, str], dict[int, str]] = {}
    expected_totals: dict[tuple[str, str], int] = {}
    order: list[tuple[str, str]] = []

    for raw_line in lines:
        if not raw_line.strip():
            continue
        kind, message_id, index, total, payload = _parse_frame(raw_line)
        key = (kind, message_id)
        if key not in grouped:
            grouped[key] = {}
            expected_totals[key] = total
            order.append(key)
        if expected_totals[key] != total:
            raise ProtocolError("inconsistent frame totals for one message")
        if index in grouped[key]:
            raise ProtocolError("duplicate frame index")
        grouped[key][index] = payload

    messages: list[KeytalkMessage] = []
    for key in order:
        total = expected_totals[key]
        chunks = grouped[key]
        missing = [str(index) for index in range(1, total + 1) if index not in chunks]
        if missing:
            raise ProtocolError(f"incomplete message missing frame(s): {', '.join(missing)}")
        payload = "".join(chunks[index] for index in range(1, total + 1))
        messages.append(
            KeytalkMessage(
                kind=key[0],
                message_id=key[1],
                payload=_decode_payload(payload),
            )
        )
    return messages
