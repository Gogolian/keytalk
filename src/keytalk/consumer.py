"""The CONSUMER side: send a prompt, receive the streamed answer.

The consumer owns a :class:`~keytalk.transport.Transport` (in production a BLE
central connected to the host).  :meth:`ConsumerClient.generate` returns the
whole answer; :meth:`ConsumerClient.stream` yields response text incrementally
as frames arrive.  Each outstanding request is tracked by ``message_id`` so the
client can multiplex several prompts over one link.
"""

from __future__ import annotations

import asyncio
import codecs
import itertools
import json
import logging
import zlib
from typing import AsyncIterator, Dict, List, Optional

from .modes import LEGACY_PROFILE, Mode, ProfileConfig, negotiate_mode, mode_id_for, make_l2cap_coc_profile
from .protocol import (
    DEFAULT_ATT_MTU,
    CHECKSUM_SIZE,
    Flags,
    Frame,
    MessageType,
    ProtocolError,
    Reassembler,
    chunk_message,
    max_payload_for_mtu,
    compute_message_checksum,
    encode_delta_payload,
    decode_delta_payload,
    encode_select_payload,
)
from .reliability import make_ack_frame
from .transport import Transport

__all__ = ["ConsumerClient", "RemoteError", "_PendingRequest"]

logger = logging.getLogger("keytalk.consumer")

DEFAULT_TIMEOUT = 300.0


class RemoteError(Exception):
    """Raised when the host returns an ERROR message for a request."""


class _PendingRequest:
    """Tracks reassembly and streaming for one in-flight request.

    Response (or error) frames are validated for ordering and pushed onto a
    queue so consumers can stream them.  Out-of-order frames (which happen when
    a notification is dropped and later retransmitted) are buffered and replayed
    in sequence rather than treated as a fatal error.  Completion is signalled
    with a sentinel so an async iterator terminates cleanly.

    When ``reassemble=True`` (used by FAST_GATT), all frames are fed through a
    :class:`Reassembler` which handles decompression and CRC32 verification;
    the decoded payload is emitted as a single chunk once the END frame arrives.
    """

    _END = object()

    def __init__(self, message_id: int, *, reassemble: bool = False) -> None:
        self.message_id = message_id
        self._queue: "asyncio.Queue[object]" = asyncio.Queue()
        self._next_seq = 0
        self._reorder: Dict[int, Frame] = {}
        self._started = False
        self._done = False
        self._reassemble = reassemble
        self._inner: Optional[Reassembler] = Reassembler() if reassemble else None
        self._decompressor: Optional[zlib.Decompress] = None
        self._running_crc: int = 0

    @property
    def ack_seq(self) -> int:
        """Next contiguous sequence number expected (cumulative ACK value)."""

        return self._next_seq

    def feed(self, frame: Frame) -> None:
        """Validate and enqueue an inbound frame for this request.

        Frames may arrive out of order or be duplicated by retransmission.  We
        buffer anything ahead of the next expected sequence number and drop
        anything already consumed, so only contiguous frames are delivered.
        """

        if self._done:
            logger.debug("frame for already-finished request %s", self.message_id)
            return

        # Duplicate of an already-consumed or already-buffered frame: ignore.
        if frame.seq < self._next_seq or frame.seq in self._reorder:
            logger.debug(
                "ignoring duplicate frame seq=%d for %s", frame.seq, self.message_id
            )
            return

        self._reorder[frame.seq] = frame
        # Deliver every frame that is now contiguous from _next_seq onwards.
        while self._next_seq in self._reorder:
            self._process(self._reorder.pop(self._next_seq))
            if self._done:
                break
            self._next_seq += 1

    def _process(self, frame: Frame) -> None:
        if frame.seq == 0:
            if not frame.is_start:
                self._fail(
                    ProtocolError(
                        f"response {self.message_id} started without a START frame"
                    )
                )
                return
            self._started = True
            if not self._reassemble and (frame.flags & Flags.COMPRESSED):
                self._decompressor = zlib.decompressobj()

        if self._reassemble:
            # Route through the inner Reassembler for decompression + CRC32.
            assert self._inner is not None
            try:
                result = self._inner.feed(frame)
            except ProtocolError as exc:
                self._fail(exc)
                return
            if result is not None:
                if result.msg_type == MessageType.ERROR:
                    self._queue.put_nowait(("error", result.payload))
                else:
                    self._queue.put_nowait(("data", result.payload))
                self._next_seq += 1  # mirrors legacy END handling
                self._done = True
                self._queue.put_nowait(self._END)
            return

        # Legacy streaming path: emit each frame payload immediately.
        if frame.msg_type == MessageType.ERROR:
            # Error payloads can also span multiple frames; accumulate until END.
            self._queue.put_nowait(("error", frame.payload))
        elif frame.msg_type == MessageType.RESPONSE:
            payload = frame.payload
            if self._decompressor is not None:
                # Strip and verify CRC32 trailer before decompressing.
                if frame.is_end and (frame.flags & Flags.CHECKSUM):
                    if len(payload) < CHECKSUM_SIZE:
                        self._fail(ProtocolError("CHECKSUM END frame payload too short"))
                        return
                    payload_data = payload[:-CHECKSUM_SIZE]
                    expected_crc = int.from_bytes(payload[-CHECKSUM_SIZE:], "big")
                    self._running_crc = zlib.crc32(payload_data, self._running_crc) & 0xFFFFFFFF
                    if self._running_crc != expected_crc:
                        self._fail(ProtocolError("CRC32 mismatch on reassembled response"))
                        return
                    payload = payload_data
                else:
                    self._running_crc = zlib.crc32(payload, self._running_crc) & 0xFFFFFFFF
                try:
                    decompressed = self._decompressor.decompress(payload)
                    if frame.is_end:
                        decompressed += self._decompressor.flush()
                    payload = decompressed
                except zlib.error as exc:
                    self._fail(ProtocolError(f"decompression failed: {exc}"))
                    return
            if payload:
                self._queue.put_nowait(("data", payload))
        else:
            self._fail(
                ProtocolError(
                    f"unexpected response type {frame.msg_type.name} "
                    f"for {self.message_id}"
                )
            )
            return

        if frame.is_end:
            self._next_seq += 1
            self._done = True
            self._queue.put_nowait(self._END)

    def _fail(self, exc: Exception) -> None:
        self._done = True
        self._queue.put_nowait(("exc", exc))
        self._queue.put_nowait(self._END)

    async def __aiter__(self) -> AsyncIterator[str]:
        # A character may be split across frames at the byte level, so decode
        # incrementally and only emit complete characters.
        decoder = codecs.getincrementaldecoder("utf-8")()
        error_parts: list[bytes] = []
        is_error = False
        while True:
            item = await self._queue.get()
            if item is self._END:
                break
            kind, value = item  # type: ignore[misc]
            if kind == "exc":
                raise value
            if kind == "error":
                is_error = True
                error_parts.append(value)
                continue
            text = decoder.decode(value)
            if text:
                yield text
        if is_error:
            raise RemoteError(b"".join(error_parts).decode("utf-8", "replace"))
        tail = decoder.decode(b"", final=True)
        if tail:
            yield tail


class ConsumerClient:
    """Sends prompts and collects responses over a transport."""

    def __init__(
        self,
        transport: Transport,
        *,
        profile: Optional[ProfileConfig] = None,
        requested_mode: str = "auto",
        mtu: int = DEFAULT_ATT_MTU,
        max_payload_size: Optional[int] = None,
        timeout: float = DEFAULT_TIMEOUT,
        compress_prompts: bool = True,
        enable_delta_messages: bool = True,
    ) -> None:
        self._transport = transport
        self._profile = profile or LEGACY_PROFILE
        # If an explicit profile was supplied, skip negotiation.
        self._requested_mode: Optional[str] = None if profile is not None else requested_mode
        self._max_payload = (
            max_payload_size
            if max_payload_size is not None
            else max_payload_for_mtu(profile.mtu if profile is not None else mtu)
        )
        if self._max_payload <= 0:
            raise ValueError("max_payload_size must be positive")
        self._timeout = timeout
        self._compress_prompts = compress_prompts
        self._enable_delta_messages = enable_delta_messages
        self._pending: Dict[int, _PendingRequest] = {}
        # Final ACK value for recently-completed messages, so a retransmitted
        # tail frame (arriving after we stopped tracking the request) can still
        # be acknowledged and the host's sender can drain.
        self._completed_acks: Dict[int, int] = {}
        # message_id 0 is reserved/avoided; ids wrap within the 16-bit space.
        self._ids = itertools.cycle(range(1, 0x10000))
        # Track conversation history for delta detection
        self._conversation_history: bytes = b""
        self._history_checksum: str = ""

    async def start(self) -> None:
        self._transport.on_receive(self._on_frame)
        await self._transport.start()
        await self._negotiate()

    async def _negotiate(self) -> None:
        """Run the Phase-1 capability handshake, if supported by the transport."""
        if self._requested_mode is None:
            return  # explicit profile supplied at construction — skip
        host_modes = await self._transport.read_caps()
        new_profile = negotiate_mode(host_modes, self._requested_mode)
        prev_mode = self._profile.mode
        self._profile = new_profile
        # For FAST_GATT and L2CAP_COC, size max_payload and switch write mode.
        if new_profile.mode == Mode.FAST_GATT:
            mtu = self._transport.mtu_size
            self._max_payload = max_payload_for_mtu(mtu)
            self._transport.configure_write_mode(write_with_response=False)
        elif new_profile.mode == Mode.L2CAP_COC:
            # PSM read and L2CAP channel open happen at the BLE transport layer;
            # the transport switch is coordinated by BleakCentralTransport when
            # running on real hardware.  For in-process tests the L2CAP transport
            # is wired directly and negotiation is bypassed via profile=.
            mtu = self._transport.mtu_size
            self._max_payload = max_payload_for_mtu(mtu)
            psm = await self._transport.read_l2cap_psm()
            if psm is not None:
                logger.info("L2CAP_COC: host PSM=%d (channel open deferred to BLE layer)", psm)
        elif new_profile.mode == Mode.CLASSIC_RFCOMM:
            # RFCOMM channel setup happens at the Classic transport layer; for
            # in-process tests the transport is wired directly via profile=.
            mtu = self._transport.mtu_size
            self._max_payload = max_payload_for_mtu(mtu)
        # Send SELECT so the host knows the agreed mode and the consumer's MTU.
        mtu = self._transport.mtu_size
        select_frame = Frame(
            msg_type=MessageType.SELECT,
            message_id=0,  # reserved control channel
            seq=0,
            payload=encode_select_payload(mode_id_for(new_profile.mode), mtu),
            flags=Flags.START | Flags.END,
        )
        await self._transport.send(select_frame.encode())
        if prev_mode != new_profile.mode:
            logger.info(
                "Bluetooth mode: %s → %s (MTU=%d, host caps=%s)",
                prev_mode.value, new_profile.mode.value, mtu,
                host_modes if host_modes is not None else "n/a (legacy host)",
            )
        else:
            logger.info(
                "Bluetooth mode: %s (MTU=%d, host caps=%s)",
                new_profile.mode.value, mtu,
                host_modes if host_modes is not None else "n/a (legacy host)",
            )

    async def close(self) -> None:
        await self._transport.close()

    async def __aenter__(self) -> "ConsumerClient":
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # -- frame intake ---------------------------------------------------------

    async def _on_frame(self, data: bytes) -> None:
        try:
            frame = Frame.decode(data)
        except ProtocolError:
            logger.exception("dropping malformed response frame")
            return
        # The consumer never expects ACK frames itself; ignore defensively.
        if frame.msg_type == MessageType.ACK:
            return
        # Control frames (CAPS, SELECT, HELLO, NAK) are not routed to pending
        # requests; ignore them so unexpected control traffic doesn't surface
        # as errors after negotiation completes.
        if frame.msg_type in (
            MessageType.CAPS,
            MessageType.SELECT,
            MessageType.HELLO,
            MessageType.NAK,
        ):
            logger.debug("ignoring control frame type %s", frame.msg_type.name)
            return
        pending = self._pending.get(frame.message_id)
        if pending is None:
            # The request already finished; re-ack so a retransmitted tail frame
            # lets the host's sender drain instead of timing out.
            ack = self._completed_acks.get(frame.message_id)
            if ack is not None:
                await self._send_ack(frame.message_id, ack)
            else:
                logger.debug("response for unknown message %s", frame.message_id)
            return
        if frame.is_start:
            logger.info("✓ Started receiving response (msg_id=%d)", frame.message_id)
        pending.feed(frame)
        # Acknowledge the highest contiguous sequence received so the host can
        # release acked frames and retransmit anything still missing.
        await self._send_ack(frame.message_id, pending.ack_seq)

    async def _send_ack(self, message_id: int, ack_seq: int) -> None:
        try:
            await self._transport.send(make_ack_frame(message_id, ack_seq).encode())
        except Exception:  # pragma: no cover - best effort over a failing link
            logger.debug("failed to send ACK for %s", message_id, exc_info=True)

    # -- public API -----------------------------------------------------------

    def _alloc_id(self) -> int:
        for _ in range(0x10000):
            candidate = next(self._ids)
            if candidate not in self._pending:
                # Reusing an id invalidates any stale completed-ack record.
                self._completed_acks.pop(candidate, None)
                return candidate
        raise RuntimeError("no free message ids available")

    async def _send_message(
        self, message_id: int, msg_type: MessageType, payload: bytes
    ) -> None:
        # Try delta encoding for prompts if enabled and we have history
        is_delta = False
        if (self._enable_delta_messages and 
            msg_type == MessageType.PROMPT and 
            self._conversation_history and 
            len(payload) > len(self._conversation_history)):
            
            # Check if the new message starts with the old conversation history
            if payload[:len(self._conversation_history)] == self._conversation_history:
                # Extract only the new part
                delta_content = payload[len(self._conversation_history):]
                original_payload_size = len(payload)
                
                # Encode with checksum reference
                payload = encode_delta_payload(self._history_checksum, delta_content)
                is_delta = True
                
                logger.info(
                    "Detected delta prompt: %d bytes -> %d bytes (%.1f%% saved via delta)",
                    original_payload_size,
                    len(payload),
                    100 * (1 - len(payload) / original_payload_size)
                )
        
        # Compress prompts to reduce BLE transmission time
        compressed = False
        original_size = len(payload)
        if self._compress_prompts and msg_type == MessageType.PROMPT and payload:
            compressed_payload = zlib.compress(payload, level=6)
            # Only use compression if it actually saves space
            if len(compressed_payload) < len(payload):
                payload = compressed_payload
                compressed = True
                logger.info(
                    "Sending %s (msg_id=%d, %d bytes -> %d bytes compressed, %.1f%%)...",
                    msg_type.name,
                    message_id,
                    original_size,
                    len(payload),
                    100 * len(payload) / original_size,
                )
            else:
                logger.info(
                    "Sending %s (msg_id=%d, %d bytes, compression skipped)...",
                    msg_type.name,
                    message_id,
                    len(payload),
                )
        else:
            logger.info(
                "Sending %s (msg_id=%d, %d bytes)...",
                msg_type.name,
                message_id,
                len(payload),
            )
        frames = chunk_message(
            msg_type,
            message_id,
            payload,
            self._max_payload,
            checksum=self._profile.mode in (Mode.FAST_GATT, Mode.L2CAP_COC, Mode.CLASSIC_RFCOMM),
        )
        # Mark first frame with appropriate flags
        if frames:
            flags = frames[0].flags
            if compressed:
                flags |= Flags.COMPRESSED
            if is_delta:
                flags |= Flags.DELTA
            if flags != frames[0].flags:
                frames[0] = Frame(
                    msg_type=frames[0].msg_type,
                    message_id=frames[0].message_id,
                    seq=frames[0].seq,
                    payload=frames[0].payload,
                    flags=flags,
                    version=frames[0].version,
                )
        for frame in frames:
            await self._transport.send(frame.encode())
        logger.info("✓ %s sent, waiting for response...", msg_type.name)

    def stream(self, prompt: str) -> AsyncIterator[str]:
        """Send ``prompt`` and yield response text pieces as they arrive."""

        return self._stream(prompt.encode("utf-8"), MessageType.PROMPT)

    async def _stream(
        self, payload: bytes, msg_type: MessageType
    ) -> AsyncIterator[str]:
        message_id = self._alloc_id()
        # FAST_GATT, L2CAP_COC and CLASSIC_RFCOMM now use the streaming path by
        # default; COMPRESSED flag on the START frame triggers incremental
        # decompression in _PendingRequest so text appears as tokens arrive.
        pending = _PendingRequest(message_id)
        self._pending[message_id] = pending
        
        # Track the original prompt for history
        original_prompt = payload
        
        try:
            await self._send_message(message_id, msg_type, payload)
            
            # Collect the response for history tracking
            response_parts: List[str] = []
            
            iterator = pending.__aiter__()
            while True:
                try:
                    piece = await iterator.__anext__()
                except StopAsyncIteration:
                    break
                response_parts.append(piece)
                yield piece
            
            # Update conversation history if this was a successful prompt
            if msg_type == MessageType.PROMPT and self._enable_delta_messages:
                response_text = "".join(response_parts)
                # Build new history: old_prompt + old_response + new_prompt + new_response
                new_history = original_prompt + response_text.encode("utf-8")
                self._conversation_history = new_history
                self._history_checksum = compute_message_checksum(new_history)
                logger.debug(
                    "Updated conversation history: %d bytes, checksum=%s",
                    len(self._conversation_history),
                    self._history_checksum
                )
        finally:
            # Remember the final ACK so late retransmissions of the tail can
            # still be acknowledged, then stop tracking the live request.
            self._completed_acks[message_id] = pending.ack_seq
            self._pending.pop(message_id, None)

    async def generate(self, prompt: str) -> str:
        """Send ``prompt`` and return the complete response text."""

        parts = [piece async for piece in self._stream(prompt.encode("utf-8"), MessageType.PROMPT)]
        return "".join(parts)

    async def list_models(self) -> List[str]:
        """Ask the host which models it can serve.

        Sends a LIST_MODELS request and parses the host's JSON reply (an object
        of the form ``{"models": [...]}``).  Returns an empty list if the host
        reports no models or sends an unexpected payload.
        """

        parts = [piece async for piece in self._stream(b"", MessageType.LIST_MODELS)]
        text = "".join(parts).strip()
        if not text:
            return []
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("host sent malformed model list: %r", text[:200])
            return []
        models = obj.get("models", []) if isinstance(obj, dict) else []
        return [str(name) for name in models if name]
