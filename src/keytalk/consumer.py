"""The CONSUMER side: send a prompt, receive the streamed answer.

The consumer owns a :class:`~keytalk.transport.Transport` (in production a BLE
central connected to the host).  :meth:`ConsumerClient.generate` returns the
whole answer; :meth:`ConsumerClient.stream` yields response text pieces
incrementally as frames arrive.  Each outstanding request is tracked by
``message_id`` so the client can multiplex several prompts over one link.

Resilience:

* Prompts are zlib-compressed before chunking (responses are compressed by the
  host and decompressed here incrementally).
* When a response stalls (link drop, lost notification) the consumer sends a
  RESUME asking the host to retransmit from the last contiguous sequence
  number, so ``stream()`` continues exactly where it left off.
* Requests that fail before producing any output are retried from scratch with
  a fresh message id; ``generate`` also retries after partial output (it can
  discard it safely).
* When a streamed request is abandoned early a CANCEL is sent so the host can
  stop generating.
* A periodic PING keeps the BLE link alive and triggers the transport's
  reconnect logic before a user request pays for it.
"""

from __future__ import annotations

import asyncio
import codecs
import itertools
import json
import logging
import zlib
from collections import OrderedDict
from typing import AsyncIterator, Dict, List, Optional

from .modes import LEGACY_PROFILE, Mode, ProfileConfig, mode_id_for, negotiate_mode
from .protocol import (
    CHECKSUM_SIZE,
    DEFAULT_ATT_MTU,
    Flags,
    Frame,
    MessageType,
    ProtocolError,
    RESUME_UNKNOWN,
    chunk_message,
    encode_cancel,
    encode_max_payload,
    encode_resume,
    encode_select_payload,
    max_payload_for_mtu,
)
from .reliability import make_ack_frame
from .transport import Transport, TransportClosed

__all__ = ["ConsumerClient", "RemoteError", "ResponseStream", "_PendingRequest"]

logger = logging.getLogger("keytalk.consumer")

DEFAULT_TIMEOUT = 300.0

#: Message id reserved for control messages (HELLO/PING/RESUME/CANCEL).
CONTROL_ID = 0

#: Bound on the out-of-order buffer for one request (frames).  Far more than
#: the sender's window ever keeps in flight; anything beyond means trouble.
MAX_REORDER = 512

#: How many completed-message ACK values to remember for late retransmissions.
MAX_COMPLETED_ACKS = 256


class RemoteError(Exception):
    """Raised when the host returns an ERROR message for a request."""


class ResponseStream:
    """Async text stream carrying one request's trailer metadata.

    The generation stats, structured ``tool_calls`` and ``finish_reason`` a
    host reports in a message's ``TIMINGS`` trailer belong to *that* request.
    They ride on the stream object rather than on shared client state, so
    concurrent requests (e.g. several HTTP connections against ``--serve``)
    cannot read each other's metadata.
    """

    def __init__(
        self,
        gen: AsyncIterator[str],
        meta: Optional[Dict[str, object]] = None,
    ) -> None:
        self._gen = gen
        self._meta: Dict[str, object] = meta if meta is not None else {}

    def __aiter__(self) -> "ResponseStream":
        return self

    async def __anext__(self) -> str:
        return await self._gen.__anext__()

    async def aclose(self) -> None:
        aclose = getattr(self._gen, "aclose", None)
        if aclose is not None:
            await aclose()

    @property
    def timings(self) -> Optional[dict]:
        """Server-side generation stats for this request, once available."""

        value = self._meta.get("timings")
        return value if isinstance(value, dict) else None

    @property
    def tool_calls(self) -> Optional[list]:
        """Structured tool calls for this request, once available."""

        value = self._meta.get("tool_calls")
        return value if isinstance(value, list) and value else None

    @property
    def finish_reason(self) -> Optional[str]:
        """Backend finish reason for this request, once available."""

        value = self._meta.get("finish_reason")
        return value if isinstance(value, str) else None


def _is_retryable(exc: BaseException) -> bool:
    """Whether a failed attempt can simply be run again from scratch."""

    if isinstance(exc, RemoteError):
        # The host no longer knows the request (e.g. it was restarted between
        # our RESUME and its replay): restartable, unlike real backend errors.
        return str(exc) == RESUME_UNKNOWN
    return isinstance(exc, (asyncio.TimeoutError, TransportClosed, ConnectionError, OSError))


class _PendingRequest:
    """Tracks reassembly and streaming for one in-flight request.

    Response (or error) frames are validated for ordering and pushed onto a
    queue so consumers can stream them.  Out-of-order frames (which happen when
    a notification is dropped and later retransmitted) are buffered and replayed
    in sequence rather than treated as a fatal error.  Completion is signalled
    with a sentinel so an async iterator terminates cleanly.

    Compressed responses (``COMPRESSED`` on the START frame) are decompressed
    incrementally.  ERROR frames are never compressed: the host flushes the
    compressed stream before switching to error frames, so data can be
    attributed to the frame type that carried it.
    """

    _END = object()

    def __init__(self, message_id: int) -> None:
        self.message_id = message_id
        self._queue: "asyncio.Queue[object]" = asyncio.Queue()
        self._next_seq = 0
        self._reorder: Dict[int, Frame] = {}
        self._started = False
        self._done = False
        self._dec = None  # incremental zlib decompressor, if compressed
        self._crc = 0  # running CRC32 over wire payloads (CHECKSUM messages)
        #: Raw TIMINGS trailer payloads (generation meta; see host._finish_message)
        self.meta_parts: List[bytes] = []
        #: Frames accepted so far; used to detect progress vs. stalls.
        self.activity = 0

    @property
    def ack_seq(self) -> int:
        """Next contiguous sequence number expected (cumulative ACK value)."""

        return self._next_seq

    @property
    def done(self) -> bool:
        return self._done

    @property
    def timings(self) -> Optional[dict]:
        """Generation stats from the message's TIMINGS trailer, if present."""

        value = self._meta().get("timings")
        return value if isinstance(value, dict) else None

    @property
    def tool_calls(self) -> Optional[list]:
        """Structured tool calls from the message's trailer, if present."""

        value = self._meta().get("tool_calls")
        return value if isinstance(value, list) and value else None

    @property
    def finish_reason(self) -> Optional[str]:
        """Backend finish reason from the message's trailer, if present."""

        value = self._meta().get("finish_reason")
        return value if isinstance(value, str) else None

    def _meta(self) -> Dict[str, object]:
        """Parsed trailer payload (``{}`` when there is none / it is malformed)."""

        if not self.meta_parts:
            return {}
        try:
            obj = json.loads(b"".join(self.meta_parts).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            logger.warning("malformed timings trailer for %s", self.message_id)
            return {}
        return obj if isinstance(obj, dict) else {}

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

        if len(self._reorder) >= MAX_REORDER:
            self._fail(
                ProtocolError(
                    f"response {self.message_id} exceeded the reorder buffer"
                )
            )
            return

        self._reorder[frame.seq] = frame
        self.activity += 1
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
            if frame.flags & Flags.COMPRESSED:
                self._dec = zlib.decompressobj()

        if frame.msg_type == MessageType.ERROR:
            # Error payloads can also span multiple frames; accumulate until
            # END.  They are plain text even when the response was compressed.
            self._queue.put_nowait(("error", frame.payload))
        elif frame.msg_type == MessageType.TIMINGS:
            # Plain-text stats trailer (never compressed); never streamed out.
            self.meta_parts.append(frame.payload)
        elif frame.msg_type == MessageType.RESPONSE:
            data = frame.payload
            # Strip and verify the CRC32 trailer carried by the frame flagged
            # CHECKSUM (buffered/checksummed senders), regardless of
            # compression.  The flag - not END - marks the trailer, because a
            # message may close on a following TIMINGS frame instead.
            if frame.flags & Flags.CHECKSUM:
                if len(data) < CHECKSUM_SIZE:
                    self._fail(
                        ProtocolError("CHECKSUM END frame payload too short")
                    )
                    return
                body, trailer = data[:-CHECKSUM_SIZE], data[-CHECKSUM_SIZE:]
                self._crc = zlib.crc32(body, self._crc) & 0xFFFFFFFF
                expected = int.from_bytes(trailer, "big")
                if self._crc != expected:
                    self._fail(
                        ProtocolError("CRC32 mismatch on reassembled response")
                    )
                    return
                data = body
            else:
                self._crc = zlib.crc32(data, self._crc) & 0xFFFFFFFF
            if self._dec is not None:
                try:
                    data = self._dec.decompress(data)
                except zlib.error:
                    self._fail(
                        ProtocolError(
                            f"failed to decompress response {self.message_id}"
                        )
                    )
                    return
            if data:
                self._queue.put_nowait(("data", data))
        else:
            self._fail(
                ProtocolError(
                    f"unexpected response type {frame.msg_type.name} "
                    f"for {self.message_id}"
                )
            )
            return

        if frame.is_end:
            if self._dec is not None:
                tail = self._dec.flush()
                if tail:
                    self._queue.put_nowait(("data", tail))
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
        max_mtu: int = 512,
        max_payload_size: Optional[int] = None,
        timeout: float = DEFAULT_TIMEOUT,
        compress_prompts: bool = True,
        retries: int = 2,
        max_resumes: int = 3,
        keepalive_interval: float = 15.0,
    ) -> None:
        self._transport = transport
        self._profile = profile or LEGACY_PROFILE
        # An explicit profile skips negotiation entirely.
        self._requested_mode: Optional[str] = (
            None if profile is not None else requested_mode
        )
        self._max_payload = (
            max_payload_size
            if max_payload_size is not None
            else max_payload_for_mtu(profile.mtu if profile is not None else mtu)
        )
        self._explicit_payload = max_payload_size is not None
        self._max_mtu = max_mtu
        if self._max_payload <= 0:
            raise ValueError("max_payload_size must be positive")
        #: Idle timeout per response piece; a stall triggers a RESUME.
        self._timeout = timeout
        self._compress_prompts = compress_prompts
        self._retries = retries
        self._max_resumes = max_resumes
        self._keepalive_interval = keepalive_interval
        self._keepalive_task: Optional["asyncio.Task[None]"] = None
        self._pending: Dict[int, _PendingRequest] = {}
        # Final ACK value for recently-completed messages, so a retransmitted
        # tail frame (arriving after we stopped tracking the request) can still
        # be acknowledged and the host's sender can drain.
        self._completed_acks: "OrderedDict[int, int]" = OrderedDict()
        # message_id 0 is reserved for control messages; request ids wrap
        # within the rest of the 16-bit space.
        self._ids = itertools.cycle(range(1, 0x10000))
        # Generation metadata (llama.cpp timings incl. MTP acceptance,
        # structured tool calls, finish reason) captured from the most
        # recently completed request's TIMINGS trailer.
        self._last_timings: Optional[dict] = None
        self._last_tool_calls: Optional[list] = None
        self._last_finish_reason: Optional[str] = None

    @property
    def last_timings(self) -> Optional[dict]:
        """Server-side generation stats for the most recent request, if any."""

        return self._last_timings

    @property
    def last_tool_calls(self) -> Optional[list]:
        """Structured tool calls for the most recent request, if any."""

        return self._last_tool_calls

    @property
    def last_finish_reason(self) -> Optional[str]:
        """Backend finish reason for the most recent request, if any."""

        return self._last_finish_reason

    async def start(self) -> None:
        self._transport.on_receive(self._on_frame)
        await self._transport.start()
        # If the transport knows the negotiated link MTU (BLE adapters do),
        # size frames to it instead of the conservative 23-byte ATT default -
        # this is a ~10x throughput win - and tell the host what we accept.
        link_mtu = int(getattr(self._transport, "mtu_size", DEFAULT_ATT_MTU) or 0)
        if link_mtu > DEFAULT_ATT_MTU and not self._explicit_payload:
            self._max_payload = max_payload_for_mtu(min(link_mtu, self._max_mtu))
            logger.info("link MTU %d: frame payload %d bytes", link_mtu, self._max_payload)
        await self._negotiate()
        await self._send_control(MessageType.HELLO, encode_max_payload(self._max_payload))
        if self._keepalive_interval > 0:
            self._keepalive_task = asyncio.ensure_future(self._keepalive_loop())

    async def _negotiate(self) -> None:
        """Run the Phase-1 capability handshake (transfer-mode negotiation).

        Reads the host's CAPS advertisement (if any), picks the best common
        mode, reconfigures this side (frame size, write mode), and tells the
        host via a SELECT frame so both ends agree for this connection.
        """

        if self._requested_mode is None:
            return  # explicit profile supplied at construction - skip
        host_modes = await self._transport.read_caps()
        new_profile = negotiate_mode(host_modes, self._requested_mode)
        prev_mode = self._profile.mode
        self._profile = new_profile
        mtu = self._transport.mtu_size
        if (
            new_profile.mode in (Mode.FAST_GATT, Mode.L2CAP_COC, Mode.CLASSIC_RFCOMM)
            and not self._explicit_payload
        ):
            # Same clamp as the link-MTU path in start(): honour --mtu.
            self._max_payload = max_payload_for_mtu(min(mtu, self._max_mtu))
        if new_profile.mode == Mode.FAST_GATT:
            self._transport.configure_write_mode(write_with_response=False)
        elif new_profile.mode == Mode.L2CAP_COC:
            # PSM read and L2CAP channel open happen at the BLE transport layer;
            # for in-process tests the L2CAP transport is wired directly.
            psm = await self._transport.read_l2cap_psm()
            if psm is not None:
                logger.info(
                    "L2CAP_COC: host PSM=%d (channel open deferred to BLE layer)",
                    psm,
                )
        # Send SELECT so the host knows the agreed mode and the consumer's MTU.
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
                "Bluetooth mode: %s -> %s (MTU=%d, host caps=%s)",
                prev_mode.value,
                new_profile.mode.value,
                mtu,
                host_modes if host_modes is not None else "n/a (legacy host)",
            )
        else:
            logger.info(
                "Bluetooth mode: %s (MTU=%d, host caps=%s)",
                new_profile.mode.value,
                mtu,
                host_modes if host_modes is not None else "n/a (legacy host)",
            )

    async def close(self) -> None:
        task = self._keepalive_task
        self._keepalive_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
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
        # The consumer never expects ACK or PING frames itself; PONG is the
        # reply to our keepalive and needs no handling.
        if frame.msg_type in (MessageType.ACK, MessageType.PING, MessageType.PONG):
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

    async def _send_control(self, msg_type: MessageType, payload: bytes = b"") -> None:
        """Send a single control message (id 0) to the host."""

        frames = chunk_message(msg_type, CONTROL_ID, payload, self._max_payload)
        for frame in frames:
            await self._transport.send(frame.encode())

    async def _send_resume(self, message_id: int, ack_seq: int) -> None:
        logger.warning(
            "response for message %s stalled; requesting retransmission from seq %d",
            message_id,
            ack_seq,
        )
        await self._send_control(MessageType.RESUME, encode_resume(message_id, ack_seq))

    def _fire_cancel(self, message_id: int) -> None:
        """Tell the host to stop generating for an abandoned request."""

        async def _run() -> None:
            try:
                await self._send_control(MessageType.CANCEL, encode_cancel(message_id))
                logger.info("cancelled request %s on the host", message_id)
            except Exception:  # noqa: BLE001 - best effort
                logger.debug("failed to send CANCEL for %s", message_id, exc_info=True)

        asyncio.ensure_future(_run())

    async def _keepalive_loop(self) -> None:
        while True:
            await asyncio.sleep(self._keepalive_interval)
            try:
                await self._send_control(MessageType.PING)
            except Exception:  # noqa: BLE001 - the next request retries anyway
                logger.debug("keepalive ping failed", exc_info=True)

    # -- public API -----------------------------------------------------------

    def _alloc_id(self) -> int:
        for _ in range(0x10000):
            candidate = next(self._ids)
            if candidate not in self._pending:
                # Reusing an id invalidates any stale completed-ack record.
                self._completed_acks.pop(candidate, None)
                return candidate
        raise RuntimeError("no free message ids available")

    def _remember_ack(self, message_id: int, ack_seq: int) -> None:
        self._completed_acks[message_id] = ack_seq
        self._completed_acks.move_to_end(message_id)
        while len(self._completed_acks) > MAX_COMPLETED_ACKS:
            self._completed_acks.popitem(last=False)

    async def _send_message(
        self, message_id: int, msg_type: MessageType, payload: bytes
    ) -> None:
        # Compress prompts/chat payloads to reduce BLE transmission time
        compressed = False
        original_size = len(payload)
        if self._compress_prompts and msg_type in (MessageType.PROMPT, MessageType.CHAT) and payload:
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
            start_flags=Flags.COMPRESSED if compressed else Flags.NONE,
        )
        for frame in frames:
            await self._transport.send(frame.encode())
        logger.info("✓ %s sent, waiting for response...", msg_type.name)

    def stream(self, prompt: str) -> ResponseStream:
        """Send ``prompt`` and yield response text pieces as they arrive.

        Transparently resumes after link drops; only retries the whole prompt
        if nothing has been yielded yet (retrying later would duplicate text).
        """

        return self._stream(prompt.encode("utf-8"), MessageType.PROMPT, retries=self._retries)

    def chat_stream(self, messages: List[dict], **params) -> ResponseStream:
        """Send structured chat ``messages`` and yield the reply incrementally.

        Unlike ``stream`` (which sends a pre-rendered prompt string), the
        messages are passed to the host as JSON so a backend with native chat
        support can render the model's own chat template server-side.
        ``params`` (e.g. ``temperature``) are forwarded to the backend.
        """

        return self._stream(
            self._chat_payload(messages, params), MessageType.CHAT, retries=self._retries
        )

    async def chat(self, messages: List[dict], **params) -> str:
        """Send structured chat ``messages`` and return the complete reply."""

        return await self._collect(self._chat_payload(messages, params), MessageType.CHAT)

    @staticmethod
    def _chat_payload(messages: List[dict], params: dict) -> bytes:
        request: Dict[str, object] = {"messages": list(messages)}
        if params:
            request["params"] = params
        return json.dumps(request, separators=(",", ":")).encode("utf-8")

    def _stream(
        self, payload: bytes, msg_type: MessageType, *, retries: int = 0
    ) -> ResponseStream:
        """Wrap the retrying text loop in a per-request metadata carrier."""

        meta: Dict[str, object] = {}
        return ResponseStream(
            self._stream_text(payload, msg_type, meta, retries=retries), meta
        )

    async def _stream_text(
        self,
        payload: bytes,
        msg_type: MessageType,
        meta: Dict[str, object],
        *,
        retries: int = 0,
    ) -> AsyncIterator[str]:
        yielded = False
        attempts = 0
        while True:
            attempt = self._attempt(payload, msg_type, meta)
            try:
                async for piece in attempt:
                    yielded = True
                    yield piece
                return
            except BaseException as exc:  # noqa: BLE001 - classified below
                if not _is_retryable(exc) or yielded or attempts >= retries:
                    raise
                attempts += 1
                logger.warning(
                    "%s failed (%s); retrying (%d/%d)",
                    msg_type.name,
                    exc,
                    attempts,
                    retries,
                )
                await asyncio.sleep(min(0.5 * attempts, 2.0))
            finally:
                # Deterministically finalize the attempt (and thus CANCEL any
                # half-received request on the host) even when this generator
                # is closed early by its caller.
                await attempt.aclose()

    async def _attempt(
        self,
        payload: bytes,
        msg_type: MessageType,
        meta: Optional[Dict[str, object]] = None,
    ) -> AsyncIterator[str]:
        message_id = self._alloc_id()
        pending = _PendingRequest(message_id)
        self._pending[message_id] = pending
        # The piece-getter is awaited with asyncio.wait() rather than
        # wait_for(): timing out must NOT cancel it, because that would close
        # the iterator and silently truncate the response on resume.
        next_piece: Optional["asyncio.Task[str]"] = None
        try:
            await self._send_message(message_id, msg_type, payload)
            iterator = pending.__aiter__()
            next_piece = asyncio.ensure_future(iterator.__anext__())
            resumes = 0
            last_activity = pending.activity
            while True:
                done, _ = await asyncio.wait({next_piece}, timeout=self._timeout)
                if not done:
                    # The stream stalled (link drop, lost notification): ask
                    # the host to retransmit from our position before failing.
                    if pending.activity != last_activity:
                        last_activity = pending.activity
                        resumes = 0
                    if resumes >= self._max_resumes:
                        raise asyncio.TimeoutError(
                            f"response for message {message_id} stalled "
                            f"after {resumes} retransmission requests"
                        )
                    resumes += 1
                    await self._send_resume(message_id, pending.ack_seq)
                    continue
                try:
                    piece = next_piece.result()
                except StopAsyncIteration:
                    # Publish on the per-request carrier first, then mirror onto
                    # the client's ``last_*`` views for library callers.
                    if meta is not None:
                        meta["timings"] = pending.timings
                        meta["tool_calls"] = pending.tool_calls
                        meta["finish_reason"] = pending.finish_reason
                    self._last_timings = pending.timings
                    self._last_tool_calls = pending.tool_calls
                    self._last_finish_reason = pending.finish_reason
                    return
                resumes = 0
                yield piece
                next_piece = asyncio.ensure_future(iterator.__anext__())
        finally:
            if next_piece is not None and not next_piece.done():
                next_piece.cancel()
            # Remember the final ACK so late retransmissions of the tail can
            # still be acknowledged, then stop tracking the live request.
            self._remember_ack(message_id, pending.ack_seq)
            self._pending.pop(message_id, None)
            if not pending.done:
                # The request was abandoned early (error, timeout, or the
                # caller stopped reading): let the host stop generating.
                self._fire_cancel(message_id)

    async def generate(self, prompt: str) -> str:
        """Send ``prompt`` and return the complete response text.

        Unlike ``stream``, failed attempts are retried even after partial
        output: nothing has been shown to the caller yet, so it is safe to
        discard and restart.
        """

        return await self._collect(prompt.encode("utf-8"), MessageType.PROMPT)

    async def _collect(self, payload: bytes, msg_type: MessageType) -> str:
        """Run a request to completion, retrying from scratch on failure.

        Safe to retry after partial output because nothing has been handed to
        the caller yet.
        """

        attempts = 0
        while True:
            attempt = self._attempt(payload, msg_type)
            try:
                parts = [piece async for piece in attempt]
                return "".join(parts)
            except BaseException as exc:  # noqa: BLE001 - classified below
                if not _is_retryable(exc) or attempts >= self._retries:
                    raise
                attempts += 1
                logger.warning(
                    "%s failed (%s); retrying (%d/%d)",
                    msg_type.name,
                    exc,
                    attempts,
                    self._retries,
                )
                await asyncio.sleep(min(0.5 * attempts, 2.0))
            finally:
                await attempt.aclose()

    async def list_models(self) -> List[str]:
        """Ask the host which models it can serve.

        Sends a LIST_MODELS request and parses the host's JSON reply (an object
        of the form ``{"models": [...]}``).  Returns an empty list if the host
        reports no models or sends an unexpected payload.
        """

        parts = [piece async for piece in self._stream(b"", MessageType.LIST_MODELS, retries=self._retries)]
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
