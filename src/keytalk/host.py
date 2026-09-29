"""The HOST side: receive prompt frames, run the LLM, stream back the answer.

The host owns a :class:`~keytalk.transport.Transport` (in production a BLE
peripheral) and an :class:`~keytalk.backends.LLMBackend`.  Incoming frames are
reassembled into prompt messages; each prompt is handled on its own task so
several consumers (or pipelined requests) can be served concurrently.  The LLM's
streamed output is re-chunked into RESPONSE frames and sent back, correlated to
the request by ``message_id``.

Reliability and recovery:

* Every outbound frame is retained until the consumer acknowledges it, so a
  RESUME request can replay whatever the consumer missed after a link drop
  (see :class:`_OutboundMessage`).  Retained frames are trimmed by cumulative
  ACKs and swept after a TTL, so memory stays bounded.
* Malformed prompt frames are answered with an immediate ERROR message (NACK)
  instead of leaving the consumer waiting for a response that will never come.
* A CANCEL message aborts the generation task for a request (e.g. when the
  consumer's HTTP client disconnects), freeing the LLM for the next request.
* A HELLO message lets the consumer negotiate the frame payload size once the
  real BLE MTU is known (the default ATT MTU only fits 13 payload bytes/frame).
* Responses are zlib-compressed incrementally; ERROR frames switch to plain
  text after the compressed stream is flushed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import zlib
from typing import Dict, List, Optional, Set

from .backends import LLMBackend, messages_to_prompt
from .modes import (
    LEGACY_PROFILE,
    Mode,
    ProfileConfig,
    make_classic_rfcomm_profile,
    make_fast_gatt_profile,
    make_l2cap_coc_profile,
    mode_for_id,
    profile_for_mode,
)
from .protocol import (
    DEFAULT_ATT_MTU,
    RESUME_UNKNOWN,
    Flags,
    Frame,
    FrameStreamEncoder,
    MessageType,
    ProtocolError,
    Reassembler,
    chunk_message,
    decode_cancel,
    decode_max_payload,
    decode_resume,
    decode_select_payload,
    max_payload_for_mtu,
)
from .reliability import ReliableSender
from .transport import Transport

__all__ = ["HostService"]

logger = logging.getLogger("keytalk.host")


class _OutboundMessage:
    """All frames of one outbound message plus its delivery state.

    Frames are kept until cumulatively acknowledged so delivery can be
    (re)started at any point: after a link drop the consumer sends a RESUME
    with the next sequence number it expects and the message is re-pumped from
    there - mid-stream or after completion.
    """

    __slots__ = (
        "message_id",
        "frames",
        "base",
        "complete",
        "delivered",
        "event",
        "sender",
        "task",
        "created",
    )

    def __init__(self, message_id: int) -> None:
        self.message_id = message_id
        self.frames: List[Frame] = []
        self.base = 0  # sequence number of frames[0]
        self.complete = False  # no further frames will be appended
        self.delivered = False  # every frame has been acknowledged
        self.event = asyncio.Event()
        self.sender: Optional[ReliableSender] = None
        self.task: Optional["asyncio.Task[None]"] = None
        self.created = time.monotonic()

    def append(self, frame: Frame) -> None:
        self.frames.append(frame)
        self.event.set()

    @property
    def end_seq(self) -> int:
        """Sequence number one past the last frame."""

        return self.base + len(self.frames)

    def trim_to(self, ack_seq: int) -> None:
        """Drop frames the peer has cumulatively acknowledged."""

        if ack_seq > self.base:
            drop = min(ack_seq - self.base, len(self.frames))
            del self.frames[:drop]
            self.base += drop


class HostService:
    """Bridges a transport to an LLM backend."""

    def __init__(
        self,
        transport: Transport,
        backend: LLMBackend,
        *,
        profile: Optional[ProfileConfig] = None,
        buffer_response: bool = False,
        mtu: int = DEFAULT_ATT_MTU,
        max_payload_size: Optional[int] = None,
        compress_responses: bool = True,
        window: Optional[int] = None,
        rto: float = 0.75,
        max_retries: int = 20,
        resume_ttl: float = 300.0,
    ) -> None:
        self._transport = transport
        self._backend = backend
        self._profile = profile or LEGACY_PROFILE
        self._buffer_response = buffer_response
        self._negotiated_mtu: int = DEFAULT_ATT_MTU
        self._max_payload = (
            max_payload_size
            if max_payload_size is not None
            else max_payload_for_mtu(profile.mtu if profile is not None else mtu)
        )
        if self._max_payload <= 0:
            raise ValueError("max_payload_size must be positive")
        # Cap for HELLO negotiation: never emit frames larger than this even if
        # the consumer claims a bigger link MTU.
        self._max_payload_cap = self._max_payload
        self._compress_responses = compress_responses
        # A profile with reliability_window=0 marks a reliable stream transport
        # (L2CAP CoC / RFCOMM) where Go-Back-N is unnecessary; the pump still
        # works there (nothing drops), so fall back to a sane window size.
        self._window = (
            window
            if window is not None
            else (self._profile.reliability_window or 64)
        )
        self._rto = rto
        self._max_retries = max_retries
        self._resume_ttl = resume_ttl
        self._reassembler = Reassembler()
        self._tasks: Set["asyncio.Task[None]"] = set()
        self._messages: Dict[int, _OutboundMessage] = {}
        # Reliable senders used by the buffered/FAST_GATT handlers (which do
        # not use the _OutboundMessage pump).
        self._senders: Dict[int, ReliableSender] = {}
        self._gen_tasks: Dict[int, "asyncio.Task[None]"] = {}
        self._started = False

    async def start(self) -> None:
        """Register the frame handler and bring the transport up."""

        self._transport.on_receive(self._on_frame)
        await self._transport.start()
        self._started = True

    async def close(self) -> None:
        """Cancel in-flight handlers and tear the transport down."""

        for task in list(self._tasks) + list(self._gen_tasks.values()):
            task.cancel()
        tasks = list(self._tasks) + list(self._gen_tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._gen_tasks.clear()
        self._messages.clear()
        await self._transport.close()
        self._started = False

    async def __aenter__(self) -> "HostService":
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # -- frame intake ---------------------------------------------------------

    async def _on_frame(self, data: bytes) -> None:
        try:
            frame = Frame.decode(data)
        except ProtocolError:
            logger.exception("dropping malformed frame")
            return

        # ACKs from the consumer drive retransmission of the response stream;
        # route them to the message for the matching id and stop here.
        if frame.msg_type == MessageType.ACK:
            state = self._messages.get(frame.message_id)
            if state is not None:
                state.trim_to(frame.seq)
                if state.sender is not None:
                    state.sender.on_ack(frame.seq)
            else:
                sender = self._senders.get(frame.message_id)
                if sender is not None:
                    sender.on_ack(frame.seq)
                else:
                    logger.debug("ACK for unknown message %s", frame.message_id)
            return

        # SELECT arrives before any prompts and configures the mode for this
        # connection; handle it directly without going through the Reassembler.
        if frame.msg_type == MessageType.SELECT:
            self._handle_select(frame)
            return

        try:
            message = self._reassembler.feed(frame)
        except ProtocolError as exc:
            logger.exception("dropping malformed frame")
            # Fail fast: without this the consumer waits out its whole timeout
            # for a response that can never come.
            self._reassembler.discard(frame.message_id)
            if frame.message_id != 0:
                self._spawn(self._send_nack(frame.message_id, str(exc)))
            return

        if message is None:
            logger.debug("Received frame fragment (message incomplete)")
            return

        try:
            await self._handle_message(message)
        except ProtocolError:
            logger.exception("dropping malformed control message")

    async def _handle_message(self, message) -> None:  # noqa: ANN001
        if message.msg_type == MessageType.PROMPT:
            if message.message_id in self._messages:
                # A duplicated prompt (e.g. a write retried across a reconnect)
                # must not start a second generation for the same request.
                logger.warning(
                    "duplicate prompt for message %s; ignoring", message.message_id
                )
                return
            logger.info(
                "Received complete prompt (msg_id=%d): %r",
                message.message_id,
                message.text()[:100],
            )
            self._purge_messages()
            self._spawn(
                self._handle_prompt(message.message_id, message.text()),
                message_id=message.message_id,
            )
        elif message.msg_type == MessageType.CHAT:
            if message.message_id in self._messages:
                logger.warning(
                    "duplicate chat request for message %s; ignoring",
                    message.message_id,
                )
                return
            self._purge_messages()
            self._spawn(
                self._handle_chat(message.message_id, message.payload),
                message_id=message.message_id,
            )
        elif message.msg_type == MessageType.LIST_MODELS:
            if message.message_id in self._messages:
                logger.warning(
                    "duplicate model-list request for %s; ignoring",
                    message.message_id,
                )
                return
            self._purge_messages()
            self._spawn(
                self._handle_list_models(message.message_id),
                message_id=message.message_id,
            )
        elif message.msg_type == MessageType.HELLO:
            self._handle_hello(message.payload)
        elif message.msg_type == MessageType.RESUME:
            target_id, ack_seq = decode_resume(message.payload)
            logger.info(
                "RESUME requested for message %s from seq %d", target_id, ack_seq
            )
            self._spawn(self._handle_resume(target_id, ack_seq))
        elif message.msg_type == MessageType.CANCEL:
            self._handle_cancel(decode_cancel(message.payload))
        elif message.msg_type == MessageType.PING:
            self._spawn(self._send_pong())
        else:
            logger.warning(
                "ignoring unexpected message type %s", message.msg_type.name
            )

    def _handle_select(self, frame: Frame) -> None:
        """Apply the mode selection sent by the consumer."""
        try:
            mode_id, reported_mtu = decode_select_payload(frame.payload)
        except ProtocolError as exc:
            logger.warning("Ignoring malformed SELECT frame: %s", exc)
            return
        try:
            mode = mode_for_id(mode_id)
        except ValueError:
            logger.warning("SELECT: unknown mode_id %d — staying on legacy", mode_id)
            return
        try:
            if mode == Mode.FAST_GATT:
                new_profile = make_fast_gatt_profile(reported_mtu)
            elif mode == Mode.L2CAP_COC:
                new_profile = make_l2cap_coc_profile(reported_mtu)
            elif mode == Mode.CLASSIC_RFCOMM:
                new_profile = make_classic_rfcomm_profile(reported_mtu)
            else:
                new_profile = profile_for_mode(mode.value)
        except ValueError as exc:
            logger.warning("SELECT: mode %r not implemented — staying on legacy: %s", mode.value, exc)
            return
        prev_mode = self._profile.mode
        self._profile = new_profile
        self._negotiated_mtu = reported_mtu
        # Only resize response frames for modes that exploit larger MTUs.
        if new_profile.mode != Mode.LEGACY:
            self._max_payload = max_payload_for_mtu(reported_mtu)
            self._max_payload_cap = max(self._max_payload_cap, self._max_payload)
        if prev_mode != new_profile.mode:
            logger.info(
                "Bluetooth mode switched: %s → %s (consumer MTU=%d, max_payload=%d)",
                prev_mode.value, new_profile.mode.value, reported_mtu, self._max_payload,
            )
        else:
            logger.info(
                "Bluetooth mode: %s (consumer MTU=%d, max_payload=%d)",
                new_profile.mode.value, reported_mtu, self._max_payload,
            )

    def _spawn(self, coro: "asyncio.coroutines", message_id: Optional[int] = None) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        if message_id is not None:
            self._gen_tasks[message_id] = task

    # -- control messages -----------------------------------------------------

    def _handle_hello(self, payload: bytes) -> None:
        """Adopt the consumer's frame payload size (capped by ours)."""

        peer_max = decode_max_payload(payload)
        new_max = min(self._max_payload_cap, peer_max)
        if new_max != self._max_payload:
            logger.info(
                "frame payload negotiated: %d -> %d bytes", self._max_payload, new_max
            )
            self._max_payload = new_max

    async def _send_pong(self) -> None:
        frame = Frame(
            msg_type=MessageType.PONG,
            message_id=0,
            seq=0,
            payload=b"",
            flags=Flags.START | Flags.END,
        )
        try:
            await self._transport.send(frame.encode())
        except Exception:  # noqa: BLE001 - keepalive is best effort
            logger.debug("failed to send PONG", exc_info=True)

    def _handle_cancel(self, target_id: int) -> None:
        """Abort generation and delivery for a request the consumer abandoned."""

        task = self._gen_tasks.pop(target_id, None)
        if task is not None:
            task.cancel()
        state = self._messages.pop(target_id, None)
        if state is not None and state.task is not None:
            state.task.cancel()
        logger.info("cancelled message %s", target_id)

    async def _handle_resume(self, target_id: int, ack_seq: int) -> None:
        state = self._messages.get(target_id)
        if state is None:
            # We no longer know this message: tell the consumer so it can
            # restart the request from scratch instead of waiting forever.
            await self._send_nack(target_id, RESUME_UNKNOWN)
            return
        await self._resume_message(state, ack_seq)

    async def _resume_message(self, state: _OutboundMessage, ack_seq: int) -> None:
        state.trim_to(ack_seq)
        sender = state.sender
        if (
            sender is not None
            and sender.alive
            and state.task is not None
            and not state.task.done()
        ):
            # Delivery is still running: drop what the peer confirms it has and
            # retransmit the rest immediately.
            sender.on_ack(ack_seq)
            sender.nudge()
            return
        # Delivery is dead (or finished): restart it from the peer's position.
        if state.task is not None and not state.task.done():
            state.task.cancel()
            await asyncio.gather(state.task, return_exceptions=True)
            state.task = None
        if not state.delivered:
            logger.info("replaying message %s from seq %d", state.message_id, ack_seq)
            self._spawn_pump(state)

    async def _send_nack(self, message_id: int, text: str) -> None:
        """Send an ERROR message for a request that will never be answered.

        Used for prompt reassembly failures (nothing was sent yet) and for
        RESUME of a message we no longer have.  In both cases the sequence
        space starts fresh at 0.
        """

        state = self._messages.get(message_id)
        if state is not None and state.end_seq > state.base:
            # A stream is already in progress; its own error handling applies.
            logger.warning("NACK skipped: message %s already streaming", message_id)
            return
        if state is None:
            state = self._new_message(message_id)
        await self._send_error_frames(state, None, message_id, text)

    # -- outbound message pumping ---------------------------------------------

    def _new_message(self, message_id: int) -> _OutboundMessage:
        state = _OutboundMessage(message_id)
        self._messages[message_id] = state
        self._spawn_pump(state)
        return state

    def _spawn_pump(self, state: _OutboundMessage) -> None:
        task = asyncio.ensure_future(self._pump(state))
        state.task = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _pump(self, state: _OutboundMessage) -> None:
        """Deliver the message's frames reliably, as fast as they appear.

        If the link dies the pump gives up but the frames are kept: the
        consumer's RESUME restarts delivery from wherever it stopped.
        """

        sender = ReliableSender(
            self._transport.send,
            window=self._window,
            rto=self._rto,
            max_retries=self._max_retries,
        )
        state.sender = sender
        sender.start()
        cursor = state.base
        try:
            while True:
                if cursor < state.base:
                    cursor = state.base
                while cursor < state.end_seq:
                    await sender.send_frame(state.frames[cursor - state.base])
                    cursor += 1
                if state.complete:
                    # Block until the consumer has acknowledged every frame so
                    # a final dropped notification is retransmitted before we
                    # consider the message delivered.
                    await sender.drain()
                    state.delivered = True
                    return
                state.event.clear()
                await state.event.wait()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - link failure; wait for RESUME
            logger.exception(
                "delivery failed for message %s; waiting for RESUME",
                state.message_id,
            )
        finally:
            sender.close()
            if state.sender is sender:
                state.sender = None

    def _purge_messages(self) -> None:
        """Drop expired delivery state so memory stays bounded."""

        now = time.monotonic()
        expired = [
            message_id
            for message_id, state in self._messages.items()
            if now - state.created > self._resume_ttl
        ]
        for message_id in expired:
            state = self._messages.pop(message_id)
            if state.task is not None and not state.task.done():
                state.task.cancel()

    # -- prompt handling ------------------------------------------------------

    async def _handle_prompt(self, message_id: int, prompt: str) -> None:
        """Dispatch prompt handling on the negotiated transfer mode.

        LEGACY and non-buffered FAST_GATT use the reliable message-pump path
        (retransmission/RESUME/TIMINGS); the buffered and stream modes use the
        mode-specific handlers kept from the transfer-mode work.
        """

        mode = self._profile.mode
        logger.info(
            "Prompt %d via %s (%d chars)", message_id, mode.value, len(prompt)
        )
        try:
            if mode == Mode.FAST_GATT and self._buffer_response:
                await self._handle_prompt_fast_gatt(message_id, prompt)
            elif mode in (Mode.L2CAP_COC, Mode.CLASSIC_RFCOMM) and self._buffer_response:
                await self._handle_prompt_stream(message_id, prompt)
            else:
                await self._handle_prompt_legacy(message_id, prompt)
        finally:
            self._gen_tasks.pop(message_id, None)

    async def _handle_prompt_legacy(self, message_id: int, prompt: str) -> None:
        state = self._new_message(message_id)
        encoder = FrameStreamEncoder(
            MessageType.RESPONSE,
            message_id,
            self._max_payload,
            compressed=self._compress_responses,
        )
        try:
            await self._generate_into(
                state, encoder, message_id, self._backend.generate(prompt)
            )
        except asyncio.CancelledError:
            self._messages.pop(message_id, None)
            raise
        except Exception as exc:  # noqa: BLE001 - report any backend failure
            logger.exception("backend failed for message %s", message_id)
            await self._send_error_frames(state, encoder, message_id, str(exc))
        finally:
            self._gen_tasks.pop(message_id, None)

    async def _handle_prompt_stream(self, message_id: int, prompt: str) -> None:
        """Collect full response, compress, CRC32, send directly (L2CAP_COC / RFCOMM path).

        Both L2CAP COC and RFCOMM provide reliable ordered streams, so the
        Go-Back-N ``ReliableSender`` is not needed; frames go to the transport directly.
        """
        logger.info(
            "msg_id=%d: buffering full response, then compress+chunk (%s)",
            message_id, self._profile.mode.value,
        )
        encoder = FrameStreamEncoder(MessageType.RESPONSE, message_id, self._max_payload)
        try:
            parts: list[bytes] = []
            async for fragment in self._backend.generate(prompt):
                if fragment:
                    parts.append(fragment.encode("utf-8"))
            full_response = b"".join(parts)
            logger.debug(
                "msg_id=%d: collected %d bytes, compressing...",
                message_id, len(full_response),
            )
            compressed = zlib.compress(full_response, level=6)
            if len(compressed) >= len(full_response):
                wire_payload = full_response
                use_compressed_flag = False
            else:
                wire_payload = compressed
                use_compressed_flag = True

            frames = chunk_message(
                MessageType.RESPONSE,
                message_id,
                wire_payload,
                self._max_payload,
                checksum=True,
            )
            if use_compressed_flag and frames:
                f0 = frames[0]
                frames[0] = Frame(
                    msg_type=f0.msg_type,
                    message_id=f0.message_id,
                    seq=f0.seq,
                    payload=f0.payload,
                    flags=f0.flags | Flags.COMPRESSED,
                    version=f0.version,
                )
            for frame in frames:
                await self._transport.send(frame.encode())
            logger.info(
                "Completed %s prompt %s (%d raw bytes → %d wire bytes, %d frames)",
                self._profile.mode.value,
                message_id, len(full_response), len(wire_payload), len(frames),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("backend failed for message %s (%s)", message_id, self._profile.mode.value)
            # Send an error message directly without a ReliableSender.
            payload = str(exc).encode("utf-8") or b"backend error"
            pieces = [
                payload[i : i + self._max_payload]
                for i in range(0, len(payload), self._max_payload)
            ] or [b""]
            for idx, piece in enumerate(pieces):
                flags = Flags.NONE
                if idx == 0 and not encoder.has_started:
                    flags |= Flags.START
                if idx == len(pieces) - 1:
                    flags |= Flags.END
                await self._transport.send(
                    Frame(
                        msg_type=MessageType.ERROR,
                        message_id=message_id,
                        seq=encoder.next_seq + idx,
                        payload=piece,
                        flags=flags,
                    ).encode()
                )

    # Keep the old name as an alias so existing test suites that reference it directly still pass.
    # Historical names kept as aliases: the streaming variants now share the
    # reliable message-pump pipeline (whole-stream compression, RESUME, TIMINGS).
    _handle_prompt_l2cap_coc = _handle_prompt_stream
    _handle_prompt_stream_chunked = _handle_prompt_legacy

    async def _handle_prompt_fast_gatt(self, message_id: int, prompt: str) -> None:
        """Collect full response, compress, checksum, then send (FAST_GATT path)."""
        logger.info(
            "msg_id=%d: buffering full response, then compress+chunk (FAST_GATT, window=%d)",
            message_id, self._profile.reliability_window,
        )
        sender = ReliableSender(
            self._transport.send,
            window=self._profile.reliability_window,
        )
        self._senders[message_id] = sender
        sender.start()
        # Dummy encoder only used for _send_error path.
        encoder = FrameStreamEncoder(MessageType.RESPONSE, message_id, self._max_payload)
        try:
            parts: list[bytes] = []
            async for fragment in self._backend.generate(prompt):
                if fragment:
                    parts.append(fragment.encode("utf-8"))
            full_response = b"".join(parts)
            logger.debug(
                "msg_id=%d: collected %d bytes, compressing...",
                message_id, len(full_response),
            )
            # Compress the full response.
            compressed = zlib.compress(full_response, level=6)
            if len(compressed) >= len(full_response):
                # Not worth it — send raw.
                wire_payload = full_response
                use_compressed_flag = False
            else:
                wire_payload = compressed
                use_compressed_flag = True

            frames = chunk_message(
                MessageType.RESPONSE,
                message_id,
                wire_payload,
                self._max_payload,
                checksum=True,
            )
            # Set COMPRESSED on the START frame if the payload is compressed.
            if use_compressed_flag and frames:
                f0 = frames[0]
                frames[0] = Frame(
                    msg_type=f0.msg_type,
                    message_id=f0.message_id,
                    seq=f0.seq,
                    payload=f0.payload,
                    flags=f0.flags | Flags.COMPRESSED,
                    version=f0.version,
                )
            for frame in frames:
                await sender.send_frame(frame)
            await sender.drain()
            logger.info(
                "Completed fast_gatt prompt %s (%d raw bytes → %d wire bytes, %d frames)",
                message_id, len(full_response), len(wire_payload), len(frames),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("backend failed for message %s (fast_gatt)", message_id)
            await self._send_error(sender, encoder, message_id, str(exc))
        finally:
            sender.close()
            self._senders.pop(message_id, None)

    async def _handle_chat(self, message_id: int, payload: bytes) -> None:
        """Handle a CHAT message: structured messages for the backend's template.

        Unlike PROMPT (a pre-rendered string), the payload is JSON of the form
        ``{"messages": [...], "params": {...}}``.  Backends with native chat
        support (``generate_messages``) receive the messages untouched so the
        model's own chat template renders them; other backends fall back to a
        flattened transcript.
        """

        try:
            request = json.loads(payload.decode("utf-8")) if payload else {}
            if not isinstance(request, dict):
                raise ValueError("chat payload must be a JSON object")
            messages = request.get("messages") or []
            params = request.get("params") or {}
            if not isinstance(params, dict):
                raise ValueError("params must be a JSON object")
        except (ValueError, UnicodeDecodeError) as exc:
            logger.warning("bad chat payload for message %s: %s", message_id, exc)
            state = self._new_message(message_id)
            await self._send_error_frames(
                state, None, message_id, f"bad chat payload: {exc}"
            )
            return
        logger.info(
            "Received chat request (msg_id=%d, %d messages)",
            message_id,
            len(messages),
        )
        state = self._new_message(message_id)
        encoder = FrameStreamEncoder(
            MessageType.RESPONSE,
            message_id,
            self._max_payload,
            compressed=self._compress_responses,
        )
        try:
            generate_messages = getattr(self._backend, "generate_messages", None)
            if generate_messages is not None:
                stream = generate_messages(messages, **params)
            else:
                stream = self._backend.generate(messages_to_prompt(messages))
            await self._generate_into(state, encoder, message_id, stream)
        except asyncio.CancelledError:
            self._messages.pop(message_id, None)
            raise
        except Exception as exc:  # noqa: BLE001 - report any backend failure
            logger.exception("backend failed for message %s", message_id)
            await self._send_error_frames(state, encoder, message_id, str(exc))
        finally:
            self._gen_tasks.pop(message_id, None)

    async def _generate_into(
        self, state: _OutboundMessage, encoder: FrameStreamEncoder, message_id: int, stream
    ) -> None:
        """Stream a backend's tokens into response frames and finish cleanly."""

        token_count = 0
        async for fragment in stream:
            if not fragment:
                continue
            token_count += 1
            if token_count % 10 == 0:  # Log every 10th token in verbose mode
                logger.debug(
                    "Received %d tokens so far (msg_id=%d)", token_count, message_id
                )
            for frame in encoder.push(fragment.encode("utf-8")):
                state.append(frame)
        self._finish_message(state, encoder, getattr(stream, "timings", None))
        logger.info(
            "Completed message %s (%d tokens generated)", message_id, token_count
        )

    def _finish_message(
        self,
        state: _OutboundMessage,
        encoder: FrameStreamEncoder,
        timings: Optional[dict] = None,
    ) -> None:
        """Emit the message tail: optional TIMINGS trailer, then the END frame.

        When the backend reports timings (llama.cpp attaches generation stats,
        incl. MTP draft acceptance), they ride along as plain-text TIMINGS
        frames at the tail of the *same* message - after the compressed stream
        is cut - so they are guaranteed to arrive before END and cannot race
        the consumer's request teardown.  Timings are telemetry: if a trailer
        frame is ever lost the consumer simply ends without stats.
        """

        if not timings:
            for frame in encoder.finish():
                state.append(frame)
        else:
            # Z_SYNC_FLUSH: every compressed byte is emitted and decodable
            # without ending the zlib stream (which END never does in this
            # path - the trailer is plain text).
            for frame in encoder.flush():
                state.append(frame)
            payload = json.dumps(
                {"timings": timings}, separators=(",", ":")
            ).encode("utf-8")
            pieces = [
                payload[i : i + self._max_payload]
                for i in range(0, len(payload), self._max_payload)
            ] or [b""]
            last = len(pieces) - 1
            for index, piece in enumerate(pieces):
                flags = Flags.NONE
                if index == 0 and not encoder.has_started:
                    flags |= Flags.START
                if index == last:
                    flags |= Flags.END
                state.append(
                    Frame(
                        msg_type=MessageType.TIMINGS,
                        message_id=state.message_id,
                        seq=state.end_seq,
                        payload=piece,
                        flags=flags,
                    )
                )
        state.complete = True
        state.event.set()

    async def _handle_list_models(self, message_id: int) -> None:
        """Answer a LIST_MODELS request with the backend's available models."""

        state = self._new_message(message_id)
        encoder = FrameStreamEncoder(
            MessageType.RESPONSE,
            message_id,
            self._max_payload,
            compressed=self._compress_responses,
        )
        try:
            try:
                names = await self._backend.list_models()
            except Exception as exc:  # noqa: BLE001 - report backend failure
                logger.exception("backend failed to list models for %s", message_id)
                await self._send_error_frames(state, encoder, message_id, str(exc))
                return
            payload = json.dumps({"models": list(names)}).encode("utf-8")
            for frame in encoder.push(payload):
                state.append(frame)
            self._finish_message(state, encoder)
            logger.info(
                "Answered model-list request %s (%d models)", message_id, len(names)
            )
        except asyncio.CancelledError:
            self._messages.pop(message_id, None)
            raise
        finally:
            self._gen_tasks.pop(message_id, None)

    async def _send_error_frames(
        self,
        state: _OutboundMessage,
        encoder: Optional[FrameStreamEncoder],
        message_id: int,
        text: str,
    ) -> None:
        # Error frames continue the same sequence stream as the (possibly
        # partial) response so the consumer's cumulative ACKs stay consistent.
        # The first frame carries START only if no response frame did.
        seq = state.end_seq
        if encoder is not None:
            # Flush pending data (and the compressor) so the compressed stream
            # is cleanly cut before the plain-text error frames begin.  This
            # may emit the START frame itself, so it must run before deciding
            # whether the error frames need one.
            for frame in encoder.flush():
                state.append(frame)
            seq = state.end_seq
        started = encoder is not None and encoder.has_started
        payload = text.encode("utf-8")
        pieces = [
            payload[i : i + self._max_payload]
            for i in range(0, len(payload), self._max_payload)
        ] or [b""]
        last = len(pieces) - 1
        for index, piece in enumerate(pieces):
            flags = Flags.NONE
            if index == 0 and not started:
                flags |= Flags.START
            if index == last:
                flags |= Flags.END
            state.append(
                Frame(
                    msg_type=MessageType.ERROR,
                    message_id=message_id,
                    seq=seq,
                    payload=piece,
                    flags=flags,
                )
            )
            seq += 1
        state.complete = True
        state.event.set()
