"""The HOST side: receive prompt frames, run the LLM, stream back the answer.

The host owns a :class:`~keytalk.transport.Transport` (in production a BLE
peripheral) and an :class:`~keytalk.backends.LLMBackend`.  Incoming frames are
reassembled into prompt messages; each prompt is handled on its own task so
several consumers (or pipelined requests) can be served concurrently.  The LLM's
streamed output is re-chunked into RESPONSE frames and sent back, correlated to
the request by ``message_id``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import zlib
from typing import Dict, Optional, Set

from .backends import LLMBackend
from .modes import LEGACY_PROFILE, Mode, ProfileConfig, make_classic_rfcomm_profile, make_fast_gatt_profile, make_l2cap_coc_profile, mode_for_id, profile_for_mode
from .protocol import (
    DEFAULT_ATT_MTU,
    FrameStreamEncoder,
    Flags,
    Frame,
    MessageType,
    ProtocolError,
    Reassembler,
    chunk_message,
    max_payload_for_mtu,
    compute_message_checksum,
    decode_delta_payload,
    decode_select_payload,
)
from .reliability import ReliableSender
from .transport import Transport

__all__ = ["HostService"]

logger = logging.getLogger("keytalk.host")


class HostService:
    """Bridges a transport to an LLM backend."""

    def __init__(
        self,
        transport: Transport,
        backend: LLMBackend,
        *,
        profile: Optional[ProfileConfig] = None,
        mtu: int = DEFAULT_ATT_MTU,
        max_payload_size: Optional[int] = None,
        buffer_response: bool = False,
    ) -> None:
        self._transport = transport
        self._backend = backend
        self._profile = profile or LEGACY_PROFILE
        self._buffer_response = buffer_response
        self._max_payload = (
            max_payload_size
            if max_payload_size is not None
            else max_payload_for_mtu(mtu)
        )
        if self._max_payload <= 0:
            raise ValueError("max_payload_size must be positive")
        self._reassembler = Reassembler()
        self._tasks: Set["asyncio.Task[None]"] = set()
        self._senders: Dict[int, ReliableSender] = {}
        self._started = False
        self._negotiated_mtu: int = DEFAULT_ATT_MTU
        # Track conversation history for delta message reconstruction
        self._conversation_history: bytes = b""
        self._history_checksum: str = ""

    async def start(self) -> None:
        """Register the frame handler and bring the transport up."""

        self._transport.on_receive(self._on_frame)
        await self._transport.start()
        self._started = True

    async def close(self) -> None:
        """Cancel in-flight handlers and tear the transport down."""

        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
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
        # route them to the sender for the matching message and stop here.
        if frame.msg_type == MessageType.ACK:
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
        except ProtocolError:
            logger.exception("dropping malformed frame")
            return

        if message is None:
            logger.debug("Received frame fragment (message incomplete)")
            return
        if message.msg_type == MessageType.PROMPT:
            logger.info("Received complete prompt (msg_id=%d): %r", message.message_id, message.text()[:100])
            self._spawn(self._handle_prompt(message.message_id, message.text()))
        elif message.msg_type == MessageType.LIST_MODELS:
            logger.info("Received model-list request (msg_id=%d)", message.message_id)
            self._spawn(self._handle_list_models(message.message_id))
        elif message.msg_type == MessageType.CANCEL:
            # Cancellation targets a prompt by id; the cooperative model here is
            # that the consumer simply stops reading.  A full implementation
            # would track and cancel the matching task.
            logger.debug("received cancel for message %s", message.message_id)
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

    def _spawn(self, coro: "asyncio.coroutines") -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # -- prompt handling ------------------------------------------------------

    async def _handle_prompt(self, message_id: int, prompt: str) -> None:
        mode = self._profile.mode
        logger.info(
            "Prompt %d via %s (%d chars)",
            message_id, mode.value, len(prompt),
        )

        if mode == Mode.FAST_GATT:
            if self._buffer_response:
                await self._handle_prompt_fast_gatt(message_id, prompt)
            else:
                await self._handle_prompt_legacy(message_id, prompt, compress=True)
        elif mode in (Mode.L2CAP_COC, Mode.CLASSIC_RFCOMM):
            if self._buffer_response:
                await self._handle_prompt_stream(message_id, prompt)
            else:
                await self._handle_prompt_stream_chunked(message_id, prompt)
        else:
            await self._handle_prompt_legacy(message_id, prompt)

    async def _handle_prompt_legacy(self, message_id: int, prompt: str, *, compress: bool = False) -> None:
        """Stream response frames one token at a time (LEGACY / default path)."""
        logger.info(
            "msg_id=%d: streaming token-by-token via GATT notify (window=%d, compress=%s)",
            message_id, self._profile.reliability_window, compress,
        )
        encoder = FrameStreamEncoder(
            MessageType.RESPONSE, message_id, self._max_payload,
            start_flags=Flags.COMPRESSED if compress else Flags.NONE,
        )
        sender = ReliableSender(
            self._transport.send,
            window=self._profile.reliability_window,
        )
        self._senders[message_id] = sender
        sender.start()
        compressor = zlib.compressobj(level=6) if compress else None
        try:
            token_count = 0
            async for fragment in self._backend.generate(prompt):
                if not fragment:
                    continue
                token_count += 1
                if token_count % 10 == 0:  # Log every 10th token in verbose mode
                    logger.debug("Received %d tokens so far (msg_id=%d)", token_count, message_id)
                data = fragment.encode("utf-8")
                if compressor is not None:
                    data = compressor.compress(data) + compressor.flush(zlib.Z_SYNC_FLUSH)
                for frame in encoder.push(data):
                    await sender.send_frame(frame)
            if compressor is not None:
                final_data = compressor.flush(zlib.Z_FINISH)
                if final_data:
                    for frame in encoder.push(final_data):
                        await sender.send_frame(frame)
            for frame in encoder.finish():
                await sender.send_frame(frame)
            # Block until the consumer has acknowledged every frame so a final
            # dropped notification is retransmitted before we move on.
            await sender.drain()
            logger.info("Completed prompt %s (%d tokens generated)", message_id, token_count)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - report any backend failure
            logger.exception("backend failed for message %s", message_id)
            await self._send_error(sender, encoder, message_id, str(exc))
        finally:
            sender.close()
            self._senders.pop(message_id, None)

    async def _handle_prompt_stream_chunked(self, message_id: int, prompt: str) -> None:
        """Stream response with per-chunk compression directly (L2CAP_COC / RFCOMM streaming path)."""
        logger.info(
            "msg_id=%d: streaming per-token with zlib compression (%s)",
            message_id, self._profile.mode.value,
        )
        encoder = FrameStreamEncoder(
            MessageType.RESPONSE, message_id, self._max_payload,
            start_flags=Flags.COMPRESSED,
        )
        compressor = zlib.compressobj(level=6)
        try:
            async for fragment in self._backend.generate(prompt):
                if not fragment:
                    continue
                data = compressor.compress(fragment.encode("utf-8")) + compressor.flush(zlib.Z_SYNC_FLUSH)
                for frame in encoder.push(data):
                    await self._transport.send(frame.encode())
            final_data = compressor.flush(zlib.Z_FINISH)
            if final_data:
                for frame in encoder.push(final_data):
                    await self._transport.send(frame.encode())
            for frame in encoder.finish():
                await self._transport.send(frame.encode())
            logger.info(
                "Completed %s prompt %s (streamed with compression)",
                self._profile.mode.value, message_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("backend failed for message %s (%s)", message_id, self._profile.mode.value)
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
    _handle_prompt_l2cap_coc = _handle_prompt_stream

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

    async def _send_error(
        self,
        sender: ReliableSender,
        encoder: FrameStreamEncoder,
        message_id: int,
        text: str,
    ) -> None:
        # Error frames continue the same sequence stream as the (possibly
        # partial) response so the consumer's cumulative ACKs stay consistent.
        # The first frame carries START only if no response frame did.
        payload = text.encode("utf-8")
        pieces = [
            payload[i : i + self._max_payload]
            for i in range(0, len(payload), self._max_payload)
        ] or [b""]
        seq = encoder.next_seq
        last = len(pieces) - 1
        try:
            for index, piece in enumerate(pieces):
                flags = Flags.NONE
                if index == 0 and not encoder.has_started:
                    flags |= Flags.START
                if index == last:
                    flags |= Flags.END
                await sender.send_frame(
                    Frame(
                        msg_type=MessageType.ERROR,
                        message_id=message_id,
                        seq=seq,
                        payload=piece,
                        flags=flags,
                    )
                )
                seq += 1
            await sender.drain()
        except Exception:  # pragma: no cover - transport already failing
            logger.exception("failed to deliver error for message %s", message_id)

    async def _handle_list_models(self, message_id: int) -> None:
        """Answer a LIST_MODELS request with the backend's available models."""

        encoder = FrameStreamEncoder(
            MessageType.RESPONSE, message_id, self._max_payload
        )
        sender = ReliableSender(self._transport.send)
        self._senders[message_id] = sender
        sender.start()
        try:
            try:
                names = await self._backend.list_models()
            except Exception as exc:  # noqa: BLE001 - report backend failure
                logger.exception("backend failed to list models for %s", message_id)
                await self._send_error(sender, encoder, message_id, str(exc))
                return
            payload = json.dumps({"models": list(names)}).encode("utf-8")
            for frame in encoder.push(payload):
                await sender.send_frame(frame)
            for frame in encoder.finish():
                await sender.send_frame(frame)
            await sender.drain()
            logger.info(
                "Answered model-list request %s (%d models)", message_id, len(names)
            )
        except asyncio.CancelledError:
            raise
        finally:
            sender.close()
            self._senders.pop(message_id, None)
