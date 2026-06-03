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
from typing import Dict, Optional, Set

from .backends import LLMBackend
from .protocol import (
    DEFAULT_ATT_MTU,
    FrameStreamEncoder,
    Flags,
    Frame,
    MessageType,
    ProtocolError,
    Reassembler,
    max_payload_for_mtu,
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
        mtu: int = DEFAULT_ATT_MTU,
        max_payload_size: Optional[int] = None,
    ) -> None:
        self._transport = transport
        self._backend = backend
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

    def _spawn(self, coro: "asyncio.coroutines") -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # -- prompt handling ------------------------------------------------------

    async def _handle_prompt(self, message_id: int, prompt: str) -> None:
        logger.info("Starting to handle prompt %s (%d chars)", message_id, len(prompt))
        encoder = FrameStreamEncoder(
            MessageType.RESPONSE, message_id, self._max_payload
        )
        sender = ReliableSender(self._transport.send)
        self._senders[message_id] = sender
        sender.start()
        try:
            token_count = 0
            async for fragment in self._backend.generate(prompt):
                if not fragment:
                    continue
                token_count += 1
                logger.debug("Received token #%d from backend (msg_id=%d)", token_count, message_id)
                for frame in encoder.push(fragment.encode("utf-8")):
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
