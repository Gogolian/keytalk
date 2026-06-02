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
import logging
from typing import Optional, Set

from .backends import LLMBackend
from .protocol import (
    DEFAULT_ATT_MTU,
    FrameStreamEncoder,
    Frame,
    MessageType,
    ProtocolError,
    Reassembler,
    chunk_message,
    max_payload_for_mtu,
)
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
            message = self._reassembler.feed(frame)
        except ProtocolError:
            logger.exception("dropping malformed frame")
            return

        if message is None:
            return
        if message.msg_type == MessageType.PROMPT:
            self._spawn(self._handle_prompt(message.message_id, message.text()))
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
        logger.debug("handling prompt %s (%d chars)", message_id, len(prompt))
        encoder = FrameStreamEncoder(
            MessageType.RESPONSE, message_id, self._max_payload
        )
        try:
            async for fragment in self._backend.generate(prompt):
                if not fragment:
                    continue
                for frame in encoder.push(fragment.encode("utf-8")):
                    await self._transport.send(frame.encode())
            for frame in encoder.finish():
                await self._transport.send(frame.encode())
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - report any backend failure
            logger.exception("backend failed for message %s", message_id)
            await self._send_error(message_id, str(exc))

    async def _send_error(self, message_id: int, text: str) -> None:
        frames = chunk_message(
            MessageType.ERROR,
            message_id,
            text.encode("utf-8"),
            self._max_payload,
        )
        try:
            for frame in frames:
                await self._transport.send(frame.encode())
        except Exception:  # pragma: no cover - transport already failing
            logger.exception("failed to deliver error for message %s", message_id)
