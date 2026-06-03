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
import logging
from typing import AsyncIterator, Dict, Optional

from .protocol import (
    DEFAULT_ATT_MTU,
    Frame,
    MessageType,
    ProtocolError,
    chunk_message,
    max_payload_for_mtu,
)
from .transport import Transport

__all__ = ["ConsumerClient", "RemoteError", "_PendingRequest"]

logger = logging.getLogger("keytalk.consumer")

DEFAULT_TIMEOUT = 300.0


class RemoteError(Exception):
    """Raised when the host returns an ERROR message for a request."""


class _PendingRequest:
    """Tracks reassembly and streaming for one in-flight request.

    Response (or error) frames are validated for ordering and pushed onto a
    queue so consumers can stream them.  Completion is signalled with a sentinel
    so an async iterator terminates cleanly.
    """

    _END = object()

    def __init__(self, message_id: int) -> None:
        self.message_id = message_id
        self._queue: "asyncio.Queue[object]" = asyncio.Queue()
        self._next_seq = 0
        self._started = False
        self._done = False

    def feed(self, frame: Frame) -> None:
        """Validate and enqueue an inbound frame for this request."""

        if self._done:
            logger.debug("frame for already-finished request %s", self.message_id)
            return

        if frame.is_start:
            self._started = True
            self._next_seq = 0
        elif not self._started:
            self._fail(
                ProtocolError(
                    f"response {self.message_id} started without a START frame"
                )
            )
            return

        if frame.seq != self._next_seq:
            self._fail(
                ProtocolError(
                    f"out-of-order response frame for {self.message_id}: "
                    f"expected {self._next_seq}, got {frame.seq}"
                )
            )
            return
        self._next_seq += 1

        if frame.msg_type == MessageType.ERROR:
            # Error payloads can also span multiple frames; accumulate until END.
            self._queue.put_nowait(("error", frame.payload))
        elif frame.msg_type == MessageType.RESPONSE:
            self._queue.put_nowait(("data", frame.payload))
        else:
            self._fail(
                ProtocolError(
                    f"unexpected response type {frame.msg_type.name} "
                    f"for {self.message_id}"
                )
            )
            return

        if frame.is_end:
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
        mtu: int = DEFAULT_ATT_MTU,
        max_payload_size: Optional[int] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._transport = transport
        self._max_payload = (
            max_payload_size
            if max_payload_size is not None
            else max_payload_for_mtu(mtu)
        )
        if self._max_payload <= 0:
            raise ValueError("max_payload_size must be positive")
        self._timeout = timeout
        self._pending: Dict[int, _PendingRequest] = {}
        # message_id 0 is reserved/avoided; ids wrap within the 16-bit space.
        self._ids = itertools.cycle(range(1, 0x10000))

    async def start(self) -> None:
        self._transport.on_receive(self._on_frame)
        await self._transport.start()

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
        pending = self._pending.get(frame.message_id)
        if pending is None:
            logger.debug("response for unknown message %s", frame.message_id)
            return
        if frame.is_start:
            logger.info("✓ Started receiving response (msg_id=%d)", frame.message_id)
        pending.feed(frame)

    # -- public API -----------------------------------------------------------

    def _alloc_id(self) -> int:
        for _ in range(0x10000):
            candidate = next(self._ids)
            if candidate not in self._pending:
                return candidate
        raise RuntimeError("no free message ids available")

    async def _send_prompt(self, message_id: int, prompt: str) -> None:
        logger.info("Sending prompt (msg_id=%d, %d chars)...", message_id, len(prompt))
        frames = chunk_message(
            MessageType.PROMPT,
            message_id,
            prompt.encode("utf-8"),
            self._max_payload,
        )
        for frame in frames:
            await self._transport.send(frame.encode())
        logger.info("✓ Prompt sent, waiting for response...")

    def stream(self, prompt: str) -> AsyncIterator[str]:
        """Send ``prompt`` and yield response text pieces as they arrive."""

        return self._stream(prompt)

    async def _stream(self, prompt: str) -> AsyncIterator[str]:
        message_id = self._alloc_id()
        pending = _PendingRequest(message_id)
        self._pending[message_id] = pending
        try:
            await self._send_prompt(message_id, prompt)
            iterator = pending.__aiter__()
            while True:
                try:
                    piece = await asyncio.wait_for(
                        iterator.__anext__(), timeout=self._timeout
                    )
                except StopAsyncIteration:
                    break
                yield piece
        finally:
            self._pending.pop(message_id, None)

    async def generate(self, prompt: str) -> str:
        """Send ``prompt`` and return the complete response text."""

        parts = [piece async for piece in self._stream(prompt)]
        return "".join(parts)
