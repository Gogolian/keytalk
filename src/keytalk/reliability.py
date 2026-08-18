"""Reliable delivery on top of the lossy host->consumer notification channel.

BLE *notifications* (the host -> consumer RESPONSE path) are fire-and-forget:
the controller may drop one if values are pushed faster than the radio can
transmit them, and there is no link-layer acknowledgement.  The consumer ->
host PROMPT path, by contrast, uses *write-with-response* and is already
acknowledged at the link layer.

This module adds an application-level reliability layer over the notification
path using cumulative acknowledgements and Go-Back-N retransmission:

* The consumer ACKs the highest contiguous sequence number it has received by
  writing a small ``ACK`` frame back over the (reliable) PROMPT channel.
* The host keeps every sent-but-unacked RESPONSE/ERROR frame in a bounded
  window and retransmits the window if an ACK does not arrive within a timeout.

The result is that a dropped notification is recovered transparently instead of
aborting the whole stream.  Everything here is transport agnostic and operates
on :class:`~keytalk.protocol.Frame` objects so it can be unit-tested with plain
bytes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable, Dict, List, Optional

from .protocol import Flags, Frame, MessageType

__all__ = ["make_ack_frame", "ReliableSender"]

logger = logging.getLogger("keytalk.reliability")

SendBytes = Callable[[bytes], Awaitable[None]]


def make_ack_frame(message_id: int, ack_seq: int) -> Frame:
    """Build a cumulative ACK frame.

    ``ack_seq`` is the next sequence number the receiver expects, i.e. every
    frame with ``seq < ack_seq`` has been received contiguously.
    """

    return Frame(
        msg_type=MessageType.ACK,
        message_id=message_id,
        seq=ack_seq,
        payload=b"",
        flags=Flags.START | Flags.END,
    )


class _Inflight:
    """Bookkeeping for a single sent-but-unacked frame."""

    __slots__ = ("data", "sent_at", "retries")

    def __init__(self, data: bytes, sent_at: float) -> None:
        self.data = data
        self.sent_at = sent_at
        self.retries = 0


class ReliableSender:
    """Sliding-window reliable sender for one outbound message.

    Frames are buffered until cumulatively acknowledged by the peer; unacked
    frames are retransmitted after ``rto`` seconds.  A bounded ``window``
    provides flow control so a slow consumer cannot be overrun.
    """

    def __init__(
        self,
        send_bytes: SendBytes,
        *,
        window: int = 32,
        rto: float = 0.75,
        max_retries: int = 10,
    ) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self._send_bytes = send_bytes
        self._window = window
        self._rto = rto
        self._max_retries = max_retries
        self._inflight: Dict[int, _Inflight] = {}
        self._failed: Optional[BaseException] = None
        self._retransmit_task: Optional["asyncio.Task[None]"] = None
        # Set whenever the window has room or the sender has failed.
        self._window_event = asyncio.Event()
        self._window_event.set()
        # Set whenever there are no inflight frames (or the sender has failed).
        self._drained_event = asyncio.Event()
        self._drained_event.set()

    def start(self) -> None:
        """Spawn the background retransmission loop."""

        if self._retransmit_task is None:
            self._retransmit_task = asyncio.ensure_future(self._retransmit_loop())

    async def send_frame(self, frame: Frame) -> None:
        """Transmit ``frame`` reliably, blocking while the window is full."""

        _logged_blocked = False
        while True:
            if self._failed is not None:
                raise self._failed
            if len(self._inflight) < self._window:
                break
            if not _logged_blocked:
                logger.debug(
                    "window full (%d/%d inflight) — waiting for ACK before seq=%d",
                    len(self._inflight), self._window, frame.seq,
                )
                _logged_blocked = True
            self._window_event.clear()
            await self._window_event.wait()

        data = frame.encode()
        self._inflight[frame.seq] = _Inflight(data, time.monotonic())
        self._drained_event.clear()
        await self._send_bytes(data)

    def on_ack(self, ack_seq: int) -> None:
        """Process a cumulative ACK: drop every frame with ``seq < ack_seq``."""

        removed = False
        for seq in list(self._inflight):
            if seq < ack_seq:
                del self._inflight[seq]
                removed = True
        if not removed:
            return
        if len(self._inflight) < self._window:
            self._window_event.set()
        if not self._inflight:
            self._drained_event.set()

    async def drain(self) -> None:
        """Wait until every sent frame has been acknowledged."""

        while self._inflight and self._failed is None:
            self._drained_event.clear()
            waiter = asyncio.ensure_future(self._drained_event.wait())
            try:
                await waiter
            finally:
                if not waiter.done():
                    waiter.cancel()
        if self._failed is not None:
            raise self._failed

    def close(self) -> None:
        """Stop the retransmission loop and release any waiters."""

        if self._retransmit_task is not None:
            self._retransmit_task.cancel()
            self._retransmit_task = None
        self._window_event.set()
        self._drained_event.set()

    # -- internals ------------------------------------------------------------

    def _fail(self, exc: BaseException) -> None:
        if self._failed is None:
            self._failed = exc
        # Wake anything blocked in send_frame / drain so the error propagates.
        self._window_event.set()
        self._drained_event.set()

    async def _retransmit_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._rto / 2)
                now = time.monotonic()
                for seq, entry in list(self._inflight.items()):
                    if now - entry.sent_at < self._rto:
                        continue
                    if entry.retries >= self._max_retries:
                        self._fail(
                            TimeoutError(
                                f"frame seq={seq} not acknowledged after "
                                f"{entry.retries} retransmissions"
                            )
                        )
                        return
                    entry.retries += 1
                    entry.sent_at = now
                    logger.debug(
                        "retransmitting seq=%d (attempt %d/%d)",
                        seq,
                        entry.retries,
                        self._max_retries,
                    )
                    try:
                        await self._send_bytes(entry.data)
                    except Exception as exc:  # noqa: BLE001 - surface to sender
                        self._fail(exc)
                        return
        except asyncio.CancelledError:
            raise
