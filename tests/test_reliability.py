"""Tests for the reliable-delivery layer over a lossy notification path.

The host -> consumer direction (RESPONSE notifications) is modelled as lossy by
dropping a configurable subset of frames the *first* time they are sent.  With
ACK/retransmission in place the stream must still arrive intact and in order.
"""

import asyncio
import unittest
from typing import AsyncIterator, List, Optional, Set

from keytalk.backends import LLMBackend, StaticBackend
from keytalk.consumer import ConsumerClient, _PendingRequest
from keytalk.host import HostService
from keytalk.protocol import Flags, Frame, MessageType
from keytalk.reliability import ReliableSender, make_ack_frame
from keytalk.transport import InMemoryTransport

TINY = 6


class _LossyTransport(InMemoryTransport):
    """Loopback transport that drops selected frames in one direction.

    Only frames whose decoded ``msg_type`` is in ``drop_types`` are eligible to
    be dropped, and each ``(message_id, seq)`` is dropped at most ``drop_count``
    times so retransmissions eventually get through.
    """

    def __init__(
        self,
        name: str = "",
        *,
        drop_seqs: Optional[Set[int]] = None,
        drop_types: Optional[Set[MessageType]] = None,
        drop_count: int = 1,
    ) -> None:
        super().__init__(name)
        self._drop_seqs = drop_seqs or set()
        self._drop_types = drop_types or {MessageType.RESPONSE}
        self._drop_count = drop_count
        self._dropped: dict = {}

    async def send(self, frame: bytes) -> None:
        try:
            decoded = Frame.decode(frame)
        except Exception:
            decoded = None
        if decoded is not None and decoded.msg_type in self._drop_types:
            key = (decoded.message_id, decoded.seq)
            if decoded.seq in self._drop_seqs and self._dropped.get(key, 0) < self._drop_count:
                self._dropped[key] = self._dropped.get(key, 0) + 1
                # Pretend the notification was sent but silently lost.
                self.sent.append(bytes(frame))
                return
        await super().send(frame)


def _lossy_loopback(drop_seqs: Set[int], drop_count: int = 1):
    host = _LossyTransport(
        "host", drop_seqs=drop_seqs, drop_types={MessageType.RESPONSE}, drop_count=drop_count
    )
    consumer = _LossyTransport("consumer")  # consumer->host path stays reliable
    host.link(consumer)
    consumer.link(host)
    return host, consumer


class ReliableSenderUnitTests(unittest.IsolatedAsyncioTestCase):
    async def test_retransmits_until_acked(self):
        sent: List[Frame] = []

        async def capture(data: bytes) -> None:
            sent.append(Frame.decode(data))

        sender = ReliableSender(capture, window=8, rto=0.05, max_retries=20)
        sender.start()
        try:
            frame = Frame(MessageType.RESPONSE, 1, 0, b"hi", Flags.START | Flags.END)
            await sender.send_frame(frame)
            # No ACK yet: the retransmit loop should resend the frame.
            await asyncio.sleep(0.2)
            self.assertGreater(len(sent), 1)
            # Once acknowledged, retransmission stops and drain returns.
            sender.on_ack(1)
            await asyncio.wait_for(sender.drain(), timeout=1.0)
            before = len(sent)
            await asyncio.sleep(0.15)
            self.assertEqual(len(sent), before)
        finally:
            sender.close()

    async def test_window_blocks_until_ack(self):
        sent: List[Frame] = []

        async def capture(data: bytes) -> None:
            sent.append(Frame.decode(data))

        sender = ReliableSender(capture, window=2, rto=5.0, max_retries=20)
        sender.start()
        try:
            await sender.send_frame(Frame(MessageType.RESPONSE, 1, 0, b"a", Flags.START))
            await sender.send_frame(Frame(MessageType.RESPONSE, 1, 1, b"b"))
            # Window is full; the third send must block until an ACK frees space.
            third = asyncio.ensure_future(
                sender.send_frame(Frame(MessageType.RESPONSE, 1, 2, b"c"))
            )
            await asyncio.sleep(0.05)
            self.assertFalse(third.done())
            sender.on_ack(1)  # acknowledges seq 0
            await asyncio.wait_for(third, timeout=1.0)
        finally:
            sender.close()

    async def test_fails_after_max_retries(self):
        async def capture(_data: bytes) -> None:
            pass

        sender = ReliableSender(capture, window=4, rto=0.02, max_retries=2)
        sender.start()
        try:
            await sender.send_frame(Frame(MessageType.RESPONSE, 1, 0, b"x", Flags.START))
            with self.assertRaises(TimeoutError):
                await asyncio.wait_for(sender.drain(), timeout=2.0)
        finally:
            sender.close()


class PendingRequestReorderTests(unittest.IsolatedAsyncioTestCase):
    def _frame(self, seq: int, payload: bytes, flags: Flags = Flags.NONE) -> Frame:
        return Frame(MessageType.RESPONSE, 7, seq, payload, flags)

    async def test_buffers_out_of_order_frames(self):
        pending = _PendingRequest(7)
        # Deliver seq 0, then 2 (ahead), then 1 (fills the gap).
        pending.feed(self._frame(0, b"a", Flags.START))
        self.assertEqual(pending.ack_seq, 1)
        pending.feed(self._frame(2, b"c", Flags.END))
        # seq 2 is buffered but not yet contiguous, so the ack stays at 1.
        self.assertEqual(pending.ack_seq, 1)
        pending.feed(self._frame(1, b"b"))
        self.assertEqual(pending.ack_seq, 3)
        pieces = [p async for p in pending]
        self.assertEqual("".join(pieces), "abc")

    async def test_ignores_duplicate_frames(self):
        pending = _PendingRequest(7)
        pending.feed(self._frame(0, b"a", Flags.START))
        pending.feed(self._frame(0, b"a", Flags.START))  # duplicate retransmit
        pending.feed(self._frame(1, b"b", Flags.END))
        self.assertEqual(pending.ack_seq, 2)
        pieces = [p async for p in pending]
        self.assertEqual("".join(pieces), "ab")


class LossyIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, text: str, drop_seqs: Set[int], drop_count: int = 1) -> str:
        host_t, consumer_t = _lossy_loopback(drop_seqs, drop_count)
        host = HostService(host_t, StaticBackend(text, 2), max_payload_size=TINY)
        consumer = ConsumerClient(consumer_t, max_payload_size=TINY, timeout=5.0)
        await host.start()
        await consumer.start()
        try:
            return await consumer.generate("go")
        finally:
            await consumer.close()
            await host.close()

    async def test_recovers_from_single_drop(self):
        text = "the quick brown fox jumps over the lazy dog"
        result = await self._run(text, drop_seqs={1})
        self.assertEqual(result, text)

    async def test_recovers_from_multiple_drops(self):
        text = "reliability over a lossy bluetooth notification channel works"
        result = await self._run(text, drop_seqs={0, 2, 3, 5})
        self.assertEqual(result, text)

    async def test_recovers_from_repeated_drops_of_same_frame(self):
        text = "persistent loss still recovers via retransmission"
        result = await self._run(text, drop_seqs={2}, drop_count=3)
        self.assertEqual(result, text)


if __name__ == "__main__":
    unittest.main()
