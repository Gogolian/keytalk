"""Reliability tests: MTU negotiation, NACK fast-failure, RESUME, and CANCEL.

These exercise the recovery paths over simulated link problems:

* HELLO negotiates the frame payload size once the real BLE MTU is known.
* A garbled prompt is answered with an immediate ERROR (NACK) instead of a
  silent stall.
* A response interrupted by a link outage is replayed on RESUME from the
  consumer's last contiguous sequence number - no matter whether the host was
  mid-stream or already done generating.
* A RESUME for a request the host no longer knows makes the consumer retry the
  whole request with a fresh id.
* Abandoning a stream early sends CANCEL so the host stops generating.
"""

import asyncio
import unittest

from keytalk.backends import LLMBackend, StaticBackend
from keytalk.consumer import ConsumerClient, RemoteError
from keytalk.host import HostService
from keytalk.protocol import (
    Frame,
    MessageType,
    max_payload_for_mtu,
)
from keytalk.transport import InMemoryTransport, create_loopback

TINY = 6


class _SimTransport(InMemoryTransport):
    """Loopback transport with controllable failure injection.

    * ``outage``: silently drop every host->consumer frame (dead radio).
    * ``drop_prompts``: swallow this many complete PROMPT messages.
    * ``corrupt_prompt_seqs``: rewrite the seq of prompt frames with these
      sequence numbers, so the host's reassembler rejects the message (as if
      the transmission were garbled).
    """

    def __init__(self, name: str = "") -> None:
        super().__init__(name)
        self.outage = False
        self.drop_prompts = 0
        self.corrupt_prompt_seqs: set = set()

    async def send(self, frame: bytes) -> None:
        try:
            decoded = Frame.decode(frame)
        except Exception:
            decoded = None
        if decoded is None:
            await super().send(frame)
            return
        if self.outage and decoded.msg_type in (
            MessageType.RESPONSE,
            MessageType.ERROR,
            MessageType.PONG,
        ):
            self.sent.append(bytes(frame))  # transmitted but lost
            return
        if decoded.msg_type == MessageType.PROMPT:
            if decoded.seq in self.corrupt_prompt_seqs:
                self.corrupt_prompt_seqs.discard(decoded.seq)
                frame = Frame(
                    msg_type=decoded.msg_type,
                    message_id=decoded.message_id,
                    seq=99,  # out of order: reassembly must fail
                    payload=decoded.payload,
                    flags=decoded.flags,
                ).encode()
            if self.drop_prompts > 0:
                if decoded.is_end:
                    self.drop_prompts -= 1
                self.sent.append(bytes(frame))  # transmitted but lost
                return
        await super().send(frame)


class _CountingBackend(LLMBackend):
    def __init__(self, text: str, piece: int = 2, delay: float = 0.0) -> None:
        self._text = text
        self._piece = piece
        self._delay = delay
        self.prompts = []
        self.count = 0

    async def generate(self, prompt: str):
        self.prompts.append(prompt)
        for i in range(0, len(self._text), self._piece):
            self.count += 1
            if self._delay:
                await asyncio.sleep(self._delay)
            yield self._text[i : i + self._piece]


class NegotiationTestBase(unittest.IsolatedAsyncioTestCase):
    async def _make(self, backend, *, host_kw=None, consumer_kw=None, sim=False):
        if sim:
            host_t, consumer_t = _SimTransport("host"), _SimTransport("consumer")
            host_t.link(consumer_t)
            consumer_t.link(host_t)
        else:
            host_t, consumer_t = create_loopback()
        host_args = {"max_payload_size": TINY, **(host_kw or {})}
        consumer_args = {
            "max_payload_size": TINY,
            "timeout": 5.0,
            "keepalive_interval": 0,
            **(consumer_kw or {}),
        }
        host = HostService(host_t, backend, **host_args)
        consumer = ConsumerClient(consumer_t, **consumer_args)
        await host.start()
        await consumer.start()
        self.addAsyncCleanup(consumer.close)
        self.addAsyncCleanup(host.close)
        return host, consumer, host_t, consumer_t


class HelloNegotiationTests(NegotiationTestBase):
    async def test_hello_shrinks_host_frame_size(self):
        host, consumer, _, _ = await self._make(
            StaticBackend("ok", 2),
            host_kw={"max_payload_size": 100},
            consumer_kw={"max_payload_size": 20},
        )
        # wait for the HELLO (sent at start) to be processed
        await asyncio.sleep(0.05)
        self.assertEqual(host._max_payload, 20)

    async def test_host_cap_wins_when_consumer_is_greedy(self):
        host, consumer, _, _ = await self._make(
            StaticBackend("ok", 2),
            host_kw={"max_payload_size": 20},
            consumer_kw={"max_payload_size": 100},
        )
        await asyncio.sleep(0.05)
        self.assertEqual(host._max_payload, 20)

    async def test_att_mtu_auto_sizes_consumer_frames(self):
        class _MtuTransport(InMemoryTransport):
            @property
            def mtu_size(self) -> int:
                return 185

        host_t, consumer_t = _MtuTransport("host"), _MtuTransport("consumer")
        host_t.link(consumer_t)
        consumer_t.link(host_t)
        host = HostService(host_t, StaticBackend("ok", 2), mtu=23)  # host capped low
        consumer = ConsumerClient(consumer_t)  # auto-detect from the transport
        await host.start()
        await consumer.start()
        self.addAsyncCleanup(consumer.close)
        self.addAsyncCleanup(host.close)
        self.assertEqual(consumer._max_payload, max_payload_for_mtu(185))
        await asyncio.sleep(0.05)
        # the host cap (MTU 23) still bounds what it emits
        self.assertEqual(host._max_payload, max_payload_for_mtu(23))
        self.assertEqual(await consumer.generate("go"), "ok")


class NackFastFailureTests(NegotiationTestBase):
    async def test_garbled_prompt_fails_fast_with_error(self):
        backend = _CountingBackend("ok")
        host, consumer, _, consumer_t = await self._make(
            backend,
            sim=True,
            consumer_kw={"compress_prompts": False, "retries": 0, "timeout": 30.0},
        )
        # Corrupt the middle prompt frame's sequence number.
        consumer_t.corrupt_prompt_seqs = {1}
        with self.assertRaises(RemoteError) as ctx:
            await asyncio.wait_for(
                consumer.generate("abcdefghijklmnop"), timeout=2.0
            )
        # The error comes from the host's reassembler, promptly - well within
        # the 30s request timeout.
        self.assertIn("out-of-order", str(ctx.exception))
        self.assertEqual(backend.prompts, [])  # nothing was generated


class ResumeTests(NegotiationTestBase):
    async def test_recovers_after_link_outage(self):
        text = "the quick brown fox jumps over the lazy dog"
        backend = _CountingBackend(text, 2)
        host, consumer, host_t, _ = await self._make(
            backend,
            sim=True,
            host_kw={"rto": 0.05, "max_retries": 3},
            consumer_kw={
                "timeout": 0.08,
                "max_resumes": 30,
                "retries": 0,
                "compress_prompts": False,
            },
        )
        host_t.outage = True  # notifications vanish: nothing arrives at all
        task = asyncio.ensure_future(consumer.generate("go"))
        await asyncio.sleep(0.3)  # host exhausts its retransmits; consumer resumes
        host_t.outage = False
        result = await asyncio.wait_for(task, timeout=5.0)
        self.assertEqual(result, text)

    async def test_resume_replays_only_the_missing_tail(self):
        text = "resuming must not repeat what we already received"
        backend = _CountingBackend(text, 2)
        host, consumer, host_t, _ = await self._make(
            backend,
            sim=True,
            host_kw={"rto": 0.05, "max_retries": 3},
            consumer_kw={
                "timeout": 0.08,
                "max_resumes": 30,
                "retries": 0,
                "compress_prompts": False,
            },
        )
        # Let the first frames through, then kill the link mid-stream.
        async def _outage_soon() -> None:
            while not any(
                Frame.decode(f).msg_type == MessageType.RESPONSE
                for f in host_t.sent[-8:]
                if len(f) >= 7
            ):
                await asyncio.sleep(0.01)
            host_t.outage = True
            await asyncio.sleep(0.3)
            host_t.outage = False

        killer = asyncio.ensure_future(_outage_soon())
        result = await asyncio.wait_for(consumer.generate("go"), timeout=5.0)
        await killer
        self.assertEqual(result, text)  # no duplicated or missing characters

    async def test_unknown_resume_retries_whole_request(self):
        text = "retry me from scratch"
        backend = _CountingBackend(text, 2)
        host, consumer, _, consumer_t = await self._make(
            backend,
            sim=True,
            consumer_kw={
                "timeout": 0.08,
                "max_resumes": 3,
                "retries": 2,
                "compress_prompts": False,
            },
        )
        consumer_t.drop_prompts = 1  # the first prompt never reaches the host
        result = await asyncio.wait_for(consumer.generate("hello"), timeout=5.0)
        self.assertEqual(result, text)
        # Only the retried prompt was generated, with the prompt intact.
        self.assertEqual(backend.prompts, ["hello"])

    async def test_stalled_response_gives_up_eventually(self):
        host, consumer, host_t, _ = await self._make(
            StaticBackend("never arrives", 2),
            sim=True,
            consumer_kw={
                "timeout": 0.05,
                "max_resumes": 2,
                "retries": 0,
                "compress_prompts": False,
            },
        )
        host_t.outage = True
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(consumer.generate("go"), timeout=5.0)


class CancelTests(NegotiationTestBase):
    async def test_abandoned_stream_cancels_generation(self):
        backend = _CountingBackend("abcdefghij" * 10, piece=1, delay=0.02)
        host, consumer, _, _ = await self._make(
            backend,
            consumer_kw={"timeout": 5.0, "compress_prompts": False},
            host_kw={"compress_responses": False},
        )
        stream = consumer.stream("go")
        first = await stream.__anext__()
        self.assertTrue(first)
        await stream.aclose()  # caller walks away mid-response
        await asyncio.sleep(0.2)  # CANCEL propagates; host aborts generation
        self.assertEqual(host._gen_tasks, {})
        stopped = backend.count
        self.assertLess(stopped, 100)  # generation stopped early
        await asyncio.sleep(0.1)
        self.assertEqual(backend.count, stopped)  # and stays stopped
        self.assertEqual(consumer._pending, {})


if __name__ == "__main__":
    unittest.main()
