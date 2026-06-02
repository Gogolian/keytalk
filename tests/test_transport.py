"""Tests for the transport abstraction and in-memory loopback."""

import asyncio
import unittest

from keytalk.transport import (
    InMemoryTransport,
    TransportClosed,
    create_loopback,
)


class InMemoryTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_delivers_to_peer(self):
        host, consumer = create_loopback()
        received = []

        async def on_recv(data):
            received.append(data)

        consumer.on_receive(on_recv)
        await host.start()
        await consumer.start()

        await host.send(b"frame-1")
        await host.send(b"frame-2")
        await asyncio.sleep(0)  # let scheduled deliveries run
        await asyncio.sleep(0)
        self.assertEqual(received, [b"frame-1", b"frame-2"])
        self.assertEqual(host.sent, [b"frame-1", b"frame-2"])

    async def test_bidirectional(self):
        host, consumer = create_loopback()
        to_host, to_consumer = [], []
        host.on_receive(lambda d: _collect(to_host, d))
        consumer.on_receive(lambda d: _collect(to_consumer, d))
        await asyncio.gather(host.start(), consumer.start())

        await consumer.send(b"ping")
        await host.send(b"pong")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertEqual(to_host, [b"ping"])
        self.assertEqual(to_consumer, [b"pong"])

    async def test_send_after_close_raises(self):
        host, consumer = create_loopback()
        await host.start()
        await host.close()
        with self.assertRaises(TransportClosed):
            await host.send(b"x")

    async def test_delivery_is_a_copy(self):
        host, consumer = create_loopback()
        seen = []
        consumer.on_receive(lambda d: _collect(seen, d))
        await asyncio.gather(host.start(), consumer.start())

        buf = bytearray(b"mutable")
        await host.send(buf)
        buf[0] = ord("X")  # mutate after send
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertEqual(seen, [b"mutable"])

    async def test_unlinked_transport_raises(self):
        lonely = InMemoryTransport("lonely")
        await lonely.start()
        with self.assertRaises(RuntimeError):
            await lonely.send(b"x")

    async def test_context_manager(self):
        host, consumer = create_loopback()
        async with host, consumer:
            seen = []
            consumer.on_receive(lambda d: _collect(seen, d))
            await host.send(b"hi")
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        self.assertEqual(seen, [b"hi"])


async def _collect(target, data):
    target.append(data)


if __name__ == "__main__":
    unittest.main()
