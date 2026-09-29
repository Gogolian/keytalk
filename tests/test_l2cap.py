"""Tests for the L2CAP CoC transport layer.

The framing and transport run over real OS socket pairs (no radio required):
length-prefixed stream links, datagram links, and a full keytalk
host<->consumer pipeline over ``LinkTransport``.  The PyObjC selector
assertions pin the macOS glue to the actual CoreBluetooth API surface so it
cannot silently rot.  The end-to-end RF path needs two machines -
``tools/l2cap_smoke.py``.
"""

import socket
import threading
import unittest

from keytalk.backends import StaticBackend
from keytalk.ble.l2cap_link import (
    CBL2CAP_AVAILABLE,
    MAC_OPEN_SELECTOR,
    MAC_PUBLISH_SELECTOR,
    DatagramLink,
    FrameReader,
    LinkClosed,
    LinkTransport,
    open_l2cap_socket,
    pack_frame,
    socket_read_exact,
    stream_link_from_socket,
)
from keytalk.consumer import ConsumerClient
from keytalk.host import HostService
from keytalk.transport import TransportClosed


class FramingUnitTests(unittest.TestCase):
    def test_pack_and_read_roundtrip(self):
        reader = FrameReader()
        self.assertEqual(reader.feed(pack_frame(b"abc")), [b"abc"])

    def test_reader_handles_partial_and_coalesced_feeds(self):
        data = pack_frame(b"hello") + pack_frame(b"") + pack_frame(b"world!")
        reader = FrameReader()
        got = []
        # feed one byte at a time - every split must be tolerated
        for i in range(len(data)):
            got.extend(reader.feed(data[i : i + 1]))
        self.assertEqual(got, [b"hello", b"", b"world!"])

    def test_reader_holds_incomplete_frame(self):
        reader = FrameReader()
        self.assertEqual(reader.feed(pack_frame(b"xyz")[:-1]), [])
        self.assertEqual(reader.feed(b"z"), [b"xyz"])

    def test_pack_rejects_oversize(self):
        with self.assertRaises(ValueError):
            pack_frame(b"x" * 0x10000)


class StreamLinkTests(unittest.TestCase):
    def test_roundtrip_both_directions(self):
        a, b = socket.socketpair()
        la, lb = stream_link_from_socket(a), stream_link_from_socket(b)
        try:
            la.send_frame(b"ping")
            self.assertEqual(lb.recv_frame(), b"ping")
            lb.send_frame(b"p" * 1000)  # larger than one socket buffer chunk
            self.assertEqual(la.recv_frame(), b"p" * 1000)
        finally:
            la.close()
            lb.close()

    def test_eof_raises_link_closed(self):
        a, b = socket.socketpair()
        la, lb = stream_link_from_socket(a), stream_link_from_socket(b)
        lb.close()
        with self.assertRaises(LinkClosed):
            la.recv_frame()

    def test_write_failure_raises_link_closed(self):
        a, b = socket.socketpair()
        la, lb = stream_link_from_socket(a), stream_link_from_socket(b)
        lb.close()
        with self.assertRaises(LinkClosed):
            for _ in range(100):  # writes may be buffered; keep writing
                la.send_frame(b"x" * 4096)
        la.close()

    def test_socket_read_exact_handles_short_reads(self):
        a, b = socket.socketpair()
        try:
            read_exact = socket_read_exact(a)

            def writer():
                for chunk in (b"ab", b"cd", b"ef"):
                    b.sendall(chunk)

            t = threading.Thread(target=writer)
            t.start()
            self.assertEqual(read_exact(6), b"abcdef")
            t.join()
        finally:
            a.close()
            b.close()


class DatagramLinkTests(unittest.TestCase):
    def test_roundtrip(self):
        a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        la, lb = DatagramLink(a), DatagramLink(b)
        try:
            la.send_frame(b"one")
            lb.send_frame(b"two")
            self.assertEqual(lb.recv_frame(), b"one")
            self.assertEqual(la.recv_frame(), b"two")
        finally:
            la.close()
            lb.close()


class LinkTransportPairTests(unittest.IsolatedAsyncioTestCase):
    """Full keytalk pipeline over stream links (what L2CAP CoC looks like)."""

    async def test_host_consumer_over_link_transports(self):
        sock_a, sock_b = socket.socketpair()
        host = HostService(
            LinkTransport(stream_link_from_socket(sock_a)),
            StaticBackend("0123456789" * 20, 3),
            max_payload_size=64,
        )
        consumer = ConsumerClient(
            LinkTransport(stream_link_from_socket(sock_b)),
            max_payload_size=64,
            timeout=5.0,
            keepalive_interval=0,
        )
        await host.start()
        await consumer.start()
        try:
            result = await consumer.generate("go")
            self.assertEqual(result, "0123456789" * 20)
        finally:
            await consumer.close()
            await host.close()

    async def test_send_after_close_raises(self):
        a, b = socket.socketpair()
        transport = LinkTransport(stream_link_from_socket(a))
        await transport.start()
        await transport.close()
        with self.assertRaises(TransportClosed):
            await transport.send(b"x")
        stream_link_from_socket(b).close()


class PlatformGlueTests(unittest.TestCase):
    def test_open_l2cap_socket_fails_cleanly_off_linux(self):
        import sys

        if sys.platform.startswith("linux"):
            self.skipTest("Linux exposes real L2CAP sockets here")
        with self.assertRaises(RuntimeError):
            open_l2cap_socket(0x1001)

    @unittest.skipUnless(CBL2CAP_AVAILABLE, "PyObjC CoreBluetooth L2CAP not available")
    def test_macos_selectors_exist(self):
        # Pin the glue to the real API surface (verified bindings, not docs).
        # The skipUnless guard already proved the framework is loadable; import
        # it explicitly so a broken PyObjC install fails loudly here.
        import CoreBluetooth

        import objc

        self.assertTrue(CoreBluetooth is not None)
        manager = objc.lookUpClass("CBPeripheralManager")
        peripheral = objc.lookUpClass("CBPeripheral")
        channel = objc.lookUpClass("CBL2CAPChannel")
        self.assertTrue(hasattr(manager, MAC_PUBLISH_SELECTOR))
        self.assertTrue(hasattr(manager, "unpublishL2CAPChannel_"))
        self.assertTrue(hasattr(peripheral, MAC_OPEN_SELECTOR))
        for prop in ("PSM", "inputStream", "outputStream"):
            self.assertTrue(hasattr(channel, prop))


if __name__ == "__main__":
    unittest.main()
