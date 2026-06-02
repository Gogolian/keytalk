"""Tests for the framing / chunking / reassembly protocol."""

import unittest

from keytalk.protocol import (
    DEFAULT_ATT_MTU,
    Flags,
    Frame,
    FrameStreamEncoder,
    MessageType,
    ProtocolError,
    Reassembler,
    chunk_message,
    max_payload_for_mtu,
)


class FrameCodecTests(unittest.TestCase):
    def test_roundtrip(self):
        frame = Frame(
            msg_type=MessageType.PROMPT,
            message_id=4242,
            seq=7,
            payload=b"hello world",
            flags=Flags.START | Flags.END,
        )
        decoded = Frame.decode(frame.encode())
        self.assertEqual(decoded, frame)
        self.assertTrue(decoded.is_start)
        self.assertTrue(decoded.is_end)

    def test_empty_payload_roundtrip(self):
        frame = Frame(MessageType.RESPONSE, 1, 0, b"", Flags.END)
        self.assertEqual(Frame.decode(frame.encode()), frame)

    def test_decode_rejects_short_frame(self):
        with self.assertRaises(ProtocolError):
            Frame.decode(b"\x01\x02")

    def test_decode_rejects_bad_version(self):
        good = Frame(MessageType.PROMPT, 1, 0, b"x").encode()
        bad = bytes([99]) + good[1:]
        with self.assertRaises(ProtocolError):
            Frame.decode(bad)

    def test_decode_rejects_unknown_type(self):
        good = bytearray(Frame(MessageType.PROMPT, 1, 0, b"x").encode())
        good[1] = 200  # unknown message type
        with self.assertRaises(ProtocolError):
            Frame.decode(bytes(good))

    def test_decode_rejects_unknown_flag_bits(self):
        good = bytearray(Frame(MessageType.PROMPT, 1, 0, b"x").encode())
        good[2] = 0xFF  # bits beyond START|END
        with self.assertRaises(ProtocolError):
            Frame.decode(bytes(good))

    def test_frame_validates_id_range(self):
        with self.assertRaises(ValueError):
            Frame(MessageType.PROMPT, 0x1_0000, 0, b"")
        with self.assertRaises(ValueError):
            Frame(MessageType.PROMPT, 0, 0x1_0000, b"")


class MtuTests(unittest.TestCase):
    def test_default_mtu_leaves_room(self):
        self.assertEqual(max_payload_for_mtu(DEFAULT_ATT_MTU), 23 - 3 - 7)

    def test_tiny_mtu_rejected(self):
        with self.assertRaises(ValueError):
            max_payload_for_mtu(9)


class ChunkMessageTests(unittest.TestCase):
    def _reassemble(self, frames):
        return b"".join(f.payload for f in frames)

    def test_single_chunk(self):
        frames = chunk_message(MessageType.PROMPT, 1, b"short", 100)
        self.assertEqual(len(frames), 1)
        self.assertTrue(frames[0].is_start and frames[0].is_end)
        self.assertEqual(self._reassemble(frames), b"short")

    def test_empty_payload_is_one_frame(self):
        frames = chunk_message(MessageType.PROMPT, 1, b"", 100)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].flags, Flags.START | Flags.END)
        self.assertEqual(frames[0].payload, b"")

    def test_exact_multiple(self):
        payload = b"abcd" * 4  # 16 bytes, max 4 -> 4 frames exactly
        frames = chunk_message(MessageType.PROMPT, 1, payload, 4)
        self.assertEqual(len(frames), 4)
        self.assertEqual([f.seq for f in frames], [0, 1, 2, 3])
        self.assertTrue(frames[0].is_start)
        self.assertFalse(frames[0].is_end)
        self.assertTrue(frames[-1].is_end)
        self.assertFalse(frames[-1].is_start)
        self.assertEqual(self._reassemble(frames), payload)

    def test_large_payload(self):
        payload = bytes(range(256)) * 40  # 10240 bytes
        frames = chunk_message(MessageType.RESPONSE, 9, payload, 20)
        self.assertEqual(len(frames), (len(payload) + 19) // 20)
        self.assertEqual(self._reassemble(frames), payload)

    def test_rejects_nonpositive_max(self):
        with self.assertRaises(ValueError):
            chunk_message(MessageType.PROMPT, 1, b"x", 0)


class ReassemblerTests(unittest.TestCase):
    def test_basic_reassembly(self):
        r = Reassembler()
        frames = chunk_message(MessageType.PROMPT, 5, b"hello there friend", 4)
        result = None
        for f in frames:
            result = r.feed(f)
        self.assertIsNotNone(result)
        self.assertEqual(result.payload, b"hello there friend")
        self.assertEqual(result.msg_type, MessageType.PROMPT)
        self.assertEqual(result.message_id, 5)
        self.assertEqual(r.pending, 0)

    def test_partial_returns_none(self):
        r = Reassembler()
        frames = chunk_message(MessageType.PROMPT, 1, b"abcdefgh", 4)
        self.assertIsNone(r.feed(frames[0]))
        self.assertEqual(r.pending, 1)

    def test_interleaved_messages(self):
        r = Reassembler()
        a = chunk_message(MessageType.PROMPT, 1, b"aaaa-bbbb", 4)
        b = chunk_message(MessageType.PROMPT, 2, b"cccc-dddd", 4)
        out = []
        # interleave frames from both messages
        for fa, fb in zip(a, b):
            ra = r.feed(fa)
            rb = r.feed(fb)
            out.extend(x for x in (ra, rb) if x is not None)
        self.assertEqual({m.message_id: m.payload for m in out},
                         {1: b"aaaa-bbbb", 2: b"cccc-dddd"})

    def test_out_of_order_raises(self):
        r = Reassembler()
        frames = chunk_message(MessageType.PROMPT, 1, b"abcdefgh", 2)
        self.assertGreaterEqual(len(frames), 3)
        r.feed(frames[0])
        with self.assertRaises(ProtocolError):
            r.feed(frames[2])  # skipped seq 1

    def test_missing_start_raises(self):
        r = Reassembler()
        orphan = Frame(MessageType.PROMPT, 1, 1, b"x", Flags.NONE)
        with self.assertRaises(ProtocolError):
            r.feed(orphan)

    def test_restart_discards_partial(self):
        r = Reassembler()
        first = chunk_message(MessageType.PROMPT, 1, b"abcdefgh", 4)
        r.feed(first[0])  # partial
        # A fresh START for the same id resets the buffer.
        again = chunk_message(MessageType.PROMPT, 1, b"zz", 4)
        result = r.feed(again[0])
        self.assertIsNotNone(result)
        self.assertEqual(result.payload, b"zz")

    def test_type_change_midstream_raises(self):
        r = Reassembler()
        start = Frame(MessageType.PROMPT, 1, 0, b"a", Flags.START)
        nxt = Frame(MessageType.RESPONSE, 1, 1, b"b", Flags.END)
        r.feed(start)
        with self.assertRaises(ProtocolError):
            r.feed(nxt)

    def test_discard(self):
        r = Reassembler()
        frames = chunk_message(MessageType.PROMPT, 1, b"abcdefgh", 4)
        r.feed(frames[0])
        r.discard(1)
        self.assertEqual(r.pending, 0)


class FrameStreamEncoderTests(unittest.TestCase):
    def _drive(self, pieces, max_size):
        enc = FrameStreamEncoder(MessageType.RESPONSE, 1, max_size)
        frames = []
        for p in pieces:
            frames.extend(enc.push(p))
        frames.extend(enc.finish())
        return frames

    def test_flags_and_reassembly(self):
        frames = self._drive([b"hello ", b"streamed ", b"world"], 4)
        self.assertTrue(frames[0].is_start)
        self.assertTrue(frames[-1].is_end)
        self.assertFalse(frames[0].is_end)
        # exactly one START and one END across the stream
        self.assertEqual(sum(f.is_start for f in frames), 1)
        self.assertEqual(sum(f.is_end for f in frames), 1)
        self.assertEqual([f.seq for f in frames], list(range(len(frames))))
        self.assertEqual(b"".join(f.payload for f in frames),
                         b"hello streamed world")
        for f in frames[:-1]:
            self.assertLessEqual(len(f.payload), 4)

    def test_empty_stream_single_frame(self):
        frames = self._drive([], 4)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].flags, Flags.START | Flags.END)
        self.assertEqual(frames[0].payload, b"")

    def test_exact_multiple_end_on_last(self):
        # 8 bytes with max 4: must still end cleanly with END on final frame.
        frames = self._drive([b"abcdefgh"], 4)
        self.assertEqual(b"".join(f.payload for f in frames), b"abcdefgh")
        self.assertTrue(frames[-1].is_end)
        self.assertEqual(sum(f.is_end for f in frames), 1)

    def test_stream_reassembles_via_reassembler(self):
        frames = self._drive([b"x" * 10, b"y" * 7], 4)
        r = Reassembler()
        result = None
        for f in frames:
            result = r.feed(f)
        self.assertEqual(result.payload, b"x" * 10 + b"y" * 7)

    def test_push_after_finish_raises(self):
        enc = FrameStreamEncoder(MessageType.RESPONSE, 1, 4)
        enc.finish()
        with self.assertRaises(RuntimeError):
            enc.push(b"x")

    def test_double_finish_raises(self):
        enc = FrameStreamEncoder(MessageType.RESPONSE, 1, 4)
        enc.finish()
        with self.assertRaises(RuntimeError):
            enc.finish()


if __name__ == "__main__":
    unittest.main()
