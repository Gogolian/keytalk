import unittest

from keytalk.protocol import ProtocolError, decode_lines, encode_text


class ProtocolTests(unittest.TestCase):
    def test_round_trip_text_message(self) -> None:
        frames = encode_text("hello over keyboard", kind="text", message_id="abc123", max_chunk_size=8)

        messages = decode_lines(frames)

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].kind, "text")
        self.assertEqual(messages[0].message_id, "abc123")
        self.assertEqual(messages[0].payload, "hello over keyboard")

    def test_multiple_messages_are_decoded_in_input_order(self) -> None:
        first = encode_text("one", kind="text", message_id="one")
        second = encode_text("two", kind="text", message_id="two")

        messages = decode_lines(first + second)

        self.assertEqual([message.message_id for message in messages], ["one", "two"])
        self.assertEqual([message.payload for message in messages], ["one", "two"])

    def test_corrupt_checksum_is_rejected(self) -> None:
        [frame] = encode_text("hello", message_id="abc123")
        corrupt = frame[:-1] + ("0" if frame[-1] != "0" else "1")

        with self.assertRaises(ProtocolError):
            decode_lines([corrupt])

    def test_missing_frame_is_rejected(self) -> None:
        frames = encode_text("hello world", message_id="abc123", max_chunk_size=4)

        with self.assertRaises(ProtocolError):
            decode_lines(frames[:-1])
