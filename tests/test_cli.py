import io
import json
import unittest
from unittest import mock

from keytalk import cli


class CliTests(unittest.TestCase):
    def _run(self, argv, stdin_text: str = ""):
        stdout = io.StringIO()
        stderr = io.StringIO()
        exit_code = cli.main(argv, stdin=io.StringIO(stdin_text), stdout=stdout, stderr=stderr)
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_encode_and_decode_commands_round_trip(self) -> None:
        exit_code, encoded_stdout, encoded_stderr = self._run(
            ["encode", "--text", "hello", "--message-id", "msg123", "--chunk-size", "3"]
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(encoded_stderr, "")

        frames = encoded_stdout.strip().splitlines()
        exit_code, decoded_stdout, decoded_stderr = self._run(["decode", *frames])

        self.assertEqual(exit_code, 0)
        self.assertEqual(decoded_stderr, "")
        self.assertEqual(decoded_stdout.strip(), "hello")

    def test_request_command_reads_prompt_from_stdin(self) -> None:
        exit_code, stdout, stderr = self._run(["request", "--model", "llama3.2"], stdin_text="hello from stdin")

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr, "")

        exit_code, request_json, _ = self._run(["receive-request", *stdout.strip().splitlines()])
        self.assertEqual(exit_code, 0)
        payload = json.loads(request_json)
        self.assertEqual(payload["prompt"], "hello from stdin")
        self.assertEqual(payload["model"], "llama3.2")

    def test_reply_command_uses_ollama_client(self) -> None:
        exit_code, request_frames, _ = self._run(
            ["request", "--model", "llama3.2", "--prompt", "What is keytalk?", "--message-id", "msg123"]
        )
        self.assertEqual(exit_code, 0)

        with mock.patch("keytalk.cli.OllamaClient.generate") as generate:
            generate.return_value = mock.Mock(
                message_id="msg123",
                model="llama3.2",
                response="A keyboard-safe relay.",
                error=None,
                to_message=lambda: mock.Mock(
                    payload=json.dumps(
                        {
                            "response": "A keyboard-safe relay.",
                            "model": "llama3.2",
                            "error": None,
                        },
                        separators=(",", ":"),
                    )
                ),
            )
            exit_code, response_frames, stderr = self._run(["reply", *request_frames.strip().splitlines()])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr, "")

        exit_code, response_json, response_stderr = self._run(["receive-response", *response_frames.strip().splitlines()])
        self.assertEqual(exit_code, 0)
        self.assertEqual(response_stderr, "")
        payload = json.loads(response_json)
        self.assertEqual(payload["response"], "A keyboard-safe relay.")
        self.assertEqual(payload["model"], "llama3.2")

    def test_decode_json_requires_json_payload(self) -> None:
        exit_code, encoded_stdout, _ = self._run(["encode", "--text", "hello", "--message-id", "msg123"])

        exit_code, _, stderr = self._run(["decode", "--json", *encoded_stdout.strip().splitlines()])

        self.assertEqual(exit_code, 1)
        self.assertIn("Expecting value", stderr)
