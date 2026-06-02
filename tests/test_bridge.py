import io
import json
import unittest
from unittest import mock

from keytalk.bridge import (
    BridgeError,
    OllamaClient,
    PromptRequest,
    PromptResponse,
    decode_prompt_request,
    decode_prompt_response,
    encode_prompt_request,
    handle_prompt_lines,
)


class BridgeTests(unittest.TestCase):
    def test_prompt_request_round_trip(self) -> None:
        frames = encode_prompt_request(
            "Summarize this text",
            model="llama3.2",
            system="be concise",
            message_id="msg123",
        )

        prompt_request = decode_prompt_request(frames)

        self.assertEqual(prompt_request.message_id, "msg123")
        self.assertEqual(prompt_request.prompt, "Summarize this text")
        self.assertEqual(prompt_request.model, "llama3.2")
        self.assertEqual(prompt_request.system, "be concise")

    def test_handle_prompt_lines_calls_responder_and_encodes_reply(self) -> None:
        frames = encode_prompt_request("ping", model="llama3.2", message_id="msg123")

        def responder(prompt_request: PromptRequest) -> PromptResponse:
            self.assertEqual(prompt_request.prompt, "ping")
            return PromptResponse(message_id=prompt_request.message_id, response="pong", model=prompt_request.model)

        response_frames = handle_prompt_lines(frames, responder=responder)
        prompt_response = decode_prompt_response(response_frames)

        self.assertEqual(prompt_response.message_id, "msg123")
        self.assertEqual(prompt_response.response, "pong")
        self.assertEqual(prompt_response.model, "llama3.2")

    def test_handle_prompt_lines_rejects_mismatched_message_ids(self) -> None:
        frames = encode_prompt_request("ping", model="llama3.2", message_id="msg123")

        with self.assertRaises(BridgeError):
            handle_prompt_lines(
                frames,
                responder=lambda _: PromptResponse(message_id="wrong", response="pong", model="llama3.2"),
            )

    @mock.patch("keytalk.bridge.request.urlopen")
    def test_ollama_client_posts_expected_payload(self, mock_urlopen: mock.Mock) -> None:
        fake_response = io.BytesIO(json.dumps({"response": "done", "model": "llama3.2"}).encode("utf-8"))
        mock_urlopen.return_value.__enter__.return_value = fake_response
        client = OllamaClient(base_url="http://ollama.example", timeout=5.0)

        response = client.generate(
            PromptRequest(message_id="msg123", prompt="ping", model="llama3.2", system="reply short")
        )

        self.assertEqual(response.response, "done")
        self.assertEqual(response.model, "llama3.2")
        sent_request = mock_urlopen.call_args.args[0]
        self.assertEqual(sent_request.full_url, "http://ollama.example/api/generate")
        self.assertEqual(sent_request.get_method(), "POST")
        body = json.loads(sent_request.data.decode("utf-8"))
        self.assertEqual(
            body,
            {
                "model": "llama3.2",
                "prompt": "ping",
                "stream": False,
                "system": "reply short",
            },
        )
