"""Request/response helpers for Ollama-backed keytalk bridges."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable
from urllib import error, request

from .protocol import KeytalkMessage, ProtocolError, decode_lines, encode_text


class BridgeError(RuntimeError):
    """Raised when a bridge request cannot be processed."""


@dataclass(frozen=True)
class PromptRequest:
    """A prompt sent over the keytalk transport."""

    message_id: str
    prompt: str
    model: str
    system: str | None = None

    def to_message(self) -> KeytalkMessage:
        payload = {
            "prompt": self.prompt,
            "model": self.model,
            "system": self.system,
        }
        return KeytalkMessage(
            kind="prompt",
            message_id=self.message_id,
            payload=json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
        )

    @classmethod
    def from_message(cls, message: KeytalkMessage) -> "PromptRequest":
        if message.kind != "prompt":
            raise BridgeError(f"expected a prompt message, got {message.kind!r}")
        try:
            data = json.loads(message.payload)
        except json.JSONDecodeError as exc:
            raise BridgeError("prompt payload is not valid JSON") from exc
        prompt = data.get("prompt")
        model = data.get("model")
        system = data.get("system")
        if not isinstance(prompt, str) or not prompt:
            raise BridgeError("prompt payload must include a non-empty prompt")
        if not isinstance(model, str) or not model:
            raise BridgeError("prompt payload must include a non-empty model")
        if system is not None and not isinstance(system, str):
            raise BridgeError("system prompt must be a string when provided")
        return cls(
            message_id=message.message_id,
            prompt=prompt,
            model=model,
            system=system,
        )


@dataclass(frozen=True)
class PromptResponse:
    """A response returned over the keytalk transport."""

    message_id: str
    response: str
    model: str
    error: str | None = None

    def to_message(self) -> KeytalkMessage:
        payload = {
            "response": self.response,
            "model": self.model,
            "error": self.error,
        }
        return KeytalkMessage(
            kind="response",
            message_id=self.message_id,
            payload=json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
        )


class OllamaClient:
    """Tiny Ollama client using the standard library."""

    def __init__(self, *, base_url: str = "http://127.0.0.1:11434", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt_request: PromptRequest) -> PromptResponse:
        payload = {
            "model": prompt_request.model,
            "prompt": prompt_request.prompt,
            "stream": False,
        }
        if prompt_request.system:
            payload["system"] = prompt_request.system

        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            url=f"{self.base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise BridgeError(f"Ollama request failed with HTTP {exc.code}: {details}") from exc
        except error.URLError as exc:
            raise BridgeError(f"unable to reach Ollama at {self.base_url}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BridgeError("Ollama returned invalid JSON") from exc

        response_text = data.get("response")
        if not isinstance(response_text, str):
            raise BridgeError("Ollama response is missing a string response field")

        model = data.get("model")
        if not isinstance(model, str) or not model:
            model = prompt_request.model

        return PromptResponse(
            message_id=prompt_request.message_id,
            response=response_text,
            model=model,
        )


def encode_prompt_request(
    prompt: str,
    *,
    model: str,
    system: str | None = None,
    message_id: str | None = None,
) -> list[str]:
    prompt_request = PromptRequest(
        message_id=message_id or "",
        prompt=prompt,
        model=model,
        system=system,
    )
    message = prompt_request.to_message()
    return encode_text(
        message.payload,
        kind=message.kind,
        message_id=message_id,
    )


def decode_prompt_request(lines: list[str]) -> PromptRequest:
    messages = decode_lines(lines)
    if len(messages) != 1:
        raise BridgeError(f"expected exactly one prompt message, got {len(messages)}")
    return PromptRequest.from_message(messages[0])


def encode_prompt_response(prompt_response: PromptResponse) -> list[str]:
    message = prompt_response.to_message()
    return encode_text(
        message.payload,
        kind=message.kind,
        message_id=message.message_id,
    )


def decode_prompt_response(lines: list[str]) -> PromptResponse:
    messages = decode_lines(lines)
    if len(messages) != 1:
        raise BridgeError(f"expected exactly one response message, got {len(messages)}")
    message = messages[0]
    if message.kind != "response":
        raise BridgeError(f"expected a response message, got {message.kind!r}")
    try:
        data = json.loads(message.payload)
    except json.JSONDecodeError as exc:
        raise BridgeError("response payload is not valid JSON") from exc
    response_text = data.get("response")
    model = data.get("model")
    response_error = data.get("error")
    if not isinstance(response_text, str):
        raise BridgeError("response payload must include a string response")
    if not isinstance(model, str) or not model:
        raise BridgeError("response payload must include a non-empty model")
    if response_error is not None and not isinstance(response_error, str):
        raise BridgeError("response error must be a string when provided")
    return PromptResponse(
        message_id=message.message_id,
        response=response_text,
        model=model,
        error=response_error,
    )


def handle_prompt_lines(
    lines: list[str],
    *,
    responder: Callable[[PromptRequest], PromptResponse],
) -> list[str]:
    """Decode a prompt, call a responder, and encode its reply."""

    prompt_request = decode_prompt_request(lines)
    prompt_response = responder(prompt_request)
    if prompt_response.message_id != prompt_request.message_id:
        raise BridgeError("response message_id must match the prompt message_id")
    return encode_prompt_response(prompt_response)
