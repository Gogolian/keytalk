"""Command line interface for keytalk."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable, TextIO

from .bridge import (
    BridgeError,
    OllamaClient,
    PromptResponse,
    decode_prompt_request,
    decode_prompt_response,
    encode_prompt_request,
    encode_prompt_response,
    handle_prompt_lines,
)
from .protocol import DEFAULT_CHUNK_SIZE, ProtocolError, decode_lines, encode_text


def _read_text_argument(value: str | None, stream: TextIO) -> str:
    if value is not None:
        return value
    return stream.read()


def _read_lines(values: list[str], stream: TextIO) -> list[str]:
    if values:
        return values
    return stream.read().splitlines()


def _write_lines(lines: Iterable[str], stream: TextIO) -> int:
    for line in lines:
        stream.write(f"{line}\n")
    return 0


def _command_encode(args: argparse.Namespace, stdin: TextIO, stdout: TextIO) -> int:
    text = _read_text_argument(args.text, stdin)
    return _write_lines(
        encode_text(
            text,
            kind=args.kind,
            message_id=args.message_id,
            max_chunk_size=args.chunk_size,
        ),
        stdout,
    )


def _command_decode(args: argparse.Namespace, stdin: TextIO, stdout: TextIO) -> int:
    messages = decode_lines(_read_lines(args.frames, stdin))
    for message in messages:
        if args.json:
            parsed = json.loads(message.payload)
            stdout.write(json.dumps(parsed, indent=2, sort_keys=True))
            stdout.write("\n")
        else:
            stdout.write(message.payload)
            stdout.write("\n")
    return 0


def _command_request(args: argparse.Namespace, stdin: TextIO, stdout: TextIO) -> int:
    prompt = _read_text_argument(args.prompt, stdin)
    return _write_lines(
        encode_prompt_request(
            prompt,
            model=args.model,
            system=args.system,
            message_id=args.message_id,
        ),
        stdout,
    )


def _command_receive_request(args: argparse.Namespace, stdin: TextIO, stdout: TextIO) -> int:
    prompt_request = decode_prompt_request(_read_lines(args.frames, stdin))
    stdout.write(json.dumps(prompt_request.__dict__, indent=2, sort_keys=True))
    stdout.write("\n")
    return 0


def _command_receive_response(args: argparse.Namespace, stdin: TextIO, stdout: TextIO) -> int:
    prompt_response = decode_prompt_response(_read_lines(args.frames, stdin))
    stdout.write(json.dumps(prompt_response.__dict__, indent=2, sort_keys=True))
    stdout.write("\n")
    return 0


def _command_reply(args: argparse.Namespace, stdin: TextIO, stdout: TextIO) -> int:
    client = OllamaClient(base_url=args.base_url, timeout=args.timeout)

    def responder(prompt_request):
        return client.generate(prompt_request)

    frames = handle_prompt_lines(_read_lines(args.frames, stdin), responder=responder)
    return _write_lines(frames, stdout)


def _command_mock_reply(args: argparse.Namespace, stdin: TextIO, stdout: TextIO) -> int:
    prompt_request = decode_prompt_request(_read_lines(args.frames, stdin))
    response = PromptResponse(
        message_id=prompt_request.message_id,
        response=args.response,
        model=prompt_request.model,
    )
    return _write_lines(encode_prompt_response(response), stdout)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="keytalk",
        description="Keyboard-safe prompt and response transport for LLM relays.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode_parser = subparsers.add_parser("encode", help="encode plain text into keytalk frames")
    encode_parser.add_argument("--text")
    encode_parser.add_argument("--kind", default="text")
    encode_parser.add_argument("--message-id")
    encode_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    encode_parser.set_defaults(handler=_command_encode)

    decode_parser = subparsers.add_parser("decode", help="decode plain text frames")
    decode_parser.add_argument("frames", nargs="*")
    decode_parser.add_argument("--json", action="store_true")
    decode_parser.set_defaults(handler=_command_decode)

    request_parser = subparsers.add_parser("request", help="encode a prompt request")
    request_parser.add_argument("--prompt")
    request_parser.add_argument("--model", required=True)
    request_parser.add_argument("--system")
    request_parser.add_argument("--message-id")
    request_parser.set_defaults(handler=_command_request)

    receive_request_parser = subparsers.add_parser("receive-request", help="decode a prompt request as JSON")
    receive_request_parser.add_argument("frames", nargs="*")
    receive_request_parser.set_defaults(handler=_command_receive_request)

    receive_response_parser = subparsers.add_parser("receive-response", help="decode a prompt response as JSON")
    receive_response_parser.add_argument("frames", nargs="*")
    receive_response_parser.set_defaults(handler=_command_receive_response)

    reply_parser = subparsers.add_parser("reply", help="decode a prompt, call Ollama, and emit response frames")
    reply_parser.add_argument("frames", nargs="*")
    reply_parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    reply_parser.add_argument("--timeout", type=float, default=30.0)
    reply_parser.set_defaults(handler=_command_reply)

    mock_reply_parser = subparsers.add_parser("mock-reply", help="emit a fixed response for a prompt request")
    mock_reply_parser.add_argument("frames", nargs="*")
    mock_reply_parser.add_argument("--response", required=True)
    mock_reply_parser.set_defaults(handler=_command_mock_reply)

    return parser


def main(
    argv: list[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    try:
        return args.handler(args, stdin, stdout)
    except (BridgeError, ProtocolError, json.JSONDecodeError) as exc:
        stderr.write(f"error: {exc}\n")
        return 1


def run_cli() -> None:
    raise SystemExit(main())
