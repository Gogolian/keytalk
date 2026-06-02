"""Command-line entry points for the keytalk host and consumer.

Usage::

    keytalk host --model llama3           # run on the Mac with Ollama
    keytalk consume --address <addr> --prompt "Hi"   # run on the other machine
    keytalk scan                          # list nearby keytalk hosts

These commands require the optional BLE dependencies (``bless`` for the host,
``bleak`` for the consumer).  The core library and tests do not.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import List, Optional

from .backends import OllamaBackend
from .consumer import ConsumerClient
from .host import HostService
from .server import DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_PORT, OllamaBridgeServer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="keytalk", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    host = sub.add_parser("host", help="run the BLE host bridging to Ollama")
    host.add_argument("--model", default="llama3", help="Ollama model name")
    host.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="base URL of the Ollama server",
    )
    host.add_argument("--name", default="keytalk", help="advertised BLE name")

    consume = sub.add_parser(
        "consume",
        help="send one prompt to a BLE host, or serve an Ollama-compatible API",
    )
    consume.add_argument("--address", required=True, help="host BLE address")
    consume.add_argument(
        "--prompt",
        help="prompt text to send (required unless --serve is given)",
    )
    consume.add_argument(
        "--timeout", type=float, default=300.0, help="response timeout (s)"
    )
    consume.add_argument(
        "--serve",
        action="store_true",
        help="run a local Ollama-compatible HTTP endpoint backed by the BLE "
        "host (point e.g. VS Code at it instead of a real Ollama server)",
    )
    consume.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="address to bind the --serve HTTP endpoint to",
    )
    consume.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="port for the --serve HTTP endpoint (Ollama's default is 11434)",
    )
    consume.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="model name advertised by the --serve endpoint",
    )

    scan = sub.add_parser("scan", help="discover nearby keytalk hosts")
    scan.add_argument(
        "--timeout", type=float, default=5.0, help="scan duration (s)"
    )
    return parser


async def _run_host(args: argparse.Namespace) -> int:
    from .ble.peripheral import BlessPeripheralTransport

    backend = OllamaBackend(model=args.model, host=args.ollama_host)
    transport = BlessPeripheralTransport(name=args.name)
    host = HostService(transport, backend)
    await host.start()
    print(f"keytalk host advertising as {args.name!r}; serving model "
          f"{args.model!r}. Press Ctrl-C to stop.", file=sys.stderr)
    try:
        await asyncio.Event().wait()  # run until interrupted
    finally:
        await host.close()
    return 0


async def _run_consume(args: argparse.Namespace) -> int:
    from .ble.central import BleakCentralTransport

    transport = BleakCentralTransport(args.address)
    client = ConsumerClient(transport, timeout=args.timeout)
    await client.start()
    try:
        if args.serve:
            return await _serve_consume(args, client)
        if args.prompt is None:
            print(
                "error: --prompt is required unless --serve is given",
                file=sys.stderr,
            )
            return 2
        async for piece in client.stream(args.prompt):
            sys.stdout.write(piece)
            sys.stdout.flush()
        sys.stdout.write("\n")
    finally:
        await client.close()
    return 0


async def _serve_consume(
    args: argparse.Namespace, client: ConsumerClient
) -> int:
    server = OllamaBridgeServer(
        client, host=args.host, port=args.port, model=args.model
    )
    await server.start()
    print(
        f"keytalk serving Ollama-compatible API on "
        f"http://{server.host}:{server.port} (model {server.model!r}); "
        f"bridging to BLE host {args.address!r}. Press Ctrl-C to stop.",
        file=sys.stderr,
    )
    try:
        await server.serve_forever()
    finally:
        await server.close()
    return 0


async def _run_scan(args: argparse.Namespace) -> int:
    from .ble.central import discover_hosts

    devices = await discover_hosts(timeout=args.timeout)
    if not devices:
        print("no keytalk hosts found", file=sys.stderr)
        return 1
    for device in devices:
        print(f"{device.address}\t{device.name or '<unknown>'}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    runners = {
        "host": _run_host,
        "consume": _run_consume,
        "scan": _run_scan,
    }
    runner = runners[args.command]
    try:
        return asyncio.run(runner(args))
    except KeyboardInterrupt:  # pragma: no cover - interactive
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
