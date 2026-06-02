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

    consume = sub.add_parser("consume", help="send one prompt to a BLE host")
    consume.add_argument("--address", required=True, help="host BLE address")
    consume.add_argument("--prompt", required=True, help="prompt text to send")
    consume.add_argument(
        "--timeout", type=float, default=300.0, help="response timeout (s)"
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
        async for piece in client.stream(args.prompt):
            sys.stdout.write(piece)
            sys.stdout.flush()
        sys.stdout.write("\n")
    finally:
        await client.close()
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
