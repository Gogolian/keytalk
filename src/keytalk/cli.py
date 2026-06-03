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
import errno
import sys
from typing import List, Optional

from .backends import OllamaBackend, LMStudioBackend
from .consumer import ConsumerClient
from .host import HostService
from .server import DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_PORT, OllamaBridgeServer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="keytalk", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    host = sub.add_parser("host", help="run the BLE host bridging to Ollama or LM Studio")
    host.add_argument("--model", default="llama3", help="model name")
    host.add_argument(
        "--backend",
        choices=["ollama", "lmstudio"],
        default="ollama",
        help="LLM backend to use (default: ollama)",
    )
    host.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="base URL of the Ollama server (for --backend=ollama)",
    )
    host.add_argument(
        "--lmstudio-host",
        default="http://localhost:1234",
        help="base URL of the LM Studio server (for --backend=lmstudio)",
    )
    host.add_argument(
        "--num-ctx",
        type=int,
        default=32768,
        help="context window (tokens) to load the Ollama model with; raise "
        "this if large agent prompts overflow the model's default context",
    )
    host.add_argument("--name", default="keytalk", help="advertised BLE name")
    host.add_argument(
        "--verbose",
        action="store_true",
        help="enable verbose logging (shows frame-by-frame details)",
    )

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
    consume.add_argument(
        "--no-compress",
        action="store_true",
        help="disable zlib compression of prompts (compression is enabled by "
        "default and typically reduces transmission time by 60-80%% for text)",
    )

    scan = sub.add_parser("scan", help="discover nearby keytalk hosts")
    scan.add_argument(
        "--timeout", type=float, default=5.0, help="scan duration (s)"
    )
    scan.add_argument(
        "--simple",
        action="store_true",
        help="output just addresses, one per line",
    )
    return parser


async def _run_host(args: argparse.Namespace) -> int:
    import logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    from .ble.peripheral import BlessPeripheralTransport

    # Select backend
    if args.backend == "lmstudio":
        backend = LMStudioBackend(model=args.model, host=args.lmstudio_host)
        backend_info = f"LM Studio: {args.lmstudio_host}"
    else:  # ollama
        backend = OllamaBackend(
            model=args.model, host=args.ollama_host, num_ctx=args.num_ctx
        )
        backend_info = f"Ollama: {args.ollama_host}"
    
    transport = BlessPeripheralTransport(name=args.name)
    host = HostService(transport, backend)
    await host.start()
    print(f"\nkeytalk host ready!\n"
          f"  Advertising as: {args.name!r}\n"
          f"  Backend: {args.backend}\n"
          f"  Model: {args.model!r}\n"
          f"  Server: {backend_info}\n"
          f"\nWaiting for consumer connections... (Press Ctrl-C to stop)\n", file=sys.stderr)
    try:
        await asyncio.Event().wait()  # run until interrupted
    finally:
        await host.close()
    return 0


async def _run_consume(args: argparse.Namespace) -> int:
    import logging
    import sys
    
    # Enable logging to stderr so it doesn't interfere with stdout output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
        stream=sys.stderr
    )
    
    from .ble.central import BleakCentralTransport

    transport = BleakCentralTransport(args.address)
    client = ConsumerClient(
        transport, 
        timeout=args.timeout,
        compress_prompts=not args.no_compress,
    )
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
    try:
        await server.start()
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(
                f"error: {args.host}:{args.port} is already in use.\n"
                f"  Another process (often a running Ollama or a previous "
                f"`keytalk consume --serve`) is listening there.\n"
                f"  Find it with:  lsof -nP -iTCP:{args.port} -sTCP:LISTEN\n"
                f"  Then stop it, or pass a different port with --port.",
                file=sys.stderr,
            )
            return 1
        raise
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
    
    if args.simple:
        # Simple output for scripting
        for device in devices:
            print(device.address)
        return 0
    
    # Detailed output
    print(f"Found {len(devices)} keytalk host(s):\n")
    for i, device in enumerate(devices, 1):
        # Get the local name from advertisement data if available
        adv_name = None
        if hasattr(device, 'metadata') and device.metadata:
            adv_name = device.metadata.get('name')
        if not adv_name and hasattr(device, 'details'):
            # Try to get advertised local name from details
            details = device.details
            if hasattr(details, 'name'):
                adv_name = details.name
        
        display_name = adv_name or device.name or '<unknown>'
        rssi = getattr(device, 'rssi', None)
        rssi_str = f" (RSSI: {rssi} dBm)" if rssi is not None else ""
        
        print(f"  [{i}] Address: {device.address}")
        print(f"      BLE Name: {display_name}{rssi_str}")
        print(f"      (Use this address with: keytalk consume --address {device.address})")
        print()
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
