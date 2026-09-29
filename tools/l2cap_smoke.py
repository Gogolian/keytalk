#!/usr/bin/env python3
"""Two-machine smoke test for the L2CAP CoC transport (real radio required).

Unit tests cover the framing/transport over socket pairs; this exercises the
actual Bluetooth L2CAP path between two machines:

    # machine A (host, runs llama-server):
    ./l2cap_smoke.py listen 0x1001 --url http://127.0.0.1:8080

    # machine B (consumer):
    ./l2cap_smoke.py connect AA:BB:CC:DD:EE:FF 0x1001 --prompt "hello"

On Linux/BlueZ this uses ``SOCK_SEQPACKET`` L2CAP sockets
(``open_l2cap_socket``).  On macOS, CoreBluetooth L2CAP channels are exposed
via ``CBPeripheralManager.publishL2CAPChannel_`` / ``CBPeripheral
openL2CAPChannel_`` and ``CBL2CAPChannel``'s NSStreams (see
``keytalk.ble.l2cap_link.MacStreamPair``); glue a channel's streams into
``StreamLink`` + ``LinkTransport`` the same way this tool does with sockets.

PSMs must be odd, 0x1001-0xFFFF (dynamic range).
"""

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from keytalk.ble.l2cap_link import (  # noqa: E402
    DatagramLink,
    LinkTransport,
    open_l2cap_socket,
)
from keytalk.consumer import ConsumerClient  # noqa: E402
from keytalk.host import HostService  # noqa: E402

PAYLOAD = 64 * 1024 // 10  # a response big enough to measure throughput


async def serve(psm: int, url: str) -> int:
    from keytalk.backends import LlamaCppBackend

    backend = LlamaCppBackend(model="smoke", host=url, n_predict=256)
    listener = open_l2cap_socket(psm, server=True)
    print(f"[l2cap] listening on PSM {psm:#x}, backend {url}")
    while True:
        sock, peer = await asyncio.get_running_loop().run_in_executor(None, listener.accept)
        print(f"[l2cap] consumer connected: {peer}")
        host = HostService(
            LinkTransport(DatagramLink(sock)), backend, max_payload_size=32 * 1024
        )
        await host.start()
        try:
            while True:
                await asyncio.sleep(1)
        finally:
            await host.close()


async def connect(address: str, psm: int, prompt: str) -> int:
    sock = open_l2cap_socket(psm, address)
    print(f"[l2cap] connected to {address} PSM {psm:#x}")
    consumer = ConsumerClient(
        LinkTransport(DatagramLink(sock)),
        max_payload_size=32 * 1024,
        timeout=60.0,
        keepalive_interval=10.0,
    )
    await consumer.start()
    try:
        start = time.monotonic()
        text = await consumer.generate(prompt)
        elapsed = time.monotonic() - start
        print(f"[l2cap] received {len(text)} chars in {elapsed:.2f}s "
              f"({len(text) / max(elapsed, 1e-9):.0f} chars/s)")
        print(f"[l2cap] timings: {consumer.last_timings}")
        print(f"[l2cap] text: {text[:200]!r}")
        return 0
    finally:
        await consumer.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    listen = sub.add_parser("listen")
    listen.add_argument("psm", type=lambda s: int(s, 0))
    listen.add_argument("--url", default="http://127.0.0.1:8080")
    conn = sub.add_parser("connect")
    conn.add_argument("address")
    conn.add_argument("psm", type=lambda s: int(s, 0))
    conn.add_argument("--prompt", default="Say hello.")
    args = parser.parse_args()
    if args.mode == "listen":
        return asyncio.run(serve(args.psm, args.url))
    return asyncio.run(connect(args.address, args.psm, args.prompt))


if __name__ == "__main__":
    sys.exit(main())
