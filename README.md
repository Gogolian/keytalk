# keytalk

Use a large language model running on **one** machine from **another** machine —
over **Bluetooth LE** instead of an IP network.

- The **HOST** (e.g. a Mac) runs an LLM such as an [Ollama](https://ollama.com)
  server and exposes it through a **custom BLE GATT service**.
- The **CONSUMER** (any nearby machine) connects over BLE, writes prompt chunks,
  and reads streamed response chunks.

No Wi-Fi, no LAN, no sockets — just a Bluetooth GATT link.

```
 HOST (Mac)                                CONSUMER
 Ollama <- HostService <- bless GATT  <=BLE=>  bleak central -> ConsumerClient
           (LLM bridge)    server                              -> your code

 PROMPT char   (write)  <-- prompt chunks --  CONSUMER
 RESPONSE char (notify) --- response chunks --> CONSUMER
```

## Why chunking?

A BLE GATT characteristic can only carry a small payload (the default ATT MTU is
23 bytes — 20 usable). Prompts and completions are much larger, so every logical
message is split into small **frames** that are written/notified one at a time
and reassembled on the far side. keytalk's framing protocol
(`keytalk.protocol`) handles message ids, sequence numbers, and `START`/`END`
boundary markers so messages survive fragmentation, interleaving, and streaming.

## How it is structured

| Module | Responsibility |
| --- | --- |
| `keytalk.protocol` | Transport-agnostic framing, chunking, reassembly, streaming encoder. |
| `keytalk.transport` | `Transport` interface + in-memory loopback used by the tests. |
| `keytalk.backends` | `LLMBackend` interface, `OllamaBackend`, and test fakes. |
| `keytalk.host` | `HostService`: prompt frames -> LLM -> streamed response frames. |
| `keytalk.consumer` | `ConsumerClient`: prompt -> frames -> reassembled/streamed reply. |
| `keytalk.ble` | Real radio adapters: `bless` peripheral (host), `bleak` central (consumer). |
| `keytalk.server` | Ollama-compatible HTTP bridge exposed by `keytalk consume --serve`. |
| `keytalk.cli` | `keytalk host` / `keytalk consume` / `keytalk scan` commands. |

The radio layer is the **only** part that needs Bluetooth. Everything else is
pure Python and is exercised end-to-end over an in-memory loopback transport, so
the protocol can be tested exhaustively without hardware.

## Install

```bash
python3 -m pip install -e .             # core library (no radio deps)
python3 -m pip install -e ".[host]"     # + bless, for running the host
python3 -m pip install -e ".[consumer]" # + bleak, for the consumer
python3 -m pip install -e ".[ble]"      # both
```

> The BLE adapters require OS Bluetooth support (CoreBluetooth on macOS, BlueZ
> on Linux). `bless` (GATT server) is only available where its backend is
> supported. The core library and the whole test-suite have **no** dependencies.

## Usage

On the **host** (with Ollama already running and a model pulled):

```bash
keytalk host --model llama3
```

On the **consumer**:

```bash
keytalk scan                                   # find the host's address
keytalk consume --address <ADDRESS> --prompt "Explain BLE GATT in one line."
```

### Ollama-compatible endpoint (`--serve`)

Many tools (editors, IDE extensions, chat front-ends) already speak the
[Ollama](https://ollama.com) HTTP API. `keytalk consume --serve` runs a local,
dependency-free Ollama-compatible HTTP server on the consumer machine and bridges
every request to the LLM running on the remote BLE host — so a tool such as
VS Code only needs to point at this port instead of a real Ollama install:

```bash
keytalk consume --address <ADDRESS> --serve            # binds 127.0.0.1:11434
keytalk consume --address <ADDRESS> --serve --port 11434 --model llama3
```

It implements the endpoints clients probe and use: `GET /` (health),
`GET /api/version`, `GET /api/tags`, `POST /api/show`, `POST /api/generate`, and
`POST /api/chat` — with both streaming (newline-delimited JSON) and
non-streaming (`"stream": false`) responses. Point your Ollama client at
`http://127.0.0.1:11434` (or whatever `--host`/`--port` you chose) and it will
transparently talk to the model over Bluetooth LE.

### Library API

```python
import asyncio
from keytalk import HostService, ConsumerClient, EchoBackend, create_loopback

async def main():
    host_t, consumer_t = create_loopback()            # swap for real BLE transports
    host = HostService(host_t, EchoBackend())
    consumer = ConsumerClient(consumer_t)
    await host.start(); await consumer.start()

    print(await consumer.generate("hello over bluetooth"))
    async for piece in consumer.stream("stream me"):  # incremental tokens
        print(piece, end="")

    await consumer.close(); await host.close()

asyncio.run(main())
```

To run against real hardware, replace the loopback transports with
`keytalk.ble.peripheral.BlessPeripheralTransport` (host) and
`keytalk.ble.central.BleakCentralTransport(address)` (consumer); the rest of the
code is identical.

## Custom GATT service

| Item | UUID |
| --- | --- |
| Service | `9a8c0001-7b1e-4f9a-8c3d-2f6b1e9a8c00` |
| PROMPT characteristic (write) | `9a8c0002-7b1e-4f9a-8c3d-2f6b1e9a8c00` |
| RESPONSE characteristic (notify) | `9a8c0003-7b1e-4f9a-8c3d-2f6b1e9a8c00` |

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The suite covers frame encode/decode and validation, chunking edge cases (empty,
exact-multiple, large payloads), reassembly (ordering, interleaving, restart,
error cases), the streaming encoder, the loopback transport, the backends and
Ollama line parsing, and full host<->consumer integration including large
payloads, Unicode split across frames, empty prompts/responses, incremental
streaming, backend errors surfacing as `RemoteError`, concurrent requests, id
reuse, and timeouts.  A dedicated suite also drives the Ollama-compatible
`--serve` bridge over a real TCP socket — discovery endpoints, streaming and
non-streaming `/api/generate` and `/api/chat`, chunked encoding, keep-alive,
malformed-request handling, and the full server → consumer → BLE loopback →
host pipeline.
