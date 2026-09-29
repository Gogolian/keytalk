# keytalk

Use a large language model running on **one** machine from **another** machine —
over **Bluetooth LE** instead of an IP network.

- The **HOST** (e.g. a Mac) runs an LLM via [Ollama](https://ollama.com),
  [LM Studio](https://lmstudio.ai), or [OpenRouter](https://openrouter.ai)
  and exposes it through a **custom BLE GATT service**.
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

**Performance & reliability**: Both directions are compressed with zlib
(prompts as a whole, responses as a stream), typically reducing transmission
size by 60-80%. The consumer reports the negotiated BLE MTU to the host
(`HELLO`), so frames use the full link capacity instead of the 23-byte default
(~10x fewer frames). Lost notifications are recovered with cumulative
ACKs + retransmission, and a dead link is survived with automatic `RESUME`
(replay from the last received frame), request retries, and `CANCEL` of
abandoned generations.

## How it is structured

| Module | Responsibility |
| --- | --- |
| `keytalk.protocol` | Transport-agnostic framing, chunking, reassembly, streaming encoder. |
| `keytalk.transport` | `Transport` interface + in-memory loopback used by the tests. |
| `keytalk.backends` | `LLMBackend` interface, `OllamaBackend`, `LMStudioBackend`, `OpenRouterBackend`, and test fakes. |
| `keytalk.backends` | `LLMBackend` interface, `OllamaBackend`, `LMStudioBackend`, `LlamaCppBackend` (llama.cpp with MTP), and test fakes. |
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

On the **host**, choose a backend:

```bash
# Ollama (default) — model must already be pulled
keytalk host --model llama3

# LM Studio — point at its local server
keytalk host --backend lmstudio --lmstudio-host http://localhost:1234 --model gemma-4-31b-it

# OpenRouter — hosted models via API key
keytalk host --backend openrouter --model anthropic/claude-3.5-sonnet --openrouter-key sk-or-...
# or set OPENROUTER_API_KEY in the environment instead of passing --openrouter-key
keytalk host --backend llamacpp --llamacpp-host http://127.0.0.1:8080   # llama-server
```

On the **consumer**:

```bash
keytalk scan                                   # find the host's address
keytalk consume --address <ADDRESS> --prompt "Explain BLE GATT in one line."
keytalk consume --address <ADDRESS> --prompt "..." --retries 3 --keepalive 10
```

Both commands accept tuning flags: `keytalk host --mtu 185 --notify-interval
0.004` controls frame sizing and notification pacing; `keytalk consume
--retries N --keepalive SECS --timeout SECS --mtu N` controls request retries,
BLE keepalive pings, the idle timeout before a stalled response is resumed, and
the cap for the auto-detected link MTU.

### llama.cpp backend (with MTP)

`--backend llamacpp` bridges to a [`llama-server`](https://github.com/ggml-org/llama.cpp)
instance using its native `/completions` and `/v1/chat/completions` endpoints.
Structured chat messages are passed through untouched so the model's own Jinja
chat template (thinking modes, `preserve_thinking`, tool roles) does the
rendering on the host.  Generation stats reported by the server - including
multi-token-prediction acceptance (`timings.draft_n` / `draft_n_accepted`) -
ride back to the consumer via the `TIMINGS` trailer and surface as
`ConsumerClient.last_timings` (and in the `--serve` done envelopes).  See
`tools/smoke_llamacpp.py` for a real-server end-to-end check.

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

## Reliability over a lossy link

BLE notifications are fire-and-forget and links drop silently, so the protocol
is built to recover rather than fail:

| Mechanism | What it does |
| --- | --- |
| ACK + retransmission | The consumer cumulatively ACKs response frames; the host retransmits whatever is unacked (Go-Back-N, bounded window doubles as flow control). |
| `RESUME` | On a stall or after a reconnect the consumer asks for retransmission from its last contiguous sequence number. The host replays from its retained frame store - mid-stream or after completion - so `stream()` continues exactly where it left off. |
| Request retries | Requests that fail before producing output are retried from scratch (`--retries`); `generate()` also retries after partial output. |
| `CANCEL` | Abandoning a stream (e.g. an HTTP client disconnecting under `--serve`) tells the host to stop generating, freeing the LLM and the link. |
| NACK fast-failure | A garbled prompt is answered with an immediate `ERROR` instead of leaving the consumer waiting out its timeout. |
| Keepalive | Periodic `PING` keeps the link fresh and exercises the transport's reconnect logic (`--keepalive`). |
| `HELLO` | The consumer announces its frame payload size once the real MTU is known; the host sizes its notifications to match. |

## Beyond GATT: L2CAP CoC

`keytalk.ble.l2cap` carries the same protocol over BLE L2CAP
Connection-Oriented Channels (Bluetooth 4.2+): reliable credit-based streams
with negotiated MTUs up to 64 KiB - roughly 10x+ GATT throughput on the same
radio.  The frame links (`StreamLink` for byte streams, `DatagramLink` for
message-preserving sockets) and `LinkTransport` are fully unit-tested over real
socket pairs; Linux/BlueZ gets `open_l2cap_socket`, macOS glue for
CoreBluetooth's `CBL2CAPChannel` is in `MacStreamPair`.  Two-machine RF testing
uses `tools/l2cap_smoke.py`.

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
reuse, and timeouts.  Dedicated suites cover compression (prompt round-trips,
streaming response decompression, clean error transitions) and reliability
(ACK/retransmission over lossy links, MTU negotiation, NACK fast-failure,
resume after link outages, whole-request retries, and CANCEL).  A final suite
drives the Ollama-compatible `--serve` bridge over a real TCP socket —
discovery endpoints, streaming and non-streaming `/api/generate` and
`/api/chat`, chunked encoding, keep-alive, malformed-request handling, and the
full server → consumer → BLE loopback → host pipeline.
