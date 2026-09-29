# keytalk — architecture and transfer modes

This replaces the original phased implementation plan, which described the
pre-transfer-mode protocol and has since been superseded. What follows is the
state of the code as it stands.

## Protocol

Transport-agnostic framing lives in `keytalk.protocol`. A *message* is an
ordered run of frames sharing a `message_id`; the first carries `START`, the
last carries `END`, and sequence numbers increase by exactly one per frame.

```
offset  size  field
0       1     version
1       1     message type
2       1     flags
3       2     message id
5       2     sequence number
7       N     payload
```

Flags: `START` (1), `END` (2), `COMPRESSED` (4, zlib), `CHECKSUM` (16, a 4-byte
CRC32 trailer on the flagged frame). Bit 8 (`DELTA`) is reserved and unused.

Two message encoders exist and they are deliberately asymmetric:

* `chunk_message()` — for a payload that is already complete. Takes
  `start_flags` so whole-message metadata (e.g. `COMPRESSED`) attaches to the
  `START` frame without post-processing the returned list.
* `FrameStreamEncoder` — for a stream of unknown total size. Compresses
  incrementally; `flush()` cuts the zlib stream without ending the message so
  plain-text frames (errors, the `TIMINGS` trailer) can follow.

**The `CHECKSUM` flag, not `END`, marks the CRC trailer.** A message can close
on a following `TIMINGS` frame instead, so receivers key off the flag.

## Transfer modes

`keytalk.modes` bundles each mode's choices — frame MTU, write-with-response,
compression codec, flow control, reliability window — into a `ProfileConfig`.
`Mode.LEGACY` is the original hard-coded behaviour and is permanently the
fallback.

| Mode | Wire id | Link | Reliability |
| --- | --- | --- | --- |
| `legacy` | 0 | BLE GATT, 23-byte ATT MTU | Go-Back-N window 32 |
| `fast_gatt` | 1 | BLE GATT, negotiated MTU | Go-Back-N window 64 |
| `l2cap_coc` | 2 | BLE L2CAP CoC, 1024 | stream, `reliability_window=0` |
| `rfcomm` | 3 | Bluetooth Classic SPP, 1024 | stream, `reliability_window=0` |

`reliability_window=0` marks a transport that is already reliable and ordered
(L2CAP CoC, RFCOMM); the message pump then falls back to a window of 64, which
costs nothing because nothing is ever dropped.

### Negotiation

1. The host advertises its supported modes on a read-only **CAPS** GATT
   characteristic. A host without it is pre-negotiation and the consumer falls
   back to legacy silently.
2. The consumer reads CAPS, picks the best common mode
   (`negotiate_mode`), and confirms it with a `SELECT` frame carrying the mode
   id and its link MTU.
3. `HELLO` then carries the largest frame payload the consumer accepts.

**Peer-reported MTUs are untrusted input.** Both ends clamp them:
`ConsumerClient` to its `max_mtu` (CLI `--mtu`, default 512) and `HostService`
to `DEFAULT_MAX_MTU` (1024) via its `max_mtu` argument. A `SELECT` claiming
65535 is clamped and logged, never sized into. `HostService` also tracks
`_window_explicit` so a mode change can retune the window unless the caller
pinned one.

## Request handling

`HostService` dispatches on the negotiated mode:

* **Streaming** — tokens are encoded into frames as they arrive (lowest
  latency).
* **Buffered** — the whole reply is collected, then compressed in one shot for
  a better ratio (`--buffer-response`, or automatically for the stream modes).

Both paths deliver through the same `_OutboundMessage` pump, so reliability is
identical: duplicate suppression, frame retention for `RESUME`, cumulative ACKs
and the `TIMINGS` trailer.

The message id is **reserved at dispatch time**, before the handler task is
spawned. Reserving inside the task body lets N duplicate writes all pass the
check, because none of them have run yet.

Generation metadata (llama.cpp timings including MTP draft acceptance,
structured `tool_calls`, `finish_reason`) rides the tail of the *same* message
as plain-text `TIMINGS` frames, after any compressed stream is cut.

## Per-request metadata

A response's metadata belongs to that response, so it rides the stream object
(`ConsumerClient.stream()` returns a `ResponseStream`). It is deliberately
**not** read from shared client state: `OllamaBridgeServer` serves requests
concurrently, and a shared `last_tool_calls` slot would let a request that
finishes late hand its metadata to whichever envelope was built next.

`ConsumerClient.last_timings` / `last_tool_calls` still exist for library and
CLI callers who want the most recent request's values; the bridge does not use
them.

## Known gaps

- **The consumer ACKs every single frame** it receives, each a write over the
  acknowledged prompt path. That per-frame round trip is the cost `fast_gatt`
  exists to avoid. ACK coalescing would be the fix.
- **`zstd`/`lz4` codecs are declared in `ProfileConfig` but never used**; zlib
  is the only implemented codec.
- **Windows L2CAP CoC and RFCOMM are best-effort** and may support only
  `legacy` + `fast_gatt`.
- **`keytalk scan --classic` is a stub** that prints a notice and falls back to
  the BLE scan; Bluetooth Classic discovery is not implemented anywhere.
- **Platform glue is untested off its own platform.** `classic/macos.py`,
  `ble/l2cap/{linux,macos}.py` and the RFCOMM/L2CAP socket code are only
  exercised on macOS by this suite.
- **Message ids are 16-bit** with wraparound reuse; `resume_ttl` bounds how long
  completed ids are retained, but a very long-lived connection with many
  in-flight requests could in principle collide.

## Development

```bash
python3 -m unittest discover -s tests -v        # full suite
python3 -m pyflakes src/keytalk tests tools     # lint
```

The core library and the whole test-suite have no runtime dependencies. See
`COMPRESSION.md` for the original compression write-up and
`tests/test_negotiation_limits.py` for the regression tests covering the
negotiation limits and buffered-mode parity described above.
