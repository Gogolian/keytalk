# keytalk — Faster Protocols & Pluggable Transfer Modes

A plan to add selectable, negotiated transfer modes to keytalk. The current
protocol becomes **mode 0 (legacy)** and remains the default and always
available. Faster protocols are added as independent, verifiable steps.

## Scope (agreed)

- **Platforms:** macOS + Linux + Windows.
- **Priority:** throughput *and* latency, equally.
- **Modes to add:** `FAST_GATT`, `L2CAP_COC`, `CLASSIC_RFCOMM` — implemented in
  separate steps.
- **Dependencies:** optional `zstd`/`lz4` as extras; the core stays zero-dep.

---

## 1. Current protocol (inferred)

**Bluetooth Low Energy (BLE) GATT** with a custom application layer on top:

- **Topology:** Host = GATT peripheral/server (via `bless`); Consumer = GATT
  central/client (via `bleak`). Custom 128-bit service `9a8c0001-…`
  (`ble/constants.py`) with two characteristics:
  - `PROMPT` (write / write-without-response): consumer → host
  - `RESPONSE` (read / **notify**): host → consumer
- **Framing** (`protocol.py`): 7-byte header (version, type, flags, `msg_id`,
  `seq`) + payload. MTU hardcoded to `DEFAULT_ATT_MTU = 23` → **only 13 payload
  bytes per frame** (a 2.3 KB prompt = 177 frames).
- **Reliability** (`reliability.py`): consumer → host uses write-with-response
  (link-reliable). host → consumer notifications are lossy, so an app-level
  **Go-Back-N** sliding window (window=32, rto=0.75s) with cumulative ACKs sent
  back over the PROMPT characteristic.
- **Compression:** zlib level 6 on **prompts only** (`COMPRESSED` flag). Plus a
  delta-encoding optimization for conversation history (`DELTA` flag +
  truncated SHA-256 reference).
- **Pacing:** host sleeps a fixed `notify_interval = 0.02s` between
  notifications (`ble/peripheral.py`).

### Bottlenecks

1. MTU never negotiated → 13 B/frame. BLE 4.2+ allows ATT MTU up to 517 (DLE
   251-byte PDUs) → **18–38× fewer frames**.
2. Fixed 20 ms notify pacing caps the response path at ~50 frames/s (~24 KB/s
   even with a large MTU).
3. Write-with-response adds a round-trip per prompt write.
4. Compression is one-directional (prompts only).
5. 7-byte header on a 13-byte payload = 35% overhead.

---

## 2. Faster protocol options (become selectable modes)

- **`FAST_GATT`** — same GATT topology, but: negotiate the ATT MTU, drop the
  fixed pacing in favor of credit/window flow control, write-without-response
  prompts with symmetric reliability, and bidirectional compression. No new
  dependencies, cross-platform. **Biggest ROI.**
- **`L2CAP_COC`** — BLE LE credit-based Connection-Oriented Channel: a reliable
  stream that bypasses ATT/GATT overhead with built-in credit flow control.
  Linux `AF_BLUETOOTH`/`BTPROTO_L2CAP` LE-PSM socket; macOS `CBL2CAPChannel`
  via pyobjc; Windows best-effort.
- **`CLASSIC_RFCOMM` / SPP** — Bluetooth Classic serial stream (~1–2 Mbps,
  inherently reliable/ordered). Linux `BTPROTO_RFCOMM`/PyBluez; macOS
  IOBluetooth via pyobjc; Windows RFCOMM. Requires classic pairing + discovery;
  all-new backend (not `bleak`/`bless`).
- **Codecs** — optional `zstd` (ratio) / `lz4` (speed) as extras, negotiated per
  mode.

---

## 3. Speed-up with data integrity

- **Throughput:** larger negotiated MTU + write-without-response + the existing
  Go-Back-N (sequence numbers already detect drops/reorders; retransmission
  recovers).
- **Integrity:** add a **message-level checksum** (CRC32C over the full
  reassembled payload, carried in the END-frame trailer) + a NAK/re-request path
  on mismatch. BLE's link layer already has a 24-bit CRC per PDU, so the main
  loss mode is *dropped frames* (handled by sequence numbers); the message
  checksum is cheap belt-and-suspenders that also guards the L2CAP/RFCOMM modes
  and app-level bugs.
- **Flow control:** replace the fixed 20 ms pacing with credit/window flow
  control so throughput scales with the link without reintroducing drops.
- **Compression:** bidirectional, with a negotiated codec.

---

## 4. Implementation plan — pluggable negotiated modes

Both sides select a mode at startup (`--mode`), negotiated on connect. Mode 0
(legacy) is the default and always available.

### Phase 0 — Mode scaffolding (no behavior change) ✅ DONE

1. ✅ New `src/keytalk/modes.py`: `Mode` enum + `ProfileConfig` dataclass (mtu,
   payload size, compression codec, flow-control style, write-with-response,
   transport factory). `LEGACY_PROFILE` constant + `profile_for_mode()` helper.
2. ✅ Added `--mode` to `host` and `consume` in `cli.py` (default `auto`);
   `ProfileConfig` threaded into `HostService` and `ConsumerClient` (stored as
   `self._profile`). Legacy path verified byte-identical.

### Phase 1 — Capability handshake + negotiation *(depends on 0)* ✅ DONE

3. ✅ Added read-only **CAPS characteristic** (`9a8c0004-…`) to the GATT service
   listing host-supported modes; old hosts lack it → consumer falls back to
   legacy transparently. New `HELLO`/`CAPS`/`SELECT`/`NAK` `MessageType`s in
   `protocol.py`; `encode_select_payload`/`decode_select_payload` (3-byte binary
   wire format). `NegotiationError` + `negotiate_mode()` + mode-ID table in
   `modes.py`. `read_caps()`/`mtu_size` hooks on `Transport` base class;
   `InMemoryTransport` accepts `caps=` for test simulation.
4. ✅ `ConsumerClient` accepts `requested_mode=` and runs `_negotiate()` at
   `start()`: reads CAPS char, picks best common mode via `negotiate_mode()`,
   sends `SELECT` frame (mode_id + MTU). `HostService` handles `SELECT` before
   the reassembler and stores `_negotiated_mtu`. CLI passes `requested_mode=`
   and `supported_modes=["legacy"]`; 19 negotiation tests in
   `tests/test_negotiation.py`.

### Phase 2 — `FAST_GATT` mode (biggest win, no new transport) *(depends on 1)* ✅ DONE

5. ✅ **Negotiate MTU:** consumer sends MTU in SELECT; host calls `make_fast_gatt_profile(mtu)`
   and sets `_max_payload = max_payload_for_mtu(mtu)` — 240–495 B/frame vs 13 B.
6. ✅ **Credit/window flow control:** `BlessPeripheralTransport` already skips sleep when
   `notify_interval=0`; `ReliableSender` window (64) now drives pacing.
7. ✅ **Write-without-response prompts:** consumer calls `transport.configure_write_mode(False)`
   after FAST_GATT negotiation; `BleakCentralTransport` overrides the hook.
   **Bidirectional compression:** host buffers full response, compresses with zlib,
   sets `COMPRESSED` flag on START frame; consumer's `_PendingRequest` uses an
   inner `Reassembler` for decompression when `reassemble=True`.
8. ✅ **Message-level CRC32 integrity:** `Flags.CHECKSUM` (bit 16) + `compute_crc32()`;
   `chunk_message(checksum=True)` appends a 4-byte CRC32 trailer; `FrameStreamEncoder`
   maintains a running CRC; `Reassembler` verifies and strips the trailer before
   decompression, raising `ProtocolError` on mismatch.
9. ✅ **In-memory throughput benchmark** in `tests/test_fast_gatt.py`: FAST_GATT achieves
   ~5–7× higher bytes/s and **631× fewer frames** than LEGACY on an 8 KiB payload.

### Phase 3 — `L2CAP_COC` mode (BLE credit-based stream) *(depends on 1)* ✅ DONE

10. ✅ New `src/keytalk/ble/l2cap/` per-platform transports: Linux
    `AF_BLUETOOTH`/`BTPROTO_L2CAP` LE-PSM socket; macOS `CBL2CAPChannel` via
    pyobjc; Windows best-effort (won't negotiate if unsupported).
    Length-prefixed framing on the reliable stream (drop Go-Back-N, keep the
    checksum). PSM exchanged via a GATT characteristic (`9a8c0005-…`).
    `L2CAPLoopbackTransport` using `socket.socketpair()` for in-process tests.
    `_handle_prompt_l2cap_coc()` on host sends frames directly (no ReliableSender).
    Consumer `reassemble=True` and `checksum=True` for L2CAP_COC like FAST_GATT.
    `BlessPeripheralTransport` advertises PSM char when `l2cap_coc` in modes;
    `BleakCentralTransport.read_l2cap_psm()` reads it. 20 tests in
    `tests/test_l2cap_coc.py`; **210× fewer frames** than LEGACY on an 8 KiB payload.

### Phase 4 — `CLASSIC_RFCOMM` / SPP mode (highest throughput) *(depends on 1)* ✅ DONE

11. ✅ New `src/keytalk/classic/` RFCOMM transports: `RFCOMMStreamTransport` base
    inheriting the 4-byte length-prefix framing from L2CAP; `RFCOMMLoopbackTransport`
    / `create_rfcomm_loopback()` for in-process tests; platform skeletons for Linux
    (`AF_BLUETOOTH/BTPROTO_RFCOMM`), macOS (IOBluetooth via pyobjc), and Windows
    (WinSock `AF_BTH`). `CLASSIC_RFCOMM_PROFILE` + `make_classic_rfcomm_profile()`
    in `modes.py`; `rfcomm` added to `_IMPLEMENTED_MODES`; host's
    `_handle_prompt_stream()` shared by L2CAP_COC and CLASSIC_RFCOMM (reliable
    ordered stream path: collect → compress → CRC32 → send direct). Consumer
    `_negotiate()`, reassemble flag, and checksum flag all include RFCOMM.
    `keytalk scan --classic` added (informative stub, falls back to BLE scan until
    Classic inquiry is implemented per-platform). `classic` extra in
    `pyproject.toml` (`PyBluez` on Linux). 28 tests in
    `tests/test_classic_rfcomm.py`; **210× fewer frames** than LEGACY on an 8 KiB
    payload.

### Cross-cutting

12. `pyproject.toml` extras: `fast` (zstd/lz4), `l2cap`, `classic` (pybluez);
    core stays zero-dep. Add `MODES.md` with a mode × platform × direction
    compatibility matrix.

---

## Affected files

- `src/keytalk/protocol.py` — integrity checksum + `HELLO/CAPS/SELECT/NAK`
  `MessageType`s; reuse `Frame`, `chunk_message`, `compute_message_checksum`.
- `src/keytalk/reliability.py` — per-mode window/rto; optionally reuse
  `ReliableSender` on the prompt path.
- `src/keytalk/transport.py` — `Transport` ABC is the seam for L2CAP/RFCOMM
  backends.
- `src/keytalk/ble/peripheral.py` / `central.py` — MTU negotiation, remove
  `notify_interval`, CAPS characteristic.
- `src/keytalk/consumer.py` / `host.py` — accept a `ProfileConfig`; drive
  handshake.
- `src/keytalk/cli.py` — `--mode` on `host`/`consume`, `scan --classic`.
- New: `src/keytalk/modes.py`, `src/keytalk/ble/l2cap/`,
  `src/keytalk/classic/`.

---

## Verification

1. **Legacy regression:** byte-identical frames vs today (unit test over
   `chunk_message`/`Reassembler`).
2. **Negotiation tests:** correct mode picked; fallback to legacy when CAPS
   absent; explicit-mode-not-offered errors clearly.
3. **Integrity test:** inject a dropped/corrupted frame → NAK + recovery;
   checksum mismatch rejected.
4. **Throughput benchmark:** `FAST_GATT` vs legacy over the in-memory transport
   (frames/s, bytes/s).
5. **Manual hardware runs** per mode on macOS↔macOS and macOS↔Linux.

---

## Decisions & open considerations

- Mode 0 (legacy) is the default and permanently supported (backward-compat
  requirement).
- Negotiation uses a *dedicated CAPS characteristic* so pre-mode hosts are
  undisturbed and consumers auto-fall-back.
- All faster modes reuse the transport-agnostic framing/checksum so
  L2CAP/RFCOMM share code with `FAST_GATT`.
- **Open:** checksum algorithm — CRC32C (fast, 4 bytes) vs reuse truncated
  SHA-256. *Rec: CRC32C.*
- **Open:** default `--mode auto` (negotiate best common) vs explicit `--mode X`
  (fail fast). *Rec: `auto` default, explicit overrides.*
- **Open:** Windows L2CAP/RFCOMM is best-effort and may support only `legacy` +
  `fast_gatt`; the compatibility matrix will make this explicit.
