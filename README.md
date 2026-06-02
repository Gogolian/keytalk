# keytalk

`keytalk` is a small, testable transport for moving prompt and response data through a channel that only looks like a generic keyboard.

## What this repository solves

This repository does **not** try to make two normal host computers magically become USB keyboards for each other over a plain USB-C cable. That is a hardware limitation: at least one side still needs a USB gadget, microcontroller, KVM, or other device that can present itself as a keyboard.

What this repository *does* provide is the software layer you need once you have that keyboard-like path:

1. Frame prompt and response payloads into keyboard-safe lines.
2. Decode those lines back into structured messages.
3. Forward prompt requests to an Ollama server on the machine doing inference.
4. Return the generated response over the same keyboard-safe transport.

The result is a minimal foundation for “keyboard-talk”: one machine can request inference, another machine can run the model, and both sides exchange data as text that can be typed through a generic keyboard device.

## Repository contents

- `/tmp/workspace/Gogolian/keytalk/keytalk/protocol.py` — framing, chunking, checksums, and message decoding.
- `/tmp/workspace/Gogolian/keytalk/keytalk/bridge.py` — prompt/response structures plus an Ollama-backed responder.
- `/tmp/workspace/Gogolian/keytalk/keytalk/cli.py` — CLI for encoding, decoding, request generation, and replying.
- `/tmp/workspace/Gogolian/keytalk/tests/` — unit tests for protocol, bridge flow, and CLI behavior.

## How the flow works

1. **Client side**
   - Encode a prompt request into `KT1|...` frames.
   - Send those characters through your keyboard-emulating link.
2. **Inference side**
   - Read the frames in a terminal or capture process.
   - Decode them into a prompt request.
   - Call a local Ollama server.
   - Encode the response into `KT1|...` frames.
3. **Client side**
   - Decode the returned response frames.
   - Display or consume the response text.

## Installation

```bash
cd /tmp/workspace/Gogolian/keytalk
python3 -m pip install -e .
```

You can also run the package without installation:

```bash
cd /tmp/workspace/Gogolian/keytalk
python3 -m keytalk --help
```

## Command examples

### Encode plain text

```bash
python3 -m keytalk encode --text "hello keyboard world" --message-id demo1
```

### Build a prompt request for another machine

```bash
python3 -m keytalk request --model llama3.2 --prompt "Summarize the attached log." --message-id req1
```

### Inspect a received prompt request

```bash
python3 -m keytalk receive-request '...frame 1...' '...frame 2...'
```

### Turn a prompt request into a response with Ollama

```bash
python3 -m keytalk reply --base-url http://127.0.0.1:11434 '...frame 1...' '...frame 2...'
```

### Decode a received response

```bash
python3 -m keytalk receive-response '...response frame 1...' '...response frame 2...'
```

### Local end-to-end smoke test without Ollama

```bash
REQUEST="$(python3 -m keytalk request --model llama3.2 --prompt 'ping' --message-id demo2)"
python3 -m keytalk mock-reply --response 'pong' $REQUEST | xargs python3 -m keytalk receive-response
```

## Running tests

```bash
cd /tmp/workspace/Gogolian/keytalk
python3 -m unittest discover -s tests -v
```

## Protocol notes

- Each frame starts with `KT1`.
- Payloads are UTF-8, base64url encoded, and chunked.
- A short SHA-256 digest protects each frame against accidental corruption.
- Multiple frames can be reassembled into one logical prompt or response.

## Suggested hardware setup

To use this with two physical machines, pair the software in this repository with one of the following:

- a Linux system configured as a USB HID gadget
- a microcontroller that can emulate a USB keyboard
- a KVM or programmable keyboard-injection device

That hardware handles the USB presentation. `keytalk` handles the framing and the inference bridge.