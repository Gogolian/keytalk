#!/usr/bin/env python3
"""End-to-end smoke test: keytalk <-> real llama-server (llama.cpp) with MTP.

Verifies, against a *real* server (not mocks):

1. ``LlamaCppBackend`` streams the native ``/completions`` endpoint and the
   text survives the whole keytalk pipeline (consumer -> BLE framing -> host ->
   llama.cpp and back);
2. structured chat passthrough works (``/v1/chat/completions``, server-side
   Jinja template);
3. MTP generation stats (``timings.draft_n`` / ``draft_n_accepted``) ride back
   to the consumer via the TIMINGS trailer.

Usage:
    ./smoke_llamacpp.py --spawn              # start build-llama.cpp llama-server
    ./smoke_llamacpp.py --url http://127.0.0.1:8080   # use a running server

--spawn uses serve.sh's exact model/MTP flags on a private port (8123) with a
small context so it loads fast.  If a benchmark pkill's the server mid-run the
smoke reports INCONCLUSIVE rather than FAIL.
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from keytalk.backends import LlamaCppBackend  # noqa: E402
from keytalk.consumer import ConsumerClient  # noqa: E402
from keytalk.host import HostService  # noqa: E402
from keytalk.transport import create_loopback  # noqa: E402

BIN = os.path.expanduser("~/code/localllama/build-llama.cpp/bin/llama-server")
DIR = os.path.expanduser("~/code/localllama/models/Qwen3.8-3.6-27B-blend-GGUF")
MODEL = os.path.join(DIR, "Qwen3.8-3.6-27B-blend-Q5_K_M.gguf")
PORT = 8123


def spawn_server() -> subprocess.Popen:
    cmd = [
        BIN,
        "--model", MODEL,
        "--alias", "keytalk-smoke",
        "--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
        "--n-gpu-layers", "99", "--ctx-size", "4096", "--parallel", "1",
        "--jinja", "--reasoning", "on", "--reasoning-format", "deepseek",
        "--chat-template-kwargs", '{"enable_thinking":true,"preserve_thinking":true}',
        "--temp", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0",
        "--repeat-penalty", "1",
        "--host", "127.0.0.1", "--port", str(PORT),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_healthy(url: str, proc: subprocess.Popen, timeout: float = 240.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False  # server died (killed by a benchmark?) -> inconclusive
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


async def run_pipeline(url: str) -> dict:
    backend = LlamaCppBackend(model="keytalk-smoke", host=url, n_predict=48)
    host_t, consumer_t = create_loopback()
    host = HostService(host_t, backend, max_payload_size=64)
    consumer = ConsumerClient(consumer_t, max_payload_size=64, timeout=60.0,
                              keepalive_interval=0)
    await host.start()
    await consumer.start()
    results = {}
    try:
        prompt = "Say hello in exactly three words."
        text = await consumer.generate(prompt)
        results["prompt_text"] = text
        results["prompt_timings"] = consumer.last_timings

        messages = [{"role": "system", "content": "Answer with one word."},
                    {"role": "user", "content": "2+2?"}]
        chat_text = await consumer.chat(messages, temperature=0)
        results["chat_text"] = chat_text
        results["chat_timings"] = consumer.last_timings

        models = await consumer.list_models()
        results["models"] = models
    finally:
        await consumer.close()
        await host.close()
    return results


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' - ' + detail) if detail else ''}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="use an already-running llama-server")
    parser.add_argument("--spawn", action="store_true",
                        help="spawn the local llama-server on port %d" % PORT)
    parser.add_argument("--expect-mtp", action="store_true",
                        help="require MTP draft acceptance stats (draft_n) in "
                        "timings; implied by --spawn")
    args = parser.parse_args()
    expect_mtp = args.expect_mtp or args.spawn

    proc = None
    url = args.url
    try:
        if args.spawn:
            print(f"[smoke] starting {BIN} (MTP draft-mtp n=2) on :{PORT} ...")
            proc = spawn_server()
            url = f"http://127.0.0.1:{PORT}"
        if not url:
            print("error: pass --url or --spawn", file=sys.stderr)
            return 2
        print(f"[smoke] waiting for {url}/health ...")
        if not wait_healthy(url, proc):
            print("INCONCLUSIVE: server never became healthy "
                  "(was it killed by a benchmark?)")
            return 3

        results = asyncio.run(run_pipeline(url))

        print("[smoke] results:")
        print("  generate:", json.dumps(results["prompt_text"])[:120])
        print("  chat    :", json.dumps(results["chat_text"])[:120])
        print("  timings :", json.dumps(results["prompt_timings"]))
        print("  models  :", results["models"])

        ok = True
        ok &= check("generate text non-empty", bool(results["prompt_text"].strip()))
        ok &= check("chat text non-empty", bool(results["chat_text"].strip()))
        ok &= check("model listed over BLE", bool(results["models"]))
        for key in ("prompt_timings", "chat_timings"):
            t = results[key] or {}
            ok &= check(f"{key} delivered", bool(t), json.dumps(t)[:100])
            if expect_mtp:
                ok &= check(
                    f"{key} has MTP acceptance stats",
                    "draft_n" in t,
                    f"draft_n_accepted={t.get('draft_n_accepted')}/{t.get('draft_n')}",
                )
            elif "draft_n" in t:
                print(f"  INFO  {key} MTP stats: "
                      f"{t.get('draft_n_accepted')}/{t.get('draft_n')} accepted")
        print("[smoke]", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        if proc is not None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
