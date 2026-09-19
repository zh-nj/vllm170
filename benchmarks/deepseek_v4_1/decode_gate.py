#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Production decode gate: exact-token prompt, streaming, no instrumentation.

The prompt is sent as raw token ids so ``usage.prompt_tokens`` is exact by
construction. Decode throughput is measured between the first and the last
streamed token, which removes the prefill/TTFT component from the number
without subtracting two separate calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request

WORDS = (
    "system kernel memory buffer thread process socket packet register cache "
    "pointer allocate schedule interrupt virtual physical address translate "
    "compile execute branch predict pipeline vector matrix tensor gradient "
    "cluster network storage device driver latency throughput bandwidth "
    "protocol checksum payload header footer stream channel partition replica "
    "consistency quorum commit rollback snapshot journal segment offset stride "
    "alignment padding overflow underflow saturation rounding precision"
).split()


def _post(url: str, payload: dict, timeout: float = 1800.0):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def make_token_ids(base: str, n_tokens: int, seed: int) -> list[int]:
    text = " ".join(
        WORDS[(i * 7 + seed) % len(WORDS)] for i in range(n_tokens * 4 + 512)
    )
    with _post(f"{base}/tokenize", {"model": "", "prompt": text}) as resp:
        ids = json.loads(resp.read())["tokens"]
    if len(ids) < n_tokens:
        raise SystemExit(f"tokenize returned only {len(ids)} < {n_tokens}")
    return ids[:n_tokens]


def run_once(
    base: str,
    model: str,
    prompt_ids: list[int],
    max_tokens: int,
    timeout: float,
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    first = None
    last = None
    n_chunks = 0
    usage = None
    with _post(f"{base}/v1/completions", payload, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            obj = json.loads(body)
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                if choice.get("text"):
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    n_chunks += 1
    if first is None or last is None or n_chunks < 2:
        raise SystemExit("stream produced fewer than two text chunks")
    intervals = n_chunks - 1
    decode_s = last - first
    return {
        "ttft_s": first - t0,
        "decode_s": decode_s,
        "intervals": intervals,
        "ms_per_token": decode_s / intervals * 1e3,
        "tok_per_s": intervals / decode_s,
        "chunks": n_chunks,
        "usage": usage,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=26663)
    ap.add_argument("--model", default="dsv41f-port")
    ap.add_argument("--prompt-tokens", type=int, default=32000)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    prompt_ids = make_token_ids(base, args.prompt_tokens, args.seed)
    digest = hashlib.sha256(
        json.dumps(prompt_ids).encode()
    ).hexdigest()  # prompt identity for cross-run comparison

    rows = []
    for i in range(args.repeats):
        row = run_once(
            base, args.model, prompt_ids, args.max_tokens, args.timeout
        )
        row["repeat"] = i
        rows.append(row)
        print(
            f"[{i}] prompt={args.prompt_tokens} out={args.max_tokens} "
            f"ttft={row['ttft_s']:.3f}s "
            f"{row['ms_per_token']:.3f} ms/token "
            f"{row['tok_per_s']:.3f} tok/s usage={row['usage']}"
        )

    summary = {
        "prompt_tokens": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "prompt_sha256": digest,
        "repeats": args.repeats,
        "ms_per_token_mean": statistics.mean(
            r["ms_per_token"] for r in rows
        ),
        "tok_per_s_mean": statistics.mean(r["tok_per_s"] for r in rows),
        "ms_per_token_min": min(r["ms_per_token"] for r in rows),
        "ms_per_token_max": max(r["ms_per_token"] for r in rows),
        "ttft_s_mean": statistics.mean(r["ttft_s"] for r in rows),
        "rows": rows,
    }
    print(
        f"MEAN {summary['ms_per_token_mean']:.3f} ms/token "
        f"{summary['tok_per_s_mean']:.3f} tok/s "
        f"(min {summary['ms_per_token_min']:.3f} "
        f"max {summary['ms_per_token_max']:.3f}) sha={digest[:16]}"
    )
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
