#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-accurate streaming decode gate.

``decode_gate.py`` divides the streaming span by the number of *chunks*, which
is only the token count when the server emits one token per chunk. With
speculative decoding the server coalesces every token of a step into one chunk,
so the chunk count under-counts tokens by the acceptance length. This gate
divides by the exact ``usage.completion_tokens`` instead and also reports the
chunk interval, which is the engine step time when drafting is on.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decode_gate import _post, make_token_ids  # noqa: E402


def run_once(base: str, model: str, prompt_ids: list[int], max_tokens: int, timeout: float):
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
    first = last = None
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
            data = json.loads(body)
            if data.get("usage"):
                usage = data["usage"]
            # Only content chunks count: the final usage chunk carries no text.
            if not (data.get("choices") and data["choices"][0].get("text")):
                continue
            now = time.perf_counter()
            if first is None:
                first = now
            last = now
            n_chunks += 1
    if first is None or last is None or usage is None:
        raise SystemExit("stream produced no tokens or no usage")
    tokens = usage["completion_tokens"]
    span = last - first
    return {
        "ttft_s": first - t0,
        "span_s": span,
        "tokens": tokens,
        "chunks": n_chunks,
        "tokens_per_chunk": tokens / n_chunks if n_chunks else float("nan"),
        "ms_per_chunk": span * 1000 / max(n_chunks - 1, 1),
        "ms_per_token": span * 1000 / max(tokens - 1, 1),
        "tok_per_s": (tokens - 1) / span if span > 0 else float("nan"),
        "usage": usage,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=26663)
    ap.add_argument("--model", default="dsv41f-port")
    ap.add_argument("--prompt-tokens", type=int, default=32000)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    prompt_ids = make_token_ids(base, args.prompt_tokens, args.seed)
    rows = []
    for i in range(args.repeats):
        row = run_once(base, args.model, prompt_ids, args.max_tokens, args.timeout)
        rows.append(row)
        print(
            f"[{i}] ttft={row['ttft_s']:.3f}s {row['tok_per_s']:.3f} tok/s "
            f"({row['ms_per_token']:.2f} ms/token) chunks={row['chunks']} "
            f"tokens/chunk={row['tokens_per_chunk']:.2f} "
            f"ms/chunk={row['ms_per_chunk']:.2f}"
        )
    summary = {
        "prompt_tokens": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "tok_per_s_mean": statistics.mean(r["tok_per_s"] for r in rows),
        "ms_per_token_mean": statistics.mean(r["ms_per_token"] for r in rows),
        "ms_per_chunk_mean": statistics.mean(r["ms_per_chunk"] for r in rows),
        "tokens_per_chunk_mean": statistics.mean(r["tokens_per_chunk"] for r in rows),
        "ttft_s_mean": statistics.mean(r["ttft_s"] for r in rows),
        "rows": rows,
    }
    print(
        f"MEAN {summary['tok_per_s_mean']:.3f} tok/s "
        f"{summary['ms_per_token_mean']:.3f} ms/token "
        f"| step {summary['ms_per_chunk_mean']:.2f} ms "
        f"x {summary['tokens_per_chunk_mean']:.3f} tokens"
    )
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
