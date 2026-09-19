#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill gate: cold-prompt TTFT and prefill throughput, per prompt length.

``--max-num-batched-tokens`` caps a prefill chunk at
``budget - max_num_new_slots_for_drafting`` tokens, so a long prompt is served
as ``ceil(len / (budget - slots))`` chunks and a larger budget buys fewer,
wider chunks. This gate is the other half of the budget decision: it reports
what the budget buys in TTFT, which the token-id gates cannot say.

Three things would otherwise make the number meaningless, so each request is
built to avoid them:

* every repeat carries a ``cache_salt`` with a per-run nonce, because prefix
  caching is on by default here and a repeated prompt is served from the cache
  (a 32K "prefill" then finishes in 0.4 s instead of 6.5 s);
* the request asks for one token and the gate idles before it, because TTFT is
  measured from the request and a previous request still decoding delays the
  next one by its own decode time (the same 4096-token prompt came out bimodal,
  0.5 s and 1.4 s, at ``--max-tokens 8``);
* the prefix-cache hit counter is read around every request, so the artifact
  records that the prompt really was prefilled rather than asserting it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decode_gate import _post, make_token_ids  # noqa: E402

HITS = "vllm:prefix_cache_hits_total"


def prefix_cache_hits(base: str) -> float:
    with urllib.request.urlopen(f"{base}/metrics", timeout=120) as resp:
        text = resp.read().decode()
    total = 0.0
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        series, _, value = line.rpartition(" ")
        name, _, _ = series.partition("{")
        if name == HITS:
            total += float(value)
    return total


def prefill_once(
    base: str,
    model: str,
    prompt_ids: list[int],
    max_tokens: int,
    cache_salt: str,
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
        "cache_salt": cache_salt,
    }
    before = prefix_cache_hits(base)
    start = time.perf_counter()
    ttft = None
    with _post(f"{base}/v1/completions", payload) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            obj = json.loads(body)
            text = "".join(
                choice.get("text") or "" for choice in obj.get("choices") or []
            )
            if text and ttft is None:
                ttft = time.perf_counter() - start
    total = time.perf_counter() - start
    return {
        "ttft_s": ttft,
        "total_s": total,
        "cache_hit_tokens": prefix_cache_hits(base) - before,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=26663)
    ap.add_argument("--model", default="dsv41f-port")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 32768])
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument(
        "--idle-s",
        type=float,
        default=2.0,
        help="sleep before each repeat so the engine is quiet when it starts",
    )
    ap.add_argument("--label", default="")
    ap.add_argument("--server-args", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    nonce = time.strftime("%H%M%S")
    prompt = make_token_ids(base, max(args.lengths), args.seed)
    rows = []
    print(
        f"{'prompt':>8} {'ttft med':>10} {'ttft spread':>12} "
        f"{'prefill tok/s':>14} {'cached':>7}"
    )
    for length in sorted(args.lengths):
        runs = []
        for index in range(args.repeats):
            time.sleep(args.idle_s)
            runs.append(
                prefill_once(
                    base,
                    args.model,
                    prompt[:length],
                    args.max_tokens,
                    f"pf{nonce}-{length}-{index}",
                )
            )
        ttfts = [run["ttft_s"] for run in runs]
        hit = max(run["cache_hit_tokens"] for run in runs)
        median = statistics.median(ttfts)
        row = {
            "length": length,
            "ttft_s": ttfts,
            "ttft_median_s": median,
            "prefill_tokens_per_s": length / median,
            "cache_hit_tokens": hit,
        }
        rows.append(row)
        print(
            f"{length:>8} {median:>10.3f} {max(ttfts) - min(ttfts):>12.3f} "
            f"{row['prefill_tokens_per_s']:>14.0f} {hit:>7}"
        )
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "label": args.label,
                    "server_args": args.server_args,
                    "seed": args.seed,
                    "repeats": args.repeats,
                    "nonce": nonce,
                    "rows": rows,
                },
                indent=1,
            )
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
