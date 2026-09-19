#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed gate: what a long prefill does to a decode running beside it.

``--max-num-batched-tokens`` is a shared budget: a step spends up to
``budget - draft_slots`` tokens on a chunked prefill and gives the rest to the
decodes. A wider chunk therefore keeps the same total prefill work for a smaller
number of longer steps, which is what makes TTFT better and is also what makes a
concurrent decode's inter-token latency choppier. This gate measures both halves
at once, so the budget can be chosen knowing the cost and not only the gain.

A long cold prompt is sent, and once it is being prefilled a short request is
sent beside it; the short request's streamed intervals are its ITL under load.
The same short request is then measured on a quiet engine as the baseline. Every
request carries its own salt nonce so the long prompt is really prefilled.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decode_gate import _post, make_token_ids  # noqa: E402


def stream(base: str, model: str, prompt_ids: list[int], max_tokens: int, salt: str):
    """Stream one request; return per-token timestamps and the first-token time."""
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
        "stream": True,
        "cache_salt": salt,
    }
    stamps: list[float] = []
    with _post(f"{base}/v1/completions", payload) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            obj = json.loads(body)
            for choice in obj.get("choices") or []:
                if choice.get("text"):
                    stamps.append(time.perf_counter())
    return stamps


def itl(stamps: list[float], base: float | None = None) -> dict:
    """Inter-token latencies, optionally restricted to a time window."""
    if len(stamps) < 2:
        return {"n": 0, "median_ms": None, "max_ms": None, "intervals_ms": []}
    pairs = [
        (a, b)
        for a, b in zip(stamps, stamps[1:])
        if base is None or (base[0] <= (a + b) / 2 <= base[1])
    ]
    if not pairs:
        return {"n": 0, "median_ms": None, "max_ms": None, "intervals_ms": []}
    intervals = [(b - a) * 1e3 for a, b in pairs]
    return {
        "n": len(intervals),
        "median_ms": statistics.median(intervals),
        "max_ms": max(intervals),
        "intervals_ms": intervals,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=26663)
    ap.add_argument("--model", default="dsv41f-port")
    ap.add_argument("--long-tokens", type=int, default=32768)
    ap.add_argument("--short-tokens", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--delay-s", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--label", default="")
    ap.add_argument("--server-args", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    nonce = time.strftime("%H%M%S")
    long_prompt = make_token_ids(base, args.long_tokens, args.seed)
    short_prompt = make_token_ids(base, args.short_tokens, args.seed + 1)

    quiet_start = time.perf_counter()
    quiet = itl(
        stream(base, args.model, short_prompt, args.max_tokens, f"q-{nonce}-base")
    )
    quiet["ttft_ms"] = quiet["intervals_ms"][0] if quiet["intervals_ms"] else None
    loaded = []
    long_ttft = []
    for index in range(args.repeat):
        time.sleep(2.0)
        holder: dict = {}

        def run_short(holder: dict = holder, index: int = index) -> None:
            holder["stamps"] = stream(
                base,
                args.model,
                short_prompt,
                args.max_tokens,
                f"m-{nonce}-short-{index}",
            )

        # Decode first, prefill second: a request that arrives while the prefill
        # runs waits for the prefill to end before it is served at all, so only
        # this order overlaps the two and shows what the chunk does to the ITL.
        thread = threading.Thread(target=run_short)
        thread.start()
        time.sleep(args.delay_s)
        long_send = time.perf_counter()
        long_stamps = stream(
            base, args.model, long_prompt, args.max_tokens, f"m-{nonce}-long-{index}"
        )
        thread.join()
        # The long request's prefill occupies [long_send, first long token]; what
        # a concurrent decode gets served inside that window is the cost of the
        # chunk size.
        lo = long_send
        hi = long_stamps[0] if long_stamps else None
        window = (lo, hi) if hi else None
        run = itl(holder["stamps"], window)
        run["short_ttft_ms"] = (
            (holder["stamps"][-1] - lo) * 1e3 if holder["stamps"] else None
        )
        run["tokens_in_window"] = sum(
            1 for t in holder["stamps"] if window and window[0] <= t <= window[1]
        )
        run["window_s"] = (hi - lo) if window else None
        run["long_ttft_s"] = (hi - lo) if window else None
        loaded.append(run)
        long_ttft.append((hi - long_send) if hi else None)
    row = {
        "quiet": quiet,
        "loaded": loaded,
        "long_ttft_s": long_ttft,
        "long_ttft_median_s": statistics.median(
            [t for t in long_ttft if t is not None]
        ),
        "quiet_start": quiet_start,
    }
    print(
        f"quiet  ITL median {quiet['median_ms']:.1f} ms  max {quiet['max_ms']:.1f} ms"
    )
    for index, run in enumerate(loaded):
        print(
            f"loaded[{index}] during prefill: {run['tokens_in_window']} decodes in "
            f"{run['window_s']:.2f} s, worst gap {run['max_ms']:.1f} ms, "
            f"median {run['median_ms']:.1f} ms"
        )
    windows = [r["window_s"] for r in loaded if r.get("window_s")]
    print(
        f"long prefill: {statistics.median(windows):.2f} s, "
        f"TTFT {row['long_ttft_median_s']:.3f} s"
    )
    loads = [r for r in loaded if r["max_ms"] is not None]
    if loads:
        print(
            f"SUMMARY during prefill: decodes "
            f"{statistics.median([r['tokens_in_window'] for r in loads]):.0f}  "
            f"worst gap {max(r['max_ms'] for r in loads):.1f} ms  "
            f"median gap {statistics.median([r['median_ms'] for r in loads]):.1f} ms  "
            f"| quiet ITL {quiet['median_ms']:.1f} ms"
        )
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "label": args.label,
                    "server_args": args.server_args,
                    "args": vars(args),
                    "nonce": nonce,
                    **row,
                },
                indent=1,
            )
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
