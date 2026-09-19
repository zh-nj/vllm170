#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark decode gate: decode throughput plus exact speculative acceptance.

Runs the same exact-token streaming gate as ``decode_gate.py`` and reads the
``vllm:spec_decode_*`` Prometheus counters immediately before and after, so one
run yields both the wall-clock tokens/s and the acceptance statistics of the
rounds that produced it. Counter deltas are exact, not sampled.
"""

from __future__ import annotations

import argparse
import json
import statistics
import urllib.request

from decode_gate import make_token_ids, run_once

COUNTERS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)
PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"


def scrape(base: str) -> dict[str, float]:
    with urllib.request.urlopen(f"{base}/metrics", timeout=120) as resp:
        text = resp.read().decode()
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("vllm:spec_decode") or line.startswith("#"):
            continue
        series, _, value = line.rpartition(" ")
        name, _, labels = series.partition("{")
        try:
            out[f"{name}|{labels}"] = float(value)
        except ValueError:
            continue
    return out


def delta(before: dict[str, float], after: dict[str, float], name: str) -> float:
    total = 0.0
    for key, value in after.items():
        key_name, _, _ = key.partition("|")
        if key_name == name:
            total += value - before.get(key, 0.0)
    return total


def per_position(before: dict[str, float], after: dict[str, float]) -> list[float]:
    rows: dict[int, float] = {}
    for key, value in after.items():
        key_name, _, labels = key.partition("|")
        if key_name != PER_POS:
            continue
        position = None
        for part in labels.strip("}").split(","):
            field, _, raw = part.partition("=")
            if field.strip() == "position":
                position = int(raw.strip().strip('"'))
        if position is not None:
            rows[position] = value - before.get(key, 0.0)
    return [rows[i] for i in sorted(rows)]


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

    before = scrape(base)
    rows = []
    for i in range(args.repeats):
        row = run_once(base, args.model, prompt_ids, args.max_tokens, args.timeout)
        row["repeat"] = i
        rows.append(row)
        print(
            f"[{i}] ttft={row['ttft_s']:.3f}s {row['ms_per_token']:.3f} ms/token "
            f"{row['tok_per_s']:.3f} tok/s"
        )
    after = scrape(base)

    drafts = delta(before, after, COUNTERS[0])
    draft_tokens = delta(before, after, COUNTERS[1])
    accepted = delta(before, after, COUNTERS[2])
    positions = per_position(before, after)

    summary = {
        "prompt_tokens": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "ms_per_token_mean": statistics.mean(r["ms_per_token"] for r in rows),
        "tok_per_s_mean": statistics.mean(r["tok_per_s"] for r in rows),
        "ms_per_token_min": min(r["ms_per_token"] for r in rows),
        "ms_per_token_max": max(r["ms_per_token"] for r in rows),
        "ttft_s_mean": statistics.mean(r["ttft_s"] for r in rows),
        "drafts": drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / draft_tokens if draft_tokens else 0.0,
        "accepted_length": (accepted + drafts) / drafts if drafts else 0.0,
        "accepted_per_pos": positions,
        "rows": rows,
    }
    print(
        f"MEAN {summary['ms_per_token_mean']:.3f} ms/token "
        f"{summary['tok_per_s_mean']:.3f} tok/s | drafts={drafts:.0f} "
        f"acceptance={summary['acceptance_rate'] * 100:.2f}% "
        f"length={summary['accepted_length']:.3f}"
    )
    print(f"accepted_per_pos={positions}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
