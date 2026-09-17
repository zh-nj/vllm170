# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure decode-sized Engram index transfer from GPU to CPU."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def _measure(call, warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        call()
        samples.append((time.perf_counter_ns() - start) / 1000.0)
    return {
        "mean_us": statistics.fmean(samples),
        "p50_us": statistics.median(samples),
        "p90_us": sorted(samples)[min(repeats - 1, int(repeats * 0.9))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--vocab", type=int, default=384_006_168)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.tokens, args.heads, args.vocab, args.repeats) <= 0:
        raise ValueError("sizes and repeats must be positive")

    device_ids = torch.randint(
        args.vocab,
        (args.tokens, args.heads),
        dtype=torch.int32,
        device="cuda",
    )
    host_i32 = torch.empty_like(device_ids, device="cpu", pin_memory=True)
    stream = torch.cuda.current_stream()

    def current() -> None:
        ids = device_ids.to(device="cpu", dtype=torch.int64)
        owned = (ids >= 0) & (ids < args.vocab)
        (ids - 0).masked_fill_(~owned, 0)

    def pinned_i32() -> None:
        host_i32.copy_(device_ids, non_blocking=True)
        stream.synchronize()

    results = {
        "current_to_i64_validate": _measure(current, args.warmup, args.repeats),
        "pinned_i32_tp1": _measure(pinned_i32, args.warmup, args.repeats),
    }
    payload = {
        "device": torch.cuda.get_device_name(),
        "tokens": args.tokens,
        "heads": args.heads,
        "results": results,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
