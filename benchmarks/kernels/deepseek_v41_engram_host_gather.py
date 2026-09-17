# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure CPU-side Engram raw-row gather alternatives.

The production full-RAM path gathers 24 rows per decode token from two
independent tensors (FP8 bytes and E8M0 scales).  This benchmark keeps the
same row width and random-access pattern while comparing index dtypes and a
single interleaved source tensor.  It is a CPU wall-clock diagnostic; the
production gate remains the no-probe PP5 endpoint benchmark.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import statistics
import time
from pathlib import Path

import torch


def _madvise(tensor: torch.Tensor, advice: int) -> None:
    page_size = os.sysconf("SC_PAGE_SIZE")
    address = tensor.data_ptr()
    start = address - address % page_size
    length = tensor.nbytes + address - start
    length = (length + page_size - 1) // page_size * page_size
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(length), advice):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _measure(call, warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        call()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "mean_ms": statistics.fmean(samples),
        "p50_ms": statistics.median(samples),
        "p90_ms": sorted(samples)[min(repeats - 1, int(repeats * 0.9))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1 << 20)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--huge-pages", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows, args.tokens, args.heads, args.dim, args.block_size) <= 0:
        raise ValueError("all sizes must be positive")
    if args.dim % args.block_size:
        raise ValueError("dim must be divisible by block-size")

    torch.manual_seed(20260914)
    random.seed(20260914)
    weight = torch.empty((args.rows, args.dim), dtype=torch.uint8)
    scales = torch.empty((args.rows, args.dim // args.block_size), dtype=torch.uint8)
    packed = torch.empty(
        (args.rows, args.dim + args.dim // args.block_size), dtype=torch.uint8
    )
    if args.huge_pages:
        for tensor in (weight, scales, packed):
            _madvise(tensor, 14)  # MADV_HUGEPAGE
    weight.random_(0, 256)
    scales.random_(0, 256)
    packed[:, : args.dim].copy_(weight)
    packed[:, args.dim :].copy_(scales)

    # Use a changing random row set on every call.  This avoids measuring only
    # the 24-row L3 footprint while preserving the production [T, H] layout.
    ids_i64 = [
        torch.randint(args.rows, (args.tokens, args.heads), dtype=torch.int64)
        for _ in range(args.warmup + args.repeats)
    ]
    ids_i32 = [item.to(torch.int32) for item in ids_i64]
    cursor = 0
    weight_out = torch.empty((args.tokens * args.heads, args.dim), dtype=torch.uint8)
    scale_out = torch.empty(
        (args.tokens * args.heads, args.dim // args.block_size), dtype=torch.uint8
    )
    packed_out = torch.empty(
        (args.tokens * args.heads, args.dim + args.dim // args.block_size),
        dtype=torch.uint8,
    )

    def next_ids(dtype: torch.dtype) -> torch.Tensor:
        nonlocal cursor
        value = ids_i64[cursor] if dtype == torch.int64 else ids_i32[cursor]
        cursor = (cursor + 1) % len(ids_i64)
        return value.reshape(-1)

    def two_i64() -> None:
        local = next_ids(torch.int64)
        torch.index_select(weight, 0, local, out=weight_out)
        torch.index_select(scales, 0, local, out=scale_out)

    def two_i32() -> None:
        local = next_ids(torch.int32)
        torch.index_select(weight, 0, local, out=weight_out)
        torch.index_select(scales, 0, local, out=scale_out)

    def packed_i32() -> None:
        local = next_ids(torch.int32)
        torch.index_select(packed, 0, local, out=packed_out)

    calls = {
        "two_index_select_i64": two_i64,
        "two_index_select_i32": two_i32,
        "one_index_select_interleaved": packed_i32,
    }
    results = {
        name: _measure(call, args.warmup, args.repeats) for name, call in calls.items()
    }
    payload = {
        "rows": args.rows,
        "tokens": args.tokens,
        "heads": args.heads,
        "dim": args.dim,
        "block_size": args.block_size,
        "bytes_allocated": weight.nbytes + scales.nbytes + packed.nbytes,
        "huge_pages": args.huge_pages,
        "results": results,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
