# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bound DSV4.1 decode gains from overlapping cache-side attention work."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch

from vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache import (
    fused_save_compress_norm,
    rope_quant_insert,
)


def _measure(
    call: Callable[[], None], warmup: int, repeats: int, calls_per_replay: int
) -> dict[str, float]:
    for _ in range(warmup):
        call()
    torch.accelerator.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        call()
        end.record()
    torch.accelerator.synchronize()
    samples = [
        start.elapsed_time(end) * 1000.0 / calls_per_replay
        for start, end in zip(starts, ends, strict=True)
    ]
    return {
        "mean_us": statistics.fmean(samples),
        "p50_us": statistics.median(samples),
        "p90_us": sorted(samples)[min(repeats - 1, int(repeats * 0.9))],
    }


def _capture(
    call: Callable[[], None], warmup: int, calls_per_replay: int
) -> Callable[[], None]:
    for _ in range(warmup):
        call()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls_per_replay):
            call()
    return graph.replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--graph-calls", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats <= 0 or args.graph_calls <= 0:
        raise ValueError(
            "warmup must be non-negative; repeats and graph-calls must be positive"
        )

    generator = torch.Generator(device="cuda").manual_seed(20260914)
    ratios = (2, 2, 2, 1)
    norm = torch.randn(512, dtype=torch.bfloat16, device="cuda", generator=generator)
    cos_sin = torch.randn(
        (65536, 64), dtype=torch.float32, device="cuda", generator=generator
    )
    positions = torch.tensor([32000], dtype=torch.int64, device="cuda")
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    req_ids = torch.zeros(1, dtype=torch.int32, device="cuda")

    layers = []
    for ratio in ratios:
        kv_score = torch.randn(
            (1, 512 * ratio), dtype=torch.float32, device="cuda", generator=generator
        )
        latent = torch.empty((1, 512), dtype=torch.bfloat16, device="cuda")
        slot = torch.zeros(1, dtype=torch.int64, device="cuda")
        state = (
            torch.randn(
                (1, 8, 1024), dtype=torch.float32, device="cuda", generator=generator
            )
            if ratio == 2
            else None
        )
        cache = torch.empty((1, 128 // ratio, 584), dtype=torch.uint8, device="cuda")
        layers.append((ratio, kv_score, latent, slot, state, cache))

    def compress_all() -> None:
        for ratio, kv_score, latent, slot, state, _ in layers:
            fused_save_compress_norm(
                kv_score,
                positions,
                state,
                slot,
                query_start_loc if state is not None else None,
                req_ids if state is not None else None,
                norm,
                1e-6,
                ratio,
                latent,
            )

    def insert_all() -> None:
        for ratio, _, latent, slot, _, cache in layers:
            rope_quant_insert(latent, positions, cos_sin, cache, slot, ratio)

    def measure_at(position: int) -> dict[str, object]:
        positions.fill_(position)
        compress_graph = _capture(compress_all, args.warmup, args.graph_calls)
        insert_graph = _capture(insert_all, args.warmup, args.graph_calls)
        compress = _measure(compress_graph, args.warmup, args.repeats, args.graph_calls)
        insert = _measure(insert_graph, args.warmup, args.repeats, args.graph_calls)
        return {
            "position": position,
            "compressor_four_sources": compress,
            "cache_insert_four_sources": insert,
            "ideal_overlap_upper_bound_us": compress["p50_us"] + insert["p50_us"],
        }

    results = [measure_at(32000), measure_at(32001)]
    average_bound = statistics.fmean(
        float(row["ideal_overlap_upper_bound_us"]) for row in results
    )
    payload = {
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "kv_source_ratios": ratios,
        "graph_calls": args.graph_calls,
        "positions": results,
        "two_token_average_ideal_overlap_upper_bound_us": average_bound,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
