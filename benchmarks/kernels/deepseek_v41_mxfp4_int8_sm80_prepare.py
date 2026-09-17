# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark DSV4.1 single-token MXFP4-int8 MoE preparation."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch

from vllm.model_executor.layers.fused_moe.experts.mxfp4_int8_sm80 import (
    make_mxfp4_int8_sm80_decode_metadata,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_int8_sm80 import (
    quantize_per_token_int8,
)

HIDDEN_SIZE = 5120
NUM_EXPERTS = 384
TOPK = 6
BLOCK_M = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--graph-calls", type=int, default=64)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def measure(
    call: Callable[[], object],
    warmup: int,
    repeats: int,
    calls_per_sample: int = 1,
) -> dict[str, float]:
    latest = None
    for _ in range(warmup):
        latest = call()
    torch.accelerator.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        latest = call()
        end.record()
    torch.accelerator.synchronize()
    assert latest is not None
    samples = [
        start.elapsed_time(end) * 1000 / calls_per_sample
        for start, end in zip(starts, ends, strict=True)
    ]
    return {
        "mean_us": statistics.fmean(samples),
        "p50_us": statistics.median(samples),
        "p90_us": percentile(samples, 0.9),
    }


def capture(
    call: Callable[[], object], warmup: int, calls_per_replay: int
) -> Callable[[], object]:
    latest = None
    for _ in range(warmup):
        latest = call()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        latest = [call() for _ in range(calls_per_replay)]

    def replay() -> object:
        graph.replay()
        return latest

    return replay


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.repeats <= 0 or args.graph_calls <= 0:
        raise ValueError(
            "warmup must be non-negative; repeats and graph-calls must be positive"
        )

    generator = torch.Generator(device="cuda").manual_seed(4242)
    hidden_states = torch.randn(
        (1, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    topk_ids = torch.randperm(
        NUM_EXPERTS,
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )[:TOPK].view(1, TOPK)

    def quant() -> object:
        return quantize_per_token_int8(hidden_states)

    def align() -> object:
        return moe_align_block_size(topk_ids, BLOCK_M, NUM_EXPERTS)

    def baseline() -> object:
        return quant(), align()

    persistent_sorted, persistent_post_pad = make_mxfp4_int8_sm80_decode_metadata(
        hidden_states.device
    )

    def persistent() -> object:
        return (
            quant(),
            persistent_sorted,
            topk_ids.view(-1),
            persistent_post_pad,
        )

    calls = {
        "quant": quant,
        "align": align,
        "quant_then_align": baseline,
        "persistent_metadata": persistent,
    }
    results = {
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "shape": {
            "tokens": 1,
            "hidden_size": HIDDEN_SIZE,
            "num_experts": NUM_EXPERTS,
            "topk": TOPK,
            "block_m": BLOCK_M,
        },
        "eager_event": {
            name: measure(call, args.warmup, args.repeats)
            for name, call in calls.items()
        },
        "graph_event": {
            name: measure(
                capture(call, args.warmup, args.graph_calls),
                args.warmup,
                args.repeats,
                args.graph_calls,
            )
            for name, call in calls.items()
        },
    }
    serialized = json.dumps(results, indent=2, sort_keys=True)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
