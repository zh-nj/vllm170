# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure separate versus merged DSV4.1 M=1 MXFP8 Q projections."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch

from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_mxfp8_marlin_linear,
    prepare_mxfp8_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
)

TOKENS = 1
Q_LORA_RANK = 1280
MAIN_Q_SIZE = 64 * 512
INDEX_Q_SIZE = 32 * 128
GROUP_SIZE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--graph-calls", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_layer(weight: torch.Tensor, scale: torch.Tensor) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.input_size_per_partition = weight.shape[1]
    layer.output_size_per_partition = weight.shape[0]
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    prepare_mxfp8_layer_for_marlin(layer)
    return layer


def apply(layer: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    return apply_mxfp8_marlin_linear(
        input=inputs,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        workspace=layer.workspace,
        size_n=layer.output_size_per_partition,
        size_k=layer.input_size_per_partition,
    )


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

    generator = torch.Generator(device="cuda").manual_seed(20260913)
    weight = (
        torch.randn(
            (MAIN_Q_SIZE + INDEX_Q_SIZE, Q_LORA_RANK),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        / 4
    ).to(torch.float8_e4m3fn)
    scale = torch.randint(
        121,
        128,
        (MAIN_Q_SIZE + INDEX_Q_SIZE, Q_LORA_RANK // GROUP_SIZE),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    main_layer = make_layer(weight[:MAIN_Q_SIZE].clone(), scale[:MAIN_Q_SIZE].clone())
    index_layer = make_layer(weight[MAIN_Q_SIZE:].clone(), scale[MAIN_Q_SIZE:].clone())
    merged_layer = make_layer(weight, scale)
    weight_bf16 = dequant_mxfp8_to_bf16(weight, scale)
    main_weight_bf16 = weight_bf16[:MAIN_Q_SIZE]
    index_weight_bf16 = weight_bf16[MAIN_Q_SIZE:]
    inputs = torch.randn(
        (TOKENS, Q_LORA_RANK),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    def main_q() -> object:
        return apply(main_layer, inputs)

    def index_q() -> object:
        return apply(index_layer, inputs)

    def separate() -> object:
        return main_q(), index_q()

    def merged() -> object:
        return apply(merged_layer, inputs)

    def main_q_bf16() -> object:
        return torch.nn.functional.linear(inputs, main_weight_bf16)

    def index_q_bf16() -> object:
        return torch.nn.functional.linear(inputs, index_weight_bf16)

    def merged_bf16() -> object:
        return torch.nn.functional.linear(inputs, weight_bf16)

    separate_out = separate()
    merged_out = merged()
    torch.accelerator.synchronize()
    torch.testing.assert_close(
        merged_out[:, :MAIN_Q_SIZE], separate_out[0], rtol=0, atol=0
    )
    torch.testing.assert_close(
        merged_out[:, MAIN_Q_SIZE:], separate_out[1], rtol=0, atol=0
    )

    calls = {
        "main_q": main_q,
        "index_q": index_q,
        "separate_main_index": separate,
        "merged_main_index": merged,
        "main_q_bf16": main_q_bf16,
        "index_q_bf16": index_q_bf16,
        "merged_bf16": merged_bf16,
    }
    results = {
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "shape": {
            "tokens": TOKENS,
            "k": Q_LORA_RANK,
            "main_n": MAIN_Q_SIZE,
            "index_n": INDEX_Q_SIZE,
            "merged_n": MAIN_Q_SIZE + INDEX_Q_SIZE,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
