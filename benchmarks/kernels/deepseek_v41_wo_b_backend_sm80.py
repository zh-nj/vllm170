# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the production-shape MXFP8 Marlin and BF16 ``wo_b`` paths."""

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


def _measure(
    call: Callable[[], object],
    warmup: int,
    repeats: int,
    calls_per_sample: int = 1,
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
        start.elapsed_time(end) * 1000.0 / calls_per_sample
        for start, end in zip(starts, ends, strict=True)
    ]
    return {
        "mean_us": statistics.fmean(samples),
        "p50_us": statistics.median(samples),
        "p90_us": sorted(samples)[min(repeats - 1, int(repeats * 0.9))],
    }


def _capture(
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="+", default=[1])
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--n", type=int, default=5120)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--graph-calls", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    generator = torch.Generator(device="cuda").manual_seed(20260914)
    packed_weight = (
        torch.randn(
            (args.n, args.k),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        / 8
    ).to(torch.float8_e4m3fn)
    scale = torch.full((args.n, args.k // 32), 127, dtype=torch.uint8, device="cuda")
    bf16_weight = dequant_mxfp8_to_bf16(packed_weight, scale)

    layer = torch.nn.Module()
    layer.input_size_per_partition = args.k
    layer.output_size_per_partition = args.n
    layer.weight = torch.nn.Parameter(packed_weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    prepare_mxfp8_layer_for_marlin(layer)

    results = []
    for m in args.m:
        inputs = torch.randn((m, args.k), dtype=torch.bfloat16, device="cuda", generator=generator)
        bf16_output = torch.empty((m, args.n), dtype=torch.bfloat16, device="cuda")

        def bf16() -> object:
            return torch.mm(inputs, bf16_weight.t(), out=bf16_output)

        def marlin() -> object:
            return apply_mxfp8_marlin_linear(
                input=inputs,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                workspace=layer.workspace,
                size_n=args.n,
                size_k=args.k,
            )

        reference = bf16()
        candidate = marlin()
        torch.accelerator.synchronize()
        torch.testing.assert_close(candidate, reference, rtol=2e-2, atol=1.0)
        results.append((m, bf16, marlin))
    calls = {}
    for m, bf16, marlin in results:
        calls[f"bf16_m{m}"] = bf16
        calls[f"marlin_mxp8_m{m}"] = marlin
    result = {
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "shape": {"ms": args.m, "n": args.n, "k": args.k},
        "graph": {
            name: _measure(
                _capture(call, args.warmup, args.graph_calls),
                args.warmup,
                args.repeats,
                args.graph_calls,
            )
            for name, call in calls.items()
        },
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
