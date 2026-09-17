# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark a fused M=1 BF16 LM-head/argmax path on SM80."""

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
from vllm.triton_utils import tl, triton


@triton.jit
def _lm_head_local_argmax_kernel(
    hidden_ptr,
    weight_ptr,
    local_values_ptr,
    local_indices_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k = k_start + offsets_k
        hidden = tl.load(hidden_ptr + k, mask=k < K, other=0.0).to(tl.bfloat16)
        weight = tl.load(
            weight_ptr + offsets_n[:, None] * K + k[None, :],
            mask=(offsets_n[:, None] < N) & (k[None, :] < K),
            other=0.0,
        ).to(tl.bfloat16)
        accumulator += tl.sum(weight * hidden[None, :], axis=1)

    # The production projection materializes BF16 logits before the sampler
    # widens them to FP32, so preserve that comparison boundary.
    scores = accumulator.to(tl.bfloat16).to(tl.float32)
    scores = tl.where(offsets_n < N, scores, -float("inf"))
    local_offset = tl.argmax(scores, axis=0)
    tl.store(local_values_ptr + pid, tl.max(scores, axis=0))
    tl.store(local_indices_ptr + pid, pid * BLOCK_N + local_offset)


@triton.jit
def _reduce_local_argmax_kernel(
    local_values_ptr,
    local_indices_ptr,
    output_ptr,
    NUM_BLOCKS: tl.constexpr,
    REDUCE_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, REDUCE_SIZE)
    values = tl.load(
        local_values_ptr + offsets,
        mask=offsets < NUM_BLOCKS,
        other=-float("inf"),
    )
    local_offset = tl.argmax(values, axis=0)
    output = tl.load(local_indices_ptr + local_offset)
    tl.store(output_ptr, output)


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


def _make_candidate(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    block_k: int,
    block_n: int,
    num_warps: int,
) -> Callable[[], object]:
    k = weight.shape[1]
    n = weight.shape[0]
    num_blocks = triton.cdiv(n, block_n)
    reduce_size = triton.next_power_of_2(num_blocks)
    local_values = torch.empty(num_blocks, dtype=torch.float32, device="cuda")
    local_indices = torch.empty(num_blocks, dtype=torch.int32, device="cuda")

    def candidate() -> object:
        _lm_head_local_argmax_kernel[(num_blocks,)](
            hidden,
            weight,
            local_values,
            local_indices,
            K=k,
            N=n,
            BLOCK_K=block_k,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )
        _reduce_local_argmax_kernel[(1,)](
            local_values,
            local_indices,
            output,
            NUM_BLOCKS=num_blocks,
            REDUCE_SIZE=reduce_size,
            num_warps=8,
        )
        return output

    return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--n", type=int, default=129280)
    parser.add_argument("--block-k", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--block-n", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--warps", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--graph-calls", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.k <= 0 or args.n <= 0:
        raise ValueError("k and n must be positive")

    generator = torch.Generator(device="cuda").manual_seed(20260914)
    hidden = torch.randn(
        (1, args.k), dtype=torch.bfloat16, device="cuda", generator=generator
    )
    # Raw values stay below the sole E4M3 NaN code.  With exponent scale 127
    # (2**0), a direct BF16 cast is the exact dequantized reference and avoids
    # a multi-GiB FP32 temporary for this full-vocabulary benchmark.
    weight_mxfp8 = torch.empty(
        (args.n, args.k), dtype=torch.float8_e4m3fn, device="cuda"
    )
    weight_mxfp8.view(torch.uint8).random_(0, 127, generator=generator)
    weight_scale = torch.full(
        (args.n, args.k // 32), 127, dtype=torch.uint8, device="cuda"
    )
    weight = weight_mxfp8.to(torch.bfloat16)
    marlin_layer = torch.nn.Module()
    marlin_layer.input_size_per_partition = args.k
    marlin_layer.output_size_per_partition = args.n
    marlin_layer.weight = torch.nn.Parameter(weight_mxfp8, requires_grad=False)
    marlin_layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    prepare_mxfp8_layer_for_marlin(marlin_layer)
    logits = torch.empty((1, args.n), dtype=torch.bfloat16, device="cuda")
    sampled_long = torch.empty(1, dtype=torch.int64, device="cuda")
    baseline_output = torch.empty(1, dtype=torch.int32, device="cuda")
    candidate_output = torch.empty_like(baseline_output)

    def projection() -> object:
        return torch.mm(hidden, weight.t(), out=logits)

    def argmax() -> object:
        torch.argmax(logits.float(), dim=-1, out=sampled_long)
        baseline_output.copy_(sampled_long)
        return baseline_output

    def baseline() -> object:
        projection()
        return argmax()

    def mxfp8_projection() -> torch.Tensor:
        return apply_mxfp8_marlin_linear(
            input=hidden,
            weight=marlin_layer.weight,
            weight_scale=marlin_layer.weight_scale,
            workspace=marlin_layer.workspace,
            size_n=args.n,
            size_k=args.k,
        )

    def mxfp8_projection_argmax() -> object:
        quantized_logits = mxfp8_projection()
        torch.argmax(quantized_logits.float(), dim=-1, out=sampled_long)
        candidate_output.copy_(sampled_long)
        return candidate_output

    baseline()
    quantized_logits = mxfp8_projection()
    torch.accelerator.synchronize()
    expected = int(baseline_output.item())
    max_abs_error = float((quantized_logits - logits).abs().max().item())
    mean_abs_error = float((quantized_logits - logits).abs().mean().item())
    quantized_top1 = int(quantized_logits.argmax(dim=-1).item())

    calls = {
        "projection": projection,
        "argmax": argmax,
        "projection_argmax": baseline,
        "mxfp8_marlin_projection": mxfp8_projection,
        "mxfp8_marlin_projection_argmax": mxfp8_projection_argmax,
    }
    graph_results = {
        name: _measure(
            _capture(call, args.warmup, args.graph_calls),
            args.warmup,
            args.repeats,
            args.graph_calls,
        )
        for name, call in calls.items()
    }

    candidates = []
    for block_k in args.block_k:
        for block_n in args.block_n:
            for num_warps in args.warps:
                candidate = _make_candidate(
                    hidden,
                    weight,
                    candidate_output,
                    block_k,
                    block_n,
                    num_warps,
                )
                candidate()
                torch.accelerator.synchronize()
                actual = int(candidate_output.item())
                graph = _capture(candidate, args.warmup, args.graph_calls)
                event = _measure(graph, args.warmup, args.repeats, args.graph_calls)
                row = {
                    "block_k": block_k,
                    "block_n": block_n,
                    "num_warps": num_warps,
                    "argmax_matches": actual == expected,
                    "expected": expected,
                    "actual": actual,
                    "graph": event,
                }
                candidates.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

    payload = {
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "shape": {"m": 1, "n": args.n, "k": args.k},
        "mxfp8_correctness": {
            "top1_matches": quantized_top1 == expected,
            "expected": expected,
            "actual": quantized_top1,
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
        },
        "baseline_graph": graph_results,
        "candidates": candidates,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
