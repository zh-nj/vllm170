# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe a two-stage FP32 grouped tail for the DeepSeek-V4 Marlin W2."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
    _marlin_moe_fp32_reduce_workspace_size,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl, triton

HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 2048
NUM_EXPERTS = 256
TOP_K = 6
GROUP_SIZE = 32
MOE_BLOCK_SIZE = 48

TAIL_CONFIGS = (
    (1024, 4),
    (1024, 8),
    (2048, 4),
    (2048, 8),
    (4096, 8),
)


@triton.jit
def _fp32_grouped_tail_kernel(
    routed_ptr,
    output_ptr,
    HIDDEN: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < HIDDEN
    base = token * TOPK * HIDDEN + offsets
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for expert_slot in tl.static_range(0, TOPK):
        acc += tl.load(
            routed_ptr + base + expert_slot * HIDDEN,
            mask=mask,
            other=0.0,
        )
    tl.store(output_ptr + token * HIDDEN + offsets, acc, mask=mask)


def _fp32_grouped_tail(
    routed: torch.Tensor,
    output: torch.Tensor,
    block_n: int,
    num_warps: int,
) -> None:
    m = output.size(0)
    _fp32_grouped_tail_kernel[(m, triton.cdiv(HIDDEN_SIZE, block_n))](
        routed,
        output,
        HIDDEN_SIZE,
        TOP_K,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=1,
    )


def _routing(m: int, seed: int, device: torch.device) -> torch.Tensor:
    if seed < 0:
        tokens = torch.arange(m, device=device, dtype=torch.int32)[:, None]
        offsets = torch.arange(TOP_K, device=device, dtype=torch.int32)[None, :]
        return (tokens * TOP_K + offsets * 43) % NUM_EXPERTS
    generator = torch.Generator(device=device).manual_seed(seed)
    scores = torch.rand((m, NUM_EXPERTS), device=device, generator=generator)
    return scores.topk(TOP_K, dim=1).indices.to(torch.int32)


def _packed_weight(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    qweight = torch.full(
        (NUM_EXPERTS, INTERMEDIATE_SIZE // 16, HIDDEN_SIZE * 2),
        0x11111111,
        dtype=torch.int32,
        device=device,
    )
    scale_bits = torch.full(
        (NUM_EXPERTS, INTERMEDIATE_SIZE // GROUP_SIZE, HIDDEN_SIZE),
        116,
        dtype=torch.uint8,
        device=device,
    )
    return qweight, scale_bits.view(torch.float8_e8m0fnu)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(math.ceil(fraction * len(ordered)) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def _measure(
    fn: Callable[[], Any], warmup: int, iterations: int, samples: int
) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()

    timings_ms: list[float] = []
    for _ in range(samples):
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        torch.accelerator.synchronize()
        timings_ms.append(start.elapsed_time(end) / iterations)
    return (
        statistics.median(timings_ms) * 1000,
        _percentile(timings_ms, 0.9) * 1000,
    )


def _benchmark_shape(
    m: int,
    routing_seed: int,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    device: torch.device,
    warmup: int,
    iterations: int,
    samples: int,
) -> list[dict[str, Any]]:
    topk_ids = _routing(m, routing_seed, device)
    generator = torch.Generator(device=device).manual_seed(routing_seed + 1000)
    topk_weights = torch.softmax(
        torch.rand((m, TOP_K), device=device, generator=generator), dim=1
    )
    sorted_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, MOE_BLOCK_SIZE, NUM_EXPERTS
    )
    activation = torch.randn(
        (m * TOP_K, INTERMEDIATE_SIZE),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    routed_bf16 = torch.empty(
        (m * TOP_K, HIDDEN_SIZE), dtype=torch.bfloat16, device=device
    )
    routed_fp32 = torch.empty(
        (m * TOP_K, HIDDEN_SIZE), dtype=torch.float32, device=device
    )
    dense_baseline = torch.empty((m, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
    dense_candidate = torch.empty_like(dense_baseline)
    marlin_workspace = marlin_make_workspace_new(device, 4)
    fp32_reduce_workspace = torch.empty(
        (
            _marlin_moe_fp32_reduce_workspace_size(
                HIDDEN_SIZE,
                sorted_ids.numel(),
                MOE_BLOCK_SIZE,
                device.index,
            )
        ),
        dtype=torch.float32,
        device=device,
    )

    def run_w2(fp32_output: torch.Tensor | None) -> None:
        ops.moe_wna16_marlin_gemm(
            activation,
            routed_bf16,
            qweight,
            None,
            scales,
            None,
            None,
            None,
            None,
            None,
            marlin_workspace,
            sorted_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            MOE_BLOCK_SIZE,
            1,
            True,
            scalar_types.float4_e2m1f,
            m * TOP_K,
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            True,
            False,
            True,
            False,
            -1,
            -1,
            -1,
            fp32_reduce_workspace,
            fp32_output,
        )

    def run_bf16_tail() -> None:
        ops.moe_sum(routed_bf16.view(m, TOP_K, HIDDEN_SIZE), dense_baseline)

    def run_baseline() -> None:
        run_w2(None)
        run_bf16_tail()

    run_baseline()
    run_w2(routed_fp32)
    torch.accelerator.synchronize()
    routed_diff = (routed_fp32 - routed_bf16.float()).abs()

    w2_bf16_p50_us, w2_bf16_p90_us = _measure(
        partial(run_w2, None), warmup, iterations, samples
    )
    w2_fp32_p50_us, w2_fp32_p90_us = _measure(
        partial(run_w2, routed_fp32), warmup, iterations, samples
    )
    bf16_tail_p50_us, bf16_tail_p90_us = _measure(
        run_bf16_tail, warmup, iterations, samples
    )
    baseline_p50_us, baseline_p90_us = _measure(
        run_baseline, warmup, iterations, samples
    )

    common = {
        "schema_version": 1,
        "timestamp_ns": time.time_ns(),
        "device": current_platform.get_device_name(),
        "compute_capability": current_platform.get_device_capability().as_version_str(),
        "sm_count": current_platform.num_compute_units(),
        "m": m,
        "routing_seed": routing_seed,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_experts": NUM_EXPERTS,
        "top_k": TOP_K,
        "moe_block_size": MOE_BLOCK_SIZE,
        "padded_rows": int(num_tokens_post_padded.item()),
        "warmup": warmup,
        "iterations": iterations,
        "samples": samples,
        "bf16_routed_mib": routed_bf16.numel() * routed_bf16.element_size() / 2**20,
        "fp32_routed_mib": routed_fp32.numel() * routed_fp32.element_size() / 2**20,
        "routed_max_abs_error": float(routed_diff.max().item()),
        "w2_bf16_p50_us": w2_bf16_p50_us,
        "w2_bf16_p90_us": w2_bf16_p90_us,
        "w2_fp32_p50_us": w2_fp32_p50_us,
        "w2_fp32_p90_us": w2_fp32_p90_us,
        "bf16_tail_p50_us": bf16_tail_p50_us,
        "bf16_tail_p90_us": bf16_tail_p90_us,
        "baseline_p50_us": baseline_p50_us,
        "baseline_p90_us": baseline_p90_us,
    }

    results: list[dict[str, Any]] = []
    for block_n, num_warps in TAIL_CONFIGS:
        row = common | {"block_n": block_n, "num_warps": num_warps}

        def run_fp32_tail(block_n: int = block_n, num_warps: int = num_warps) -> None:
            _fp32_grouped_tail(routed_fp32, dense_candidate, block_n, num_warps)

        def run_candidate() -> None:
            run_w2(routed_fp32)
            run_fp32_tail()

        try:
            run_candidate()
            torch.accelerator.synchronize()
            dense_diff = (dense_candidate.float() - dense_baseline.float()).abs()
            tail_p50_us, tail_p90_us = _measure(
                run_fp32_tail, warmup, iterations, samples
            )
            candidate_p50_us, candidate_p90_us = _measure(
                run_candidate, warmup, iterations, samples
            )
            row |= {
                "status": "passed",
                "dense_max_abs_error": float(dense_diff.max().item()),
                "dense_mean_abs_error": float(dense_diff.mean().item()),
                "dense_mismatch_count": int(torch.count_nonzero(dense_diff).item()),
                "fp32_tail_p50_us": tail_p50_us,
                "fp32_tail_p90_us": tail_p90_us,
                "candidate_p50_us": candidate_p50_us,
                "candidate_p90_us": candidate_p90_us,
                "speedup_vs_baseline": baseline_p50_us / candidate_p50_us,
            }
        except RuntimeError as error:
            row |= {"status": "rejected", "error": str(error).splitlines()[0]}
        results.append(row)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", nargs="+", type=int, default=[1666, 1667, 2000, 4048])
    parser.add_argument("--routing-seeds", nargs="+", type=int, default=[-1, 8, 9])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.m) <= 0 or args.warmup < 0 or args.iterations <= 0 or args.samples <= 0:
        raise ValueError("M, iterations, and samples must be positive")
    if not current_platform.is_cuda():
        raise RuntimeError("this benchmark requires CUDA")
    capability = current_platform.get_device_capability()
    if capability is None or capability.to_int() != 80:
        raise RuntimeError(f"expected SM80, got {capability}")

    device = torch.device("cuda", torch.accelerator.current_device_index())
    qweight, scales = _packed_weight(device)
    output = args.output.open("w") if args.output is not None else None
    try:
        for m in args.m:
            for routing_seed in args.routing_seeds:
                rows = _benchmark_shape(
                    m,
                    routing_seed,
                    qweight,
                    scales,
                    device,
                    args.warmup,
                    args.iterations,
                    args.samples,
                )
                for row in rows:
                    line = json.dumps(row, sort_keys=True)
                    print(line, flush=True)
                    if output is not None:
                        output.write(line + "\n")
                        output.flush()
    finally:
        if output is not None:
            output.close()


if __name__ == "__main__":
    main()
