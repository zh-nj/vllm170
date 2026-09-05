# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep Marlin MXFP4 execution configs for DeepSeek-V4/GLM MoE shapes."""

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

HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 2048
NUM_EXPERTS = 256
TOP_K = 6
GROUP_SIZE = 32
MOE_BLOCK_SIZE = 48

EXEC_CONFIGS = (
    ("auto", -1, -1, -1),
    ("k64_n256_b1", 64, 256, 1),
    ("k64_n256_b2", 64, 256, 2),
    ("k64_n128_b1", 64, 128, 1),
    ("k64_n128_b2", 64, 128, 2),
    ("k128_n64_b1", 128, 64, 1),
    ("k128_n64_b2", 128, 64, 2),
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
    size_k: int, size_n: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    qweight = torch.full(
        (NUM_EXPERTS, size_k // 16, size_n * 2),
        0x11111111,
        dtype=torch.int32,
        device=device,
    )
    scale_bits = torch.full(
        (NUM_EXPERTS, size_k // GROUP_SIZE, size_n),
        116,
        dtype=torch.uint8,
        device=device,
    )
    return qweight, scale_bits.view(torch.float8_e8m0fnu)


def _stage_shape(stage: str, m: int) -> tuple[int, int, int, int]:
    if stage == "w13":
        return m, HIDDEN_SIZE, 2 * INTERMEDIATE_SIZE, TOP_K
    if stage == "w2":
        return m * TOP_K, INTERMEDIATE_SIZE, HIDDEN_SIZE, 1
    raise ValueError(f"unknown stage: {stage}")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(math.ceil(fraction * len(ordered)) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def _measure(
    fn: Callable[[], torch.Tensor], warmup: int, iterations: int, samples: int
) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()

    timings_ms: list[float] = []
    torch.accelerator.reset_peak_memory_stats()
    allocated_before = torch.accelerator.memory_allocated()
    for _ in range(samples):
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        torch.accelerator.synchronize()
        timings_ms.append(start.elapsed_time(end) / iterations)
    peak_temp_mib = (
        torch.accelerator.max_memory_allocated() - allocated_before
    ) / 2**20
    return (
        statistics.median(timings_ms) * 1000,
        _percentile(timings_ms, 0.9) * 1000,
        peak_temp_mib,
    )


def _benchmark_shape(
    stage: str,
    m: int,
    routing_seed: int,
    moe_block_size: int,
    device: torch.device,
    warmup: int,
    iterations: int,
    samples: int,
    correctness_atol: float,
    fp32_reduce_workspace_mode: str,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    exec_configs: tuple[tuple[str, int, int, int], ...],
) -> list[dict[str, Any]]:
    size_m, size_k, size_n, kernel_top_k = _stage_shape(stage, m)
    topk_ids = _routing(m, routing_seed, device)
    generator = torch.Generator(device=device).manual_seed(routing_seed + 1000)
    topk_weights = torch.rand(
        (m, TOP_K), dtype=torch.float32, device=device, generator=generator
    )
    sorted_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, moe_block_size, NUM_EXPERTS
    )
    input = torch.randn(
        (size_m, size_k), dtype=torch.bfloat16, device=device, generator=generator
    )
    output = torch.empty(
        (size_m * kernel_top_k, size_n), dtype=torch.bfloat16, device=device
    )
    workspace = marlin_make_workspace_new(device, 4)
    fp32_reduce_workspace = None
    if fp32_reduce_workspace_mode == "caller":
        fp32_reduce_workspace = torch.empty(
            (
                _marlin_moe_fp32_reduce_workspace_size(
                    size_n,
                    sorted_ids.numel(),
                    moe_block_size,
                    device.index,
                )
            ),
            dtype=torch.float32,
            device=device,
        )

    def run(thread_k: int, thread_n: int, blocks_per_sm: int) -> torch.Tensor:
        return ops.moe_wna16_marlin_gemm(
            input,
            output,
            qweight,
            None,
            scales,
            None,
            None,
            None,
            None,
            None,
            workspace,
            sorted_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            moe_block_size,
            kernel_top_k,
            stage == "w2",
            scalar_types.float4_e2m1f,
            size_m,
            size_n,
            size_k,
            True,
            False,
            True,
            False,
            thread_k,
            thread_n,
            blocks_per_sm,
            fp32_reduce_workspace,
        )

    run(-1, -1, -1)
    torch.accelerator.synchronize()
    reference = output.clone()
    padded_rows = int(num_tokens_post_padded.item())
    useful_rows = m * TOP_K
    common = {
        "schema_version": 1,
        "timestamp_ns": time.time_ns(),
        "device": current_platform.get_device_name(),
        "compute_capability": current_platform.get_device_capability().as_version_str(),
        "sm_count": current_platform.num_compute_units(),
        "stage": stage,
        "m": m,
        "size_m": size_m,
        "size_n": size_n,
        "size_k": size_k,
        "num_experts": NUM_EXPERTS,
        "top_k": TOP_K,
        "moe_block_size": moe_block_size,
        "routing_seed": routing_seed,
        "padded_rows": padded_rows,
        "useful_row_pct": 100 * useful_rows / padded_rows,
        "warmup": warmup,
        "iterations": iterations,
        "samples": samples,
        "fp32_reduce_workspace": fp32_reduce_workspace_mode,
    }
    results: list[dict[str, Any]] = []
    auto_latency_us = 0.0
    for name, thread_k, thread_n, blocks_per_sm in exec_configs:
        row = common | {
            "config": name,
            "thread_k": thread_k,
            "thread_n": thread_n,
            "blocks_per_sm": blocks_per_sm,
        }
        try:
            run(thread_k, thread_n, blocks_per_sm)
            torch.accelerator.synchronize()
            diff = (output.float() - reference.float()).abs()
            max_abs_error = float(diff.max().item())
            mean_abs_error = float(diff.mean().item())
            if not math.isfinite(max_abs_error) or max_abs_error > correctness_atol:
                raise RuntimeError(
                    f"max_abs_error={max_abs_error} exceeds {correctness_atol}"
                )
            latency_p50_us, latency_p90_us, peak_temp_mib = _measure(
                partial(run, thread_k, thread_n, blocks_per_sm),
                warmup,
                iterations,
                samples,
            )
            if name == "auto":
                auto_latency_us = latency_p50_us
            row |= {
                "status": "passed",
                "max_abs_error": max_abs_error,
                "mean_abs_error": mean_abs_error,
                "latency_p50_us": latency_p50_us,
                "latency_p90_us": latency_p90_us,
                "peak_temp_mib": peak_temp_mib,
            }
        except RuntimeError as error:
            row |= {"status": "rejected", "error": str(error).splitlines()[0]}
        results.append(row)

    for row in results:
        if row["status"] == "passed":
            row["speedup_vs_auto"] = auto_latency_us / row["latency_p50_us"]
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("w13", "w2", "both"), default="both")
    parser.add_argument("--m", nargs="+", type=int, default=[1000, 1666, 1667, 2000])
    parser.add_argument("--routing-seeds", nargs="+", type=int, default=[-1, 8, 9])
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--intermediate-size", type=int, default=INTERMEDIATE_SIZE)
    parser.add_argument("--num-experts", type=int, default=NUM_EXPERTS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=tuple(config[0] for config in EXEC_CONFIGS),
        default=[config[0] for config in EXEC_CONFIGS],
    )
    parser.add_argument(
        "--block-sizes",
        nargs="+",
        type=int,
        default=[MOE_BLOCK_SIZE],
        choices=(8, 16, 32, 48, 64),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--correctness-atol", type=float, default=1e-5)
    parser.add_argument(
        "--fp32-reduce-workspace",
        choices=("caller", "internal"),
        default="caller",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_EXPERTS, TOP_K
    HIDDEN_SIZE = args.hidden_size
    INTERMEDIATE_SIZE = args.intermediate_size
    NUM_EXPERTS = args.num_experts
    TOP_K = args.top_k
    if min(args.m) <= 0 or args.warmup < 0 or args.iterations <= 0 or args.samples <= 0:
        raise ValueError("M, iterations, and samples must be positive")
    if not current_platform.is_cuda():
        raise RuntimeError("this benchmark requires CUDA")
    capability = current_platform.get_device_capability()
    if capability is None or capability.to_int() != 80:
        raise RuntimeError(f"expected SM80, got {capability}")

    device = torch.device("cuda", torch.accelerator.current_device_index())
    stages = ("w13", "w2") if args.stage == "both" else (args.stage,)
    exec_configs = tuple(
        config for config in EXEC_CONFIGS if config[0] in args.configs
    )
    output = args.output.open("w") if args.output is not None else None
    try:
        for stage in stages:
            _, size_k, size_n, _ = _stage_shape(stage, 1)
            qweight, scales = _packed_weight(size_k, size_n, device)
            for moe_block_size in args.block_sizes:
                for m in args.m:
                    for routing_seed in args.routing_seeds:
                        rows = _benchmark_shape(
                            stage,
                            m,
                            routing_seed,
                            moe_block_size,
                            device,
                            args.warmup,
                            args.iterations,
                            args.samples,
                            args.correctness_atol,
                            args.fp32_reduce_workspace,
                            qweight,
                            scales,
                            exec_configs,
                        )
                        for row in rows:
                            line = json.dumps(row, sort_keys=True)
                            print(line, flush=True)
                            if output is not None:
                                output.write(line + "\n")
                                output.flush()
            del qweight, scales
            torch.accelerator.empty_cache()
    finally:
        if output is not None:
            output.close()


if __name__ == "__main__":
    main()
