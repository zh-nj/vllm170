# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark Qwen4Exp FP8 Marlin MoE decode block sizes on SM80."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from flashinfer.testing import bench_gpu_time_with_cudagraph

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
    _marlin_moe_fp32_reduce_workspace_size,
    _marlin_moe_max_sorted_tokens,
    fused_marlin_moe,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    marlin_quant_fp8_torch,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

HIDDEN_SIZE = 2560
INTERMEDIATE_SIZE = 640
NUM_EXPERTS = 512
TOP_K = 10
GROUP_SIZE = 128


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(math.ceil(fraction * len(ordered)) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def _replicated_fp8_weight(
    size_n: int,
    size_k: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight = (
        torch.randn(
            (size_n, size_k),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        / 10
    )
    reference_transposed, packed, scales = marlin_quant_fp8_torch(
        weight,
        GROUP_SIZE,
    )
    reference = reference_transposed.T.contiguous()
    packed = packed.unsqueeze(0).expand(NUM_EXPERTS, -1, -1).contiguous()
    scales = scales.unsqueeze(0).expand(NUM_EXPERTS, -1, -1).contiguous()
    return reference, packed, scales


def _make_routing(m: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(m, dtype=torch.int32, device="cuda")[:, None]
    offsets = torch.arange(TOP_K, dtype=torch.int32, device="cuda")[None, :]
    topk_ids = (rows * 67 + offsets * 29 + seed * 101) % NUM_EXPERTS
    topk_weights = torch.linspace(
        0.02,
        0.18,
        m * TOP_K,
        dtype=torch.float32,
        device="cuda",
    ).reshape(m, TOP_K)
    return topk_ids.contiguous(), topk_weights


def _reference(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    gate_up = F.linear(hidden_states, w13)
    gate, up = gate_up.chunk(2, dim=-1)
    one_expert = F.linear(F.silu(gate) * up, w2)
    return (one_expert * topk_weights.sum(dim=-1, keepdim=True)).to(hidden_states.dtype)


def _benchmark_case(
    m: int,
    decode_block_size: int,
    hidden_states: torch.Tensor,
    w13_packed: torch.Tensor,
    w2_packed: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    reference: torch.Tensor,
    repeat_time_ms: int,
    round_index: int,
) -> dict[str, float | int | str]:
    max_sorted_tokens = _marlin_moe_max_sorted_tokens(
        m,
        TOP_K,
        NUM_EXPERTS,
        decode_block_size,
    )
    cache13 = torch.empty(
        (m * TOP_K * max(2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),),
        dtype=torch.bfloat16,
        device="cuda",
    )
    cache2 = torch.empty(
        (m * TOP_K, INTERMEDIATE_SIZE),
        dtype=torch.bfloat16,
        device="cuda",
    )
    fp32_workspace = torch.empty(
        _marlin_moe_fp32_reduce_workspace_size(
            HIDDEN_SIZE,
            max_sorted_tokens,
            decode_block_size,
        ),
        dtype=torch.float32,
        device="cuda",
    )
    marlin_workspace = marlin_make_workspace_new(torch.device("cuda"), 4)
    output = torch.empty_like(hidden_states)

    def run(input_tensor: torch.Tensor) -> torch.Tensor:
        return fused_marlin_moe(
            input_tensor,
            w13_packed,
            w2_packed,
            None,
            None,
            w13_scales,
            w2_scales,
            topk_weights,
            topk_ids,
            scalar_types.float8_e4m3fn.id,
            workspace=marlin_workspace,
            intermediate_cache13=cache13,
            intermediate_cache2=cache2,
            fp32_reduce_workspace=fp32_workspace,
            output=output,
            activation=MoEActivation.SILU,
            decode_block_size=decode_block_size,
        )

    actual = run(hidden_states).clone()
    torch.accelerator.synchronize()
    torch.testing.assert_close(actual, reference, atol=5e-2, rtol=2e-2)
    error = (actual.float() - reference.float()).abs()

    timings_ms = bench_gpu_time_with_cudagraph(
        run,
        repeat_time_ms=repeat_time_ms,
        cold_l2_cache=True,
        input_args=(hidden_states,),
    )
    return {
        "status": "passed",
        "m": m,
        "round": round_index,
        "block_size": decode_block_size,
        "padded_rows": max_sorted_tokens,
        "latency_p50_us": statistics.median(timings_ms) * 1e3,
        "latency_p90_us": _percentile(timings_ms, 0.9) * 1e3,
        "max_abs_error": float(error.max().item()),
        "mean_abs_error": float(error.mean().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", nargs="+", type=int, default=[1])
    parser.add_argument("--block-sizes", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeat-time-ms", type=int, default=300)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.accelerator.is_available():
        raise RuntimeError("CUDA is required")
    if not current_platform.is_device_capability((8, 0)):
        raise RuntimeError("This benchmark targets SM80")

    torch.set_default_device("cuda")
    torch.manual_seed(7)
    generator = torch.Generator(device="cuda").manual_seed(7)
    w13_ref, w13_packed, w13_scales = _replicated_fp8_weight(
        2 * INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
        generator,
    )
    w2_ref, w2_packed, w2_scales = _replicated_fp8_weight(
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        generator,
    )

    rows: list[dict[str, float | int | str]] = []
    for round_index in range(args.rounds):
        block_sizes = (
            list(reversed(args.block_sizes)) if round_index % 2 else args.block_sizes
        )
        for m in args.m:
            hidden_states = (
                torch.randn(
                    (m, HIDDEN_SIZE),
                    dtype=torch.bfloat16,
                    device="cuda",
                    generator=generator,
                )
                / 10
            )
            topk_ids, topk_weights = _make_routing(m, round_index)
            reference = _reference(hidden_states, w13_ref, w2_ref, topk_weights)
            for block_size in block_sizes:
                rows.append(
                    _benchmark_case(
                        m,
                        block_size,
                        hidden_states,
                        w13_packed,
                        w2_packed,
                        w13_scales,
                        w2_scales,
                        topk_ids,
                        topk_weights,
                        reference,
                        args.repeat_time_ms,
                        round_index,
                    )
                )

    baseline = {
        (int(row["round"]), int(row["m"])): float(row["latency_p50_us"])
        for row in rows
        if int(row["block_size"]) == args.block_sizes[0]
    }
    for row in rows:
        baseline_key = (int(row["round"]), int(row["m"]))
        row["speedup_vs_first"] = baseline[baseline_key] / float(row["latency_p50_us"])

    summaries = []
    for m in args.m:
        for block_size in args.block_sizes:
            matching = [
                row
                for row in rows
                if int(row["m"]) == m and int(row["block_size"]) == block_size
            ]
            summaries.append(
                {
                    "m": m,
                    "block_size": block_size,
                    "rounds": len(matching),
                    "median_of_p50_us": statistics.median(
                        float(row["latency_p50_us"]) for row in matching
                    ),
                    "median_speedup_vs_first": statistics.median(
                        float(row["speedup_vs_first"]) for row in matching
                    ),
                }
            )

    result = {
        "schema_version": 1,
        "timestamp_ns": time.time_ns(),
        "device": current_platform.get_device_name(),
        "compute_capability": (
            current_platform.get_device_capability().as_version_str()
        ),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "shape": {
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "num_experts": NUM_EXPERTS,
            "top_k": TOP_K,
            "group_size": GROUP_SIZE,
            "weight_dtype": "fp8_e4m3fn",
            "activation_dtype": "bfloat16",
        },
        "timing": "CUDA Graph/Event fallback with cold-L2 input rotation",
        "summary": summaries,
        "results": rows,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
