# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark DeepSeek-V4 wo_b FP8 Marlin against cached BF16 on SM8x."""

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    marlin_quant_fp8_torch,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

KernelFn = Callable[[], torch.Tensor]


def _measure(
    fn: KernelFn, warmup: int, iterations: int, samples: int
) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    timings = []
    for _ in range(samples):
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        torch.accelerator.synchronize()
        timings.append(start.elapsed_time(end) * 1000.0 / iterations)
    return statistics.median(timings), sorted(timings)[int(0.9 * (samples - 1))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, nargs="+", default=[1000, 1666, 1667, 2000])
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    weight = (
        torch.randn((args.n, args.k), device="cuda", generator=generator) / args.k**0.5
    ).to(torch.bfloat16)
    weight_bf16, weight_marlin, scales = marlin_quant_fp8_torch(weight, args.group_size)
    del weight
    workspace = marlin_make_workspace_new(weight_marlin.device)

    rows = []
    for size_m in args.m:
        a = torch.randn(
            (size_m, args.k),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )

        def marlin(a: torch.Tensor = a, size_m: int = size_m) -> torch.Tensor:
            return ops.marlin_gemm(
                a=a,
                c=None,
                b_q_weight=weight_marlin,
                b_bias=None,
                b_scales=scales,
                a_scales=None,
                global_scale=None,
                b_zeros=None,
                g_idx=None,
                perm=None,
                workspace=workspace,
                b_q_type=scalar_types.float8_e4m3fn,
                size_m=size_m,
                size_n=args.n,
                size_k=args.k,
                use_atomic_add=False,
                use_fp32_reduce=True,
            )

        def bf16(a: torch.Tensor = a) -> torch.Tensor:
            return torch.mm(a, weight_bf16)

        marlin_out = marlin()
        bf16_out = bf16()
        torch.accelerator.synchronize()
        torch.testing.assert_close(marlin_out, bf16_out, atol=1e-1, rtol=2e-2)

        marlin_us, marlin_p90_us = _measure(
            marlin, args.warmup, args.iterations, args.samples
        )
        bf16_us, bf16_p90_us = _measure(
            bf16, args.warmup, args.iterations, args.samples
        )
        row = {
            "device": current_platform.get_device_name(),
            "m": size_m,
            "k": args.k,
            "n": args.n,
            "group_size": args.group_size,
            "marlin_us": marlin_us,
            "marlin_p90_us": marlin_p90_us,
            "bf16_us": bf16_us,
            "bf16_p90_us": bf16_p90_us,
            "speedup": marlin_us / bf16_us,
            "bf16_weight_bytes": weight_bf16.numel() * weight_bf16.element_size(),
            "correct": True,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
