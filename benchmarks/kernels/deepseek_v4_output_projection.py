# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark and tune DeepSeek-V4 wo_b FP8 Marlin on SM8x."""

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

MARLIN_CONFIGS = {
    "auto": (-1, -1, -1),
    "k128_n128_t256": (128, 128, 256),
    "k64_n256_t256": (64, 256, 256),
    "k64_n128_t128": (64, 128, 128),
    "k128_n64_t128": (128, 64, 128),
}


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
    parser.add_argument(
        "--configs", nargs="+", choices=MARLIN_CONFIGS, default=["auto"]
    )
    parser.add_argument(
        "--sms",
        type=int,
        nargs="+",
        default=[-1],
        help="SM counts to scan; -1 uses every visible SM.",
    )
    parser.add_argument(
        "--reduce", nargs="+", choices=("fp32", "bf16"), default=["fp32"]
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device_sms = torch.cuda.get_device_properties(0).multi_processor_count
    invalid_sms = [sms for sms in args.sms if sms != -1 and not 1 <= sms <= device_sms]
    if invalid_sms:
        parser.error(f"--sms must be -1 or in [1, {device_sms}]: {invalid_sms}")

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

        def marlin(
            config: tuple[int, int, int],
            sms: int,
            use_fp32_reduce: bool,
            a: torch.Tensor = a,
            size_m: int = size_m,
        ) -> torch.Tensor:
            thread_k, thread_n, num_threads = config
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
                use_fp32_reduce=use_fp32_reduce,
                thread_k=thread_k,
                thread_n=thread_n,
                num_threads=num_threads,
                sms=sms,
            )

        def bf16(a: torch.Tensor = a) -> torch.Tensor:
            return torch.mm(a, weight_bf16)

        bf16_out = bf16()
        bf16_us, bf16_p90_us = _measure(
            bf16, args.warmup, args.iterations, args.samples
        )
        for config_name in args.configs:
            config = MARLIN_CONFIGS[config_name]
            for sms in args.sms:
                for reduce_dtype in args.reduce:
                    candidate = lambda config=config, sms=sms, reduce_dtype=(
                        reduce_dtype
                    ): marlin(config, sms, reduce_dtype == "fp32")
                    try:
                        marlin_out = candidate()
                    except RuntimeError as error:
                        if "Unsupported shapes" not in str(error):
                            raise
                        row = {
                            "device": current_platform.get_device_name(),
                            "m": size_m,
                            "k": args.k,
                            "n": args.n,
                            "group_size": args.group_size,
                            "config": config_name,
                            "sms": sms,
                            "reduce_dtype": reduce_dtype,
                            "status": "unsupported",
                        }
                        rows.append(row)
                        print(json.dumps(row, sort_keys=True), flush=True)
                        continue
                    torch.accelerator.synchronize()
                    torch.testing.assert_close(
                        marlin_out, bf16_out, atol=1e-1, rtol=2e-2
                    )
                    abs_error = (marlin_out.float() - bf16_out.float()).abs()
                    marlin_us, marlin_p90_us = _measure(
                        candidate, args.warmup, args.iterations, args.samples
                    )
                    row = {
                        "device": current_platform.get_device_name(),
                        "m": size_m,
                        "k": args.k,
                        "n": args.n,
                        "group_size": args.group_size,
                        "config": config_name,
                        "sms": sms,
                        "reduce_dtype": reduce_dtype,
                        "marlin_us": marlin_us,
                        "marlin_p90_us": marlin_p90_us,
                        "bf16_us": bf16_us,
                        "bf16_p90_us": bf16_p90_us,
                        "speedup": marlin_us / bf16_us,
                        "max_abs_error": abs_error.max().item(),
                        "mean_abs_error": abs_error.mean().item(),
                        "bf16_weight_bytes": (
                            weight_bf16.numel() * weight_bf16.element_size()
                        ),
                        "correct": True,
                        "status": "ok",
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
