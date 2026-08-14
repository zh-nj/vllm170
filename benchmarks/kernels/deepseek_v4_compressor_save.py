# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tune DeepSeek-V4 compressor state-save launches on SM8x."""

import argparse
import json
import statistics
from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch

from vllm.models.deepseek_v4.common.ops.save_partial_states import (
    _save_partial_states_kernel,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton

KernelFn = Callable[[], None]


def _launch(
    num_warps: int,
    *,
    outputs: dict[int, torch.Tensor],
    size_m: int,
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    head_size: int,
    compress_ratio: int,
) -> None:
    state_cache = outputs[num_warps]
    _save_partial_states_kernel[(size_m,)](
        kv,
        kv.stride(0),
        score,
        score.stride(0),
        ape,
        ape.stride(0),
        positions,
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        slot_mapping,
        block_size,
        HEAD_SIZE=head_size,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_size),
        STATE_WIDTH=head_size,
        COMPRESS_RATIO=compress_ratio,
        num_warps=num_warps,
    )


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
    parser.add_argument("--m", type=int, nargs="+", default=[1666, 1667, 2000])
    parser.add_argument("--warps", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    rows = []
    shapes = (
        ("c4_mla", 1024, 4),
        ("c4_indexer", 256, 4),
        ("c128_mla", 512, 128),
    )
    for shape_name, head_size, compress_ratio in shapes:
        block_size = 4 if compress_ratio == 4 else 8
        for size_m in args.m:
            kv = torch.randn(
                (size_m, head_size),
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
            score = torch.randn_like(kv)
            ape = torch.randn(
                (compress_ratio, head_size),
                dtype=torch.float32,
                device="cuda",
                generator=generator,
            )
            positions = torch.arange(size_m, dtype=torch.int64, device="cuda")
            slot_mapping = torch.arange(size_m, dtype=torch.int64, device="cuda")
            num_blocks = triton.cdiv(size_m, block_size)
            outputs = {
                num_warps: torch.zeros(
                    (num_blocks, block_size, 2 * head_size),
                    dtype=torch.float32,
                    device="cuda",
                )
                for num_warps in args.warps
            }

            launch = partial(
                _launch,
                outputs=outputs,
                size_m=size_m,
                kv=kv,
                score=score,
                ape=ape,
                positions=positions,
                slot_mapping=slot_mapping,
                block_size=block_size,
                head_size=head_size,
                compress_ratio=compress_ratio,
            )

            launch(4)
            reference = outputs[4].clone()
            for num_warps in args.warps:
                launch(num_warps)
                torch.accelerator.synchronize()
                torch.testing.assert_close(
                    outputs[num_warps], reference, atol=0, rtol=0
                )
                median_us, p90_us = _measure(
                    partial(launch, num_warps),
                    args.warmup,
                    args.iterations,
                    args.samples,
                )
                row = {
                    "device": current_platform.get_device_name(),
                    "shape": shape_name,
                    "m": size_m,
                    "head_size": head_size,
                    "compress_ratio": compress_ratio,
                    "num_warps": num_warps,
                    "median_us": median_us,
                    "p90_us": p90_us,
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
