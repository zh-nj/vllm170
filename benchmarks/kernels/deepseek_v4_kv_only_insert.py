# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark the DeepSeek-V4 DSpark KV-only cache insertion path."""

import argparse
import json
import statistics
from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch

import vllm._custom_ops  # noqa: F401
from vllm.platforms import current_platform

HEAD_DIM = 512
HEAD_BYTES = 584
NUM_HEADS = 64
PADDED_HEADS = 64
ROPE_DIM = 64

KernelFn = Callable[[], None]


def _make_cos_sin_cache(max_position: int) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000.0
        ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32, device="cuda") / ROPE_DIM)
    )
    positions = torch.arange(max_position, dtype=torch.float32, device="cuda")
    frequencies = torch.einsum("i,j->ij", positions, inv_freq)
    return torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)


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


def _old_kernel(
    q: torch.Tensor,
    kv: torch.Tensor,
    q_out: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    block_size: int,
) -> None:
    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_out(
        q,
        kv,
        q_out,
        cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        PADDED_HEADS,
        1e-6,
        block_size,
    )


def _old_request_path(
    kv: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    block_size: int,
) -> None:
    dummy_q = torch.zeros(
        (kv.shape[0], NUM_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        dummy_q,
        kv,
        cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        PADDED_HEADS,
        1e-6,
        block_size,
    )


def _kv_only(
    kv: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    block_size: int,
) -> None:
    torch.ops._C.deepseek_v4_kv_rope_quant_insert(
        kv,
        cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        block_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m", type=int, nargs="+", default=[7, 8, 32, 64, 256, 1667, 2048]
    )
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    cos_sin_cache = _make_cos_sin_cache(max(4096, max(args.m)))
    rows = []
    for size_m in args.m:
        kv = torch.randn(
            (size_m, HEAD_DIM),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        q = torch.zeros(
            (size_m, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
        )
        q_out = torch.empty_like(q)
        positions = torch.arange(size_m, dtype=torch.int64, device="cuda")
        slot_mapping = torch.arange(size_m, dtype=torch.int64, device="cuda")
        num_blocks = (size_m + args.block_size - 1) // args.block_size + 1
        cache_reference = torch.zeros(
            (num_blocks, args.block_size * HEAD_BYTES),
            dtype=torch.uint8,
            device="cuda",
        )
        cache_actual = torch.zeros_like(cache_reference)

        old_kernel = partial(
            _old_kernel,
            q,
            kv,
            q_out,
            cache_reference,
            slot_mapping,
            positions,
            cos_sin_cache,
            args.block_size,
        )
        old_request_path = partial(
            _old_request_path,
            kv,
            cache_reference,
            slot_mapping,
            positions,
            cos_sin_cache,
            args.block_size,
        )
        kv_only = partial(
            _kv_only,
            kv,
            cache_actual,
            slot_mapping,
            positions,
            cos_sin_cache,
            args.block_size,
        )

        old_kernel()
        kv_only()
        torch.accelerator.synchronize()
        torch.testing.assert_close(cache_actual, cache_reference, rtol=0, atol=0)

        old_kernel_us, old_kernel_p90_us = _measure(
            old_kernel, args.warmup, args.iterations, args.samples
        )
        old_request_us, old_request_p90_us = _measure(
            old_request_path, args.warmup, args.iterations, args.samples
        )
        kv_only_us, kv_only_p90_us = _measure(
            kv_only, args.warmup, args.iterations, args.samples
        )
        row = {
            "device": current_platform.get_device_name(),
            "m": size_m,
            "old_kernel_us": old_kernel_us,
            "old_kernel_p90_us": old_kernel_p90_us,
            "old_request_us": old_request_us,
            "old_request_p90_us": old_request_p90_us,
            "kv_only_us": kv_only_us,
            "kv_only_p90_us": kv_only_p90_us,
            "kernel_speedup": old_kernel_us / kv_only_us,
            "request_speedup": old_request_us / kv_only_us,
            "dummy_q_bytes": q.numel() * q.element_size(),
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
