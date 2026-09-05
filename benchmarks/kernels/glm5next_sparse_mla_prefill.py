# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark the GLM-5.3 K-pool sparse MLA prefill kernel on SM8x."""

import argparse
import json
import statistics
from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _rocm_sparse_attn_prefill_ragged_triton,
)

KernelFn = Callable[[], torch.Tensor]
CONFIGS = (
    (1, 32, 4),
    (2, 32, 4),
    (4, 16, 4),
    (4, 32, 4),
    (8, 16, 4),
    (8, 32, 4),
    (8, 32, 8),
    (16, 16, 4),
    (16, 32, 4),
    (16, 32, 8),
)
HEADS = 64
HEAD_DIM = 512
NOPE_DIM = 512
ROPE_DIM = 0
TOPK = 2048


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


def _inputs(
    num_queries: int, context_start: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = (
        torch.randn((num_queries, HEADS, HEAD_DIM), device="cuda", generator=generator)
        * 0.125
    ).to(torch.bfloat16)
    kv_rows = context_start + num_queries
    kv_bf16 = (
        torch.randn((kv_rows, HEAD_DIM), device="cuda", generator=generator) * 0.125
    ).to(torch.bfloat16)
    kv_scale = torch.tensor(0.002, dtype=torch.float32, device="cuda")
    kv = (kv_bf16 / kv_scale).to(current_platform.fp8_dtype())

    positions = torch.arange(
        context_start,
        context_start + num_queries,
        dtype=torch.int32,
        device="cuda",
    )
    lengths = torch.minimum(positions + 1, positions.new_tensor(TOPK))
    indptr = torch.zeros(num_queries + 1, dtype=torch.int32, device="cuda")
    torch.cumsum(lengths, dim=0, out=indptr[1:])
    total_indices = int(indptr[-1].item())
    indices = torch.arange(total_indices, dtype=torch.int32, device="cuda")
    indices.mul_(15485863).add_(32452843).remainder_(kv_rows)
    return q, kv, kv_scale, indices, indptr


def _launch(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_scale: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    output: torch.Tensor,
    config: tuple[int, int, int],
) -> torch.Tensor:
    block_h, block_k, num_warps = config
    return _rocm_sparse_attn_prefill_ragged_triton(
        q=q,
        kv=kv,
        indices=indices,
        indptr=indptr,
        scale=HEAD_DIM**-0.5,
        attn_sink=None,
        nope_head_dim=NOPE_DIM,
        rope_head_dim=ROPE_DIM,
        output=output,
        kv_scale=kv_scale,
        block_h=block_h,
        block_k=block_k,
        num_warps=num_warps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, nargs="+", default=[2041, 1557])
    parser.add_argument(
        "--context-start", type=int, nargs="+", default=[0, 2041, 4082, 6123]
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument(
        "--config",
        action="append",
        help="Limit the sweep to BLOCK_H,BLOCK_K,num_warps (repeatable).",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    configs = CONFIGS
    if args.config:
        configs = tuple(
            tuple(int(value) for value in config.split(","))
            for config in args.config
        )
        if any(len(config) != 3 for config in configs):
            parser.error("--config must contain three comma-separated integers")

    rows = []
    for num_queries in args.m:
        for context_start in args.context_start:
            if context_start + num_queries > 8192:
                continue
            q, kv, kv_scale, indices, indptr = _inputs(
                num_queries, context_start, args.seed
            )
            output = torch.empty_like(q)
            baseline = partial(
                _launch,
                q,
                kv,
                kv_scale,
                indices,
                indptr,
                output,
                (16, 32, 4),
            )
            reference = baseline().clone()
            torch.accelerator.synchronize()

            for config in configs:
                candidate_fn = partial(
                    _launch,
                    q,
                    kv,
                    kv_scale,
                    indices,
                    indptr,
                    output,
                    config,
                )
                candidate = candidate_fn()
                torch.accelerator.synchronize()
                torch.testing.assert_close(
                    candidate, reference, atol=2e-2, rtol=2e-2
                )
                latency_us, p90_us = _measure(
                    candidate_fn, args.warmup, args.iterations, args.samples
                )
                row = {
                    "device": current_platform.get_device_name(),
                    "dtype": str(kv.dtype),
                    "m": num_queries,
                    "context_start": context_start,
                    "min_length": int((indptr[1:] - indptr[:-1]).min().item()),
                    "max_length": int((indptr[1:] - indptr[:-1]).max().item()),
                    "block_h": config[0],
                    "block_k": config[1],
                    "num_warps": config[2],
                    "latency_us": latency_us,
                    "p90_us": p90_us,
                    "correct": True,
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as output_file:
            for row in rows:
                output_file.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
