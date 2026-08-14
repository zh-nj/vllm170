# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark DeepSeek-V4 sparse MLA prefill on SM8x real request shapes."""

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
CONFIGS = ((8, 16, 4), (8, 32, 4), (16, 16, 4), (16, 16, 8), (16, 32, 4))
HEADS = 64
HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
WINDOW = 128
TOPK = 512


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


def _lengths(num_queries: int, context_start: int, compress_ratio: int) -> torch.Tensor:
    positions = torch.arange(
        context_start,
        context_start + num_queries,
        dtype=torch.int32,
        device="cuda",
    )
    compressed = torch.minimum(
        (positions + 1) // compress_ratio, positions.new_tensor(TOPK)
    )
    swa = torch.minimum(positions + 1, positions.new_tensor(WINDOW))
    return compressed + swa


def _inputs(
    num_queries: int, context_start: int, compress_ratio: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = (
        torch.randn((num_queries, HEADS, HEAD_DIM), device="cuda", generator=generator)
        * 0.125
    ).to(torch.bfloat16)
    lengths = _lengths(num_queries, context_start, compress_ratio)
    indptr = torch.zeros(num_queries + 1, dtype=torch.int32, device="cuda")
    torch.cumsum(lengths, dim=0, out=indptr[1:])
    total_indices = int(indptr[-1].item())
    kv_rows = max(context_start + num_queries, int(lengths.max().item()))
    kv = (
        torch.randn((kv_rows, HEAD_DIM), device="cuda", generator=generator) * 0.125
    ).to(torch.bfloat16)
    indices = torch.arange(total_indices, dtype=torch.int32, device="cuda")
    indices.remainder_(kv_rows)
    sink = torch.full((HEADS,), -float("inf"), dtype=torch.float32, device="cuda")
    return q, kv, indices, indptr, sink


def _launch(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    sink: torch.Tensor,
    output: torch.Tensor | None,
    config: tuple[int, int, int],
) -> torch.Tensor:
    block_h, block_k, num_warps = config
    return _rocm_sparse_attn_prefill_ragged_triton(
        q=q,
        kv=kv,
        indices=indices,
        indptr=indptr,
        scale=HEAD_DIM**-0.5,
        attn_sink=sink,
        nope_head_dim=NOPE_DIM,
        rope_head_dim=ROPE_DIM,
        output=output,
        block_h=block_h,
        block_k=block_k,
        num_warps=num_warps,
    )


def _launch_copied(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    sink: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    temporary = _launch(q, kv, indices, indptr, sink, None, (16, 16, 4))
    output.copy_(temporary)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, nargs="+", default=[1666, 1667, 2000])
    parser.add_argument("--context-start", type=int, nargs="+", default=[0, 1667, 3334])
    parser.add_argument("--compress-ratio", type=int, nargs="+", default=[4, 128])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for compress_ratio in args.compress_ratio:
        for num_queries in args.m:
            for context_start in args.context_start:
                q, kv, indices, indptr, sink = _inputs(
                    num_queries, context_start, compress_ratio, args.seed
                )
                direct_output = torch.empty_like(q)
                copied_output = torch.empty_like(q)

                copied = partial(
                    _launch_copied, q, kv, indices, indptr, sink, copied_output
                )

                reference = copied()
                baseline_direct = partial(
                    _launch,
                    q,
                    kv,
                    indices,
                    indptr,
                    sink,
                    direct_output,
                    (16, 16, 4),
                )
                candidate = baseline_direct()
                torch.accelerator.synchronize()
                torch.testing.assert_close(candidate, reference, atol=0, rtol=0)
                copied_us, copied_p90_us = _measure(
                    copied, args.warmup, args.iterations, args.samples
                )

                for config in CONFIGS:
                    direct = partial(
                        _launch,
                        q,
                        kv,
                        indices,
                        indptr,
                        sink,
                        direct_output,
                        config,
                    )
                    candidate = direct()
                    torch.accelerator.synchronize()
                    torch.testing.assert_close(
                        candidate, reference, atol=2e-2, rtol=2e-2
                    )
                    direct_us, direct_p90_us = _measure(
                        direct, args.warmup, args.iterations, args.samples
                    )
                    row = {
                        "device": current_platform.get_device_name(),
                        "m": num_queries,
                        "context_start": context_start,
                        "compress_ratio": compress_ratio,
                        "min_length": int((indptr[1:] - indptr[:-1]).min().item()),
                        "max_length": int((indptr[1:] - indptr[:-1]).max().item()),
                        "block_h": config[0],
                        "block_k": config[1],
                        "num_warps": config[2],
                        "copied_us": copied_us,
                        "copied_p90_us": copied_p90_us,
                        "direct_us": direct_us,
                        "direct_p90_us": direct_p90_us,
                        "speedup": copied_us / direct_us,
                        "output_bytes": (
                            direct_output.numel() * direct_output.element_size()
                        ),
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
