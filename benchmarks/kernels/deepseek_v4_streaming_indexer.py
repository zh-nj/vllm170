# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare full-logits and streaming DeepSeek-V4 prefill top-k on SM8x."""

import argparse
import json
from collections.abc import Callable

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.v1.attention.ops.mqa_logits_triton import (
    fp8_mqa_logits_streaming_topk_triton,
    fp8_mqa_logits_triton,
)

TopKFn = Callable[[], tuple[torch.Tensor, torch.Tensor]]


def _measure(fn: TopKFn, warmup: int, iterations: int) -> tuple[float, int]:
    for _ in range(warmup):
        outputs = fn()
    torch.accelerator.synchronize()
    del outputs
    torch.accelerator.empty_cache()

    baseline_bytes = torch.accelerator.memory_allocated()
    torch.accelerator.reset_peak_memory_stats()
    outputs = fn()
    torch.accelerator.synchronize()
    peak_bytes = torch.accelerator.max_memory_allocated() - baseline_bytes
    del outputs

    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        outputs = fn()
    end.record()
    torch.accelerator.synchronize()
    del outputs
    return start.elapsed_time(end) / iterations, peak_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--keys", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    q = torch.randn(
        (args.rows, args.heads, args.head_dim),
        device="cuda",
        generator=generator,
    ).to(torch.float8_e4m3fn)
    k = torch.randn((args.keys, args.head_dim), device="cuda", generator=generator).to(
        torch.float8_e4m3fn
    )
    scales = torch.rand((args.keys,), device="cuda", generator=generator) + 0.5
    weights = torch.randn((args.rows, args.heads), device="cuda", generator=generator)
    starts = torch.zeros(args.rows, dtype=torch.int32, device="cuda")
    ends = torch.full((args.rows,), args.keys, dtype=torch.int32, device="cuda")

    def full_topk() -> tuple[torch.Tensor, torch.Tensor]:
        logits = fp8_mqa_logits_triton(
            q, (k, scales), weights, starts, ends, clean_logits=False
        )
        indices = torch.empty((args.rows, args.topk), dtype=torch.int32, device="cuda")
        ops.top_k_per_row_prefill(
            logits,
            starts,
            ends,
            indices,
            args.rows,
            logits.stride(0),
            logits.stride(1),
            args.topk,
        )
        scores = logits.gather(1, indices.to(torch.int64))
        return scores, indices

    def streaming_topk() -> tuple[torch.Tensor, torch.Tensor]:
        return fp8_mqa_logits_streaming_topk_triton(
            q,
            (k, scales),
            weights,
            starts,
            ends,
            args.topk,
            tile_size=args.tile_size,
        )

    reference_logits = fp8_mqa_logits_triton(
        q, (k, scales), weights, starts, ends, clean_logits=False
    )
    reference_scores, reference_indices = reference_logits.topk(args.topk, dim=1)
    full_scores, full_indices = full_topk()
    streaming_scores, streaming_indices = streaming_topk()
    torch.accelerator.synchronize()
    if not torch.equal(
        reference_indices.to(torch.int32).sort().values,
        full_indices.sort().values,
    ):
        raise AssertionError("full-logits selector is not mathematically exact")
    if not torch.equal(
        reference_indices.to(torch.int32).sort().values,
        streaming_indices.sort().values,
    ):
        raise AssertionError("streaming top-k selected a different index set")
    torch.testing.assert_close(streaming_scores, reference_scores, rtol=0, atol=0)
    del (
        reference_logits,
        reference_scores,
        reference_indices,
        full_scores,
        full_indices,
        streaming_scores,
        streaming_indices,
    )

    full_ms, full_peak_bytes = _measure(full_topk, args.warmup, args.iterations)
    streaming_ms, streaming_peak_bytes = _measure(
        streaming_topk, args.warmup, args.iterations
    )
    result = {
        "device": current_platform.get_device_name(),
        "rows": args.rows,
        "keys": args.keys,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "topk": args.topk,
        "tile_size": args.tile_size,
        "full_ms": full_ms,
        "streaming_ms": streaming_ms,
        "latency_ratio": streaming_ms / full_ms,
        "full_peak_bytes": full_peak_bytes,
        "streaming_peak_bytes": streaming_peak_bytes,
        "peak_memory_ratio": streaming_peak_bytes / full_peak_bytes,
        "sets_equal": True,
        "scores_equal": True,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
