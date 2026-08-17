# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare full-logits and streaming DeepSeek-V4 prefill top-k on SM8x."""

import argparse
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.v1.attention.ops.mqa_logits_triton import (
    _fp8_mqa_logits_bf16_impl,
    _prepare_streaming_tile_kernel,
    _select_prefill_kv_group,
    fp8_mqa_logits_streaming_topk_triton,
    fp8_mqa_logits_triton,
)

T = TypeVar("T")


def _measure(fn: Callable[[], T], warmup: int, iterations: int) -> tuple[float, int]:
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


@contextmanager
def _record_stage(
    events: dict[str, list[tuple[torch.Event, torch.Event]]],
    name: str,
) -> Iterator[None]:
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        events.setdefault(name, []).append((start, end))


def _stage_times(
    events: dict[str, list[tuple[torch.Event, torch.Event]]],
) -> dict[str, float]:
    torch.accelerator.synchronize()
    return {
        name: sum(start.elapsed_time(end) for start, end in pairs)
        for name, pairs in events.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--keys", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument(
        "--full-row-chunk",
        type=int,
        default=0,
        help="Bound rows per full-logits call, matching the serving logits cap.",
    )
    parser.add_argument(
        "--streaming-row-chunk",
        type=int,
        default=0,
        help="Bound rows per streaming call while retaining K tiling.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Report one-call CUDA-event stage timings for both paths.",
    )
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

    def full_topk(*, gather_scores: bool = False) -> tuple[torch.Tensor, ...]:
        row_chunk = args.full_row_chunk or args.rows
        indices = torch.empty((args.rows, args.topk), dtype=torch.int32, device="cuda")
        scores = (
            torch.empty((args.rows, args.topk), dtype=torch.float32, device="cuda")
            if gather_scores
            else None
        )
        for row_start in range(0, args.rows, row_chunk):
            row_end = min(row_start + row_chunk, args.rows)
            row_slice = slice(row_start, row_end)
            logits = fp8_mqa_logits_triton(
                q[row_slice],
                (k, scales),
                weights[row_slice],
                starts[row_slice],
                ends[row_slice],
                clean_logits=False,
            )
            chunk_indices = indices[row_slice]
            ops.top_k_per_row_prefill(
                logits,
                starts[row_slice],
                ends[row_slice],
                chunk_indices,
                row_end - row_start,
                logits.stride(0),
                logits.stride(1),
                args.topk,
            )
            if scores is not None:
                scores[row_slice] = logits.gather(1, chunk_indices.to(torch.int64))
        return (indices,) if scores is None else (scores, indices)

    def streaming_topk() -> tuple[torch.Tensor, torch.Tensor]:
        row_chunk = args.streaming_row_chunk or args.rows
        scores = torch.empty((args.rows, args.topk), dtype=torch.float32, device="cuda")
        indices = torch.empty((args.rows, args.topk), dtype=torch.int32, device="cuda")
        for row_start in range(0, args.rows, row_chunk):
            row_end = min(row_start + row_chunk, args.rows)
            row_slice = slice(row_start, row_end)
            scores[row_slice], indices[row_slice] = (
                fp8_mqa_logits_streaming_topk_triton(
                    q[row_slice],
                    (k, scales),
                    weights[row_slice],
                    starts[row_slice],
                    ends[row_slice],
                    args.topk,
                    tile_size=args.tile_size,
                )
            )
        return scores, indices

    reference_logits = fp8_mqa_logits_triton(
        q, (k, scales), weights, starts, ends, clean_logits=False
    )
    reference_scores, reference_indices = reference_logits.topk(args.topk, dim=1)
    full_scores, full_indices = full_topk(gather_scores=True)
    streaming_scores, streaming_indices = streaming_topk()
    torch.accelerator.synchronize()
    torch.testing.assert_close(
        full_scores.sort(dim=1, descending=True).values,
        reference_scores,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        streaming_scores.sort(dim=1, descending=True).values,
        reference_scores,
        rtol=0,
        atol=0,
    )
    del (
        reference_logits,
        reference_scores,
        reference_indices,
        full_scores,
        full_indices,
        streaming_scores,
        streaming_indices,
    )

    full_ms, full_peak_bytes = _measure(
        lambda: full_topk(gather_scores=False), args.warmup, args.iterations
    )
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
        "full_row_chunk": args.full_row_chunk or args.rows,
        "streaming_row_chunk": args.streaming_row_chunk or args.rows,
        "full_ms": full_ms,
        "streaming_ms": streaming_ms,
        "latency_ratio": streaming_ms / full_ms,
        "full_peak_bytes": full_peak_bytes,
        "streaming_peak_bytes": streaming_peak_bytes,
        "peak_memory_ratio": streaming_peak_bytes / full_peak_bytes,
        "sets_equal": True,
        "scores_equal": True,
    }
    if args.breakdown:
        full_events: dict[str, list[tuple[torch.Event, torch.Event]]] = {}
        full_row_chunk = args.full_row_chunk or args.rows
        full_indices = torch.empty(
            (args.rows, args.topk), dtype=torch.int32, device="cuda"
        )
        for row_start in range(0, args.rows, full_row_chunk):
            row_end = min(row_start + full_row_chunk, args.rows)
            row_slice = slice(row_start, row_end)
            with _record_stage(full_events, "q_decode"):
                q_bf16 = q[row_slice].to(torch.bfloat16)
            with _record_stage(full_events, "k_decode"):
                k_bf16 = k.to(torch.bfloat16)
            kv_group = _select_prefill_kv_group(row_end - row_start, args.keys)
            with _record_stage(full_events, "score"):
                logits = _fp8_mqa_logits_bf16_impl(
                    q_bf16,
                    k_bf16,
                    scales,
                    weights[row_slice],
                    starts[row_slice],
                    ends[row_slice],
                    kv_group,
                )
            with _record_stage(full_events, "select"):
                chunk_indices = full_indices[row_slice]
                ops.top_k_per_row_prefill(
                    logits,
                    starts[row_slice],
                    ends[row_slice],
                    chunk_indices,
                    row_end - row_start,
                    logits.stride(0),
                    logits.stride(1),
                    args.topk,
                )

        streaming_events: dict[str, list[tuple[torch.Event, torch.Event]]] = {}
        streaming_row_chunk = args.streaming_row_chunk or args.rows
        for row_start in range(0, args.rows, streaming_row_chunk):
            row_end = min(row_start + streaming_row_chunk, args.rows)
            row_slice = slice(row_start, row_end)
            chunk_rows = row_end - row_start
            with _record_stage(streaming_events, "q_decode"):
                q_bf16 = q[row_slice].to(torch.bfloat16)
            tile_width_max = min(args.tile_size, args.keys)
            k_bf16 = torch.empty(
                (tile_width_max, args.head_dim),
                dtype=torch.bfloat16,
                device="cuda",
            )
            tile_logits_storage = torch.empty(
                (chunk_rows, tile_width_max), dtype=torch.float32, device="cuda"
            )
            tile_starts = torch.empty(chunk_rows, dtype=torch.int32, device="cuda")
            tile_ends = torch.empty_like(tile_starts)
            best_scores = torch.empty(
                (chunk_rows, args.topk), dtype=torch.float32, device="cuda"
            )
            next_scores = torch.empty_like(best_scores)
            best_indices = torch.empty(
                (chunk_rows, args.topk), dtype=torch.int32, device="cuda"
            )
            next_indices = torch.empty_like(best_indices)
            kv_group = _select_prefill_kv_group(chunk_rows, args.keys)
            if tile_width_max < 128 * kv_group:
                kv_group = 1
            for tile_start in range(0, args.keys, args.tile_size):
                tile_end = min(tile_start + args.tile_size, args.keys)
                tile_width = tile_end - tile_start

                with _record_stage(streaming_events, "bounds"):
                    _prepare_streaming_tile_kernel[(chunk_rows,)](
                        starts[row_slice],
                        ends[row_slice],
                        tile_starts,
                        tile_ends,
                        tile_start,
                        tile_width,
                        num_warps=4,
                    )
                k_bf16_slice = k_bf16[:tile_width]
                with _record_stage(streaming_events, "k_decode"):
                    k_bf16_slice.copy_(k[tile_start:tile_end])
                tile_logits = tile_logits_storage[:, :tile_width]
                with _record_stage(streaming_events, "score"):
                    tile_logits = _fp8_mqa_logits_bf16_impl(
                        q_bf16,
                        k_bf16_slice,
                        scales[tile_start:tile_end],
                        weights[row_slice],
                        tile_starts,
                        tile_ends,
                        kv_group,
                        logits_out=tile_logits,
                    )
                with _record_stage(streaming_events, "select_merge"):
                    torch.ops._C.top_k_per_row_prefill_update_candidates(
                        tile_logits,
                        tile_starts,
                        tile_ends,
                        starts[row_slice],
                        best_scores,
                        best_indices,
                        next_scores,
                        next_indices,
                        tile_start,
                        chunk_rows,
                        tile_logits.stride(0),
                        tile_logits.stride(1),
                        args.topk,
                        tile_start > 0,
                    )
                best_scores, next_scores = next_scores, best_scores
                best_indices, next_indices = next_indices, best_indices

        result["full_stage_ms"] = _stage_times(full_events)
        result["streaming_stage_ms"] = _stage_times(streaming_events)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
