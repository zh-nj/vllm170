# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-Event benchmark for the SM80 paged indexer-logits kernel."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

import vllm._custom_ops  # noqa: F401  (register persistent top-k for imports)
from vllm.triton_utils import triton
from vllm.v1.attention.ops import mqa_logits_triton


def _measure(call, warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        call()
    torch.accelerator.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        call()
        end.record()
    torch.accelerator.synchronize()
    samples = [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]
    return {
        "mean_us": statistics.fmean(samples) * 1000.0,
        "p50_us": statistics.median(samples) * 1000.0,
        "p90_us": sorted(samples)[min(repeats - 1, int(repeats * 0.9))] * 1000.0,
    }


def _graph(call, warmup: int, graph_calls: int):
    for _ in range(warmup):
        call()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_calls):
            call()
    return graph.replay


def _make_call(q, cache, weights, context_lens, block_table, context):
    def call() -> None:
        mqa_logits_triton.fp8_paged_mqa_logits_triton(
            q,
            cache,
            weights,
            context_lens,
            block_table,
            max_model_len=context,
            clean_logits=False,
        )

    return call


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=int, nargs="+", default=[8192, 32768, 65536])
    parser.add_argument("--stages", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--graph-calls", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.block_size not in (64, 128):
        raise ValueError("block-size must be 64 or 128")

    torch.manual_seed(20260914)
    num_heads, head_dim = 64, 128
    q = torch.randn((1, 1, num_heads, head_dim), device="cuda").to(torch.float8_e4m3fn)
    weights = torch.randn((1, num_heads), device="cuda", dtype=torch.float32)
    rows = []
    for context in args.contexts:
        num_blocks = triton.cdiv(context, args.block_size)
        cache = torch.zeros(
            (num_blocks, args.block_size, 1, head_dim + 4),
            device="cuda",
            dtype=torch.uint8,
        )
        cache[..., :head_dim].random_(0, 255)
        scales = torch.ones(
            (num_blocks, args.block_size), device="cuda", dtype=torch.float32
        )
        cache[..., head_dim:].view(torch.float32).copy_(
            scales.view(num_blocks, args.block_size, 1, 1)
        )
        context_lens = torch.tensor([context], device="cuda", dtype=torch.int32)
        block_table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(
            1, -1
        )

        for num_stages in args.stages:
            mqa_logits_triton._fp8_paged_mqa_logits_kernel.configs = [
                triton.Config({}, num_warps=4, num_stages=num_stages)
            ]
            mqa_logits_triton._fp8_paged_mqa_logits_kernel.cache.clear()

            call = _make_call(q, cache, weights, context_lens, block_table, context)

            # Trigger compilation/configuration before graph capture.
            call()
            graph_call = _graph(call, args.warmup, args.graph_calls)
            event = _measure(graph_call, args.warmup, args.repeats)
            for key in event:
                event[key] /= args.graph_calls
            row = {
                "context": context,
                "block_size": args.block_size,
                "num_blocks": num_blocks,
                "num_stages": num_stages,
                "graph": event,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    serialized = json.dumps(rows, indent=2, sort_keys=True)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
