# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark Qwen4Exp sparse prefill attention on its production SM80 shape."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.platforms import current_platform
from vllm.triton_utils import triton


@dataclass(frozen=True)
class KernelConfig:
    block_n: int
    num_warps: int
    num_stages: int

    @property
    def name(self) -> str:
        return f"bn{self.block_n}_w{self.num_warps}_s{self.num_stages}"


CONFIGS = (
    KernelConfig(16, 2, 2),
    KernelConfig(16, 4, 2),
    KernelConfig(32, 2, 2),
    KernelConfig(32, 4, 2),
    KernelConfig(32, 8, 2),
    KernelConfig(64, 2, 1),
    KernelConfig(64, 2, 2),
    KernelConfig(64, 2, 3),
    KernelConfig(64, 4, 2),
    KernelConfig(64, 8, 2),
    KernelConfig(128, 4, 2),
    KernelConfig(128, 8, 2),
)


def _launch_sparse_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor,
    config: KernelConfig,
) -> None:
    group_size = q.shape[1] // k_cache.shape[2]
    block_m = triton.next_power_of_2(group_size)
    num_tiles = triton.cdiv(logical_indices.shape[1], config.block_n)
    qsa_ops._qsa_sparse_paged_gqa_splitk_kernel[(q.shape[0], k_cache.shape[2], 1)](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        out,
        out,
        out,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        logical_indices.stride(0),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        TOPK=logical_indices.shape[1],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=1,
        NUM_TILES=num_tiles,
        BLOCK_M=block_m,
        BLOCK_N=config.block_n,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )


def _event_bench(fn: Callable[[], None], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    samples = []
    for _ in range(7):
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return statistics.median(samples)


def _run_zeroed(out: torch.Tensor, fn: Callable[[], None]) -> None:
    out.zero_()
    fn()


def _make_inputs(
    *,
    rows: int,
    context: int,
    max_context: int,
    page_size: int,
    selection_width: int,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(20260902 + context)
    num_query_heads = 24
    num_kv_heads = 2
    head_dim = 256
    num_blocks = math.ceil(max_context / page_size)
    q = torch.randn(
        rows,
        num_query_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k_cache = torch.randn(
        num_blocks,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    v_cache = torch.randn(
        num_blocks,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    block_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda").view(1, -1)
    token_to_req = torch.zeros(rows, dtype=torch.int32, device="cuda")
    query_positions = torch.arange(
        context - rows,
        context,
        dtype=torch.int64,
        device="cuda",
    )
    columns = torch.arange(selection_width, dtype=torch.int64, device="cuda")
    valid_counts = torch.minimum(query_positions + 1, query_positions.new_tensor(2051))
    logical_indices = (
        columns.unsqueeze(0) * 1543 + query_positions.unsqueeze(1) * 31
    ) % (query_positions.unsqueeze(1) + 1)
    logical_indices = torch.where(
        columns.unsqueeze(0) < valid_counts.unsqueeze(1),
        logical_indices,
        -1,
    ).to(torch.int32)
    return q, k_cache, v_cache, logical_indices, block_table, token_to_req


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--contexts", type=int, nargs="+", default=[2048, 32768])
    parser.add_argument("--max-context", type=int, default=34816)
    parser.add_argument("--page-size", type=int, default=1568)
    parser.add_argument("--selection-width", type=int, default=2051)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if current_platform.get_device_capability() != (8, 0):
        raise RuntimeError("This benchmark is intended for the SM80 production path")

    results = []
    for context in args.contexts:
        inputs = _make_inputs(
            rows=args.rows,
            context=context,
            max_context=args.max_context,
            page_size=args.page_size,
            selection_width=args.selection_width,
        )
        q, k_cache, v_cache, logical_indices, block_table, token_to_req = inputs
        baseline_out = torch.empty_like(q)
        baseline_fn = partial(
            qsa_ops.qsa_sparse_paged_attention,
            q,
            k_cache,
            v_cache,
            logical_indices,
            block_table,
            token_to_req,
            baseline_out,
        )
        baseline_fn()
        torch.accelerator.synchronize()
        baseline_ms = _event_bench(baseline_fn, args.warmup, args.iterations)

        zeroed_fn = partial(_run_zeroed, baseline_out, baseline_fn)
        zeroed_ms = _event_bench(zeroed_fn, args.warmup, args.iterations)
        flops = 4 * args.rows * 24 * args.selection_width * 256
        results.append(
            {
                "context": context,
                "config": "production_bn64_w2_s2",
                "zero_output": False,
                "latency_ms": baseline_ms,
                "tflops": flops / baseline_ms / 1e9,
                "correct": True,
            }
        )
        results.append(
            {
                "context": context,
                "config": "production_bn64_w2_s2",
                "zero_output": True,
                "latency_ms": zeroed_ms,
                "tflops": flops / zeroed_ms / 1e9,
                "correct": True,
            }
        )

        for config in CONFIGS:
            candidate_out = torch.empty_like(q)
            candidate_fn = partial(
                _launch_sparse_attention,
                q,
                k_cache,
                v_cache,
                logical_indices,
                block_table,
                token_to_req,
                candidate_out,
                config,
            )
            candidate_fn()
            torch.accelerator.synchronize()
            torch.testing.assert_close(
                candidate_out,
                baseline_out,
                rtol=2e-2,
                atol=2e-2,
            )
            latency_ms = _event_bench(candidate_fn, args.warmup, args.iterations)
            results.append(
                {
                    "context": context,
                    "config": config.name,
                    "zero_output": False,
                    "latency_ms": latency_ms,
                    "tflops": flops / latency_ms / 1e9,
                    "correct": True,
                    "relative_to_production": latency_ms / baseline_ms,
                }
            )

    metadata = {
        "device": current_platform.get_device_name(),
        "capability": current_platform.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "rows": args.rows,
        "contexts": args.contexts,
        "max_context": args.max_context,
        "page_size": args.page_size,
        "selection_width": args.selection_width,
        "timer": "cuda_event_median_of_7_groups",
    }
    payload = json.dumps({"metadata": metadata, "results": results}, indent=2)
    if args.output is not None:
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
