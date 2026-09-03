# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark Qwen4Exp QSA prefill index selection on its SM80 shape."""

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
from vllm.triton_utils import tl, triton


@dataclass(frozen=True)
class ScoreConfig:
    block_n: int
    tiles_per_program: int
    num_warps: int
    num_stages: int

    @property
    def name(self) -> str:
        return (
            f"bn{self.block_n}_tpp{self.tiles_per_program}_"
            f"w{self.num_warps}_s{self.num_stages}"
        )


SCORE_CONFIGS = (
    ScoreConfig(32, 4, 2, 2),
    ScoreConfig(32, 8, 2, 2),
    ScoreConfig(32, 16, 2, 2),
    ScoreConfig(64, 2, 2, 2),
    ScoreConfig(64, 4, 2, 2),
    ScoreConfig(64, 8, 2, 1),
    ScoreConfig(64, 8, 2, 2),
    ScoreConfig(64, 8, 2, 3),
    ScoreConfig(64, 8, 4, 2),
    ScoreConfig(64, 16, 2, 2),
    ScoreConfig(128, 2, 4, 2),
    ScoreConfig(128, 4, 4, 2),
    ScoreConfig(128, 8, 4, 2),
)


@dataclass(frozen=True)
class RowTileConfig:
    block_m: int
    block_n: int
    tiles_per_program: int
    num_warps: int
    num_stages: int

    @property
    def name(self) -> str:
        return (
            f"bm{self.block_m}_bn{self.block_n}_tpp{self.tiles_per_program}_"
            f"w{self.num_warps}_s{self.num_stages}"
        )


ROW_TILE_CONFIGS = (
    RowTileConfig(4, 32, 4, 4, 2),
    RowTileConfig(4, 32, 8, 4, 2),
    RowTileConfig(4, 64, 2, 4, 2),
    RowTileConfig(4, 64, 4, 4, 2),
    RowTileConfig(4, 64, 8, 4, 2),
    RowTileConfig(8, 32, 4, 4, 2),
    RowTileConfig(8, 32, 8, 4, 2),
    RowTileConfig(8, 64, 2, 4, 2),
    RowTileConfig(8, 64, 4, 4, 2),
    RowTileConfig(8, 64, 8, 8, 2),
    RowTileConfig(16, 32, 4, 8, 2),
    RowTileConfig(16, 32, 8, 8, 2),
    RowTileConfig(16, 64, 2, 8, 2),
    RowTileConfig(16, 64, 4, 8, 2),
    RowTileConfig(16, 64, 4, 8, 1),
    RowTileConfig(16, 64, 4, 8, 3),
    RowTileConfig(16, 64, 8, 8, 2),
    RowTileConfig(32, 32, 2, 8, 2),
    RowTileConfig(32, 32, 4, 8, 2),
    RowTileConfig(32, 64, 1, 8, 2),
    RowTileConfig(32, 64, 2, 8, 2),
    RowTileConfig(32, 64, 4, 8, 2),
)


@triton.jit
def _qsa_mqa_paged_rowtile_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_table_page,
    stride_logits_row,
    num_rows,
    num_columns,
    num_pages,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    STAGES: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_valid = row_offsets < num_rows
    query_positions = tl.load(
        query_positions_ptr + row_offsets,
        mask=row_valid,
        other=-1,
    )
    sequence_length = tl.load(sequence_lengths_ptr)
    visible = tl.where(
        row_valid,
        tl.minimum(
            (query_positions + 1) // COMPRESS_RATIO,
            sequence_length // COMPRESS_RATIO,
        ),
        0,
    )
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + row_offsets, visible, mask=row_valid)

    tile_start = tl.program_id(1) * TILES_PER_PROG
    max_visible = tl.max(visible, axis=0)
    if tile_start * BLOCK_N >= max_visible:
        return
    tile_end = tl.minimum(tile_start + TILES_PER_PROG, tl.cdiv(max_visible, BLOCK_N))
    tile_end = tl.minimum(tile_end, tl.cdiv(num_columns, BLOCK_N))

    packed = tl.arange(0, BLOCK_M * NUM_HEADS)
    packed_rows = tl.program_id(0) * BLOCK_M + packed // NUM_HEADS
    packed_heads = packed % NUM_HEADS
    dims = tl.arange(0, BLOCK_D)
    query = tl.load(
        q_ptr
        + packed_rows[None, :] * stride_q_row
        + packed_heads[None, :] * stride_q_head
        + dims[:, None] * stride_q_dim,
        mask=(packed_rows[None, :] < num_rows) & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(tile_start, tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < max_visible
        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr + logical_page * stride_table_page,
            mask=live,
            other=-1,
        )
        page_valid = live & (physical_page >= 0) & (physical_page < num_pages)
        safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=page_valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.reshape(scores, (BLOCK_N, BLOCK_M, NUM_HEADS))
        scores = tl.sum(tl.maximum(scores, 0.0), axis=2) / score_divisor
        output_valid = (
            page_valid[:, None]
            & row_valid[None, :]
            & (columns[:, None] < visible[None, :])
        )
        tl.store(
            logits_ptr + columns[:, None] + row_offsets[None, :] * stride_logits_row,
            tl.where(output_valid, scores, -float("inf")),
            mask=output_valid,
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


def _stabilize_gpu() -> None:
    lhs = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    rhs = torch.randn_like(lhs)
    for _ in range(12):
        torch.mm(lhs, rhs)
    torch.accelerator.synchronize()
    del lhs, rhs


def _make_inputs(
    *,
    rows: int,
    context: int,
    max_context: int,
    page_size: int,
    compress_ratio: int,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(20260902 + context)
    num_heads = 4
    head_dim = 128
    compressed_capacity = math.ceil(max_context / compress_ratio)
    num_pages = math.ceil(compressed_capacity / page_size)
    q = torch.randn(
        rows,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k_cache = torch.randn(
        num_pages,
        page_size,
        1,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    page_table = torch.arange(num_pages, dtype=torch.int32, device="cuda").view(1, -1)
    token_to_req = torch.zeros(rows, dtype=torch.int32, device="cuda")
    query_positions = torch.arange(
        context - rows,
        context,
        dtype=torch.int64,
        device="cuda",
    )
    sequence_lengths = torch.tensor([context], dtype=torch.int32, device="cuda")
    return (
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
    )


def _launch_score(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    compress_ratio: int,
    config: ScoreConfig,
) -> None:
    columns = logits.shape[1]
    block_d = max(16, triton.next_power_of_2(q.shape[2]))
    max_n = max(16, triton.next_power_of_2(q.shape[1]))
    qsa_ops._qsa_mqa_paged_kernel[
        (
            q.shape[0],
            triton.cdiv(columns, config.block_n * config.tiles_per_program),
        )
    ](
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(3),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        q.shape[0],
        columns,
        k_cache.shape[0],
        page_table.shape[0],
        math.sqrt(q.shape[2]),
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        BLOCK_N=config.block_n,
        BLOCK_D=block_d,
        TILES_PER_PROG=config.tiles_per_program,
        STAGES=config.num_stages,
        MAX_N=max_n,
        COMPRESS_RATIO=compress_ratio,
        num_warps=config.num_warps,
    )


def _launch_rowtile_score(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    compress_ratio: int,
    config: RowTileConfig,
) -> None:
    if page_table.shape[0] != 1:
        raise ValueError("The row-tiled benchmark requires one request")
    columns = logits.shape[1]
    block_d = max(16, triton.next_power_of_2(q.shape[2]))
    _qsa_mqa_paged_rowtile_kernel[
        (
            triton.cdiv(q.shape[0], config.block_m),
            triton.cdiv(columns, config.block_n * config.tiles_per_program),
        )
    ](
        q,
        k_cache,
        page_table,
        query_positions,
        sequence_lengths,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(3),
        page_table.stride(1),
        logits.stride(0),
        q.shape[0],
        columns,
        k_cache.shape[0],
        math.sqrt(q.shape[2]),
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_D=block_d,
        TILES_PER_PROG=config.tiles_per_program,
        STAGES=config.num_stages,
        COMPRESS_RATIO=compress_ratio,
        num_warps=config.num_warps,
    )


def _run_topk(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    blocks: torch.Tensor,
    workspace: torch.Tensor,
) -> None:
    torch.ops._C.persistent_topk(
        logits,
        visible_blocks,
        blocks,
        workspace,
        blocks.shape[1],
        logits.shape[1],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--contexts", type=int, nargs="+", default=[2048, 32768])
    parser.add_argument("--max-context", type=int, default=34816)
    parser.add_argument("--page-size", type=int, default=1568)
    parser.add_argument("--token-topk", type=int, default=2048)
    parser.add_argument("--compress-ratio", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if current_platform.get_device_capability() != (8, 0):
        raise RuntimeError("This benchmark is intended for the SM80 production path")
    if args.rows > min(args.contexts):
        raise ValueError("Every context must contain at least one full query chunk")

    _stabilize_gpu()
    results: list[dict[str, object]] = []
    production_config = ScoreConfig(64, 8, 2, 2)
    for context in args.contexts:
        inputs = _make_inputs(
            rows=args.rows,
            context=context,
            max_context=args.max_context,
            page_size=args.page_size,
            compress_ratio=args.compress_ratio,
        )
        q, k_cache, page_table, token_to_req, query_positions, sequence_lengths = inputs
        columns = page_table.shape[1] * k_cache.shape[1]
        logits = torch.empty((args.rows, columns), dtype=torch.float32, device="cuda")
        visible_blocks = torch.empty(args.rows, dtype=torch.int32, device="cuda")

        production_kernel = partial(
            _launch_score,
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            sequence_lengths,
            logits,
            visible_blocks,
            args.compress_ratio,
            production_config,
        )
        production_kernel()
        torch.accelerator.synchronize()
        reference_logits = logits.clone()
        reference_visible = visible_blocks.clone()

        production_score = partial(
            qsa_ops.qsa_mqa_paged,
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            sequence_lengths,
            args.compress_ratio,
        )
        results.append(
            {
                "context": context,
                "phase": "score_call",
                "config": "production",
                "latency_ms": _event_bench(
                    production_score, args.warmup, args.iterations
                ),
                "correct": True,
            }
        )
        production_kernel_ms = _event_bench(
            production_kernel, args.warmup, args.iterations
        )
        results.append(
            {
                "context": context,
                "phase": "score_kernel",
                "config": production_config.name,
                "latency_ms": production_kernel_ms,
                "correct": True,
            }
        )

        for config in SCORE_CONFIGS:
            candidate_logits = torch.empty_like(logits)
            candidate_visible = torch.empty_like(visible_blocks)
            candidate = partial(
                _launch_score,
                q,
                k_cache,
                page_table,
                token_to_req,
                query_positions,
                sequence_lengths,
                candidate_logits,
                candidate_visible,
                args.compress_ratio,
                config,
            )
            candidate()
            torch.accelerator.synchronize()
            torch.testing.assert_close(candidate_visible, reference_visible)
            visible_columns = torch.arange(columns, device="cuda").unsqueeze(0)
            visible = visible_columns < reference_visible.unsqueeze(1)
            torch.testing.assert_close(
                candidate_logits[visible],
                reference_logits[visible],
                rtol=1e-3,
                atol=1e-3,
            )
            latency_ms = _event_bench(candidate, args.warmup, args.iterations)
            results.append(
                {
                    "context": context,
                    "phase": "score_kernel",
                    "config": config.name,
                    "latency_ms": latency_ms,
                    "relative_to_production": latency_ms / production_kernel_ms,
                    "correct": True,
                }
            )

        for config in ROW_TILE_CONFIGS:
            candidate_logits = torch.empty_like(logits)
            candidate_visible = torch.empty_like(visible_blocks)
            candidate = partial(
                _launch_rowtile_score,
                q,
                k_cache,
                page_table,
                query_positions,
                sequence_lengths,
                candidate_logits,
                candidate_visible,
                args.compress_ratio,
                config,
            )
            candidate()
            torch.accelerator.synchronize()
            torch.testing.assert_close(candidate_visible, reference_visible)
            visible_columns = torch.arange(columns, device="cuda").unsqueeze(0)
            visible = visible_columns < reference_visible.unsqueeze(1)
            torch.testing.assert_close(
                candidate_logits[visible],
                reference_logits[visible],
                rtol=1e-3,
                atol=1e-3,
            )
            latency_ms = _event_bench(candidate, args.warmup, args.iterations)
            results.append(
                {
                    "context": context,
                    "phase": "score_rowtile_kernel",
                    "config": config.name,
                    "latency_ms": latency_ms,
                    "relative_to_production": latency_ms / production_kernel_ms,
                    "correct": True,
                }
            )

        block_topk = args.token_topk // args.compress_ratio
        blocks = torch.empty((args.rows, block_topk), dtype=torch.int32, device="cuda")
        workspace = torch.empty(
            qsa_ops._TOPK_WORKSPACE_BYTES, dtype=torch.uint8, device="cuda"
        )
        topk = partial(_run_topk, logits, visible_blocks, blocks, workspace)
        topk()
        torch.accelerator.synchronize()
        topk_ms = _event_bench(topk, args.warmup, args.iterations)
        results.append(
            {
                "context": context,
                "phase": "topk",
                "config": "persistent_topk",
                "latency_ms": topk_ms,
                "correct": True,
            }
        )

        expanded = torch.empty(
            (args.rows, args.token_topk + args.compress_ratio - 1),
            dtype=torch.int32,
            device="cuda",
        )
        expand = partial(
            qsa_ops.expand_qsa_block_indices_cuda,
            blocks,
            query_positions,
            sequence_lengths,
            token_to_req,
            args.compress_ratio,
            args.token_topk,
            expanded,
        )
        expand_ms = _event_bench(expand, args.warmup, args.iterations)
        results.append(
            {
                "context": context,
                "phase": "expand",
                "config": "production",
                "latency_ms": expand_ms,
                "correct": True,
            }
        )

        selected = torch.empty_like(expanded)
        selection = partial(
            qsa_ops.qsa_select_paged_tokens,
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            sequence_lengths,
            args.token_topk,
            args.compress_ratio,
            selected,
        )
        selection()
        torch.accelerator.synchronize()
        torch.testing.assert_close(
            selected.sort(dim=1).values,
            expanded.sort(dim=1).values,
        )
        results.append(
            {
                "context": context,
                "phase": "selection_call",
                "config": "production",
                "latency_ms": _event_bench(selection, args.warmup, args.iterations),
                "correct": True,
            }
        )

    payload = {
        "device": current_platform.get_device_name(),
        "capability": list(current_platform.get_device_capability()),
        "timing": "CUDA events; median of 7 groups",
        "shape": {
            "rows": args.rows,
            "max_context": args.max_context,
            "page_size": args.page_size,
            "token_topk": args.token_topk,
            "compress_ratio": args.compress_ratio,
            "heads": 4,
            "head_dim": 128,
        },
        "results": results,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
