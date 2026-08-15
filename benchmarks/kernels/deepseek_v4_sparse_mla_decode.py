# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tune DeepSeek-V4 split sparse MLA decode on CUDA SM8x."""

import argparse
import gc
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch

from vllm.models.deepseek_v4.common.ops import quantize_and_insert_k_cache
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.v1.attention.ops.fp8_sm80 import get_e4m3fn_bf16_lut
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _sparse_attn_decode_partial_kernel,
    _sparse_attn_decode_ragged_kernel,
    _sparse_attn_decode_reduce_kernel,
)

KernelFn = Callable[[], object]
PartialConfig = tuple[int, int, int]
ReduceConfig = tuple[int, int]

HEADS = 64
HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
COMB_DIM = NOPE_DIM + ROPE_DIM
BLOCK_SIZE = 256
PARTIAL_CONFIGS: tuple[PartialConfig, ...] = (
    (8, 16, 4),
    (8, 16, 8),
    (8, 32, 4),
    (8, 32, 8),
    (16, 16, 4),
    (16, 16, 8),
    (16, 32, 4),
    (16, 32, 8),
)
REDUCE_CONFIGS: tuple[ReduceConfig, ...] = tuple(
    (block_h, num_warps) for block_h in (1, 2, 4, 8) for num_warps in (1, 2, 4, 8)
)


def _measure_graph(
    fn: KernelFn, warmup: int, iterations: int, samples: int
) -> tuple[float, float]:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(warmup):
        graph.replay()
    torch.accelerator.synchronize()
    timings = []
    for _ in range(samples):
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000.0 / iterations)
    del graph
    timings.sort()
    return statistics.median(timings), timings[int(0.9 * (samples - 1))]


def _make_cache(rows: int, generator: torch.Generator) -> torch.Tensor:
    num_blocks = triton.cdiv(rows, BLOCK_SIZE)
    cache = torch.zeros(
        num_blocks,
        BLOCK_SIZE,
        584,
        dtype=torch.uint8,
        device="cuda",
    )
    values = (
        torch.randn(
            rows,
            HEAD_DIM,
            device="cuda",
            generator=generator,
        )
        * 0.125
    ).to(torch.bfloat16)
    slots = torch.arange(rows, dtype=torch.int64, device="cuda")
    quantize_and_insert_k_cache(values, cache, slots, BLOCK_SIZE)
    return cache


class DecodeCase:
    def __init__(
        self,
        num_queries: int,
        main_len: int,
        extra_len: int,
        splits: int,
        seed: int,
    ) -> None:
        self.num_queries = num_queries
        self.main_len = main_len
        self.extra_len = extra_len
        self.splits = splits
        rows = max(4096, num_queries * max(main_len, extra_len, 1))
        generator = torch.Generator(device="cuda").manual_seed(seed)
        self.q = torch.randn(
            num_queries,
            HEADS,
            HEAD_DIM,
            device="cuda",
            generator=generator,
        ).to(torch.bfloat16)
        self.main_cache = _make_cache(rows, generator)
        self.main_indices = torch.arange(
            num_queries * main_len,
            dtype=torch.int32,
            device="cuda",
        ).remainder_(rows)
        self.main_indptr = torch.arange(
            0,
            (num_queries + 1) * main_len,
            main_len,
            dtype=torch.int32,
            device="cuda",
        )
        self.extra_cache = _make_cache(rows, generator)
        self.extra_indices = torch.arange(
            num_queries * extra_len,
            dtype=torch.int32,
            device="cuda",
        ).remainder_(rows)
        self.extra_indptr = (
            torch.arange(
                0,
                (num_queries + 1) * extra_len,
                extra_len,
                dtype=torch.int32,
                device="cuda",
            )
            if extra_len > 0
            else torch.zeros(num_queries + 1, dtype=torch.int32, device="cuda")
        )
        self.sink = torch.randn(HEADS, device="cuda", generator=generator)
        self.lut = get_e4m3fn_bf16_lut(self.q.device)
        self.out = torch.empty_like(self.q)
        self.copied_out = torch.empty_like(self.q)
        self.part_m = torch.empty(
            num_queries, splits, HEADS, dtype=torch.float32, device="cuda"
        )
        self.part_l = torch.empty_like(self.part_m)
        self.part_acc = torch.empty(
            num_queries,
            splits,
            HEADS,
            COMB_DIM,
            dtype=torch.float32,
            device="cuda",
        )

    def baseline(self) -> None:
        _sparse_attn_decode_ragged_kernel[(self.num_queries, triton.cdiv(HEADS, 16))](
            self.q,
            self.main_cache,
            self.main_indices,
            self.main_indptr,
            self.extra_cache,
            self.extra_indices,
            self.extra_indptr,
            self.sink,
            self.lut,
            self.out,
            self.q.stride(0),
            self.q.stride(1),
            self.out.stride(0),
            self.out.stride(1),
            self.main_cache.stride(0),
            self.extra_cache.stride(0),
            self.main_cache.shape[0] * BLOCK_SIZE,
            self.extra_cache.shape[0] * BLOCK_SIZE,
            BLOCK_SIZE,
            BLOCK_SIZE,
            HEAD_DIM**-0.5,
            HEADS,
            HAS_ATTN_SINK=True,
            HAS_EXTRA=self.extra_len > 0,
            NOPE_DIM=NOPE_DIM,
            NOPE_BLOCK=512,
            ROPE_DIM=ROPE_DIM,
            IS_FNUZ_MAIN=False,
            IS_FNUZ_EXTRA=False,
            BLOCK_H=16,
            BLOCK_K=16,
            num_warps=8,
        )

    def partial(self, config: PartialConfig) -> None:
        block_h, block_k, num_warps = config
        _sparse_attn_decode_partial_kernel[
            (
                self.num_queries,
                self.splits,
                triton.cdiv(HEADS, block_h),
            )
        ](
            self.q,
            self.main_cache,
            self.main_indices,
            self.main_indptr,
            self.extra_cache,
            self.extra_indices,
            self.extra_indptr,
            self.part_m,
            self.part_l,
            self.part_acc,
            self.lut,
            self.q.stride(0),
            self.q.stride(1),
            self.main_cache.stride(0),
            self.extra_cache.stride(0),
            self.part_m.stride(0),
            self.part_m.stride(1),
            self.part_acc.stride(0),
            self.part_acc.stride(1),
            self.part_acc.stride(2),
            self.main_cache.shape[0] * BLOCK_SIZE,
            self.extra_cache.shape[0] * BLOCK_SIZE,
            BLOCK_SIZE,
            BLOCK_SIZE,
            HEAD_DIM**-0.5,
            HEADS,
            HAS_EXTRA=self.extra_len > 0,
            NOPE_DIM=NOPE_DIM,
            NOPE_BLOCK=512,
            ROPE_DIM=ROPE_DIM,
            IS_FNUZ_MAIN=False,
            IS_FNUZ_EXTRA=False,
            BLOCK_H=block_h,
            BLOCK_K=block_k,
            NUM_SPLITS=self.splits,
            NUM_STAGES=1,
            num_warps=num_warps,
        )

    def reduce(self, config: ReduceConfig) -> None:
        block_h, num_warps = config
        _sparse_attn_decode_reduce_kernel[
            (self.num_queries, triton.cdiv(HEADS, block_h))
        ](
            self.part_m,
            self.part_l,
            self.part_acc,
            self.sink,
            self.out,
            self.out.stride(0),
            self.out.stride(1),
            self.part_m.stride(0),
            self.part_m.stride(1),
            self.part_acc.stride(0),
            self.part_acc.stride(1),
            self.part_acc.stride(2),
            HEADS,
            HAS_ATTN_SINK=True,
            COMB_DIM=COMB_DIM,
            BLOCK_H=block_h,
            NUM_SPLITS=self.splits,
            SPLITS_PAD=triton.next_power_of_2(self.splits),
            num_warps=num_warps,
        )

    def split(self, partial: PartialConfig, reduce: ReduceConfig) -> None:
        self.partial(partial)
        self.reduce(reduce)

    def split_copied(self, partial: PartialConfig, reduce: ReduceConfig) -> None:
        self.split(partial, reduce)
        self.copied_out.copy_(self.out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--main-len", type=int, default=128)
    parser.add_argument("--extra-len", type=int, default=512)
    parser.add_argument("--splits", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    case = DecodeCase(
        args.queries,
        args.main_len,
        args.extra_len,
        args.splits,
        args.seed,
    )
    case.baseline()
    torch.accelerator.synchronize()
    reference = case.out.clone()
    baseline_us, baseline_p90_us = _measure_graph(
        case.baseline, args.warmup, args.iterations, args.samples
    )
    rows: list[dict[str, object]] = [
        {
            "stage": "baseline",
            "device": current_platform.get_device_name(),
            "queries": args.queries,
            "main_len": args.main_len,
            "extra_len": args.extra_len,
            "graph_us": baseline_us,
            "graph_p90_us": baseline_p90_us,
        }
    ]

    partial_timings: dict[PartialConfig, float] = {}
    for config in PARTIAL_CONFIGS:
        case.partial(config)
        torch.accelerator.synchronize()
        timing, p90 = _measure_graph(
            lambda config=config: case.partial(config),
            args.warmup,
            args.iterations,
            args.samples,
        )
        partial_timings[config] = timing
        rows.append(
            {
                "stage": "partial",
                "block_h": config[0],
                "block_k": config[1],
                "num_warps": config[2],
                "graph_us": timing,
                "graph_p90_us": p90,
            }
        )

    seed_partial = min(partial_timings, key=partial_timings.__getitem__)
    case.partial(seed_partial)
    torch.accelerator.synchronize()
    reduce_timings: dict[ReduceConfig, float] = {}
    for config in REDUCE_CONFIGS:
        timing, p90 = _measure_graph(
            lambda config=config: case.reduce(config),
            args.warmup,
            args.iterations,
            args.samples,
        )
        reduce_timings[config] = timing
        rows.append(
            {
                "stage": "reduce",
                "block_h": config[0],
                "num_warps": config[1],
                "graph_us": timing,
                "graph_p90_us": p90,
            }
        )

    best_reduce = min(reduce_timings, key=reduce_timings.__getitem__)
    case.split(seed_partial, best_reduce)
    torch.accelerator.synchronize()
    delta = (case.out.float() - reference.float()).abs()
    split_us, split_p90_us = _measure_graph(
        lambda: case.split(seed_partial, best_reduce),
        args.warmup,
        args.iterations,
        args.samples,
    )
    split_copied_us, split_copied_p90_us = _measure_graph(
        lambda: case.split_copied(seed_partial, best_reduce),
        args.warmup,
        args.iterations,
        args.samples,
    )
    rows.append(
        {
            "stage": "combined",
            "partial": seed_partial,
            "reduce": best_reduce,
            "graph_us": split_us,
            "graph_p90_us": split_p90_us,
            "speedup": baseline_us / split_us,
            "max_abs": delta.max().item(),
            "mean_abs": delta.mean().item(),
            "workspace_bytes": (
                case.part_m.nbytes + case.part_l.nbytes + case.part_acc.nbytes
            ),
        }
    )
    rows.append(
        {
            "stage": "combined_copied_output",
            "partial": seed_partial,
            "reduce": best_reduce,
            "graph_us": split_copied_us,
            "graph_p90_us": split_copied_p90_us,
            "caller_output_speedup": split_copied_us / split_us,
            "copy_us": split_copied_us - split_us,
        }
    )

    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    gc.collect()


if __name__ == "__main__":
    main()
