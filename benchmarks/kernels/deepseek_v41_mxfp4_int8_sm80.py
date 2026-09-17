# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark DSV4.1 SM80 MXFP4-int8 expert GEMM tiles."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)

HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 2304
NUM_EXPERTS = 384
TOPK = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[232, 256])
    parser.add_argument("--block-m", type=int, nargs="+", default=[16])
    parser.add_argument("--block-n", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--graph-calls", type=int, default=40)
    parser.add_argument("--rotate-experts", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def measure(call, warmup: int, repeats: int) -> dict[str, float]:
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
        "mean_ms": statistics.fmean(samples),
        "p50_ms": statistics.median(samples),
        "p90_ms": percentile(samples, 0.9),
    }


def capture(call: Callable[[], None], warmup: int, graph_calls: int):
    for _ in range(warmup):
        call()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_calls):
            call()

    return graph.replay


def capture_many(calls: list[Callable[[], None]], warmup: int):
    for _ in range(warmup):
        for call in calls:
            call()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()

    return graph.replay


def make_gemm_call(
    a_q: torch.Tensor,
    a_scale: torch.Tensor,
    weight: torch.Tensor,
    shift2: torch.Tensor,
    channel_scale: torch.Tensor,
    output: torch.Tensor,
    routed_out: torch.Tensor,
    routing_weights: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_post_pad: torch.Tensor,
    num_routed: int,
    block_m: int,
    block_n: int,
    source_rows_are_slots: bool,
) -> Callable[[], None]:
    def call() -> None:
        torch.ops._moe_C.dsv4_mxfp4_int8_sm80_moe_gemm(
            a_q,
            a_scale,
            weight,
            shift2,
            channel_scale,
            output,
            routed_out,
            routing_weights,
            sorted_ids,
            expert_ids,
            num_post_pad,
            num_routed,
            block_m,
            block_n,
            source_rows_are_slots,
        )

    return call


def make_reduce_call(
    routed_out: torch.Tensor,
    routing_weights: torch.Tensor,
    output: torch.Tensor,
    num_routed: int,
) -> Callable[[], None]:
    def call() -> None:
        torch.ops._moe_C.dsv4_mxfp4_int8_sm80_moe_reduce(
            routed_out,
            routing_weights,
            output,
            num_routed,
        )

    return call


def check_tile_equivalence(
    a13_q: torch.Tensor,
    a13_scale: torch.Tensor,
    w13: torch.Tensor,
    w13_shift2: torch.Tensor,
    w13_scale: torch.Tensor,
    a2_q: torch.Tensor,
    a2_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_shift2: torch.Tensor,
    w2_scale: torch.Tensor,
    output: torch.Tensor,
    routing_weights: torch.Tensor,
    routing_ids: torch.Tensor,
    num_routed: int,
    tiles: list[tuple[int, int]],
) -> None:
    gate_up_reference = torch.empty(
        (num_routed, INTERMEDIATE_SIZE * 2), dtype=torch.bfloat16, device="cuda"
    )
    routed_out_reference = torch.empty(
        (num_routed, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
    )
    gate_up_candidate = torch.empty_like(gate_up_reference)
    routed_out_candidate = torch.empty_like(routed_out_reference)

    reference_tile = (16, 128)
    validation_tiles = [
        reference_tile,
        *(tile for tile in tiles if tile != reference_tile),
    ]
    for tile_index, (block_m, block_n) in enumerate(validation_tiles):
        sorted_ids, expert_ids, num_post_pad = moe_align_block_size(
            routing_ids, block_m, NUM_EXPERTS
        )
        gate_up = gate_up_reference if tile_index == 0 else gate_up_candidate
        routed_out = routed_out_reference if tile_index == 0 else routed_out_candidate
        make_gemm_call(
            a13_q,
            a13_scale,
            w13,
            w13_shift2,
            w13_scale,
            gate_up,
            gate_up,
            routing_weights,
            sorted_ids,
            expert_ids,
            num_post_pad,
            num_routed,
            block_m,
            block_n,
            False,
        )()
        make_gemm_call(
            a2_q,
            a2_scale,
            w2,
            w2_shift2,
            w2_scale,
            output,
            routed_out,
            routing_weights,
            sorted_ids,
            expert_ids,
            num_post_pad,
            num_routed,
            block_m,
            block_n,
            True,
        )()
        torch.accelerator.synchronize()
        if tile_index > 0:
            torch.testing.assert_close(
                gate_up_candidate, gate_up_reference, rtol=0, atol=0
            )
            torch.testing.assert_close(
                routed_out_candidate, routed_out_reference, rtol=0, atol=0
            )


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.repeats <= 0 or args.graph_calls <= 0:
        raise ValueError(
            "warmup must be non-negative; repeats and graph-calls must be positive"
        )
    if any(tokens <= 0 for tokens in args.tokens):
        raise ValueError("tokens must be positive")
    if any(block_m not in (16, 64, 128) for block_m in args.block_m):
        raise ValueError("block-m must contain only supported 16, 64, or 128 tiles")
    if any(block_n not in (64, 128) for block_n in args.block_n):
        raise ValueError("block-n must contain only supported 64 or 128 tiles")
    supported_tiles = {(16, 64), (16, 128), (64, 128), (128, 128)}
    tiles = [
        (block_m, block_n)
        for block_m in args.block_m
        for block_n in args.block_n
        if (block_m, block_n) in supported_tiles
    ]
    if not tiles:
        raise ValueError("no supported block-m/block-n tile combinations requested")

    device = torch.device("cuda")
    torch.manual_seed(4242)
    w13 = torch.empty(
        (NUM_EXPERTS, HIDDEN_SIZE // 32, INTERMEDIATE_SIZE * 2 // 8, 128),
        dtype=torch.uint8,
        device=device,
    )
    w13.random_(0, 256)
    w13_shift2 = torch.empty(
        (NUM_EXPERTS, HIDDEN_SIZE // 128, INTERMEDIATE_SIZE * 2 // 8, 8),
        dtype=torch.uint8,
        device=device,
    )
    w13_shift2.random_(0, 256)
    w13_scale = torch.ones(
        (NUM_EXPERTS, INTERMEDIATE_SIZE * 2), dtype=torch.float32, device=device
    )
    w2 = torch.empty(
        (NUM_EXPERTS, INTERMEDIATE_SIZE // 32, HIDDEN_SIZE // 8, 128),
        dtype=torch.uint8,
        device=device,
    )
    w2.random_(0, 256)
    w2_shift2 = torch.empty(
        (NUM_EXPERTS, INTERMEDIATE_SIZE // 128, HIDDEN_SIZE // 8, 8),
        dtype=torch.uint8,
        device=device,
    )
    w2_shift2.random_(0, 256)
    w2_scale = torch.ones(
        (NUM_EXPERTS, HIDDEN_SIZE), dtype=torch.float32, device=device
    )

    results = []
    for tokens in args.tokens:
        routing_ids = (
            torch.rand((tokens, NUM_EXPERTS), dtype=torch.float32, device=device)
            .topk(TOPK, dim=1)
            .indices.to(torch.int32)
        )
        routing_weights = torch.full(
            (tokens, TOPK), 1 / TOPK, dtype=torch.float32, device=device
        )
        num_routed = tokens * TOPK
        a13_q = torch.empty((tokens, HIDDEN_SIZE), dtype=torch.int8, device=device)
        a13_q.random_(-127, 128)
        a13_scale = torch.ones((tokens,), dtype=torch.float32, device=device)
        gate_up = torch.empty(
            (num_routed, INTERMEDIATE_SIZE * 2), dtype=torch.bfloat16, device=device
        )
        a2_q = torch.empty(
            (num_routed, INTERMEDIATE_SIZE), dtype=torch.int8, device=device
        )
        a2_q.random_(-127, 128)
        a2_scale = torch.ones((num_routed,), dtype=torch.float32, device=device)
        routed_out = torch.empty(
            (num_routed, HIDDEN_SIZE), dtype=torch.bfloat16, device=device
        )
        output = torch.empty((tokens, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)

        check_tile_equivalence(
            a13_q,
            a13_scale,
            w13,
            w13_shift2,
            w13_scale,
            a2_q,
            a2_scale,
            w2,
            w2_shift2,
            w2_scale,
            output,
            routing_weights,
            routing_ids,
            num_routed,
            tiles,
        )

        for block_m, block_n in tiles:
            sorted_ids, expert_ids, num_post_pad = moe_align_block_size(
                routing_ids, block_m, NUM_EXPERTS
            )
            w13_call = make_gemm_call(
                a13_q,
                a13_scale,
                w13,
                w13_shift2,
                w13_scale,
                gate_up,
                gate_up,
                routing_weights,
                sorted_ids,
                expert_ids,
                num_post_pad,
                num_routed,
                block_m,
                block_n,
                False,
            )
            w2_call = make_gemm_call(
                a2_q,
                a2_scale,
                w2,
                w2_shift2,
                w2_scale,
                output,
                routed_out,
                routing_weights,
                sorted_ids,
                expert_ids,
                num_post_pad,
                num_routed,
                block_m,
                block_n,
                True,
            )
            reduce_call = make_reduce_call(
                routed_out,
                routing_weights,
                output,
                num_routed,
            )
            graph_w13_call = None
            graph_w2_call = None
            if args.rotate_experts:
                if tokens != 1:
                    raise ValueError("--rotate-experts currently requires --tokens 1")
                w13_calls = []
                w2_calls = []
                for call_index in range(args.graph_calls):
                    routed_ids = (
                        (
                            torch.arange(TOPK, dtype=torch.int32, device=device)
                            + call_index * TOPK
                        )
                        .remainder(NUM_EXPERTS)
                        .view(1, TOPK)
                    )
                    routed_sorted, routed_experts, routed_post_pad = (
                        moe_align_block_size(routed_ids, block_m, NUM_EXPERTS)
                    )
                    w13_calls.append(
                        make_gemm_call(
                            a13_q,
                            a13_scale,
                            w13,
                            w13_shift2,
                            w13_scale,
                            gate_up,
                            gate_up,
                            routing_weights,
                            routed_sorted,
                            routed_experts,
                            routed_post_pad,
                            num_routed,
                            block_m,
                            block_n,
                            False,
                        )
                    )
                    w2_calls.append(
                        make_gemm_call(
                            a2_q,
                            a2_scale,
                            w2,
                            w2_shift2,
                            w2_scale,
                            output,
                            routed_out,
                            routing_weights,
                            routed_sorted,
                            routed_experts,
                            routed_post_pad,
                            num_routed,
                            block_m,
                            block_n,
                            True,
                        )
                    )
                graph_w13_call = capture_many(w13_calls, args.warmup)
                graph_w2_call = capture_many(w2_calls, args.warmup)
            eager_w13 = measure(w13_call, args.warmup, args.repeats)
            eager_w2 = measure(w2_call, args.warmup, args.repeats)
            eager_reduce = measure(reduce_call, args.warmup, args.repeats)
            graph_w13 = measure(
                graph_w13_call or capture(w13_call, args.warmup, args.graph_calls),
                args.warmup,
                args.repeats,
            )
            graph_w2 = measure(
                graph_w2_call or capture(w2_call, args.warmup, args.graph_calls),
                args.warmup,
                args.repeats,
            )
            graph_reduce = measure(
                capture(reduce_call, args.warmup, args.graph_calls),
                args.warmup,
                args.repeats,
            )
            for values in (graph_w13, graph_w2, graph_reduce):
                for key in tuple(values):
                    values[key] /= args.graph_calls
            results.append(
                {
                    "tokens": tokens,
                    "num_experts": NUM_EXPERTS,
                    "block_m": block_m,
                    "block_n": block_n,
                    "rotate_experts": args.rotate_experts,
                    "eager_event": {
                        "w13": eager_w13,
                        "w2": eager_w2,
                        "reduce": eager_reduce,
                    },
                    "graph_event": {
                        "w13": graph_w13,
                        "w2": graph_w2,
                        "reduce": graph_reduce,
                    },
                }
            )

    serialized = json.dumps(results, indent=2, sort_keys=True)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
