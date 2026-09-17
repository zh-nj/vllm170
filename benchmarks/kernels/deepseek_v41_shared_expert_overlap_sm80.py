# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate shared-expert stream overlap for DSV4.1 decode on SM80."""

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
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_mxfp8_marlin_linear,
    prepare_mxfp8_layer_for_marlin,
)

HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 2304
NUM_EXPERTS = 384
TOPK = 6


def _measure(call: Callable[[], object], warmup: int, repeats: int) -> dict[str, float]:
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
    samples = [
        start.elapsed_time(end) * 1000.0
        for start, end in zip(starts, ends, strict=True)
    ]
    return {
        "mean_us": statistics.fmean(samples),
        "p50_us": statistics.median(samples),
        "p90_us": sorted(samples)[min(repeats - 1, int(repeats * 0.9))],
    }


def _make_marlin_layer(n: int, k: int, generator: torch.Generator) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n
    layer.weight = torch.nn.Parameter(
        torch.randn(
            (n, k), dtype=torch.bfloat16, device="cuda", generator=generator
        ).to(torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.randint(
            121,
            128,
            (n, k // 32),
            dtype=torch.uint8,
            device="cuda",
            generator=generator,
        ),
        requires_grad=False,
    )
    prepare_mxfp8_layer_for_marlin(layer)
    return layer


def _apply_marlin(layer: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    return apply_mxfp8_marlin_linear(
        input=inputs,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        workspace=layer.workspace,
        size_n=layer.output_size_per_partition,
        size_k=layer.input_size_per_partition,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.layers <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("layers and repeats must be positive; warmup may be zero")

    generator = torch.Generator(device="cuda").manual_seed(20260914)
    hidden = torch.randn(
        (1, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    shared_w13 = [
        _make_marlin_layer(2 * INTERMEDIATE_SIZE, HIDDEN_SIZE, generator)
        for _ in range(args.layers)
    ]
    shared_w2 = [
        _make_marlin_layer(HIDDEN_SIZE, INTERMEDIATE_SIZE, generator)
        for _ in range(args.layers)
    ]
    shared_intermediates = [
        torch.empty((1, INTERMEDIATE_SIZE), dtype=torch.bfloat16, device="cuda")
        for _ in range(args.layers)
    ]
    shared_outputs = [torch.empty_like(hidden) for _ in range(args.layers)]

    w13 = torch.empty(
        (NUM_EXPERTS, HIDDEN_SIZE // 32, INTERMEDIATE_SIZE * 2 // 8, 128),
        dtype=torch.uint8,
        device="cuda",
    ).random_(0, 256)
    w13_shift2 = torch.empty(
        (NUM_EXPERTS, HIDDEN_SIZE // 128, INTERMEDIATE_SIZE * 2 // 8, 8),
        dtype=torch.uint8,
        device="cuda",
    ).random_(0, 256)
    w13_scale = torch.ones(
        (NUM_EXPERTS, INTERMEDIATE_SIZE * 2), dtype=torch.float32, device="cuda"
    )
    w2 = torch.empty(
        (NUM_EXPERTS, INTERMEDIATE_SIZE // 32, HIDDEN_SIZE // 8, 128),
        dtype=torch.uint8,
        device="cuda",
    ).random_(0, 256)
    w2_shift2 = torch.empty(
        (NUM_EXPERTS, INTERMEDIATE_SIZE // 128, HIDDEN_SIZE // 8, 8),
        dtype=torch.uint8,
        device="cuda",
    ).random_(0, 256)
    w2_scale = torch.ones(
        (NUM_EXPERTS, HIDDEN_SIZE), dtype=torch.float32, device="cuda"
    )
    routing_weights = torch.full(
        (1, TOPK), 1 / TOPK, dtype=torch.float32, device="cuda"
    )
    a13_q = torch.empty((1, HIDDEN_SIZE), dtype=torch.int8, device="cuda").random_(
        -127, 128
    )
    a13_scale = torch.ones((1,), dtype=torch.float32, device="cuda")
    gate_up = torch.empty(
        (TOPK, 2 * INTERMEDIATE_SIZE), dtype=torch.bfloat16, device="cuda"
    )
    a2_q = torch.empty(
        (TOPK, INTERMEDIATE_SIZE), dtype=torch.int8, device="cuda"
    ).random_(-127, 128)
    a2_scale = torch.ones((TOPK,), dtype=torch.float32, device="cuda")
    routed_out = torch.empty((TOPK, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda")
    output = torch.empty((1, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda")
    route_metadata = []
    for layer_index in range(args.layers):
        routing_ids = (
            (torch.arange(TOPK, dtype=torch.int32, device="cuda") + layer_index * TOPK)
            .remainder(NUM_EXPERTS)
            .view(1, TOPK)
        )
        route_metadata.append(
            (*moe_align_block_size(routing_ids, 16, NUM_EXPERTS), TOPK)
        )

    def shared(layer_index: int) -> None:
        gate = _apply_marlin(shared_w13[layer_index], hidden)
        intermediate = shared_intermediates[layer_index]
        torch.ops._C.silu_and_mul_with_clamp(intermediate, gate, 10.0, 1.0, 0.0)
        shared_outputs[layer_index].copy_(
            _apply_marlin(shared_w2[layer_index], intermediate)
        )

    def routed(layer_index: int) -> None:
        sorted_ids, expert_ids, num_post_pad, num_routed = route_metadata[layer_index]
        torch.ops._moe_C.dsv4_mxfp4_int8_sm80_moe_gemm(
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
            16,
            128,
            False,
        )
        torch.ops._moe_C.dsv4_mxfp4_int8_sm80_moe_gemm(
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
            16,
            128,
            True,
        )

    aux = torch.cuda.Stream()
    input_ready = torch.cuda.Event()
    output_ready = torch.cuda.Event()

    def serial_all() -> None:
        for layer_index in range(args.layers):
            shared(layer_index)
            routed(layer_index)

    def overlap_all() -> None:
        main_stream = torch.cuda.current_stream()
        for layer_index in range(args.layers):
            input_ready.record(main_stream)
            with torch.cuda.stream(aux):
                aux.wait_event(input_ready)
                shared(layer_index)
                output_ready.record(aux)
            routed(layer_index)
            main_stream.wait_event(output_ready)

    def shared_all() -> None:
        for layer_index in range(args.layers):
            shared(layer_index)

    def routed_all() -> None:
        for layer_index in range(args.layers):
            routed(layer_index)

    calls = {
        "shared_only": shared_all,
        "routed_w13_w2_only": routed_all,
        "serial_shared_then_routed": serial_all,
        "overlap_shared_and_routed": overlap_all,
    }
    results = {}
    for name, call in calls.items():
        graph = torch.cuda.CUDAGraph()
        for _ in range(args.warmup):
            call()
        torch.accelerator.synchronize()
        with torch.cuda.graph(graph):
            call()
        timing = _measure(graph.replay, args.warmup, args.repeats)
        results[name] = {key: value / args.layers for key, value in timing.items()}

    payload = {
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "layers_per_replay": args.layers,
        "shape": {
            "tokens": 1,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "routed_experts": NUM_EXPERTS,
            "topk": TOPK,
        },
        "graph_event_per_layer": results,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
