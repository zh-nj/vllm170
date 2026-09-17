# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure the SM80 DSV4.1 M=1 output-projection components."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch

from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_mxfp8_marlin_linear,
    prepare_mxfp8_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _fused_inverse_rope_gptj


@triton.jit
def _grouped_m1_mm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    groups,
    K,
    N,
    stride_x_g,
    stride_w_g,
    stride_y_g,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_g = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs_k
        x = tl.load(x_ptr + pid_g * stride_x_g + k, mask=k < K, other=0.0)
        w = tl.load(
            w_ptr + pid_g * stride_w_g + k[:, None] * N + offs_n[None, :],
            mask=(k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.reshape(tl.dot(x[None, :], w, out_dtype=tl.float32), (BLOCK_N,))
    tl.store(y_ptr + pid_g * stride_y_g + offs_n, acc.to(tl.bfloat16), mask=offs_n < N)


def grouped_m1_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    block_k: int,
    block_n: int,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    groups, _, k = x.shape
    n = weight.shape[-1]
    _grouped_m1_mm_kernel[(groups, triton.cdiv(n, block_n))](
        x,
        weight,
        out,
        groups,
        k,
        n,
        x.stride(0),
        weight.stride(0),
        out.stride(0),
        BLOCK_K=block_k,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def measure(
    call: Callable[[], None],
    warmup: int,
    repeats: int,
    calls_per_sample: int = 1,
) -> dict[str, float]:
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
        start.elapsed_time(end) / calls_per_sample
        for start, end in zip(starts, ends, strict=True)
    ]
    return {
        "mean_ms": statistics.fmean(samples),
        "p50_ms": statistics.median(samples),
        "p90_ms": sorted(samples)[min(len(samples) - 1, int(len(samples) * 0.9))],
    }


def benchmark(
    name: str,
    call: Callable[[], None],
    warmup: int,
    repeats: int,
    graph_calls: int,
) -> dict:
    event = measure(call, warmup, repeats)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_calls):
            call()
    graph_replay = measure(graph.replay, warmup, repeats, graph_calls)
    return {"component": name, "event": event, "graph_replay": graph_replay}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--graph-calls", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats <= 0 or args.graph_calls <= 0:
        raise ValueError(
            "warmup must be non-negative; repeats and graph-calls must be positive"
        )

    torch.manual_seed(20260913)
    device = torch.device("cuda")
    tokens, heads, head_dim = 1, 64, 512
    groups, rank, rope_dim = 8, 1024, 64
    group_input = heads // groups * head_dim
    o = torch.randn((tokens, heads, head_dim), dtype=torch.bfloat16, device=device)
    positions = torch.tensor([32000], dtype=torch.int64, device=device)
    cos_sin = torch.randn((65536, rope_dim), dtype=torch.float32, device=device)
    inverse = torch.empty_like(o)
    grouped_weight_mxfp8 = torch.randn(
        (groups, rank, group_input), dtype=torch.bfloat16, device=device
    ).to(torch.float8_e4m3fn)
    grouped_weight_scale = torch.full(
        (groups, rank, group_input // 32),
        127,
        dtype=torch.uint8,
        device=device,
    )
    grouped_weight = torch.stack(
        [
            dequant_mxfp8_to_bf16(grouped_weight_mxfp8[g], grouped_weight_scale[g])
            for g in range(groups)
        ]
    )
    marlin_layers = []
    for g in range(groups):
        layer = torch.nn.Module()
        layer.input_size_per_partition = group_input
        layer.output_size_per_partition = rank
        layer.weight = torch.nn.Parameter(
            grouped_weight_mxfp8[g].clone(), requires_grad=False
        )
        layer.weight_scale = torch.nn.Parameter(
            grouped_weight_scale[g].clone(), requires_grad=False
        )
        prepare_mxfp8_layer_for_marlin(layer)
        marlin_layers.append(layer)
    z = torch.empty((tokens, groups, rank), dtype=torch.bfloat16, device=device)
    z_flat = z.view(tokens, groups * rank)
    wo_b = torch.randn((groups * rank, 5120), dtype=torch.bfloat16, device=device)
    output = torch.empty((tokens, 5120), dtype=torch.bfloat16, device=device)

    def inverse_rope() -> None:
        _fused_inverse_rope_gptj(o, positions, cos_sin, rope_dim)

    def inverse_rope_into() -> None:
        result = _fused_inverse_rope_gptj(o, positions, cos_sin, rope_dim)
        inverse.copy_(result)

    def wo_a_einsum() -> None:
        z.copy_(
            torch.einsum(
                "tgd,grd->tgr",
                inverse.view(tokens, groups, group_input),
                grouped_weight,
            )
        )

    def wo_a_bmm() -> None:
        z.copy_(
            torch.bmm(
                inverse.view(tokens, groups, group_input).transpose(0, 1),
                grouped_weight.transpose(1, 2),
            ).transpose(0, 1)
        )

    def wo_a_marlin() -> None:
        grouped_input = inverse.view(tokens, groups, group_input)
        for g, layer in enumerate(marlin_layers):
            z[:, g].copy_(
                apply_mxfp8_marlin_linear(
                    input=grouped_input[:, g],
                    weight=layer.weight,
                    weight_scale=layer.weight_scale,
                    workspace=layer.workspace,
                    size_n=rank,
                    size_k=group_input,
                )
            )

    def wo_b_mm() -> None:
        torch.mm(z_flat, wo_b, out=output)

    def full_bf16() -> None:
        inverse_rope_into()
        wo_a_einsum()
        wo_b_mm()

    inverse_rope_into()
    wo_a_bmm()
    wo_a_reference = z.clone()
    wo_a_marlin()
    torch.accelerator.synchronize()
    torch.testing.assert_close(z, wo_a_reference, rtol=2e-2, atol=1.0)

    results = [
        benchmark(
            "inverse_rope", inverse_rope, args.warmup, args.repeats, args.graph_calls
        ),
        benchmark(
            "inverse_rope_copy",
            inverse_rope_into,
            args.warmup,
            args.repeats,
            args.graph_calls,
        ),
        benchmark(
            "wo_a_einsum", wo_a_einsum, args.warmup, args.repeats, args.graph_calls
        ),
        benchmark("wo_a_bmm", wo_a_bmm, args.warmup, args.repeats, args.graph_calls),
        benchmark(
            "wo_a_8x_marlin",
            wo_a_marlin,
            args.warmup,
            args.repeats,
            args.graph_calls,
        ),
        benchmark("wo_b_bf16_mm", wo_b_mm, args.warmup, args.repeats, args.graph_calls),
        benchmark("full_bf16", full_bf16, args.warmup, args.repeats, args.graph_calls),
    ]
    serialized = json.dumps(results, indent=2, sort_keys=True)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
