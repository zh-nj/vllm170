# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-Event benchmark for fused SwiGLU + per-token INT8 quantization."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from vllm.model_executor.layers.quantization.utils.mxfp4_int8_sm80 import (
    quantize_per_token_int8,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _silu_mul_quant_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    M,
    K,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    mask = cols < K
    gate = tl.load(x_ptr + row * (2 * K) + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + row * (2 * K) + K + cols, mask=mask, other=0.0).to(tl.float32)
    # Match the C++ path: SiLU is narrowed to BF16 before multiplying by up,
    # and the result is narrowed to BF16 before the quantizer reads it.
    activated = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16)
    values = (activated.to(tl.float32) * up).to(tl.bfloat16).to(tl.float32)
    max_abs = tl.max(tl.abs(tl.where(mask, values, 0.0)), axis=0)
    scale = tl.where(max_abs == 0.0, 1.0, max_abs / 127.0)
    q_i32 = tl.inline_asm_elementwise(
        "cvt.rni.s32.f32 $0, $1;",
        "=r,f",
        [values / scale],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    q_i32 = tl.minimum(tl.maximum(q_i32, -127), 127).to(tl.int8)
    tl.store(q_ptr + row * K + cols, q_i32, mask=mask)
    tl.store(scale_ptr + row, scale)


def fused_silu_mul_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, k2 = x.shape
    if x.dtype != torch.bfloat16 or not x.is_cuda or k2 % 2:
        raise ValueError("expected CUDA BF16 [M, 2*K] input")
    k = k2 // 2
    q = torch.empty((m, k), device=x.device, dtype=torch.int8)
    scale = torch.empty((m,), device=x.device, dtype=torch.float32)
    _silu_mul_quant_kernel[(m,)](x, q, scale, m, k, BLOCK_K=triton.next_power_of_2(k))
    return q, scale


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
        "mean_ms": statistics.fmean(samples),
        "p50_ms": statistics.median(samples),
        "p90_ms": sorted(samples)[min(repeats - 1, int(repeats * 0.9))],
    }


def _capture(call, warmup: int, graph_calls: int):
    for _ in range(warmup):
        call()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_calls):
            call()
    return graph.replay


def _make_calls(x, activation, quantized, scale_out):
    def separate() -> None:
        torch.ops._C.silu_and_mul(activation, x)
        q, scale = quantize_per_token_int8(activation)
        quantized.copy_(q)
        scale_out.copy_(scale)

    def fused() -> None:
        q, scale = fused_silu_mul_quant(x)
        quantized.copy_(q)
        scale_out.copy_(scale)

    return separate, fused


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[6])
    parser.add_argument("--k", type=int, default=2304)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--graph-calls", type=int, default=40)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.k <= 0 or any(rows <= 0 for rows in args.rows):
        raise ValueError("rows and k must be positive")

    torch.manual_seed(4242)
    results = []
    for rows in args.rows:
        x = torch.randn((rows, args.k * 2), device="cuda", dtype=torch.bfloat16)
        activation = torch.empty((rows, args.k), device="cuda", dtype=torch.bfloat16)
        separate_q = torch.empty((rows, args.k), device="cuda", dtype=torch.int8)
        separate_scale = torch.empty((rows,), device="cuda", dtype=torch.float32)

        separate, fused = _make_calls(x, activation, separate_q, separate_scale)

        separate()
        reference_q = separate_q.clone()
        reference_scale = separate_scale.clone()
        fused()
        torch.accelerator.synchronize()
        torch.testing.assert_close(separate_q, reference_q, rtol=0, atol=0)
        torch.testing.assert_close(separate_scale, reference_scale, rtol=0, atol=0)

        separate_graph = _capture(separate, args.warmup, args.graph_calls)
        fused_graph = _capture(fused, args.warmup, args.graph_calls)
        separate_event = _measure(separate_graph, args.warmup, args.repeats)
        fused_event = _measure(fused_graph, args.warmup, args.repeats)
        for values in (separate_event, fused_event):
            for key in values:
                values[key] /= args.graph_calls
        results.append(
            {
                "rows": rows,
                "k": args.k,
                "graph_calls": args.graph_calls,
                "separate": separate_event,
                "fused": fused_event,
                "speedup": separate_event["p50_ms"] / fused_event["p50_ms"] - 1.0,
            }
        )

    serialized = json.dumps(results, indent=2, sort_keys=True)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
