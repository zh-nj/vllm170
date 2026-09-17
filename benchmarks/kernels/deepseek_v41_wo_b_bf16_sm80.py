# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark M=1 BF16 ``wo_b`` GEMM alternatives on SM80."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _m1_mm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    K,
    N,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs_k
        x = tl.load(x_ptr + k, mask=k < K, other=0.0).to(tl.bfloat16)
        w = tl.load(
            w_ptr + k[:, None] * N + offs_n[None, :],
            mask=(k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.bfloat16)
        acc += tl.reshape(tl.dot(x[None, :], w, out_dtype=tl.float32), (BLOCK_N,))
    tl.store(y_ptr + offs_n, acc.to(tl.bfloat16), mask=offs_n < N)


@triton.jit
def _m1_mm_bf16_acc_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    K,
    N,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N,), dtype=tl.bfloat16)
    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs_k
        x = tl.load(x_ptr + k, mask=k < K, other=0.0).to(tl.bfloat16)
        w = tl.load(
            w_ptr + k[:, None] * N + offs_n[None, :],
            mask=(k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.bfloat16)
        acc = (
            acc
            + tl.reshape(tl.dot(x[None, :], w, out_dtype=tl.float32), (BLOCK_N,)).to(
                tl.bfloat16
            )
        ).to(tl.bfloat16)
    tl.store(y_ptr + offs_n, acc, mask=offs_n < N)


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


def _make_candidate(
    x,
    weight,
    output,
    k,
    n,
    block_k,
    block_n,
    num_warps,
    num_stages,
):
    def candidate() -> None:
        _m1_mm_kernel[(triton.cdiv(n, block_n),)](
            x,
            weight,
            output,
            k,
            n,
            BLOCK_K=block_k,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--n", type=int, default=5120)
    parser.add_argument("--block-k", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--block-n", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--warps", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--stages", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--graph-calls", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(20260914)
    x = torch.randn((1, args.k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((args.k, args.n), device="cuda", dtype=torch.bfloat16)
    reference = torch.mm(x, weight)
    results = []
    for block_k in args.block_k:
        for block_n in args.block_n:
            for num_warps in args.warps:
                for num_stages in args.stages:
                    if block_k <= 0 or block_n <= 0:
                        raise ValueError("block sizes must be positive")
                    output = torch.empty_like(reference)

                    candidate = _make_candidate(
                        x,
                        weight,
                        output,
                        args.k,
                        args.n,
                        block_k,
                        block_n,
                        num_warps,
                        num_stages,
                    )

                    candidate()
                    torch.accelerator.synchronize()
                    # The reduction order differs from cuBLAS.  BF16 output
                    # must stay within one ulp plus the usual GEMM tolerance.
                    torch.testing.assert_close(output, reference, rtol=2e-2, atol=1.0)
                    graph_candidate = _graph(candidate, args.warmup, args.graph_calls)
                    graph_event = _measure(graph_candidate, args.warmup, args.repeats)
                    for key in graph_event:
                        graph_event[key] /= args.graph_calls
                    row = {
                        "block_k": block_k,
                        "block_n": block_n,
                        "num_warps": num_warps,
                        "num_stages": num_stages,
                        "graph": graph_event,
                    }
                    results.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)

    def baseline() -> None:
        torch.mm(x, weight, out=reference)

    baseline_graph = _graph(baseline, args.warmup, args.graph_calls)
    baseline_event = _measure(baseline_graph, args.warmup, args.repeats)
    for key in baseline_event:
        baseline_event[key] /= args.graph_calls
    payload = {"k": args.k, "n": args.n, "baseline": baseline_event, "results": results}
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
