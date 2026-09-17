# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the DSV4.1 M=1 TileLang mHC prenorm projection on SM80."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch

from vllm.model_executor.kernels.mhc.tilelang import _torch_hc_prenorm_gemm
from vllm.model_executor.kernels.mhc.tilelang_kernels import (
    hc_prenorm_gemm_tilelang,
    mhc_fused_tilelang,
    mhc_post_tilelang,
)
from vllm.model_executor.kernels.mhc.warmup import MHC_PRE_NORM_KERNEL

HIDDEN_SIZE = 5120
HC_MULT = 4
N_OUT = HC_MULT * (HC_MULT + 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--tile-n", type=int, nargs="+", default=[4])
    parser.add_argument("--n-thr", type=int, nargs="+", default=[1024])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def measure(call: Callable[[], None], warmup: int, repeats: int) -> dict[str, float]:
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


def main() -> None:
    args = parse_args()
    if args.tokens <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("tokens/repeats must be positive and warmup non-negative")
    if any(tile_n <= 0 or tile_n > 32 for tile_n in args.tile_n):
        raise ValueError("tile-n must be in [1, 32]")
    if any(n_thr not in (128, 256, 512, 1024) for n_thr in args.n_thr):
        raise ValueError("n-thr must be one of 128, 256, 512, or 1024")

    torch.manual_seed(20260913)
    device = torch.device("cuda")
    k = HC_MULT * HIDDEN_SIZE
    x = torch.randn((args.tokens, k), dtype=torch.bfloat16, device=device)
    fn = torch.randn((N_OUT, k), dtype=torch.float32, device=device) * 1e-4
    reference_out = torch.empty(
        (1, args.tokens, N_OUT), dtype=torch.float32, device=device
    )
    reference_sqrsum = torch.empty((1, args.tokens), dtype=torch.float32, device=device)
    _torch_hc_prenorm_gemm(x, fn, reference_out, reference_sqrsum)

    results = []
    for n_thr in args.n_thr:
        if k % n_thr:
            continue
        for tile_n in args.tile_n:
            out = torch.empty_like(reference_out)
            sqrsum = torch.empty_like(reference_sqrsum)

            call = partial(
                hc_prenorm_gemm_tilelang,
                x,
                fn,
                out,
                sqrsum,
                HIDDEN_SIZE,
                HC_MULT,
                N_OUT,
                n_thr,
                tile_n,
                1,
            )

            call()
            torch.accelerator.synchronize()
            torch.testing.assert_close(out, reference_out, atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(sqrsum, reference_sqrsum, atol=8.0, rtol=5e-4)
            event = measure(call, args.warmup, args.repeats)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
            graph_replay = measure(graph.replay, args.warmup, args.repeats)
            results.append(
                {
                    "component": "prenorm_gemm",
                    "tokens": args.tokens,
                    "hidden_size": HIDDEN_SIZE,
                    "hc_mult": HC_MULT,
                    "n_out": N_OUT,
                    "n_thr": n_thr,
                    "tile_n": tile_n,
                    "event": event,
                    "graph_replay": graph_replay,
                }
            )

    residual = x.view(args.tokens, HC_MULT, HIDDEN_SIZE)
    hc_scale = torch.randn((3,), dtype=torch.float32, device=device) * 0.1
    hc_base = torch.randn((N_OUT,), dtype=torch.float32, device=device) * 0.1
    norm_weight = torch.randn((HIDDEN_SIZE,), dtype=torch.bfloat16, device=device)
    pre_mix = torch.randn((args.tokens, HC_MULT), dtype=torch.float32, device=device)
    post_mix = torch.empty_like(pre_mix)
    comb_mix = torch.empty(
        (args.tokens, HC_MULT * HC_MULT), dtype=torch.float32, device=device
    )
    layer_input = torch.empty(
        (args.tokens, HIDDEN_SIZE), dtype=torch.bfloat16, device=device
    )
    next_pre_mix = torch.empty_like(pre_mix)

    def pre_norm_finalize() -> None:
        MHC_PRE_NORM_KERNEL(
            reference_out,
            reference_sqrsum,
            hc_scale,
            hc_base,
            residual,
            post_mix,
            comb_mix,
            layer_input,
            norm_weight,
            pre_mix,
            next_pre_mix,
            hidden_size=HIDDEN_SIZE,
            rms_eps=1e-20,
            hc_pre_eps=1e-6,
            hc_sinkhorn_eps=1e-6,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=20,
            norm_eps=1e-20,
            hc_mult=HC_MULT,
            use_pre_mix_in=True,
            save_pre_mix=True,
            rms_numel=k,
        )

    pre_norm_finalize()
    torch.accelerator.synchronize()
    finalize_event = measure(pre_norm_finalize, args.warmup, args.repeats)
    finalize_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(finalize_graph):
        pre_norm_finalize()
    finalize_replay = measure(finalize_graph.replay, args.warmup, args.repeats)
    results.append(
        {
            "component": "pre_norm_finalize",
            "tokens": args.tokens,
            "hidden_size": HIDDEN_SIZE,
            "hc_mult": HC_MULT,
            "event": finalize_event,
            "graph_replay": finalize_replay,
        }
    )

    attention_output = torch.randn(
        (args.tokens, HIDDEN_SIZE), dtype=torch.bfloat16, device=device
    )
    residual_out = torch.empty_like(residual)

    def post() -> None:
        mhc_post_tilelang(
            comb_mix.view(args.tokens, HC_MULT, HC_MULT),
            residual,
            post_mix,
            attention_output,
            residual_out,
            HC_MULT,
            HIDDEN_SIZE,
        )

    post()
    torch.accelerator.synchronize()
    post_event = measure(post, args.warmup, args.repeats)
    post_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(post_graph):
        post()
    post_replay = measure(post_graph.replay, args.warmup, args.repeats)
    results.append(
        {
            "component": "post",
            "tokens": args.tokens,
            "hidden_size": HIDDEN_SIZE,
            "hc_mult": HC_MULT,
            "event": post_event,
            "graph_replay": post_replay,
        }
    )

    separate_mix = torch.empty_like(reference_out)
    separate_sqrsum = torch.empty_like(reference_sqrsum)
    fused_mix = torch.empty((4, args.tokens, N_OUT), dtype=torch.float32, device=device)
    fused_sqrsum = torch.empty((4, args.tokens), dtype=torch.float32, device=device)
    fused_residual = torch.empty_like(residual)

    def separate_post_projection() -> None:
        post()
        hc_prenorm_gemm_tilelang(
            residual_out.view(args.tokens, k),
            fn,
            separate_mix,
            separate_sqrsum,
            HIDDEN_SIZE,
            HC_MULT,
            N_OUT,
            1024,
            4,
            1,
        )

    def fused_post_projection() -> None:
        mhc_fused_tilelang(
            comb_mix.view(args.tokens, HC_MULT, HC_MULT),
            residual,
            post_mix,
            attention_output,
            fn.view(N_OUT, HC_MULT, HIDDEN_SIZE),
            fused_mix,
            fused_sqrsum,
            fused_residual,
            HC_MULT,
            HIDDEN_SIZE,
            N_OUT,
            tile_n=2,
            split_k=4,
        )

    separate_post_projection()
    fused_post_projection()
    torch.accelerator.synchronize()
    torch.testing.assert_close(fused_residual, residual_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        fused_mix.sum(dim=0), separate_mix[0], atol=0.1, rtol=2e-2
    )
    torch.testing.assert_close(
        fused_sqrsum.sum(dim=0), separate_sqrsum[0], atol=32.0, rtol=2e-3
    )
    for component, call in (
        ("separate_post_projection", separate_post_projection),
        ("fused_post_projection", fused_post_projection),
    ):
        event = measure(call, args.warmup, args.repeats)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graph_replay = measure(graph.replay, args.warmup, args.repeats)
        results.append(
            {
                "component": component,
                "tokens": args.tokens,
                "hidden_size": HIDDEN_SIZE,
                "hc_mult": HC_MULT,
                "event": event,
                "graph_replay": graph_replay,
            }
        )

    serialized = json.dumps(results, indent=2, sort_keys=True)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
