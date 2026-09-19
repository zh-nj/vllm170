# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split the mHC prenorm cost into its GEMM and its fused epilogue.

The decode path used to run the prenorm GEMM with n_splits=1, i.e. six
threadblocks for the whole 24 x 20480 reduction. This sweeps the split factor
for the bare GEMM and times the epilogue on its own, so the remaining time can
be attributed instead of guessed.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from vllm.model_executor.kernels.mhc.tilelang import _tilelang_hc_prenorm_gemm
from vllm.model_executor.kernels.mhc.warmup import MHC_PRE_NORM_KERNEL

HIDDEN_SIZE = 5120
HC_MULT = 4
N_OUT = 24
INPUT_SIZE = HC_MULT * HIDDEN_SIZE


def graph_time(call, calls: int, samples: int) -> float:
    graph = torch.cuda.CUDAGraph()
    for _ in range(3):
        call()
    torch.accelerator.synchronize()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            call()
    torch.accelerator.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        graph.replay()
        end.record()
    torch.accelerator.synchronize()
    return statistics.median(
        s.elapsed_time(e) * 1000.0 / calls for s, e in zip(starts, ends)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, nargs="+", default=[1, 2, 4, 5, 10, 20])
    ap.add_argument("--graph-calls", type=int, default=40)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(20260918)
    device = torch.device("cuda")
    x = torch.randn((1, INPUT_SIZE), dtype=torch.bfloat16, device=device)
    fn = torch.randn((N_OUT, INPUT_SIZE), dtype=torch.float32, device=device) * 1e-4
    hc_scale = torch.ones((3,), dtype=torch.float32, device=device)
    hc_base = torch.zeros((N_OUT,), dtype=torch.float32, device=device)
    residual = torch.randn((1, HC_MULT, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
    norm_weight = torch.ones((HIDDEN_SIZE,), dtype=torch.bfloat16, device=device)
    post = torch.empty((1, HC_MULT), dtype=torch.float32, device=device)
    comb = torch.empty((1, HC_MULT * HC_MULT), dtype=torch.float32, device=device)
    layer_input = torch.empty((1, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
    next_pre_mix = torch.empty((1, HC_MULT), dtype=torch.float32, device=device)

    gemm = {}
    reference = None
    for splits in args.splits:
        if INPUT_SIZE % splits or (INPUT_SIZE // splits) % 1024:
            gemm[splits] = None
            continue
        mixes = torch.empty(splits, 1, N_OUT, dtype=torch.float32, device=device)
        sqrsum = torch.empty(splits, 1, dtype=torch.float32, device=device)

        def call(m=mixes, s=sqrsum):
            _tilelang_hc_prenorm_gemm(x, fn, m, s, INPUT_SIZE, 1, n_splits=m.shape[0])

        call()
        torch.accelerator.synchronize()
        total = mixes.sum(dim=0)
        if reference is None:
            reference = total.clone()
        else:
            torch.testing.assert_close(total, reference, rtol=2e-4, atol=2e-4)
        gemm[splits] = graph_time(call, args.graph_calls, args.samples)

    # Epilogue alone, with a 20-split input, so its own cost is visible.
    mixes = torch.randn((20, 1, N_OUT), dtype=torch.float32, device=device)
    sqrsum = torch.rand((20, 1), dtype=torch.float32, device=device) * INPUT_SIZE

    def epilogue():
        MHC_PRE_NORM_KERNEL(
            mixes,
            sqrsum,
            hc_scale,
            hc_base,
            residual,
            post,
            comb,
            layer_input,
            norm_weight,
            post,
            next_pre_mix,
            hidden_size=HIDDEN_SIZE,
            rms_eps=1e-6,
            hc_pre_eps=1e-6,
            hc_sinkhorn_eps=1e-6,
            hc_post_mult_value=1.0,
            sinkhorn_repeat=20,
            norm_eps=1e-6,
            hc_mult=HC_MULT,
            use_pre_mix_in=False,
            save_pre_mix=True,
            rms_numel=INPUT_SIZE,
        )

    epilogue_us = graph_time(epilogue, args.graph_calls, args.samples)
    result = {
        "gemm_us_by_splits": {str(k): v for k, v in gemm.items()},
        "epilogue_us": epilogue_us,
        "w13_bytes_per_call": N_OUT * INPUT_SIZE * 4,
    }
    print(json.dumps(result, indent=2))
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
