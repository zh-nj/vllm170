# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time the two mHC entry points exactly as a decode layer calls them.

A decoder layer calls ``mhc_post_tilelang`` and ``mhc_pre_delayed_tilelang``
twice each, so the per-layer cost is 2 * (pre + post). The published component
benchmark times the inner GEMM and the elementwise stages separately; this
times the production call as the model makes it, inside one CUDA graph, so the
per-token total follows without any assumption about how many kernels hide
inside each entry point.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from vllm.model_executor.kernels.mhc.tilelang import (
    mhc_post_tilelang,
    mhc_pre_delayed_tilelang,
)

HIDDEN_SIZE = 5120
HC_MULT = 4
N_OUT = 24


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
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--graph-calls", type=int, default=40)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(20260918)
    device = torch.device("cuda")
    t = args.tokens
    residual = torch.randn(
        (t, HC_MULT, HIDDEN_SIZE), dtype=torch.bfloat16, device=device
    )
    fn = torch.randn((N_OUT, HC_MULT * HIDDEN_SIZE), dtype=torch.float32, device=device) * 1e-4
    hc_scale = torch.ones((3,), dtype=torch.float32, device=device)
    hc_base = torch.zeros((N_OUT,), dtype=torch.float32, device=device)
    norm_weight = torch.ones((HIDDEN_SIZE,), dtype=torch.bfloat16, device=device)
    # fn projects the whole hc_mult*hidden state, so leave x=None exactly as
    # the layer does and let the entry point view the residual.
    pre_mix = torch.zeros((t, HC_MULT), dtype=torch.float32, device=device)

    pre = mhc_pre_delayed_tilelang(
        residual,
        fn,
        hc_scale,
        hc_base,
        1e-6,
        1e-6,
        1e-6,
        1.0,
        20,
        pre_mix=pre_mix,
        norm_weight=norm_weight,
        norm_eps=1e-6,
    )
    post_mix, res_mix, x_out = pre[0], pre[1], pre[2]

    def pre_call():
        mhc_pre_delayed_tilelang(
            residual,
            fn,
            hc_scale,
            hc_base,
            1e-6,
            1e-6,
            1e-6,
            1.0,
            20,
            pre_mix=pre_mix,
            norm_weight=norm_weight,
            norm_eps=1e-6,
        )

    def post_call():
        # Shape-compatible with the tensors mhc_pre_delayed hands the layer.
        mhc_post_tilelang(x_out, residual, post_mix, res_mix)

    pre_us = graph_time(pre_call, args.graph_calls, args.samples)
    post_us = graph_time(post_call, args.graph_calls, args.samples)
    per_layer = 2 * (pre_us + post_us)
    result = {
        "tokens": t,
        "pre_delayed_us": pre_us,
        "post_us": post_us,
        "calls_per_layer": 4,
        "ms_per_layer": per_layer / 1000.0,
        "ms_per_token_40_layers": per_layer * 40 / 1000.0,
    }
    print(json.dumps(result, indent=2))
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
