# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Control for the MXFP4 MoE GEMM: read the same expert slabs, do no math.

The grouped GEMM for one decode token touches six experts' packed W13 and W2
slabs, scattered across the expert array. This kernel walks exactly those
addresses with a plain coalesced read so the two can be compared directly: the
gap is then the GEMM's own inefficiency, not the DRAM pattern a repacked layout
would have to fix anyway.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _slab_read_kernel(src, ids, out, slab_numel, n_ids, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    per_id = tl.num_programs(0) // n_ids
    slot = tl.minimum(pid // per_id, n_ids - 1)
    expert = tl.load(ids + slot).to(tl.int64)
    base = expert * slab_numel
    offs = (pid % per_id) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < slab_numel
    values = tl.load(src + base + offs, mask=mask, other=0)
    acc = tl.sum(values.to(tl.int32), axis=0)
    if acc == 1234567:
        tl.store(out, acc)


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
    per_call = sorted(s.elapsed_time(e) * 1000.0 / calls for s, e in zip(starts, ends))
    return statistics.median(per_call)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--graph-calls", type=int, default=40)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(7)
    device = torch.device("cuda")
    e, topk = args.experts, args.topk

    w13_bytes = 160 * 576 * 128
    w2_bytes = 72 * 640 * 128
    w13_flat = torch.empty(e * w13_bytes, dtype=torch.uint8, device=device).random_(
        0, 256
    )
    w2_flat = torch.empty(e * w2_bytes, dtype=torch.uint8, device=device).random_(
        0, 256
    )
    ids = torch.arange(topk, dtype=torch.int32, device=device)
    sink = torch.zeros(1, dtype=torch.int32, device=device)

    def read_w13():
        _slab_read_kernel[(topk * triton.cdiv(w13_bytes, args.block),)](
            w13_flat, ids, sink, w13_bytes, topk, BLOCK=args.block
        )

    def read_w2():
        _slab_read_kernel[(topk * triton.cdiv(w2_bytes, args.block),)](
            w2_flat, ids, sink, w2_bytes, topk, BLOCK=args.block
        )

    def read_both():
        read_w13()
        read_w2()

    traffic_w13 = topk * w13_bytes
    traffic_w2 = topk * w2_bytes
    result = {
        "experts": e,
        "topk": topk,
        "w13_slab_bytes_per_expert": w13_bytes,
        "w2_slab_bytes_per_expert": w2_bytes,
        "read_w13_us": graph_time(read_w13, args.graph_calls, args.samples),
        "read_w2_us": graph_time(read_w2, args.graph_calls, args.samples),
        "read_both_us": graph_time(read_both, args.graph_calls, args.samples),
        "traffic_w13": traffic_w13,
        "traffic_w2": traffic_w2,
    }
    result["read_w13_gb_s"] = round(
        traffic_w13 / (result["read_w13_us"] * 1e-6) / 1e9, 1
    )
    result["read_w2_gb_s"] = round(traffic_w2 / (result["read_w2_us"] * 1e-6) / 1e9, 1)
    result["read_both_gb_s"] = round(
        (traffic_w13 + traffic_w2) / (result["read_both_us"] * 1e-6) / 1e9, 1
    )
    print(json.dumps(result, indent=2))
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
