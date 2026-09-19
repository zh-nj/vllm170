# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read the packed MXFP4 expert slabs in the GEMM's own access pattern.

``deepseek_v41_moe_slab_read_control.py`` streams each slab contiguously with a
flat grid; the grouped GEMM instead runs a small grid where every CTA walks the
K dimension in 2 KiB chunks with a 72 KiB stride, because the packed layout
interleaves K/32 blocks with N/8 groups. This separates the two: if this
strided, GEMM-shaped reader also lands near the GEMM's bandwidth, the gap is
the access pattern; if it reaches the flat reader's bandwidth, the gap is the
kernel's inner loop.
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
def _strided_slab_kernel(
    src,
    ids,
    out,
    k_blocks,
    k_stride,
    n_tiles,
    tile_bytes,
    n_groups8,
    BLOCK: tl.constexpr,
):
    """Grid = (#experts x #n_tiles); each program walks all K for its N tile."""
    pid = tl.program_id(0)
    slot = pid // n_tiles
    n_tile = pid - slot * n_tiles
    expert = tl.load(ids + slot).to(tl.int64)
    base = expert * (k_blocks * k_stride)
    n_group = n_tile * (tile_bytes // 128)
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for k_block in tl.range(0, k_blocks):
        offs = (
            base
            + k_block * k_stride
            + n_group * 128
            + tl.arange(0, BLOCK)
        )
        acc += tl.load(src + offs).to(tl.int32)
    if tl.sum(acc, axis=0) == 1234567:
        tl.store(out, 0)


@triton.jit
def _flat_slab_kernel(src, ids, out, slab_numel, n_ids, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    per_id = tl.num_programs(0) // n_ids
    slot = tl.minimum(pid // per_id, n_ids - 1)
    expert = tl.load(ids + slot).to(tl.int64)
    offs = (pid % per_id) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < slab_numel
    values = tl.load(src + expert * slab_numel + offs, mask=mask, other=0)
    if tl.sum(values.to(tl.int32), axis=0) == 1234567:
        tl.store(out, 0)


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
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--n-tile-bytes", type=int, default=2048)
    ap.add_argument("--graph-calls", type=int, default=40)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(7)
    device = torch.device("cuda")
    e, topk = args.experts, args.topk
    # W13 packed layout from the probe: [E, K/32, N/8, 128] uint8.
    k_dim, n_dim = 5120, 4608
    k_blocks = k_dim // 32
    n_groups8 = n_dim // 8
    block_bytes = 128
    k_stride = n_groups8 * block_bytes
    # A 128-wide N tile is 16 n-groups of 128 B; the row stride is the whole
    # packed N extent. 36 tiles cover N=4608 for one expert, so the grid is the
    # GEMM's own 6 x 36.
    n_tiles = k_stride // args.n_tile_bytes

    flat = torch.empty(
        e * k_blocks * k_stride, dtype=torch.uint8, device=device
    ).random_(0, 256)
    ids = torch.arange(topk, dtype=torch.int32, device=device)
    sink = torch.zeros(1, dtype=torch.int32, device=device)
    # Walk the whole K extent like the GEMM's 40 K-tiles of 4 k-blocks each.
    k_blocks_per_call = k_blocks

    def strided():
        _strided_slab_kernel[(topk * n_tiles,)](
            flat,
            ids,
            sink,
            k_blocks_per_call,
            k_stride,
            n_tiles,
            args.n_tile_bytes,
            n_groups8,
            BLOCK=args.n_tile_bytes,
        )

    slab_total = k_blocks * k_stride
    def flatcall():
        _flat_slab_kernel[(topk * triton.cdiv(slab_total, 4096),)](
            flat, ids, sink, slab_total, topk, BLOCK=4096
        )

    traffic = topk * k_blocks_per_call * k_stride
    strided_us = graph_time(strided, args.graph_calls, args.samples)
    # Scale the flat read to the same byte volume for an apples-to-apples rate.
    flat_us = graph_time(flatcall, args.graph_calls, args.samples)
    result = {
        "topk": topk,
        "grid_strided": topk * n_tiles,
        "n_tile_bytes": args.n_tile_bytes,
        "strided_us": strided_us,
        "strided_gb_s": round(traffic / (strided_us * 1e-6) / 1e9, 1),
        "flat_us_full_slab": flat_us,
        "flat_gb_s": round(topk * slab_total / (flat_us * 1e-6) / 1e9, 1),
    }
    print(json.dumps(result, indent=2))
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
