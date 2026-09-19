# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Can a plain kernel read AND decode the packed MXFP4 weights at full rate?

``deepseek_v41_moe_pattern_control.py`` showed the grouped GEMM's own addresses
are readable at 964 GB/s with no compute. This adds the one piece of real work
the GEMM does per byte -- the E2M1 nibble decode the CUTLASS kernel performs
with ``lookup_mxfp4_x2_byte_perm`` -- and nothing else: no smem staging, no
fragment layout, no tensor-core operand shaping.

If this still lands near the pure-read rate, then a GEMV-shaped replacement for
the M=1 grouped GEMM can approach that rate too, and the gap is the kernel's
staging rather than the decode arithmetic.
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
def _decode4(u):
    """E2M1 magnitude table {0,1,2,3,4,6,8,12} indexed by the low 3 bits."""
    m = u & 7
    return tl.where(
        m == 0, 0,
        tl.where(m == 1, 1,
                 tl.where(m == 2, 2,
                          tl.where(m == 3, 3,
                                   tl.where(m == 4, 4,
                                            tl.where(m == 5, 6,
                                                     tl.where(m == 6, 8, 12)))))))


@triton.jit
def _strided_decode_kernel(
    src, ids, out, k_blocks, k_stride, n_tiles, tile_bytes,
    BLOCK: tl.constexpr, DECODE: tl.constexpr,
):
    pid = tl.program_id(0)
    slot = pid // n_tiles
    n_tile = pid - slot * n_tiles
    expert = tl.load(ids + slot).to(tl.int64)
    base = expert * (k_blocks * k_stride)
    n_group = n_tile * (tile_bytes // 128)
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for k_block in tl.range(0, k_blocks):
        offs = base + k_block * k_stride + n_group * 128 + tl.arange(0, BLOCK)
        q = tl.load(src + offs).to(tl.int32)
        if DECODE:
            lo = q & 15
            hi = (q >> 4) & 15
            vlo = tl.where(lo < 8, _decode4(lo), -_decode4(lo - 8))
            vhi = tl.where(hi < 8, _decode4(hi), -_decode4(hi - 8))
            acc += vlo + vhi
        else:
            acc += q
    if tl.sum(acc, axis=0) == 1234567:
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
    k_blocks, n_dim, block_bytes = 160, 4608, 128
    k_stride = (n_dim // 8) * block_bytes
    n_tiles = k_stride // args.n_tile_bytes

    src = torch.empty(e * k_blocks * k_stride, dtype=torch.uint8, device=device)
    src.random_(0, 256)
    ids = torch.arange(topk, dtype=torch.int32, device=device)
    sink = torch.zeros(1, dtype=torch.int32, device=device)
    traffic = topk * k_blocks * k_stride

    def make(decode: bool):
        def call():
            _strided_decode_kernel[(topk * n_tiles,)](
                src, ids, sink, k_blocks, k_stride, n_tiles, args.n_tile_bytes,
                BLOCK=args.n_tile_bytes, DECODE=decode,
            )
        return call

    read_us = graph_time(make(False), args.graph_calls, args.samples)
    dec_us = graph_time(make(True), args.graph_calls, args.samples)
    result = {
        "topk": topk,
        "grid": topk * n_tiles,
        "bytes": traffic,
        "read_us": read_us,
        "read_gb_s": round(traffic / (read_us * 1e-6) / 1e9, 1),
        "read_decode_us": dec_us,
        "read_decode_gb_s": round(traffic / (dec_us * 1e-6) / 1e9, 1),
        "decode_overhead_us": dec_us - read_us,
    }
    print(json.dumps(result, indent=2))
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
