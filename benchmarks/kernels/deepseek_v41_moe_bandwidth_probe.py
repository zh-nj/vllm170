# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Is the SM80 MXFP4 grouped GEMM bandwidth-bound at the production M=1 shape?

Runs the production W13/W2 kernels through the same ``_moe_C`` entry point the
server uses, then reads exactly the same weight bytes with a plain streaming
kernel. The ratio between the two is the kernel's headroom over pure DRAM
traffic -- the Ceiling that any rewrite of the data path has to beat.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deepseek_v41_mxfp4_int8_sm80 import (  # noqa: E402
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    make_gemm_call,
)

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)


@triton.jit
def _stream_read_kernel(src, out, numel, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    values = tl.load(src + offs, mask=mask, other=0)
    acc = tl.sum(values.to(tl.int32), axis=0)
    if acc == 1234567:
        tl.store(out, acc)


def graph_time(call, calls: int, samples: int) -> dict[str, float]:
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
    return {
        "p50_us": statistics.median(per_call),
        "p90_us": per_call[min(len(per_call) - 1, int(len(per_call) * 0.9))],
        "min_us": per_call[0],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--block-m", type=int, default=16)
    ap.add_argument("--block-n", type=int, default=128)
    ap.add_argument("--graph-calls", type=int, default=40)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(4242)
    e, topk = args.experts, args.topk

    w13 = torch.empty(
        (e, HIDDEN_SIZE // 32, INTERMEDIATE_SIZE * 2 // 8, 128),
        dtype=torch.uint8,
        device=device,
    ).random_(0, 256)
    w13_shift2 = torch.empty(
        (e, HIDDEN_SIZE // 128, INTERMEDIATE_SIZE * 2 // 8, 8),
        dtype=torch.uint8,
        device=device,
    ).random_(0, 256)
    w13_scale = torch.ones(
        (e, INTERMEDIATE_SIZE * 2), dtype=torch.float32, device=device
    )
    w2 = torch.empty(
        (e, INTERMEDIATE_SIZE // 32, HIDDEN_SIZE // 8, 128),
        dtype=torch.uint8,
        device=device,
    ).random_(0, 256)
    w2_shift2 = torch.empty(
        (e, INTERMEDIATE_SIZE // 128, HIDDEN_SIZE // 8, 8),
        dtype=torch.uint8,
        device=device,
    ).random_(0, 256)
    w2_scale = torch.ones((e, HIDDEN_SIZE), dtype=torch.float32, device=device)

    # Per-expert bytes the kernel must move for one expert slot.
    bytes_per_expert = (
        w13[0].numel()
        + w13_shift2[0].numel()
        + w13_scale[0].numel() * 4
        + w2[0].numel()
        + w2_shift2[0].numel()
        + w2_scale[0].numel() * 4
    )
    # One token, top-k experts; pad to a 16-row block exactly like serving.
    # Distinct expert ids per slot keep the byte accounting exact: every slot
    # owns one block-M tile, so CTA count scales with slots and nothing is
    # merged or L2-shared between slots.
    tokens = args.tokens
    routing_ids = (
        torch.arange(tokens * topk, dtype=torch.int32, device=device)
        .reshape(tokens, topk)
        % e
    )
    routing_weights = torch.full(
        (tokens, topk), 1.0 / topk, dtype=torch.float32, device=device
    )
    sorted_ids, expert_ids, num_post_pad = moe_align_block_size(
        routing_ids, args.block_m, e
    )
    num_routed = tokens * topk
    traffic = bytes_per_expert * num_routed

    a13_q = torch.randint(
        -8, 8, (tokens, HIDDEN_SIZE), dtype=torch.int8, device=device
    )
    a13_scale = torch.ones((tokens,), dtype=torch.float32, device=device)
    a2_q = torch.randint(
        -8, 8, (num_routed, INTERMEDIATE_SIZE), dtype=torch.int8, device=device
    )
    a2_scale = torch.ones((num_routed,), dtype=torch.float32, device=device)
    gate_up = torch.empty(
        (num_routed, INTERMEDIATE_SIZE * 2), dtype=torch.bfloat16, device=device
    )
    routed_out = torch.empty(
        (num_routed, HIDDEN_SIZE), dtype=torch.bfloat16, device=device
    )
    out = torch.empty((tokens, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)

    w13_call = make_gemm_call(
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
        args.block_m,
        args.block_n,
        False,
    )
    w2_call = make_gemm_call(
        a2_q,
        a2_scale,
        w2,
        w2_shift2,
        w2_scale,
        out,
        routed_out,
        routing_weights,
        sorted_ids,
        expert_ids,
        num_post_pad,
        num_routed,
        args.block_m,
        args.block_n,
        True,
    )

    def both():
        w13_call()
        w2_call()

    def read_only():
        flat = scratch
        _stream_read_kernel[(triton.cdiv(flat.numel(), 4096),)](
            flat, sink, flat.numel(), BLOCK=4096
        )

    # Control: read the same number of bytes with a streaming kernel.
    scratch = torch.empty(traffic, dtype=torch.uint8, device=device).random_(0, 256)
    sink = torch.zeros(1, dtype=torch.int32, device=device)

    result = {
        "experts": e,
        "topk": topk,
        "tokens": tokens,
        "tile": [args.block_m, args.block_n],
        "bytes_per_expert": bytes_per_expert,
        "expert_traffic_bytes": traffic,
        "w13": graph_time(w13_call, args.graph_calls, args.samples),
        "w2": graph_time(w2_call, args.graph_calls, args.samples),
        "both": graph_time(both, args.graph_calls, args.samples),
        "stream_read_same_bytes": graph_time(
            read_only, args.graph_calls, args.samples
        ),
    }
    for key in ("w13", "w2", "both", "stream_read_same_bytes"):
        result[key]["effective_gb_s"] = round(
            traffic / (result[key]["p50_us"] * 1e-6) / 1e9, 1
        )
    result["w13_only_gb_s"] = round(
        (bytes_per_expert * topk / 2) / (result["w13"]["p50_us"] * 1e-6) / 1e9, 1
    )
    print(json.dumps(result, indent=2))
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
