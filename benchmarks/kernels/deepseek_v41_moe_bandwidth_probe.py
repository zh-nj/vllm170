# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How much of the SM80 MXFP4 grouped GEMM's weight traffic is actually useful?

Runs the production W13/W2 kernels through the same ``_moe_C`` entry point the
server uses, then reads the same *logical* number of weight bytes with a plain
streaming kernel. The ratio is the kernel's headroom over pure DRAM traffic.

Byte accounting rules (corrected in schema 2):

* W13 and W2 have different footprints, so each stage is measured against its
  own bytes. W13 is about 2x W2, never half of their sum.
* Weight bytes are deduplicated over the experts the routing actually hits.
  Repeated hits on one expert do not multiply its footprint.
* ``active_weight_gb_s`` is a *logical effective bandwidth* over the bytes the
  kernel logically must read. It is not a measured DRAM/HBM bandwidth: it
  cannot see L2 hits, re-reads, padding or partial-line effects.

The streaming control in this file reads a freshly allocated contiguous buffer
of the same logical size. It is **not** an address-identical comparison against
the production GEMM; for address-sharing controls see
``deepseek_v41_moe_slab_read_control.py`` and friends.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deepseek_v41_moe_harness import (  # noqa: E402
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    STAGE_SHAPES,
    W2_STAGE,
    W13_STAGE,
    compute_route_work,
    resident_bytes,
)
from deepseek_v41_mxfp4_int8_sm80 import make_gemm_call  # noqa: E402

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
from vllm.triton_utils import tl, triton  # noqa: E402

SCHEMA_VERSION = 2


@triton.jit
def _stream_read_kernel(src, out, numel, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    values = tl.load(src + offs, mask=mask, other=0)
    acc = tl.sum(values.to(tl.int32), axis=0)
    if acc == 1234567:
        tl.store(out, acc)


def graph_time(call, calls: int, samples: int) -> dict[str, float]:
    """Event-bracketed CUDA-graph replay, normalised per call."""
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
    ap = argparse.ArgumentParser(description=__doc__)
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

    tokens = args.tokens
    # Distinct expert ids per slot keep the deduplicated accounting exact and
    # equal to the naive per-slot count; reuse is handled by the harness.
    routing_ids = (
        torch.arange(tokens * topk, dtype=torch.int32, device=device).reshape(
            tokens, topk
        )
        % e
    )
    routing_weights = torch.full(
        (tokens, topk), 1.0 / topk, dtype=torch.float32, device=device
    )
    sorted_ids, expert_ids, num_post_pad = moe_align_block_size(
        routing_ids, args.block_m, e
    )
    num_routed = tokens * topk

    work = compute_route_work(
        routing_ids,
        args.block_m,
        {
            W13_STAGE: (w13, w13_shift2, w13_scale),
            W2_STAGE: (w2, w2_shift2, w2_scale),
        },
    )
    w13_bytes = work.stage_bytes[W13_STAGE]["total"]
    w2_bytes = work.stage_bytes[W2_STAGE]["total"]
    both_bytes = w13_bytes + w2_bytes

    a13_q = torch.randint(-8, 8, (tokens, HIDDEN_SIZE), dtype=torch.int8, device=device)
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

    # Control: same logical byte count, freshly allocated contiguous storage.
    # Not the production addresses and not the production access pattern.
    scratch = torch.empty(both_bytes, dtype=torch.uint8, device=device).random_(0, 256)
    sink = torch.zeros(1, dtype=torch.int32, device=device)

    def contiguous_read_control():
        _stream_read_kernel[(triton.cdiv(scratch.numel(), 4096),)](
            scratch, sink, scratch.numel(), BLOCK=4096
        )

    timings = {
        W13_STAGE: graph_time(w13_call, args.graph_calls, args.samples),
        W2_STAGE: graph_time(w2_call, args.graph_calls, args.samples),
        "both": graph_time(both, args.graph_calls, args.samples),
        "contiguous_read_control": graph_time(
            contiguous_read_control, args.graph_calls, args.samples
        ),
    }
    traffic = {
        W13_STAGE: w13_bytes,
        W2_STAGE: w2_bytes,
        "both": both_bytes,
        "contiguous_read_control": both_bytes,
    }
    for key, timing in timings.items():
        timing["logical_bytes"] = traffic[key]
        timing["effective_gb_s"] = round(
            traffic[key] / (timing["p50_us"] * 1e-6) / 1e9, 3
        )
        timing["useful_flops"] = (
            2 * num_routed * STAGE_SHAPES[key][0] * STAGE_SHAPES[key][1]
            if key in STAGE_SHAPES
            else None
        )
        timing["useful_tflops"] = (
            round(timing["useful_flops"] / (timing["p50_us"] * 1e-6) / 1e12, 4)
            if timing["useful_flops"]
            else None
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "schema_changes": {
            "1->2": (
                "bytes_per_expert/expert_traffic_bytes/effective_gb_s applied the "
                "W13+W2 sum to each stage and divided the sum by two to estimate "
                "W13. Replaced by per-stage deduplicated bytes "
                "(w13_weight_bytes/w2_weight_bytes). Old rows must not be merged "
                "with schema 2 rows."
            ),
            "w13_only_gb_s": (
                "removed: it multiplied the combined per-expert footprint by "
                "topk and halved it. Use active_weight_gb_s.w13 instead."
            ),
            "stream_read_same_bytes": (
                "renamed to contiguous_read_control; it never shared addresses "
                "with the GEMM."
            ),
        },
        "experts": e,
        "topk": topk,
        "tokens": tokens,
        "tile": [args.block_m, args.block_n],
        "num_routed": num_routed,
        "active_experts": len(work.active_experts),
        "expert_token_histogram": work.expert_token_histogram,
        "m_tiles": work.m_tiles,
        "padded_rows": work.padded_rows,
        "w13_weight_bytes": w13_bytes,
        "w2_weight_bytes": w2_bytes,
        "both_weight_bytes": both_bytes,
        "w13_bytes_detail": work.stage_bytes[W13_STAGE],
        "w2_bytes_detail": work.stage_bytes[W2_STAGE],
        "w13_w2_size_ratio": round(w13_bytes / w2_bytes, 4),
        "resident_bytes": resident_bytes(
            w13, w13_shift2, w13_scale, w2, w2_shift2, w2_scale
        ),
        "logical_io_bytes": None,
        "timings": timings,
        "notes": (
            "active weight bytes are deduplicated over active experts; "
            "effective_gb_s is a logical effective bandwidth over those bytes, "
            "not a measured DRAM/HBM figure. contiguous_read_control reads a "
            "freshly allocated buffer of the same logical size."
        ),
    }
    print(json.dumps(result, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
