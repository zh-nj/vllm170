# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recover the packed MXFP4 weight byte -> (k, n) mapping from the op itself.

Puts one nibble at a time into the packed tensor, runs the production grouped
GEMM, and reads back which output column moved and by how much. Two runs per
position (activation ramp by k%100 and by k//100) pin the exact k index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)

HIDDEN = 5120
INTER = 2304
NUM_EXPERTS = 8
TOPK = 6
N_DIM = INTER * 2


def run_op(w13, a_q, scale):
    device = w13.device
    w13_shift2 = torch.zeros(
        (NUM_EXPERTS, HIDDEN // 128, N_DIM // 8, 8), dtype=torch.uint8, device=device
    )
    w13_scale = torch.ones((NUM_EXPERTS, N_DIM), dtype=torch.float32, device=device)
    a_scale = torch.ones((1,), dtype=torch.float32, device=device)
    gate_up = torch.empty((TOPK, N_DIM), dtype=torch.bfloat16, device=device)
    routing_ids = torch.arange(TOPK, dtype=torch.int32, device=device).reshape(1, TOPK)
    routing_weights = torch.full(
        (1, TOPK), 1.0 / TOPK, dtype=torch.float32, device=device
    )
    sorted_ids, expert_ids, num_post_pad = moe_align_block_size(
        routing_ids, 16, NUM_EXPERTS
    )
    torch.ops._moe_C.dsv4_mxfp4_int8_sm80_moe_gemm(
        a_q,
        a_scale,
        w13,
        w13_shift2,
        w13_scale,
        gate_up,
        gate_up,
        routing_weights,
        sorted_ids,
        expert_ids,
        num_post_pad,
        TOPK,
        16,
        128,
        False,
    )
    torch.accelerator.synchronize()
    return gate_up[0].float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=128)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    k_dim = HIDDEN
    shape = (NUM_EXPERTS, k_dim // 32, N_DIM // 8, 128)

    k_idx = torch.arange(k_dim)
    a_mod = (1 + (k_idx % 100)).to(torch.int8).reshape(1, k_dim).to(device)
    a_div = (1 + (k_idx // 100)).to(torch.int8).reshape(1, k_dim).to(device)

    rows = []
    for half in (0, 1):
        for byte in range(args.bytes):
            w13 = torch.zeros(shape, dtype=torch.uint8, device=device)
            w13[0, 0, 0, byte] = 1 if half == 0 else 0x10
            out_mod = run_op(w13, a_mod, 1.0)
            out_div = run_op(w13, a_div, 1.0)
            n = int(out_mod.abs().argmax())
            v_mod = float(out_mod[n])
            v_div = float(out_div[n])
            if abs(v_mod) < 0.5:
                rows.append({"byte": byte, "half": half, "n": -1, "k": -1})
                continue
            k = 100 * (round(v_div) - 1) + (round(v_mod) - 1)
            rows.append({"byte": byte, "half": half, "n": n, "k": int(k)})

    ok = [r for r in rows if r["n"] >= 0]
    n_of_byte = {}
    for r in rows:
        n_of_byte.setdefault(r["byte"], set()).add(r["n"])
    summary = {
        "positions": len(rows),
        "resolved": len(ok),
        "byte_to_n_first16": {str(b): sorted(n_of_byte.get(b, [])) for b in range(16)},
        "byte_to_n_last16": {
            str(b): sorted(n_of_byte.get(b, [])) for b in range(112, 128)
        },
        "byte_half_to_k_first16": {
            f"{r['byte']}:{r['half']}": r["k"] for r in rows if r["byte"] < 8
        },
        "rows": rows,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    args.output.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
