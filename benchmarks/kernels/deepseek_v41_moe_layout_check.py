# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode the SM80 MXFP4-int8 grouped layout on the host and check it.

The packed weight layout is only documented by the CUTLASS kernel that consumes
it. Before writing any replacement kernel this reconstructs the dense int8
weight from the packed tensors and compares a full GEMM against the production
``_moe_C`` op. If the byte->(k, n) mapping or the E2M1 table is wrong, the
comparison fails loudly instead of producing a subtly wrong model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

HIDDEN = 5120
INTER = 2304
NUM_EXPERTS = 8
TOPK = 6


def decode_e2m1(nib: torch.Tensor) -> torch.Tensor:
    """E2M1 nibble -> int8, mirroring lookup_mxfp4_x1_byte_perm (shift = 0)."""
    table = torch.tensor([0, 1, 2, 3, 4, 6, 8, 12], dtype=torch.int32)
    mag = table[(nib & 7).long()]
    sign = torch.where((nib & 8) != 0, -1, 1)
    return (mag * sign).to(torch.int32)


def unpack_w13(packed: torch.Tensor, k_dim: int, n_dim: int) -> torch.Tensor:
    """[E, K/32, N/8, 128] uint8 -> [E, N, K] int8.

    Byte b of a 128-byte chunk covers n_rel = b // 16 and, within the 16-byte
    per-n group, quad = (b % 16) // 4, j = b % 4: the low nibble is k_rel =
    8*quad + 2*j and the high nibble is k_rel + 1.
    """
    e = packed.shape[0]
    view = packed.view(e, k_dim // 32, n_dim // 8, 8, 4, 4)
    lo = decode_e2m1(view & 0xF)
    hi = decode_e2m1((view >> 4) & 0xF)
    quad = torch.arange(4)
    j = torch.arange(4)
    k_rel_lo = (8 * quad[:, None] + 2 * j[None, :])
    w = torch.empty(e, k_dim // 32, n_dim // 8, 8, 4, 4, 2, dtype=torch.int32)
    w[..., 0] = lo
    w[..., 1] = hi
    # (E, K/32, N/8, n_rel, quad, j, half) -> (E, K, N)
    return w.permute(0, 1, 4, 5, 6, 2, 3).reshape(
        e, k_dim // 32, 4, 4, 2, n_dim
    ).reshape(e, k_dim, n_dim)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(3)
    device = torch.device("cuda")
    e = NUM_EXPERTS
    w13 = torch.empty(
        (e, HIDDEN // 32, (INTER * 2) // 8, 128), dtype=torch.uint8, device=device
    ).random_(0, 256)
    w13_shift2 = torch.zeros(
        (e, HIDDEN // 128, (INTER * 2) // 8, 8), dtype=torch.uint8, device=device
    )
    w13_scale = torch.ones((e, INTER * 2), dtype=torch.float32, device=device)

    a_q = torch.randint(-8, 8, (1, HIDDEN), dtype=torch.int8, device=device)
    a_scale = torch.ones((1,), dtype=torch.float32, device=device)
    gate_up = torch.empty((TOPK, INTER * 2), dtype=torch.bfloat16, device=device)

    routing_ids = torch.arange(TOPK, dtype=torch.int32, device=device).reshape(1, TOPK)
    routing_weights = torch.full((1, TOPK), 1.0 / TOPK, dtype=torch.float32, device=device)
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    sorted_ids, expert_ids, num_post_pad = moe_align_block_size(routing_ids, 16, e)
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

    dense = unpack_w13(w13.cpu(), HIDDEN, INTER * 2).to(torch.int64)
    a = a_q.cpu().to(torch.int64)
    slot_experts = expert_ids[sorted_ids[::16].long()].long()
    reference = []
    for slot in range(TOPK):
        expert = int(slot_experts[slot])
        acc = (a[0] @ dense[expert]).to(torch.float32)
        reference.append(acc.to(torch.bfloat16))
    reference = torch.stack(reference)

    got = gate_up.cpu()
    match = torch.equal(reference, got)
    max_delta = (reference.float() - got.float()).abs().max().item()
    result = {
        "bit_exact": bool(match),
        "max_abs_delta": max_delta,
        "slot_experts": [int(x) for x in slot_experts],
        "sample_reference": [float(x) for x in reference[0, :6]],
        "sample_op": [float(x) for x in got[0, :6]],
    }
    print(json.dumps(result, indent=2))
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
