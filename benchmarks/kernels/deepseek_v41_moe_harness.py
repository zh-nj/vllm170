# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline acceptance harness for the DSV4.1-Flash SM80 MXFP4-INT8 routed MoE.

Scope is deliberately narrow: the two grouped GEMM stages (W13, W2) and the
routing-weighted reduce of the ``mxfp4_int8_sm80`` backend at the DSV4.1-F TP1
expert shape (H=5120, I=2304). It is an offline measuring stick, not a kernel.

Design notes
------------
* Importing this module must not require CUDA, vLLM, or ``deep_gemm``. The pure
  helpers (packed-weight decoding, routing, metrics, byte accounting, JSONL
  schema) are usable on CPU so they can be self-checked on a machine without a
  GPU. ``moe_align_block_size`` is imported lazily inside :func:`align_routes`.
* The reference decodes the *production* packed weights (``b_mxfp4`` /
  ``b_shift2`` / ``b_channel_scale``) instead of re-deriving them from the MXFP4
  checkpoint. The MXFP4 -> INT8 conversion is an approximate remap, so a
  checkpoint-exact dequantisation is a different operator, not a reference for
  this kernel.
* ``calc_diff`` follows the definition used by DeepGEMM
  (``deep_gemm/testing/numeric.py``, deepseek-ai/DeepGEMM commit 057ca59,
  MIT licensed -- see that repository's LICENSE). The formula is restated here
  rather than imported so that this module keeps zero DeepGEMM imports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 1

HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 2304
NUM_EXPERTS = 384
SUPPORTED_TOPK = (3, 6)
SUPPORTED_TILES: tuple[tuple[int, int], ...] = (
    (16, 64),
    (16, 128),
    (64, 128),
    (128, 128),
)
WORKSPACE_ALIGNMENT = 128

# Stage geometry. ``source_rows_are_slots`` is the kernel's own flag: it selects
# whether the A operand is indexed by routed slot (W2) or by token (W13).
W13_STAGE = "w13"
W2_STAGE = "w2"
REDUCE_STAGE = "reduce"
STAGE_SHAPES: dict[str, tuple[int, int, bool]] = {
    W13_STAGE: (2 * INTERMEDIATE_SIZE, HIDDEN_SIZE, False),
    W2_STAGE: (HIDDEN_SIZE, INTERMEDIATE_SIZE, True),
}

# ---------------------------------------------------------------------------
# Packed-weight format
# ---------------------------------------------------------------------------

# ``(n % 8) * 4 + thread_in_group`` selects a 4-byte lane; the j-th byte of that
# lane carries the k-pair given by this permutation.
_PACK_BYTE_PERMUTATION = (0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15)
# Magnitudes of the stored code, already carrying the E2M1 x2 factor.
_E2M1_MAG_X2 = (0, 1, 2, 3, 4, 6, 8, 12)
_MAX_SHIFT = 3
# |code| <= 12 * 2**3 = 96.
MAX_ABS_INT8_WEIGHT = _E2M1_MAG_X2[-1] << _MAX_SHIFT
# Activation quantization clamps to +-127 (see ``quantize_per_token_int8``).
MAX_ABS_INT8_ACTIVATION = 127
INT32_MAX = 2**31 - 1


def int32_accumulator_bound(k: int) -> int:
    """Worst-case ``|sum_k a*w|`` for the documented operand ranges.

    The MMA uses ``OpMultiplyAddSaturate``, so an out-of-range accumulator would
    saturate instead of wrapping and the integer reference would stop matching.
    With ``|a| <= 127``, ``|w| <= 96`` and ``K <= 5120`` the bound is 62.4M,
    about 34x below ``INT32_MAX``, so saturation is unreachable for any input
    this backend accepts.
    """
    return MAX_ABS_INT8_ACTIVATION * MAX_ABS_INT8_WEIGHT * k


def _mag_table(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(_E2M1_MAG_X2, device=device, dtype=dtype)


def decode_packed_weights(
    b_mxfp4: torch.Tensor,
    b_shift2: torch.Tensor,
    *,
    k: int | None = None,
    n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode packed expert weights into the signed INT8 values the GEMM uses.

    Independent of the production ``_pack_codes_coalesced`` code path: the lane
    permutation and nibble order are re-derived from the on-disk layout.

    Args:
        b_mxfp4: uint8 ``[E, K/32, ceil(N/8), 128]`` packed code nibbles.
        b_shift2: uint8 ``[E, ceil((K/32)/4), ceil(N/8), 8]`` 2-bit shift fields.
        k: real K; defaults to the full K implied by ``b_mxfp4``.
        n: real N; defaults to ``ceil(N/8) * 8`` (the padded width).

    Returns:
        ``(weights, shifts)`` with shapes ``[E, N, K]`` (int8, signed) and
        ``[E, N, K//32]`` (int16). The float weight a channel is multiplied by is
        ``weights * b_channel_scale[e, n]``; the E2M1 halving is already folded
        into the production channel scale, so it must not be applied again.
    """
    if b_mxfp4.dim() != 4 or b_mxfp4.shape[-1] != 128:
        raise ValueError(
            f"b_mxfp4 must be [E, K/32, ceil(N/8), 128], got {tuple(b_mxfp4.shape)}"
        )
    if b_shift2.dim() != 4 or b_shift2.shape[-1] != 8:
        raise ValueError(
            f"b_shift2 must be [E, ceil(Kblocks/4), ceil(N/8), 8], got "
            f"{tuple(b_shift2.shape)}"
        )
    experts, k_blocks, n_groups8, _ = b_mxfp4.shape
    if b_shift2.shape[0] != experts or b_shift2.shape[1] < (k_blocks + 3) // 4:
        raise ValueError(
            f"b_shift2 does not cover b_mxfp4 K blocks: {tuple(b_shift2.shape)} vs "
            f"{tuple(b_mxfp4.shape)}"
        )
    if n is not None and n_groups8 != (n + 7) // 8:
        raise ValueError(f"b_mxfp4 N groups {n_groups8} do not cover N={n}")
    full_k = k_blocks * 32
    if k is not None and (k > full_k or k % 32 != 0):
        raise ValueError(f"K must be a multiple of 32 and <= {full_k}, got {k}")

    device = b_mxfp4.device
    packed = b_mxfp4.contiguous().view(experts, k_blocks, n_groups8, 32, 4)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    # codes[e, kb, ng, n_in_group, k32]
    codes = torch.zeros(
        (experts, k_blocks, n_groups8, 8, 32), dtype=torch.uint8, device=device
    )
    n_in_group = torch.arange(8, device=device)
    for thread_in_group in range(4):
        lane = n_in_group * 4 + thread_in_group
        for j in range(4):
            k_pair = _PACK_BYTE_PERMUTATION[thread_in_group * 4 + j]
            codes[:, :, :, :, 2 * k_pair] = low[:, :, :, lane, j]
            codes[:, :, :, :, 2 * k_pair + 1] = high[:, :, :, lane, j]

    shifts = _decode_shifts(b_shift2, k_blocks)
    # shifts[e, kb, ng, n_in_group] -> broadcast over k32.
    magnitude = _mag_table(device, torch.int16)[(codes & 0x07).long()]
    negative = (codes & 0x08) != 0
    signed = torch.where(negative, -magnitude, magnitude)
    weights = (signed * (1 << shifts)[..., None]).to(torch.int8)

    # [E, kb, ng, 8, 32] -> [E, ng*8, kb*32]
    weights = weights.permute(0, 2, 3, 1, 4).reshape(
        experts, n_groups8 * 8, k_blocks * 32
    )
    shifts = shifts.permute(0, 2, 3, 1).reshape(experts, n_groups8 * 8, k_blocks)
    if n is not None:
        weights = weights[:, :n]
        shifts = shifts[:, :n]
    if k is not None and k != full_k:
        weights = weights[:, :, :k]
        shifts = shifts[:, :, : k // 32]
    return weights.contiguous(), shifts.contiguous()


def _decode_shifts(b_shift2: torch.Tensor, k_blocks: int) -> torch.Tensor:
    """Expand the packed 2-bit shift fields into ``[E, K/32, ng, 8]``."""
    kb_index = torch.arange(k_blocks, device=b_shift2.device)
    packed = b_shift2[:, kb_index // 4]  # [E, kb, ng, 8]
    bit_offset = (2 * (kb_index % 4)).view(1, k_blocks, 1, 1)
    return ((packed >> bit_offset) & 0x03).to(torch.int16)


def decode_packed_weights_scalar(
    b_mxfp4: torch.Tensor,
    b_shift2: torch.Tensor,
    *,
    n: int,
    k: int,
) -> torch.Tensor:
    """Scalar restatement of the format, used to cross-check the vector op.

    Written directly from the documented layout with explicit loops so that a
    shared indexing mistake between this and :func:`decode_packed_weights` would
    have to be made twice, in two different shapes of code.
    """
    experts = b_mxfp4.shape[0]
    out = torch.zeros((experts, n, k), dtype=torch.int8, device=b_mxfp4.device)
    u = b_mxfp4
    s = b_shift2
    for e in range(experts):
        for nn in range(n):
            for kb in range(k // 32):
                field = int(s[e, kb // 4, nn // 8, nn % 8].item())
                shift = (field >> (2 * (kb % 4))) & 0x03
                for t in range(4):
                    lane = (nn % 8) * 4 + t
                    for j in range(4):
                        byte = int(u[e, kb, nn // 8, lane * 4 + j].item())
                        k_pair = _PACK_BYTE_PERMUTATION[t * 4 + j]
                        for nibble, kk in (
                            (byte & 0x0F, 2 * k_pair),
                            ((byte >> 4) & 0x0F, 2 * k_pair + 1),
                        ):
                            mag = _E2M1_MAG_X2[nibble & 0x07]
                            value = -mag if nibble & 0x08 else mag
                            out[e, nn, kb * 32 + kk] = value << shift
    return out


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def make_routing_ids(
    tokens: int,
    topk: int,
    num_experts: int,
    pattern: str,
    seed: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build ``[tokens, topk]`` int32 expert ids with no in-token repeats.

    ``uniform`` spreads tokens over all experts, ``hot`` concentrates most
    slots on a fixed shared expert set, ``sparse`` touches a small random subset
    and leaves the rest of the experts empty.
    """
    if tokens <= 0:
        raise ValueError(f"tokens must be positive, got {tokens}")
    if topk not in SUPPORTED_TOPK:
        raise ValueError(f"topk must be one of {SUPPORTED_TOPK}, got {topk}")
    if num_experts < topk:
        raise ValueError(
            f"num_experts ({num_experts}) must be >= topk ({topk}) so that a "
            "token never selects the same expert twice"
        )
    generator = torch.Generator().manual_seed(int(seed))
    if pattern == "uniform":
        ids = torch.rand((tokens, num_experts), generator=generator).topk(topk, dim=1)
        return ids.indices.to(torch.int32).to(device)
    if pattern == "hot":
        hot_count = max(topk, min(num_experts, 8))
        hot = torch.randperm(num_experts, generator=generator)[:hot_count]
        draw = torch.rand((tokens, hot_count), generator=generator).topk(topk, dim=1)
        return hot[draw.indices].to(torch.int32).to(device)
    if pattern == "sparse":
        active = max(topk, min(num_experts, max(topk, tokens)))
        pool = torch.randperm(num_experts, generator=generator)[:active]
        draw = torch.rand((tokens, active), generator=generator).topk(topk, dim=1)
        return pool[draw.indices].to(torch.int32).to(device)
    raise ValueError(f"unknown route pattern {pattern!r}")


def make_routing_ids_from_slots(
    slot_experts: Sequence[int],
    topk: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Explicit slot-major expert ids, for hand-built routing fixtures."""
    flat = torch.tensor(list(slot_experts), dtype=torch.int32)
    if flat.numel() % topk != 0:
        raise ValueError("slot count must be a multiple of topk")
    tokens = flat.numel() // topk
    ids = flat.view(tokens, topk)
    for token in range(tokens):
        row = ids[token].tolist()
        if len(set(row)) != topk:
            raise ValueError(f"token {token} repeats an expert: {row}")
    return ids.to(device)


def token_expert_exclusivity(routing_ids: torch.Tensor) -> bool:
    """True when no token selects the same expert twice."""
    return bool(
        (routing_ids.sort(dim=1).values.diff(dim=1) != 0).all().item()
        if routing_ids.shape[1] > 1
        else True
    )


def route_hash(routing_ids: torch.Tensor) -> str:
    data = routing_ids.detach().to(torch.int32).cpu().contiguous()
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()[:16]


def align_routes(
    routing_ids: torch.Tensor, block_m: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Production alignment. Imported lazily to keep this module CUDA-free."""
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    return moe_align_block_size(routing_ids, block_m, num_experts)


# ---------------------------------------------------------------------------
# Byte and work accounting (see the harness contract, section 6.1)
# ---------------------------------------------------------------------------


def stage_weight_bytes(
    b_mxfp4: torch.Tensor,
    b_shift2: torch.Tensor,
    b_channel_scale: torch.Tensor,
    experts: Iterable[int],
) -> dict[str, int]:
    """Logical bytes of one stage's weights over a *deduplicated* expert set.

    Counts the tensors an expert's grouped GEMM must read: packed codes, shift
    fields and the per-channel scale. Sharing/allocation size is reported
    separately by :func:`resident_bytes`.
    """
    unique = sorted({int(e) for e in experts})
    if not unique:
        return {"packed": 0, "shift2": 0, "channel_scale": 0, "total": 0}
    index = torch.tensor(unique, dtype=torch.long, device=b_mxfp4.device)
    packed = int(b_mxfp4.index_select(0, index).numel())
    shift2 = int(b_shift2.index_select(0, index).numel())
    scale = int(b_channel_scale.index_select(0, index).numel()) * (
        b_channel_scale.element_size()
    )
    return {
        "packed": packed,
        "shift2": shift2,
        "channel_scale": scale,
        "total": packed + shift2 + scale,
    }


def resident_bytes(*tensors: torch.Tensor | None) -> int:
    """Allocation size of logical tensors; not a DRAM-traffic estimate."""
    total = 0
    for tensor in tensors:
        if tensor is None:
            continue
        total += tensor.numel() * tensor.element_size()
    return total


@dataclass
class RouteWork:
    """Per-case routing work and byte accounting."""

    active_experts: list[int]
    expert_token_histogram: list[int]
    m_tiles: int
    padded_rows: int
    slots: int
    stage_bytes: dict[str, dict[str, int]]
    useful_flops: dict[str, int]

    def as_row(self) -> dict[str, Any]:
        return {
            "active_experts": len(self.active_experts),
            "expert_token_histogram": self.expert_token_histogram,
            "m_tiles": self.m_tiles,
            "padded_rows": self.padded_rows,
            "slots": self.slots,
        }


def compute_route_work(
    routing_ids: torch.Tensor,
    block_m: int,
    weights: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> RouteWork:
    """Derive active experts, padding and per-stage bytes from the routing ids.

    ``m_e`` is the number of routed slots an expert receives, so
    ``m_tiles = sum_e ceil(m_e / block_m)``. Weight bytes are deduplicated over
    the active expert set: a token that reuses an expert does not double count
    that expert's footprint.
    """
    flat = routing_ids.detach().reshape(-1).to(torch.int64).cpu()
    num_experts = int(flat.max().item()) + 1 if flat.numel() else 0
    histogram = torch.bincount(flat, minlength=num_experts) if flat.numel() else None
    active = [
        int(e)
        for e in (
            histogram.nonzero().flatten().tolist() if histogram is not None else []
        )
    ]
    counts = [int(histogram[e].item()) for e in active] if histogram is not None else []
    m_tiles = sum((count + block_m - 1) // block_m for count in counts)
    padded_rows = sum(((count + block_m - 1) // block_m) * block_m for count in counts)
    slots = int(flat.numel())
    stage_bytes = {
        stage: stage_weight_bytes(*weights[stage], active) for stage in weights
    }
    useful_flops = {
        stage: 2 * slots * n * k for stage, (n, k, _) in STAGE_SHAPES.items()
    }
    return RouteWork(
        active_experts=active,
        expert_token_histogram=counts,
        m_tiles=m_tiles,
        padded_rows=padded_rows,
        slots=slots,
        stage_bytes=stage_bytes,
        useful_flops=useful_flops,
    )


# ---------------------------------------------------------------------------
# Independent reference
# ---------------------------------------------------------------------------


def reference_stage(
    a_q: torch.Tensor,
    a_scale: torch.Tensor,
    b_mxfp4: torch.Tensor,
    b_shift2: torch.Tensor,
    b_channel_scale: torch.Tensor,
    routing_ids: torch.Tensor,
    *,
    stage: str,
    chunk_n: int = 512,
    compute_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Exact grouped-GEMM reference producing slot-indexed BF16 rows.

    Mirrors the production epilogue exactly:

    ``slot = sorted_token_ids[padded_row]``; ``source_row = slot`` for W2 and
    ``slot // topk`` for W13; ``value = (float32(acc) * a_scale[source_row]) *
    b_channel_scale[expert, n]``; store with round-to-nearest-even BF16.

    The integer accumulator is computed in ``compute_dtype``. With the default
    float64 the dot product is *exact*: A entries are in [-127, 127] and decoded
    weights in [-96, 96], so every accumulated sum is an integer bounded by
    ``127 * 96 * K`` (62.4M for W13), far below 2**53.

    Args:
        a_q: int8 ``[rows, K]``; rows are tokens when ``source_rows_are_slots``
            is False and routed slots otherwise.
        a_scale: float32 ``[rows]`` per-row activation scale.
        routing_ids: int32 ``[tokens, topk]``.
        stage: ``"w13"`` or ``"w2"``.
        chunk_n: output columns decoded and multiplied per pass.

    Returns:
        BF16 ``[tokens * topk, N]``; rows beyond ``tokens * topk`` are untouched
        because the caller supplies exactly that many.
    """
    if stage not in STAGE_SHAPES:
        raise ValueError(f"stage must be one of {tuple(STAGE_SHAPES)}, got {stage!r}")
    n, k, source_rows_are_slots = STAGE_SHAPES[stage]
    topk = routing_ids.shape[1]
    slot_experts = routing_ids.detach().reshape(-1).to(torch.int64)
    num_slots = int(slot_experts.numel())
    if a_q.shape[1] != k:
        raise ValueError(f"{stage} expects K={k}, got {tuple(a_q.shape)}")

    out = torch.empty((num_slots, n), dtype=torch.bfloat16, device=a_q.device)
    if num_slots == 0:
        return out
    slot_index = torch.arange(num_slots, device=a_q.device, dtype=torch.int64)
    source_row = slot_index if source_rows_are_slots else slot_index // topk
    rows = a_q.shape[0]
    if int(source_row.max().item()) >= rows:
        raise ValueError(
            f"{stage} reference needs {int(source_row.max().item()) + 1} A rows, "
            f"got {rows}"
        )

    for expert in sorted({int(e) for e in slot_experts.cpu().tolist()}):
        slots = (slot_experts == expert).nonzero(as_tuple=True)[0]
        src = source_row.index_select(0, slots)
        if int(src.max().item()) >= rows:
            raise ValueError(f"expert {expert} references A row {int(src.max())}")
        activation = a_q.index_select(0, src).to(compute_dtype)
        scale = a_scale.index_select(0, src).to(torch.float32)
        expert_weights, _ = decode_packed_weights(
            b_mxfp4[expert : expert + 1],
            b_shift2[expert : expert + 1],
            k=k,
            n=n,
        )
        expert_weights = expert_weights[0]
        for start in range(0, n, chunk_n):
            stop = min(start + chunk_n, n)
            codes = expert_weights[start:stop].to(compute_dtype)
            acc = activation @ codes.t()
            acc32 = acc.to(torch.float32)
            value = (acc32 * scale.unsqueeze(1)) * b_channel_scale[
                expert, start:stop
            ].unsqueeze(0)
            out[slots, start:stop] = value.to(torch.bfloat16)
    return out


def reference_reduce(
    routed_out: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    accumulate_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Routing-weighted reduce over the top-k slots of each token.

    The kernel walks ``k`` ascending from ``0.0f`` in float32, so the reference
    uses the same order. ``accumulate_dtype`` selects float32 (structural match,
    the default) or float64 (rounding-free cross-check). The kernel contracts
    ``acc + x * w`` into an FFMA; with float32 accumulation the two differ by at
    most a final-ulp, which is why bitwise expectations are only claimed for
    fixtures whose products and partial sums are exactly representable.
    """
    topk = routing_weights.shape[1]
    if routed_out.shape[0] < routing_weights.shape[0] * topk:
        raise ValueError("routed_out does not cover every routed slot")
    tokens = routing_weights.shape[0]
    out = torch.empty(
        (tokens, routed_out.shape[1]), dtype=routed_out.dtype, device=routed_out.device
    )
    acc = torch.zeros(
        (tokens, routed_out.shape[1]),
        dtype=accumulate_dtype,
        device=routed_out.device,
    )
    for k in range(topk):
        slots = routed_out[k : tokens * topk : topk].to(accumulate_dtype)
        acc = acc + slots * routing_weights[:, k].to(accumulate_dtype).unsqueeze(1)
    out.copy_(acc.to(routed_out.dtype))
    return out


def bf16_spacing(magnitude: torch.Tensor) -> torch.Tensor:
    """Spacing of the bf16 grid at ``|magnitude|``.

    bf16 keeps 7 mantissa bits, so for ``|x|`` in ``[2**e, 2**(e+1))`` the
    spacing is ``2**(e-7)``. Note the *relative* spacing is therefore between
    ``2**-8`` and ``2**-7``, not a single constant.
    """
    floored = magnitude.abs().to(torch.float64).clamp_min(2.0**-126)
    exponent = torch.floor(torch.log2(floored))
    return torch.pow(torch.tensor(2.0, dtype=torch.float64), exponent - 7)


def reduce_term_scale(routed_out: torch.Tensor, routing_weights: torch.Tensor) -> float:
    """``max_n sum_k |slot_value * weight|`` -- the reduce's own scale.

    This is the denominator that makes a reduce tolerance scale-free. It is
    computed from the reference operands, never from the result.
    """
    topk = routing_weights.shape[1]
    tokens = routing_weights.shape[0]
    if routed_out.shape[0] < tokens * topk:
        raise ValueError("routed_out does not cover every routed slot")
    total = torch.zeros(
        (tokens, routed_out.shape[1]), dtype=torch.float64, device=routed_out.device
    )
    for k in range(topk):
        slot = routed_out[k : tokens * topk : topk].to(torch.float64)
        total = total + slot.abs() * routing_weights[:, k].to(torch.float64).unsqueeze(
            1
        )
    return float(total.max().item())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """``1 - 2<x, y> / (||x||^2 + ||y||^2)``; 0.0 when both are all zero."""
    x = x.detach().to(torch.float64)
    y = y.detach().to(torch.float64)
    denominator = float((x * x + y * y).sum().item())
    if denominator == 0.0:
        return 0.0
    return float(1.0 - 2.0 * float((x * y).sum().item()) / denominator)


def rel_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    """``||candidate - reference|| / ||reference||``."""
    difference = candidate.detach().to(torch.float64) - reference.detach().to(
        torch.float64
    )
    denominator = float(reference.detach().to(torch.float64).norm().item())
    if denominator == 0.0:
        return 0.0 if float(difference.norm().item()) == 0.0 else math.inf
    return float(difference.norm().item()) / denominator


def first_mismatch(
    candidate: torch.Tensor, reference: torch.Tensor
) -> list[int] | None:
    different = (candidate != reference).reshape(-1)
    index = int(different.to(torch.uint8).argmax().item()) if different.any() else -1
    if index < 0:
        return None
    return [int(v) for v in torch.unravel_index(torch.tensor(index), candidate.shape)]


@dataclass
class Comparison:
    """Correctness metrics for one candidate/reference pair."""

    finite: bool
    max_abs: float
    rel_l2: float
    calc_diff: float
    exact_fraction: float
    num_mismatch: int
    first_mismatch: list[int] | None
    worst_expert: int | None
    worst_expert_rel_l2: float | None
    term_scale: float | None = None
    max_bf16_steps: float | None = None
    max_step_excess: float | None = None

    @property
    def max_step_excess_over_term_scale(self) -> float | None:
        """Worst mismatch left after subtracting one bf16 step, over the terms.

        Both sides round an fp32 accumulator to bf16 once, so

            |candidate - reference| <= spacing(max(|c|, |r|)) + |Ac - Ar|

        where ``Ac``/``Ar`` are the two fp32 accumulators. Subtracting one
        spacing removes the rounding term; what is left is the accumulation
        disagreement, which is a few 2**-24 of the sum of absolute terms. That
        residual, divided by the term scale, is scale-free and stays meaningful
        when the top-k slots cancel and the result is near zero.
        """
        if self.max_step_excess is None:
            return None
        if self.term_scale is None or self.term_scale <= 0.0:
            return None
        return self.max_step_excess / self.term_scale

    @property
    def max_abs_over_term_scale(self) -> float | None:
        """``max_abs`` relative to the sum of absolute terms that produced it.

        ``rel_l2`` divides by the *result*, which is meaningless when the top-k
        slots cancel: a result near zero can carry a large absolute error while
        still being the correctly rounded answer. Dividing by the term scale
        instead is what makes the bound scale-free.
        """
        if self.term_scale is None or self.term_scale <= 0.0:
            return None
        return self.max_abs / self.term_scale

    def as_row(self) -> dict[str, Any]:
        return {
            "finite": self.finite,
            "max_abs": self.max_abs,
            "rel_l2": self.rel_l2,
            "calc_diff": self.calc_diff,
            "exact_fraction": self.exact_fraction,
            "num_mismatch": self.num_mismatch,
            "first_mismatch": self.first_mismatch,
            "worst_expert": self.worst_expert,
            "worst_expert_rel_l2": self.worst_expert_rel_l2,
            "term_scale": self.term_scale,
            "max_abs_over_term_scale": self.max_abs_over_term_scale,
            "max_bf16_steps": self.max_bf16_steps,
            "max_step_excess": self.max_step_excess,
            "max_step_excess_over_term_scale": self.max_step_excess_over_term_scale,
        }


def compare(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    slot_experts: torch.Tensor | None = None,
    rows: int | None = None,
    term_scale: float | None = None,
    measure_bf16_steps: bool = False,
) -> Comparison:
    """Full metric set. Non-finite values fail before any tolerance is applied."""
    if candidate.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: {tuple(candidate.shape)} vs {tuple(reference.shape)}"
        )
    view_c = candidate
    view_r = reference
    if rows is not None:
        view_c = candidate[:rows]
        view_r = reference[:rows]
    finite = bool(
        torch.isfinite(view_c).all().item() and torch.isfinite(view_r).all().item()
    )
    difference = (view_c.to(torch.float64) - view_r.to(torch.float64)).abs()
    max_bf16_steps = None
    max_step_excess = None
    if measure_bf16_steps and difference.numel():
        # Spacing is taken at the larger of the two, so one step is the gap
        # between adjacent bf16 values at that magnitude.
        spacing = bf16_spacing(torch.maximum(view_c.abs(), view_r.abs()))
        max_bf16_steps = float((difference / spacing).max().item())
        max_step_excess = float((difference - spacing).clamp_min(0.0).max().item())
    worst_expert = None
    worst_expert_rel_l2 = None
    if slot_experts is not None and view_c.shape[0]:
        experts = slot_experts[: view_c.shape[0]].to(torch.int64)
        best = -1.0
        for expert in torch.unique(experts).tolist():
            mask = experts == expert
            value = rel_l2(view_c[mask], view_r[mask])
            if value > best:
                best = value
                worst_expert = int(expert)
        worst_expert_rel_l2 = best if worst_expert is not None else None
    return Comparison(
        finite=finite,
        max_abs=float(difference.max().item()) if difference.numel() else 0.0,
        rel_l2=rel_l2(view_c, view_r),
        calc_diff=calc_diff(view_c, view_r),
        exact_fraction=(
            float((view_c == view_r).to(torch.float64).mean().item())
            if view_c.numel()
            else 1.0
        ),
        num_mismatch=int((view_c != view_r).sum().item()),
        first_mismatch=first_mismatch(view_c, view_r),
        worst_expert=worst_expert,
        worst_expert_rel_l2=worst_expert_rel_l2,
        term_scale=term_scale,
        max_bf16_steps=max_bf16_steps,
        max_step_excess=max_step_excess,
    )


def classify(
    comparison: Comparison,
    *,
    exact_required: bool,
    threshold: Threshold | None,
) -> tuple[str, str | None]:
    """Map metrics onto a status plus a machine-readable reason.

    Returns one of ``pass`` / ``fail`` / ``uncalibrated``. Callers add
    ``skipped`` when a case never ran.
    """
    if not comparison.finite:
        return "fail", "non_finite_values"
    if exact_required:
        if comparison.num_mismatch:
            return "fail", "exact_mismatch"
        return "pass", None
    if threshold is None or not threshold.calibrated:
        return "uncalibrated", "threshold_not_calibrated"
    if comparison.calc_diff > threshold.calc_diff:
        return "fail", "calc_diff_exceeded"
    residual = comparison.max_step_excess_over_term_scale
    if residual is None:
        return "fail", "step_excess_metric_missing"
    if residual > threshold.max_step_excess_over_term_scale:
        return "fail", "accumulation_error_exceeds_fp32_slack"
    return "pass", None


@dataclass(frozen=True)
class Threshold:
    """Tolerance bound plus the evidence that produced it.

    An absolute ``max_abs`` bound is not usable: routed MoE outputs scale with
    the activation magnitude, so it would be vacuous for small data and
    unreachable for large data. The bound is on

        (|candidate - reference| - one_bf16_step) / term_scale

    i.e. the part of the mismatch that the single bf16 rounding cannot explain,
    measured against the magnitude of the terms that produced it. For a fp32
    accumulator with n terms the residual is at most ``n * 2**-24`` of the term
    scale; ``2**-20`` leaves room for a handful of terms while still being a
    million times tighter than a routing or weighting mistake, which moves the
    result by order 1 of the term scale.
    """

    threshold_id: str
    calc_diff: float
    max_step_excess_over_term_scale: float
    calibrated: bool
    calibration_run: str | None = None
    note: str | None = None

    def as_row(self) -> dict[str, Any]:
        return {
            "threshold_id": self.threshold_id,
            "threshold_calc_diff": self.calc_diff,
            "threshold_max_step_excess_over_term_scale": (
                self.max_step_excess_over_term_scale
            ),
            "threshold_calibrated": self.calibrated,
            "threshold_calibration_run": self.calibration_run,
            "threshold_note": self.note,
        }


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


@dataclass
class Timing:
    """Raw CUDA-graph replay samples, in microseconds, per kernel invocation."""

    samples_us: list[float]
    repeats: int
    calls_per_repeat: int

    @property
    def calls_per_graph(self) -> int:
        return self.repeats * self.calls_per_repeat

    @property
    def p50_us(self) -> float:
        return statistics.median(self.samples_us) if self.samples_us else math.nan

    @property
    def p90_us(self) -> float:
        if not self.samples_us:
            return math.nan
        ordered = sorted(self.samples_us)
        return ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]

    @property
    def min_us(self) -> float:
        return min(self.samples_us) if self.samples_us else math.nan

    def as_row(self) -> dict[str, Any]:
        return {
            "samples_us": [round(v, 4) for v in self.samples_us],
            "p50_us": round(self.p50_us, 4) if self.samples_us else None,
            "p90_us": round(self.p90_us, 4) if self.samples_us else None,
            "min_us": round(self.min_us, 4) if self.samples_us else None,
            "num_samples": len(self.samples_us),
            "calls_per_graph": self.calls_per_graph,
        }


def capture_graph(call: Callable[[], None], *, warmup: int, graph_calls: int):
    """Capture ``graph_calls`` invocations; all warmup happens outside."""
    for _ in range(max(1, warmup)):
        call()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_calls):
            call()
    torch.accelerator.synchronize()
    return graph


def time_graph(
    call: Callable[[], None],
    *,
    warmup: int,
    repeats: int,
    samples: int,
    calls_per_repeat: int = 1,
    graph=None,
) -> Timing:
    """Event-bracketed CUDA-graph replay, normalised per kernel invocation.

    ``call`` performs ``calls_per_repeat`` kernel invocations (1 for a plain
    repeated GEMM, ``graph_calls`` for a pre-built expert-rotation sequence) and
    is replayed ``repeats`` times inside the graph, matching the original
    benchmark's graph-call convention.

    Import, compilation, allocation, data generation and reference computation
    must all have happened before this is called; capture happens once, outside
    the timing region.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if calls_per_repeat <= 0:
        raise ValueError("calls_per_repeat must be positive")
    calls_per_graph = repeats * calls_per_repeat
    if graph is None:
        graph = capture_graph(call, warmup=warmup, graph_calls=repeats)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        graph.replay()
        end.record()
    torch.accelerator.synchronize()
    per_call = [
        start.elapsed_time(end) * 1000.0 / calls_per_graph
        for start, end in zip(starts, ends, strict=True)
    ]
    return Timing(
        samples_us=per_call,
        repeats=repeats,
        calls_per_repeat=calls_per_repeat,
    )


def interleaved_ab(
    baseline_call: Callable[[], None],
    candidate_call: Callable[[], None],
    *,
    rounds: int,
    warmup: int,
    graph_calls: int,
    samples: int,
) -> dict[str, Any]:
    """Alternate baseline/candidate as A/B/B/A and keep every round.

    Returns per-round p50/p90 plus the baseline self-comparison that decides
    whether a difference is measurable at all.
    """
    if rounds < 3:
        raise ValueError("at least 3 independent rounds are required")
    baseline_graph = capture_graph(
        baseline_call, warmup=warmup, graph_calls=graph_calls
    )
    candidate_graph = capture_graph(
        candidate_call, warmup=warmup, graph_calls=graph_calls
    )
    baseline_rounds: list[Timing] = []
    candidate_rounds: list[Timing] = []
    for index in range(rounds):
        order = (
            ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
        )
        for which in order:
            if which == "baseline":
                baseline_rounds.append(
                    time_graph(
                        baseline_call,
                        warmup=0,
                        graph_calls=graph_calls,
                        samples=samples,
                        graph=baseline_graph,
                    )
                )
            else:
                candidate_rounds.append(
                    time_graph(
                        candidate_call,
                        warmup=0,
                        graph_calls=graph_calls,
                        samples=samples,
                        graph=candidate_graph,
                    )
                )
    baseline_p50 = [t.p50_us for t in baseline_rounds]
    baseline_p90 = [t.p90_us for t in baseline_rounds]
    candidate_p50 = [t.p50_us for t in candidate_rounds]
    candidate_p90 = [t.p90_us for t in candidate_rounds]
    self_spread = (
        (max(baseline_p50) - min(baseline_p50)) / statistics.median(baseline_p50)
        if len(baseline_p50) > 1 and statistics.median(baseline_p50) > 0
        else 0.0
    )
    ratio = (
        statistics.median(candidate_p50) / statistics.median(baseline_p50)
        if statistics.median(baseline_p50) > 0
        else math.nan
    )
    return {
        "rounds": rounds,
        "baseline_p50_us": [round(v, 4) for v in baseline_p50],
        "baseline_p90_us": [round(v, 4) for v in baseline_p90],
        "candidate_p50_us": [round(v, 4) for v in candidate_p50],
        "candidate_p90_us": [round(v, 4) for v in candidate_p90],
        "baseline_median_p50_us": round(statistics.median(baseline_p50), 4),
        "candidate_median_p50_us": round(statistics.median(candidate_p50), 4),
        "baseline_ratio": round(ratio, 4) if not math.isnan(ratio) else None,
        "baseline_self_spread": round(self_spread, 4),
        "measurement_status": (
            "inconclusive" if abs(ratio - 1.0) <= self_spread else "ok"
        ),
    }


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


def _pci_bus_id_for(gpu_uuid: str | None) -> str | None:
    """Resolve the PCI bus id of the selected device, matching on UUID.

    ``nvidia-smi`` is not usable on the CMP 170HX hosts (the CLI cannot talk to
    the driver), but NVML is, so this goes through ``pynvml``.
    """
    if not gpu_uuid:
        return None
    try:
        import pynvml

        pynvml.nvmlInit()
        wanted = gpu_uuid if gpu_uuid.startswith("GPU-") else f"GPU-{gpu_uuid}"
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode()
            if uuid != wanted:
                continue
            bus = pynvml.nvmlDeviceGetPciInfo(handle).busId
            return bus.decode() if isinstance(bus, bytes) else bus
    except Exception:
        return None
    return None


def _clock_state() -> dict[str, Any] | None:
    """Clocks, power limit and ECC mode, as far as NVML will report them."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        state: dict[str, Any] = {}
        for key, clock in (
            ("sm_clock_mhz", pynvml.NVML_CLOCK_SM),
            ("mem_clock_mhz", pynvml.NVML_CLOCK_MEM),
        ):
            try:
                state[key] = int(pynvml.nvmlDeviceGetClockInfo(handle, clock))
            except Exception:
                state[key] = None
        try:
            state["power_limit_mw"] = int(
                pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            )
        except Exception:
            state["power_limit_mw"] = None
        try:
            state["ecc_enabled"] = bool(
                pynvml.nvmlDeviceGetEccMode(handle)[0] == pynvml.NVML_FEATURE_ENABLED
            )
        except Exception:
            state["ecc_enabled"] = None
        return state
    except Exception:
        return None


def environment() -> dict[str, Any]:
    """Versions and identifiers needed to make a result reproducible."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception:  # pragma: no cover - triton is optional here
        info["triton"] = None
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["device_count"] = torch.accelerator.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_capability"] = list(torch.cuda.get_device_capability(0))
        try:
            info["gpu_uuid"] = str(torch.cuda.get_device_properties(0).uuid)
        except Exception:
            info["gpu_uuid"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        info["pci_bus_id"] = _pci_bus_id_for(info.get("gpu_uuid"))
        info["clock_and_power"] = _clock_state()
    else:
        info["cuda"] = None
    info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    info["cuda_device_order"] = os.environ.get("CUDA_DEVICE_ORDER")
    info["relevant_env"] = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith("VLLM_DEEPSEEK_V4_SM80") or key.startswith("MXFP4")
    }
    return info


def git_head(repo: Path | None = None) -> str | None:
    try:
        root = repo or Path(__file__).resolve().parents[2]
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def dirty_diff_hash(repo: Path | None = None) -> str | None:
    try:
        root = repo or Path(__file__).resolve().parents[2]
        out = subprocess.run(
            ["git", "-C", str(root), "diff"],
            capture_output=True,
            text=True,
            check=True,
        )
        return hashlib.sha256(out.stdout.encode()).hexdigest()[:16]
    except Exception:
        return None


def sanitize(value: Any) -> Any:
    """Make a value JSON-safe: no NaN/Infinity tokens in the output."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    return value


@dataclass
class ResultRow:
    """One case/stage/variant result. Field set follows the harness contract."""

    run_id: str
    case_id: str
    stage: str
    variant: str
    values: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        row = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "stage": self.stage,
            "variant": self.variant,
        }
        row.update(self.values)
        return sanitize(row)


class JsonlWriter:
    """Append-only JSONL sink; one line per case/stage/variant."""

    def __init__(self, path: Path | str | None):
        self.path = Path(path) if path is not None else None
        self._handle = None

    def __enter__(self) -> JsonlWriter:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
        return self

    def write(self, row: dict[str, Any]) -> None:
        line = json.dumps(sanitize(row), sort_keys=True)
        if self._handle is not None:
            self._handle.write(line + "\n")
            self._handle.flush()
        else:
            print(line)

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def write_summary(path: Path | str, rows: list[dict[str, Any]], **extra: Any) -> None:
    """Atomically write the run summary, keeping every failing case visible."""
    target = Path(path)
    statuses: dict[str, int] = {}
    for row in rows:
        status = str(
            row.get("correctness_status") or row.get("measurement_status") or "unknown"
        )
        statuses[status] = statuses.get(status, 0) + 1
    failures = [
        {
            "case_id": row.get("case_id"),
            "stage": row.get("stage"),
            "variant": row.get("variant"),
            "correctness_status": row.get("correctness_status"),
            "skip_or_failure_reason": row.get("skip_or_failure_reason"),
        }
        for row in rows
        if row.get("correctness_status") == "fail"
        or row.get("skip_or_failure_reason") is not None
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "num_rows": len(rows),
        "status_counts": statuses,
        "failures_and_skips": failures,
        "all_passed": bool(rows) and not failures,
        **extra,
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(sanitize(summary), indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


# ---------------------------------------------------------------------------
# Packed-weight fixtures
# ---------------------------------------------------------------------------

# Hand-derived anchor. Only lanes 0..4 are written; every other byte is zero.
# lane = (n % 8) * 4 + thread_in_group, byte j of a lane covers k-pair
# _PACK_BYTE_PERMUTATION[thread_in_group * 4 + j]; the low nibble is the even k
# of the pair and the high nibble the odd k.
MANUAL_ANCHOR_K = 128
MANUAL_ANCHOR_N = 8
MANUAL_ANCHOR_LANES: dict[int, tuple[int, int, int, int]] = {
    0: (0x01, 0x12, 0x00, 0x8F),  # (n=0, t=0): +1, +2, zeros, -12
    1: (0x23, 0x00, 0xF4, 0x00),  # (n=0, t=1)
    2: (0x45, 0x00, 0x00, 0x00),  # (n=0, t=2)
    3: (0x06, 0x00, 0x00, 0x00),  # (n=0, t=3)
    4: (0x08, 0x00, 0x00, 0x00),  # (n=1, t=0): negative zero
}
# shift2[0, 0, 0, n_in_group] packs the four kb fields as 2 bits each: field
# kb%4 is at bit offset 2*(kb%4), so n=0 gets shifts (0, 1, 2, 3) and n=1 gets
# (3, 3, 0, 0). All four shift values are therefore exercised.
MANUAL_ANCHOR_SHIFTS = {0: 0b11_10_01_00, 1: 0b00_00_11_11}
# (n, k, decoded int8). Derived by hand from the lane table above.
# n=0 reaches its k values through four different lanes, so the byte
# permutation is exercised: lane 0 (t=0) covers k pairs 0,1,8,9; lane 1 (t=1)
# covers 2,3,10,11; lane 2 (t=2) covers 4,5,12,13; lane 3 (t=3) covers
# 6,7,14,15. n=1 is the only value served by lane 4 (group_id 1, t=0).
MANUAL_ANCHOR_EXPECTED: list[tuple[int, int, int]] = [
    # n = 0, kb = 0 (shift 0): every magnitude and sign-bit case.
    (0, 0, 1),  # code 0x1 -> +1
    (0, 1, 0),  # code 0x0 -> positive zero
    (0, 2, 2),  # code 0x2 -> +2
    (0, 4, 3),  # code 0x3, via lane 1 -> +3
    (0, 8, 6),  # code 0x5, via lane 2 -> magnitude index 5 -> +6
    (0, 12, 8),  # code 0x6, via lane 3 -> +8
    (0, 18, -12),  # code 0xF -> magnitude 12, sign bit -> -12
    (0, 19, 0),  # code 0x8 -> negative zero decodes to 0
    # n = 0, kb = 1 (shift 1)
    (0, 32, 2),
    (0, 36, 6),
    (0, 50, -24),
    # n = 0, kb = 2 (shift 2)
    (0, 64, 4),
    (0, 70, 0),  # k32 = 6 lives in lane 1 byte 1, which is zero
    (0, 72, 24),  # lane 2 byte 0 low nibble -> +6, shifted by 2
    (0, 76, 32),  # lane 3 byte 0 low nibble -> +8, shifted by 2
    (0, 82, -48),
    # n = 0, kb = 3 (shift 3), the largest representable magnitude
    (0, 96, 8),
    (0, 100, 24),
    (0, 114, -96),
    # n = 1: lane 4 stores code 0x8 (negative zero) at k32 = 0 in every kb, so
    # all four shift fields must decode to 0. Unwritten bytes stay 0 as well.
    (1, 0, 0),
    (1, 32, 0),
    (1, 64, 0),
    (1, 96, 0),
    (1, 1, 0),
]


def manual_anchor_tensors(
    *, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hand-built packed bytes for K=128, N=8, one expert.

    Covers the lane permutation, both nibbles of a byte, positive/negative zero,
    and all four shift fields. Built from literal bytes, never by the production
    packing code, so agreement with :func:`decode_packed_weights` is evidence
    about the format rather than a self-fulfilling round trip.
    """
    b_mxfp4 = torch.zeros((1, 4, 1, 128), dtype=torch.uint8, device=device)
    for lane, four_bytes in MANUAL_ANCHOR_LANES.items():
        for kb in range(4):
            for j, value in enumerate(four_bytes):
                b_mxfp4[0, kb, 0, lane * 4 + j] = value
    b_shift2 = torch.zeros((1, 1, 1, 8), dtype=torch.uint8, device=device)
    for n_in_group, packed in MANUAL_ANCHOR_SHIFTS.items():
        b_shift2[0, 0, 0, n_in_group] = packed
    return b_mxfp4, b_shift2


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def self_check(*, device: torch.device | str = "cpu") -> dict[str, Any]:
    """CPU-only checks for the metrics, the format and the byte accounting.

    Runnable without CUDA so that the offline facility can be validated on a
    machine with no GPU.
    """
    results: dict[str, Any] = {}

    a = manual_anchor_tensors(device=device)
    decoded, shifts = decode_packed_weights(
        a[0], a[1], n=MANUAL_ANCHOR_N, k=MANUAL_ANCHOR_K
    )
    scalar = decode_packed_weights_scalar(
        a[0], a[1], n=MANUAL_ANCHOR_N, k=MANUAL_ANCHOR_K
    )
    vector_ok = all(
        int(decoded[0, n, k].item()) == expected
        for n, k, expected in MANUAL_ANCHOR_EXPECTED
    )
    scalar_ok = all(
        int(scalar[0, n, k].item()) == expected
        for n, k, expected in MANUAL_ANCHOR_EXPECTED
    )
    results["manual_anchor_vector"] = vector_ok
    results["manual_anchor_scalar"] = scalar_ok
    results["manual_anchor_matches_scalar"] = bool(torch.equal(decoded, scalar))
    results["decoded_range"] = [int(decoded.min().item()), int(decoded.max().item())]
    results["shift_range"] = [int(shifts.min().item()), int(shifts.max().item())]

    # Metrics: identical tensors, zero reference, NaN handling.
    same = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    results["calc_diff_identity"] = calc_diff(same, same) == 0.0
    results["rel_l2_identity"] = rel_l2(same, same) == 0.0
    zero = torch.zeros(2, 2, dtype=torch.bfloat16)
    results["calc_diff_both_zero"] = calc_diff(zero, zero) == 0.0
    results["rel_l2_zero_reference"] = rel_l2(zero, zero) == 0.0
    results["rel_l2_reference_zero_candidate_nonzero"] = math.isinf(rel_l2(same, zero))
    adjacent = torch.tensor([[1.0], [2.0]], dtype=torch.bfloat16)
    next_up = torch.tensor([[1.0078125], [2.015625]], dtype=torch.bfloat16)
    two_up = torch.tensor([[1.015625], [2.03125]], dtype=torch.bfloat16)
    one = compare(adjacent, next_up, term_scale=1.0, measure_bf16_steps=True)
    two = compare(adjacent, two_up, term_scale=1.0, measure_bf16_steps=True)
    same = compare(adjacent, adjacent, term_scale=1.0, measure_bf16_steps=True)
    results["one_step_has_no_excess"] = (
        one.max_bf16_steps == 1.0 and one.max_step_excess == 0.0
    )
    results["two_steps_leaves_excess"] = (
        two.max_bf16_steps == 2.0 and two.max_step_excess > 0.0
    )
    results["identical_has_zero_steps"] = (
        same.max_bf16_steps == 0.0 and same.max_step_excess == 0.0
    )
    # The relative spacing of bf16 is not a single constant: it is 2**-7 just
    # above a power of two and approaches 2**-8 at the top of a binade. Getting
    # this wrong is what makes a naive max_abs budget too tight by 2x.
    relative = [
        float((bf16_spacing(torch.tensor([value])) / value).item())
        for value in (1.0, 1.9999, 2.0, 3.9999)
    ]
    results["bf16_relative_spacing_range"] = all(
        2.0**-8 < ratio <= 2.0**-7 + 1e-12 for ratio in relative
    )
    results["bf16_relative_spacing_endpoints"] = (
        abs(relative[0] - 2.0**-7) < 1e-12 and abs(relative[1] - 2.0**-8) < 1e-4
    )
    nan_row = compare(
        torch.tensor([[float("nan")]], dtype=torch.bfloat16),
        torch.tensor([[1.0]], dtype=torch.bfloat16),
    )
    results["non_finite_is_fail"] = (
        classify(nan_row, exact_required=True, threshold=None)[0] == "fail"
    )

    # Byte accounting: the same expert hit twice must not double count.
    weights = {
        W13_STAGE: (
            torch.zeros((4, 160, 576, 128), dtype=torch.uint8),
            torch.zeros((4, 40, 576, 8), dtype=torch.uint8),
            torch.zeros((4, 4608), dtype=torch.float32),
        ),
        W2_STAGE: (
            torch.zeros((4, 72, 640, 128), dtype=torch.uint8),
            torch.zeros((4, 18, 640, 8), dtype=torch.uint8),
            torch.zeros((4, 5120), dtype=torch.float32),
        ),
    }
    single = compute_route_work(
        torch.tensor([[0, 1, 2]], dtype=torch.int32), 16, weights
    )
    # Two tokens hitting the same three experts: the expert footprint must not
    # change, while slot-derived work doubles.
    repeated = compute_route_work(
        torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.int32), 16, weights
    )
    one_expert = compute_route_work(
        torch.tensor([[0, 0, 0]], dtype=torch.int32), 16, weights
    )
    results["active_experts_deduplicated"] = (
        len(single.active_experts) == 3
        and len(repeated.active_experts) == 3
        and len(one_expert.active_experts) == 1
    )
    results["active_bytes_ignore_reuse"] = (
        single.stage_bytes[W13_STAGE]["total"]
        == repeated.stage_bytes[W13_STAGE]["total"]
    )
    results["active_bytes_drop_with_fewer_experts"] = (
        one_expert.stage_bytes[W13_STAGE]["total"]
        < single.stage_bytes[W13_STAGE]["total"]
    )
    results["slots_count_every_hit"] = (
        single.slots == 3 and repeated.slots == 6 and one_expert.slots == 3
    )
    results["useful_flops_track_slots"] = (
        repeated.useful_flops[W13_STAGE] == 2 * single.useful_flops[W13_STAGE]
    )
    results["m_tiles_track_block_m"] = (
        compute_route_work(
            torch.tensor([[0] * 16], dtype=torch.int32), 16, weights
        ).m_tiles
        == 1
        and compute_route_work(
            torch.tensor([[0] * 16 + [1]], dtype=torch.int32), 16, weights
        ).m_tiles
        == 2
    )

    expected_w13_bytes = 160 * 576 * 128 + 40 * 576 * 8 + 4608 * 4
    results["w13_bytes_formula"] = (
        stage_weight_bytes(*weights[W13_STAGE], [0])["total"] == expected_w13_bytes
    )
    results["w13_w2_bytes_are_not_equal"] = (
        weights[W13_STAGE][0][0].numel() + weights[W13_STAGE][1][0].numel()
        != weights[W2_STAGE][0][0].numel() + weights[W2_STAGE][1][0].numel()
    )

    # Mandated CPU arithmetic check: hold the hit experts fixed and grow M.
    growth = []
    for tokens in (1, 2, 3, 4):
        ids = torch.tensor([[0, 1, 2, 3]] * tokens, dtype=torch.int32)
        growth.append(compute_route_work(ids, 16, weights))
    results["active_bytes_constant_over_M"] = (
        len({row.stage_bytes[W13_STAGE]["total"] for row in growth}) == 1
    )
    results["useful_flops_grow_with_slots"] = [
        row.useful_flops[W13_STAGE] for row in growth
    ] == [2 * tokens * 4 * 4608 * 5120 for tokens in (1, 2, 3, 4)]
    results["w13_plus_w2_bytes_additive"] = all(
        row.stage_bytes[W13_STAGE]["total"] + row.stage_bytes[W2_STAGE]["total"]
        == sum(
            row.stage_bytes[stage][key]
            for stage in (W13_STAGE, W2_STAGE)
            for key in ("packed", "shift2", "channel_scale")
        )
        for row in growth
    ) and (
        growth[0].stage_bytes[W13_STAGE]["total"]
        == 4 * stage_weight_bytes(*weights[W13_STAGE], [0])["total"]
        and growth[0].stage_bytes[W2_STAGE]["total"]
        == 4 * stage_weight_bytes(*weights[W2_STAGE], [0])["total"]
        and growth[0].stage_bytes[W13_STAGE]["total"]
        + growth[0].stage_bytes[W2_STAGE]["total"]
        == 4
        * (
            stage_weight_bytes(*weights[W13_STAGE], [0])["total"]
            + stage_weight_bytes(*weights[W2_STAGE], [0])["total"]
        )
    )
    # Crossing a block_m boundary must add a tile.
    cross = [
        compute_route_work(torch.tensor([[0] * count], dtype=torch.int32), 16, weights)
        for count in (15, 16, 17, 32, 33)
    ]
    results["m_tiles_cross_block_m"] = [row.m_tiles for row in cross] == [1, 1, 2, 2, 3]
    results["padded_rows_cross_block_m"] = [row.padded_rows for row in cross] == [
        16,
        16,
        32,
        32,
        48,
    ]

    # The INT32 accumulator can never saturate for the accepted input range.
    results["int32_bound_w13"] = int32_accumulator_bound(STAGE_SHAPES[W13_STAGE][1])
    results["int32_bound_w2"] = int32_accumulator_bound(STAGE_SHAPES[W2_STAGE][1])
    results["int32_no_saturation"] = all(
        int32_accumulator_bound(k) < INT32_MAX for _, k, _ in STAGE_SHAPES.values()
    )

    # Routing: no in-token expert repeats for every pattern.
    for pattern in ("uniform", "hot", "sparse"):
        ids = make_routing_ids(8, 3, 32, pattern, 4242)
        results[f"routing_exclusive_{pattern}"] = token_expert_exclusivity(ids)
    results["routing_rejects_duplicate_experts"] = False
    try:
        make_routing_ids_from_slots([0, 0, 1], topk=3)
    except ValueError:
        results["routing_rejects_duplicate_experts"] = True
    try:
        make_routing_ids(1, 6, 4, "uniform", 0)
    except ValueError:
        results["routing_rejects_experts_lt_topk"] = True
    else:
        results["routing_rejects_experts_lt_topk"] = False

    # Reduce: exact data is bit-exact, and weights are applied exactly once.
    routed = torch.tensor(
        [[1.0], [2.0], [4.0]], dtype=torch.bfloat16
    )  # slots of one token, topk=3
    w = torch.tensor([[0.25, 0.25, 0.5]], dtype=torch.float32)
    reduced = reference_reduce(routed, w)
    results["reduce_power_of_two_exact"] = bool(
        torch.equal(reduced, torch.tensor([[2.75]], dtype=torch.bfloat16))
    )
    zero_weight = reference_reduce(
        routed, torch.tensor([[0.0, 0.5, 0.5]], dtype=torch.float32)
    )
    results["reduce_zero_weight_slot"] = bool(
        torch.equal(zero_weight, torch.tensor([[3.0]], dtype=torch.bfloat16))
    )
    results["reduce_is_not_renormalised"] = bool(
        torch.equal(
            reference_reduce(
                routed, torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)
            ),
            torch.tensor([[3.5]], dtype=torch.bfloat16),
        )
    )

    results["all_ok"] = all(
        value
        for key, value in results.items()
        if key not in ("decoded_range", "shift_range")
    )
    return results


# ---------------------------------------------------------------------------
# CLI for the self-check
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="run the CPU-only metric/format/accounting self-check",
    )
    parser.add_argument("--json", type=Path, help="also write the self-check result")
    args = parser.parse_args(argv)
    if not args.self_check:
        parser.error("nothing to do; pass --self-check")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = self_check(device=device)
    print(json.dumps(results, indent=2, sort_keys=True))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return 0 if results["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
