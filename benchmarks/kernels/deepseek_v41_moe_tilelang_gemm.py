# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang grouped GEMM candidate for the DSV4.1-Flash SM80 routed MoE.

Scope: the two routed grouped GEMM stages (W13, W2) of the
``mxfp4_int8_sm80`` backend on SM80 only. This module is an offline experiment:
it does not register a backend, change a default or touch any production path.

Contract reproduced from ``csrc/libtorch_stable/moe/mxfp4_int8_sm80``:

* A is INT8 ``[rows, K]`` with ``rows = tokens`` (W13) or ``rows = slots`` (W2),
  a per-row FP32 scale, and the A row of slot ``s`` is ``s // topk`` for W13
  and ``s`` for W2.
* B stays in the production packed format ``[E, K/32, ceil(N/8), 128]`` uint8
  plus ``[E, ceil(K/32/4), ceil(N/8), 8]`` 2-bit shift fields; the kernel
  decodes it to INT8, so the packed bytes are read exactly once per pass and
  no expanded INT8 copy of the weights exists anywhere.
* The accumulator is INT32 (m16n8k32 INT8 MMA); the operand ranges keep
  ``|acc| <= 127 * 96 * K < INT32_MAX`` so saturation is unnecessary.
* Epilogue: ``rne_bf16(float32(acc) * a_scale[src_row] * channel_scale[e, n])``
  with the activation scale applied first.
* Writes go to the caller's ``routed_out[slot, :]``; rows whose
  ``sorted_ids`` entry is not a valid slot are not written.

The scalar/LUT decoders reconstruct the documented layout. The optional
``prmt`` decoder imports a CUDA word-level helper into the TileLang kernel;
all variants are checked against the independent structural reference.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deepseek_v41_moe_harness import (  # noqa: E402
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    W2_STAGE,
    W13_STAGE,
    align_routes,
    make_routing_ids,
    reference_stage,
)

from vllm.tilelang_utils import T, tilelang  # noqa: E402

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
}

# Magnitudes of the stored E2M1 code, already carrying the x2 factor, packed
# four bits per entry: nibble i holds ``_MAG_X2[i]``. Stored in the signed
# int32 form of ``0xC8643210`` so the literal fits TIR's int32 constants; an
# arithmetic shift keeps the requested nibble valid under the ``& 0xF``.
_MAG_PACKED = 0xC8643210 - (1 << 32)

# Per-inner-byte index arithmetic: for k_pair ``p`` the packed byte of the
# ``nn``-th row of an 8-row group is ``nn * 16 + ((p % 8) // 2) * 4 +
# (p // 8) * 2 + (p % 2)``.
_BYTES_PER_N_GROUP = 16

_PRMT_DECODE_SOURCE = r"""
#include <stdint.h>
__device__ __forceinline__ uint32_t dsv41_negate_i8x4(uint32_t magnitude) {
  return (0x80808080u - magnitude) ^ 0x80808080u;
}
__device__ __forceinline__ uint32_t dsv41_decode_x1(uint32_t q4, uint32_t shift) {
  uint32_t magnitude = __byte_perm(0x03020100u, 0x0c080604u, q4) << shift;
  uint32_t negative = dsv41_negate_i8x4(magnitude);
  uint32_t select = 0x3210u | ((q4 & 0x8888u) >> 1);
  return __byte_perm(magnitude, negative, select);
}
__device__ __forceinline__ uint64_t dsv41_decode_x2(uint32_t q4_pair, uint32_t shift) {
  uint32_t lo = dsv41_decode_x1(q4_pair, shift);
  uint32_t hi = dsv41_decode_x1(q4_pair >> 16, shift);
  return (static_cast<uint64_t>(hi) << 32) | lo;
}
"""


def align_capacity(num_valid_tokens: int, block_m: int, num_experts: int) -> int:
    """``moe_align_block_size``'s ``sorted_token_ids`` capacity."""
    if num_valid_tokens < num_experts:
        return num_valid_tokens * block_m
    return num_valid_tokens + num_experts * (block_m - 1)


@dataclass(frozen=True)
class GemmConfig:
    """Compile-time configuration of one candidate kernel."""

    n: int
    k: int
    num_experts: int
    topk: int
    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    threads: int
    source_rows_are_slots: bool
    decode: str = "mxfp4"

    @property
    def label(self) -> str:
        stage = "w2" if self.source_rows_are_slots else "w13"
        suffix = "" if self.decode == "mxfp4" else f"-{self.decode}"
        return (
            f"{stage}-{self.block_m}x{self.block_n}x{self.block_k}"
            f"-s{self.num_stages}-t{self.threads}{suffix}"
        )


def build_kernel(
    config: GemmConfig,
    m_in: int,
    max_m: int,
    max_m_tiles: int,
    slots: int,
):
    """Compile one grouped GEMM variant for a fixed set of runtime shapes."""
    n, k = config.n, config.k
    num_experts = config.num_experts
    topk = config.topk
    block_m, block_n, block_k = config.block_m, config.block_n, config.block_k
    num_stages, threads = config.num_stages, config.threads
    source_rows_are_slots = config.source_rows_are_slots
    decode = config.decode

    k_blocks = k // 32
    shift_stride = (k_blocks + 3) // 4
    n_groups8 = n // 8
    kb_tile = block_k // 32
    ng_tile = block_n // 8
    src_div = 1 if source_rows_are_slots else topk
    k_iters = k // block_k
    assert kb_tile in (1, 2, 4), f"block_k {block_k} must be 32, 64 or 128"
    assert n % block_n == 0 and k % block_k == 0 and kb_tile <= 4
    assert decode in ("mxfp4", "lut", "luta", "prmt", "raw")

    def byte_offset(nn, kk32):
        """Packed byte holding both nibbles of ``k_pair = kk32 // 2``."""
        pair = kk32 // 2
        return (
            (nn % 8) * _BYTES_PER_N_GROUP
            + ((pair % 8) // 2) * 4
            + (pair // 8) * 2
            + (pair % 2)
        )

    @T.prim_func
    def grouped_gemm(
        a_q: T.Tensor((m_in, k), "int8"),
        a_scale: T.Tensor((m_in,), "float32"),
        b_packed: T.Tensor((num_experts, k_blocks, n_groups8, 128), "uint8"),
        b_shift2: T.Tensor((num_experts, shift_stride, n_groups8, 8), "uint8"),
        b_channel_scale: T.Tensor((num_experts, n), "float32"),
        sorted_ids: T.Tensor((max_m,), "int32"),
        expert_ids: T.Tensor((max_m_tiles,), "int32"),
        num_post_pad: T.Tensor((1,), "int32"),
        routed_out: T.Tensor((slots, n), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(n, block_n), max_m_tiles, threads=threads) as (bx, by):
            if decode == "prmt":
                T.import_source(_PRMT_DECODE_SOURCE)
            m0 = by * block_m
            if m0 < num_post_pad[0]:
                expert = expert_ids[by]
                ng0 = bx * ng_tile

                a_s = T.alloc_shared((block_m, block_k), "int8")
                p_s = T.alloc_shared((kb_tile, ng_tile, 128), "uint8")
                sh_s = T.alloc_shared((ng_tile, 8), "uint8")
                asc_s = T.alloc_shared((block_m,), "float32")
                row_s = T.alloc_shared((block_m,), "int32")
                lut_s = T.alloc_shared((4 * 256,), "int32")
                b_frag = T.alloc_fragment((block_n, block_k), "int8")
                c_i = T.alloc_fragment((block_m, block_n), "int32")
                c_f = T.alloc_fragment((block_m, block_n), "float32")

                if decode in ("lut", "luta"):
                    # (shift, packed byte) -> the byte's two decoded INT8
                    # values in the low half-word. Built once per CTA.
                    for idx in T.Parallel(4 * 256):
                        lut_shift = idx // 256
                        lut_byte = idx % 256
                        lo_code = lut_byte & 0xF
                        hi_code = (lut_byte >> 4) & 0xF
                        lo_mag = (_MAG_PACKED >> (4 * (lo_code & 7))) & 0xF
                        hi_mag = (_MAG_PACKED >> (4 * (hi_code & 7))) & 0xF
                        lo_val = (
                            T.if_then_else((lo_code & 8) != 0, -lo_mag, lo_mag)
                            << lut_shift
                        )
                        hi_val = (
                            T.if_then_else((hi_code & 8) != 0, -hi_mag, hi_mag)
                            << lut_shift
                        )
                        lut_s[idx] = (lo_val & 0xFF) | ((hi_val & 0xFF) << 8)

                for i in T.Parallel(block_m):
                    row_s[i] = (
                        T.if_then_else(m0 + i < num_post_pad[0], sorted_ids[m0 + i], 0)
                        // src_div
                    )
                for i in T.Parallel(block_m):
                    asc_s[i] = a_scale[row_s[i]]

                T.clear(c_i)
                # Keep all three shared copies ahead of their consumers.
                # Depth one has no overlap; greater depths expand the buffers.
                for ko in T.Pipelined(
                    k_iters,
                    order=[0, 1, 2, 3, 4],
                    stage=[0, 0, 0, max(0, num_stages - 1), max(0, num_stages - 1)],
                ):
                    for i, kk in T.Parallel(block_m, block_k):
                        a_s[i, kk] = a_q[row_s[i], ko * block_k + kk]
                    T.copy(b_packed[expert, ko * kb_tile, ng0, 0], p_s)
                    T.copy(b_shift2[expert, (ko * kb_tile) // 4, ng0, 0], sh_s)
                    if decode == "mxfp4":
                        for nn, kk in T.Parallel(block_n, block_k):
                            kk32 = kk % 32
                            kb_rel = kk // 32
                            nib = kk32 % 2
                            code = T.Cast(
                                "int32",
                                (
                                    p_s[kb_rel, nn // 8, byte_offset(nn, kk32)]
                                    >> (4 * nib)
                                )
                                & 0xF,
                            )
                            mag = (_MAG_PACKED >> (4 * (code & 7))) & 0xF
                            shift = T.Cast(
                                "int32",
                                (
                                    sh_s[nn // 8, nn % 8]
                                    >> (2 * ((ko * kb_tile + kb_rel) % 4))
                                )
                                & 3,
                            )
                            b_frag[nn, kk] = T.Cast(
                                "int8",
                                T.if_then_else((code & 8) != 0, -mag, mag) << shift,
                            )
                    elif decode == "lut":
                        # One shared lookup per packed byte yields both of its
                        # decoded INT8 values, so the per-element cost is a
                        # smem load plus two byte extracts instead of a
                        # magnitude table shift, a sign select and a shift.
                        for nn, kk in T.Parallel(block_n, block_k):
                            kk32 = kk % 32
                            kb_rel = kk // 32
                            nib = kk32 % 2
                            shift = T.Cast(
                                "int32",
                                (
                                    sh_s[nn // 8, nn % 8]
                                    >> (2 * ((ko * kb_tile + kb_rel) % 4))
                                )
                                & 3,
                            )
                            pair = lut_s[
                                shift * 256
                                + T.Cast(
                                    "int32", p_s[kb_rel, nn // 8, byte_offset(nn, kk32)]
                                )
                            ]
                            b_frag[nn, kk] = T.Cast(
                                "int8",
                                T.if_then_else(
                                    nib == 0, pair & 0xFF, (pair >> 8) & 0xFF
                                ),
                            )
                    elif decode == "luta":
                        # Same lookup, but the loop walks the packed layout
                        # directly: the byte address is affine in the loop
                        # variables and k_pair is the closed form of the byte
                        # permutation, so no per-element index arithmetic is
                        # left. Each packed byte writes both of its values.
                        for nn, kb_rel, t, j, nib in T.Parallel(
                            block_n, kb_tile, 4, 4, 2
                        ):
                            b_frag[
                                nn,
                                kb_rel * 32 + (j // 2) * 16 + t * 4 + (j % 2) * 2 + nib,
                            ] = T.Cast(
                                "int8",
                                T.if_then_else(
                                    nib == 0,
                                    lut_s[
                                        T.Cast(
                                            "int32",
                                            (
                                                sh_s[nn // 8, nn % 8]
                                                >> (2 * ((ko * kb_tile + kb_rel) % 4))
                                            )
                                            & 3,
                                        )
                                        * 256
                                        + T.Cast(
                                            "int32",
                                            p_s[
                                                kb_rel,
                                                nn // 8,
                                                (nn % 8) * _BYTES_PER_N_GROUP
                                                + t * 4
                                                + j,
                                            ],
                                        )
                                    ]
                                    & 0xFF,
                                    (
                                        lut_s[
                                            T.Cast(
                                                "int32",
                                                (
                                                    sh_s[nn // 8, nn % 8]
                                                    >> (
                                                        2
                                                        * ((ko * kb_tile + kb_rel) % 4)
                                                    )
                                                )
                                                & 3,
                                            )
                                            * 256
                                            + T.Cast(
                                                "int32",
                                                p_s[
                                                    kb_rel,
                                                    nn // 8,
                                                    (nn % 8) * _BYTES_PER_N_GROUP
                                                    + t * 4
                                                    + j,
                                                ],
                                            )
                                        ]
                                        >> 8
                                    )
                                    & 0xFF,
                                ),
                            )
                    elif decode == "prmt":
                        # Decode four packed bytes with the CUDA PRMT helper.
                        # The output bytes span two separate 16-column K regions.
                        for nn, kb_rel, t, j, nib in T.Parallel(
                            block_n, kb_tile, 4, 4, 2
                        ):
                            shift = T.Cast(
                                "int32",
                                (
                                    sh_s[nn // 8, nn % 8]
                                    >> (2 * ((ko * kb_tile + kb_rel) % 4))
                                )
                                & 3,
                            )
                            base = (nn % 8) * _BYTES_PER_N_GROUP + t * 4
                            word = (
                                T.Cast("uint32", p_s[kb_rel, nn // 8, base + 0])
                                | (
                                    T.Cast("uint32", p_s[kb_rel, nn // 8, base + 1])
                                    << 8
                                )
                                | (
                                    T.Cast("uint32", p_s[kb_rel, nn // 8, base + 2])
                                    << 16
                                )
                                | (
                                    T.Cast("uint32", p_s[kb_rel, nn // 8, base + 3])
                                    << 24
                                )
                            )
                            decoded = T.call_extern(
                                "uint64", "dsv41_decode_x2", word, shift
                            )
                            b_frag[
                                nn,
                                kb_rel * 32 + (j // 2) * 16 + t * 4 + (j % 2) * 2 + nib,
                            ] = T.Cast("int8", (decoded >> (8 * (j * 2 + nib))) & 0xFF)
                    else:
                        # Deliberately wrong ablation; compiler simplification
                        # can change traffic and resources as well as decoding.
                        for nn, kk in T.Parallel(block_n, block_k):
                            b_frag[nn, kk] = T.Cast(
                                "int8", p_s[kk // 32, nn // 8, (nn % 8) * 16 + kk % 16]
                            )
                    T.gemm(a_s, b_frag, c_i, transpose_B=True)

                for i, j in T.Parallel(block_m, block_n):
                    c_f[i, j] = (
                        T.Cast("float32", c_i[i, j])
                        * asc_s[i]
                        * b_channel_scale[expert, ng0 * 8 + j]
                    )
                for i, j in T.Parallel(block_m, block_n):
                    slot = sorted_ids[m0 + i]
                    if (m0 + i < num_post_pad[0]) and (slot < slots):
                        routed_out[slot, ng0 * 8 + j] = T.Cast("bfloat16", c_f[i, j])

    return grouped_gemm


@dataclass
class TileLangMoeGemm:
    """A compiled candidate plus the metadata its launch needs."""

    config: GemmConfig
    kernel: Any
    max_m: int
    max_m_tiles: int
    slots: int
    m_in: int
    compile_seconds: float = 0.0
    source: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        config: GemmConfig,
        *,
        tokens: int,
        slots: int,
    ) -> "TileLangMoeGemm":
        m_in = slots if config.source_rows_are_slots else tokens
        max_m = align_capacity(slots, config.block_m, config.num_experts)
        max_m_tiles = (max_m + config.block_m - 1) // config.block_m
        start = time.perf_counter()
        func = build_kernel(config, m_in, max_m, max_m_tiles, slots)
        kernel = tilelang.compile(func, pass_configs=PASS_CONFIGS)
        seconds = time.perf_counter() - start
        return cls(
            config=config,
            kernel=kernel,
            max_m=max_m,
            max_m_tiles=max_m_tiles,
            slots=slots,
            m_in=m_in,
            compile_seconds=seconds,
            source=kernel.get_kernel_source(),
        )

    def run(
        self,
        a_q: torch.Tensor,
        a_scale: torch.Tensor,
        b_packed: torch.Tensor,
        b_shift2: torch.Tensor,
        b_channel_scale: torch.Tensor,
        routing_ids: torch.Tensor,
        routed_out: torch.Tensor,
        *,
        sorted_ids: torch.Tensor | None = None,
        expert_ids: torch.Tensor | None = None,
        num_post_pad: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if sorted_ids is None:
            sorted_ids, expert_ids, num_post_pad = align_routes(
                routing_ids, self.config.block_m, self.config.num_experts
            )
        self.kernel(
            a_q,
            a_scale,
            b_packed,
            b_shift2,
            b_channel_scale,
            sorted_ids,
            expert_ids,
            num_post_pad,
            routed_out,
        )
        return {
            "sorted_ids": sorted_ids,
            "expert_ids": expert_ids,
            "num_post_pad": num_post_pad,
        }


# ---------------------------------------------------------------------------
# Offline correctness driver
# ---------------------------------------------------------------------------


def candidate_routes(
    tokens: int, topk: int, num_experts: int, pattern: str, seed: int
) -> torch.Tensor:
    if pattern == "tile-boundary":
        # Exactly tokens slots per selected expert: M15/16/17 cross BM16.
        return make_routing_ids(1, topk, num_experts, "uniform", seed).repeat(tokens, 1)
    return make_routing_ids(tokens, topk, num_experts, pattern, seed)


def activation_scales(rows: int, device: torch.device, mode: str) -> torch.Tensor:
    if mode == "unit":
        return torch.ones(rows, dtype=torch.float32, device=device)
    return (
        0.037
        + torch.arange(rows, dtype=torch.float32, device=device).remainder(997) / 811
    )


def build_packed_weights(
    num_experts: int,
    stage: str,
    device: torch.device,
    *,
    seed: int = 0,
    scale_mode: str = "varied",
) -> dict[str, torch.Tensor]:
    """Deterministic synthetic weights in the production packed layout."""
    if stage == W13_STAGE:
        n, k = 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE
    else:
        n, k = HIDDEN_SIZE, INTERMEDIATE_SIZE
    generator = torch.Generator(device=device).manual_seed(seed)
    packed = torch.empty(
        (num_experts, k // 32, n // 8, 128), dtype=torch.uint8, device=device
    ).random_(0, 256, generator=generator)
    shift2 = torch.empty(
        (num_experts, (k // 32 + 3) // 4, n // 8, 8),
        dtype=torch.uint8,
        device=device,
    ).random_(0, 256, generator=generator)
    if scale_mode == "unit":
        channel_scale = torch.ones((num_experts, n), dtype=torch.float32, device=device)
    elif scale_mode == "varied":
        channel_scale = (
            0.0313
            + torch.arange(num_experts * n, device=device, dtype=torch.float32)
            .remainder(997)
            .reshape(num_experts, n)
            / 811.0
        )
    else:
        raise ValueError(f"unknown scale mode {scale_mode!r}")
    return {
        "packed": packed,
        "shift2": shift2,
        "channel_scale": channel_scale,
    }


def check_case(
    *,
    stage: str,
    num_experts: int,
    topk: int,
    tokens: int,
    pattern: str,
    seed: int,
    scale_mode: str = "varied",
    config_kwargs: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Run one candidate against the independent reference, bit for bit."""
    if stage == W13_STAGE:
        n, k, slots_are_source = 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE, False
    else:
        n, k, slots_are_source = HIDDEN_SIZE, INTERMEDIATE_SIZE, True
    config = GemmConfig(
        n=n,
        k=k,
        num_experts=num_experts,
        topk=topk,
        source_rows_are_slots=slots_are_source,
        **config_kwargs,
    )
    slots = tokens * topk
    routing_ids = candidate_routes(tokens, topk, num_experts, pattern, seed).to(device)
    weights = build_packed_weights(
        num_experts, stage, device, seed=seed, scale_mode=scale_mode
    )
    m_in = slots if slots_are_source else tokens
    generator = torch.Generator(device=device).manual_seed(seed)
    a_q = torch.empty((m_in, k), dtype=torch.int8, device=device).random_(
        -127, 128, generator=generator
    )
    a_scale = activation_scales(m_in, device, scale_mode)

    gemm = TileLangMoeGemm.create(config, tokens=tokens, slots=slots)
    routed_out = torch.full(
        (slots, n), float("nan"), dtype=torch.bfloat16, device=device
    )
    meta = gemm.run(
        a_q,
        a_scale,
        weights["packed"],
        weights["shift2"],
        weights["channel_scale"],
        routing_ids,
        routed_out,
    )
    reference = reference_stage(
        a_q,
        a_scale,
        weights["packed"],
        weights["shift2"],
        weights["channel_scale"],
        routing_ids,
        stage=stage,
    )
    mismatch = (routed_out != reference).nonzero()
    exact = int(mismatch.shape[0] == 0)
    result = {
        "stage": stage,
        "config": config.label,
        "experts": num_experts,
        "topk": topk,
        "tokens": tokens,
        "pattern": pattern,
        "seed": seed,
        "block_m": config.block_m,
        "block_n": config.block_n,
        "block_k": config.block_k,
        "num_stages": config.num_stages,
        "threads": config.threads,
        "bitwise_exact": bool(exact),
        "mismatches": int(mismatch.shape[0]),
        "elements": int(routed_out.numel()),
        "compile_seconds": gemm.compile_seconds,
        "max_m": gemm.max_m,
        "max_m_tiles": gemm.max_m_tiles,
    }
    if mismatch.shape[0]:
        first = mismatch[0].tolist()
        result["first_mismatch"] = first
        result["candidate"] = float(routed_out[first[0], first[1]])
        result["reference"] = float(reference[first[0], first[1]])
        result["nan_in_candidate"] = int(
            torch.isnan(routed_out.to(torch.float32)).sum()
        )
    del meta
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=[W13_STAGE, W2_STAGE, "both"], default="both"
    )
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--route-pattern", default="uniform")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--scale-mode", choices=["varied", "unit"], default="varied")
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument(
        "--decode",
        choices=["mxfp4", "lut", "luta", "prmt", "raw"],
        default="mxfp4",
    )
    parser.add_argument("--dump-dir", type=str, default="")
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA is required for this candidate driver", file=sys.stderr)
        return 2
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(0)
    if capability != (8, 0):
        print(f"expected an SM80 device, got {capability}", file=sys.stderr)
        return 2
    stages = [W13_STAGE, W2_STAGE] if args.stage == "both" else [args.stage]
    config_kwargs = {
        "block_m": args.block_m,
        "block_n": args.block_n,
        "block_k": args.block_k,
        "num_stages": args.num_stages,
        "threads": args.threads,
        "decode": args.decode,
    }
    results = []
    failed = 0
    for stage in stages:
        started = time.perf_counter()
        result = check_case(
            stage=stage,
            num_experts=args.experts,
            topk=args.topk,
            tokens=args.tokens,
            pattern=args.route_pattern,
            seed=args.seed,
            scale_mode=args.scale_mode,
            config_kwargs=config_kwargs,
            device=device,
        )
        result["wall_seconds"] = time.perf_counter() - started
        results.append(result)
        status = "PASS" if result["bitwise_exact"] else "FAIL"
        print(
            f"{status} {result['config']} E{result['experts']} k{result['topk']} "
            f"M{result['tokens']} {result['pattern']} seed{result['seed']} "
            f"exact={result['bitwise_exact']} mismatches={result['mismatches']} "
            f"compile={result['compile_seconds']:.1f}s"
        )
        if not result["bitwise_exact"]:
            failed += 1
            print("  first mismatch:", result.get("first_mismatch"))
    if args.dump_dir:
        out = Path(args.dump_dir)
        out.mkdir(parents=True, exist_ok=True)
        for stage in stages:
            config = GemmConfig(
                n=2 * INTERMEDIATE_SIZE if stage == W13_STAGE else HIDDEN_SIZE,
                k=HIDDEN_SIZE if stage == W13_STAGE else INTERMEDIATE_SIZE,
                num_experts=args.experts,
                topk=args.topk,
                source_rows_are_slots=stage == W2_STAGE,
                **config_kwargs,
            )
            gemm = TileLangMoeGemm.create(
                config, tokens=args.tokens, slots=args.topk * args.tokens
            )
            (out / f"{config.label}.cu").write_text(gemm.source)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
