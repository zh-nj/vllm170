# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frozen M=1, E=384, top-k6 PRMT grouped GEMMs for DSV41F on SM80.

Imported only after the model/backend/device opt-in gate. No runtime search or
benchmark dependencies; callers own all tensors and routing metadata.
"""

# T.Tensor annotations are evaluated by the TileLang parser.
# mypy: disable-error-code="valid-type"

from functools import cache

import torch

from vllm.model_executor.layers.fused_moe.experts.mxfp4_int8_sm80 import (
    _DECODE_BLOCK_M,
    _DECODE_NUM_EXPERTS,
    _HIDDEN_SIZE,
    _INTERMEDIATE_SIZE,
    _TARGET_TOPK,
)
from vllm.tilelang_utils import T, tilelang

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
}

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


# Packed MXFP4 geometry. One packed block covers `_PACKED_K_GROUP` K elements
# along the reduction axis, and one shift word carries a 2-bit decode shift per
# block for `_SHIFT_WORD_K_BLOCKS` consecutive blocks. The shift word is the
# scale-factor analogue for this format: it varies along K, so it must be
# applied per K tile and cannot be folded into the epilogue.
_PACKED_K_GROUP = 32
_SHIFT_WORD_K_BLOCKS = 4
_SHIFT_WORD_K = _PACKED_K_GROUP * _SHIFT_WORD_K_BLOCKS
_SHIFT_BITS = 2
# One scale group spans this many output channels; the packed row and the shift
# word are both segmented by it.
_N_GROUP = 8
_PACKED_ROW_BYTES = 128


def _k_blocks(k: int) -> int:
    """Number of packed blocks along a K axis of ``k`` elements."""
    return k // _PACKED_K_GROUP


def _shift_stride(k: int) -> int:
    """Shift-word rows needed for a K axis of ``k`` elements."""
    return (_k_blocks(k) + _SHIFT_WORD_K_BLOCKS - 1) // _SHIFT_WORD_K_BLOCKS


def _shift_word_of(k_block: int) -> int:
    """Index of the shift word holding ``k_block``'s decode shift."""
    return k_block // _SHIFT_WORD_K_BLOCKS


def _shift_slot_of(k_block: int) -> int:
    """2-bit slot of ``k_block`` inside its shift word."""
    return k_block % _SHIFT_WORD_K_BLOCKS


def _packed_shapes(n: int, k: int, num_experts: int):
    """Expected (packed, shift, channel-scale) shapes for one expert weight."""
    return (
        (num_experts, _k_blocks(k), n // _N_GROUP, _PACKED_ROW_BYTES),
        (num_experts, _shift_stride(k), n // _N_GROUP, _N_GROUP),
        (num_experts, n),
    )


def _build_decode_gemm(source_rows_are_slots: bool):
    num_experts, topk = _DECODE_NUM_EXPERTS, _TARGET_TOPK
    slots = _TARGET_TOPK
    n, k = (
        (_HIDDEN_SIZE, _INTERMEDIATE_SIZE)
        if source_rows_are_slots
        else (2 * _INTERMEDIATE_SIZE, _HIDDEN_SIZE)
    )
    m_in = slots if source_rows_are_slots else 1
    block_m = _DECODE_BLOCK_M
    max_m, max_m_tiles = slots * block_m, slots
    block_n, block_k = 64, 128
    threads = 128
    num_stages = 3 if source_rows_are_slots else 2
    k_blocks = _k_blocks(k)
    shift_stride = _shift_stride(k)
    n_groups8 = n // _N_GROUP
    kb_tile, ng_tile = block_k // _PACKED_K_GROUP, block_n // _N_GROUP
    src_div = 1 if source_rows_are_slots else topk
    k_iters = k // block_k

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
                b_frag = T.alloc_fragment((block_n, block_k), "int8")
                c_i = T.alloc_fragment((block_m, block_n), "int32")
                c_f = T.alloc_fragment((block_m, block_n), "float32")

                for i in T.Parallel(block_m):
                    row_s[i] = (
                        T.if_then_else(
                            m0 + i < num_post_pad[0], sorted_ids[m0 + i], slots
                        )
                        // src_div
                    )
                for i in T.Parallel(block_m):
                    # Mask padding explicitly, preserving zero-fill async loads.
                    asc_s[i] = T.if_then_else(
                        (row_s[i] >= 0) & (row_s[i] < m_in), a_scale[row_s[i]], 0.0
                    )

                T.clear(c_i)
                # Keep all three shared copies ahead of their consumers.
                for ko in T.Pipelined(
                    k_iters,
                    order=[0, 1, 2, 3, 4],
                    stage=[0, 0, 0, num_stages - 1, num_stages - 1],
                ):
                    k_block0 = ko * kb_tile
                    for i, kk in T.Parallel(block_m, block_k):
                        a_s[i, kk] = T.if_then_else(
                            (row_s[i] >= 0) & (row_s[i] < m_in),
                            a_q[row_s[i], ko * block_k + kk],
                            T.int8(0),
                        )
                    T.copy(b_packed[expert, k_block0, ng0, 0], p_s)
                    T.copy(b_shift2[expert, _shift_word_of(k_block0), ng0, 0], sh_s)
                    # Decode four packed bytes with the CUDA PRMT helper.
                    # The output bytes span two separate 16-column K regions.
                    for nn, kb_rel, t in T.Parallel(block_n, kb_tile, 4):
                        # Scale staging is [group][channel within group]; the
                        # K-dependent decode shift lives in the byte's 2-bit
                        # slot for this K block.
                        n_group, channel_in_group = nn // _N_GROUP, nn % _N_GROUP
                        shift = T.Cast(
                            "int32",
                            (
                                sh_s[n_group, channel_in_group]
                                >> (_SHIFT_BITS * _shift_slot_of(k_block0 + kb_rel))
                            )
                            & 3,
                        )
                        base = channel_in_group * 16 + t * 4
                        word = (
                            T.Cast("uint32", p_s[kb_rel, n_group, base + 0])
                            | (T.Cast("uint32", p_s[kb_rel, n_group, base + 1]) << 8)
                            | (T.Cast("uint32", p_s[kb_rel, n_group, base + 2]) << 16)
                            | (T.Cast("uint32", p_s[kb_rel, nn // 8, base + 3]) << 24)
                        )
                        decoded = T.call_extern(
                            "uint64", "dsv41_decode_x2", word, shift
                        )
                        # Keep byte extraction out of the vectorizer: 0.1.15
                        # otherwise broadcasts each packed input into 8 lanes.
                        for byte in T.unroll(8):
                            b_frag[
                                nn, kb_rel * 32 + (byte // 4) * 16 + t * 4 + byte % 4
                            ] = T.Cast("int8", (decoded >> (8 * byte)) & 0xFF)
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


@cache
def compile_decode_gemms(device_index: int):
    """Compile once per worker device during weight loading, before capture."""
    with torch.accelerator.device_index(device_index):
        return tuple(
            tilelang.compile(_build_decode_gemm(slots), pass_configs=PASS_CONFIGS)
            for slots in (False, True)
        )


def prepare_decode_gemms(layer, w13_scale, w2_scale, sorted_ids, num_post_pad):
    """Warm the adapters using a bounded temporary workspace at model load.

    The kernel hardcodes its geometry at compile time, so everything it reads
    is checked against the packed layout here; a shape or granularity mismatch
    must fail at load rather than silently address the wrong scale.
    """
    device = layer.w13_weight.device
    num_experts = _DECODE_NUM_EXPERTS
    for name, tensor, shape, dtype in (
        ("sorted_ids", sorted_ids, (_TARGET_TOPK * _DECODE_BLOCK_M,), torch.int32),
        ("num_post_pad", num_post_pad, (1,), torch.int32),
    ):
        if (
            tensor is None
            or tensor.shape != shape
            or tensor.dtype != dtype
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"DSV41F TileLang requires {name} as a contiguous "
                f"{tuple(shape)} {dtype} tensor on {device}"
            )
    for weight, shift, scale, n, k in (
        (layer.w13_weight, layer.w13_mxfp4_int8_shift2, w13_scale, 4608, 5120),
        (layer.w2_weight, layer.w2_mxfp4_int8_shift2, w2_scale, 5120, 2304),
    ):
        for name, tensor, shape, dtype in zip(
            ("packed weight", "shift", "channel scale"),
            (weight, shift, scale),
            _packed_shapes(n, k, num_experts),
            (torch.uint8, torch.uint8, torch.float32),
        ):
            if (
                tensor is None
                or tensor.shape != shape
                or tensor.dtype != dtype
                or tensor.device != device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"DSV41F TileLang requires the packed SM80 {name} for "
                    f"N={n} K={k}: expected {tuple(shape)} {dtype} on {device}"
                )
    kernels = compile_decode_gemms(device.index)
    slots = _TARGET_TOPK
    expert_ids = torch.arange(slots, dtype=torch.int32, device=device)
    for kernel, weight, shift, scale, rows, k, n in (
        (
            kernels[0],
            layer.w13_weight,
            layer.w13_mxfp4_int8_shift2,
            w13_scale,
            1,
            _HIDDEN_SIZE,
            2 * _INTERMEDIATE_SIZE,
        ),
        (
            kernels[1],
            layer.w2_weight,
            layer.w2_mxfp4_int8_shift2,
            w2_scale,
            slots,
            _INTERMEDIATE_SIZE,
            _HIDDEN_SIZE,
        ),
    ):
        a = torch.zeros((rows, k), dtype=torch.int8, device=device)
        a_scale = torch.ones(rows, dtype=torch.float32, device=device)
        output = torch.empty((slots, n), dtype=torch.bfloat16, device=device)
        kernel(
            a,
            a_scale,
            weight,
            shift,
            scale,
            sorted_ids,
            expert_ids,
            num_post_pad,
            output,
        )
    return kernels
