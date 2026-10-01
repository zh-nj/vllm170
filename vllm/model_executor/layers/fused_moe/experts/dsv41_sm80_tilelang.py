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


def _build_decode_gemm(source_rows_are_slots: bool):
    n, k = (5120, 2304) if source_rows_are_slots else (4608, 5120)
    num_experts, topk, slots = 384, 6, 6
    m_in = slots if source_rows_are_slots else 1
    max_m, max_m_tiles = 96, 6
    block_m, block_n, block_k = 16, 64, 128
    threads = 128
    num_stages = 3 if source_rows_are_slots else 2
    k_blocks = k // 32
    shift_stride = (k_blocks + 3) // 4
    n_groups8 = n // 8
    kb_tile, ng_tile = block_k // 32, block_n // 8
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
                    for i, kk in T.Parallel(block_m, block_k):
                        a_s[i, kk] = T.if_then_else(
                            (row_s[i] >= 0) & (row_s[i] < m_in),
                            a_q[row_s[i], ko * block_k + kk],
                            T.int8(0),
                        )
                    T.copy(b_packed[expert, ko * kb_tile, ng0, 0], p_s)
                    T.copy(b_shift2[expert, (ko * kb_tile) // 4, ng0, 0], sh_s)
                    # Decode four packed bytes with the CUDA PRMT helper.
                    # The output bytes span two separate 16-column K regions.
                    for nn, kb_rel, t in T.Parallel(block_n, kb_tile, 4):
                        shift = T.Cast(
                            "int32",
                            (
                                sh_s[nn // 8, nn % 8]
                                >> (2 * ((ko * kb_tile + kb_rel) % 4))
                            )
                            & 3,
                        )
                        base = (nn % 8) * 16 + t * 4
                        word = (
                            T.Cast("uint32", p_s[kb_rel, nn // 8, base + 0])
                            | (T.Cast("uint32", p_s[kb_rel, nn // 8, base + 1]) << 8)
                            | (T.Cast("uint32", p_s[kb_rel, nn // 8, base + 2]) << 16)
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
    """Warm the adapters using a bounded temporary workspace at model load."""
    device = layer.w13_weight.device
    for weight, shift, scale, n, k in (
        (layer.w13_weight, layer.w13_mxfp4_int8_shift2, w13_scale, 4608, 5120),
        (layer.w2_weight, layer.w2_mxfp4_int8_shift2, w2_scale, 5120, 2304),
    ):
        for tensor, shape, dtype in (
            (weight, (384, k // 32, n // 8, 128), torch.uint8),
            (shift, (384, k // 128, n // 8, 8), torch.uint8),
            (scale, (384, n), torch.float32),
        ):
            if (
                tensor is None
                or tensor.shape != shape
                or tensor.dtype != dtype
                or tensor.device != device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "DSV41F TileLang requires the packed SM80 weight layout"
                )
    kernels = compile_decode_gemms(device.index)
    expert_ids = torch.arange(6, dtype=torch.int32, device=device)
    for kernel, weight, shift, scale, rows, k, n in (
        (
            kernels[0],
            layer.w13_weight,
            layer.w13_mxfp4_int8_shift2,
            w13_scale,
            1,
            5120,
            4608,
        ),
        (
            kernels[1],
            layer.w2_weight,
            layer.w2_mxfp4_int8_shift2,
            w2_scale,
            6,
            2304,
            5120,
        ),
    ):
        a = torch.zeros((rows, k), dtype=torch.int8, device=device)
        a_scale = torch.ones(rows, dtype=torch.float32, device=device)
        output = torch.empty((6, n), dtype=torch.bfloat16, device=device)
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
