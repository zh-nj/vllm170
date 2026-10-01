# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import importlib
import math
import os
from collections.abc import Callable
from importlib.util import find_spec

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.profiler.phase_profile_hook import optional_phase_profile_stage
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import LayerNameType
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerPrefillChunkMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.attention.ops.fp8_sm80 import (
    _decode_fp8_lut,
    get_e4m3fn_bf16_lut,
    get_scaled_e4m3fn_bf16_lut,
)
from vllm.v1.worker.workspace import current_workspace_manager

if current_platform.is_rocm():
    from vllm.platforms.rocm import _ON_GFX942, _ON_GFX950
else:
    _ON_GFX942 = False
    _ON_GFX950 = False


@functools.cache
def _get_aiter_topk_ops() -> tuple[Callable[..., None], Callable[..., None]] | None:
    try:
        from aiter.ops.topk import (
            top_k_per_row_decode,
            top_k_per_row_prefill,
        )
    except ImportError:
        return None
    return top_k_per_row_prefill, top_k_per_row_decode


_GFX950_C4A_AITER_MAX_COMPRESSED_SEQ_LEN = 64 * 1024
_GFX950_C4A_NATIVE_MAX_ROWS = 256


def _get_aiter_top_k_kernel(
    *,
    is_prefill: bool,
    compress_ratio: int,
    num_rows: int,
    max_valid_seq_len: int | None = None,
    on_gfx950: bool = _ON_GFX950,
) -> Callable[..., None] | None:
    if compress_ratio <= 1 or not on_gfx950:
        return None

    if not is_prefill:
        assert max_valid_seq_len is not None
        # AITER v0.1.19 decode is one-block only. This measured gfx950
        # FP32/k=1024 compressed-row boundary is independent of the native
        # split-count boundary in sampler.cu.
        if (
            num_rows <= _GFX950_C4A_NATIVE_MAX_ROWS
            and max_valid_seq_len > _GFX950_C4A_AITER_MAX_COMPRESSED_SEQ_LEN
        ):
            return None

    topk_ops = _get_aiter_topk_ops()
    if topk_ops is None:
        return None
    return topk_ops[0] if is_prefill else topk_ops[1]


@triton.jit
def _localize_aiter_prefill_topk_kernel(
    indices_ptr,
    row_starts_ptr,
    indices_stride,
    num_topk,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_topk
    indices = tl.load(indices_ptr + row_idx * indices_stride + offsets, mask=mask)
    row_start = tl.load(row_starts_ptr + row_idx)
    localized = tl.where(indices < 0, indices, indices - row_start)
    tl.store(
        indices_ptr + row_idx * indices_stride + offsets,
        localized,
        mask=mask,
    )


def _localize_aiter_prefill_topk(
    indices: torch.Tensor,
    row_starts: torch.Tensor,
) -> None:
    num_rows, num_topk = indices.shape
    block_size = 256
    _localize_aiter_prefill_topk_kernel[(num_rows, triton.cdiv(num_topk, block_size))](
        indices,
        row_starts,
        indices.stride(0),
        num_topk,
        BLOCK_SIZE=block_size,
    )


def _launch_aiter_top_k_per_row_prefill(
    top_k_per_row_prefill: Callable[..., None],
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    topk_tokens: int,
) -> None:
    top_k_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        indices,
        None,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        k=topk_tokens,
    )
    _localize_aiter_prefill_topk(indices, row_starts)


def _launch_aiter_top_k_per_row_decode(
    top_k_per_row_decode: Callable[..., None],
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    topk_tokens: int,
) -> None:
    top_k_per_row_decode(
        logits,
        1,
        seq_lens.reshape(-1),
        indices,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        k=topk_tokens,
    )


@triton.jit
def _indexer_k_quant_and_cache_kernel(
    k_ptr,  # [num_tokens, head_dim]
    kv_cache_ptr,  # [n_blks, blk_size//tile_block, head_dim // 16B, tile_block, 16B]
    # [n_blocks, blk_size, head_dim]
    kv_cache_scale_ptr,  # [n_blks, blk_size]
    slot_mapping_ptr,  # [num_tokens]
    kv_cache_scale_stride,
    kv_cache_value_stride,
    block_size,
    num_tokens,
    head_dim: tl.constexpr,
    LAYOUT: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
    IS_FNUZ: tl.constexpr,
    USE_UE8M0: tl.constexpr,
):
    tid = tl.program_id(0)
    offset = tl.arange(0, head_dim)
    if LAYOUT == "SHUFFLE":
        tile_offset = (
            offset // HEAD_TILE_SIZE * BLOCK_TILE_SIZE * HEAD_TILE_SIZE
            + offset % HEAD_TILE_SIZE
        )
    else:
        tile_offset = offset
    tile_store_offset = tile_offset
    # for idx in tl.range(tid, num_tokens, n_program):
    src_ptr = k_ptr + tid * head_dim
    slot_id = tl.load(slot_mapping_ptr + tid)
    if slot_id < 0:
        return
    # The packed KV layout makes per-block strides large
    # enough that block_id * stride can exceed 32-bit range.
    block_id = (slot_id // block_size).to(tl.int64)
    block_offset = slot_id % block_size
    tile_block_id = block_offset // BLOCK_TILE_SIZE
    tile_block_offset = block_offset % BLOCK_TILE_SIZE
    val = tl.load(src_ptr + offset)
    amax = tl.max(val.abs(), axis=-1).to(tl.float32)
    if IS_FNUZ:
        scale = tl.maximum(1e-4, amax) / 224.0
    else:
        scale = tl.maximum(1e-4, amax) / 448.0

    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    fp8_val = (val.to(tl.float32) / scale).to(kv_cache_ptr.type.element_ty)
    if LAYOUT == "SHUFFLE":
        dst_ptr = (
            kv_cache_ptr
            + block_id * kv_cache_value_stride
            + tile_block_id * BLOCK_TILE_SIZE * head_dim
            + tile_block_offset * HEAD_TILE_SIZE
        )
    else:
        dst_ptr = (
            kv_cache_ptr + block_id * kv_cache_value_stride + block_offset * head_dim
        )
    tl.store(dst_ptr + tile_store_offset, fp8_val)
    dst_scale_ptr = kv_cache_scale_ptr + block_id * kv_cache_scale_stride + block_offset
    tl.store(dst_scale_ptr, scale)


def indexer_k_quant_and_cache_triton(
    k: torch.Tensor,
    kv_cache: torch.Tensor,  # [num_blocks, block_size, head_dim + 4]
    slot_mapping: torch.Tensor,
    quant_block_size,
    scale_fmt,
    block_tile_size=16,
    head_tile_size=16,
):
    num_blocks = kv_cache.shape[0]
    head_dim = k.shape[-1]
    num_tokens = slot_mapping.shape[0]
    block_size = kv_cache.shape[1]
    # In real layout, we store the first portion as kv cache value
    # and second portion as kv cache scale
    kv_cache = kv_cache.view(num_blocks, -1)
    fp8_dtype = current_platform.fp8_dtype()
    kv_cache_value = kv_cache[:, : block_size * head_dim].view(fp8_dtype)
    kv_cache_scale = kv_cache[:, block_size * head_dim :].view(torch.float32)
    head_tile_size = head_tile_size // kv_cache.element_size()
    layout = "NORMAL" if block_size == 1 else "SHUFFLE"
    grid = (num_tokens,)
    _indexer_k_quant_and_cache_kernel[grid](
        k,
        kv_cache_value,
        kv_cache_scale,
        slot_mapping,
        kv_cache_scale.stride(0),
        kv_cache_value.stride(0),
        block_size,
        num_tokens,
        head_dim,
        layout,
        block_tile_size,
        head_tile_size,
        IS_FNUZ=current_platform.fp8_dtype() == torch.float8_e4m3fnuz,
        USE_UE8M0=scale_fmt == "ue8m0",
    )


@triton.jit
def _cp_gather_indexer_quant_cache_kernel(
    kv_cache_ptr,  # [n_blks,blk_size//tile_blk,head_dim//16B,tile_blk,16B]
    # [n_blks, blk_size, head_dim]
    kv_cache_scale_ptr,  # [n_blks, blk_size]
    k_fp8_ptr,  # [num_tokens, head_dim]
    k_scale_ptr,  # [num_tokens]
    block_table_ptr,  # [batch_size, block_table_stride]
    cu_seqlen_ptr,  # [batch_size + 1]
    token_to_seq_ptr,  # [num_tokens]
    block_size,
    block_table_stride,
    kv_cache_stride,
    kv_cache_scale_stride,
    LAYOUT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    NUM_BATCHES: tl.constexpr,
    BLOCK_TABLE_WIDTH: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    tid = tl.program_id(0)
    offset = tl.arange(0, HEAD_DIM)
    valid_tid = tid < NUM_TOKENS
    batch_id = tl.load(token_to_seq_ptr + tid, mask=valid_tid, other=-1)
    valid_batch = (batch_id >= 0) & (batch_id < NUM_BATCHES)
    safe_batch_id = tl.where(valid_batch, batch_id, 0)
    batch_start = tl.load(cu_seqlen_ptr + safe_batch_id, mask=valid_batch, other=0)
    batch_end = tl.load(cu_seqlen_ptr + safe_batch_id + 1, mask=valid_batch, other=0)
    batch_offset = tid - batch_start
    valid_token = valid_tid & valid_batch & (tid >= batch_start) & (tid < batch_end)
    if not valid_token:
        return
    block_table_id = batch_offset // block_size
    block_offset = batch_offset % block_size
    valid_block_table = (
        valid_token
        & (block_table_id >= 0)
        & (block_table_id < BLOCK_TABLE_WIDTH)
        & (block_offset >= 0)
        & (block_offset < block_size)
    )
    safe_block_table_id = tl.where(valid_block_table, block_table_id, 0)
    block_table_offset = safe_batch_id * block_table_stride + safe_block_table_id
    block_id = tl.load(
        block_table_ptr + block_table_offset, mask=valid_block_table, other=-1
    )
    valid_block = valid_block_table & (block_id >= 0) & (block_id < NUM_BLOCKS)
    # The packed KV layout makes per-block strides large
    # enough that block_id * stride can exceed 32-bit range.
    safe_block_id = tl.where(valid_block, block_id, 0).to(tl.int64)
    safe_block_offset = tl.where(valid_block, block_offset, 0)
    tiled_block_offset = safe_block_offset % BLOCK_TILE_SIZE
    if LAYOUT == "SHUFFLE":
        src_cache_offset = (
            safe_block_id * kv_cache_stride
            + (safe_block_offset // BLOCK_TILE_SIZE) * HEAD_DIM * BLOCK_TILE_SIZE
            + tiled_block_offset * HEAD_TILE_SIZE
        )
    else:
        src_cache_offset = (
            safe_block_id * kv_cache_stride + safe_block_offset * HEAD_DIM
        )
    src_scale_offset = safe_block_id * kv_cache_scale_stride + safe_block_offset
    dst_offset = tid * HEAD_DIM
    src_scale_ptr = kv_cache_scale_ptr + src_scale_offset
    src_cache_ptr = kv_cache_ptr + src_cache_offset
    dst_k_ptr = k_fp8_ptr + dst_offset
    scale_val = tl.load(src_scale_ptr, mask=valid_block, other=0.0)
    tl.store(k_scale_ptr + tid, scale_val)
    if LAYOUT == "SHUFFLE":
        tiled_src_offset = (
            offset // HEAD_TILE_SIZE * HEAD_TILE_SIZE * BLOCK_TILE_SIZE
            + offset % HEAD_TILE_SIZE
        )
    else:
        tiled_src_offset = offset
    val = tl.load(src_cache_ptr + tiled_src_offset)
    tl.store(dst_k_ptr + offset, val, mask=valid_block)


@triton.jit(do_not_specialize=["num_batches"])
def _cp_gather_indexer_quant_cache_gfx950_kernel(
    kv_cache_ptr,  # [n_blks,blk_size//tile_blk,head_dim//16B,tile_blk,16B]
    # [n_blks, blk_size, head_dim]
    kv_cache_scale_ptr,  # [n_blks, blk_size]
    k_fp8_ptr,  # [num_tokens, head_dim]
    k_scale_ptr,  # [num_tokens]
    block_table_ptr,  # [batch_size, block_table_stride]
    cu_seqlen_ptr,  # [batch_size + 1]
    token_to_seq_ptr,  # [num_tokens]
    block_size,
    block_table_stride,
    kv_cache_stride,
    kv_cache_scale_stride,
    LAYOUT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
    num_batches,
    BLOCK_TABLE_WIDTH: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    tid = tl.program_id(0)
    offset = tl.arange(0, HEAD_DIM)
    batch_id = tl.load(token_to_seq_ptr + tid)
    valid_batch = (batch_id >= 0) & (batch_id < num_batches)
    safe_batch_id = tl.where(valid_batch, batch_id, 0)
    batch_start = tl.load(cu_seqlen_ptr + safe_batch_id, mask=valid_batch, other=0)
    batch_end = tl.load(cu_seqlen_ptr + safe_batch_id + 1, mask=valid_batch, other=0)
    batch_offset = tid - batch_start
    valid_token = valid_batch & (tid >= batch_start) & (tid < batch_end)
    if not valid_token:
        return
    block_table_id = batch_offset // block_size
    block_offset = batch_offset % block_size
    valid_block_table = (
        valid_token
        & (block_table_id >= 0)
        & (block_table_id < BLOCK_TABLE_WIDTH)
        & (block_offset >= 0)
        & (block_offset < block_size)
    )
    safe_block_table_id = tl.where(valid_block_table, block_table_id, 0)
    block_table_offset = safe_batch_id * block_table_stride + safe_block_table_id
    block_id = tl.load(
        block_table_ptr + block_table_offset, mask=valid_block_table, other=-1
    )
    valid_block = valid_block_table & (block_id >= 0) & (block_id < NUM_BLOCKS)
    # The packed KV layout makes per-block strides large
    # enough that block_id * stride can exceed 32-bit range.
    safe_block_id = tl.where(valid_block, block_id, 0).to(tl.int64)
    safe_block_offset = tl.where(valid_block, block_offset, 0)
    tiled_block_offset = safe_block_offset % BLOCK_TILE_SIZE
    if LAYOUT == "SHUFFLE":
        src_cache_offset = (
            safe_block_id * kv_cache_stride
            + (safe_block_offset // BLOCK_TILE_SIZE) * HEAD_DIM * BLOCK_TILE_SIZE
            + tiled_block_offset * HEAD_TILE_SIZE
        )
    else:
        src_cache_offset = (
            safe_block_id * kv_cache_stride + safe_block_offset * HEAD_DIM
        )
    src_scale_offset = safe_block_id * kv_cache_scale_stride + safe_block_offset
    dst_offset = tid * HEAD_DIM
    src_scale_ptr = kv_cache_scale_ptr + src_scale_offset
    src_cache_ptr = kv_cache_ptr + src_cache_offset
    dst_k_ptr = k_fp8_ptr + dst_offset
    scale_val = tl.load(src_scale_ptr, mask=valid_block, other=0.0)
    tl.store(k_scale_ptr + tid, scale_val)
    if LAYOUT == "SHUFFLE":
        tiled_src_offset = (
            offset // HEAD_TILE_SIZE * HEAD_TILE_SIZE * BLOCK_TILE_SIZE
            + offset % HEAD_TILE_SIZE
        )
    else:
        tiled_src_offset = offset
    val = tl.load(src_cache_ptr + tiled_src_offset)
    tl.store(dst_k_ptr + offset, val, mask=valid_block)


def cp_gather_indexer_k_quant_cache_triton(
    k_cache: torch.Tensor,  # [num_blocks, block_size, head_dim + 4]
    k_fp8: torch.Tensor,
    k_fp8_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlen: torch.Tensor,
    token_to_seq: torch.Tensor,
    block_tile_size: int = 16,
    head_tile_size: int = 16,
):
    num_tokens = k_fp8.size(0)
    block_size = k_cache.size(1)
    block_table_stride = block_table.stride(0)
    head_dim = k_fp8.shape[-1]
    num_blocks = k_cache.shape[0]
    # we assume the kv cache already been split to 2 portion
    k_cache = k_cache.view(num_blocks, -1)
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_value = k_cache[:, : block_size * head_dim].view(fp8_dtype)
    k_cache_scale = k_cache[:, block_size * head_dim :].view(torch.float32)
    grid = (num_tokens,)
    k_fp8_scale = k_fp8_scale.view(torch.float32)
    layout = "NORMAL" if block_size == 1 else "SHUFFLE"
    kernel_args = (
        k_cache_value,
        k_cache_scale,
        k_fp8,
        k_fp8_scale,
        block_table,
        cu_seqlen,
        token_to_seq,
        block_size,
        block_table_stride,
        k_cache_value.stride(0),
        k_cache_scale.stride(0),
        layout,
        head_dim,
        block_tile_size,
        head_tile_size,
    )
    if _ON_GFX950:
        _cp_gather_indexer_quant_cache_gfx950_kernel[grid](
            *kernel_args,
            cu_seqlen.shape[0] - 1,
            block_table.shape[1],
            num_blocks,
        )
    else:
        _cp_gather_indexer_quant_cache_kernel[grid](
            *kernel_args,
            num_tokens,
            cu_seqlen.shape[0] - 1,
            block_table.shape[1],
            num_blocks,
        )


# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L156
def fp8_paged_mqa_logits_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):
    from vllm.utils.math_utils import cdiv

    fp8_dtype = current_platform.fp8_dtype()
    batch_size, next_n, _, dim = q.size()
    if next_n == 1:
        block_size = kv_cache.shape[1]
        logits = torch.full(
            [batch_size, max_model_len],
            float("-inf"),
            device=q.device,
            dtype=torch.float32,
        )
        if context_lens.dim() > 1:
            context_lens = context_lens.squeeze(-1)
        kv_cache_flat = kv_cache.view(-1, block_size * (dim + 4))
        for i in range(batch_size):
            q_i = q[i, 0].to(torch.float32)
            q_scale = weights[i]
            seq_len = int(context_lens[i].item())
            assert seq_len <= max_model_len
            num_pages = cdiv(seq_len, block_size)
            padded_seq_len = num_pages * block_size
            pages = block_tables[i, :num_pages]
            cache = kv_cache_flat[pages]
            scale_offset = block_size * dim
            cache_value = (
                cache[..., :scale_offset].view(dtype=fp8_dtype).to(torch.float32)
            )
            cache_scale = (
                cache[..., scale_offset:].view(dtype=torch.float32).contiguous()
            )
            cache_value = cache_value.view(padded_seq_len, dim)
            cache_scale = cache_scale.view(padded_seq_len)
            score = F.linear(cache_value, q_i)
            score = F.relu(score)
            score *= q_scale[None, :]
            score = score.sum(dim=1)
            score *= cache_scale
            logits[i, :seq_len] = score[:seq_len]
        return logits

    kv_cache, scale = kv_cache[..., :dim], kv_cache[..., dim:]
    scale = scale.contiguous().view(torch.float)
    q = q.float()
    kv_cache = kv_cache.view(fp8_dtype).float() * scale
    num_block, block_size, _, dim = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    for i in range(batch_size):
        context_len = context_lens[i]
        if context_len.ndim == 0:
            context_len_i = int(context_len.item())
            q_offsets = torch.arange(
                context_len_i - next_n, context_len_i, device=q.device
            )
            context_limit = torch.full(
                (next_n,), context_len_i, dtype=torch.int32, device=q.device
            )
        else:
            context_limit = context_len.to(device=q.device, dtype=torch.int32)
            q_offsets = context_limit - 1
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        max_context_len = int(context_limit.max().item())
        for block_rk in range(cdiv(max_context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size, (block_rk + 1) * block_size, device=q.device
            )
            mask = (k_offsets[None, :] < context_limit[:, None]) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits


@functools.lru_cache
def paged_mqa_logits_module():
    paged_mqa_logits_module_path = None
    if find_spec("aiter.ops.triton.pa_mqa_logits") is not None:
        paged_mqa_logits_module_path = "aiter.ops.triton.pa_mqa_logits"
    elif find_spec("aiter.ops.triton.attention.pa_mqa_logits") is not None:
        paged_mqa_logits_module_path = "aiter.ops.triton.attention.pa_mqa_logits"

    if paged_mqa_logits_module_path is not None:
        try:
            module = importlib.import_module(paged_mqa_logits_module_path)
            return module
        except ImportError:
            return None
    return None


def rocm_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Compute FP8 MQA logits using paged KV-cache.

    Args:
        q_fp8: Query tensor of shape [B, next_n, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv_cache_fp8: Paged KV-cache in packed FP8+scale layout with shape
            [num_blocks, block_size, 1, D+4], dtype `torch.uint8`. The last
            4 bytes per (block,pos) store the `float` dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """
    from vllm._aiter_ops import rocm_aiter_ops

    aiter_paged_mqa_logits_module = None
    # if rocm_aiter_ops.is_enabled():
    batch_size, next_n = q_fp8.shape[:2]
    block_size = kv_cache_fp8.shape[1]

    if rocm_aiter_ops.is_enabled() or rocm_aiter_ops.is_rdna_aiter_enabled():
        aiter_paged_mqa_logits_module = paged_mqa_logits_module()

    if aiter_paged_mqa_logits_module is not None:
        if _ON_GFX942 or _ON_GFX950:
            deepgemm_fp8_paged_mqa_logits = (
                aiter_paged_mqa_logits_module.deepgemm_fp8_paged_mqa_logits
            )
            batch_size, next_n, heads, _ = q_fp8.shape
            (out_logits,) = current_workspace_manager().get_simultaneous(
                ((batch_size * next_n, max_model_len), torch.float32),
            )
            deepgemm_fp8_paged_mqa_logits(
                q_fp8,
                kv_cache_fp8,
                weights,
                out_logits,
                context_lens,
                block_tables,
                max_model_len,
                ChunkK=256,
                Preshuffle=block_size > 1,
                KVBlockSize=block_size,
                WavePerEU=2,
            )
            out_logits.nan_to_num_(float("-inf"))
            return out_logits
        deepgemm_fp8_paged_mqa_logits_stage1 = (
            aiter_paged_mqa_logits_module.deepgemm_fp8_paged_mqa_logits_stage1
        )
        batch_size, next_n, heads, _ = q_fp8.shape
        (out_qk,) = current_workspace_manager().get_simultaneous(
            ((heads, batch_size * next_n, max_model_len), torch.float32),
        )
        out_qk.fill_(float("-inf"))
        deepgemm_fp8_paged_mqa_logits_stage1(
            q_fp8,
            kv_cache_fp8,
            weights,
            out_qk,
            context_lens,
            block_tables,
            max_model_len,
            ChunkQ=heads,
        )
        return out_qk.sum(dim=0)
    else:
        return fp8_paged_mqa_logits_torch(
            q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len
        )


# Take from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L84
def fp8_mqa_logits_torch(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    k_fp8, scale = kv
    seq_len_kv = k_fp8.shape[0]
    k = k_fp8.to(torch.bfloat16)
    q = q.to(torch.bfloat16)
    device = q.device

    mask_lo = (
        torch.arange(0, seq_len_kv, device=device)[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device=device)[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi

    # ``score`` is [H, M, N]; ``scale`` is the per-KV-token scale, which
    # vLLM callers hand us as ``[N, 1]`` (a ``[N, 4]`` uint8 buffer cast
    # to fp32). PyTorch right-aligns dimensions for broadcasting, so a
    # naked ``score * scale`` would align ``scale``'s leading dim with
    # ``score``'s M dim and raise a shape mismatch. Flatten to ``[N]`` so
    # broadcasting lines up with the last dim of ``score``.
    score = torch.einsum("mhd,nd->hmn", q, k).float() * scale.reshape(-1)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    return logits


@functools.lru_cache
def mqa_logits_module():
    mqa_logits_module_path = None
    if find_spec("aiter.ops.triton.fp8_mqa_logits") is not None:
        mqa_logits_module_path = "aiter.ops.triton.fp8_mqa_logits"
    elif find_spec("aiter.ops.triton.attention.fp8_mqa_logits") is not None:
        mqa_logits_module_path = "aiter.ops.triton.attention.fp8_mqa_logits"

    if mqa_logits_module_path is not None:
        try:
            module = importlib.import_module(mqa_logits_module_path)
            return module
        except ImportError:
            return None
    return None


def rocm_fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """

    from vllm._aiter_ops import rocm_aiter_ops

    k_fp8, scale = kv

    if _ON_GFX942 and rocm_aiter_ops.is_enabled():
        from aiter.ops.flydsl import flydsl_fp8_mqa_logits

        return flydsl_fp8_mqa_logits(
            q, k_fp8, scale, weights, cu_seqlen_ks, cu_seqlen_ke
        )

    aiter_mqa_logits_module = None
    if rocm_aiter_ops.is_enabled() or rocm_aiter_ops.is_rdna_aiter_enabled():
        aiter_mqa_logits_module = mqa_logits_module()

    if aiter_mqa_logits_module is not None:
        fp8_mqa_logits = aiter_mqa_logits_module.fp8_mqa_logits
        return fp8_mqa_logits(q, k_fp8, scale, weights, cu_seqlen_ks, cu_seqlen_ke)
    else:
        return fp8_mqa_logits_torch(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)


def _max_decode_logits_rows(num_batched_tokens: int) -> int:
    """Upper bound on decode rows the paged-MQA logits buffer can ever hold.

    ``rocm_fp8_paged_mqa_logits`` sizes its workspace as
    ``(batch_size * next_n, max_model_len)``. ``batch_size`` is bounded by
    ``max_num_seqs`` and ``next_n`` by ``1 + num_speculative_tokens``, which is
    far tighter than ``max_num_batched_tokens`` -- 192 vs 16384 for a typical
    32-seq DSpark-5 deployment. The loose bound is harmless at short contexts
    but scales with ``max_model_len``, so at the model's full context it asks
    for tens of TiB and the engine cannot start. Take whichever valid bound is
    smaller; the workspace is locked after profiling, so it must not be under-
    estimated.
    """
    try:
        vllm_config = get_current_vllm_config()
    except Exception:
        return num_batched_tokens
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    max_num_seqs = getattr(scheduler_config, "max_num_seqs", None)
    if not max_num_seqs:
        return num_batched_tokens
    speculative_config = getattr(vllm_config, "speculative_config", None)
    num_spec = getattr(speculative_config, "num_speculative_tokens", 0) or 0
    return min(num_batched_tokens, max_num_seqs * (1 + num_spec))


def rocm_aiter_sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor | None,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool = False,
    compress_ratio: int = 1,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 0,
    candidate_write: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer


def _gather_indexer_chunk_kv_cache(
    chunk: DeepseekV32IndexerPrefillChunkMetadata,
    kv_cache: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
) -> None:
    if chunk.skip_kv_gather:
        return
    cp_gather_indexer_k_quant_cache_triton(
        kv_cache,
        k_fp8,
        k_scale,
        chunk.block_table,
        chunk.cu_seq_lens,
        token_to_seq=chunk.token_to_seq,
    )


@eager_break_during_capture
def rocm_aiter_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor | None,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool = False,
    compress_ratio: int = 1,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 0,
    candidate_write: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    from vllm.utils.torch_utils import _resolve_layer_name

    k_cache_prefix = _resolve_layer_name(k_cache_prefix)
    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Profiling early-exit: reserve memory to account for runtime
        # allocations. Must be in the real impl, not the fake impl —
        # torch.compile calls the fake impl under FakeTensor mode where
        # workspace manager operations on the locked real workspace
        # would corrupt PyTorch's dispatch state.
        workspace_manager = current_workspace_manager()

        # Prefill k_fp8 and k_scale buffers, used by
        # rocm_aiter_sparse_attn_indexer's prefill path
        workspace_manager.get_simultaneous(
            ((total_seq_lens, head_dim), fp8_dtype),
            ((total_seq_lens, 4), torch.uint8),
        )

        # Decode logits buffer, used by rocm_fp8_paged_mqa_logits.
        decode_rows = _max_decode_logits_rows(hidden_states.shape[0])
        if _ON_GFX942 or _ON_GFX950:
            workspace_manager.get_simultaneous(
                ((decode_rows, max_model_len), torch.float32),
            )
        else:
            workspace_manager.get_simultaneous(
                (
                    (q_fp8.shape[1], decode_rows, max_model_len),
                    torch.float32,
                ),
            )
        # Transient logits tensor peak memory, produced by
        # rocm_fp8_mqa_logits (prefill) and rocm_fp8_paged_mqa_logits
        # (decode). Prefill logits are bounded by
        # VLLM_SPARSE_INDEXER_MAX_LOGITS_MB via chunking in
        # split_indexer_prefill_chunks; decode logits are smaller.
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return rocm_aiter_sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_fp8,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            compress_ratio,
            candidate_blocks,
            candidate_block_size,
            candidate_write,
        )
    layer_attn_metadata = attn_metadata[k_cache_prefix]
    assert isinstance(layer_attn_metadata, DeepseekV32IndexerMetadata)
    assert topk_indices_buffer is not None
    assert scale_fmt is not None
    slot_mapping = layer_attn_metadata.slot_mapping
    has_decode = layer_attn_metadata.num_decodes > 0
    has_prefill = layer_attn_metadata.num_prefills > 0
    num_decode_tokens = layer_attn_metadata.num_decode_tokens
    topk_indices_buffer[: hidden_states.shape[0]] = -1

    # during speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]
    elif not skip_k_cache_insert:
        raise ValueError("k must be provided when skip_k_cache_insert is False")

    if not skip_k_cache_insert:
        indexer_k_quant_and_cache_triton(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )

    if has_prefill:
        prefill_metadata = layer_attn_metadata.prefill
        assert prefill_metadata is not None

        workspace_manager = current_workspace_manager()
        k_fp8_full, k_scale_full = workspace_manager.get_simultaneous(
            ((total_seq_lens, head_dim), fp8_dtype),
            ((total_seq_lens, 4), torch.uint8),
        )
        for chunk in prefill_metadata.chunks:
            k_fp8 = k_fp8_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]
            _gather_indexer_chunk_kv_cache(chunk, kv_cache, k_fp8, k_scale)
            logits = rocm_fp8_mqa_logits(
                q_fp8[chunk.token_start : chunk.token_end],
                (k_fp8, k_scale.view(torch.float32)),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
            )
            if candidate_blocks is not None:
                from vllm.model_executor.layers.sparse_attn_indexer import (
                    _apply_candidate_mask,
                    _select_candidate_blocks,
                )

                chunk_candidates = candidate_blocks[chunk.token_start : chunk.token_end]
                if candidate_write:
                    _select_candidate_blocks(
                        logits,
                        chunk.cu_seqlen_ks,
                        chunk.cu_seqlen_ke,
                        chunk_candidates.shape[1],
                        candidate_block_size,
                        chunk_candidates,
                    )
                else:
                    _apply_candidate_mask(
                        logits,
                        chunk.cu_seqlen_ks,
                        chunk.cu_seqlen_ke,
                        chunk_candidates,
                        candidate_block_size,
                    )
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            num_rows = logits.shape[0]

            aiter_topk_kernel = _get_aiter_top_k_kernel(
                is_prefill=True,
                compress_ratio=compress_ratio,
                num_rows=num_rows,
            )
            if aiter_topk_kernel is not None:
                _launch_aiter_top_k_per_row_prefill(
                    aiter_topk_kernel,
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_indices,
                    topk_tokens,
                )
            else:
                torch.ops._C.top_k_per_row_prefill(
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )

    if has_decode:
        decode_metadata = layer_attn_metadata.decode
        assert decode_metadata is not None
        # kv_cache size requirement [num_block, block_size, n_head, head_dim],
        # we only have [num_block, block_size, head_dim],
        kv_cache = kv_cache.unsqueeze(-2)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens)
            padded_q_fp8_decode_tokens = pack_seq_triton(
                q_fp8[:num_decode_tokens], decode_lens
            )
        else:
            padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_fp8.shape[1:]
            )
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_fp8_decode_tokens.shape[0]
        next_n = padded_q_fp8_decode_tokens.shape[1]
        assert batch_size == decode_metadata.seq_lens.shape[0]
        num_padded_tokens = batch_size * next_n

        logits = rocm_fp8_paged_mqa_logits(
            padded_q_fp8_decode_tokens,
            kv_cache,
            weights[:num_padded_tokens],
            decode_metadata.seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=max_model_len,
        )

        if candidate_blocks is not None:
            from vllm.model_executor.layers.sparse_attn_indexer import (
                _apply_candidate_mask,
                _select_candidate_blocks,
            )

            num_rows = logits.shape[0]
            visible = decode_metadata.seq_lens.reshape(-1)
            if visible.numel() != num_rows:
                visible = visible.repeat_interleave(next_n)
            visible = visible[:num_rows].to(torch.int64)
            row_starts = torch.zeros_like(visible)
            decode_candidates = candidate_blocks[:num_rows]
            if candidate_write:
                _select_candidate_blocks(
                    logits,
                    row_starts,
                    visible,
                    decode_candidates.shape[1],
                    candidate_block_size,
                    decode_candidates,
                )
            else:
                _apply_candidate_mask(
                    logits,
                    row_starts,
                    visible,
                    decode_candidates,
                    candidate_block_size,
                )

        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
        num_rows = logits.shape[0]

        # FULL graphs are not keyed by context length, so use a replay-safe
        # upper bound when choosing the captured kernel.
        if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL:
            max_compressed_seq_len = max_model_len
        else:
            max_compressed_seq_len = layer_attn_metadata.max_seq_len // compress_ratio
        aiter_topk_kernel = _get_aiter_top_k_kernel(
            is_prefill=False,
            compress_ratio=compress_ratio,
            num_rows=num_rows,
            max_valid_seq_len=max_compressed_seq_len,
        )
        if aiter_topk_kernel is not None:
            _launch_aiter_top_k_per_row_decode(
                aiter_topk_kernel,
                logits,
                decode_metadata.seq_lens,
                topk_indices,
                topk_tokens,
            )
        else:
            torch.ops._C.top_k_per_row_decode(
                logits,
                next_n,
                decode_metadata.seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, next_n, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def _decode_e8m0_scales(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.float8_e8m0fnu:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        return _upcast_e8m0_to_fp32(scale).contiguous()
    if scale.dtype == torch.uint8:
        # MXFP8 parameters preserve E8M0 scales as their raw exponent bytes.
        # They are biased exponents, not numeric uint8 scale values.
        return torch.exp2(scale.to(torch.int16).to(torch.float32) - 127.0)
    return scale.to(torch.float32)


def _expand_2d_block_scales(
    scale: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    scale = _decode_e8m0_scales(scale)
    row_blocks, col_blocks = scale.shape[-2:]
    row_block = math.ceil(rows / row_blocks)
    col_block = math.ceil(cols / col_blocks)
    scale = torch.repeat_interleave(scale, row_block, dim=-2)[..., :rows, :]
    scale = torch.repeat_interleave(scale, col_block, dim=-1)[..., :, :cols]
    return scale


@triton.jit
def _inverse_rope_gptj_kernel(
    o_ptr,  # [T, H, D] input
    out_ptr,  # [T, H, D] bf16 output
    pos_ptr,  # [T] positions
    cos_sin_ptr,  # [P, rope_dim] fp32 (cos[:half] | sin[half:])
    s_t,
    s_h,  # input row strides (last dim contiguous)
    os_t,
    os_h,  # output row strides
    cs_stride,  # cos_sin_cache row stride
    NOPE: tl.constexpr,  # non-rope head dims (passed through)
    HALF: tl.constexpr,  # rope_dim // 2
    BLOCK_NOPE: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    """Fused inverse GPT-J RoPE on the trailing rope_dim of each (token, head).

    Mirrors ``DeepseekV4ScalingRotaryEmbedding.forward_native(inverse=True)``
    for the GPT-J (non-neox) layout, writing bf16 directly. Replaces the
    clone + index_select + repeat_interleave + neg + stack + cat + cast chain
    (~10 small kernels) with a single launch.
    """
    t = tl.program_id(0)
    h = tl.program_id(1)
    in_base = t * s_t + h * s_h
    out_base = t * os_t + h * os_h

    # NoPE lanes pass through unchanged (only cast to bf16).
    n = tl.arange(0, BLOCK_NOPE)
    nmask = n < NOPE
    vals = tl.load(o_ptr + in_base + n, mask=nmask)
    tl.store(out_ptr + out_base + n, vals.to(tl.bfloat16), mask=nmask)

    # RoPE lanes: out_even = a*cos + b*sin, out_odd = b*cos - a*sin
    # (a = even lane, b = odd lane; sin negated for the inverse rotation).
    pos = tl.load(pos_ptr + t).to(tl.int64)
    k = tl.arange(0, BLOCK_HALF)
    kmask = k < HALF
    a = tl.load(o_ptr + in_base + NOPE + 2 * k, mask=kmask).to(tl.float32)
    b = tl.load(o_ptr + in_base + NOPE + 2 * k + 1, mask=kmask).to(tl.float32)
    cos = tl.load(cos_sin_ptr + pos * cs_stride + k, mask=kmask)
    sin = tl.load(cos_sin_ptr + pos * cs_stride + HALF + k, mask=kmask)
    out_even = a * cos + b * sin
    out_odd = b * cos - a * sin
    tl.store(out_ptr + out_base + NOPE + 2 * k, out_even.to(tl.bfloat16), mask=kmask)
    tl.store(out_ptr + out_base + NOPE + 2 * k + 1, out_odd.to(tl.bfloat16), mask=kmask)


def _fused_inverse_rope_gptj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_head_dim: int,
) -> torch.Tensor:
    """bf16 inverse GPT-J RoPE via a single fused Triton kernel."""
    assert o.dim() == 3 and o.stride(-1) == 1, (
        "_fused_inverse_rope_gptj expects a [T, H, D] input with a contiguous last dim"
    )
    assert rope_head_dim > 0 and rope_head_dim % 2 == 0, (
        f"_fused_inverse_rope_gptj expects an even rope_head_dim, got {rope_head_dim}"
    )
    assert cos_sin_cache.shape[-1] == rope_head_dim, (
        "_fused_inverse_rope_gptj expects cos_sin_cache laid out as "
        f"[P, {rope_head_dim}] = cos | sin, got {tuple(cos_sin_cache.shape)}"
    )
    num_tokens, num_heads, head_dim = o.shape
    out = torch.empty(
        (num_tokens, num_heads, head_dim), dtype=torch.bfloat16, device=o.device
    )
    if num_tokens == 0:
        return out
    _inverse_rope_gptj_kernel[(num_tokens, num_heads)](
        o,
        out,
        positions,
        cos_sin_cache,
        o.stride(0),
        o.stride(1),
        out.stride(0),
        out.stride(1),
        cos_sin_cache.stride(0),
        NOPE=head_dim - rope_head_dim,
        HALF=rope_head_dim // 2,
        BLOCK_NOPE=triton.next_power_of_2(head_dim - rope_head_dim),
        BLOCK_HALF=triton.next_power_of_2(rope_head_dim // 2),
    )
    return out


def _get_cached_wo_a_bf16(
    wo_a: torch.nn.Module,
    n_local_groups: int,
    o_lora_rank: int,
    hidden_dim: int,
) -> torch.Tensor:
    """Dequantize wo_a to bf16 once and cache it on the module.

    wo_a weights are static, so the fp8 -> fp32 -> (* block scale) -> bf16
    dequant only needs to run once. Recomputing it every decode step shows up
    in the profile as the largest copy/mul kernels (``direct_copy float`` ~55us
    and ``MulFunctor float`` ~31us per two layers). SGLang / ATOM keep wo_a in
    bf16 and feed a plain bf16 GEMM; this mirrors that.
    """
    cached = getattr(wo_a, "_dsv4_wo_a_bf16", None)
    if cached is not None:
        return cached
    bmm_weight = getattr(wo_a, "bmm_weight_bf16", None)
    if bmm_weight is not None:
        expected_shape = (n_local_groups, o_lora_rank, hidden_dim)
        assert bmm_weight.shape == expected_shape, (
            f"Expected preprocessed wo_a BMM weight {expected_shape}, got "
            f"{tuple(bmm_weight.shape)}"
        )
        wo_a._dsv4_wo_a_bf16 = bmm_weight
        return bmm_weight
    if wo_a.weight.element_size() >= 2:
        cached = wo_a.weight.view(n_local_groups, o_lora_rank, hidden_dim).to(
            torch.bfloat16
        )
        wo_a._dsv4_wo_a_bf16 = cached
        return cached
    wo_a_scale_param = getattr(wo_a, "weight_scale_inv", None)
    if wo_a_scale_param is None:
        # ModelOpt MXFP8 stores the multiplicative E8M0 scale without the
        # historical ``_inv`` suffix.
        wo_a_scale_param = getattr(wo_a, "weight_scale", None)
    if wo_a_scale_param is not None:
        wo_a_weight = wo_a.weight.view(n_local_groups, o_lora_rank, hidden_dim).to(
            torch.float32
        )
        wo_a_scale = _expand_2d_block_scales(
            wo_a_scale_param.view(n_local_groups, -1, wo_a_scale_param.shape[-1]),
            o_lora_rank,
            hidden_dim,
        )
        cached = (wo_a_weight * wo_a_scale).to(torch.bfloat16)
    else:
        cached = wo_a.weight.view(n_local_groups, o_lora_rank, hidden_dim).to(
            torch.bfloat16
        )
    wo_a._dsv4_wo_a_bf16 = cached
    return cached


def rocm_inv_rope_einsum(
    rotary_emb: torch.nn.Module,
    o: torch.Tensor,
    positions: torch.Tensor,
    rope_head_dim: int,
    n_local_groups: int,
    o_lora_rank: int,
    wo_a: torch.nn.Module,
) -> torch.Tensor:
    """Inverse-RoPE + WO_A bmm path used on ROCm.

    Fuses the inverse GPT-J RoPE into one Triton kernel and caches the bf16
    wo_a weight so the per-step dequant disappears.
    """
    with optional_phase_profile_stage("attention.output_projection.inv_rope", o):
        o_ref = _fused_inverse_rope_gptj(
            o, positions, rotary_emb.cos_sin_cache, rope_head_dim
        )
    o_ref = o_ref.view(o.shape[0], n_local_groups, -1)

    wo_a_weight = _get_cached_wo_a_bf16(
        wo_a, n_local_groups, o_lora_rank, o_ref.shape[-1]
    )

    with optional_phase_profile_stage("attention.output_projection.wo_a", o_ref):
        return torch.einsum("tgd,grd->tgr", o_ref, wo_a_weight)


# Name of the env var that overrides the sparse-prefill BLOCK_K. Checked with
# `os.environ` directly so a value left at its default can be told apart from
# one the operator exported on purpose.
_PREFILL_BLOCK_K_ENV = "VLLM_DEEPSEEK_V4_SPARSE_PREFILL_BLOCK_K"

# Shared memory the tuned sm_80 sparse-prefill tiling needs, read from Triton's
# own compiled-kernel metadata on a CMP 170HX: block_h=64 / block_k=64 /
# num_stages=2 with bf16 KV emits 139264 B (136 KB) against the 166912 B
# available. The fp8 lookup-table path needs more than 163 KB for a tiling of
# this width, which is why that path is excluded separately below.
_SM80_PREFILL_TUNED_SMEM_BYTES = 139264


@functools.lru_cache
def _prefill_smem_limit_bytes() -> int:
    """Opt-in shared memory per block for the current device, 0 if unknown.

    Mirrors v1/attention/backends/flex_attention.py: ROCm does not expose
    `shared_memory_per_block_optin`, so the attribute is read defensively and a
    missing value disables the sm_80 tuning rather than guessing.
    """
    if not torch.cuda.is_available():
        return 0
    props = torch.cuda.get_device_properties()
    return int(getattr(props, "shared_memory_per_block_optin", 0))


_DSV4_SPARSE_NOPE_DIM = 448
_DSV4_SPARSE_ROPE_DIM = 64


def _validate_sparse_dims(
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    op_name: str,
) -> None:
    assert head_dim > 0, f"{op_name} expected a positive head_dim, got {head_dim}"
    assert nope_head_dim > 0, (
        f"{op_name} expected a positive NoPE dimension, got {nope_head_dim}"
    )
    assert rope_head_dim >= 0, (
        f"{op_name} expected a non-negative RoPE dimension, got {rope_head_dim}"
    )
    assert head_dim == nope_head_dim + rope_head_dim, (
        f"{op_name} expected head_dim={nope_head_dim + rope_head_dim}, got {head_dim}"
    )


def _validate_dsv4_sparse_dims(
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    op_name: str,
) -> None:
    _validate_sparse_dims(head_dim, nope_head_dim, rope_head_dim, op_name)
    assert (
        nope_head_dim == _DSV4_SPARSE_NOPE_DIM
        and rope_head_dim == _DSV4_SPARSE_ROPE_DIM
    ), (
        f"{op_name} expects {_DSV4_SPARSE_NOPE_DIM} NoPE dims and "
        f"{_DSV4_SPARSE_ROPE_DIM} RoPE dims"
    )


@triton.jit
def _pack_dense_prefix_to_ragged_kernel(
    indices_ptr,
    lengths_ptr,
    indptr_ptr,
    out_ptr,
    indices_stride0,
    num_rows_limit,
    row_width,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    row_len = tl.load(lengths_ptr + row_idx)
    if block_idx * BLOCK_SIZE >= row_len:
        return

    mask = offsets < row_len
    safe_offsets = tl.where(offsets < row_width, offsets, 0)
    vals = tl.load(
        indices_ptr + row_idx * indices_stride0 + safe_offsets,
        mask=mask & (offsets < row_width),
        other=-1,
    ).to(tl.int32)
    if num_rows_limit >= 0:
        vals = tl.where((vals >= 0) & (vals < num_rows_limit), vals, -1)

    out_start = tl.load(indptr_ptr + row_idx)
    tl.store(out_ptr + out_start + offsets, vals, mask=mask)


def build_ragged_indices_from_dense(
    indices: torch.Tensor,
    lengths: torch.Tensor,
    num_rows: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = indices.reshape(indices.shape[0], -1)
    lengths = lengths.to(device=indices.device, dtype=torch.int32).reshape(-1)
    assert lengths.numel() == indices.shape[0], (
        f"Expected one length per row, got {lengths.shape} for indices {indices.shape}"
    )

    max_width = indices.shape[1] if indices.ndim == 2 else 0
    lengths = lengths.clamp(min=0, max=max_width).contiguous()

    with optional_phase_profile_stage("sparse_mla.ragged_indptr", indices):
        indptr = torch.zeros(
            indices.shape[0] + 1, dtype=torch.int32, device=indices.device
        )
        torch.cumsum(lengths, dim=0, out=indptr[1:])

    if indices.numel() == 0:
        flat = torch.empty(0, dtype=torch.int32, device=indices.device)
    else:
        flat = torch.empty(
            indices.shape[0] * max_width,
            dtype=torch.int32,
            device=indices.device,
        )
        if flat.numel() > 0:
            block_size = 128
            with optional_phase_profile_stage("sparse_mla.ragged_pack", indices):
                _pack_dense_prefix_to_ragged_kernel[
                    (indices.shape[0], triton.cdiv(max_width, block_size))
                ](
                    indices,
                    lengths,
                    indptr,
                    flat,
                    indices.stride(0),
                    int(num_rows),
                    max_width,
                    BLOCK_SIZE=block_size,
                )

    return flat, indptr


def _as_int32_contiguous_1d(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.int32 and x.ndim == 1 and x.is_contiguous():
        return x
    return x.to(torch.int32).contiguous()


@triton.jit
def _sparse_kv_row_offset(slot, stride):
    # A global token slot fits in int32, but its byte/element offset may not.
    return slot.to(tl.int64) * stride


# Paged KV-cache token layout written by ``quantize_and_insert_k_cache``: 448
# fp8 values followed by 64 bf16 rope values (576 bytes of data per token), with
# one UE8M0 byte-scale per 64-wide quant block (7 used, 1 padding) in a separate
# per-token area after the block's token data.
_SPARSE_CACHE_FP8_DIM = tl.constexpr(448)
_SPARSE_CACHE_QUANT_BLOCK = tl.constexpr(64)
_SPARSE_CACHE_QUANT_BLOCKS = tl.constexpr(7)
_SPARSE_CACHE_ROPE_DIM = tl.constexpr(64)
_SPARSE_CACHE_SCALE_DIM = tl.constexpr(8)
_SPARSE_CACHE_TOKEN_BYTES = tl.constexpr(448 + 64 * 2)

# A direct KV index encodes ``tag_base + request * req_stride + position`` so one
# int32 array can address both the paged cache (tagged) and the gathered
# sliding-window rows (untagged). req_stride is the largest context the model
# allows and tag_base sits far above any request*stride+position.
# Direct paged-cache reads take the compressed source plus the sliding-window
# source: (cache, block table, lens, block size, tag base[, request stride]).
SparseDirectKvSource = tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]
SparseDirectKv = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    int,
    SparseDirectKvSource,
]

SPARSE_DIRECT_KV_TAG_BASE = 1 << 27
SPARSE_DIRECT_KV_SWA_TAG_BASE = 1 << 28
SPARSE_DIRECT_KV_REQ_STRIDE = 1 << 20


@triton.jit
def _sparse_direct_kv_tile(
    slot,
    taken,
    cache_ptr,
    cache_bt_ptr,
    cache_lens_ptr,
    cache_block_size,
    cache_block_stride,
    cache_max_blocks,
    tag_base,
    req_stride,
    dim_offsets,
    scaled_lut_ptr,
    FP8_DIM: tl.constexpr,
    TOKEN_BYTES: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    N_QUANT_BLOCKS: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    DECODE_PROBE: tl.constexpr,
):
    """Assemble one [BLOCK_K, BLOCK_D] bf16 tile straight from the paged cache.

    ``slot`` carries ``tag_base + request * req_stride + position`` for entries to
    read from the cache; the position is a compressed-context position, so the
    paged address is ``position // cache_block_size`` through the request's block
    table. Layout and dequant follow ``dequantize_and_gather_k_cache``: 448 fp8
    values scaled by one UE8M0 byte per 64-wide quant block, then 64 bf16 rope
    values stored verbatim. Decoding through the bf16 lookup table (instead of an
    fp32 decode) keeps the tile in bf16, which is what lets BLOCK_K stay wide
    enough for the KV loads to run at the bandwidth the gathered path reaches.
    """
    rel = slot - tag_base
    req = rel // req_stride
    pos = rel - req * req_stride
    n_blocks = pos // cache_block_size
    in_block = pos - n_blocks * cache_block_size

    length = tl.load(cache_lens_ptr + req, mask=taken, other=0)
    ok = taken & (pos >= 0) & (pos < length)
    physical = tl.load(
        cache_bt_ptr + req * cache_max_blocks + n_blocks, mask=ok, other=0
    )
    block_ptr = cache_ptr + physical.to(tl.int64) * cache_block_stride
    row = block_ptr + in_block * TOKEN_BYTES
    # Scales live after the block's whole token-data area, one 8-byte group per
    # token, so they are addressed from the block and not from the token row.
    scale_row = block_ptr + cache_block_size * TOKEN_BYTES + in_block * SCALE_DIM

    fp8_mask = dim_offsets < FP8_DIM
    quantized = tl.load(
        row[:, None] + dim_offsets[None, :],
        mask=ok[:, None] & fp8_mask[None, :],
        other=0,
    )
    # One gathered byte per element: segment `d // QUANT_BLOCK` of the token holds
    # that dimension's UE8M0 scale (2 ** (stored - 127)); byte 7 is padding and is
    # never read because the fp8 dimensions stop before it.
    scale_bytes = tl.load(
        scale_row[:, None] + (dim_offsets // QUANT_BLOCK)[None, :],
        mask=ok[:, None] & fp8_mask[None, :],
        other=127,
    )
    # Dequantize with one gather per element: the table is indexed by
    # `byte | scale << 8` and already holds the scaled value, so no per-element
    # exponent arithmetic is needed.
    index = quantized.to(tl.int32) | (scale_bytes.to(tl.int32) << 8)
    if DECODE_PROBE:
        values = index.to(tl.bfloat16)
    else:
        values = tl.load(scaled_lut_ptr + index).to(tl.bfloat16, bitcast=True)

    rope = tl.load(
        (row + FP8_DIM).to(tl.pointer_type(tl.bfloat16))[:, None]
        + (dim_offsets - FP8_DIM)[None, :],
        mask=ok[:, None] & ~fp8_mask[None, :],
        other=0,
    )
    return tl.where(fp8_mask[None, :], values, rope)


@triton.jit
def _sparse_attn_prefill_ragged_kernel(
    q_ptr,
    kv_ptr,
    kv_scale_ptr,
    fp8_lut_ptr,
    kv_indices_ptr,
    kv_indptr_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    kv_stride_n,
    kv_stride_d,
    out_stride_t,
    out_stride_h,
    out_stride_d,
    num_heads,
    head_dim,
    num_kv,
    scale,
    cache_ptr,
    cache_bt_ptr,
    cache_lens_ptr,
    cache_block_size,
    cache_block_stride,
    cache_max_blocks,
    direct_tag_base,
    direct_req_stride,
    swa_cache_ptr,
    swa_bt_ptr,
    swa_lens_ptr,
    swa_block_size,
    swa_block_stride,
    swa_max_blocks,
    direct_swa_tag_base,
    scaled_lut_ptr,
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    split_id,
    pm_stride0,
    pm_stride_s,
    pa_stride0,
    pa_stride_s,
    pa_stride_h,
    DIRECT_KV: tl.constexpr,
    DECODE_PROBE: tl.constexpr,
    STORE_PARTIAL: tl.constexpr,
    KV_IS_FP8: tl.constexpr,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # None keeps Triton's own loop default, i.e. exactly the pre-tuning kernel.
    NUM_STAGES: tl.constexpr = None,
):
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim

    q = tl.load(
        q_ptr
        + query_idx * q_stride_t
        + head_offsets[:, None] * q_stride_h
        + dim_offsets[None, :] * q_stride_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    kv_start = tl.load(kv_indptr_ptr + query_idx)
    kv_end = tl.load(kv_indptr_ptr + query_idx + 1)
    kv_len = kv_end - kv_start

    k_offsets = tl.arange(0, BLOCK_K)
    slot = tl.load(
        kv_indices_ptr + kv_start + k_offsets, mask=k_offsets < kv_len, other=-1
    )
    # NUM_STAGES > 1 software-pipelines the gathered KV loads. On sm_80 the
    # gather is the bottleneck (the kernel is DRAM-bound), and pipelining plus a
    # wider BLOCK_K lifts it from ~500 GB/s to ~1100 GB/s of the 1361 GB/s
    # achievable peak. Measured 1.53x on CMP 170HX at s_q=2048, topk=512.
    for k_start in tl.range(0, kv_len, BLOCK_K, num_stages=NUM_STAGES):
        k_pos = k_start + k_offsets
        in_range = k_pos < kv_len
        if DIRECT_KV:
            # Every entry is tagged with the cache it belongs to, so compressed
            # positions and sliding-window positions are both read straight from
            # their paged cache: nothing is gathered and the prefill workspace
            # holds nothing that grows with the context. One unpack path serves
            # both sources, which is what keeps the tile wide enough to run the
            # cache loads at full bandwidth.
            is_swa = slot >= direct_swa_tag_base
            tagged = slot >= direct_tag_base
            valid = in_range & (slot >= 0) & tagged
            kv = _sparse_direct_kv_tile(
                slot,
                valid,
                tl.where(is_swa, swa_cache_ptr, cache_ptr),
                tl.where(is_swa, swa_bt_ptr, cache_bt_ptr),
                tl.where(is_swa, swa_lens_ptr, cache_lens_ptr),
                tl.where(is_swa, swa_block_size, cache_block_size),
                tl.where(is_swa, swa_block_stride, cache_block_stride),
                tl.where(is_swa, swa_max_blocks, cache_max_blocks),
                tl.where(is_swa, direct_swa_tag_base, direct_tag_base),
                direct_req_stride,
                dim_offsets,
                scaled_lut_ptr,
                FP8_DIM=_SPARSE_CACHE_FP8_DIM,
                TOKEN_BYTES=_SPARSE_CACHE_TOKEN_BYTES,
                QUANT_BLOCK=_SPARSE_CACHE_QUANT_BLOCK,
                N_QUANT_BLOCKS=_SPARSE_CACHE_QUANT_BLOCKS,
                SCALE_DIM=_SPARSE_CACHE_SCALE_DIM,
                DECODE_PROBE=DECODE_PROBE,
            )
        else:
            valid = in_range & (slot >= 0) & (slot < num_kv)
            safe_slot = tl.where(valid, slot, 0)

            kv = tl.load(
                kv_ptr
                + _sparse_kv_row_offset(safe_slot[:, None], kv_stride_n)
                + dim_offsets[None, :] * kv_stride_d,
                mask=valid[:, None] & dim_mask[None, :],
                other=0.0,
            )
            if KV_IS_FP8:
                kv = (
                    _decode_fp8_lut(kv, False, fp8_lut_ptr) * tl.load(kv_scale_ptr)
                ).to(q.dtype)

        next_k_pos = k_start + BLOCK_K + k_offsets
        slot = tl.load(
            kv_indices_ptr + kv_start + next_k_pos, mask=next_k_pos < kv_len, other=-1
        )

        scores = tl.dot(q, tl.trans(kv)) * scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    if STORE_PARTIAL:
        # Split-K: hand this block's raw online-softmax state to the reduce
        # kernel, which applies the attention sink and normalizes once.
        pm_base = query_idx * pm_stride0 + split_id * pm_stride_s + head_offsets
        tl.store(part_m_ptr + pm_base, m_i, mask=head_mask)
        tl.store(part_l_ptr + pm_base, l_i, mask=head_mask)
        acc_base = (
            part_acc_ptr
            + query_idx * pa_stride0
            + split_id * pa_stride_s
            + head_offsets[:, None] * pa_stride_h
        )
        tl.store(
            acc_base + dim_offsets[None, :],
            acc,
            mask=head_mask[:, None] & dim_mask[None, :],
        )
    elif HAS_ATTN_SINK:
        sink = tl.load(
            attn_sink_ptr + head_offsets, mask=head_mask, other=neg_large
        ).to(tl.float32)
        m_final = tl.maximum(m_i, sink)
        alpha = tl.exp(m_i - m_final)
        l_final = l_i * alpha + tl.exp(sink - m_final)
        denom = tl.maximum(l_final, 1.0e-30)
        out = tl.where(
            l_final[:, None] > 0.0,
            (acc * alpha[:, None]) / denom[:, None],
            0.0,
        )
        tl.store(
            out_ptr
            + query_idx * out_stride_t
            + head_offsets[:, None] * out_stride_h
            + dim_offsets[None, :] * out_stride_d,
            out,
            mask=head_mask[:, None] & dim_mask[None, :],
        )
    else:
        denom = tl.maximum(l_i, 1.0e-30)
        out = tl.where(l_i[:, None] > 0.0, acc / denom[:, None], 0.0)
        tl.store(
            out_ptr
            + query_idx * out_stride_t
            + head_offsets[:, None] * out_stride_h
            + dim_offsets[None, :] * out_stride_d,
            out,
            mask=head_mask[:, None] & dim_mask[None, :],
        )


@triton.jit
def _decode_e8m0_scales_triton(encoded_scales):
    scale_bits = encoded_scales.to(tl.int32) << 23
    scale_bits = tl.where(encoded_scales == 0, 1 << 22, scale_bits)
    return scale_bits.to(tl.float32, bitcast=True)


@triton.jit
def _load_fp8_ds_mla_gfx950_nope_exact_chunk(
    token_data_ptr,
    token_scale_ptr,
    valid,
    CHUNK_START: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_FNUZ: tl.constexpr,
):
    offsets = CHUNK_START + tl.arange(0, CHUNK_SIZE)
    x_uint8 = tl.load(
        token_data_ptr[:, None] + offsets[None, :],
        mask=valid[:, None],
        other=0,
    )
    scale_offsets = CHUNK_START // 64 + tl.arange(0, CHUNK_SIZE // 64)
    encoded_scales = tl.load(
        token_scale_ptr[:, None] + scale_offsets[None, :],
        mask=valid[:, None],
        other=127,
    )
    scales = _decode_e8m0_scales_triton(encoded_scales)
    scales = tl.broadcast_to(scales[:, :, None], (BLOCK_K, CHUNK_SIZE // 64, 64))
    scales = tl.reshape(scales, (BLOCK_K, CHUNK_SIZE))
    if IS_FNUZ:
        x_f32 = x_uint8.to(tl.float8e4b8, bitcast=True).to(tl.bfloat16).to(tl.float32)
    else:
        x_f32 = x_uint8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    value = (x_f32 * scales).to(tl.bfloat16)
    zero = tl.zeros((BLOCK_K, CHUNK_SIZE), dtype=tl.bfloat16)
    return tl.where(valid[:, None], value, zero)


@triton.jit
def _load_fp8_ds_mla_gfx950_tail128(
    token_data_ptr,
    token_scale_ptr,
    valid,
    NOPE_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_FNUZ: tl.constexpr,
):
    tail_offsets = 384 + tl.arange(0, 128)
    nope_mask = tail_offsets < NOPE_DIM
    x_uint8 = tl.load(
        token_data_ptr[:, None] + tail_offsets[None, :],
        mask=valid[:, None] & nope_mask[None, :],
        other=0,
    )
    scale_offsets = 6 + tl.arange(0, 2)
    scale_mask = scale_offsets < NOPE_DIM // 64
    encoded_scales = tl.load(
        token_scale_ptr[:, None] + scale_offsets[None, :],
        mask=valid[:, None] & scale_mask[None, :],
        other=127,
    )
    scales = _decode_e8m0_scales_triton(encoded_scales)
    scales = tl.broadcast_to(scales[:, :, None], (BLOCK_K, 2, 64))
    scales = tl.reshape(scales, (BLOCK_K, 128))
    if IS_FNUZ:
        x_f32 = x_uint8.to(tl.float8e4b8, bitcast=True).to(tl.bfloat16).to(tl.float32)
    else:
        x_f32 = x_uint8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    nope = (x_f32 * scales).to(tl.bfloat16)

    rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
    rope = tl.load(
        rope_ptr[:, None] + (tail_offsets[None, :] - NOPE_DIM),
        mask=valid[:, None] & ~nope_mask[None, :],
        other=0.0,
    )
    value = tl.where(nope_mask[None, :], nope, rope)
    zero = tl.zeros((BLOCK_K, 128), dtype=tl.bfloat16)
    return tl.where(valid[:, None], value, zero)


@triton.jit
def _sparse_attn_decode_ragged_kernel(
    q_ptr,
    main_cache_ptr,
    main_indices_ptr,
    main_indptr_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    extra_indptr_ptr,
    attn_sink_ptr,
    fp8_lut_ptr,
    out_ptr,
    q_stride0,
    q_stride1,
    out_stride0,
    out_stride1,
    main_cache_stride0,
    extra_cache_stride0,
    main_num_rows,
    extra_num_rows,
    main_block_size,
    extra_block_size,
    scale,
    num_heads,
    HAS_ATTN_SINK: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    # SWA K-cache (main): C++ encoder writes FNUZ on gfx942, OCP on gfx950.
    # Compressed K-cache (extra): Triton encoder writes OCP everywhere.
    IS_FNUZ_MAIN: tl.constexpr,
    IS_FNUZ_EXTRA: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    nope_offsets = tl.arange(0, NOPE_BLOCK)
    nope_mask = nope_offsets < NOPE_DIM
    rope_offsets = tl.arange(0, ROPE_DIM)

    q_row_ptr = q_ptr + query_idx * q_stride0 + head_offsets[:, None] * q_stride1
    q_nope = tl.load(
        q_row_ptr + nope_offsets[None, :],
        mask=head_mask[:, None] & nope_mask[None, :],
        other=0.0,
    )
    q_rope = tl.load(
        q_row_ptr + NOPE_DIM + rope_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_nope = tl.zeros((BLOCK_H, NOPE_BLOCK), dtype=tl.float32)
    acc_rope = tl.zeros((BLOCK_H, ROPE_DIM), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)

    main_start = tl.load(main_indptr_ptr + query_idx)
    main_end = tl.load(main_indptr_ptr + query_idx + 1)
    main_len = main_end - main_start

    zero_nope = tl.zeros((BLOCK_K, NOPE_BLOCK), dtype=tl.bfloat16)
    zero_rope = tl.zeros((BLOCK_K, ROPE_DIM), dtype=tl.bfloat16)

    for k_start in tl.range(0, main_len, BLOCK_K):
        k_pos = k_start + k_offsets
        in_range = k_pos < main_len
        slot = tl.load(main_indices_ptr + main_start + k_pos, mask=in_range, other=-1)
        valid = in_range & (slot >= 0) & (slot < main_num_rows)
        safe_slot = tl.where(valid, slot, 0)

        block_idx = safe_slot // main_block_size
        pos_in_block = safe_slot % main_block_size
        cache_block_ptr = main_cache_ptr + block_idx.to(tl.int64) * main_cache_stride0
        token_data_ptr = cache_block_ptr + pos_in_block * 576
        token_scale_ptr = cache_block_ptr + main_block_size * 576 + pos_in_block * 8

        x_uint8 = tl.load(
            token_data_ptr[:, None] + nope_offsets[None, :],
            mask=valid[:, None] & nope_mask[None, :],
            other=0,
        )
        k_vals = _decode_fp8_lut(x_uint8, IS_FNUZ_MAIN, fp8_lut_ptr)
        encoded_scales = tl.load(
            token_scale_ptr[:, None] + nope_offsets[None, :] // 64,
            mask=valid[:, None] & nope_mask[None, :],
            other=127,
        )
        scales = tl.exp2(encoded_scales.to(tl.float32) - 127.0)
        k_nope = k_vals.to(tl.bfloat16) * scales.to(tl.bfloat16)
        k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_nope)
        k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)

        rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
        k_rope = tl.load(
            rope_ptr[:, None] + rope_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        k_rope = tl.where(valid[:, None], k_rope, zero_rope)
        k_rope = tl.where(k_rope == k_rope, k_rope, zero_rope)

        scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(q_rope, tl.trans(k_rope))
        scores *= scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc_nope = acc_nope * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
        acc_rope = acc_rope * alpha[:, None] + tl.dot(p.to(k_rope.dtype), k_rope)
        m_i = m_new
        l_i = l_new

    if HAS_EXTRA:
        extra_start = tl.load(extra_indptr_ptr + query_idx)
        extra_end = tl.load(extra_indptr_ptr + query_idx + 1)
        extra_len = extra_end - extra_start

        for k_start in tl.range(0, extra_len, BLOCK_K):
            k_pos = k_start + k_offsets
            in_range = k_pos < extra_len
            slot = tl.load(
                extra_indices_ptr + extra_start + k_pos, mask=in_range, other=-1
            )
            valid = in_range & (slot >= 0) & (slot < extra_num_rows)
            safe_slot = tl.where(valid, slot, 0)

            block_idx = safe_slot // extra_block_size
            pos_in_block = safe_slot % extra_block_size
            cache_block_ptr = (
                extra_cache_ptr + block_idx.to(tl.int64) * extra_cache_stride0
            )
            token_data_ptr = cache_block_ptr + pos_in_block * 576
            token_scale_ptr = (
                cache_block_ptr + extra_block_size * 576 + pos_in_block * 8
            )

            x_uint8 = tl.load(
                token_data_ptr[:, None] + nope_offsets[None, :],
                mask=valid[:, None] & nope_mask[None, :],
                other=0,
            )
            k_vals = _decode_fp8_lut(x_uint8, IS_FNUZ_EXTRA, fp8_lut_ptr)
            encoded_scales = tl.load(
                token_scale_ptr[:, None] + nope_offsets[None, :] // 64,
                mask=valid[:, None] & nope_mask[None, :],
                other=127,
            )
            scales = tl.exp2(encoded_scales.to(tl.float32) - 127.0)
            k_nope = k_vals.to(tl.bfloat16) * scales.to(tl.bfloat16)
            k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_nope)
            k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)

            rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
            k_rope = tl.load(
                rope_ptr[:, None] + rope_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            k_rope = tl.where(valid[:, None], k_rope, zero_rope)
            k_rope = tl.where(k_rope == k_rope, k_rope, zero_rope)

            scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(
                q_rope,
                tl.trans(k_rope),
            )
            scores *= scale
            scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

            m_block = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_block)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=1)

            acc_nope = acc_nope * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
            acc_rope = acc_rope * alpha[:, None] + tl.dot(p.to(k_rope.dtype), k_rope)
            m_i = m_new
            l_i = l_new

    if HAS_ATTN_SINK:
        sink = tl.load(
            attn_sink_ptr + head_offsets, mask=head_mask, other=neg_large
        ).to(tl.float32)
        m_final = tl.maximum(m_i, sink)
        alpha = tl.exp(m_i - m_final)
        l_final = l_i * alpha + tl.exp(sink - m_final)
        denom = tl.maximum(l_final, 1.0e-30)
        out_nope = tl.where(
            l_final[:, None] > 0.0,
            (acc_nope * alpha[:, None]) / denom[:, None],
            0.0,
        )
        out_rope = tl.where(
            l_final[:, None] > 0.0,
            (acc_rope * alpha[:, None]) / denom[:, None],
            0.0,
        )
    else:
        denom = tl.maximum(l_i, 1.0e-30)
        out_nope = tl.where(l_i[:, None] > 0.0, acc_nope / denom[:, None], 0.0)
        out_rope = tl.where(l_i[:, None] > 0.0, acc_rope / denom[:, None], 0.0)

    out_row_ptr = (
        out_ptr + query_idx * out_stride0 + head_offsets[:, None] * out_stride1
    )
    tl.store(
        out_row_ptr + nope_offsets[None, :],
        out_nope,
        mask=head_mask[:, None] & nope_mask[None, :],
    )
    tl.store(
        out_row_ptr + NOPE_DIM + rope_offsets[None, :],
        out_rope,
        mask=head_mask[:, None],
    )


@triton.jit
def _sparse_attn_decode_partial_kernel(
    q_ptr,
    main_cache_ptr,
    main_indices_ptr,
    main_indptr_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    extra_indptr_ptr,
    main_block_table_ptr,
    main_seq_lens_ptr,
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    fp8_lut_ptr,
    q_stride0,
    q_stride1,
    main_cache_stride0,
    extra_cache_stride0,
    main_block_table_stride,
    pm_stride0,
    pm_stride_s,
    pa_stride0,
    pa_stride_s,
    pa_stride_h,
    main_num_rows,
    extra_num_rows,
    main_block_size,
    extra_block_size,
    main_window_size,
    scale,
    num_heads,
    HAS_EXTRA: tl.constexpr,
    HAS_PAGED_MAIN: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    # `main_cache` is the SWA K-cache (written by the C++ encoder, FNUZ on
    # gfx942 / OCP on gfx950). `extra_cache` is the compressed K-cache
    # (Triton encoder, OCP on every platform). Reading both with the same
    # `IS_FNUZ` would decode one of them with the wrong FNUZ/OCP scale ratio.
    IS_FNUZ_MAIN: tl.constexpr,
    IS_FNUZ_EXTRA: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    query_idx = tl.program_id(0)
    split_id = tl.program_id(1)
    pid_h = tl.program_id(2)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    nope_offsets = tl.arange(0, NOPE_BLOCK)
    nope_mask = nope_offsets < NOPE_DIM
    rope_offsets = tl.arange(0, ROPE_DIM)

    q_row_ptr = q_ptr + query_idx * q_stride0 + head_offsets[:, None] * q_stride1
    q_nope = tl.load(
        q_row_ptr + nope_offsets[None, :],
        mask=head_mask[:, None] & nope_mask[None, :],
        other=0.0,
    )
    q_rope = tl.load(
        q_row_ptr + NOPE_DIM + rope_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_nope = tl.zeros((BLOCK_H, NOPE_BLOCK), dtype=tl.float32)
    acc_rope = tl.zeros((BLOCK_H, ROPE_DIM), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)

    zero_nope = tl.zeros((BLOCK_K, NOPE_BLOCK), dtype=tl.bfloat16)
    zero_rope = tl.zeros((BLOCK_K, ROPE_DIM), dtype=tl.bfloat16)

    # Each split processes a contiguous slice of this query's main (SWA) and
    # extra (topk) segments. Slices are handled independently so a block never
    # straddles the main/extra boundary.
    if HAS_PAGED_MAIN:
        main_seq_len = tl.load(main_seq_lens_ptr + query_idx)
        main_start = 0
        main_len = tl.minimum(main_seq_len, main_window_size)
        main_pos_start = main_seq_len - main_len
    else:
        main_start = tl.load(main_indptr_ptr + query_idx)
        main_end = tl.load(main_indptr_ptr + query_idx + 1)
        main_len = main_end - main_start
        main_pos_start = 0
    main_chunk = (main_len + NUM_SPLITS - 1) // NUM_SPLITS
    main_lo = split_id * main_chunk
    main_hi = tl.minimum(main_lo + main_chunk, main_len)

    for k_start in tl.range(main_lo, main_hi, BLOCK_K, num_stages=NUM_STAGES):
        k_pos = k_start + k_offsets
        in_range = k_pos < main_hi
        if HAS_PAGED_MAIN:
            logical_pos = main_pos_start + k_pos
            page_idx = logical_pos // main_block_size
            block_id = tl.load(
                main_block_table_ptr + query_idx * main_block_table_stride + page_idx,
                mask=in_range,
                other=-1,
            )
            slot = block_id * main_block_size + logical_pos % main_block_size
        else:
            slot = tl.load(
                main_indices_ptr + main_start + k_pos, mask=in_range, other=-1
            )
        valid = in_range & (slot >= 0) & (slot < main_num_rows)
        safe_slot = tl.where(valid, slot, 0)

        block_idx = safe_slot // main_block_size
        pos_in_block = safe_slot % main_block_size
        cache_block_ptr = main_cache_ptr + block_idx.to(tl.int64) * main_cache_stride0
        token_data_ptr = cache_block_ptr + pos_in_block * 576
        token_scale_ptr = cache_block_ptr + main_block_size * 576 + pos_in_block * 8

        x_uint8 = tl.load(
            token_data_ptr[:, None] + nope_offsets[None, :],
            mask=valid[:, None] & nope_mask[None, :],
            other=0,
        )
        k_vals = _decode_fp8_lut(x_uint8, IS_FNUZ_MAIN, fp8_lut_ptr)
        encoded_scales = tl.load(
            token_scale_ptr[:, None] + nope_offsets[None, :] // 64,
            mask=valid[:, None] & nope_mask[None, :],
            other=127,
        )
        scales = tl.exp2(encoded_scales.to(tl.float32) - 127.0)
        k_nope = k_vals.to(tl.bfloat16) * scales.to(tl.bfloat16)
        k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_nope)
        k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)

        rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
        k_rope = tl.load(
            rope_ptr[:, None] + rope_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        k_rope = tl.where(valid[:, None], k_rope, zero_rope)
        k_rope = tl.where(k_rope == k_rope, k_rope, zero_rope)

        scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(q_rope, tl.trans(k_rope))
        scores *= scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc_nope = acc_nope * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
        acc_rope = acc_rope * alpha[:, None] + tl.dot(p.to(k_rope.dtype), k_rope)
        m_i = m_new
        l_i = l_new

    if HAS_EXTRA:
        extra_start = tl.load(extra_indptr_ptr + query_idx)
        extra_end = tl.load(extra_indptr_ptr + query_idx + 1)
        extra_len = extra_end - extra_start
        extra_chunk = (extra_len + NUM_SPLITS - 1) // NUM_SPLITS
        extra_lo = split_id * extra_chunk
        extra_hi = tl.minimum(extra_lo + extra_chunk, extra_len)

        for k_start in tl.range(extra_lo, extra_hi, BLOCK_K, num_stages=NUM_STAGES):
            k_pos = k_start + k_offsets
            in_range = k_pos < extra_hi
            slot = tl.load(
                extra_indices_ptr + extra_start + k_pos, mask=in_range, other=-1
            )
            valid = in_range & (slot >= 0) & (slot < extra_num_rows)
            safe_slot = tl.where(valid, slot, 0)

            block_idx = safe_slot // extra_block_size
            pos_in_block = safe_slot % extra_block_size
            cache_block_ptr = (
                extra_cache_ptr + block_idx.to(tl.int64) * extra_cache_stride0
            )
            token_data_ptr = cache_block_ptr + pos_in_block * 576
            token_scale_ptr = (
                cache_block_ptr + extra_block_size * 576 + pos_in_block * 8
            )

            x_uint8 = tl.load(
                token_data_ptr[:, None] + nope_offsets[None, :],
                mask=valid[:, None] & nope_mask[None, :],
                other=0,
            )
            k_vals = _decode_fp8_lut(x_uint8, IS_FNUZ_EXTRA, fp8_lut_ptr)
            encoded_scales = tl.load(
                token_scale_ptr[:, None] + nope_offsets[None, :] // 64,
                mask=valid[:, None] & nope_mask[None, :],
                other=127,
            )
            scales = tl.exp2(encoded_scales.to(tl.float32) - 127.0)
            k_nope = k_vals.to(tl.bfloat16) * scales.to(tl.bfloat16)
            k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_nope)
            k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)

            rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
            k_rope = tl.load(
                rope_ptr[:, None] + rope_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            k_rope = tl.where(valid[:, None], k_rope, zero_rope)
            k_rope = tl.where(k_rope == k_rope, k_rope, zero_rope)

            scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(
                q_rope,
                tl.trans(k_rope),
            )
            scores *= scale
            scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

            m_block = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_block)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=1)

            acc_nope = acc_nope * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
            acc_rope = acc_rope * alpha[:, None] + tl.dot(p.to(k_rope.dtype), k_rope)
            m_i = m_new
            l_i = l_new

    # Store raw (un-normalized) partial state for this split. Softmax sink and
    # final normalization happen in the reduce kernel.
    pm_base = query_idx * pm_stride0 + split_id * pm_stride_s + head_offsets
    tl.store(part_m_ptr + pm_base, m_i, mask=head_mask)
    tl.store(part_l_ptr + pm_base, l_i, mask=head_mask)
    acc_base = (
        part_acc_ptr
        + query_idx * pa_stride0
        + split_id * pa_stride_s
        + head_offsets[:, None] * pa_stride_h
    )
    tl.store(
        acc_base + nope_offsets[None, :],
        acc_nope,
        mask=head_mask[:, None] & nope_mask[None, :],
    )
    tl.store(
        acc_base + NOPE_DIM + rope_offsets[None, :],
        acc_rope,
        mask=head_mask[:, None],
    )


@triton.jit
def _sparse_attn_decode_gfx950_partial_loaded_tile(
    q_combined,
    cache_ptr,
    slot,
    valid,
    cache_stride0,
    scale: tl.constexpr,
    head_mask,
    m_i,
    l_i,
    acc_nope_0a,
    acc_nope_0b,
    acc_nope_1,
    acc_tail,
    BLOCK_SIZE: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_FNUZ: tl.constexpr,
    TRUST_EXTRA_CACHE_NAN_FREE: tl.constexpr,
):
    safe_slot = tl.where(valid, slot, 0)
    block_idx = safe_slot // BLOCK_SIZE
    pos_in_block = safe_slot % BLOCK_SIZE
    cache_block_ptr = cache_ptr + block_idx.to(tl.int64) * cache_stride0
    token_data_ptr = cache_block_ptr + pos_in_block * 576
    token_scale_ptr = cache_block_ptr + BLOCK_SIZE * 576 + pos_in_block * 8
    k_nope_0a = _load_fp8_ds_mla_gfx950_nope_exact_chunk(
        token_data_ptr,
        token_scale_ptr,
        valid,
        0,
        128,
        BLOCK_K,
        IS_FNUZ,
    )
    k_nope_0b = _load_fp8_ds_mla_gfx950_nope_exact_chunk(
        token_data_ptr,
        token_scale_ptr,
        valid,
        128,
        128,
        BLOCK_K,
        IS_FNUZ,
    )
    k_nope_1 = _load_fp8_ds_mla_gfx950_nope_exact_chunk(
        token_data_ptr,
        token_scale_ptr,
        valid,
        256,
        128,
        BLOCK_K,
        IS_FNUZ,
    )
    k_tail = _load_fp8_ds_mla_gfx950_tail128(
        token_data_ptr,
        token_scale_ptr,
        valid,
        NOPE_DIM,
        BLOCK_K,
        IS_FNUZ,
    )
    if not TRUST_EXTRA_CACHE_NAN_FREE:
        zero = tl.zeros((BLOCK_K, 128), dtype=tl.bfloat16)
        k_nope_0a = tl.where(k_nope_0a == k_nope_0a, k_nope_0a, zero)
        k_nope_0b = tl.where(k_nope_0b == k_nope_0b, k_nope_0b, zero)
        k_nope_1 = tl.where(k_nope_1 == k_nope_1, k_nope_1, zero)
        k_tail = tl.where(k_tail == k_tail, k_tail, zero)
    k_nope_0 = tl.cat(k_nope_0a, k_nope_0b, dim=1)
    k_tail_256 = tl.cat(k_nope_1, k_tail, dim=1)
    k_combined = tl.cat(k_nope_0, k_tail_256, dim=1)

    scores = tl.dot(q_combined, tl.trans(k_combined))
    scores *= scale * 1.4426950408889634
    scores = tl.where(
        head_mask[:, None] & valid[None, :],
        scores,
        -3.4028234663852886e38,
    )
    m_block = tl.max(scores, axis=1)
    m_new = tl.maximum(m_i, m_block)
    alpha = tl.exp2(m_i - m_new)
    p = tl.exp2(scores - m_new[:, None])
    p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
    l_new = l_i * alpha + tl.sum(p, axis=1)
    p_bf16 = p.to(k_nope_0a.dtype)
    acc_nope_0a = acc_nope_0a * alpha[:, None] + tl.dot(p_bf16, k_nope_0a)
    acc_nope_0b = acc_nope_0b * alpha[:, None] + tl.dot(p_bf16, k_nope_0b)
    acc_nope_1 = acc_nope_1 * alpha[:, None] + tl.dot(p_bf16, k_nope_1)
    acc_tail = acc_tail * alpha[:, None] + tl.dot(p_bf16, k_tail)
    return (
        m_new,
        l_new,
        acc_nope_0a,
        acc_nope_0b,
        acc_nope_1,
        acc_tail,
    )


@triton.jit
def _sparse_attn_decode_gfx950_partial_kernel(
    q_ptr,
    main_cache_ptr,
    main_indices_ptr,
    main_indptr_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    extra_indptr_ptr,
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    q_stride0: tl.constexpr,
    q_stride1: tl.constexpr,
    main_cache_stride0: tl.constexpr,
    extra_cache_stride0: tl.constexpr,
    main_num_rows,
    extra_num_rows,
    MAIN_BLOCK_SIZE: tl.constexpr,
    EXTRA_BLOCK_SIZE: tl.constexpr,
    scale: tl.constexpr,
    num_heads: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    IS_FNUZ_MAIN: tl.constexpr,
    IS_FNUZ_EXTRA: tl.constexpr,
    TRUST_EXTRA_CACHE_NAN_FREE: tl.constexpr,
    ADAPTIVE_SPLITS: tl.constexpr,
    ONE_WAVE_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    query_idx = tl.program_id(0)
    split_id = tl.program_id(1)
    pid_h = tl.program_id(2)

    tl.static_assert(NOPE_DIM == 448)
    tl.static_assert(ROPE_DIM == 64)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    if num_heads % BLOCK_H == 0:
        head_mask = tl.full((BLOCK_H,), True, tl.int1)
    else:
        head_mask = head_offsets < num_heads
    neg_large = -3.4028234663852886e38

    if ADAPTIVE_SPLITS:
        main_start = tl.load(main_indptr_ptr + query_idx)
        main_end = tl.load(main_indptr_ptr + query_idx + 1)
        main_len = main_end - main_start
        if HAS_EXTRA:
            extra_start = tl.load(extra_indptr_ptr + query_idx)
            extra_end = tl.load(extra_indptr_ptr + query_idx + 1)
            extra_len = extra_end - extra_start
        else:
            extra_start = 0
            extra_len = 0
        split4_span: tl.constexpr = 4 * BLOCK_K
        split4_iters = (main_len + split4_span - 1) // split4_span
        split4_iters += (extra_len + split4_span - 1) // split4_span
        use_four_splits = split4_iters <= 3
        work_splits = NUM_SPLITS
        if ONE_WAVE_SPLITS > 4 and ONE_WAVE_SPLITS < NUM_SPLITS:
            one_wave_span: tl.constexpr = ONE_WAVE_SPLITS * BLOCK_K
            one_wave_iters = (main_len + one_wave_span - 1) // one_wave_span
            one_wave_iters += (extra_len + one_wave_span - 1) // one_wave_span
            work_splits = tl.where(one_wave_iters <= 3, ONE_WAVE_SPLITS, work_splits)
        work_splits = tl.where(use_four_splits, 4, work_splits)
        if split_id >= work_splits:
            pm_base = (query_idx * NUM_SPLITS + split_id) * num_heads + head_offsets
            tl.store(part_m_ptr + pm_base, neg_large, mask=head_mask)
            tl.store(part_l_ptr + pm_base, 0.0, mask=head_mask)
            return
    else:
        work_splits = NUM_SPLITS

    nope_offsets_0a = tl.arange(0, 128)
    nope_offsets_0b = 128 + tl.arange(0, 128)
    nope_offsets_0 = tl.arange(0, 256)
    tail_offsets = 256 + tl.arange(0, 256)
    nope_offsets_1 = 256 + tl.arange(0, 128)
    tail_offsets_128 = 384 + tl.arange(0, 128)

    q_row_ptr = q_ptr + query_idx * q_stride0 + head_offsets[:, None] * q_stride1
    q_nope_0 = tl.load(
        q_row_ptr + nope_offsets_0[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )
    q_tail = tl.load(
        q_row_ptr + tail_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )
    q_combined = tl.cat(q_nope_0, q_tail, dim=1)

    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_nope_0a = tl.zeros((BLOCK_H, 128), dtype=tl.float32)
    acc_nope_0b = tl.zeros((BLOCK_H, 128), dtype=tl.float32)
    acc_nope_1 = tl.zeros((BLOCK_H, 128), dtype=tl.float32)
    acc_tail = tl.zeros((BLOCK_H, 128), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)

    if not ADAPTIVE_SPLITS:
        main_start = tl.load(main_indptr_ptr + query_idx)
        main_end = tl.load(main_indptr_ptr + query_idx + 1)
        main_len = main_end - main_start
    main_chunk = (main_len + work_splits - 1) // work_splits
    main_lo = split_id * main_chunk
    main_hi = tl.minimum(main_lo + main_chunk, main_len)

    for k_start in tl.range(
        main_lo,
        main_hi,
        BLOCK_K,
        num_stages=NUM_STAGES,
    ):
        k_pos = k_start + k_offsets
        in_range = k_pos < main_hi
        slot = tl.load(main_indices_ptr + main_start + k_pos, mask=in_range, other=-1)
        valid = in_range & (slot >= 0) & (slot < main_num_rows)
        (
            m_i,
            l_i,
            acc_nope_0a,
            acc_nope_0b,
            acc_nope_1,
            acc_tail,
        ) = _sparse_attn_decode_gfx950_partial_loaded_tile(
            q_combined,
            main_cache_ptr,
            slot,
            valid,
            main_cache_stride0,
            scale,
            head_mask,
            m_i,
            l_i,
            acc_nope_0a,
            acc_nope_0b,
            acc_nope_1,
            acc_tail,
            MAIN_BLOCK_SIZE,
            NOPE_DIM,
            BLOCK_K,
            IS_FNUZ_MAIN,
            False,
        )

    if HAS_EXTRA:
        if not ADAPTIVE_SPLITS:
            extra_start = tl.load(extra_indptr_ptr + query_idx)
            extra_end = tl.load(extra_indptr_ptr + query_idx + 1)
            extra_len = extra_end - extra_start
        extra_chunk = (extra_len + work_splits - 1) // work_splits
        extra_lo = split_id * extra_chunk
        extra_hi = tl.minimum(extra_lo + extra_chunk, extra_len)

        outer_block_k: tl.constexpr = 2 * BLOCK_K
        outer_k_offsets = tl.arange(0, outer_block_k)
        extra_hi_full = (
            extra_lo + ((extra_hi - extra_lo) // outer_block_k) * outer_block_k
        )
        for k_start in tl.range(
            extra_lo,
            extra_hi_full,
            outer_block_k,
            num_stages=NUM_STAGES,
        ):
            slot = tl.load(extra_indices_ptr + extra_start + k_start + outer_k_offsets)
            valid = (slot >= 0) & (slot < extra_num_rows)
            slot_pairs = tl.trans(tl.reshape(slot, (2, BLOCK_K)))
            valid_pairs = tl.trans(tl.reshape(valid, (2, BLOCK_K)))
            slot_lo, slot_hi = tl.split(slot_pairs)
            valid_lo, valid_hi = tl.split(valid_pairs)
            (
                m_i,
                l_i,
                acc_nope_0a,
                acc_nope_0b,
                acc_nope_1,
                acc_tail,
            ) = _sparse_attn_decode_gfx950_partial_loaded_tile(
                q_combined,
                extra_cache_ptr,
                slot_lo,
                valid_lo,
                extra_cache_stride0,
                scale,
                head_mask,
                m_i,
                l_i,
                acc_nope_0a,
                acc_nope_0b,
                acc_nope_1,
                acc_tail,
                EXTRA_BLOCK_SIZE,
                NOPE_DIM,
                BLOCK_K,
                IS_FNUZ_EXTRA,
                TRUST_EXTRA_CACHE_NAN_FREE,
            )
            (
                m_i,
                l_i,
                acc_nope_0a,
                acc_nope_0b,
                acc_nope_1,
                acc_tail,
            ) = _sparse_attn_decode_gfx950_partial_loaded_tile(
                q_combined,
                extra_cache_ptr,
                slot_hi,
                valid_hi,
                extra_cache_stride0,
                scale,
                head_mask,
                m_i,
                l_i,
                acc_nope_0a,
                acc_nope_0b,
                acc_nope_1,
                acc_tail,
                EXTRA_BLOCK_SIZE,
                NOPE_DIM,
                BLOCK_K,
                IS_FNUZ_EXTRA,
                TRUST_EXTRA_CACHE_NAN_FREE,
            )
        for tail_idx in tl.static_range(2):
            tail_start = extra_hi_full + tail_idx * BLOCK_K
            if tail_start < extra_hi:
                k_pos = tail_start + k_offsets
                in_range = k_pos < extra_hi
                slot = tl.load(
                    extra_indices_ptr + extra_start + k_pos,
                    mask=in_range,
                    other=-1,
                )
                valid = in_range & (slot >= 0) & (slot < extra_num_rows)
                (
                    m_i,
                    l_i,
                    acc_nope_0a,
                    acc_nope_0b,
                    acc_nope_1,
                    acc_tail,
                ) = _sparse_attn_decode_gfx950_partial_loaded_tile(
                    q_combined,
                    extra_cache_ptr,
                    slot,
                    valid,
                    extra_cache_stride0,
                    scale,
                    head_mask,
                    m_i,
                    l_i,
                    acc_nope_0a,
                    acc_nope_0b,
                    acc_nope_1,
                    acc_tail,
                    EXTRA_BLOCK_SIZE,
                    NOPE_DIM,
                    BLOCK_K,
                    IS_FNUZ_EXTRA,
                    TRUST_EXTRA_CACHE_NAN_FREE,
                )

    pm_base = (query_idx * NUM_SPLITS + split_id) * num_heads + head_offsets
    m_store = tl.where(l_i > 0.0, m_i * 0.6931471805599453, neg_large)
    tl.store(part_m_ptr + pm_base, m_store, mask=head_mask)
    tl.store(part_l_ptr + pm_base, l_i, mask=head_mask)
    acc_base = part_acc_ptr + (
        (query_idx * NUM_SPLITS + split_id) * num_heads + head_offsets[:, None]
    ) * (NOPE_DIM + ROPE_DIM)
    tl.store(
        acc_base + nope_offsets_0a[None, :],
        acc_nope_0a,
        mask=head_mask[:, None],
    )
    tl.store(
        acc_base + nope_offsets_0b[None, :],
        acc_nope_0b,
        mask=head_mask[:, None],
    )
    tl.store(
        acc_base + nope_offsets_1[None, :],
        acc_nope_1,
        mask=head_mask[:, None],
    )
    tl.store(
        acc_base + tail_offsets_128[None, :],
        acc_tail,
        mask=head_mask[:, None],
    )


@triton.jit
def _sparse_attn_decode_reduce_kernel(
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    attn_sink_ptr,
    out_ptr,
    out_stride0,
    out_stride1,
    pm_stride0,
    pm_stride_s,
    pa_stride0,
    pa_stride_s,
    pa_stride_h,
    num_heads,
    HAS_ATTN_SINK: tl.constexpr,
    ADAPTIVE_SPLITS: tl.constexpr,
    COMB_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SPLITS_PAD: tl.constexpr,
):
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    comb_offsets = tl.arange(0, COMB_DIM)
    # SPLITS_PAD is NUM_SPLITS rounded up to a power of two so the parallel
    # split-axis load is a legal arange for any split count; padding lanes are
    # masked off.
    split_offsets = tl.arange(0, SPLITS_PAD)
    split_mask = split_offsets < NUM_SPLITS

    neg_large = -3.4028234663852886e38

    # Phase 1: load every split's running max/sum at once and reduce the max
    # in parallel (tl.max over the split axis) instead of walking the splits
    # serially. This breaks the long online-softmax dependency chain that made
    # the reduce latency-bound.
    load_mask = split_mask[:, None] & head_mask[None, :]
    pm_split = (
        part_m_ptr
        + query_idx * pm_stride0
        + split_offsets[:, None] * pm_stride_s
        + head_offsets[None, :]
    )
    m_all = tl.load(pm_split, mask=load_mask, other=neg_large)  # [S, H]
    l_all = tl.load(
        part_l_ptr
        + query_idx * pm_stride0
        + split_offsets[:, None] * pm_stride_s
        + head_offsets[None, :],
        mask=load_mask,
        other=0.0,
    )

    m_comb = tl.max(m_all, axis=0)  # [H]
    if HAS_ATTN_SINK:
        sink = tl.load(
            attn_sink_ptr + head_offsets, mask=head_mask, other=neg_large
        ).to(tl.float32)
        m_final = tl.maximum(m_comb, sink)
    else:
        m_final = m_comb

    w_all = tl.exp(m_all - m_final[None, :])  # [S, H]
    w_all = tl.where(load_mask, w_all, 0.0)
    l_final = tl.sum(w_all * l_all, axis=0)  # [H]
    if HAS_ATTN_SINK:
        l_final = l_final + tl.exp(sink - m_final)
    denom = tl.maximum(l_final, 1.0e-30)

    # Phase 2: weighted sum of the per-split accumulators. The combine weight
    # for each split only depends on the (already known) global max, so the
    # acc loads carry no cross-split dependency and the compiler can pipeline
    # them; only the cheap FMA into `acc` is loop-carried.
    acc = tl.zeros((BLOCK_H, COMB_DIM), dtype=tl.float32)
    for s in tl.static_range(NUM_SPLITS):
        m_s = tl.load(
            part_m_ptr + query_idx * pm_stride0 + s * pm_stride_s + head_offsets,
            mask=head_mask,
            other=neg_large,
        )
        w_s = tl.exp(m_s - m_final)
        if ADAPTIVE_SPLITS:
            active_split = m_s > neg_large
            w_s = tl.where(head_mask & active_split, w_s, 0.0)
        acc_base = (
            part_acc_ptr
            + query_idx * pa_stride0
            + s * pa_stride_s
            + head_offsets[:, None] * pa_stride_h
        )
        if ADAPTIVE_SPLITS:
            acc_s = tl.load(
                acc_base + comb_offsets[None, :],
                mask=head_mask[:, None] & active_split[:, None],
                other=0.0,
            )
        else:
            acc_s = tl.load(
                acc_base + comb_offsets[None, :],
                mask=head_mask[:, None],
                other=0.0,
            )
        acc += w_s[:, None] * acc_s

    out = tl.where(l_final[:, None] > 0.0, acc / denom[:, None], 0.0)

    out_row_ptr = (
        out_ptr + query_idx * out_stride0 + head_offsets[:, None] * out_stride1
    )
    tl.store(
        out_row_ptr + comb_offsets[None, :],
        out,
        mask=head_mask[:, None],
    )


def _rocm_sparse_attn_prefill_ragged_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    output: torch.Tensor | None = None,
    kv_scale: torch.Tensor | None = None,
    direct_kv: SparseDirectKv | None = None,
    partial_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None = None,
    *,
    block_h: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    assert q.ndim == 3, f"expected q=[sq,h,d], got {q.shape}"
    assert kv.ndim == 2, f"expected kv=[skv,d], got {kv.shape}"
    assert indices.ndim == 1, f"expected indices=[nnz], got {indices.shape}"
    assert indptr.ndim == 1, f"expected indptr=[sq+1], got {indptr.shape}"
    assert not q.is_cpu and not kv.is_cpu and not indices.is_cpu and not indptr.is_cpu

    kv_is_fp8 = kv.dtype == current_platform.fp8_dtype()
    if kv_is_fp8:
        assert kv_scale is not None, "FP8 sparse MLA requires a KV cache scale"
        kv = kv.view(torch.uint8)
        fp8_lut = get_e4m3fn_bf16_lut(q.device)
    elif direct_kv is not None:
        # The direct reader decodes packed rows itself, through the same table.
        fp8_lut = get_e4m3fn_bf16_lut(q.device)
    else:
        fp8_lut = torch.empty(1, dtype=torch.bfloat16, device=q.device)
        if kv_scale is None:
            kv_scale = torch.ones(1, dtype=torch.float32, device=q.device)

    # The direct reader dequantizes with a table that already has the block scale
    # baked in; the probe mode skips it (wrong values, same read shape).
    scaled_lut = (
        get_scaled_e4m3fn_bf16_lut(q.device)
        if direct_kv is not None and not envs.VLLM_DSV41_PREFILL_DIRECT_DECODE_PROBE
        else q
    )

    indices = _as_int32_contiguous_1d(indices)
    indptr = _as_int32_contiguous_1d(indptr)
    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.contiguous()

    num_queries, num_heads, head_dim = q.shape
    assert indptr.numel() == num_queries + 1, (
        f"expected indptr shape [{num_queries + 1}], got {indptr.shape}"
    )
    _validate_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "_rocm_sparse_attn_prefill_ragged_triton",
    )

    block_d = triton.next_power_of_2(head_dim)
    # CUDA sm_8x runs this kernel through the ROCm-derived path, so it never got
    # a tiling of its own. Interleaved A/B (round-robin, to cancel thermal
    # drift), under CUDA graph replay on CMP 170HX, sm_80, 70 SMs, at the
    # production shape (s_q=2048, topk=512, h_q=64, head_dim=512, bf16 KV),
    # with the device asserted to be a CMP 170HX:
    #     block_h=16 block_k=32 num_warps=4  (pre-change)  3428 us  40.1 TFLOPS
    #     block_h=32 block_k=64 num_warps=8 num_stages=2   2826 us  48.6 TFLOPS
    #     block_h=64 block_k=64 num_warps=8 num_stages=2   2269 us  60.6 TFLOPS
    # i.e. 1.51x over the pre-change tiling, max relative output deviation
    # 6.0e-3. (Note the pre-change default is block_k=32, taken from the envs.py
    # fallback; block_k=16 is only reachable by exporting the env var as 0, and
    # using that as a baseline overstates the gain.)
    #
    # The kernel is DRAM-bound: with one head-block per CTA the same KV
    # selection is re-read once per head-block, a measured 4.3x traffic
    # amplification. block_h=64 puts all 64 heads in one CTA, so each query
    # token's KV is read exactly once, and the software-pipelined gather plus
    # the wider BLOCK_K lift it toward the 1361 GB/s achievable peak. An earlier
    # revision stopped at block_h=32 because block_h=64 had only been tried
    # without pipelining, where it looks like a register-wall dead end.
    #
    # Only the bf16 KV path is tuned. The fp8 lookup-table path stages an extra
    # bf16 copy of the gathered tile and needs well over the available shared
    # memory at this width (Triton raises OutOfResources). Everything outside
    # the tuned case keeps its previous behaviour exactly, so ROCm
    # (gfx942/gfx950) and the fp8 path are untouched.
    #
    # Exporting VLLM_DEEPSEEK_V4_SPARSE_PREFILL_BLOCK_K opts out of the whole
    # sm_80 tuning -- not just BLOCK_K -- so the pre-tuning behaviour stays
    # reachable for A/B work and as an operator escape hatch. Note that the
    # cmp170hx PP3 launchers in benchmarks/deepseek_v4/ export it at 32, so they
    # opt out by default; the V4.1 PP5 launcher does not and gets the tuning.
    # Explicit caller-supplied kernel arguments always win as well.
    #
    # is_device_capability_family(80) is true for every 8.x part, but the tuning
    # is sized for a datacentre Ampere SM: the 170HX allows 166912 B of opt-in
    # shared memory while an sm_86 consumer part allows 101376 B, which the
    # tiling does not fit in. Gate on the actual limit so it can never be
    # selected where it cannot compile.
    tuning_opted_out = _PREFILL_BLOCK_K_ENV in os.environ or block_h is not None
    sm80_prefill = (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(80)
        and _prefill_smem_limit_bytes() >= _SM80_PREFILL_TUNED_SMEM_BYTES
        # Exactly the shape the tiling was measured on. BLOCK_D is
        # next_power_of_2(head_dim), so a 576-wide head (V3.2) would double the
        # gathered tile to BLOCK_D=1024 and blow the budget the check above
        # cleared -- the smem figure is only known for BLOCK_D=512.
        and head_dim == 512
        and not kv_is_fp8
        and not tuning_opted_out
    )
    if block_h is None:
        block_h = 64 if sm80_prefill else 16
    if num_warps is None:
        num_warps = 8 if sm80_prefill else 4
    if num_stages is None:
        # None -> tl.range keeps Triton's default, matching the pre-tuning kernel.
        num_stages = 2 if sm80_prefill else None
    if block_k is None:
        # The env var default (32) is the ROCm tuning; on sm_80 the wider
        # BLOCK_K is what lets the pipelined gather reach ~1100 GB/s.
        block_k = 64 if sm80_prefill else envs.VLLM_DEEPSEEK_V4_SPARSE_PREFILL_BLOCK_K
        if block_k not in (0, 16, 32, 64):
            raise ValueError(
                "VLLM_DEEPSEEK_V4_SPARSE_PREFILL_BLOCK_K must be 0, 16, 32, or 64"
            )
        if block_k == 0:
            block_k = 16 if head_dim >= 256 else 32
    if output is None:
        output = torch.empty_like(q, dtype=torch.bfloat16)
    else:
        assert output.shape == q.shape, (
            f"expected output shape {q.shape}, got {output.shape}"
        )
        assert output.dtype == torch.bfloat16 and output.device == q.device
        assert output.stride(-1) == 1
    if direct_kv is None:
        direct_cache, direct_bt, direct_lens = kv, indices, indptr
        direct_block_size = direct_block_stride = direct_max_blocks = 1
        direct_tag_base = direct_req_stride = 1
        direct_swa_cache, direct_swa_bt, direct_swa_lens = kv, indices, indptr
        direct_swa_block_size = direct_swa_block_stride = direct_swa_max_blocks = 1
        direct_swa_tag_base = 1
    else:
        # The direct reader unpacks the packed row itself and its fp8 decode is
        # the OCP one, so it cannot serve a platform whose caches are FNUZ.
        assert not kv_is_fp8, "direct KV reads never take an fp8 kv argument"
        assert not current_platform.is_fp8_fnuz(), (
            "direct KV reads support OCP fp8 caches only"
        )
        (
            direct_cache,
            direct_bt,
            direct_lens,
            direct_block_size,
            direct_tag_base,
            direct_req_stride,
            direct_swa,
        ) = direct_kv
        direct_block_stride = direct_cache.stride(0)
        direct_max_blocks = direct_bt.shape[1]
        (
            direct_swa_cache,
            direct_swa_bt,
            direct_swa_lens,
            direct_swa_block_size,
            direct_swa_tag_base,
        ) = direct_swa
        direct_swa_block_stride = direct_swa_cache.stride(0)
        direct_swa_max_blocks = direct_swa_bt.shape[1]
        # The assembled tile and the masked sliding-window tile both live in
        # shared memory, so the direct read runs without software pipelining and
        # with a narrower KV tile than the gathered path.
        num_stages = envs.VLLM_DSV41_PREFILL_DIRECT_STAGES
        block_k = min(block_k, envs.VLLM_DSV41_PREFILL_DIRECT_BLOCK_K)

    if partial_out is None:
        part_m = part_l = part_acc = output
        split_id = 0
        pm_stride0 = pm_stride_s = pa_stride0 = pa_stride_s = pa_stride_h = 1
    else:
        part_m, part_l, part_acc, split_id = partial_out
        pm_stride0, pm_stride_s = part_m.stride(0), part_m.stride(1)
        pa_stride0, pa_stride_s = part_acc.stride(0), part_acc.stride(1)
        pa_stride_h = part_acc.stride(2)

    _sparse_attn_prefill_ragged_kernel[(num_queries, triton.cdiv(num_heads, block_h))](
        q,
        kv,
        kv_scale,
        fp8_lut,
        indices,
        indptr,
        attn_sink,
        output,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        num_heads,
        head_dim,
        kv.shape[0],
        float(scale),
        direct_cache,
        direct_bt,
        direct_lens,
        direct_block_size,
        direct_block_stride,
        direct_max_blocks,
        direct_tag_base,
        direct_req_stride,
        direct_swa_cache,
        direct_swa_bt,
        direct_swa_lens,
        direct_swa_block_size,
        direct_swa_block_stride,
        direct_swa_max_blocks,
        direct_swa_tag_base,
        scaled_lut,
        part_m,
        part_l,
        part_acc,
        split_id,
        pm_stride0,
        pm_stride_s,
        pa_stride0,
        pa_stride_s,
        pa_stride_h,
        DIRECT_KV=direct_kv is not None,
        STORE_PARTIAL=partial_out is not None,
        DECODE_PROBE=bool(envs.VLLM_DSV41_PREFILL_DIRECT_DECODE_PROBE),
        KV_IS_FP8=kv_is_fp8,
        HAS_ATTN_SINK=has_attn_sink,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        NUM_STAGES=num_stages,
        num_warps=num_warps,
    )
    return output


def _rocm_sparse_attn_prefill_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    topk_length: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    kv_scale: torch.Tensor | None = None,
    direct_kv: SparseDirectKv | None = None,
    partial_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None = None,
    block_h: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
        indices,
        topk_length
        if topk_length is not None
        else (indices >= 0).sum(dim=-1, dtype=torch.int32),
        # Tagged direct-KV entries carry request/position rather than a row of the
        # gathered buffer, so the row-range clamp must not run: the kernel checks
        # each entry against the thing it actually addresses.
        num_rows=-1 if direct_kv is not None else kv.shape[0],
    )
    return _rocm_sparse_attn_prefill_ragged_triton(
        q=q,
        kv=kv,
        indices=ragged_indices,
        indptr=ragged_indptr,
        scale=scale,
        attn_sink=attn_sink,
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        output=output,
        kv_scale=kv_scale,
        direct_kv=direct_kv,
        partial_out=partial_out,
        block_h=block_h,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )


@functools.lru_cache
def _decode_cu_count() -> int:
    try:
        return torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:
        return 256  # For gfx950 arch, gated behind a fallback path for other archs.


def _decode_partial_iters(
    avg_main_len: float, avg_extra_len: float, splits: int, block_k: int
) -> int:
    """BLOCK_K iterations one partial workgroup walks for ``splits`` splits.

    Each split processes ``ceil(seg_len / splits)`` tokens of a segment, walked
    ``BLOCK_K`` at a time, and the main/extra segments are handled separately.
    """
    main_iters = (
        math.ceil(math.ceil(avg_main_len / splits) / block_k) if avg_main_len > 0 else 0
    )
    extra_iters = (
        math.ceil(math.ceil(avg_extra_len / splits) / block_k)
        if avg_extra_len > 0
        else 0
    )
    return main_iters + extra_iters


def _decode_num_splits(
    num_queries: int,
    heads_blocks: int,
    avg_main_len: float = 0.0,
    avg_extra_len: float = 0.0,
    block_k: int = 32,
) -> int:
    """Pick a flash-decode split count to keep the GPU busy across batch sizes.

    Decode launches only ``num_queries * heads_blocks`` workgroups otherwise,
    which severely under-fills the device for the low-concurrency regime that
    dominates latency. Splitting the KV sequence adds parallelism.

    We model the relative partial-kernel latency for a given split count ``s``
    as ``waves * (1/s + mu)`` where ``waves = ceil(base * s / CU)`` and ``mu``
    is a small per-wave overhead penalty:

      - ``waves / s`` captures the partial compute: each wave walks roughly
        ``total_tokens / s`` tokens and there are ``waves`` of them, so dividing
        by ``s`` makes more splits cheaper *until* they spill into extra waves.
      - ``mu * waves`` charges per-wave launch/tail overhead so we do not
        over-split into many mostly-idle waves (e.g. batch 224 on 256 CUs is
        best left at 1 split rather than 8 splits across 7 waves).

    The minimiser naturally prefers split counts that pack the device into full
    waves (``base * s`` near a multiple of ``CU``) and falls back to 1 split
    once the batch already fills the device. Ties favour the smaller split
    count (less reduce work).

    Finally we "snap down" the chosen split count to the smallest value that
    yields the same wave count *and* the same per-workgroup BLOCK_K iteration
    count. Because latency tracks iteration count (not raw token count), extra
    splits that do not lower the iteration count add only reduce/HBM overhead
    for no parallelism gain (e.g. batch 24: s8 and s10 both walk 4 extra iters
    in one wave, so s8 is strictly better). Snapping needs the average segment
    lengths, which the caller derives sync-free from the ragged index sizes.
    """
    base = max(1, num_queries * heads_blocks)
    # Target ~1 workgroup per CU: enough to fill the device while keeping the
    # reduce cost (which grows with split count) small. Tuned on gfx950.
    cu = max(1, _decode_cu_count())
    # Per-wave overhead penalty: higher values discourage split counts that
    # spill into extra GPU waves. Tuned on gfx950.
    mu = 0.04
    best_splits = 1
    best_cost = None
    # Search up to 16 splits; beyond that the reduce/HBM overhead dominates.
    for splits in range(1, 17):
        waves = (base * splits + cu - 1) // cu
        cost = waves * (1.0 / splits + mu)
        if best_cost is None or cost < best_cost - 1e-9:
            best_splits = splits
            best_cost = cost

    if best_splits > 1 and (avg_main_len > 0 or avg_extra_len > 0):
        target_waves = (base * best_splits + cu - 1) // cu
        target_iters = _decode_partial_iters(
            avg_main_len, avg_extra_len, best_splits, block_k
        )
        for splits in range(1, best_splits):
            waves = (base * splits + cu - 1) // cu
            iters = _decode_partial_iters(avg_main_len, avg_extra_len, splits, block_k)
            if waves == target_waves and iters == target_iters:
                best_splits = splits
                break
    return best_splits


def _decode_gfx950_num_splits(
    num_queries: int,
    heads_blocks: int,
    avg_main_len: float = 0.0,
    avg_extra_len: float = 0.0,
    block_k: int = 32,
) -> int:
    base = max(1, num_queries * heads_blocks)
    cu = max(1, _decode_cu_count())
    target_workgroups = 2 * cu
    num_splits = min(
        32,
        max(
            1,
            math.ceil(target_workgroups / base),
        ),
    )
    if (
        base >= 16
        and num_splits > 4
        and _decode_partial_iters(avg_main_len, avg_extra_len, 4, block_k) <= 3
    ):
        return 4
    if 16 <= base < 64:
        one_wave_splits = max(1, cu // base)
        one_wave_iters = _decode_partial_iters(
            avg_main_len, avg_extra_len, one_wave_splits, block_k
        )
        target_waves = 1 if one_wave_iters <= 9 else 2
        num_splits = min(num_splits, max(1, target_waves * cu // base))
    if base >= 16 and num_splits > 1:
        target_waves = (base * num_splits + cu - 1) // cu
        target_iters = _decode_partial_iters(
            avg_main_len, avg_extra_len, num_splits, block_k
        )
        for splits in range(1, num_splits):
            waves = (base * splits + cu - 1) // cu
            iters = _decode_partial_iters(avg_main_len, avg_extra_len, splits, block_k)
            if waves == target_waves and iters == target_iters:
                return splits
    return num_splits


def _rocm_sparse_attn_decode_ragged_triton(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    main_indptr: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_indptr: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    extra_cache_nan_free: bool = False,
    adaptive_splits: bool = False,
    main_block_table: torch.Tensor | None = None,
    main_seq_lens: torch.Tensor | None = None,
    main_window_size: int = 0,
) -> torch.Tensor:
    assert q.ndim == 3, f"expected q=[b,h,d], got {q.shape}"
    assert main_cache.ndim == 3, (
        f"expected main_cache=[blocks,block,bytes], got {main_cache.shape}"
    )
    assert main_indices.ndim == 1, (
        f"expected main_indices=[nnz], got {main_indices.shape}"
    )
    assert main_indptr.ndim == 1, f"expected main_indptr=[b+1], got {main_indptr.shape}"
    assert (
        not q.is_cpu
        and not main_cache.is_cpu
        and not main_indices.is_cpu
        and not main_indptr.is_cpu
    )

    main_indices = _as_int32_contiguous_1d(main_indices)
    main_indptr = _as_int32_contiguous_1d(main_indptr)
    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.contiguous()

    num_queries, num_heads, head_dim = q.shape
    use_paged_main = main_block_table is not None and main_seq_lens is not None
    if use_paged_main:
        assert main_block_table is not None
        assert main_seq_lens is not None
        assert main_block_table.ndim == 2
        assert main_seq_lens.numel() == num_queries
        assert main_window_size > 0
    else:
        assert main_indptr.numel() == num_queries + 1, (
            f"expected main_indptr shape [{num_queries + 1}], got {main_indptr.shape}"
        )
        main_block_table = main_indices
        main_seq_lens = main_indptr
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "_rocm_sparse_attn_decode_ragged_triton",
    )

    has_extra = (
        extra_cache is not None
        and extra_indices is not None
        and extra_indptr is not None
    )
    assert not extra_cache_nan_free or (_ON_GFX950 and has_extra), (
        "extra_cache_nan_free requires a gfx950 compressed cache with trusted "
        "canonical-writer provenance"
    )
    if has_extra:
        assert extra_cache is not None
        assert extra_indices is not None
        assert extra_indptr is not None
        assert extra_indices.ndim == 1, (
            f"expected extra_indices=[nnz], got {extra_indices.shape}"
        )
        assert extra_indptr.ndim == 1, (
            f"expected extra_indptr=[b+1], got {extra_indptr.shape}"
        )
        extra_indices = _as_int32_contiguous_1d(extra_indices)
        extra_indptr = _as_int32_contiguous_1d(extra_indptr)
        assert extra_indptr.numel() == num_queries + 1, (
            f"expected extra_indptr shape [{num_queries + 1}], got {extra_indptr.shape}"
        )
    else:
        extra_cache = main_cache
        extra_indices = torch.empty(0, device=q.device, dtype=torch.int32)
        extra_indptr = torch.zeros(num_queries + 1, device=q.device, dtype=torch.int32)

    block_h = 16
    if out is None:
        out = torch.empty_like(q, dtype=torch.bfloat16)
    else:
        assert out.shape == q.shape, f"expected out shape {q.shape}, got {out.shape}"
        assert out.device == q.device, (
            f"expected out on device {q.device}, got {out.device}"
        )
        assert out.dtype == torch.bfloat16, (
            f"expected out dtype {torch.bfloat16}, got {out.dtype}"
        )
    heads_blocks = triton.cdiv(num_heads, block_h)
    nope_block = triton.next_power_of_2(nope_head_dim)
    comb_dim = nope_head_dim + rope_head_dim
    is_fnuz = current_platform.is_fp8_fnuz()
    fp8_lut = get_e4m3fn_bf16_lut(q.device)

    use_sm80_split = (
        envs.VLLM_DEEPSEEK_V4_SPARSE_DECODE_SPLIT
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(80)
    )
    assert not use_paged_main or use_sm80_split, (
        "paged-direct SWA decode requires the SM80 split decode kernel"
    )
    if not (_ON_GFX942 or _ON_GFX950 or use_sm80_split):
        block_k = 16 if head_dim >= 256 else 32
        _sparse_attn_decode_ragged_kernel[(num_queries, heads_blocks)](
            q,
            main_cache,
            main_indices,
            main_indptr,
            extra_cache,
            extra_indices,
            extra_indptr,
            attn_sink,
            fp8_lut,
            out,
            q.stride(0),
            q.stride(1),
            out.stride(0),
            out.stride(1),
            main_cache.stride(0),
            extra_cache.stride(0),
            main_cache.shape[0] * main_cache.shape[1],
            extra_cache.shape[0] * extra_cache.shape[1],
            main_cache.shape[1],
            extra_cache.shape[1],
            scale,
            num_heads,
            HAS_ATTN_SINK=has_attn_sink,
            HAS_EXTRA=has_extra,
            NOPE_DIM=nope_head_dim,
            NOPE_BLOCK=nope_block,
            ROPE_DIM=rope_head_dim,
            IS_FNUZ_MAIN=is_fnuz,
            IS_FNUZ_EXTRA=False,
            BLOCK_H=block_h,
            BLOCK_K=block_k,
            num_warps=8,
        )
        return out

    block_k = 32  # KV tokens walked per split-K iteration. Tuned on gfx950.
    sm80_num_splits = (
        envs.VLLM_DEEPSEEK_V4_SM80_DECODE_NUM_SPLITS if use_sm80_split else 0
    )
    if not 0 <= sm80_num_splits <= 64:
        raise ValueError(
            "VLLM_DEEPSEEK_V4_SM80_DECODE_NUM_SPLITS must be between 0 and 64"
        )
    if sm80_num_splits:
        num_splits = sm80_num_splits
    elif _ON_GFX950:
        inv_q = 1.0 / max(1, num_queries)
        avg_main_len = (
            float(main_window_size) if use_paged_main else main_indices.numel() * inv_q
        )
        avg_extra_len = (extra_indices.numel() * inv_q) if has_extra else 0.0
        num_splits = _decode_gfx950_num_splits(
            num_queries,
            heads_blocks,
            avg_main_len,
            avg_extra_len,
            block_k,
        )
    else:
        # Average per-query segment lengths, read sync-free from the ragged
        # index sizes, let the split heuristic avoid over-splitting.
        inv_q = 1.0 / max(1, num_queries)
        avg_main_len = (
            float(main_window_size) if use_paged_main else main_indices.numel() * inv_q
        )
        avg_extra_len = (extra_indices.numel() * inv_q) if has_extra else 0.0
        num_splits = _decode_num_splits(
            num_queries, heads_blocks, avg_main_len, avg_extra_len, block_k
        )

    base_workgroups = num_queries * heads_blocks
    adaptive_splits = (
        _ON_GFX950 and adaptive_splits and base_workgroups >= 16 and num_splits > 4
    )
    one_wave_splits = (
        max(1, _decode_cu_count() // base_workgroups)
        if adaptive_splits and 16 <= base_workgroups < 64
        else num_splits
    )

    part_shapes = (
        ((num_queries, num_splits, num_heads), torch.float32),
        ((num_queries, num_splits, num_heads), torch.float32),
        ((num_queries, num_splits, num_heads, comb_dim), torch.float32),
    )
    if use_sm80_split:
        part_m, part_l, part_acc = current_workspace_manager().get_simultaneous(
            *part_shapes
        )
        if num_queries == 1:
            block_k = 16
            reduce_block_h, reduce_num_warps = 1, 8
        else:
            reduce_block_h = 2 if num_queries >= 8 else 1
            reduce_num_warps = 4
    else:
        part_m = torch.empty(part_shapes[0][0], dtype=torch.float32, device=q.device)
        part_l = torch.empty(part_shapes[1][0], dtype=torch.float32, device=q.device)
        part_acc = torch.empty(part_shapes[2][0], dtype=torch.float32, device=q.device)
        reduce_block_h, reduce_num_warps = 1, 4

    if _ON_GFX950:
        _sparse_attn_decode_gfx950_partial_kernel[
            (num_queries, num_splits, heads_blocks)
        ](
            q,
            main_cache,
            main_indices,
            main_indptr,
            extra_cache,
            extra_indices,
            extra_indptr,
            part_m,
            part_l,
            part_acc,
            q.stride(0),
            q.stride(1),
            main_cache.stride(0),
            extra_cache.stride(0),
            main_cache.shape[0] * main_cache.shape[1],
            extra_cache.shape[0] * extra_cache.shape[1],
            main_cache.shape[1],
            extra_cache.shape[1],
            scale,
            num_heads,
            HAS_EXTRA=has_extra,
            NOPE_DIM=nope_head_dim,
            ROPE_DIM=rope_head_dim,
            IS_FNUZ_MAIN=is_fnuz,
            IS_FNUZ_EXTRA=False,
            TRUST_EXTRA_CACHE_NAN_FREE=extra_cache_nan_free,
            ADAPTIVE_SPLITS=adaptive_splits,
            ONE_WAVE_SPLITS=one_wave_splits,
            BLOCK_H=block_h,
            BLOCK_K=block_k,
            NUM_SPLITS=num_splits,
            NUM_STAGES=1,
            num_warps=4,
            waves_per_eu=0,
        )
    else:
        num_warps = (
            8
            if use_sm80_split
            and num_queries == 6
            and num_heads == 64
            and num_splits == 3
            and has_extra
            and extra_indices.numel() == num_queries * 512
            else 4
        )
        _sparse_attn_decode_partial_kernel[(num_queries, num_splits, heads_blocks)](
            q,
            main_cache,
            main_indices,
            main_indptr,
            extra_cache,
            extra_indices,
            extra_indptr,
            main_block_table,
            main_seq_lens,
            part_m,
            part_l,
            part_acc,
            fp8_lut,
            q.stride(0),
            q.stride(1),
            main_cache.stride(0),
            extra_cache.stride(0),
            main_block_table.stride(0),
            part_m.stride(0),
            part_m.stride(1),
            part_acc.stride(0),
            part_acc.stride(1),
            part_acc.stride(2),
            main_cache.shape[0] * main_cache.shape[1],
            extra_cache.shape[0] * extra_cache.shape[1],
            main_cache.shape[1],
            extra_cache.shape[1],
            main_window_size,
            scale,
            num_heads,
            HAS_EXTRA=has_extra,
            HAS_PAGED_MAIN=use_paged_main,
            NOPE_DIM=nope_head_dim,
            NOPE_BLOCK=nope_block,
            ROPE_DIM=rope_head_dim,
            # main_cache = swa_k_cache (C++ encoder, FNUZ on gfx942 / OCP on gfx950).
            # extra_cache = compressed kv_cache (Triton encoder, OCP everywhere).
            # Reading both with a single IS_FNUZ would decode one of them with the
            # wrong FNUZ/OCP scale ratio (~1.87×).
            IS_FNUZ_MAIN=is_fnuz,
            IS_FNUZ_EXTRA=False,
            BLOCK_H=block_h,
            BLOCK_K=block_k,
            NUM_SPLITS=num_splits,
            NUM_STAGES=1,
            num_warps=num_warps,
        )

    reduce_heads_blocks = triton.cdiv(num_heads, reduce_block_h)
    _sparse_attn_decode_reduce_kernel[(num_queries, reduce_heads_blocks)](
        part_m,
        part_l,
        part_acc,
        attn_sink,
        out,
        out.stride(0),
        out.stride(1),
        part_m.stride(0),
        part_m.stride(1),
        part_acc.stride(0),
        part_acc.stride(1),
        part_acc.stride(2),
        num_heads,
        HAS_ATTN_SINK=has_attn_sink,
        ADAPTIVE_SPLITS=adaptive_splits,
        COMB_DIM=comb_dim,
        BLOCK_H=reduce_block_h,
        NUM_SPLITS=num_splits,
        SPLITS_PAD=triton.next_power_of_2(num_splits),
        num_warps=reduce_num_warps,
    )
    return out


def _rocm_sparse_attn_decode_triton(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    main_lengths: torch.Tensor | None = None,
    extra_lengths: torch.Tensor | None = None,
    main_ragged_indices: torch.Tensor | None = None,
    main_ragged_indptr: torch.Tensor | None = None,
    extra_ragged_indices: torch.Tensor | None = None,
    extra_ragged_indptr: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    extra_cache_nan_free: bool = False,
    adaptive_splits: bool = False,
    main_block_table: torch.Tensor | None = None,
    main_seq_lens: torch.Tensor | None = None,
    main_window_size: int = 0,
) -> torch.Tensor:
    use_paged_main = main_block_table is not None and main_seq_lens is not None
    if not use_paged_main and (
        main_ragged_indices is None or main_ragged_indptr is None
    ):
        main_ragged_indices, main_ragged_indptr = build_ragged_indices_from_dense(
            main_indices,
            main_lengths
            if main_lengths is not None
            else (main_indices >= 0).sum(dim=-1, dtype=torch.int32),
            num_rows=main_cache.shape[0] * main_cache.shape[1],
        )
    elif use_paged_main:
        main_ragged_indices = main_indices.reshape(-1)
        main_ragged_indptr = main_lengths

    assert main_ragged_indices is not None
    assert main_ragged_indptr is not None

    if (
        (extra_ragged_indices is None or extra_ragged_indptr is None)
        and extra_cache is not None
        and extra_indices is not None
    ):
        extra_ragged_indices, extra_ragged_indptr = build_ragged_indices_from_dense(
            extra_indices,
            extra_lengths
            if extra_lengths is not None
            else (extra_indices >= 0).sum(dim=-1, dtype=torch.int32),
            num_rows=extra_cache.shape[0] * extra_cache.shape[1],
        )

    return _rocm_sparse_attn_decode_ragged_triton(
        q=q,
        main_cache=main_cache,
        main_indices=main_ragged_indices,
        main_indptr=main_ragged_indptr,
        scale=scale,
        attn_sink=attn_sink,
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        extra_cache=extra_cache,
        extra_indices=extra_ragged_indices,
        extra_indptr=extra_ragged_indptr,
        out=out,
        extra_cache_nan_free=extra_cache_nan_free,
        adaptive_splits=adaptive_splits,
        main_block_table=main_block_table,
        main_seq_lens=main_seq_lens,
        main_window_size=main_window_size,
    )


def rocm_sparse_attn_reduce_partials(
    part_m: torch.Tensor,
    part_l: torch.Tensor,
    part_acc: torch.Tensor,
    attn_sink: torch.Tensor | None,
    output: torch.Tensor,
    num_heads: int,
    comb_dim: int,
    adaptive_splits: bool = True,
) -> None:
    """Combine per-split ``(m, l, acc)`` states into the attention output.

    Split-K prefill stores raw online-softmax state per support region instead of
    a normalized row; this applies the attention sink, re-weights every split by
    its own running max and normalizes once. Splits a query never touched carry
    ``m == -inf`` and are dropped when ``adaptive_splits`` is set.
    """
    num_splits = part_m.shape[1]
    assert part_m.shape[:2] == part_l.shape[:2] == part_acc.shape[:2], (
        f"partial shapes disagree: {part_m.shape}, {part_l.shape}, {part_acc.shape}"
    )
    reduce_block_h = 2 if output.shape[0] >= 8 else 1
    if attn_sink is None:
        attn_sink = torch.empty(1, dtype=torch.float32, device=output.device)
    else:
        attn_sink = attn_sink.contiguous()
    _sparse_attn_decode_reduce_kernel[
        (output.shape[0], triton.cdiv(num_heads, reduce_block_h))
    ](
        part_m,
        part_l,
        part_acc,
        attn_sink,
        output,
        output.stride(0),
        output.stride(1),
        part_m.stride(0),
        part_m.stride(1),
        part_acc.stride(0),
        part_acc.stride(1),
        part_acc.stride(2),
        num_heads,
        HAS_ATTN_SINK=attn_sink.numel() > 1,
        ADAPTIVE_SPLITS=adaptive_splits,
        COMB_DIM=comb_dim,
        BLOCK_H=reduce_block_h,
        NUM_SPLITS=num_splits,
        SPLITS_PAD=triton.next_power_of_2(num_splits),
        num_warps=4,
    )


def rocm_sparse_attn_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor | None,
    topk_length: torch.Tensor | None,
    scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    attn_sink: torch.Tensor | None,
    output: torch.Tensor,
    ragged_indices: torch.Tensor | None = None,
    ragged_indptr: torch.Tensor | None = None,
    kv_scale: torch.Tensor | None = None,
    direct_kv: SparseDirectKv | None = None,
    partial_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None = None,
    block_h: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> None:
    assert kv.ndim == 3 and kv.shape[1] == 1, (
        f"ROCm Triton sparse prefill expects kv=[skv,1,d], got {kv.shape}"
    )
    _validate_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "rocm_sparse_attn_prefill",
    )
    caller_output = (
        output if envs.VLLM_DEEPSEEK_V4_SPARSE_PREFILL_CALLER_OUTPUT else None
    )
    if ragged_indices is not None and ragged_indptr is not None:
        output_chunk = _rocm_sparse_attn_prefill_ragged_triton(
            q=q,
            kv=kv.squeeze(1),
            indices=ragged_indices,
            indptr=ragged_indptr,
            scale=scale,
            attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
            nope_head_dim=nope_head_dim,
            rope_head_dim=rope_head_dim,
            output=caller_output,
            kv_scale=kv_scale,
            direct_kv=direct_kv,
            partial_out=partial_out,
            block_h=block_h,
            block_k=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        assert indices is not None
        indices_2d = indices.reshape(indices.shape[0], -1)
        output_chunk = _rocm_sparse_attn_prefill_triton(
            q=q,
            kv=kv.squeeze(1),
            indices=indices_2d,
            scale=scale,
            attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
            nope_head_dim=nope_head_dim,
            rope_head_dim=rope_head_dim,
            topk_length=topk_length,
            output=caller_output,
            kv_scale=kv_scale,
            direct_kv=direct_kv,
            partial_out=partial_out,
            block_h=block_h,
            block_k=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    if caller_output is None:
        output.copy_(output_chunk)


def rocm_sparse_attn_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    swa_k_cache: torch.Tensor,
    swa_only: bool,
    topk_indices: torch.Tensor | None,
    topk_lens: torch.Tensor | None,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_ragged_indices: torch.Tensor | None,
    swa_ragged_indptr: torch.Tensor | None,
    topk_ragged_indices: torch.Tensor | None,
    topk_ragged_indptr: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    output: torch.Tensor,
    extra_cache_nan_free: bool = False,
    adaptive_splits: bool = False,
    swa_block_table: torch.Tensor | None = None,
    swa_seq_lens: torch.Tensor | None = None,
    swa_window_size: int = 0,
    swa_paged_direct: bool = False,
) -> None:
    assert swa_k_cache.dtype == torch.uint8, (
        "ROCm Triton sparse decode expects uint8 fp8_ds_mla SWA cache, "
        f"got {swa_k_cache.dtype}"
    )
    _validate_dsv4_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "rocm_sparse_attn_decode",
    )

    main_indices = swa_indices.reshape(swa_indices.shape[0], -1)

    extra_cache = None
    extra_indices = None
    if not swa_only:
        assert kv_cache is not None
        assert topk_indices is not None or (
            topk_ragged_indices is not None and topk_ragged_indptr is not None
        )
        assert kv_cache.dtype == torch.uint8, (
            "ROCm Triton sparse decode expects uint8 fp8_ds_mla extra cache, "
            f"got {kv_cache.dtype}"
        )
        extra_cache = kv_cache
        if topk_indices is not None:
            extra_indices = topk_indices.reshape(topk_indices.shape[0], -1)

    use_sm80_caller_output = (
        envs.VLLM_DEEPSEEK_V4_SPARSE_DECODE_CALLER_OUTPUT
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(80)
    )
    direct_out = (
        output
        if output.dtype == torch.bfloat16 and (_ON_GFX950 or use_sm80_caller_output)
        else None
    )
    attn_out = _rocm_sparse_attn_decode_triton(
        q=q,
        main_cache=swa_k_cache,
        main_indices=main_indices,
        scale=scale,
        attn_sink=None if attn_sink is None else attn_sink[: q.shape[1]],
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        main_lengths=swa_lens,
        extra_lengths=topk_lens,
        main_ragged_indices=swa_ragged_indices,
        main_ragged_indptr=swa_ragged_indptr,
        extra_ragged_indices=topk_ragged_indices,
        extra_ragged_indptr=topk_ragged_indptr,
        out=direct_out,
        extra_cache_nan_free=extra_cache_nan_free,
        adaptive_splits=adaptive_splits,
        main_block_table=swa_block_table if swa_paged_direct else None,
        main_seq_lens=swa_seq_lens if swa_paged_direct else None,
        main_window_size=swa_window_size if swa_paged_direct else 0,
    )
    if direct_out is None:
        output.copy_(attn_out.to(output.dtype))
