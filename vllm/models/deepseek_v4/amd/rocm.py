# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from dataclasses import dataclass
from typing import cast

import torch

from vllm import envs
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.deepseek_v4.attention import (
    DeepseekV4Attention,
    get_sparse_prefill_reqs_per_chunk,
)
from vllm.models.deepseek_v4.common.ops import dequantize_and_gather_k_cache
from vllm.models.deepseek_v4.phase_profile import phase_profile_stage
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLAMetadata,
    DeepseekV4SparseMLABackend,
    DeepseekV4SparseMLAMetadataBuilder,
    build_c128a_topk_metadata,
)
from vllm.platforms import current_platform
from vllm.platforms.rocm import _ON_GFX950
from vllm.profiler.phase_profile_hook import optional_phase_profile_stage
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backend import (
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWAMetadata,
    DeepseekSparseSWAMetadataBuilder,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    build_ragged_indices_from_dense,
    rocm_inv_rope_einsum,
    rocm_sparse_attn_decode,
    rocm_sparse_attn_prefill,
)
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)


def _sparse_prefill_reqs_per_chunk(max_reqs: int, M: int, head_dim: int) -> int:
    return get_sparse_prefill_reqs_per_chunk(
        max_reqs=max_reqs,
        num_kv_tokens=M,
        head_dim=head_dim,
        budget_mb=envs.VLLM_DEEPSEEK_V4_SPARSE_PREFILL_TEMP_MB,
    )


def _trust_dsv4_extra_cache_nan_free(
    kv_cache_dtype: str,
    has_kv_transfer: bool,
    has_extra_cache: bool,
) -> bool:
    return (
        _ON_GFX950
        and kv_cache_dtype == "fp8_ds_mla"
        and not has_kv_transfer
        and has_extra_cache
    )


def _build_indptr_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    lengths = lengths.to(dtype=torch.int32).contiguous()
    indptr = torch.zeros(lengths.shape[0] + 1, dtype=torch.int32, device=lengths.device)
    torch.cumsum(lengths, dim=0, out=indptr[1:])
    return indptr


def apply_pre_quantized_block_scaled_mm(
    linear: torch.nn.Module,
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
) -> torch.Tensor:
    """Block-scaled fp8 GEMM on pre-quantized activations.

    The fused q/kv norm kernel writes fp8 qr + per-1x128 scales; this
    drives the linear's block-scaled GEMM directly with them, bypassing
    apply_weights which would re-quantize the fp8 input. Only valid for
    the wq_b-style column/replicated linears: their output is the local
    TP shard, so no all-reduce is needed.
    """
    from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
        FP8BlockParams,
    )

    params = FP8BlockParams.from_layer(linear)
    weight_scale = (
        params.weight_scale
        if params.weight_scale_inv is None
        else params.weight_scale_inv
    )
    kernel = linear.quant_method.fp8_linear
    out = kernel.apply_block_scaled_mm(
        A=x_fp8, B=params.weight, As=x_scale, Bs=weight_scale
    )
    return out.to(dtype=kernel.config.out_dtype)


# ROCm sparse prefill keeps this dense combine local so AMD-specific SWA changes
# do not touch the shared DeepSeek V4 cache utilities.
_SPARSE_PREFILL_TOPK_ALIGNMENT = 128


@triton.jit
def _combine_topk_swa_indices_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    M,
    N,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    TOPK_WIDTH: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    base = tl.load(query_start_loc_ptr)
    query_start = tl.load(query_start_loc_ptr + batch_idx) - base
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1) - base
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    gather_len = tl.load(gather_lens_ptr + batch_idx)
    start_pos = seq_len - query_len
    gather_start = seq_len - gather_len

    for token_idx in range(query_start + worker_id, query_end, num_workers):
        token_idx_in_query = token_idx - query_start
        pos = start_pos + token_idx_in_query
        topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K)
        swa_len = tl.minimum(pos + 1, WINDOW_SIZE)

        topk_offset = tl.arange(0, PADDED_TOP_K)
        topk_mask = topk_offset < topk_len
        safe_topk_offset = tl.where(topk_offset < TOPK_WIDTH, topk_offset, 0)
        topk_indices = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + safe_topk_offset,
            mask=topk_mask,
            other=-1,
        )
        valid_topk = (topk_indices >= 0) & (topk_indices < N)
        topk_indices = tl.where(valid_topk, topk_indices + M * batch_idx, -1)
        tl.store(
            combined_indices_ptr + token_idx * combined_indices_stride + topk_offset,
            topk_indices,
            mask=topk_mask,
        )

        swa_offset = tl.arange(0, WINDOW_SIZE)
        tl.store(
            combined_indices_ptr
            + token_idx * combined_indices_stride
            + topk_len
            + swa_offset,
            M * batch_idx + N + swa_offset + pos - swa_len + 1 - gather_start,
            mask=swa_offset < swa_len,
        )

        tl.store(combined_lens_ptr + token_idx, topk_len + swa_len)


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    if out is None:
        combined_indices = torch.full(
            (num_tokens, combined_topk),
            fill_value=-1,
            dtype=torch.int32,
            device=topk_indices.device,
        )
        combined_lens = torch.empty(
            num_tokens, dtype=torch.int32, device=topk_indices.device
        )
    else:
        combined_indices, combined_lens = out

    num_workers = 128
    _combine_topk_swa_indices_kernel[(num_reqs, num_workers)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        topk_indices,
        topk_indices.stride(0),
        query_start_loc,
        seq_lens,
        gather_lens,
        M,
        N,
        TOP_K=topk,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=window_size,
        TOPK_WIDTH=topk_indices.shape[-1],
        PADDED_TOP_K=triton.next_power_of_2(topk_indices.shape[-1]),
    )
    return combined_indices, combined_lens


@triton.jit
def _compute_topk_lens_kernel(
    topk_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    topk,
    is_valid_token_ptr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    is_valid_token = tl.load(is_valid_token_ptr + token_idx)

    count = tl.zeros((), dtype=tl.int32)
    for i in range(0, topk, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < topk
        local_idx = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
            other=-1,
        )
        count += tl.sum((local_idx >= 0).to(tl.int32), axis=0)

    tl.store(topk_lens_ptr + token_idx, tl.where(is_valid_token, count, 0))


@triton.jit
def _pack_global_topk_ragged_kernel(
    global_topk_ragged_ptr,
    topk_indptr_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    topk,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    out_start = tl.load(topk_indptr_ptr + token_idx)
    out_end = tl.load(topk_indptr_ptr + token_idx + 1)
    out_len = out_end - out_start
    if block_idx * BLOCK_SIZE >= out_len:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    mask = (offset < out_len) & (offset < topk)
    local_idx = tl.load(
        topk_indices_ptr + token_idx * topk_indices_stride + offset,
        mask=mask,
        other=-1,
    )
    valid = mask & (local_idx >= 0)
    block_indices = local_idx // block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=valid,
        other=0,
    )
    block_offsets = local_idx % block_size
    slot_ids = tl.where(valid, block_numbers * block_size + block_offsets, -1)
    tl.store(global_topk_ragged_ptr + out_start + offset, slot_ids, mask=mask)


def compute_global_topk_ragged_indices_and_indptr(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    topk = topk_indices.shape[1]

    topk_lens = torch.empty(num_tokens, dtype=torch.int32, device=topk_indices.device)
    _compute_topk_lens_kernel[(num_tokens,)](
        topk_lens,
        topk_indices,
        topk_indices.stride(0),
        topk,
        is_valid_token,
        TRITON_BLOCK_SIZE=1024,
    )

    topk_indptr = _build_indptr_from_lengths(topk_lens)
    global_topk_ragged = torch.empty(
        num_tokens * topk,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    if global_topk_ragged.numel() > 0:
        block = 128
        _pack_global_topk_ragged_kernel[(num_tokens, triton.cdiv(topk, block))](
            global_topk_ragged,
            topk_indptr,
            topk_indices,
            topk_indices.stride(0),
            token_to_req_indices,
            block_table,
            block_table.stride(0),
            block_size,
            topk,
            BLOCK_SIZE=block,
        )
    return global_topk_ragged, topk_indptr, topk_lens


@triton.jit
def _build_c128_decode_ragged_indptr_kernel(
    decode_lens_ptr,
    ragged_indptr_ptr,
    positions_ptr,
    slot_mapping_ptr,
    compress_ratio,
    max_compressed_tokens,
    NUM_TOKENS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    token_idx = tl.arange(0, BLOCK_ROWS)
    in_bounds = token_idx < NUM_TOKENS
    position = tl.load(positions_ptr + token_idx, mask=in_bounds, other=0)
    is_valid = tl.load(slot_mapping_ptr + token_idx, mask=in_bounds, other=-1) >= 0
    num_compressed = tl.minimum(
        (position + 1) // compress_ratio,
        max_compressed_tokens,
    )
    decode_len = tl.where(in_bounds & is_valid, num_compressed, 0).to(tl.int32)
    tl.store(decode_lens_ptr + token_idx, decode_len, mask=in_bounds)
    inclusive = tl.cumsum(decode_len, axis=0)
    tl.store(ragged_indptr_ptr, 0)
    tl.store(ragged_indptr_ptr + token_idx + 1, inclusive, mask=in_bounds)


@triton.jit
def _build_c128_decode_ragged_indices_kernel(
    ragged_indices_ptr,
    ragged_indptr_ptr,
    decode_lens_ptr,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    ACTIVE_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    decode_len = tl.load(decode_lens_ptr + token_idx)
    if block_idx * BLOCK_SIZE >= decode_len:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    out_start = tl.load(ragged_indptr_ptr + token_idx)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < decode_len) & (offsets < ACTIVE_WIDTH)
    block_indices = offsets // block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=mask,
        other=-1,
    )
    slot_ids = block_numbers * block_size + offsets % block_size
    tl.store(ragged_indices_ptr + out_start + offsets, slot_ids, mask=mask)


def build_c128_decode_ragged_direct(
    *,
    positions: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    slot_mapping: torch.Tensor,
    compress_ratio: int,
    active_topk_width: int,
    max_topk_width: int,
    decode_lens_buffer: torch.Tensor,
    ragged_indices_buffer: torch.Tensor,
    ragged_indptr_buffer: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build C128 decode metadata directly in graph-stable ragged buffers."""
    num_tokens = positions.shape[0]
    assert num_tokens > 0
    assert active_topk_width <= max_topk_width
    assert decode_lens_buffer.numel() >= num_tokens
    assert ragged_indptr_buffer.numel() >= num_tokens + 1
    assert ragged_indices_buffer.numel() >= num_tokens * max_topk_width

    decode_lens = decode_lens_buffer[:num_tokens]
    ragged_indptr = ragged_indptr_buffer[: num_tokens + 1]
    block_rows = triton.next_power_of_2(num_tokens)
    _build_c128_decode_ragged_indptr_kernel[(1,)](
        decode_lens,
        ragged_indptr,
        positions,
        slot_mapping,
        compress_ratio,
        active_topk_width,
        NUM_TOKENS=num_tokens,
        BLOCK_ROWS=block_rows,
    )

    max_entries = max(num_tokens * max_topk_width, 1)
    ragged_indices = ragged_indices_buffer[:max_entries]
    block = 256
    _build_c128_decode_ragged_indices_kernel[
        (num_tokens, triton.cdiv(active_topk_width, block))
    ](
        ragged_indices,
        ragged_indptr,
        decode_lens,
        token_to_req_indices,
        block_table,
        block_table.stride(0),
        block_size,
        ACTIVE_WIDTH=active_topk_width,
        BLOCK_SIZE=block,
    )
    return ragged_indices, ragged_indptr, decode_lens


def _copy_ragged_to_graph_buffers(
    ragged_indices: torch.Tensor,
    ragged_indptr: torch.Tensor,
    ragged_indices_buffer: torch.Tensor,
    ragged_indptr_buffer: torch.Tensor,
    num_rows: int,
    max_entries_per_row: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy dynamic ragged metadata into persistent CUDA graph buffers.

    FULL decode graphs capture kernel argument addresses. Keep the returned
    tensors backed by stable storage, while indptr continues to bound reads.
    """
    indptr_out = ragged_indptr_buffer[: num_rows + 1]
    indptr_out.copy_(ragged_indptr, non_blocking=True)

    max_entries = max(num_rows * max_entries_per_row, 1)
    ragged_out = ragged_indices_buffer[:max_entries]
    source_entries = ragged_indices.numel()
    if source_entries > 0:
        ragged_out[:source_entries].copy_(ragged_indices, non_blocking=True)
    if _ON_GFX950:
        # Preserve the graph-stable base pointer while exposing source capacity
        # to the sync-free split selector; indptr still carries the true NNZ.
        ragged_out = ragged_out[: max(source_entries, 1)]
    return ragged_out, indptr_out


@triton.jit
def _build_swa_ragged_indptr_kernel(
    swa_lens_ptr,
    ragged_indptr_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    token_to_req_indices_ptr,
    is_valid_token_ptr,
    window_size,
    NUM_TOKENS: tl.constexpr,
    NON_CAUSAL: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    token_idx = tl.arange(0, BLOCK_ROWS)
    in_bounds = token_idx < NUM_TOKENS
    is_valid = tl.load(is_valid_token_ptr + token_idx, mask=in_bounds, other=0)
    req_idx = tl.load(
        token_to_req_indices_ptr + token_idx,
        mask=in_bounds & is_valid,
        other=0,
    )
    query_start = tl.load(
        query_start_loc_ptr + req_idx,
        mask=in_bounds & is_valid,
        other=0,
    )
    query_end = tl.load(
        query_start_loc_ptr + req_idx + 1,
        mask=in_bounds & is_valid,
        other=0,
    )
    seq_len = tl.load(
        seq_lens_ptr + req_idx,
        mask=in_bounds & is_valid,
        other=0,
    )
    query_len = query_end - query_start
    prefix_len = seq_len - query_len
    if NON_CAUSAL:
        start_pos = tl.maximum(prefix_len - window_size, 0)
        end_pos = seq_len
    else:
        pos = prefix_len + token_idx - query_start
        start_pos = tl.maximum(pos - window_size + 1, 0)
        end_pos = pos + 1
    swa_len = tl.where(in_bounds & is_valid, end_pos - start_pos, 0).to(tl.int32)
    tl.store(swa_lens_ptr + token_idx, swa_len, mask=in_bounds)
    inclusive = tl.cumsum(swa_len, axis=0)
    tl.store(ragged_indptr_ptr, 0)
    tl.store(ragged_indptr_ptr + token_idx + 1, inclusive, mask=in_bounds)


@triton.jit
def _build_swa_ragged_indices_kernel(
    ragged_indices_ptr,
    ragged_indptr_ptr,
    swa_lens_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    token_to_req_indices_ptr,
    is_valid_token_ptr,
    block_table_ptr,
    block_table_stride,
    window_size,
    block_size,
    INDEX_WIDTH: tl.constexpr,
    NON_CAUSAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    is_valid = tl.load(is_valid_token_ptr + token_idx)
    if not is_valid:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + req_idx)
    prefix_len = seq_len - query_len
    if NON_CAUSAL:
        start_pos = tl.maximum(prefix_len - window_size, 0)
    else:
        pos = prefix_len + token_idx - query_start
        start_pos = tl.maximum(pos - window_size + 1, 0)

    swa_len = tl.load(swa_lens_ptr + token_idx)
    out_start = tl.load(ragged_indptr_ptr + token_idx)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < swa_len) & (offsets < INDEX_WIDTH)
    pos_offset = start_pos + offsets
    block_indices = pos_offset // block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=mask,
        other=-1,
    )
    slot_ids = block_numbers * block_size + pos_offset % block_size
    tl.store(ragged_indices_ptr + out_start + offsets, slot_ids, mask=mask)


def build_swa_ragged_indices_direct(
    *,
    num_decode_tokens: int,
    window_size: int,
    index_width: int,
    non_causal: bool,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    is_valid_token: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    swa_lens: torch.Tensor,
    ragged_indices_buffer: torch.Tensor,
    ragged_indptr_buffer: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build decode SWA ragged metadata directly in graph-stable buffers."""
    assert 0 < num_decode_tokens <= swa_lens.numel()
    assert ragged_indptr_buffer.numel() >= num_decode_tokens + 1
    assert ragged_indices_buffer.numel() >= num_decode_tokens * index_width

    ragged_indptr = ragged_indptr_buffer[: num_decode_tokens + 1]
    block_rows = triton.next_power_of_2(num_decode_tokens)
    _build_swa_ragged_indptr_kernel[(1,)](
        swa_lens,
        ragged_indptr,
        query_start_loc,
        seq_lens,
        token_to_req_indices,
        is_valid_token,
        window_size,
        NUM_TOKENS=num_decode_tokens,
        NON_CAUSAL=non_causal,
        BLOCK_ROWS=block_rows,
    )

    max_entries = max(num_decode_tokens * index_width, 1)
    ragged_indices = ragged_indices_buffer[:max_entries]
    block = 256
    _build_swa_ragged_indices_kernel[
        (num_decode_tokens, triton.cdiv(index_width, block))
    ](
        ragged_indices,
        ragged_indptr,
        swa_lens,
        query_start_loc,
        seq_lens,
        token_to_req_indices,
        is_valid_token,
        block_table,
        block_table.stride(0),
        window_size,
        block_size,
        INDEX_WIDTH=index_width,
        NON_CAUSAL=non_causal,
        BLOCK_SIZE=block,
    )
    return ragged_indices, ragged_indptr


@dataclass
class DeepseekV4ROCMAiterMLASparseMetadata(DeepseekV4FlashMLAMetadata):
    """ROCm-specific DeepSeek V4 metadata carrying ragged decode topk."""

    c128a_decode_topk_ragged_indices: torch.Tensor | None = None
    c128a_decode_topk_ragged_indptr: torch.Tensor | None = None
    for_cudagraph_capture: bool = False


@dataclass
class DeepseekV4ROCMAiterSparseSWAMetadata(DeepseekSparseSWAMetadata):
    decode_swa_ragged_indices: torch.Tensor | None = None
    decode_swa_ragged_indptr: torch.Tensor | None = None


class DeepseekV4ROCMAiterMLASparseMetadataBuilder(DeepseekV4SparseMLAMetadataBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_direct_ragged_c128 = not self._requires_dense_c128a_decode_indices()
        self.c128a_decode_topk_ragged_indices_buffer: torch.Tensor | None = None
        self.c128a_decode_topk_ragged_indptr_buffer: torch.Tensor | None = None
        self._direct_c128a_decode_ragged: tuple[torch.Tensor, torch.Tensor] | None = (
            None
        )
        if self.compress_ratio == 128:
            max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
            self.c128a_decode_topk_ragged_indices_buffer = torch.empty(
                max_tokens * self.c128a_max_compressed,
                dtype=torch.int32,
                device=self.device,
            )
            self.c128a_decode_topk_ragged_indptr_buffer = torch.empty(
                max_tokens + 1,
                dtype=torch.int32,
                device=self.device,
            )

    def _build_c128a_metadata(
        self,
        cm: CommonAttentionMetadata,
        req_id_per_token: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        if not self._use_direct_ragged_c128:
            return super()._build_c128a_metadata(cm, req_id_per_token)

        (num_decodes, _, num_decode_tokens, num_prefill_tokens) = (
            split_decodes_and_prefills(
                cm,
                decode_threshold=self.reorder_batch_threshold or 1,
            )
        )
        num_total = num_decode_tokens + num_prefill_tokens
        if num_total == 0:
            return {}

        assert cm.positions is not None
        active_topk_width = self._get_c128a_active_topk_width(cm.max_seq_len)
        block_size = self.kv_cache_spec.block_size // self.compress_ratio
        result: dict[str, torch.Tensor | None] = {}

        if num_decode_tokens > 0:
            assert self.c128a_decode_topk_ragged_indices_buffer is not None
            assert self.c128a_decode_topk_ragged_indptr_buffer is not None
            with optional_phase_profile_stage(
                "target.attn_metadata.mla.c128.direct_ragged",
                cm.slot_mapping,
            ):
                ragged_indices, ragged_indptr, decode_lens = (
                    build_c128_decode_ragged_direct(
                        positions=cm.positions[:num_decode_tokens],
                        token_to_req_indices=req_id_per_token[:num_decode_tokens],
                        block_table=cm.block_table_tensor[:num_decodes],
                        block_size=block_size,
                        slot_mapping=cm.slot_mapping[:num_decode_tokens],
                        compress_ratio=self.compress_ratio,
                        active_topk_width=active_topk_width,
                        max_topk_width=self.c128a_max_compressed,
                        decode_lens_buffer=self.c128a_decode_lens_buffer,
                        ragged_indices_buffer=(
                            self.c128a_decode_topk_ragged_indices_buffer
                        ),
                        ragged_indptr_buffer=(
                            self.c128a_decode_topk_ragged_indptr_buffer
                        ),
                    )
                )
            self._direct_c128a_decode_ragged = ragged_indices, ragged_indptr
            result["c128a_decode_topk_lens"] = decode_lens

        if num_prefill_tokens > 0:
            _, _, prefill_local = build_c128a_topk_metadata(
                cm.positions[num_decode_tokens:num_total],
                self.compress_ratio,
                0,
                req_id_per_token[num_decode_tokens:num_total],
                cm.block_table_tensor,
                block_size,
                cm.slot_mapping[num_decode_tokens:num_total],
                self.c128a_global_decode_buffer,
                self.c128a_decode_lens_buffer,
                self.c128a_prefill_buffer,
                max_compressed_tokens=active_topk_width,
            )
            result["c128a_prefill_topk_indices"] = prefill_local
        return result

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4ROCMAiterMLASparseMetadata:
        self._direct_c128a_decode_ragged = None
        base = super().build(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )

        ragged_indices = None
        ragged_indptr = None
        dense_decode = base.c128a_global_decode_topk_indices
        decode_lens = base.c128a_decode_topk_lens
        if self._direct_c128a_decode_ragged is not None:
            ragged_indices, ragged_indptr = self._direct_c128a_decode_ragged
        elif dense_decode is not None and decode_lens is not None:
            with optional_phase_profile_stage(
                "target.attn_metadata.mla.c128.ragged_pack", dense_decode
            ):
                ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
                    dense_decode.reshape(dense_decode.shape[0], -1),
                    decode_lens,
                )
            assert self.c128a_decode_topk_ragged_indices_buffer is not None
            assert self.c128a_decode_topk_ragged_indptr_buffer is not None
            with optional_phase_profile_stage(
                "target.attn_metadata.mla.c128.ragged_copy", dense_decode
            ):
                ragged_indices, ragged_indptr = _copy_ragged_to_graph_buffers(
                    ragged_indices,
                    ragged_indptr,
                    self.c128a_decode_topk_ragged_indices_buffer,
                    self.c128a_decode_topk_ragged_indptr_buffer,
                    dense_decode.shape[0],
                    self.c128a_max_compressed,
                )

        return DeepseekV4ROCMAiterMLASparseMetadata(
            **vars(base),
            c128a_decode_topk_ragged_indices=ragged_indices,
            c128a_decode_topk_ragged_indptr=ragged_indptr,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> DeepseekV4ROCMAiterMLASparseMetadata:
        metadata = cast(
            DeepseekV4ROCMAiterMLASparseMetadata,
            super().build_for_cudagraph_capture(common_attn_metadata),
        )
        metadata.for_cudagraph_capture = _ON_GFX950
        return metadata


class DeepseekV4ROCMAiterSparseSWAMetadataBuilder(DeepseekSparseSWAMetadataBuilder):
    # Keep fused multi-step decode disabled until update_draft_decode_metadata()
    # also refreshes the ROCm-specific ragged SWA indices and indptrs.
    supports_draft_decode_metadata_update = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_direct_ragged_swa = not self._requires_dense_decode_swa_indices()
        max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        # The non-causal (DSpark draft) path widens each token's SWA index list
        # to ``noncausal_index_width`` (>= window_size), so size the persistent
        # ragged buffer to the wider bound to cover both causal and non-causal.
        swa_index_width = max(self.window_size, self.noncausal_index_width)
        self.decode_swa_ragged_indices_buffer = torch.empty(
            max_tokens * swa_index_width,
            dtype=torch.int32,
            device=self.device,
        )
        self.decode_swa_ragged_indptr_buffer = torch.empty(
            max_tokens + 1,
            dtype=torch.int32,
            device=self.device,
        )
        self._direct_decode_swa_ragged: tuple[torch.Tensor, torch.Tensor] | None = None

    def _build_decode_swa_indices(
        self,
        *,
        num_decode_tokens: int,
        non_causal: bool,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        token_to_req_indices: torch.Tensor,
        is_valid_token: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        if not self._use_direct_ragged_swa:
            return super()._build_decode_swa_indices(
                num_decode_tokens=num_decode_tokens,
                non_causal=non_causal,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                token_to_req_indices=token_to_req_indices,
                is_valid_token=is_valid_token,
                block_table=block_table,
            )

        mode = "noncausal" if non_causal else "causal"
        index_width = self.noncausal_index_width if non_causal else self.window_size
        with optional_phase_profile_stage(
            f"target.attn_metadata.swa.{mode}.direct_ragged", query_start_loc
        ):
            self._direct_decode_swa_ragged = build_swa_ragged_indices_direct(
                num_decode_tokens=num_decode_tokens,
                window_size=self.window_size,
                index_width=index_width,
                non_causal=non_causal,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                token_to_req_indices=token_to_req_indices,
                is_valid_token=is_valid_token,
                block_table=block_table,
                block_size=self.block_size,
                swa_lens=self.decode_swa_lens,
                ragged_indices_buffer=self.decode_swa_ragged_indices_buffer,
                ragged_indptr_buffer=self.decode_swa_ragged_indptr_buffer,
            )
        return self.decode_swa_indices[:, :, :0]

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4ROCMAiterSparseSWAMetadata:
        self._direct_decode_swa_ragged = None
        base = super().build(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )

        ragged_indices = None
        ragged_indptr = None
        if self._direct_decode_swa_ragged is not None:
            ragged_indices, ragged_indptr = self._direct_decode_swa_ragged
        elif (
            base.num_decode_tokens > 0
            and base.decode_swa_indices is not None
            and base.decode_swa_lens is not None
        ):
            ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
                base.decode_swa_indices.reshape(
                    base.num_decode_tokens, base.decode_swa_width
                ),
                base.decode_swa_lens,
            )
            ragged_indices, ragged_indptr = _copy_ragged_to_graph_buffers(
                ragged_indices,
                ragged_indptr,
                self.decode_swa_ragged_indices_buffer,
                self.decode_swa_ragged_indptr_buffer,
                base.num_decode_tokens,
                base.decode_swa_width,
            )

        return DeepseekV4ROCMAiterSparseSWAMetadata(
            **vars(base),
            decode_swa_ragged_indices=ragged_indices,
            decode_swa_ragged_indptr=ragged_indptr,
        )


class DeepseekV4ROCMAiterMLASparseBackend(DeepseekV4SparseMLABackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_FLASHMLA_SPARSE_DSV4"

    @staticmethod
    def get_builder_cls() -> type["DeepseekV4ROCMAiterMLASparseMetadataBuilder"]:
        return DeepseekV4ROCMAiterMLASparseMetadataBuilder


class DeepseekV4ROCMAiterMLAAttention(DeepseekV4Attention):
    """ROCm sparse MLA attention layer for DeepSeek V4."""

    backend_cls = DeepseekV4ROCMAiterMLASparseBackend

    def __init__(self, *args, **kwargs):
        vllm_config = args[0] if args else kwargs["vllm_config"]
        super().__init__(*args, **kwargs)
        self._has_kv_transfer = vllm_config.kv_transfer_config is not None
        # Block scale for the preshuffled weight; None = not preshuffled.
        self._wqa_wkv_scale: torch.Tensor | None = None
        self._wo_b_scale: torch.Tensor | None = None
        self._fused_compressor_weight: torch.Tensor | None
        self.register_buffer("_fused_compressor_weight", None, persistent=False)
        self._fused_compressor_split_sizes: tuple[int, int] | None = None

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def prepare_attn_preshuffle(self) -> None:
        from vllm._aiter_ops import rocm_aiter_ops

        if not rocm_aiter_ops.is_enabled():
            return
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )
        from vllm.model_executor.utils import replace_parameter

        def _prep(linear) -> torch.Tensor | None:
            w = getattr(linear, "weight", None)
            if w is None or w.dim() != 2:
                return None
            # K % 128 (group-128 quant) and N % 16 (shuffle_weight) must hold.
            if w.shape[-1] % 128 != 0 or w.shape[0] % 16 != 0:
                return None
            ws = getattr(linear, "weight_scale_inv", None)  # per-block scale
            if ws is None:
                return None
            if ws.dtype == torch.float8_e8m0fnu:
                ws = _upcast_e8m0_to_fp32(ws).contiguous()
            # Shuffle the weight in place (single weight, no unshuffled copy).
            replace_parameter(
                linear,
                "weight",
                rocm_aiter_ops.shuffle_weight(w.data, layout=(16, 16)),
            )
            return ws

        self._wqa_wkv_scale = _prep(self.fused_wqa_wkv)
        self._wo_b_scale = _prep(self.wo_b)

    def prepare_compressor_gemm_fusion(self) -> bool:
        if self._fused_compressor_weight is not None:
            return False

        from vllm.model_executor.offloader import NoopOffloader, get_offloader

        if not isinstance(get_offloader(), NoopOffloader):
            logger.warning_once(
                "DeepSeek V4 compressor GEMM fusion is incompatible with "
                "weight offloading and will remain disabled."
            )
            return False

        compressor = self.compressor
        indexer = self.indexer
        if compressor is None or indexer is None:
            return False

        main_weight = compressor.fused_wkv_wgate.weight
        indexer_weight = indexer.compressor.fused_wkv_wgate.weight
        if main_weight.ndim != 2 or indexer_weight.ndim != 2:
            raise ValueError("DeepSeek V4 compressor weights must be matrices")
        if main_weight.shape[1] != indexer_weight.shape[1]:
            raise ValueError("DeepSeek V4 compressor weights must share K")
        if main_weight.dtype != indexer_weight.dtype:
            raise ValueError("DeepSeek V4 compressor weights must share dtype")
        if main_weight.device != indexer_weight.device:
            raise ValueError("DeepSeek V4 compressor weights must share device")

        main_size = main_weight.shape[0]
        indexer_size = indexer_weight.shape[0]
        fused_weight = torch.cat((main_weight, indexer_weight), dim=0)
        with torch.no_grad():
            main_weight.set_(fused_weight[:main_size])
            indexer_weight.set_(fused_weight[main_size:])

        self._fused_compressor_weight = fused_weight
        self._fused_compressor_split_sizes = (main_size, indexer_size)
        return True

    def _bpre_attn_gemm(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        x: torch.Tensor,
        reduce_tp: bool,
    ) -> torch.Tensor:
        from vllm._aiter_ops import rocm_aiter_ops

        x_fp8, x_scale = rocm_aiter_ops.group_fp8_quant(x, transpose_scale=True)
        out = rocm_aiter_ops.gemm_a8w8_blockscale_bpreshuffle(
            x_fp8, weight, x_scale, scale, output_dtype=x.dtype
        )
        if reduce_tp and get_tensor_model_parallel_world_size() > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out

    def _fused_wqa_wkv_gemm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._wqa_wkv_scale is not None and hidden_states.dim() == 2:
            return self._bpre_attn_gemm(
                self.fused_wqa_wkv.weight, self._wqa_wkv_scale, hidden_states, False
            )
        return super()._fused_wqa_wkv_gemm(hidden_states)

    def _run_parallel_input_projections(
        self, hidden_states: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        fused_weight = self._fused_compressor_weight
        split_sizes = self._fused_compressor_split_sizes
        if fused_weight is None or split_sizes is None:
            return super()._run_parallel_input_projections(hidden_states)

        indexer = self.indexer
        if indexer is None:
            raise RuntimeError("Fused compressor weight requires a C4 indexer")

        qr_kv = self._fused_wqa_wkv_gemm(hidden_states)
        fused_scores = torch.mm(
            hidden_states,
            fused_weight.T,
            out_dtype=torch.float32,
        )
        kv_score, indexer_kv_score = fused_scores.split(split_sizes, dim=-1)
        indexer_weights, _ = indexer.weights_proj(hidden_states)
        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    @functools.cached_property
    def _wq_b_uses_aiter_block_scaled(self) -> bool:
        """True when both wq_b GEMMs run the aiter block-scaled fp8 kernel.

        Cached: the linear kernels and the aiter env gates are fixed once
        the model is built, so this is evaluated at the first forward
        only.

        The fused norm+quant path is only valid if the quant and GEMM it
        replaces are exactly the aiter ones; otherwise fall back to the
        shared path.
        """
        from vllm._aiter_ops import rocm_aiter_ops
        from vllm.model_executor.kernels.linear.scaled_mm import (
            Fp8BlockScaledMMLinearKernel,
        )

        if not rocm_aiter_ops.is_linear_fp8_enabled():
            return False

        linears = [self.wq_b]
        if self.indexer is not None:
            linears.append(self.indexer.wq_b)
        for linear in linears:
            kernel = getattr(getattr(linear, "quant_method", None), "fp8_linear", None)
            if not isinstance(kernel, Fp8BlockScaledMMLinearKernel):
                return False
        return True

    def _split_qkv_and_norm(
        self, qr_kv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Fuse q/kv RMSNorm + per-1x128 fp8 q quant into one aiter kernel.

        The shared path norms q and kv in one triton kernel and the wq_b
        linears then re-read the bf16 qr to quantize it. The aiter kernel
        computes both RMSNorms (fp32 accumulate) and the fp8 group quant
        in a single pass, writing fp8 qr + group scales directly; both
        wq_b GEMMs (attention and indexer) then consume that pair and
        skip their own input quant. kv stays bf16: the fused insert
        kernel RoPE/quantizes it itself. Falls back to the shared path
        when the aiter linear path is not active.
        """
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        if not (
            qr.dim() == 2
            and qr.shape[0] > 0
            and self.q_lora_rank % 128 == 0
            and self._wq_b_uses_aiter_block_scaled
        ):
            return super()._split_qkv_and_norm(qr_kv)

        from vllm._aiter_ops import rocm_aiter_ops

        return rocm_aiter_ops.fused_qk_rmsnorm_group_quant(
            q=qr,
            q_weight=self.q_norm.weight.data,
            q_epsilon=self.eps,
            kv=kv,
            kv_weight=self.kv_norm.weight.data,
            kv_epsilon=self.eps,
            group_size=128,
            transpose_scale=False,
        )

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # ROCm BF16 reference wo_a path (inverse RoPE + einsum) + wo_b.
        with phase_profile_stage("attention.output_projection.inv_rope_wo_a", o):
            z = rocm_inv_rope_einsum(
                self.rotary_emb,
                o,
                positions,
                self.rope_head_dim,
                self.n_local_groups,
                self.o_lora_rank,
                self.wo_a,
            )
        zf = z.flatten(1)
        with phase_profile_stage("attention.output_projection.wo_b", zf):
            if self._wo_b_scale is not None and zf.dim() == 2:
                return self._bpre_attn_gemm(
                    self.wo_b.weight, self._wo_b_scale, zf, True
                )
            return self.wo_b(zf)

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # Warmup dummy run: no real metadata. Reserve the same bf16
            # gather workspace _forward_prefill would; the dequantize / topk
            # / sparse_fwd kernels are skipped this step.
            swa_only = self.compress_ratio <= 1
            N = (
                0
                if swa_only
                else (self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio
            )
            M = N + self.window_size + self.max_num_batched_tokens
            if swa_only:
                top_k = 0
            elif self.compress_ratio == 128:
                top_k = N
            else:
                assert self.topk_indices_buffer is not None
                top_k = self.topk_indices_buffer.shape[-1]
            combined_topk = round_up(top_k + self.window_size, 128)
            reqs_per_chunk = _sparse_prefill_reqs_per_chunk(
                self.PREFILL_CHUNK_SIZE,
                M,
                q.shape[-1],
            )
            current_workspace_manager().get_simultaneous(
                ((reqs_per_chunk, M, q.shape[-1]), torch.bfloat16),
                ((self.max_num_batched_tokens, combined_topk), torch.int32),
                ((self.max_num_batched_tokens,), torch.int32),
            )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        rocm_metadata = cast(
            DeepseekV4ROCMAiterMLASparseMetadata | None,
            attn_metadata.get(self.prefix),
        )
        swa_metadata = cast(
            DeepseekV4ROCMAiterSparseSWAMetadata | None,
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=rocm_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=rocm_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
                adaptive_splits=(
                    _ON_GFX950
                    and not swa_only
                    and self.compress_ratio == 128
                    and rocm_metadata is not None
                    and rocm_metadata.for_cudagraph_capture
                ),
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
        adaptive_splits: bool,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        topk_ragged_indices = None
        topk_ragged_indptr = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                with phase_profile_stage("sparse_mla.topk_metadata", q, flow="decode"):
                    (
                        topk_ragged_indices,
                        topk_ragged_indptr,
                        topk_lens,
                    ) = compute_global_topk_ragged_indices_and_indptr(
                        self.topk_indices_buffer[:num_decode_tokens],
                        swa_metadata.token_to_req_indices,
                        attn_metadata.block_table[:num_decodes],
                        block_size,
                        is_valid,
                    )
            else:
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens
                topk_ragged_indices = attn_metadata.c128a_decode_topk_ragged_indices
                topk_ragged_indptr = attn_metadata.c128a_decode_topk_ragged_indptr

        rocm_sparse_attn_decode(
            q=q,
            kv_cache=kv_cache,
            swa_k_cache=self.swa_cache_layer.kv_cache,
            swa_only=swa_only,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
            swa_indices=swa_metadata.decode_swa_indices,
            swa_lens=swa_metadata.decode_swa_lens,
            swa_ragged_indices=swa_metadata.decode_swa_ragged_indices,
            swa_ragged_indptr=swa_metadata.decode_swa_ragged_indptr,
            topk_ragged_indices=topk_ragged_indices,
            topk_ragged_indptr=topk_ragged_indptr,
            attn_sink=self.attn_sink,
            scale=self.scale,
            head_dim=self.head_dim,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            output=output,
            adaptive_splits=adaptive_splits,
            extra_cache_nan_free=_trust_dsv4_extra_cache_nan_free(
                self.kv_cache_dtype,
                self._has_kv_transfer,
                not swa_only and kv_cache is not None,
            ),
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            assert topk_indices is not None
            top_k = topk_indices.shape[-1]
            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
        else:
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            N = 0

        M = N + self.window_size + self.max_num_batched_tokens
        reqs_per_chunk = _sparse_prefill_reqs_per_chunk(
            self.PREFILL_CHUNK_SIZE,
            M,
            q.shape[-1],
        )
        num_chunks = (num_prefills + reqs_per_chunk - 1) // reqs_per_chunk

        workspace_manager = current_workspace_manager()
        combined_topk = round_up(top_k + self.window_size, 128)
        kv, combined_indices_out, combined_lens_out = (
            workspace_manager.get_simultaneous(
                ((reqs_per_chunk, M, q.shape[-1]), torch.bfloat16),
                ((self.max_num_batched_tokens, combined_topk), torch.int32),
                ((self.max_num_batched_tokens,), torch.int32),
            )
        )
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * reqs_per_chunk
            chunk_end = min(chunk_start + reqs_per_chunk, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                assert attn_metadata is not None
                assert compressed_k_cache is not None
                block_table = attn_metadata.block_table[num_decodes:]
                # compressed_k_cache is OCP on every platform (Triton encoder).
                with phase_profile_stage(
                    "sparse_mla.compressed_gather", q, flow="prefill"
                ):
                    dequantize_and_gather_k_cache(
                        kv[:chunk_size],
                        compressed_k_cache,
                        seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                        gather_lens=None,
                        block_table=block_table[chunk_start:chunk_end],
                        block_size=attn_metadata.block_size // self.compress_ratio,
                        offset=0,
                        use_fnuz=False,
                    )

            swa_block_table = swa_metadata.block_table[num_decodes:]
            with phase_profile_stage("sparse_mla.swa_gather", q, flow="prefill"):
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    swa_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end],
                    gather_lens=gather_lens[chunk_start:chunk_end],
                    block_table=swa_block_table[chunk_start:chunk_end],
                    block_size=swa_metadata.block_size,
                    offset=N,
                    use_fnuz=current_platform.is_fp8_fnuz(),
                )

            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )
            combined_indices_chunk = combined_indices_out[: query_end - query_start]
            combined_lens_chunk = combined_lens_out[: query_end - query_start]

            with phase_profile_stage("sparse_mla.combine_indices", q, flow="prefill"):
                combined_indices, combined_lens = combine_topk_swa_indices(
                    topk_indices[query_start:query_end],
                    query_start_loc[
                        num_decodes + chunk_start : num_decodes + chunk_end + 1
                    ],
                    seq_lens[chunk_start:chunk_end],
                    gather_lens[chunk_start:chunk_end],
                    self.window_size,
                    self.compress_ratio,
                    top_k,
                    M,
                    N,
                    out=(combined_indices_chunk, combined_lens_chunk),
                )
            with phase_profile_stage("sparse_mla.attention", q, flow="prefill"):
                rocm_sparse_attn_prefill(
                    q=q[query_start:query_end],
                    kv=kv.view(-1, 1, q.shape[-1]),
                    indices=combined_indices,
                    topk_length=combined_lens,
                    scale=self.scale,
                    head_dim=self.head_dim,
                    nope_head_dim=self.nope_head_dim,
                    rope_head_dim=self.rope_head_dim,
                    attn_sink=self.attn_sink,
                    output=output[query_start:query_end],
                )
