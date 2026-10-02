# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import os
from dataclasses import dataclass
from typing import cast

import torch

import vllm.envs as envs
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.deepseek_v4.amd.rocm import (
    build_swa_ragged_indices_direct as _build_swa_ragged_indices_direct,
)
from vllm.models.deepseek_v4_1 import row_probe
from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
from vllm.models.deepseek_v4_1.common.ops import dequantize_and_gather_k_cache
from vllm.models.deepseek_v4_1.sparse_mla import (
    DeepseekV4FlashMLAMetadata,
    DeepseekV4SparseMLABackend,
    DeepseekV4SparseMLAMetadataBuilder,
    DeepseekV41SparseSWAMetadataBuilder,
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
    DeepseekSparseSWABackend,
    DeepseekSparseSWAMetadata,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    SPARSE_DIRECT_KV_REQ_STRIDE,
    SPARSE_DIRECT_KV_SWA_TAG_BASE,
    SPARSE_DIRECT_KV_TAG_BASE,
    build_ragged_indices_from_dense,
    rocm_inv_rope_einsum,
    rocm_sparse_attn_decode,
    rocm_sparse_attn_prefill,
    rocm_sparse_attn_reduce_partials,
)
from vllm.v1.worker.workspace import (
    PREFILL_WORKSPACE_LANE,
    current_workspace_manager,
    use_workspace_lane,
)

logger = init_logger(__name__)

# Prefill workspace admission counters, per worker process: how often a step was
# sent to the direct paged-cache reader and how often a batch was chunked down to
# fit the fixed capacity. Logged on the first occurrence and then periodically so
# an operator can see the workspace contract being exercised.
_ADMISSION_COUNTS = {"direct": 0, "chunked": 0}
_ADMISSION_LOG_EVERY = 64

# The prefill gather asks for its worst case one scheduler chunk at a time, so a
# long prompt grows the lane's buffer once per chunk (each growth reallocates and
# empties the allocator cache). Rounding the gathered width up to this many
# columns turns that into one growth per few chunks -- a 262144-token prefill
# went from 262 growths to 33 -- at the cost of at most this many columns of
# slack per row (512 * 2 bytes each). M is both the tensor width and the row
# stride the combined indices use, so the rounded value must be the one passed
# to both.
_PREFILL_WORKSPACE_COLUMN_STEP = 8192


def _prefill_gather_width(
    context_rows: int, window_size: int, max_num_batched_tokens: int
) -> int:
    """Rounded workspace width for one prefill gather row."""
    return round_up(
        context_rows + window_size + max_num_batched_tokens,
        _PREFILL_WORKSPACE_COLUMN_STEP,
    )


def _prefill_split_k_partials(
    manager, capacity: int, num_passes: int, n_heads: int, comb_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed-shape ``(m, l, acc)`` partials for split-K prefill.

    Both the warmup and the serving path go through here, so a lane locked after
    capture can never meet a key the warmup did not create. The buffers are sized
    for the whole chunk budget once and sliced per chunk by the caller; they are
    keyed per ``(ubatch, lane)`` and are not lane offsets, so they cannot overlap
    the gathered rows.
    """
    shape = (capacity, num_passes, n_heads)
    return (
        manager.get_persistent("dsv41_split_k_m", shape, torch.float32),
        manager.get_persistent("dsv41_split_k_l", shape, torch.float32),
        manager.get_persistent("dsv41_split_k_acc", (*shape, comb_dim), torch.float32),
    )


def _prefill_gather_budget_bytes(
    budget_tokens: int,
    compress_ratio: int,
    window_size: int,
    max_num_batched_tokens: int,
    head_dim: int,
    element_bytes: int,
) -> int:
    """Bytes one gather row may occupy under a token-denominated budget.

    The knob counts context tokens while a gather row holds *compressed* rows, so
    the same division the warmup reservation applies has to be applied here; a
    compressed layer would otherwise be admitted to a row wider than it reserved.
    """
    if budget_tokens <= 0:
        return 0
    context_rows = budget_tokens // max(1, compress_ratio)
    return (
        _prefill_gather_width(context_rows, window_size, max_num_batched_tokens)
        * head_dim
        * element_bytes
    )


def _prefill_gather_plan(
    context_rows: int,
    num_prefills: int,
    window_size: int,
    max_num_batched_tokens: int,
    head_dim: int,
    element_bytes: int,
    budget_bytes: int,
    max_rows: int,
) -> tuple[int, bool]:
    """Rows one prefill chunk may gather, and whether the gather is admitted.

    A fixed budget turns the gather workspace into an admission-controlled
    resource instead of a growable one: a batch whose context does not fit a
    single row is served by the direct paged-cache reader, and a batch that fits
    fewer than ``max_rows`` rows is split into correspondingly more chunks.
    ``budget_bytes <= 0`` keeps the unbounded behaviour (grow on demand).

    Returns:
        ``(rows, admit_direct)``. ``rows`` is at least 1 and at most ``max_rows``.
    """
    if budget_bytes <= 0:
        return min(max_rows, max(1, num_prefills)), False
    row_bytes = (
        _prefill_gather_width(context_rows, window_size, max_num_batched_tokens)
        * head_dim
        * element_bytes
    )
    if row_bytes > budget_bytes:
        return 1, True
    fits = max(1, budget_bytes // row_bytes)
    return max(1, min(max_rows, max(1, num_prefills), fits)), False


_SHAPE_PROBE_TRIGGER = "/tmp/dsv41_fastprefill_shapes_on"
_SHAPE_PROBE_BASE = "/tmp/dsv41_fastprefill_shapes"


def _shape_probe(prefix, q, positions, swa_metadata, attn_metadata, compress_ratio):
    """Dump the bounded prefill's row bookkeeping, then synchronize.

    Temporary diagnostic for the switch-on CUDA fault: the reported site is a
    one-element division, so the error is sticky and belongs to an earlier
    launch. Writing the shapes first shows whether metadata and activations
    disagree, and the synchronize attributes any pending fault to whatever ran
    before this point instead of to the division after it.
    """
    if not os.path.exists(_SHAPE_PROBE_TRIGGER):
        return
    rank = torch.accelerator.current_device_index()
    fields = [
        prefix,
        f"q={q.shape[0]}",
        f"pos={positions.numel()}",
        f"ndec={swa_metadata.num_decodes}",
        f"ndec_tok={swa_metadata.num_decode_tokens}",
        f"npref={swa_metadata.num_prefills}",
        f"npref_tok={swa_metadata.num_prefill_tokens}",
    ]
    lengths = swa_metadata.prefill_gather_lens
    if lengths is not None:
        fields.append(f"gather_len={lengths.numel()}:{int(lengths.max().item())}")
    seq = swa_metadata.prefill_seq_lens
    if seq is not None:
        fields.append(f"seq={seq.numel()}:{int(seq.max().item())}")
    slots = swa_metadata.slot_mapping
    if slots is not None:
        fields.append(f"slots={slots.numel()}")
    if attn_metadata is not None:
        prefill_tokens = getattr(
            attn_metadata,
            "num_prefill_tokens",
            getattr(attn_metadata, "num_actual_tokens", -1),
        )
        fields.append(f"mpref_tok={prefill_tokens}")
        fields.append(f"mmax_seq={getattr(attn_metadata, 'max_seq_len', -1)}")
        fields.append(f"ratio={compress_ratio}")
    with open(f"{_SHAPE_PROBE_BASE}.rank{rank}.tsv", "a") as handle:
        handle.write("\t".join(str(value) for value in fields) + "\n")
    torch.accelerator.synchronize()


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
    direct_kv: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine compressed-attention and sliding-window indices with Torch.

    The Triton implementation inherited from DeepSeek V4 launches a
    two-dimensional grid with 128 workers per request.  On gfx950 it can issue
    an out-of-bounds access for V4.1's mixed prefill metadata (including the
    synthetic mixed-token warmup).  This path is prefill-only and the tensors
    are small, so use ordinary Torch indexing until a gfx950-safe fused kernel
    is available.
    """
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        fill_value=-1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )

    # query_start_loc may have a non-zero base for a narrowed mixed batch.
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    req_ids = torch.repeat_interleave(
        torch.arange(seq_lens.shape[0], device=seq_lens.device), query_lens
    )
    query_starts = query_start_loc[:-1] - query_start_loc[0]
    token_offsets = torch.arange(num_tokens, device=seq_lens.device) - (
        torch.repeat_interleave(query_starts, query_lens)
    )
    positions = seq_lens[req_ids] - query_lens[req_ids] + token_offsets

    logical_topk_width = min(topk, topk_indices.shape[1])
    topk_lens = torch.minimum(
        (positions + 1) // compress_ratio,
        torch.full_like(positions, logical_topk_width),
    ).clamp_min(0)
    topk_offsets = torch.arange(logical_topk_width, device=seq_lens.device)
    topk_mask = topk_offsets[None, :] < topk_lens[:, None]
    topk_values = topk_indices[:, :logical_topk_width].to(torch.int32)
    topk_valid = topk_mask & (topk_values >= 0) & (topk_values < N)
    if direct_kv:
        # Tagged compressed positions: the attention kernel rebuilds (request,
        # compressed position) from the tag and reads the paged cache itself, so
        # nothing has to be gathered for them and the sliding window turns into
        # the only region of the workspace. That is what keeps the workspace from
        # growing with the context length.
        compressed_values = (
            topk_values
            + SPARSE_DIRECT_KV_TAG_BASE
            + req_ids[:, None] * SPARSE_DIRECT_KV_REQ_STRIDE
        )
    else:
        compressed_values = topk_values + (M * req_ids).to(torch.int32)[:, None]
    combined_indices[:, :logical_topk_width] = torch.where(
        topk_valid,
        compressed_values.to(torch.int32),
        -1,
    )

    swa_lens = torch.minimum(
        positions + 1, torch.full_like(positions, window_size)
    ).clamp_min(0)
    swa_offsets = torch.arange(window_size, device=seq_lens.device)
    swa_mask = swa_offsets[None, :] < swa_lens[:, None]
    swa_columns = topk_lens[:, None] + swa_offsets[None, :]
    gather_starts = seq_lens - gather_lens
    if direct_kv:
        # Sliding-window entries carry their logical token position in the SWA
        # cache, so the kernel can address that cache directly too: neither source
        # is gathered, and the workspace stops depending on the context length.
        swa_values = (
            SPARSE_DIRECT_KV_SWA_TAG_BASE
            + req_ids[:, None] * SPARSE_DIRECT_KV_REQ_STRIDE
            + swa_offsets[None, :]
            + positions[:, None]
            - swa_lens[:, None]
            + 1
        ).to(torch.int32)
    else:
        swa_values = (
            M * req_ids[:, None]
            + N
            + swa_offsets[None, :]
            + positions[:, None]
            - swa_lens[:, None]
            + 1
            - gather_starts[req_ids, None]
        ).to(torch.int32)
    rows = torch.arange(num_tokens, device=seq_lens.device)[:, None].expand_as(
        swa_columns
    )
    flat_dst = rows[swa_mask] * combined_topk + swa_columns[swa_mask]
    combined_indices.view(-1).index_copy_(0, flat_dst, swa_values[swa_mask])
    combined_lens.copy_((topk_lens + swa_lens).to(torch.int32))
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


def _prefill_context_rows(
    seq_lens_cpu: torch.Tensor | None,
    compress_ratio: int,
    fallback_rows: int,
) -> int:
    """Compressed context rows the prefill gather buffer has to hold.

    Sizing this by ``max_model_len // compress_ratio`` reserves a full-context
    gather on every rank, whatever the batch is; the batch metadata carries the
    real bound (``prefill_seq_lens_cpu`` is the CPU mirror, so reading it costs
    no sync). Without metadata the caller supplies a fallback.
    """
    if seq_lens_cpu is None or seq_lens_cpu.numel() == 0:
        return max(0, fallback_rows)
    longest = int(seq_lens_cpu.max().item())
    ratio = max(1, compress_ratio)
    return max(0, longest // ratio)


@dataclass
class DeepseekV4ROCMAiterSparseSWAMetadata(DeepseekSparseSWAMetadata):
    decode_swa_ragged_indices: torch.Tensor | None = None
    decode_swa_ragged_indptr: torch.Tensor | None = None


@triton.jit
def _build_small_swa_ragged_kernel(
    swa_lens_ptr,
    ragged_indptr_ptr,
    ragged_indices_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    token_to_req_indices_ptr,
    is_valid_token_ptr,
    block_table_ptr,
    block_table_stride,
    window_size,
    block_size,
    NUM_TOKENS: tl.constexpr,
    INDEX_WIDTH: tl.constexpr,
    NON_CAUSAL: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    token_idx = tl.program_id(0)
    rows = tl.arange(0, BLOCK_ROWS)
    is_valid = tl.load(is_valid_token_ptr + rows, rows < NUM_TOKENS, other=0)
    mask = (rows < NUM_TOKENS) & is_valid
    req_idx = tl.load(token_to_req_indices_ptr + rows, mask, other=0)
    query_start = tl.load(query_start_loc_ptr + req_idx, mask, other=0)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1, mask, other=0)
    seq_len = tl.load(seq_lens_ptr + req_idx, mask, other=0)
    prefix_len = seq_len - (query_end - query_start)
    if NON_CAUSAL:
        start_pos = tl.maximum(prefix_len - window_size, 0)
        end_pos = seq_len
    else:
        pos = prefix_len + rows - query_start
        start_pos = tl.maximum(pos - window_size + 1, 0)
        end_pos = pos + 1
    lengths = tl.where(mask, end_pos - start_pos, 0).to(tl.int32)
    # Small decode batches can recompute the prefix in each CTA, avoiding a
    # separate indptr launch and any inter-CTA dependency.
    out_start = tl.sum(tl.where(rows < token_idx, lengths, 0), axis=0)
    length = tl.sum(tl.where(rows == token_idx, lengths, 0), axis=0)
    row_start = tl.sum(tl.where(rows == token_idx, start_pos, 0), axis=0)
    row_req = tl.sum(tl.where(rows == token_idx, req_idx, 0), axis=0)
    tl.store(swa_lens_ptr + token_idx, length)
    tl.store(ragged_indptr_ptr + token_idx, out_start)
    if token_idx == NUM_TOKENS - 1:
        tl.store(ragged_indptr_ptr + NUM_TOKENS, out_start + length)

    offsets = tl.arange(0, BLOCK_COLS)
    index_mask = (offsets < length) & (offsets < INDEX_WIDTH)
    positions = row_start + offsets
    block_numbers = tl.load(
        block_table_ptr + row_req * block_table_stride + positions // block_size,
        index_mask,
        other=-1,
    )
    slot_ids = block_numbers * block_size + positions % block_size
    tl.store(ragged_indices_ptr + out_start + offsets, slot_ids, index_mask)


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
    use_fused: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optionally fuse small V4.1 decode ragged metadata in persistent buffers."""
    if not (use_fused and 0 < num_decode_tokens <= 48 and 0 < index_width <= 256):
        return _build_swa_ragged_indices_direct(
            num_decode_tokens=num_decode_tokens,
            window_size=window_size,
            index_width=index_width,
            non_causal=non_causal,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            token_to_req_indices=token_to_req_indices,
            is_valid_token=is_valid_token,
            block_table=block_table,
            block_size=block_size,
            swa_lens=swa_lens,
            ragged_indices_buffer=ragged_indices_buffer,
            ragged_indptr_buffer=ragged_indptr_buffer,
        )
    assert num_decode_tokens <= swa_lens.numel()
    assert ragged_indptr_buffer.numel() >= num_decode_tokens + 1
    assert ragged_indices_buffer.numel() >= num_decode_tokens * index_width
    ragged_indptr = ragged_indptr_buffer[: num_decode_tokens + 1]
    ragged_indices = ragged_indices_buffer[: num_decode_tokens * index_width]
    _build_small_swa_ragged_kernel[(num_decode_tokens,)](
        swa_lens,
        ragged_indptr,
        ragged_indices,
        query_start_loc,
        seq_lens,
        token_to_req_indices,
        is_valid_token,
        block_table,
        block_table.stride(0),
        window_size,
        block_size,
        NUM_TOKENS=num_decode_tokens,
        INDEX_WIDTH=index_width,
        NON_CAUSAL=non_causal,
        BLOCK_ROWS=triton.next_power_of_2(num_decode_tokens),
        BLOCK_COLS=triton.next_power_of_2(index_width),
        num_warps=4,
    )
    return ragged_indices, ragged_indptr


class DeepseekV4ROCMAiterSparseSWAMetadataBuilder(DeepseekV41SparseSWAMetadataBuilder):
    # Keep fused multi-step decode disabled until update_draft_decode_metadata()
    # also refreshes the ROCm-specific ragged SWA indices and indptrs.
    supports_draft_decode_metadata_update = False

    def _requires_dense_decode_swa_indices(self) -> bool:
        return not (
            envs.VLLM_DEEPSEEK_V4_DIRECT_RAGGED_SWA
            and current_platform.is_device_capability_family(80)
        )

    def _requires_decode_swa_lens_tail_clear(self) -> bool:
        return not self._use_direct_ragged_swa

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        additional_config = self.vllm_config.additional_config
        fused_swa_metadata = (
            additional_config.get("dsv41_sm80_fused_swa_metadata", False)
            if isinstance(additional_config, dict)
            else False
        )
        if not isinstance(fused_swa_metadata, bool):
            raise ValueError("dsv41_sm80_fused_swa_metadata must be a JSON boolean")
        self._use_fused_swa_metadata = (
            fused_swa_metadata
            and current_platform.is_cuda()
            and self.device.type == "cuda"
            and current_platform.is_device_capability(80, self.device.index or 0)
        )
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
        self._use_direct_ragged_swa = (
            envs.VLLM_DEEPSEEK_V4_DIRECT_RAGGED_SWA
            and current_platform.is_device_capability_family(80)
        )
        self._direct_decode_swa_ragged: tuple[torch.Tensor, torch.Tensor] | None = None

    def _get_is_valid_token(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        slot_mapping: torch.Tensor,
        num_decode_tokens: int,
        num_prefill_tokens: int,
    ) -> torch.Tensor:
        if not self._use_fused_swa_metadata:
            return super()._get_is_valid_token(
                common_attn_metadata,
                slot_mapping,
                num_decode_tokens,
                num_prefill_tokens,
            )
        is_valid_token = self.is_valid_token[: slot_mapping.shape[0]]
        torch.ge(slot_mapping, 0, out=is_valid_token)
        return is_valid_token

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

        index_width = self.noncausal_index_width if non_causal else self.window_size
        with optional_phase_profile_stage(
            "target.attn_metadata.swa.direct_ragged", query_start_loc
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
                use_fused=self._use_fused_swa_metadata,
            )
        # DSpark decode consumes the ragged fields; dense rows remain unused.
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
    def get_builder_cls() -> type[DeepseekV4SparseMLAMetadataBuilder]:
        return DeepseekV4SparseMLAMetadataBuilder


class DeepseekV41ROCMAiterSparseSWABackend(DeepseekSparseSWABackend):
    @staticmethod
    def get_builder_cls() -> type["DeepseekV4ROCMAiterSparseSWAMetadataBuilder"]:
        return DeepseekV4ROCMAiterSparseSWAMetadataBuilder


class DeepseekV41ROCMAiterMLAAttention(DeepseekV4Attention):
    """ROCm sparse MLA attention layer for DeepSeek V4.1."""

    _combine_prefill_indices = staticmethod(combine_topk_swa_indices)
    backend_cls = DeepseekV4ROCMAiterMLASparseBackend
    swa_backend_cls = DeepseekV41ROCMAiterSparseSWABackend

    def __init__(self, *args, **kwargs):
        vllm_config = args[0] if args else kwargs["vllm_config"]
        super().__init__(*args, **kwargs)
        # CUDA executes WO_A with a quantized grouped-BMM kernel.  ROCm's
        # correctness path below dequantizes WO_A once and uses torch.einsum,
        # so retain the ordinary MXFP8 linear kernel for post-load processing
        # instead of asking for the CUDA-only BMM kernel.
        self.wo_a.is_bmm = False
        self._direct_kv_logged = False
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
        # V4.1 derives index keys from the source compressor's emitted latent
        # and has no nested ``indexer.compressor``.  Keep the projections
        # separate and use the shared linear/PyTorch correctness path.
        return False

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
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        return super()._run_parallel_input_projections(hidden_states)

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
        if self._wo_b_scale is not None and zf.dim() == 2:
            result = self._bpre_attn_gemm(self.wo_b.weight, self._wo_b_scale, zf, True)
        else:
            result = self.wo_b(zf)
        return result

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
            swa_only = self.compress_ratio == 0
            # The reservation *is* the capacity admission control enforces, so the
            # profiler accounts for exactly what serving may ever gather.
            reserved = self._prefill_workspace_tokens()
            if swa_only:
                N = 0
            elif reserved > 0:
                N = min(reserved, self.max_model_len) // max(1, self.compress_ratio)
            else:
                N = (
                    self.max_model_len + self.compress_ratio - 1
                ) // self.compress_ratio
            M = _prefill_gather_width(N, self.window_size, self.max_num_batched_tokens)
            # One row, not the per-chunk cap: a single long prompt is the shape
            # this reservation is for, and a prefill batch that needs more rows
            # grows the lane later. Growth is safe here because the lane is
            # reserved for eager-only work (see PREFILL_WORKSPACE_LANE); the
            # cost is that the memory profiler sees one row's worth of this
            # buffer, so KV sizing that relies on the profiler should leave the
            # rest of the cap as headroom (or pin KV bytes, as this deployment
            # does).
            with use_workspace_lane(PREFILL_WORKSPACE_LANE):
                workspace_manager = current_workspace_manager()
                workspace_manager.get_simultaneous(
                    ((1, M, q.shape[-1]), torch.bfloat16),
                )
                # Admission control can switch any prefill over to the direct
                # reader, so its stub belongs to the warmup set: a lock taken
                # after capture must not meet a key warmup never created.
                workspace_manager.get_persistent(
                    "dsv41_direct_kv_stub",
                    (1, 1, q.shape[-1]),
                    torch.bfloat16,
                    zero_init=True,
                )
                if envs.VLLM_DSV41_PREFILL_SPLIT_K and not swa_only:
                    # Split-K partials are created here, not on the first real
                    # request: the lane may be locked after capture, and its
                    # capacity reservation must account for them.
                    _prefill_split_k_partials(
                        workspace_manager,
                        max(1, self.max_num_batched_tokens),
                        2,
                        self.n_heads,
                        self.nope_head_dim + self.rope_head_dim,
                    )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        rocm_metadata = cast(
            DeepseekV4FlashMLAMetadata | None,
            attn_metadata.get(self.compressed_cache_prefix)
            if self.compressed_cache_prefix is not None
            else None,
        )
        swa_metadata = cast(
            DeepseekV4ROCMAiterSparseSWAMetadata | None,
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio == 0
        self_kv_cache = None if swa_only else self._compressed_kv_cache()
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
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_lens = None
        topk_ragged_indices = None
        topk_ragged_indptr = None
        if not swa_only:
            # Local indices filled by the index-source layer's indexer.
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            assert self.topk_indices_buffer is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
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

        rocm_sparse_attn_decode(
            q=q,
            kv_cache=kv_cache,
            swa_k_cache=self.swa_cache_layer.kv_cache,
            swa_only=swa_only,
            topk_indices=None,
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
            extra_cache_nan_free=_trust_dsv4_extra_cache_nan_free(
                self.kv_cache_dtype,
                self._has_kv_transfer,
                not swa_only and kv_cache is not None,
            ),
        )

    def _prefill_workspace_tokens(self) -> int:
        """Context tokens one gather row may cover; 0 disables the fixed budget."""
        reserved = envs.VLLM_DSV41_PREFILL_WORKSPACE_TOKENS
        return 0 if reserved < 0 else min(reserved, self.max_model_len)

    def _prefill_workspace_budget_bytes(self) -> int:
        """Byte budget for the prefill gather lane; 0 means unbounded.

        Token-denominated, so operators keep sizing it in context tokens; the
        compressed-row division matches the warmup reservation.
        """
        return _prefill_gather_budget_bytes(
            self._prefill_workspace_tokens(),
            self.compress_ratio,
            self.window_size,
            self.max_num_batched_tokens,
            self.head_dim,
            2,
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
    ) -> None:
        swa_only = attn_metadata is None
        _shape_probe(
            self.prefix, q, positions, swa_metadata, attn_metadata, self.compress_ratio
        )

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

        # Local indices filled by the index source; SWA-only layers pass
        # top_k=0 and never read them.
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[num_decode_tokens:]
        topk_indices = topk_indices[:num_prefill_tokens]
        if not swa_only:
            top_k = topk_indices.shape[-1]
            N = _prefill_context_rows(
                swa_metadata.prefill_seq_lens_cpu,
                self.compress_ratio,
                (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio,
            )
        else:
            top_k = 0
            N = 0

        # Direct KV reads keep the compressed context out of the workspace: the
        # attention kernel reads those positions from the paged fp8 cache itself,
        # so only the sliding-window rows are gathered and the workspace width no
        # longer follows the context length. They cost ~18% prefill throughput, so
        # the default is to gather, and "auto" only switches a request over once
        # its gathered workspace would be large.
        # N counts compressed rows, so compare in context tokens.
        direct_mode = envs.VLLM_DSV41_PREFILL_DIRECT_KV
        use_direct_kv = not swa_only and (
            direct_mode == 1
            or (
                direct_mode == 2
                and N * self.compress_ratio
                > envs.VLLM_DSV41_PREFILL_DIRECT_KV_THRESHOLD_TOKENS
            )
        )
        if use_direct_kv and not self._direct_kv_logged:
            self._direct_kv_logged = True
            logger.info_once(
                "DeepSeek V4.1 prefill reads compressed KV from the paged cache "
                "(mode %d, %d context tokens).",
                direct_mode,
                N,
            )
        # One row per prefill request in this batch, bounded by the fixed budget:
        # a context that does not fit a single row is admitted to the direct
        # paged-cache reader instead of growing the buffer, and a batch that fits
        # fewer rows is split into more chunks. The combined indices address rows
        # as ``M * req_id``, so fewer rows change nothing else.
        rows, admit_direct = _prefill_gather_plan(
            N,
            num_prefills,
            self.window_size,
            self.max_num_batched_tokens,
            self.head_dim,
            2,
            self._prefill_workspace_budget_bytes(),
            self.PREFILL_CHUNK_SIZE,
        )
        if admit_direct and not use_direct_kv and current_platform.is_fp8_fnuz():
            # The direct reader unpacks OCP fp8 rows; a FNUZ cache cannot use it.
            # Keep gathering (the budget is advisory there) rather than entering a
            # path the kernel refuses.
            logger.info_once(
                "DeepSeek V4.1 prefill workspace budget (%d tokens) is below one "
                "context row, but this platform's fp8 caches are FNUZ, so the "
                "gather lane is allowed to grow past the budget.",
                self._prefill_workspace_tokens(),
            )
        elif admit_direct and not use_direct_kv:
            use_direct_kv = True
            _ADMISSION_COUNTS["direct"] += 1
            if (
                _ADMISSION_COUNTS["direct"] == 1
                or _ADMISSION_COUNTS["direct"] % _ADMISSION_LOG_EVERY == 0
            ):
                logger.info(
                    "DeepSeek V4.1 prefill workspace budget (%d tokens) is smaller "
                    "than one %d-token context row; reading the paged cache "
                    "directly (%d step(s) so far).",
                    self._prefill_workspace_tokens(),
                    N * self.compress_ratio,
                    _ADMISSION_COUNTS["direct"],
                )
        elif not use_direct_kv and rows < min(self.PREFILL_CHUNK_SIZE, num_prefills):
            _ADMISSION_COUNTS["chunked"] += 1
            if _ADMISSION_COUNTS["chunked"] % _ADMISSION_LOG_EVERY == 1:
                logger.info(
                    "DeepSeek V4.1 prefill workspace budget (%d tokens) fits %d of "
                    "%d concurrent prefill rows; the batch is split into more "
                    "chunks (%d time(s) so far).",
                    self._prefill_workspace_tokens(),
                    rows,
                    num_prefills,
                    _ADMISSION_COUNTS["chunked"],
                )

        if use_direct_kv:
            assert attn_metadata is not None and compressed_k_cache is not None
            direct_block_table = attn_metadata.block_table[num_decodes:]
            direct_block_size = attn_metadata.block_size // self.compress_ratio
            direct_swa_block_table = swa_metadata.block_table[num_decodes:]

        M = _prefill_gather_width(
            0 if use_direct_kv else N, self.window_size, self.max_num_batched_tokens
        )
        num_chunks = (num_prefills + rows - 1) // rows
        if use_direct_kv:
            # Nothing is gathered, so the prefill lane is not touched at all and
            # the workspace stops depending on the context length. The kernel
            # never dereferences this stub: every valid index is tagged.
            with use_workspace_lane(PREFILL_WORKSPACE_LANE):
                kv = current_workspace_manager().get_persistent(
                    "dsv41_direct_kv_stub",
                    (1, 1, q.shape[-1]),
                    torch.bfloat16,
                    zero_init=True,
                )
        else:
            workspace_manager = current_workspace_manager()
            with use_workspace_lane(PREFILL_WORKSPACE_LANE):
                kv = workspace_manager.get_simultaneous(
                    ((rows, M, q.shape[-1]), torch.bfloat16),
                )[0]
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * rows
            chunk_end = min(chunk_start + rows, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not use_direct_kv:
                if not swa_only:
                    assert attn_metadata is not None
                    assert compressed_k_cache is not None
                    block_table = attn_metadata.block_table[num_decodes:]
                    # compressed_k_cache is OCP on every platform (Triton encoder).
                    with optional_phase_profile_stage(
                        "sparse_mla.gather_compressed", q
                    ):
                        dequantize_and_gather_k_cache(
                            kv[:chunk_size],
                            compressed_k_cache,
                            seq_lens=seq_lens[chunk_start:chunk_end]
                            // self.compress_ratio,
                            gather_lens=None,
                            block_table=block_table[chunk_start:chunk_end],
                            block_size=attn_metadata.block_size // self.compress_ratio,
                            offset=0,
                            use_fnuz=False,
                        )

                swa_block_table = swa_metadata.block_table[num_decodes:]
                with optional_phase_profile_stage("sparse_mla.gather_swa", q):
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

            # Split-K walks the support in passes and reduces the raw
            # (m, l, acc) states afterwards, which bounds the gathered rows; the
            # compressed support and the sliding window are independent regions,
            # so they become one pass each.
            split_k = (
                bool(envs.VLLM_DSV41_PREFILL_SPLIT_K)
                and not use_direct_kv
                and not swa_only
                and top_k > 0
            )
            passes: tuple[tuple[int, int], ...] = (
                ((0, top_k), (self.window_size, 0))
                if split_k
                else ((self.window_size, top_k),)
            )
            part_m = part_l = part_acc = None
            if split_k:
                num_chunk_queries = query_end - query_start
                # Persistent, fixed-shape resources: one allocation for the whole
                # manager lifetime, sliced per chunk. They are keyed per
                # (ubatch, lane) and never handed out as lane offsets, so they
                # cannot overlap the gathered rows the way a second
                # get_simultaneous call in the same lane would.
                with use_workspace_lane(PREFILL_WORKSPACE_LANE):
                    part_m, part_l, part_acc = _prefill_split_k_partials(
                        current_workspace_manager(),
                        max(1, self.max_num_batched_tokens),
                        len(passes),
                        self.n_heads,
                        self.nope_head_dim + self.rope_head_dim,
                    )
                part_m = part_m[:num_chunk_queries]
                part_l = part_l[:num_chunk_queries]
                part_acc = part_acc[:num_chunk_queries]
            combined_indices = combined_lens = None
            for split_id, (pass_window, pass_topk) in enumerate(passes):
                if split_k or split_id == 0:
                    with optional_phase_profile_stage("sparse_mla.combine_indices", q):
                        combined_indices, combined_lens = self._combine_prefill_indices(
                            topk_indices[query_start:query_end],
                            query_start_loc[
                                num_decodes + chunk_start : num_decodes + chunk_end + 1
                            ],
                            seq_lens[chunk_start:chunk_end],
                            gather_lens[chunk_start:chunk_end],
                            pass_window,
                            self.compress_ratio,
                            pass_topk,
                            M,
                            N,
                            direct_kv=use_direct_kv,
                        )
                assert combined_indices is not None and combined_lens is not None
                if row_probe.active():
                    # The support domain this chunk attends to, per query row, plus
                    # the context width the gather buffer was sized for. Those are
                    # the two quantities every sparse-MLA launch sizes itself from.
                    chunk_positions = positions[query_start:query_end]
                    if combined_indices.shape[0] == chunk_positions.numel():
                        row_probe.capture(
                            f"{self.prefix}.prefill.support_indices",
                            combined_indices,
                            chunk_positions,
                        )
                        row_probe.capture(
                            f"{self.prefix}.prefill.support_lens",
                            combined_lens,
                            chunk_positions,
                        )
                    row_probe.capture(
                        f"{self.prefix}.prefill.context_rows",
                        torch.full(
                            (chunk_positions.numel(), 1),
                            float(N),
                            dtype=torch.float32,
                            device=q.device,
                        ),
                        chunk_positions,
                    )
                with optional_phase_profile_stage("sparse_mla.prefill", q):
                    rocm_sparse_attn_prefill(
                        q=q[query_start:query_end],
                        kv=kv.view(-1, 1, q.shape[-1]),
                        indices=combined_indices,
                        topk_length=combined_lens,
                        scale=self.scale,
                        head_dim=self.head_dim,
                        nope_head_dim=self.nope_head_dim,
                        rope_head_dim=self.rope_head_dim,
                        attn_sink=None if split_k else self.attn_sink,
                        output=output[query_start:query_end],
                        partial_out=(
                            (part_m, part_l, part_acc, split_id) if split_k else None
                        ),
                        direct_kv=(
                            (
                                compressed_k_cache,
                                direct_block_table[chunk_start:chunk_end],
                                seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                                direct_block_size,
                                SPARSE_DIRECT_KV_TAG_BASE,
                                SPARSE_DIRECT_KV_REQ_STRIDE,
                                (
                                    swa_k_cache,
                                    direct_swa_block_table[chunk_start:chunk_end],
                                    seq_lens[chunk_start:chunk_end],
                                    swa_metadata.block_size,
                                    SPARSE_DIRECT_KV_SWA_TAG_BASE,
                                ),
                            )
                            if use_direct_kv
                            else None
                        ),
                    )
            if split_k:
                with optional_phase_profile_stage("sparse_mla.reduce_splits", q):
                    rocm_sparse_attn_reduce_partials(
                        part_m,
                        part_l,
                        part_acc,
                        self.attn_sink,
                        output[query_start:query_end],
                        self.n_heads,
                        self.nope_head_dim + self.rope_head_dim,
                    )
