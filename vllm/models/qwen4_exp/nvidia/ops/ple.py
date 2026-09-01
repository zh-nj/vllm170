# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA kernels for CPU-offloaded Qwen4Exp PLE embeddings."""

import torch

from vllm.triton_utils import tl, triton

_PLE_CHANNELS = 10240
_SHORT_CONV_KERNEL_SIZE = 4
_SHORT_CONV_DILATION = 3
_SHORT_CONV_STATE_LEN = 9
_NGRAM_SIZE = 3
_HEADS_PER_NGRAM = 8
_NGRAM_HEADS = 16
_SPEC_QUERY_LEN = 4


@triton.jit
def _round_bf16_to_fp32(value):
    bits = value.to(tl.int32, bitcast=True)
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded_bits = (bits + rounding_bias) & -65536
    return rounded_bits.to(tl.float32, bitcast=True)


@triton.jit
def _ple_ngram_hash_from_tokens_kernel(
    input_ids_ptr,
    ngram_context_ptr,
    request_indices_ptr,
    token_offsets_ptr,
    multipliers_ptr,
    vocab_sizes_ptr,
    head_offsets_ptr,
    output_ptr,
    num_outputs,
    eos_token_id,
    stride_context_row,
    stride_context_col,
    HEADS_PER_NGRAM: tl.constexpr,
    NGRAM_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    output_index = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = output_index < num_outputs
    token_index = output_index // NGRAM_HEADS
    head_index = output_index % NGRAM_HEADS
    request_index = tl.load(request_indices_ptr + token_index, mask=mask, other=0)
    token_offset = tl.load(token_offsets_ptr + token_index, mask=mask, other=0)
    context_base = request_index * stride_context_row

    current = tl.load(input_ids_ptr + token_index, mask=mask, other=eos_token_id)
    previous_1 = tl.load(
        input_ids_ptr + token_index - 1,
        mask=mask & (token_offset >= 1),
        other=eos_token_id,
    )
    previous_1 = tl.where(
        token_offset >= 1,
        previous_1,
        tl.load(
            ngram_context_ptr + context_base + stride_context_col,
            mask=mask,
            other=eos_token_id,
        ),
    )
    previous_2 = tl.load(
        input_ids_ptr + token_index - 2,
        mask=mask & (token_offset >= 2),
        other=eos_token_id,
    )
    context_2 = tl.load(
        ngram_context_ptr
        + context_base
        + tl.minimum(token_offset, 1) * stride_context_col,
        mask=mask & (token_offset < 2),
        other=eos_token_id,
    )
    previous_2 = tl.where(token_offset >= 2, previous_2, context_2)
    previous_2 = tl.where(
        (previous_2 == eos_token_id) | (previous_1 == eos_token_id),
        eos_token_id,
        previous_2,
    )

    multiplier_0 = tl.load(multipliers_ptr)
    multiplier_1 = tl.load(multipliers_ptr + 1)
    multiplier_2 = tl.load(multipliers_ptr + 2)
    mixed = (current * multiplier_0) ^ (previous_1 * multiplier_1)
    mixed_3 = mixed ^ (previous_2 * multiplier_2)
    mixed = tl.where(head_index < HEADS_PER_NGRAM, mixed, mixed_3)
    vocab_size = tl.load(vocab_sizes_ptr + head_index, mask=mask, other=1)
    head_offset = tl.load(head_offsets_ptr + head_index, mask=mask, other=0)
    tl.store(
        output_ptr + output_index,
        mixed % vocab_size + head_offset,
        mask=mask,
    )


def can_use_fused_ple_ngram_hash(
    input_ids: torch.Tensor,
    ngram_context: torch.Tensor,
    request_indices: torch.Tensor,
    token_offsets: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    head_offsets: torch.Tensor,
) -> bool:
    """Return whether inputs match the fixed Qwen4Exp n-gram layout."""
    num_tokens = input_ids.numel()
    num_requests = ngram_context.shape[0] if ngram_context.ndim == 2 else -1
    return (
        input_ids.is_cuda
        and input_ids.dtype == torch.int64
        and input_ids.ndim == 1
        and input_ids.is_contiguous()
        and ngram_context.is_cuda
        and ngram_context.dtype == torch.int64
        and ngram_context.shape == (num_requests, _NGRAM_SIZE - 1)
        and ngram_context.is_contiguous()
        and request_indices.is_cuda
        and request_indices.dtype == torch.int64
        and request_indices.shape == (num_tokens,)
        and request_indices.is_contiguous()
        and token_offsets.is_cuda
        and token_offsets.dtype == torch.int64
        and token_offsets.shape == (num_tokens,)
        and token_offsets.is_contiguous()
        and multipliers.is_cuda
        and multipliers.dtype == torch.int64
        and multipliers.numel() == _NGRAM_SIZE
        and vocab_sizes.is_cuda
        and vocab_sizes.dtype == torch.int64
        and vocab_sizes.numel() == _NGRAM_HEADS
        and head_offsets.is_cuda
        and head_offsets.dtype == torch.int64
        and head_offsets.numel() == _NGRAM_HEADS
    )


def fused_ple_ngram_hash(
    input_ids: torch.Tensor,
    ngram_context: torch.Tensor,
    request_indices: torch.Tensor,
    token_offsets: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    head_offsets: torch.Tensor,
    eos_token_id: int,
) -> torch.Tensor:
    """Hash Qwen4Exp two- and three-token windows in one kernel."""
    if not can_use_fused_ple_ngram_hash(
        input_ids,
        ngram_context,
        request_indices,
        token_offsets,
        multipliers,
        vocab_sizes,
        head_offsets,
    ):
        raise ValueError("unsupported input for fused PLE n-gram hash")
    output = torch.empty(
        (input_ids.numel(), _NGRAM_HEADS),
        device=input_ids.device,
        dtype=torch.int64,
    )
    if output.numel():
        block_size = 256
        _ple_ngram_hash_from_tokens_kernel[(triton.cdiv(output.numel(), block_size),)](
            input_ids,
            ngram_context,
            request_indices,
            token_offsets,
            multipliers,
            vocab_sizes,
            head_offsets,
            output,
            output.numel(),
            eos_token_id,
            ngram_context.stride(0),
            ngram_context.stride(1),
            HEADS_PER_NGRAM=_HEADS_PER_NGRAM,
            NGRAM_HEADS=_NGRAM_HEADS,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
    return output


@triton.jit
def _ple_prefill_short_conv_kernel(
    x_ptr,
    state_ptr,
    state_indices_ptr,
    has_initial_ptr,
    request_indices_ptr,
    token_offsets_ptr,
    weight_ptr,
    output_ptr,
    num_tokens,
    stride_x_token,
    stride_x_channel,
    stride_state_slot,
    stride_state_channel,
    stride_state_col,
    stride_weight_channel,
    stride_weight_tap,
    stride_output_token,
    stride_output_channel,
    CHANNELS: tl.constexpr,
    STATE_LEN: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    DILATION: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
) -> None:
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)[:, None]
    channel = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    token_mask = token < num_tokens
    mask = token_mask & (channel < CHANNELS)
    request = tl.load(request_indices_ptr + token, mask=token_mask, other=0)
    token_offset = tl.load(token_offsets_ptr + token, mask=token_mask, other=0)
    state_index = tl.load(state_indices_ptr + request, mask=token_mask, other=0)
    valid_state = state_index != 0
    has_initial = tl.load(has_initial_ptr + request, mask=token_mask, other=0).to(
        tl.int1
    )
    has_initial &= valid_state
    acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)

    for tap in range(KERNEL_SIZE):
        lag = STATE_LEN - tap * DILATION
        from_x = token_offset >= lag
        x_value = tl.load(
            x_ptr + (token - lag) * stride_x_token + channel * stride_x_channel,
            mask=mask & from_x,
            other=0.0,
        ).to(tl.float32)
        state_col = STATE_LEN + token_offset - lag
        state_value = tl.load(
            state_ptr
            + state_index * stride_state_slot
            + channel * stride_state_channel
            + state_col * stride_state_col,
            mask=mask & ~from_x & has_initial,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr + channel * stride_weight_channel + tap * stride_weight_tap,
            mask=channel < CHANNELS,
            other=0.0,
        ).to(tl.float32)
        acc += (x_value + state_value) * weight

    conv = _round_bf16_to_fp32(acc)
    output = conv * tl.sigmoid(conv)
    tl.store(
        output_ptr + token * stride_output_token + channel * stride_output_channel,
        output,
        mask=mask & valid_state,
    )
    tl.store(
        output_ptr + token * stride_output_token + channel * stride_output_channel,
        0.0,
        mask=mask & ~valid_state,
    )


@triton.jit
def _ple_prefill_short_conv_state_kernel(
    x_ptr,
    state_ptr,
    state_indices_ptr,
    has_initial_ptr,
    query_start_loc_ptr,
    lengths_ptr,
    num_requests,
    stride_x_token,
    stride_x_channel,
    stride_state_slot,
    stride_state_channel,
    stride_state_col,
    CHANNELS: tl.constexpr,
    STATE_LEN: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
) -> None:
    request = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)[:, None]
    state_col = tl.arange(0, BLOCK_S)[None, :]
    request_mask = request < num_requests
    mask = request_mask & (channel < CHANNELS) & (state_col < STATE_LEN)
    state_index = tl.load(state_indices_ptr + request, mask=request_mask, other=0)
    valid_state = state_index != 0
    has_initial = tl.load(has_initial_ptr + request, mask=request_mask, other=0).to(
        tl.int1
    )
    query_start = tl.load(query_start_loc_ptr + request, mask=request_mask, other=0)
    length = tl.load(lengths_ptr + request, mask=request_mask, other=0)

    history_position = length + state_col
    from_x = history_position >= STATE_LEN
    old_value = tl.load(
        state_ptr
        + state_index * stride_state_slot
        + channel * stride_state_channel
        + history_position * stride_state_col,
        mask=mask & ~from_x & has_initial & valid_state,
        other=0.0,
    )
    x_offset = history_position - STATE_LEN
    x_value = tl.load(
        x_ptr + (query_start + x_offset) * stride_x_token + channel * stride_x_channel,
        mask=mask & from_x & (x_offset < length),
        other=0.0,
    )
    tl.store(
        state_ptr
        + state_index * stride_state_slot
        + channel * stride_state_channel
        + state_col * stride_state_col,
        tl.where(from_x, x_value, old_value),
        mask=mask & valid_state & (length > 0),
    )


def can_use_fused_ple_prefill_short_conv(
    x: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial: torch.Tensor,
    request_indices: torch.Tensor,
    token_offsets: torch.Tensor,
    query_start_loc: torch.Tensor,
    lengths: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    """Return whether inputs match the SM8x Qwen4Exp prefill contract."""
    num_requests = state_indices.numel()
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.shape[1] == _PLE_CHANNELS
        and x.is_contiguous()
        and state.is_cuda
        and state.dtype == x.dtype
        and state.ndim == 3
        and state.shape[1] == _PLE_CHANNELS
        and state.shape[2] >= _SHORT_CONV_STATE_LEN
        and state_indices.is_cuda
        and state_indices.dtype == torch.int64
        and state_indices.shape == (num_requests,)
        and state_indices.is_contiguous()
        and has_initial.is_cuda
        and has_initial.dtype == torch.bool
        and has_initial.shape == (num_requests,)
        and has_initial.is_contiguous()
        and request_indices.is_cuda
        and request_indices.dtype == torch.int64
        and request_indices.shape == (x.shape[0],)
        and request_indices.is_contiguous()
        and token_offsets.is_cuda
        and token_offsets.dtype == torch.int64
        and token_offsets.shape == (x.shape[0],)
        and token_offsets.is_contiguous()
        and query_start_loc.is_cuda
        and query_start_loc.dtype == torch.int64
        and query_start_loc.shape == (num_requests + 1,)
        and query_start_loc.is_contiguous()
        and lengths.is_cuda
        and lengths.dtype == torch.int64
        and lengths.shape == (num_requests,)
        and lengths.is_contiguous()
        and weight.is_cuda
        and weight.dtype == x.dtype
        and weight.shape == (_PLE_CHANNELS, _SHORT_CONV_KERNEL_SIZE)
        and weight.is_contiguous()
        and torch.cuda.get_device_capability(x.device)[0] == 8
    )


def fused_ple_prefill_short_conv(
    x: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial: torch.Tensor,
    request_indices: torch.Tensor,
    token_offsets: torch.Tensor,
    query_start_loc: torch.Tensor,
    lengths: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Apply PLE prefill short-conv and commit its state without packing."""
    if not can_use_fused_ple_prefill_short_conv(
        x,
        state,
        state_indices,
        has_initial,
        request_indices,
        token_offsets,
        query_start_loc,
        lengths,
        weight,
    ):
        raise ValueError("unsupported input for fused PLE prefill short-conv")

    output = torch.empty_like(x)
    if x.shape[0]:
        block_t = 8
        block_c = 128
        _ple_prefill_short_conv_kernel[
            (
                triton.cdiv(x.shape[0], block_t),
                triton.cdiv(x.shape[1], block_c),
            )
        ](
            x,
            state,
            state_indices,
            has_initial,
            request_indices,
            token_offsets,
            weight,
            output,
            x.shape[0],
            x.stride(0),
            x.stride(1),
            state.stride(0),
            state.stride(1),
            state.stride(2),
            weight.stride(0),
            weight.stride(1),
            output.stride(0),
            output.stride(1),
            CHANNELS=_PLE_CHANNELS,
            STATE_LEN=_SHORT_CONV_STATE_LEN,
            KERNEL_SIZE=_SHORT_CONV_KERNEL_SIZE,
            DILATION=_SHORT_CONV_DILATION,
            BLOCK_T=block_t,
            BLOCK_C=block_c,
            num_warps=4,
        )

    num_requests = state_indices.numel()
    if num_requests:
        block_c = 256
        block_s = triton.next_power_of_2(_SHORT_CONV_STATE_LEN)
        _ple_prefill_short_conv_state_kernel[
            (num_requests, triton.cdiv(x.shape[1], block_c))
        ](
            x,
            state,
            state_indices,
            has_initial,
            query_start_loc,
            lengths,
            num_requests,
            x.stride(0),
            x.stride(1),
            state.stride(0),
            state.stride(1),
            state.stride(2),
            CHANNELS=_PLE_CHANNELS,
            STATE_LEN=_SHORT_CONV_STATE_LEN,
            BLOCK_C=block_c,
            BLOCK_S=block_s,
            num_warps=8,
        )
    return output


@triton.jit
def _ple_spec_short_conv_kernel(
    x_ptr,
    state_ptr,
    state_indices_ptr,
    query_start_loc_ptr,
    num_accepted_ptr,
    weight_ptr,
    output_ptr,
    stride_x_token,
    stride_x_channel,
    stride_state_slot,
    stride_state_channel,
    stride_state_col,
    stride_weight_channel,
    stride_weight_tap,
    stride_output_token,
    stride_output_channel,
    state_capacity,
    CHANNELS: tl.constexpr,
    STATE_LEN: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    DILATION: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
) -> None:
    channel = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    token_offset = tl.arange(0, QUERY_LEN)[:, None]
    channel_mask = channel < CHANNELS
    state_index = tl.load(state_indices_ptr)
    valid_state = state_index != 0
    query_start = tl.load(query_start_loc_ptr)
    query_end = tl.load(query_start_loc_ptr + 1)
    query_length = tl.minimum(tl.maximum(query_end - query_start, 0), QUERY_LEN)
    rollback = tl.minimum(tl.maximum(tl.load(num_accepted_ptr) - 1, 0), QUERY_LEN - 1)
    valid_token = token_offset < query_length
    acc = tl.zeros((QUERY_LEN, BLOCK_C), dtype=tl.float32)

    for tap in range(KERNEL_SIZE):
        history_position = token_offset + tap * DILATION
        from_x = history_position >= STATE_LEN
        x_offset = history_position - STATE_LEN
        x_value = tl.load(
            x_ptr
            + (query_start + x_offset) * stride_x_token
            + channel * stride_x_channel,
            mask=channel_mask
            & valid_token
            & from_x
            & (x_offset >= 0)
            & (x_offset < query_length),
            other=0.0,
        ).to(tl.float32)
        state_col = rollback + history_position
        state_value = tl.load(
            state_ptr
            + state_index * stride_state_slot
            + channel * stride_state_channel
            + state_col * stride_state_col,
            mask=channel_mask
            & valid_token
            & ~from_x
            & valid_state
            & (state_col < state_capacity),
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr + channel * stride_weight_channel + tap * stride_weight_tap,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        acc += (x_value + state_value) * weight

    conv = _round_bf16_to_fp32(acc)
    output = conv * tl.sigmoid(conv)
    tl.store(
        output_ptr
        + (query_start + token_offset) * stride_output_token
        + channel * stride_output_channel,
        tl.where(valid_token & valid_state, output, 0.0),
        mask=channel_mask,
    )

    state_col = tl.arange(0, BLOCK_STATE)[:, None]
    source_position = state_col + 1
    from_x = source_position >= STATE_LEN
    x_offset = source_position - STATE_LEN
    x_value = tl.load(
        x_ptr + (query_start + x_offset) * stride_x_token + channel * stride_x_channel,
        mask=channel_mask & from_x & (x_offset >= 0) & (x_offset < query_length),
        other=0.0,
    )
    source_col = rollback + source_position
    state_value = tl.load(
        state_ptr
        + state_index * stride_state_slot
        + channel * stride_state_channel
        + source_col * stride_state_col,
        mask=channel_mask & ~from_x & valid_state & (source_col < state_capacity),
        other=0.0,
    )
    update_length = STATE_LEN + query_length - 1
    tl.store(
        state_ptr
        + state_index * stride_state_slot
        + channel * stride_state_channel
        + state_col * stride_state_col,
        tl.where(from_x, x_value, state_value),
        mask=channel_mask
        & valid_state
        & (state_col < update_length)
        & (state_col < state_capacity),
    )


def can_use_fused_ple_spec_short_conv(
    x: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    weight: torch.Tensor,
    spec_query_len: int,
) -> bool:
    """Return whether inputs match the single-request MTP3 PLE contract."""
    return (
        spec_query_len == _SPEC_QUERY_LEN
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.shape == (_SPEC_QUERY_LEN, _PLE_CHANNELS)
        and x.is_contiguous()
        and state.is_cuda
        and state.dtype == x.dtype
        and state.ndim == 3
        and state.shape[1] == _PLE_CHANNELS
        and state.shape[2] >= _SHORT_CONV_STATE_LEN + _SPEC_QUERY_LEN - 1
        and state_indices.is_cuda
        and state_indices.dtype in (torch.int32, torch.int64)
        and state_indices.shape == (1,)
        and state_indices.is_contiguous()
        and query_start_loc.is_cuda
        and query_start_loc.dtype in (torch.int32, torch.int64)
        and query_start_loc.shape == (2,)
        and query_start_loc.is_contiguous()
        and num_accepted_tokens.is_cuda
        and num_accepted_tokens.dtype in (torch.int32, torch.int64)
        and num_accepted_tokens.numel() >= 1
        and num_accepted_tokens.is_contiguous()
        and weight.is_cuda
        and weight.dtype == x.dtype
        and weight.shape == (_PLE_CHANNELS, _SHORT_CONV_KERNEL_SIZE)
        and weight.is_contiguous()
        and torch.cuda.get_device_capability(x.device)[0] == 8
    )


def fused_ple_spec_short_conv(
    x: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    weight: torch.Tensor,
    spec_query_len: int,
) -> torch.Tensor:
    """Apply single-request MTP3 PLE short-conv and retain draft state."""
    if not can_use_fused_ple_spec_short_conv(
        x,
        state,
        state_indices,
        query_start_loc,
        num_accepted_tokens,
        weight,
        spec_query_len,
    ):
        raise ValueError("unsupported input for fused PLE speculative short-conv")
    output = torch.empty_like(x)
    block_c = 128
    _ple_spec_short_conv_kernel[(triton.cdiv(x.shape[1], block_c),)](
        x,
        state,
        state_indices,
        query_start_loc,
        num_accepted_tokens,
        weight,
        output,
        x.stride(0),
        x.stride(1),
        state.stride(0),
        state.stride(1),
        state.stride(2),
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        state.shape[2],
        CHANNELS=_PLE_CHANNELS,
        STATE_LEN=_SHORT_CONV_STATE_LEN,
        STATE_CAPACITY=_SHORT_CONV_STATE_LEN + _SPEC_QUERY_LEN - 1,
        QUERY_LEN=_SPEC_QUERY_LEN,
        KERNEL_SIZE=_SHORT_CONV_KERNEL_SIZE,
        DILATION=_SHORT_CONV_DILATION,
        BLOCK_C=block_c,
        BLOCK_STATE=triton.next_power_of_2(_SHORT_CONV_STATE_LEN + _SPEC_QUERY_LEN - 1),
        num_warps=4,
    )
    return output


@triton.jit
def _gather_pinned_ple_embedding_kernel(
    weight_ptr,
    ids_ptr,
    output_ptr,
    embedding_dim,
    vocab_start,
    vocab_end,
    scale,
    is_fp8: tl.constexpr,
    fp8_e4b15: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    global_id = tl.load(ids_ptr + row)
    in_range = (global_id >= vocab_start) & (global_id < vocab_end)
    local_id = tl.where(in_range, global_id - vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < embedding_dim

    if is_fp8:
        if fp8_e4b15:
            raw_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
            raw = tl.load(
                raw_ptr + local_id * embedding_dim + offsets,
                mask=mask,
                other=0,
            ).to(tl.int32)
            sign = (raw >> 7) & 1
            exponent = (raw >> 3) & 0xF
            mantissa = raw & 0x7
            mantissa_fraction = mantissa.to(tl.float32) * (1.0 / 8.0)
            values = (1.0 + mantissa_fraction) * tl.exp2((exponent - 7).to(tl.float32))
            values = tl.where(
                exponent == 0,
                mantissa_fraction * (2.0**-6),
                values,
            )
            values = tl.where(
                (exponent == 0xF) & (mantissa == 0x7),
                0.0,
                values,
            )
            values = tl.where(sign != 0, -values, values).to(tl.bfloat16)
        else:
            fp8_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
            values = tl.load(
                fp8_ptr + local_id * embedding_dim + offsets,
                mask=mask,
                other=0.0,
            ).to(tl.bfloat16)
    else:
        bf16_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.bfloat16))
        values = tl.load(
            bf16_ptr + local_id * embedding_dim + offsets,
            mask=mask,
            other=0.0,
        )

    tl.store(
        output_ptr + row * embedding_dim + offsets,
        tl.where(in_range, values * scale, 0.0),
        mask=mask,
    )


def gather_pinned_ple_embedding(
    weight: torch.Tensor,
    input_ids: torch.Tensor,
    output: torch.Tensor,
    *,
    vocab_start: int,
    vocab_end: int,
    scale: float = 1.0,
) -> None:
    """Gather PLE rows from pinned host storage into a CUDA tensor."""
    if weight.device.type != "cpu" or not weight.is_pinned():
        raise ValueError("PLE UVA gather requires a pinned CPU weight")
    if weight.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError(f"unsupported PLE UVA weight dtype: {weight.dtype}")
    if not input_ids.is_cuda or not output.is_cuda:
        raise ValueError("PLE UVA gather requires CUDA input IDs and output")
    if weight.ndim != 2 or not weight.is_contiguous():
        raise ValueError("PLE UVA weight must be a contiguous matrix")
    expected_shape = (*input_ids.shape, weight.shape[1])
    if tuple(output.shape) != expected_shape or not output.is_contiguous():
        raise ValueError(
            f"PLE UVA output must be contiguous with shape {expected_shape}"
        )

    flat_ids = input_ids.reshape(-1)
    if not flat_ids.numel():
        return
    if flat_ids.dtype != torch.int64:
        raise TypeError("PLE UVA input IDs must be int64")
    device_index = input_ids.device.index or 0
    fp8_e4b15 = torch.cuda.get_device_capability(device_index) < (8, 9)
    _gather_pinned_ple_embedding_kernel[(flat_ids.numel(),)](
        weight.data_ptr(),
        flat_ids,
        output,
        weight.shape[1],
        vocab_start,
        vocab_end,
        scale,
        is_fp8=weight.dtype == torch.float8_e4m3fn,
        fp8_e4b15=fp8_e4b15,
        BLOCK_D=triton.next_power_of_2(weight.shape[1]),
    )


__all__ = [
    "can_use_fused_ple_ngram_hash",
    "can_use_fused_ple_prefill_short_conv",
    "can_use_fused_ple_spec_short_conv",
    "fused_ple_ngram_hash",
    "fused_ple_prefill_short_conv",
    "fused_ple_spec_short_conv",
    "gather_pinned_ple_embedding",
]
