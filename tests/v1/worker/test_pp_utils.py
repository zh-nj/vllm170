# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.pp_utils import (
    _snapshot_draft_tokens_for_pp,
    compute_need_sampled_mask,
    pack_speculative_pp_relay,
    pad_sampled_tokens,
    unpack_speculative_pp_relay,
)


def _batch(num_computed, prefill_len, num_scheduled):
    return Mock(
        num_reqs=len(num_computed),
        num_computed_tokens_np=np.array(num_computed, dtype=np.int32),
        prefill_len_np=np.array(prefill_len, dtype=np.int32),
        num_scheduled_tokens=np.array(num_scheduled, dtype=np.int32),
    )


def test_need_sampled_mask_excludes_non_final_prefill_chunks():
    batch = _batch(
        num_computed=[512, 1000],
        prefill_len=[4096, 1004],
        num_scheduled=[448, 4],
    )

    mask = compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [False, True]


def test_need_sampled_mask_returns_none_when_no_row_samples():
    batch = _batch(
        num_computed=[0, 512],
        prefill_len=[4096, 4096],
        num_scheduled=[448, 448],
    )

    assert compute_need_sampled_mask(batch) is None


def test_need_sampled_mask_keeps_decode_past_length_cap():
    """Decode rows must be relayed while the scheduler still runs them."""
    batch = _batch(
        num_computed=[14176],
        prefill_len=[12175],
        num_scheduled=[8],
    )

    mask = compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True]


def test_need_sampled_mask_handles_decode_before_prefill_chunk():
    batch = _batch(
        num_computed=[10, 512],
        prefill_len=[8, 4096],
        num_scheduled=[1, 448],
    )

    mask = compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True, False]


def test_pad_sampled_tokens_matches_fixed_collective_width():
    sampled = torch.tensor([[11], [22]], dtype=torch.int64)

    padded = pad_sampled_tokens(sampled, 4)

    assert padded.tolist() == [[11, -1, -1, -1], [22, -1, -1, -1]]
    assert padded.dtype == torch.int64


def test_pad_sampled_tokens_preserves_full_width_buffer():
    sampled = torch.tensor([[11, 12, 13]], dtype=torch.int64)

    assert pad_sampled_tokens(sampled, 3) is sampled


def test_pad_sampled_tokens_rejects_collective_count_mismatch():
    sampled = torch.tensor([[11, 12]], dtype=torch.int64)

    with pytest.raises(ValueError, match="exceeds PP buffer width"):
        pad_sampled_tokens(sampled, 1)


def test_snapshot_draft_tokens_for_pp_owns_storage():
    draft_tokens = torch.tensor([[11, 12, 13]], dtype=torch.int64)

    snapshot = _snapshot_draft_tokens_for_pp(draft_tokens)
    draft_tokens.fill_(-1)

    assert snapshot.tolist() == [[11, 12, 13]]
    assert snapshot.data_ptr() != draft_tokens.data_ptr()


def test_speculative_pp_relay_round_trip_uses_one_fixed_width_buffer():
    sampled = torch.tensor([[21], [31]], dtype=torch.int64)
    drafts = torch.tensor([[22, 23, 24], [32, 33, 34]], dtype=torch.int64)
    num_sampled = torch.tensor([2, 4], dtype=torch.int32)
    num_rejected = torch.tensor([2, 0], dtype=torch.int32)

    relay = pack_speculative_pp_relay(
        sampled, drafts, num_sampled, num_rejected, max_sample_len=4
    )
    unpacked = unpack_speculative_pp_relay(relay, max_sample_len=4)

    assert relay.shape == (2, 9)
    assert unpacked[0].tolist() == [[21, -1, -1, -1], [31, -1, -1, -1]]
    assert unpacked[1].tolist() == [2, 4]
    assert unpacked[2].tolist() == [2, 0]
    assert unpacked[3].tolist() == [[22, 23, 24], [32, 33, 34]]


def test_unpack_speculative_pp_relay_returns_contiguous_counts():
    relay = torch.arange(18, dtype=torch.int64).reshape(2, 9)

    _, num_sampled, num_rejected, _ = unpack_speculative_pp_relay(
        relay, max_sample_len=4
    )

    assert num_sampled.is_contiguous()
    assert num_rejected.is_contiguous()
