# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.gpu.pp_utils import pad_sampled_tokens


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
