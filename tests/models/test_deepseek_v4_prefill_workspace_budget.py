# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.models.deepseek_v4.attention import (
    get_sparse_prefill_reqs_per_chunk,
)


@pytest.mark.parametrize(
    ("num_kv_tokens", "budget_mb", "expected"),
    [
        (65_536, 512, 4),
        (131_072, 512, 4),
        (262_144, 512, 2),
        (524_288, 512, 1),
        (65_536, 128, 2),
        (1_048_576, 128, 1),
        (1_048_576, 0, 4),
    ],
)
def test_sparse_prefill_reqs_follow_byte_budget(
    num_kv_tokens: int,
    budget_mb: int,
    expected: int,
) -> None:
    assert (
        get_sparse_prefill_reqs_per_chunk(
            max_reqs=4,
            num_kv_tokens=num_kv_tokens,
            head_dim=512,
            budget_mb=budget_mb,
        )
        == expected
    )


def test_sparse_prefill_workspace_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="positive"):
        get_sparse_prefill_reqs_per_chunk(
            max_reqs=4,
            num_kv_tokens=0,
            head_dim=512,
            budget_mb=512,
        )
