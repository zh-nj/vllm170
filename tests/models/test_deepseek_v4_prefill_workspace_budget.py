# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.models.deepseek_v4.attention import (
    get_sparse_prefill_reqs_per_chunk as get_dsv4_reqs_per_chunk,
)
from vllm.models.deepseek_v4_1.attention import (
    get_sparse_prefill_reqs_per_chunk as get_dsv41_reqs_per_chunk,
)
from vllm.models.deepseek_v4_1.attention import (
    get_sparse_prefill_workspace_layout,
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
@pytest.mark.parametrize(
    "get_reqs_per_chunk",
    [get_dsv4_reqs_per_chunk, get_dsv41_reqs_per_chunk],
)
def test_sparse_prefill_reqs_follow_byte_budget(
    num_kv_tokens: int,
    budget_mb: int,
    expected: int,
    get_reqs_per_chunk,
) -> None:
    assert (
        get_reqs_per_chunk(
            max_reqs=4,
            num_kv_tokens=num_kv_tokens,
            head_dim=512,
            budget_mb=budget_mb,
        )
        == expected
    )


@pytest.mark.parametrize(
    "get_reqs_per_chunk",
    [get_dsv4_reqs_per_chunk, get_dsv41_reqs_per_chunk],
)
def test_sparse_prefill_workspace_rejects_invalid_dimensions(
    get_reqs_per_chunk,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        get_reqs_per_chunk(
            max_reqs=4,
            num_kv_tokens=0,
            head_dim=512,
            budget_mb=512,
        )


@pytest.mark.parametrize(
    ("num_prefills", "num_kv_tokens", "expected"),
    [
        (1, 32_768, (1, 1_048_576)),
        (4, 32_768, (4, 32_768)),
        (2, 200_000, (2, 200_000)),
        (2, 300_000, (1, 1_048_576)),
    ],
)
def test_v41_sparse_prefill_workspace_stabilizes_single_row(
    num_prefills: int,
    num_kv_tokens: int,
    expected: tuple[int, int],
) -> None:
    assert (
        get_sparse_prefill_workspace_layout(
            num_prefills=num_prefills,
            max_reqs=4,
            num_kv_tokens=num_kv_tokens,
            max_num_kv_tokens=1_048_576,
            head_dim=512,
            budget_mb=512,
        )
        == expected
    )
