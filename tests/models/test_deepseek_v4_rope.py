# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.models.deepseek_v4.common.rope import (
    resolve_deepseek_v4_rope_cache_extent,
)


@pytest.mark.parametrize(
    ("max_model_len", "model_limit", "expected"),
    [
        (131_072, 1_048_576, 131_072),
        (1_048_576, 1_048_576, 1_048_576),
        (2_000_000, 1_048_576, 1_048_576),
    ],
)
def test_rope_cache_extent_uses_deployment_limit(
    max_model_len: int,
    model_limit: int,
    expected: int,
) -> None:
    assert resolve_deepseek_v4_rope_cache_extent(max_model_len, model_limit) == expected


def test_rope_cache_extent_rejects_nonpositive_length() -> None:
    with pytest.raises(ValueError, match="positive"):
        resolve_deepseek_v4_rope_cache_extent(0, 1_048_576)
