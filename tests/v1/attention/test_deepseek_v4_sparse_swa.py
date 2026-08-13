# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla import sparse_swa


@pytest.mark.skipif(
    not (current_platform.is_cuda() and not current_platform.has_device_capability(90)),
    reason="SM80 CUDA is required",
)
def test_sm80_tile_scheduler_does_not_require_flashmla(monkeypatch):
    get_mla_metadata = Mock(side_effect=AssertionError("FlashMLA was called"))
    monkeypatch.setattr(sparse_swa, "get_mla_metadata", get_mla_metadata)
    builder = sparse_swa.DeepseekSparseSWAMetadataBuilder.__new__(
        sparse_swa.DeepseekSparseSWAMetadataBuilder
    )
    builder._layer_types = {
        sparse_swa._LAYER_TYPE_SWAONLY,
        sparse_swa._LAYER_TYPE_C4A,
        sparse_swa._LAYER_TYPE_C128A,
    }

    schedulers = builder.build_tile_scheduler(num_decode_tokens=1)

    assert schedulers == {
        sparse_swa._LAYER_TYPE_SWAONLY: None,
        sparse_swa._LAYER_TYPE_C4A: None,
        sparse_swa._LAYER_TYPE_C128A: None,
    }
    get_mla_metadata.assert_not_called()
