# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import cast

import torch

from vllm.config import SpeculativeConfig
from vllm.models.deepseek_v4.eager_scratch import DeepseekV4EagerScratchPool
from vllm.models.deepseek_v4.nvidia.model import _needs_pre_hc_hidden_states


def test_indexer_q_scratch_covers_fp8_and_fp4() -> None:
    pool = DeepseekV4EagerScratchPool(
        max_num_tokens=4,
        padded_q_heads=8,
        q_head_dim=16,
        index_q_heads=2,
        index_q_head_dim=32,
        index_q_dtype=torch.float8_e4m3fn,
        index_topk=8,
        device="cpu",
    )

    fp8_values, fp8_weights = pool.indexer_q_outputs(3, use_fp4=False)
    fp4_values, fp4_scales, fp4_weights = pool.indexer_q_outputs(3, use_fp4=True)

    assert fp8_values.shape == (3, 2, 32)
    assert fp8_values.dtype == torch.float8_e4m3fn
    assert fp8_weights.shape == (3, 2)
    assert fp4_values.shape == (3, 2, 16)
    assert fp4_scales.shape == (3, 2, 1)
    assert fp4_weights.shape == (3, 2)
    assert pool.indexer_q_outputs(3, use_fp4=False)[0] is fp8_values
    assert pool.indexer_q_outputs(3, use_fp4=True)[0] is fp4_values


class _SpecConfig:
    def __init__(self, *, eagle: bool, dspark: bool, draft_model: bool) -> None:
        self._eagle = eagle
        self._dspark = dspark
        self._draft_model = draft_model

    def use_eagle(self) -> bool:
        return self._eagle

    def use_dspark(self) -> bool:
        return self._dspark

    def uses_draft_model(self) -> bool:
        return self._draft_model


def test_pre_hc_hidden_states_skip_dspark_only() -> None:
    dspark = _SpecConfig(eagle=True, dspark=True, draft_model=False)
    mtp = _SpecConfig(eagle=True, dspark=False, draft_model=False)
    generic_draft = _SpecConfig(eagle=False, dspark=False, draft_model=True)

    assert not _needs_pre_hc_hidden_states(None)
    assert not _needs_pre_hc_hidden_states(cast(SpeculativeConfig, dspark))
    assert _needs_pre_hc_hidden_states(cast(SpeculativeConfig, mtp))
    assert _needs_pre_hc_hidden_states(cast(SpeculativeConfig, generic_draft))
