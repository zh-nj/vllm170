# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import cast

import pytest
import torch
import torch.nn as nn

from vllm.config import SpeculativeConfig
from vllm.models.deepseek_v4.nvidia.dspark import DSparkDeepseekV4ForCausalLM
from vllm.models.deepseek_v4.nvidia.model import _needs_pre_hc_hidden_states


def _make_draft(confidence_head: nn.Module | None) -> DSparkDeepseekV4ForCausalLM:
    draft = DSparkDeepseekV4ForCausalLM.__new__(DSparkDeepseekV4ForCausalLM)
    nn.Module.__init__(draft)
    draft.model = SimpleNamespace(confidence_head=confidence_head)
    return draft


@pytest.mark.cpu_test
def test_dspark_loads_confidence_weight_only_when_audit_head_exists() -> None:
    checkpoint_name = "mtp.2.confidence_head.proj.weight"

    disabled = _make_draft(None)
    assert disabled._remap_dspark_name(checkpoint_name) is None

    enabled = _make_draft(nn.Identity())
    assert (
        enabled._remap_dspark_name(checkpoint_name)
        == "model.confidence_head.proj.weight"
    )


@pytest.mark.cpu_test
def test_dspark_confidence_uses_hidden_and_markov_features_in_fp32() -> None:
    class ConfidenceHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(3, 1, bias=False, dtype=torch.float32)

        def forward(
            self, hidden: torch.Tensor, markov_embed: torch.Tensor
        ) -> torch.Tensor:
            inputs = torch.cat([hidden, markov_embed], dim=-1).float()
            return self.proj(inputs).squeeze(-1)

    head = ConfidenceHead()
    with torch.no_grad():
        head.proj.weight.copy_(torch.tensor([[1.0, 2.0, 3.0]]))
    draft = _make_draft(head)

    confidence = draft.compute_confidence(
        torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
        torch.tensor([[3.0]], dtype=torch.bfloat16),
    )

    torch.testing.assert_close(confidence, torch.sigmoid(torch.tensor([14.0])))


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
