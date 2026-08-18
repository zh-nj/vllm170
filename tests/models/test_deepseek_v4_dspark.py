# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.models.deepseek_v4.nvidia.dspark import DSparkDeepseekV4ForCausalLM


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

    enabled = _make_draft(nn.Linear(3, 1, bias=False, dtype=torch.float32))
    assert (
        enabled._remap_dspark_name(checkpoint_name)
        == "model.confidence_head.weight"
    )


@pytest.mark.cpu_test
def test_dspark_confidence_uses_hidden_and_markov_features_in_fp32() -> None:
    head = nn.Linear(3, 1, bias=False, dtype=torch.float32)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 2.0, 3.0]]))
    draft = _make_draft(head)

    confidence = draft.compute_draft_confidence(
        torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
        torch.tensor([[3.0]], dtype=torch.bfloat16),
    )

    torch.testing.assert_close(confidence, torch.tensor([14.0]))
