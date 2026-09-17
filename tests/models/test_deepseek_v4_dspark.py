# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm.config import KernelConfig, SpeculativeConfig, VllmConfig
from vllm.config.speculative import verify_distributed_dspark_stage_ranks
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.models.deepseek_v4.nvidia.dspark import DSparkDeepseekV4ForCausalLM
from vllm.models.deepseek_v4.nvidia.model import _needs_pre_hc_hidden_states
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import (
    _get_dspark_kernel_config,
    _is_materialized_embedding,
)


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


def test_pp_missing_layer_is_not_a_materialized_embedding() -> None:
    assert _is_materialized_embedding(torch.nn.Embedding(4, 8))
    assert not _is_materialized_embedding(PPMissingLayer())


@pytest.mark.cpu_test
def test_dspark_uses_independent_moe_backend() -> None:
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(moe_backend="marlin"),
        kernel_config=KernelConfig(moe_backend="mxfp4_int8_sm80"),
    )

    draft_kernel_config = _get_dspark_kernel_config(cast(VllmConfig, vllm_config))

    assert draft_kernel_config.moe_backend == "marlin"
    assert vllm_config.kernel_config.moe_backend == "mxfp4_int8_sm80"


def test_distributed_dspark_stage_ranks_accept_pp1_through_pp3() -> None:
    verify_distributed_dspark_stage_ranks([1, 2, 3], 3, 5)


@pytest.mark.parametrize(
    ("computed", "scheduled", "prefill", "expected"),
    [
        ([0], [4], [10], True),
        ([0], [10], [10], False),
        ([0, 10], [4, 1], [10, 11], False),
        ([], [], [], False),
    ],
)
def test_distributed_dspark_skips_only_intermediate_prefill(
    computed: list[int],
    scheduled: list[int],
    prefill: list[int],
    expected: bool,
) -> None:
    input_batch = SimpleNamespace(
        num_reqs=len(computed),
        num_computed_prefill_tokens_np=np.asarray(computed, dtype=np.int32),
        num_scheduled_tokens=np.asarray(scheduled, dtype=np.int32),
        prefill_len_np=np.asarray(prefill, dtype=np.int32),
    )

    assert GPUModelRunner._is_all_reqs_chunked_prefill(input_batch) is expected


@pytest.mark.parametrize(
    "stage_ranks",
    ([1, 2], [1, 1, 3], [2, 1, 3], [1, 2, 4], [-1, 1, 2]),
)
def test_distributed_dspark_stage_ranks_reject_invalid_layout(stage_ranks) -> None:
    with pytest.raises(ValueError):
        verify_distributed_dspark_stage_ranks(stage_ranks, 3, 5)


def test_distributed_dspark_accepts_matching_batch_identity() -> None:
    speculator = DSparkSpeculator.__new__(DSparkSpeculator)
    speculator._distributed_step_id = 17
    input_batch = SimpleNamespace(req_ids=["req-a", "req-b"], num_reqs=2)

    speculator._validate_distributed_package(
        {
            "scheduler_step_id": 17,
            "dummy_run": False,
            "req_ids": ("req-a", "req-b"),
            "num_reqs": 2,
        },
        input_batch,
    )


def test_distributed_dspark_ignores_local_dummy_request_ids() -> None:
    speculator = DSparkSpeculator.__new__(DSparkSpeculator)
    speculator._distributed_step_id = -1
    input_batch = SimpleNamespace(req_ids=["local-dummy"], num_reqs=1)

    speculator._validate_distributed_package(
        {
            "scheduler_step_id": -1,
            "dummy_run": True,
            "req_ids": ("remote-dummy",),
            "num_reqs": 1,
        },
        input_batch,
    )


@pytest.mark.parametrize(
    ("package", "match"),
    [
        (
            {
                "scheduler_step_id": 18,
                "dummy_run": False,
                "req_ids": ("req-a",),
                "num_reqs": 1,
            },
            "scheduler step mismatch",
        ),
        (
            {
                "scheduler_step_id": 17,
                "dummy_run": False,
                "req_ids": ("req-b",),
                "num_reqs": 1,
            },
            "request batch mismatch",
        ),
        (
            {
                "scheduler_step_id": 17,
                "dummy_run": False,
                "req_ids": ("req-a",),
                "num_reqs": 2,
            },
            "request count mismatch",
        ),
        (
            {
                "scheduler_step_id": 17,
                "dummy_run": True,
                "req_ids": ("req-a",),
                "num_reqs": 1,
            },
            "dummy-run marker mismatch",
        ),
    ],
)
def test_distributed_dspark_rejects_batch_identity_drift(
    package: dict[str, object], match: str
) -> None:
    speculator = DSparkSpeculator.__new__(DSparkSpeculator)
    speculator._distributed_step_id = 17
    input_batch = SimpleNamespace(req_ids=["req-a"], num_reqs=1)

    with pytest.raises(RuntimeError, match=match):
        speculator._validate_distributed_package(package, input_batch)
