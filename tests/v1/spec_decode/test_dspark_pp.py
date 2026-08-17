# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
    DSparkSpeculator,
    _collect_q1_drafter_audit_records,
    _get_q1_drafter_audit_eligible_rows,
)

pytestmark = pytest.mark.cpu_test


def test_q1_drafter_audit_skips_only_nonfinal_prefill() -> None:
    input_batch = type(
        "InputBatchStub",
        (),
        {
            "num_computed_prefill_tokens_np": np.array([0, 8, 0]),
            "num_scheduled_tokens": np.array([8, 2, 1]),
            "prefill_len_np": np.array([16, 10, 0]),
            "is_prefilling_np": np.array([True, True, False]),
        },
    )()

    np.testing.assert_array_equal(
        _get_q1_drafter_audit_eligible_rows(input_batch),
        np.array([False, True, True]),
    )


def test_collect_q1_drafter_audit_records() -> None:
    records = _collect_q1_drafter_audit_records(
        base_logits=torch.tensor([[1.0, 4.0, 3.0, 0.0], [4.0, 3.0, 2.0, 1.0]]),
        markov_bias=torch.tensor(
            [[0.0, -1.0, 2.0, -2.0], [0.0, 0.0, 0.0, 0.0]]
        ),
        final_logits=torch.tensor([[1.0, 3.0, 5.0, -2.0], [4.0, 3.0, 2.0, 1.0]]),
        draft_ids=torch.tensor([2, 0]),
        previous_ids=torch.tensor([7, 8]),
        positions=torch.tensor([100, 200]),
        temperatures=torch.tensor([0.0, 0.0]),
        eligible_rows=torch.tensor([True, False]),
        max_rows=2,
    )

    assert records == [
        {
            "request_index": 0,
            "position": 100,
            "predicted_position": 101,
            "previous_token_id": 7,
            "draft_token_id": 2,
            "base_top1_id": 1,
            "base_top2_id": 2,
            "bias_top1_id": 2,
            "bias_top2_id": 0,
            "final_top1_id": 2,
            "final_top2_id": 1,
            "draft_base_rank": 2,
            "draft_base_rank_end": 2,
            "base_top1_final_rank": 2,
            "base_top1_final_rank_end": 2,
            "temperature": 0.0,
            "draft_base_logit": 3.0,
            "draft_bias": 2.0,
            "draft_final_logit": 5.0,
            "base_top1_logit": 4.0,
            "base_top2_logit": 3.0,
            "base_top2_margin": 1.0,
            "bias_top1_logit": 2.0,
            "bias_top2_logit": 0.0,
            "final_top1_logit": 5.0,
            "final_top2_logit": 3.0,
            "final_top2_margin": 2.0,
            "base_top1_bias": -1.0,
            "base_top1_final_logit": 3.0,
            "markov_changed_argmax": True,
            "sample_matches_final_top1": True,
            "sample_matches_final_max": True,
            "markov_scale_000_top1_id": 1,
            "markov_scale_050_top1_id": 2,
            "markov_scale_075_top1_id": 2,
            "markov_scale_100_top1_id": 2,
            "markov_scale_125_top1_id": 2,
            "markov_scale_150_top1_id": 2,
        }
    ]


def test_dspark_packs_aux_states_into_pp_workspace():
    class CombineModel:
        packed_hidden_states: torch.Tensor | None = None

        def combine_hidden_states(
            self, packed_hidden_states: torch.Tensor
        ) -> torch.Tensor:
            self.packed_hidden_states = packed_hidden_states
            width = packed_hidden_states.shape[-1] // 2
            return packed_hidden_states[:, :width] + packed_hidden_states[:, width:]

    speculator = DSparkSpeculator.__new__(DSparkSpeculator)
    speculator.model = CombineModel()
    speculator.dtype = torch.float32
    speculator.device = torch.device("cpu")
    speculator._target_hidden_states_workspace = None
    workspace = torch.empty(32)
    assert speculator.set_target_hidden_states_workspace(workspace)
    aux_hidden_states = [
        torch.arange(6, dtype=torch.float32).view(3, 2),
        torch.full((3, 2), 10.0),
    ]

    hidden_states = speculator._prepare_context_hidden_states(
        torch.empty((3, 2)),
        aux_hidden_states,
        num_target_tokens=2,
    )

    packed = speculator.model.packed_hidden_states
    assert packed is not None
    assert packed.data_ptr() == speculator._target_hidden_states_workspace.data_ptr()
    torch.testing.assert_close(hidden_states, (aux_hidden_states[0] + 10)[:2])


def test_dspark_without_workspace_uses_draft_width_buffer():
    class CombineModel:
        def combine_hidden_states(
            self, packed_hidden_states: torch.Tensor
        ) -> torch.Tensor:
            width = packed_hidden_states.shape[-1] // 2
            return packed_hidden_states[:, :width] + packed_hidden_states[:, width:]

    speculator = DSparkSpeculator.__new__(DSparkSpeculator)
    speculator.model = CombineModel()
    speculator.hidden_states = torch.empty((3, 2))
    speculator._target_hidden_states_workspace = None
    aux_hidden_states = [
        torch.arange(6, dtype=torch.float32).view(3, 2),
        torch.full((3, 2), 10.0),
    ]

    hidden_states = speculator._prepare_context_hidden_states(
        torch.empty((3, 2)),
        aux_hidden_states,
        num_target_tokens=2,
    )

    assert hidden_states.data_ptr() == speculator.hidden_states.data_ptr()
    torch.testing.assert_close(hidden_states, (aux_hidden_states[0] + 10)[:2])


def test_dspark_rejects_incompatible_pp_workspace():
    speculator = DSparkSpeculator.__new__(DSparkSpeculator)
    speculator.dtype = torch.float16
    speculator.device = torch.device("cpu")
    speculator._target_hidden_states_workspace = None

    assert not speculator.set_target_hidden_states_workspace(torch.empty(4))
    assert speculator._target_hidden_states_workspace is None
