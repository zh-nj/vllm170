# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

pytestmark = pytest.mark.cpu_test


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
