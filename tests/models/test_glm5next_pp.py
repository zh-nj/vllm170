# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.models.glm5next.nvidia.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
    _should_materialize_target_embedding,
)


def test_glm5next_pp_buffers_preserve_mhc_and_dflash_state() -> None:
    model = object.__new__(Glm5NextModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        hidden_size=16,
        mhc=True,
        mhc_num_residual_streams=4,
    )
    model._set_aux_hidden_state_layers((6, 15, 25, 34, 43))

    intermediate = model.make_empty_intermediate_tensors(
        batch_size=3,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    assert intermediate["hidden_states"].shape == (3, 4, 16)
    assert intermediate["aux_hidden_states"].shape == (3, 5, 16)


def test_glm5next_target_declares_pp_and_dflash_interfaces() -> None:
    assert Glm5NextForCausalLM.supports_pp
    assert Glm5NextForCausalLM.supports_eagle3
    assert Glm5NextForConditionalGeneration.supports_eagle3
    assert callable(Glm5NextForCausalLM.make_empty_intermediate_tensors)


def test_glm5next_dflash_pp_last_rank_has_target_embedding() -> None:
    dflash_config = SimpleNamespace(speculative_config=SimpleNamespace(method="dflash"))
    eagle_config = SimpleNamespace(speculative_config=SimpleNamespace(method="eagle"))

    assert _should_materialize_target_embedding(
        dflash_config, is_first_rank=False, is_last_rank=True
    )
    assert not _should_materialize_target_embedding(
        eagle_config, is_first_rank=False, is_last_rank=True
    )
    assert _should_materialize_target_embedding(
        eagle_config, is_first_rank=True, is_last_rank=False
    )
    assert not _should_materialize_target_embedding(
        dflash_config, is_first_rank=False, is_last_rank=False
    )
