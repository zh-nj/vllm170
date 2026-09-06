# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import torch

import vllm.models.glm5next.nvidia.model as glm5next_model
from vllm.model_executor.models.utils import StageMissingLayer
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


def test_glm5next_multimodal_pp_tower_only_on_first_rank(monkeypatch) -> None:
    class DummyLanguageModel(torch.nn.Module):
        def make_empty_intermediate_tensors(self):
            return None

    pp_group = SimpleNamespace(is_first_rank=True)
    multimodal_config = SimpleNamespace(
        mm_encoder_tp_mode="weights",
        is_multimodal_pruning_enabled=lambda: False,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                text_config=SimpleNamespace(),
                vision_config=SimpleNamespace(rms_norm_eps=1e-6),
            ),
            multimodal_config=multimodal_config,
        )
    )

    monkeypatch.setattr(glm5next_model, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(
        Glm5NextForConditionalGeneration,
        "_mark_tower_model",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        Glm5NextForConditionalGeneration,
        "_mark_language_model",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        glm5next_model,
        "Glm5NextVisionTransformer",
        lambda *args, **kwargs: torch.nn.Identity(),
    )
    monkeypatch.setattr(
        glm5next_model,
        "init_vllm_registered_model",
        lambda **kwargs: DummyLanguageModel(),
    )

    first_stage = Glm5NextForConditionalGeneration(vllm_config=vllm_config)
    assert isinstance(first_stage.visual, torch.nn.Identity)

    pp_group.is_first_rank = False
    later_stage = Glm5NextForConditionalGeneration(vllm_config=vllm_config)
    assert isinstance(later_stage.visual, StageMissingLayer)
    assert later_stage._tower_model_names == []
