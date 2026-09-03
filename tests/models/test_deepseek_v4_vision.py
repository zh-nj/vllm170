# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import torch


def test_deepseek_v4_loads_vision_tower_only_on_first_pp_rank() -> None:
    from vllm.models.deepseek_v4.nvidia.vl_model import (
        DeepseekV4ForConditionalGeneration,
    )

    class DummyLanguageModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.make_empty_intermediate_tensors = None

    multimodal_config = SimpleNamespace(
        mm_encoder_only=False,
        get_limit_per_prompt=lambda modality: 1,
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            vision_n_layers=32,
            hidden_size=16,
            expert_dtype="fp4",
        ),
        multimodal_config=multimodal_config,
        dtype=torch.bfloat16,
        get_multimodal_config=lambda: multimodal_config,
    )
    vllm_config = SimpleNamespace(model_config=model_config)
    pp_group = SimpleNamespace(is_first_rank=False)

    with (
        patch(
            "vllm.models.deepseek_v4.nvidia.vl_model.get_pp_group",
            return_value=pp_group,
        ),
        patch(
            "vllm.models.deepseek_v4.nvidia.vl_model.DeepseekV4ViT",
            side_effect=AssertionError("vision tower must not be constructed"),
        ) as vision_cls,
        patch(
            "vllm.models.deepseek_v4.nvidia.vl_model.DeepseekV4Aligner",
            side_effect=AssertionError("aligner must not be constructed"),
        ) as aligner_cls,
        patch(
            "vllm.models.deepseek_v4.nvidia.vl_model.init_vllm_registered_model",
            return_value=DummyLanguageModel(),
        ),
    ):
        model = DeepseekV4ForConditionalGeneration(vllm_config=vllm_config)

    vision_cls.assert_not_called()
    aligner_cls.assert_not_called()
    assert model.vision is None
    assert model.aligner is None
    assert model.image_start is None
    assert model.image_end is None
    assert model.image_newline is None
    assert model.image_pad is None
    assert model._tower_model_names == []
    assert (
        model.hf_to_vllm_mapper.apply_list(
            ["vision.weight", "aligner.weight", "image_start"]
        )
        == []
    )
