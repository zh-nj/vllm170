# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch


def test_decode_raw_ids_preserve_text_and_sentinel_embeddings() -> None:
    from vllm.models.deepseek_v4.common.mm_preprocess import IMAGE_SENTINEL_BASE_ID
    from vllm.models.deepseek_v4.nvidia.vl_model import (
        DeepseekV4ForConditionalGeneration,
    )

    class DummyLanguageModel(torch.nn.Module):
        def embed_input_ids(self, input_ids):
            return input_ids[:, None].expand(-1, 4).float()

    model = SimpleNamespace(
        language_model=DummyLanguageModel(),
        image_start=torch.full((4,), 10.0),
        image_pad=torch.full((4,), 11.0),
        image_newline=torch.full((4,), 13.0),
        image_end=torch.full((4,), 14.0),
    )
    input_ids = torch.tensor([7, IMAGE_SENTINEL_BASE_ID, IMAGE_SENTINEL_BASE_ID + 2])

    external = DeepseekV4ForConditionalGeneration.embed_input_ids(
        model,
        input_ids,
        multimodal_embeddings=[],
        is_multimodal=torch.zeros(3, dtype=torch.bool),
    )
    internal = DeepseekV4ForConditionalGeneration.embed_input_ids(model, input_ids)

    torch.testing.assert_close(internal, external, rtol=0, atol=0)


@torch.no_grad()
def test_deepseek_v4_vision_selects_decode_embedding_path() -> None:
    from vllm.models.deepseek_v4.nvidia.vl_model import (
        DeepseekV4ForConditionalGeneration,
    )

    input_ids = torch.tensor([7])
    positions = torch.tensor([0])
    embedded = torch.randn(1, 4)
    supplied = torch.randn(1, 4)

    for is_first_rank, inputs_embeds, expected_embed_call in (
        (True, None, True),
        (True, supplied, False),
        (False, None, False),
    ):
        model = SimpleNamespace(
            embed_input_ids=MagicMock(return_value=embedded),
            language_model=MagicMock(return_value=torch.empty(0)),
        )
        with patch(
            "vllm.models.deepseek_v4.nvidia.vl_model.get_pp_group",
            return_value=SimpleNamespace(is_first_rank=is_first_rank),
        ):
            DeepseekV4ForConditionalGeneration.forward(
                model,
                input_ids,
                positions,
                inputs_embeds=inputs_embeds,
            )

        if expected_embed_call:
            model.embed_input_ids.assert_called_once_with(input_ids)
        else:
            model.embed_input_ids.assert_not_called()
        args = model.language_model.call_args.args
        assert args[0] is input_ids
        assert args[1] is positions
        assert args[2] is None
        expected_inputs_embeds = embedded if expected_embed_call else inputs_embeds
        assert args[3] is expected_inputs_embeds


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
