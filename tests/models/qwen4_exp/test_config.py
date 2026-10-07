# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models.config import (
    Qwen3_5ForConditionalGenerationConfig,
    Qwen4ExpForConditionalGenerationConfig,
)
from vllm.models.qwen4_exp.config import (
    Qwen4ExpConfig,
    Qwen4ExpTextConfig,
)
from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


def _text_config(**kwargs) -> Qwen4ExpTextConfig:
    values = {
        "vocab_size": 64,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "num_experts": 0,
        "hc_count": 2,
        "hc_lowrank": 4,
        "ple_layer_ids": [1],
        "mtp_num_hidden_layers": 1,
        "mtp": {"hybrid": True},
    }
    values.update(kwargs)
    return Qwen4ExpTextConfig(**values)


@pytest.mark.parametrize(
    ("capability", "quantization", "hidden_size", "expected"),
    [
        (True, "modelopt_fp4", 2560, 16),
        (False, "modelopt_fp4", 2560, None),
        (True, "fp8", 2560, None),
        (True, "modelopt_fp4", 4096, None),
    ],
)
def test_qwen4_exp_marlin_decode_block_size_is_exact_sm80_tuning(
    capability: bool,
    quantization: str,
    hidden_size: int,
    expected: int | None,
) -> None:
    from vllm.models.qwen4_exp.nvidia import model as model_module

    config = SimpleNamespace(
        hidden_size=hidden_size,
        moe_intermediate_size=640,
        num_experts=512,
        num_experts_per_tok=10,
    )
    quant_config = SimpleNamespace(get_name=lambda: quantization)

    with patch.object(
        model_module.current_platform,
        "is_device_capability",
        return_value=capability,
    ):
        actual = model_module._qwen4_exp_marlin_moe_decode_block_size(
            config,
            quant_config,
        )

    assert actual == expected


@pytest.mark.parametrize(
    ("enabled", "speculation", "sm80", "tp_size", "quantization", "standard"),
    [
        (True, False, True, 1, "inc", True),
        (False, False, True, 1, "inc", False),
        (True, True, True, 1, "inc", False),
        (True, False, False, 1, "inc", False),
        (True, False, True, 2, "inc", False),
        (True, False, True, 1, "modelopt_fp4", False),
    ],
)
def test_standard_gdn_preserves_speculation_and_non_target_paths(
    enabled, speculation, sm80, tp_size, quantization, standard, monkeypatch
) -> None:
    from vllm.models.qwen4_exp.nvidia import model as model_module

    def initialize(layer, *args):
        torch.nn.Module.__init__(layer)
        layer.tp_size = tp_size
        layer.head_k_dim = layer.head_v_dim = 128
        layer.enable_fused_gdn_decode = True
        layer.in_proj_qkvz = layer.in_proj_ba = None
        layer.gqa_interleaved_layout = False

    config = SimpleNamespace(hidden_size=2560)
    vllm_config = SimpleNamespace(
        speculative_config=object() if speculation else None,
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        quant_config=SimpleNamespace(get_name=lambda: quantization),
    )
    monkeypatch.setattr(
        model_module.envs, "VLLM_QWEN4_EXP_GDN_STANDARD_DECODE", enabled
    )
    with (
        patch.object(model_module.QwenGatedDeltaNetAttention, "__init__", initialize),
        patch.object(
            model_module.current_platform, "is_device_capability", return_value=sm80
        ),
        patch.object(
            model_module,
            "can_use_qwen4_exp_sm80_gdn_input_projection",
            return_value=False,
        ),
    ):
        layer = model_module.Qwen4ExpGatedDeltaNetAttention(config, vllm_config)
    assert layer.enable_fused_gdn_decode is not standard


def test_qwen4_exp_loads_vision_tower_only_on_first_pp_rank() -> None:
    from vllm.model_executor.models.utils import StageMissingLayer
    from vllm.models.qwen4_exp.nvidia.model import (
        Qwen4ExpForConditionalGeneration,
    )

    class DummyLanguageModel(torch.nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            self.model = SimpleNamespace(layers=[])
            self.make_empty_intermediate_tensors = None

    multimodal_config = SimpleNamespace(
        language_model_only=False,
        mm_encoder_only=False,
        mm_encoder_tp_mode="weights",
    )
    model_config = SimpleNamespace(
        hf_config=Qwen4ExpConfig(text_config=_text_config(ple_layer_ids=[])),
        multimodal_config=multimodal_config,
        get_multimodal_config=lambda: multimodal_config,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        quant_config=None,
    )
    pp_group = SimpleNamespace(is_first_rank=False)

    with (
        patch(
            "vllm.models.qwen4_exp.nvidia.model.get_pp_group",
            return_value=pp_group,
        ),
        patch.object(
            Qwen4ExpForConditionalGeneration,
            "_init_video_pruning",
        ),
        patch(
            "vllm.models.qwen4_exp.nvidia.model.cached_tokenizer_from_config",
            return_value=None,
        ),
        patch(
            "vllm.models.qwen4_exp.nvidia.model.Qwen4ExpForCausalLM",
            DummyLanguageModel,
        ),
    ):
        model = Qwen4ExpForConditionalGeneration(vllm_config=vllm_config)

    assert isinstance(model.visual, StageMissingLayer)
    assert model._tower_model_names == []


def test_qwen4_exp_mtp_runs_locally_on_last_target_pp_rank() -> None:
    from vllm.models.qwen4_exp.nvidia.mtp import (
        Qwen4ExpMultiTokenPredictor,
    )

    model = object.__new__(Qwen4ExpMultiTokenPredictor)
    torch.nn.Module.__init__(model)
    model.hc_count = 2
    model.hidden_size = 4
    model.num_mtp_layers = 1
    model.embed_tokens = torch.nn.Embedding(2, 4)
    model.pre_fc_norm_embedding = torch.nn.Identity()
    model.pre_fc_norm_hidden = torch.nn.Identity()
    model.fc_embedding = torch.nn.Identity()
    model.fc_hidden = torch.nn.Identity()
    model.layers = [
        lambda **kwargs: (
            kwargs["hidden_states"],
            kwargs["hidden_states"],
            torch.zeros(kwargs["hidden_states"].shape[0], 2),
        ),
    ]
    model.hyper_connection_mixer = SimpleNamespace(
        combine_and_mix=lambda hidden_states, block_output, injection: (
            hidden_states,
            hidden_states.unflatten(-1, (2, 4)).mean(-2),
            None,
        ),
    )
    multi_hidden = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    input_ids = torch.tensor([0, 1])
    expected_multi_hidden = (
        multi_hidden.unflatten(-1, (2, 4)) + model.embed_tokens(input_ids).unsqueeze(-2)
    ).flatten(-2)
    pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=True)

    with patch(
        "vllm.models.qwen4_exp.nvidia.mtp.get_pp_group",
        return_value=pp_group,
        create=True,
    ):
        sample_hidden, returned_multi_hidden = model.forward(
            input_ids=input_ids,
            positions=torch.arange(2),
            hidden_states=multi_hidden,
        )

    torch.testing.assert_close(
        sample_hidden,
        expected_multi_hidden.unflatten(-1, (2, 4)).mean(dim=-2),
    )
    torch.testing.assert_close(returned_multi_hidden, expected_multi_hidden)


@pytest.mark.parametrize("wrapped_config", [False, True])
def test_qwen4_exp_mtp_override_sets_draft_config(
    wrapped_config: bool,
) -> None:
    text_config = _text_config(
        architectures=["Qwen4ExpForCausalLM"],
        index_share_for_mtp_iteration=True,
    )
    config = (
        Qwen4ExpConfig(
            architectures=["Qwen4ExpForConditionalGeneration"],
            text_config=text_config,
        )
        if wrapped_config
        else text_config
    )

    draft_config = SpeculativeConfig.hf_config_override(config)

    assert draft_config.index_share_for_mtp_iteration is True
    assert draft_config.model_type == "qwen4_exp_mtp"
    assert draft_config.architectures == ["Qwen4ExpMTP"]
    assert draft_config.hc_mult == 2
    assert draft_config.n_predict == 1


@pytest.mark.parametrize("ple_layer_ids", [[1], []])
def test_qwen4_exp_allows_pipeline_parallel(ple_layer_ids) -> None:
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=_text_config(ple_layer_ids=ple_layer_ids),
            multimodal_config=None,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2, enable_dbo=False, ubatch_size=1
        ),
        speculative_config=None,
    )
    with patch.object(
        Qwen3_5ForConditionalGenerationConfig, "verify_and_update_config"
    ):
        Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(vllm_config)


@pytest.mark.parametrize(
    ("configured_mode", "expected_mode"),
    [
        (None, None),
        (CUDAGraphMode.FULL, CUDAGraphMode.FULL),
        (
            CUDAGraphMode.FULL_AND_PIECEWISE,
            CUDAGraphMode.FULL_AND_PIECEWISE,
        ),
        (CUDAGraphMode.NONE, CUDAGraphMode.NONE),
    ],
)
def test_qwen4_exp_cpu_ple_preserves_graph_mode(
    monkeypatch: pytest.MonkeyPatch,
    configured_mode: CUDAGraphMode | None,
    expected_mode: CUDAGraphMode,
) -> None:
    monkeypatch.setenv("VLLM_QWEN4_EXP_PLE_CPU_OFFLOAD", "1")
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=_text_config(),
            multimodal_config=None,
        ),
        parallel_config=SimpleNamespace(enable_dbo=False, ubatch_size=1),
        speculative_config=None,
        compilation_config=SimpleNamespace(cudagraph_mode=configured_mode),
    )

    with patch.object(
        Qwen3_5ForConditionalGenerationConfig, "verify_and_update_config"
    ):
        Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(vllm_config)

    assert vllm_config.compilation_config.cudagraph_mode == expected_mode


def test_qwen4_exp_model_state_prepares_ngram_context() -> None:
    model_state = object.__new__(Qwen4ExpModelState)
    model_state.uses_ngram_embedding = True
    model_state.ngram_context_len = 3
    model_state.ngram_eos_token_id = 99
    model_state.ngram_context = torch.empty((4, 3), dtype=torch.int32)
    model_state.ngram_context_offsets = torch.arange(-3, 0, dtype=torch.int64)
    model_state.ple_query_start_loc = torch.empty(5, dtype=torch.int32)

    input_batch = SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=3,
        idx_mapping=torch.tensor([1, 0]),
        query_start_loc=torch.tensor([0, 2, 3, 3], dtype=torch.int32),
    )
    req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([3, 1])),
        all_token_ids=SimpleNamespace(
            gpu=torch.tensor([[1, 2, 3, 4], [20, 21, 22, 23]], dtype=torch.int32)
        ),
    )

    with patch.object(MambaHybridModelState, "prepare_inputs", return_value={}):
        model_inputs = model_state.prepare_inputs(input_batch, req_states)

    torch.testing.assert_close(
        model_inputs["query_start_loc"],
        torch.tensor([0, 2, 3, 3], dtype=torch.int32),
    )
    torch.testing.assert_close(
        model_inputs["ngram_context"],
        torch.tensor([[99, 99, 20], [1, 2, 3], [99, 99, 99]], dtype=torch.int32),
    )


def test_qwen4_exp_model_state_prepares_stable_dummy_ngram_inputs() -> None:
    model_state = object.__new__(Qwen4ExpModelState)
    model_state.uses_ngram_embedding = True
    model_state.ngram_eos_token_id = 99
    model_state.ngram_context = torch.empty((4, 3), dtype=torch.int32)
    model_state.ple_query_start_loc = torch.empty(5, dtype=torch.int32)

    with patch.object(MambaHybridModelState, "prepare_dummy_inputs", return_value={}):
        first = model_state.prepare_dummy_inputs(num_reqs=3, num_tokens=8)
        query_start_loc_ptr = first["query_start_loc"].data_ptr()
        ngram_context_ptr = first["ngram_context"].data_ptr()
        second = model_state.prepare_dummy_inputs(num_reqs=3, num_tokens=8)

    torch.testing.assert_close(
        second["query_start_loc"], torch.tensor([0, 2, 5, 8], dtype=torch.int32)
    )
    torch.testing.assert_close(
        second["ngram_context"], torch.full((3, 3), 99, dtype=torch.int32)
    )
    assert second["query_start_loc"].data_ptr() == query_start_loc_ptr
    assert second["ngram_context"].data_ptr() == ngram_context_ptr


@pytest.mark.parametrize(
    ("pp_rank", "ple_layer_ids", "uses_ngram_embedding", "raises"),
    [
        (0, [2], True, False),
        (1, [2], False, False),
        (1, [25], True, True),
    ],
)
def test_qwen4_exp_model_state_limits_ple_to_first_pp_rank(
    pp_rank: int,
    ple_layer_ids: list[int],
    uses_ngram_embedding: bool,
    raises: bool,
) -> None:
    def init_base_state(state, vllm_config, model, encoder_cache, device) -> None:
        state.model_config = vllm_config.model_config
        state.max_num_reqs = 4
        state.device = device

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                ple_layer_ids=ple_layer_ids,
                num_hidden_layers=48,
                ngram_size=3,
                eos_token_id=99,
            ),
        ),
    )
    pp_group = SimpleNamespace(
        ranks=[0, 1],
        rank_in_group=pp_rank,
        is_first_rank=pp_rank == 0,
    )
    error_context = (
        pytest.raises(RuntimeError, match="first pipeline rank")
        if raises
        else nullcontext()
    )
    with (
        patch.object(MambaHybridModelState, "__init__", init_base_state),
        patch(
            "vllm.models.qwen4_exp.nvidia.model_state.get_pp_group",
            return_value=pp_group,
        ),
        error_context,
    ):
        model_state = Qwen4ExpModelState(
            vllm_config,
            torch.nn.Identity(),
            None,
            torch.device("cpu"),
        )
    if not raises:
        assert model_state.uses_ngram_embedding is uses_ngram_embedding
