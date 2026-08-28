# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer GQA builder: reorder threshold under DCP with spec decode."""

import pytest

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("FlashInfer backend requires a CUDA platform.", allow_module_level=True)

import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.config import SpeculativeConfig, set_current_vllm_config
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends import flashinfer as flashinfer_backend
from vllm.v1.attention.backends.flashinfer import (
    FlashInferDecodeKernel,
    FlashInferMetadataBuilder,
)
from vllm.v1.attention.backends.utils import PerLayerParameters
from vllm.v1.kv_cache_interface import FullAttentionSpec

TEST_MODEL = "hmellor/tiny-random-LlamaForCausalLM"


def test_flashinfer_gqa_dcp_spec_decode_clamps_reorder_threshold(monkeypatch):
    """trtllm-gen decode receives no cp_rank/global-seq-len information, so its
    end-aligned causal mask is wrong for q_len > 1 over the DCP-interleaved
    local KV shard. The builder must keep reorder_batch_threshold at 1 under
    DCP so spec queries take the (DCP-aware) prefill path instead.
    """
    vllm_config = create_vllm_config(model_name=TEST_MODEL, max_model_len=1024)
    vllm_config.parallel_config.decode_context_parallel_size = 2
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=3
    )

    monkeypatch.setattr(
        flashinfer_backend, "can_use_trtllm_attention", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        FlashInferMetadataBuilder,
        "_get_flashinfer_trtllm_api_decode_kernel",
        staticmethod(lambda: FlashInferDecodeKernel.TRTLLM_GEN),
    )
    monkeypatch.setattr(
        flashinfer_backend,
        "get_per_layer_parameters",
        lambda *args, **kwargs: {
            "layer.0": PerLayerParameters(
                window_left=-1, logits_soft_cap=None, sm_scale=0.1, has_sinks=False
            )
        },
    )

    kv_cache_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        ),
        head_size=vllm_config.model_config.get_head_size(),
        dtype=vllm_config.model_config.dtype,
    )
    with set_current_vllm_config(vllm_config):
        builder = FlashInferMetadataBuilder(
            kv_cache_spec,
            ["layer.0"],
            vllm_config,
            torch.device("cpu"),
        )

    # Guard against passing vacuously with the kernel disabled.
    assert (
        builder.flashinfer_trtllm_api_decode_kernel == FlashInferDecodeKernel.TRTLLM_GEN
    )
    assert builder.reorder_batch_threshold == 1


@pytest.mark.parametrize("use_non_causal", [False, True], ids=["causal", "noncausal"])
@pytest.mark.parametrize("num_speculative_tokens", [3, 5])
def test_flashinfer_sm80_supports_spec_decode_full_graph(
    monkeypatch, num_speculative_tokens, use_non_causal
):
    vllm_config = create_vllm_config(model_name=TEST_MODEL, max_model_len=1024)
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=num_speculative_tokens
    )
    vllm_config.attention_config.use_non_causal = use_non_causal

    monkeypatch.setattr(
        current_platform,
        "is_device_capability",
        lambda capability, device_id=0: capability == 80,
    )
    monkeypatch.setattr(
        flashinfer_backend, "can_use_trtllm_attention", lambda *args, **kwargs: False
    )
    monkeypatch.setattr(
        flashinfer_backend,
        "get_per_layer_parameters",
        lambda *args, **kwargs: {
            "layer.0": PerLayerParameters(
                window_left=-1, logits_soft_cap=None, sm_scale=0.1, has_sinks=False
            )
        },
    )

    kv_cache_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        ),
        head_size=vllm_config.model_config.get_head_size(),
        dtype=vllm_config.model_config.dtype,
    )
    with set_current_vllm_config(vllm_config):
        builder = FlashInferMetadataBuilder(
            kv_cache_spec,
            ["layer.0"],
            vllm_config,
            torch.device("cpu"),
        )

    expected_reorder_threshold = 1 if use_non_causal else num_speculative_tokens + 1
    assert builder.reorder_batch_threshold == expected_reorder_threshold
    assert (
        builder.get_cudagraph_support(vllm_config, kv_cache_spec)
        == AttentionCGSupport.UNIFORM_BATCH
    )


def test_flashinfer_sm80_noncausal_dcp_does_not_claim_full_graph(monkeypatch):
    vllm_config = create_vllm_config(model_name=TEST_MODEL, max_model_len=1024)
    vllm_config.attention_config.use_non_causal = True
    vllm_config.parallel_config.decode_context_parallel_size = 2
    monkeypatch.setattr(
        current_platform,
        "is_device_capability",
        lambda capability, device_id=0: capability == 80,
    )
    monkeypatch.setattr(
        flashinfer_backend, "can_use_trtllm_attention", lambda *args, **kwargs: False
    )
    kv_cache_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        ),
        head_size=vllm_config.model_config.get_head_size(),
        dtype=vllm_config.model_config.dtype,
    )

    assert (
        FlashInferMetadataBuilder.get_cudagraph_support(vllm_config, kv_cache_spec)
        == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )


def test_flashinfer_noncausal_cudagraph_wrappers_are_shape_stable(monkeypatch):
    vllm_config = create_vllm_config(model_name=TEST_MODEL, max_model_len=1024)
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=5
    )
    vllm_config.attention_config.use_non_causal = True

    monkeypatch.setattr(
        current_platform,
        "is_device_capability",
        lambda capability, device_id=0: capability == 80,
    )
    monkeypatch.setattr(
        flashinfer_backend, "can_use_trtllm_attention", lambda *args, **kwargs: False
    )
    monkeypatch.setattr(
        flashinfer_backend,
        "get_per_layer_parameters",
        lambda *args, **kwargs: {
            "layer.0": PerLayerParameters(
                window_left=-1, logits_soft_cap=None, sm_scale=0.1, has_sinks=False
            )
        },
    )

    wrappers = []

    def make_wrapper(*args, **kwargs):
        wrapper = object()
        wrappers.append((wrapper, args, kwargs))
        return wrapper

    monkeypatch.setattr(
        flashinfer_backend,
        "BatchPrefillWithPagedKVCacheWrapper",
        make_wrapper,
    )

    kv_cache_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        ),
        head_size=vllm_config.model_config.get_head_size(),
        dtype=vllm_config.model_config.dtype,
    )
    with set_current_vllm_config(vllm_config):
        builder = FlashInferMetadataBuilder(
            kv_cache_spec,
            ["layer.0"],
            vllm_config,
            torch.device("cpu"),
        )
    builder.set_workspace_buffer(torch.empty(1, dtype=torch.uint8))

    with set_current_vllm_config(vllm_config):
        wrapper_4x7 = builder._get_prefill_wrapper(
            causal=False,
            batch_size=4,
            q_len_per_req=7,
            use_cudagraph=True,
        )
        assert (
            builder._get_prefill_wrapper(
                causal=False,
                batch_size=4,
                q_len_per_req=7,
                use_cudagraph=True,
            )
            is wrapper_4x7
        )
        wrapper_1x7 = builder._get_prefill_wrapper(
            causal=False,
            batch_size=1,
            q_len_per_req=7,
            use_cudagraph=True,
        )

    assert wrapper_1x7 is not wrapper_4x7
    assert len(wrappers) == 2
    _, _, kwargs_4x7 = wrappers[0]
    assert kwargs_4x7["use_cuda_graph"] is True
    assert kwargs_4x7["qo_indptr_buf"].shape == (5,)
    assert kwargs_4x7["paged_kv_indptr_buf"].shape == (5,)
    assert kwargs_4x7["paged_kv_last_page_len_buf"].shape == (4,)
    assert kwargs_4x7["paged_kv_indices_buf"].shape == (
        4 * builder.max_num_pages_per_req,
    )
