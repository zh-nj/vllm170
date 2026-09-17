# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy

import torch.nn as nn

from vllm.config import KernelConfig, ModelConfig, VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = init_logger(__name__)


def _is_materialized_embedding(module: nn.Module | None) -> bool:
    return getattr(module, "weight", None) is not None


def _get_dspark_kernel_config(vllm_config: VllmConfig) -> KernelConfig:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.moe_backend is None:
        return vllm_config.kernel_config
    return replace(
        vllm_config.kernel_config,
        moe_backend=speculative_config.moe_backend,
    )


def _resolve_dspark_attention_backend(
    draft_model_config: ModelConfig,
    draft_backend: AttentionBackendEnum | None,
    target_backend: AttentionBackendEnum | None,
) -> AttentionBackendEnum | None:
    if draft_backend is not None:
        return draft_backend
    # DeepSeek-V4(.1) draft layers share the target's KV-cache layout. Other
    # DSpark architectures may use a different attention kind.
    if draft_model_config.hf_config.model_type in ("deepseek_v4", "deepseek_v41"):
        if target_backend is not None:
            logger.info_once(
                "Using the target model's %s attention backend for the "
                "DeepSeek-V4 DSpark drafter.",
                target_backend.name,
            )
        return target_backend
    return None


def dspark_backbone_vllm_config(vllm_config: VllmConfig) -> VllmConfig:
    """Size draft attention for all query positions without changing verification."""
    speculative_config = vllm_config.speculative_config
    if speculative_config is None:
        return vllm_config
    query_tokens = speculative_config.dspark_num_query_tokens
    if (
        query_tokens is None
        or query_tokens == speculative_config.num_speculative_tokens
    ):
        return vllm_config
    draft_config = copy.copy(vllm_config)
    draft_config.speculative_config = copy.copy(speculative_config)
    draft_config.speculative_config.num_speculative_tokens = query_tokens
    return draft_config


def load_dspark_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    vllm_config = dspark_backbone_vllm_config(vllm_config)
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.model_loader import get_model
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal
    from vllm.model_executor.models.utils import get_draft_quant_config
    from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
        _should_share,
        get_target_lm_head,
    )

    draft_attention_backend = _resolve_dspark_attention_backend(
        draft_model_config,
        speculative_config.attention_backend,
        vllm_config.attention_config.backend,
    )

    draft_vllm_config = replace(
        vllm_config,
        kernel_config=_get_dspark_kernel_config(vllm_config),
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=draft_attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
    )
    # VllmConfig post-init restores the target's quant config because the target
    # config is retained for DSpark's target-layer metadata, so we must override it.
    draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)

    with set_model_tag("dspark_head"):
        draft_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    is_pp = get_pp_group().world_size != 1

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = draft_model.model
    target_vocab_size = vllm_config.model_config.get_vocab_size()

    target_embed = getattr(target_inner, "embed_tokens", None)
    draft_embed = getattr(draft_inner, "embed_tokens", None)
    is_dspark_head_rank = getattr(draft_model, "is_dspark_head_rank", True)
    is_dspark_embedding_rank = getattr(draft_model, "is_dspark_embedding_rank", True)
    if (
        is_dspark_embedding_rank
        and _is_materialized_embedding(target_embed)
        and draft_model_config.get_vocab_size() <= target_vocab_size
        and _should_share(
            draft_model, "has_own_embed_tokens", draft_embed, target_embed
        )
    ):
        if draft_embed is not None:
            del draft_inner.embed_tokens
        draft_inner.embed_tokens = target_embed
    elif (
        is_pp
        and is_dspark_embedding_rank
        and not _is_materialized_embedding(draft_embed)
    ):
        raise RuntimeError(
            "DSpark+PP requires a materialized draft embedding on its first "
            "draft stage."
        )

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(draft_model, "lm_head", None)
    draft_output_vocab_size = (
        getattr(draft_model_config.hf_config, "draft_vocab_size", None)
        or draft_model_config.get_vocab_size()
    )
    if (
        target_lm_head is not None
        and draft_output_vocab_size == target_vocab_size
        and _should_share(draft_model, "has_own_lm_head", draft_lm_head, target_lm_head)
    ):
        if draft_lm_head is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head
    elif is_pp and is_dspark_head_rank:
        raise RuntimeError(
            "DSpark+PP expects the target lm_head on the last pipeline rank."
        )

    return draft_model
