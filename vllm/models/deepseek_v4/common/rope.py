# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepseekV4 rotary embedding initialization."""

from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding


def resolve_deepseek_v4_rope_cache_extent(
    max_model_len: int,
    model_max_position_embeddings: int,
) -> int:
    if max_model_len <= 0:
        raise ValueError(
            f"DeepSeek V4 max_model_len must be positive, got {max_model_len}."
        )
    return min(max_model_len, model_max_position_embeddings)


def build_deepseek_v4_rope(
    config,
    *,
    head_dim: int,
    rope_head_dim: int,
    max_position_embeddings: int,
    max_model_len: int,
    compress_ratio: int,
) -> RotaryEmbedding:
    rope_parameters = config.rope_parameters
    rope_parameters["rope_theta"] = (
        config.compress_rope_theta if compress_ratio > 1 else config.rope_theta
    )
    if rope_parameters["rope_type"] != "default":
        rope_parameters["rope_type"] = (
            "deepseek_yarn"
            if rope_parameters.get("apply_yarn_scaling", True)
            else "deepseek_llama_scaling"
        )
    rope_parameters["mscale"] = 0  # Disable mscale
    rope_parameters["mscale_all_dim"] = 0  # Disable mscale
    rope_parameters["is_deepseek_v4"] = True
    rope_parameters["rope_dim"] = rope_head_dim
    cache_extent = resolve_deepseek_v4_rope_cache_extent(
        max_model_len,
        max_position_embeddings,
    )
    return get_rope(
        head_dim,
        max_position=cache_extent,
        rope_parameters=rope_parameters,
        is_neox_style=False,
    )
