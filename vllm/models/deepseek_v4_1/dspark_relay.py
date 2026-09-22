# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in relay reduction for the 40-layer V4.1 Flash target."""

from vllm.config import VllmConfig


def compact_relay_enabled(config: VllmConfig) -> bool:
    extra = config.additional_config
    if (
        not isinstance(extra, dict)
        or extra.get("dsv41_dspark_compact_relay") is not True
    ):
        return False
    spec = config.speculative_config
    target = config.model_config.hf_config
    return (
        getattr(target, "model_type", None) == "deepseek_v41"
        and getattr(target, "num_hidden_layers", None) == 40
        and config.parallel_config.tensor_parallel_size == 1
        and spec is not None
        and spec.method == "dspark"
        and spec.use_distributed_dspark()
        and getattr(spec.draft_model_config.hf_config, "model_type", None)
        == "deepseek_v41"
    )
