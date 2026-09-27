# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in relay reduction for the 40-layer V4.1 Flash target."""

from vllm.config import VllmConfig


def _v41_distributed_dspark(config: VllmConfig) -> bool:
    """Whether this engine runs the PP5 distributed V4.1 draft relay at all."""
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


def _relay_flag(config: VllmConfig, key: str) -> bool:
    extra = config.additional_config
    return isinstance(extra, dict) and extra.get(key) is True


def compact_relay_enabled(config: VllmConfig) -> bool:
    return _relay_flag(config, "dsv41_dspark_compact_relay") and (
        _v41_distributed_dspark(config)
    )


def prepare_ahead_enabled(config: VllmConfig) -> bool:
    """Whether the head sends draft context/control before the layer state.

    All three draft stages derive their context KV and attention metadata from
    the same head-side context, so no stage's prepare depends on the layer
    below it. Sending that package separately lets each stage prepare while its
    upstream neighbour is still running the query that produces the state it
    will consume.
    """
    return _relay_flag(config, "dsv41_dspark_prepare_ahead") and (
        _v41_distributed_dspark(config)
    )
