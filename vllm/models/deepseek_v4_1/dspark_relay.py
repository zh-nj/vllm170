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

    On by default; ``dsv41_dspark_prepare_ahead: false`` in the additional
    config keeps the single package. A non-boolean value is rejected rather
    than read as false, so a misspelled opt-out cannot silently keep the split.
    """
    if not _v41_distributed_dspark(config):
        return False
    extra = config.additional_config
    if not isinstance(extra, dict):
        return True
    value = extra.get("dsv41_dspark_prepare_ahead", True)
    if not isinstance(value, bool):
        raise ValueError(f"dsv41_dspark_prepare_ahead must be a boolean, got {value!r}")
    return value
