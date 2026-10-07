# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp MTP preparation without device-to-host indexing waits."""

import torch

from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


def qwen_mtp_pp_optimizations_enabled(
    vllm_config: VllmConfig,
    device: torch.device,
    feature: str = "qwen4_exp_mtp_pp_optimizations",
) -> bool:
    extra = vllm_config.additional_config
    enabled = extra.get(feature, False) if isinstance(extra, dict) else False
    if not isinstance(enabled, bool):
        raise ValueError(f"{feature} must be a JSON boolean")
    if not enabled:
        return False
    model = vllm_config.model_config
    hf = model.hf_text_config
    parallel = vllm_config.parallel_config
    speculative = vllm_config.speculative_config
    return bool(
        enabled
        and current_platform.is_cuda()
        and device.type == "cuda"
        and current_platform.is_device_capability(80, device.index or 0)
        and getattr(model.hf_config, "model_type", None) == "qwen4_exp"
        and getattr(hf, "hidden_size", None) == 2560
        and getattr(hf, "num_hidden_layers", None) == 48
        and getattr(hf, "num_experts", None) == 512
        and model.quantization == "inc"
        and model.dtype == torch.bfloat16
        and parallel.pipeline_parallel_size == 2
        and parallel.tensor_parallel_size == 1
        and parallel.data_parallel_size == 1
        and parallel.prefill_context_parallel_size == 1
        and speculative is not None
        and speculative.method == "mtp"
        and (
            speculative.num_speculative_tokens == 3
            or (
                speculative.num_speculative_tokens == 2
                and feature == "qwen4_exp_mtp_pp_optimizations"
            )
        )
        and not speculative.enable_adaptive_verification
        and vllm_config.scheduler_config.max_num_seqs <= 4
        and vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
    )


@triton.jit
def _copy_pp_draft_tokens_kernel(
    destination,
    source,
    idx_mapping,
    destination_stride: tl.constexpr,
    destination_column_stride: tl.constexpr,
    source_stride: tl.constexpr,
    source_column_stride: tl.constexpr,
    num_drafts: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    index = tl.load(idx_mapping + row)
    columns = tl.arange(0, block)
    values = tl.load(
        source + row * source_stride + columns * source_column_stride,
        mask=(index >= 0) & (columns < num_drafts),
        other=0,
    )
    tl.store(
        destination + index * destination_stride + columns * destination_column_stride,
        values,
        mask=(index >= 0) & (columns < num_drafts),
    )


def copy_pp_draft_tokens(
    destination: torch.Tensor, source: torch.Tensor, idx_mapping: torch.Tensor
) -> None:
    """Copy valid relay rows, skipping freed request slots marked with -1."""
    num_reqs, num_drafts = source.shape
    if num_reqs == 0 or num_drafts == 0:
        return
    _copy_pp_draft_tokens_kernel[(num_reqs,)](
        destination,
        source,
        idx_mapping,
        destination.stride(0),
        destination.stride(1),
        source.stride(0),
        source.stride(1),
        num_drafts,
        triton.next_power_of_2(num_drafts),
        num_warps=1,
    )
