# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    prepare_fp8_layer_for_marlin,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

logger = init_logger(__name__)

_MTP_LM_HEAD_SHAPE = (248320, 2560)
_MTP_LM_HEAD_GROUP_SIZE = 128
_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


@torch.no_grad()
def _quantize_groupwise_fp8(
    weight: torch.Tensor,
    group_size: int = _MTP_LM_HEAD_GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    size_n, size_k = weight.shape
    if size_k % group_size != 0:
        raise ValueError(f"K={size_k} must be divisible by group_size={group_size}")

    quantized = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (size_n, size_k // group_size),
        dtype=torch.float32,
        device=weight.device,
    )
    rows_per_chunk = 2048
    for start in range(0, size_n, rows_per_chunk):
        end = min(start + rows_per_chunk, size_n)
        chunk = weight[start:end].reshape(end - start, -1, group_size)
        scale = (chunk.abs().amax(dim=2).float() / _FP8_MAX).clamp_min_(1e-8)
        scales[start:end] = scale
        quantized[start:end] = (
            (chunk / scale[:, :, None])
            .clamp_(-_FP8_MAX, _FP8_MAX)
            .to(torch.float8_e4m3fn)
            .reshape(end - start, size_k)
        )
    return quantized, scales


class Qwen4ExpMTPFP8LMHeadMethod(UnquantizedEmbeddingMethod):
    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if getattr(layer, "_qwen4_exp_mtp_fp8_prepared", False):
            return
        if tuple(layer.weight.shape) != _MTP_LM_HEAD_SHAPE:
            shape = tuple(layer.weight.shape)
            raise ValueError(f"unexpected MTP LM head shape: {shape}")

        size_n, size_k = _MTP_LM_HEAD_SHAPE
        quantized, scales = _quantize_groupwise_fp8(layer.weight)
        replace_parameter(layer, "weight", quantized)
        layer.register_parameter(
            "weight_scale", nn.Parameter(scales, requires_grad=False)
        )
        layer.output_size_per_partition = size_n
        layer.input_size_per_partition = size_k
        layer.orig_dtype = torch.bfloat16
        layer.logical_widths = [size_n]
        layer.weight_block_size = (1, _MTP_LM_HEAD_GROUP_SIZE)
        prepare_fp8_layer_for_marlin(layer, size_k_first=False)
        layer._qwen4_exp_mtp_fp8_prepared = True

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        size_n, size_k = _MTP_LM_HEAD_SHAPE
        return apply_fp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=size_n,
            size_k=size_k,
            bias=bias,
        )

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("the draft-only FP8 LM head cannot be used as an embedding")


def can_use_qwen4_exp_mtp_fp8_lm_head(
    lm_head: ParallelLMHead,
    *,
    tied: bool,
) -> bool:
    return (
        envs.VLLM_QWEN4_EXP_MTP_FP8_LM_HEAD
        and not envs.VLLM_BATCH_INVARIANT
        and not tied
        and current_platform.is_device_capability((8, 0))
        and lm_head.tp_size == 1
        and type(lm_head.quant_method) is UnquantizedEmbeddingMethod
        and lm_head.params_dtype == torch.bfloat16
        and lm_head.bias is None
        and tuple(lm_head.weight.shape) == _MTP_LM_HEAD_SHAPE
    )


def enable_qwen4_exp_mtp_fp8_lm_head(
    lm_head: ParallelLMHead,
    *,
    tied: bool,
) -> bool:
    if not can_use_qwen4_exp_mtp_fp8_lm_head(lm_head, tied=tied):
        return False
    lm_head.quant_method = Qwen4ExpMTPFP8LMHeadMethod()
    logger.warning_once("Using groupwise FP8 Marlin for the Qwen4Exp MTP draft LM head")
    return True


__all__ = [
    "Qwen4ExpMTPFP8LMHeadMethod",
    "can_use_qwen4_exp_mtp_fp8_lm_head",
    "enable_qwen4_exp_mtp_fp8_lm_head",
]
