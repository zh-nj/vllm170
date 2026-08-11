# SPDX-License-Identifier: Apache-2.0
"""Experimental DSV4 MXFP4 -> INT8 modular MoE backend for SM80."""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_int8_sm80 import (
    quantize_per_token_int8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
)
from vllm.platforms import current_platform

_HIDDEN_SIZE = 4096
_INTERMEDIATE_SIZE = 2048
_TOPK = 6
_TILE_TABLE: tuple[tuple[int, int, int], ...] = (
    (1, 16, 128),
    (1024, 16, 64),
    (2048, 64, 128),
    (1 << 30, 128, 128),
)


def select_mxfp4_int8_sm80_tile(
    num_tokens: int,
    topk: int,
    num_experts: int,
) -> tuple[int, int]:
    """Static starting point; the server benchmark must validate this table."""
    average = max(1, (num_tokens * topk + num_experts - 1) // num_experts)
    max_block_m = min(128, max(16, 1 << (average - 1).bit_length()))
    for max_tokens, block_m, block_n in _TILE_TABLE:
        if num_tokens <= max_tokens and block_m <= max_block_m:
            return block_m, block_n
    return 16, 128


class Mxfp4Int8Sm80Experts(mk.FusedMoEExpertsModular):
    """SM80 INT8-MMA experts for the fixed DSV4 TP1 expert shape.

    Pipeline parallelism is outside the MoE tensor layout, so PP3 is supported
    while TP/EP/DP/PCP/SP sharding deliberately fails closed.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        layer: torch.nn.Module,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        if moe_config.in_dtype != torch.bfloat16:
            raise ValueError(
                "mxfp4_int8_sm80 currently requires BF16 model activations; "
                f"got {moe_config.in_dtype}"
            )
        if (
            moe_config.hidden_dim != _HIDDEN_SIZE
            or moe_config.intermediate_size_per_partition != _INTERMEDIATE_SIZE
            or moe_config.experts_per_token != _TOPK
        ):
            raise ValueError(
                "mxfp4_int8_sm80 currently requires DSV4 TP1 shape "
                f"H={_HIDDEN_SIZE}, I={_INTERMEDIATE_SIZE}, topk={_TOPK}; got "
                f"H={moe_config.hidden_dim}, "
                f"I={moe_config.intermediate_size_per_partition}, "
                f"topk={moe_config.experts_per_token}"
            )
        self.w13_shift2 = layer.w13_mxfp4_int8_shift2
        self.w2_shift2 = layer.w2_mxfp4_int8_shift2
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        capability = current_platform.get_device_capability()
        return (
            current_platform.is_cuda()
            and capability is not None
            and capability.major == 8
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return weight_key == kMxfp4Static and activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(config: FusedMoEParallelConfig) -> bool:
        return (
            config.tp_size == 1
            and config.pcp_size == 1
            and config.dp_size == 1
            and config.ep_size == 1
            and config.sp_size == 1
            and not config.use_ep
        )

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method == RoutingMethodType.DeepseekV4

    @staticmethod
    def _supports_shape(hidden_dim: int) -> bool:
        return hidden_dim == _HIDDEN_SIZE

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        return (
            w1.size(0),
            a1.size(0),
            _INTERMEDIATE_SIZE * 2,
            _HIDDEN_SIZE,
            topk_ids.size(1),
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # The modular runner normally aliases its final output with workspace13.
        # Keep routed W2 slots in workspace2 so the in-place reduction never
        # overwrites slots belonging to later tokens. workspace13 is only the
        # activation output, whose contents are dead after A2 quantization.
        return (
            (M * topk, _INTERMEDIATE_SIZE),
            (M * topk, max(2 * _INTERMEDIATE_SIZE, _HIDDEN_SIZE)),
            (M, _HIDDEN_SIZE),
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        if apply_router_weight_on_input:
            raise ValueError("mxfp4_int8_sm80 applies routing weights in its W2 reduce")
        if expert_map is not None:
            raise ValueError("mxfp4_int8_sm80 does not support expert parallelism")
        if a1q_scale is not None or a2_scale is not None:
            raise ValueError("mxfp4_int8_sm80 quantizes activations internally")
        if hidden_states.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
            raise TypeError("mxfp4_int8_sm80 requires BF16 input and output tensors")
        assert self.w1_scale is not None
        assert self.w2_scale is not None

        m = hidden_states.size(0)
        topk = topk_ids.size(1)
        num_experts = w1.size(0)
        block_m, block_n = select_mxfp4_int8_sm80_tile(m, topk, num_experts)
        routing_ids = topk_ids.to(torch.int32).contiguous()
        routing_weights = topk_weights.to(torch.float32).contiguous()
        sorted_ids, expert_ids, num_post_pad = moe_align_block_size(
            routing_ids,
            block_m,
            global_num_experts,
        )

        gate_up = workspace2.view(-1, workspace2.size(-1))[
            : m * topk, : 2 * _INTERMEDIATE_SIZE
        ]
        intermediate = workspace13.view(-1, workspace13.size(-1))[
            : m * topk, :_INTERMEDIATE_SIZE
        ]
        # Reuse the W13 output after activation. A2 quantization is enqueued
        # before W2 on the same stream, so overwriting this storage is ordered.
        routed_out = workspace2.view(-1, workspace2.size(-1))[: m * topk, :_HIDDEN_SIZE]

        a13_q, a13_scale = quantize_per_token_int8(hidden_states)
        torch.ops._moe_C.dsv4_mxfp4_int8_sm80_moe_gemm(
            a13_q,
            a13_scale,
            w1,
            self.w13_shift2,
            self.w1_scale,
            gate_up,
            gate_up,
            routing_weights,
            sorted_ids,
            expert_ids,
            num_post_pad,
            m * topk,
            block_m,
            block_n,
            False,
        )
        self.activation(
            activation,
            intermediate,
            gate_up,
            clamp_limit=self.quant_config.gemm1_clamp_limit,
            alpha=(
                self.quant_config.gemm1_alpha
                if self.quant_config.gemm1_alpha is not None
                else 1.0
            ),
            beta=(
                self.quant_config.gemm1_beta
                if self.quant_config.gemm1_beta is not None
                else 0.0
            ),
        )

        a2_q, a2_token_scale = quantize_per_token_int8(intermediate)
        torch.ops._moe_C.dsv4_mxfp4_int8_sm80_moe_gemm(
            a2_q,
            a2_token_scale,
            w2,
            self.w2_shift2,
            self.w2_scale,
            output,
            routed_out,
            routing_weights,
            sorted_ids,
            expert_ids,
            num_post_pad,
            m * topk,
            block_m,
            block_n,
            True,
        )
        torch.ops._moe_C.dsv4_mxfp4_int8_sm80_moe_reduce(
            routed_out,
            routing_weights,
            output,
            m * topk,
        )
