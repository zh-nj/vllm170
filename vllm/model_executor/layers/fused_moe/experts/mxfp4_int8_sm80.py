# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental DSV4.1 MXFP4 -> INT8 modular MoE backend for SM80."""

from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
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
from vllm.profiler.phase_profile_hook import optional_phase_profile_stage

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_HIDDEN_SIZE = 5120
_INTERMEDIATE_SIZE = 2304
_TARGET_TOPK = 6
_DRAFT_TOPK = 3
_DECODE_NUM_EXPERTS = 384
_DECODE_BLOCK_M = 16
_TILE_TABLE: tuple[tuple[int, int, int], ...] = (
    (1, 16, 128),
    (1024, 16, 128),
    (2048, 64, 128),
    (1 << 30, 128, 128),
)


# Block-M candidates the kernel is instantiated for, ascending.
_BLOCK_M_CANDIDATES: tuple[int, ...] = (16, 64, 128)
_BLOCK_N = 128
# The single-token, top-k 6 decode step reads one 4-bit expert slab per slot, so
# its grouped GEMM is DRAM-latency bound rather than bandwidth bound: halving
# block-N doubles the CTA count and buys about 12% on the W13+W2 pair in
# isolation. Only the decode tile is overridable -- larger batches pick
# block-M 64/128, whose supported block-N set is different.
_DECODE_BLOCK_N = envs.VLLM_DEEPSEEK_V4_SM80_MOE_DECODE_BLOCK_N
_SUPPORTED_DECODE_BLOCK_N: tuple[int, ...] = (64, 128)


def select_mxfp4_int8_sm80_tile(
    num_tokens: int,
    topk: int,
    num_experts: int,
) -> tuple[int, int]:
    """Smallest supported block_m that covers the rows each expert receives.

    Sizing block_m to the *average* rows per expert keeps one M tile per expert,
    so its mxfp4 weights are read from DRAM once per step. The previous table
    lookup capped block_m at the largest power of two <= the average and had no
    entry covering (2048, 1<<30) with community block_m, so 2048- and
    4096-token prefill chunks silently fell through to the ``16`` fallback --
    the slowest tiling measured (20.6 ms/layer against 17.8 ms for block_m=64
    at 2048 tokens on a CMP 170HX, and worse at 4096).
    """
    average = max(1, (num_tokens * topk + num_experts - 1) // num_experts)
    block_m = _BLOCK_M_CANDIDATES[-1]
    for candidate in _BLOCK_M_CANDIDATES:
        if candidate >= average:
            block_m = candidate
            break
    if block_m != _DECODE_BLOCK_M:
        return block_m, _BLOCK_N
    if _DECODE_BLOCK_N not in _SUPPORTED_DECODE_BLOCK_N:
        raise ValueError(
            f"unsupported MXFP4-int8 SM80 decode block-N {_DECODE_BLOCK_N}; "
            f"supported values are {_SUPPORTED_DECODE_BLOCK_N}"
        )
    return block_m, _DECODE_BLOCK_N


def dsv41_sm80_tilelang_enabled(
    config: "VllmConfig | None", moe: FusedMoEConfig
) -> bool:
    """Opt in only the DSV41F target experts; shape alone is not identity."""
    if config is None or config.model_config is None:
        return False
    hf_config = config.model_config.hf_config
    if getattr(hf_config, "model_type", None) != "deepseek_v41":
        return False
    extra = config.additional_config
    value = (
        extra.get("dsv41_sm80_moe_tilelang", False)
        if isinstance(extra, dict)
        else False
    )
    if not isinstance(value, bool):
        raise ValueError("dsv41_sm80_moe_tilelang must be a JSON boolean")
    return (
        value
        and getattr(hf_config, "hidden_size", None) == _HIDDEN_SIZE
        and getattr(hf_config, "moe_intermediate_size", None) == _INTERMEDIATE_SIZE
        and getattr(hf_config, "n_routed_experts", None) == _DECODE_NUM_EXPERTS
        and moe.num_experts == moe.num_local_experts == _DECODE_NUM_EXPERTS
        and moe.experts_per_token == _TARGET_TOPK
        and moe.hidden_dim == _HIDDEN_SIZE
        and moe.intermediate_size_per_partition == _INTERMEDIATE_SIZE
        and moe.in_dtype == torch.bfloat16
        and moe.activation == MoEActivation.SILU
        and not moe.has_bias
        and not moe.is_lora_enabled
        and Mxfp4Int8Sm80Experts._supports_parallel_config(moe.moe_parallel_config)
    )


def make_mxfp4_int8_sm80_decode_metadata(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the immutable M=1, top-k 6, block-M 16 route layout."""
    sorted_ids = torch.full(
        (_TARGET_TOPK * _DECODE_BLOCK_M,),
        _TARGET_TOPK,
        dtype=torch.int32,
        device=device,
    )
    sorted_ids[::_DECODE_BLOCK_M] = torch.arange(
        _TARGET_TOPK, dtype=torch.int32, device=device
    )
    num_post_pad = torch.tensor(
        [_TARGET_TOPK * _DECODE_BLOCK_M], dtype=torch.int32, device=device
    )
    return sorted_ids, num_post_pad


class Mxfp4Int8Sm80Experts(mk.FusedMoEExpertsModular):
    """SM80 INT8-MMA experts for the fixed DSV4.1 TP1 expert shape.

    Pipeline parallelism is outside the MoE tensor layout, so PP3 is supported
    while TP/EP/DP/PCP/SP sharding deliberately fails closed.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
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
            or moe_config.experts_per_token not in (_DRAFT_TOPK, _TARGET_TOPK)
        ):
            raise ValueError(
                "mxfp4_int8_sm80 currently requires DSV4.1 TP1 shape "
                f"H={_HIDDEN_SIZE}, I={_INTERMEDIATE_SIZE}, topk=3 or 6; got "
                f"H={moe_config.hidden_dim}, "
                f"I={moe_config.intermediate_size_per_partition}, "
                f"topk={moe_config.experts_per_token}"
            )
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        self.w13_shift2: torch.Tensor | None = None
        self.w2_shift2: torch.Tensor | None = None
        self.decode_sorted_ids: torch.Tensor | None = None
        self.decode_num_post_pad: torch.Tensor | None = None
        self._tilelang_requested = dsv41_sm80_tilelang_enabled(
            get_current_vllm_config_or_none(), moe_config
        )
        self._tilelang_decode_gemms = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.w13_shift2 = layer.w13_mxfp4_int8_shift2
        self.w2_shift2 = layer.w2_mxfp4_int8_shift2
        self.decode_sorted_ids, self.decode_num_post_pad = (
            make_mxfp4_int8_sm80_decode_metadata(self.w13_shift2.device)
        )
        self._tilelang_decode_gemms = None
        if self._tilelang_requested and current_platform.is_cuda():
            capability = current_platform.get_device_capability(
                self.w13_shift2.device.index
            )
            if (
                capability is not None
                and capability.major == 8
                and capability.minor == 0
            ):
                from vllm.model_executor.layers.fused_moe.experts import (
                    dsv41_sm80_tilelang,
                )

                self._tilelang_decode_gemms = dsv41_sm80_tilelang.prepare_decode_gemms(
                    layer,
                    self.w1_scale,
                    self.w2_scale,
                    self.decode_sorted_ids,
                    self.decode_num_post_pad,
                )
                logger.info_once(
                    "DSV41F SM80 TileLang PRMT enabled for M=1, E=384, top-k6; "
                    "other shapes use CUTLASS."
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
        # overwrites slots belonging to later tokens. Pack the W13 output and
        # activation into disjoint flat regions of workspace13 so both views
        # have the tight row stride required by the CUDA kernels.
        return (
            (M * topk, 3 * _INTERMEDIATE_SIZE),
            (M * topk, _HIDDEN_SIZE),
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
        assert self.w13_shift2 is not None
        assert self.w2_shift2 is not None

        m = hidden_states.size(0)
        topk = topk_ids.size(1)
        num_experts = w1.size(0)
        block_m, block_n = select_mxfp4_int8_sm80_tile(m, topk, num_experts)
        with optional_phase_profile_stage("moe.mxfp4_sm80.routing_cast", hidden_states):
            routing_ids = topk_ids.to(torch.int32).contiguous()
            routing_weights = topk_weights.to(torch.float32).contiguous()
        use_persistent_decode_metadata = (
            m == 1
            and topk == _TARGET_TOPK
            and num_experts == global_num_experts == _DECODE_NUM_EXPERTS
            and block_m == _DECODE_BLOCK_M
        )
        if use_persistent_decode_metadata:
            assert self.decode_sorted_ids is not None
            assert self.decode_num_post_pad is not None
            sorted_ids = self.decode_sorted_ids
            expert_ids = routing_ids.view(-1)
            num_post_pad = self.decode_num_post_pad
        else:
            with optional_phase_profile_stage("moe.mxfp4_sm80.align", hidden_states):
                sorted_ids, expert_ids, num_post_pad = moe_align_block_size(
                    routing_ids,
                    block_m,
                    global_num_experts,
                )

        decode_gemms = (
            self._tilelang_decode_gemms if use_persistent_decode_metadata else None
        )

        num_routed = m * topk
        scratch = workspace13.flatten()
        gate_up_numel = num_routed * 2 * _INTERMEDIATE_SIZE
        gate_up = scratch[:gate_up_numel].view(num_routed, 2 * _INTERMEDIATE_SIZE)
        intermediate = scratch[
            gate_up_numel : gate_up_numel + num_routed * _INTERMEDIATE_SIZE
        ].view(num_routed, _INTERMEDIATE_SIZE)
        # Reuse the W13 output after activation. A2 quantization is enqueued
        # before W2 on the same stream, so overwriting this storage is ordered.
        routed_out = workspace2.view(-1, _HIDDEN_SIZE)[:num_routed]

        with optional_phase_profile_stage("moe.mxfp4_sm80.w13_quant", hidden_states):
            a13_q, a13_scale = quantize_per_token_int8(hidden_states)
        with optional_phase_profile_stage("moe.mxfp4_sm80.w13_gemm", hidden_states):
            if decode_gemms is not None:
                decode_gemms[0](
                    a13_q,
                    a13_scale,
                    w1,
                    self.w13_shift2,
                    self.w1_scale,
                    sorted_ids,
                    expert_ids,
                    num_post_pad,
                    gate_up,
                )
            else:
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
        with optional_phase_profile_stage("moe.mxfp4_sm80.activation", gate_up):
            self.activation(activation, intermediate, gate_up)

        with optional_phase_profile_stage("moe.mxfp4_sm80.w2_quant", intermediate):
            a2_q, a2_token_scale = quantize_per_token_int8(intermediate)
        with optional_phase_profile_stage("moe.mxfp4_sm80.w2_gemm", intermediate):
            if decode_gemms is not None:
                decode_gemms[1](
                    a2_q,
                    a2_token_scale,
                    w2,
                    self.w2_shift2,
                    self.w2_scale,
                    sorted_ids,
                    expert_ids,
                    num_post_pad,
                    routed_out,
                )
            else:
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
        with optional_phase_profile_stage("moe.mxfp4_sm80.reduce", routed_out):
            torch.ops._moe_C.dsv4_mxfp4_int8_sm80_moe_reduce(
                routed_out,
                routing_weights,
                output,
                m * topk,
            )
