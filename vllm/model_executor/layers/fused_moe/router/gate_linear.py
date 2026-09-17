# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch.nn.parameter import Parameter

import vllm._custom_ops as ops
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


@PluggableLayer.register("gate_linear")
class GateLinear(ReplicatedLinear):
    """MoE gate linear layer with multi-tier GEMM dispatch:

    1. cuteDSL ll_bf16_gemm (SM90+, M<=16, bf16 in, fp32 out,
       K divisible by 8)
    2. DSV3 specialized kernel (SM90+, M<=16, H=7168 E=256/384, H=6144 E=256)
    3. fp32 specialized kernel  (SM90+, bf16/fp32 in, fp32 out, M<=32,
       (H, E) in {(3072, 256), (6144, 128), (6144, 256)})
    4. experimental bf16x3 CuteDSL kernel (opt-in, SM100, bf16 in, fp32 weight)
    5. cuBLAS bf16×bf16→fp32 (SM90+ + bf16 weight + fp32 out_dtype)
    6. F.linear via ReplicatedLinear (ultimate fallback)

    The ``out_dtype`` attribute is mutable and can be set after init
    (e.g. when the required dtype depends on the expert quantization
    method which is only known later).
    """

    # Dimensions supported by the DSV3 specialized kernel.
    # Valid (hidden_size, num_experts) combinations:
    #   (7168, 256) -> DeepSeek-V3,  (7168, 384) -> Kimi-K2,
    #   (6144, 256) -> GLM-5
    DSV3_SUPPORTED_NUM_EXPERTS = [256, 384]
    DSV3_SUPPORTED_HIDDEN_SIZES = [7168, 6144]
    # num_experts=384 is only instantiated for hidden_size=7168.
    DSV3_UNSUPPORTED_SHAPES = {(6144, 384)}

    # (hidden_size, num_experts) pairs with an instantiated fp32 kernel:
    #   (3072, 256) -> MiniMax-M2/M2.5,  (6144, 128) -> MiniMax-M3
    FP32_SUPPORTED_SHAPES = {(3072, 256), (6144, 128), (6144, 256)}
    FP32_MAX_TOKENS = 32

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        out_dtype: torch.dtype | None = None,
        params_dtype: torch.dtype | None = None,
        force_fp32_compute: bool = False,
        prefix: str = "",
    ):
        is_hopper = current_platform.is_device_capability((9, 0))
        is_blackwell = current_platform.is_device_capability_family(100)
        can_use_specialized_kernels = (
            current_platform.is_cuda() and (is_hopper or is_blackwell) and not bias
        )

        # If fp32 compute is required and no specialized kernel is available,
        # store weights in fp32 so the fallback linear path computes in fp32.
        if force_fp32_compute and not can_use_specialized_kernels:
            params_dtype = torch.float32

        super().__init__(
            input_size,
            output_size,
            bias=bias,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=prefix,
        )
        self.out_dtype = out_dtype

        # DSV3 specialized kernel eligibility (SM90+, exact dims)
        self.allow_specialized_router_gemm = can_use_specialized_kernels
        self.allow_dsv3_router_gemm = (
            self.allow_specialized_router_gemm
            and self.weight.dtype == torch.bfloat16
            and output_size in self.DSV3_SUPPORTED_NUM_EXPERTS
            and input_size in self.DSV3_SUPPORTED_HIDDEN_SIZES
            and (input_size, output_size) not in self.DSV3_UNSUPPORTED_SHAPES
        )
        # See https://github.com/vllm-project/vllm/pull/44217
        # for more details.
        self._dsv3_max_batch = 16 if is_hopper else 8

        # fp32 specialized kernel eligibility (SM90+, exact dims, fp32 weight)
        vllm_config = get_current_vllm_config_or_none()
        enable_bf16x3_router_gemm = (
            vllm_config is not None
            and vllm_config.kernel_config.enable_bf16x3_router_gemm
        )
        self.allow_fp32_router_gemm = (
            not bias
            and self.weight.dtype == torch.float32
            and current_platform.is_cuda()
            and (is_hopper or is_blackwell)
            and (input_size, output_size) in self.FP32_SUPPORTED_SHAPES
        )
        self.allow_bf16x3_router_gemm = (
            not bias
            and self.weight.dtype == torch.float32
            and current_platform.is_cuda()
            and is_blackwell
            and input_size % 8 == 0
            and enable_bf16x3_router_gemm
        )
        if self.allow_bf16x3_router_gemm:
            logger.info_once("Enabled experimental SM100 BF16x3 router GEMM.")

        # Fused bf16 x bf16 -> fp32 GEMM eligibility. torch.mm's out_dtype
        # epilogue folds the fp32 cast into the GEMM, removing the standalone
        # bf16->fp32 copy kernel that otherwise runs before grouped_topk. This is
        # the plain cuBLAS (CUDA) / hipBLASLt (ROCm) out_dtype epilogue, so it
        # applies on any CUDA-alike device (no bias, since torch.mm has no bias
        # term). The specialized-kernel gate above excludes family-120 Blackwell
        # (GB10 / DGX Spark), which this tier still covers. See #49921.
        self._router_gemm_no_bias = not bias
        self._router_gemm_cublas_capable = (
            current_platform.is_cuda() or current_platform.is_rocm()
        ) and self._router_gemm_no_bias
        self.allow_cublas_router_gemm = (
            self._router_gemm_cublas_capable
            and self.weight.dtype == torch.bfloat16
            and self.out_dtype == torch.float32
        )

        # cuteDSL ll_bf16_gemm eligibility. Any dims supported, but SM90+ required bc:
        # 1. PDL support. Both dot-product and split-K kernels.
        # 2. Thread Block Clusters. Split-K kernel for cross-CTA reduction.
        self.allow_ll_bf16_gemm = False
        if can_use_specialized_kernels:
            from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
                is_available,
            )

            self.allow_ll_bf16_gemm = (
                self.weight.dtype == torch.bfloat16
                and self.out_dtype == torch.float32
                and is_available()
            )

    def set_out_dtype(self, out_dtype: torch.dtype) -> None:
        """Set output dtype for the router logits after init.

        Useful when the required dtype depends on the expert quantization
        method which is only known after the gate is constructed.
        """
        if self.out_dtype is not None:
            raise ValueError("out_dtype has already been set")
        self.out_dtype = out_dtype

        if (
            not self.allow_cublas_router_gemm
            and self._router_gemm_cublas_capable
            and out_dtype == torch.float32
        ):
            self.allow_cublas_router_gemm = self.weight.dtype == torch.bfloat16

        # out_dtype may start as None -> recompute eligibility here
        if self.allow_specialized_router_gemm:
            from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
                is_available,
            )

            self.allow_ll_bf16_gemm = (
                self.weight.dtype == torch.bfloat16
                and out_dtype == torch.float32
                and is_available()
            )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        # Tier 1: cuteDSL ll_bf16_gemm (SM90+, any dims)
        if self.allow_ll_bf16_gemm and x.shape[0] <= 16 and x.dtype == torch.bfloat16:
            from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
                ll_bf16_gemm,
            )

            output = ll_bf16_gemm(x, self.weight)
            return output, None

        # Tier 2: DSV3 specialized kernel (fallback for when cuteDSL unavailable)
        if self.allow_dsv3_router_gemm and x.shape[0] <= self._dsv3_max_batch:
            output = ops.dsv3_router_gemm(
                hidden_states=x,
                router_weight=self.weight,
                output_dtype=self.out_dtype,
            )
            return output, None

        # Tier 3: fp32 specialized kernel (H=3072, E=256, M<=32)
        # Dispatch is wrapped in a custom op so that torch.compile/CUDA-graph
        # capture does not freeze the runtime num_tokens branch.
        if self.allow_fp32_router_gemm and x.dtype in (
            torch.float32,
            torch.bfloat16,
        ):
            output = torch.ops.vllm.fp32_router_gemm_dispatch(
                x, self.weight, self.allow_bf16x3_router_gemm
            )
            return output, None

        # Tier 4: experimental bf16x3 CuteDSL kernel for fp32 router weights
        if self.allow_bf16x3_router_gemm and x.dtype == torch.bfloat16:
            from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_cutedsl import (  # noqa: E501
                bf16x3_router_gemm,
            )

            output = bf16x3_router_gemm(x, self.weight)
            return output, None

        # Tier 5: cuBLAS bf16→fp32
        if self.allow_cublas_router_gemm and x.dtype == torch.bfloat16:
            output = _router_gemm_pinned(x, self.weight)
            return output, None

        # Tier 6: F.linear (ReplicatedLinear)
        if self.out_dtype is not None and x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        output, output_bias = super().forward(x)
        if self.out_dtype is not None and output.dtype != self.out_dtype:
            output = output.to(self.out_dtype)
        return output, output_bias


_FP32_ROUTER_GEMM_MAX_TOKENS = GateLinear.FP32_MAX_TOKENS

# The MoE router GEMM is the one GEMM in a DeepSeek-V4.1 forward whose numerics
# depend on how many tokens happen to share the batch, so the same request gets
# different logits at different concurrency and a near-tie can flip (at width 8
# this cost every prompt of the four-cell gate its correct answer, 30/48).
#
# cuBLAS picks its kernel from the problem shape: for layers.1.ffn.gate
# (5120 -> 384, bf16 in, fp32 out) M in [14, 64] is one kernel and 65..128,
# 129..192, ... are others, with the two disagreeing in thousands of entries by
# up to 3.4e-6. A solo request is always in the M<=64 regime, so pinning every
# batch to that regime makes the co-batched result identical to the solo one
# instead of swapping in a new summation order, and it leaves every answer the
# solo path produces today untouched.
#
# VLLM_DSV41_ROUTER_GEMM_ALIGN selects how:
#   ""      no pin (previous behaviour)
#   "split" M<=64 as-is, M>64 through a 16-way split-K Triton kernel that
#           reproduces that regime bit-for-bit (measured M=14..8192)
#   "chunk" call cuBLAS once per 64 rows instead; equally exact but one launch
#           per chunk, kept as the reference implementation for the kernel
_ROUTER_CHUNK_M = 64


def _router_gemm_canonical_chunked(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Keep every router GEMM inside cuBLAS's M<=64 regime.

    cuBLAS picks its kernel from the problem shape, so a token's router logits
    depend on how many tokens share the batch. For layers.1.ffn.gate (5120->384)
    the production call is bit-identical for M in [14, 64] and changes at 65,
    129, 193 ...; a 5-request batch of 14-token prompts is M=70, which is why a
    co-batched prefill disagrees with a solo one.

    Calling cuBLAS once per 64 rows (zero-padding the tail, which the measurements
    show is exact) keeps every call in the M<=64 regime, so each row gets exactly
    the numerics a solo request gets today -- invariant across batches without
    inventing a new summation order. Decode and prefills up to 64 tokens need no
    chunking at all, so the cost falls only on longer prefills.
    """
    m, k = x.shape
    if m <= _ROUTER_CHUNK_M:
        return torch.mm(x, weight.T, out_dtype=torch.float32)
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    for start in range(0, m, _ROUTER_CHUNK_M):
        part = x[start : start + _ROUTER_CHUNK_M]
        rows = part.shape[0]
        if rows < _ROUTER_CHUNK_M:
            part = torch.cat([part, part.new_zeros(_ROUTER_CHUNK_M - rows, k)], dim=0)
        chunk = torch.mm(part, weight.T, out_dtype=torch.float32)
        out[start : start + rows] = chunk[:rows]
    return out


_ROUTER_SPLIT_K = 16

# Cached transposed router weights, keyed by the weight tensor's address. The
# weight itself is stored alongside so the address cannot be recycled by another
# layer's gate.
_ROUTER_W_CACHE: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


@triton.jit
def _router_gemm_split_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """Router GEMM with the same K-split structure cuBLAS uses for M<=64.

    cuBLAS's small-M kernel splits K into 16 segments, accumulates each in
    increasing K order, and adds the 16 partials in order. Reproducing that
    structure makes this kernel bit-identical to the canonical result a solo
    request gets today (verified for M=14..8192), while a single launch keeps
    long prefills cheap -- unlike chunking cuBLAS, which is also exact but costs
    one launch per 64 rows.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    per_split = tl.cdiv(tl.cdiv(K, BLOCK_K), SPLIT_K)
    n_mask = None if EVEN_N else offs_n[None, :] < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for split in tl.static_range(SPLIT_K):
        part = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for i in range(0, per_split):
            k = (split * per_split + i) * BLOCK_K + offs_k
            a_mask = m_mask[:, None] if EVEN_K else m_mask[:, None] & (k[None, :] < K)
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
                mask=a_mask,
                other=0.0,
            )
            b_ptrs = b_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            if EVEN_K and EVEN_N:
                b = tl.load(b_ptrs)
            elif EVEN_K:
                b = tl.load(b_ptrs, mask=n_mask, other=0.0)
            elif EVEN_N:
                b = tl.load(b_ptrs, mask=k[:, None] < K, other=0.0)
            else:
                b = tl.load(b_ptrs, mask=(k[:, None] < K) & n_mask, other=0.0)
            part = tl.dot(a, b, part)
        acc += part
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=m_mask[:, None] if EVEN_N else m_mask[:, None] & (offs_n[None, :] < N),
    )


def _router_gemm_split_k(
    x: torch.Tensor,
    weight: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
) -> torch.Tensor:
    key = weight.data_ptr()
    entry = _ROUTER_W_CACHE.get(key)
    if entry is None or entry[0] is not weight:
        entry = (weight, weight.t().contiguous())
        _ROUTER_W_CACHE[key] = entry
    bt = entry[1]
    m, k = x.shape
    n = bt.shape[1]
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _router_gemm_split_kernel[grid](
        x,
        bt,
        out,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        bt.stride(0),
        bt.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SPLIT_K=_ROUTER_SPLIT_K,
        EVEN_K=(triton.cdiv(k, block_k) % _ROUTER_SPLIT_K == 0),
        EVEN_N=(n % block_n == 0),
        num_warps=4,
    )
    return out


def _router_gemm_pinned(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Keep every router GEMM inside cuBLAS's M<=64 regime.

    M<=64 already is that regime, so the plain call is canonical and costs
    nothing; above it the split-K kernel reproduces the same regime bit-for-bit
    (measured M=14..8192), which is what makes a batched prefill agree with a
    solo one without changing any answer the solo path produces today.

    The regime was characterised on SM80, which is the only architecture where
    this tier is the router's exclusive path; elsewhere the same shape is served
    by a different cuBLAS kernel, so the pin stays off rather than swapping in
    numerics nobody measured.
    """
    from vllm import envs

    mode = envs.VLLM_DSV41_ROUTER_GEMM_ALIGN
    if not mode or not current_platform.is_device_capability((8, 0)):
        return torch.mm(x, weight.T, out_dtype=torch.float32)
    if mode == "chunk":
        return _router_gemm_canonical_chunked(x, weight)
    if x.shape[0] <= _ROUTER_CHUNK_M:
        return torch.mm(x, weight.T, out_dtype=torch.float32)
    return _router_gemm_split_k(x, weight)


def fp32_router_gemm_dispatch_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    allow_bf16x3_router_gemm: bool,
) -> torch.Tensor:
    """
    Dynamically run fp32 specialized gemm if num_tokens <= FP32_MAX_TOKENS,
    otherwise optionally run the experimental BF16x3 kernel for medium/large
    SM100 router batches, then fall back to F.linear.
    This must be wrapped in a custom op because our torch.compile integration
    does not support runtime dispatching on num_tokens.
    """
    if x.shape[0] <= _FP32_ROUTER_GEMM_MAX_TOKENS:
        return ops.fp32_router_gemm(x, weight)

    if allow_bf16x3_router_gemm and x.dtype == torch.bfloat16:
        from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_cutedsl import (  # noqa: E501
            bf16x3_router_gemm,
        )

        return bf16x3_router_gemm(x, weight)

    return torch.nn.functional.linear(x.float(), weight)


def fp32_router_gemm_dispatch_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    allow_bf16x3_router_gemm: bool,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.float32)


direct_register_custom_op(
    op_name="fp32_router_gemm_dispatch",
    op_func=fp32_router_gemm_dispatch_impl,
    fake_impl=fp32_router_gemm_dispatch_fake,
)
