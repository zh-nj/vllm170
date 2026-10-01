# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SwiGLU and numerical INT8 A2 quantization for opted-in DSV41F on SM80."""

# T.Tensor annotations are evaluated by the TileLang parser.
# mypy: disable-error-code="valid-type"

from functools import cache

import torch

from vllm.tilelang_utils import T, tilelang

INTERMEDIATE_SIZE = 2304
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
}

_MATH_SOURCE = r"""
#include <cuda_bf16.h>
#include <stdint.h>
template <bool Fast>
__device__ __forceinline__ float dsv41_swiglu_rounded(float gate, float up) {
  float activated;
  if constexpr (Fast) {
    float denom = 1.0f + __expf(-gate);
    asm("div.full.f32 %0, %1, %2;" : "=f"(activated) : "f"(gate), "f"(denom));
  } else {
    activated = gate / (1.0f + expf(-gate));
  }
  activated = __bfloat162float(__float2bfloat16_rn(activated));
  return __bfloat162float(__float2bfloat16_rn(activated * up));
}
template <bool Fast, bool Clamp>
__device__ __forceinline__ float dsv41_swiglu_input_rounded(
    float gate, float up, float limit) {
  if constexpr (Clamp) {
    gate = __bfloat162float(__float2bfloat16_rn(fminf(gate, limit)));
    up = __bfloat162float(__float2bfloat16_rn(fmaxf(fminf(up, limit), -limit)));
  }
  return dsv41_swiglu_rounded<Fast>(gate, up);
}
__device__ __forceinline__ int8_t dsv41_quant_i8(float value, float scale) {
  float normalized;
  asm("div.full.f32 %0, %1, %2;" : "=f"(normalized) : "f"(value), "f"(scale));
  int q = __float2int_rn(normalized);
  return static_cast<int8_t>(min(max(q, -127), 127));
}
__device__ __forceinline__ float dsv41_quant_scale(float amax) {
  float scale;
  asm("div.full.f32 %0, %1, %2;" : "=f"(scale) : "f"(amax), "f"(127.0f));
  return scale == 0.0f ? 1.0f : scale;
}
"""


def build_swiglu_int8(
    *, threads: int = 256, fast_math: bool = True, clamp_limit: float | None = None
):
    """Preserve both CUDA BF16 rounds and Triton's full-range INT8 division."""
    if threads not in (128, 256, 512, 1024):
        raise ValueError("threads must be 128, 256, 512, or 1024")
    if clamp_limit not in (None, 10.0):
        raise ValueError("the frozen candidate supports only no clamp or clamp=10")
    rows = T.dynamic("rows")
    block_k = tilelang.next_power_of_2(INTERMEDIATE_SIZE)
    fast = "true" if fast_math else "false"
    clamp = "false" if clamp_limit is None else "true"
    func = f"dsv41_swiglu_input_rounded<{fast}, {clamp}>"
    limit = 0.0 if clamp_limit is None else clamp_limit

    @T.prim_func
    def kernel(
        gate_up: T.Tensor((rows, 2 * INTERMEDIATE_SIZE), "bfloat16"),
        out: T.Tensor((rows, INTERMEDIATE_SIZE), "int8"),
        scale: T.Tensor((rows,), "float32"),
    ):
        with T.Kernel(rows, threads=threads) as row:
            T.import_source(_MATH_SOURCE)
            values = T.alloc_fragment((block_k,), "float32")
            magnitudes = T.alloc_fragment((block_k,), "float32")
            amax = T.alloc_fragment((1,), "float32")
            for col in T.Parallel(block_k):
                values[col] = T.if_then_else(
                    col < INTERMEDIATE_SIZE,
                    T.call_extern(
                        "float32",
                        func,
                        T.Cast("float32", gate_up[row, col]),
                        T.Cast("float32", gate_up[row, INTERMEDIATE_SIZE + col]),
                        T.float32(limit),
                    ),
                    0.0,
                )
                magnitudes[col] = T.abs(values[col])
            T.reduce_max(magnitudes, amax, dim=0)
            for col in T.Parallel(block_k):
                sf = T.call_extern("float32", "dsv41_quant_scale", amax[0])
                if col < INTERMEDIATE_SIZE:
                    out[row, col] = T.call_extern(
                        "int8", "dsv41_quant_i8", values[col], sf
                    )
                if col == 0:
                    scale[row] = sf

    return kernel


@cache
def compile_activation_quant(device_index: int | None, clamp_limit: float | None):
    """Compile the frozen 256-thread candidate once per worker/device."""
    with torch.accelerator.device_index(device_index):
        if torch.cuda.get_device_capability() != (8, 0):
            raise ValueError("DSV41F TileLang activation requires SM80")
        return tilelang.compile(
            build_swiglu_int8(clamp_limit=clamp_limit), pass_configs=PASS_CONFIGS
        )


def prepare_activation_quant(device: torch.device, clamp_limit: float | None):
    """Warm the adapter at model load, before forward or Graph capture."""
    kernel = compile_activation_quant(device.index, clamp_limit)
    gate_up = torch.zeros(
        (6, 2 * INTERMEDIATE_SIZE), device=device, dtype=torch.bfloat16
    )
    q = torch.empty((6, INTERMEDIATE_SIZE), device=device, dtype=torch.int8)
    scale = torch.empty((6,), device=device, dtype=torch.float32)
    kernel(gate_up, q, scale)
    return kernel
