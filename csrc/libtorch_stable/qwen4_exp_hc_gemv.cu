// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "core/registration.h"
#include "libtorch_stable/torch_utils.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kOutputSize = 336;
constexpr int kHiddenSize = 10240;
constexpr int kVectorWidth = 8;

template <int kTokens, int kRowsPerBlock, int kThreads>
__global__ __launch_bounds__(kThreads) void qwen4ExpHcDownKernel(
    __nv_bfloat16 const* __restrict__ input,
    __nv_bfloat16 const* __restrict__ weight,
    __nv_bfloat16* __restrict__ output) {
  constexpr int kWarps = kThreads / 32;
  const int row_base = blockIdx.x * kRowsPerBlock;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  float sums[kRowsPerBlock][kTokens] = {};

  for (int offset = threadIdx.x * kVectorWidth; offset < kHiddenSize;
       offset += kThreads * kVectorWidth) {
    uint4 input_vectors[kTokens];
#pragma unroll
    for (int token = 0; token < kTokens; ++token) {
      input_vectors[token] =
          *reinterpret_cast<uint4 const*>(input + token * kHiddenSize + offset);
    }

#pragma unroll
    for (int row_offset = 0; row_offset < kRowsPerBlock; ++row_offset) {
      const int row = row_base + row_offset;
      const uint4 weight_vector = *reinterpret_cast<uint4 const*>(
          weight + static_cast<int64_t>(row) * kHiddenSize + offset);
      auto const* weight_values =
          reinterpret_cast<__nv_bfloat16 const*>(&weight_vector);

#pragma unroll
      for (int token = 0; token < kTokens; ++token) {
        auto const* input_values =
            reinterpret_cast<__nv_bfloat16 const*>(&input_vectors[token]);
#pragma unroll
        for (int index = 0; index < kVectorWidth; ++index) {
          sums[row_offset][token] = fmaf(__bfloat162float(input_values[index]),
                                         __bfloat162float(weight_values[index]),
                                         sums[row_offset][token]);
        }
      }
    }
  }

  __shared__ float warp_sums[kRowsPerBlock][kTokens][kWarps];
#pragma unroll
  for (int row_offset = 0; row_offset < kRowsPerBlock; ++row_offset) {
#pragma unroll
    for (int token = 0; token < kTokens; ++token) {
      float value = sums[row_offset][token];
#pragma unroll
      for (int delta = 16; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, delta);
      }
      if (lane == 0) {
        warp_sums[row_offset][token][warp] = value;
      }
    }
  }
  __syncthreads();

  if (warp == 0) {
#pragma unroll
    for (int row_offset = 0; row_offset < kRowsPerBlock; ++row_offset) {
#pragma unroll
      for (int token = 0; token < kTokens; ++token) {
        float value = lane < kWarps ? warp_sums[row_offset][token][lane] : 0.0f;
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {
          value += __shfl_down_sync(0xffffffff, value, delta);
        }
        if (lane == 0) {
          output[token * kOutputSize + row_base + row_offset] =
              __float2bfloat16_rn(value);
        }
      }
    }
  }
}

void qwen4_exp_hc_sm80_down(torch::stable::Tensor& output,
                            torch::stable::Tensor const& input,
                            torch::stable::Tensor const& weight) {
  STD_TORCH_CHECK(output.is_cuda() && input.is_cuda() && weight.is_cuda(),
                  "qwen4_exp_hc_sm80_down: tensors must be CUDA tensors");
  STD_TORCH_CHECK(output.get_device_index() == input.get_device_index() &&
                      output.get_device_index() == weight.get_device_index(),
                  "qwen4_exp_hc_sm80_down: tensors must be on the same device");
  STD_TORCH_CHECK(
      output.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
          input.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
          weight.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "qwen4_exp_hc_sm80_down: tensors must be bfloat16");
  STD_TORCH_CHECK(
      output.is_contiguous() && input.is_contiguous() && weight.is_contiguous(),
      "qwen4_exp_hc_sm80_down: tensors must be contiguous");
  STD_TORCH_CHECK(input.dim() == 2 &&
                      (input.size(0) == 1 || input.size(0) == 4) &&
                      input.size(1) == kHiddenSize,
                  "qwen4_exp_hc_sm80_down: input must have shape [1, 10240] "
                  "or [4, 10240]");
  STD_TORCH_CHECK(
      weight.dim() == 2 && weight.size(0) == kOutputSize &&
          weight.size(1) == kHiddenSize,
      "qwen4_exp_hc_sm80_down: weight must have shape [336, 10240]");
  STD_TORCH_CHECK(output.dim() == 2 && output.size(0) == input.size(0) &&
                      output.size(1) == kOutputSize,
                  "qwen4_exp_hc_sm80_down: output must have shape [M, 336] "
                  "matching input M");

  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  auto* properties = get_device_prop();
  STD_TORCH_CHECK(properties->major == 8 && properties->minor == 0,
                  "qwen4_exp_hc_sm80_down: requires SM80, got SM",
                  properties->major, properties->minor);

  auto const* input_ptr =
      reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr());
  auto const* weight_ptr =
      reinterpret_cast<__nv_bfloat16 const*>(weight.data_ptr());
  auto* output_ptr = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
  STD_TORCH_CHECK(
      reinterpret_cast<uintptr_t>(input_ptr) % alignof(uint4) == 0 &&
          reinterpret_cast<uintptr_t>(weight_ptr) % alignof(uint4) == 0,
      "qwen4_exp_hc_sm80_down: input and weight must be 16-byte "
      "aligned");

  const cudaStream_t stream = get_current_cuda_stream(input.get_device_index());
  if (input.size(0) == 1) {
    constexpr int kRowsPerBlock = 2;
    constexpr int kThreads = 512;
    qwen4ExpHcDownKernel<1, kRowsPerBlock, kThreads>
        <<<kOutputSize / kRowsPerBlock, kThreads, 0, stream>>>(
            input_ptr, weight_ptr, output_ptr);
  } else {
    constexpr int kRowsPerBlock = 1;
    constexpr int kThreads = 256;
    qwen4ExpHcDownKernel<4, kRowsPerBlock, kThreads>
        <<<kOutputSize / kRowsPerBlock, kThreads, 0, stream>>>(
            input_ptr, weight_ptr, output_ptr);
  }
}

}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_C, qwen4_exp_hc_ops) {
  qwen4_exp_hc_ops.def(
      "qwen4_exp_hc_sm80_down(Tensor! output, Tensor input, Tensor weight) "
      "-> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, qwen4_exp_hc_ops) {
  qwen4_exp_hc_ops.impl("qwen4_exp_hc_sm80_down",
                        TORCH_BOX(&qwen4_exp_hc_sm80_down));
}
