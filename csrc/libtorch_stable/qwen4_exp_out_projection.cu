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

constexpr int kTokens = 4;
constexpr int kOutputSize = 2560;
constexpr int kHiddenSize = 6144;
constexpr int kRowsPerBlock = 4;
constexpr int kThreads = 64;
constexpr int kVectorWidth = 8;

__global__ __launch_bounds__(kThreads) void qwen4ExpOutProjectionKernel(
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
      const uint4 weight_vector = *reinterpret_cast<uint4 const*>(
          weight + static_cast<int64_t>(row_base + row_offset) * kHiddenSize +
          offset);
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

void checkTensor(torch::stable::Tensor const& tensor, int64_t rows,
                 int64_t columns, int device_index, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  STD_TORCH_CHECK(tensor.get_device_index() == device_index, name,
                  " must be on the input device");
  STD_TORCH_CHECK(
      tensor.scalar_type() == torch::headeronly::ScalarType::BFloat16, name,
      " must be bfloat16");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  STD_TORCH_CHECK(
      tensor.dim() == 2 && tensor.size(0) == rows && tensor.size(1) == columns,
      name, " has an unexpected shape");
}

void qwen4_exp_sm80_out_projection(torch::stable::Tensor& output,
                                   torch::stable::Tensor const& input,
                                   torch::stable::Tensor const& weight) {
  STD_TORCH_CHECK(input.is_cuda(),
                  "qwen4_exp_sm80_out_projection: input must be CUDA");
  const int device_index = input.get_device_index();
  checkTensor(input, kTokens, kHiddenSize, device_index, "input");
  checkTensor(weight, kOutputSize, kHiddenSize, device_index, "weight");
  checkTensor(output, kTokens, kOutputSize, device_index, "output");

  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  auto* properties = get_device_prop();
  STD_TORCH_CHECK(properties->major == 8 && properties->minor == 0,
                  "qwen4_exp_sm80_out_projection requires SM80, got SM",
                  properties->major, properties->minor);

  auto const* input_ptr =
      reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr());
  auto const* weight_ptr =
      reinterpret_cast<__nv_bfloat16 const*>(weight.data_ptr());
  STD_TORCH_CHECK(
      reinterpret_cast<uintptr_t>(input_ptr) % alignof(uint4) == 0 &&
          reinterpret_cast<uintptr_t>(weight_ptr) % alignof(uint4) == 0,
      "qwen4_exp_sm80_out_projection inputs must be 16-byte aligned");

  constexpr int kBlocks = kOutputSize / kRowsPerBlock;
  qwen4ExpOutProjectionKernel<<<kBlocks, kThreads, 0,
                                get_current_cuda_stream(device_index)>>>(
      input_ptr, weight_ptr,
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()));
}

}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_C, qwen4_exp_out_projection_ops) {
  qwen4_exp_out_projection_ops.def(
      "qwen4_exp_sm80_out_projection(Tensor! output, Tensor input, "
      "Tensor weight) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, qwen4_exp_out_projection_ops) {
  qwen4_exp_out_projection_ops.impl("qwen4_exp_sm80_out_projection",
                                    TORCH_BOX(&qwen4_exp_sm80_out_projection));
}
