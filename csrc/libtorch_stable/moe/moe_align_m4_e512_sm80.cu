// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "core/registration.h"
#include "libtorch_stable/torch_utils.h"

#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>

namespace {

constexpr int kTokens = 4;
constexpr int kTopK = 10;
constexpr int kRoutes = kTokens * kTopK;
constexpr int kExperts = 512;
constexpr int kBlockSize = 8;
constexpr int kMaxSorted = kRoutes * kBlockSize;
constexpr int kThreads = kExperts;

__global__ __launch_bounds__(kThreads) void moeAlignM4E512Kernel(
    int32_t const* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ num_tokens_post_pad) {
  using BlockScan = cub::BlockScan<int32_t, kThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ int32_t counts[kExperts];
  __shared__ int32_t offsets[kExperts];
  __shared__ int32_t cursors[kExperts];

  const int tid = threadIdx.x;
  counts[tid] = 0;
  cursors[tid] = 0;
  if (tid < kRoutes) {
    expert_ids[tid] = -1;
  }
  if (tid < kMaxSorted) {
    sorted_token_ids[tid] = kRoutes;
  }
  __syncthreads();

  if (tid < kRoutes) {
    const int expert = topk_ids[tid];
    if (expert >= 0 && expert < kExperts) {
      atomicAdd(counts + expert, 1);
    }
  }
  __syncthreads();

  const int padded_count =
      ((counts[tid] + kBlockSize - 1) / kBlockSize) * kBlockSize;
  int offset;
  int total;
  BlockScan(scan_storage).ExclusiveSum(padded_count, offset, total);
  offsets[tid] = offset;
  if (tid == kThreads - 1) {
    *num_tokens_post_pad = total;
  }
  for (int block = 0; block < padded_count / kBlockSize; ++block) {
    expert_ids[offset / kBlockSize + block] = tid;
  }
  __syncthreads();

  if (tid < kRoutes) {
    const int expert = topk_ids[tid];
    if (expert >= 0 && expert < kExperts) {
      const int rank = atomicAdd(cursors + expert, 1);
      sorted_token_ids[offsets[expert] + rank] = tid;
    }
  }
}

void checkTensor(torch::stable::Tensor const& tensor, int64_t rows,
                 int64_t columns, int device_index, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  STD_TORCH_CHECK(tensor.get_device_index() == device_index, name,
                  " must be on the topk_ids device");
  STD_TORCH_CHECK(tensor.scalar_type() == torch::headeronly::ScalarType::Int,
                  name, " must be int32");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  STD_TORCH_CHECK(tensor.dim() == (columns == 0 ? 1 : 2) &&
                      tensor.size(0) == rows &&
                      (columns == 0 || tensor.size(1) == columns),
                  name, " has an unexpected shape");
}

void moe_align_block_size_m4_e512(torch::stable::Tensor const& topk_ids,
                                  torch::stable::Tensor& sorted_token_ids,
                                  torch::stable::Tensor& expert_ids,
                                  torch::stable::Tensor& num_tokens_post_pad) {
  STD_TORCH_CHECK(topk_ids.is_cuda(), "topk_ids must be CUDA");
  const int device_index = topk_ids.get_device_index();
  checkTensor(topk_ids, kTokens, kTopK, device_index, "topk_ids");
  checkTensor(sorted_token_ids, kMaxSorted, 0, device_index,
              "sorted_token_ids");
  checkTensor(expert_ids, kRoutes, 0, device_index, "expert_ids");
  checkTensor(num_tokens_post_pad, 1, 0, device_index, "num_tokens_post_pad");

  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  auto* properties = get_device_prop();
  STD_TORCH_CHECK(properties->major == 8 && properties->minor == 0,
                  "moe_align_block_size_m4_e512 requires SM80, got SM",
                  properties->major, properties->minor);

  moeAlignM4E512Kernel<<<1, kThreads, 0,
                         get_current_cuda_stream(device_index)>>>(
      reinterpret_cast<int32_t const*>(topk_ids.const_data_ptr()),
      reinterpret_cast<int32_t*>(sorted_token_ids.mutable_data_ptr()),
      reinterpret_cast<int32_t*>(expert_ids.mutable_data_ptr()),
      reinterpret_cast<int32_t*>(num_tokens_post_pad.mutable_data_ptr()));
}

}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_C, moe_align_m4_e512_ops) {
  moe_align_m4_e512_ops.def(
      "moe_align_block_size_m4_e512(Tensor topk_ids, "
      "Tensor! sorted_token_ids, Tensor! expert_ids, "
      "Tensor! num_tokens_post_pad) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, moe_align_m4_e512_ops) {
  moe_align_m4_e512_ops.impl("moe_align_block_size_m4_e512",
                             TORCH_BOX(&moe_align_block_size_m4_e512));
}
