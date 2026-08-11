// SPDX-License-Identifier: Apache-2.0
// vLLM stable-ABI adapter for the SGLang-derived SM80 MXFP4-int8 kernels.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "libtorch_stable/torch_utils.h"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "mxfp4_moe_cutlass_core.cuh"

#include <algorithm>
#include <cstdint>

namespace {

using namespace mxfp4_int8::cutlass_core;

constexpr int kMxfp4Int8MoeReduceWarpsPerBlock = 8;
constexpr int kMxfp4Int8MoeReduceVec = 16;
constexpr int kDsv4HiddenSize = 4096;
constexpr int kDsv4IntermediateSize = 2048;
constexpr int kDsv4TopK = 6;

template <typename T>
constexpr T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

void check_cuda(cudaError_t status, const char* message) {
  STD_TORCH_CHECK(status == cudaSuccess, message, ": ",
                  cudaGetErrorString(status));
}

template <int BlockM, int BlockN, bool SourceRowsAreSlots>
struct Mxfp4Int8MoeKernelSelector;

#define SGL_MXFP4_INT8_MOE_SELECT_KERNEL(TM, TN)                                  \
  template <>                                                                     \
  struct Mxfp4Int8MoeKernelSelector<TM, TN, false> {                              \
    using Kernel = Mxfp4PackedBGroupedScaledBf16CutlassKernel_##TM##x##TN##x128##_W13; \
  };                                                                              \
  template <>                                                                     \
  struct Mxfp4Int8MoeKernelSelector<TM, TN, true> {                               \
    using Kernel = Mxfp4PackedBGroupedScaledBf16CutlassKernel_##TM##x##TN##x128##_W2;  \
  }

SGL_MXFP4_INT8_MOE_SELECT_KERNEL(16, 32);
SGL_MXFP4_INT8_MOE_SELECT_KERNEL(16, 64);
SGL_MXFP4_INT8_MOE_SELECT_KERNEL(16, 128);
SGL_MXFP4_INT8_MOE_SELECT_KERNEL(32, 32);
SGL_MXFP4_INT8_MOE_SELECT_KERNEL(32, 64);
SGL_MXFP4_INT8_MOE_SELECT_KERNEL(32, 128);
SGL_MXFP4_INT8_MOE_SELECT_KERNEL(64, 32);
SGL_MXFP4_INT8_MOE_SELECT_KERNEL(64, 64);
SGL_MXFP4_INT8_MOE_SELECT_KERNEL(64, 128);
SGL_MXFP4_INT8_MOE_SELECT_KERNEL(128, 32);
SGL_MXFP4_INT8_MOE_SELECT_KERNEL(128, 64);
SGL_MXFP4_INT8_MOE_SELECT_KERNEL(128, 128);

#undef SGL_MXFP4_INT8_MOE_SELECT_KERNEL

template <typename Kernel>
void set_max_dynamic_smem_if_needed() {
  constexpr int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if constexpr (smem_size > (48 << 10)) {
    cudaError_t status = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    STD_TORCH_CHECK(
        status == cudaSuccess,
        "failed to set MXFP4-int8 MoE CUTLASS dynamic shared memory attribute: ",
        cudaGetErrorString(status));
  }
}

union Mxfp4Int8MoeBf16Pack16 {
  uint4 v;
  __nv_bfloat16 u16[8];
};

template <int TopK>
__global__ void reduce_moe_slots_bf16_vec_kernel(
    const __nv_bfloat16* __restrict__ slots,
    const float* __restrict__ topk_weights,
    __nv_bfloat16* __restrict__ out,
    int M,
    int N,
    int total_valid_slots) {
  int warp_id = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int token = blockIdx.y * kMxfp4Int8MoeReduceWarpsPerBlock + warp_id;
  if (token >= M) {
    return;
  }

  int n_chunks = N / kMxfp4Int8MoeReduceVec;
  int slot_base = token * TopK;

  for (int chunk = blockIdx.x * 32 + lane; chunk < n_chunks; chunk += gridDim.x * 32) {
    int n = chunk * kMxfp4Int8MoeReduceVec;
    float acc[kMxfp4Int8MoeReduceVec];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kMxfp4Int8MoeReduceVec; ++i) {
      acc[i] = 0.0f;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < TopK; ++k) {
      int slot = slot_base + k;
      if (slot < total_valid_slots) {
        float weight = topk_weights[slot];
        CUTLASS_PRAGMA_UNROLL
        for (int pack_idx = 0; pack_idx < 2; ++pack_idx) {
          Mxfp4Int8MoeBf16Pack16 pack{
              *reinterpret_cast<const uint4*>(
                  slots + static_cast<int64_t>(slot) * N + n + pack_idx * 8)};
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < 8; ++i) {
            acc[pack_idx * 8 + i] += __bfloat162float(pack.u16[i]) * weight;
          }
        }
      }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int pack_idx = 0; pack_idx < 2; ++pack_idx) {
      Mxfp4Int8MoeBf16Pack16 out_pack;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < 8; ++i) {
        out_pack.u16[i] = __float2bfloat16_rn(acc[pack_idx * 8 + i]);
      }
      *reinterpret_cast<uint4*>(
          out + static_cast<int64_t>(token) * N + n + pack_idx * 8) = out_pack.v;
    }
  }
}

template <int HiddenSize, int IntermediateSize, int TopK, int BlockM, int BlockN, bool SourceRowsAreSlots>
struct Mxfp4Int8MoeGemm {
  using BaseKernel = typename Mxfp4Int8MoeKernelSelector<BlockM, BlockN, SourceRowsAreSlots>::Kernel;
  static constexpr int kN = SourceRowsAreSlots ? HiddenSize : (IntermediateSize * 2);
  static constexpr int kK = SourceRowsAreSlots ? IntermediateSize : HiddenSize;
  using Kernel = Mxfp4PackedBGroupedGemmKernel<
      typename BaseKernel::Mma,
      typename BaseKernel::Epilogue,
      SourceRowsAreSlots,
      kK,
      kN>;
  static constexpr int kBlockM = Kernel::Mma::Shape::kM;
  static constexpr int kBlockN = Kernel::Mma::Shape::kN;

  static void run(
      torch::stable::Tensor const& a_q,
      torch::stable::Tensor const& a_scale,
      torch::stable::Tensor const& b_mxfp4,
      torch::stable::Tensor const& b_shift2,
      torch::stable::Tensor const& b_channel_scale,
      torch::stable::Tensor& out,
      torch::stable::Tensor& routed_out,
      torch::stable::Tensor const& topk_weights,
      torch::stable::Tensor const& sorted_token_ids,
      torch::stable::Tensor const& expert_ids,
      torch::stable::Tensor const& num_tokens_post_padded,
      int64_t num_valid_tokens,
      int64_t multi_processor_count) {
    using ScalarType = torch::headeronly::ScalarType;
    STD_TORCH_CHECK(a_q.is_cuda() && a_scale.is_cuda() && b_mxfp4.is_cuda() &&
                        b_shift2.is_cuda() && b_channel_scale.is_cuda() &&
                        out.is_cuda() && routed_out.is_cuda() &&
                        topk_weights.is_cuda() && sorted_token_ids.is_cuda() &&
                        expert_ids.is_cuda() && num_tokens_post_padded.is_cuda(),
                    "MXFP4-int8 MoE tensors must be CUDA tensors");
    const int device_index = a_q.get_device_index();
    STD_TORCH_CHECK(
        a_scale.get_device_index() == device_index &&
            b_mxfp4.get_device_index() == device_index &&
            b_shift2.get_device_index() == device_index &&
            b_channel_scale.get_device_index() == device_index &&
            out.get_device_index() == device_index &&
            routed_out.get_device_index() == device_index &&
            topk_weights.get_device_index() == device_index &&
            sorted_token_ids.get_device_index() == device_index &&
            expert_ids.get_device_index() == device_index &&
            num_tokens_post_padded.get_device_index() == device_index,
        "MXFP4-int8 MoE tensors must be on the same CUDA device");
    STD_TORCH_CHECK(a_q.is_contiguous() && a_scale.is_contiguous() &&
                        b_mxfp4.is_contiguous() && b_shift2.is_contiguous() &&
                        b_channel_scale.is_contiguous() && out.is_contiguous() &&
                        routed_out.is_contiguous() && topk_weights.is_contiguous() &&
                        sorted_token_ids.is_contiguous() && expert_ids.is_contiguous() &&
                        num_tokens_post_padded.is_contiguous(),
                    "MXFP4-int8 MoE tensors must be contiguous");
    STD_TORCH_CHECK(a_q.scalar_type() == ScalarType::Char,
                    "a_q must be int8");
    STD_TORCH_CHECK(a_scale.scalar_type() == ScalarType::Float &&
                        b_channel_scale.scalar_type() == ScalarType::Float &&
                        topk_weights.scalar_type() == ScalarType::Float,
                    "activation, channel, and routing scales must be float32");
    STD_TORCH_CHECK(b_mxfp4.scalar_type() == ScalarType::Byte &&
                        b_shift2.scalar_type() == ScalarType::Byte,
                    "packed weight and shift tensors must be uint8");
    STD_TORCH_CHECK(sorted_token_ids.scalar_type() == ScalarType::Int &&
                        expert_ids.scalar_type() == ScalarType::Int &&
                        num_tokens_post_padded.scalar_type() == ScalarType::Int,
                    "routing index tensors must be int32");
    STD_TORCH_CHECK(out.scalar_type() == ScalarType::BFloat16 &&
                        routed_out.scalar_type() == ScalarType::BFloat16,
                    "outputs must be bfloat16");
    STD_TORCH_CHECK(a_q.dim() == 2 && a_q.size(1) == kK,
                    "a_q must have shape [M, ", kK, "]");
    STD_TORCH_CHECK(a_scale.dim() == 1 && a_scale.size(0) == a_q.size(0),
                    "a_scale must have one value per a_q row");
    STD_TORCH_CHECK(b_mxfp4.dim() == 4 && b_mxfp4.size(3) == 128,
                    "b_mxfp4 must have shape [E, K/32, ceil(N/8), 128]");
    STD_TORCH_CHECK(b_shift2.dim() == 4 && b_shift2.size(3) == 8,
                    "b_shift2 must have shape [E, ceil(Kblocks/4), ceil(N/8), 8]");
    STD_TORCH_CHECK(b_channel_scale.dim() == 2 &&
                        b_channel_scale.size(0) == b_mxfp4.size(0) &&
                        b_channel_scale.size(1) == kN,
                    "b_channel_scale shape mismatch");
    STD_TORCH_CHECK(topk_weights.dim() == 2 &&
                        topk_weights.size(1) == TopK,
                    "topk_weights must have shape [M, ", TopK, "]");
    STD_TORCH_CHECK(sorted_token_ids.dim() == 1 && expert_ids.dim() == 1 &&
                        num_tokens_post_padded.numel() == 1,
                    "invalid routing metadata shapes");

    const int64_t m_in = a_q.size(0);
    const int64_t num_experts = b_mxfp4.size(0);
    const int64_t k_blocks = b_mxfp4.size(1);
    const int64_t n_groups8 = b_mxfp4.size(2);
    const int64_t shift_stride = b_shift2.size(1);

    if constexpr (SourceRowsAreSlots) {
      STD_TORCH_CHECK(out.dim() == 2 && out.size(1) == kN &&
                          out.size(0) * TopK >= num_valid_tokens,
                      "W2 output rows do not cover routed slots");
      STD_TORCH_CHECK(m_in >= num_valid_tokens,
                      "W2 input rows do not cover routed slots");
    } else {
      STD_TORCH_CHECK(out.dim() == 2 && out.size(0) == num_valid_tokens &&
                          out.size(1) == kN,
                      "W13 output must have shape [valid_slots, ", kN, "]");
    }
    STD_TORCH_CHECK(routed_out.dim() == 2 &&
                        routed_out.size(0) >= num_valid_tokens &&
                        routed_out.size(1) == kN,
                    "routed_out does not cover routed slots");

    STD_TORCH_CHECK(kK % 128 == 0, "K must be divisible by 128");
    STD_TORCH_CHECK(k_blocks == kK / 32, "b_mxfp4 K blocks mismatch");
    STD_TORCH_CHECK(shift_stride >= (k_blocks + 3) / 4,
                    "b_shift2 stride mismatch");
    STD_TORCH_CHECK(n_groups8 == (kN + 7) / 8,
                    "packed N group mismatch");
    STD_TORCH_CHECK(num_valid_tokens >= 0,
                    "num_valid_tokens must be non-negative");
    STD_TORCH_CHECK(multi_processor_count > 0,
                    "multi_processor_count must be positive");
    STD_TORCH_CHECK(topk_weights.size(0) * TopK >= num_valid_tokens,
                    "topk_weights does not cover routed slots");

    // Match vLLM's moe_align_block_size capacity formula exactly. The
    // original SGLang integration used an extra sentinel expert; vLLM does
    // not, so carrying E+1 here would overrun the routing buffers.
    const int64_t num_align_experts = num_experts;
    const int64_t max_m = (num_valid_tokens < num_align_experts)
        ? num_valid_tokens * kBlockM
        : num_valid_tokens + num_align_experts * (kBlockM - 1);
    const int grid_m_tiles = static_cast<int>(
        ceil_div(max_m, static_cast<int64_t>(kBlockM)));
    const int grid_n_tiles = static_cast<int>(ceil_div(
        static_cast<int64_t>(kN), static_cast<int64_t>(kBlockN)));
    STD_TORCH_CHECK(
        sorted_token_ids.size(0) >= max_m,
        "sorted_token_ids capacity is smaller than maximum aligned token count");
    STD_TORCH_CHECK(
        expert_ids.size(0) >= grid_m_tiles,
        "expert_ids capacity is smaller than maximum M tile count");

    const cudaStream_t stream = get_current_cuda_stream(a_q.get_device_index());

    const bool persistent =
        kBlockM >= 128 &&
        kBlockN >= 64 &&
        grid_m_tiles * grid_n_tiles > multi_processor_count * 2;

    cutlass::gemm::GemmCoord problem_size(
        static_cast<int>(max_m),
        kN,
        kK);
    typename Kernel::Params params{
        problem_size,
        {static_cast<const int8_t*>(a_q.data_ptr()), kK},
        static_cast<const uint8_t*>(b_mxfp4.data_ptr()),
        static_cast<const uint8_t*>(b_shift2.data_ptr()),
        static_cast<int>(b_shift2.size(2)),
        b_mxfp4.size(1) * b_mxfp4.size(2) * b_mxfp4.size(3),
        b_shift2.size(1) * b_shift2.size(2) * b_shift2.size(3),
        static_cast<const int32_t*>(expert_ids.data_ptr()),
        static_cast<const int32_t*>(sorted_token_ids.data_ptr()),
        static_cast<const int32_t*>(num_tokens_post_padded.data_ptr()),
        static_cast<int>(num_valid_tokens),
        TopK,
        grid_m_tiles,
        grid_n_tiles,
        persistent,
        typename Kernel::Epilogue::OutputTileIterator::Params(cutlass::layout::RowMajor(kN)),
        {nullptr, cutlass::layout::RowMajor(kN)},
        typename Kernel::Epilogue::OutputTileIterator::Params(
            cutlass::layout::RowMajor(kN),
            static_cast<const float*>(a_scale.data_ptr()),
            static_cast<const float*>(b_channel_scale.data_ptr()),
            static_cast<const int32_t*>(sorted_token_ids.data_ptr()),
            static_cast<const int32_t*>(expert_ids.data_ptr()),
            static_cast<int>(num_valid_tokens),
            kBlockM,
            TopK,
            kN),
        {reinterpret_cast<__nv_bfloat16*>(routed_out.mutable_data_ptr()), cutlass::layout::RowMajor(kN)},
        {}};

    dim3 grid;
    if (persistent) {
      int blocks = multi_processor_count * 2;
      int total_tiles = grid_m_tiles * grid_n_tiles;
      grid = dim3(static_cast<unsigned>(std::min(blocks, total_tiles)), 1, 1);
    } else {
      grid = dim3(static_cast<unsigned>(grid_m_tiles), static_cast<unsigned>(grid_n_tiles), 1);
    }

    dim3 block(Kernel::kThreadCount, 1, 1);
    constexpr int smem_size = int(sizeof(typename Kernel::SharedStorage));
    set_max_dynamic_smem_if_needed<Kernel>();
    cutlass::Kernel<Kernel><<<grid, block, smem_size, stream>>>(params);
    check_cuda(cudaGetLastError(), "MXFP4-int8 MoE GEMM launch failed");
  }

  static void run_reduce(
      torch::stable::Tensor const& routed_out,
      torch::stable::Tensor const& topk_weights,
      torch::stable::Tensor& out,
      int64_t num_valid_tokens) {
    using ScalarType = torch::headeronly::ScalarType;
    STD_TORCH_CHECK(SourceRowsAreSlots,
                    "MXFP4-int8 MoE reduce is only valid for W2");
    STD_TORCH_CHECK(routed_out.is_cuda() && topk_weights.is_cuda() &&
                        out.is_cuda(),
                    "MXFP4-int8 MoE reduce tensors must be CUDA tensors");
    STD_TORCH_CHECK(
        routed_out.get_device_index() == topk_weights.get_device_index() &&
            routed_out.get_device_index() == out.get_device_index(),
        "MXFP4-int8 MoE reduce tensors must be on the same CUDA device");
    STD_TORCH_CHECK(routed_out.is_contiguous() && topk_weights.is_contiguous() &&
                        out.is_contiguous(),
                    "MXFP4-int8 MoE reduce tensors must be contiguous");
    STD_TORCH_CHECK(routed_out.scalar_type() == ScalarType::BFloat16 &&
                        out.scalar_type() == ScalarType::BFloat16 &&
                        topk_weights.scalar_type() == ScalarType::Float,
                    "reduce expects BF16 slots/output and FP32 routing weights");
    STD_TORCH_CHECK(routed_out.dim() == 2 && routed_out.size(1) == kN &&
                        routed_out.size(0) >= num_valid_tokens,
                    "routed_out does not cover valid routed slots");
    STD_TORCH_CHECK(topk_weights.dim() == 2 &&
                        topk_weights.size(1) == TopK &&
                        topk_weights.size(0) * TopK >= num_valid_tokens,
                    "topk_weights does not cover routed slots");
    STD_TORCH_CHECK(out.dim() == 2 && out.size(1) == kN &&
                        out.size(0) * TopK >= num_valid_tokens,
                    "out rows do not cover routed slots");
    STD_TORCH_CHECK(num_valid_tokens >= 0,
                    "num_valid_tokens must be non-negative");
    STD_TORCH_CHECK(kN % kMxfp4Int8MoeReduceVec == 0,
                    "W2 reduce requires N divisible by 16");

    const cudaStream_t stream =
        get_current_cuda_stream(routed_out.get_device_index());
    int m_out = static_cast<int>(out.size(0));
    if constexpr (SourceRowsAreSlots) {
      int n_chunks = kN / kMxfp4Int8MoeReduceVec;
      int grid_x = ceil_div(n_chunks, 32);
      grid_x = std::min(grid_x, 65535);
      int grid_y = ceil_div(m_out, kMxfp4Int8MoeReduceWarpsPerBlock);
      grid_y = std::min(grid_y, 65535);
      reduce_moe_slots_bf16_vec_kernel<TopK>
          <<<dim3(static_cast<unsigned>(grid_x),
                  static_cast<unsigned>(grid_y), 1),
             dim3(kMxfp4Int8MoeReduceWarpsPerBlock * 32, 1, 1), 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(routed_out.data_ptr()),
          static_cast<const float*>(topk_weights.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr()),
          m_out,
          kN,
          static_cast<int>(num_valid_tokens));
    }
    check_cuda(cudaGetLastError(), "MXFP4-int8 MoE reduce launch failed");
  }
};

}  // namespace

namespace {

template <int BlockM, int BlockN, bool SourceRowsAreSlots>
void launch_dsv4_mxfp4_int8_gemm(
    torch::stable::Tensor const& a_q,
    torch::stable::Tensor const& a_scale,
    torch::stable::Tensor const& b_mxfp4,
    torch::stable::Tensor const& b_shift2,
    torch::stable::Tensor const& b_channel_scale,
    torch::stable::Tensor& out,
    torch::stable::Tensor& routed_out,
    torch::stable::Tensor const& topk_weights,
    torch::stable::Tensor const& sorted_token_ids,
    torch::stable::Tensor const& expert_ids,
    torch::stable::Tensor const& num_tokens_post_padded,
    int64_t num_valid_tokens, int64_t multi_processor_count) {
  using Gemm = Mxfp4Int8MoeGemm<
      kDsv4HiddenSize, kDsv4IntermediateSize, kDsv4TopK, BlockM, BlockN,
      SourceRowsAreSlots>;
  Gemm::run(a_q, a_scale, b_mxfp4, b_shift2, b_channel_scale, out,
            routed_out, topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded, num_valid_tokens, multi_processor_count);
}

template <bool SourceRowsAreSlots>
void dispatch_dsv4_mxfp4_int8_gemm(
    torch::stable::Tensor const& a_q,
    torch::stable::Tensor const& a_scale,
    torch::stable::Tensor const& b_mxfp4,
    torch::stable::Tensor const& b_shift2,
    torch::stable::Tensor const& b_channel_scale,
    torch::stable::Tensor& out,
    torch::stable::Tensor& routed_out,
    torch::stable::Tensor const& topk_weights,
    torch::stable::Tensor const& sorted_token_ids,
    torch::stable::Tensor const& expert_ids,
    torch::stable::Tensor const& num_tokens_post_padded,
    int64_t num_valid_tokens, int64_t multi_processor_count,
    int64_t block_m, int64_t block_n) {
#define VLLM_DISPATCH_MXFP4_INT8_TILE(BM, BN)                              \
  if (block_m == BM && block_n == BN) {                                    \
    launch_dsv4_mxfp4_int8_gemm<BM, BN, SourceRowsAreSlots>(               \
        a_q, a_scale, b_mxfp4, b_shift2, b_channel_scale, out, routed_out, \
        topk_weights, sorted_token_ids, expert_ids,                        \
        num_tokens_post_padded, num_valid_tokens, multi_processor_count);  \
    return;                                                                \
  }
  VLLM_DISPATCH_MXFP4_INT8_TILE(16, 64)
  VLLM_DISPATCH_MXFP4_INT8_TILE(16, 128)
  VLLM_DISPATCH_MXFP4_INT8_TILE(64, 128)
  VLLM_DISPATCH_MXFP4_INT8_TILE(128, 128)
#undef VLLM_DISPATCH_MXFP4_INT8_TILE
  STD_TORCH_CHECK(false, "unsupported MXFP4-int8 SM80 tile ", block_m,
                  "x", block_n,
                  "; supported tiles are 16x64, 16x128, 64x128, 128x128");
}

void dsv4_mxfp4_int8_sm80_moe_gemm(
    torch::stable::Tensor const& a_q,
    torch::stable::Tensor const& a_scale,
    torch::stable::Tensor const& b_mxfp4,
    torch::stable::Tensor const& b_shift2,
    torch::stable::Tensor const& b_channel_scale,
    torch::stable::Tensor& out,
    torch::stable::Tensor& routed_out,
    torch::stable::Tensor const& topk_weights,
    torch::stable::Tensor const& sorted_token_ids,
    torch::stable::Tensor const& expert_ids,
    torch::stable::Tensor const& num_tokens_post_padded,
    int64_t num_valid_tokens, int64_t block_m, int64_t block_n,
    bool source_rows_are_slots) {
  const torch::stable::accelerator::DeviceGuard device_guard(
      a_q.get_device_index());
  auto* props = get_device_prop();
  STD_TORCH_CHECK(props->major == 8,
                  "mxfp4_int8_sm80 requires an SM80-family GPU, got SM",
                  props->major, props->minor);
  const int64_t multi_processor_count = props->multiProcessorCount;
  if (source_rows_are_slots) {
    dispatch_dsv4_mxfp4_int8_gemm<true>(
        a_q, a_scale, b_mxfp4, b_shift2, b_channel_scale, out, routed_out,
        topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
        num_valid_tokens, multi_processor_count, block_m, block_n);
  } else {
    dispatch_dsv4_mxfp4_int8_gemm<false>(
        a_q, a_scale, b_mxfp4, b_shift2, b_channel_scale, out, routed_out,
        topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
        num_valid_tokens, multi_processor_count, block_m, block_n);
  }
}

void dsv4_mxfp4_int8_sm80_moe_reduce(
    torch::stable::Tensor const& routed_out,
    torch::stable::Tensor const& topk_weights,
    torch::stable::Tensor& out,
    int64_t num_valid_tokens) {
  const torch::stable::accelerator::DeviceGuard device_guard(
      routed_out.get_device_index());
  auto* props = get_device_prop();
  STD_TORCH_CHECK(props->major == 8,
                  "mxfp4_int8_sm80 requires an SM80-family GPU, got SM",
                  props->major, props->minor);
  using Gemm = Mxfp4Int8MoeGemm<
      kDsv4HiddenSize, kDsv4IntermediateSize, kDsv4TopK, 16, 128, true>;
  Gemm::run_reduce(routed_out, topk_weights, out, num_valid_tokens);
}

}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_moe_C, m) {
  m.def(
      "dsv4_mxfp4_int8_sm80_moe_gemm(Tensor a_q, Tensor a_scale, "
      "Tensor b_mxfp4, Tensor b_shift2, Tensor b_channel_scale, Tensor! out, "
      "Tensor! routed_out, Tensor topk_weights, Tensor sorted_token_ids, "
      "Tensor expert_ids, Tensor num_tokens_post_padded, int num_valid_tokens, "
      "int block_m, int block_n, bool source_rows_are_slots) -> ()");
  m.def(
      "dsv4_mxfp4_int8_sm80_moe_reduce(Tensor routed_out, "
      "Tensor topk_weights, Tensor! out, int num_valid_tokens) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_moe_C, CUDA, m) {
  m.impl("dsv4_mxfp4_int8_sm80_moe_gemm",
         TORCH_BOX(&dsv4_mxfp4_int8_sm80_moe_gemm));
  m.impl("dsv4_mxfp4_int8_sm80_moe_reduce",
         TORCH_BOX(&dsv4_mxfp4_int8_sm80_moe_reduce));
}
