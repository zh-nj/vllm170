# TileKernels 算子可用性复检（tilelang 0.1.15）

日期：2026-10-01。环境从 tilelang 0.1.12 升到 0.1.15 后的重新梳理。承接

- [vllm170-sm80-fp8-dense-tilekernels-feasibility-2026-10-01.md](vllm170-sm80-fp8-dense-tilekernels-feasibility-2026-10-01.md)（原始可行性研究）
- [vllm170-tilekernels-operator-survey-2026-10-01.md](vllm170-tilekernels-operator-survey-2026-10-01.md)（0.1.12 下的逐算子梳理与复核）
- `build(deps): upgrade tilelang to 0.1.15`（本次环境变更）

分工：**逐算子清单与逐行引用以 0.1.12 那份梳理为准**，本文只处理环境变更带来的差异、在 0.1.15 上重做的验证，以及 0.1.15 新增能力。核查对象仍是 `/mnt/data/apps/vllma100/TileKernels` HEAD `66258df`。

## 结论

1. **上游已经对齐**：`git ls-remote` 显示 upstream `main` 就是 `66258df6175d2f630ffecb04c5ab66bff8a2ae6a`，与本地检出同一提交。不存在"本地落后、上游已有 GEMM"的可能，本地即为最新。

2. **原文的核心结论在 0.1.15 上全部不变**：TileKernels 没有 dense/grouped GEMM 主循环；它是反量化、scale 布局、量化融合与数值参考的资源，不是 FP8 linear 的替代品。P0 的 `cast_back` 生成代码在 0.1.15 上结构与 0.1.12 一致，MXFP8 契约（uint8 行主序 scale、`<<23` 解码、按 32 组广播）逐项复现。

3. **import 阻碍的根因查清了**：那不是 TileKernels 的缺陷，而是它要求 0.1.13+ 的 API。TileLang v0.1.13 做了"multi-backend language-dialect refactor，用静态的 per-backend re-export 取代运行时激活的 language facade"——`tilelang.cuda.language` 正是这次重构的产物，0.1.12 早于它。升到 0.1.15 后 **33/33 模块原生导入，不再需要 shim**。

4. **0.1.15 新增了一个理想形状的算子，但 SM80 用不了**。`T.gemm_blockscaled` 的语义正是 `C (+)= (A*SFA) @ (B*SFB)`，即块量化 GEMM；但它只在 **SM100（TCGEN05）与 SM120（mma.sync）** 上有实现，SM80 上下沉时直接报错。因此"SM80 必须自己把反量化写进 GEMM tile"这一结论不但不变，还多了一条来自新 API 的反证。

5. **reducer API 解锁的算子在 SM80 上全部可用**：`norm_forward`、`norm_backward`、`swiglu_backward`、`moe.topk_gate`、`moe.moe_topk_gate_backward` 现在都能 lower，14 个候选 **14/14 lower、13/14 通过 nvcc sm_80**。唯一失败项仍是随机舍入，且 0.1.15 把它变成了一条明确的编译期断言。

## 一、0.1.15 兼容性变更对 TileKernels 的影响

v0.1.15 发布说明的兼容性条目与本库逐条对照：

| 0.1.15 变更 | TileKernels 用法 | 影响 |
| --- | --- | --- |
| 移除 `T.Pipelined` 的 `sync`/`group` 参数 | 0 处使用 | 无 |
| 移除 `T.gemm_sp` 的 `k_pack` | 0 处使用 | 无 |
| `T.symbolic` 降级为废弃别名，改用 `T.dynamic` | 0 处使用 `T.symbolic`；全部用 `T.dynamic` | 无 |
| 追踪期查询线程数的 kernel 必须在 `T.Kernel` 显式传 `threads=` | 5 个 mhc CUDA 文件存在不传 `threads=` 的调用（见下） | **已上机排除**（见下） |
| 256-bit 全局 load/store 限制在 SM100+ | SM80 目标本就不会生成 | 无（反而是保护） |
| kernel 缓存参数改存 JSON、二进制缓存目录改为不可变、哈希校验强制 | — | 升级后首次编译为冷缓存 |

**待定风险：mhc 组的 `threads=` 省略。** `mhc/head_compute_mix_cuda.py:27,62`、`mhc/sinkhorn_cuda.py:24,76`、`mhc/expand_cuda.py:15,40`、`mhc/norm_fn_cuda.py:21,44`、`mhc/pre_split_mixes_cuda.py:35,106` 都是 `T.Kernel(...)` 不带 `threads=`，依赖 TileLang 自行推断。0.1.15 只对"追踪期查询线程数"的 kernel 强制要求显式 `threads=`，因此是否真的破坏取决于这些 kernel 是否查询线程数。

**已上机验证：省略 `threads=` 在 0.1.15 上不构成破坏。** 离线探针当时判定不了，因为这些 kernel 用 `T.const()` 风格标注（`expand_cuda.py:6-9`），而探针要替换 `tilelang.jit` 才能跳过设备查询，`T.const()` 在非 eager builder 下直接抛 `JITNoBuilderError`。在有卡环境改用真实 `tilelang.jit` 后，两个独立的无 `threads=` kernel 均**编译并运行正确**（空闲 SM80，UUID `GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216`）：

| kernel | 覆盖 | 结果 |
| --- | --- | --- |
| `mhc/expand_cuda.py:15,40` `expand_to_mhc_fwd` | n=64/32/17/64 × h=5120/4096/7168/128 | 4/4 与 `torch/mhc.py:4` 参考**逐位相等** |
| `mhc/norm_fn_cuda.py:21` `mhc_fn_normw_merge_fwd` | 两个 `n_blk` 分支 + 不整齐尾部（m=64 n=8192/7000、m=8 n=256、m=1 n=513） | 4/4 **逐位相等** |
| `mhc/norm_fn_cuda.py:44` `mhc_fn_normw_merge_bwd`（用 `finalize_reducer`） | m=32/16/4 × n=4096/1024/700 | `fn_grad` **逐位相等**，`normw_grad` 差 ~1e-6（fp32 累加顺序） |

结论：0.1.15 的 `threads=` 限制只作用于"追踪期查询线程数"的 kernel，这两个不属于此列。**另外发现一个不是版本问题的契约陷阱**：`norm_fn_cuda.py:57` 是 `fn_grad[i_m, i_n] += ...`，即 `fn_grad` 是**累加目标**、调用方必须预先清零；用 NaN 填充去测会得到全 NaN 并误判为 kernel 缺写。`head_compute_mix`/`sinkhorn`/`pre_split_mixes` 三个文件本轮未逐一验证。

## 二、P0 契约在 0.1.15 上的复验

`cast_back` MXFP8（H=5120、sf=(1,32)、packed UE8M0）在 0.1.15 下重新生成的 SM80 代码，与 0.1.12 结构一致：

```cuda
extern "C" __global__ void __launch_bounds__(128, 1) cast_back_kernel_kernel(
    bfloat16_t* __restrict__ out, const fp8_e4_t* __restrict__ x,
    const uchar* __restrict__ x_sf, int num_tokens, int sf_stride) {
  // scale 解码：UE8M0 bit trick，一次入 shared 的 f32
  uint v_ = ((uint)x_sf[ i_1*128 + blockIdx.x*sf_stride + threadIdx.x ]) << (uint)23;
  ((float*)sf_shared)[ i_1*128 + threadIdx.x ] = (*(float *)(&(v_)));
  // 广播：每 4 个线程共享一个 32 元素组的 scale
  make_float4(((float*)sf_shared)[ i_2*32 + (threadIdx.x >> 2) ], ...);
  // 反量化：fp8 -> f32 -> 乘 scale -> bf16
  __tl_cvt_fp8x2_to_float2(..., __NV_E4M3);  ...  __float22bfloat162_rn(...);
}
```

三条结论在 0.1.15 上原样成立：

- kernel 侧 scale 仍是 **`const uchar*` + 动态字节 stride**，int32 打包只是 host 侧视图；
- vLLM 的 `weight_scale[N, K/32]` uint8 与其**布局同构**，适配仍是调用方一次 `.view(torch.int32)`（前提 `K % 128 == 0` 且连续）；
- scale 解码一次入 shared、按 32 组广播，这正是"解码 tile 在 shared/register 内消费"的参考实现。

唯一差异是 0.1.15 把 `x_shared_local_cast` / `sf_shared_local_cast_1` 声明移进了循环体内，指令序列未变。

### UE8M0 byte-0 边界：在 float32 下复核

之前的结论是"只有 byte 0 不一致"。本轮用 float32 逐字节复核（`numpy`），确认如此：

| E8M0 byte | TileKernels `(sf << 23)` 视作 float32 | vLLM `exp2(sf - 127)`（float32） |
| --- | --- | --- |
| **0** | **0.0** | **2^-127 ≈ 5.877e-39** |
| 1 | 2^-126 | 2^-126 |
| 126 / 127 / 254 | 0.5 / 1.0 / 2^127 | 相同 |
| 255 | inf | inf（float32 下 2^128 上溢） |

一个方法论提醒：用 Python `math.exp2`（双精度）比较时，byte 255 会假报不一致（双精度 2^128 = 3.4028e38，转 float32 才是 inf）。**在 float32 语义下只有 byte 0 是真差异**，其余 255 个字节完全一致。生产数据里若出现 E8M0==0，TileKernels 给 0 而 vLLM 参考给 2^-127，这是契约差异，会改变零权重块的语义。

## 三、0.1.15 新增：`T.gemm_blockscaled`（MXFP8 的理想形状，但 SM80 不可用）

0.1.15 引入统一块量化 GEMM，语义为 `C (+)= (A * SFA) @ (B * SFB)`，带 `k_start`、`sf_a_granularity_k`、`sf_b_granularity_k`、`sf_layout` 参数。docstring 与源码给出的后端支持：

- **SM100（Blackwell）**：TCGEN05 `kind::mxf8f6f4.block_scale`，A/B 为 shared 中的 FP8/FP6/FP4，**SFA/SFB 是已在 tensor memory 里的 E8M0 scale**，必须给 `mbar`，可选 `use_2cta`。
- **SM120**：`mma.sync.m16n8k64.kind::mxf4nvf4.block_scale`，**E2M1 操作数 + UE4M3 scale**，FP32 累加，A/B 与打包 scale word 在 shared；该路径忽略 `mbar`。
- **SM80**：无实现。

实测（sm_80 目标下沉）：

```text
InternalError: Block-scaled GEMM requires either Blackwell SM100 TCGEN5MMA
(A/B in shared memory, C in tensor memory) or SM120 mma.sync (A in shared
memory or a fragment, B in shared memory, C in a fragment), but got
target={"kind":"cuda",...,"arch":"sm_80"}, A scope=shared.dyn, B
scope=shared.dyn, C scope=local.fragment, M=64, N=64, K=64.
```

值得记下的设计取向：`src/op/gemm_blockscaled.cc:91-97` 明确选择**报错而不是退化成 dense GEMM**，理由是"a dense GEMM lowering would silently drop the SFA/SFB scale factors"。也就是说 0.1.15 不会给我们一条静默丢 scale 的错误路径。

**对目标的意义**：即使不谈 SM80，SM120 那条路是 MXF4/NVFP4+UE4M3，并非我们的 MXFP8-E4M3+E8M0；能覆盖 E8M0 的只有 SM100。所以 0.1.15 提供的这块新能力对本目标的 SM80 分支是**明确不可用**，但它划清了边界：将来若迁到 SM100，块量化 GEMM 是现成的；留在 SM80 就必须自写反量化进 tile。

同一版本里 FP8 MMA dispatcher 未变：FP8 仍指向 `cute::SM89_16x8x32_*`（`tl_templates/cuda/instruction/mma.h:237-252`），BF16/INT8 保留 SM80 路径（`:206`、`:210`）。新出现的 `mma_block_scale.h` 是 SM120 `kMxf4nvf4` 专用，与 SM80 无关。

## 四、可用性矩阵更新

### 4.1 状态变化（0.1.12 → 0.1.15）

| 算子 | 0.1.12 | 0.1.15 | 变化原因 |
| --- | --- | --- | --- |
| `cast_back`（P0，MXFP8 等 4 种 scale 配置） | lower ✅ / nvcc ✅ | lower ✅ / nvcc ✅ | 无变化；契约已复验 |
| `cast`（非随机舍入） | ✅ | ✅ | — |
| `cast`（`stochastic_rounding=True`） | lower ✅ / nvcc ❌ `.rs` | lower ✅ / nvcc ❌ **static_assert** | 报错更早更明确 |
| `per_token_cast` / `per_block_cast` / `per_channel_cast` | ✅ | ✅ | — |
| `swiglu_forward` | ✅ | ✅ | — |
| **`swiglu_backward`** | ❌ reducer_init | **✅** | reducer API |
| **`norm_forward`** | ❌ reducer_init | **✅** | reducer API |
| **`norm_backward`** | ❌ reducer_init | **✅** | reducer API |
| **`moe.topk_gate`** | ❌ reducer_init | **✅** | reducer API |
| **`moe.moe_topk_gate_backward`** | ❌ reducer_init | **✅** | reducer API |
| `moe.normalize_weight` / `moe.topk_gate_forward` | ✅ | ✅ | — |
| `batched_transpose_weight_sf` / `per_block_cast_lossless` | ✅ | ✅ | — |
| mhc 组（8 个 CUDA kernel） | ❌ reducer_init | 未验证（见 §1 的 `threads=` 风险） | reducer API 已解除，但签名构造未做 |

0.1.15 上的完整扫描（sm_80 目标）：**14/14 lower 通过，13/14 通过 `nvcc -arch=sm_80 -cubin`**。唯一失败是随机舍入：

```text
cuda_fp8.h:293: static_assert failed: "Stochastic rounding f32-to-FP8 requires sm_100a or sm_103a"
```

这比 0.1.12 的 ptxas `.rs not supported` 更好——编译期就拒绝，且说明了该特性在 0.1.15 里连 SM90 都不支持。**SM80 上随机舍入仍不可用**，`per_token_cast(..., stochastic_cast=True)` 启用即编译失败。

### 4.2 未变的分级判断

0.1.12 那份梳理的优先级表（P0 `cast_back` + `common.py`、P1 量化融合与 `testing/numeric.py`、P2 `batched_transpose_weight_sf`、旁支 engram/mhc/moe 路由）**逐行仍然有效**，本文不重复。唯一需要补充的是：P1 里的 `norm_forward` / `swiglu_backward` 从"0.1.12 不可用"变成了"0.1.15 可用且已通过 sm_80 编译"。

技术层面的可借鉴项也不变，其中两条仍然最重要：

- **`cast_back` 的"scale 解码入 shared → 每 4 线程广播一组 → fp8 转 f32 再乘"**：0.1.15 上复验通过，可直接作为 W8A16/MXFP8 GEMM 内部反量化的参考实现。
- **`moe.moe_topk_gate_forward_cuda.py:239-244` 的 XOR-butterfly + 字典序 tie-break**：确定性、仅用 shuffle，现随 `moe.topk_gate` 的解禁一同可用。

## 五、建议

1. **0.1.15 不改变 SM80 的技术路线**：`cast_back` 作反量化与数值参考、`common.py` 作 scale 布局契约、GEMM 主循环自写。新出现的 `T.gemm_blockscaled` 反而把"SM80 没有块量化 GEMM"这件事从"文档没写"变成了"编译器明确拒绝"。
2. **现在可以做的候选扩大了**：原本计划里排除的 `norm_forward` / `swiglu_backward` 现在能编到 sm_80，如果融合 RMSNorm+量化或 SwiGLU 反传进入范围，可以纳入评估。但注意它们的量化模式仍只接受 `(1,32)`（`norm_forward_cuda.py:25`）。
3. **仍不要在 SM80 上打开随机舍入**，并保持 `set_pdl(False)`、`disable_tma=True` 的调用习惯。
4. **升级后第一次编译是冷缓存**（0.1.15 改了缓存格式与目录语义），并且**已提交的 DSV41F 内核 codegen 与 0.1.12 不逐字节一致**（见 `build(deps)` 提交），其 8.05%/7.85% 收益需要有卡重测。
5. **若要复用 mhc 组任何 kernel，先确认 `threads=` 省略在 0.1.15 上是否成立**（§1）。

## 六、本轮证据边界

- 产出为**源码审计 + CPU 数值核对 + 无 GPU 的 SM80 离线 lower/nvcc 编译**，外加一次上游比对（`git ls-remote`，网络恢复后进行）。**没有执行任何 kernel、没有数值验收、没有性能数字**；本机无可用 GPU（`nvidia-smi` 无法与驱动通信）。
- 「能 lower」≠「能编译」≠「数值正确」≠「性能达标」。本轮把前两级做实，后三级仍未验证。
- mhc 组的 8 个 CUDA kernel 未做签名正确的构建测试；`T.const()` 风格标注使离线探针无法替代 `tilelang.jit`。
- codegen 摘录基于 H=5120 单一形状；其它形状会走 `cast_back_cuda.py:26-41` 的其它 TILE_K/TILE_M 分支。
- 0.1.15 的发布说明来自 GitHub Releases（v0.1.15，2026-09-30 发布）；兼容性条目按发布说明原文对照本地代码。
- 如后续复制实现，保留 TileKernels MIT 版权与许可声明。
