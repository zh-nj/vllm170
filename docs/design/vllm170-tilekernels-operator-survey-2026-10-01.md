# TileKernels 算子可用性梳理（SM80 FP8 dense linear 视角）

日期：2026-10-01。配套阅读：[vllm170-sm80-fp8-dense-tilekernels-feasibility-2026-10-01.md](vllm170-sm80-fp8-dense-tilekernels-feasibility-2026-10-01.md)。

本文是对上述可行性文档的**复核与补充**，不是重写。原文的核心结论成立；本文修正其中两处判断，补齐它未覆盖的文件，并首次给出 SM80 离线编译证据。

核查对象：`/mnt/data/apps/vllma100/TileKernels`，HEAD `66258df6175d2f630ffecb04c5ab66bff8a2ae6a`，工作树干净，本轮未改动其任何文件。vLLM 侧 HEAD `ccedc943fe`。

## 结论

1. **"没有 dense/grouped GEMM 主循环"成立，且比原文更强**：全库 20 633 行 Python 中 `T.gemm` / `T.mm` / `wgmma` / `tcgen05` / `ldmatrix` 命中数为 **0**。唯一的 `einsum` 出现在 `torch/` 参考实现里（`torch/mhc.py:62,78,109`、`torch/engram.py:205`），且是 `torch.einsum` 而非 `T.einsum`。TileKernels 不提供可替换 FP8 linear 的现成算子。

2. **"当前阻挡是 import"需要修正两处**：
   - import 阻碍**确实只有 2 行**，且是纯模块路径改名。加一行 `sys.modules` shim 后，本审计探测的 31 个模块在 tilelang 0.1.12 下**全部导入成功**，包括原文点名会失败的 `norm_forward_cuda`。
   - 但真正卡住 P1 算子的是**reducer API 换代**，不是 import：`T.reducer_init` / `T.reducer_update` 在 0.1.12 中不存在，且 `finalize_reducer` 签名不兼容。共 **13 个文件**依赖它。

3. **新增可比证据：SM80 离线 codegen + nvcc 编译双层验证**。用 `tilelang.engine.lower(func, target={"kind":"cuda","arch":"sm_80"})` 在 `Target(...)` 上下文内，无 GPU、无驱动即可下降出 CUDA 源码；再用本机 nvcc 13.3 `-arch=sm_80 -cubin` 验证能否真正编译。实测 18 个候选中 **16 个可 lower**，其中抽检的 8 个里 **7 个通过 sm_80 编译**。

   **唯一失败项是一条实打实的 SM80 硬限制**：`cast_cuda.py` 的随机舍入路径 `ptxas` 报 `Feature '.rs' not supported on .target 'sm_80'` —— 它依赖 PTX `cvt.rs`（`tl_templates/cuda/cuda_fp8.h:440`），该指令在 SM80 上不存在。

   其余通过项的生成代码中 `mma.sync` / `cp.async` / `cp.async.bulk` / `griddepcontrol` / `SM89` / `ldmatrix` 命中数均为 0 —— 这些量化算子是纯 LDG/STG + ALU，架构中立。

4. **关键新发现：MXFP8 反量化的 kernel 级契约是 uint8 行主序，不是 int32**。生成代码中 `x_sf` 声明为 `const uchar*` 配动态字节 stride。原文所述"CUDA packed scale 为 int32，与 vLLM MXFP8 uint8 输入不能直接互换"在 **API 签名层**成立，但在**内存布局层不成立**：两者同构，适配是调用方一次 `.view(torch.int32)`，不需要改 kernel。

## 一、核查方法（相对原文新增的手段）

原文只有一处导入检查（`norm_forward_cuda.py`）和若干临时 GPU 探针。本轮补充三项可复现手段：

**1. 全包 import 可达性探测。** 注入 `sys.modules['tilelang.cuda.language'] = tilelang.language` 后逐个导入 53 个模块，区分"自身代码失败"与"父包 `__init__` 连带失败"。

**2. 无 GPU 的 SM80 离线 codegen + 编译验证。** 这是本轮最有价值的手段，可直接回答原文遗留的"SM80 codegen 仍需逐项核实"：

```python
import torch, tilelang
from tvm.target import Target
import tilelang.engine as eng
torch.cuda.get_device_capability = lambda *a, **k: (8, 0)   # 无驱动时伪造
TGT = {"kind": "cuda", "arch": "sm_80"}
def fake_jit(*ja, **jk):                                     # 绕开 jit，取回 prim_func
    def deco(fn):
        def w(*a, **kw): return fn(*a, **kw)
        w.__wrapped__ = fn; return w
    return deco
tilelang.jit = fake_jit
tilelang.cuda.language = tilelang.language
# ...
with Target(TGT):
    src = eng.lower(prim_func, target=TGT).kernel_source    # 纯 CUDA C，不需要 nvcc/驱动
```

注意 target 必须用 JSON dict 形式；`"cuda -arch=sm_80"` 字符串形式在 0.1.12 已被移除。`get_best_vectorize_size`（`quant/common.py:13-16`）在 trace 期读 `torch.cuda.get_device_capability()`，无驱动时必须打桩。

拿到 CUDA 源码后再过一层真实编译（nvcc 不需要 GPU，只需驱动无关的 ptxas）：

```bash
nvcc -arch=sm_80 -std=c++17 -cubin -o k.cubin k.cu -w \
     -I$TLDIR/src -I$TLDIR/3rdparty/cutlass/include
```

**"能 lower" 与 "能编译" 是两个不同强度的事实，前者会漏掉 PTX 指令集的架构门禁** —— 本轮的随机舍入项正是被后者才抓出来的。

**3. 生成代码逐行核对。** 不依赖文档或注释，直接读 SM80 CUDA 源码确认契约。

## 二、两个真实阻碍

### 2.1 import：2 行，纯路径改名（影响面是整个包）

```python
# tile_kernels/quant/norm_forward_cuda.py:4
# tile_kernels/quant/norm_backward_cuda.py:4
from tilelang.cuda import language as T   # 0.1.12 中 tilelang.cuda 无 language
```

`tilelang.cuda` 在 0.1.12 的属性只有 `codegen / execution_backend / intrinsics / op / pipeline / target / transform`。

**影响面比原文描述的大**：失败链是
`tile_kernels/__init__.py:3` → `quant/__init__.py:12` → `norm_forward_kernel.py:8` → `norm_forward_cuda.py:4`。
因为任何子模块导入都会先执行父包 `__init__`，所以 `import tile_kernels` 整体失败，连带 `tile_kernels.testing.*`（校验/基准工具）与 `tile_kernels.torch.*`（PyTorch 参考实现）全部不可达。

**但这一层是表面的**：注入 shim 后 31/31 模块导入成功，说明 0.1.12 与 TileKernels 的 API 面在 import 层完全兼容。原文"不代表改一行 import 后所有算子就能使用"的告诫方向正确，但真正的原因不是 import。

### 2.2 reducer API 换代（0.1.12 的真正硬缺口）

| API | tilelang 0.1.12 | TileKernels 用法 |
| --- | --- | --- |
| `T.reducer_init` | **不存在** | 多处 |
| `T.reducer_update` | **不存在** | 多处 |
| `T.finalize_reducer` | `(reducer, batch=1)` | `(reducer, result, batch=N)` |
| `T.alloc_reducer` | 存在 | 兼容 |
| `T.sort` / `T.argsort` | **不存在** | 未使用 |

依赖新版 reducer API 的 **13 个文件**：

```text
quant/norm_forward_cuda.py (4)      quant/norm_backward_cuda.py (4)
quant/swiglu_backward_cuda.py (2)   moe/topk_gate_cuda.py (4)
moe/moe_topk_gate_backward_cuda.py (8)
mhc/{post,pre_apply_mix,pre_split_mixes,head_compute_mix,norm_fn}_cuda.py
mhc/{norm_fn,head_compute_mix,pre_split_mixes}_asc.py
```

实测后果：`norm_forward`、`moe.topk_gate` 在 trace 期直接 `AttributeError: module 'tilelang.language' has no attribute 'reducer_init'`。**这是必须升 tilelang 才能解决的部分，与 import 改名不是一回事。**

注意 `finalize_reducer` 的 arity 差异会**绑错参数**（0.1.12 把第二个位置参数当 `batch`），不是静默错值而是报错，所以风险是显式的。

### 2.3 PDL 在 SM80 是硬编译错误

`have_pdl(target)` 要求 `major >= 9`，否则后端 `ICHECK` 直接失败。TileKernels 默认 `use_pdl=False`（`config.py:7,110-112`），且 `if use_pdl:` 是 Python 层分支，所以默认路径安全。**在 SM80 上不得调用 `set_pdl(True)`。**

## 三、SM80 离线编译验证结果

TileLang 0.1.12 + sm_80 target。`lower` = 能下降出 CUDA 源码；`nvcc` = `-arch=sm_80 -cubin` 真实编译。

| 算子 | 配置 | lower | nvcc sm_80 | 说明 |
| --- | --- | --- | --- | --- |
| `cast_back` | MXFP8 packed-ue8m0 sf=(1,32) | ✅ 3589B | **✅** | P0 目标，直接可用 |
| `cast_back` | fp32 sf=(1,32) | ✅ 3526B | — | |
| `cast_back` | per-token fp32 sf=(1,H) | ✅ 3011B | — | |
| `cast_back` | 128×128 block | ✅ 4129B | — | |
| `cast` | bf16→fp8，非随机 | ✅ 2357B | **✅** | |
| `cast` | fp32→fp8，**随机舍入** | ✅ 2793B | **❌ `.rs` 不支持** | 见下方说明 |
| `per_token_cast` | bf16→mxfp8 (1,32) | ✅ 6416B | **✅** | |
| `per_token_cast` | bf16→fp8 (1,H) | ✅ 4869B | — | |
| `per_block_cast` | bf16→mxfp8 (32,32) | ✅ 10308B | **✅** | |
| `per_channel_cast` | fp8→fp8 (128,128) | ✅ 7421B | — | |
| `swiglu_forward` | H=5120 → mxfp8 | ✅ 8893B | **✅** | |
| `batched_transpose_weight_sf` | block 32 | ✅ 8282B | **✅** | |
| `per_block_cast_lossless` | fp4→fp8 | ✅ 6927B | **✅** | |
| `moe.normalize_weight` | k=8 | ✅ 2349B | — | |
| `moe.topk_gate_forward` | k=8, E=256 | ✅ 7203B | — | |
| `norm_forward` | H=5120 → mxfp8 | ❌ reducer_init | — | 需 tilelang ≥0.1.15 |
| `moe.topk_gate` | E=256, k=8 | ❌ reducer_init | — | 同上 |
| `norm_backward` / `swiglu_backward` | — | ❌（源码依赖 reducer_init） | — | 同上 |

**16/18 lower 通过；抽检 8 个中 7 个通过 sm_80 编译。** 所有通过（非随机舍入）项的生成代码中 `mma.sync`、`cp.async`、`cp.async.bulk`、`griddepcontrol`、`SM89`、`ldmatrix` 命中数均为 0 —— 与原文"FP8 MMA dispatcher 使用 `cute::SM89_*`"（`tilelang/src/tl_templates/cuda/instruction/mma.h:237-253`）并不矛盾：**这些算子根本不走 tensor core**，因此不受 SM89 dispatcher 限制。

### 唯一确认的 SM80 硬限制：随机舍入

`stochastic_rounding=True` 时生成 `__tl_cvt_f32x4_to_e4m3x4_rs_sat(...)`，其实现是内联 PTX（`cuda_fp8.h:440`）：

```cuda
asm("cvt.rs.satfinite.e4m3x4.f32 %0, {%1, %2, %3, %4}, %5;" ...);
```

ptxas 对 sm_80 报 `Feature '.rs' not supported on .target 'sm_80'`。**PTX 的随机舍入 `cvt.rs` 在 SM80 上不存在，TileLang 也没有软件回退。** 因此：

- `cast(..., stochastic_rounding=True)` 在 SM80 上不可用；
- `get_per_token_cast_kernel_cuda(..., stochastic_cast=True)`（`per_token_cast_cuda.py:42`）同理，**启用即编译失败**；
- 若 SM80 上确需随机舍入，必须自行用整数哈希 + 手动进位实现，不能复用这条路径。

**证据边界：本轮验证到"能下降出 SM80 CUDA 源码"并抽检"能用 nvcc 编成 sm_80 cubin"，没有执行、没有数值验收、没有性能数字。** 本机当前 `nvidia-smi` 无法与驱动通信，无 GPU 可用。

## 四、MXFP8 契约：生成代码实测

`cast_back` MXFP8（H=5120, sf=(1,32), packed ue8m0）下降出的 SM80 内核，关键片段：

```cuda
extern "C" __global__ void __launch_bounds__(128, 1) cast_back_kernel_kernel(
    bfloat16_t* __restrict__ out, const fp8_e4_t* __restrict__ x,
    const uchar* __restrict__ x_sf, int num_tokens, int sf_stride) {
  ...
  // 1) scale 解码：UE8M0 bit trick，只做一次，落在 shared 的 f32
  uint v_ = ((uint)x_sf[ i_1*128 + blockIdx.x*sf_stride + threadIdx.x ]) << (uint)23;
  ((float*)sf_shared)[ i_1*128 + threadIdx.x ] = (*(float *)(&(v_)));
  // 2) 广播：每 4 个线程共享同一个 group scale（32 元素 / 8 元素每线程）
  *(float4*)(sf_shared_local_cast_1 + vec_copy*4) =
      make_float4(f4(sf_shared[i_2*32 + (threadIdx.x >> 2)]));
  // 3) 反量化：fp8 -> f32 -> 乘 scale -> bf16
  ((float2*)(&__3))[0] = __tl_cvt_fp8x2_to_float2(..., __NV_E4M3);
  __2.x = (__3.x * v__2.x); ...
  (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(...);
}
```

三条可直接复用的结论：

**(a) kernel 消费的是 `uint8` 行主序 scale。** 形参就是 `const uchar* x_sf` + 动态 `int sf_stride`（字节单位）。"int32 打包"只存在于 host 侧（`common.py:111-113` 把 int32 view 成 uint8），kernel 从不做 4 字节打包语义。

**(b) 因此 vLLM MXFP8 与 TileKernels 在内存布局上同构。** vLLM 契约见 `vllm/model_executor/kernels/linear/mxfp8/Mxfp8LinearKernel.py:15-19`："FP8-E4M3 weights with **uint8 (E8M0) per-block scales at block size 32**"。

适配推导：`weight_scale[N, K/32]` uint8 连续张量 → 调用方 `.view(torch.int32)` →
shape `(N, K/128)`、stride `(K/128, 1)`；`stride(0) != 1` 故不触发 `use_tma_aligned_col_major_sf`（`common.py:106-108`）；再经 `common.py:113` 的 `view(torch.uint8)` 回到 `(N, K/32)`、stride `(K/32, 1)`，与 kernel 期望的字节 stride 完全一致。
**前提：K % 128 == 0 且张量连续。** 目标形状 5120 / 7168 / 4096 均满足。若 K/32 不是 4 的倍数，`get_sf_shape`（`common.py:193-194`）会按 4 对齐补 K，bytes 就不再与 vLLM 布局逐字节对齐，此时不能直接 view。

**(c) 原文的 UE8M0 边界结论复核为正确，但引用路径需更正。**
`common.py:306` 用 `T.reinterpret(T.uint32(sf) << 23, T.float32)`，即 sf≥1 时为 `2^(sf-127)`、**sf==0 时为 0**；vLLM 参考在 `vllm/model_executor/layers/quantization/utils/mxfp8_utils.py:237` 的 `dequant_mxfp8_to_bf16` 内用 `torch.exp2(scales - 127.0)`，sf==0 时为 `2^-127`。原文写 `common.py:304`（那是 `def` 行）与 `mxfp8_utils.py:230`（该文件在 `kernels/linear/mxfp8/` 下不存在；真实路径在 `layers/quantization/utils/`，`exp2` 在第 237 行）。**结论不变：只有 byte 0 不一致，其余 1/126/127/254/255 相同。**

**因此：把 MXFP8 权重 scale 直接喂给 `cast_back` 时，必须确认生产数据中 E8M0 byte 不为 0，或显式处理 sf==0 的语义差异。**

## 五、按可用性分级的算子清单（更新原文表格）

原文表格的行仍然有效，此处只列**需要修正或补充**的部分。

### 新增：原文遗漏但与本目标相关的文件

| 优先级 | 算子 / 文件 | 可借鉴内容 | 限制 |
| --- | --- | --- | --- |
| ~~P0~~ **不可用** | `quant/cast_cuda.py:12`（`cast_kernel.py:8` 派发） | 哈希 rbits 驱动的**随机舍入** `T.cast(..., round='rs', rbits=...)`（`cast_cuda.py:24-38,70-83`） | **SM80 硬限制**：走 PTX `cvt.rs`，ptxas 对 sm_80 报 `Feature '.rs' not supported`。非随机路径正常。若确需随机舍入须自行用整数哈希实现 |
| **P0** | `quant/per_token_cast_and_cast_back_cuda.py:15` | 量化→反量化往返探针，含 `T.reduce_absmax` 的**分组 amax**（`:68-69`）与 group-wise scale 施加（`:77-80`） | 供误差归因用，不是生产路径 |
| P1 | `tile_kernels/testing/numeric.py`（74 行） | 全套断言词汇：`assert_equal` 字节级比对（dtype/shape/stride/device，`:5-30`）、`calc_diff` fp64 归一化平方误差（`:33-38`）、`check_bias` 二项符号检验（`:41-64`） | 无 ULP 指标；纯 torch，**可直接抄** |
| P1 | `tile_kernels/testing/pytest/benchmark.py` + `testing/bench.py` + `testing/pytest/common.py` | 分片 JSONL 基线库、`-K` 过滤器、`ratio>1+threshold && delta>=0.8µs` 双条件回归门、geomean 报告 | 默认 `backend='cupti'` 只测**纯 kernel 时间，不含 launch 间隔**；需换 `backend='event'/'cudagraph'`；**仓库不附带任何基线数据** |
| P1 | `tile_kernels/torch/cast.py`（349 行） | MXFP8 数值 oracle：`cast`（`:122-318`）、`cast_back`（`:46-66`）、`transpose_weight_sf`（`:69-110`，产出 deep_gemm UE8M0 sf0/sf1 布局） | 依赖 `quant/common.py`（模块级 import tilelang）+ `config.is_ascend`；建议 vendored 到自有 `_ref.py` |
| P2 | `mhc/post_cuda.py`、`pre_apply_mix_cuda.py`、`pre_split_mixes_cuda.py`、`head_compute_mix_cuda.py` | 批量 reducer（`batch=N` 一次 AllReduce）、持久化 CTA（`T.Persistent`）、`T.annotate_layout`+`T.Fragment` 手排 128-bit 向量化 | 全部依赖 reducer_init → **0.1.12 不可用**；mHC 领域专属 |
| 旁支 | `engram/engram_gate_fwd_cuda.py:110-147` | **手写 cp.async 多缓冲流水线**（`T.async_copy` + `ptx_wait_group(N)` + 释放后 bank 复用预取）；双 RMSNorm + dot + 门控 + 残差融合，一次 launch | `TL_DISABLE_THREAD_STORAGE_SYNC: True` → **手工 `T.sync_threads()`，照抄易出静默竞态**；调优只覆盖 hidden 4096/7168 |
| 旁支 | `moe/moe_topk_gate_forward_cuda.py:239-250` | **XOR-butterfly top-k + 字典序 tie-break**：`other>cur \|\| (== && other_idx<idx)` 构成全序 ⇒ 确定性、仅用 shuffle、与 lane 映射无关。SM80 合法，实测可 lower | 该算子本身非 DSV41 路由契约（sqrtsoftplus + 强制 renorm），但 **butterfly 技巧可独立移植** |

### 明确非发现（原文未点明）

- 全库无 tensor core 代码：无 MMA、无 K/N 分块主循环、无 `ldmatrix`、无 TMA descriptor/mbarrier/cluster/DSMEM。
- 全库无确定性归约工具、无 CTA swizzle 助手、无通用 ragged/tail 助手；确定性只靠"固定顺序的两阶段 partial 归约"（`engram_grad_w_reduce_cuda.py:51-57`、`engram_sinkhorn_cuda.py:174-187`）逐案实现。
- `config.py:14-16` 的 `get_best_vectorize_size` 有显式 `major < 10 → 16 字节` 分支，说明库**感知架构**；README 的 "SM90/SM100" 是支持声明而非硬性门禁，但也意味着 **SM80 路径未经该库自身测试**（测试套件无任何 compute capability 门禁）。

## 六、对原可行性文档的具体修正

| 原文位置 | 原文表述 | 修正 |
| --- | --- | --- |
| L8-9 | "本地 TileKernels 没有 dense/grouped GEMM 主循环" | **成立**，且更强：全库 0 处 `T.gemm/T.mm/wgmma/tcgen05/ldmatrix` |
| L20-29 | import 阻碍（只点名 `norm_forward_cuda.py:4`） | 阻碍是 **2 行**；且因父包 `__init__` 连带失败，**整个包**（含 `testing/`、`torch/`）不可导入；shim 后 31/31 通过 |
| L29 | "不代表改一行 import 后所有算子就能使用" | 方向正确但归因不准：真正限制是 **reducer API 换代**（13 个文件），与 import 无关 |
| L38 | "与 vLLM MXFP8 uint8 输入不能直接互换" | API 签名层成立，**内存布局层不成立**：kernel 收 uint8 行主序，适配是调用方一次 `.view(torch.int32)`，前提 K%128==0 且连续 |
| L72 | `common.py:304` 的 bit trick | 实际在 `common.py:306`（304 是 `def` 行）；结论不变 |
| L73 | vLLM `mxfp8_utils.py:230` | 路径应为 `vllm/model_executor/layers/quantization/utils/mxfp8_utils.py`，`exp2(scales-127)` 在 **:237**（`dequant_mxfp8_to_bf16`）；边界结论不变 |
| L86 | 临时 `(N=4096,K=5120)` 探针"未完成…" | 本轮提供可复现的**离线 SM80 codegen** 通道，无需 GPU 即可做形状/配置覆盖面检查；但**执行与性能验收仍待有卡时进行** |
| L92 | "在现有 TileLang 0.1.12 中写独立 W8A16 候选，参考 cast_back 和 common" | **已被本轮证据支持**：`cast_back` 的 4 种 scale 配置在 0.1.12 + sm80 下均可下降，其中 MXFP8 配置已通过 nvcc sm_80 编译，可直接作为解量化/数值参考基线 |
| L37-48 表格 | 算子清单 | 遗漏 `cast_cuda.py`、`cast_kernel.py`、`per_token_cast_and_cast_back_cuda.py`、`norm_backward_cuda.py`、`swiglu_backward_cuda.py` 及整个 `mhc/`+`engram/` CUDA 集、`testing/`、`torch/`。其中 `cast_cuda.py` 的随机舍入路径**在 SM80 上不可编译**（PTX `cvt.rs`），是一条原文未记录的硬限制 |

## 七、建议

1. **P0 参考物已就绪，先取 `cast_back` + `common.py`。** MXFP8 配置已通过 sm_80 编译，另 4 种 scale 契约（E4M3-sf / fp32-sf / per-token / 2D-block）至少能下降。生成代码给出了"scale 解码入 shared → 每 4 线程广播一组 → fp8 转 f32 再乘"的完整模式 —— 这正是原文建议 3 "解码 tile 在 shared/register 内消费"的参考实现。
2. **把 tilelang 升级与 import 改名分开决策。** 若只需要 `cast_back`/`cast`/`per_*_cast`/`swiglu_forward`/`per_block_cast_lossless`/`batched_transpose_weight_sf`，**0.1.12 够用**（16/18 通过），只需改 2 行 import 路径。若还需要 `norm_forward`/`norm_backward`/`swiglu_backward`/`topk_gate` 或用 mhc，则必须升到 ≥0.1.15 —— 不要为一个 reducer API 升级生产环境，除非确有对应算子需求。
3. **SM80 上避开随机舍入路径。** `cast(..., stochastic_rounding=True)` 与 `per_token_cast(..., stochastic_cast=True)` 会编译失败（PTX `cvt.rs` 不存在于 SM80）。这条限制不影响 `cast_back`，但会影响任何"量化 activation 时用随机舍入提升精度"的方案。
4. **先验 byte-0 边界再决定是否复用 `transform_sf`。** 生产 MXFP8 权重 scale 中若出现 E8M0==0，`<<23` 得 0 而 vLLM 参考给 `2^-127`；这是**契约差异不是 bug**，但会改变零权重块的语义。
5. **数值验收用 vendored `torch/cast.py` 作 oracle**，并沿用它的双层断言纪律：量化数据对 torch 参考**逐字节**比对，容差比对只用于"解量化结果 vs 原始未量化张量"（`tests/quant/test_cast_back.py:121-127`，e4m3 阈值 1e-3）。量子化误差与 kernel 误差不能混为一谈。
6. **不要照抄 `engram/` 的手工同步内核。** 它们关闭了自动 barrier 与 OOB 诊断，正确性依赖编译器默认的 safe-memory legalization —— 跨版本升级会漂移。
7. **SM80 上禁用 PDL**（`set_pdl(True)` 会硬编译失败），并保留 `disable_tma=True` 的调用习惯。
8. **上机第一步建议做"能编译+能跑"的最小闭环**：把已验证可编译的 `cast_back` MXFP8 与 vendored `torch/cast.py` 参考对拍，用真实权重与非单位 scale 覆盖 E8M0 的 0/1/254/255 及 E4M3 的符号/零/subnormal。这一步不需要 GEMM，却能先把数据契约钉死。

## 八、本轮证据边界

- 产出为**源码审计 + 导入探测 + 无 GPU 的 SM80 离线 codegen + nvcc sm_80 编译抽检**。**没有**安装依赖、没有执行任何 kernel、没有数值验收、没有性能数字。
- 能 lower ≠ 能编译 ≠ 数值正确 ≠ 可执行 ≠ 性能达标。本轮已把前两级做实（随机舍入项正是被第二级否掉的），**后三级完全未验证**。SM80 上的实际可执行性与数值仍需上机确认。
- 未通过 nvcc 抽检的项（`cast_back` 的其余 3 种 scale 配置、`per_token_cast` per-token 变体、`per_channel_cast`）只有 lower 级证据，编译门禁未查。
- 生成代码解读基于 H=5120 的单一形状；其它形状的 tile 选择（`cast_back_cuda.py:26-41` 的 TILE_K/TILE_M 分支）会改变共享内存占用与向量化宽度。
- 调查期间本机 `nvidia-smi` 无法与驱动通信，无 GPU 可用于验收。
- 如后续复制实现，保留 TileKernels MIT 版权与许可声明。
