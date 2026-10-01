# 反量化进 tile：从 `T.gemm_blockscaled` 吸收什么（SM80）

日期：2026-10-01。承接 [vllm170-tilekernels-operator-resurvey-tilelang015-2026-10-01.md](vllm170-tilekernels-operator-resurvey-tilelang015-2026-10-01.md) §3 与
[vllm170-sm80-fp8-dense-tilekernels-feasibility-2026-10-01.md](vllm170-sm80-fp8-dense-tilekernels-feasibility-2026-10-01.md)。

前提（本文不再论证）：SM80 没有原生 FP8/FP4 tensor core，本项目的路线是**反量化进 tile**——在 shared/register 内消费量化权重，用 BF16/INT8 MMA 计算。因此 `T.gemm_blockscaled` 在 SM80 上不可调用这件事**不是重点**；重点是它的**计算流程与数据组织有没有可吸收的东西**。

结论先给：**计算流程抄不到，数据组织值得吸收。** 前者是因为 TileLang 里根本不存在软件反量化 GEMM 实现；后者是本轮升级对本项目最实在的产出。

## 一、为什么"计算流程"抄不到：三个后端全部委托硬件

TileLang 为块量化 GEMM 准备了三套实现，**没有一套在软件里做 scale 乘法**：

| 后端 | 指令 / 机制 | scale 在哪里被施加 | 证据 |
| --- | --- | --- | --- |
| SM100 | TCGEN05 `kind::mxf8f6f4.block_scale` | 硬件；SFA/SFB **必须已经在 tensor memory**，`mbar` 必需 | `cuda/language/gemm_op.py` gemm_blockscaled docstring |
| SM120 | `mma.sync.m16n8k64.kind::mxf4nvf4.block_scale` | 硬件；E2M1 操作数 + UE4M3 scale，打包 scale word 驻留 shared | 同上 |
| Ascend | `asc_mmad_mx` | 硬件 MX slot；docstring 原文："the scaling is implicit in the hardware MX slots, **so no SF pointer is passed here**" | `ascend/op/gemm/gemm_mad_blockscaled.py:74` |
| SM80 / SM90 | 无 | — | 下沉直接 `LOG(FATAL)` |

所以 SM80 上需要的那段"取 scale → 广播 → 乘进操作数"的循环，**TileLang 不提供任何参考实现**。这不是文档缺失，而是所有后端都把这一层交给了硬件。原可行性文档"SM80 的 GEMM 主循环、权重布局和调度仍需自己实现"的判断因此得到的是硬证据，而不是推测。

反过来说：**任何声称从 TileLang 借用块量化 GEMM 主循环的想法，在 SM80 上都没有可借之物。** 唯一现存的软件参考仍是我们自己的 `dsv41_sm80_tilelang.py`（INT8 fragment 解码 + `T.gemm`）与 TileKernels 的 `cast_back`（scale 解码入 shared + 按组广播）。

## 二、值得吸收的四条契约

`T.gemm_blockscaled` 的语义是架构中立的、且写得非常明确：

```text
C (+)= (A * SFA) @ (B * SFB)
```

`_gemm_blockscaled_impl` / `gemm_blockscaled` 的 docstring 原文：

> Scale factors apply to blocks along the reduction axis: `k_start` is the logical K-axis start of this tile and `sf_*_granularity_k` gives the number of K elements covered by one scale factor.

这四条契约对我们的 SM80 kernel 是直接可用的设计：

**1. granularity 是一等参数，不是硬编码。** `sf_a_granularity_k` / `sf_b_granularity_k` 显式传入，MXFP8=32、NVFP4=16 由调用方决定。我们的 W8A16/MXFP8 候选应当照抄这个形态，而不是把 32 写死在索引表达式里。

**2. `k_start` 给出 tile 的逻辑 K 起点。** 这让 K 方向流水线的每个 tile 不必自己重推 scale 索引——正是我们 `dsv41_sm80_tilelang.py:112-137` 目前用手算 `ko` 推导 scale 偏移所缺的那层抽象。把它提成显式参数，索引算术和断言都能收敛到一处。

**3. scale 是带 layout 的一等 tile buffer，驻留 shared，按 `(row, k_word)` 索引**，而不是逐元素去 global 取。这与 `cast_back` 复验出来的"解码一次入 shared、按 32 组广播"是同一个思路，区别只是 TileLang 把它形式化成了 buffer + layout + 断言。

**4. fail-loudly 是写进设计的。** 源码注释（`src/op/gemm_blockscaled.cc:97`）：

> a dense GEMM lowering would silently drop the SFA/SFB scale factors

我们自己的 kernel 也应当具备这个性质：断言 `BK % granularity == 0`、断言 scale buffer 形状等于 `ceil(K / granularity)`、断言 K 方向 tile 数与 scale 数一致。当前 `dsv41_sm80_tilelang.py` 的 Python 包装层没有 shape 断言（这是上一轮复核就点出的隐患），这套契约正好补上。

## 三、可直接抄的 scale 布局

CUDA dialect 接受两种 `sf_layout`（`cuda/language/gemm_op.py`：`"rowmajor"` 或 `"blockscaled_chunk_kmajor"`）。

### 3.1 `rowmajor`：最接近我们现状，建议作为 SM80 起点

```python
SFA_data[..., SFA_base_m + scale_m, SFA_base_k + scale_a_word_k]
```

即 scale tile 按 `[rows][k_word]` 朴素二维索引。逻辑上与我们 `cast_back` 里 `sf_shared[i, j]` 的用法一致，**SM80 直接可用**，建议作为默认。

### 3.2 `blockscaled_chunk_kmajor`：一个 bank/atom 友好的 K-major 变体

这是 SM120 "full-tile package contract" 唯一接受的布局（`mma_sm120_macro_generator.py:137-138` 强制）。核心公式（`_tile_kmajor_scale_word`，`:563-566`）：

```python
offset(row, kblock) = kblock * tile_rows + (row % 32) * (tile_rows // 32) + row // 32
```

`tile_rows = 128` 时展开为 `kblock*128 + (row % 32)*4 + (row // 32)`。读法：**4 路交错**——`row // 32` 选 sub-lane（0–3），`row % 32` 选 stride-4 的位置。也就是说 128 行的 scale 不是按行连续排，而是把行按 32 一组切四组、再交叉铺开。注释原文："K-major storage groups rows as `[row % 32][row // 32]` within each K atom."

这个公式的价值不在于我们要在 SM80 上原样使用（它的消费者是 SM120 宏生成器，SM80 上没有消费者），而在于它是**一个已推导好的、避免 bank conflict 的 scale tile 布局样例**。我们的 kernel 只要 scale 走 shared，就会碰到同一个问题，可以直接从这个结构出发做变体。

最终物理寻址在 SM120 上还做了一次四分拆：`SFA_data[base_m + word // 4, base_k + word % 4]`（`mma_sm120_macro_generator.py:627-635`，同样的表达式在 `:741`、`:748` 的 fragment 路径复用）。这一层是 SM120 的内存像，不必照搬。

### 3.3 打包粒度：一个 uint32 装 4 个 scale

```python
packed_word_k = sf_granularity_k * 4        # 一个 word 覆盖的 K 元素数
word_k = (k_start + micro_size_k * ki) // packed_word_k
```

因为一个 scale 覆盖 `granularity` 个 K 元素，而一个 word 覆盖 `granularity * 4`，所以**每个 uint32 装 4 个 scale**。SM120 上 `granularity=16`（NVFP4）→ `packed_word_k = 64 = micro_size_k`（`m16n8k64`），即**一个 word 恰好覆盖一个 MMA K-step**，这是硬件驱动的取整。

对我们的意义：这个"word 宽度对齐 MMA K-step"的规则**不能照搬**。SM80 用 BF16 MMA（`m16n8k16`）而 MXFP8 granularity=32，一个 scale 跨两个 MMA K-step，所以 SM80 的自然单位应当按 **`BK / granularity` 个 scale 每 tile** 来组织，而不是凑一个 word。这条差异必须显式记住，否则会照抄出一个与 hardware 对齐需求无关的打包方式。

### 3.4 另一份可参照的物理布局描述

Ascend 的 SF layout helper 把概念写得很干净（`layout/ascend.py:95-105`）：

> A logical `[..., MN, K]` scale tile maps to physical `[..., MN/16, K / (16 / dtype.bits), 16, 16 / dtype.bits]`. For uint8/int8 SF this is `[..., MN/16, K/2, 16, 2]`.

即 **逻辑 `[MN, K_blocks]` → 物理按 16 行分块、块内再按 K 维细分的四维张量**。这是我们设计 SM80 scale staging 时可以直接借用的形状语言（16 行分块的理由是 MMA 的 M/N 原子尺寸）。

## 四、对现有两个 kernel 的具体建议

### `dsv41_sm80_tilelang.py`（已提交的 SM80 候选）

- 把 `ko` 手算的 scale 偏移改成显式的 `(k_start, sf_b_granularity_k)` 契约，索引算术集中一处。
- scale staging 由隐式推导改为 `[rows][k_word]` 显式布局（§3.1）。
- 补 Python 包装层断言：shape/dtype/连续性与 `ceil(K/granularity)`，把目前"只有 apply 层门禁保证编译期常量一致"的隐患堵上。

### 未来的 SM80 W8A16 / MXFP8 候选

- 契约照抄：`sf_b_granularity_k = 32`、`k_start`、断言 `BK % 32 == 0`、scale buffer 形状 `ceil(K/32)`。
- 布局从 `rowmajor` 起步；若 profiling 显示 scale 访问是瓶颈，再按 §3.2 的结构做 bank-aware 变体。
- 反量化循环本身**没有可抄的参考**，仍需自写；`cast_back` 的生成代码（scale 解码入 shared → 每 4 线程广播一组 → fp8 转 f32 再乘）仍是目前最好的起点。

## 五、明确不要吸收的部分

- **`sf_layout="blockscaled_chunk_kmajor"` 在 SM80 上无消费者**：它只在 SM120 宏生成器里被解释。写上不会生效，只会误导。
- **`make_blockscaled_gemm_layout` 是 SM100 的 TMEM 布局**（docstring："Build the TMEM store layout for the C accumulator"），与 SM80 无关。
- **不要把 SM120 的 scale 语义当成 MXFP8 的等价物**：SM120 那条路是 **E2M1 操作数 + UE4M3 scale**（`kind::mxf4nvf4`），不是 E4M3 + E8M0。能覆盖 E8M0 的只有 SM100 的 `mxf8f6f4`。三者不可混用。
- **不要照搬 §3.3 的打包取整**（见上文差异说明）。

## 六、证据边界

- 结论来自对已安装 tilelang 0.1.15 的源码阅读（`language/gemm_op.py`、`cuda/language/gemm_op.py`、`tileop/gemm_blockscaled/`、`cuda/intrinsics/macro/mma_sm120_macro_generator.py`、`src/op/gemm_blockscaled.cc`、`src/cuda/op/gemm_blockscaled.cc`、`layout/ascend.py`、`ascend/op/gemm/gemm_mad_blockscaled.py`），加上一次 sm_80 目标下沉实测。
- **没有构造可运行的块量化 GEMM**：SM80 无实现，SM100/SM120 无对应硬件。§3.2 的布局公式是从宏生成器源码读出的，**未在硬件上验证其 bank conflict 表现**。
- **没有执行任何 kernel、没有性能数据**；本机无可用 GPU。
- 行号引用对应已安装的 tilelang 0.1.15。
