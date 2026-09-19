# DSV4.1-Flash @ 5×CMP170HX：无 DSpark decode 环节计量与热点定位（2026-09-18）

目标：**不开启 DSpark**，把单请求 decode 的每一环量出来，确定热点，优化到
**50 token/s**。

本轮结论先行：decode 的 92% 是五个 PP rank 的 model CUDA Graph replay；
把每个 rank 的 replay 按“每 token 必须搬运的权重字节”折算，整条 decode 只跑到
**约 40% 的可达 DRAM 带宽**。热点不是某个可以换 tile 的小 kernel，而是
**M=1 grouped GEMM 的执行效率**，其中最大单项是 routed MoE：
它搬运的字节和一次纯读取完全相同，却慢 **2.06 倍**。

---

## 1. 基线（本轮实测，无插桩）

服务：PP5/TP1、`mxfp4_int8_sm80`、`TRITON_MLA_SPARSE_DSV41`、FULL graph、
`max-num-batched-tokens=2048`、`max-num-seqs=8`、**`PORT_DSPARK_TOKENS=0`**
（启动器现在会完全省略 `--speculative-config`，不再保留 draft stage 的权重和 KV）。
工作负载：精确 32,000 token 输入（以 token id 数组发送，`usage.prompt_tokens`
按构造精确）、64 输出、贪心、无插桩流式计时。

| 轮次 | 配置 | ms/token | token/s |
| --- | --- | ---: | ---: |
| 3 次 | 无插桩 | 24.105 / 24.396 / **24.284**（均值） | 41.18 |
| 3 次 | profiler 待命但未开启 | 23.967 / 24.144 / 24.970（均值 24.360） | 41.06 |
| 3 次 | 收工前恢复生产默认后复测 | **24.493**（min 24.410 / max 24.552） | 40.83 |

比 2026-09-14 记录的 28.046 ms/token（35.66 token/s）快 13.4%：那是
`max_num_seqs=8` 构造期门控把单请求快路径关掉的回退被修掉之后的正常水平。
**到 50 token/s 还差 4.28 ms/token（−17.6%）。**

---

## 2. 计量方法：这台机器上没有内核级 profiler

必须先记录一条硬约束，它决定了后面所有证据的形式：

| 工具 | 结果 |
| --- | --- |
| `ncu` | `ERR_NVCMPGPU`（2026-09-14 已记录） |
| `nsys profile` | 能跑完，但报告 “does not contain CUDA kernel data”（GPU0/GPU1 都一样） |
| `torch.profiler`（kineto/CUPTI） | `CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED (42)`，CUDA activity 全部缺失 |

CMP 170HX 是矿卡固件，**CUPTI 直接拒绝为该设备采集**。因此本项目的证据只有三类：

1. **CUDA Event**（`vllm/models/deepseek_v4/phase_profile.py`）——图外边界与 eager 细分；
2. **真实 shape 的隔离 CUDA Graph 微基准**——图内 kernel 的绝对时间和带宽；
3. **无插桩端到端门禁**——唯一有资格判定收益的口径。

本轮给 profiler 补了两样东西，让“哪些时间没被任何 scope 覆盖”可算：
`cpu_start_ns`（每条记录的宿主起始时刻）与 `step.host_total`（每步的宿主步长）。
两者在图外边界上是可靠的；图内 scope 在 CUDA Graph capture 期间被
`_is_compiling_or_capturing()` 主动关闭，所以**图内不可能用 Event 细分**——
这不是工具没配好，是 capture 语义决定的，也是必须转向隔离微基准的原因。

---

## 3. decode 一层的账本（PP 串行）

五个 rank 对单个请求是严格串行的，stage 的关键路径份额是**各 rank 每步之和**。
括起来的父 scope 含子项，不重复相加。

| 边界 | PP 串行 ms/step | 生产占比 | 说明 |
| --- | ---: | ---: | --- |
| `target.model_graph_replay` | **22.482** | **92.3%** | P0 4.355 / P1 4.105 / P2 4.023 / P3 5.095 / P4 4.904 |
| `target.sampling_total` | 1.832 | 7.5% | 只发生在 PP4；`logits` 1.131 + `sample` 0.392 |
| `target.graph_input_staging` | 0.176 | 0.7% | 已不是热点（09-14 记为 4.95 ms 的口径与今天的代码不同） |
| `target.model_inputs` | 0.639（最大 rank） | — | 宿主侧组织输入 |
| `target.pp_recv_copy` / `wait` | 0.148 / 0.125（最大 rank） | — | 不是一阶项 |
| `target.attn_metadata` | 见下 | — | 父 scope 被事件密度放大，只有叶子可信 |

`target.attn_metadata` 的父 scope 在 P1–P4 上读到 15–17 ms/step，而同一 scope 的
宿主耗时只有 0.58 ms/step，叶子项（`swa.causal.decode_indices` 1.75、
`validity` 0.77、`ragged_indptr` 0.75、`ragged_pack` 0.70、`decode_tail_clear` 0.47、
`token_map` 0.42 ms/step）加起来也远超 24.36 ms 总预算里剩下的空间。
**这些是插桩自身的产物**：每步 `flush_phase_profile()` 会做一次全设备
synchronize，步内密集的事件记录之间会出现宿主间隙，父 scope 的 GPU 时间戳差
因此被拉长。该轮端到端被放大到 178 ms/token，只有 `model_graph_replay`（单次
graph launch，无嵌套）和 `sampling_total` 的绝对值可以采信。

排 P0–P2 与 P3/P4 的 replay 差 25%（4.02 对 5.10 ms），说明按层切分本身还有
余量，但单请求必须串行穿过五个 rank，**重分层不能减少这 22.48 ms 的总和**。

---

## 4. 每 token 到底要搬多少权重（来自 checkpoint，不是估算）

直接解析 48 个 safetensors header 得到张量字节数（全模型 497.19 GiB）。

| 组件 | 每层 | ×40 层/token | 备注 |
| --- | ---: | ---: | --- |
| routed experts | 6.72 GiB（384 个专家） | **4.32 GB** | 只读 top-6/384 = 1/64 |
| attention 投影 | 123.98 MB | **4.96 GB** | `wq_b` 41.94 + `wo_b` 41.94 + `wo_a` 33.55 + `wq_a` 6.55 |
| shared expert | 35.4 MB | 1.42 GB | 每 token 全读 |
| router + mHC | 7.8 MB | 0.31 GB | |
| LM head | — | 1.32 GB | 只在 PP4，每 token 一次 |
| **合计** | | **≈12.3 GB/token** | |

M=1 时每个 GEMM 都是 matrix-vector，**没有算术复用**，所以 decode 的物理下界
就是这张表的字节数除以可达带宽。

---

## 5. 可达带宽：先把分母量准

同一块卡（物理 GPU1，`CUDA_DEVICE_ORDER=PCI_BUS_ID`）：

| 测法 | 有效带宽 |
| --- | ---: |
| fp16 只读求和，512 MiB | **1261 GB/s** |
| `copy_`（读+写），512 MiB | 1281 GB/s |
| 空载重测（另一时段） | 1636 / 1583 GB/s |
| **expert slab 纯读取**（见 §6） | **1207 GB/s** |

取保守值 **1207 GB/s** 作为分母。⇒ 12.3 GB/token 的**下界是 10.2 ms/token
≈ 98 token/s**。当前 24.28 ms 意味着整条 decode 只用到 **40%**；50 token/s
（20 ms）只需要 **48.5%**。

---

## 6. 热点：routed MoE 的字节和纯读取完全一样，但慢 2.06 倍

新增两个探针（都走生产用的 `_moe_C` 入口，`E=384/topk=6` 的真实 layout，
`M=1`，6 个不同专家，装在单个 CUDA Graph 里取 P50）：

- `benchmarks/kernels/deepseek_v41_moe_bandwidth_probe.py`：跑真实 grouped GEMM；
- `benchmarks/kernels/deepseek_v41_moe_slab_read_control.py`：**读同一批地址**，
  每个专家一个连续 slab，不做 unpack、不做 MMA。

| 项目 | 字节 | 时间 | 有效带宽 | 相对 slab 读取 |
| --- | ---: | ---: | ---: | ---: |
| W13 GEMM（16×128，生产） | 72.0 MB | 126.9 us | 567 GB/s | 46% |
| W2 GEMM（16×128，生产） | 36.0 MB | 60.8 us | 592 GB/s | 49% |
| **W13+W2（16×128，生产）** | **108.1 MB** | **199.3 us** | **542 GB/s** | **45%** |
| W13+W2（16×64） | 108.1 MB | 175.0 us | 617 GB/s | 51% |
| W13+W2（64×128） | 108.1 MB | 391.5 us | 276 GB/s | 23% |
| W13+W2（128×128） | 108.1 MB | 473.8 us | 228 GB/s | 19% |
| **slab 纯读取（对照）** | 108.1 MB | **88.0 us** | **1207 GB/s** | 100% |

**结论一：DRAM 访问模式不是问题。** 同样的地址、同样的分散 expert slab，
纯读取能跑满 1207 GB/s。差距全部来自 kernel 自身的执行。

**结论二：不是 tile 能修的。** 换 tile 最好也只有 617 GB/s，仍然只有对照的 51%。

**结论三：6 slot 时是延迟受限。** 用不同 token 数把 slot 数从 6 拉到 48
（每个 slot 独占一个 block-M tile，字节数与 CTA 数同步增长）：

| slots | CTA 数 | W13 | W2 |
| ---: | ---: | ---: | ---: |
| 6（生产） | 216 | 557 GB/s | 590 GB/s |
| 12 | 432 | 687 GB/s | 720 GB/s |
| 24 | 864 | 767 GB/s | 755 GB/s |
| 48 | 1728 | 811 GB/s | 778 GB/s |

slot 数 8 倍只换回 1.46 倍带宽，并且饱和在约 810 GB/s（对照的 67%）——
说明除延迟外还有第二条与 unpack / smem 流水相关的上限。生产固定 6 slot，
所以第一条（延迟）才是当前损失的那一半。

生产 shape 下每个 rank 每层在 MoE 上花 187.7 us，40 层即 **7.51 ms/token**。
按 slab 读取的 88.0 us/layer 折算，**这里压着 3.99 ms/token**——几乎正好是
到 50 token/s 所需的 4.28 ms。

### 6.1 kernel 结构（供下一步改）

`csrc/libtorch_stable/moe/mxfp4_int8_sm80/`：生产 decode 走 `16x128x128_W13/W2`，
由 `Mxfp4PackedBMmaCoreForShape<16, 16, 128, 32, 128>` 生成，即
**WarpCount=(1,4)，每 CTA 只有 128 线程**；`kStages=3`；
权重经 `cp_async_zfill<16>` 进 smem，再由 `load_packed_b_warp_fragment_shared`
做 nibble→int8 展开后送 `mma.m16n8k32.s8`。上机测得 136 registers/thread，
`65536/(136×128)=3.76`，即每 SM 最多 3 个 CTA ⇒ **384 线程 / 2048 = 19% 占用率**，
而 6 slot 恰好只产生 216 个 CTA（3.09 个/SM），没有任何多余 CTA 可以用来掩盖延迟。
可动的方向是每 CTA 线程数 / 流水级数 / unpack 路径，而不是继续扫 tile。

---

## 7. 其余可归因项（全部为本轮实测）

下表把 §3 的 22.48 ms replay 用真实 shape 的隔离 CUDA Graph 拆开，全部是本轮
在同一批硬件上重测的数（括号内为字节/token 与有效带宽）。

| 组件 | 每层 us | ms/token | 字节/token | 有效带宽 | 占可达带宽 |
| --- | ---: | ---: | ---: | ---: | ---: |
| routed MoE W13+W2（16×128） | 187.7 | **7.51** | 4.32 GB | 575 GB/s | **48%** |
| `wo_a`（8 组 BF16 BMM） | 59.3 | 2.37 | 1.34 GB | 566 GB/s | 47% |
| main Q `wq_b`（MXFP8） | 49.5 | 1.98 | 1.68 GB | 847 GB/s | 70% |
| mHC（prenorm 16.4 + finalize 15.4 + post 14.3） | 46.1 | **1.84** | ≈0.04 GB | ≈50 GB/s | **≈4%** |
| `wo_b`（MXFP8 Marlin） | 40.1 | 1.60 | 1.68 GB | 1046 GB/s | 87% |
| shared expert（与 routed 重叠后的边际成本） | 13.3 | 0.53 | 1.42 GB | — | — |
| indexer decode（32K，只在 8 个 index-source 层） | 21.3 × 8 | 0.17 | — | — | — |
| inverse-RoPE + 物化 | 4.8 | 0.19 | — | — | — |
| LM head（PP4，`mxfp8_marlin_projection`） | — | 0.54 | 1.32 GB | **1237 GB/s** | **≈100%** |
| greedy argmax（融合进上一条） | — | 0.04 | — | — | — |

`sampling_total` 在 PP4 上是 1.832 ms/token，而 LM head + argmax 只占 0.58 ms：
**余下约 1.25 ms 是 sampling 侧的其它工作**，下一轮要拆。

### 7.1 已经确认没有余量的项

`wo_b`（87%）、main Q（70%）、LM head（≈100%）都贴近可达带宽；
`benchmarks/kernels/deepseek_v41_output_projection_sm80.py` 也确认把 `wo_a`
换成 8 路 MXFP8 Marlin 反而更慢（115.5 对 59.3 us）。
indexer 与 shared expert 的绝对值都 <0.6 ms/token。

### 7.2 未被解释的残差

上表 ×40 层的主观测量合计约 16.1 ms，加上 PP4 的 sampling 1.83 ms 得 17.9 ms，
距实际 24.28 ms 还有 **约 6.4 ms 没有独立归因**：sparse MLA attention/reduce
（09-14 测得 31.4 us × 38 层 ≈ 1.26 ms）、C0/SWA attention、KV insert、
norm/rotary/router/reduce 以及图内调度间隙。**这 6.4 ms 是本轮之后必须补测的
部分**——在它被拆开之前，不能声称账本已经闭合。

---

## 8. 热点排序与下一步

| 优先级 | 对象 | 现在的效率 | 可回收 |
| --- | --- | ---: | ---: |
| **P0** | routed MoE W13/W2（M=1，6 slot） | 45–51% 可达带宽 | 最多 3.99 ms/token |
| **P0** | `wo_a` 分组 BMM | 47% | 最多 1.26 ms/token |
| **P1** | mHC 三个小 kernel（N=24，延迟受限） | ≈4% 可达带宽 | 最多 1.5 ms/token |
| **P1** | sampling 侧非 LM-head 的 ≈1.25 ms | 未知 | 需先拆 |
| P1 | §7.2 尚未归因的 ≈6.4 ms | 未知 | 需先补测 |
| P2 | main Q（70%）、`wo_b`（87%）、LM head（100%） | 接近上限 | ≈0 |
| — | attn metadata / staging / PP recv | — | 已 <0.7 ms，不再是对象 |

**下一步（按此顺序）：**

1. 用隔离探针（不需要重启服务）改 MoE kernel 的执行参数：每 CTA 线程数、
   `kStages`、nibble 展开路径；判据是 W13+W2 从 542 GB/s 向 88 us（1207 GB/s）
   靠近，**不设 tile 扫描**（§6 已证明 tile 不是答案）。
2. 同法处理 `wo_a`：8 个分组小 BMM 合成一次启动、提高 CTA 数。
3. 拆 mHC 的三个 kernel（自己的 TileLang 代码，改动成本低于 CUTLASS）。
4. 补齐 §7.2 的 6.4 ms，让账本闭合。
5. 每个候选都必须回到无插桩 32K/64 门禁（≥3 repeats，2% 门槛）判定。

### 本轮已进入代码的门禁项与负结果

`VLLM_DEEPSEEK_V4_SM80_MOE_DECODE_BLOCK_N`（默认 128）：只对单 token、top-k 6
的 decode tile 生效，prefill 的 block-M 64/128 不受影响（第一版没有隔离，
prefill 选到 64×64 直接启动失败，已修）。隔离测得 16×64 把 W13+W2 从 199.3
降到 175.0 us（−12%），但**端到端没有兑现**：

| 配置 | repeats | ms/token 均值 | token/s |
| --- | ---: | ---: | ---: |
| 16×128（生产默认） | 6 | 24.322 | 41.12 |
| 16×64 | 10 | **23.924** | 41.80 |

只快 **1.64%**，低于本项目 2% 的立项门槛，而且 min/max（23.45–24.64）说明
单轮离散度就有 5%。**默认值保持 128**；多出来的 CTA 与 shared-expert 流共享
设备，抵消了孤立 kernel 的收益，这与 09-14 记录的 16×64 端到端回退同源。

---

## 9. 原始记录

- `/tmp/dsv41_dec_prof/baseline_target_only.json` — 无 DSpark 无插桩基线
- `/tmp/dsv41_dec_prof/baseline_armed_off.json` — profiler 待命基线
- `/tmp/dsv41_dec_prof/boundary_prof.json` — 插桩轮端到端（仅用于说明扰动）
- `/tmp/dsv41_dec_prof/rank*.jsonl` — 边界 Event 原始记录
- `/tmp/dsv41_dec_prof/boundary_ledger.json` — §3 账本
- `/tmp/dsv41_dec_prof/moe_slab_control.json` — §6 对照
- `/tmp/dsv41_dec_prof/moe_bw_probe.json`、`tile_*.json`、`moe_bw_m*.json` — §6 微基准
- `/tmp/dsv41_dec_prof/output_proj.json` — §7 `wo_a`/`wo_b`
- `/tmp/dsv41_dec_prof/q_proj.json` — §7 main Q
- `/tmp/dsv41_dec_prof/lm_head.json` — §7 LM head
- `/tmp/dsv41_dec_prof/deepseek_v41_shared_expert_overlap_sm80.json` — §7 shared expert
- `/tmp/dsv41_dec_prof/deepseek_v41_mhc_sm80.json` — §7 mHC
- `/tmp/dsv41_dec_prof/deepseek_v41_indexer_decode_sm80.json` — §7 indexer
- `/tmp/dsv41_dec_prof/moe_bn64_e2e.json`、`moe_bn64_e2e2.json` — 16×64 端到端负结果
- `benchmarks/deepseek_v4_1/decode_gate.py` — 端到端门禁
- `benchmarks/deepseek_v4_1/analyze_decode_profile.py` — 账本生成
- `benchmarks/deepseek_v4_1/stop_pp5.sh` — 不会误杀调用者自身的停机脚本
- `benchmarks/kernels/deepseek_v41_moe_bandwidth_probe.py` — MoE 带宽探针
- `benchmarks/kernels/deepseek_v41_moe_slab_read_control.py` — 同地址纯读取对照

---

## 10. 第二轮：MoE 执行效率的资源级归因（2026-09-18 续）

第一轮把 MoE 定为 P0（108.1 MB/layer 只用 542 GB/s，而同地址纯读取 1207 GB/s）。
本轮把差距拆开，并逐个验证"资源/异步折叠"类的候选。

### 10.1 差距的一半是访存模式，另一半是没被折叠的内循环

新增 `deepseek_v41_moe_pattern_control.py`：用**完全相同的地址、相同的 216 个
CTA 网格、相同的 2 KiB 分块 + 72 KiB 跨步**把 W13 的 packed 权重读一遍，
但不做 unpack、不做 smem、不做 MMA。

| 读法 | 时间/layer | 有效带宽 |
| --- | ---: | ---: |
| 平坦连续读（2880 程序/专家） | 57.3 us | 1235 GB/s |
| **GEMM 同款分块跨步读（216 程序）** | **73.4 us** | **964 GB/s** |
| **实际 GEMM（16×128）** | **126.9 us** | **567 GB/s** |

⇒ 访存模式只解释 28% 的差距（1235→964）。**剩下的 53.5 us/layer 全部是
`cp.async` → smem → nibble 展开 → MMA 这条链上没有被访存掩盖的执行时间**，
40 层合计 **2.14 ms/token**。这就是"耗时折叠"要吃掉的部分。

### 10.2 资源账（`cuobjdump -res-usage`）

`cutlass::Kernel` 不带 `__launch_bounds__`，所以占用率完全由寄存器分配器决定。
为 decode tile 加了一个带 `__launch_bounds__` 的启动器
（`Mxfp4PackedBGroupedKernel`，默认 `MXFP4_MOE_MIN_BLOCKS_PER_SM=4`）：

| tile | warp | REG | STACK | CTAs/SM（寄存器） |
| --- | --- | ---: | ---: | ---: |
| 16×64 | 16×16 | 116 | 0 | 4 |
| **16×128（decode 生产）** | **16×32** | **124** | **0** | **4** |
| 64×128 | 32×64 | 128 | 184 | 4 |
| 128×128 | 64×64 | 128 | 1312 | 4 |

改之前 16×128 是 136 registers ⇒ 3 CTAs/SM；现在 124 ⇒ 4 CTAs/SM 且 **无 spill**。
smem 每 CTA：`operand_A` 6.1 KiB + `operand_B_packed` 8 KiB×3 + `operand_B_shift`
0.4 KiB ≈ 31 KiB（级数 3），所以 163 KiB/31 KiB = 5，寄存器仍是绑定约束。

**隔离收益只有 3.8%**（W13+W2 199.3 → 191.7 us）。384 → 512 线程/SM 的占用率
提升并没有兑现成带宽，说明该 kernel 不是"线程不够"型延迟受限。

### 10.3 加深 cp.async 流水：反而更差

把 `kStages` 从写死的 3 改成 `MXFP4_MOE_PIPELINE_STAGES` 后实测：

| 级数 | smem/CTA | REG | W13+W2 | 相对 3 级 |
| ---: | ---: | ---: | ---: | ---: |
| **3（生产）** | ~31 KiB | 124 | **191.7 us** | 基线 |
| 4 | ~41 KiB | 124 | 205.6 us | +7.2% |
| 5 | ~52 KiB | 124 | 213.8 us | +11.5% |
| 6 | ~62 KiB | 124 | 229.8 us | +19.9% |

级数每 +1，smem 每 CTA 多 8.4 KiB，`163/smem` 从 5 掉到 3，**占用率损失大于
流水线加深的收益**。⇒ 想靠"再深几级"折叠时间这条路已经封死；要折叠必须
**降低内循环的指令数**（每搬运 1 字节要 ~3 条指令展开 nibble），而不是加缓冲。

### 10.4 下一步（MoE）

已经排除：tile 扫描（16×64/16×128/64×128/128×128）、占用率（3→4 CTA/SM）、
流水级数（3→6）、访存模式（同款跨步读能到 964 GB/s）。
剩下唯一没有被证否的方向是**换掉这条每字节 ~3 指令的展开链**：M=1 时
`mma.m16n8k32` 的 16 行里只有 1 行是真的，而权重仍要走
smem 分片 + `lookup_mxfp4_x2_byte_perm` 展开。候选是给单 token 走一条
SIMT + `__dp4a` 的 GEMV 路径（8 个 nibble 用 2 条 dp4a 吃掉，省掉 smem 中转和
fragment 布局转换）。判据仍是同地址读取的 73.4 us。

### 10.5 decode 到底跑几层？——40 层全部执行（直接层普查）

模型确实有"编码段 0–19 / 解码段 20–39"的结构（`ced.py`：层 20 是
`publish_decoder_kv` 的源头），但它是 **CED prefill** 机制：prefill 时编码段处理
全部 prompt token，解码段只重放一个 ≤128 token 的有界后缀。它**不改变 decode**，
而且在本部署里没有开启（启动器不传 `additional_config.ced_prefill`，
`ced_metadata is None` ⇒ `ced_step is None` ⇒ 走常规整层循环）。

为了让"每 token 读多少权重"这件事不依赖读代码，本轮用 `PORT_CUDAGRAPH_MODE=NONE`
把 decode 变成 eager（CUDA Graph 会吞掉 `layer.*` scope），跑一次真实 decode 并做
**逐层普查**（`benchmarks/deepseek_v4_1/census_decode_layers.py`）：

| PP rank | 该 rank 在 **decode** 中执行的层 |
| --- | --- |
| PP0 | 0,1,2,3,4,5,6,7 |
| PP1 | 8,9,10,11,12,13,14,15 |
| PP2 | 16,17,18,19,20,21,22,23 |
| PP3 | 24,25,26,27,28,29,30,31 |
| PP4 | 32,33,34,35,36,37,38,39 |

**一个 decode step 里 40 层全部出现**，编码段 20 层与解码段 20 层都在其中。
`target.model_forward` 的每 rank 值为 P0 64.92 / P1 63.21 / P2 65.91 /
P3 68.64 / P4 66.88 ms（eager，5 rank 串行 329.6 ms/step），五个 rank
量级一致 —— 如果编码段在 decode 被跳过，PP0/PP1 应该是 0。

⇒ §4 的每 token 12.3 GB 权重搬运（40 层）成立；§5 的 10.2 ms/token 下界、
以及"当前只用到约 40% 可达带宽"的结论不变。层内相对构成（eager 口径，
只可用于排序）：`attn.total` 36.8%、`moe.routed_experts` 20.6%、
`moe.shared_experts` 9.3%、`hc.*` 12.3%，其余为 metadata/norm/router/glue。

### 10.6 两个新的可调项及其门禁结果

两个宏都**默认等于生产值**，所以在默认配置下生成的代码与改动前一致：

| 宏 | 默认 | 作用 |
| --- | ---: | --- |
| `MXFP4_MOE_MIN_BLOCKS_PER_SM` | 0 | 0 = 保持 `cutlass::Kernel`（无 `__launch_bounds__`）；>0 启用带上界的启动器 |
| `MXFP4_MOE_PIPELINE_STAGES` | 3 | cp.async 流水级数 |

占用率候选的**无插桩端到端门禁**（同一 32K/64 prompt、同一 sha）：

| 配置 | repeats | ms/token 均值 | token/s |
| --- | ---: | ---: | ---: |
| 生产默认（无 `__launch_bounds__`，3 CTA/SM） | 9（历次汇总） | 24.379 | 41.02 |
| `MIN_BLOCKS=4`（124 reg，4 CTA/SM，无 spill） | 16 | **24.173** | 41.33 |

只有 **+0.85%**，低于 2% 门槛 ⇒ **默认回到 0**，生产的 launch 与改动前完全一致；
宏保留是因为重写内循环后第一个要重调的就是占用率。

> 顺带修掉一个真实缺陷：`set_max_dynamic_smem_if_needed` 原来只对
> `cutlass::Kernel<Kernel>` 调 `cudaFuncSetAttribute`。换成新启动器后，smem > 48 KiB
> 的 prefill tile（128×128 的 `operand_A` 是 48 KiB）直接
> `MXFP4-int8 MoE GEMM launch failed: invalid argument`，启动三连失败。现在两个
> 启动器符号各自 opt-in。

### 10.7 decode 能不能只跑 20 层？（"是否必须"）

结论：**以当前权重和当前服务形态，必须跑满 40 层**。四条独立证据：

1. **因果性**：解码段每一层的输入就是同一 token 的编码段输出。层 20 处的
   `publish_decoder_kv(attn, normalized, positions)` 把编码段 token 投到解码段的
   global main/index KV；token t+1 的产生依赖 token t 走完 0–19 层。跳过编码段
   等价于换一个模型。
2. **代码门控**：唯一能在层 20 短路整批的机制是 `ced_step`，而
   `CEDMetadata.prepare` 第一行就是 `if not batch.has_prefill: return None`
   （`ced.py:126`）。纯 decode 批次永远拿不到 `ced_step`，于是
   `model.py` 的循环照常跑 `start_layer..end_layer`。
3. **CED 的作用域**：CED 减掉的是"解码段 20 层 × **全部 prefill token**"，
   换成"解码段 20 层 × 每请求 ≤128 token"，并且本部署根本没开
   （启动器不传 `additional_config.ced_prefill`）。它对 decode 没有任何作用。
4. **量级自证**：如果 decode 能跳过编码段，五个 rank 的 replay 之和会从
   22.48 ms 掉到约 11 ms，端到端约 13 ms ⇒ 77 token/s。实测是 24.3 ms。

⇒ 每 token 12.3 GB 的权重搬运是**必须发生**的，§5 的 10.2 ms/token 下界成立；
优化空间只能来自"同样的字节搬得更快"，不能来自"少搬一半"。

> 顺带记录一个口径变化：本轮末复测生产默认配置为 24.729 ms/token（13 次汇总，
> 24.103–25.426），比本轮开头的 24.284–24.493 高约 1%。同机同 prompt
> （sha 不变）在数小时内出现这种漂移，所以所有 A/B 都必须用**同一时间窗内**
> 的对照，跨小时的绝对值不可比。

---

## 11. 第三轮：口径修正 + MoE 成为唯一可达杠杆（2026-09-18 续）

本轮先纠正第一版账本里的两个**实际错误**，再把 MoE 上剩下的参数空间逐个证否。

### 11.1 修正一：`wo_a` 早已跑满，不是热点

第一轮把 `wo_a` 记为 33.55 MB / 59.29 us = 566 GB/s（47%），据此列为"第二热点、
可回收 1.26 ms"。这是错的：33.55 MB 是 **checkpoint 里 FP8 权重的字节数**，
而运行期 `_get_cached_wo_a_bf16`（`rocm_aiter_mla_sparse.py:1302`）把它
**一次性反量化成 BF16 [8,1024,4096]**，每 token 实际搬运的是
**67.1 MB/layer**。

| 组件 | 运行期 byte/layer | 每层 us | 有效带宽 | 占 1207 GB/s |
| --- | ---: | ---: | ---: | ---: |
| `wo_a`（BF16 分组 einsum） | **67.1 MB** | 59.3 | **1132 GB/s** | **92%** |

⇒ `wo_a` 已经贴着上限，**从候选名单里删除**。

### 11.2 修正二：每 token 权重搬运是 13.7 GB，不是 12.3 GB

按运行期工作集重算（40 层 + LM head）：

| 组件 | 每层 MB | ×40 = GB/token | 每层 us | 有效带宽 | 占可达 |
| --- | ---: | ---: | ---: | ---: | ---: |
| routed MoE W13+W2（packed MXFP4） | 108.1 | **4.32** | 187.7 | **576 GB/s** | **48%** |
| `wo_a`（BF16） | 67.1 | 2.68 | 59.3 | 1132 GB/s | 92% |
| `wo_b`（MXFP8 Marlin） | 41.94 | 1.68 | 40.1 | 1046 GB/s | 85% |
| main Q `wq_b`（MXFP8） | 41.94 | 1.68 | 49.5 | 847 GB/s | 70% |
| shared expert（FP8） | 35.4 | 1.42 | 与 routed 重叠，边际 13.3 | — | — |
| `wq_a`（FP8） | 6.55 | 0.26 | 未单独测 | — | — |
| router + mHC | 7.8 | 0.31 | — | — | — |
| LM head（MXFP8） | — | 1.32 | 535 | 1237 GB/s | ~100% |
| **合计** | | **≈13.7 GB/token** | | | |

⇒ 下界 = 13.7 GB ÷ 1207 GB/s = **11.3 ms/token ≈ 88 token/s**；50 token/s 只需 57%
的可达带宽。当前 24.3 ms 是 **47%**。

### 11.3 决定性归因：只有 MoE 是异类

把 replay（22.48 ms）拆成"权重流"和"非权重计算"两类：

| 类别 | 字节 | 时间 | 有效带宽 |
| --- | ---: | ---: | ---: |
| routed MoE | 4.32 GB | 7.51 ms | **576 GB/s（48%）** |
| **其余全部权重流**（wo_a/wo_b/Q/shared/wq_a/mHC） | 8.03 GB | ≈8.7 ms | **≈930 GB/s（77%）** |
| 非权重计算（sparse MLA、indexer、norm、router、activation、reduce、KV insert、glue） | — | ≈4.8 ms | — |
| sampling（LM head + 其它，PP4） | 1.32 GB | 1.83 ms | — |

**除了 MoE，整条 decode 的权重搬运都在 77% 可达带宽上；MoE 是唯一掉到 48% 的
组件。** 而且按绝对值排序，MoE 的 3.5 ms 可回收量是其余所有单项（≤2.4 ms）
之和还多。⇒ **不修 MoE 内核，50 token/s 到不了**，这不是"再找几个小项"能补上的。

### 11.4 mHC 的真实成本（本轮实测）

一个 decoder layer 调 4 次：`hc.post_prev`、`hc.pre_attn`、`hc.post_attn`、
`hc.pre_ffn`，即 **2× `mhc_pre_delayed_tilelang` + 2× `mhc_post_tilelang`**。
用生产入口（不是内部子 kernel）在单个 CUDA Graph 里实测：

| 入口 | graph P50 |
| --- | ---: |
| `mhc_pre_delayed_tilelang` | 17.59 us |
| `mhc_post_tilelang` | 4.99 us |
| 每层（4 次调用） | 45.2 us |
| **每 token（40 层）** | **1.81 ms** |

mHC 每层只搬 3.93 MB（两个 FP32 的 `hc_*_fn`），即 **87 GB/s（7%）**——它是
N=24、K=20480 的极窄输出，6 个 CTA 打满不了 70 个 SM。但总量只有 1.81 ms，
且上一轮的 post+GEMM 融合候选端到端已回退（见 §7 历史记录）。

### 11.5 MoE 参数空间：本轮又关掉两个

| 候选 | 机制 | 结果 |
| --- | --- | --- |
| `ThreadblockK` 128 → 256 / 512 | 提高 `kWarpGemmIterations`，让更多 MMA 去掩盖每批 B 的 smem 载入+nibble 解码 | **编译失败**：CUTLASS `tensor_op_multiplicand_sm75.h:92` 要求 kCrosswise ≤ 一条 smem cache line，int8 下 K=256 直接违反 |
| 边界检查快路径 | 用 `run_mma<true>`（NoNBoundsCheck） | **本来就在用**：`(shape.n % kN) == 0`，W13 的 4608 与 W2 的 5120 都整除 128 |
| 更深 cp.async（3→6 级） | 加深流水 | 已证否（§10.3），smem 每级 +8.4 KiB 使占用率 5→3 |
| `__launch_bounds__` 4 CTA/SM | 提高占用率 | 已证否（§10.6），端到端 +0.85% |

于是 MoE 上剩下的唯一方向是**主循环内部的 load/decode/MMA 交织**：当前
`run_mma`（`mxfp4_cutlass_core.cuh:1008-1083`）先把整级 `kWarpGemmIterations`
个 B fragment 全部 smem-load + 解码完，再进 MMA 循环，中间没有跨级软件流水。
实测每级 3.17 us，而纯访存只占 1.84 us —— **1.33 us/级 × 40 级 = 53 us/layer
（2.1 ms/token）是"没被折叠"的那部分**。

### 11.6 为什么不能简单用 Triton GEMV 替代

写了 `deepseek_v41_moe_decode_control.py`：在与 GEMM 完全相同的地址与网格上读，
并加上 E2M1 nibble 解码（纯 `tl.where` 表）。

| 同地址同网格 | 时间 | 带宽 |
| --- | ---: | ---: |
| 只读 | 74.8 us | 946 GB/s |
| **读 + 表查解码** | **839.0 us** | **84 GB/s** |

朴素表查解码比纯读贵 **10 倍**，因此一个 Triton GEMV 会比现在的 CUTLASS 还慢。
CUTLASS 用 `lookup_mxfp4_x2_byte_perm`（PRMT 类字节置换）把解码压到约
15 条指令/4 字节，Triton 没有对应的 SIMD-in-register 原语。**MoE 的替代实现
必须写在 CUDA C++ 里，并沿用 PRMT 式解码。**

### 11.7 新增：packed 布局的权威映射（实测标定）

布局原先只隐含在 CUTLASS 代码里。本轮用 `deepseek_v41_moe_layout_calibrate.py`
逐 nibble 标定（把单个 nibble 放进 packed 张量、跑生产 op、读回是哪个输出列动了），
**256/256 个位置全部解出**（E=8、topk=6、block_m=16、16×128 tile）：

- W13 packed 张量形状 `[E, K/32, N/8, 128]` uint8；一个 128 字节 chunk 覆盖
  **8 个 n × 32 个 k**。
- `n_rel = byte // 16`，即 `n = 8*n_group + byte//16`。
- `r = byte % 16` 时，`k_rel = 2*f(r) + half`，其中 **`f(r)` 是把 r 的第 2、3 位
  互换**：`f(r) = (r & 3) | ((r & 4) << 1) | ((r & 8) >> 1)`。
  （实测：r=0..7 → k_base/2 = 0,1,8,9,2,3,10,11；r=8..15 → 4,5,12,13,6,7,14,15。）
- 解码值 = `±table[nib & 7] << shift`，`table = {0,1,2,3,4,6,8,12}`（E2M1），
  `shift = (shift2[e, kb//4, n_group, n_rel] >> (2*(kb%4))) & 3`。
- 累加是 **int32**（`mma.s8`），加法可交换 ⇒ **任何求和顺序都给出逐位相同的结果**，
  这一点让未来的替代实现不必担心数值漂移；真正的风险只在最后的
  `a_scale * b_channel_scale` 浮点乘法顺序。

（`deepseek_v41_moe_layout_check.py` 里我最初猜的映射是错的，标定把它纠正过来；
该脚本保留了对照。）

### 11.8 本轮结论

1. `wo_a` 从候选名单删除（92%，已满）。
2. 每 token 搬运口径改为 **13.7 GB**，下界 **11.3 ms ≈ 88 token/s**。
3. **MoE 是唯一 ≥3 ms 的杠杆**；其余全部在 70–100% 可达带宽上或总量 <2 ms。
4. MoE 的 2.1 ms 缺口是**主循环没有跨级软件流水**，不是 tile/占用率/流水级数/
   访存模式/边界检查的问题——这些本轮及上轮已逐个证否。
5. 替代实现必须走 CUDA C++ + PRMT 式解码；朴素 Triton 表查会慢 10 倍。

原始记录：`/tmp/dsv41_dec_prof/moe_layout_cal.json`、
`moe_decode_control.json`、`mhc_decode.json`、`moe_restored.json`。

### 11.9 操作教训：微基准不要与服务并发跑

本轮在服务运行期间把探针跑在物理 GPU1 上（PP0 所在卡），服务在 11:37:48 抛
`RuntimeError: Cannot start a receive while PP receive buffer is receiving;
expected free.`（`pp_utils.py:103`），五个 worker 全部退出、EngineCore 陷入
`No available shared memory broadcast block found in 60 seconds` 空转，
之后所有门禁请求挂死。重启后恢复（24.755 ms/token，4 次，与收工值 24.729 一致）。

**结论：CUDA Graph 微基准必须在服务停止时跑。** 本轮的布局标定（512 次连续小
GEMM）正好落在崩溃时间点上，是最可能的外部扰动源；PP 收发缓冲的严格时序守卫
在共享 GPU 被抢占时不宽容。

---

## 12. 第四轮：MoE 的差距被证明是硬件吞吐，mHC 拿到第一个过门槛的改动（2026-09-18 续）

### 12.1 MoE：去掉 nibble 展开，直接量出它的份额

给 `lookup_mxfp4_x2_byte_perm` 加了一个**只用于计时**的旁路（把展开换成原始字拷贝，
数值故意错），在同一探针下对比：

| 16×128、6 slot | W13 | W2 | both |
| --- | ---: | ---: | ---: |
| 生产（含 E2M1 展开） | 126.23 us | 60.70 | 199.24 |
| **去掉展开** | **84.36 us** | **47.49** | **136.70** |
| 同地址纯读取（无 kernel 计算） | 73.4 us | — | — |

⇒ **E2M1 展开独占 62.5 us/layer（199 us 的 31%，40 层合计 2.5 ms/token）**，
而剩下的访存+其它只有 136.7 us。

### 12.2 为什么这部分收不回来

试了两条明确针对"展开"的路，都无效：

| 候选 | 机制 | 结果 |
| --- | --- | --- |
| 把多个 K 步的 packed 字**先全部读进寄存器再统一展开**（`MXFP4_MOE_DECODE_BATCH`=2/4），拉长独立链窗口 | 展开链有 ~6 级依赖，怀疑是调度不够 | **199.14 / 199.17 vs 基线 199.24**，完全不变 |
| 提高 `ThreadblockK` 128→256/512，让更多 MMA 掩盖解码 | 每批 B 对应更多 MMA | 上轮已证否（CUTLASS crosswise 布局编译失败） |

BATCH 实验说明**不是调度问题**：编译器本来就把那些读排好了。既然重排无效，
剩下的解释是**吞吐**：`__byte_perm`（PRMT）在 GA100 上是四分之一速率的管道。
核算：W13 需要 70.8e6/4 × 4 = **70.8M 条 PRMT**，分摊到 70 个 SM 是 1.01M 条；
按 16 条/cycle 算是 63k cycle = 45 us —— 与实测的 +41.9 us 吻合。

**结论：在 MXFP4→INT8 模拟这条路上，SM80（没有原生 FP4 tensor core）的 MoE
已经贴着硬件下限。** 换 tile、换占用率、换流水级数、换 K 块、重排解码都不动它。
真要再快只能换数值格式，而"预展开成 INT8"已算过：字节翻倍后 216 MB/layer
即使按 1200 GB/s 也要 180 us，比现在的 199 us 只好 10%，还要多占 1.7 GB/rank 权重显存。

### 12.3 mHC：decode 路径的 split-K 被 `is_deep_gemm_supported()` 关掉了

`mhc_pre_delayed_tilelang` 里：

```python
use_deep_gemm = is_deep_gemm_supported()
n_splits = compute_mhc_pre_num_splits(input_size, num_tokens) if use_deep_gemm else 1
...
_tilelang_hc_prenorm_gemm(x, fn, mixes, sqrsum, input_size, 1)   # 最后一参 = n_splits
```

**本机 `is_deep_gemm_supported()` 返回 False**（CMP 170HX / SM80），于是走 else 分支，
`n_splits = 1`。prenorm 投影是 `[1, 20480] x [24, 20480]^T`——只有 24 个输出，
`tile_n=4` 意味着 **整个 20480 长度的归约只有 6 个 threadblock**，70 个 SM 闲着 64 个，
1.97 MB fp32 权重搬了 17.6 us（约 120 GB/s）。

核函数本来就支持 split-K（grid 是 `T.Kernel(num_tokens, n_tiles, n_splits)`），
融合 epilogue 也本来就对 split 维求和（`mixes[j] += gemm_out_mul[i_split, i, j]`），
只是那条 else 分支把它钉死成 1。本轮的改动：

1. 新增 `_tilelang_hc_prenorm_num_splits()`：decode（`num_tokens < 128`）且
   `input_size % 1024 == 0` 时，取 `input_size//1024` 的、不超过 10 的最大因子
   （小批量分支把 `n_thr` 钉在 1024，所以 split 必须整除 `input_size//1024`）。
2. `_tilelang_hc_prenorm_gemm` 的小批量分支去掉 `n_splits == 1` 的限制，并加上
   `(x.shape[1] // n_splits) % 1024 == 0` 的可分性条件。
3. `MHC_PRE_NORM_KERNEL` 的 `n_splits` 由 `tensors[0].shape[0]` 自动取得，无需传参。
4. `get_warmup_keys` 的 else 分支原来硬编码 `[1]`，改成走同一个 fold 函数，
   否则第一个 decode 请求会触发一次 JIT 编译尖峰。

**隔离实测**（生产入口，单个 CUDA Graph）：

| | GEMM 单独 | 融合 epilogue | `mhc_pre_delayed` 合计 |
| --- | ---: | ---: | ---: |
| 改动前（n_splits=1） | 7.71 us | 8.22 us | **17.59 us** |
| 改动后（n_splits=10） | **3.97 us** | 8.22 us | **13.67 us** |

split 扫描：1→7.71、2→5.79、4→4.38、5→4.22、10→**3.97**、20→4.48（20 反而回退）。
⇒ 每 token：`2 × (17.59 + 5.02)` = 45.2 us/layer → `2 × (13.67 + 5.02)` = 37.4 us/layer，
即 **1.81 → 1.50 ms/token（−0.31 ms）**。

顺带定位到改动后剩下的那半：**融合 epilogue 8.22 us**，
grid 是 `T.Kernel(num_tokens=1, threads=96)`，里面是 19 次串行 Sinkhorn，
每次 2 次 (4×4) 的块级归约 = 38 次跨线程归约，每次约 300 cycle ⇒ 约 8 us，与实测吻合。
把 `threads` 从 96 降到 64/32 只回收 0.67 us（8.22→7.55），**不值得改**，已放弃。

### 12.4 数值等价性（必查：split 改变了 fp32 累加顺序）

用 monkeypatch 把 `_tilelang_hc_prenorm_num_splits` 强制回 1，与 split=10 逐张量对比：

| 输出 | 逐位相同 | 最大绝对差 | 最大相对差 |
| --- | --- | ---: | ---: |
| `post_mix` | **是** | 0 | 0 |
| `res_mix` | 否 | 2.98e-08 | **9.65e-08** |
| `layer_input` | **是** | 0 | 0 |
| `next_pre_mix` | **是** | 0 | 0 |

最差 9.7e-08 相对误差，远在 bf16 分辨率之下。

### 12.5 端到端门禁：同窗口 A/B 过门槛，但跨窗口无法复现

| 配置 | repeats | ms/token 均值 | token/s |
| --- | ---: | ---: | ---: |
| 对照（`_TILELANG_SPLIT_TARGET = 1`，即改动前行为） | 14 | 24.592 | 40.66 |
| 候选（split=10）**紧邻同一时间窗** | 14 | **24.048** | 41.58 |
| 候选，另一个（更晚的）时间窗 | 32 | 25.267（sd 0.762） | 39.58 |

**同一时间窗 A/B = −2.21%（过 2% 门槛）；但换到更晚的窗口重测，整机慢了约 5%，
同一个候选反而读到 25.27。** 本机（含桌面/风扇控制等常驻负载）的小时级漂移
（±3–5%）大于本改动的效果，**端到端无法给出可信的效应量**。

因此保留该改动的依据是**确定性的隔离测量**（同一组件、同一图、17.59 → 13.67 us，
−22%）加上已验证的数值等价性——把它从 6 个 threadblock 提到 60 个、其余不变，
不存在变慢的机制。**但不应把 −2.21% 当作已确认的生产收益写进结论。**

### 12.6 本轮结论

1. **MoE 已经到底**：E2M1 展开占 31%（62.5 us/layer），是 PRMT 管道吞吐；
   重排、tile、占用率、流水级数、K 块全部证否。SM80 没有原生 FP4 tensor core，
   这是模拟代价。
2. **mHC 拿到本项目第一个能过门槛的候选**（隔离 −22%，同窗口端到端 −2.21%），
   根因是 `is_deep_gemm_supported()` 为 False 时 split-K 被钉死为 1。
3. 改动后 mHC 剩下的半壁是那个**单 CTA、38 次跨线程归约的融合 epilogue（8.22 us）**，
   降线程数只值 0.67 us。

原始记录：`/tmp/dsv41_dec_prof/mhc_split_probe.json`、`mhc_split10.json`、
`mhc_off.json`、`mhc_on.json`、`mhc_final*.json`、`moe_nodecode.json`、`moe_b2/b4.json`。

---

## 13. 第五轮：换卡到 2–6、确认 P2P 不可用、并给 split=32 一个否定的结论（2026-09-18 续）

### 13.1 服务迁移到物理 GPU 2–6

本机现在有 7 张卡：PCI0 = RTX 3080 Ti（桌面），**PCI1–6 = 6× CMP 170HX**。
`nvidia-smi topo` 显示 PCI2–6 两两 **PIX**（同一颗 PCIe switch）、同属 NUMA node 1，
而 PCI1 在 node 0。启动器默认从 `1,2,3,4,5` 改为 **`2,3,4,5,6`**，
把 PCI1 留给 burn-in/验证。

| 配置 | 32K/64 无插桩 |
| --- | ---: |
| PCI 1,2,3,4,5（跨 NUMA） | 24.28–24.49 ms/token |
| **PCI 2,3,4,5,6（同 switch/同 NUMA）** | **23.798 ms/token（10 次）** |

同一时间窗的边界账本（PP 串行）：`target.model_graph_replay` **22.011 ms**
（P0 3.883 / P1 4.052 / P2 4.287 / P3 4.982 / P4 4.807），`sampling_total` 1.719 ms，
两者相加 23.73 与端到端 23.80 闭合。

### 13.2 P2P：CMP 卡上彻底不可用（实测）

同 switch 让人以为可以开 P2P。直接测 `torch.cuda.can_device_access_peer(i, j)`：

```
dev0: - n n n n
dev1: n - n n n
dev2: n n - n n
dev3: n n n - n
dev4: n n n n -
```

**全部为 False**，即使五张卡在同一颗 PCIe switch 下。CMP 170HX 的固件关掉了 P2P，
所以 `NCCL_P2P_DISABLE=1` 是必须的，不是保守设置。这条排除掉了"换卡后可以用
GPU 直连降低 PP 交接"的想法。

### 13.3 尝试在 CUDA Graph 内部记事件（失败，已回退）

decode 的 22.0 ms replay 里还有约 4.8 ms 从未被独立归因。唯一的希望是
**CUDA event record 本身是可被 capture 的节点**：如果 capture 时把事件写进图，
每次 replay 都会被重新打时间戳，就能拿到图内账本。

实现了 `VLLM_DEEPSEEK_V4_PHASE_PROFILE_IN_GRAPH`（并在启动器加
`PORT_CUDAGRAPH_SIZES` 以便只捕获 `[1]` 这一张图），结果 **warmup 阶段
NCCL 直接失败**（`ncclUnhandledCudaError / Cuda failure 2 'out of memory'`），
三次启动全部失败。该开关已从代码中删除——留一个会让服务起不来的测量开关
比没有它更糟。图内账本仍未取得。

### 13.4 split=32：三点 A-B-A 判定为**无效果**，不采纳

线索来自第一轮：更早的 CMP 启动器（`run_cmp170hx_pp5.sh`）把
`VLLM_DEEPSEEK_V4_SM80_DECODE_NUM_SPLITS` 钉成 32，而 port 启动器根本没设它
（默认 0 = 走搜索上限 16 的 heuristic）。第一轮在旧代码上记过"isolated
31.821→26.760 us（−15.9%），端到端 +3.59%"。本轮把该旋钮暴露到 port 启动器
（默认仍为 0）并做门禁：

| 点 | 配置 | ms/token | 窗口 |
| --- | --- | ---: | --- |
| H1 | heuristic | 23.798 | 14:0x |
| H2 | heuristic | 24.273 | 14:5x |
| **S** | **split=32** | **23.762** | 15:2x |
| H3 | heuristic | 23.385 | 15:4x |

heuristic 三点均值 **23.819**（极差 0.887 ms = 3.7%），split=32 为 23.762
⇒ **−0.24%，无效果**。相邻窗口那一次读到的 −2.11% 纯粹是 H2 处的一个慢窗口。

**结论：不采纳，port 启动器默认保持 0。** 这条同时给出本项目的测量纪律：
**小时级漂移（3.7%）大于这类改动（±2%），单次 A/B 会给出错误结论；
必须用 A-B-A 三点或等价的交错设计。**

### 13.5 第五轮小结

- 服务已迁到 **PCI 2–6（同 PCIe switch、同 NUMA）**，基线 **23.798 ms/token（42.04 tok/s）**。
- P2P 在 CMP 卡上不可用（实测），PP 交接只能走主机。
- 图内事件法失败并已移除；4.8 ms 未归因部分仍未有账本。
- split=32 经三点 A-B-A 判定无效果，未采纳。
- 基础设施：`run_port_pp5.sh` 支持 `PORT_CUDAGRAPH_SIZES` 与
  `VLLM_DEEPSEEK_V4_SM80_DECODE_NUM_SPLITS`；`/logs/` 加入 `.gitignore`
  （原先测量记录放在 `/tmp`，本轮被系统清理过一次，已改存 `logs/decode_prof/`）。
