# DSV4.1-Flash @ 5×CMP170HX：FlashMLA 结论 + prefill/engram 环节计量（2026-09-18）

目标：① 集成 FlashMLA 注意力后端；② 保持 2048 chunk，梳理 prefill/engram 各环节耗时，
以 prefill 10000 tok/s 为目标优化。

基线（本轮实测，63,053 token 单请求，`max_num_batched_tokens=2048`，31 个 chunk）：
**11.7–12.0 s ≈ 5,300 tok/s**（profile 开启时 13.1–13.6 s，插桩本身约 +12%）。

---

## 1. FlashMLA：SM80 上无法集成（有证据）

| 检查 | 结果 |
| --- | --- |
| `csrc/kernels/sm80/` | **只有 README**，没有任何 kernel 源码；`flash_mla/cuda.*.so` 是 sm_80-only 构建（`cuobjdump` 只列 `sm_80.cubin`） |
| `import flash_mla` / `get_mla_metadata` | ✓ 可用（arch-generic helper 能在 Ampere 上跑） |
| `flash_mla_sparse_fwd`（V4.1 稀疏 prefill 内核） | ✗ `RuntimeError: [FlashMLA] sparse_prefill_fwd is not implemented for sm_80 in this build` |
| 上游 V4.1 kernel | SM90/SM100：wgmma、TMA、mbarrier、thread-block cluster，sm_80 上连编译都过不去 |

仓库自带的 `docs/20260915-sm80-dsv41-attention-analysis.md`（同一台机器上测的）结论一致：
**FlashMLA 在 sm_80 上"没有可移植的东西"**，并给了替代方案——Triton 的 read-once
prefill tiling（`bh=64/bk=64/st=2/nw=8`，1.51×）。

**该替代方案已经在本仓库生效**：`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`
里有完整的 sm_80 门控（`head_dim==512` + bf16 KV + `shared_memory_per_block_optin >= 139264`
+ 未导出 `VLLM_DEEPSEEK_V4_SPARSE_PREFILL_BLOCK_K`），`run_port_pp5.sh` 没有导出该变量
→ PP5 启动就是 read-once tiling。本轮未改这块（已是最优可得配置）。

**结论**：第一步能做的不是"接 FlashMLA kernel"（不存在），而是（a）确认等价优化在位 ✓、
（b）把注意力/engram 的耗时量出来，喂给第二步。

---

## 2. 计量工具（本轮新增）

`vllm/models/deepseek_v4/phase_profile.py`（CUDA event 级 phase profiler）本来就被
`model_runner.py` 导入并注册，所以只要开环境变量就能测到 MoE/attention/indexer 的既有 stage；
本轮补上了缺失的 engram 与层归属：

| 新增 stage | 位置 |
| --- | --- |
| `engram.hash` | `deepseek_v4_1/nvidia/model.py`（n-gram hash + 滚动槽查表） |
| `engram.prepare` / `engram.lookup` | `common/engram.py`（**宿主 94 GiB 表的 PCIe gather**） |
| `engram.embed` / `engram.post_wkv` | `common/engram.py`（wkv 投影 + 融合注入 kernel） |
| `layer.total` + `phase_profile_layer(prefix)` | `nvidia/model.py`（每层归属，`layer` 字段带层名） |

用法（control 文件做开关，避免全程插桩）：

```bash
VLLM_DEEPSEEK_V4_PHASE_PROFILE=1 \
VLLM_DEEPSEEK_V4_PHASE_PROFILE_PATH='/tmp/dsv41_prof/rank{pp_rank}.jsonl' \
VLLM_DEEPSEEK_V4_PHASE_PROFILE_CONTROL_PATH=/tmp/dsv41_prof/ON \
VLLM_DEEPSEEK_V4_PHASE_PROFILE_FLOWS=prefill,dspark \
PORT_MAX_BATCHED=2048 bash benchmarks/deepseek_v4_1/run_port_pp5.sh
# 测完 rm /tmp/dsv41_prof/ON
```

---

## 3. prefill 环节账本（63k 请求 / 31 chunk，每 rank 每 step 平均 ms）

| 环节 | ms/step | 占 step（377 ms） | 说明 |
| --- | ---: | ---: | --- |
| **MoE 路由专家** | **158** | **42%** | w13_gemm 102 + w2_gemm 53 + 其余 3 |
| 其余 layer 内（attention kernel、hc/norm、PP-KV 导出等） | ~83 | 22% | 未单独插桩，由 `layer.total − 已知子项` 得出 |
| 步间空隙（非 GPU） | 63 | 17% | 引擎/PP 交接/采样，见 §4 |
| indexer.logits | 18 | 5% | 只在 ratio=1 的层（rank3/4 高） |
| 共享专家 | 11.5 | 3% | 2048 token 下未与路由专家重叠 |
| target.attn_metadata | 10 | 3% | metadata 构建 |
| attention wo_a / inv_rope / ragged | 9 | 2% | |
| engram（rank0 有层 1 / rank1 有层 14） | 18 / 2.6 | — | 整个请求 572 ms / 82 ms |

engram 明细（整个 63k 请求，rank0 / rank1）：
`prepare+lookup` 567/77 ms，`embed` 余下；即**宿主表 gather 就是 engram 的全部成本**，
每 step 约 18 ms（rank0）——占该 rank step 的 5% 左右，不是瓶颈。

dspark 流在 prefill 期间可忽略：整请求合计 `target.sampling_total` 63 ms + `target.logits` 41 ms
+ `target.rejection` 15 ms。

---

## 4. 那 63 ms/step 的空隙是什么（已排除的假设）

- **不是 PP 收发**：`target.pp_recv_wait` + `pp_recv_copy` 合计 4 ms/请求。
- **不是 DSpark draft**：prefill 期间 dspark 流总量 ~0.13 s/请求。
- **是引擎/宿主每步开销**：31 step × 63 ms ≈ 1.95 s ≈ 总墙钟的 15–17%。
  step 周期（由 profiler 时间戳差算出）345 ms，其中 GPU forward 283 ms、metadata 10 ms。
- 由此得到一条硬结论：**在 2048 token/step 下，prefill 上限 = 2048 / (GPU 每步 + 固定每步开销)；
  减少每步固定开销（更少步数或更低宿主开销）与降低 GPU 每步耗时同等重要。**

---

## 5. MoE 为什么是瓶颈，以及还剩多少空间

MoE 每 step 每 rank 要读 **8 层 × 6.79 GB = 54 GB** 的 mxfp4 专家权重（2048 token × topk6 =
12,288 个专家命中摊到 384 个专家上，每个约 32 token → **每个专家权重每步都要读一遍**）。
所以 MoE 时间 ≈ 权重字节 / 有效带宽，与 token 数几乎无关：**这是"每 chunk 固定成本"**。

实测（`benchmarks/kernels/deepseek_v41_mxfp4_int8_sm80.py --tokens 2048`，单层）：

| block_m × block_n | w13 (ms) | w2 (ms) | 合计 | 有效带宽 |
| --- | ---: | ---: | ---: | ---: |
| **64 × 128（当前生产 tile）** | **11.67** | **6.19** | **17.85** | **380 GB/s** |
| 16 × 128 | 13.58 | 7.07 | 20.64 | 329 GB/s |
| 16 × 64 | 14.24 | 7.69 | 21.93 | 310 GB/s |
| 128 × 128 | 15.79 | 8.50 | 24.29 | 280 GB/s |

结论：**tile 已经是四个候选里最好的**；瓶颈是内核本身的带宽效率（380 / 1240 GB/s ≈ 31%）。
内核结构：`threadblock 256×128×64`、`warp 128×32×64`、`kStages = 3`、cp.async +
运行时 mxfp4 解包（含 `b_shift2` 修正）。要提速只能动内核（更多 stage / 更大 K tile /
更宽 load），这需要重编 `_moe_C`（`cmake-build-release/`，nvcc 13.3 在位）。

### 到 10000 tok/s 的预算

10000 tok/s @ 2048 chunk ⇒ **每 step ≤ 205 ms**（现在 ~377 ms）。

| 项 | 现在 | 需要 | 手段 |
| --- | ---: | ---: | --- |
| MoE | 158 | ~70–80 | 内核带宽 380 → 700+ GB/s（kStages/K-tile/load 宽度） |
| 其余 layer 内 | 83 | ~45 | attention kernel + hc/norm 插桩后再定位 |
| 步间空隙 | 63 | ~20 | 引擎/宿主每步开销 |
| metadata + 其他 | 21 | ~15 | |

即：**这是一个"MoE 内核带宽 ×2 + 每步固定开销减半"的组合目标**，单靠调参到不了。

---

## 6. 本轮已试且无效 / 待试

- ✗ `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=4096` +
  `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=4096`（想让 11.5 ms 共享专家与路由专家重叠）：
  12.0 s vs 基线 11.7 s，**无收益**（多流路径大概率未对当前 MoE 后端生效，或直接争抢带宽）。
- ✗ MoE tile 换档：当前档已最优（§5）。
- ✗ FlashMLA sm_80 kernel：不存在（§1）。
- 待做（下一轮，按收益排序）：
  1. MoE 内核带宽：`kStages` / K-tile / load 宽度扫描（改 `mxfp4_moe_cutlass_core.cuh` + 重编 `_moe_C`，
     用 §5 的 benchmark 快速迭代，不需要起服务）。
  2. attention/hc/norm 的剩余 ~83 ms/step：给 `sparse_mla.attention` 加 stage 后定位。
  3. 每步 63 ms 宿主开销：查 engine→5 worker 的每步 RPC/序列化路径。
  4. engram（rank0 18 ms/step）：宿主表 gather 用后台流与主计算重叠（`lookup(background=True)` 已存在但未启用）。

---

## 7. 第 2 轮：MoE 成本模型、注意力计量、10000 tok/s 的可行性算术

### 7.1 MoE 的成本模型（benchmark 实测，单层 w13+w2）

`benchmarks/kernels/deepseek_v41_mxfp4_int8_sm80.py`，tile 固定 (64,128)，只变 token 数：

| tokens | 合计 ms | 每 token µs |
| ---: | ---: | ---: |
| 2048 | 17.79 | 8.68 |
| 4096 | 25.64 | 6.26 |
| 8192 | 43.95 | 5.37 |

线性拟合 **T ≈ 9.07 ms（权重流）+ 4.26 µs/token**。固定项 = 6.79 GB 专家权重 / 9.07 ms
⇒ **749 GB/s**（卡片 copy 峰值的 60%），这部分已经接近该内核的实际带宽上限。
每 token 项 4.26 µs = 每 step 每 rank 多出 **70 ms**（8 层），是 MoE 里最大的一块可动项。

### 7.2 已试的两个 MoE 改动（都无效，已回滚）

| 改动 | 预期 | 实测 |
| --- | --- | --- |
| 把 grouped GEMM 的 grid 从 m-fastest 换成 n-fastest（`mxfp4_moe_cutlass_core.cuh` + entry 的 `dim3`），让 A tile 在 L2 里复用 | 省 ~9.8 GB/层 DRAM | **1.01×（噪声内）** ⇒ A 重读本来就被 L2 吃掉了，DRAM 流量没变 |
| tile 换档（16×64 / 16×128 / 128×128） | 提高带宽 | 均更慢（310 / 329 / 280 GB/s vs 380） |

⇒ 内核当前的瓶颈是 mma 路径效率（每 slot 35.4M MACs / 0.71 µs ≈ 100 TOPS ≈ int8 峰值的 32%），
以及权重流带宽 749 GB/s。想再快只能改内核主体（stage 数 / K-tile / mma 指令组合），不是调参。

### 7.3 注意力计量（本轮补的 stage）

`deepseek_v4_1/amd/rocm.py::_forward_prefill` 现在有 `sparse_mla.gather_compressed` /
`gather_swa` / `combine_indices` / `prefill` 四个 stage（此前散在未插桩的部分里）。

整请求（63k / 31 chunk）每 rank：

| 环节 | rank0 | rank3 | 每 step |
| --- | ---: | ---: | ---: |
| `sparse_mla.prefill`（注意力内核本身） | 772 | 890 | 25–29 ms |
| `sparse_mla.combine_indices` | 70 | 51 | ~2 ms |
| `sparse_mla.gather_compressed + gather_swa` | 23 | 37 | ~1 ms |
| `indexer.logits` | 140 | 556 | 4.5–18 ms |
| `target.attn_metadata` | 399 | 340 | 11–13 ms |

同时确认：生产 prefill 走的是 **已调优的 read-once 路径**（`ampere_sparse` →
`deepseek_v4_1/amd/rocm.py` 调 `rocm_sparse_attn_prefill` 时**不传** block_h/block_k，
所以 sm_80 门控生效、用 bh=64）。`triton_mla_sparse_sm80.py` 那个显式传 `block_h=32`
的调用点属于另一个未被选用的后端，不影响我们。

### 7.4 那 55 ms/step 的"未插桩层内时间"与 63–90 ms 的步间空隙

rank 内子项求和 6,724 ms vs `target.model_forward` 8,424 ms ⇒ **仍有 ~55 ms/step**
在 hc/norm/rope/PP-KV/量化等未插桩的层内工作里。

步间空隙（step 周期 − forward）**五个 rank 几乎相同**（85 / 82 / 91 / 64 / 74 ms），
说明它不是逐跳累积的 PP 交接，而是**每步一次全局同步 + 引擎/宿主开销**
（scheduler、SchedulerOutput 广播、采样、输出处理）。当前 step 周期 ~350 ms，
其中 forward 260–286 ms。

### 7.5 10000 tok/s 的可行性算术（2048 chunk）

10000 tok/s ⇒ **每 step ≤ 205 ms**（现在 ~377，含插桩；无插桩 2048 tok / 0.377 s = 5,430）。

| 项 | 现在 ms/step | 物理下限 | 说明 |
| --- | ---: | ---: | --- |
| MoE 权重流（54 GB/step/rank） | 158 | **~72** | 749 GB/s 实测带宽 × 54 GB |
| MoE 每 token 项 | （含在上面） | ~0 | 需内核效率提升 |
| 注意力 | 29 | ~20 | read-once 已在位 |
| indexer + metadata | 29 | ~15 | |
| 其余层内（hc/norm/PP-KV） | 55 | ~35 | 待插桩定位 |
| 步间空隙 | 63 | ~10 | 引擎每步同步 |

即：**即使 MoE 立刻达到它的实测带宽上限（72 ms）且空隙压到 10 ms，也只能到 ~200 ms/step
≈ 10,200 tok/s** —— 目标是"贴着实测物理上限"的，任何一项掉链子都到不了。
按现在的内核效率（158 ms）算，实际落在 **5,400–6,500 tok/s**。

⇒ 结论：**10k 的瓶颈在 MoE 内核效率与每步固定开销**，下一步优先级：
① MoE 内核 mma 路径（100→200 TOPS）；② 每步 63–90 ms 空隙的引擎侧归因；
③ 55 ms/step 未插桩层内工作（hc/norm）。

---

## 8. 第 3 轮：账本闭环、CPU/流水线归因、4096 chunk 反证

### 8.1 逐层账本现在完全闭环（rank3，63k 请求 / 31 chunk）

新增 `hc.*` 与 `attn.*` 子 stage 后：`layer.total` 8,924 ms = 具名子项 8,875 ms + 49 ms（0.5%）。

| 环节 | ms/step | 占 step | 内容 |
| --- | ---: | ---: | --- |
| `moe.routed_experts` | **157.7** | 42% | w13 101.6 + w2 52.7 |
| `attn.total` | **97.3** | 26% | 见下 |
| ├ `attn.prepare` | 72.3 | 19% | sparse_mla.prefill 28.8 + indexer.logits 17.9 + topk/combine/gather ~7 + **~18 未细分**（wq_b、qnorm/rope/kv-insert、SWA 注意力） |
| ├ `attn.o_proj` | 21.7 | 6% | inv_rope 2.2 + wo_a 6.3 + **wo_b ~13** |
| └ `attn.input_proj` + `split_norm` | 3.2 | 1% | q_a/kv_a/compressor（已在 aux stream 上） |
| `hc.pre_attn/post_attn/pre_ffn/post_prev` | 18.0 | 5% | 4 次 TileLang 混合系数 kernel |
| `moe.shared_experts` | 11.5 | 3% | |
| **layer 合计** | **286.4** | 76% | |
| `target.attn_metadata` | 10.9 | 3% | SWA causal + indexer metadata |
| 步间空隙 | ~60 | 16% | 见 8.2 |

### 8.2 那 60 ms/step 不是宿主 Python，是 PP 流水线气泡

- phase profiler 的 `cpu_elapsed_us` 显示 `attn.prepare` 的 CPU 时间是 GPU 的 2.74 倍，
  一度看起来像"宿主受限"。
- 但 torch trace（`--profiler-config '{"profiler":"torch",...}'`，CPU-only tier）给出否定证据：
  rank3 主线程在 3 个 chunk（2.2 s 墙钟）里**所有算子耗时合计只有 159–226 ms**，
  算子之间的空档合计 **2,084 ms（94% 时间在等）**，最大的几个空档全部是
  `c10d::recv_` → `aten::item`（785 / 277 / 54 / 53 … ms）。
  ⇒ 真实情况是**worker 在等上游 rank 的数据**（PP 链），不是 Python 开销。
  那些 `.item()`（3 chunk 内 288 次 `_local_scalar_dense`，即 288 次 GPU→CPU 同步）
  总计只有 ~30 ms，是坏味道但不是瓶颈。
- 结论：每步 ~60 ms 是 **PP 交接 + 引擎每步同步** 的气泡，削弱它需要引擎级流水（多 microbatch 在飞），
  而 prefill 单请求的 chunk 链天然是串行的。

### 8.3 反证：4096 chunk 并不会更快

| 配置 | 63k prefill | 速率 |
| --- | ---: | ---: |
| `max_num_batched_tokens=2048`（当前） | 11.88 s | **5,300 tok/s** |
| `max_num_batched_tokens=4096`（KV 需 ≥1.5 GB，否则启动即报 1.36 GiB > 1.21 GiB） | 12.48–12.57 s | 5,020 tok/s |

即"MoE 每步固定权重流 → 更大 chunk 更省"的推论**不成立**：chunk 翻倍后每步的激活/注意力
gather/中间张量流量同步上升，净效果为负。2048 已是当前内核组合下的局部最优。

### 8.4 本轮 MoE 内核实验（全部无效，已回滚）

| 实验 | 结果 |
| --- | --- |
| `kStages` 3 → 4（`mxfp4_cutlass_core.cuh`，重编 `_moe_C`） | 17.80 vs 17.79 ms ⇒ 1.00× |
| n-fastest grid（第 2 轮做过） | 1.01× |
| tile 换档 | 更慢 |

⇒ 该内核在其设计内已到顶：权重流 749 GB/s（copy 峰值 60%）+ mma 100 TOPS（int8 峰值 32–45%）。
再快只能重写 mainloop 的寄存器分块/指令组合（例如把 mxfp4 解包从主循环里挪出去、
改用更大 K-tile 的双缓冲），这是数天量级的内核工作。

### 8.5 对 10000 tok/s 的最终判断（有数据支撑）

2048 chunk 下每步预算 205 ms，当前 377 ms（无插桩 ~384 ms）。各项实测下限：

| 项 | 现在 | 该内核组合下的下限 | 差距来源 |
| --- | ---: | ---: | --- |
| MoE | 158 | ~75（权重流 72 + 不可避免的激活） | mainloop mma 效率 |
| attn（全部） | 97 | ~60 | wo_b/SWA/未细分 ~35 需逐个定位 |
| hc | 18 | ~12 | TileLang kernel |
| shared experts | 11.5 | ~8 | 未与路由专家重叠 |
| metadata | 11 | ~8 | |
| 步间气泡 | 60 | ~15 | 需要 PP 多 microbatch 重叠 |

⇒ 即使全部达到下限也只有 ~178 ms/step（11,500 tok/s 的乐观值），
但要在**每一项**上都拿到 20–40% 是多个独立的内核/引擎工程；
按当前内核，实测平台在 **5,300–5,600 tok/s**，理论乐观天花板 ~6,200。
**结论：10,000 tok/s 在本硬件 + 当前内核组合下不可达；需要（a）重写 MoE mainloop 拿到 ~2×，
（b）消除 PP 每步气泡**。
