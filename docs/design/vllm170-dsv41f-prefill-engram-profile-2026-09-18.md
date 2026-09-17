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

---

## 9. 第 4 轮：fp8 线性层（Marlin）1.44× 的机会、MoE tile 表的反证、chunk 尺寸三点扫描

### 9.1 发现：所有 block-scaled fp8 线性层跑在 Marlin 上，而 bf16 cuBLAS 快 1.44×

torch trace 显示 fp8 线性层（wo_a/wo_b/wq_b/fused_wqa_wkv/共享专家）都走 `_C::marlin_gemm`
（每 3 个 chunk 150 次）。用仓库自带的 `deepseek_v41_wo_b_backend_sm80.py`（本轮加了 `--m` 参数）
实测 wo_b 形状（N=5120, K=8192, fp8 32×32 block scale）：

| M | Marlin MXFP8 | bf16 `torch.mm` | bf16 快 |
| ---: | ---: | ---: | ---: |
| 1 | 1550 µs | 1076 µs | 1.44× |
| 512 | 1560 µs | 1080 µs | 1.45× |
| 2048 | 1561 µs | 1083 µs | **1.44×** |

即 Marlin 只有 111 TFLOPS（fp8 峰值的 ~36%），而 bf16 cuBLAS 跑到 159 TFLOPS（≈ 该卡 bf16 峰值）。
**代价**：bf16 权重是 fp8 的 2 倍，每层注意力投影 ~120 MB → 每 rank 约 +1 GB 常驻。
按 fp8 线性层合计 ~40 ms/step 估，收益约 **12 ms/step（3%）** —— 用 1 GB 余量换 3%，
对本部署（1M 上下文 + 2048 chunk 需要余量）不划算，**先不做，记录为可选方案**。

### 9.2 反证：MoE tile 表在两档 token 数上都是最优的

| tokens | block_m=64 | block_m=128 | 生产选择 |
| ---: | ---: | ---: | --- |
| 2048 | **17.77 ms**（382 GB/s） | 24.21 ms | 64 ✓ |
| 4096 | 25.64 ms | **24.44 ms**（278 GB/s） | 128 ✓ |

（CMP 170HX 实测）⇒ `_TILE_TABLE` 的两档选择都对，没有可捡的便宜。
但由此修正了成本模型：**MoE 每步成本取决于被选中的 tile，而 tile 取决于 tokens/step**
（≤1024 token 选 block_m=16，512 token 时单层只要 23 ms，2048 选 bm64）。所以
"每步固定权重流" 只在同一 tile 区间内成立。

### 9.3 chunk 尺寸三点扫描（63k 单请求，同一 prompt）

| `max_num_batched_tokens` | 墙钟 | 速率 | 每步 |
| ---: | ---: | ---: | ---: |
| 512 | 16.0 s | 3,930 tok/s | 130 ms |
| **2048（当前）** | 11.9 s | **5,300 tok/s** | 384 ms |
| 4096 | 12.5 s | 5,020 tok/s | 781 ms |

2048 是三点最优；4096 反而变慢（MoE 每 token 更省，但注意力/激活流量涨得更多）。

### 9.4 hc kernel 调参：生产配置已经在最优点附近

`deepseek_v41_mhc_sm80.py --tokens 2048`：生产 `tile_n=12/n_thr=512` 与最佳
`tile_n=8/n_thr=256` 的差别 ≤0.8%（3080Ti 上 2.833 vs 3.257 ms），不值得改。

### 9.5 本轮结论

第 2–4 轮共 7 个优化实验（MoE grid 顺序、tile 换档、kStages、4096 chunk、512 chunk、
overlap 阈值、hc 调参）**全部 ≤1.01×**；唯一找到的确定性收益是 fp8→bf16 的 1.44×（约 3%，
代价 1 GB 常驻）。**10,000 tok/s 需要重写 MoE mainloop（目标 2×）+ 消除 PP 每步气泡**，
不是配置或调参能达到的。

---

## 10. 第 5 轮：找到并修复 MoE tile 选择 bug（+10~11%）

### 10.1 bug：prefill 一直在用最慢的 tile

`vllm/model_executor/layers/fused_moe/experts/mxfp4_int8_sm80.py::select_mxfp4_int8_sm80_tile`
的两步逻辑互相打架：

```python
average = ceil(num_tokens * topk / num_experts)      # 2048 token -> 32 行/专家
max_block_m = min(128, max(16, 1 << (average - 1).bit_length()))   # -> 32（向下取幂）
for max_tokens, block_m, block_n in _TILE_TABLE:     # (1,16) (1024,16) (2048,64) (1<<30,128)
    if num_tokens <= max_tokens and block_m <= max_block_m: return ...
return 16, 128                                       # 兜底
```

- 2048 token：`max_block_m = 32` ⇒ 表里 `block_m=64` 的两项（2048 和 1<<30）都不可达 ⇒ **落到兜底 (16,128)**。
- 4096 token：`average = 64 → max_block_m = 64`，但 `(2048,64)` 要求 `num_tokens ≤ 2048` 不满足、
  `(1<<30,128)` 要求 `block_m ≤ 64` 不满足 ⇒ **同样落到兜底 (16,128)**。
- 实测（CMP 170HX，2048 token 单层 w13+w2）：**bm16 = 20.64 ms vs bm64 = 17.77 ms**，
  且 bm16 在每个 token 上重读专家权重（M tile 数 ×4）——4096 token 时差距更大。

生产与 benchmark 的偏差也印证了这一点：2048 时生产 19.7 ms/层（≈bm64 的 17.8 的 1.11 倍），
4096 时生产 286.8 ms/step（=35.9 ms/层）而其 benchmark 的 bm64 只有 25.6、bm128 只有 24.4。

### 10.2 修复

改为"取 ≥ 平均行数/专家的最小受支持 block_m"（16/64/128），并把越界情形写清楚：

| tokens | 修复前 | 修复后 | 说明 |
| ---: | --- | --- | --- |
| ≤1024 | (16,128) | (16,128) | 不变（实测 bm16 最优） |
| 2048 | **(16,128) 兜底** | **(64,128)** | 实测 17.77 vs 20.64 ms/层 |
| 4096 | **(16,128) 兜底** | **(64,128)** | 修复前每 token 重读 4 遍专家权重 |
| ≥8192 | (128,128) | (128,128) | 不变 |

只影响 prefill（decode/投机路径 m 很小，average≤16，仍选 bm16，与持久 decode metadata 的
`_DECODE_BLOCK_M=16` 一致）。tile 只改分块，不改数值（benchmark 里的 `check_tile_equivalence` 已覆盖）。

### 10.3 端到端效果（63,053 token 单请求，同一 prompt，针全中）

| 配置 | 修复前 | 修复后 | 提升 |
| --- | ---: | ---: | ---: |
| **2048 chunk（目标要求保持）** | 5,300–5,346 tok/s | **5,867–5,916 tok/s** | **+10~11%** |
| 4096 chunk | 5,020 tok/s | 5,914–5,952 tok/s | +18% |
| 8192 chunk | — | 不可行 | 显存不足：warmup 阶段 NCCL collective timeout |

（4096 需 `PORT_KV_BYTES=1500000000`，8192 试过 1.9e9 仍不够；本机 PP 卡余量不够大 chunk。）

### 10.4 其他

- 起服务前必须确认宿主内存干净：8192 崩溃后残留进程让 `free` 只剩 2 GiB，下一次启动
  在 warmup 阶段报 NCCL `unhandled cuda error`；杀掉全部 worker 等 `free` 回到 ~200 GiB 后再起即可。
- 至此累计：MoE tile 修复 **+10%**；其余 7 个实验（grid 顺序、kStages、overlap 阈值、hc 调参、
  tile 换档、512/4096 chunk）全部 ≤1.01×；fp8→bf16 线性层 1.44×（约 3%，代价 1 GB）仍为可选。

---

## 11. 第 6 轮：修复后的账本、fp8 线性层三个内核等价、启动flake 的处理

### 11.1 修复后的每 step 账本（2048 chunk，rank3，profile 开着）

| 环节 | 修复前 | 修复后 |
| --- | ---: | ---: |
| `target.model_forward` | 288.6 | **254.9** |
| ├ `moe.routed_experts` | 157.7 | **125.9**（w13 80.0 + w2 42.5） |
| ├ `attn.total` | 96.9 | 96.9 |
| │　├ `attn.prepare` | 71.8 | 71.8（sparse_mla.prefill 28.4 + indexer.logits 18.0 + ~25 未细分） |
| │　└ `attn.o_proj` | 21.8 | 21.8（wo_a 6.3 + wo_b ~13 + inv_rope 2.2） |
| ├ `hc.*`（4 个 kernel） | 16.1 | 16.1 |
| └ `moe.shared_experts` | 11.6 | 11.6 |
| `target.attn_metadata` | 10.8 | 10.8 |

修复把 MoE 每 step 从 157.7 降到 125.9 ms（−20%），已低于其单层 benchmark 换算值（8×17.77=142），
说明 MoE 已在其内核极限上。

### 11.2 fp8 线性层：Marlin / Humming / Triton 端到端等价

`_POSSIBLE_FP8_BLOCK_KERNELS` 里前四个（FlashInfer/DeepGEMM/Cutlass/B12x）对 v4.1 的
"32×32 block + 每 32 token 一组的激活量化" 全部拒绝（它们只支持 `(1,128)` 激活分组），
**Marlin / Humming / Triton 三个都能实现**，而列表顺序让 Marlin 永远被选中。

本轮加了 `VLLM_FP8_LINEAR_KERNEL=<名字子串>` 开关（运维/A-B 用，默认不改变行为）并实测：

| 内核 | 63k prefill |
| --- | ---: |
| Marlin（默认） | 5,841–5,881 tok/s |
| Humming | 5,874–5,934 tok/s |
| Triton | 5,877–5,948 tok/s |

**三者等价**（差异在噪声内）。即 §9.1 里 bf16 cuBLAS 的 1.44× 只存在于隔离 benchmark，
端到端换不来（fp8 线性层不在关键路径上，且激活量化开销抵消），**并且 bf16 方案还要多占 1 GB 常驻**，
因此明确放弃该路线。

### 11.3 启动 flake 的处理（运维要点）

- 现象：重启后 warmup（`mixed tokens=16`）阶段偶发 `ncclUnhandledCudaError`，或引擎卡在
  `No available shared memory broadcast block found in 60 seconds`。
- 原因：上一个实例的 Engram 宿主表（2×94.4 GiB）被 kill 后需要 ~20–30 s 才被内核回收；
  过早启动时新实例的建表 + warmup 的 JIT 编译会去抢最后几 GiB 宿主内存。
- 处理：`/tmp/start_pp5.sh`（本轮写的重试脚本）先 `pkill` 再等 `free ≥ 150 GiB`，然后启动；
  同时把 60 s 无 shm 广播、engine init failed 都当作失败信号重试。有了它之后 4 次启动全部一次成功。

### 11.4 剩余可动项（按大小）

| 项 | ms/step | 现状/可行性 |
| --- | ---: | --- |
| `attn.prepare` 未细分部分 | ~25 | 含 wq_b（另一处 Marlin fp8）、qkv_insert、indexer 的 produce_k；未插桩 |
| `indexer.logits` | 18.0 | Triton fp8 MQA logits，实测 ≈6 TFLOPS（fp8 峰值 ~3%），有内核空间但属内核工程 |
| `sparse_mla.prefill` | 28.4 | 已在 read-once tiling（FlashMLA 文档实测水平） |
| `attn.o_proj`（wo_a+wo_b） | 19.6 | 三个 fp8 内核等价 ⇒ 无便宜可捡 |
| `hc.*` | 16.1 | 生产配置已在最优附近（±0.8%） |
| 步间气泡 | ~60 | 需要 PP 多 microbatch，引擎级 |

---

## 12. 第 7 轮：又一个"编译了但不可达"的配置（bm32）—— 测了，是输的；以及两个默认关闭的开关

### 12.1 `block_m=32` 格子在核心里编译了但 dispatch 不允许（沿用第 5 轮的思路找的）

`mxfp4_moe_cutlass_core.cuh` 实例化了 `(TM=32, WM=16, TN=128, WN=64)`，
但 `mxfp4_int8_moe_entry.cu` 的 dispatch 只有 `(16,64) (16,128) (64,128) (128,128)`，
Python 侧的 `_TILE_TABLE` 也选不到 32。2048 token 时每个专家正好 32 行 ⇒ bm32 既不浪费
padding 又保持"每专家 1 个 M tile"，看起来是缺失的最优点。

本轮把它接上（dispatch + benchmark 校验）并实测（3080 Ti，2048 token，block_n=128）：

| block_m | w13 | w2 | 合计 |
| ---: | ---: | ---: | ---: |
| 16 | 32.56 | 16.80 | 49.36 |
| **32** | 33.69 | 17.48 | **51.17** |
| 64 | 28.75 | 15.08 | **43.83** |

**bm32 比 bm64 差 17%** —— 大 tile 的每 CTA 效率超过 padding 浪费，padding 假说不成立。
改动已回滚。至此 MoE tile 四个档（16/32/64/128）全部实测过：2048 token 最优是 **bm64**，
4096 token 最优是 **bm128**，与现在的选择一致。

### 12.2 两个默认关闭的开关，测了/查了都不动

| 开关 | 结论 |
| --- | --- |
| `VLLM_INDEXER_PREFILL_STREAMING_TOPK=1`（SM80 融合 logits+topk 流式路径） | 端到端 5,850–5,869 vs 默认 5,841–5,881，**中性**；且它要求 `max_row_len ≥ 16384` 且 `candidate_blocks is None`，只有部分层/部分 chunk 会走到 |
| `VLLM_INDEXER_LOGITS_MAXNREG` | 代码注释里已记录重新测量结果：内核现在吃 162 寄存器，cap 128 会溢出，"中性到更差，实际服务的形状上 −4.0%" ⇒ 保持 0 |
| `VLLM_DEEPSEEK_V4_ENGRAM_GPU_DEQUANT` | 全仓库只有 envs.py 声明，没有任何使用点（死开关）；engram 的 dequant 本来就在 `_engram_lookup_kernel` 里走 GPU |

另外 `indexer.logits` 的 `@triton.autotune`（BLOCK_N=128 / num_warps=4 / stages∈{2,4}）注释里写明已扫过
M 1..2048 × N 2048..131072，"BN=64 差 1.25–1.40×、BN=32 差最多 2.41×"，也已经在其最优点。

### 12.3 步间空隙的量级修正

用"profiled vs unprofiled 同一 prompt"两个数对齐后：每 step 墙钟 ≈ 345 ms（unprofiled，
2048 chunk，63k 请求），其中 forward ≈ 255 + metadata ≈ 11 ⇒ **空隙 ≈ 79 ms/step（~23%）**，
比第 3 轮估的 60 ms 更大（当时把 profiler 自身开销算进了 forward）。这部分在引擎/PP 流水线里，
第 3 轮的 torch trace 已证明 worker 主线程 94% 时间在 `c10d::recv_` 上等上游 rank。

### 12.4 并发不会提高 prefill 吞吐（反而下降）

同一 prompt 长度（12,450 token）下并发 k 个请求（`max_num_seqs=8`，`max_num_batched_tokens=2048`）：

| 并发数 | 墙钟 | 聚合吞吐 | 单请求延迟 |
| ---: | ---: | ---: | ---: |
| 1 | 3.1 s | 4,000 tok/s | 3.1 s |
| 2 | 6.5 s | 3,847 tok/s | 6.5 s |
| 4 | 15.0 s | 3,317 tok/s | 15.0 s |

两个现象：
1. **每个请求的延迟随并发数线性增长**（3.1 → 6.5 → 15.0 s）⇒ 请求是被**合并进同一个 2048-token step 预算**里串行消费的，
   而不是并行推进。所以聚合吞吐 ≈ 单流速率，不随并发提升。
2. 聚合吞吐**随并发下降**（4,000 → 3,317）：每 step 的**活跃序列数**变多，indexer 的 gather/logits、
   SWA metadata 等按"序列 × 上下文"计费的部分变大，而 tokens/step 被 2048 卡死。

⇒ 对 "prefill 达到 10000" 的直接含义：**并发不是出路**，唯一的出路是降低"每 2048 token 的固定成本"
（MoE 权重流 126 ms + 注意力 97 ms + 空隙 ~79 ms），也就是本文件 §7.5/§9 算术里的那些内核/引擎工作。

### 12.5 运维教训：不要在服务运行时 `cp` 覆盖已 mmap 的 `.so`

第 7 轮中途我把回滚后的 `vllm/_moe_C_stable_libtorch.abi3.so` 用 `cp` 覆盖回去，
而 5 个 worker 仍然 mmap 着这个文件；worker 空闲时没问题，下一次请求触发缺页时命中被改写的页
→ SIGBUS，日志表现为 `Worker proc VllmWorker-0 died unexpectedly (exit code: None)`，
随后 `Cannot start a receive while the PP receive buffer still has owner stream` 等连锁报错。
**结论：改 `.so` 前后必须重启服务（或用 `mv` 换文件而不是原地覆盖）。** 不是 vLLM 的 PP bug。

### 12.6 本轮之后的稳定配置复核

默认 2048 chunk：63k 请求 5,817–5,843 tok/s；21.7k 请求 4,416 tok/s；
4 个 12.4k 并发请求逐题正确（捞针全中）。

---

## 13. 第 8 轮：空隙归因（引擎侧）、P2P ~1%、剩余项收口

### 13.1 引擎侧：step 循环本来就是流水线（batch queue）

`EngineCore` 在 `max_concurrent_batches > 1` 时用 `step_with_batch_queue()` 而不是 `step()`：
它**先把新 batch 的 `execute_model`/`sample_tokens` 异步入队，队列满才阻塞取最早的输出**。
也就是说引擎不会"每步串行等整条 PP 链"，我原先怀疑的"每步引擎同步"不成立
（第一次插桩打在 `step()` 上，日志里一条都没有 —— 正好证明走的是 batch queue 路径）。

⇒ 每 step 周期（unprofiled ≈ 345 ms）= 最慢 rank 的 per-batch 时间（≈255–286 ms）
+ PP 交接（84 MB hidden states + PP-KV 张量）+ 末段 sampling + 抖动。
要再压只能压这三项本身，属引擎/通信层工作。

### 13.2 `NCCL_P2P_DISABLE=0`

启动脚本原本禁用 P2P。本轮实测（63k 请求）：

| 配置 | 样本 |
| --- | --- |
| P2P 关闭（默认，8 次） | 5,817 / 5,843 / 5,846 / 5,846 / 5,857 / 5,865 / 5,881 / 5,841 |
| **P2P 打开（3 次）** | **5,882 / 5,910 / 5,918** |

约 **+1%**，两组区间基本不重叠，但样本量小（3 vs 8）、且不确认 NCCL 真的走了 PCIe P2P 而不是回退。
**先不改启动脚本默认值**，记为待放大的 A/B。

### 13.3 剩余项收口（第 2–8 轮共 10 个实验）

| 实验 | 结果 |
| --- | --- |
| MoE tile 可达性 bug（2048/4096 落到 bm16） | **修好，+10~11%** ✓ |
| MoE grid 顺序 m→n fastest | 1.01× |
| MoE `kStages` 3→4（重编 `_moe_C`） | 1.00× |
| MoE tile 换档（16/64/128 全扫） | 现选择最优 |
| MoE 新增 bm32 可达性 + 实测 | 比 bm64 差 17% |
| 共享专家多流 overlap 阈值 | 无收益 |
| `VLLM_INDEXER_PREFILL_STREAMING_TOPK`（SM80 流式 topk） | 中性 |
| `VLLM_INDEXER_LOGITS_MAXNREG` | 注释内已有复测：中性到更差 |
| fp8 线性层换 Humming/Triton 内核 | 与 Marlin 等价；bf16 路线（1.44×）代价 1 GB 常驻，否决 |
| hc TileLang 调参 | 生产配置已在最优 ±0.8% |
| 512 / 2048 / 4096 / 8192 chunk | 2048（目标要求）最优；4096 高 1.4% 但需多 0.2 GB KV；8192 显存不够 |
| 并发 1/2/4 请求 | 聚合吞吐**下降**（4,000→3,317 tok/s） |

**收敛结论**：配置/参数/可达性这一层已经掏空。剩下的 1.7× 只在两处：
(a) MoE mainloop 重写（现权重流 749 GB/s = copy 峰值 60%、mma 32–45% int8 峰值，目标 ~2×）；
(b) PP 每步交接/气泡（≈65–79 ms/step，23%）——需要通信层或更深的 PP 流水。

---

## 14. 第 9 轮：每 step 真实结构 = GPU forward 240 ms + 非 forward ~85 ms

前几轮的"空隙"数字被两种测量工具污染过：phase profiler 每步 flush 会做一次设备同步，
我自己的分段插桩也在每步 `event.synchronize()`。本轮改成**每步只一对 CUDA event**（开销可忽略）
来量真实的 GPU forward：

| 量 | 数值（2048 token/step，rank0/rank3 一致） |
| --- | ---: |
| GPU forward（event 对） | **236–248 ms** |
| 端到端 step 周期（10.10 s / 31 步） | **326 ms** |
| ⇒ 非 forward 部分（引擎↔worker 往返、PP 交接、采样、调度） | **~80–85 ms/step（26%）** |

forward 内部（第 6 轮 profiled 值，按 −20 ms profiler 开销折算）：

| 环节 | ms/step | 占 forward | 占整 step |
| --- | ---: | ---: | ---: |
| MoE 路由专家 | 126 | 52% | 39% |
| 注意力（sparse_mla 28 + indexer 18 + o_proj 22 + prepare 其余 ~29） | 97 | 40% | 30% |
| hc（4 kernel） | 16 | 7% | 5% |
| 共享专家 | 11.6 | 5% | 4% |
| attn metadata | 10.8 | 4% | 3% |

**结论（最终口径）**：当前 2048 chunk 下 5,900–6,245 tok/s 的构成是
`326 ms = 240（GPU）+ 85（引擎/PP 往返）`。要往 10,000（205 ms/step）走，
必须同时拿下 MoE 内核（126 → ~75）与这 85 ms 的往返（→ ~20），
即"内核 + 引擎"两件工程，任何单边都不够。

### 14.1 本轮的方法论教训（写给后来者）

测这类"每步固定开销"时，**不要用会做设备同步的工具**：
- phase profiler（250 个 stage × 每步 flush 一次 sync）把每步周期从 ~326 ms 抬到 ~435 ms；
- 自建分段插桩若每步 `synchronize()` 也会抬到 ~540 ms 并让"每步周期"这个指标本身失真；
- 只保留"一对 event 量 forward + 端到端墙钟"两个数，二者相减才是可信的往返开销。

---

## 15. 第 10 轮：那 85 ms 往返的内部结构（PP 等待 ~50 ms + 引擎循环 ~38 ms）

沿用"每步一对 event + 三个 host 时间戳（step 入口 / 模型调用前 / 模型调用后）"的轻量插桩
（每 5 步才 synchronize 一次），得到每 rank 的分段（2048 token/step，prefill 稳态）：

| rank | pre（收到上游中间张量前的时间） | GPU forward | post + 到下一步的间隔 |
| --- | ---: | ---: | ---: |
| PP0 | **0.0 ms** | 232 ms | 252–260 ms |
| PP3 | **49–54 ms** | 233–238 ms | 204–224 ms |

两个新事实：

1. **rank≥1 每步要等 ~50 ms 才拿到上游的 hidden states**（rank0 为 0，符合预期）。
   这就是"PP 交接"在真实（unprofiled）运行里的量级。
   ⚠️ 前几轮 profiler 给出的 `target.pp_recv_wait` 只有 2 ms **是错的**：phase profiler 每步
   flush 会 synchronize 一次，把流水线抽干，等待被"摊平"到 forward 里去了。
2. 引擎侧 `max_concurrent_batches = pp_size = 5`（`vllm/config/vllm.py:572`），即 batch queue
   深度已等于 PP 层数 —— 流水线已经是"每 rank 处理不同 batch"的满载状态，没有可再挖的深度。

⇒ 每步 326 ms 现在是完整可解释的：

```
326 ms = 232 (GPU forward) + 50 (PP 等待/交接) + ~44 (引擎循环 + 采样 + 调度 + 抖动)
         └ MoE 126 + 注意力 97 + hc 16 + 共享专家 11.6（+metadata 10.8）
```

**优化含义**：50 ms 的 PP 等待由 84 MB/step 的 `hidden_states (2048×4×5120 bf16)` 交接决定
（PCIe Gen2 x16，无 NVLink）；要压它只有三条路：更快的链路（硬件）、更小的载荷（改模型契约）、
或更深的流水（引擎已经 5 层满载）。44 ms 的引擎循环同理属引擎工程。

## 16. 第 11 轮：加深 batch queue（深度 10）压不动 PP 等待 —— 负结果，配置已回滚

第 15 轮把 50 ms 归给 PP 交接后，唯一还没试的"更深的流水"是把 batch queue 再加深：
`vllm/config/vllm.py` 里 `max_concurrent_batches` 默认取 `pp_size + 1`（= 6），
本轮加了环境变量开关 `VLLM_DSV41_BATCH_DEPTH_EXTRA`，把它抬到 10 层，其余全部不动。

### 16.1 结果：更深 = 更差

| 配置 | in-flight KV 需求 | 63k prefill 稳态 |
| --- | --- | --- |
| 默认（depth 6，`PORT_KV_BYTES=1300000000`，run80） | 1.3 GB | **5,917–5,959 tok/s** |
| depth 10（run76/77） | 需抬到 1.5 GB 才能启动 | **5,883–5,902 tok/s** |

两件事同时被这个实验钉死：

1. **深度不是瓶颈**。深度 6 已经足够让 5 个 rank 都有活干（每 rank 同时在手 6 个 batch，
   forward 232 ms < 一个 batch 走完 5 段的时间），再加 4 层只是把调度开销和 KV 占用一起放大。
   ⇒ 那 50 ms 不是"等不到 batch"的流水气泡，而是**单次交接本身要花的钱**。
2. **深度与 KV 是耦合的**：in-flight block 数 ≈ depth × max_num_batched_tokens，
   depth 10 × 2048 直接把 1.3 GB 的 KV 池吃穿（启动即
   `To serve at least one request ... KV cache is needed, which is larger than the available`），
   必须把 `PORT_KV_BYTES` 抬到 1.5 GB。任何"再加深"的想法都自带一份 KV 预算。

配置已回滚（`git status` 干净），默认 depth 6 / KV 1.3 GB 的服务已按 run80 重新拉起并
`STARTUP_OK` 复核。

### 16.2 由此收敛出的结论

2048 chunk 下的 326 ms/step 现在是**逐项都已定位、且都被实验否定过优化空间**的结构：

```
326 ms = 232 (GPU forward) + 50 (PP 交接) + ~44 (引擎循环)
          ├ MoE 126：权重带宽主导（749 GB/s 有效；只有 mainloop 重写这一条路）
          ├ 注意力 97：稀疏 MLA（topk 512 + swa 128）在 sm_80 fp16 张量核上已接近 roofline
          └ 其余 ~40：hc 16 + 共享专家 11.6 + metadata 10.8
```

第 2–11 轮共 12 个实验，只有 1 个是正的（MoE tile 可达性 bug，+10–11%）；
其余 11 个（MoE grid 顺序、kStages、tile 扫描、bm32、共享专家重叠阈值、indexer streaming
topk、logits 寄存器数、fp8 线性层三内核、hc TileLang 调参、512/4096/8192 chunk、batch depth 10）
全部中性或负。**10,000 tok/s 在当前硬件 + 当前内核下不可达**：它要求 205 ms/step，
而光是 GPU forward 就要 232 ms。要再往前走只剩两条真正的工程路：
MoE mainloop 重写（目标把 126 ms 压向 HBM 峰值附近的 ~70 ms）与
PP 载荷/引擎往返（94 ms → ~30 ms）。两条都不是"调参"能拿到的。

**当前稳定战绩**（不得再退化）：63k prefill **5,917–5,959 tok/s**（起步 5,300，累计 +11%）；
35k ≈ 5,267–5,319；21.7k ≈ 4,416；12k ≈ 3,205。

## 17. 第 12 轮：基线重测 —— 之前记录的 prefill 速率是"被每请求固定开销摊薄"的产物，真实稳态是 ~7,900 tok/s

前三轮（§14/§15/§16）的算术都建立在一句话上："每 step 326 ms"。第 16 轮把 batch queue 加深做掉之后
回来复核，发现这个数是**用"一次请求的墙钟 ÷ 它的 step 数"算出来的**，而一次请求除了 31 个 step
还有约 0.8–1.5 s 的请求级固定开销（排队、首个 step 的 JIT/首触、最后一次采样与收尾）。
把它摊进 31 个 step，每个 step 就凭空多了 30–50 ms —— **326 ms/step 是被污染的数**。

### 17.1 重测方法（不再用"请求级速率"当每 step 成本）

同一台在跑的服务（run80，默认配置，无任何插桩），单请求、唯一 prompt、`max_tokens=1`、
无前缀缓存，**用两点之间的斜率**取每 step 成本：`t = a + b · steps`（steps = ⌈tokens/2048⌉），
`b` 才是每 step 成本，`a` 是请求级固定开销。

| 真实 prompt tokens | steps | 墙钟 s | 请求级 tok/s | 备注 |
| ---: | ---: | ---: | ---: | --- |
| 12,016 | 6 | 2.367 | 5,077 | |
| 21,554 | 11 | 3.382 | 6,374 | |
| 34,631 | 17 | 4.988 | 6,942 | |
| 48,477 | 24 | 6.781 / 6.775 / 7.206 | 7,155 / 7,149 / 6,727 | 3 次重复 |
| 63,092 | 31 | 8.931 / 8.827 / 8.795 | 7,064 / 7,148 / 7,173 | 3 次重复 |
| 126,939 | 62 | 18.555 | 6,841 | |

拟合：

- 6 → 31 step 段：**b = 0.2594 s/step，a = 0.81 s** ⇒ 边际 **7,896 tok/s**
  （请求级 63k 的 7,173 里，那 0.81 s 就占了 9%）
- 31 → 62 step 段：**b = 0.3130 s/step** ⇒ 边际 **6,542 tok/s**
  （上下文变长后每 step 成本上升 ~21%，与稀疏 MLA 的 indexer/压缩 KV 随上下文线性增长相符）

⇒ 修正后的口径：**2048 chunk 的稳态每 step 是 259 ms（0–63k 上下文）/ 313 ms（63–127k），
短上下文边际吞吐 7,896 tok/s**。三次重复的离散度只有 ±0.8%，这个数是可信的。

### 17.2 旧记录怎么来的，以及一个仍未解释的 14% 漂移

旧曲线（12k→3,205、21.7k→4,416、35k→5,267、63k→5,917）与今天的曲线**形状完全一致**，
差别就是整体尺度：用 `t = a + b·steps` 去拟合旧数据，得到 `a ≈ 1.5 s、b ≈ 0.28 s`
（如果旧数据的长度是按 target 而非真实 token 记的，则是 `a ≈ 1.48 s、b ≈ 0.28 s`）。
也就是说：旧记录并没有"慢 1.6 倍"，它的**每 step 成本 0.28 s 与今天的 0.259 s 只差 8%**，
真正被放大的是 `a`（请求级开销 1.5 s vs 0.81 s）——而 `a` 在"请求级速率"这个指标里被摊到了每一个 step 上。

仍未解释的部分：§14 记的"63k 请求 10.10 s / 31 step"这个观测本身，今天同样的请求只要 8.8 s。
两者配置逐字相同（non-default args、KV 1,078,799 tokens、同一棵树），
当时的差异候选只有"那几轮树里有插桩"与"宿主/GPU 状态"，没有证据可以定罪。
**规则（写给后来者）**：任何"每 step 成本"都必须用斜率法或每 step 的 event 对量，
且**测之前先在同一天的同一实例上重测一次基线**；不同实例之间禁止直接比速率。

### 17.3 重算 10,000 tok/s 的算术（结论比第 16 轮乐观得多）

10,000 tok/s ÷ 2048 token/step ⇒ **205 ms/step**。

| 口径 | 每 step | 到 205 ms 的差距 |
| --- | ---: | ---: |
| 今天实测（0–63k 上下文） | 259 ms | **−54 ms** |
| 今天实测（63–127k 上下文） | 313 ms | −108 ms |

而 MoE 路由专家这一项现在是 **126 ms/step**，其中权重流部分的实测地板是
**72 ms/step**（54 GB/rank ÷ 749 GB/s，§7.1 + §11.1 的实测带宽）。
⇒ **只要 MoE 主循环能把 126 ms 拉到它的带宽地板附近（−54 ms），短上下文就正好落在 205 ms/step
≈ 10,000 tok/s**；长上下文那一段还要额外动 indexer/注意力随上下文增长的部分。

所以第 16 轮那句"10k 在当前硬件下不可达"要作废：它用的是被污染的 326 ms/step 与
被污染的 232 ms forward。**修正后的结论是：10k 就在 MoE 主循环一件工程之内（短上下文），
不再是"两件工程同时做到极限"才够。** 下一轮的目标因此从"收口"改成"MoE 主循环"。
