# dsv41f 移植决策书：撤销本项目 dsv41 提交 + 移植 cpu-hybrid 分支（2026-09-17）

## 0. 目标（用户原话）

> 撤销本项目关于 dsv41f 推理的提交，全面移植 deepseek-v41-cpu-hybrid 分支的 dsv41f 推理
> 用于 5 卡 170hx 以 pp 模式推理，不需要 cpu 混推，保留本项目分布式 dspark

**安全网（已执行，未改动任何代码）**

| 对象 | 值 |
| --- | --- |
| 备份 tag | `backup/main-20260917-pre-dsv41f-port` → `2f1b9caa7d` |
| 工作分支 | `port/dsv41f-hybrid` → `2f1b9caa7d`（与 main 同点，尚未切换） |
| 当前所在分支 | `main`（用户未提交改动原样保留） |

---

## 1. 侦察结论：三条硬事实（决定工作量与风险）

### F1　该分支**没有** `mxfp4_int8_sm80`，本项目有

```
$ ls .worktrees/dsv41-cpu-hybrid/vllm/model_executor/layers/fused_moe/experts/ | grep -i mxfp4
aiter_mxfp4_w4a16_moe.py   aiter_mxfp4_w4a8_moe.py   cpu_mxfp4.py   trtllm_mxfp4_moe.py
$ grep -rl "mxfp4_int8_sm80\|Mxfp4Int8Sm80" .worktrees/dsv41-cpu-hybrid/{vllm,csrc}   # 空
```

本项目的 `mxfp4_int8_sm80`（Python expert + `csrc/libtorch_stable/moe/mxfp4_int8_sm80/*`）
是 **PP5 纯 GPU、SM80、384 专家** 的生产 MoE 后端，也是本轮"不需要 cpu 混推"后唯一能承接
MXFP4 路由专家的路径之一（另一条是 Marlin）。

**推论**：字面意义上的"全面替换 + 去掉 CPU 混推"会**删掉本项目自己的 GPU MoE 后端**，
而替换源里没有等价物 ⇒ MoE 只能回落 Marlin，等于把 V4.1/PP5 的推理链换回 DSv4/PP3 时代的方案。

### F2　逐文件对比：本项目在 8 个文件上明显更厚（替换即回退）

| 文件 | 本项目 | cpu-hybrid | 本项目多出的内容（例） |
| --- | ---: | ---: | --- |
| `amd/rocm.py` | **1310** | 906 | direct-ragged SWA / paged decode SWA / caller-output / split-32 / topk 转换缓存 |
| `attention.py` | **1406** | 1229 | 上述开关的接入与预热、短上下文 indexer 门 |
| `common/engram.py` | **1821** | 1145 | SSD 转换文件预载 + `POSIX_FADV_DONTNEED` + 全量主存驻留 + GPU 反量化 staging |
| `nvidia/model.py` | **1482** | 1287 | PP 压缩 KV 副本（latent+index K 传递）、candidate 两级过滤、lookahead 窗口 |
| `nvidia/dspark.py` | **750** | 586 | 分布式 draft 分阶段放置、私有 embed、topk 审计 |
| `compressor.py` / `sparse_mla.py` / `indexer_k_store.py` | 352/261/285 | 323/231/254 | 单请求 token-map 快路径、相位剖析、PP 归一化 index K 导入 |
| `nvidia/model_state.py` | 122 | **333** | （唯一被对方领先的文件：CPU 混推的 model-state 钩子——正是本轮不要的部分） |

对方**独有的文件只有 10 个**，其中 7 个是 CPU 混推专属：

```
cpu_moe.py  expert_cache.py  cache_policy.py  cache_feedback.py
host_memory.py  hybrid.py  hybrid_runtime.py          ← CPU 混推，本轮不要
ced.py  pp_kv.py  ampere/prefill_metadata.py           ← 与部署形态无关，可移植
```

**推论**：真正值得移的"分支独有推理代码"只有 **3 个新文件 + DSpark Q/K 解耦 + 若干通用小改**；
其余替换都是净回退。

### F3　整体 revert 的爆炸半径远超 dsv41

9 个 dsv41 提交共触及约 200 个文件，其中**只有 ~54 个在 `deepseek_v4_1/` 内**，其余 ~148 个是共享基建：

| 提交 | 文件数 | dsv41 相关 | 共享基建 |
| --- | ---: | ---: | ---: |
| `7039dab0f3` V4.1 五卡 SM80 | 129 | 34 | **95** |
| `a1a6a2dda5` Optimize PP5 | 46 | 12 | **34** |
| `2f1b9caa7d` indexer top-k 确定性 | 6 | 0 | **6** |
| `1df00d0c60` Marlin/router 几何 | 5 | 1 | 4 |
| 其余 5 个 | 21 | 9 | 12 |

共享基建里含：`csrc/libtorch_stable/moe/mxfp4_int8_sm80/*`、`vllm/model_executor/kernels/mhc/*`
（V4/V4.1 解码器的 tilelang MHC 核）、`vllm/v1/worker/gpu/*`、`vllm/v1/attention/*`、
tokenizer/parser for `deepseek_v41`、以及多处 GLM/Qwen 共用的图捕获与 PP 逻辑。

**推论**：逐提交 `git revert` 会连带打断 GLM-5.3 / Qwen4Exp / DeepSeek-V4（非 1）等其它模型，
不是"只影响 dsv41"的操作。

---

## 2. 两条路线

### 路线 A：字面执行（全量替换）

1. 在新分支从 `7783087333`（dsv41 之前的基线）出发；
2. 把 `vllm/models/deepseek_v4_1/*` 等整段换成该分支版本，剔除 `cpu_*.py / expert_cache.py /
   cache_policy.py / cache_feedback.py / host_memory.py / hybrid*.py / cpu_mxfp4.py / csrc/cpu/dsv41/*`；
3. MoE 回落 Marlin（该分支唯一可用的 SM80 MXFP4 GPU 路径）；
4. 把 `7af2170e0b` 分布式 DSpark 在新区线上重做（它的父提交被 revert，无法直接 cherry-pick）；
5. 处理 0.13 基座 → 0.27 基座的 API 适配（`build_attn_metadata`、KV manager、model_runner、
   `AttentionGroup`、`create_metadata_builders` 等签名差异）。

**代价**：失去 F2 表里全部 8 项本项目已领先的能力（含稀疏 prefill tiling 1.51×、
确定性 indexer top-k、Engram SSD 预载、mxfp4_int8_sm80），并把 GLM/Qwen 共用的基建暴露在风险里。
**收益**：得到 CED / pp_kv / prefill_metadata 这 3 个文件 + DSpark Q/K 解耦。

### 路线 B：精准移植（推荐）

保留本项目 5 卡纯 GPU PP5 基座，只把该分支**与部署形态无关**的机制搬过来，分三批：

**批次 1（小而独立，收益确定，风险低）**

| 项 | 来源 | 落点 |
| --- | --- | --- |
| DSpark 宽度解禁（V4.1 免整除校验）+ `dspark_num_query_tokens` Q/K 解耦 + `dspark_backbone_vllm_config` | `648218d25`、`05922aa2b` | `config/speculative.py`、`config/vllm.py`、`spec_decode/dspark/{speculator,utils}.py` |
| `kv_cache_tokens` 物理 KV 预算（精确 block 公式 + 引擎侧预留 + 差多少 GiB 报错） | `05922aa2b` | `config/cache.py`、`v1/core/kv_cache_utils.py`、`sched/scheduler.py`、`engine/arg_utils.py` |
| 每步对全部 running 请求回收 SWA 块 | `05922aa2b` | `sched/scheduler.py`、`v1/core/kv_cache_manager.py` |
| indexer prefill logits 容量取 2 的幂（分配复用） | `05922aa2b` | `v1/attention/ops/mqa_logits_triton.py`、`model_executor/layers/sparse_attn_indexer.py` |
| PP 各 stage 后端 KV layout 取交集 | `05922aa2b` | `v1/attention/backends/utils.py` |

**批次 2（中等，prefill 结构）**

| 项 | 来源 | 注意 |
| --- | --- | --- |
| PP KV 传「已打包 fp8 行 + scatter」+ 中继走本地副本 | `pp_kv.py`（新文件）+ `nvidia/model.py` 接入 | 载荷 11.5→2.6 KB/token；**必须常驻复用缓冲**（失败案例：指针拓扑不一致 → acceptance 塌到 0–2.4%） |
| SM80 融合索引构造（单 launch 构建 topk+SWA 行） | `ampere/prefill_metadata.py` + `ampere_sparse.py::_combine_prefill_indices` | 本项目已有 caller-owned 工作区，只取融合与去同步的思路 |

**批次 3（大，需与 PP 重分层联合决策）**

| 项 | 来源 | 前置 |
| --- | --- | --- |
| CED prefill（encoder 0–19 全量 + decoder 20–39 有界 replay ≤128 token） | `ced.py`（新文件）+ `model_state.py` 接入 + `model_runner.py` 的 `skip_compiled` 门 + `dspark` 的 CED draft context | 8,8,8,8,8 全驻留下上界 ≈7%（关键路径仍是 rank0/1 的 8 层）；只有配合专家卸载或重分层才值 ~1.5–2× |

**全程保留**：本项目分布式 DSpark（`7af2170e0b` + 工作区未提交的 rank-local draft 图捕获，
其 launcher 注释记录 108/108 正确、单流 2.174s → 1.544s / −29%）。

---

## 3. 待确认

路线 A 与 B 的工作量与风险差一个数量级，且 A 会造成 F1/F2/F3 三项净损失。
按 E 的默认选择继续，或明确指定 A。

另需确认一件事：**工作区里未提交的分布式 DSpark 改动**（`run_cmp170hx_pp5.sh`、
`dspark/speculator.py` +368、`dflash/cudagraph.py`）是否先由你提交到 main，再在其上做移植？
（我未改动这三个文件；若由我提交，会把它们单独作为一笔提交，不混进移植提交。）

---

## 4. 执行进度（路线 A，已落地第一笔提交）

**分支**：`port/dsv41f-hybrid`（基于 `7783087333`，即 dsv41 之前的基线；9 个 dsv41 提交不在历史里）
**提交**：`99d653fde9` — 183 文件，+31538/−1578

### 已完成

| 项 | 结果 |
| --- | --- |
| 安全网 | tag `backup/main-20260917-pre-dsv41f-port`、分支 `backup/wip-dspark-graph-20260917`（含被移除的未提交分布式 DSpark 图捕获改动，可随时恢复） |
| 工作区未提交改动 | 已按指示移除（备份在上一行分支） |
| dsv41 实现 | `vllm/models/deepseek_v4_1/**` 整体取分支版本（32 个文件） |
| CPU 混推剔除 | 删除 6 个模块；`nvidia/model.py`、`nvidia/model_state.py`、`nvidia/vl_model.py` 三处引用已剥离 |
| 分布式 DSpark | 保留（本项目实现），并叠加分支的 Q/K 解耦增强 |
| 0.27 适配 | `is_cutedsl_supported()`（SM90 门）、注册 `TRITON_MLA_SPARSE_DSV41` 后端 |
| 验证 | 18 个模块导入通过；模型类解析 `DeepseekV41ForCausalLM`；后端解析通过；DSpark 配置矩阵 K∈{3,5,7,10} 与 Q=5/K=3 可构造，K=7 已解禁，Q<K 仍拒绝 |

### 尚未完成（下一轮）

1. **真机跑通（最关键）**：5 卡 PP 加载模型 + 推理。目前只到"导入/配置层通过"，
   尚未证明权重加载与一次前向。
2. **launcher 适配**：`benchmarks/deepseek_v4_1/run_cmp170hx_pp5.sh` 是为主项目模型写的
   （env 旋钮、engram 配置），需与分支实现对齐。
3. **Engram 集成风险**：分支的 `Engram` 只读 `engram_config.cpu_offload`，
   不认 `cpu_cache_gb` / `ssd_offload_path`（本项目的全量主存 + SSD 转换文件方案）。
   直接用本项目 launcher 会让它退回 pinned host 常驻（每 rank ≈94 GiB），
   需要么改 launcher、么补这三个字段的语义。
4. **`prepare_full_cudagraph_inputs` 钩子**：本项目的 runner 会调用它做 Engram staging；
   分支模型没有对应实现（默认 no-op），FULL cudagraph 下需要确认是否可行。
5. **共享基建里的 dsv41 本地优化**：稀疏 prefill tiling、indexer top-k 确定性、
   Marlin/router 几何固定仍是本项目版本（它们是共享文件、被 GLM/Qwen 共用，
   字面 revert 会连带打断其它模型）。需要你决定是否也要退回。

---

## 5. 第 2 轮：结构性发现与跨 rank KV 补齐

### 5.1 硬结论：分支的 dsv41 实现不支持「KV 共享组跨 rank」的 PP 切分

| 证据 | 位置 |
| --- | --- |
| 消费层找不到源层 cache 时**直接抛错** | `attention.py`：`NotImplementedError("...PP splits inside a v4.1 kv-sharing group are not supported.")` |
| `pp_kv_transfer` 的计划**显式拒绝** boundary ≥ `candidate_source_layer_id`(20) | `pp_kv.py::SharedKVTransferPlan.for_stage`：`raise ValueError("PP KV replicas require an encoder group with one index source")` |
| 实测 | `8,8,8,8,8` 的边界 24、32 被拒；`[7,8,25]` 通过 |

原因：解码组 21–39 有 4 个 index source（24/28/32/36），而分支的实现假设「源层与其消费者同 rank」，
这正是它把 21 层解码段放末段（`[7,8,25]`）的前提——而该前提**只在 CPU/主机专家下成立**
（21 层 ×≈7.4GB ≈155GB，单卡 63GB 装不下）。

**因此"5 卡 PP + 全域 GPU 专家 + 分支 dsv41 实现"必须补上跨 rank 的 KV 共享传递。**

### 5.2 已补齐（第 2 轮提交）

在分支模型内并入本项目的跨 rank KV 机制：

- `attention.py`：新增 `DeepseekV4PPCompressedCache`；源层记录 `_last_compressed_latent`；
  indexer 增加 `export_normalized_k` / `_last_normalized_k`
- `common/ops/indexer_k_store.py`：采用本项目版本（新增 `input_is_normalized` /
  `return_normalized`，向后兼容）
- `nvidia/model.py`：`_pp_incoming_{source,index,topk,candidates}` / `_pp_outgoing_*` 计划、
  4 个 PP 张量、`_import_pp_attention_state` / `_append_pp_attention_state`，并接入 forward
- 8,8,8,8,8 的计划实测：rank0/1 不发不收；rank1→2 传 topk+latent(源14)；
  rank2→3 传 latent(源20)+归一化 index K+候选块；rank3→4 中继同一组（源 20 在 rank2）

### 5.3 运行期适配（第 2 轮）

- 分支模型用的 4 个 kernel wrapper（`_COMPUTE_SWA_INDICES_AND_LENS_KERNEL`、
  `_COMPRESSED_SLOT_MAPPING_KERNEL`、`_PREPARE_UNIFORM_DECODE_KERNEL`、
  `_COMPUTE_DSPARK_NONCAUSAL_SWA_INDICES_KERNEL`）在本项目的后端文件里不存在
  → warmup 注册改为容错（缺失只跳过预编译，不影响正确性），并保留原有 wrapper 的注册
- 新增 `benchmarks/deepseek_v4_1/run_port_pp5.sh`：5 卡 PP5 启动脚本，
  不开 CED/`pp_kv_transfer`（两者在 8,8,8,8,8 下均不可用），保留分布式 DSpark

### 5.4 仍未完成

1. **真机跑通**：仍在加载/启动阶段排错（当前已越过 import 层）。
2. **CED 在 5 卡上不可用**：其约束（层 19 与全部解码层同段）在纯 GPU 驻留下无法满足，
   `ced_prefill` 只能保持关闭。
3. 分支 engram 只认 `cpu_offload`，不认 `cpu_cache_gb`/`ssd_offload_path`。

## 6. 第 3 轮：真机跑通（已完成）

### 6.1 最后一个启动阻塞：`prepare_attn` 签名

```
TypeError: DefaultModelState.prepare_attn() takes from 7 to 8 positional arguments but 9 were given
  at vllm/models/deepseek_v4_1/nvidia/model_state.py:142
```

分支时代的 `ModelState.prepare_attn` 多一个 `ubatch_idx` 参数，本项目的
`DefaultModelState.prepare_attn`（以及 `model_runner.py:1819`、`cudagraph_utils.py:711`
两个调用点）都没有。该参数在本仓库中只出现在这个 override 里，直接删除即可。
删掉后引擎正常启动：加载 → KV 预留 → CUDA graph 捕获 → warmup → `Application startup
complete`（engine init 73.7 s）。

### 6.2 显存包线：PP2 只剩 ~1.1 GiB，2048 chunk 必 OOM

各 rank 稳态占用（`nvidia-smi`，单卡 65024 MiB）：

| rank | GPU | weights（加载报告） | 稳态占用 | 空闲 | 峰值 |
|------|-----|--------------------|----------|------|------|
| PP0  | 1   | 55.10 GiB          | 60204 MiB | 4.8 GiB | 60650 |
| PP1  | 2   | 57.48 GiB          | 63154 MiB | 1.9 GiB | 63540 |
| PP2  | 3   | 56.06 GiB          | 63918 MiB | **1.1 GiB** | **64368** |
| PP3  | 4   | 56.06 GiB          | 63538 MiB | 1.5 GiB | 63986 |
| PP4  | 5   | 54.92 GiB          | 62432 MiB | 2.6 GiB | 62816 |

默认 `--max-num-batched-tokens 2048` 时首次长提示词（6841 token，chunked prefill）在 PP2 上：

```
[rank2] memory allocation failed with OOM on device 2 while trying to allocate
134217728 bytes (free: 118226944, total: 67725885440)
→ RuntimeError: torch_call_dispatcher("aten::new_empty"...) failed
   at .../torch/csrc/stable/ops.h, line 939
   （栈：attention.py:916 fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert）
```

即权重之外只剩 ~1.1 GiB 设备空闲，而 1024 chunk 的一次 prefill 峰值就要 ~450 MiB，
2048 chunk 直接翻倍超出。**结论：1024 是这台机器的安全 chunk；实际用
`expandable_segments:True` 把约 4.8 GiB 的碎片化缓存池回收后再试 2048。**

### 6.3 已验证的行为（run10，`PORT_MAX_BATCHED=1024`）

- 单请求 43 token 提示 → 答案正确（Paris），含 reasoning 段
- 单请求 300 token 生成：11.5 s ≈ 26 tok/s，代码片段正确
- 6841 token 长上下文大海捞针：**答对**（`QUARTZ-9182`），3.8 s
- 800 token 生成：33.1 s ≈ **24.2 tok/s**
- 8 路并发（提示 1.9k–10.6k token，各 120 token 输出）：31.7 s 全部完成，8/8 捞针答对，
  峰值显存与单请求持平（PP2 64372 MiB）
- DSpark 投机：drafts 91 / draft tokens 455（=K×drafts，K=5 生效）/ accepted 250，
  **接受率 54.9%**，逐位 0.86/0.66/0.48/0.43/0.36，平均接受长度 3.78
  —— 对比移植前本项目分布式 DSpark 的 17.71%，接受率已恢复到正常水平

### 6.4 显存账本：63,918 MiB 是怎么来的

三个数据源：`nvidia-smi` 的 per-process 占用、`MemorySnapshot`/`memory_profiling` 的
`torch` 账本（用 `VLLM_DSV41_MEM_DEBUG` 临时插桩打印）、以及分配器 block 直方图
（`torch.cuda.memory_snapshot()`）。PP2 = rank2 = GPU3，标称 65,024 MiB / torch 侧 63.07 GiB。

| 项 | GiB | MiB | 来源 |
| --- | --- | --- | --- |
| CUDA context + 驱动 + NCCL（non-torch） | 0.47 → 0.55 | 481 → 560 | init snapshot（62.6/63.07）；稳态 `device_used − torch_reserved` |
| 权重（8 层 + 本 rank 的 draft stage） | 56.06 | 57,405 | `Model loading took` |
| KV cache（`kv_cache_memory_bytes=1.3e9`） | 1.21 | 1,240 | `reserved ... for KV Cache` |
| CUDA graphs（FULL_DECODE_ONLY，7 档） | 0.32 | 328 | `Graph capturing finished` |
| 常驻工作区/激活池（本 rank 实测激活峰值 1.28 GiB，含 `max_model_len` 尺寸的 indexer workspace） | ~3.8 | ~3,900 | `torch_alloc` 61.414 − 上三项 57.59 |
| torch 已缓存但空闲（可复用） | 0.33 | 338 | `torch_reserved − torch_alloc` |

权重那 57,405 MiB 的主体是**每层路由专家**：block 直方图显示 rank2 上是
`4320 MiB × 9 + 2160 MiB × 8`，而 4320 MiB = 384 专家 × 2 × 2304 × 5120 × 0.5 B（fp4），
2160 MiB = 384 × 5120 × 2304 × 0.5 B，正好对上 `moe_intermediate_size=2304`、
`expert_dtype=fp4`；剩下 ~500 个 32–64 MiB 小块（合计 ~2.3 GiB）是每层的 scales/norm/hc。

关键推论（解释"为什么 2048 chunk 必 OOM"）：**稳态空闲 = 65,024 − 63,892 ≈ 1.1 GiB，且其中
0.33 GiB 处于碎片化缓存池**。默认分配器要 128 MiB 连续块时只能 `cudaMalloc`（free 118 MiB →
失败）；1024 chunk 的 prefill 峰值只要 ~450 MiB，2048 chunk 直接翻倍。

**还测到一个反直觉的杠杆**：把 `--max-model-len` 从 1M 降到 262k，PP0 的
`Available KV cache memory` 从 3.62 GiB 涨到 5.38 GiB —— 涨的 ~1.8 GiB 是 SM80 稀疏
indexer 的常驻 workspace（`(max_model_len, index_head_dim=128)` bf16 = 256 MiB/份）。
也就是说 **1M 上下文不是白拿的，它每卡吃掉约 1.8 GiB 的 KV/激活余量**。

