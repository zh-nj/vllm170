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
