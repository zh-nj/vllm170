# DeepSeek-V4.1 CPU 混合分支流程对照与可吸收收益点（2026-09-17）

对照对象：`/mnt/data/apps/vllma100/vllm-backport` 的 `origin/codex/deepseek-v41-cpu-hybrid`
（HEAD `648218d25`，工作区 `/mnt/data/apps/vllma100/.worktrees/dsv41-cpu-hybrid`）。
本体：`/mnt/data/apps/vllma100/vllm170`（`main`，vLLM 0.27.2.dev6 基座，PP5 全驻留）。

两者面向同一块卡（NVIDIA CMP 170HX，SM80/GA100，单卡 63.07 GiB 可用，PCIe Gen2 x16、
无 NVLink），但选择了**完全不同的部署形态**：

| 维度 | cpu-hybrid 分支 | vllm170 |
| --- | --- | --- |
| 并行 | **PP3**，`pipeline_layers=[7,8,25]`（层 15–39 全在末段） | **PP5**，`8,8,8,8,8`（`VLLM_PP_LAYER_PARTITION`） |
| 专家放置 | 层 20–39 的 384 专家**全部常驻主机**（IK AVX2 MXFP4，≈134.5 GiB），GPU 只放一份**动态专家缓存**（把命中专家重打包成 **Marlin** 布局执行） | 40 层专家**全部驻留显存**，生产后端是 **`mxfp4_int8_sm80`**（`run_cmp170hx_pp5.sh:67`，profile 文档亦记「MoE 仍为 mxfp4_int8_sm80，没有使用 Marlin」；Marlin 只用于 dense MXFP8 线性层）。每层 ≈7.4 GB，每卡 8 层 ≈59.2 GiB |
| 主机内存 | 加载后 worker ≈344 GiB PSS，建议 512 GiB；host LRU 默认 12 GiB | Engram（≈94.42 GiB/rank）驻留**普通（非 pinned）**主存；GPU 侧 ≈24.75 MiB pinned staging + ≈48.75 MiB device（staging 24.75 + `staged_rows` 24.0，后者是 FULL graph 实际读的常驻缓冲） |
| KV | `--kv-cache-tokens 1048576`（新增的**物理 token 预算**），`max-model-len 524288` | `--kv-cache-memory-bytes 1.3e9`（≈1,078,799 token），`max-model-len 1048576` |
| 前缀缓存 | **强制关闭**（CED/专家缓存的前置条件） | PP5 启动脚本也是 `--no-enable-prefix-caching` |
| CUDA Graph | `FULL_DECODE_ONLY` | `FULL` + breakable/piecewise |
| DSpark | `num_speculative_tokens=3`，`dspark_num_query_tokens=5`（新增） | `num_speculative_tokens=5`（= checkpoint block size），draft 分布在 PP1/2/3、heads 在 PP4 |
| 实测吞吐 | **分支内没有任何 3 卡端到端吞吐数字**（`.gitignore` 忽略 `docs/validation/`，原始数据在外部 `/root/optim/*.md`） | 1K TTFT p50 555.6 ms、32K/256 6.738 s（有效 prefill ≈4,933.8 tok/s）；decode 有两套口径：CMP graph 轮 19.55 tok/s，2026-09-14 重启后生产基线 **35.657 tok/s（28.046 ms/token）**——且 `max_num_seqs=1→8` 之后的最终值**至今未复测**（文档明说），真值在 25.8–37 tok/s 之间（§4.2b） |

> 阅读提示：本文中「已实测」均给出文件/文档出处；「推算」是我基于两边代码与已发布数字做的上界估计，
> 标注为 ≈ 的区间不可当作承诺值。

---

## 0. 结论摘要

**该分支真正新增（相对 vllm170 缺失）的能力只有 4 类**：CED prefill（编码/解码分离 + 有界 replay）、
主机侧原生 MXFP4 专家 + 动态 GPU 专家缓存、PP KV 以「已打包 584B 行」形式传递、DSpark 查询宽度与
提案宽度解耦。其余大量文件是它自带基座或与 vllm170 同源的代码。

按「对 vllm170 的收益 / 代价」排序：

| 级别 | 收益点 | 预期收益 | 工作量 | 风险 |
| --- | --- | --- | --- | --- |
| **P0** | DSpark 宽度解禁 + Q/K 解耦（`dspark_num_query_tokens` + backbone config） | 让宽度扫描回到常规路径；K=7 在 1K 是 2.63×（32K 上 q5 仍最优）；Q>K 为待验证假设 | 小（≈80 行 + 测试） | 低（已验证当前 K=7 被拒） |
| **P0'** | （不是吸收项）**PP5 分布式 DSpark acceptance 塌陷**：11.281 tok/s / 17.71%，同轮 target-only 25.778 tok/s（生产基线 35.657 tok/s） | 直接决定 PP5 上 DSpark 是否该继续用 | 定位为主 | — |
| **P0** | `kv_cache_tokens` 物理 token 预算 + 精确 block 公式 + 引擎侧预留 | 1M 容量可判定、可复现；不再靠 `kv-cache-memory-bytes` 试错 | 中（≈200 行） | 低 |
| **P1** | 主机内存护栏（`MemAvailable` + cgroup `memory.max` 分块校验）用于 94 GB×2 Engram 加载 | 把「host OOM-kill」变成可读错误；补上 full-RAM 模式**当前完全没有**的资源门禁 | 小（≈60 行） | 低 |
| **P1** | Engram full-RAM 冷启动自举与文件完整性校验 | 现在冷机首次启动会静默退化成「94 GiB checkpoint 读进 pageable RAM」；3 点采样判完整可能静默用错表 | 小–中 | 低 |
| **P1** | 每步对**所有** running 请求回收 SWA 块（`recycle_sliding_windows`） | 紧 KV 下的抢占/抖动减少；KV 池仅 1.21 GiB（1.3e9 B）时更明显 | 小（≈15 行） | 低 |
| **P1** | PP KV 改为传「已打包 fp8 行 + scatter」而非 latent + 接收端重做 RoPE/量化 | PP 载荷 ≈11.5 KB/token → ≈2.6 KB/token；每边界每 chunk 少 2 个 kernel | 中（≈250 行） | 中（KV 预算/布局耦合） |
| **P2** | indexer prefill logits 分配按 2 的幂取容量（`round_allocations`） | 长上下文增长时分配器复用，减少碎片 | 极小（≈10 行） | 低 |
| **P2** | `PREFILL_CHUNK_SIZE = min(4, max_num_seqs)` | **不需吸收**：vllm170 已有的按实际批长决定 gather 请求数 + 512 MiB 上限更强（PP2 空闲显存 ≈507 MiB → 3,459 MiB） | — | — |
| **P2** | CED prefill（编码/解码分离 + 有界 replay） | **在 8,8,8,8,8 全驻留下 ≈0–8%**；在「专家卸载/重排分层」下可达 ~1.5–2× | 大（跨 rank replay、suffix 缓冲、每请求隔离） | 高 |
| **P3** | 共享专家旁路流（`cpu_shared_stream`）、fused Markov argmax、PP KV layout 交集 | 1–3% / 仅 TP>1 有效 / 健壮性 | 小–中 | 低 |
| **战略** | 移植「主机原生专家 + 动态 GPU 专家缓存」整体形态（PP3、3 卡跑 1M） | 未知（该分支未发布吞吐）；需先跑它的微基准 | 很大 | 高 |

**已经具备、不要重复做的**（详见 §5）：PP 压缩 KV 副本（latent+normalized index K 传递）、DSpark
non-final chunk 的 context-only 快路径与 KV-only insert、Engram 主机驻留 + pinned staging + GPU 反量化、
低峰值（原地/staged）专家重打包路径（`mxfp4.py:819-829` 的
`prepare_moe_mxfp4_layer_for_marlin_inplace`，注意生产后端是 `mxfp4_int8_sm80`，这条只在切到
Marlin 时生效）、单 launch 的 `combine_topk_swa_indices`（且带 caller 工作区）、int64 router、
PP draft 词嵌入私有副本。

**vllm170 领先、但有两个例外**：indexer top-k 确定性、Marlin M-tile/router GEMM 几何固定、
稀疏 prefill 流水化 tiling + `TEMP_MB` 预算、Engram SSD 预加载 + `posix_fadvise(DONTNEED)`、
caller-owned 索引工作区、阶段化 profiling。
例外一：**分布式 DSpark 放置（`dspark_pipeline_stage_ranks`）目前是负收益**（§4.2b），
不要把它当作可输出的经验。例外二：**Engram full-RAM 模式的资源门禁与冷启动自举反而不如该分支的
`check_host_headroom` + 分块校验**（§3.4、§7）。

---

## 1. 该分支的模型结构事实（两边同源，先对齐术语）

来自 `/media/z/data6/models/DeepSeek-V4.1-Flash/config.json`（vllm170 直接用该文件实测读出）：

```
num_hidden_layers = 40, hidden_size = 5120, hc_mult = 4, head_dim = 512
sliding_window = 128, index_topk = 512, index_head_dim = 128, index_n_heads = 32
compress_ratios = [0,0, 2×18, 1×20, 0,0,0]        # L0-1 纯 SWA；L2-19 ratio-2；L20-39 ratio-1；MTP 3 层 ratio-0
kv_source_layer_ids   = [2, 8, 14, 20]
index_source_layer_ids= [2, 8, 14, 20, 24, 28, 32, 36]
candidate_source_layer_id = 20, candidate_topk_blocks = 2048, candidate_block_size = 8
dspark_block_size = 5, dspark_target_layer_ids = [37,38,39], dspark_markov_rank = 256
dspark_n_routed_experts = 128, dspark_num_experts_per_tok = 3
engram: 2 张表 384,006,168 / 384,016,682 行 × head_dim 256
```

关键结构推论（两边代码都依赖它，也是 CED 的正确性基础）：

1. **层 21–39 没有自己的压缩 KV cache**，它们读 L20 发布的压缩 KV + indexer topk（`kv_source_for(20)=20`）。
2. **每个压缩层仍各有自己的 128 token SWA cache**（`sparse_swa` 的 `swa_len`），所以层 21–39 的
   「长期状态」在 L20，而「短期状态」只有最近 128 个 token。
3. 因此 **L20–L39 这半个模型对每个 token 的输出，只依赖「L20 的压缩 KV（全序列）」+「最近 128 个
   token 的 hidden」** —— 这正是 CED 能成立的原因（§2.2）。

---

## 2. Prefill 环节对照

### 2.1 该分支的 prefill 流水

入口与普通路径一致（scheduler → chunked prefill），差别在 `DeepseekV4Model` 里挂了两个新模块
（`.worktrees/dsv41-cpu-hybrid/vllm/models/deepseek_v4_1/nvidia/model.py:537-580`）：

- `pp_shared_kv = IncrementalPPKV(...)`：当 `additional_config["pp_kv_transfer"]` 打开时构造，
  为每个 PP 边界创建「压缩 KV 副本」缓存（`pp_kv.py:143-213`）。
- `ced_prefill = CEDPrefill(config.sliding_window)`：仅末段 rank，且要求
  `start_layer < 20 and end_layer == 40`（否则抛错）。

每步 prefill 的形态：

```
rank 0..k-1（编码段）: layers 0..19 全 token 前向（各自 SWA + L2/8/14 压缩 KV + indexer）
        └─ 边界：把「新增的压缩 KV 行(584B) + topk indices」塞进原有 PP 消息（不额外开通信）
末段 rank: L20 用编码段最终 normalized hidden 调 publish_decoder_kv() → 把全序列投影进解码段
           的压缩 KV（pp_kv.py 之外，见 ced.py:239-247）
        └─ 非最终 chunk：解码段 20..39 直接跳过，返回零 hidden（ced.py:334-335）
        └─ 最终 chunk：把每请求保留的 ≤128 token suffix 重放一遍 20..39，产出
           hidden / aux_hidden / MTP hidden（ced.py:337-397）
```

### 2.2 CED prefill（该分支最有价值的 prefill 机制）

**机制**（`.worktrees/dsv41-cpu-hybrid/vllm/models/deepseek_v4_1/ced.py`）：

- `SuffixBuffer(width=sliding_window=128)`：每个请求只保留「最近 ≤128 个 token」的
  `hidden / pre_mix / slots / positions / input_ids`，跨 chunk 拼接；要求
  **从 position 0 连续**（`ced.py:47-70`，`start != next_position` 直接抛错）。
- `publish_decoder_kv(attn, normalized, positions)`（`ced.py:239-247`）：用 L20 的
  `fused_wkv_wgate` + compressor + `insert_cache` + indexer `_produce_k`，把**每一个编码 token**
  投影进解码段共享的压缩 KV / indexer K。这一步不含 attention、不含 MoE，只有一层投影。
- `CEDPrefill.forward`（`ced.py:253-397`）：非最终 chunk 在 publish 之后直接返回零张量；
  最终 chunk 才把各请求的 suffix 拼起来，用 `build_attn_metadata` 重建一份**只覆盖 suffix**的
  attention metadata（`ced.py:170-236`），跑 20..39，再把结果按请求切回各自输出位置、填
  `_mtp_hidden_buffer` 与 `auxiliary_output`。
- SWA gather 长度被显式夹到 replay 长度：`metadata.prefill_gather_lens = min(prefill_gather_lens, bounds)`
  （`ced.py:224-236`），保证 replay 只读 128 token 窗口。
- 附带把 CED 的 replay 结果交给 DSpark：`CEDDraftContext` + `take_ced_draft_context()`
  （`ced.py:262-267`，使用方见 `dspark/speculator.py:172-190`）——**在 propose 之前就把 draft 的
  context KV 写好**，省掉 draft 侧的重复投影。

**正确性依据**：§1 的三条结构事实。因为解码段 21–39 无自有压缩 KV、其 SWA 只看 128 token，
重放最近 128 个 token 就能恢复解码段全部状态。

**前置条件**（都是从代码里读出来的硬约束，不是建议）：

| 约束 | 出处 |
| --- | --- |
| 必须 v2 model runner | `ced.py:25-45` |
| TP=1，且**不能**用 ubatching/DBO | `ced.py:31-36` |
| **必须关闭 prefix caching**（suffix 必须从 position 0 连续） | `ced.py:34-36`、`hybrid.py:58-59` |
| 只支持 DSpark 投机 | `ced.py:38-43` |
| 末段 rank 必须持有 19–39 层 | `model.py:568-575`、`hybrid.py:27-38` |
| 现只实现 Ampere attention 后端 | `ced.py:271-278` |

**收益推算（vllm170 视角，这是关键判断）**

- 该分支的 `[7,8,25]` 分层：中间 chunk 的关键路径 = `max(7, 8, ~1) ≈ 8 层`；若不 CED，
  末段要串 25 层 → 完全不可行。**CED 是这套分层的使能条件，而不是一个可叠加的百分比优化。**
- vllm170 的 `8,8,8,8,8` 全驻留：中间 chunk 仍然 rank0/rank1 各跑 8 层 → **关键路径不变**；
  能省的只有 rank2→3→4 两跳 PP 传输（≈41 KB/token × 2048 token × 2 ≈ 168 MB/chunk，
  按该分支自己的实测链路 ~5.9 GB/s 折算 ≈28 ms）与 rank3/4 的空转。
  以 32K prefill 6.522 s / 16 chunk ≈ 406 ms/chunk 计，**上界 ≈7%**，且 1K 短 prefill 几乎为 0。
- 想让 CED 真正变成 1.5–2×，必须让解码段 20 层在 prefill 期间可以被「赶出」关键路径，也就是
  **把解码段专家卸载到主机**（回到该分支的形态）或重新分层——而全驻留下每卡最多 8 层
  （7.4 GB×8=59.2 GiB < 63.07 GiB），根本排不出「编码 4 卡 / 解码 1 卡」的分层。
- **口径说明（避免误读 2× 这个数）**：CED 砍掉的是「解码段 20 层 × 全部 prefill token」，
  换成「解码段 20 层 × 每请求 ≤128 token」（32K prompt 上解码段的 token 量降约 250×）。
  在**没有 PP**（单卡/TP）时端到端上限就是「解码段占整模型 FLOPs 的比例」，接近 2×；
  在**PP 下**这个上限还要再被「最慢 rank 不包含解码段」这个条件打折 ——
  vllm170 的 `8,8,8,8,8` 恰好不满足（rank0/rank1 仍跑 8 层编码层），所以只剩传输与空转的收益。
- 结论：**除非同时接受专家卸载或牺牲上下文/并发，CED 对当前 vllm170 部署的净收益很小，
  而实现与回归成本很高**（跨 3 个 rank 的 replay、每请求 suffix 缓冲、与 PP 消息/图的耦合）。
  建议先做 §7 的测量，再决定。
- **一份支持这个判断的旁证**：vllm170 自己的 PP5 decode profile 显示瓶颈就在
  **PP0/PP1**（`target.model_forward` 均值 PP0 73.315 / PP1 73.024 ms，对 PP2–PP4 的
  68.162–70.957 ms；189 轮中「最慢 rank」PP0 占 100 次、PP1 占 66 次），
  归因于早期 index-source 层（2/8/14）与 Engram 常驻层 —— 也就是**前 16 层**。
  该文档给的下一步建议正是「小范围 PP 分层扫描 + 串行关键路径复核」，
  而 CED 恰恰是把「分层可以往解码段倾斜」这件事变成可能的前提。
  换言之：**CED 与「PP 重分层」是一个联合决策，单独做 CED 没有意义。**
  （同一个证据也说明：`9/8/8/8/7` 这种更优分层目前因为 PP0 只剩 ≈6.8 GiB 而拿不到，
  这又回到「每卡最多 8 层」的权重硬约束。）

### 2.3 PP KV 传递：从「latent + 接收端重做」改为「已打包行 + scatter」

两边都把「上游源层的压缩 KV」跨 PP 传递，但**传的东西不同**：

| | vllm170（现状） | cpu-hybrid |
| --- | --- | --- |
| 传的载荷 | `_PP_COMPRESSED_LATENT` bf16 `[T,512]` = **1024 B/token**；`_PP_NORMALIZED_INDEX_K` bf16 `[T,128]` = **256 B/token**；`_PP_TOPK_INDICES` int32 `[T,512]` = **2048 B/token**；`_PP_CANDIDATE_BLOCKS` int32 `[T,2048]` = **8192 B/token**（仅当候选源在上游 rank） | `shared_kv_rows` uint8 `[T,584]` = **584 B/token**；`shared_topk_indices` int32 `[T,512]` = **2048 B/token**；不传 index K、不传候选块 |
| 接收端动作 | `rope_quant_insert(...)` + `indexer_k_norm_rope_store(...)` 两个 kernel（`nvidia/model.py:711-760`） | 一个 triton scatter（`_copy_packed_kv_rows`，每行一个 program，1024 线程覆盖 584 B，`pp_kv.py:44-101`） |
| 中继（多跳） | 中继方把 latent 原样再传，**每跳都会重做一次 RoPE+量化+写入** | 中继方从**本地副本缓存**重新打包（`export_chunk` 从 replica 的 `kv_cache` 读），不再做 RoPE/量化 |
| 副本是否占 KV 预算 | 是（`DeepseekV4PPCompressedCache.get_kv_cache_spec`） | 是，且带 `tokens_per_state=compress_ratios[source]`，按压缩比摊薄（`pp_kv.py:127-140`） |

**可吸收点**：把 vllm170 的 PP KV 载荷换成「已打包 fp8_ds_mla 行 + scatter」。收益：
① 载荷 11.5 KB/token → 2.6 KB/token（前提：解码段同段的候选块不需要跨边界，或另想办法）；
② 每边界每 chunk 少 2 个 kernel，中继跳不再重复 RoPE/量化；
③ 中继语义变简单（行是不可变的最终形态）。
代价：要求 `use_fp8_ds_mla_layout`（vllm170 已满足）；副本 spec 会进入 KV 预算（需与 §2.5 一起算）。
**按当前实测的 PP 占比（hidden state 40 KB/token 才是大头：`(T, hc_mult=4, 5120)` bf16），
这一项在单请求 32K prefill 上大约只有 1–3%**，但在高并发/长上下文、PP 传输开始进 critical path 时
会放大。建议列入「等待 profile 证据」的 P1/P2。

补充两条证据：① 2632 B/token/crossing 这个数已经过硬核对 —— 它是 `584 + 4×index_topk`，
而 `index_topk=512` 直接来自 checkpoint `config.json`（不是从 benchmark 反推）；
② 「为什么必须有 pp_kv_transfer」在树里最清楚的表述在该分支的 harness 前置校验里
（`benchmark/kernels/cpu/benchmark_dsv41_hybrid.py:806-817`）：**任何落在压缩组内部的 stage
边界都是非法的**，它与 `pp_kv.py:28-44` 的三个判据逐条一致，并在 harness 层同时钉住了
「CED 要求层 19 与整个解码段在末段」这条规则。

### 2.4 prefill 里的零碎项

1. **`round_logits_allocations`**（混合分支新加，`mqa_logits_triton.py:544-566` +
   `sparse_attn_indexer.py`）：把 indexer prefill logits 的存储容量按
   `triton.next_power_of_2(N)` 分配再切片（返回 strided view，消费端显式传 stride），
   使「上下文增长」时反复复用同一批尺寸，降低分配器碎片/抖动。
   注意两点：① 这是**分配复用**轴，不是**峰值**轴 —— vllm170 已经用
   `VLLM_DSV4_LOGITS_ROW_CHUNK=128` 分 128 行处理把 `[M,N]` fp32 峰值钉住
   （并预留 `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=512`），所以吸收它是再叠一层；
   ② 代价是 `N` 略大于 2 的幂时 transient 可能翻倍，需与 row-chunk 一起调。
   10 行改动，1M 上下文下值得试。
2. **`PREFILL_CHUNK_SIZE = min(PREFILL_CHUNK_SIZE, max_num_seqs)`**（混合
   `attention.py:215-220`）：该值是 `[PREFILL_CHUNK_SIZE, M, head_dim]` bf16 gather 工作区的
   请求倍数，`M = N + window_size + max_num_batched_tokens`、`N = ceil(max_model_len/ratio)` ——
   在 `max_model_len=524288` 下单请求切片就 ≈540 MB，类默认 4 会预留 ≈2.2 GB，
   所以它不得不按并发收口。
   **vllm170 已经有一个更强的版本**：sparse-prefill 的 gather 字节预算用「按 1M 最坏情况预热一行
   + 512 MiB 上限」，运行时再按**本批实际最大长度**决定每次 gather 的请求数 ——
   效果是把 1M 配置的 PP2 空闲显存从 ≈507 MiB 抬到 **3,459 MiB**，并且
   4×32K 墙钟从 128.681 s 降到 **94.327 s（−26.7%）**。
   所以这一项**不需要吸收**，只需确认两边的分块语义一致。
3. **PP 边界 KV 拷贝的分配纪律**：`pack_kv_rows` 每步新建 `[T,584]` 设备张量
   （`pp_kv.py:79`），不在图内固定地址；若吸收 §2.3，**必须改成常驻复用缓冲**。
   这不是"优化建议"而是硬要求 —— vllm170 自己已经因为这个踩过一次事故：
   第一版 draft non-causal SWA metadata 复用在做图捕获时按组分配独立缓冲、重放时却共享了
   第一组的指针，**acceptance 直接塌到 0–2.4%**（`vllm170-dsv4-sm80-dspark-profile-2026-08-15.md`
   §P0 实机结论），文档因此把「**捕获与重放的指针拓扑一致**」写成独立正确性门禁；
   2026-09-15 的 workspace 二次扩容 IMA 是同一族问题。这类故障**不崩、只静默变慢/变错**，
   所以吸收任何"跨步复用缓冲"的方案时都要显式加这条门禁。
   另注：该分支的 `export_chunk` 在非末段 rank 上**每步无条件执行**（decode 也传），
   所以缓冲纪律在 decode 路径同样重要。
4. **不要照抄的两处**（它们是为 64 GB、纯 PCIe 卡做的「用时间换显存」妥协，不是收益）：
   - 每个编码层每个 chunk 一次 `torch.cuda.current_stream().synchronize()`
     （`ampere_sparse.py:45-50`，注释自陈「Bound live eager intermediates after removing Torch's
     implicit syncs」）：这是该分支**最大的未量化串行化成本**，恰好抵消 CED 想要的层间重叠。
     而且 prefill 上**有两个独立的逐层同步点** —— 另一个在 CPU 专家路径
     `cpu_moe.py:311-314`（pinned staging 拷完后必须同步，CPU 调用才能读到数据），
     每 128 行块每层一次。两者叠加正是「prefill 无法与自身重叠」的根因。
     vllm170 的 `combine_topk_swa_indices` 有 caller-owned `out` 工作区
     （`common/ops/cache_utils.py:521-566`），本来就不需要第一下 —— 仅作参考，
     除非某个 shape 上仍看到分配峰值。
   - CED 打开后 prefill 被强制走 eager/跳过 compiled 路径
     （`model_runner.py:1624-1632`，`skip_compiled = has_prefill and requires_eager_prefill`），
     图收益部分回吐 —— 吸收 CED 时必须单独核算这部分损失。

### 2.5 KV 容量规划：`kv_cache_tokens`（建议 P0 吸收）

混合分支新增了一个**物理 token 预算**开关：

- `CacheConfig.kv_cache_tokens`（`vllm/config/cache.py:240-245`）：按「容量 = 该 token 数 + 有界
  的每请求状态 + 在途 scratch + block 对齐」分配物理 KV，并明确禁止与
  `kv_cache_memory_bytes` / `num_gpu_blocks_override` 同用（`:313-320`）。
- `get_kv_cache_configs` 里走一条**显式公式**而不是 profile 猜测：
  `blocks = max(ceil(max_memory_usage_bytes(budget_config)/pool_bytes_per_block) * num_reqs
  + (num_reqs-1)*len(groups), ceil(total_bytes * target / (pool_bytes_per_block * max_len))) + 2`
  （`vllm/v1/core/kv_cache_utils.py:2687-2706`），并对 sliding-window / chunked-local 组用更紧的
  `_token_budget_num_blocks`（`:2404-2447`）。
- 引擎侧**显式预留**并在装不下时报出具体差多少 GiB（`:2713-2733`），
  日志打印「Reserving X GiB KV per worker for N aggregate tokens and up to M active requests」。

**为什么对 vllm170 有价值**：vllm170 现在的容量是「给字节数 → 从启动日志反推能装多少 token」
（`docs/design/vllm170-dsv41-dspark-pp5-memory-2026-09-14.md` 里明确记录了
「400 MiB/1 GiB 预算不能宣称支持 1M，实际配置必须经过启动日志的 token 容量检查」这一段教训）。
换成 token 预算后：1M 目标变成可声明、可复现、可被 CI 断言，且剩余显存（Engram staging、
graph workspace）的边界更清楚。

**注意**：若同时吸收 §2.3 的 PP 副本，副本会按其 `tokens_per_state` 进入这份预算（
`pp_kv.py:127-140`），要一并核算。

---

## 3. Decode 环节对照

### 3.1 该分支的 decode 流水（三路并行 + 一个流屏障）

每层 MoE 的形态（`nvidia/model.py:142-172`，`.worktrees/dsv41-cpu-hybrid`）：

```
router(gate) → fused_topk_bias(indices_type=int64)
  ├─ 旁路流 A：shared_experts（纯 GPU，cpu_shared_stream）
  ├─ 旁路流 B：GPU 专家缓存（fused_marlin_moe + expert_map 跳过非驻留专家）
  └─ 主流 C：cudaLaunchHostFunc(main_stream, run, task) → 原生 CPU MXFP4 MoE
              （D2H hidden/ids/routes → CPU 计算 → H2D）
  → main.wait_stream(B) ; output.add_(cached) ; output.add_(shared)
```

要点：`cudaLaunchHostFunc` 让 CPU 专家调用成为**整条主流的前置屏障**（回调返回前，
后续入队的 GPU 工作不会开始，`csrc/cpu/dsv41/cuda_host_moe.cpp:47-49`）。因此只有
「回调之前就已经入队到旁路流」的工作（GPU 专家缓存、shared experts）能藏在 CPU 计算窗口里；
下一层的 attention/归一化无法躲。这是该形态的固有气泡。

CPU 侧实现（`vllm/model_executor/layers/fused_moe/experts/cpu_mxfp4.py` +
`csrc/cpu/dsv41/{ggml_moe.cpp,compact_moe.h,compact_avx2.h}`）：
- 一个 C ABI（`dsv41_moe_create`）+ 两种执行器：`CompactMoE`（tokens ≤ 64 的 decode 默认）
  与 GGML graph（prefill 走 graph，`ggml_moe.cpp:316-334`）；
- 路由按 `(expert, position)` 排序后分组，up/gate 按 128 行分块、`schedule(dynamic,1)` 并行，
  down 每线程一组、累加用 FP32 乘 + FP64 加（`compact_moe.h:229-243`），与 GGML 路径 **bit-exact**
  （测试 `atol=0, rtol=0`）；
- 权重按 `GGML_TYPE_MXFP4_R8`（8 行交织、136 B/块的 E8M0+E2M1 布局）**一次成型**，
  还能反向 `export_weight` 把专家原样搬回 GPU 做 Marlin 打包；
- 线程数可运行时切换（prefill/decode 分开：`cpu_phase_threads`），worker 进程被
  `sched_setaffinity` 钉到物理核，`OMP_WAIT_POLICY=ACTIVE`/`GOMP_SPINCOUNT=1e7`。

### 3.2 动态 GPU 专家缓存 + 代价模型（该分支 decode 的核心）

- 每层一个 `GPUExpertCache`：4 个 Marlin 张量（w13/w2/weight/scale）、`expert_map`（384 长，
  −1 表示不驻留，交给 Marlin 跳过）、`membership`（把已驻留路由标成 `-2` 哨兵，让 CPU 侧跳过）。
- `select()` **保持已在槽位里的专家不动**，只把新专家填进空槽；集合不变则零开销。
- 准入是**代价受限**的（`cache_policy.TailCachePolicy`）：
  `benefit = Δ(route 频率) × horizon_steps × cpu_call_ms`，`cost` 取
  `transfer_ms`（已在 pinned LRU 里）或 `repack_transfer_ms`（需从 CPU 权重重新导出+打包）；
  仅当 `benefit > safety_factor(2.0) × cost` 且未超预算/交换次数时才换。
  SM80 上的实测常数：**transfer 3.2 ms / repack 6.4 ms**（`hybrid_runtime.py:263-282`），
  反推链路 ≈18.8 MiB / 3.2 ms ≈ **5.9 GB/s** —— 这个数字对 vllm170 的 PP/Engram 传输估算同样适用。
- 三个循环：prompt 尾部校准（`calibrate` → 按 count 选 top-capacity）、prefill 段自适应
  （`adapt`，每 chunk 一次，≥64 token）、decode 段反馈（`refresh_decode`，每 64 步、每步轮询一层，
  用 in-graph 的 `_record_decode_routes` triton 直方图统计**包含 GPU 命中**的完整路由分布）。
- 默认生产参数：`decode_max_swaps=2`、`decode_budget_ms=8.0`、`feedback_weight=0.9`、
  `expected_tokens_per_step=2.5`（开 DSpark）、host LRU 32 专家/层 ≈12 GiB。

**对 vllm170 的意义**：vllm170 全驻留、没有专家缓存，这套机制**不能直接迁移**。可迁移的是
其中两个「方法论」：① **用带时间预算的代价模型做缓存决策**（而不是纯 LRU）；
② **in-graph 精确直方图**统计真实访问分布——后者对 vllm170 的 Engram 主机读路径有潜在价值
（见 §3.5）。

### 3.3 内存规划：KV 预留 → 专家容量

`initialize_hybrid_cache`（`hybrid_runtime.py:156-310`）的顺序值得抄：
① 先用真实 shape dummy run 量出 transient 峰值（`(num_tokens,1)`、`(num_tokens,num_reqs)`、
`(decode_tokens,num_reqs)`）；
② 读回引擎**实际**分配的 KV 字节；
③ `reserve = GiB/2 + max(GiB, transient, cudagraph_memory_estimate)`；
④ `available = max(0, min(profiled, free − reserved_for_others − transient) − reserve)`；
⑤ all-gather 各 rank 的 available → 按 8 对齐、优先本地设备地均分给 20 层（`hybrid.py:100-116`）；
⑥ 层配额 <8 就置 0（该层全 CPU），OOM 无兜底（靠 clamp/skip 降级）。
**对 vllm170 的直接价值**：③ 里的 `cudagraph_memory_estimate` 与 ① 的「多 shape dummy run」
正是 vllm170 目前用 `--gpu-memory-utilization 0.98` 一刀切替代的东西；在 63.07 GiB 卡上，
把 graph workspace 与 transient 分开量、再给 KV 一个确定的数，比 0.98 更稳。

### 3.4 主机内存与加载

- `host_memory.check_host_headroom()`：`MemAvailable < 64 GiB + 需要量` 直接抛错，
  并且从 `/proc/self/cgroup` 向上遍历 `memory.max`，`used + need + 8 GiB > limit` 也抛错——
  **在分配之前**失败（`host_memory.py:23-49`）。
- `RegisteredMapping`：`mmap` 精确尺寸匿名映射 + 按 ≥64 MiB 页对齐分片**增量** `cudaHostRegister`；
  注释写明「DMA 拷贝不能跨两个注册段，专家 LRU 按第一维切片，所以每行必须落在一个段内」。
- Engram 加载：64 MiB 分块拷贝，每块前 `check_host_headroom`，并用
  `MADV_PAGEOUT` 回收已被消费的 checkpoint 文件页（`common/engram.py` 的
  `_engram_checkpoint_pageout`，注释强调「MADV_PAGEOUT 回收干净文件页且不丢私有修改，
  不要用 MADV_DONTNEED」）。
- **对 vllm170 的价值（这里 vllm170 有三个真实缺口，不是"锦上添花"）**：
  vllm170 的 Engram 已经是 94.42 GiB×2 主机驻留 + SSD 转换文件 + `posix_fadvise(DONTNEED)`
  分块预载（commit `8f0516c36b`），**主体思路已具备且不弱**；但：
  ① **资源门禁缺失**：`engram.py` 里的 30%-物理内存 / 50%-设备两条检查都在
     `if not ssd_offload_path` 之内，而两个 Engram rank 的 2×94.42 GiB 配置恰恰走
     `ssd_offload_path` —— 也就是说**最需要护栏的模式没有任何护栏**；运行栈现在处于
     ~225 GiB used / 22 GiB available + 2 GiB swap，对一个 94 GiB 匿名表是实打实的风险。
     `check_host_headroom` 的 `MemAvailable` + cgroup `memory.max` 分块校验正好补这个洞。
  ② **冷启动不能自举**：full-RAM 模式下 `ssd_num_embeddings == 0` ⇒
     `param.engram_ssd_tensor is None` ⇒ SSD 写分支不执行，同时 `engram_ssd_ready` 被强制为
     `_ssd_ready and not _full_ram`。冷机（`.u8` 文件不存在）时预载被跳过、也不会生成文件，
     退化成**把 94 GiB checkpoint 直接读进 pageable RAM** —— 正是该 commit 想避免的峰值，
     而且**静默**（只少一行 " (preloaded from SSD)" 日志）。SSD 文件只是升级路径，不是冷启动路径。
  ③ **完整性判据过弱**：`_ssd_file_has_data` 只做「大小相等 + 三个 4096B 窗口各含一个非零字节」
     的 3 点采样；半写文件只要这三点非零就被当作完整，随后 `_ram_preloaded` 抑制 checkpoint 加载
     ⇒ **静默用错表**。该分支的做法（加载时按 64 MiB 分块、每块校验 headroom、回收已消费页）
     天然要求「边读边校验」，可以直接借鉴。
  半块结论：**P1-1（护栏）与 P1-4（自举/完整性）不是可选优化，而是当前配置下的正确性/稳定性缺口。**

### 3.5 明确不可迁移 / 需改造的部分

| 机制 | 结论 |
| --- | --- |
| CPU 原生 MXFP4 专家（IK/GGML） | vllm170 全驻留前提相反；要迁移等于换部署形态（见 §6） |
| CUDA host callback 桥 | 只在「CPU 侧有活」时有意义 |
| 动态专家缓存 + 反馈直方图 | 无专家缓存则无用；直方图思路可转用于 Engram 热点行 |
| `MPS` 跨进程 GPU 访问、CPU 亲和/自旋调优 | 形态不同，不适用 |
| 专家缓存指标（log-only，无 Prometheus） | 无对应对象 |

---

## 4. DSpark 环节对照

### 4.1 该分支做了什么

1. **`dspark_num_query_tokens`（Q）与 `num_speculative_tokens`（K）解耦**
   （`vllm/config/speculative.py:442-452`）：Q 是「backbone 查询块长度（含预测锚点）」，
   可以大于 K，从而**在 proposal 只取前缀的同时保持 checkpoint 训练时的查询块宽度**；
   要求 greedy DSpark + `sample_from_anchor=True` + 定长验证。
   默认值：`spec.setdefault("num_speculative_tokens", 3)` +
   `spec.setdefault("dspark_num_query_tokens", max(5, int(spec["num_speculative_tokens"])))`
   （`vllm/models/deepseek_v4_1/hybrid.py:81-83`），即 K=3→Q=5、K=5→Q=5、K=7→Q=7。
   *注*：checkpoint 的 `dspark_block_size=5`，而 vllm170 里 `num_query_per_req == num_speculative_steps`
   （`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:111-117`），即两者被绑死。
2. **配套的 draft attention 尺寸覆盖**：`dspark_backbone_vllm_config()` 复制一份 VllmConfig 并把
   `speculative_config.num_speculative_tokens` 改成 Q，交给 `load_dspark_model`
   （`dspark/utils.py:35-52`）——否则 Q≠K 时 draft attention 的 metadata/slot 会按 K 分配而越界。
   同时 `VllmConfig` 的 `num_speculative_tokens` 消耗处也返回 Q（`config/vllm.py:631-635`）。
3. **允许 V4.1 的 K 超过 checkpoint block size**（commit `648218d25`）：
   把「`num_speculative_tokens` 必须被 `n_predict` 整除」这条校验对
   `model_type == "deepseek_v41"` 的 DSpark 豁免，并把默认 `n_predict` 保持为
   `dspark_block_size` 而不是 MTP 层数。
4. **DSpark + PP 的词嵌入**：`has_own_embed_tokens = not get_pp_group().is_first_rank`，
   非首段 rank 从同一 checkpoint 加载一份**私有** embed 权重（`nvidia/dspark.py:298-320, 409-420, 553-560`）；
   并要求 aux hidden states 全部落在末段（`nvidia/model.py:616-630`）。
5. **CED → draft context**：propose 之前把 CED 重放出的 suffix 的 aux hidden 组合成 draft context，
   直接 `precompute_and_store_context_kv`（`dspark/speculator.py:172-190`）。
6. greedy 路径的 **fused Markov argmax + vocab shard**（`spec_decode/dspark/markov_argmax.py`、
   `use_local_argmax_reduction`）：把 draft logits 的通信从 O(vocab) 降到 O(2·tp)。

### 4.2 vllm170 当前状态与实测

- `num_query_per_req` 跟随 K（见上）；`dspark_pipeline_stage_ranks=[1,2,3]` 把 mtp.0/1/2 分到 PP1/2/3、
  heads 在 PP4（`benchmarks/deepseek_v4_1/run_cmp170hx_pp5.sh:100-112`，
  `docs/design/vllm170-dsv41-dspark-pp5-memory-2026-09-14.md`）——
  **这是该分支没有的能力**（它把整个 drafter 放末段）。
- **实测宽度数据（权威版：`docs/design/vllm170-dsv4-sm80-dspark-acceptance-p0-2026-08-17.md` §4/§6，
  PP3 `15,15,13`、32K、greedy、graph）**：

| K | pooled acceptance | 平均接受长度 | ms/round | decode tok/s |
| --- | --- | --- | --- | --- |
| 3 | 68.25% | 3.051 | 30.219 | 100.28 |
| 4 | 58.84% | 3.388 | 30.095 | 110.08 |
| **5** | **49.29%** | **3.465** | **30.949** | **110.92** |
| 7 | 36.53% | 3.557 | 32.317 | 108.58 |

> ⚠️ **不要横向比较 acceptance%**：该口径满足
> `平均接受长度 = 1 + K × acceptance_rate`（分母含全部 K 个草稿位），
> 所以 K 越小 acceptance% 越高是**算术必然**，不代表起草器更好。
> 例如 K=3 的 68.25% 看着最高，但平均接受长度 3.051 **低于** K=5 的 3.465。
> 跨 K 比较只能看「平均接受长度 + ms/round + tok/s」这三列。

  **结论很重要，且和直觉相反：32K 上 q5 才是吞吐最优点。** q7 相对 q5 平均接受长度只 +2.60%，
  但 ms/round +4.23% → decode **−2.15%**；q3 的 round 快 2.36% 但接受长度低 13.35% → decode
  **−9.59%**。所以启动脚本默认已从 q7 改成 q5。
  另一头的证据（`profiles/deepseek_v4/maxseq1-1m-ab/20260816T012158Z/summary.md`，PP3 + `max_num_seqs=1`
  + DSpark-7 对 target-only）：**1K 时 DSpark 2.63×（159.3 vs 60.6 tok/s，acceptance 70.13%）**、
  32K 1.45×、128K 1.88×；prefill 吞吐几乎不变（+0.59%/−0.04%/+0.95%）。
  即 **K=7 的价值集中在短上下文**（1K 70% 接受率），到 32K 以上主要靠 target 侧收益。
  概率路径（post-RNG 9 seed，32K）同向：general q5 94.39 vs q7 94.50 tok/s（+0.12%），
  agentic q5 91.69 vs q7 76.84（**−16.19%**）。
  **口径警告**：以上全部是 **DSv4 世代**（DeepSeek-V4-Flash-0731，E=256，`K=4096`，Marlin MoE）
  + **PP3** + `max_num_seqs=1` 的数字；V4.1/PP5 是 E=384、hidden 5120、`mxfp4_int8_sm80`、
  draft `topk=3`。仓库自己的文档（`vllm170-dsv41-pp5-decode-profile-2026-09-12.md` §计量方法）
  明确禁止直接搬运，这里只用于**方向性判断**。
- **当前 K=7 会被配置校验直接拒绝**（本机实测）：

```
$ .venv/bin/python -c "SpeculativeConfig(model=..., method='dspark', num_speculative_tokens=7, ...)"
K=7: REJECTED -> ValidationError: num_speculative_tokens:7 must be divisible by n_predict=5
```

  原因链：`7039dab0f3` 起 `n_predict = dspark_block_size = 5`
  （`vllm/config/speculative.py:1484-1488`）→ 整除校验 `:1530-1536`。
  August 那批 q3/q4 扫描是在**临时把这条硬错误降级成 warning** 之后跑出来的，扫描完又恢复了
  （`acceptance-p0` §6，与上游 issue #50012 同源）——也就是说 **q3/q4/q7 的扫描能力目前是「一次性手工
  打开」的，代码里没有留下开关**。这是本次对照里最确定、最便宜的一个收益点。

### 4.2b 顺带发现：PP5 分布式 DSpark 目前是负收益（vllm170 内部问题）

对照该分支时暴露出一件比「吸不吸收」更紧急的事：**vllm170 的两套 DSpark 放置，效果完全相反**。

| 形态 | 32K/64 输出 | acceptance | 相对 target-only |
| --- | --- | --- | --- |
| PP3 `15,15,13`，三层 draft 全部同段（Aug） | — | q5 49.29% | **1.45–2.63× 加速** |
| PP5 `8,8,8,8,8`，`dspark_pipeline_stage_ranks=[1,2,3]`（Sep-15） | **11.281 tok/s** | **17.71%**（平均接受 1.886） | 同轮同 shape target-only **25.778 tok/s** |

出处：`docs/design/vllm170-dsv41-dspark-pp5-memory-2026-09-14.md` §2026-09-15（文档自己写明
「不得表述为 DSpark 已达到生产收益门禁」）。

**基线要取对**：同一份 PP5 decode profile 里还有两个目标侧数字 ——
未插桩 FULL 单请求 32K/64 为 **19.63–20.03 tok/s（≈50 ms/token）**，
而重启后的生产基线是 **35.657 tok/s（28.046 ms/token）**。
11.281 tok/s 相对同轮 25.778 是 **2.3× 慢**，相对生产基线是 **3.2× 慢**。
该文档把任何候选的准入线定为 **≥0.561 ms/token（生产路径的 2%）**——评估下面所有收益点都用这个门槛。

同时该分支的文档记录：**PP5 的两条单请求 SWA 快路径都显式排除了非因果/DSpark**，
而它们的收益（fused causal single-token SWA +3.29%、paged-direct SWA −27.3% 每次调用、
32K/64 +10.28%、累计 **+59.7%**）**全部是在 DSpark 关闭时测的**
（`vllm170-dsv41-pp5-decode-profile-2026-09-12.md:280,306`）——
即 PP5 上的 DSpark 拿不到这套收益。

**更关键的三个事实（2026-09-17 实机核对）**：

1. **PP5 启动脚本把 draft 设成 eager**（`DSV41_DSPARK_ENFORCE_EAGER=1`，`run_cmp170hx_pp5.sh:71`），
   也就是说 **PP5 上「draft 走 CUDA graph」这条路从未跑通**，只有 eager 版本有数据（就是那 11.281 tok/s）。
2. **今天有人用 `DSV41_DSPARK_ENFORCE_EAGER=0` 尝试打开它，进程崩了**：
   `AssertionError: cudagraph_manager should only be None during profile run, where need_eager must be True`
   @ `vllm/v1/worker/gpu/dp_utils.py:186`，调用栈
   `dspark/speculator.py:567 participate_distributed` ← `model_runner._dummy_run`。
   那正是**未提交的那份工作区改动**新加的代码路径。所以它不是"写了但没开"，
   而是一个**当前失败中的未完成项**。
3. **PP5 的 target 其实也没跑 FULL 图**：日志每次启动都会打印
   `CUDAGraphMode.FULL is not supported with DeepseekV41ROCMAiterSparseSWABackend backend
   (support: AttentionCGSupport.UNIFORM_BATCH); setting cudagraph_mode=FULL_DECODE_ONLY`。
   推论：分布式 draft 图代码自己那条 `supports_full` 检查（要求 `>= UNIFORM_BATCH`）在同一个
   SWA 后端上也会回落 —— **即使把上面那个断言修好，draft 图大概率仍然开不起来**，
   除非先让该 SWA 后端支持 FULL。

**对照价值**：该分支**没有**分布式 draft —— 它把整只 drafter 放末段（`has_own_embed_tokens =
not is_first_rank`、`_set_aux_hidden_state_layers` 要求 aux 全在末段），PP3 分层下天然成立。
所以当 PP5 分布式 DSpark 的 acceptance 只有 PP3 同配置的 1/3 时，**该分支的「drafter 全放末段」
是一个现成的对照组**（代价：末段多 ≈7.4 GiB 权重 + 每 chunk 20 MiB 的 `main_x` 传输，
`dspark-pp5-memory` §Draft 权重构成）。建议：先定位 PP5 acceptance 塌陷是 relay/槽位还是
metadata 路径问题，再决定是修分布式还是改回末段共置。

### 4.3 可吸收项

1. **P0：V4.1 的整除豁免 + Q/K 解耦**（低成本、先把能力打开，再用你们现成的 harness 定值）。
   收益的正确表述：
   ① **恢复可扫描性** —— 现在 q3/q4/q7 只能靠临时改校验跑（`acceptance-p0` §6 记录了这一操作），
   把 V4.1 的豁免做成正式开关后，宽度扫描才能进 CI/常规回归；
   ② **K=7 在短上下文是明确收益**（1K：2.63×、acceptance 70.13%），在 32K 上按现有实测
   **不占优**（q5 110.92 vs q7 108.58 tok/s），所以它是「按上下文分档」的旋钮而不是全局改动；
   ③ **Q>K 是一个尚未验证的假设**：让 draft backbone 跑训练宽度 Q=5、而只提案 K=3/4，
   理论上既保住非因果查询块的训练条件、又把 target 侧 verify 从 q6 降到 q4
   （32K→64K target 重放 29.811→32.802 ms，而 draft proposal 只有 ≈10.3 ms 且几乎不随上下文增长）。
   但你们已有数据是反向线索：q5 的前三位接受率（0.61/0.58/0.58）**低于** q3（0.68/0.64/0.61），
   所以 Q=5/K=3 未必胜过 Q=3/K=3。**必须用 `VLLM_DEEPSEEK_V4_DSPARK_Q1_AUDIT` /
   `dspark-width-audit` 实测，不能只看文档措辞。**
   **Q 必须精确地传播到这 4 处**（漏一处就会静默越界）：
   ① `num_query_per_req`（draft 查询宽度）；② `_anchor_idx = arange(max_num_reqs) * Q`
   （图内读每请求第 0 列的唯一 CG-safe 方式）；③ `num_lookahead_tokens = Q`（KV 预留，
   `vllm/config/vllm.py:620-650`）；④ `max_num_new_slots_for_drafting = Q - 1`（调度预算，
   `speculative.py:1866-1883`）。vllm170 现在是 ③=`K`、④=`K-1`（`speculative.py:1928-1958` 表格中
   DSpark = K−1），②③④ 都要一起改。
   **打开 K=7 的连带成本**（改 K 属启动期改动，不是运行时旋钮）：target 与 draft 两套 CUDA graph
   全部重捕获（捕获尺寸按 `max_num_seqs×(K+1)` 增长）；`input_batch` 的
   `BLOCK_SIZE = next_pow2(K + num_new_sampled)` 与 fused Markov 的 `n_spec=K` 会重新特化；
   非因果 SWA 索引宽度 `align64(window + K)` 变化（`compressor_utils.py:19-28`）触发 JIT 重预热；
   scheduler 每请求扣 `Q-1` 个槽位；PP 每步多传 2 个 int64/请求。
   另：**不要**对 DSpark 打开 `num_speculative_tokens_per_batch_size`（会产生非均匀
   `decode_query_len`，与 DSpark 的均匀查询假设冲突）。
   单请求长上下文还要注意：Triton GEMV 快路径要求 `M = K+1 ≤ 8`（`gemv_triton.py:20-33`），
   K=7 恰好等于 8 是上限；8 并发时 `M=8×8` 早已落在 cuBLAS，不受影响。
2. **P1：CED → draft context 的「先发布再 propose」**。vllm170 已有 non-final chunk 的
   context-only 快路径与 KV-only insert（`VLLM_DEEPSEEK_V4_DSPARK_KV_ONLY_INSERT`），
   语义接近；若将来实现 CED，这一条是天然配套。
3. **P3：fused Markov argmax / `use_local_argmax_reduction`**。前者只在 greedy 起草时省掉
   「物化 [T,vocab] bias」；vllm170 生产 PP5 脚本默认就是 greedy（`DSV41_DSPARK_DRAFT_SAMPLE_METHOD=greedy`），
   draft 图整体只有 ≈2.7 ms（`vllm170-dsv4-sm80-dspark-profile-2026-08-15.md` 结论第 5 条），收益有限。
   **后者（vocab shard + local argmax）在 vllm170 上不要开**：它只在 TP>1 有意义 ——
   实测 Markov GEMM 被 8 路切分后带宽从 1457 GB/s 掉到 807 GB/s
   （`benchmarks/kernels/benchmark_dsv4_sm80.py:2785-2790`），而在 TP=1 下只有一份分片，
   reduction 退化成 no-op：**既拿不到通信收益，也不会造成那种带宽损失**。
   该分支强行打开它只是因为混合形态固定 TP1、想省通信，对 vllm170 无意义。4. **已具备**：PP draft 私有 embed（`nvidia/dspark.py:556-620` 的 `_TARGET_EMBED_KEYS` 分支）。

---

## 5. 收益点清单（可执行版）

### P0-1　DSpark：解禁 K>block_size 并解耦查询宽度

- **机制**：见 §4.3-1。三处改动：`n_predict` 校验对 V4.1 DSpark 豁免；
  新增 `dspark_num_query_tokens`；`dspark_backbone_vllm_config()` 让 draft 侧按 Q 建 config。
  另需把 `num_query_per_req` 从 `num_speculative_steps` 改为「Q（若给出）」，
  并让 draft KV slot 数按 Q 而非 K 计算（该分支的 `get_num_extra_drafting_slots` 也一并返回 Q，`speculative.py:1866-1883`）。
- **证据**：本机实测 K=7 被拒；`profiles/deepseek_v4/dspark-width-audit`、`dspark-q1-audit` 已有 q3/q4/q5/q7 对比口径。
- **预期**：长上下文可把 verify 宽度降到 3–4，target q(K+1) 的 attention 成本近似线性下降
  （该分支自身文档把 target 侧列为 DSpark 第一成本项）；短上下文恢复 K=7 → 平均接受长度
  从 3.85 → 4.96（1K 实测，仅供参考）。
- **工作量/风险**：小；风险低（默认不变，纯新增开关）。前置：`sample_from_anchor=True` + 定长验证
  （vllm170 PP5 脚本已 `enable_adaptive_verification: false`，符合）。
- **移植时要特别小心的三点**（都是该分支已经踩过或留下的口子）：
  ① **Q≠K 时「KV 预留」与「图尺寸」由两个不同的数字驱动** —— 该分支刻意让
     `num_lookahead_tokens = Q` 而 `uniform_decode_query_len` 仍是 `1+K`
     （`config/vllm.py:653-671` 有专门注释说明这是两份不同契约）。这是 Q≠K 之后
     最可能出现尺寸类 bug 的地方，移植时应把两者写进同一个断言/测试。
  ② 该分支的**两个半改动各只有一个间接保护**，移植时请补齐：没有测试断言
     `hf_config.n_predict == dspark_block_size`（所以"保持 n_predict 默认值"这半可能静默回归）；
     `tests/test_config.py:1130-1151` 是在构造**之后**才 patch `method`，绕过了 `__post_init__`，
     因此 `max_num_new_slots_for_drafting = Q-1` 这一支也没有直接覆盖。
  ③ `dspark_num_query_tokens` **没有 `ge=1` 下界**，正确性完全靠相对的 `Q >= K` 校验
     （`speculative.py:1809-1812`）；显式加下界更稳。

### P0-2　`kv_cache_tokens`：物理 KV 的 token 预算

- **机制**：见 §2.5。要点是**显式 block 公式 + 引擎侧预留 + 明确的“差多少 GiB”报错**，
  以及对 sliding-window/chunked-local 组的更紧界。
- **证据**：vllm170 文档明确记录了「字节预算不能宣称 1M，必须靠启动日志反推」这一痛点。
- **预期**：容量可声明、可断言；让 `--gpu-memory-utilization` 不再是唯一旋钮。
- **工作量/风险**：中（≈200 行，主要在 `kv_cache_utils`/`config/cache.py`/`arg_utils`）；
  注意与 `--kv-cache-memory-bytes` 互斥、与 PP 副本状态的核算。

### P1-1　主机内存护栏（Engram 加载）

- **机制**：`check_host_headroom(additional)`：`MemAvailable ≥ 64 GiB + need`，
  且沿 cgroup 逐级校验 `memory.current + need + 8 GiB ≤ memory.max`，**在分配前抛错**；
  预载按 64–256 MiB 分块，每块前校验一次。
- **证据**：vllm170 Engram 每 rank ≈94.42 GiB、两 rank 合计 ≈189 GiB，主机 503 GiB；
  仅有的「>30% 物理内存 / 50% 设备」两条检查都在 `if not ssd_offload_path` 之内
  （`engram.py:869-886`），也就是**生产用的 full-RAM 模式完全没有门禁**，
  只靠启动脚本一行注释提醒；当前运行栈 ~225 GiB used / 22 GiB available + 2 GiB swap。
- **预期**：把 OOM-kill 变成可读错误；对反复加载/卸载的调试循环尤其有价值。
- **工作量/风险**：小（≈60 行），低。

### P1-2　每步回收所有 running 请求的 SWA 块

- **机制**：`schedule()` 开头、做任何分配决策**之前**，对全部 running 请求
  `recycle_sliding_windows(request)`（`processed = num_computed − in_flight`，只动
  `SlidingWindowManager/ChunkedLocalAttentionManager`）。
  vllm170 目前只在 `allocate_slots()` 里逐请求回收（`kv_cache_manager.py:503`），
  先被处理的请求无法受益于后面请求稍后才释放的窗口。
- **证据**：vllm170 KV 池 1.3e9 B（≈1,078,799 token）、`max-num-seqs 8`、1M 并发系数 1.0288。
- **预期**：紧 KV 下减少「本可继续却因窗口未回收而被抢占」的情况；量级取决于 SWA 层占比
  （每个压缩层都有自己的 128-token 窗口，共 40 层，块粒度 128）。
- **工作量/风险**：小；**注意**该分支把这一步 gate 在 `kv_cache_tokens is not None` 上，
  而预算计算对「隐式混合路径」同样生效 —— 移植时应无条件执行（或与 `connector is None` 一起判断），
  否则会复制它这个不一致。

### P1-3　PP KV 传「已打包行」

- **机制/证据/预期**：见 §2.3。载荷 11.5→2.6 KB/token，少 2 个 kernel/边界/chunk，中继不再重做 RoPE。
- **工作量/风险**：中；需处理候选块、KV 预算核算、常驻缓冲以保持图地址稳定。

### P1-4　Engram full-RAM 的冷启动自举与文件完整性

- **机制**：① 冷启动时若转换文件（`.u8`）不存在，应有明确的「先生成转换文件」路径，或退化为
  「分块 + 回收 page cache」的 checkpoint 直读（即该分支的 `MADV_PAGEOUT` 分块路径），
  而不是静默把 94 GiB 读进 pageable RAM；② 把 `_ssd_file_has_data` 的 3 点采样升级为
  带完成标记/校验和的判据，或至少在采样失败时**拒绝**抑制 checkpoint 加载。
- **证据**（§3.4 三条缺口）：`_full_ram` 下 `ssd_num_embeddings == 0` ⇒ 不会有 SSD 写分支，
  同时 `engram_ssd_ready = _ssd_ready and not _full_ram` ⇒ 冷机既不预载也不生成；
  3 点采样（大小相等 + 三个 4096B 窗口非全零）可能把半写文件判为完整。
- **预期**：消除「静默慢 10 倍加载」与「静默用错表」两类事故。
- **工作量/风险**：小–中；低。

### P2 级

- **logits 容量取 2 的幂**（§2.4-1，10 行）。
- **`PREFILL_CHUNK_SIZE` 按并发收口**（§2.4-2，1 行，需核对 vllm170 自研分块的语义）。
- **共享专家旁路流**：把 shared experts 放到独立 `torch.cuda.Stream` 与 routed MoE 并行
  （`nvidia/model.py:161-171`）。vllm170 decode 里 FFN 占 41.5%（`layer.ffn ≈147.09 ms/round`），
  shared expert 只是其中一部分，且 FULL graph 下要处理 fork/join —— **建议作为一个便宜的 A/B 实验**，
  预期 0–3%，失败就丢。
- **PP KV layout 交集**：当各 stage 的后端支持集合不一致时（vllm170 的分布式 DSpark 让部分 rank
  跑 draft attention），现在是 `assert all(names == supported_layouts[0])`
  （`vllm/v1/attention/backends/utils.py:266-269`）会直接失败；该分支改成交集 + PP 才放宽
  （`utils.py:253-280`）。属于健壮性兜底。

### P2/CED（条件性）

见 §2.2 的收益推算：**在 8,8,8,8,8 全驻留下不要先做**。只有在下面任一条件成立时才值得投入：
① 接受把解码段专家卸载到主机（换形态）；② 重新分层能让编码段每卡层数 <8；
③ 实测 PP 传输占到 chunk 时间的 15% 以上。

---

## 6. 战略项：要不要吸收「主机专家 + 动态专家缓存」这套形态

这是该分支的立身之本，也是它与 vllm170 的根本分歧：

- **它的取舍**：把 20 层 ×384 专家的 ≈134.5 GiB 放到主机，用原生 AVX2 内核算，GPU 只保留
  一份按代价准入的热点缓存；换来的是**只用 3 张卡**就能跑 1M 上下文，且 PP3 分层可以把
  层 15–39 全部塞进末段（配合 CED，末段在中间 chunk 只做一层投影）。
- **vllm170 的取舍**：全驻留 + PP5，单请求时延低（5 段串行）、无需 512 GiB 主机内存和本地编译，
  但 5 张卡只服务一条流水，且每卡 59.2/63.07 GiB 几乎无余量（`--gpu-memory-utilization 0.98`）。
- **一个容易被忽略的算术**：如果机器上有 5–6 张卡，该分支的形态是「3 卡一条流水」，
  理论上可以起**两份独立 PP3 实例**（聚合吞吐翻倍，代价是单请求时延变差、以及两份额度共享
 主机 CPU 与 PCIe）。vllm170 的 PP5 形态没有任何这种冗余：要么 5 卡跑一条，要么少卡跑不了
  （8 层/卡是权重决定的硬约束）。**如果业务是「多并发中等上下文」而不是「单条 1M」，
  这个形态差异可能比任何单点优化都值钱**——前提仍是先量化。
- **判断依据缺失**：该分支**没有发布任何 3 卡端到端吞吐**（README 只有内存数字 344 GiB/512 GiB/188–189 GiB，
  且 `docs/validation/` 被 gitignore、原始数据在外部机器上）。所以「效率比较高」目前是**未量化的**。
- **建议路径（不盲目移植）**：
  1. 先把该分支的**微基准**搬到 170HX 机器上跑：
     `benchmarks/kernels/cpu/benchmark_dsv41_cpu_moe.py`（CPU 专家算力与线程标定）、
     `benchmark_dsv41_cache_transfer.py`（3.2/6.4 ms 是否复现）、
     `benchmark_dsv41_hybrid.py`（单请求 prefill/decode 吞吐）。
     这些脚本不依赖它的服务形态，能直接给出「CPU 侧一层 MoE 要多少 ms / 每步要传多少 MiB」。
  2. 用它自己的公式做纸面推演：每步每层 CPU 调用 `cpu_call_ms≈0.3 ms`（开 DSpark）→
     20 层 × 0.3 = **6 ms/step 的 CPU 串行预算**，必须完全藏在 GPU 侧工作背后；
     对照 vllm170 现在的 decode ≈51 ms/token（19.55 tok/s），理论上有空间，但 DSpark 下
     每步的 GPU 工作也会被打断在 host callback 上（§3.1）。
  3. 只有当 ① 复现出可接受的 CPU 时间、② 你确实需要「3 卡跑 1M」或「5 卡跑两份实例」时，
     再考虑整体移植（那是一个独立项目，不是补丁）。
- **若真要走这条路，先记下两个已知边界**（都是该分支当前实现里的事实，不需要硬件就能复现）：
  ① **≤16 token 的 prefill 尾块会静默走 decode 异步路径** —— 门槛来自 decode 图形状
  （`async_tokens = max(16, min(128, max_num_seqs × uniform_decode_query_len))` = 16），
  而排除条件只对 `shape[0] > 16` 生效，所以「prompt 长度 = chunk*k + r（1≤r≤16）」的尾块
  用的是 host-callback 路径而非 128 行分块路径（`cpu_moe.py:334-338`）——
  可复现，值得进正确性扫描；
  ② CPU 专家单次调用有 `max_num_batched_tokens` 硬上限（`cpu_mxfp4.py:536-537`）。

---

## 7. 建议的验证顺序（都能用现成 harness）

> **统一验收门槛**：PP5 decode profile 把任何候选的准入线定为
> **≥0.561 ms/token（生产路径的 2%）**，且必须在**未插桩的 FULL 轮**里复测
> （插桩本身会把 32K/64 从 ≈19.8 打到 11.1，再到 1.6 tok/s）。下面每一项都用这个口径判定。

0. **先解决 PP5 分布式 DSpark 的 acceptance 塌陷与 draft 图**（§4.2b）：同样 K=5，PP3 同置 49.29%，
   PP5 分布式只有 17.71%（11.281 tok/s，对同轮 target-only 25.778 慢 2.3×）。要分清四件事：
   ① relay/槽位映射（`package["idx_mapping"]`、generation 过滤）；
   ② 非因果/DSpark 被排除在 PP5 两条 SWA 快路径之外（那两条合计 +59.7%，但都是在 DSpark 关闭时测的）；
   ③ 同步 4 跳 ring 的固定开销；
   ④ **draft 只能跑 eager**——用 `=0` 打开图会崩在 `dp_utils.py:186`，
   而且该 SWA 后端只支持 `UNIFORM_BATCH`，target 自己都被降级成 `FULL_DECODE_ONLY`，
   draft 图大概率也上不去。在定论前不要按 DSpark 收益做容量规划。
1. **DSpark 宽度实验**（P0-1 之后）：用 `profiles/deepseek_v4/dspark-width-audit` 的口径在 **V4.1/PP5** 上
   重跑 K=3/4/5/7 与 Q∈{5,K}（注意 PP3/V4-Flash 的数字不可外推），输出 acceptance、平均接受长度、
   decode tok/s、PP 各段 ms；目标一是**消掉 PP5 与 PP3 的 acceptance 差**，
   二是回答「Q=5/K=3 是否比 K=5 更快」。
2. **KV 预算对照**（P0-2）：同一 prompt 集下比较 `--kv-cache-memory-bytes 1.3e9` 与
   `--kv-cache-tokens 1048576` 的 token 容量、并发系数与抢占次数。
   （注意 1.3 GB 只够**一条** 1M 请求，不是八条。）
3. **PP 载荷测量**（决定 P1-3 值不值）：在现网 PP5 上用 phase profiler 量
   `pp.send/recv` 占 chunk 时间的比例；若 <10%，P1-3 降级。
4. **CED 可行性测量**（决定 P2-CED）：量「编码段 8 层」与「解码段 8 层」各自的 chunk 时间，
   以及 PP 两跳的实际耗时；若解码段不是关键路径也省不下传输，就不做。
5. **共享专家旁路流 A/B**（P2）：FULL graph 下 fork/join 是否可行，端到端 ±。
6. **Engram 门禁与冷启动**（P1-1/P1-4）：在 `DSV41_ENGRAM_CPU_CACHE_GB=96` 下删掉
   `.u8` 转换文件后启动一次，确认是否退化成 94 GiB checkpoint 直读；并核对
   `_ssd_file_has_data` 的 3 点采样在真实文件上是否够用。
7. **量化该分支机制本身**（若决定吸收 CED / PP-KV / 索引构造，用它自带的 harness 在自己机器上出数）：
   `benchmarks/kernels/cpu/benchmark_dsv41_hybrid.py` 已内置 A/B 开关 ——
   `--ced`、`--pp-kv-transfer`、`--prefill-index-sweep {torch,fused}`、`--prefill-chunk-sweep`、
   `--replay-cache-sweep {0,1}`、`--record-expert-routes`、`--profile-prefill`、`--threads`；
   输出键把数字映射回机制：`ttft_seconds`/`prefill_tokens_per_second`、`ced_before`/`ced_after`、
   `prefill_index_impl`、`replay_cache_enabled`、`prefill_stage_intervals`、`expert_route_histograms`。
   它自带三重门禁（输出长度、`metrics.is_corrupted`、logprob 有限性）与 CED 一致性门
   （token 相同 + `"42"` 存在 + max|Δlogprob| < 0.05）。
   ⚠️ 该 harness 默认 `--graph-mode PIECEWISE`（而混合默认是 `FULL_DECODE_ONLY`）——
   这正是 commit 需要额外加 `skip_compiled` 门的原因，跑之前先统一图模式。

---

## 8. 反向输出（vllm170 → 该分支）

该分支若继续演进，以下 vllm170 已有而它没有的东西值得它吸收（**但第 1 条现在要先打问号**）：

1. ~~`dspark_pipeline_stage_ranks` 分布式 DSpark~~ **撤回此建议**：该能力在 vllm170 自己这边
   目前是负收益（PP5 11.281 tok/s / acceptance 17.71%，慢于 target-only 25.778 tok/s，
   见 §4.2b）。**在 vllm170 定位清楚之前，不应把它作为「可输出的经验」推荐给该分支**；
   该分支把整只 drafter 放末段反而是一个更稳的对照点。
2. **indexer top-k 确定性**（集合与顺序）与 **router GEMM / GEMM 几何固定**
   ——它那边同样是 SM80 + 同样的近似平局问题。
3. **稀疏 prefill 的流水化 tiling + `TEMP_MB` 预算**、**caller-owned 索引工作区**
   （它现在的 `combine_topk_swa_indices` 每次新建 `[T, width]`，只有一次同步收口）。
4. **Engram 转换缓存文件 + `posix_fadvise(DONTNEED)`**：虽然 vllm170 的这条路径有
   §3.4 的三个缺口（无门禁、冷启动不自举、完整性判据弱），但「转换文件 + 冷启动可复用」
   这一层比该分支的 `MADV_PAGEOUT` 更适合反复加载；建议它补上自举与校验后再互抄。
   **若真要互抄，这套设计有 3 条硬不变量必须同时满足**（`tests/kernels/test_engram.py` 已钉住）：
   full-RAM 模式下表**不得 pinned**（`:690`）、**权重加载器必须是 no-op**（`:691-693`）、
   以及 staging 的 pinned/CUDA 归属（`:750-751`）——该分支现在用的是
   `empty_registered`（匿名 mmap + `cudaHostRegister`），与第 1 条直接冲突，
   属于「形态不同」而非「改改就行」。
5. **阶段化 phase profiler**（`VLLM_DEEPSEEK_V4_PHASE_PROFILE_*`）——它只有 CUDA Event 级诊断。

---

## 9. 对照时必须注意的口径差异（不要抹平）

| # | 差异 | 影响 |
| --- | --- | --- |
| 1 | 该分支的 MoE 是 CPU 原生 AVX2 MXFP4；vllm170 生产是 **`mxfp4_int8_sm80`**，Marlin 只做 dense MXFP8（`run_cmp170hx_pp5.sh:67`、`pp5-decode-profile` §MoE/MXFP4） | 两边「专家核」根本不同，任何 tile/shape 结论都不能互推 |
| 2 | 代际不同：DSv4（V4-Flash，E=256，`K=4096`，Marlin MoE）vs V4.1（E=384，hidden 5120，draft topk=3） | 仓库自己的文档禁止搬运；本报告已就地标注哪些数字属 DSv4 世代 |
| 2b | **同一件事在两代的结论是相反的**：DSv4/PP3 **否决**了 `mxfp4_int8_sm80`（「实测更慢且为近似路径」）而保留 Marlin MXFP4；V4.1/PP5 反过来把 `mxfp4_int8_sm80` 作为生产后端、MoE 放弃 Marlin；稀疏 prefill 的最佳 `block_k` 也从 DSv4 的 32 变成 V4.1 的 64（`rocm_aiter_mla_sparse.py:2878`） | 任何 tile/后端结论都不能跨代引用；该分支用的是**第三种**（CPU AVX2 MXFP4），三者互不可推 |
| 3 | `1.3 GB` KV 是「一条 1M 请求」的预算，不是八条 | 讨论并发时必须区分 |
| 4 | 启动脚本写 `cudagraph_mode:"FULL"`，实际每次启动被降级：`CUDAGraphMode.FULL is not supported with DeepseekV41ROCMAiterSparseSWABackend (support: UNIFORM_BATCH); setting cudagraph_mode=FULL_DECODE_ONLY` | 讨论图收益不要按 FULL 记账；同一 flag 在 PP3/PP5 生效模式不同；也是 draft 图开不起来的根因之一 |
| 5 | 四个开关"名存实亡"：`VLLM_PP_REUSE_RECV_BUFFER` / `VLLM_DEEPSEEK_V4_DSPARK_KV_ONLY_INSERT` 的库默认值与文档相反（现为 True / 1，所以启动脚本里的显式导出已冗余）；`VLLM_DEEPSEEK_V4_PERSIST_SINGLE_REQ_VALIDITY` 仍在启动脚本里导出但快路径已被 `409f7b2b93` 作为正确性 bug 删除；三个 **Marlin MoE 库级默认**（`VLLM_MARLIN_MOE_BLOCK_SIZE=48`、`..._MIN_M=1000`、`..._CALLER_FP32_WORKSPACE=1`，`envs.py:1316-1324`）对当前 `mxfp4_int8_sm80` 部署完全惰性，但任何走 Marlin MoE 的配置都会继承。另：KV-only insert 的生效条件是 `cache_dtype == torch.uint8`（即 `fp8_ds_mla`），**旧的分支仍在**（每次调用白分配 `[M,64,512]` BF16 dummy Q，`dspark.py:367-405`），所以 274.31→16.76 us/layer 那个收益只对 uint8 缓存成立 | 按文档复现「对照组」会静默拿到「候选组」；找优化点时不要再看这些路径；换 KV dtype 时那笔收益会消失 |
| 6 | 文件名的误导：`vllm/models/deepseek_v4_1/amd/rocm.py` 是 **SM80 CUDA** 上真正在跑的 attention 类（`DeepseekV41AmpereMLAAttention` 继承它），`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` 里才是 **SM80 稀疏 prefill/decode 的 Triton tiling**（9-15 两个 prefill commit 只动这个文件）。同类的**同名不同义**还有 `num_lookahead_tokens`：`VllmConfig.num_lookahead_tokens`（KV 预留，`config/vllm.py:620-650`）与 `speculative.py:1470-1475` 往 draft hf_config 上写的同名属性（对 DeepSeek/DSpark 是 no-op，只有 mlp_speculator/medusa 会读） | 对照该分支的 `ampere/*` 时要找对文件，否则会得出"vllm170 没有对应实现"的错误结论；grep `num_lookahead_tokens` 时会同时命中两个不同契约 |
| 7 | `VLLM_DFLASH_EAGER_TARGET`：`model_runner.py:1711-1724` 说 PP DSpark 需要它开启，但 PP5 启动脚本默认 **0** | 单请求 PP5 DSpark 实际是「draft eager + target graph」，与注释冲突；这是未解决的正确性/性能耦合 |
| 8 | 分布式 draft 的 CUDA graph 代码在**工作区未提交**，启动脚本 `enforce_eager:true` 让它从不生效；2026-09-17 用 `=0` 试开即在 `dp_utils.py:186` 断言崩溃 | 不要把「代码里有图」当成「生产开了图」；这是一个**失败中的未完成项** |
| 9 | acceptance% 与平均接受长度满足 `mal = 1 + K × acceptance` | 跨 K 比 acceptance% 是算术错觉（见 §4.2） |
| 10 | profile 语料整体是 operator-scope：`kernel_profile_accepted=false`、`phase_timing_accepted=false`，phase 文件只有算子 µs（`round_summary: null`） | 不要拿 `phase_*.md` 当吞吐；不要嵌套相加 |
| 11 | 短窗口 tok/s 与种子方差：8–16 token 的窗口可低到 4.7 ms（出现 1483.50 tok/s 的假值）；PP3 并发 4 的种子方差 ±42%，多组配对 CI 跨零 | 端点结论必须用未插桩、多 repeat、固定种子的口径 |
| 12 | 311 条含 `acceptance_rate` 的记录里 **103 条为 0**（早期/target-only 期） | 直接对 `profiles/**/result.jsonl` 求平均没有意义 |
| 13 | CMP 170HX 无 CUPTI/NCU/sanitizer：`compute-sanitizer` 直接返回 `GPU debugging features are disabled`（memcheck 从未跑过），`ERR_NVCMPGPU` / `CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED` 封死 NCU。所有性能数字基于 CUDA Event + 精确 shape 微基准 + 端到端 A/B，插桩本身会把 19.81 tok/s 打到 11.10/1.61 | 跨启动、跨工具的百分比只能当方向；没有 occupancy / 寄存器 / 访存 / 竞态证据 |
| 14 | **上游与该分支都没有 DeepSeek-V4.x 的 DSpark 参考数字**：树内所有 acceptance 指标都来自 Qwen3-4B / Gemma-4-12B / Qwen3.6-35B，V4-Flash-DSpark 的 eval 只断言宽松的 `min_acceptance_length: 1.2`；DSpark 的 CI 跑在 `h200_35gb`/`mi300_1`，**从不跑 SM80、也从不跑 PP>1** | 你们 q3/q4/q5/q7 那批数据是自测资产（也是唯一的），`min_accepted ≥ 4.8` 那种高接受率参考来自 Kimi-K3（不同架构、K=8 + block 验证），不可类比 |

---

## 附：本次核对用到的关键文件

**cpu-hybrid 分支**（`.worktrees/dsv41-cpu-hybrid`）
`vllm/models/deepseek_v4_1/{hybrid.py,ced.py,pp_kv.py,cpu_moe.py,expert_cache.py,cache_policy.py,cache_feedback.py,host_memory.py,hybrid_runtime.py,nvidia/model.py,nvidia/dspark.py,ampere/prefill_metadata.py,ampere/ampere_sparse.py,common/engram.py}`、
`vllm/v1/worker/gpu/spec_decode/dspark/{speculator.py,utils.py,markov_argmax.py}`、
`vllm/config/{speculative.py,cache.py,vllm.py}`、`vllm/v1/core/{kv_cache_utils.py,kv_cache_manager.py,sched/scheduler.py}`、
`vllm/v1/attention/ops/mqa_logits_triton.py`、`vllm/model_executor/layers/sparse_attn_indexer.py`、
`csrc/cpu/dsv41/*`、`benchmarks/kernels/cpu/*`、`README.md`。

**vllm170**
`benchmarks/deepseek_v4_1/run_cmp170hx_pp5.sh`、
`docs/design/vllm170-dsv41-engram-pp5-profile-2026-09-11.md`、
`docs/design/vllm170-dsv41-pp5-decode-profile-2026-09-12.md`、
`docs/design/vllm170-dsv41-dspark-pp5-memory-2026-09-14.md`、
`docs/design/vllm170-dsv4-sm80-dspark-profile-2026-08-15.md`、
`profiles/deepseek_v4/dspark-{q1-audit,width-audit,q5-promotion}/*/result.jsonl`、
`vllm/models/deepseek_v4_1/{nvidia/model.py,nvidia/dspark.py,attention.py,common/engram.py,common/ops/cache_utils.py}`、
`vllm/v1/core/{kv_cache_manager.py,sched/scheduler.py}`、`vllm/config/speculative.py`。
