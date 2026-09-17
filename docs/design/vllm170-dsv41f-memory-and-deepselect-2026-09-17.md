# dsv41f-PP5 的"3.8 GiB 常驻"到底是什么，以及 DeepSelect 能接在哪（2026-09-17）

## 1. 结论先行

1. PP2 上"权重/KV/图"之外的 **3.82 GiB 常驻 torch 分配不是碎片、也不是泄漏**，其中
   **约 1.76 GiB 是 indexer 的全局最坏情况工作区**（`max_model_len × min(max_num_seqs,40)`
   定尺），剩余约 2.0 GiB 是首次 prefill 之后被缓存分配器**留住的激活池**（实测峰值 1.28 GiB）
   与 draft（DSpark）侧常驻。碎片只有 0.33 GiB 那一项。
2. 这份工作区**确实"按全局维度而不是本 rank 的能力"分配**，和"每 rank 只应分配持有层的
   激活空间"是同一类问题：8,388,608 个压缩槽 = 假设 **8 条各自 1M token 的 prefill 同时进行**；
   而这块卡的 KV cache 总共只装得下 1,171,051 token，物理上不可能出现这种 batch。
   ⇒ 预分配比可达上界大 **7.2×**，每 rank 浪费 ≈ **0.92 GiB**（gather）+ 0.25 GiB（反量化）
   + 0.09 GiB（logits）。
3. **DeepSelect 不解决显存**（它只做"选择"，不做 K gather），它解决的是 **indexer top-k 的耗时**：
   本机实测（CMP 170HX，sm_80 cubin）在我们真实形状上比 `persistent_topk` 快
   **1.1–2.0×（decode）/ 13.3×（prefill 行数 128）**。而且我们**默认根本没开** SM80 流式融合
   top-k（`VLLM_INDEXER_PREFILL_STREAMING_TOPK` 默认 0），prefill 走的就是可被直接替换的
   `ops.top_k_per_row_prefill`。

## 2. 3.82 GiB 的逐项拆解

PP2 = rank2 = GPU3，稳态 `torch_alloc = 61.414 GiB`（有插桩实测），其中
weights 56.06 + KV 1.21 + CUDA graphs 0.32 = 57.59 GiB，余 **3.82 GiB**。

### 2.1 indexer 工作区：≈1.46 GiB（全局最坏情况定尺）

`vllm/v1/worker/workspace.py` 的 workspace manager 是**进程内单例**（`_manager`），
`get_simultaneous()` 只会把同一块 buffer 增长到各调用点的最大值 —— 所以这是**每 rank 一份**，
不是每层一份（K pool 变体 `SparseAttnIndexerKpool` 本项目未使用）。它的尺寸来自：

| 项 | 代码 | 尺寸（我们 1M / max_num_seqs=8） |
| --- | --- | --- |
| gather values | `_gather_workspace_shapes(max_total_seq_len, 128, fp8)`（`sparse_attn_indexer.py:717`） | 8,388,608 × 128 B = **1.074 GB** |
| gather scales | 同上 `(total_seq_lens, 4)` uint8 | 33.5 MB |
| logits | `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=96` | 96 MiB |
| Triton 反量化 K | `(max_model_len, 128)` bf16（`sparse_attn_indexer.py:517`） | **256 MiB** |
| Triton Q/去量化缓冲 | `(max_num_batched_tokens, q_rows, 128)` bf16 | ~33 MB |
| radix top-k | `RADIX_TOPK_WORKSPACE_SIZE` | 1 MiB |
| **合计** | | **≈1.46 GiB** |

其中 `max_total_seq_len` 的来源（`vllm/v1/attention/backends/mla/indexer.py:623`）：

```python
def get_max_prefill_buffer_size(vllm_config):
    # NOTE(Chen): 40 是魔法数… flashmla_sparse 的 workspace 是 5 * max_model_len…
    return max_model_len * min(max_num_seqs, 40)      # = 1,048,576 × 8 = 8,388,608
```

再被 `// compress_ratio`（`deepseek_v4_1/attention.py:1205`）后传给 `SparseAttnIndexer`：
ratio=1 的层（20–39）拿到 8,388,608，ratio=2 的层（2–19）拿到 4,194,304。

### 2.2 实测佐证：max_model_len 1M → 262k，每 rank 少 1.76 GiB

| run | max_model_len | PP0 `Available KV cache memory`（profiler 口径） |
| --- | --- | --- |
| run15 | 1,048,576 | 3.62 GiB |
| run16 | 262,144 | 5.38 GiB |

只改 max_model_len，其他不变 ⇒ **每 rank 有 ≈1.76 GiB 与 max_model_len 成正比**，
和 §2.1 里按 max_model_len 定尺的 1.074 GB + 256 MiB 同量级（差额来自 ratio=2 层的第二套
签名与对齐取整）。

### 2.3 余下的 ≈2.0 GiB

* 激活池：profiler 在 PP2 实测 dummy batch 的 `torch peak increase = 1.28 GiB`；
  缓存分配器保留这些 block（`cached` 只有 0.33 GiB，说明大部分是**活着的** workspace/激活）；
* DSpark draft：draft stage 落在 PP1–3，其权重 + 缓冲 + draft KV；
  佐证：权重加载量 PP1 57.48 / PP2 56.06 / PP3 56.06 GiB，而三者的 8 层 checkpoint 权重都是
  55.07 GiB（层 16–23 精确求和），差额 2.41 / 0.99 / 0.99 GiB 只能来自 draft 侧。

## 3. 修法 A：按本 rank 实际能装下的量定尺（推荐先做）

**核心**：gather buffer 只需要覆盖"这一层 indexer 的 K cache 里**可能同时存在的压缩槽**"。
KV cache 的真实容量在运行时已知（`DeepseekV4IndexerCache.kv_cache` 的形状），
所以正确上界是 `num_blocks × block_size`（该层压缩槽数），而不是
`max_model_len × max_num_seqs // ratio`。

```python
# deepseek_v4_1/attention.py:1205 附近（草图）
raw = get_max_prefill_buffer_size(vllm_config) // self.compress_ratio
slots = int(self.k_cache_shape[0]) * int(self.k_cache_shape[1])   # 该层 K cache 的槽数
self.max_total_seq_len = min(raw, slots)
```

预期收益（每 rank）：

| 项 | 现在 | 改后 | 省 |
| --- | --- | --- | --- |
| gather values + scales | 1.107 GB | ~0.15 GB（按 1.17M token 容量） | **0.92 GiB** |
| Triton 反量化 K | 256 MiB（按 max_model_len） | 按 tile（几 MiB） | **0.25 GiB** |
| logits | 96 MiB | 8–16 MiB（配合分块） | 0.08 GiB |
| **合计** | | | **≈1.2 GiB/rank** |

PP2 的空闲从 ~1.1 GiB 涨到 ~2.3 GiB ⇒ **2048-token prefill chunk（峰值多 ~0.5 GiB）就装得下了**，
而 2048 chunk 正是 9-17 那次 OOM 的直接原因。

**风险**：上界必须严格成立（越界会写出垃圾 attention）。验证手段：
`tests/models/test_deepseek_v41_ced.py` 里的 suffix/gather 单测 + 长上下文捞针 + 8 路并发压测，
并在 set 之后断言 `total_seq_lens <= slots`。

## 4. DeepSelect 研究报告

### 4.1 是什么

`/mnt/data/apps/vllma100/DeepSelect`：DeepSeek 官方开源的 **DSA（稀疏注意力）TopK 内核库**
（v1.0.0，2026-09-10），覆盖两类负载：
Lightning Indexer（bf16，`topk ≤ 4096`）与 sampler（fp32，vocab≈128K）。
**sm_80 tier 就是为 CMP 170HX 做的**（`DEEP_SELECT_CUDA_ARCHS=80`；LDG.128 直取 + 软件模拟
`setp/add.rn.bf16x2`；用 occupancy=2 隐藏延迟），仓库里的 `.so` 已经是 sm_80 cubin。

### 4.2 本机实测（我在这台机器上跑的，`CUDA_VISIBLE_DEVICES=1`，fp32 logits，topk=512）

| 形状 | torch.topk | vLLM `persistent_topk` | DeepSelect | DeepSelect vs vLLM |
| --- | --- | --- | --- | --- |
| R=1, V=8192 | 0.097 ms | 0.048 ms | 0.029 ms | 1.69× |
| R=1, V=65536 | 0.134 ms | 0.053 ms | 0.048 ms | 1.12× |
| R=8, V=65536 | 0.151 ms | 0.064 ms | 0.047 ms | 1.36× |
| R=8, V=262144 | 0.202 ms | 0.189 ms | 0.124 ms | 1.53× |
| R=16, V=262144 | 0.272 ms | 0.266 ms | 0.135 ms | 1.97× |
| **R=128, V=262144** | 1.331 ms | 3.066 ms | **0.230 ms** | **13.3×** |

正确性：与 `torch.topk` 的 values 全等（`values_ok=True`）。
`get_stride_requirement() = (1024, 32)` 字节 —— **输入行 stride 必须是 1024 B 的整数倍**
（fp32 即 256 元素对齐），输出索引行 stride 32 B 的整数倍（512×int32=2048 B ✓）。

### 4.3 接在哪：三个候选点

| 位置 | 现状 | 替换方式 | 收益/风险 |
| --- | --- | --- | --- |
| **prefill** `sparse_attn_indexer.py:896` `ops.top_k_per_row_prefill(logits, cu_seqlen_ks, cu_seqlen_ke, topk_indices, …)` | 非融合，logits 已物化（我们**没开** `VLLM_INDEXER_PREFILL_STREAMING_TOPK`，默认 0） | `deep_select.topk(logits, 512, end=cu_seqlen_ke, output_idx=topk_indices, idx_oob_fill_value=-1, return_value=False)` | **收益最大（行数多时 13×）**；`ks` 在 DCP 关闭时恒为 0，`begin` 不支持不影响我们 |
| **decode** `sparse_attn_indexer.py:1125` `torch.ops._C.persistent_topk` | 已按 `{512,1024,2048}` 分流，带 1 MiB radix workspace | 同上，`end=seq_lens`，`output_idx=topk_indices` 切片（stride 2048 B ✓） | 1.1–2.0×；单步占比小（~1%） |
| **sampler** top-k（fp32, vocab 129280） | `torch.topk` 类路径 | DeepSelect 的 sampler 场景 | 小；只在请求显式带 `top_k` 时生效 |

**显存方面 DeepSelect 基本不省**：它替掉的是选择（radix workspace 仅 1 MiB），
gather/反量化/logits 这些大块是"算 logits"必需的，仍要按 §3 单独修。

### 4.4 约束与风险清单（接入前必须处理）

1. **stride 1024 B 对齐**：`logits` 的行 stride 必须按 256 个 fp32 元素对齐分配（或 padding 拷一次，
   拷一次 1 行的带宽 ≈ 直接算 logits 的量级，不划算）→ 建议在 logits 分配处把
   `stride(0)` 向上取整到 256。
2. **NaN 默认 abort**：`abort_when_nan_found=True` 会 `trap()` 整个进程 → 生产必须显式传
   `False`（此时它会往 `indices[b][0]` 写 0x3F3F3F3F）。
3. **-INF padding 的已知限制**：当第 k 大值恰好是 `-INF` 时可能返回越界索引。
   我们的 logits 有大量掩码位（candidate mask / 非因果位）——**必须专门构造用例验证**：
   掩码位是否恰好能落到第 k 名。
4. **顺序语义**：`sorted_index=True` 会按索引升序输出并**显著变慢**（大 CTA、occupancy 1）。
   下游（SWA gather）是否依赖顺序需要先确认；若依赖则要付费。
5. **`begin` 未支持**：DCP/PCP 开启时每行 `ks>0`，直接用不了（我们当前 DCP=1、PCP=1，OK）。
6. 小 batch（≤6 行）在 SM80 上没有 cluster tier，只有 1.4–4.7×，decode 收益有限。

### 4.5 建议的落地顺序

1. **A（显存，先做）**：§3 的工作区定尺 —— 1 处代码 + 断言，直接解 2048 chunk OOM，
   让 PP2 空闲从 1.1 → 2.3 GiB。验证：启动后 `nvidia-smi` per-process + 捞针 + 8 路并发。
2. **B（DeepSelect，后做）**：先接 **prefill** 分支（收益 13×、位置最干净），
   用 `VLLM_DSV41_USE_DEEP_SELECT=1` 开关灰度；再评估 decode / sampler。
   验证：`tests/models/test_deepseek_v41_ced.py` + 长上下文捞针 + DSpark 接受率不退化
   （当前 51–55%）+ 与 `persistent_topk` 的索引集合逐位对比（掩码位用例）。
3. **C（可选）**：把 `VLLM_INDEXER_PREFILL_STREAMING_TOPK` 的融合路径与 "triton logits +
   DeepSelect" 做一次正面对比，决定 prefill 最终走哪条（前者省 logits 物化显存，
   后者快；§3 做完后前者的显存优势会变小）。
