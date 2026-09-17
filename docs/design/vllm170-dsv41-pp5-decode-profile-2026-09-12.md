# DSV4.1 Flash PP5 decode CUDA Event 计量（2026-09-12）

## 结论

当前 PP5、单请求、32K 输入、64 token 输出的无埋点 FULL 服务为约
19.81 token/s，即约 50.48 ms/token。模型 GPU 图回放在 CUDA Event 上的
PP 串行工作量约 21.49 ms/token；二者不可直接相减为某一个 kernel 的耗时，
中间还包含 PP 调度与传输、主机侧 metadata、输出复制和请求服务开销。

事件细分表明，后续优化优先级应为：

1. metadata builder 的重复构造和主机/GPU 边界（PP1--PP4 每轮约 6.8--7.6 ms
   的 CUDA scope，CPU scope 也约 6.9--7.7 ms）；
2. attention 的 `q_index_compress_mla` 路径（PP 串行约 67.98 ms/轮，占模型
   Event 工作 19.2%），进一步拆分 indexer query、candidate compact 和
   sparse MLA 之间的未归因部分；
3. FFN/MoE（`layer.ffn` 约 147.09 ms/轮，占 41.5%；routed experts 约
   72.34 ms/轮），重点看 MXFP4 的 quant、align、W13/W2 GEMM 及跨 stream
   overlap；
4. mHC（attention-pre + FFN-pre 约 36.80 ms/轮，占 10.4%），再考虑
   TileLang kernel 融合或 workspace 复用。

FULL graph 的 `lookup_total` 约 3.20 ms/轮；eager scope 中
`lookup_total` 约 2.20 ms、`model.engram_prepare` 约 2.51 ms，合计约
4.71 ms/轮，在 eager 模型 Event 工作中约 1.3%。FULL graph 还要单独计入
约 5.04 ms/轮的 graph input staging，不能把它和 graph replay 相加成 kernel
耗时。Engram 仍不是当前首要热点；sampling 约 1.3 ms/轮，也不是第一
优化对象。

## 测试配置和正确性

- 硬件：CUDA_VISIBLE_DEVICES=`1,2,3,4,5`，5 张 CMP 170HX，PCI 顺序映射
  为 PP0--PP4；TP=1，PP=5。
- 模型：`/media/z/data6/models/DeepSeek-V4.1-Flash`。
- KV：FP8，block size 128，显式 KV cache 1,000,000,000 bytes。
- MoE：`mxfp4_int8_sm80`。
- Engram：SSD 源 `/media/z/data6/vllm-engram-cache`，PP0/PP1 各将约
  94.42 GiB 表完整预加载到主存，GPU 仅保留约 24.75 MiB pinned +
  24.75 MiB staging；未修改 swap。
- 调度：`max-num-batched-tokens=2048`，`max-num-seqs=1`，FULL 请求会被
  `DeepseekV41ROCMAiterSparseSWABackend` 降级为 `FULL_DECODE_ONLY`。
- 工作负载：精确 32,000 prompt tokens，单请求，64 completion tokens，
  3 repeats；每次 repeat 前使用同长度 8-token warmup。
- 三轮 32K 请求均返回 200，usage 均为 `prompt_tokens=32000`、
  `completion_tokens=64`。另发起短中文请求，返回“答：CUDA Event计时用于
  测量”，证明服务路径和 tokenizer/采样路径正常。

## 计量方法

`vllm/models/deepseek_v4/phase_profile.py` 使用 `torch.cuda.Event`
（`enable_timing=True`）在同一 CUDA stream 上记录开始/结束事件；每个
decode step 只做一次设备同步，再读取所有事件，避免每个子阶段单独
`synchronize`。每个 scope 同时记录 `cpu_elapsed_us`，用于识别
`Work.wait()`、metadata builder 等 GPU Event 看不到的主机阻塞。

FULL graph 轮只打开图外边界：
`target.attn_metadata`、`target.model_inputs`、PP receive wait/copy、
Engram staging、`target.model_graph_replay`、logits 和 sample。eager 轮
打开模型、层、attention/indexer、MoE、Engram 的细分 scope。eager 的事件
记录本身会改变服务调度，因此只用于相对热点排序。

历史对照采用
[`vllm170-dsv4-sm80-decode-profile-2026-08-14.md`](vllm170-dsv4-sm80-decode-profile-2026-08-14.md)、
[`vllm170-dsv4-sm80-sparse-decode-p1-2026-08-15.md`](vllm170-dsv4-sm80-sparse-decode-p1-2026-08-15.md)
和
[`vllm170-dsv4-sm80-streaming-indexer-2026-08-16.md`](vllm170-dsv4-sm80-streaming-indexer-2026-08-16.md)。
这些文档中的 PP3、Marlin 和 SM80 特定参数只作为“先量 parent、再拆
children、最后回到 graph/e2e 门禁”的方法参考，不直接移植数值。

原始 JSONL 和生成报告保存在：

- `/tmp/dsv41_decode_profile_20260912/graph_phase/`
- `/tmp/dsv41_decode_profile_20260912/graph_analysis/phase_decode.md`
- `/tmp/dsv41_decode_profile_20260912/eager_phase/`
- `/tmp/dsv41_decode_profile_20260912/eager_analysis/phase_decode.md`
- `/tmp/dsv41_decode_profile_20260912/eager_phase2/`
- `/tmp/dsv41_decode_profile_20260912/eager_analysis2/phase_decode.md`
- `/tmp/dsv41_decode_profile_20260912/eager_phase3/`
- `/tmp/dsv41_decode_profile_20260912/eager_analysis3/phase_decode.md`

## 吞吐对照

| 轮次 | 模式 | decode token/s | decode ms/token | 说明 |
| --- | --- | ---: | ---: | --- |
| 3 repeats | FULL，无事件 | 19.63 / 19.78 / 20.03 | 50.99 / 50.56 / 49.93 | 生产参考 |
| 3 repeats | FULL，图外 CUDA Event | 11.10 / 10.94 / 11.04 | 90.05 / 91.40 / 90.58 | 仅边界计量 |
| 3 repeats | eager，全部细分 Event | 1.61 / 1.59 / 1.61 | 621.85 / 626.97 / 619.26 | 仅热点排序 |
| 3 repeats | eager，定向细分 Event | 2.76 / 2.78 / 2.75 | 362.52 / 360.29 / 364.25 | 仅 metadata/indexer/sparse MLA |

细分采集的吞吐不能用于评价优化收益；任何候选改动都必须回到无埋点
FULL 轮，以 50.48 ms/token 的基线复测。

## FULL graph 边界预算

以下是 CUDA Event 的 PP 串行工作和关键边界。不同 scope 可能嵌套，不能
把同一列全部相加。

| Scope | PP 串行/关键值 ms | 解释 |
| --- | ---: | --- |
| `target.model_graph_replay` | 21.488 | 五个 PP rank 的 graph replay 工作和 |
| `target.attn_metadata` | PP1 9.075；PP2 6.588；PP4 6.363 | metadata scope，PP1 最慢 |
| `target.graph_input_staging` | 5.043 | FULL replay 前输入/Engram staging 的跨 rank 工作和 |
| `engram.lookup_total` | 3.201 | PP0+PP1，每轮一次 |
| `target.sampling_total` | 1.699（关键 PP4） | logits + sample 的总 scope |
| `target.logits` / `target.sample` | 1.072 / 0.326（PP4） | sampling 子阶段 |
| `target.model_inputs` | 0.788（关键 PP1） | 模型参数组织和 PP 输入准备 |
| `target.pp_recv_copy` / `wait` | 0.135 / 0.117（关键 PP4） | CUDA scope；CPU wait 另计 |

Engram graph 轮的主要子项为 host raw gather 1.296 ms、indices-to-CPU
0.480 ms、GPU dequant 0.305 ms、host gather 0.281 ms、raw H2D 0.181 ms、
staging wait 0.081 ms（均为 PP0+PP1 每轮工作和）。这些是可观测的数据搬运
链，当前总量小于 metadata 和 attention/MoE。

## eager 模型拆分

| 模块/范围 | PP 串行工作 ms/轮 | 模型 Event 工作占比 |
| --- | ---: | ---: |
| `target.model_forward` | 354.292 | 100.0% |
| `layer.total` | 342.455 | 96.7% |
| `layer.ffn` | 147.092 | 41.5% |
| `layer.attention` | 143.746 | 40.6% |
| `moe.routed_experts` | 72.343 | 20.4% |
| `attention.q_index_compress_mla` | 67.975 | 19.2% |
| `moe.modular.kernel` | 53.457 | 15.1% |
| `attention.output_projection` | 33.176 | 9.4% |
| `moe.shared_experts` | 28.913 | 8.2% |
| `attention.sparse_mla` | 25.088 | 7.1% |
| mHC attention-pre + FFN-pre | 36.804 | 10.4% |

`layer.total` 的平均值为 8.54--8.71 ms/layer（compression ratio 1/2/0
分别为 8.54/8.56/8.71 ms），因此旧 DSV4F/SM70 的 Marlin block-M 参数
不能直接套用到当前 MXFP4 SM80 后端。

### Attention/indexer

在 `q_index_compress_mla` 内，当前可见的 PP 串行工作为：

| Scope | ms/轮 | 备注 |
| --- | ---: | --- |
| `attention.indexer` | 8.704 | 8 个 index-source layer 的父 scope |
| `attention.indexer_query` | 5.905 | query 侧，仍有 parent/child overlap |
| `indexer.paged_logits` | 3.050 | Triton paged logits |
| `indexer.candidate_blocks` | 0.978 | candidate compact/filter |
| `indexer.topk` | 0.916 | top-k |
| `attention.sparse_mla` | 25.088 | sparse MLA 父 scope |
| `sparse_mla.attention` | 13.667 | sparse MLA kernel 子 scope |
| `attention.output_projection` | 33.176 | inverse-RoPE/wo_a/wo_b |

`attention.indexer` 的子项相加会超过 parent，这是异步 stream 和嵌套 scope
的正常现象。已增加 buffer 初始化、mask/compact 的独立 scope，并保留已有
stream join 计量用于解释 parent residual；现有数据仍不支持先改 top-k launch
参数。

### 二次 CUDA Event 拆分（2026-09-12）

在相同 32K 输入/64 token 输出、PP5、单请求条件下，追加以下事件：SWA
`ragged_indptr`/`ragged_pack`，固定 graph buffer 的 `ragged_copy_indptr`/
`ragged_copy_indices`，以及 indexer 的 `topk_buffer_clear`、candidate
`select`/`mask`、decode `pack`/`unpack`。原始记录和汇总位于
`eager_phase2`/`eager_analysis2`；该轮仍是 eager+事件诊断，绝对吞吐和
parent 时间不用于生产 A/B。

PP2 是 metadata 的最慢诊断 rank。每次 SWA builder 调用（每轮 4 次）为：

| 子阶段 | CUDA Event 均值/调用 | 说明 |
| --- | ---: | --- |
| `target.attn_metadata.swa.ragged_build` | 0.580 ms | dense→临时 ragged 的完整范围 |
| `sparse_mla.ragged_indptr` | 0.108 ms | lengths clamp/cumsum/indptr |
| `sparse_mla.ragged_pack` | 0.095 ms | dense-prefix Triton pack |
| `target.attn_metadata.swa.ragged_copy` | 0.421 ms | 临时 ragged→固定 graph 地址 |
| `target.attn_metadata.swa.ragged_copy_indptr` | 0.055 ms | indptr copy |
| `target.attn_metadata.swa.ragged_copy_indices` | 0.067 ms | indices copy |

因此已标记 GPU 子项只解释约 0.325 ms，`ragged_build` 仍有约 0.255 ms、
`ragged_copy` 仍有约 0.299 ms 的 launch/allocator/host 间隙；这些 residual
不是单个 kernel 的时间。PP1/PP3/PP4 的排序一致（build 约 0.541/0.561/
0.549 ms，copy 约 0.391/0.406/0.394 ms），说明该结论不是单一 rank 异常。

Indexer 的新子项均低于当前 attention parent：candidate `select` 在 PP2
约 0.283 ms/调用，`mask` 在 PP3/PP4 约 0.189 ms/调用，top-k buffer clear
约 0.067 ms/调用；该 workload 没有触发 decode pack/unpack。它们不支持
作为下一步 P0。当前可执行方向是研究 SWA dense→ragged 的 direct-ragged
写入（去掉临时 dense/临时 ragged 和二次 copy），并用无埋点
`FULL_DECODE_ONLY` 端到端门禁验证；不要先调整 top-k launch 参数。

### 三次定向 CUDA Event 拆分（2026-09-13）

为降低“全量 scope 本身改变调度”的干扰，第三轮只保留 metadata、indexer
和 sparse MLA 相关事件，并关闭 `record_function`。3 次精确 32K/64 请求均
返回 HTTP 200，decode 为 2.76/2.78/2.75 token/s；该吞吐仍只表示采样开销，
不能与无埋点 FULL 基线比较优化收益。

SWA metadata 的排序在较低埋点密度下保持不变。PP1--PP3 每次 builder 的
`ragged_build` 为 0.293--0.300 ms，`ragged_copy` 为 0.356--0.372 ms；最慢
PP4 分别为 0.372 和 0.462 ms。PP4 每轮 4 次 builder，单这两个 parent
scope 就约 3.34 ms/轮。`token_map`、`decode_indices` 和 `validity` 在 PP4
另为 0.386/0.213/0.127 ms/调用。由此，优先消除四组 builder 重复构造和
临时 ragged→固定 graph buffer 的二次复制，结论不依赖第二轮的高密度
埋点。

第三轮也首次拆开了 global top-k dense→ragged 转换：

| 子阶段 | PP5 串行 ms/轮 | 模型 Event 工作占比 |
| --- | ---: | ---: |
| `sparse_mla.topk_lens` | 1.055 | 0.4% |
| `sparse_mla.topk_indptr` | 1.145 | 0.5% |
| `sparse_mla.topk_ragged_pack` | 1.153 | 0.5% |
| 三项合计 | 3.353 | 1.4% |

`sparse_mla.index_convert` 全部路径为 7.890 ms/轮。其中每个 rank 的
index-source 层会出现 cache miss：PP0 每轮 1 次，PP1--PP4 每轮各 2 次，
共 9 次；其余可复用层共 29 次 cache hit。miss 路径约 6.66 ms/轮，hit
路径仍约 1.20 ms/轮。三段 global-ragged kernel 解释 miss 路径约一半，
剩余约 3.31 ms/轮包含分配、Python/launch 间隙及 parent 边界，不能归因
为单一 kernel。

因此新的优先级仍是：先做 SWA metadata direct-ragged/caller-owned buffer，
再研究 `index_convert` 的 miss-path workspace 和 hit-path 固定地址复用；
candidate select/mask、top-k clear 和 global-ragged 单项均不足以独立列为 P0。

### Direct-ragged SWA 实现与无埋点 A/B（2026-09-13）

V4.1 的 SM80 metadata builder 已改为直接把 decode SWA 的 lengths、indptr
和 indices 写入 caller-owned、graph-stable buffer，不再构造临时 dense
indices，也不再经过临时 ragged tensor 和第二次 copy。该路径由
`VLLM_DEEPSEEK_V4_DIRECT_RAGGED_SWA=1` 控制，PP5 启动器默认启用，仍可设为
`0` 回退到原实现。V4 AMD builder 中原有 direct-ragged 实现缺失的 capability
gate 也已补齐。

kernel correctness 覆盖 causal/non-causal 两种路径：direct 与原
dense→ragged 的 lengths、indptr、indices 完全一致，且返回 view 保持
caller buffer 地址。生产门禁使用同一模型、prompt hash、精确 32K 输入、
64 token 输出、3 repeats、无 Event 的 `FULL_DECODE_ONLY` 服务：

| SWA metadata 路径 | decode token/s | 均值 | 相对原路径 |
| --- | --- | ---: | ---: |
| 原 dense→ragged→copy | 20.16 / 20.26 / 20.16 | 20.20 | 基线 |
| direct-ragged | 22.82 / 23.10 / 22.80 | 22.91 | +13.4% |

关闭 direct-ragged 的第二轮复测均值为 19.39 token/s，说明服务级结果存在
一定的主机和调度波动，但没有出现收益反转。两组请求均为 HTTP 200，usage
均是 32,000/64；最终常驻服务的短中文 CUDA Event 问答也语义正确。原始
记录位于：

- `/tmp/dsv41_decode_profile_20260912/direct_ab/direct_off.jsonl`
- `/tmp/dsv41_decode_profile_20260912/direct_ab/direct_off_env0.jsonl`
- `/tmp/dsv41_decode_profile_20260912/direct_ab/direct_on.jsonl`

Direct-ragged 消除了此前最大的 dense/ragged 物化，但没有消除 metadata
builder 中对单请求稳定字段的逐轮重建。后续仍按 metadata 子项排序推进，未
提前转向 `sparse_mla.index_convert`。

### 单请求 metadata 持久化与单-token融合（2026-09-13）

在 `max-num-seqs=1` 的当前服务 shape 下，SWA、MLA 和 Compressor builder
原先仍会逐轮构造全零 token-to-request map。先只持久化 SWA map，再把同一
门禁扩展到 MLA/Compressor；多请求配置和运行时多请求均保留动态路径。
同一套详细 Event scope 的前后结果为：

| Event 子项 | 优化前 ms/调用 | 优化后 ms/调用 | 变化 |
| --- | ---: | ---: | ---: |
| SWA token map | 0.600 | 0.059 | -90.2% |
| MLA c1 token map | 0.506 | 0.058 | -88.5% |
| MLA c2 token map | 0.587 | 0.054 | -90.8% |
| Compressor token map | 0.494 | 0.056 | -88.8% |
| metadata parent（关键 rank） | 15.685 ms/轮 | 13.737 ms/轮 | -12.4% |

无 Event 的 3 次 32K/64 生产门禁从 22.07 token/s 提升到 26.21
token/s（仅 SWA map），再到 27.51 token/s（所有 map），累计 +24.6%。原始
记录位于：

- `/tmp/dsv41_decode_profile_20260913/metadata_direct/endpoint_off.jsonl`
- `/tmp/dsv41_decode_profile_20260913/metadata_persist_on/endpoint_on.jsonl`
- `/tmp/dsv41_decode_profile_20260913/metadata_all_maps/endpoint.jsonl`

持久化后，SWA `decode_indices` 成为最大的已解释子项。追加 Event 将
direct-ragged 拆为 indptr 和 indices 两个 kernel；分别约 0.16 和
0.17--0.18 ms/调用，二者接近各半。因此为 causal、单 decode token 增加
一个融合 kernel，同时写 lengths、indptr 和 ragged indices；多 token 和
non-causal/DSpark 继续使用原实现。无 Event A/B 为 27.54 → 28.45 token/s，
提升 3.29%。同埋点下 `decode_indices` 从约 1.16 降到 0.82 ms/调用，metadata
parent 从 15.25 降到 13.93 ms/轮。原始记录位于：

- `/tmp/dsv41_decode_profile_20260913/swa_direct_split/endpoint_baseline.jsonl`
- `/tmp/dsv41_decode_profile_20260913/swa_single_fused/endpoint.jsonl`

下一已测热点是每个 rank 重复 4 次的 SWA validity。纯单请求 decode 的有效
mask 恒为全真，因此以独立环境门禁持久化；混合 prefill、多请求或 token 数
不匹配时回退到 `slot_mapping >= 0`。Event 从约 0.17 降到 0.054--0.056
ms/调用，关键 metadata parent 从 13.93 降到 13.40 ms/轮；无 Event A/B 为
28.45 → 31.10 token/s，提升 9.34%。原始记录位于：

- `/tmp/dsv41_decode_profile_20260913/swa_validity_persist/endpoint.jsonl`
- `/tmp/dsv41_decode_profile_20260913/swa_validity_persist/analysis/phase_decode.md`

随后按剩余子项依次验证两项结构性消除。第一项仅在 direct-ragged 路径跳过
未被 consumer 暴露的 `decode_swa_lens` 尾部清零；dense fallback 仍保留原
清零语义。Event 中该 scope 从约 0.114--0.117 ms/调用降到空 scope 的
0.028--0.029 ms/调用，metadata parent 从 13.40 降到 12.84 ms/轮；无 Event
门禁从 31.10 提升到 31.96 token/s，提升 2.77%。

第二项针对真正剩余热点：每个 PP rank、每轮四次展开 2,048 个 SWA 物理
slot。四个 KV group 共用 BlockPool，但分别分配物理 block id，因此不能共享
最终索引。单请求、单-token causal decode 改为让 SM80 split-attention 直接
读取该组自己的 `seq_len + block_table`，在 attention tile 内映射逻辑位置，
完全省去四组 ragged-index 展开；多请求、prefill、non-causal/DSpark 和非
SM80 路径不变。

| 指标 | paged-direct 前 | paged-direct 后 | 变化 |
| --- | ---: | ---: | ---: |
| SWA builder | 2.021 ms/调用 | 1.470 ms/调用 | -27.3% |
| metadata parent（关键 rank） | 12.836 ms/轮 | 10.416 ms/轮 | -18.9% |
| 32K/64 decode | 31.96 token/s | 35.25 token/s | +10.28% |

无 Event 三轮为 35.48 / 35.21 / 35.06 token/s；相对最初单请求 metadata
持久化前的 22.07 token/s 累计提升 59.7%。paged 与 ragged attention 的隔离
输出一致，既有 split/caller-output 测试通过，FULL_DECODE_ONLY graph 捕获和
短中文语义请求也通过。原始记录位于：

- `/tmp/dsv41_decode_profile_20260913/swa_tail_skip/endpoint.jsonl`
- `/tmp/dsv41_decode_profile_20260913/swa_tail_skip/analysis/phase_decode.md`
- `/tmp/dsv41_decode_profile_20260913/swa_paged_direct/endpoint.jsonl`
- `/tmp/dsv41_decode_profile_20260913/swa_paged_direct/event_result.jsonl`
- `/tmp/dsv41_decode_profile_20260913/swa_paged_direct/analysis/phase_decode.md`

以上 Event parent 的绝对值会随子 scope 密度明显变化，只在相同埋点集合中
比较；生产收益始终以无控制文件的端到端请求为准。

### FULL graph 复测、indexer wait 排除与 sparse-attention split（2026-09-13）

metadata 优化收口后重新采集了 FULL graph 边界和一次诊断专用 eager 全量
Event。FULL graph 只暴露图外边界：PP 串行 `target.model_graph_replay` 为
21.735 ms/轮，graph input staging 为 5.071 ms/轮，Engram lookup 为
2.973 ms/轮；Python 内部 attention/FFN scope 已被图捕获，不能从该轮继续
拆分。因此用 eager Event 只做热点排序，不把其绝对值当生产延迟：

| eager parent/scope | PP 串行 ms/轮 | eager 模型 Event 占比 |
| --- | ---: | ---: |
| `target.model_forward` | 368.921 | 100.0% |
| `attention.total` | 157.162 | 42.6% |
| `layer.ffn` | 147.980 | 40.1% |
| `attention.q_index_compress_mla` | 80.442 | 21.8% |
| `attention.output_projection` | 33.665 | 9.1% |
| `attention.sparse_mla` | 28.770 | 7.8% |
| `attention.qkv_insert_compress` | 22.853 | 6.2% |
| `attention.input_gemm_overlap` | 20.031 | 5.4% |
| `sparse_mla.attention` | 14.656 | 4.0% |
| `attention.indexer` | 10.011 | 2.7% |

随后只保留 indexer、cache 和 join scope 做窄 Event 轮。该轮
`attention.qkv_insert_compress.join` 与
`attention.indexer_prepare_cache.join` 分别只有 0.122 和 0.120 ms/轮；
单-token decode 没有 pack/unpack。`attention.indexer` 的 query、paged
logits、candidate 和 top-k 已分别解释 4.353、3.897、1.936 和
1.125 ms/轮。剩余差值主要是 eager custom-op/launch 边界和 Event 本身的
扰动，不存在一个应优先消除的隐藏 stream wait，因而不先改 top-k 参数。

也隔离验证过把单-token top-k lengths、indptr 和 ragged pack 合成一个
kernel：CUDA Graph replay 从 6.471 降到 4.391 us，但按每轮 9 次 miss 估算
只节省约 0.019 ms，不到 graph model 工作的 0.1%。该实验已回退，没有进入
生产代码。

按下一个可解释主热点转向 `sparse_mla.attention`。原 split heuristic 的
搜索上限为 16，且注释明确是 gfx950 调参值；SM80 单请求生产 shape 没有独立
参数。新增 `VLLM_DEEPSEEK_V4_SM80_DECODE_NUM_SPLITS`，通用默认 `0` 保留原
heuristic，PP5 启动器使用 32，并限制显式值为 1--64。真实 CMP170HX、
`q=[1,64,512]`、paged SWA window 128 的 CUDA Graph replay 结果为：

| 生产 attention shape | split=16 | split=32 | 变化 | 模型层数 |
| --- | ---: | ---: | ---: | ---: |
| SWA128 only | 15.635 us | 20.896 us | +33.7% | 2 |
| SWA128 + top-k512 | 31.821 us | 26.760 us | -15.9% | 38 |

两种 shape 的所有候选输出都与 split=16 在 `atol=rtol=2e-2` 内一致。按
2/38 层粗加权，split partial+reduce 总量约下降 14.7%；局部 SWA-only
回退不能忽略，但不足以抵消 38 个 compressed 层的收益。早先 16-head
隔离 shape 得出的 split=48 不适用于本模型的 64-head 生产路径，未采用。

最终仍用同一 prompt hash、精确 32K/64、3 repeats、无 Event 的
FULL_DECODE_ONLY 服务门禁：

| 参数 | decode token/s | 均值 | 相对 split=16 |
| --- | --- | ---: | ---: |
| 原 heuristic（实际 16） | 35.475 / 35.208 / 35.063 | 35.249 | 基线 |
| SM80 split=32 | 36.489 / 36.448 / 36.602 | 36.513 | +3.59% |

split=32 的 1K/64 三轮为 34.129 / 34.265 / 34.343 token/s，均返回精确
1,000/64 usage；短中文请求返回“CUDA Event计时用于在CUDA流中记录时间戳，
以精确测量GPU内核或操作之间的执行耗时”，语义正确。原始端到端记录位于：

- `/tmp/dsv41_decode_profile_20260913/post_metadata_full/`
- `/tmp/dsv41_decode_profile_20260913/eager_attention_full/`
- `/tmp/dsv41_decode_profile_20260913/eager_indexer_wait/`
- `/tmp/dsv41_decode_profile_20260913/sm80_split32_full/endpoint_32k_verified.jsonl`
- `/tmp/dsv41_decode_profile_20260913/sm80_split32_full/endpoint_1k.jsonl`
- `/tmp/dsv41_decode_profile_20260913/sm80_split32_full/semantic.json`

优化后的下一 attention 候选是 qkv/input GEMM 和 Q projection；其中 input
GEMM join 只有 0.756 ms/轮，已证明不是等待热点。若隔离 GEMM 没有可复现的
SM80 收益，就按总 parent 排序转向 40.1% 的 FFN/MoE，而不是继续追逐
sub-ms 的 indexer/top-k 或 Engram 子项。

### MoE/MXFP4

MXFP4 子 scope 是诊断值，不应与 `layer.ffn` 或 `moe.routed_experts`
相加。每轮可见的 PP 串行工作为：

| Scope | ms/轮 |
| --- | ---: |
| `moe.mxfp4_sm80.w13_quant` / `w13_gemm` | 5.965 / 6.965 |
| `moe.mxfp4_sm80.w2_quant` / `w2_gemm` | 5.706 / 4.180 |
| `moe.mxfp4_sm80.align` | 5.070 |
| `moe.mxfp4_sm80.activation` | 0.963 |
| `moe.mxfp4_sm80.reduce` | 1.929 |
| `moe.shared_experts.w13` / `w2` | 7.300 / 7.029 |

优先检查 quant + align 是否重复读写，以及 routed/shared 两条 stream 的
等待是否暴露；历史 Marlin 的 block-M、BF16 cache 和 workspace 结论不应
直接迁移。

## PP 负载和平衡

### 按热点继续拆分：FFN、mHC 与 output projection（2026-09-13）

这一轮严格使用 CUDA Event 与 CUDA Graph replay，所有单卡微基准均在真实
CMP170HX（物理 GPU1）上运行，shape 固定为 V4.1 decode 的 M=1。端到端门禁仍
使用 PP5、同一 prompt hash、精确 1K/32K 输入和 64 output token；微基准不能
替代该门禁。

#### FFN/MoE

单专家 dense 路径不可取。W13 的 `M=1,N=4608,K=5120` Graph P50 为
33.577 us，W2 的 `M=1,N=5120,K=2304` 为 18.944 us；按 6 个 routed expert
的下界已经分别达到 201.5 us 和 113.7 us，均劣于 grouped MXFP4 路径。
因此没有把 grouped kernel 改成逐专家 dense GEMM。

当前生产后端支持
`(block-M, block-N)=(16,64),(16,128),(64,128),(128,128)`。重启后确认物理
GPU0 是 RTX 3080 Ti Laptop，不再用于 CMP170HX 性能结论；下表在物理 GPU1
的 CMP170HX 上按真实 `E=384,M=1,topk=6` 重测，每次 Graph replay 包含 40 个
连续调用。16/32 是临时暴露已有模板后取得的诊断值，不是生产支持项。

| block-M/N | W13 | W2 |
| --- | ---: | ---: |
| 16/32（诊断） | 121.280 | 64.947 |
| 16/64 | **114.637** | **55.987** |
| 16/128 | 126.438 | 60.851 |

16/32 的两段合计慢于 16/64，已撤销其 dispatch 暴露。16/64 的孤立 W13+W2
合计为 170.624 us，相对生产 16/128 的 187.289 us 快 8.9%；但无埋点 PP5
端到端的干净 A/B 得到相反结果：同一 32K prompt、64 output、各 5 次请求，
16/128 的 decode 吞吐均值为 36.270 tok/s、decode 时间 1.7374 s，16/64
分别为 35.284 tok/s、1.7865 s，即吞吐回退 2.72%、decode 时间增加 2.83%。
这表明孤立 kernel 的 tile 收益没有落在全图关键路径上；较小 block-N 增加的
CTA 数还可能挤压 routed/shared 两条 stream 的并发。16/64 已回退，生产默认
保持 16/128，不再继续扫这一组 tile。

随后验证了把 routed W13 输出的 `SiLU+mul` 与 W2 输入 INT8 per-token quant
融合。融合输出与现有 BF16 activation→INT8 quant 位级一致；物理 GPU1 的
CMP170HX、真实 6 routed rows、`I=2304`、每个 Graph 64 次调用下，Graph P50
从 4.656 us 降至 2.688 us（-42.3%）。但 40 个 MoE 层合计只回收约
0.079 ms，不到 21.735 ms 模型 Graph 的 0.4%，未达到 5% 生产解释力门禁，
故原型已移除。这再次说明 eager scope 中 activation/quant 的毫秒级数字主要
用于排序，不能直接外推生产 Graph 收益。

最后对占主要时间的 grouped kernel 本体做了同 shape 的 warp/stage 定向扫描：

| 16x128 kernel | W13 Graph P50 | W2 Graph P50 | 合计相对 3-stage |
| --- | ---: | ---: | ---: |
| warp-N=32，3-stage（生产） | 126.438 us | 60.851 us | 基线 |
| warp-N=64，3-stage | 130.406 us | 62.746 us | +3.13% |
| warp-N=32，2-stage | 127.565 us | 62.106 us | +1.27% |
| warp-N=32，4-stage | 124.877 us | 60.032 us | -1.27% |

warp-N=64 和 2-stage 在 isolated Graph 已回退。4-stage 的理论上限只有约
0.09 ms/轮，不到模型 Graph 的 0.5%，不足以承担更高 shared-memory 压力，
因此也未进入 PP5 门禁。最终动态库已恢复并复测为 warp-N=32、3-stage；这条
CUTLASS 参数线收口，下一步若继续优化 routed experts，必须是能改变 W13/W2
整条数据流的结构方案，而不是继续扩散 tile/warp/stage 小参数。

`moe.shared_experts.wait` 只有 1.039 ms/轮、0.3%，共享专家的父 scope
不是可单独优化的 GEMM 热点。

原始记录：

- `/tmp/dsv41_decode_profile_20260913/ffn_decode/grouped_supported_tiles_m1.json`
- `/tmp/dsv41_decode_profile_20260913/ffn_decode/dense_w13/`
- `/tmp/dsv41_decode_profile_20260913/ffn_decode/dense_w2/`
- `/tmp/dsv41_decode_profile_20260913/moe_m1_16x64_full/`
- `/tmp/dsv41_decode_profile_20260914/ffn_decode/grouped_tiles_m1_e384_graph40.json`
- `/tmp/dsv41_decode_profile_20260914/ffn_decode/activation_quant_fused_graph64.json`
- `/tmp/dsv41_decode_profile_20260914/ffn_decode/grouped_tile_16x128_warpn64_graph40.json`
- `/tmp/dsv41_decode_profile_20260914/ffn_decode/grouped_tile_16x128_stages2_graph40.json`
- `/tmp/dsv41_decode_profile_20260914/ffn_decode/grouped_tile_16x128_stages4_graph40.json`
- `/tmp/dsv41_decode_profile_20260914/ffn_decode/grouped_tile_16x128_restored_graph40_r200.json`
- `/tmp/dsv41_decode_profile_20260914/moe_tile_ab/baseline_16x128_32k.jsonl`
- `/tmp/dsv41_decode_profile_20260914/moe_tile_ab/candidate_16x64_32k.jsonl`

#### MoE decode preparation

`moe.mxfp4_sm80.align` 的生产 decode 输入固定为
`M=1,topk=6,E=384,block-M=16`。通用 `moe_align_block_size` 在该 shape
上仍为 384 个专家分配
cumsum workspace，并执行 1024-thread count/prefix/fill kernel 和后续 sort
kernel；但 DSV4.1 router 保证 6 个 expert id 唯一，所以每个 route 固定占一个
16-row block 即可。`sorted_ids` 的 96 个元素和 `num_post_pad=96` 现在在权重
加载后按层持久化；动态 `expert_ids` 直接使用 int32 router 输出的一维 view。
prefill 和所有非该固定 shape 仍走通用 align。

CMP170HX 上以 64 次调用装入单个 CUDA Graph、300 个 Event 样本得到：

| M=1 prepare | eager P50 | Graph P50 |
| --- | ---: | ---: |
| quant + 通用 align | 226.304 us | 8.016 us |
| 持久 metadata + quant | 131.072 us | 2.320 us |
| 通用 align 单独 | 116.736 us | 5.792 us |

Graph 口径 prepare 减少 71.1%。另一个把 metadata 写入融合到 quant kernel 的
原型为 2.592 us，慢于持久化方案，未进入生产。持久 metadata 与通用 align
生成的有效 `(expert_id, route_slot)` 集合逐项一致，CMP170HX 定点测试总计
7 passed。

无 Event 的紧邻 PP5 32K/64 门禁从 34.576 提升到 34.691 tok/s（+0.33%，
三次均值），decode 时间从 1.8224 降到 1.8163 s；该幅度接近运行噪声，只能
作为无回退门禁，不能宣称稳定的端到端加速。1K/64 候选三次均值为
34.460 tok/s；中文语义请求正确回答 100 摄氏度。由于改动明确消除了每个 MoE
层的两个 decode kernel，且不影响 prefill，候选保留。

原始记录：

- `/tmp/dsv41_decode_profile_20260913/ffn_decode/prepare_m1_fused_graph64.json`
- `/tmp/dsv41_decode_profile_20260913/ffn_decode/prepare_m1_persistent_graph64.json`
- `/tmp/dsv41_decode_profile_20260913/moe_persistent_metadata/baseline_32k.jsonl`
- `/tmp/dsv41_decode_profile_20260913/moe_persistent_metadata/candidate_32k.jsonl`
- `/tmp/dsv41_decode_profile_20260913/moe_persistent_metadata/candidate_1k.jsonl`

#### mHC

生产 shape 是 `M=1,N=24,K=20480`，现有 TileLang 特殊分支
`n_thr=1024,tile-N=4` 已是本轮扫描中最稳的配置。Graph P50 为：

| tile-N | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 24 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| us | 21.504 | 35.840 | 21.504 | **20.480** | 23.552 | 29.696 | 29.696 | 37.888 |

固定 tile-N=4 扫 `n_thr=128/256/512/1024` 没有稳定超过默认值。进一步拆分
得到 Graph P50：prenorm GEMM 16.384 us、pre-norm finalize 15.360 us、post
14.336 us。历史已有的 post+GEMM fused kernel 在孤立测试中为 15.360 us，
相对独立 post+GEMM 的 19.456 us 约快 21%，并通过 V4.1 carried pre-mix
逐张量正确性测试；但接入生产后 32K/64 仅 36.2766 vs 36.2759 tok/s，
1K/64 下降约 0.93%，故融合候选已回退。结论是 mHC 父 scope 的约 36.8 ms/轮
主要包含异步 Python/调度和嵌套 scope 边界，不能把该绝对值当成三个设备 kernel
的可加和；当前不再修改 mHC 参数。

原始记录：

- `/tmp/dsv41_decode_profile_20260913/mhc_decode/tile_n_m1.json`
- `/tmp/dsv41_decode_profile_20260913/mhc_decode/n_thr_m1.json`
- `/tmp/dsv41_decode_profile_20260913/mhc_decode/components_m1.json`
- `/tmp/dsv41_decode_profile_20260913/mhc_decode/fused_candidate_m1.json`

#### output projection

旧 eager 全量 Event 已把 `attention.output_projection` 拆到其内部：PP5
串行父 scope 33.665 ms/轮，其中 `inv_rope_wo_a` 18.961 ms、`wo_b`
7.299 ms、`wo_a` 6.611 ms、`inv_rope` 4.928 ms。真实 M=1 的独立 Graph
P50 为 inverse-RoPE 14.336 us、wo_a grouped BMM 64.512 us、BF16 wo_b
矩阵乘 76.800 us；把 einsum 写法换成显式 BMM 在 Graph 中仍为 64.512 us，
没有可复现收益。当前 SM80/ROCm-compatible 路径已在加载时缓存 `wo_a` 的
BF16 权重，且历史 BF16 `wo_b` cache 会增加显存，不能为 1K/32K decode
冒险引入。故 output projection 继续作为观测热点，暂不改生产实现。

原始记录：

- `/tmp/dsv41_decode_profile_20260913/output_projection_m1.json`
- `/tmp/dsv41_decode_profile_20260913/eager_attention_full/analysis/phase_decode.md`

#### q-index/compressor follow-up

进一步核对真实 shape 后，compressor 本体和 compressed-cache insert 只有
0.633/0.432 ms/轮，不能解释 `q_index_compress_mla` 的 80.442 ms 父 scope。
主 Q 与 index Q 读取同一个 1280 维归一化输入；前者 shape 为
`M=1,K=1280,N=32768`，后者只在 8 个 index-source 层执行，shape 为
`M=1,K=1280,N=4096`。使用当前 dense MXFP8 linear 后端的 Graph P50 为：

| Q projection | Graph P50 |
| --- | ---: |
| main Q | 45.696 us |
| index Q | 9.856 us |
| 两个独立调用 | 56.000 us |
| 合并为 `N=36864` | 49.280 us |

合并只节省 6.72 us/index-source layer，8 层合计约 53.8 us/轮，不到
`target.model_forward` 的 0.2%。这同时证明 eager Event 中
`indexer.q_projection` 的 1.494 ms/轮不能直接解释为 GEMM 时间。为该收益改造
checkpoint mapping 和量化权重生命周期不合算，合并方案不接入生产。
MoE 已放弃 Marlin；这里测量的是服务本来就在使用的 dense MXFP8 linear，
没有改变 MoE backend。q-index/compressor 当前没有一阶 kernel 候选，后续回到
`attention.sparse_mla` 父范围和 PP0/PP1 critical path，而不是继续调
compressor、index Q 或 top-k 小项。

原始记录：

- `/tmp/dsv41_decode_profile_20260913/q_projection_m1_marlin.json`

eager 的每-rank `target.model_forward` 均值为：PP0 73.315、PP1 73.024、
PP2 68.833、PP3 68.162、PP4 70.957 ms；189 个稳态 round 中，最慢 rank
出现次数为 PP0:100、PP1:66、PP2:4、PP3:7、PP4:12。PP0/PP1 的高耗时与
早期 index-source layer（特别是 layer 2、8、14）以及 Engram 所在层有关。
在改 kernel 前应做 PP partition 小范围 sweep，并重新检查串行 PP critical
path；不能只看五个 rank 的平均值。

## 优化门禁

1. 每个候选保留同一 prompt hash、32K/64 token workload 和 3 repeats。
2. correctness：HTTP 200、精确 prompt/completion usage、短中文语义请求；
   模型改动另做对应单测或 golden token 检查。
3. 性能：无埋点 FULL 的 decode ms/token，不能用 eager Event 数字代替。
4. 计量：CUDA Event、CPU scope、PP rank、P50/P90 一起记录；嵌套 scope
   按 parent/child 分开，不做重复加总。
5. 排序优先选择能解释至少 5% 生产路径的边界；Engram 和 sampling 只有
   在其他路径下降到同量级后再优化。

## 2026-09-14 重启后生产复测与热点重排

本轮重新确认物理 GPU0 是 RTX 3080 Ti Laptop（SM86），没有用于服务；服务按
PCI 顺序固定使用物理 GPU1--5 的五块 CMP 170HX（SM80），PP=5、TP=1、
`max-num-batched-tokens=2048`、`FULL_DECODE_ONLY`。MoE 仍为
`mxfp4_int8_sm80`，没有使用 Marlin；Engram 在 PP0/PP1 各完整驻留约
94.42 GiB 主存，GPU 仅有双槽 dequant staging。

### 生产端到端边界

无埋点生产服务用同一个精确 32K prompt、64 output 连续跑 5 次；五次输入/输出
usage 均为 32000/64，prompt SHA256 均为
`3e20f41dd9acd613d3d775455d5d7119495e6a1642fe99454eeea7f1dbc5b74f`。
decode 吞吐均值 35.657 tok/s，中位数 35.609 tok/s；按 63 个 token interval
计算，均值为 **28.046 ms/token**，中位数 28.083 ms/token，范围
27.772--28.274 ms/token。

只保留 Graph 边界 Event 的探针运行覆盖 3 次、189 个稳态 decode token。它使
HTTP 端耗时放大到 72.35 ms/token，因此只用 CUDA Event 区间来分解边界；同一
PP stage 按 rank 串行求和，父子 scope 不重复相加。该边界与无埋点生产均值几乎
完全闭合：

| 生产边界 | ms/token | 生产占比 | 说明 |
| --- | ---: | ---: | --- |
| 模型 Graph replay（PP0--PP4 串行） | 21.166 | 75.47% | 五个 rank 分别 4.244/3.976/3.888/4.642/4.416 ms |
| Graph input staging（PP0--PP4） | 4.954 | 17.66% | 包含下方 Engram lookup，不能再与之相加 |
| 其中 Engram lookup（PP0+PP1） | 3.100 | 11.05% | staging 的 62.6% |
| 最终 rank sampling total | 1.837 | 6.55% | logits 1.125 ms，sample 0.390 ms；均是父子关系 |
| 未闭合残差 | 0.089 | 0.32% | 由 28.046 减去三个顶层边界所得 |

Graph replay 的每轮 P50/P90 为 21.773/21.820 ms。189 轮最慢单 rank
出现次数为 PP0:70、PP3:116、PP4:3；但单 token 要串行经过全部 PP rank，
因此仅重分层不能减少 21.166 ms 总和。PP receive copy/wait 的最大 rank 也只有
0.145/0.127 ms/token，不是当前一阶热点。

Engram 的 3.100 ms/token 可继续拆为 host raw gather 1.272、indices-to-CPU
0.482、GPU dequant 0.302、host gather 0.277、raw H2D 0.163、staging wait
0.074 ms/token。它已经不是“可忽略”，但仍排在两个主计算热点之后；后续应研究
双缓冲和跨 stage 重叠，不改变完整 RAM 驻留方案，也不启用 swap。

原始记录：

- `/tmp/dsv41_decode_profile_20260914/rebaseline/endpoint_32k_64_r5.jsonl`
- `/tmp/dsv41_decode_profile_20260914/rebaseline/graph/phase_decode.md`
- `/tmp/dsv41_decode_profile_20260914/rebaseline/graph/phase_summary.json`

### Graph 内部的生产-shape 校准

完整 eager Event 探针同样覆盖 189 个 token，但把 model forward 放大到
371.663 ms/token，绝对值不能回填线上预算。它只作为结构交叉检查：attention
占 43.5%、FFN 38.8%、mHC pre 合计 10.2%、output projection 9.3%；
`q_index_compress_mla` 父 scope 占 22.3%。为得到可用于排序的绝对量，本轮在
物理 GPU1 的 CMP 170HX 上对当前代码和真实 M=1 shape 用 CUDA Graph/Event
重新校准。下表是 Graph 内部预算的混合归因，不是另一组服务 wall time：

| Graph 内部项目 | 估计 ms/token | Graph replay 占比 | 生产端占比 | 证据边界 |
| --- | ---: | ---: | ---: | --- |
| routed MoE W13+W2 | 7.493 | 35.4% | 26.7% | E=384、top-k=6；40 层，126.464+60.851 us/layer |
| output projection | 4.174 | 19.7% | 14.9% | 40 层；inverse/copy 4.787、BF16 wo_a 59.443、生产 MXFP8-Marlin wo_b 40.115 us/layer |
| main Q + 8 层 index Q | 2.076 | 9.8% | 7.4% | main Q 49.920 us/layer，index Q 9.856 us/layer |
| compressed sparse MLA | 1.219 | 5.8% | 4.3% | 38 个压缩层乘 32.092 us；ragged shape 校准，非 exact paged 路径 |
| persistent MoE prepare | 0.090 | 0.4% | 0.3% | 2.240 us/layer；通用 align 已从生产单 token 路径消除 |
| 其余 Graph 工作 | 4.773 | 22.6% | 17.0% | input/cache/indexer、route/activation/reduce、mHC、C0 attention 和 glue |

MoE 的 7.493 ms 只含 routed W13/W2 GEMM，不含 route、quant、activation 和
reduce，是可信下界；其中 W13 约 5.059 ms、W2 约 2.434 ms。output projection
的生产路径中 `wo_a` einsum 和显式 BMM 都约 59.44 us/layer。后续对象级 selector
核对确认 `wo_b` 实际使用 `MarlinMxfp8LinearKernel`，而不是早期诊断误判的
emulation/BF16 路径；真实 shape 的 MXFP8-Marlin 为 40.115 us/layer，BF16
矩阵乘为 75.200 us/layer。修正后的完整路径估计约 104.35 us/layer，即
4.174 ms/token。因此真正目标仍是跨 `wo_a`/`wo_b` 的数据流，而不是 inverse
RoPE 物化或 einsum/BMM 写法。

Q projection 中 main Q 明显大于 index Q。只在 8 个 index-source layer 合并
两个投影，每层最多节省约 6.9 us，总计约 0.055 ms/token，不足生产端 0.2%，
不值得改 checkpoint mapping。sparse MLA 的 32.092 us 校准已验证最大绝对误差
0.000122；生产是 paged-SWA，而当前微基准是 ragged block-256，所以该项只用于
排序，不能宣称是 exact 服务耗时。

原始记录：

- `/tmp/dsv41_decode_profile_20260914/rebaseline/eager/phase_decode.md`
- `/tmp/dsv41_decode_profile_20260914/rebaseline/micro/moe_m1_16x128_graph40.json`
- `/tmp/dsv41_decode_profile_20260914/rebaseline/micro/moe_prepare_m1_graph64.json`
- `/tmp/dsv41_decode_profile_20260914/rebaseline/micro/output_projection_m1_production_graph40.json`
- `/tmp/dsv41_decode_profile_20260914/rebaseline/micro/q_projection_m1.json`
- `/tmp/dsv41_decode_profile_20260914/rebaseline/micro/sparse_mla_m1_s32.json`

### 当前优化目标顺序

1. **P0 routed MoE W13/W2**：约 7.493 ms/token，是最大可直接归因热点。
   下一步先用低密度 Event/NCU 确认完整 routed path 的 critical boundary，再做
   能减少权重流量或 grouped GEMM 工作量的结构候选。不要再扫 Marlin、16x64
   tile 或通用 align：16x64 端到端已回退 2.72%，persistent prepare 仅
   0.090 ms/token。
2. **P0 output projection `wo_a`/`wo_b`**：约 4.174 ms/token。优先研究
   caller-owned workspace、后端选择或 `wo_a -> wo_b` 数据流融合；inverse-RoPE
   和 einsum/BMM 改写已经排除。任何 BF16 cache/量化候选必须同时过显存门禁。
3. **P1 attention Q/cache/sparse 全链**：可直接校准的 Q 与 compressed sparse
   MLA 合计约 3.295 ms/token，另有未独立归因的 cache/indexer glue。先补 exact
   paged production-shape 微基准和低密度 q-index 子边界，不先调 top-k；eager
   中 top-k/candidate 仅约 0.3%/0.4% model scope。
4. **P1 Engram staging**：lookup 3.100 ms/token。目标是重叠 host gather、dequant
   与 H2D，而不是更改完整主存驻留或使用 swap；必须以无埋点端到端 A/B 判断，
   因为同步 Event 会扰动异步流水。
5. **P2 sampling/logits**：sampling total 1.837 ms/token，其中 logits 1.125 ms；
   等前三项下降后再做。
6. **P2 mHC 与 metadata**：eager 相对构成仍显示 mHC pre 约 10.2%，但现有融合
   候选端到端中性/回退；Graph metadata 最大 rank 为 2.523 ms/token，却与 input
   staging 重叠且受探针同步影响。两者都必须先建立低密度 critical-path 证据，
   不按放大的 eager 数字直接修改。

后续每个候选仍以同一 32K/64 prompt、至少 3 repeats、无埋点 decode
ms/token 为唯一性能门禁；CUDA Event 只负责解释变化来源。只有候选能解释至少
约 5% 生产路径时才进入实现和端到端 A/B。

## 2026-09-14 P0/P1 候选门禁收口

本轮以重启后无埋点 32K/64 五次基线 `28.046 ms/token` 为准，2% 收益要求至少
节省 `0.561 ms/token`。微基准均在物理 GPU1 的 CMP 170HX（SM80）上使用
CUDA Graph/Event；GPU0 是 RTX 3080 Ti Laptop，不进入任何性能结论。

### routed MoE 组合候选

先后隔离验证了 on-demand B-fragment MMA、block-N 256，以及二者组合。最终组合
同时采用 expert-major grid、on-demand MMA 和 16x256 tile；与紧邻的生产
16x128 基线比较：

| routed W13+W2 | Graph P50/layer | 40 层折算 | 端到端上限 |
| --- | ---: | ---: | ---: |
| 生产 16x128 | 126.566 + 61.133 = 187.699 us | 7.508 ms/token | 基线 |
| on-demand + 16x256 | 120.755 + 58.112 = 178.867 us | 7.155 ms/token | 约 +1.26% |

组合只回收约 `0.353 ms/token`，说明两个局部收益共享同一权重读取/CTA 调度来源，
不能线性叠加。它未达到 2%，没有进入耗时的 PP5 服务 A/B；三处 C++ 候选、
16x256 dispatch 和动态库均已恢复为生产默认。共享专家也已确认生产使用 aux
stream 重叠：串行 routed+shared 为 246.682 us/layer，现有 overlap 为
213.478 us/layer；共享尾部只剩 13.824 us/layer，40 层理论最多回收
`0.553 ms/token`（1.97%），同样低于门槛。

原始记录：

- `/tmp/dsv41_decode_opt_20260914/moe_p0_compound/baseline_16x128_rotate.json`
- `/tmp/dsv41_decode_opt_20260914/moe_p0_compound/ondemand_blockn256_rotate.json`
- `/tmp/dsv41_decode_opt_20260914/moe_p0_compound/restored_default.json`
- `/tmp/dsv41_decode_opt_20260914/shared_expert/overlap_gate.json`

### output projection、Q、sparse MLA 与 sampling

对象级 backend 核对和真实 M=1 shape 重新校准排除了旧预算中的两个误差：

- `wo_b` 已经是 MXFP8 Marlin，Graph P50 `40.115 us`，BF16 为
  `75.200 us`；不需要修复 selector。
- `wo_a` 的 8 路 MXFP8 Marlin 为 `115.968 us`，相对 BF16 grouped BMM 的
  `59.443 us` 慢约 95%，包括 atomic-add 变体也没有改善。
- inverse-RoPE 加输出物化只有 `4.787 us/layer`，即使完全消除也只回收
  `0.192 ms/token`；因此没有再实现低胜算的自定义融合 GEMM。
- main Q 的生产 MXFP8 Marlin 为 `45.107 us/layer`，BF16 为
  `73.267 us/layer`；index Q 的 BF16 虽从 `9.728` 降到 `9.114 us/layer`，
  但只出现 8 次，收益不足 `0.005 ms/token`。
- exact paged sparse MLA 的生产 attention/reduce 组合为
  `31.406 us/layer`，38 层约 `1.193 ms/token`；当前 16x16/4-warp 配置仍最优。
- LM-head projection + greedy argmax 的生产路径为 `1042.816 us/token`，融合
  原型为 `953.024 us/token`，只节省约 `0.090 ms/token`（0.32%）。

全 40 层 `fused_wqa_wkv` 的真实 `M=1,K=5120,N=1792` MXFP8 Marlin 为
`13.645 us/layer`，总计约 `0.546 ms/token`；BF16 为 `17.766 us/layer`。
这给 attention 输入主投影提供了严格上界：即使完全消除也低于 2%，不再单独
调它。Engram 内的 pinned-i32 D2H 与 host-gather 小项合计理论收益也低于门槛，
而此前整条早预取服务 A/B 仅快 0.71%，继续保持完整 RAM 驻留且不使用 swap。

原始记录：

- `/tmp/dsv41_decode_opt_20260914/output_projection/wo_b_backend.json`
- `/tmp/dsv41_decode_opt_20260914/output_projection/wo_a_marlin_gate.json`
- `/tmp/dsv41_decode_opt_20260914/attention_q/bf16_gate.json`
- `/tmp/dsv41_decode_opt_20260914/attention_q/input_qr_kv_backend.json`
- `/tmp/dsv41_decode_opt_20260914/attention_p1/sparse_mla_paged_32k_s32.jsonl`
- `/tmp/dsv41_decode_opt_20260914/sampling_p0/lm_head_greedy.json`
- `/tmp/dsv41_decode_opt_20260914/engram_p1/indices_d2h.json`

### 更新后的优化方向

本轮没有新的生产实现通过 2% 门禁。routed MoE 仍是最大的单项，但 tile、
stage、warp、on-demand 和组合参数线均已收口；output、Q、sparse MLA、sampling
及 Engram 的现有单点候选也都受上界约束。下一轮不应继续散扫小 kernel，应该：

1. 用低密度 Event 重新建立 `21.166 ms/token` model Graph 内的互斥、可加和
   边界，重点覆盖 routed/shared 重叠后的尾部、mHC 与 attention 输入/cache glue；
2. 只研究能跨至少两个现有边界的结构方案，例如跨层 workspace/调度复用或减少
   graph break；候选先证明能解释 `>=0.561 ms/token` 再编码；
3. 所有晋升仍使用同一精确 32K/64、至少 3 次无埋点服务 A/B 和语义正确性门禁。

### routed MoE 硬件约束与 16x256 端到端门禁

尝试用 Nsight Compute 对生产 `M=1,top-k=6,block-M/N=16x128` 的 W13/W2
采集 SpeedOfLight、MemoryWorkloadAnalysis 和 Occupancy 计数器，但 CMP170HX
固件直接返回 `ERR_NVCMPGPU`；因此本项目不能把 NCU 计数器作为可用证据，后续
仍以 CUDA Graph/Event、静态资源和端到端结果为准。

静态资源显示 16x128 的 W13/W2 分别使用 136/130 registers/thread；每个 CTA
为 128 threads，单 token 的 6 个专家分别形成 216/240 个 CTA。按 MXFP4 packed
weight、shift2 和 channel scale 的不可省读取下界估算，每层 W13/W2 至少读取
68.66/34.39 MiB；结合 126.566/61.133 us 的 Graph P50，有效下界约为
568.8/589.9 GB/s。该结果不能替代硬件计数器，但说明后续若仍优化此 kernel，
必须改变权重读取或 CTA 调度，而不是处理已经很小的 route/quant/reduce。

为验证 CTA 调度对 routed/shared 双流的影响，补做了 16x256 候选。它把 CTA
数减半，并在紧邻 Graph 微基准中将 W13+W2 从 187.648 降到
178.637 us/layer（局部 +4.80%）。同一精确 32K/64 prompt 的无埋点 PP5
端到端结果为：

| 16x256 轮次 | repeats | decode ms/token | 相对 16x128 基线 |
| --- | ---: | ---: | ---: |
| 首轮 | 5 | 27.625 | +1.50% |
| 热态追加 | 5 | 27.805 | +0.86% |
| 合并 | 10 | 27.715 | +1.18% |

基线为 16x128 的 5 次均值 28.046 ms/token。候选在三种统计口径下都未达到
2% 门槛，已回退源码、selector 和动态库；生产继续使用 16x128。原始记录：

- `/tmp/dsv41_decode_opt_20260914/moe_p2_ncu/`
- `/tmp/dsv41_decode_opt_20260914/moe_p2_static/resource_usage.txt`
- `/tmp/dsv41_decode_opt_20260914/moe_p3_blockn256_e2e/micro.json`
- `/tmp/dsv41_decode_opt_20260914/moe_p3_blockn256_e2e/candidate_32k_r5.jsonl`
- `/tmp/dsv41_decode_opt_20260914/moe_p3_blockn256_e2e/candidate_32k_r5_hot.jsonl`
