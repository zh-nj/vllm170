# DeepSeek-V4 CMP170HX DSpark 分解计量与优化研究

## 结论

本轮在三张 CMP170HX、PP3/TP1、DSpark-7 上把 DSpark 流程从 target q8
verification 中独立出来，并把 prompt 中间 chunk 的 proposal 与稳态 decode proposal
分开计量。生产 Graph 端点仍以关闭 phase profiler 的 benchmark 为准；模块归因使用
CUDA Event，不能把受同步扰动的模块值外推为端点收益。

全局执行链中，随上下文增长的首要对象仍是 target q8：32K 到 64K 时 PP3 replay
由 29.811 ms 增至 32.802 ms。DSpark proposal 本身基本不随上下文增长，分别为
10.265 ms 和 10.347 ms；其最大可见固定成本不是已捕获的 draft graph，而是 graph
外的 metadata 和 context-KV。

DSpark 内部优化顺序为：

1. P0：non-final prefill 的 context-only fast path。保留 hidden combine 和三层
   context-KV 写入，只在最终 prefill chunk 生成可消费的 draft proposal。
2. P0：draft non-causal SWA metadata。复用跨两个同构 group 的公共映射，并直接更新
   Graph 固定地址，禁止增加按上下文增长的大块常驻 buffer。
3. P0：context-KV。增加 KV-only 投影，消除被丢弃的 Q，并融合
   KV norm、RoPE、FP8 insert，复用 caller-owned 小型 scratch。
4. P1：target logits/rejection 的 Graph 化与融合，以及严格的 acceptance 审计。
5. P2：draft graph 内三层 backbone、Markov 和 M=7 Marlin。整个 replay 当前只有
   约 2.7 ms，必须先做隔离上限评估，不能依据 eager 细粒度值直接重写 kernel。

## 测量边界

| 项目 | 配置 |
| --- | --- |
| 模型 | `/mnt/data6/models/DeepSeek-V4-Flash-0731` |
| GPU | 3 x NVIDIA CMP170HX，SM80，PP3/TP1 |
| target 层分配 | PP0/PP1/PP2 = 15/15/13 |
| draft | PP2 上完整 3 层，7 个 speculative tokens |
| 服务参数 | `max_num_batched_tokens=1715`，`max_num_seqs=8` |
| 生产 32K | `profiles/deepseek_v4/20260814T151048Z`，5 repeats |
| 生产 64K | `profiles/deepseek_v4/20260814T151702Z`，3 repeats |
| 32K Graph 边界 | `profiles/deepseek_v4/20260814T161318Z` |
| 64K Graph 边界 | `profiles/deepseek_v4/20260814T161730Z` |
| 三层 eager detail | `profiles/deepseek_v4/20260814T160226Z` |
| 单层 eager detail | `profiles/deepseek_v4/20260814T160552Z` |

CMP170HX 的 CUPTI 返回 `CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED`，所以没有 NCU/CUPTI
kernel counter。本轮接受的 GPU 模块计时源是 CUDA Event。低扰动 Graph run 只打开
顶层 stage whitelist；eager detail 打开三层和 Markov 子范围，用于结构排序。所有服务
进程在 run 结束后由 harness 清理。

## 执行链与计量口径

```text
target prepare（PP worker 可并行）
  -> PP0 q8 target
  -> PP1 q8 target
  -> PP2 q8 target
  -> target logits
  -> sample 或 rejection
  -> draft.propose
       hidden combine/copy
       prepare query/context inputs
       3 x context KV projection/norm/insert
       dispatch
       build non-causal draft metadata
       replay 3-layer draft + logits + Markov
```

同一请求的 target verification 必须串行经过 PP0、PP1、PP2，因此 q8 model range
按对齐轮次求和。worker-local target prepare 可以重叠，依赖路径只取最慢 rank；把三卡
prepare 全部相加只表示跨 rank 工作量，不是墙钟时间。`total`、`stage`、`detail` 是
不同嵌套层级，不得交叉相加。

原报告将 prefill proposal 与 decode proposal 混在一个均值中，本轮增加
`step_kind=prefill|decode`、target token shape、draft position 和 stage whitelist。
Graph run 只保留 production-shaped 外边界，detail run 用于解释边界内部构成。

## 生产端点真值

| 上下文 | repeats | ms/round mean | ms/round p50 | acceptance mean | accepted length mean | decode tok/s mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32K | 5 | 34.964 | 34.938 | 0.2357 | 2.650 | 73.006 |
| 64K | 3 | 37.000 | 36.961 | 0.1548 | 2.084 | 54.012 |

这里的 `accepted length = 1 + accepted speculative tokens / round`。两组请求的输出
轨迹不同，因此 acceptance 只描述本次服务结果，不能把 32K 到 64K 的下降全部归因于
上下文长度。性能 A/B 必须保持 prompt hash、seed、输出策略和样本数一致，并同时报告
`ms/round` 与 accepted length。

## Graph 顶层分解

下表来自打开少量 CUDA Event 的 Graph run，绝对值受同步扰动；它用于比较模块、上下文
增长和边界，而不是替代上一节的生产端点。

| 模块 | 32K | 64K | 64K/32K | 判断 |
| --- | ---: | ---: | ---: | --- |
| target prepare critical rank | 5.886 ms / PP1 | 7.125 ms / PP1 | 1.210x | 随上下文增长 |
| target q8 PP3 serial replay | 29.811 ms | 32.802 ms | 1.100x | 全局首要增长项 |
| target sampling/rejection | 2.411 ms | 2.371 ms | 0.983x | 固定小批次成本 |
| draft proposal total | 10.265 ms | 10.347 ms | 1.008x | 基本与上下文无关 |
| 可见依赖路径 | 48.373 ms | 52.645 ms | 1.088x | 不含 host/PP gap/copy |
| profiler-on endpoint | 75.395 ms | 79.027 ms | 1.048x | 仅显示 profiler 扰动 |

32K 的 worker prepare 工作量总和为 14.341 ms，但依赖路径只计 PP1 的 5.886 ms；
64K 对应为 15.530 ms 与 7.125 ms。分析器同时输出两种口径，防止把并行 worker 工作量
误当成墙钟时间。

### Steady decode proposal

这些 stage 是 `draft.propose_total` 内部按程序顺序出现的同级外边界。

| proposal 子阶段 | 32K | 64K | 32K proposal 占比 | 结论 |
| --- | ---: | ---: | ---: | --- |
| captured draft graph | 2.656 ms | 2.776 ms | 25.9% | 三层 backbone、logits、Markov 全部在内 |
| draft metadata | 2.649 ms | 2.616 ms | 25.8% | 最大 graph-external P0 |
| context-KV | 2.327 ms | 2.335 ms | 22.7% | 第二个 graph-external P0 |
| hidden prepare | 0.725 ms | 0.730 ms | 7.1% | aux hidden combine/copy |
| input prepare | 0.591 ms | 0.591 ms | 5.8% | query/context/slot 映射 |
| dispatch | 0.047 ms | 0.046 ms | 0.5% | 不值得单独优化 |
| parent residual | 1.270 ms | 1.253 ms | 12.4% | Python/launch/未标记间隙，不是单 kernel |
| proposal total | 10.265 ms | 10.347 ms | 100.0% | - |

把整个 captured graph 完全消除的理论上限也只有约 2.7 ms；metadata 与 context-KV
合计约 5.0 ms，并且处于 graph 外，更适合先优化。

## Draft metadata 深拆

eager detail 中 `draft.metadata` 为 3.237 ms/稳态 proposal。通用 metadata hook 的
历史名称仍是 `target.attn_metadata.*`，但这些记录处于 `dspark` flow 和
`draft.metadata` parent 内，实际来自 draft 的 non-causal attention group。

| 子范围 | 调用/round | 每次 | 每 round | 关系 |
| --- | ---: | ---: | ---: | --- |
| non-causal SWA builder | 2 | 1.126 ms | 2.253 ms | 两个 draft KV group parent |
| common metadata | 5 | 0.024 ms | 0.118 ms | 每 KV group 一次 |
| builder decode indices | 2 | 0.350 ms | 0.700 ms | builder child |
| builder direct ragged | 2 | 0.187 ms | 0.375 ms | builder child |
| builder token map | 2 | 0.183 ms | 0.366 ms | builder child |
| builder validity | 2 | 0.096 ms | 0.191 ms | builder child |
| builder tile scheduler | 2 | 0.036 ms | 0.071 ms | builder child |

detail parent 比低 stage run 的 2.649 ms 高 22.2%，说明内部 Event 会放大 enqueue
间隙；子范围只用于排序。下一原型应让两个同构 non-causal group 共享 token map、validity
和长度计算，并直接写入已有 Graph persistent ragged buffer。若两个 group 的 block table
不同，只共享公共 request/query 几何，不共享最终 slot id。所有新增 workspace 应来自
现有 metadata buffer 或小型 caller-owned scratch。

## Context-KV 深拆

每次 proposal 都把 target aux hidden 合并后，为三个 draft layer 写入 context KV。
detail run 的三层合计如下：

| 子范围 | 每层均值 | 三层/round | 说明 |
| --- | ---: | ---: | --- |
| fused `wqa_wkv` input GEMM | 0.220 ms | 0.660 ms | Q 部分随后被丢弃 |
| KV norm | 0.108 ms | 0.323 ms | 三个小 kernel |
| RoPE/quant/cache insert | 0.146 ms | 0.438 ms | 当前还传入 dummy Q |
| 已标记子范围合计 | - | 1.421 ms | parent 为 2.264 ms |
| parent residual | - | 0.843 ms | 分配、zero、launch 与 Event 扰动 |

当前实现为每层调用融合 `wqa_wkv`，切掉并丢弃 q-lora 输出；insert 前再创建
`dummy_q`，调用同时处理 Q/KV 的融合插入算子。P0 原型应提供 context 专用的 KV-only
投影和 KV-only insert，并尝试把 norm、RoPE、FP8 quant、paged write 合成一个 kernel。
这条路线减少算术、临时 tensor 和 launch，不需要缓存额外 BF16 权重。

## Target logits 与 rejection

低 stage 32K/64K 的 target sampling parent 分别为 2.411/2.371 ms，其中 logits 为
0.826/0.813 ms，rejection 为 1.366/1.331 ms。eager detail 将 rejection 继续拆为：

| rejection 子范围 | mean |
| --- | ---: |
| verify total | 0.755 ms |
| verify kernel | 0.495 ms |
| accepted/rejected counts | 0.105 ms |
| apply sampling params | 0.097 ms |
| input views | 0.015 ms |

这些范围小于 metadata/context-KV，先不引入 BF16 atomic 或放宽数学语义。后续只研究
Graph 捕获 target logits/rejection，或在保持 greedy/probabilistic rejection 完全一致的
前提下融合 verify 与 counts。

## Draft graph 内部诊断

三层 eager detail 会把 production graph 拆开，并为大量子范围插入 Event。此时
`draft.generate_total` 膨胀到 30.930 ms，而生产形态的整个 replay 只有约 2.7 ms，
所以以下数据只能决定研究顺序：

| draft layer | total | attention | FFN | mHC-attn | mHC-FFN |
| --- | ---: | ---: | ---: | ---: | ---: |
| 43 | 7.941 ms | 2.721 ms | 3.902 ms | 0.431 ms | 0.510 ms |
| 44 | 7.881 ms | 2.678 ms | 3.842 ms | 0.491 ms | 0.492 ms |
| 45 | 7.968 ms | 2.700 ms | 3.914 ms | 0.490 ms | 0.483 ms |

Markov 的 detail parent 为 5.627 ms。position 0 bias 约 0.080 ms，position 1--6
各约 0.417--0.421 ms；每步 sample 约 0.076--0.086 ms。后六步存在数据依赖，适合研究
固定 DSpark-7 的 persistent/fused Markov loop，但必须以整个 2.7 ms replay 的 A/B 为
晋级门，不能按 5.627 ms 预测收益。

draft MoE 的实际 shape 是 `M=7, K=4096, N=2048, E=256, topk=6, block-M=8`。
eager detail 中 W13/W2/align/activation/reduce 分别为 0.227/0.168/0.139/0.062/
0.054 ms/层，routing padded-row useful ratio 为 41.3%。这支持增加 M=7 隔离 shape sweep，
但其优先级低于 graph 外的 5 ms 固定成本。

## Prefill 中的 DSpark 工作

DSpark proposal 当前在每个 target prompt chunk 后都运行。32K 为 19 x 1667 + 327，
64K 为 38 x 1667 + 654：

| 上下文 | proposal calls | proposal total | graph replay total | metadata total | context-KV total | hidden total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32K | 20 | 133.626 ms | 53.592 ms | 12.752 ms | 29.143 ms | 32.127 ms |
| 64K | 39 | 257.100 ms | 104.879 ms | 21.154 ms | 57.169 ms | 63.491 ms |

这些是 profiler-on Event scope 总量，不是可直接相减的 TTFT。源码确认 non-final prefill
使用已知 `next_prefill_tokens` 作为 bonus token，而 PP 的
`compute_need_sampled_mask()` 明确排除“不产生 sample 的非最终 prefill chunk”。因此
中间 chunk 生成的 draft token 没有下游消费者；但 context-KV 写入是下一 chunk 的缓存
基础，不能跳过。

候选 context-only fast path 在中间 chunk 保留 hidden prepare、input/context slot 准备和
三层 context-KV，只跳过 draft query metadata、graph replay、Markov/sample 和无用输出
传递。按当前 Event scope，单看 metadata + graph 的研究上限约为 32K 的 63 ms、64K 的
123 ms TTFT。该数字不是承诺收益。晋级前必须验证：

- 1/2/3/19/38 个 chunk 边界的最终 token 与完整输出 hash；
- 三层 draft KV cache 在进入首轮 decode 前逐 slot parity；
- PP 延迟 broadcast、请求取消、slot reuse 和 prefix-cache 命中；
- 最终 prefill chunk 仍生成完整 DSpark-7 proposal；
- 32K/64K/131K 的 TTFT A/B/A，显存峰值不得增加。

## 优化验收顺序

| 优先级 | 原型 | 核心验收 | 显存约束 |
| --- | --- | --- | --- |
| P0-A | non-final prefill context-only | token/hash/KV parity；TTFT A/B/A | 不新增持久 buffer |
| P0-B | 两组 draft metadata 公共几何复用 | metadata CUDA Event 与 32K/64K ms/round | 复用现有 Graph buffer |
| P0-C | KV-only RoPE + quant + paged insert | cache 字节一致；32K/64K Graph A/B | 不新增 workspace |
| P1-A | target logits/rejection Graph 化 | greedy/probabilistic acceptance parity | 固定 B1/q8 buffer |
| P1-B | acceptance/宽度扫描 | 同 prompt/seed 策略下 tok/s 与质量 | 计入 Graph 宽度成本 |
| P2 | M=7 Marlin/Markov/draft layer | 整个 draft graph replay A/B | 不缓存 BF16 大权重 |

优化报告同时记录 `ms/round`、accepted length、tok/s、prompt/output hash、PP rank、
context 和峰值显存。任何只改善受扰动 eager 子范围、但不能改善 profiler-off Graph
端点的方案都不进入默认配置。

## P0 实机结论

三项 P0 均先通过局部正确性和 Graph 烟测，再以关闭 profiler 的生产形态决定是否晋升。
P0-A 与 P0-B 已完整撤回，P0-C 仅晋升 context KV 插入子项；`wqa_wkv` 中被丢弃的
Q 投影也完成了独立原型，但因显存交换比不足而撤回。

| 原型 | 局部结果 | 生产 A/B | 决定 |
| --- | --- | --- | --- |
| P0-A non-final context-only | 32K 跳过 19 次无消费者的 metadata/graph，Event scope 少约 63.1 ms | A/B/A 的 TTFT 为 5.816/5.821/5.841 s，B 没有可复现收益 | 撤回 |
| P0-B metadata geometry 复用 | metadata 初测约 91.9 -> 62.2 us | 修正 Graph capture/replay 指针拓扑后，B 的 35.001 ms/round 位于 A1 34.782 与 A2 35.521 之间 | 撤回 |
| P0-C KV-only insert | cache 逐字节一致；prefill insert 274.31 -> 16.76 us/层，decode insert 11.86 -> 8.53 us/层 | 512-token B/A/B：38.275/40.348/38.741 ms/round；64K 均值 36.872 ms/round | 晋升 |

P0-B 首版曾因 capture 时为两个 metadata group 分配独立 buffer、replay 时却共享第一组
指针，导致 acceptance 降至约 0--2.4%。修正为 capture/replay 使用完全相同的指针所有权
后 acceptance 恢复。以后任何 Graph metadata 复用都必须把“捕获与重放的指针拓扑一致”
列为独立正确性门。

P0-C 新增专用 CUDA op，只执行 KV RoPE、UE8M0 FP8 量化和 paged cache 写入。旧路径
为 context token 创建 `[M,64,512]` BF16 dummy Q，并生成同形状的无用 Q 输出；新路径
不申请 caller workspace，也不增加持久显存。CMP170HX 独立计量中，含 dummy Q 分配的
旧路径与新路径在 M=7 时为 69.51/16.90 us，在 M=1667 时为 316.19/17.18 us。
复现命令为
`CUDA_VISIBLE_DEVICES=1 .venv/bin/python benchmarks/kernels/deepseek_v4_kv_only_insert.py`。

关键证据目录：

- Graph candidate/baseline：`20260814T234835Z` / `20260814T235326Z`；
- 32K 64-token A/B/A：`20260814T235640Z` / `20260814T235943Z` /
  `20260815T000245Z`，短样本结果落在噪声区；
- 32K 512-token B/A/B：`20260815T000553Z` / `20260815T000828Z` /
  `20260815T001120Z`，两侧候选合并 38.508 ms/round，相对基线快 4.56%；
- 64K candidate：`20260815T001422Z`，三次 TTFT 均值 11.793 s，峰值显存与既有
  cache 预算一致，无额外常驻分配。

CMP170HX 启动脚本默认打开
`VLLM_DEEPSEEK_V4_DSPARK_KV_ONLY_INSERT=1`；全局环境默认仍为 0，可在启动时显式设为
0 回滚。该路由只覆盖 `fp8_ds_mla` 的 `uint8` cache，BF16 和 per-tensor FP8 路径保持
原实现。

KV-only projection 也完成了隔离原型，但没有晋升。由于完整 WQA|WKV 已重排为 Marlin
packed 布局，无法安全做输出列零复制切片；原型为三个 draft layer 各保留一份独立 FP8
WKV。`20260815T002603Z` 的 audit 证明独立投影与融合投影的 KV 切片 BF16 逐元素完全
一致。相同 stage whitelist 的 candidate/baseline 为 `20260815T002910Z` /
`20260815T003159Z`：

- prefill input GEMM 为 97.72/209.05 us/层，context-KV 为 459.01/790.48 us，
  proposal 为 4.695/5.025 ms；
- decode input GEMM 为 17.04/15.48 us/层，context-KV 为 120.02/115.26 us，
  proposal 为 2.867/2.844 ms；
- PP2 workload 峰值由 63,112 增至 63,128 MiB。

只在大 M 使用独立 WKV 可避开 decode 退化，但 32K 的 TTFT Event scope 上限约为
20 x 0.330 ms = 6.6 ms，且这部分与 target prefill 流水重叠。该收益不足以交换 16 MiB
常驻显存，因此原型代码已撤回。若以后 Marlin 提供 packed-N 子视图或单份权重的 shard
GEMM，再重新评估零常驻增量实现。
