# DeepSeek-V4 CMP170HX decode 分解计量与优化研究

## 结论

本轮在三张 CMP170HX、PP3、DSpark-7、32K 上下文下，将 decode 分成两种互补口径：

- CUDA Graph 生产口径负责回答真实服务延迟。q8 target 的 PP0、PP1、PP2 串行
  replay 为 30.643 ms/轮，端到端中位数为 81.450 ms/轮。
- eager CUDA Event 口径负责回答模块构成。其 q8 target 串行时间为
  508.385 ms/轮，是 Graph 的 16.6 倍，因此只能用于同轮模块排序和同构层对比，
  不能作为生产延迟或收益预测。

研究优先级为：

1. P0：C4 的 Q/indexer/compressor/sparse-MLA 链路。C4 attention 每层为
   6.939 ms，比 C128 的 3.774 ms 慢 83.9%，而 FFN 只慢 0.6%。
2. P0：Graph 前的 attention metadata。生产路径最慢 rank 为 7.843 ms/轮，
   已是 target replay 之外最大的可见 decode 阶段。细分后两个 SWA group 合计
   2.957 ms/轮，是 builder 中最大的同类 parent。SWA 与 C128 direct-ragged
   均已通过 32K A/B/A/B 和 64K 稳定性门，提升为 CMP170HX 默认路径。
3. P1：MoE q8 路径内的真实暴露尾部。MoE parent 很大，但 shared experts 的
   join 只占 target model 的 0.3%，不能把其完整 aux-stream compute 当成可直接
   回收的 wall time。
4. P1：attention output projection。它在三种压缩族中都约为 1.1 ms/层，
   占诊断模型时间 9.4%；只研究低常驻显存的 FP8/融合方案。
5. P2：两处 mHC pre-op，共占 8.4%；top-k、shared activation 和 shared join
   均不是当前首要对象。

## 测量边界

| 项目 | 配置 |
| --- | --- |
| 模型 | `/media/z/data6/models/DeepSeek-V4-Flash-0731` |
| GPU | 3 x NVIDIA CMP170HX，SM80，PP3/TP1 |
| 层分配 | PP0/PP1/PP2 = 15/15/13 |
| 上下文与输出 | 32,000 prompt tokens，64 output tokens，3 repeats |
| 推测解码 | DSpark-7；target round 最多携带 8 tokens |
| 服务参数 | `max_num_batched_tokens=1715`，`max_num_seqs=8` |
| eager 证据 | `profiles/deepseek_v4/20260814T113235Z` |
| Graph 证据 | `profiles/deepseek_v4/20260814T114134Z` |
| P0 子阶段证据 | `profiles/deepseek_v4/20260814T130930Z` |
| direct-ragged 32K A/B/A/B | `20260814T141928Z` / `142242Z` / `142526Z` / `142831Z` |
| direct-ragged 64K A/B | `20260814T143153Z` / `143455Z` |
| direct-ragged detail | `profiles/deepseek_v4/20260814T143824Z` |
| direct-ragged 去 dense 32K | `profiles/deepseek_v4/20260814T144832Z` |
| direct-ragged 去 dense 8192-token warmup | `profiles/deepseek_v4/20260814T145153Z` |
| C128 direct-ragged 32K A/B/A/B | `20260814T150145Z` / `150508Z` / `150748Z` / `151048Z` |
| C128 direct-ragged 64K A/B | `20260814T152000Z` / `151702Z` |
| C128 direct-ragged detail | `profiles/deepseek_v4/20260814T152322Z` |

CMP170HX 上 CUPTI 返回 `CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED`。因此本轮没有
kernel timeline，也不从 PyTorch CPU scope 推断 GPU kernel 时间。模块细分使用
CUDA Event；Graph replay 和端点数据用于生产口径。两次运行均通过 workload 和
phase timing 验收，服务结束后已清理全部 worker。

## Decode 执行链

一轮 target verification 的依赖链为：

```text
model inputs / attention metadata
  -> PP0 target(q8)
  -> PP1 target(q8)
  -> PP2 target(q8)
  -> final norm / head / target logits
  -> DSpark rejection

每层 target
  -> mHC attention pre
  -> attention
       input GEMM overlap
       -> Q norm
       -> C-family Q/index/compressor/MLA
       -> output projection(inv_rope, wo_a, wo_b)
  -> mHC FFN pre
  -> MoE
       routed experts on current stream
       || shared experts W13 -> activation -> W2 on aux stream
       -> shared join -> combine
```

PP rank 在 prefill 中可形成流水重叠，但同一 autoregressive target pass 必须经过
PP0、PP1、PP2。因此 decode target 的模型时间应按对齐轮次求三段之和；单 rank
最大值只用于判断 stage balance。旧的“跨 rank 一律取 max”口径会把整轮模型工作
误报为约 11 ms，已在分析器中纠正。

## Graph 生产路径

| 指标 | mean | p50 | p90 |
| --- | ---: | ---: | ---: |
| PP0 q8 replay | 10.982 ms | 10.961 ms | 11.348 ms |
| PP1 q8 replay | 10.419 ms | 10.402 ms | 10.891 ms |
| PP2 q8 replay | 9.242 ms | 8.530 ms | 11.478 ms |
| PP3 串行 q8 replay | 30.643 ms | 29.850 ms | 33.486 ms |
| 最慢 rank attention metadata | 7.843 ms | - | - |
| 最慢 rank model inputs | 0.305 ms | - | - |

本轮共捕获 81 个 q8 target rounds。PP0 在 61 轮中最慢，PP1 为 4 轮，PP2 为
16 轮。端点三次 `ms_per_round` 为 79.997、82.214、81.450 ms，中位数
81.450 ms；对应 acceptance rate 为 0.2500、0.1939、0.2308。端点值还包含
DSpark draft/rejection、调度和通信，不能与 target replay 简单相减来归因。

Graph 将层内算子收进 replay，内部 stage marker 不可见。因此 Graph 报告的
`detail_complete=false` 是预期行为，不是验收失败；生产 A/B 应以 replay 和端点
数据为主。

## Eager 层级分解

以下占比以 508.385 ms/target round 为分母，仅用于诊断排序。父子范围有嵌套，
同一列不得相加。

| 范围 | 每 target round | 模型占比 | 解释 |
| --- | ---: | ---: | --- |
| 43 层总计 | 499.285 ms | 98.2% | 模型主要成本 |
| attention modules | 227.174 ms | 44.7% | 当前最大一级模块 |
| FFN modules | 205.372 ms | 40.4% | 主要由 MoE parent 构成 |
| MoE parent | 197.497 ms | 38.8% | 含 routed/shared 并行和嵌套范围 |
| Q/index/compressor/MLA parent | 124.262 ms | 24.4% | C4 差距的主要来源 |
| attention output projection | 47.557 ms | 9.4% | 压缩族无关的固定成本 |
| mHC attention pre | 21.297 ms | 4.2% | 小 M、逐层重复 |
| mHC FFN pre | 21.189 ms | 4.2% | 小 M、逐层重复 |
| sparse MLA parent | 30.636 ms | 6.0% | 含 metadata 和 attention |
| input GEMM overlap | 22.932 ms | 4.5% | 与其他 stream 有重叠 |

层内 sibling 的覆盖率为 95.1%；三个 PP rank 的模型层级覆盖率为
98.2%--98.7%。这说明 attention、FFN、mHC 四个 sibling 已覆盖主要层耗时。
不能再把它们的子范围加到 parent 上。

## C1/C4/C128 对照

下表是每层、每 q8 round 的 eager CUDA Event 均值。

| 压缩族 | 层数 | layer total | attention | FFN | mHC-attn | mHC-FFN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 2 | 9.281 ms | 2.995 ms | 4.755 ms | 0.486 ms | 0.493 ms |
| C4 | 21 | 13.284 ms | 6.939 ms | 4.792 ms | 0.495 ms | 0.493 ms |
| C128 | 20 | 10.088 ms | 3.774 ms | 4.762 ms | 0.496 ms | 0.493 ms |

C4 相对 C128：layer total 慢 31.7%，attention 慢 83.9%，但 FFN 只慢
0.6%。因此差距不是层位置或 MoE 波动，而是 C4 attention 特有路径。

### C4 attention 子范围

| 范围 | 每 C4 层 q8 | 关系 |
| --- | ---: | --- |
| Q/index/compressor/MLA parent | 4.417 ms | 首要 parent |
| indexer select parent | 1.002 ms | 含 logits/top-k |
| paged logits | 0.389 ms | select child |
| top-k | 0.117 ms | select child，仅全模型 0.5% |
| query prepare | 0.318 ms | 与 compressor 分支重叠 |
| compressed-cache update | 0.582 ms | 与 query 分支重叠 |
| sparse MLA parent | 0.975 ms | 含 metadata/attention |
| top-k metadata | 0.294 ms | sparse MLA child |
| sparse attention | 0.402 ms | sparse MLA child |
| output projection | 1.108 ms | C1/C128 同样约 1.1 ms |

P0 不应从 top-k 开始：它只占 target model 的 0.5%，paged logits 也只有 1.6%。
后续定点 profile 已补齐 stream join：C4 层的 input-GEMM join、QICM join 和
indexer query/cache join 分别只有 42.6、35.5 和 24.0 us。三处合计仍不到
0.11 ms/层，不能解释 4.374 ms 的 Q/index/compressor/MLA parent。当前 C4 的
真实首要对象是 2.190 ms 的 indexer branch，其次才是 0.931 ms 的 output
projection；继续减少已被隐藏的 stream 尾部不会产生主要收益。

## P0 attention metadata 深拆

`20260814T130930Z` 只记录 `model.layers.16` 的层内 detail，但 metadata 不受层
过滤，因此完整覆盖 PP0/PP1/PP2 的五个 KV cache group。三次 32K 请求均通过
phase/workload 验收。PP1 是本轮 metadata 最慢 rank：

| 范围 | 调用/target round | CUDA Event/次 | 每轮合计 | builders 占比 |
| --- | ---: | ---: | ---: | ---: |
| `target.attn_metadata` | 1 | 7.583 ms | 7.583 ms | - |
| `prepare_attn` | 1 | 7.323 ms | 7.323 ms | - |
| `builders` | 1 | 7.057 ms | 7.057 ms | 100.0% |
| SWA group 1 | 1 | 1.501 ms | 1.501 ms | 21.3% |
| SWA group 2 | 1 | 1.456 ms | 1.456 ms | 20.6% |
| MLA group 0（C4 + C128） | 2 | 0.878 ms | 1.755 ms | 24.9% |
| indexer group 0 | 1 | 0.532 ms | 0.532 ms | 7.5% |
| compressor group 3/4 | 3 | 0.151/0.216 ms | 0.519 ms | 7.4% |
| 五次 common metadata | 5 | 约 0.024 ms | 0.119 ms | 1.7% |
| 未独立标记的 Python/launch 间隙 | - | - | 1.175 ms | 16.7% |

最后一行是 parent residual，不是一个可直接优化的 kernel。细粒度 CUDA Event 会
增加 host enqueue 间隙，因此子阶段用于排序，不把它们的绝对值外推为生产收益；
7.583 ms 的 parent 与独立 Graph 口径的 7.843 ms 接近，确认 metadata 本身是
生产 P0，而不是 eager-only 假象。

### 两个 SWA group 的内部构成

PP1 每轮调用两次同构 SWA builder；下表把两次调用相加：

| 子阶段 | 每轮合计 | 两个 SWA parent 占比 | 判断 |
| --- | ---: | ---: | --- |
| ragged pack（clamp/cumsum/pack） | 0.520 ms | 17.6% | 最大已标记子阶段 |
| token-to-request map | 0.365 ms | 12.3% | 两组重复构造 |
| dense SWA indices/lens | 0.285 ms | 9.7% | 为 ragged 再物化 dense |
| ragged persistent-buffer copy | 0.255 ms | 8.6% | Graph 地址稳定所需的二次写入 |
| validity map | 0.191 ms | 6.5% | 两组重复构造 |
| tile scheduler holder | 0.075 ms | 2.6% | 低优先级 |
| prefill-common fields | 0.031 ms | 1.0% | decode 中可忽略 |
| parent residual | 1.235 ms | 41.8% | Python、分配和 launch 间隙混合 |

优化前链路为：

```text
dense SWA indices + lengths
  -> clamp lengths
  -> cumsum 得到 indptr
  -> dense-prefix pack 到临时 ragged
  -> copy 到 CUDA Graph 固定地址 ragged buffer
```

对 CMP170HX Triton decode 而言，attention 在 ragged 指针存在时不会消费 dense
indices；dense 只是 builder 的中转格式。现实现增加 SM8x 专属的 two-stage
direct-ragged：第一阶段用一个 Triton program 生成 lengths/indptr，第二阶段从
block table 把 slot id 直接写入既有 persistent ragged buffer。它删除 dense
materialization、临时 ragged 分配和二次 copy，且不增加常驻显存；实际 ROCm 和
非 SM8x 设备继续走原路径。

### Direct-ragged 晋级结果

CMP170HX 隔离 q8 测量中，完整 causal 链为 `236.4 -> 81.9 us`，non-causal
DSpark 链为 `240.2 -> 81.3 us`，分别为 2.89x 和 2.96x。32K 无 profiler
Graph A/B/A/B 每组五次：

| 路径 | 样本 | mean | p50 | 范围 |
| --- | ---: | ---: | ---: | ---: |
| control A1+A2 | 10 | 34.137 ms | 34.556 ms | 32.348--35.652 ms |
| direct B1+B2 | 10 | 33.578 ms | 34.150 ms | 31.374--34.797 ms |
| direct 相对 control | - | -1.64% | -1.17% | - |

TTFT 未改变，workload peak 显存在 A/B 中逐 MiB 一致。64K 三次 A/B 的 p50 为
33.331/33.262 ms；candidate 有一次 35.436 ms 异常值，因此 64K 只判定稳定且
无显存回退，不宣称提速。三次 32K detail 中所有 PP rank 的旧
`ragged_pack/ragged_copy` 事件均为零；新 `direct_ragged` 覆盖 PP0/PP1 的 causal
和 PP2 的 causal/non-causal 路径。PP1/PP2 SWA builder 从旧基线约
1.48/1.42 ms/次降到 1.17/1.17 ms/次。

CMP170HX 启动脚本默认设置 `VLLM_DEEPSEEK_V4_DIRECT_RAGGED_SWA=1`，设置为
`0` 可立即回退。平台无关环境默认仍关闭，避免把 SM80 结果外推到 ROCm。

更进一步，当前 `max_num_batched_tokens=1715` 时，一个
`[1715,4096] int32` dense decode buffer 为 26.8 MiB；两个 SWA group 共
53.6 MiB/rank。direct builder 现在通过构造期 capability 把这块 causal dense
buffer 的宽度设为零，同时继续避免 DSpark 首次 non-causal build 的 lazy dense
allocation；`=0` 回退时仍分配并使用原缓冲。该变更不先分配再释放，也不增加
workspace。

32K 三卡实机冒烟 `20260814T144832Z` 完成 64-token decode，物理 workload peak
为 58272/57462/63112 MiB；相对去除 causal dense 前的 58274/57464/63116 MiB
只下降 2--4 MiB。说明 CUDA allocator/Graph pool 复用了释放空间，因此只认定
逻辑分配消除和物理峰值不增加，不把 53.6 MiB 理论容量当作 `nvidia-smi` 实测
降幅。

`max_num_batched_tokens=8192` 时本次额外消除的 causal dense 上限为
256 MiB/rank，但 `20260814T145153Z` 仍在 PP2
`compress_norm_rope_store_triton` mixed-token warmup OOM，物理峰值为
60718/59424/64384 MiB。它不足以改变 8192 不可用的结论；后续容量专题应处理
compressor/warmup 峰值，而不是继续挤 metadata 小缓冲。

### MLA metadata 子阶段

MLA group 0 每轮依次构建 C128 和 C4。各 rank 的子阶段范围如下：

| 子阶段 | 每次 CUDA Event | 判断 |
| --- | ---: | --- |
| C128 ragged pack | 0.258--0.277 ms | C128 最大明确子阶段 |
| C128 ragged copy | 0.116--0.123 ms | 可并入 caller-owned direct 输出 |
| C128 global metadata | 0.178--0.186 ms | 保留独立优化对象 |
| C128 compressed slot mapping | 0.148--0.167 ms | 与 C4 算法相同但输出不同 |
| C4 compressed slot mapping | 0.150--0.154 ms | 可审计复用，不能指针别名 |
| C4/C128 token map | 0.038--0.198 ms | 跨 rank 波动大，只作次要对象 |

C4 与 C128 的 compressed slot mapping 虽然输入逻辑接近，但 consumer 需要不同
压缩率和各自固定 Graph 地址，不能共享输出指针。只允许 caller buffer 内的独立
写入或显式 copy。

### C128 direct-ragged 晋级结果

C128 原路径先生成 padded dense global slot ids，再做 clamp/cumsum/pack 和固定地址
copy。新路径保留原 position、压缩 block-table 和 invalid-token 语义，用 two-stage
Triton 直接生成 lens/indptr 与 global slot ids；prefill local indices 仍走原 kernel。
SM80 直接路径在构造期把未消费的 dense decode buffer 宽度设为零，独立环境开关
关闭时完整恢复原路径。

32K q8 隔离完整链为 `464.4 -> 159.9 us`，即 2.91x。无 profiler Graph
A/B/A/B 每组五次：

| 路径 | 样本 | mean | p50 | 范围 |
| --- | ---: | ---: | ---: | ---: |
| control A1+A2 | 10 | 35.484 ms | 35.518 ms | 34.854--35.843 ms |
| direct B1+B2 | 10 | 34.956 ms | 34.889 ms | 34.413--35.912 ms |
| direct 相对 control | - | -1.49% | -1.77% | - |

64K 有效 control/direct 各三次，mean 为 36.963/37.000 ms，p50 为
37.174/36.961 ms，只判定稳定，不宣称提速。第一次 64K control
`20260814T151353Z` 出现一次 PP1 asynchronous illegal memory access，但紧接着的
同配置三次全部通过，因此记为未复现异常，不归因给任一路径，也不纳入性能统计。

detail `20260814T152322Z` 中三个 PP rank 的旧 C128 `ragged_pack/ragged_copy`
均为零，`direct_ragged` 为 217--223 us/次，C128 parent 为 322--329 us/次。
`max_num_batched_tokens=1715`、部署长度 131K 时消除的 dense buffer 为
6.7 MiB/rank；8192-token 上限时为 32 MiB/rank。32K 和 64K 的物理采样均为
PP0/PP2 不变、PP1 低 48 MiB，但 allocator/Graph pool 会放大逻辑变化，故不把
48 MiB 全部归因给该 tensor。

CMP170HX 启动器默认设置 `VLLM_DEEPSEEK_V4_DIRECT_RAGGED_C128=1`，`=0` 可独立
回退。平台无关默认仍为关闭。

### 已否决的 P0 原型

| 原型 | 局部结果 | Graph 服务 A/B | 结论 |
| --- | --- | --- | --- |
| C4/C128 compressed slot pointer alias | 第五次请求触发 PP1 illegal memory access | 未通过稳定性门 | 永久否决跨 builder 指针别名 |
| 共享计算后 copy 到各自 caller buffer | PP0 MLA builder 491.170 -> 468.569 us/次 | 35.778 -> 35.770 ms/round（-0.02%） | 收益落入噪声，已移除 |
| caller-owned ragged pack，删除末尾 copy | PP0/PP1 SWA builder -14.4%/-12.5%，PP2 基本持平 | 36.009 ms；两个 control 为 35.778/35.495 ms | 局部赢但 E2E 退化 0.6%--1.4%，已移除 |

最后一个实验的 A/B/A 各跑五次，A1、B、A2 的 p50 分别为 35.701、36.010、
35.407 ms/round。说明单独改变 allocator/copy 所得收益不够稳定；下一原型必须把
`dense -> ragged -> copy` 整条链缩成 direct-ragged，不能再次提交同一局部改动。
以上三个未晋级原型的代码和开关均已撤回；direct-ragged 是唯一晋级路径，且没有
新增常驻 workspace。

## MoE q8 分解

| 范围 | 每 target round | 模型占比 | MoE parent 占比 |
| --- | ---: | ---: | ---: |
| MoE parent | 197.497 ms | 38.8% | 100% |
| routed experts parent | 93.118 ms | 18.3% | 47.1% |
| modular kernel | 61.624 ms | 12.1% | 31.2% |
| shared experts compute | 37.850 ms | 7.4% | 19.2% |
| shared experts join | 1.483 ms | 0.3% | 0.8% |
| routed W13 | 11.659 ms | 2.3% | 5.9% |
| routed W2 | 8.160 ms | 1.6% | 4.1% |

`shared experts compute` 在辅助 stream 上运行并与 routed experts 重叠；真正暴露在
主 stream 上的是 join 尾部。join 只有 34.4 us/层，说明 shared branch 当前基本
被隐藏。优化 shared W13/W2 的 isolated latency 不等于能回收同等 E2E 时间。

Marlin W13/W2 仍是 MoE parent 的嵌套诊断，不应和 modular kernel、routed parent
相加。当前 parent remainder 还包含 custom-op 内部工作和跨 stream overlap；它不
是“未知 kernel”总量。P1 应先增加 routed/shared 起止关系和 Graph 外最小重放，
再基于暴露尾部选择 q8 shape，而不是沿用 prefill 的大 M `thread_k/thread_n` 表。

## Output projection 与显存约束

隔离 M=8、FP8 group-128、CMP170HX 测量如下：

| shape | FP8 Marlin | cached BF16 | 差距 | 本 rank 常驻代价（15 层） |
| --- | ---: | ---: | ---: | ---: |
| wo_b 8x8192x4096 | 55.78 us | 49.18 us | FP8 慢 13.4% | 960 MiB |
| shared W13 8x4096x4096 | 56.03 us | 30.10 us | FP8 慢 86.2% | 480 MiB |
| shared W2 8x2048x4096 | 55.69 us | 22.25 us | FP8 慢 150.3% | 240 MiB |

三项均通过 FP8/BF16 数值对照。BF16 只是延迟上界，不是候选实现：仅 `wo_b`
就在 PP0、PP1 各增加 960 MiB，换取每次 isolated GEMM 约 6.6 us；shared W13/W2
再增加 720 MiB/rank，而 shared branch 又没有显著暴露尾部。因此继续拒绝任何
per-layer persistent BF16 cache。

output projection 的后续方向限定为：M=8 FP8 Marlin shape 调优、caller-owned
短生命周期 workspace、`inv_rope + wo_a` 同步边界削减，以及不增加大块常驻权重
的融合。所有方案必须记录每 rank peak/profile memory；不能用上下文容量换速度。

## 优化与验收顺序

### P0-A：C4 同步尾部定位

1. record/wait 已测明不是主尾部；保留 marker，停止继续微调 join。
2. 对 2.190 ms 的 indexer branch 分开 paged-logits tile、selection 和 cache update，
   在 2K、8K、32K、64K 上下文扫描 q8/q6。
3. 只对 Graph 中仍暴露的 indexer 阶段做融合或 SM80 Triton 重写。
4. output projection 保持 P1，除非 indexer 优化后它升为关键路径。

### P0-B：attention metadata

1. SWA two-stage direct-ragged 已晋级，保留 `=0` 回滚和固定 Graph 地址。
2. causal dense decode buffer 已在构造期跳过；8192 仍在 compressor warmup OOM。
3. C128 direct-ragged 已晋级并跳过 dense decode buffer；slot mapping 保持独立输出。
4. 继续区分 CPU 构造、GPU kernel 和 launch gap；CMP170HX 无 CUPTI 时用 CPU scope
   加 CUDA Event 双口径，并以无 profiler Graph A/B 作为晋级门。

### P1-A：MoE q8 暴露路径

1. 测 routed parent 与 shared join 的真实关键路径，而不是累计两个 stream。
2. 为 M=8 的 routed modular kernel 建独立 shape 基线。
3. 只有 W13/W2 位于关键路径时才调 `thread_k/thread_n/blocks_per_sm`。
4. W13+activation 和 W2+reduce 仍遵守既有 CTA 边界结论，不做简单 epilogue 拼接。

### P1-B：output projection 与 mHC

1. 建 M=8 FP8 shape 表，优先降低 launch 和 workspace 成本。
2. 研究 `inv_rope + wo_a` 的低显存融合；`wo_b` 保持 FP8 权重。
3. 对两处 mHC pre-op 做 q8 专用 TileLang/SM80 microbenchmark。

每个候选必须同时通过：原实现数值对照、q8/q6、32K 与至少一个更长上下文、
CUDA Graph 预热后不少于 5 次服务 A/B、DSpark acceptance/output hash 审计、每 rank
显存增量记录。禁止以 eager marker 路径的绝对时间宣称生产收益。
