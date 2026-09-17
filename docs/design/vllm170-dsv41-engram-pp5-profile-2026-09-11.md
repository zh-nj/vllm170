# DeepSeek-V4.1-Flash PP5 Prefill / Engram 分解计量

## 结论

本轮在 5 张 CMP170HX、PP5/TP1、Engram 全量主存驻留下，完成了 1,000
prompt tokens / 256 output tokens 的端点基线、eager CUDA Event 分解，以及
GPU 反量化 P0、head-major staging P1 的 A/B/A/B 验证。

- 无 profiler 的端点基线为 TTFT 688.4 ms、decode 3.495 tok/s。
- 稳态 prefill 中，每张含 Engram 的 PP rank、每个 488--512 token chunk 的
  `engram.lookup_total` 平均为 33.610 ms。CPU 反量化为 21.723 ms，占 lookup
  64.6%；主存随机 gather 为 9.583 ms，占 28.5%。
- 以两段流水线的实际 stage 时间做反事实调度，Engram lookup 占计算关键路径
  12.62%，完整 Engram 链占 13.13%。只去掉 CPU 反量化的理论上限约为
  55.2 ms，即稳定端点 TTFT 的 7.8%。最终实现还同时消除了高级索引产生的
  pageable gather 临时张量，所以端点收益可以超过“只去掉反量化”的反事实上限。
- decode 的五段串行 target model 为 252.274 ms/token；细粒度计量中的完整
  Engram 链约 4.0 ms/token，仅占 1.6%。attention 为 103.636 ms/token，
  是 decode 的首要模块；最热层集中在 checkpoint 声明的 index source layers。
  对这 8 个 index-source 层做低扰动细分后，attention 子集为 58.156 ms/token；
  可观测叶子中 `indexer.paged_logits`、`sparse_mla.attention`、
  `sparse_mla.index_convert`、`attention.indexer_query` 分别为 4.088、4.240、
  3.021、4.590 ms/token。这些数值只覆盖 whitelist 层，不能当作 38 个压缩层
  的全模型合计。
- 当前 8/8/8/8/8 分层不是 prefill 的首要问题。约束每段 7--9 层枚举后，
  最佳候选只降低 0.2% 的两-chunk 计算关键路径。单请求 decode 又必须串行通过
  五段，单纯平衡各 rank 最大值不会等比例降低 token latency。
- P0 已实现并通过反向顺序复测。丢弃每轮首个样本后，control/candidate 合并
  TTFT p50 为 666.8/596.9 ms，缩短 69.9 ms（10.48%）；1k/256 decode p50
  为 3.500/3.505 tok/s，未出现回退。
- P0 候选路径的稳定 `engram.lookup_total` p50 为 5.125 ms/chunk/rank，比原路径
  33.223 ms 降低 84.6%（6.48x）。P1 再把 staging 改为 head-major，随机
  gather mean 由 3.742 降至 3.413 ms，lookup mean 由 5.149 降至 4.813 ms；
  两-chunk 关键路径只再缩短 0.71 ms，说明 Engram 已不值得引入高复杂度优化。
- P2 已实现 decode top-k 转换缓存。模型原来在 38 个压缩 attention 层逐层重复
  local-to-global 转换；按 index-source 和 PP 边界失效后，PP5 每 token 只需 9 次。
  真实 C2/64、C1/128 块大小微基准均约 0.202 ms/次，理论去掉约 5.9 ms/token。
  profiler 关闭的 1k/256 候选三次为 3.681/3.622/3.592 tok/s（median 3.622），
  相比重启前最新 P1 单次 3.527 tok/s 为 +2.68%；因主机在对照复测前重启，
  该端点比较是跨启动证据，不宣称为同轮因果 A/B。
- P3 已恢复 SM80 sparse decode 的 caller-output 直写。CMP170HX 目标 kernel
  bit-exact 微基准从 0.3266 降到 0.2567 ms（每层节省 0.0727 ms）；端点
  1k/256 三次为 3.734/3.588/3.604 tok/s（mean/median 3.642/3.604），与
  旧代码同启动的 3.632/3.622 tok/s 相比仅在噪声内变化，因此只判定无回退。
- FULL decode graph 把 1k/256 decode 提升到约 19.5 tok/s 后，eager profile 中
  `attention.indexer` 尚未解释的 5.6 ms/token 已不再是生产热点证据。CMP170HX
  不支持 Nsight CUDA trace；candidate select 与四次 consumer mask 的 Event
  微基准上限约 0.65 ms/token，因此本轮没有继续改 candidate kernel。
- PP5 调度采用 `max-num-batched-tokens=2048`、`long-prefill-token-threshold=256`
  和 adaptive long prefill。1K prefill 中位 TTFT 为 591.6 ms，32K prefill-only
  中位为 6.522 s；32K/256 decode 中位为 19.550 tok/s。固定 256 虽把 1K 降至
  559.3 ms，却把 32K 拉长到 9.951 s，不能作为全局默认。
- 精确 1K adaptive shape 的 FFN 细分确认 W13/W2 GEMM 占 modular kernel
  81.7%，是 prefill 的第一优化对象。把 `M<=1024` 的 SM80 MXFP4-int8 tile 从
  `16x64` 改为 `16x128` 后，五个代表 shape 的 W13+W2 Event p50 合计改善
  4.9%--6.2%；1K/2 TTFT 中位由 591.6 降至 555.6 ms（-6.10%），1K/256
  TTFT 中位由 657.4 降至 629.5 ms（-4.25%），decode 无回退。

因此优化顺序为：

1. P0：保留完整表在主存，仅把 gather 后的 FP8 行和 ue8m0 scales 放入小型
   pinned staging buffer，传到 GPU 后用 Triton 融合反量化和写入/累加。
2. P1：优化主存随机 gather；先量唯一行比例，再决定是否去重，并评估直接写入
   pinned output 的 CPU gather。NUMA 本地性已较好，不是当前主因。
3. P2：复用同一 index-source 分组的转换结果；每个 DBO ubatch 独立，模型 forward
   开始和 index-source 更新后失效。该项已实现并完成单元、微基准和端点验证。
4. P4：按当前 `mxfp4_int8_sm80` shape 拆 FFN，并用逐元素一致性、精确 shape
   CUDA Event 和端到端 A/B 调 tile；`M<=1024` 的 block-N 128 已通过门禁并保留。
5. 下一步：继续分析 W13/W2 kernel 本身；CMP170HX 不支持 Nsight CUDA trace，
   使用 CUDA Event、生成代码检查和端到端 A/B，不用旧 eager 父段残差指导改动。
6. 只有在并发流水吞吐成为目标、并确认单层显存后，再实测 PP 重分层。

## P0 实现与 A/B/A/B

本轮随后实现了上述 P0。全局环境默认仍关闭，通过
`VLLM_DEEPSEEK_V4_ENGRAM_GPU_DEQUANT=1` 启用；PP5 启动器默认打开，设置为 `0`
可回退原 CPU dequant 路径。每个 Engram rank 预分配两个复用槽，共
6.19 MiB pinned host staging 和 6.19 MiB device staging。槽内 DMA 和 GPU kernel
完成前由 CUDA event 保护，CPU 不会提前覆写仍在使用的 pinned buffer。

相同 prompt、关闭 profiler、1,000-token warmup 后，每轮执行 5 次两-token
请求。首个样本仍有 warmup 尾部，比较均丢弃首个样本：

| 顺序 | 路径 | TTFT 后四次范围 | TTFT p50 |
| --- | --- | ---: | ---: |
| A1 | CPU dequant control | 665.7--673.5 ms | 668.4 ms |
| B1 | GPU dequant candidate | 594.0--597.2 ms | 596.1 ms |
| A2 | CPU dequant control | 661.6--675.4 ms | 662.3 ms |
| B2 | GPU dequant candidate | 593.1--611.6 ms | 599.9 ms |
| A1 + A2 | CPU dequant control，n=8 | 661.6--675.4 ms | 666.8 ms |
| B1 + B2 | GPU dequant candidate，n=8 | 593.1--611.6 ms | 596.9 ms |

合并 p50 缩短 69.9 ms（10.48%），均值缩短 68.7 ms（10.30%）。A2/B2
反向顺序与 A1/B1 结论一致，排除了单向测试顺序带来的主要偏差。

完整 1k/256 各三次中，decode throughput p50 为 3.500/3.505 tok/s，candidate
相对 control 为 +0.15%，属于持平；candidate 三次输出 SHA256 一致，并与 control
的稳定输出哈希一致。

隔离 lookup 在物理 GPU0 的 24-head 生产形状上保持 bit-exact。该测试使用
1,048,576-row（0.258 GiB）普通主存表，因此只验证路径和趋势，不代表 94 GiB 表：

| tokens | CPU dequant p50 | GPU dequant p50 | speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.651 ms | 0.484 ms | 1.35x |
| 488 | 8.700 ms | 1.078 ms | 8.07x |
| 512 | 7.590 ms | 1.206 ms | 6.29x |

完整 A/B/A/B 原始记录位于 `/tmp/dsv41-gpu-dequant-ab-20260911`，候选分阶段
trace 位于 `/tmp/dsv41-gpu-dequant-profile-20260911-b`。

### 候选分阶段结果

分阶段服务只启用 Engram 和 `target.model_forward` 的 CUDA Event，关闭
`record_function`。丢弃首个请求的两个 prefill chunks 后，PP0/PP1 共 16 个稳定
事件如下：

| stage | mean/chunk/rank | p50 | 范围 | lookup mean 占比 |
| --- | ---: | ---: | ---: | ---: |
| `engram.lookup_total` | 5.149 ms | 5.125 ms | 4.557--5.679 ms | 100.0% |
| `host_ram_gather_raw` | 4.236 ms | 4.204 ms | 3.699--4.748 ms | 82.3% |
| `host_ram_gather` | 3.742 ms | 3.730 ms | 3.283--4.181 ms | 72.7% |
| `indices_to_cpu` | 0.312 ms | 0.303 ms | 0.251--0.380 ms | 6.1% |
| `raw_h2d` | 0.532 ms | 0.533 ms | 0.489--0.582 ms | 10.3% |
| `gpu_dequant` | 0.032 ms | 0.031 ms | 0.026--0.042 ms | 0.6% |
| `staging_wait` | 0.026 ms | 0.024 ms | 0.024--0.038 ms | 0.5% |

旧路径与候选路径使用相同 512/488-token chunk 形状。lookup p50
33.223 -> 5.125 ms，降低 84.6%；随机 gather mean 9.583 -> 3.742 ms，除去
反量化迁移外，`index_select(..., out=pinned_staging)` 还消除了高级索引产生的
pageable 临时张量。候选路径稳定两-chunk 计算关键路径中，完整 lookup 只剩
10.23--11.04 ms/request，平均 10.62 ms；下一轮继续优化 Engram 的收益上限已经
显著收窄。

### P1 head-major staging

每个 `(n-gram size, head)` 使用互不相交、约 16M rows 的 bucket。P0 按 token-major
次序 gather，会在 24 个相距约 4 GiB 的 bucket 之间逐行跳转；P1 只把 pinned
staging 的源行改为 head-major，并在 Triton kernel 中映射回原 token-major 输出，
不改变输出布局或模型结果。

先对同一个精确 1k prompt 离线重放 hash，两个 512/488-token chunk 的每层唯一率
分别为 93.75% 和 94.67%。重复主要来自 bigram；trigram 唯一率为
98.57%--99.02%，4-gram 为 99.80%--100%。因此不采用需要排序和 inverse map 的
deduplicate 路径。

在 3.094 GiB、24-head 模拟表上，单线程、轮换随机 index sets 的 gather p50 从
11.455 降至 3.441 ms（3.33x）。真实 94.42 GiB 表的稳定 trace 收益明显更小：

| stage | P0 token-major mean | P1 head-major mean | 变化 |
| --- | ---: | ---: | ---: |
| `engram.host_ram_gather` | 3.742 ms | 3.413 ms | -8.8% |
| `engram.host_ram_gather_raw` | 4.236 ms | 3.899 ms | -8.0% |
| `engram.lookup_total` | 5.149 ms | 4.813 ms | -6.5% |
| 两-chunk lookup 关键路径 | 10.616 ms | 9.904 ms | -6.7% |

这说明小表冷访问微基准会高估地址分组收益，必须以完整表 trace 为准。head-major
只涉及一个 CPU 排列和一个 GPU 源行映射，保留该低复杂度收益；不继续实现全行
去重或专用线程池。

关闭 profiler 的正常服务上，P1 重启后连续执行十次 1k/2 请求。丢弃第一个冷请求，
九次 TTFT mean/median 为 594.635/589.641 ms；相比 P0 A/B/A/B 的稳定样本
598.494/596.934 ms，分别降低 3.859/7.292 ms（0.64%/1.22%）。方向与阶段
trace 一致，但端点差值大于 Engram trace 的 0.712 ms/request，说明其中包含服务
重启和系统抖动；因此只把阶段 trace 用作 head-major 的因果收益，把端点结果作为
无回退检查，不将 7.292 ms 全部归因于该改动。

## 测量边界

| 项目 | 配置 |
| --- | --- |
| 模型 | `/media/z/data6/models/DeepSeek-V4.1-Flash` |
| GPU | physical GPU 1--5，5 x NVIDIA CMP170HX，PP5/TP1 |
| 层分配 | PP0--PP4 = 8/8/8/8/8 |
| Engram layers | 1、14，分别位于 PP0、PP1 |
| Engram table | 两表各约 94.42 GiB，完整驻留普通主存 |
| 输入/输出 | 精确 1,000 prompt tokens / 256 output tokens |
| 调度 | `max_num_batched_tokens=512`、`max_num_seqs=1` |
| 执行 | eager、prefix caching 关闭、无 speculative decoding |
| KV cache | FP8，`kv_cache_memory_bytes=1000000000` |
| 端点基线 | `/tmp/dsv41-baseline-1k256-20260911.txt` |
| 定向 profile | `/tmp/dsv41-profile-20260911` |
| 细分 profile | `/tmp/dsv41-profile-detail-20260911` |
| P2 attention profile | `/tmp/dsv41-attention-profile-20260911-p2` |
| P4 indexer profiles | `/tmp/dsv41-indexer-profile-20260912`, `...b`, `...c` |
| decode top-k cache candidate | `/tmp/dsv41-topk-cache-candidate-1k256.jsonl` |

细分 profile 只打开本轮新增的 model/layer/Engram stage，并关闭
`record_function`。CUDA Event 会同步并扰动短 decode 路径，因此无 profiler
端点负责生产速度，细分结果只负责结构归因和同轮排序。

早期 profile 客户端写入 `result.jsonl` 的 `pp_size=3` 和
`max_num_batched_tokens=1715` 是当时的脚本缺省元数据，不是服务配置。实际启动命令、
五个 worker trace（PP rank 0--4）以及 prefill token count 512/488 共同确认该轮为
PP5 和 512-token chunk。客户端现已从 `DSV4_PP_SIZE`、`DSV4_TP_SIZE` 和
`DSV4_MAX_BATCHED_TOKENS` 记录实际值；P0 A/B/A/B 新记录均为 PP5/TP1/512。

## 端点结果与 profiler 扰动

| 口径 | TTFT | decode | 说明 |
| --- | ---: | ---: | --- |
| profiler 关闭 | 688.4 ms | 3.495 tok/s | 生产基线 |
| profiler 关闭，P1 head-major | 594.635 ms mean / 589.641 ms median（9 次稳态） | 仅测 2-token smoke | 无回退检查；相比 P0 稳态 p50 596.934 ms |
| profiler 关闭，最终 P1 1k/256 | 672.734 ms | 3.527 tok/s | 新增埋点关闭时的完整端点检查 |
| profiler 关闭，decode top-k cache | 652.612/672.502/685.672 ms | 3.681/3.622/3.592 tok/s | 三次候选；控制复测被主机重启中断 |
| 定向 profiler | 776.1 ms | 3.439 tok/s | 少量 stage marker |
| 细分 profiler，首次完整请求 | 873.3 ms | 3.196 tok/s | 含冷态/JIT 和密集 marker |
| 细分 profiler，后两次短请求 | 711.7 / 708.8 ms | 不使用 | 稳态 prefill 样本 |

细分 profiler 把 decode token interval 从基线约 286.1 ms 增至 312.9 ms，增加
9.4%。因此不能用 3.196 tok/s 判断生产回退，也不能把细分 stage 的绝对值直接
外推为 decode 收益。稳定 prefill 的后两次 TTFT 与基线只差约 3%，适合用于本轮
prefill 结构分析。

## Prefill 执行链

```text
input/token positions
  -> GPU n-gram hash
  -> 对当前 PP rank 的 Engram 表执行 prepare_embeddings
       GPU indices -> CPU int64 indices
       -> 普通主存随机 gather FP8 rows + ue8m0 scales
       -> CPU FP32 展开/乘 scale -> BF16
       -> BF16 H2D + staged_rows add
  -> 8 x decoder layer
       attention
       -> Engram layer: WKV + gate/mHC 注入
       -> FFN
  -> PP send/receive
```

lookup 在进入 decoder layer 循环前同步完成；`wkv` 和 `post_wkv` 位于对应
decoder layer 的 `layer.total` 内。因此下表中的 parent/child 不能重复相加。

### 稳态 Engram 分解

样本为后两次请求的四个 prefill chunks（step 260/261/263/264），PP0 与 PP1
各一张表，共 8 个事件。

| stage | mean/chunk/rank | p50 | 范围 | lookup 占比 |
| --- | ---: | ---: | ---: | ---: |
| `engram.lookup_total` | 33.610 ms | 33.223 ms | 31.567--36.157 ms | 100.0% |
| `host_ram_gather_dequant` | 31.899 ms | 31.480 ms | 29.844--34.469 ms | 94.9% |
| `host_dequant` | 21.723 ms | 20.724 ms | 20.489--24.612 ms | 64.6% |
| `host_ram_gather` | 9.583 ms | 9.631 ms | 8.730--10.439 ms | 28.5% |
| `indices_to_cpu` | 0.300 ms | 0.290 ms | 0.269--0.339 ms | 0.9% |
| `h2d_add` | 1.435 ms | 1.441 ms | 1.329--1.481 ms | 4.3% |
| `wkv` | 1.354 ms | 1.353 ms | 1.299--1.412 ms | lookup 外 |
| `post_wkv` | 0.056 ms | 0.056 ms | 0.051--0.059 ms | lookup 外 |
| `hash_ids` | 0.149 ms | 0.148 ms | 0.142--0.162 ms | lookup 外 |

这些 host stage 的 CPU wall mean 与 CUDA Event mean 相差不超过 0.034 ms（例如
`host_dequant` 为 21.758/21.723 ms），确认这里测到的是同步 host 阻塞，而不是
GPU 队列中无关工作的假归因。

首次完整 profile 的 lookup 曾达到约 50--70 ms/chunk，后续相同 prompt 的稳定
p50 为 33.223 ms。优化基线采用稳定样本，首次样本仅作为冷态尾延迟证据。

### PP stage 构成

以下是每个请求两个 chunk 的时间之和，再对后两次稳定请求取均值。

| PP rank | model forward | layer total | attention | FFN | Engram lookup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 255.195 ms | 185.020 ms | 66.290 ms | 102.302 ms | 68.396 ms |
| 1 | 241.789 ms | 174.162 ms | 67.765 ms | 90.776 ms | 66.044 ms |
| 2 | 169.873 ms | 169.151 ms | 67.313 ms | 89.027 ms | - |
| 3 | 179.667 ms | 178.669 ms | 72.294 ms | 92.766 ms | - |
| 4 | 180.419 ms | 179.202 ms | 78.036 ms | 88.122 ms | - |

去掉 lookup 后，PP0/PP1 分别约为 186.8/175.7 ms，与后三段的
169.9--180.4 ms 接近。当前 prefill stage 不平衡主要由 Engram lookup 产生，
不是 8 层均分本身。

全部五段的 layer work 中，FFN 为 463.0 ms/request，attention 为
351.7 ms/request，分别占 layer total 的 52.2% 和 39.7%。完成 Engram P0 后，
FFN 是下一项 prefill 聚合热点；单层中 layer 1、32、8、24 最热。

### 两-chunk 关键路径反事实

对每个请求，以 `target.model_forward` 的实测 stage duration 按标准 PP 依赖递推：

```text
end[chunk, rank] = max(end[chunk - 1, rank], end[chunk, rank - 1])
                   + duration[chunk, rank]
```

两次请求的计算关键路径为 627.898/623.602 ms，均值 625.750 ms。与稳定端点
711.7/708.8 ms 的差额包含调度、PP 通信、采样、HTTP 和未标记间隙。

| 反事实 | 关键路径减少 | 计算关键路径占比 | 稳定端点上限 |
| --- | ---: | ---: | ---: |
| 去掉完整 Engram 链 | 82.1 ms | 13.13% | 11.6% |
| 去掉 lookup | 79.0 ms | 12.62% | 11.1% |
| 去掉 CPU dequant | 55.2 ms | 8.82% | 7.8% |
| 去掉 host gather | 27.7 ms | 4.43% | 3.9% |
| 去掉 BF16 H2D/add | 4.3 ms | 0.68% | 0.6% |

这些是把该 stage 时间直接置零得到的上限，不是收益预测。若只替换 CPU 反量化，
raw H2D 和新 kernel 会吃掉部分 55.2 ms；实际 P0 还把高级索引 gather 改成直接
写 pinned staging，因此合并端点收益达到 69.9 ms。

## Engram 主存与 NUMA 证据

转换文件的每行由 256-byte FP8 weight 和 8-byte ue8m0 scale 组成：

| 表 | rows | 文件总量 | 所在 worker RSS |
| --- | ---: | ---: | ---: |
| layer 1 / PP0 | 384,006,168 | 94.415 GiB | 96.708 GiB |
| layer 14 / PP1 | 384,016,682 | 94.418 GiB | 96.716 GiB |

两表原始数据合计 188.833 GiB。其余模型 worker、engine、API server、页表和系统
进程使整机当前达到约 225 GiB used、22 GiB available。系统配置为 2 GiB swap，
本轮未修改 swap；profile 期间曾为 0 KiB，恢复最终服务后系统 swap 重新接近满载，
但 PP0/PP1 worker 自身仅分别约 49/13 MiB `VmSwap`，主要 swap 使用来自既有系统进程，
不是 Engram 表被换出。

物理 GPU1/PP0 位于 NUMA0，物理 GPU2--5/PP1--PP4 位于 NUMA1。worker 的 CPU
affinity 已按 GPU 本地节点放置，实际匿名页分布为：

| worker | NUMA0 | NUMA1 | 本地页比例 |
| --- | ---: | ---: | ---: |
| PP0 | 86.022 GiB | 10.694 GiB | 88.9% |
| PP1 | 8.316 GiB | 88.407 GiB | 91.4% |

因此跨 NUMA 页约占 9%--11%，有优化余量，但它只能影响 9.583 ms 的 gather 子段，
无法解释 21.723 ms 的 CPU dequant。两个大 worker 的 `AnonHugePages` 都为 0；
huge-page/madvise 只能作为 gather 的独立 A/B，不能先于 GPU dequant。

## Decode 分解

完整 1k/256 细分 trace 中，原始请求的 decode step 为 2--256，共 255 个间隔；
后续三个短请求的 step 259/262/265 已排除。

| 范围 | 五段串行时间/token | target model 占比 |
| --- | ---: | ---: |
| target model forward | 252.274 ms | 100.0% |
| layer total | 239.427 ms | 94.9% |
| attention | 103.636 ms | 41.1% |
| FFN | 77.207 ms | 30.6% |
| Engram lookup | 2.738 ms | 1.1% |
| Engram lookup + hash + WKV + post | 4.025 ms | 1.6% |

这组绝对值受细粒度 marker 扰动，只用于结构排序。低扰动定向 profile 中完整
Engram 链约为 3.42 ms/token，占串行模型约 1.4%，结论一致：Engram 不是 decode
首要热点。

| PP rank | model forward/token | attention/token | FFN/token |
| ---: | ---: | ---: | ---: |
| 0 | 38.701 ms | 14.655 ms | 11.663 ms |
| 1 | 52.488 ms | 21.631 ms | 15.112 ms |
| 2 | 49.857 ms | 20.675 ms | 15.757 ms |
| 3 | 52.024 ms | 21.770 ms | 16.485 ms |
| 4 | 59.204 ms | 24.906 ms | 18.190 ms |

PP4 是最慢 stage，但当前单请求 decode 必须串行执行五段，总和而不是最大 rank
决定 token latency。stage 最大值只在多 microbatch/多请求流水吞吐下成为直接瓶颈。

最热 decode layers 为：

| layer | total/token | attention | FFN |
| ---: | ---: | ---: | ---: |
| 32 | 10.014 ms | 5.371 ms | 2.797 ms |
| 20 | 8.586 ms | 5.078 ms | 1.997 ms |
| 8 | 8.326 ms | 4.842 ms | 2.104 ms |
| 36 | 8.174 ms | 4.283 ms | 2.175 ms |
| 24 | 8.155 ms | 4.386 ms | 2.300 ms |
| 14 | 8.033 ms | 4.131 ms | 1.746 ms |

checkpoint 的 `index_source_layer_ids` 为 `[2, 8, 14, 20, 24, 28, 32, 36]`；
上表六层全部属于该集合。下一轮 decode 计量应在这些层内部继续拆
indexer/select、compressed-cache update、sparse MLA 和 output projection，而不是
继续细拆 Engram。

## 优化方案

### P0（已完成）：GPU 反量化，完整 Engram 表仍只驻留主存

当前 CPU 会把 gather 后的 FP8 行先扩展成 BF16，再传输到 GPU。每个 token、每张
Engram 表有 24 个 hash heads：

- raw weight + scales：`24 * (256 + 8) = 6,336 bytes/token`；
- BF16 rows：`24 * 256 * 2 = 12,288 bytes/token`；
- 512-token chunk 分别为 3.094 MiB 和 6.000 MiB，raw H2D 少 48.4%。

已实现链路为：

```text
普通主存完整表
  -> gather raw uint8 rows/scales 到复用的 pinned host staging（约 3.1 MiB/slot）
  -> async raw H2D 到小型 device staging
  -> Triton FP8 + ue8m0 dequant
  -> 直接写入或累加 staged_rows
```

该方案不把 Engram 表放入显存，只增加 MiB 级暂存；也不 pin 两张 94 GiB 大表。
实现位于可回退开关后，已完成逐元素数值一致性、无 profiler A/B/A/B 和候选
分阶段计量。

### P1（head-major 已完成）：主存 gather

GPU dequant 后，head-major 将稳定随机 gather 进一步降至 3.413 ms/chunk，
但它仍是主要 Engram 子段。本轮结论为：

1. 行唯一率 93.75%--94.67%，不做 deduplicate + inverse map。
2. `index_select(..., out=pinned_buffer)` 和 head-major 已完成，避免 pageable 临时
   tensor，并减少跨 bucket 地址跳转。
3. PP0/PP1 大表 VMA 分别仍有约 12.9%/11.1% 页位于远端 NUMA node；即便完全
   消除，作用范围也只有当前 3.413 ms gather，不应优先于 decode attention。
4. 暂不修改系统 NUMA/THP 策略，也不为不足 10 ms/request 的剩余 Engram 关键路径
   引入常驻 CPU 线程池。

### P1：decode index-source attention（已完成计量）

本轮对 checkpoint 的 8 个 `index_source_layer_ids`（2、8、14、20、24、28、32、36）
启用 whitelist，每层累计 255 个 decode step。P2 trace 的五段 target model 为
256.566 ms/token；该绝对值含 CUDA Event marker 扰动，只用于同一轮内部排序。
选中层的 attention parent 合计 58.156 ms/token，约占 target model 的 22.7%；
父段不能和子段相加，下面的叶子只用于定位工作量。尤其
`sparse_mla.index_convert` 和 `sparse_mla.attention` 实际在全部 38 个压缩层运行，
表中数值只是 8 个 whitelist 层的合计：

| stage | 选中层合计 ms/token | 说明 |
| --- | ---: | --- |
| `attention.q_index_compress_mla` | 37.212 | Q/index/compressor/MLA 的联合父段 |
| `attention.input_gemm_overlap` | 9.334 | 三路输入投影和 stream join 的父段 |
| `attention.indexer_prepare_cache` | 6.999 | index query 与压缩 cache 准备父段 |
| `attention.sparse_mla` | 9.695 | sparse MLA 父段 |
| `attention.output_projection` | 4.867 | inverse-RoPE / `wo_a` / `wo_b` |
| `attention.indexer_query` | 4.590 | indexer query 构造 |
| `indexer.paged_logits` | 4.088 | decode paged MQA logits |
| `sparse_mla.attention` | 4.240 | sparse MLA decode kernel |
| `sparse_mla.index_convert` | 3.021 | local top-k 到 paged global index |
| `indexer.topk` | 1.292 | top-k select |
| `attention.qkv_insert` | 2.855 | Q projection / KV insertion |
| `attention.input_gemm.qr_kv` | 2.439 | fused WQA/WKV GEMM |
| `attention.compressor` | 0.809 | 4 个 index-source 层（C-ratio=1/2） |
| `attention.compressed_cache_insert` | 0.550 | 4 个 index-source 层（C-ratio=1/2） |

按 compression ratio 聚合时，C-ratio=1 的 5 层 attention 平均 7.584 ms/layer，
C-ratio=2 的 3 层平均 6.745 ms/layer；按层排序的最大三项是 layer 32
(8.966 ms)、layer 20 (8.347 ms)、layer 36 (7.389 ms)。PP4 的 layer 32 和 36
合计约 16.35 ms/token，是 source-layer attention 最重的 rank，但单请求仍是五段
串行，不能把最大 rank 时间直接当成单请求 latency。

### P2：decode top-k 转换缓存（已实现）

源码追踪确认，local top-k 只在 index-source 层更新；该 source 后面的 consumer
层会重复使用同一 top-k 和 compressed cache block table。原实现仍在每个压缩层
执行 lens、cumsum、local-to-global pack，并分配三个中间 tensor。按 8/8/8/8/8
分层统计：

| PP rank | 压缩层数 | 缓存后转换次数 |
| ---: | ---: | ---: |
| 0 | 6 | 1 |
| 1 | 8 | 2 |
| 2 | 8 | 2 |
| 3 | 8 | 2 |
| 4 | 8 | 2 |
| 合计 | 38 | 9 |

PP2 必须为从 PP1 输入的 source 14 再转换一次，因为 global slot id 对应本 rank 的
compressed cache block table，不能直接跨 PP 复用。因此正确口径是 38 -> 9，
而不是 38 -> 8。

服务所用的物理 GPU1（CMP 170HX，SM80）上用 1 decode token、top-k=512、生产
物理块大小做 CUDA Event 微基准，先以 Torch reference 逐元素校验输出，再执行
30 次 warmup 和 200 次计量：

| compression | block size | p50 | p90 | mean |
| --- | ---: | ---: | ---: | ---: |
| C2 | 64 | 0.2017 ms | 0.2284 ms | 0.2060 ms |
| C1 | 128 | 0.2017 ms | 0.2273 ms | 0.2029 ms |

去掉 29 次重复转换的微基准上限约 5.9 ms/token。实现通过
`VLLM_DEEPSEEK_V4_DECODE_TOPK_CACHE` 控制：全局默认关闭，PP5 启动器默认开启；
缓存按 DBO ubatch 隔离，在每次 model forward 开始及每个 index-source 更新后失效。
只有同一 source group 内、block size 和 decode batch shape 相同的 consumer 才命中。

正确性单元测试覆盖 ubatch 隔离、source 失效和 ROCm/SM80 共用 backend 的实际
`_forward_decode` 复用路径。profiler 关闭的候选结果如下：

| workload | 样本 | 结果 |
| --- | ---: | --- |
| 1k/2 | 3 | TTFT 704.495/588.533/591.284 ms；后两次 mean 589.908 ms |
| 1k/256 | 3 | decode 3.681/3.622/3.592 tok/s；median 3.622 tok/s |

最新 P1 未缓存完整请求为 3.527 tok/s，候选 median 提高 2.68%，折算 decode
latency 从 283.5 降至 276.1 ms/token，差约 7.4 ms/token。该量级与微基准的
5.9 ms/token 接近，但控制复测开始前主机于 2026-09-12 06:12 重启，旧服务和
`/tmp` control 文件消失，因此这里只把结果记为跨启动性能证据；不能替代同轮
A/B/A/B。

### P3：恢复 SM80 sparse decode 直接输出（已实现）

源码和历史追踪发现，`VLLM_DEEPSEEK_V4_SPARSE_DECODE_CALLER_OUTPUT=1` 虽然仍是
默认值，并被启动器文档和测试声明为启用，但后续 gfx950 优化把实际条件收窄为
`_ON_GFX950`。因此 SM80 一直让 Triton kernel 写临时 BF16 tensor，再逐层复制到
调用方 output；该开关在 CMP 170HX 上事实上失效。

修复后，gfx950 行为保持不变；仅 CUDA SM8x 在开关开启且 output 为 BF16 时把
调用方 tensor 直接传给 kernel。物理 GPU1 上以生产 decode 形状（1 token、64 heads、
SWA 128 + extra 512）比较相同 kernel，输出逐元素完全一致：

| 路径 | p50 | p90 | mean |
| --- | ---: | ---: | ---: |
| 临时 tensor + copy | 0.3205 ms | 0.3564 ms | 0.3266 ms |
| 直接写 output | 0.2478 ms | 0.2806 ms | 0.2567 ms |

微基准节省 0.0727 ms/layer；38 个压缩层的理论上限约 2.76 ms/token。现有 split
路径也在同一目标卡复核：C2/64 从 0.3215 降至 0.2365 ms，C1/128 从 0.3174
降至 0.2273 ms，因此保留 split，不调整 tile。回归测试覆盖 q=1/2/4/8、参考
输出以及 CUDA Graph 捕获/回放，并断言 SM80 kernel 实际收到调用方 output。

重载后、profiler 关闭的 PP5 端点结果：

| workload | 样本 | 结果 |
| --- | ---: | --- |
| 1k/2 | 3 | TTFT 675.333/594.714/594.329 ms；median 594.714 ms |
| 1k/256 | 3 | 3.734/3.588/3.604 tok/s；mean 3.642，median 3.604 tok/s |

旧代码虽然配置记录为 caller-output 开启，实际仍走临时 tensor；其同次主机启动的
三次 1k/256 mean/median 为 3.632/3.622 tok/s。修复后 mean 提高 0.29%，median
降低 0.48%，都远小于单请求波动，端点证据只能判定为无可见回退，不能声称显著
提速。保留修复的依据是目标 kernel 的 bit-exact 稳定收益、去除逐层分配和复制，
以及恢复已有开关的真实语义。若需要端点级因果量化，应增加样本并做交替重启 A/B。

### P4：indexer 父段边界与 candidate block 计量（已完成）

为避免把父段和子段相加，在同一 PP5 配置下对 8 个 index-source 层各采集 255 个
decode step；profile 使用 CUDA Event、关闭 Torch `record_function`，因此绝对值只
用于内部排序。第二轮只增加 `attention.indexer` 父段，第三轮再增加
`indexer.candidate_blocks`，并在每次重载后确认两张 Engram 表仍为普通主存驻留。

第三轮选中层合计（按 255 个输出 token 归一化）如下：

| stage | 合计 ms/token | 口径 |
| --- | ---: | --- |
| `attention.indexer` | 13.102 | indexer_op 父段，含 metadata/候选处理 |
| `indexer.paged_logits` | 4.566 | 8 个 source 的 paged MQA logits |
| `indexer.q_projection` | 2.100 | 8 个 source 的 index Q projection |
| `indexer.q_rope_quant` | 1.760 | RoPE + FP8 quant |
| `indexer.topk` | 1.416 | persistent top-k |
| `indexer.candidate_blocks` | 1.536 | layer 20/24/28/32/36，均值约 0.307 ms/source |
| `attention.input_gemm.indexer_weights` | 1.625 | 与主 QKV 投影并行，不能加到父段关键路径 |
| `indexer.k_projection` + `k_norm_rope_store` | 1.396 | 仅 4 个 ratio-2 K-owning source |

candidate 明细为 layer 20/24/28/32/36 分别 0.409/0.274/0.242/0.344/0.267
ms/call；因此两级 candidate 选择/掩码是本轮新增分段中最大的单个子项。paged logits
仍是最大单叶子，但在 CMP170HX 上对 1--512 个 CTA 扫描 2/4 warps 和 1--4 stages
后，最佳配置随 CTA 数反复切换，差异通常在 ±1--2 个 Event 量化单位；8 warps
还出现 0.001953 BF16 规约差异。当前不能安全替换 A100 上的默认 4 warps/2--4
stages。

`attention.indexer` 父段 13.102 与已量到的 paged logits、top-k、candidate
合计 7.518 ms/token 相差约 5.6 ms/token；Q projection/quant 不在该父段内。
剩余部分由 `indexer_op` 中尚未分段的 metadata、buffer/边界处理和 eager-break/
stream 调度组成，尚未证明存在可删除的重复计算。下一轮如需继续，只应单独给这些
边界操作加 stage 或用 nsys/CUPTI 分 stream，不应先改 kernel 配置。

更新后的优先级：

1. `sparse_mla.index_convert` 的主要结构性浪费已通过 source-group 缓存消除；
   单次 kernel 约 0.2 ms，不再优先做复杂融合。
2. `sparse_mla.attention` 已恢复 SM80 直接输出且当前 split 明显更快；现有数据不
   支持继续调整 split/tile。
3. `attention.indexer` 是当前已量到的最大 index-source 父段；candidate block
   约 1.536 ms/token，paged logits 约 4.566 ms/token，Q projection+quant 约
   3.860 ms/token。先用 nsys/CUPTI 分清 metadata/eager-break 差额，再决定是否
   优化 candidate mask 或 paged cache 访问。
4. `attention.input_gemm_overlap` 是 9.334 ms 的父段，但其子 GEMM 仅 2.439 ms
   （`qr_kv`），剩余部分包含 aux-stream 调度和等待，不能据父段直接推导 GEMM
   优化收益；需要 nsys/CUPTI 级 stream 关系后再改。
5. `indexer.topk`、compressor 和 Engram decode 各自低于 1.3 ms/token，不作为
   下一轮首选热点。

新增 stage 覆盖真实 SM80 调用链：

```text
Q/index/compressor parent
  -> input GEMM / query / compressed-cache update
  -> indexer paged logits / top-k
  -> sparse MLA index conversion / attention
  -> output projection
```

端点仍以 profiler 关闭的 1k/256 重复请求为准。细分 profiler 只判断 parent 内部
排序，不能用其绝对时间宣称生产收益。原始 trace 位于
`/tmp/dsv41-attention-profile-20260911-p2`。

### 暂不优先：PP 重分层

基于稳定逐层时间，枚举每段 7--9 层的连续划分：

- 最佳 prefill 候选 `7/8/8/9/8` 或 `7/8/9/8/8` 仅比 8 层均分降低 0.2%；
- decode 最大 stage 的理论最佳候选 `9/8/8/8/7` 可降低 9.0%，但不会等比例降低
  单请求五段串行 latency，并且 PP0 当前只剩约 6.8 GiB 显存；
- 较保守的 `8/8/9/8/7` 也需两个中间 rank 各增加一层，最大 stage 只降约 2%。

在没有逐层权重/运行时显存预算和并发 workload 前，不应为这点理论收益承担 OOM
风险。

## 2026-09-12：FULL decode graph 与 2048-token 调度上限

项目启动器已把 `max-num-batched-tokens` 默认值从 512 调到 2048，并请求
`cudagraph_mode=FULL`。当前 `DeepseekV41ROCMAiterSparseSWABackend` 只声明
`UNIFORM_BATCH` 支持，因此 vLLM 自动把运行模式收敛为
`FULL_DECODE_ONLY`：单请求 decode 使用 FULL graph，prefill 不进入 FULL graph。

普通主存 Engram lookup 包含 GPU hash 回读、CPU gather、H2D 和主机同步，不能在
单体 CUDA Graph 内捕获。为此，ModelState 在 FULL graph 捕获和每次回放前先准备
当前 token 的 Engram 行，图内只读取固定 `staged_rows`。真实 PP5 重载结果：

- PP0--PP4 均完成 FULL graph 捕获；图内存分别约 0.07/0.08/0.07/0.07/0.07
  GiB，没有再出现 `operation not permitted when stream is capturing`；
- PP0/PP1 的两张 94.42 GiB Engram 表仍完整驻留普通主存；2048-token 配置使每个
  Engram rank 的双缓冲增至 24.75 MiB pinned host + 24.75 MiB device；
- `/health` 返回 200；3-token 与 16-token decode 请求均成功，第二个请求之后没有
  新增 CUDA Graph、Engram 或运行时异常。

本次运行日志位于
`/tmp/dsv41-full-graph-mbt2048-20260912/server-after-outside-staging.log`，服务保持在
`http://127.0.0.1:26662`。

### 32K 输入 / 256 输出验收

在上述服务上先执行一次精确 32,000-token / 8-token 预热，再以同一个精确
32,000-token prompt 重复三次固定 256-token 流式请求。prefix cache 关闭，
`ignore_eos=true`；TTFT 是请求发出到首个非空 token，decode 速率为首 token 后
255 个 token 除以对应流式时间。`prompt_tokens` 和 `completion_tokens` 均取服务端
usage：

| 样本 | TTFT | 有效 prefill | decode | 总耗时 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 6.817 s | 4,693.9 tok/s | 19.208 tok/s | 20.093 s |
| 2 | 6.305 s | 5,075.6 tok/s | 19.505 tok/s | 19.378 s |
| 3 | 6.359 s | 5,032.0 tok/s | 20.054 tok/s | 19.075 s |
| mean | 6.494 s | 4,933.8 tok/s | 19.589 tok/s | 19.515 s |
| median | 6.359 s | 5,032.0 tok/s | 19.505 tok/s | 19.378 s |

这里的有效 prefill 吞吐按 `32000 / TTFT` 计算，包含 HTTP、排队和首 token，属于
端点口径，不是纯 kernel 吞吐。

长上下文语义使用放在 50% 深度的唯一 recovery-code needle，并保持精确 32,000
输入 token。同一固定 256 输出请求在 6.290 s 给出首 token、有效 prefill
5,087.2 tok/s、decode 19.892 tok/s；答案首先正确给出 `jade-orbit-7319`。由于固定
长度请求故意忽略 EOS，正确答案后的强制续写不作为语义质量判断。独立的正常 EOS
严格检查返回且只返回 `RECOVERY=jade-orbit-7319`，因此语义验收通过。

原始结果：

- `/tmp/dsv41-fullgraph-32k256-20260912.jsonl`
- `/tmp/dsv41-fullgraph-32k256-needle-20260912.json`
- `/tmp/dsv41-fullgraph-32k-semantic-strict-20260912.json`

测试结束后服务 `/health` 仍为 200，服务日志没有新增 CUDA、Engram、graph capture
或 worker 异常。

## 2026-09-12：FULL graph 热点复核与自适应 prefill chunk

### 放弃旧 eager indexer 残差作为 P0

旧 profile 的 `attention.indexer=13.102 ms/token` 来自 eager + CUDA Event
分段；当时完整请求 decode 只有约 3.6 tok/s。启用 FULL decode graph 后，正常
1k/256 已达到约 19.5 tok/s，即整个 token 约 51 ms。因而旧父段减去子段得到的
5.6 ms/token 不能跨执行模式外推为当前生产热点。

尝试用 Nsight Systems 对 FULL graph 做 CUDA graph node trace，服务请求仍达到
19.601 tok/s，但报告只有 NVTX、没有 CUDA API/kernel 数据。独立 `cuda-sw`
smoke 明确返回 `ERR_NVCMPGPU`：Nsight CUDA tracing 不支持 NVIDIA CMP。该机器
后续应继续使用 CUDA Event、精确 shape 微基准和端到端 A/B，不能依赖 nsys/ncu
counter。

物理 GPU1 上用生产 decode shape 复测 candidate block：candidate select 在
32K/64K cache width 的 p50 为 0.168--0.183 ms，consumer mask p50 为
0.108--0.123 ms。按一个 source select 加四个 consumer mask 估算，总上限约
0.65 ms/token，不足以支持继续改 candidate kernel。本轮因此不做源码变更。

原始文件：

- `/tmp/dsv41-nsys-fullgraph-20260912/fullgraph.nsys-rep`
- `/tmp/nsys-cuda-sw-smoke.nsys-rep`
- `/tmp/dsv41-nsys-fullgraph-20260912/workload.jsonl`

### 1K 固定 chunk sweep

相同精确 1,000-token prompt、固定 seed、prefix cache 关闭、无 profiler；prefill
每组 7 次，decode 每组 3 次：

| max batched tokens | prefill TTFT mean | prefill TTFT median | decode mean | decode median |
| ---: | ---: | ---: | ---: | ---: |
| 200 | 626.1 ms | 617.6 ms | 19.135 tok/s | 19.156 tok/s |
| 256 | 577.0 ms | 559.3 ms | 20.162 tok/s | 20.152 tok/s |
| 512 | 597.5 ms | 590.3 ms | 20.544 tok/s | 20.396 tok/s |

`256` 把 1K 输入分为接近四个均衡 chunk，是固定上限中的最佳值。`200` 的五个
chunk 已明显更慢；`224` 同样需要五个 chunk，`192` 需要六个，因此不再为这两个
被夹住的点各付一次全模型/Engram 重载成本。

但固定 `256` 的 32K prefill-only 三次 TTFT 为 11.349/9.950/9.951 s，median
9.951 s；相对 2048 基线 median 6.359 s 回退 56.5%，不能写成全局默认。

### 自适应 256 -> 2048

现有 scheduler 已提供 contention-aware adaptive long prefill。最终配置保持
workspace/budget 上限 2048，将前期 PP 在途 chunk 限制为 256；PP5 单请求在前
五个 chunk 后若没有竞争，后续自动使用完整 2048 budget：

```text
--max-num-batched-tokens 2048
--long-prefill-token-threshold 256
--enable-adaptive-long-prefill
```

端点结果：

| workload | 样本 | TTFT mean | TTFT median | decode mean | decode median |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1K / 2 | 7 | 598.0 ms | 591.6 ms | - | - |
| 1K / 256 | 3 | 648.3 ms | 657.4 ms | 19.111 tok/s | 19.450 tok/s |
| 32K / 2 | 3 | 6.636 s | 6.522 s | - | - |
| 32K / 256 | 3 | 6.654 s | 6.738 s | 19.577 tok/s | 19.550 tok/s |

此前同日、确认服务端为固定 2048 的 1,024/256 control 中位 TTFT 为 887.5 ms、
decode 为 19.560 tok/s。prompt 相差 24 token，不能把 TTFT 差值当作严格同 shape
A/B，但 1K 约 33% 的收益远大于该输入差异；decode 则持平。32K prefill-only
相对固定 2048 的 6.359 s 只回退 2.6%，完整 32K/256 TTFT 的差异约 6%。

内容级门禁使用 24,659-token needle prompt，正确返回
`quantum-lighthouse-4471`。固定 seed 的输出字节哈希在当前量化 CUDA 路径下跨
请求并不稳定，因此哈希仅用于发现异常，不作为语义等价判据。

原始结果位于 `/tmp/dsv41-chunk-sweep-20260912/`。最终启动器将 adaptive 作为
默认，可用 `DSV41_ENABLE_ADAPTIVE_LONG_PREFILL=0` 显式关闭，并可通过
`DSV41_LONG_PREFILL_TOKEN_THRESHOLD` 调整阈值。

### Prefill FFN 细分与 MXFP4-int8 tile

在 adaptive 配置上开启 eager CUDA Event 阶段计量，只记录 prefill，使用精确
1,000-token prompt、3 个请求；每个请求为 `256+256+256+232` 四个 chunk。五个
PP rank 共得到 7,200 条记录，每个新增 stage 都有 480 条，即
`3 requests x 4 chunks x 40 layers`，覆盖完整。

关键 PP0 上每个 stage 有 96 次调用。`layer.ffn` 合计 522.824 ms，
`moe.routed_experts` 为 429.083 ms，`moe.modular.kernel` 为 392.856 ms。其中：

| stage | total | modular kernel share |
| --- | ---: | ---: |
| W13 GEMM | 210.706 ms | 53.6% |
| W2 GEMM | 110.505 ms | 28.1% |
| W13 quant | 14.213 ms | 3.6% |
| align | 12.217 ms | 3.1% |
| activation + W2 quant + reduce | 4.965 ms | 1.3% |

因此 W13/W2 合计占 modular kernel 81.7%，应先调 GEMM，不应先融合十几到几十
微秒的尾部步骤。新增 benchmark 使用 DSV4.1 固定 TP1 shape
`H=5120, I=2304, E=256, topk=6`，随机初始化输入与权重，并在计时前逐元素比较
所有候选 tile 的 W13/W2 输出。物理 GPU1、20 次 warmup、100 次 Event 重复：

| M | 16x64 W13+W2 p50 | 16x128 W13+W2 p50 | 改善 |
| ---: | ---: | ---: | ---: |
| 232 | 5.786 ms | 5.500 ms | 4.9% |
| 256 | 5.890 ms | 5.586 ms | 5.2% |
| 512 | 6.525 ms | 6.140 ms | 5.9% |
| 1000 | 11.443 ms | 10.765 ms | 5.9% |
| 1024 | 11.652 ms | 10.926 ms | 6.2% |

所有 shape 的两个 tile 均 bit-exact，因此把静态表的 `M<=1024` 档从 `16x64`
改为 `16x128`。另测 block-M 16/64/128：232--1024 均为 16 最快，2048 为 64
最快，128 全程较慢；当前按平均 expert occupancy 限制 block-M 的逻辑保留。

无 profiler 端点门禁如下。1K/2 两边均为 7 次；旧 1K/256 为 3 次、新路径为
7 次，因此完整生成的百分比只用于判定无回退，不把它当成严格交替 A/B：

| workload | 旧 mean / median | 新 mean / median | median 变化 |
| --- | ---: | ---: | ---: |
| 1K/2 TTFT | 598.0 / 591.6 ms | 566.5 / 555.6 ms | -6.10% |
| 1K/256 TTFT | 648.3 / 657.4 ms | 633.4 / 629.5 ms | -4.25% |
| 1K/256 decode | 19.111 / 19.450 tok/s | 19.955 / 20.096 tok/s | +3.32% |

24,659-token needle 再次正确返回 `quantum-lighthouse-4471`，TTFT 5.985 s；服务
完成请求后健康。原始数据：

- `/tmp/dsv41-ffn-profile-20260912/`
- `/tmp/dsv41-moe-tile128-20260912/blockn-final.json`
- `/tmp/dsv41-moe-tile128-20260912/blockm-sweep.json`
- `/tmp/dsv41-moe-tile128-20260912/1k-2.jsonl`
- `/tmp/dsv41-moe-tile128-20260912/1k-256.jsonl`

## 本轮验收结果

1. correctness：GPU 路径在 1/7/256 tokens、前后台 grid 和双缓冲槽复用下逐元素
   对齐当前 BF16 结果；完整 Engram kernel suite 通过。
2. memory：两张完整表仍只在 PP0/PP1 主存；512-token 性能测量时每个 Engram
   rank 新增 6.19 MiB pinned host + 6.19 MiB device staging，当前 2048-token
   服务对应 24.75 MiB + 24.75 MiB，没有 GiB 级 Engram HBM 常驻，未改动 swap。
3. stage：稳定 `engram.lookup_total` p50 从 33.223 降至 5.125 ms/chunk，并分别
   记录了 gather、raw H2D、GPU dequant 和 staging wait。
4. endpoint：相同 prompt 和输出策略下完成无 profiler A/B/A/B；合并稳定 TTFT
   p50 改善 10.48%；新增 decode top-k cache 候选的 1k/256 median 为
   3.622 tok/s，但因主机重启只有跨启动 control。恢复 sparse decode 直接输出后，
   1k/256 mean/median 为 3.642/3.604 tok/s，端点波动大于预期收益，仅判定无回退；
   FULL graph + adaptive + block-N 128 的 1k/256 mean/median 已达到
   19.955/20.096 tok/s。
5. regression：全量主存新旧路径、SSD/FIFO、小 cache、TP shard 和无 offload
   原测试均通过；全局开关默认关闭，PP5 项目启动器默认开启并可显式回退。
6. service：最终无 profiler、top-k cache、sparse decode split/caller-output 和
   FULL decode graph 均开启的服务在 `http://127.0.0.1:26662` 健康；调度为
   `max-num-batched-tokens=2048`、adaptive threshold 256。完整 1K/256 和
   32K/256 请求以及长上下文 needle 均通过。

本轮最终验证为 `tests/kernels/test_engram.py` 74 passed，
`tests/test_envs.py` 60 passed，`tests/kernels/core/test_fused_q_kv_rmsnorm.py`
13 passed / 60 skipped；decode top-k cache 新增 3 个单元测试通过，和环境测试合并
为 63 passed。SM80 sparse decode 的 q=1/2/4/8、caller-output 开/关及 CUDA
Graph 回归为 8 passed。新增 profiler/cache/caller-output/candidate-block 调用链通过
ruff check、ruff format、compileall 和实际 PP5 1k/256 请求。
`tests/kernels/test_compressor_kv_cache.py`
全文件已有 67 passed / 17 skipped；其中 `test_cutedsl_full_cache_store[True-4]`
和 `[True-128]` 单独串行复跑仍在 NVVM `sm_80` 编译阶段失败，未进入 kernel
执行，属于现有 CuTeDSL/CUDA 工具链阻塞，不是本轮 Python 埋点引起。
`bash -n` 与 `git diff --check` 通过。
