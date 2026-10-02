# DSV41F CED prefill 第二轮：真实路由、MoE 细分与 tile 反例

本轮保持 CED 开启、DSpark 关闭，完成 MoE/attention 细分计时、跨 PP
时间线、真实路由捕获和一个已有 CUTLASS tile 的同实例 A/B。主要热点
确认在 routed MoE 的 W13/W2 GEMM。固定 `block_m=128` 在周期 32K 输入
上降低 TTFT **5.81%**，但在异质代码语料上增加 TTFT **4.38%**，因此
**不接入生产，不修改推理代码**。正式服务恢复原配置，health=200。

原始数据及可执行脚本：
`artifacts/dsv41f-ced-prefill-round2-20261002/`。
基线 HEAD 为 `4df3a185db133e63934d5739cc18582a215be609`；开始时的
22 个既有 dirty 文件逐个校验保留。诊断脚本与实际加载源码由
`launch.json` 保存 SHA256；`baseline-state.json` 保存原正式配置。

## 工作负载与计时边界

- 模型 `/media/z/data6/models/DeepSeek-V4.1-Flash`，SM80 CMP170HX，
  PP5/TP1/DP1，分区 `8,8,8,8,8`。V2 model runner，eager CED prefill。
- `max-num-batched-tokens=2048`、`max-num-seqs=8`，FP8 KV，2.7 GB/rank，
  prefix caching 开启；每次冷请求独立 `cache_salt`，metrics 命中增量为零。
- 保留已有 MoE TileLang、fused SWA metadata，target prepare-ahead 关闭，
  实际 argv 没有 speculative config；所有 speculative counters 增量为零。
- 第一类输入复用此前固定 32768 token 周期 prompt，仅 65 种 token；
  greedy、ignore EOS，输出 16 token，与前轮参考逐个比较。
- 第二类输入从七个项目源码文件构造代码归档，嵌入检索事实，截取为
  32768 token，共 **2521 种 token**。输出 64 token，与同实例 baseline
  逐个比较，并验证答案包含 `ced-tile128-orion-20261002`。
  输入、文件列表和 SHA256 见 `corpus-prompt.json`，不是通用质量评测集。
- TTFT 是客户端收到首个 token-ID chunk 的时间，包含调度、CED encoder、
  suffix replay、PP、采样和发送；不是单个 kernel 时间。
- 所有 A/B 使用同一私有 loopback 26662 实例，候选单独预热，正式 A/B
  不记录 CUDA Events。两个 arm 使用相同诊断 wrapper，只改变 tile 选择。
  不使用跨重启的时间差宣称优化收益。

## MoE 的实际热点

事件只在 eager pure prefill 时记录，请求结束后一次 RPC 读取；细分只覆盖
PP0–PP2，PP3/PP4 保留 model 父区间。子区间包含 GPU stream 上的 enqueue
空档与等待，不能全部当作纯 kernel 时间，也不能与父区间相加。

周期输入的 MoE profile TTFT 为 4.0852 s，以下为整请求累计区间，单位 ms：

| rank | model.total | moe.total | routed W13 | routed W2 | activation | W2 quant | router gate | router select |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PP0 | 3407.27 | 1756.69 | 878.52 | 468.23 | 15.01 | 8.17 | 52.19 | 10.58 |
| PP1 | 3279.97 | 1704.31 | 870.31 | 462.25 | 14.83 | 7.93 | 51.25 | 11.15 |
| PP2 | 1706.99 | 964.68 | 490.00 | 260.09 | 8.64 | 4.79 | 28.15 | 6.69 |

W13/W2 两段占 PP0/PP1 前向区间约 **40%**，占 MoE 区间约 **77%–78%**。
其中 PP0 W13 约为 W2 的 1.88 倍；router 和量化均明显小于 GEMM。
这比继续扩大上一轮大行 activation fusion 更值得优先处理。

代码语料同一 baseline 的 GEMM 区间显著增大：

| rank | model.total | moe.total | routed W13 | routed W2 |
| --- | ---: | ---: | ---: | ---: |
| PP0 | 4412.73 | 2804.67 | 1578.66 | 829.92 |
| PP1 | 4288.12 | 2710.35 | 1532.33 | 808.48 |
| PP2 | 2166.14 | 1433.87 | 801.72 | 423.62 |

在这个输入中，两段 GEMM 占 PP0/PP1 前向约 **55%**。输入内容会改变
expert 分布及实际工作量；固定 token 数不能代表固定 MoE 成本。

shared expert 的阶段存在 aux stream 重叠，`moe.shared_experts`、外层
module hook 和 main-stream routed 区间可能重叠。汇总不把它们的和当作
串行耗时；`routing_cast` 也不等于 router GEMM。

## attention fence：host 等待不能当作可删除成本

保留原 `_combine_prefill_indices` 中的 eager stream synchronize，只在诊断
中把它与 combine kernel 分开记录。host duration 用 perf counter，GPU
gap 用同一 stream 上包围同步点的 CUDA Events。原同步语义没有改变。

| rank | sync host wait ms | sync GPU gap ms | combine kernel ms |
| --- | ---: | ---: | ---: |
| PP0 | 2194.32 | 8.37 | 28.30 |
| PP1 | 2207.39 | 8.16 | 28.88 |
| PP2 | 1146.20 | 4.16 | 16.73 |

host 在同步点等待此前排队的 GPU 工作；GPU start event 也排在这些工作
之后。因而 2.2 s host wait 不能作为删除同步即可获得的 2.2 s 收益。
GPU gap 也不是纯同步指令成本，包含同步完成后的 host enqueue 间隙。
删除 fence 会改变后续排队和临时张量生命周期，本轮没有做这个实验。

attention profile 的 TTFT 为 4.1031 s，比一次无插桩 baseline 高 2.56%；
MoE profile 高 2.12%，root profile 高 0.13%。这些是单样本诊断比较，
不是插桩开销的成对因果测量。allocator 的诊断 peak allocated 为
PP0/PP1 约 62.47/60.90 GB，也不能替代删除 fence 后的峰值验证。

## PP 时间线

`prefill-timeline.svg` 和 PNG 使用各 rank 的 CPU-anchored CUDA Events。
anchor 的 host 包围区间最大 **0.0641 ms**；同机 perf counter 可跨进程
比较。图是近似对齐的区间证据，不是 Nsight 的逐 kernel trace。

![32K prefill PP 时间线](../../artifacts/dsv41f-ced-prefill-round2-20261002/prefill-timeline.svg)

五 rank 均有 22 个 prefill model 调用。结合上一轮已记录的 chunk 布局，
前六次为启动窗口，之后为 bulk，最后一次执行 decoder suffix replay。
本轮 root hook 的调用实参走 kwargs，`rows=null`；颜色按调用序号标注，
不把空字段伪装成直接测得的 token 数。

root profile 的 PP0/PP1 累计区间为 3309.72/3205.61 ms，PP2 为
1665.26 ms，PP3/PP4 为 85.56/80.37 ms。时间线显示 bulk 阶段 PP0
持续执行，PP1/PP2 随上游逐 chunk 前进，PP3/PP4 主要执行最后 replay。
从首个 model span 到最后一个结束约 **3.885 s**，客户端 TTFT 为
4.0056 s；差额包括 model span 外的准备与采样/交付。

不能把 rank 累计区间相加作为 prefill 延迟，也不能把启动窗口的全部
GPU 工作当作固定 overhead。本轮没有改 adaptive-long-prefill 的并发策略。

## 实验：已有 block-M 64 与 128

生产 selector 只按 `ceil(M*topk/E)` 选 block-M；在 `(2048,6,384)` 下
选择 64。已有预编译 CUDA kernel 同时支持 128，故无需重新编译。
私有候选只在 pure prefill 且精确匹配上述形状时返回 `(128,128)`；
decode、mixed、draft 和其他 M 均回原 selector。

周期输入初筛两组 AB/BA 改善 5.79%，随后四组交错 A/B：

| 配对 | baseline TTFT s | BM128 TTFT s | TTFT 降低 |
| --- | ---: | ---: | ---: |
| 0 | 4.01088 | 3.78775 | 5.56% |
| 1 | 4.01134 | 3.81122 | 4.99% |
| 2 | 4.03858 | 3.76879 | 6.68% |
| 3 | 4.01524 | 3.77283 | 6.04% |
| arm 中位数 | **4.01329** | **3.78029** | **5.81%** |

这个结果在固定周期输入上成立。用代码语料复核后出现相反结果：

| 配对 | baseline TTFT s | BM128 TTFT s | TTFT 增加 |
| --- | ---: | ---: | ---: |
| AB | 5.16334 | 5.39173 | 4.42% |
| BA | 5.15174 | 5.37556 | 4.34% |
| arm 中位数 | **5.15754** | **5.38364** | **4.38%** |

两组代码请求候选都退化，所有 64 个输出 IDs 一致且检索成功。周期
64K 单组探查为 7.48995→7.03556 s，TTFT 降低 6.07%；只有一组，不作为
独立验收结论。129/2049/2050-token fallback 两 arm IDs 一致，candidate
计数为零。这些通过不改变异质输入的失败结论。

## 路由与 padding：为什么收益反转

每个 encoder layer 只捕获第一次遇到的 M=2048 路由；共 layers 0–19。
IDs/weights 在设备上 clone，请求结束后才读取 histogram、stride 和唯一
行数，未把 GPU→CPU 读取放进正式 A/B。真实 IDs stride 均为 `[6,1]`。

周期输入每层有 **45–138** 个活跃专家，代码输入有 **236–347** 个。
因此旧材料中“所有 token 共享 6 个 expert”的结论不能用于当前负载。
这里只观测一个 chunk/layer，不能宣称是整个 32K 的全部路由。

令 `c_e` 为专家接收的 slot 数，`T_B=Σ ceil(c_e/B)`，`P_B=B*T_B`。
每层有效 slot 为 12288。对捕获的 20 层计算：

| 20 层捕获样本 | 周期输入 | 代码输入 |
| --- | ---: | ---: |
| BM64 M tile 数 | 4647 | 8137 |
| BM128 M tile 数 | 2882 | 6722 |
| M tile 数减少 | 37.98% | 17.39% |
| BM64 padded rows | 297408 | 520768 |
| BM128 padded rows | 368896 | 860416 |
| padding 后工作量增加 | 24.04% | 65.22% |
| BM64 有效行利用率 | 82.63% | 47.19% |
| BM128 有效行利用率 | 66.62% | 28.56% |

更大的 M tile 减少重复 weight tile 请求，同时在低 occupancy 专家上
计算更多 padding。周期输入的请求数节省较大，代码输入的 padding 代价
更大；这解释了为什么只按平均 `M*topk/E` 或 shape 选择固定 BM128
无法安全推广。tile 数是逻辑请求数，L2 会复用数据，不能直接当作实测
HBM 字节数。源代码和 histogram 证明 padding/tile 工作量变化，但本轮
没有候选的细分 kernel profile，不能把 TTFT 退化精确归因为“全部来自
padding”或宣称某段达到 compute wall。

离线重算及每层结果见 `analyze.py`、`route-costs.json/csv`。脚本校验每层
histogram 总数、层集合和两种 tile 的计数关系，不新增硬件扫描。

## 对照此前成功优化的推广边界

对实际加载的 `_moe_C_stable_libtorch.abi3.so` 做 cuobjdump，摘要和 binary
SHA256 见 `tile-resources.json`；完整 SASS 和 resource usage 已保存。

| kernel | block-M | registers/thread | stack/local | 静态指令 | PRMT | IMMA |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| W13 | 64 | 184 | 0/0 | 9604 | 284 | 128 |
| W13 | 128 | 238 | 0/0 | 13700 | 308 | 256 |
| W2 | 64 | 184 | 0/0 | 9092 | 284 | 128 |
| W2 | 128 | 238 | 0/0 | 12772 | 308 | 256 |

这些是 CUTLASS prefill kernel，不是之前的 q6 sparse attention 或 M6
TileLang GEMM。两种 tile 都无 spill，不能机械应用“4 warp 改 8 warp
消除寄存器溢出”。静态 PRMT/指令数也不能代替动态执行成本；mainloop
与 epilogue 的执行频率不同。当前真正需要同时控制的是 weight tile
复读和低 occupancy 专家的 padding，不只减少几条 packed-word 指令。

## 正确性、证据修正和生产恢复

私有 raw `Experts.apply` 检查在周期输入上覆盖 20 个 encoder layer 的
首个 M=2048 chunk，候选 BF16 输出与 baseline **数值相等**。它使用
`(candidate == expected).all()`，不比较有符号零的位模式。原 runner 的
`all_bitwise_exact` 字段及打印标签过强，实际含义在
`measured-record-annotations.json` 中明确修正；保留原测量文件及 SHA256，
不修改已加载的诊断脚本来伪装成更强检查。

所有候选正式请求还检查最终 IDs。它们支持本轮数值/交付一致性，不构成
全模型逐位证书，也不是 CED 相对于完整 prefill 的语义评测。

正式 `dsv41f-manual.service` 已恢复，MainPID **69930**，五个 worker
72664–72668，实际物理 GPU UUID、模型路径、argv/env 和 unit cgroup
均核对。诊断 unit inactive，26662 无 listener，26661 health=200，
dev RPC=404；没有 extension、dev mode、PYTHONPATH 或 profile 开关。

| 正式恢复检查 | 结果 |
| --- | --- |
| 冷 32K，预热后两次 | 3.97965/4.04836 s，中位 4.01400 s |
| prefix hit | 32640 token，保留 128-token suffix，TTFT 0.41270 s，IDs 一致 |
| 129/2049/2050 和 64K | 正常结束；64K 7.56275 s，仅作交付检查 |
| 单请求 256-token decode | 51.10 token/s，256 IDs 与既有参考逐个一致 |
| 并发两请求、各 128 token | 两路 IDs 均与既有参考一致 |
| 异质代码语料 32K | 冷 TTFT 5.21250 s，64 IDs 与诊断 baseline 一致，检索成功 |
| speculation / 源码 / prior dirty | metrics 增量零，启动源码摘要一致，22 个既有文件保留 |

恢复时间是新进程的交付验证，不作为候选收益。`production-validation.json`、
`production-prefill.json`、`production-corpus.json` 保存检查细节；另核对
正式实际 argv 与本轮原始 baseline argv 完全相同。新增测量工具通过 ruff/check、format
和语法检查，`git diff --check` 通过；本轮没有推理改动，不重复上一轮已
通过的 CED/scheduler pytest，也没有新增 tests 中的一次性 benchmark。

## 下一项已收敛的工作

后续优先研究**设备上按真实路由 occupancy 组织 GEMM 工作**，避免为
选择 tile 增加 host synchronize。attention metadata 只有 prefill/decode
信息，不能判断 expert diversity；路由分布要从 topk IDs 或已有 align
过程获取。第一步应保存实际 IDs/weights，并在离线 harness 重放这两种
分布，对比 weight 请求、padding 和候选分项计时，再做一次有限 serving
AB/BA。当前 histogram 不能恢复原始行顺序，不应冒充精确路由重放。

不根据这两个输入直接制定一个通用 active-expert 阈值，不启用固定 BM128，
也不删除 attention fence。下轮只有同时在集中和分散路由上获得端到端
收益、且保持输出与非目标路径正确，才进入生产推广。

离线重算不需要 GPU 或服务重启：

```bash
.venv/bin/python artifacts/dsv41f-ced-prefill-round2-20261002/analyze.py
.venv/bin/python artifacts/dsv41f-ced-prefill-round2-20261002/inspect_binary.py
.venv/bin/ruff check artifacts/dsv41f-ced-prefill-round2-20261002/*.py
.venv/bin/ruff format --check artifacts/dsv41f-ced-prefill-round2-20261002/*.py
git diff --check
```

`run_experiment.py` 和 `validate_candidate.py` 要求对应私有诊断实例；
不用于正式服务，不在正式服务上开放诊断 RPC。此轮只进行一次诊断加载，
同实例完成筛选和反例验证，之后一次恢复加载，没有 q/tile/stage 扫参。
