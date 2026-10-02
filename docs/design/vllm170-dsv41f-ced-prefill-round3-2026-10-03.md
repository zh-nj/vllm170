# DSV41F CED prefill 第三轮：BM128 配对与 BM64 尾块

## 结论

在 CED 开启、DSpark 关闭、PP5/TP1/DP1、2048-token chunk 的同一私有服务内，
混合分块通过两类 32K 输入的四组交错配对 A/B：

| 输入 | BM64 TTFT 中位数 | 混合分块 TTFT 中位数 | TTFT 降幅 | 配对降幅范围 |
| --- | ---: | ---: | ---: | ---: |
| 周期输入 | 4.029953 s | 3.725555 s | 7.553% | 7.233%–7.786% |
| 代码语料 | 5.166592 s | 4.894445 s | 5.267% | 4.739%–5.585% |

16 次计时请求均为独立 cache salt、零 prefix hit，输出分别与既有基线的
16/64 个 token ID 完全一致。两类输入的首个完整 2048-token chunk，
20 个 encoder 层的真实 `Experts.apply` 输出均有限且 BF16 逐位一致。
没有将离线 kernel 降幅当成端到端降幅。

已加入默认关闭的 `dsv41_sm80_moe_hybrid_prefill`，通过正式服务的
`--additional-config` 显式启用。正式服务交付验证记录见本轮 artifacts。

## 为什么上一轮失败，本轮有效

上一轮把所有专家块改为 BM128，代码语料的 padding 工作量增加 65.2%，
最终 TTFT 变慢 4.38%。本轮先保留原始 BM64 对齐，再按专家将相邻两个
BM64 块配成一个 BM128 块；奇数个 BM64 块的最后一块仍运行 BM64。

对于专家的有效行数 `c`，令 `b = ceil(c / 64)`：

```text
BM128 块数 = floor(b / 2)
BM64 尾块数 = b % 2
混合 padded rows = 128 * floor(b / 2) + 64 * (b % 2) = 64 * b
混合 M tile 数 = ceil(b / 2) = ceil(c / 128)
```

真实服务捕获的首个完整 chunk，聚合 20 层，得到：

| 输入 | 原 BM64 M tiles | 混合 M tiles | M tiles 降幅 | 原 BM64 padded rows | 混合 padded rows | 固定 BM128 padded rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 周期输入 | 4647 | 2882 | 37.98% | 297408 | 297408 | 368896 |
| 代码语料 | 8137 | 6722 | 17.39% | 520768 | 520768 | 860416 |

同一输入的 baseline/hybrid 捕获保留完整路由 ID、权重和行顺序，20 层全部相同。
这些数值是首个完整 chunk 的观测，不是整段 32K 的累计流量。

两种 tile 复用已有 CUTLASS 实例。BM128 同时切换到已有 persistent 调度，
因此收益包含 tile 合并与调度变化；没有独立测出 HBM 字节或带宽，不能把
全部收益定量归因于减少权重重读。整个流程没有增加有效 padding 行数。

## 离线实验与真实热点

仅使用闲置的物理 GPU1（CMP170HX、SM80、1728 MHz HBM）；
正式模型所在 GPU2–6 没有同期请求。使用 CUDA Event 与 CUDA Graph，
不依赖本卡不支持的 CUPTI/NCU。14 项独立 packed-INT8/FP64-dot 参考检查覆盖
W13/W2、M=1/63/64/65/127/128/129、非单位 scale 与输出 sentinel。

计时复现上一轮实际路由直方图，但行顺序、激活和权重均为合成数据。
周期输入的 W13/W2 降低 20.3%–24.3%，代码语料降低 6.2%–10.9%。
候选计时包含 partition、metadata copy 和两次 GEMM；原有 BM64 alignment
属于两臂共同前置步骤。离线结果只作为进入 serving A/B 的依据。

真实服务的细分事件进一步确认收益来自目标 GEMM。以下是 PP0 整段 32K
的主流 CUDA Event span；插桩运行只用于定位，不替代上面的无细分插桩 A/B：

| 输入 | 阶段 | BM64 | 混合 | 变化 |
| --- | --- | ---: | ---: | ---: |
| 周期输入 | W13+W2 | 1366.638 ms | 1146.133 ms | −16.13% |
| 周期输入 | align，含候选 partition/copy | 9.518 ms | 12.357 ms | +2.839 ms |
| 代码语料 | W13+W2 | 2438.663 ms | 2280.584 ms | −6.48% |
| 代码语料 | align，含候选 partition/copy | 9.642 ms | 12.478 ms | +2.836 ms |

额外设备端路由成本小于 GEMM 节省量。PP1/PP2 同方向，PP3/PP4 主要运行
CED decoder replay，不命中 M=2048 纯 prefill 候选。一次 32K 请求在 PP0/1/2
分别命中 120/120/60 个 layer-chunk；起始小 chunk 和尾块走原路径。
PP 有重叠，不能把 rank span 相加当作 TTFT。

主流事件区间可能包含 enqueue 间隙和 stream 依赖；没有宣称它们等于
纯 kernel active time。端到端主结论来自单服务内的四组 AB/BA 配对。

## 接入与验证边界

`dsv41_sm80_hybrid_prefill.py` 在 GPU 上扫描 BM64 专家块，生成配对和尾块
索引，再复制元数据。没有 GPU→CPU 决策；两次 GEMM 保留原始全局 token slot，
向同一 routed output 的不相交行写入。W13 与 W2 都使用同一组分块元数据。
scale、SwiGLU、INT8 量化、router 权重与最终 reduce 保持既有计算方式。

静态门控检查模型身份/维度、E384/top-k6、BF16/SILU、无 bias/LoRA、SM80，
以及 CED、V2 runner、无 DSpark、PP5/TP1/DP1、无 microbatch、2048/8 scheduler。
动态门控还要求 M2048、BM64/BN128、全局 E384，以及无 decode 的 eager prefill。
dummy、CUDA Graph、mixed batch、其他形状继续使用既有路径。
每个 expert 实例持有约 0.4 MB 的独立 scratch，加载时预热 metadata kernel。

接入后运行 39 个定向检查，包括生产 `Experts.apply` 的完整
quantize → W13 → activation → W2 → reduce、输出/workspace alias、热专家到
均匀专家的切换，以及 Graph 更新后的旧尾块清除。生产 apply 的输出及
全部 W2 slots 逐位一致。完整 pre-commit（含 mypy、accelerator API 和 Markdown 检查）、
Ruff check/format 与 diff check 通过。

开关默认关闭；本机正式服务显式启用。回滚时将
`dsv41_sm80_moe_hybrid_prefill` 设为 `false` 并重启同一服务。
两类输入有稳定收益，不代表所有路由分布都更快；完全没有可配对块时，
额外 metadata 与空 BM128 launch 仍有成本。没有扩展到未经验证的 chunk、
拓扑或 speculative 路径。

## 正式服务交付

最终 `dsv41f-manual.service` 在 26661 健康，模型为
`/media/z/data6/models/DeepSeek-V4.1-Flash`，GPU2–6 的五个 PP worker
均归属于该 unit。CED 开启，DSpark 关闭，hybrid 开关为 true；
诊断 extension、dev mode、PYTHONPATH 和私有 26662 listener 均已清除。

| 交付检查 | 结果 |
| --- | --- |
| 冷 32K，预热后两次 | TTFT 中位数 3.676855 s，16 个 IDs 与原基线一致 |
| 冷 32K 代码语料 | 4.855345 s，64 个 IDs 与原基线一致 |
| prefix hit | 32640 tokens，128-token suffix；0.423752 s，IDs 一致 |
| 129/2049/2050 尾块 | IDs 与上一轮正式基线一致 |
| 冷 64K 单次 | 6.899752 s，IDs 与上一轮正式基线一致 |
| 无 DSpark decode | 256 个 IDs 一致；预热后 50.9659 tok/s |
| 双请求 decode | 两个请求各 128 个 IDs，与原基线一致 |
| 状态核对 | 模型、完整 argv、五张卡/cgroup、17 个源 SHA 均吻合 |

这些重载后的数值用于确认交付；收益百分比仍来自私有同进程配对 A/B。
64K 和特殊尾块是单次正确性/交付检查，部分既有 indexer/metadata kernel
发生首次 JIT，不能据此声称相应形状有确定的性能改善。

初次正式加载后的验证在读取日志时发现断言错误：`logger.info_once` 默认
`scope="local"`，本机仅 PP0 打印，所以应检查一条而非五条日志。
完整 hooks 随后要求补充类型标注及 accelerator API，最终源码通过检查后
重新加载正式服务并完成上述请求。前次启动/验证日志保存在
`production-prelint-*`，最终启动与验证使用 `production-*`。

## 证据与复现

本轮目录：`artifacts/dsv41f-ced-prefill-round3-20261003/`。

- `offline.json`、`offline_measured.py.txt`：14 项独立检查、8 个 kernel A/B。
- `launch.json`、`experiment.json`：私有服务参数/源文件 SHA、四组两输入 A/B。
- `supplement.json`、`profile-summary.json`：真实语料逐位检查、完整路由和细分 span。
- `private-executed-sha-validation.json`：私有服务停机前的执行源 hash 核对。
- `serving_measured_benchmark.py.txt`、`start_service_private.py.txt`：接入前实际执行快照。
- `integration-tests-final.log`：最终源码 39 passed，18.72 s；
  `pre-commit-final.log`：完整 hooks 通过。
- `production-launch.json`、`production-validation.json`、`production-prefill.json`：正式服务交付。
- `baseline-state.json`、`workspace-preservation.json`、`round3-existing-files.patch`：工作区边界。

初始 23 个已有 dirty 文件中，21 个逐字节不变；其余两个只增加本轮代码，
原有全部行保留。没有提交。此前 FP8 dense TileLang 路线不属于本轮。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
  .venv/bin/python -m pytest tests/kernels/quantization/test_mxfp4_int8_sm80.py \
  -k 'hybrid_prefill or dsv41_sm80_tilelang_gate' -q
```

benchmark 已改为直接导入正式 helper，避免实验与生产实现漂移。已有 kernel
计时对应保存的原型执行快照；本轮没有为代码提取重复整套性能扫描。
