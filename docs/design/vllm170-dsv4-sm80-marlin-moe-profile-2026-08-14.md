# DeepSeek V4 SM80 Marlin MXFP4 MoE P0 分解与优化（2026-08-14）

## 结论

5,000-token prefill 的 P0 已完成首轮实机优化：仅当 MoE token batch
`M >= 1000` 时，将 Marlin expert alignment 的 `block_size_m` 固定为 48；
小 batch 和 decode 继续使用原有 shape heuristic。三张 CMP170HX、PP3/TP1、
DSpark 7-token、CUDA Graph 部署路径的 3 次重复 A/B 中：

- TTFT 中位数由 `1619.252 ms` 降至 `1602.793 ms`，改善 `1.02%`；
- PP0/PP1/PP2 的 Marlin MoE 父范围分别改善
  `1.93% / 4.08% / 3.62%`；
- W13 分别改善 `1.89% / 4.58% / 4.61%`，W2 分别改善
  `2.87% / 5.35% / 4.81%`；
- decode 中位数为 `263.339 ms` 对 `264.152 ms`，差异 `+0.31%`，属于
  未改变 decode block 策略后的运行波动，不归因于本优化。

因此，CMP170HX 启动器默认使用：

```bash
VLLM_MARLIN_MOE_BLOCK_SIZE=48
VLLM_MARLIN_MOE_BLOCK_SIZE_MIN_M=1000
```

设置 `VLLM_MARLIN_MOE_BLOCK_SIZE=0` 可完整回退到原 heuristic。通用 Marlin
默认行为没有改变，只有本项目的 CMP170HX 启动器选择该实测配置。

## 测量边界与证据

所有 A/B 都使用同一模型、prompt hash、精确 5,000 server tokens、8 completion
tokens、三张 CMP170HX、PP3/TP1、`15,15,13` target layers、MXFP4 Marlin MoE
和 DSpark width 7。每个服务实例先用相同上下文预热，再在一个 profiler window
内连续执行 3 次请求。

| 目的 | 原策略 | 候选 | 执行模式 |
| --- | --- | --- | --- |
| Marlin 子阶段归因 | `20260814T050439Z` | `20260814T051035Z` | eager + CUDA Events |
| 部署路径验收 | `20260814T052032Z` | `20260814T052250Z` | CUDA Graph + CUDA Events |

原始证据位于 `profiles/deepseek_v4/<run-id>/`，其中包含请求结果、三 rank
phase JSONL、按 phase 的 Markdown、`phase_summary.json`、server log 和验证状态。
四个 run 均满足精确 token 数、PP3 artifacts、三次 workload 和 prefill 明细完整性。

详细模式为每个执行 step 同步 CUDA Events，用于研究模块归因，不代表生产 decode
吞吐；Graph 模式用于最终端到端门禁。当前 CMP 驱动返回
`CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED`，因此这些结论是 operator-range/event
结论，不声称 NCU 指标、occupancy 或单个 CUDA kernel 性能。

## 实际调用链与计量范围

```text
FusedMoERunner
  router_gate
  dispatch
  router_select
  routed_experts
    modular.prepare
    modular.workspace
    modular.kernel
      marlin.align
      marlin.w13_gemm
      marlin.activation
      marlin.w2_gemm
      marlin.reduce
    modular.finalize
  combine
```

`routed_experts`、`modular.kernel` 是嵌套父范围；它们不能与子阶段相加。
分析器只把 gate、dispatch、select、prepare、workspace、align、W13、activation、
W2、reduce、finalize 和 combine 当成相互独立的 additive ranges，并单独报告父范围
及 residual。PP stages 并行，也不能跨 rank 相加。

### DSV4 实际形状

- hidden size `K=4096`，intermediate size `N=2048`；
- `E=256` routed experts，per-token `topk=6`；
- scheduler 将 5,000 tokens 切为 `M=2000, 2000, 1000`；
- W13 输出为 gate/up 两个 shard，W2 再投影回 hidden size；
- W13/W2 都使用 `use_fp32_reduce=True` 和 `use_atomic_add=False`。

## 原策略的 Marlin 子阶段分解

以下是 eager 原策略 3 次请求的每请求平均值。比例以各 rank 的完整
`moe.fused_route_experts` 父范围为分母。

| PP rank | MoE parent | W13 | W2 | W13+W2 | activation | reduce | gate | align | 其他/父范围 residual |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 380.019 ms | 208.738 ms | 113.502 ms | 84.8% | 4.574 ms | 3.596 ms | 2.593 ms | 1.317 ms | 11.6% |
| 1 | 358.136 ms | 198.189 ms | 107.874 ms | 85.5% | 3.965 ms | 3.032 ms | 2.446 ms | 0.979 ms | 11.3% |
| 2 | 296.865 ms | 161.560 ms | 87.566 ms | 83.9% | 3.486 ms | 2.657 ms | 2.079 ms | 1.188 ms | 12.5% |

结论很明确：W13 与 W2 合计约占 MoE `84%–86%`，W13 单项占约
`55%`，W2 占约 `30%`。activation、reduce、routing、alignment 均不是一阶
热点。父范围剩余 `11%–13%` 主要落在 eager/custom-op 外围和未展开的 launch
间隙；在没有图内 kernel 证据前，不把它解释成某一个 GPU kernel。

PP0 的完整模型分解也与此一致：每请求 model forward `771.946 ms`，FFN
`380.269 ms`，attention `315.156 ms`。Marlin 是当前 prefill 的首要优化对象，
但优化目标应直接指向两次 GEMM，而不是先融合低占比 bookkeeping。

## Block-M 研究

Marlin 先按 expert 排列 `M * topk` 行，并将每个 expert 的行数 padding 到
`block_size_m`。原 heuristic 在本形状选择：

| M | 原 block | PP0 有效行率 | PP1 有效行率 | PP2 有效行率 |
| ---: | ---: | ---: | ---: | ---: |
| 2000 | 64 | 71.93% | 72.41% | 78.98% |
| 1000 | 32 | 76.03% | 78.55% | 88.09% |

候选 48 的 Graph 实测为：

| M | 候选 block | PP0 有效行率 | PP1 有效行率 | PP2 有效行率 |
| ---: | ---: | ---: | ---: | ---: |
| 2000 | 48 | 75.36% | 78.28% | 83.74% |
| 1000 | 48 | 64.97% | 70.55% | 83.71% |

`M=2000` 时，48 同时减少 padding 并改善 tile 调度；`M=1000` 时有效行率
反而降低，但 GEMM 总体仍受益，说明 padding ratio 不能独立决定最佳 block。
最终策略必须以 W13/W2 和 TTFT 实测为准。

### 候选筛选

| 候选 | 结果 | 决策 |
| --- | --- | --- |
| 全局 block 32 | 2K padding 改善，但 W13 增加约 3.6%–6.3%，W2 增加约 3.0%–5.0%，单次 TTFT `1775 ms` | 拒绝 |
| 全局 block 48 | prefill 有收益，但会污染 decode/small-M 策略 | 拒绝全局启用 |
| block 48 且 `M>=1000` | eager TTFT 中位数改善 `3.47%`；Graph TTFT 中位数改善 `1.02%`，三 rank MoE 均改善 | P0 采用 |

Graph A/B 的分 rank 每请求结果如下：

| PP rank | MoE 原/新 | MoE 改善 | W13 原/新 | W13 改善 | W2 原/新 | W2 改善 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 364.499 / 357.448 ms | 1.93% | 199.910 / 196.135 ms | 1.89% | 109.261 / 106.128 ms | 2.87% |
| 1 | 345.218 / 331.141 ms | 4.08% | 190.734 / 181.994 ms | 4.58% | 103.986 / 98.424 ms | 5.35% |
| 2 | 280.619 / 270.467 ms | 3.62% | 152.735 / 145.686 ms | 4.61% | 82.729 / 78.750 ms | 4.81% |

## 第二轮研究结果

### P0：W13/W2 Marlin 执行配置扫描已完成

已使用真实 DSV4 W13/W2 形状、三种路由和
`M={1000,1666,1667,2000,4048}` 扫描 large-batch selector 的三组 thread shape
及 1/2 blocks per SM。auto 与强制 `64x128,b2` 在全部 shape 上的中位数差异不超过
0.11%，其他候选普遍回退，因此生产继续使用 auto，不增加私有强制表。

### P0：FP32 global-reduce 临时缓冲复用已完成

C++ op 现在接受可选 caller workspace；modular workspace 为 W13/W2 预留一个共享
FP32 尾区，未传时仍保留原内部申请。`M=1667` 的逻辑需求为 13.125 MiB/rank，
隔离测量的热路径临时分配峰值由 14.0 MiB 降为 0。PP3 clean Graph A/B/A 的
caller pooled TTFT 为 1260.607 ms，internal control 为 1260.645 ms，差异
-0.003%，因此结论是地址/生命周期改善、性能中性。默认启用，环境变量可回退。

### P1：融合研究结论

W13 的 gate/up halves 跨 output-column CTA，W2 的一个 token 的 6 个 expert rows
跨 expert CTA，因此二者都不能作为现有 Marlin `write_result` 的简单 epilogue 分支。
`M=1667` 时，真正融合各可消除约 156.28 MiB/层中间写回和读取，但 standalone
activation/reduce 仅约 103.5/83.8 us。下一步先研究 W2 两阶段 FP32 grouped-tail，
未通过联合范围 3%、模型输出和 PP3 Graph 门禁则停止；不采用 BF16 atomic 或在
W2 load 中重复计算 activation。

完整 shape 表、workspace ABI/A/B/A、融合数据流和验收门槛见
[`vllm170-dsv4-sm80-marlin-moe-config-workspace-2026-08-14.md`](vllm170-dsv4-sm80-marlin-moe-config-workspace-2026-08-14.md)。

### 明确不推进

- **BF16 atomic add**：Marlin 明确禁止 SM8x BF16 原生 atomic-add 路径，且
  `N >= 2048` 也不满足现有启用条件；本模型 W13/W2 正好处于该边界。
- **全局固定 block 48**：会改变 decode/small batch；当前只允许 `M>=1000`。
- **直接迁移 SM70 kernel/tile 参数**：算法证据可参考，硬件配置必须在 SM80
  重测。
- **`mxfp4_int8_sm80` 替换 Marlin**：此前短 prefill 实测更慢且为近似路径，
  未通过生产 promotion gate。
- **先优化 router/align/activation/reduce 小项**：单项上限不足，不能绕开占比
  约 85% 的两次 GEMM。

## 验收状态与限制

- 原/候选 Graph A/B 各 3 次，精确 5,000 prompt tokens 和 8 completion tokens；
- 三个 PP rank 的 prefill 子阶段完整，workload 和 phase timing 全部验收；
- 强制 block 48 的真实 CUDA MXFP4 Marlin reference case 通过；
- block selector 单元测试覆盖默认策略、M threshold、合法值和非法值；
- CUPTI 不可用，所以 occupancy、register pressure、memory throughput 仍是待未来
  可用的 isolated NCU/CUDA Event microbenchmark 补齐的证据缺口；
- block-M 结论覆盖 `M=1000/2000`；第二轮 config 表额外覆盖
  `M=1666/1667/4048`。32K、128K、不同并发和 prefix cache 命中仍需单独验收，
  不能外推收益比例。
