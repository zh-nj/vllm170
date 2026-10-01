# DSV4.1-Flash target 验证截距：复核与组件测量（2026-10-01）

## 1. 33.87 ms 究竟是什么

**33.87 ms 是 q≥2 回归线外推到 q=0 的截距，不是进入验证的一次操作耗时。**
从 9 月 28 日原始逐 rank JSON 重算，未先四舍五入：

```text
T_on(q) = 33.8650 + 2.4265 q              q=2..6
T_off(1) = 26.3800
T_on(1) 外推 = 36.2915                   未实际运行此状态
外推差额 Δ = 36.2915 - 26.3800 = 9.9115

33.8650 = 26.3800 + 9.9115 - 2.4265
          已有单 token 成本 + 外推差额 - 一个 q 的斜率项
```

所以先有两项不同的问题：已有单 token 成本为什么是 26.38 ms，以及 q≥2 的
曲线为什么外推到 36.29 ms。后者才与“进入验证”有关，但这个差额仍然只是外推，
不能直接命名为已经测到的固定进入成本。FULL Graph、小 batch GEMM 配置、路由、
indexer/top-k 和计时范围都可能改变它。

原文的 33.87 + 2.42q 使用 `verify_cost_model.py` 对五个点做最小二乘；
`20260928_verify_sweep/analyze.py` 用 q2/q6 两端点，得到 33.7025 + 2.31875q。
两种拟合不要混用。原始 OLS 最大残差 0.9275 ms；逐一删除 q2..6 一个宽度，
截距范围 32.6515–34.8100 ms、外推差额范围 9.1025–10.6675 ms。
这是拟合敏感性，不是统计置信区间。

原始数据和新分析分别在：

- `profiles/deepseek_v4_1/20260928_verify_sweep/q{1..6}/step_timing/step_timing.json`
- `artifacts/dsv41f-verify-intercept-20261001/sweep-analysis.json`
- `benchmarks/deepseek_v4_1/verify_intercept_analysis.py`

## 2. 先按 rank 拆开，不需要重新扫描

对每个 rank 的 q2..6 独立拟合，与该 rank 的 DSpark-off q1 比较：

| rank / 层 | q1 实测 ms | 截距 b ms | 斜率 a ms/q | 外推差额 b+a−q1 ms |
| --- | ---: | ---: | ---: | ---: |
| PP0 / 0–7 | 4.970 | 5.186 | 0.5350 | 0.7510 |
| PP1 / 8–15 | 5.260 | 5.746 | 0.4915 | 0.9775 |
| PP2 / 16–23 | 5.460 | 7.121 | 0.3440 | 2.0050 |
| PP3 / 24–31 | 5.130 | 8.354 | 0.3090 | 3.5330 |
| PP4 / 32–39 | 5.560 | 7.458 | 0.7470 | 2.6450 |
| 合计 | 26.380 | 33.865 | 2.4265 | 9.9115 |

**PP2–PP4 占外推差额的 82.6%。** 五个 rank 都有八层，但差额并不均匀。
这要求解释 decoder/indexer 拓扑、设备差异和计时边界，不能只说所有层同时进入了
另一种 attention 模式。PP3 的 q3 target 比 q2 多 1.11 ms，q6 却比 q5 少
0.36 ms，也显示逐 rank 曲线并非严格线性。

六个 q 的 prompt hash 相同，q≥2 的三个重复请求输出及接受结果分别一致。
但 q1 没有 q≥2 的显式 64-token warmup。每个 rank 的值来自各自 10-step
窗口的中位数；q6 五个 rank 有效窗口只有 15/14/14/21/21 个，窗口未跨 rank
对齐。**五个中位数之和不是一次真实 PP wall interval，也没有数学上必然为
wall time 上界的保证。** 本文保持这个历史统计口径，不把它变成端到端耗时。

## 3. target 计时包含什么

源码核对：`vllm/v1/worker/gpu/model_runner.py` 的 `forward_start()` 在
graph 输入 staging/replay 之前，而 `forward_end()` 在下一次
`sample_tokens()` 入口，并非 graph replay 返回的位置。非末级 rank 的
`gpu_worker.execute_model()` 在这之间还会启动 PP `isend_tensor_dict()`。

```text
attn_metadata / model_inputs / PP recv
    ↓                         这些在 target 起点之前
forward_start
    graph_input_staging → FULL graph replay → 返回 execute_model
    PP send 启动 / worker-executor 交接 / 可能的 stream 空隙
forward_end（sample_tokens 入口）
    ↓                         logits、采样、draft 在 target 终点之后
```

CUDA Event span 会包含两个 event 之间 stream 没有 GPU 工作的时间，不能自动
等同于 kernel 忙碌时间。上述交接可能留下空隙，但历史扫描没有 graph 结束与
sample_tokens 入口之间的独立计时，**本轮不把这部分空隙假定为已经测出的 ms**。

q1/q6 `attn_meta` 的 rank 和分别约 5.34/24.56 ms；它们位于 target 起点之前，
不能把 19.22 ms 的增长加进 9.91 ms。它可以通过调度与重叠改变整轮时间。
LM head/logits 同样在采样阶段，不能列入 target 内的矩阵账本。

SM80 的 q1 和 q≥2 均可进入同一 `_forward_decode`。`uniform_token_count=q`
证明 Graph 捕获了不同 shape，不能单独证明存在一项额外的完整 attention
进入操作。实际 kernel 确实按 q 改变配置：sparse attention 的 `block_k`
从 16 变为 32、reduce 的 warps 从 8 变为 4；strict BF16 GEMM/BMM
也有只适用于 M=1 的配置桶。

## 4. 组件短测：固定模型工作和可验证的配置跳变

没有重启 PP5，也没有改推理代码。在空闲 SM80 卡
`GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216` 上使用 checkpoint 实际权重，
隐藏状态、KV、路由为固定 seed 合成输入。Marlin full-K、strict BF16 math
打开，MoE TileLang 关闭，接近历史扫描的主要算子配置。

每个 case 捕获一次调用，热身后用 CUDA Events 测 3 轮 × 20 样本；每个样本前
在计时区外清掉 64 MiB L2。偶数/奇数轮反转 q 顺序。报告使用 **174 个组件×q
case**，均通过 eager 与 Graph replay 的逐位一致、输出覆盖及全 capture 后
再次 replay 检查。这是 harness 一致性检查，不是另一个算子参考实现的数值验证。
没有 CUPTI/Nsight kernel timeline；微秒级 event 值存在约 1 µs 的量化。

主要结果，每次调用单位 µs：

| 组件 | q1 | q2 | q6 | 与固定项的关系 |
| --- | ---: | ---: | ---: | --- |
| fused wq_a+wkv | 29.696 | 29.696 | 29.696 | 基本固定，40 层 |
| wq_b | 59.392 | 59.392 | 59.392 | 基本固定，40 层 |
| wo_a | 60.416 | 74.752 | 74.752 | q1→2 有配置跳变，40 层 |
| wo_b | 70.656 | 70.656 | 70.656 | 基本固定，40 层 |
| router GEMM | 69.632 | 94.208 | 94.208 | q1→2 有配置跳变，40 层 |
| mHC pre | 31.744 | 30.720 | 29.696 | 两次/层，基本固定 |
| mHC post | 9.216 | 9.216 | 9.216 | 两次/层，基本固定 |
| shared expert 串行调用 | 64.512 | 64.512 | 64.512 | 40 层，但生产与 routed 重叠 |
| C1 compressor 投影 | 31.744 | 121.856 | 121.856 | 仅 source 20；与其他输入投影并行 |
| C2 compressor fused 投影 | 121.856 | 122.880 | 122.880 | source 2/8/14，基本固定 |
| indexer weights 投影 | 113.664 | 113.664 | 113.664 | 8 个 index source，与输入投影并行 |
| indexer query 投影 | 14.848 | 14.336 | 14.336 | 8 个 index source |
| SWA128+top512 attention | 29.696 | 67.584 | 124.928 | 38 层的代表形状，随 q 增长 |
| SWA-only128 attention | 23.552 | 25.600 | 39.936 | 2 层的代表形状 |

这里只对重复次数明确的矩阵调用做量级折算：

- 四个实际 attention 投影在 q1 合计约 **8.81 ms/40 层**，q2 约 9.38 ms。
  不应再把未融合的 wq_a 和 wkv 各自相加。
- router 在 q1 约 **2.79 ms/40 层**；mHC pre/post 约 **3.28 ms/40 层**。
  即使只有一个 token，也已经有大量固定模型工作。
- 按 q2..6 同样回归，router 的外推差额约 **0.963 ms/40 层**，wo_a 约
  **0.573 ms/40 层**。约 1.54 ms 的隔离算子配置效应有实测支持。
  这是对历史差额的量级线索，不能称为同 campaign 已闭合的 15.5% 分账。
- C1 compressor 自身跳变约 0.090 ms，但 q1 的 indexer weights 已花
  0.114 ms，且二者使用辅助 stream 并行，不能把独立调用差直接加到关键路径。
  C1 只有一个 KV source，不能乘以 40，也不能用它代表三个 C2 source。
- shared expert 单独串行成本约 2.58 ms/40 层，生产与 routed experts 并行，
  不能再作为独立的 2.58 ms 加进 target。

MoE 使用相同 36 个实际专家的权重，构造三个可知路由：

| 每个新 token 增加的专家 | q1 µs | q2 µs | q6 µs | 40 层拟合斜率 ms/q |
| ---: | ---: | ---: | ---: | ---: |
| 0：全部共享 6 个专家 | 175.104 | 180.224 | 180.224 | 0.000 |
| 3：D=6+3(q−1) | 175.104 | 248.832 | 493.568 | 2.429 |
| 6：D=6q | 175.104 | 300.032 | 782.336 | 4.825 |

专家多样性是重要机制，且它的量级足够解释大量增长；但 sparse attention
本身的 38 层拟合斜率约 **0.619 ms/q**，SWA-only 还有约 0.009 ms/q。
再加 indexer，**总斜率不是只由专家决定**。表中的 3 新专家路由是人为构造，
不能据其斜率接近 2.4265 就认定生产路由每 token 增加 3 个专家。

## 5. 新发现：FULL Graph indexer 捕获上限影响回归截距

源码链条已经核实，并且这些相关源文件与历史 HEAD `05e703f694` 无差异：

1. `model_states/default.py` 的 `for_capture=True` 使用配置的
   `max_model_len=1048576`，而非当前请求的实际 32K 长度。
2. indexer metadata 保存该 `max_seq_len`；Triton paged logits 使用它决定
   输出宽度/grid。`clean_logits=False`，所以没有完整 logits 的 -inf 预填充；
   多余 CTA 会退出，但仍有启动/调度成本。
3. SM80 下 q1..5 使用 `persistent_topk`；q6 命中 DSpark exact-top-k 分支。
   使用相同捕获宽度不等于使用相同 top-k kernel。
4. C1 source 20 还选 candidate blocks，source 24/28/32/36 做 candidate mask；
   这些操作的部分工作也按捕获宽度展开。

在实际上下文仍是 32K、block table 维持最大宽度的合成 KV 上测量，包含
paged logits 和对应 top-k；只改变传入的输出宽度。q1/q2 使用原生查询布局，
q3..6 按 SM80 metadata builder 展平成 `(q, 1)`：

| C1 indexer（无 candidate） | q1 µs | q2 µs | q3 µs | q4 µs | q5 µs | q6 µs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 实际 32K 宽度 | 72.704 | 80.896 | 80.896 | 86.016 | 92.160 | 123.904 |
| FULL 1M 捕获宽度 | 128.000 | 249.856 | 344.064 | 451.584 | 559.616 | 145.408 |

**q6 的降低是 top-k 分支切换，不是上下文变短或少验证了 token。** 对这个组件
强行拟合 q2..6，会产生约 0.220 ms 的外推差额，同时最大残差约 0.209 ms；
这不能叫一个固定进入动作。

按实际拓扑分别测 3 个 C2 无 candidate、1 个 C1 candidate-select、4 个
C1 candidate-mask，再求合成组件和：

| q | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| indexer 代表调用和 ms | 1.040 | 1.836 | 2.533 | 3.277 | 4.022 | 1.405 |

这个和拟合 q2..5 是 `0.361 + 0.730q`，外推差额仅 0.051 ms；把 q6 也纳入
却变为 `2.364 + 0.063q`，外推差额 **1.386 ms**，残差达到约 1.34 ms。
这是一个具体反例：**kernel 切换造成的弯曲会进入拟合截距，尽管没有新增
1.386 ms 的固定工作。** 不能把这些合成输入的数值直接扣掉历史 9.9115 ms；
真实候选集、top-k 分布、PP 依赖和设备时钟不同。

历史 total 中 q5 残差 +0.9275、q6 残差 −0.8090，与这种方向相容。
这只是相容性，不能据此宣称历史残差已经全部归因。

## 6. 一次现有服务边界核验

现有 `dsv41f-manual.service` 仍为 PP5/q6，没有重启。对历史相同 32K prompt
做两次 128-token 请求，第二次首 token 返回后才打开已有 phase control，
结束立即移除。只用于确认 replay 的范围，不能复刻历史拟合：当前服务有
prefix cache、KV 2.7 GB、MoE TileLang opt-in，历史扫描 KV 1.3 GB 且显式
设置了 Marlin full-K/strict BF16/strict prefill tail。

21 次 rank-local q6 FULL Graph replay 的 GPU 中位数：

| rank | PP0 | PP1 | PP2 | PP3 | PP4 | 中位数之和 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| replay ms | 6.482 | 7.998 | 8.684 | 9.331 | 9.682 | 42.177 |
| staging ms | 0.017 | 0.034 | 0.023 | 0.038 | 0.040 | 0.152 |

replay 的 host scope 只有每 rank 0.148–0.269 ms，说明这里测到的是实际
GPU Graph 区间，不是把 metadata builder 的宿主耗时装进了 replay。
但 phase instrumentation 把 decode 从 1.532 s 拉到 5.694 s，约 3.7 倍。
**这次 profile 不提供生产吞吐或历史差额的标定。** 两次首 token 延迟还受
prefix cache 命中不同影响，不比较 TTFT。

两次请求 128 个输出 token ID 和 usage 一致；服务 MainPID/ExecStart 前后
一致，health 200，control 已移除，最终 running/waiting 均为 0。
当前 GPU 设备清单也显示 PP2/PP4 所在卡 HBM 上限 1458 MHz，其余生产卡
1728 MHz；这是当前状态，历史 campaign 未提供足够设备时钟记录，不能把当前
设备差异倒推为历史差额的确定原因。

## 7. 现在能闭合到哪里

**已经闭合：** 33.865 的统计构成、9.9115 在各 rank 的分布、target 边界、
router/wo_a 配置跳变、固定矩阵工作、attention 的非专家边际成本、FULL Graph
indexer 的最大宽度与 q6 top-k 切换。33.87 的形成是“原有模型成本 + 模式/形状
差异 + 曲线弯曲的外推效应”，不是单一进入动作。

**没有闭合：** 历史 9.9115 ms 的完整、可加、同 campaign kernel 账本。
历史六次扫描没有图内组件 trace，当前服务配置也不同。剩余差额不能自动命名为
launch、attention 或专家成本；本轮不重跑昂贵的六次重启来制造可比性。

若后续需要彻底闭合，只需一个匹配历史配置的隔离诊断进程，测 q1/q2/q6 三个
形状；同 stream 同时记录 graph 起止、sample_tokens 入口，图内按不重叠区间
计时。捕获只存在于诊断进程，并取实际每层 routed IDs、可见 context、top-k
分支及设备时钟。不能用当前组件结果直接减旧 target，当成已经观测的剩余项。

同时撤回历史分析中的三个过度推断：斜率不能直接观测到约 2.4 个新专家；
低于先前 microbenchmark 的 223 GB/s 不是“不可能”的带宽；只有 q6 的无插桩
对照不能证明插桩对所有 q 只抬高度而不改变形状。

## 8. 复现、失败记录和交付

```bash
.venv/bin/python -m benchmarks.deepseek_v4_1.verify_intercept_analysis \
  --sweep profiles/deepseek_v4_1/20260928_verify_sweep \
  --output artifacts/dsv41f-verify-intercept-20261001/sweep-analysis.json

# components 可选 core / extra / indexer / fused；out-dir 必须尚不存在。
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python -m benchmarks.kernels.deepseek_v41_verify_component_probe \
  --model /media/z/data6/models/DeepSeek-V4.1-Flash \
  --components indexer --samples 20 --out-dir /tmp/dsv41-indexer-probe

.venv/bin/python -m pytest -q \
  tests/benchmarks/test_deepseek_v41_verify_cost_model.py \
  tests/benchmarks/test_deepseek_v41_step_timing_log.py
```

全部证据在 `artifacts/dsv41f-verify-intercept-20261001/`：

- `components-fixed/`、`components-extra-fixed/`、`components-indexer-flat/`、
  `components-fused-fixed/` 保留每轮原始样本、实际权重来源和 SHA256。
- `component-analysis.json`、`indexer-topology-analysis.json` 保留拟合计算。
- `boundary-requests.json`、`boundary-trace/`、`boundary-summary.json` 保留
  服务请求和 Graph 边界证据；这些时序样本没有跨 rank 对齐。
- `gpu-clock-inventory.csv`、`service-before.txt`、`service-final.txt` 保留
  最终硬件与服务状态。

初次 core harness 在多个 Graph capture 后遇到 IMA；预留并锁定共用 workspace
后全部 replay 检查通过。这个现象与 capture 后 scratch 地址变化相容，但未做
core dump 根因证明，不把它称为生产缺陷。另保留 extra 缺少 VllmConfig 的构造
错误、错误假定 C1 有 wgate 的 KeyError、一次 pytest 路径拼写错误及修复后日志。
`components-indexer/` 另保留 48 个原生布局探索 case，报告数字采用与 SM80
展平规则一致的 `components-indexer-flat/`，不混用两组结果。
失败进程只使用空闲卡，没有影响生产 worker。

交付只涉及 benchmark、分析、现有 CPU 测试和文档；原字节模型的数值/JSON
键保持兼容，输出措辞改为条件假设。工作区已有的推理修改保持原样，FP8 dense
继续使用 Marlin。本轮没有提交或部署推理改动。

验证：上述 CPU suite **18 passed**；六个本轮源文件的 pre-commit 全部通过，
包括 ruff、format、markdownlint 和 torch API 检查。174 个用于报告的 GPU
capture case 检查全部通过；128-token 请求对照逐位相同。新增纯分析器的测试
覆盖截距恒等式、逐 rank 可加拟合和缺失基线/斜率的拒绝行为。
