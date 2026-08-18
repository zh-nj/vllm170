# DeepSeek V4 SM80 DSpark 接受率 P0 验证

## 结论

本轮没有发现 SM80 noncausal sparse MLA 元数据或 Triton attention 的正确性错误；
SM120 曾出现的 mHC 首层 broadcast 数值差异也不能在 CMP170HX 上复现为稳定的
DSpark 接受率收益，因此生产默认继续使用 broadcast 路径。

32K、q7、greedy、固定请求 seed 的生产 Graph 实测表明：

- DSpark pooled acceptance 为 `36.530%`，mean accepted length 为 `3.557`；
- 逐位置无条件接受率为
  `56.16% / 52.05% / 50.23% / 37.90% / 33.33% / 15.98% / 10.05%`；
- 损失从第一个 draft token 已经出现，不是只发生在 q6/q7 的 top-k cutoff 尾部；
- DSpark decode 为 `108.58 tok/s`，target-only 为 `61.96 tok/s`，提升 `1.75x`；
- 相同 32K prompt、`seed=4242`、temperature 0 下，target-only 六次请求已有四个
  不同输出哈希，DSpark 六次请求有六个不同哈希，且两组共享一个完整输出哈希。

所以当前不能把约 60% 或本轮 36.5% 的接受率直接归因于 SM80 q6 top-k 漏选，也不能
用单次 target-only 输出哈希作为语义金标准。下一阶段应先定位 target logits/草稿首位置
分叉，再研究 q6/q7 尾部；继续只优化精确 top-k 不会解决 q1 约 44% 的拒绝。

后续 q1 审计确认了这一判断：四组 1K/32K、Graph/eager 数据共有 40 次 q1 拒绝，
draft token 只有 1 次进入 target top-2。所有生产 sampler 与独立 PyTorch argmax 的差异
都是 top-2 完全同分，没有非同分错误。进一步做生产形态 A/B 后，q5 相对 q7 的六次
重复均值 decode 提升 `2.15%`，ms/round 降低 `4.23%`。因此启动器的 CMP170HX 实测
默认已从 q7 调整为 q5。官方模型 README 明确推荐 q7，所以 q7 仍是支持的生产候选，
不能把 `dspark_block_size=5` 解释成 q7 超出 DSpark 合同。

base/Markov 分解进一步覆盖 109 个 q1 轮次：最终 draft 命中 63 次，假设禁用 Markov
只命中 58 次，Markov 共救回 12 次、破坏 7 次。当前不应关闭或缩放 Markov；后续正确
对象是 drafter base predictor 的逐层数值一致性，而不是继续修改 rejection、top-k 或
Graph 路由。

## 验证配置

- 模型：`/media/z/data6/models/DeepSeek-V4-Flash-0731`
- GPU：物理卡 `1,2,3`，三张 CMP170HX，PP3/TP1
- PP 分层：`15,15,13`
- attention backend：`TRITON_MLA_SPARSE_DSV4`
- KV cache：FP8，block size 256，prefix cache 关闭
- `max_model_len=131072`
- `max_num_batched_tokens=1715`，`max_num_seqs=8`
- DSpark：q7、精确 top-k、Graph、temperature 0
- workload：精确 32768 prompt token、128 output token、六次重复
- prompt hash：
  `b5849387b36fa334059924da239dc703b55d264b8b13ceb5ffe9f33c7d9dd5a8`

## 1. SM80 noncausal sparse MLA 正确性

新增测试直接调用 Ampere backend 使用的 direct ragged metadata builder，并覆盖：

1. 小型与生产 q7 draft block 的 future-pointing index 集合；
2. FP8 UE8M0 paged cache；
3. 64 query heads 的生产 Triton sparse MLA decode；
4. dequant BF16 reference 对照；
5. causal reference 负对照，避免测试误把 causal 结果当成正确。

实机命令：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
  tests/v1/attention/test_dspark_noncausal_sparse_mla.py \
  -q -rs -k ampere
```

结果为 `3 passed, 28 deselected`。因此当前没有证据支持“SM80 direct ragged 漏掉
future draft token”这一假设。

## 2. mHC broadcast 对照

首层 mHC reference 路径显式构造 `[M,hc_mult,H]` residual，并调用原
`mhc_pre_tilelang`。生产 broadcast 路径仍是默认值，reference 只作为审计回滚路径。

### 隔离 kernel

`M=1667` 的 CMP170HX 单次样本：

| 路径 | 耗时 |
| --- | ---: |
| broadcast | 623.62 us |
| expanded reference | 1750.02 us |

reference 慢 `2.81x`。四个输出的最大绝对差分别为
`0 / 1.19e-7 / 5.96e-8 / 7.8125e-3`，说明两条路径并非逐 bit 等价，因此继续做了
真实模型 A/B/A，而不是只依赖 microtest。

### 32K A/B/A

| Arm | 路径 | pooled acceptance | mean accepted length | ms/round | decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| A1 | broadcast | 38.592% | 3.701 | 32.147 | 112.67 |
| B | expanded reference | 30.184% | 3.113 | 32.517 | 98.33 |
| A2 | broadcast | 31.452% | 3.202 | 32.779 | 98.06 |

B 相对 A2 仅低 `1.27 pp`，而 A1 与 A2 自身相差 `7.14 pp`；六次重复的每个 arm 都有
六个不同输出哈希。reference 没有形成超出控制臂漂移的正收益，且隔离性能显著更差，
因此拒绝晋升。

证据目录：

- `profiles/deepseek_v4/mhc-broadcast-ab/a1-broadcast/20260817T051535Z`
- `profiles/deepseek_v4/mhc-broadcast-ab/b-reference/20260817T052338Z`
- `profiles/deepseek_v4/mhc-broadcast-ab/a2-broadcast/20260817T052731Z`

`b-reference/20260817T052002Z` 是 reference 原型首次接错输入 shape 后触发断言的失败
工程记录，不计入性能 arm。

## 3. 固定 seed 与逐位置接受率

原 workload 的 `--seed` 只用于生成 prompt，没有传入 completions 请求；同时 Prometheus
的 `num_accepted_tokens_per_pos` 被主动忽略。本轮修复为：

- 请求显式传递 `seed`；
- 采集每个 draft position 的 counter；
- 对前后 metrics snapshot 做逐位置 delta；
- JSONL 同时保存计数和以 `num_drafts` 为分母的无条件接受率；
- profiler 父进程显式继承启动脚本关键默认值，避免 JSONL 记录与 server child 不一致。

行为测试为 `6 passed`，Ruff 检查通过。

### DSpark q7

证据：
`profiles/deepseek_v4/dspark-fixed-seed-per-pos/20260817T053441Z/result.jsonl`

| 指标 | 值 |
| --- | ---: |
| drafts / draft tokens / accepted | 219 / 1533 / 560 |
| pooled acceptance | 36.530% |
| mean accepted length | 3.557 |
| mean / median decode | 108.58 / 106.29 tok/s |
| mean TTFT | 5.905 s |
| 唯一输出哈希 | 6 / 6 |

逐位置接受计数为 `123 / 114 / 110 / 83 / 73 / 35 / 22`，除以 219 个 draft rounds
得到开头列出的逐位置接受率。各次总体接受率范围为 `31.36%` 到 `47.00%`，短样本仍有
明显轨迹方差。

### Target-only

启动脚本现在允许 `DSV4_DSPARK_TOKENS=0`，此时完全省略 speculative config，便于使用
同一套配置生成 target-only 对照。

证据：
`profiles/deepseek_v4/target-only-fixed-seed/20260817T053809Z/result.jsonl`

| 指标 | 值 |
| --- | ---: |
| mean / median decode | 61.96 / 62.14 tok/s |
| mean TTFT | 5.961 s |
| 唯一输出哈希 | 4 / 6 |
| 众数输出出现次数 | 3 / 6 |

DSpark 六个输出中的一个与 target-only 众数输出完整哈希相同。该证据说明 DSpark 可以
复现 target-only 轨迹，但也说明 temperature 0 加固定 seed 仍不足以让当前 PP3 稀疏模型
成为逐 token 确定的语义 oracle。

workload 峰值显存为：

| 路径 | PP0 | PP1 | PP2 |
| --- | ---: | ---: | ---: |
| DSpark q7 | 57596 MiB | 56682 MiB | 62310 MiB |
| target-only | 57400 MiB | 56436 MiB | 50436 MiB |

DSpark 的主要常驻/Graph 增量集中在 PP2，约 `11.6 GiB`；这不是本轮接受率修复对象，但
应进入后续上下文容量专题。

## 4. q1 sampler 审计与 checkpoint 原生宽度

新增的 q1 审计直接位于 rejection sampler 内部，读取已经完成 temperature 处理的 target
logits、实际 draft token、生产 rejection 输出与接受长度，同时用 PyTorch 计算 top-2。
它默认关闭、受样本数限制，并且会同步 GPU，因此审计数据不能作为生产绝对性能结果。
若 phase-profile 控制文件存在，审计只在正式测量窗口内取样，避免 Graph capture 和 warmup
提前耗尽预算。

### 1K/32K Graph/eager

| 上下文 | 执行 | q1 接受 | q1 拒绝 | 拒绝时 draft 在 target top-2 | sampler/torch 非同分差异 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1K | Graph | 16/27 | 11 | 0/11 | 0 |
| 1K | eager | 15/27 | 12 | 0/12 | 0 |
| 32K | Graph | 23/36 | 13 | 0/13 | 0 |
| 32K | eager | 20/34 | 14 | 1/14 | 0 |

1K 的 Graph 与 eager 前 11 个拒绝 token pair 完全一致，说明 Graph 会影响近同分边界后的
生成轨迹，但不是低 q1 接受率的主因。40 次拒绝中 39 次 draft 都不在 target top-2，问题
应继续向 drafter 的输入、base logits 与 Markov bias 分解，而不是 target argmax 或 top-k
集合。

证据目录：

- `profiles/deepseek_v4/dspark-q1-audit/graph-1k/20260817T055751Z`
- `profiles/deepseek_v4/dspark-q1-audit/eager-1k/20260817T060026Z`
- `profiles/deepseek_v4/dspark-q1-audit/graph-32k/20260817T060321Z`
- `profiles/deepseek_v4/dspark-q1-audit/eager-32k/20260817T060537Z`

### q5/q7 生产形态对照

checkpoint 的 `config.json` 声明 `dspark_block_size=5`，但官方模型 README 推荐
`num_speculative_tokens=7`。因此不能再把 q6/q7 的低无条件接受率归因于“超出原生
block”；它首先受更长前缀的连乘口径影响。关闭 q1 审计后，32K、128 output、Graph、
固定请求 seed 的六次重复为：

| 指标 | q5 | q7 | q5 相对 q7 |
| --- | ---: | ---: | ---: |
| pooled acceptance | 49.292% | 36.530% | +12.762 pp |
| mean accepted length | 3.465 | 3.557 | -2.60% |
| mean ms/round | 30.949 | 32.317 | -4.23% |
| mean decode | 110.92 tok/s | 108.58 tok/s | +2.15% |
| PP2 workload peak | 62138 MiB | 62310 MiB | -172 MiB |

在这组 greedy A/B 中，q5 少验证两个边际收益较低的尾部 token，虽然每轮平均接受长度
略降，但轮次更快，最终 decode 吞吐稳定提升，同时减少 PP2 Graph/工作区显存。该结果
支持 q5 作为本机实测默认，不否定官方 q7 配置；probabilistic 模式仍需独立 q5/q7 A/B。
TTFT 差异约 0.8%，处于本机运行漂移内，不作为晋升依据。

后续 RNG-domain 修复后的 32K、9-seed probabilistic A/B 已补齐。general 中 q7 的
accepted length 提高 `5.78%`，但 ms/round 增加 `5.38%`，合并 decode 仅 `+0.12%`；
agentic `top_p=0.95` 中 accepted length 下降 `15.43%`，合并 decode 下降 `16.19%`。
因此 q7 仍是官方支持宽度，但没有通过本机 probabilistic 晋升门，q5 继续作为
CMP170HX 默认。

证据：

- q5：`profiles/deepseek_v4/dspark-q5-promotion/20260817T061441Z`
- q7：`profiles/deepseek_v4/dspark-fixed-seed-per-pos/20260817T053441Z`
- probabilistic q7：`profiles/deepseek_v4/probabilistic-q7-post-rng`

## 5. q1 base logits 与 Markov bias 分解

checkpoint 自带的 `inference/model.py` 给出了原始 DSpark 合同：block 第一个位置放当前
anchor，后四个位置放 noise token；每个位置先计算 base logits，再把前一 token 对应的
Markov bias 相加并顺序采样。vLLM 当前 `sample_from_anchor=True` 与该参考一致。

hidden-state 接口也没有发现一层偏移。参考实现取 target layer 40/41/42 的层后状态；
vLLM 的捕获编号表示层边界，因此正确映射为 41/42/43，启动日志也确认三张 PP rank
使用了这一配置。

新增 drafter 审计记录：

- base、bias 和实际 BF16 加法后的 final top-2；
- draft token 在 base logits 中的严格 rank 与同分区间；
- base top-1 经 Markov 修正后的 rank；
- target 侧 draft token 的完整严格 rank 与同分区间；
- base-only 是否会命中 target，以及 Markov 是救回还是破坏 q1。

审计只采 temperature 0 的可消费 proposal：跳过无下游消费者的中间 prefill chunk，但
保留最终 prefill chunk 产生的第一组 draft。它在 eager 下同步 129,280 词表，以下运行的
耗时不能用于生产性能对比。

| 运行 | 配对轮次 | final q1 | base-only q1 | Markov 改变 top-1 | 救回 / 破坏 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1K eager | 32 | 19 (59.38%) | 17 (53.12%) | 16 | 3 / 1 |
| 32K eager A | 42 | 23 (54.76%) | 23 (54.76%) | 24 | 6 / 6 |
| 32K eager B | 35 | 21 (60.00%) | 18 (51.43%) | 17 | 3 / 0 |
| 合计 | 109 | 63 (57.80%) | 58 (53.21%) | 57 | 12 / 7 |

Markov head 经常改变 base argmax，但三条轨迹合计净救回 5 个 q1；直接禁用 Markov 会
降低而不是提高接受率。32K B 的 14 次拒绝中，draft 严格 target rank 中位数为 2、均值
为 2.79，其中 9 次位于 target top-2，说明当前剩余误差更多是 base predictor 与 target
在近邻候选间排序不同，而不是 Markov 整体失效。

修正版 32K B 从 position 32768 开始，共 36 条 drafter 与 35 条 target 记录；最后一条
proposal 因达到输出上限没有 target 消费，符合调度语义。唯一的 draft ID 与 PyTorch
final top-1 ID 差异发生在两个 token 的 BF16 final logit 都为 19.0，属于同分选择。

证据：

- 1K：`profiles/deepseek_v4/dspark-q1-stage-audit/eager-1k/20260817T063607Z`
- 32K A：`profiles/deepseek_v4/dspark-q1-stage-audit/eager-32k/20260817T063830Z`
- 修正版 32K B：
  `profiles/deepseek_v4/dspark-q1-stage-audit/eager-32k-final/20260817T065010Z`

## 后续门槛

1. 对同一组捕获输入逐层比较 `main_proj/main_norm`、三个 draft layer、hc_head/norm 和
   LM head 的高精度 reference，先判断 base predictor 是否存在实现数值漂移；不得根据
   单条生成轨迹调 Markov scale。
2. 使用固定任务集做 target-only 与 q5 的语义评分；输出哈希只保留为轨迹证据，不能替代
   质量门。
3. 保持 Markov 和 q5 生产默认。q6/q7 exact top-k 继续作为 selector 集合正确性门，但
   不再作为当前生产接受率优化主线。

## 6. 进一步排除：head 精度、SWA 路径与 proposal 宽度

官方 checkpoint 的 `inference/model.py` 将 BF16 LM-head 权重放在 FP32 parameter 中，
并用 FP32 `F.linear` 计算 logits。vLLM 可通过 `head_dtype=float32` 使用 BF16 输入/权重、
FP32 输出累加而不常驻一份 FP32 权重，因此先做了相同固定 workload 的诊断：

| 配置 | 上下文 | q1 | 完整接受率 | decode |
| --- | ---: | ---: | ---: | ---: |
| 默认 BF16 head | 1K | 19/33 | 59.38% | 15.21 tok/s |
| FP32 head | 1K | 21/32 | 62.50% | 14.67 tok/s |
| 默认 BF16 head | 32K | 23/35 | 55.43% | 13.00 tok/s |
| FP32 head | 32K | 28/39 | 45.13% | 11.58 tok/s |

FP32 head 会改变近同分候选和后续生成轨迹；32K 虽提高 q1，却让 q2 及以后明显下降，
完整接受率和 decode 均退化。它不是可晋升的接受率修复，生产继续使用模型 dtype。
证据为 `profiles/deepseek_v4/dspark-fp32-head-audit/{1k,32k}`。

关闭 `VLLM_DEEPSEEK_V4_DIRECT_RAGGED_SWA` 后，1K 的输出 hash、95/160 接受 token、
五个位置接受率均与既有默认路径逐项相同，说明 SM80 direct-ragged SWA 不是当前损失源。
关闭整个 PP receive reuse，或只恢复 DSpark dedicated hidden buffer，都会让近同分轨迹
发生变化，但完整接受率仍约 59%，没有系统性恢复；该差异更符合地址/launch 改变放大
边界排序，而不是 aux hidden 生命周期覆盖。对应证据：

- `profiles/deepseek_v4/dspark-ragged-audit/gathered-1k`；
- `profiles/deepseek_v4/dspark-pp-recv-audit/reuse-off-1k`；
- `profiles/deepseek_v4/dspark-pp-recv-audit/recv-on-workspace-off-1k`。

上游 issue #50012 报告了短于 checkpoint block 的 q3 生产经验。为避免把不同 proposal
宽度的接受百分比直接比较，本机临时把 `num_speculative_tokens < dspark_block_size` 的
硬错误降为 warning，完成 q3/q4 Graph 扫描后恢复原校验；没有把实验开关留入代码。

| 32K Graph | repeats | pooled acceptance | accepted length | ms/round | decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| q3 | 3 | 68.254% | 3.051 | 30.219 | 100.28 tok/s |
| q4 | 3 | 58.836% | 3.388 | 30.095 | 110.08 tok/s |
| q5 | 6 | 49.292% | 3.521 | 30.949 | 110.92 tok/s |

q3 的 round 快 2.36%，但 accepted length 少 13.35%，最终 decode 比 q5 慢 9.59%；
q4 的平均 decode 仍比 q5 慢 0.75%，没有足够收益进入 A/B/A。由此可见“约 60% 接受率”
不能脱离 draft 宽度解释：q5 分母更大，但每轮实际前进 token 更多。生产默认继续保持
CMP170HX 实测 q5。证据目录为
`profiles/deepseek_v4/dspark-width-audit/{q3-graph-32k,q4-graph-32k}`。

剩余主线仍是同一捕获输入的逐层高精度对照。优先顺序为 `main_proj/main_norm`、三层
draft backbone、`hc_head/norm`；LM head、direct-ragged SWA、PP workspace 和 q3/q4
已经由本轮实机对照降级，不应继续围绕这些变量做无证据调参。

## 7. Markov scale 同输入旁路审计

为避免“改 scale 后轨迹分叉”污染结论，q1 drafter audit 增加了默认关闭的旁路网格：
对同一份 base logits 与 Markov bias 计算 scale `0/0.5/0.75/1/1.25/1.5` 的 BF16
top-1，但实际 proposal 始终保持 checkpoint 原生 scale 1。每轮再与 rejection sampler
看到的 target ID 对齐，因此不同 scale 共享相同输入和 target logits。

| 上下文 | 配对 | scale 0 | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 31 | 25 | 23 | 21 | 20 | 20 | 19 |
| 32K | 36 | 22 | 22 | 21 | 23 | 22 | 22 |
| 128K | 42 | 18 | 21 | 23 | 29 | 29 | 27 |
| 合计 | 109 | 65 | 66 | 65 | **72** | 71 | 68 |

1K 单轨迹会误导为“关闭 Markov 最优”，但 32K 已反转，128K 则以 scale 1/1.25
明显领先；109 个新配对中原生 scale 1 为 66.06%，是网格最优。长上下文还出现较多
target BF16 cutoff 同分：sampler ID 与 `torch.topk` 首项不一致为 1K 的 1/31、32K 的
5/36、128K 的 8/42，因此分析以生产 sampler ID 为准，不能把 top-k tie 顺序误记为
拒绝。

结论是保留 checkpoint 原生 Markov scale 1，不增加运行时 scale 旋钮。旁路字段仅在
已有 `VLLM_DEEPSEEK_V4_DSPARK_Q1_AUDIT=1`、eager 诊断时产生，生产路径没有额外
kernel、workspace 或显存。证据目录：

- `profiles/deepseek_v4/dspark-markov-scale-audit/1k`；
- `profiles/deepseek_v4/dspark-markov-scale-audit/32k`；
- `profiles/deepseek_v4/dspark-markov-scale-audit/128k`。

## 8. 逐层/快路排除与统计口径收敛

继续按 q1 实际链路核验后，没有发现可晋升的数值修复：

- `main_proj` 的真实形状 `M x 12288 x 4096` 在 CMP170HX 上用 FP8 Marlin 对 BF16
  反量化参考，`M=1/5/8` 的最大绝对误差为
  `0.0078125/0.015625/0.015625`；
- `main_norm` 的 `M=1/5/8, H=4096` BF16 C++ kernel 与 FP32 RMSNorm 后舍入结果
  逐元素一致；
- mHC pre/post、hc_head 的 32 个相关 TileLang/FP32 参考用例通过，Ampere
  noncausal sparse MLA 的 3 个可执行参考用例通过；
- checkpoint 的 target layer `[40,41,42]` 正确转换成层后边界 `(41,42,43)`；
  `15,15,13` 与 `15,16,12` 两种 PP3 分层都把三份 aux state 放在 PP2；
- 关闭 DSpark KV-only context insert 后仍为 `95/160`、accepted length
  `3.96875`；关闭 mHC pre-broadcast 后更与默认组输出 hash、逐位置 `19/32`
  全部相同。

对应服务证据为
`profiles/deepseek_v4/dspark-kv-insert-audit/generic-1k` 与
`profiles/deepseek_v4/dspark-mhc-pre-audit/broadcast-off-1k`。前者会改变 BF16
近同分后的生成轨迹，但没有系统性接受收益；后者与默认路径逐项相同。因此生产继续
启用两条快路。

profiling JSONL 进一步增加
`conditional_per_position_acceptance_rates`：第 0 位以 draft rounds 为分母，后续
位置以“已到达前一位置”的计数为分母。原有
`per_position_acceptance_rates` 仍是以全部 rounds 为分母的无条件存活率。以后报告
q5 必须同时看 accepted length、ms/round、decode tok/s 和这两组逐位置指标，不能把
单独的总体 `accepted_tokens / draft_tokens` 当成 DSpark 质量结论。

既有 32K q5 六次运行合并后，无条件逐位置为
`66.81%/56.19%/47.79%/40.71%/34.96%`；换成条件口径则为
`66.81%/84.11%/85.04%/85.19%/85.87%`。主要损失集中在第一个 proposal，后四位
一旦到达后都保持约 84%--86%。同输入 Markov 审计也表明原生 scale 1 相对 scale 0
把 q1 命中从 65/109 提升到 72/109：修复 13 行、损伤 6 行，净增 7 行。因此关闭
Markov 并不是 q1 优化。

当前结论是本机实测 `q5 + checkpoint Markov scale 1` 已达到 accepted length
`3.521`、约 `110.92 tok/s` 的 32K Graph 基线；继续为抬高“接受百分比”调 scale、
head dtype 或 proposal 宽度会降低实际吞吐。后续性能优化应以 q5 的 `ms/round` 为目标，
而不是继续寻找已无证据支持的 SM80 接受率 bug。
