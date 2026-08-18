# DSV4F SM80 PP3 probabilistic DSpark 验证与优化

## 结论

本轮在三张 CMP170HX、PP3/TP1、CMP170HX 实测默认 DSpark q5 上完成了
`temperature=1.0` 的 probabilistic 专项实机门禁。普通请求固定
`top_p=1.0`，agentic 请求固定 `top_p=0.95`。

- 必须把 speculative config 的 `draft_sample_method` 显式设为
  `probabilistic`；只设置请求 temperature 会让 draft 继续走 greedy。
- 已修复 draft Gumbel 与 target rejection Gumbel 复用同一
  `(request seed, position)` 随机域的问题。该问题基本不改变期望接受率，但会破坏
  speculative output 与 target 分布的一致性；修复后 draft 使用独立 RNG domain。
- 1K/32K/128K/256K 的 24 组 DSpark 请求相对 target-only 均有 decode
  加速，范围为 `1.19x-1.59x`；这些运行发生在 RNG 修复前，只能作为性能和任务
  观测基线，不能作为修复后 exact-distribution 的 promotion 证据。
- 长上下文检索 DSpark 与 target-only 都是 `22/24`；GSM8K 32 题和 12 次
  工具调用没有出现 probabilistic DSpark 质量回退信号，但样本规模不足以宣称
  统计等价或生产任务全覆盖，且同样需要在独立 RNG domain 下复验。
- 修复后的 32K、9-seed q5 实测总体接受率为 general `43.57%`、agentic
  `43.15%`，但 mean accepted length 仍为 `3.179/3.158`。条件逐位置接受率约
  `68%-80%`；总体百分比低主要来自 q5 前缀连乘，不能解释成 rejection 实现退化。
- 官方 README 推荐的 q7 也已在修复后完成同 workload A/B。general 合并 decode
  与 q5 基本持平（`+0.12%`），agentic `top_p=0.95` 则下降 `16.19%`；q7 没有通过
  本机 probabilistic 晋升门。
- agentic target 的全词表 top-p 处理约为 `1.066 ms/round`，是
  probabilistic 专属的首要 P0。general 的 `top_p=1.0` 已直接跳过该路径。
- 两个低风险候选都没有通过 promotion gate：复用 exponential noise 临时张量
  的收益太小，32K 服务复测方向不利；把 B6 强制切到现有 Triton top-p 则更慢且
  cutoff mask 不完全一致。两项均未进入生产代码。

因此当前可用配置仍是 `q5 + probabilistic`，但结论限定在本文硬件、拓扑、采样参数和
测试集。当前没有证据支持通过 temperature、draft top-k/top-p、Markov scale、固定
截短或 confidence 截短提高真实吞吐；后续应优化 q5 的 ms/round 和 draft predictor，
而不是为了抬高单个接受率百分比牺牲 accepted length。

## 配置合同

服务端：

```bash
CUDA_VISIBLE_DEVICES=1,2,3 \
DSV4_DSPARK_TOKENS=5 \
DSV4_DSPARK_DRAFT_SAMPLE_METHOD=probabilistic \
DSV4_MODE=production \
benchmarks/deepseek_v4/run_cmp170hx_pp3.sh
```

1M-capable 单请求测试使用 `run_cmp170hx_pp3_1m.sh`，对应 PP 分层
`15,16,12`、每 rank 2.7 GB KV cache 和 `max_num_seqs=1`。本轮验证请求严格采用：

| 场景 | temperature | top_p |
|---|---:|---:|
| general | 1.0 | 1.0 |
| agentic | 1.0 | 0.95 |

draft probabilistic 路径当前只采用 temperature，不在 proposal 内重复应用 top-p。
target rejection 对 target logits 应用完整采样参数，所以输出仍服从 target 采样分布；
代价是 draft 与 target 分布不完全相同时接受率可能下降。基准工具将 scenario、
temperature、top_p、seed 和 draft sample method 写入结果，避免把 greedy draft 的数据
误标为 probabilistic。

## max 思考下 temperature 0/1 对照

为避免把采样策略和 agentic cutoff 混在一起，本组只改变 temperature，固定 q5
probabilistic DSpark、general、`top_p=1.0`、32,768 个完整 chat-template prompt tokens、
128 output tokens，并对 seed 8400-8408 做九组配对。偶数 seed 按 temperature 0→1，
奇数 seed 按 1→0 执行，降低顺序偏差。

`reasoning_effort=max` 不是只写入结果元数据：同一条短消息经 `/tokenize` 验证，
low/high 均为 10 tokens，max 为 89 tokens；正式请求使用
`/v1/chat/completions`，并由 harness 解析 reasoning 与 content 两种增量。正式 prompt
长度是在渲染 max chat template 后精确补齐到 32,768，而不是先构造 raw prompt 再附加
思考前缀。

| temperature | accepted / draft | pooled 接受率 | accepted length | 逐位置接受率 |
|---:|---:|---:|---:|---|
| 0 | 794 / 1820 | 43.626% | 3.181 | 78.02% / 61.54% / 35.71% / 25.82% / 17.03% |
| 1 | 778 / 1845 | 42.168% | 3.108 | 75.88% / 55.83% / 36.86% / 26.29% / 15.99% |

temperature 1 相对 0 的 pooled 差值为 `-1.458 pp`。按九个相同 seed 分别计算后，
配对均值差为 `-1.609 pp`，配对标准差为 `10.035 pp`，近似 95% 区间为
`[-9.32 pp, +6.10 pp]`。当前样本不支持“temperature 0 会系统性提高接受率”或
“temperature 1 导致明显接受率退化”；相比温度效应，prompt/生成轨迹方差更大。
采样温度仍应由任务语义决定，不能为了这约 1.5 pp 的点估计改成 greedy。

本轮 128-token 输出上限会让 max reasoning 在给出最终答案前被截断；needle 通过数
为 4/9 与 6/9，因此这组短输出只用于接受率，不构成语义质量对照。原始证据位于：

`profiles/deepseek_v4/probabilistic-temp-max-reasoning/sync-temp{0,1}-*.json`

首次常规异步服务在完成一个 temperature 0 样本后，下一条 temperature 1 请求的
16,830-token prefill 处出现 PP1 illegal memory access。干净服务的 temperature 1
单请求可通过；启用 `CUDA_LAUNCH_BLOCKING=1` 后，18 个交替请求全部通过且服务健康。
这表明温度 1 本身不是确定性触发条件，但 PP 接收缓冲/异步 kernel 生命周期仍有独立
稳定性风险。同步运行不改变 sampler 数学，故接受率计数保留；其 ms/round、TTFT 和
tok/s 均不作为性能数据。

## RNG 独立性修复与 p/q overlap 审计

原实现的 draft Gumbel 和 target rejection Gumbel 使用同一 request seed 和绝对 position。
两次随机量因此相关。它不会明显改变 `min(p, q)` 的期望接受率，却会让 rejection 后的
输出偏离 target 分布。合成 `V=128, B=262144` 验证为：

| RNG | acceptance | target output TV | chi-square，df=127 |
|---|---:|---:|---:|
| 共享 domain | 56.565% | 0.03266 | 1654.66 |
| draft/target 独立 domain | 56.493% | 0.00643 | 112.15 |

修复为 target 保持 domain 0，draft 使用带盐的 domain 1；默认 greedy 和 target-only
路径不变。GPU rejection suite 的 30 项测试通过，其中包含“draft RNG 改变但 target
边际分布保持”的回归用例。

实机 audit 同时保存 target 分布 `p`、draft 分布 `q`、精确 overlap、实际接受事件、
support mismatch 和同输入 counterfactual。两批 32K 轨迹合并如下：

| 场景 | reached | 实际接受 | 精确 overlap | draft temp=0.9 | draft top_p=.95 | q 落在 target support 外 | target-support oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| general / top_p=1 | 377 | 63.395% | 65.833% | 65.740% | 65.586% | 0 | 65.833% |
| agentic / top_p=.95 | 373 | 70.777% | 70.167% | 70.243% | 70.594% | 14.372% | 76.786% |

实际接受与 `sum(min(p, q))` 在有限轨迹误差内一致，没有发现 rejection stride、索引、
temperature 或 cutoff 漏处理。general 中 temperature/top-p shaping 均不改善 overlap；
agentic 中 draft 自己做 top_p=.95 只提高约 `0.43 pp`，而完整 target-support oracle 的
理论上限约 `+6.62 pp`。后者必须先知道 target support，不能作为免费 proposal 优化。
top-k 1/2/4/8/16/32/64 的同输入扫描也表明压尖 q 普遍降低 overlap。

修复后新增 32K、9-seed、128 output 的生产 Graph 统计。总体接受率是“接受草稿数 /
所有五个 proposal 槽”，后四位未到达也在分母中；条件接受率则只在到达该位置的行上
计量：

| 场景 | pooled 总体接受率 | accepted length | p0 | p1\|到达 | p2\|到达 | p3\|到达 | p4\|到达 |
|---|---:|---:|---:|---:|---:|---:|---:|
| general | 43.571% | 3.179 | 72.80% | 74.34% | 71.57% | 80.14% | 68.14% |
| agentic | 43.152% | 3.158 | 69.57% | 74.22% | 77.37% | 78.91% | 73.28% |

所以约 `43%` 的数字不是“每个实际验证位置只接受 43%”，而是 q5 前缀存活率展开后的
汇总。真实优化门仍是 accepted length、ms/round 和 decode tok/s。

原始证据：

- `profiles/deepseek_v4/probabilistic-acceptance-audit/counterfactual`
- `profiles/deepseek_v4/probabilistic-acceptance-audit/proposal-grid`
- `profiles/deepseek_v4/probabilistic-q5-submit-ab/q5`

## 官方 DSpark 参考对拍

为区分“SM80 实现错误”和“DSpark predictor 本身与 target 分布有距离”，新增了一次
真实 1K proposal 输入捕获，并用官方 checkpoint 的 `DSparkBlock` 架构、4708 个相关
参数和等价量化数学逐层重放。checkpoint 中三层权重仍命名为 `mtp.0/1/2`，这只是官方
文件的历史参数命名空间；本轮运行、捕获和比较的对象都是 DSpark，不是 MTP。

官方发布的 TileLang linear kernel 在 CMP170HX 上会请求 SM89 FP8 MMA 并直接报设备
断言，无法作为 SM80 可执行基线。比较工具因此保留官方 DSpark 层结构和权重，在
SM80 上自动替换为 FP32 累加的等价 FP8/FP4 反量化、sparse attention 和 mHC 数学参考。
这不是官方 SM89 kernel 的位级重放，适合做语义一致性诊断，不用于性能计量。

| 对拍对象 | 结果 |
|---|---:|
| `main_x` | cosine `0.999522`，RMSE `0.002237` |
| final head hidden | cosine `0.999557`，RMSE `0.003411` |
| base logits | 5 行分布 overlap `0.9765-0.9797`，top32 `32/32` |
| Markov bias | 5 行 overlap `0.9951-0.9963`，top1 `5/5` |
| final logits | 5 行 overlap `0.9671-0.9971`，top1 `5/5` |

final-logits 比较复用了捕获轨迹的同一组 previous draft ID，因此隔离的是 predictor 数学，
不是随机采样轨迹。vLLM 与官方参考的逐位置 TV 仅约 `0.3-3.3 pp`，明显小于前述
DSpark q/target p 的条件接受差距。当前证据否定了“SM80 DSpark 实现损坏导致约 60%
接受率”这一假设；主要损失来自 predictor 与 target 的内在分布距离，以及 pooled 指标
对未到达尾部位置的计数方式。

复现工具和结果：

- `benchmarks/deepseek_v4/compare_dspark_official_reference.py`
- `profiles/deepseek_v4/dspark-official-reference/vllm-1k.pt`
- `profiles/deepseek_v4/dspark-official-reference/official-vs-vllm.json`

## q5/q7 probabilistic 对照

官方模型 README 的 vLLM 示例推荐 q7，checkpoint config 同时声明
`dspark_block_size=5`。两者都属于 DSpark；不能根据字段名假定 q7 非法，也不能根据
官方 greedy 示例假定 q7 在 probabilistic 下必然更快。为此使用与 q5 完全相同的
32,768-token needle workload、seed `8400-8408`、128 output 和生产 Graph 做了 q7
补测。吞吐按 9 次请求的总 token / 总 decode 时间合并。

| 场景 | 宽度 | pooled 接受率 | accepted length | ms/round | 合并 decode |
|---|---:|---:|---:|---:|---:|
| general | q5 | 43.571% | 3.179 | 33.268 | 94.39 tok/s |
| general | q7 | 33.747% | 3.362 | 35.058 | 94.50 tok/s |
| agentic | q5 | 43.152% | 3.158 | 33.875 | 91.69 tok/s |
| agentic | q7 | 23.865% | 2.671 | 34.511 | 76.84 tok/s |

general 中 q7 的 accepted length `+5.78%` 恰好被 ms/round `+5.38%` 抵消；吞吐差
`+0.12%` 没有晋升价值。agentic 中 q7 的 accepted length `-15.43%`，ms/round
`+1.88%`，最终吞吐 `-16.19%`。两种场景 needle 都为 `9/9`，但这里只是回归哨兵。
生产默认继续使用 q5。

第一次 q7 服务在先前客户端中断之后、进入 agentic prefill 时出现一次 PP1 illegal
memory access。干净重启后同一 seed 和完整 9-seed agentic 组均成功，故未把该事件归因
于 q7/top-p；其日志保留为稳定性观察项，不计入上述 A/B。

原始证据：

- `profiles/deepseek_v4/probabilistic-q5-submit-ab/q5`
- `profiles/deepseek_v4/probabilistic-q7-post-rng`

## checkpoint confidence head 与宽度否决

checkpoint 包含 `mtp.2.confidence_head.proj.weight`。审计模式按官方实现，用 hc_head
pre-norm hidden 与当前 Markov embedding 拼接后做 FP32 projection；生产默认不加载该
权重，也不增加 workspace。32K 的 415 个真实记录（另有 5 个 profile warmup）得到：

| 场景 | reached | confidence AUC | 精确 overlap AUC | confidence/overlap 相关系数 | 最低/最高五分位实际接受 |
|---|---:|---:|---:|---:|---:|
| general | 124 | 0.766 | 0.905 | 0.605 | 50% / 100% |
| agentic | 124 | 0.694 | 0.871 | 0.502 | 54.2% / 96.0% |
| pooled | 248 | 0.736 | 0.891 | 0.555 | 51.0% / 100% |

confidence 有预测力，但不能增加 `p/q` overlap。离线阈值 `logit >= -1` 可把平均提交
宽度从 `5` 降至 `4.349`，平均接受草稿仅从 `2.193` 降至 `2.169`。为验证这是否能转化
成性能，使用 vLLM dynamic SD 做了有效的 `q5-generate/q4-submit`：draft 仍按 q5
计算，只把前四个 token 交给 target，区别于会改变 block 布局的直接 q4 配置。

| 场景 | arm | pooled 接受率 | accepted length | pooled ms/round | mean decode tok/s | needle |
|---|---|---:|---:|---:|---:|---:|
| general | q5→q4 | 53.057% | 3.122 | 35.358 | 92.25 | 9/9 |
| general | q5 full | 43.571% | 3.179 | 33.268 | 100.57 | 9/9 |
| agentic | q5→q4 | 55.208% | 3.208 | 36.553 | 90.67 | 9/9 |
| agentic | q5 full | 43.152% | 3.158 | 33.875 | 104.62 | 9/9 |

九个 seed 的 decode 轨迹方差很大，不能把均值差解释成精确 kernel 收益；但截短没有
降低 ms/round，且两个场景都没有吞吐晋升证据。当前 full Graph/padding 下少提交一行
不会消除 q5 draft backbone，反而引入 dynamic schedule 形态。confidence 动态宽度因此
停留在审计结论，不增加生产开关。原始证据为：

- `profiles/deepseek_v4/probabilistic-acceptance-audit/confidence`
- `profiles/deepseek_v4/probabilistic-q5-submit-ab`

## 长上下文服务对照

每个上下文、每个场景使用 seed 170/171/172，单请求，128 output tokens。吞吐按三次
运行的总 output tokens / 总 decode 时间合并；接受率和 accepted length 按真实 draft
计数合并，不平均百分比。

| 上下文 | 场景 | DSpark tok/s | target tok/s | 加速 | 接受率 | accepted length | ms/round | 检索 DS / target |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1K | general | 87.44 | 60.14 | 1.454x | 38.33% | 2.917 | 33.01 | 3/3 / 3/3 |
| 1K | agentic | 70.75 | 59.42 | 1.191x | 27.16% | 2.358 | 33.24 | 3/3 / 3/3 |
| 32K | general | 94.31 | 60.02 | 1.571x | 43.47% | 3.174 | 33.39 | 2/3 / 3/3 |
| 32K | agentic | 93.36 | 59.41 | 1.571x | 43.80% | 3.190 | 33.73 | 3/3 / 3/3 |
| 128K | general | 92.91 | 58.56 | 1.586x | 44.83% | 3.242 | 34.17 | 2/3 / 2/3 |
| 128K | agentic | 86.06 | 57.84 | 1.488x | 41.44% | 3.072 | 35.42 | 3/3 / 3/3 |
| 256K | general | 75.98 | 57.82 | 1.314x | 35.36% | 2.768 | 36.34 | 3/3 / 2/3 |
| 256K | agentic | 79.27 | 57.29 | 1.384x | 38.18% | 2.909 | 36.41 | 3/3 / 3/3 |

DSpark 与 target-only 的 mean TTFT 差值在约 `-0.10s` 到 `+0.20s` 内，远小于
长 prompt 本身的 0.5-69 秒范围，不能归因为 speculative sampling。256K 的
accepted length 和吞吐下降说明长上下文 target q6 扫描仍是增长项，而不是
probabilistic sampler 自身随上下文增长。

原始证据：

- `profiles/deepseek_v4/probabilistic-p0/dspark-longctx`
- `profiles/deepseek_v4/probabilistic-p0/target-longctx`

## 语义与 agentic 门禁

GSM8K 使用 32 题、5-shot、256 max tokens、general 采样参数：

| 配置 | 正确 | invalid | output tok/s |
|---|---:|---:|---:|
| DSpark probabilistic | 31/32 | 0 | 75.76 |
| target-only | 30/32 | 0 | 47.10 |

一题差异属于 temperature 1.0 的随机轨迹，不能解释为 DSpark 提升质量。agentic
测试覆盖 weather、calculator、documentation search 和 order lookup 四种工具，每项
重复三次：DSpark 与 target-only 都是 `12/12` 正确工具名和必需参数；DSpark 用时
8.60 秒，target-only 用时 15.95 秒。DSpark 这 12 次请求的合并接受率为 `89.61%`，
accepted length 为 `5.480`。

原始证据：

- `profiles/deepseek_v4/probabilistic-p0/quality/dspark-gsm8k.json`
- `profiles/deepseek_v4/probabilistic-p0/quality/target-gsm8k.json`
- `profiles/deepseek_v4/probabilistic-p0/quality/dspark-agentic.json`
- `profiles/deepseek_v4/probabilistic-p0/quality/target-agentic.json`

这些是回归哨兵，不是生产质量的充分证明。生产准入还需扩展到真实 agent trajectory、
多轮工具结果回填、结构化 schema、失败恢复和更大规模的随机种子。

## probabilistic 分解

32K agentic 的 eager CUDA Event profile 完整覆盖三个 PP rank。CUPTI 在 CMP 设备上
报告不支持，因此不使用 kernel table；阶段计量仍通过 CUDA Event 验收。稳定 decode
每轮的关键路径为：

| 阶段 | 每轮耗时 |
|---|---:|
| PP3 target q6 串行模型 | 约 23.60 ms |
| DSpark proposal | 47.24 ms |
| draft backbone | 31.36 ms |
| draft Markov total | 6.82 ms |
| target sampling total | 3.00 ms |
| target rejection | 2.06 ms |
| agentic apply sampling params | 1.066 ms |
| draft Markov sample | 0.289 ms/position，约 1.45 ms/round |

父子 range 相互重叠，表中不能直接求和。相对既有 greedy profile，probabilistic 的
额外开销约为 `2.1 ms/round`：draft 五个位置的分布采样约增加 1.0 ms，target
probabilistic rejection 约增加 1.0 ms。完整证据位于：

`profiles/deepseek_v4/probabilistic-p0/profile-32k-agentic/20260817T091628Z`

## P0 候选与准入结果

### exponential-noise 临时张量

现有 FP32 sampling 保留 draft probs，并克隆一份约 1.97 MiB 的 B1/V129280 张量做
exponential race。复用 noise tensor 的隔离测试由 127.78 us 降到 105.38 us，临时
峰值由约 1.97 MiB 降到 1.48 MiB；B8 性能持平。理论 q5 上限只有约
`0.11 ms/round`，且不增加或减少常驻显存。

32K 三 seed 服务复测没有确认收益：

| 场景 | baseline ms/round | candidate ms/round | baseline tok/s | candidate tok/s |
|---|---:|---:|---:|---:|
| general | 33.386 | 34.184 | 94.31 | 85.74 |
| agentic | 33.728 | 34.479 | 93.36 | 92.86 |

两组的接受轨迹不同，吞吐不能单独归因；但 ms/round 同样没有通过“不退化”门禁。
候选已回退，原始复测保留在
`profiles/deepseek_v4/probabilistic-p0/candidate-32k`。

### B6 Triton top-p 路由

q5 target verification 产生 6 行 logits，通用 sampler 在行数小于 8 时选择 PyTorch
精确 sort。CMP170HX、B6/V129280、top_p 0.95 的五次隔离中位数为：

| 路径 | 中位耗时 |
|---|---:|
| PyTorch sort | 552.05 us |
| 现有 Triton pivot | 903.83 us |

Triton 慢 63.7%，而且随机输入的 6 行中有 5 行 mask 与 PyTorch 不完全相同，逐行
cutoff 差异为 22/0/2/11/9/12 个 token。因此不降低通用阈值，也不把近似 pivot
结果引入 probabilistic target 分布。

## 后续优化顺序

1. P0：实现 B1-B8、V129280 的精确 top-p 专用原型。目标是融合 BF16→FP32、
   cutoff 选择和最终 mask 写回，严格定义同分 token 处理；必须与 target-only 使用相同
   语义，并覆盖 CUDA Graph、显式 seed、logprobs 和 grammar/penalty fallback。
2. P1：研究 Markov `embedding + BF16 rank-256 projection + base add` 融合。当前
   projection 约 0.13 ms/position；只有 q5 整轮和服务 A/B 都获益才晋升。量化权重会
   改变 proposal 分布，必须额外做接受率与质量门禁。
3. P1：继续优化 target q6 backbone。它仍占每轮绝大多数时间，且 128K→256K 的
   sparse indexer 增长会压低 probabilistic 和 greedy 两种 DSpark 的收益。
4. P2：扩大质量集。至少加入真实多轮 agentic trajectory、工具结果回填、JSON schema
   和 1K/32K/128K/256K 多 seed 统计置信区间，再讨论生产级质量声明。

所有 kernel 候选继续遵守：隔离正确性和 CUDA Graph micro → 相同 prompt/seed 的
服务 A/B/A → acceptance/accepted length/ms-per-round/decode 联合判定 → 显存增量门禁。
