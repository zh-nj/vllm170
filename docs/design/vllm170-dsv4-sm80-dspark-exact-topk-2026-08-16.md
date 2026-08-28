# CMP170HX DSpark q6/q8 精确 top-k

日期：2026-08-16

## 结论

CMP170HX 的 SM80 `persistent_topk` 在 DeepSeek-V4-Flash DSpark q6/q8 真实
logits 上存在少量集合漏选。生产启动器改为仅在以下条件同时满足时使用
`top_k_per_row_decode` 精确 radix selector：

- SM80；
- 单请求 DSpark q6 或 q8 target verification；
- `topk_tokens=512`；
- `VLLM_DEEPSEEK_V4_DSPARK_EXACT_TOPK=1`。

其余 decode shape 继续使用原 cooperative/persistent 路由。该开关在项目环境定义中
默认开启，启动器无需重复设置；可显式设为 `0` 回滚。

## 真实 logits 正确性审计

审计使用三张 CMP170HX、PP3、eager。审计器可在 persistent 或 exact selector
返回后按行调用独立 `torch.topk` 数学参考，并分别检查 index 集合、score 多重集、
index 合法性与唯一性、短行 padding、cutoff 同分和非有限值。32K persistent 历史结果
保留如下；生产 exact 路径的四档覆盖见下一节。

| 配置 | layer 样本 | logits 行 | 漏选候选/总候选 | recall | 漏选行 |
| --- | ---: | ---: | ---: | ---: | ---: |
| q6，DSpark-5 | 24 | 144 | 28/73,728 | 99.9620% | 未单独汇总 |
| q8，DSpark-7 | 24 | 192 | 19/98,304 | 99.9807% | 3 |

q8 的漏选分布为 PP0 18 个、PP1 1 个、PP2 0 个；最差单行漏选 11 个。漏选数量虽
小，但它会改变 sparse MLA 的候选集合，属于正确性差异，不能仅凭平均 recall 忽略。

## Exact 路径四档真实 logits 覆盖

固定三张 CMP170HX、PP3 `15,16,12`、`max_model_len=1,048,576`、
`max_num_seqs=1`、DSpark-7、FP8 indexer cache、eager，并开启 exact 生产路由。
每档发送一条精确 token 数的确定性针检索请求，生成 16 token。审计同步 GPU，以下
耗时不能用于性能结论。

| 原始上下文 | q8 压缩长度 | 样本 | 行 | 候选位置 | index 差异 | cutoff 同分行 | 严格 score 漏选 | score 多重集差异 | 语义 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1K | 约 256 | 0 | 0 | 0 | 不适用 | 不适用 | 不适用 | 不适用 | 1/1 针正确 |
| 32K | 8192–8198 | 105 | 840 | 430,080 | 0 | 0 | 0 | 0 | 1/1 针正确 |
| 128K | 32768–32774 | 105 | 840 | 430,080 | 0 | 1 | 0 | 0 | 1/1 针正确 |
| 256K | 65536–65543 | 105 | 840 | 430,080 | 219,961 | 784 | 0 | 0 | 1/1 针正确 |

1K 生产路径按设计使用 dense/短上下文跳过，不执行 sparse-indexer q8 top-k，因此没有
候选漏选面。隔离 kernel 另行覆盖 q6/q8、native/flat 布局和四档 C4 有效长度；1K
有效候选不足 512 时，所有有效 index 与 `torch.topk` 一致，剩余位置均为 `-1`。

256K 的 index 差异全部发生在第 512 名同分的行。每行候选均合法且唯一，没有选入
低于数学 cutoff 的 score，候选 score 多重集与 `torch.topk` 完全一致；因此属于同分
候选互换，不是数学 top-k 漏选。三档 sparse 路径合计覆盖 315 个 layer/round 样本、
2,520 行和 1,290,240 个候选位置，严格 score 漏选、score 多重集差异、非法/重复
index、padding 错误和 NaN/Inf 均为 0。

证据目录为
`profiles/deepseek_v4/dspark-exact-topk-coverage-20260816/`。该结论限定于上述硬件、
PP3、单请求、DSpark7/q8、top-k 512 和实测轨迹；不扩张为所有 batch、随机轨迹、
DCP 或任意数值输入的绝对保证。

复测 persistent selector 时必须关闭精确生产路由：

```bash
VLLM_DEEPSEEK_V4_DSPARK_EXACT_TOPK=0 \
VLLM_DEEPSEEK_V4_DSPARK_TOPK_AUDIT=1 \
VLLM_DEEPSEEK_V4_DSPARK_TOPK_AUDIT_MIN_LEN=5000 \
VLLM_DEEPSEEK_V4_DSPARK_TOPK_AUDIT_SAMPLES=8 \
DSV4_DSPARK_TOKENS=7 \
benchmarks/deepseek_v4/profile_cmp170hx_pp3.sh --enforce-eager
```

审计会同步 GPU，只用于正确性取证，不能用于性能结论。exact 路径可保持生产开关为
`1`，直接对独立 `torch.topk` reference 取证；要复测 persistent 近似路径才设为 `0`。
旧的
`VLLM_DEEPSEEK_V4_Q6_TOPK_AUDIT*` 环境变量保留为兼容别名。

## 32K Graph A/B/A

服务配置固定为三张 CMP170HX、PP3 `15,16,12`、`max_model_len=1,048,576`、
`max_num_seqs=1`、DSpark-7、CUDA Graph。测试使用同一条精确 32,768-token prompt，
`temperature=0`、`seed=0`、自然 EOS 和最多 64 个输出 token；每组先丢弃一次 warmup，
正式重复 6 次。prompt 包含口令 `orbital-falcon-long-4242`，要求输出先复述口令并继续
长回答。

| 阶段 | selector | pooled acceptance | median token/s | median ms/round | mean ms/round | 语义 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A1 | persistent | 22.59% | 70.77 | 36.97 | 36.35 | 6/6 口令正确 |
| B | exact | 26.32% | 80.23 | 36.08 | 35.63 | 6/6 口令正确 |
| A2 | persistent | 24.56% | 78.91 | 36.81 | 36.56 | 6/6 口令正确 |

相对 A1/A2 的平均值，B 组 median ms/round 改善 2.20%，mean ms/round 改善
2.27%。acceptance 和 token/s 会受 selector 改变后的生成轨迹影响，因此它们只作为
服务结果记录；`ms/round` 是本次 selector 成本判断的主要指标。所有正式请求均输出
64 token 并取回正确口令。

精确路径成功完成 PP3 FULL_AND_PIECEWISE CUDA Graph 捕获。隔离 selector 扫描也显示
q6/q8 的差值仅为微秒级，并随有效序列长度变化；服务 A/B/A 是最终准入依据。

## 显存与回滚

精确 selector 直接写入 caller-owned top-k 输出，不申请 persistent selector 的
1 MiB workspace。没有新增模型常驻、KV cache 或 CUDA Graph 预算。这里不把 workspace
复用的逻辑差值宣称为可用上下文容量收益；准入条件只是显存不增加。

回滚：

```bash
VLLM_DEEPSEEK_V4_DSPARK_EXACT_TOPK=0 \
benchmarks/deepseek_v4/run_cmp170hx_pp3_1m.sh
```

更改 speculative width、batch shape、top-k 或设备架构后不会进入该精确路由；如需
扩展范围，必须重新捕获真实 logits、验证数学集合并执行 Graph A/B/A。
