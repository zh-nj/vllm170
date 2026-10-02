# DSV41F 普通 decode：第一轮基线、PP 实验与热点边界

日期：2026-10-02。基线提交：`4df3a185db133e63934d5739cc18582a215be609`，
另含本轮开始时已有的 19 个未提交路径。原始证据位于
`artifacts/dsv41f-decode-q1-round1-20261002/`。

本轮按最新指令关闭 DSpark，集中测普通 decode。单请求基线为
**50.894 tok/s、19.649 ms/token**。将已有 PP 提前准备扩展到普通 decode 后，
四组正式配对全部回退，中位数 **−1.02%**，因此撤回扩展，没有新增生产优化。
原有 attention 8 warp、M1 MoE 和 metadata 优化继续保留。

## 1. 实际执行路径与测量口径

模型为 `/media/z/data6/models/DeepSeek-V4.1-Flash`，PP5/TP1/DP1，
五张 SM80 CMP170HX，服务名称 `dsv41f`。保留 max-seqs=8、
max-batched-tokens=2048、max-model-len=1048576、KV FP8/2.7 GB、CED 和 prefix cache。
不使用 `--speculative-config`。优先测单请求延迟，补双请求输出回归。

诊断 worker 的实际 inventory 确认五个 rank 均为：

- `speculative_config=None`、`runner.speculator=None`；
- `decode_query_len=1`、`num_speculative_steps=0`；
- FULL decode Graph 的 tokens/requests 为 1/1、2/2、4/4、8/8，
  `uniform_token_count=1`。请求结束后计数 running/waiting 为零。

因此单请求每一步确实只做一个 token 的普通 decode。
FULL 的配置经 backend 约束成为 `FULL_DECODE_ONLY`；prefill 不被误称为 FULL Graph。

使用历史 campaign 留存的同一份 **32768 个 prompt IDs**，每次 greedy 输出 256 个
token，`ignore_eos=True`，固定预算全部完成。指标为客户端 SSE 首个含 token 的
chunk 之后，剩余 delta token IDs / 首尾 chunk 时间；整个首 chunk 被排除。
它是客户端交付速度，不是 GPU step timer，也不含 TTFT。
所有 token（包括非文本 token）都计入。

普通 decode 的 SSE chunk 仍可能包含两个 token，这是 detokenization/交付聚合，
不能据此判断启用了 speculation。初版 harness 错误地要求每个 chunk 宽度均为 1，
已修正为保留 chunk 宽度并通过 engine inventory 和 speculation counters 核实路径。
失败记录保存在 `ab-initial-sse-width-failure.log`。

此次配置与 9 月 28 日的 q1 扫描不同，不将 35.55 tok/s、26.38 ms 或回归截距
作为本轮性能基线；此前 DSpark q6 的吞吐也不能用作普通 decode 的优化收益。

## 2. 实验：推广 PP 提前准备到无 speculation

原 `_dsv41_sm80_target_prepare_ahead()` 仅适用于既定 DSV41F/SM80/V2/PP5/TP1、
固定 DSpark5 条件。候选只允许 `speculative_config is None`，继续沿用原 lazy
PP receive callback；没有改模型 kernel、通信 payload 或采样依赖。

只加载一次诊断实例，arm 通过 RPC 改 `_defer_pp_recv`，每次切换前确认无运行/等待
请求。两侧各 warmup 一次，然后按 AB、BA、AB、BA 做四组、八次正式请求。
性能请求关闭 step timing 和 phase profiling；保留 GPU clocks/power 及 memory PSI。

| 同实例、32K/256、无 DSpark | 基线：原 receive | 候选：提前准备 | 差异 |
| --- | ---: | ---: | ---: |
| 客户端 decode 中位数 | 50.893697 tok/s | 50.376908 tok/s | −1.0154% |
| 每输出 token 中位数 | 19.648854 ms | 19.850470 ms | +0.201616 ms |

四组配对吞吐变化依次为 **−0.6201%、−1.4626%、−0.9191%、−1.0218%**。
方向一致，拒绝推广。全部 warmup、正式请求与短诊断的输出 IDs 都与基线逐位一致；
双请求 128-token 回归也在两侧逐位一致。speculation counter 没有增长。

候选源代码和 patch 已归档至 `rejected/`。仅还原本轮修改的
`vllm/v1/worker/gpu_worker.py` 与 `tests/v1/worker/test_pp_utils.py`，
两文件与本轮前快照及 HEAD 一致。没有撤回此前已验证的 DSpark5 优化。
生产无 DSpark 配置显式设置 `dsv41_sm80_target_prepare_ahead=false`。

## 3. 为什么没有收益：等待移动，前向边界不变

正式性能结束后，两侧各做一个 32K/128 的轻量 step-timing 请求。
只取 `FULL && tokens=requests=1`，每 rank 去掉开头 8 个和末尾 2 个样本，
五个 rank 各 117 个。按普通 decode 顺序对齐，先逐步求五 rank 的前向区间之和，
再求中位数；不是五个 rank 各自中位数直接求和。

| 插桩诊断 | 基线 | 提前准备 |
| --- | ---: | ---: |
| 对齐后五 rank 前向区间中位数 | 18.206720 ms | 18.237440 ms |
| 同请求客户端 ms/token | 20.083458 | 19.776278 |

诊断客户端数据方向与正式配对不同，不用它决定是否接受优化。
**基线前向区间占同次插桩客户端周期约 90.7%**，候选没有降低这项主要成本。

前向 event 从 graph staging 之前开始，在 `sample_tokens()` 入口结束；非末级 rank
还包含 PP send 启动及 worker/executor 交接空隙。18.21 ms 不能全部称为 kernel
忙碌时间，也不能与旧扫描的 target span 相减来构造收益。

| 基线、每 rank 中位数 ms | PP0 | PP1 | PP2 | PP3 | PP4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 前向 event 区间 | 3.440 | 3.602 | 3.780 | 3.522 | 3.869 |
| metadata 宿主区间 | 1.777 | 1.713 | 3.024 | 2.869 | 2.803 |
| metadata CUDA event 区间 | 0.157 | 0.164 | 0.065 | 2.865 | 2.799 |

开启提前准备后，PP1–4 宿主 receive 等待中位数从
4.13/8.64/11.04/9.99 ms 降为 1.28/2.84/4.39/5.82 ms，
但宿主 model-inputs 区间从 0.16/0.24/0.28/0.26 ms 增为
1.45/2.98/4.54/5.96 ms。等待上游计算仍存在，只是发生位置变了。
metadata 宿主时间、GPU event 时间不同也说明重叠关系发生变化。
原 receive 路径已能遮住相当一部分准备成本，不能照搬 q6 的获益结论。

`prepare` 包含 metadata/inputs，receive/inputs 还可能包含上游等待；这些区间
跨 rank 重叠。禁止将它们相加成新的端到端账本，也不把 event 区间全部当作
metadata kernel 的净 GPU 工作时间。原始数据及分析见 `ab.json`、`analysis.json`。

## 4. 短 phase 诊断确认边界，尚未闭合图内算子占比

候选关闭状态下额外做一次 32K/32 请求，仅打开 decode phase profile。
输出 IDs 与基线前 32 个逐位一致。五 rank 各有 31 个 Graph replay，phase step IDs
与顺序一致；去掉开始 5 个和末尾 1 个，逐步求和后中位数为 **18.895872 ms**。
Graph staging 的 rank-local event 中位数仅为 0.017–0.046 ms。
这支持继续追查模型 Graph 内部的主要成本。

逐阶段 event 同步和 JSONL 写入把客户端速度降到 **6.50 tok/s**。
metadata 也被强烈扰动；这些 trace 不参与收益判定，也不将 phase replay 与轻量
step-timing 的两个不同运行相减。

当前 profiler 在编译/Graph capture 时不插入图内阶段事件。启动后打开它，只能
测整个 replay、staging、图外 metadata、logits 和采样，无法观测 FULL Graph 内各
MoE/attention/projection 的当前占比。phase sampling 也有扰动，不作净采样成本结论。
`phase-analysis.json`、`phase-graph-aligned.json` 保留统计和这个限制。

下一次需要的是一次有边界的、诊断专用的 **实际 q1 Graph 内分段计时**：
分别观察不重叠的 projection、sparse attention/indexer、MoE、mHC 和其余部分，
同时记录整个 Graph 对照。图内 instrumentation 的影响必须通过无插桩 replay
与输出 IDs 验证。不能拿旧 isolated q1 数字直接分摊当前 18.2 ms。

## 5. 已有优化的普通 decode 可达性与下一处具体线索

| 已有机制 | 无 DSpark 的实际条件 | 本轮结论 |
| --- | --- | --- |
| sparse attention partial 8 warp | q1、H64、splits≤16、top512 或短 SWA | 已被现有条件覆盖，保留；本轮没有新做 4/8 warp 归因 |
| M1 TileLang MoE、W13 packed-word | 单请求 m=1、DSV41F E384/top6、H5120/I2304、SM80、显式 opt-in | 原 q6 主路径不适用的 M1 分支现在可达；已经在本轮基线中 |
| activation/quant 融合 | 原有精确 gate 和 MoE 分支 | 保留；不能把旧 TileLang 版本或旧 full-apply 速度转成当前收益 |
| fused/direct-ragged SWA metadata | causal 普通 decode、小查询/token-map 门及 graph-stable workspace | 已可达；保留；双请求与 prefill 的 fallback 不变 |
| PP 提前准备 | 原固定 DSpark5 门 | 无 DSpark 推广失败，撤回 |
| FP8 dense linear | 原 Marlin 路径 | 本轮未重启已放弃的 TileLang 路线 |

发现一处源码可验证的 q1 配置不一致：`rocm_aiter_mla_sparse.py` 在选择 splits 时
先用 **BK32** 估算，之后 q1 partial 才切换为 **BK16**。从真实源码 AST 提取
纯选择函数，以本卡 74 SM、H64/4 head-blocks 计算：

| q1 形状 | 当前 splits | 按实际 BK16 选择 | 实际每 split 的迭代数 |
| --- | ---: | ---: | --- |
| SWA128，无 extra | 4 | 8 | 2 → 1 |
| SWA133，无 extra | 5 | 9 | 2 → 1 |
| SWA128 + top512 | 16 | 16 | 不变 |

q2/4/6/8 在上述形状的选择不变。证据为 `split-heuristic-audit.json`。
它是具体候选线索，**不是性能缺陷已证实**：更多 splits 增加 reduce 工作，
尤其 top512 主路径没有变化，需先做小矩阵 partial+reduce 测量，评估整模型上限。
只有 kernel 和实际 serving 均获益，才改变默认并推广到满足相同条件的路径。
本轮没有为此再次 reload 或跑大范围 sweep。

## 6. 生产恢复、验证和复现

正式服务仍为 `dsv41f-manual.service`、公开 `0.0.0.0:26661`，关闭 DSpark。
生产 argv 无 speculative config、worker extension；无 dev mode 或诊断 PYTHONPATH，
phase/step timing 均关闭。诊断 unit 已停止，26662 关闭。
模型加载、五 rank CUDA ownership、cgroup、监听 PID、实际模型路径和运行源码 SHA256
均由 `production-validation.json` 核验，不能只用 health 代替服务身份验证。

生产验证取一次 warmup 和三次 32K/256 普通 decode，再补双请求 32K/128。
输出 IDs 要与本轮对应基线一致，完整预算、usage、speculation counters 一并检查。
重启后的验证速度只描述当前可用服务，不是配对因果 A/B。
正式服务三次验证速度为 **51.045 / 50.845 / 51.215 tok/s**，中位数
**51.044627 tok/s、19.590701 ms/token**；五个 rank 的进程均属于
MainPID **3223630** 的 service cgroup，模型身份、26661 listener 与实际 GPU UUID
均匹配。单请求及双请求的 token IDs 与对应诊断基线逐位相同，speculation
counter 无增长。服务健康检查为 200，诊断 RPC 为 404。

首次 warmup 期间记录到 Engram lookup、mHC prenorm/combine/post-update 的 JIT
警告，发生在正式三次请求之前，不据此宣称所有 cold/prefill shape 都已预热。
初版生产核验脚本错误地要求显式 `VLLM_USE_V2_MODEL_RUNNER=1`；实际通过自动选择
启用 V2，已改为同时核对真实启动日志，未因此变更或重启服务。

候选 CPU suite **35 passed**，相关 hooks 通过；撤回后原 suite **31 passed**，
本文 pre-commit hooks 通过。
本轮开始时的 19 个已有 dirty 路径 SHA256 全部保持一致，没有顺带提交或撤回它们。

```bash
# 对保存的 A/B 重新统计，无 GPU 工作：
.venv/bin/python artifacts/dsv41f-decode-q1-round1-20261002/analyze_ab.py

# 验证原 PP 门和回退，不需要重新加载模型：
.venv/bin/python -m pytest tests/v1/worker/test_pp_utils.py -q
```

启动命令和逐文件 SHA256 在 `launch.json` / `production-launch.json`。
诊断扩展、性能 harness、失败候选、单次阶段 trace、完整 request/token IDs 与
已恢复生产的验证都保留在同一目录，避免把不同配置的数字混用。
