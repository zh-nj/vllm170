# DSV4.1-Flash：target 验证的 q 扫描与代价模型（2026-09-28）

目标：把 **DSpark=5 的 6-token target 验证**与 **关闭 DSpark 的 1-token target 验证**放进同一套
口径，比较模型计算量、attention/index/metadata、PP 通信与轮间等待，建立理论模型并指出实际差距。
本轮**不改推理代码**（只补了一处测量归属，见 §5）。

> **本文修正了同日早先版本的结论。** 早先版本用 2026-09-18 ledger 的 24.284 ms（另一套配置、
> DSpark 关闭）给模型定标，声称"q=6 的增量全部由专家重读解释、路由期望 34.6 个不同专家"。
> 本轮 q 扫描证明该结论错误：实测斜率只有独立路由预测的一半。正确的结论见下。

复现：`benchmarks/deepseek_v4_1/verify_cost_model.py`；
`profiles/deepseek_v4_1/20260928_verify_sweep/`（6 次重启的原始日志与 request）；
断言见 `tests/benchmarks/test_deepseek_v41_verify_cost_model.py`。

---

## 0. 结论

1. **DSpark-on 的 target 时间对 q 近乎完美线性**：`target_sum = 33.87 + 2.42·q` ms
   （q=2..6，最大残差 0.92 ms）。**每多验证一个 token 只要 2.42 ms。**
2. **进入验证路径本身有 9.92 ms 固定代价**：把同一条直线外推到 q=1 得 36.30 ms，
   而真正关闭 DSpark 的 q=1 只花 26.38 ms。这个差在没有任何 draft token 时就已存在。
3. 机制的代码依据：DSpark 打开后 `decode_query_len = 1 + num_speculative_tokens`，
   FULL graph 以 `uniform_token_count=decode_query_len` 捕获
   （`vllm/v1/worker/gpu/cudagraph_utils.py:225-280`，`varlen_decode=False`），
   于是**每一步都走多 query 的 attention 路径**，不再是单 query 的 decode 路径。
   同一处签名也解释了 metadata：PP0–PP3 的 `attn_meta` 从 q=1 的 0.13–0.29 ms
   **跳到** q=2 的 3.09–6.26 ms 之后就不再随 q 变——是模式切换，不是 q 比例。
4. **独立路由假设被证伪**：若 6 个 token 各自独立取 top-6，斜率应为 4.85 ms/token；
   实测 2.42，对应**每个新增 token 只带进约 2.4 个新专家**，即被验证的连续 token 之间
   专家高度重合。
5. **权重字节只能解释边际，解释不了量级**：按实测的 2.4 专家/token 反推，q=6 只剩
   6.69 GB 非专家权重对应 30.0 ms，等于 223 GB/s——低于本卡任何实测带宽（400–1207 GB/s）。
   所以 step 的绝对高度里有很大一块是延迟/启动/依赖，不是带宽。
6. **DSpark=1 是净亏**：q=2 每输出 token 38.59 ms，比不开 DSpark 的 28.12 ms 更慢
   （即使接受率 128/128=100%）。q=3 起才转正（23.77 ms），q=6 为 13.53 ms。

---

## 1. 本轮实验：q 扫描

同一配置（PP5、chunk 2048、CED、FULL graph、`mxfp4_int8_sm80`、32K 固定 prompt/256 输出、
`VLLM_DEEPSEEK_V4_STEP_TIMING_EVERY=10` 低扰动计时），**只改 `PORT_DSPARK_TOKENS`**，
每个 q 重启一次；q=1 用 `PORT_DSPARK_TOKENS=0`（完全不传 speculative config）。
每个 q 采 5 个 rank 的 `target` CUDA event span 中位数并求和——本项目既定的"PP 串行和"口径。

## 2. 实测曲线

| q | target 和 ms | period ms | token/轮 | decode tok/s | 每输出 token ms | 接受 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1（关闭） | 26.38 | 28.12 | 1.000 | 35.55 | 28.12 | — |
| 2 | 38.34 | 76.88 | 1.992 | 25.40 | 38.59 | 128/128 |
| 3 | 41.21 | 71.31 | 3.000 | 42.37 | 23.77 | 170/170 |
| 4 | 43.77 | 72.42 | 3.923 | 54.78 | 18.46 | 193/195 |
| 5 | 46.92 | 76.70 | 4.904 | 65.12 | 15.64 | 203/208 |
| 6 | 47.61 | 78.41 | 5.795 | 74.81 | 13.53 | 211/220 |

- **斜率 2.42 ms/token、截距 33.87 ms、最大残差 0.92 ms**（q≥2）。
- **固定代价 9.92 ms** = 直线在 q=1 的值 36.30 − 实测 DSpark-off 26.38。
- 接受率在小 q 时接近完美（q=2 为 100%，q=3 为 100%），随 q 下降（q=6 为 95.9%）。
- period 在 q≥2 后基本平坦（71–78 ms）：**多验证 token 几乎不增加整轮时间**，
  因为 target 只占一部分，且 draft 链长度不变。

## 3. 字节模型：成立的部分与被证伪的部分

字节账本仍然有效（来自 checkpoint header，decode ledger §4）：每步固定读 6.69 GB
（attention 投影 4.96 + shared expert 1.42 + router/mHC 0.31），每个不同专家 0.72 GB
（40 层合计），MoE 带宽随 slot 数从 575 GB/s（6 slot）升到约 790 GB/s（36 slot）。

| | 独立路由 | 实测隐含 |
| --- | ---: | ---: |
| 每个新增 token 带来的新专家 | 6.0 | **≈2.4** |
| 预测斜率 | 4.85 ms/token | **实测 2.42 ms/token** |
| q=6 不同专家/层 | 34.6 | ≈17.9 |
| q=6 专家字节 | 24.93 GB | ≈12.9 GB |
| q=6 留给 6.69 GB 非专家权重的带宽 | 420 GB/s（合理） | **223 GB/s（不可能）** |

- **专家重读是真机制**：它决定边际成本，且量级（约 2.4 专家/token）只有独立路由的 40%，
  说明连续 draft token 的路由高度相关。
- **但它不是全部**：把实测斜率代入后，非专家部分需要的带宽低于本卡任何实测值。
  换言之，**权重字节决定"每多一个 token 贵多少"，不决定"一步本身多贵"**。
  后者主要是多 query 路径的固定延迟（§0 第 2、3 条）。

## 4. 四条轴的同口径对照

| 轴 | q=1（DSpark 关闭） | q=6（DSpark=5） |
| --- | --- | --- |
| **模型计算量** | rank 和 26.38 ms | 47.61 ms（1.81×）；其中 9.92 ms 是进入验证路径的固定代价，其余是 5 个 token × 2.42 ms |
| **attention / index / metadata** | `attn_meta` 0.13/0.24/0.22/0.29/**4.46** ms | 3.12/6.17/5.29/5.04/**4.94** ms |
| **PP 通信** | `pp_recv_wait`+`copy` ≈0.27 ms/rank | ≈0.32–0.39 ms/rank（payload 6×，时间几乎不变） |
| **轮间等待** | 无（一步即一轮，28.12 ms） | sampling 3.6 + draft 链约 26 ms；整轮 78.41 ms |

- **metadata 的真相**：PP0–PP3 在 q=1→q=2 就从 0.2 ms 跳到 5 ms，之后不再随 q 增长，
  是"单 query → 多 query"的路径切换；PP4 在 q=1 就已经是 4.46 ms（该 rank 特有的固定开销）。
  在 q=6 这些 metadata 大多藏在 8.3–9.5 ms 的 upstream 前向之后，只有 PP0 的 3.12 ms 暴露。
- **PP 通信不是差别所在**：payload 涨 6 倍而交接时间不变，固定成本主导，占整轮约 2%。
  真正的 PP 代价是五段前向严格串行。
- **轮间等待**：q≥2 每轮多出 sampling + draft 链约 29 ms，与 q 无关；它摊薄在
  token/轮 上，这正是 q 越大越划算的原因（q=2 摊不薄，所以净亏）。

## 5. 5 个未决项的逐条结论

| # | 原未决项 | 结论 |
| ---: | --- | --- |
| 1 | D=34.6 是反推、无直接观测 | **已解决（结论反转）**。q 扫描证明独立路由假设错误：实测斜率 2.42 ms/token ⇒ 约 2.4 新专家/token。直接捕获仍不可行：`--enable-return-routed-experts` 在 `PP>1` 被显式拒绝（`vllm/config/vllm.py:1140`），本布局无法开启；q 扫描是不改代码下能给出的最强检验。 |
| 2 | q=1 的 metadata 没有同口径数据 | **已解决**。同配置实测 `attn_meta` = 0.13/0.24/0.22/0.29/4.46 ms，与 q=6 的 3.12/6.17/5.29/5.04/4.94 直接可比，见 §4。 |
| 3 | q=6 分解来自插桩运行 | **已解决**。整条曲线来自同一轮 campaign、同一插桩；q=6 period 78.41 ms 与同配置无插桩生产值 71.386 ms 相差 **+9.8%**，即插桩只抬高量级、不改变形状。跨配置折算不再需要。 |
| 4 | PP4 多出的 ~1.9 ms 归因未验证 | **已解决（机制被排除）**。同配置每 rank 增长 1.66/1.62/1.71/1.91/**2.10**，PP4 仍最快，超出 PP0–PP2 平均约 2.5 ms。但**辅助隐状态拷贝不是原因**：3 层 × 6 行 × 5120 × 2 B = 184 KB/step，约 0.2 µs。PP4 的可测特征是它在 q=1 就有 4.46 ms 的 `attn_meta`（PP0–PP3 只有 0.2），属该 rank 固定开销，与 q 无关。 |
| 5 | q=1 锚点跨配置 | **已解决**。同配置 q=1 = 26.38 ms（target 和）/ 28.12 ms（period）/ 35.55 tok/s；ledger 的 24.284 ms 是另一套配置。跨配置差 **15%**，确认不能混用。 |

## 6. 对 DSpark 的工程含义

- **q=1（DSpark=1）不要开**：38.59 ms/token，比不开 DSpark 慢 37%，即使 100% 接受。
- **收支平衡在 q≈2.5–3**；q=6 的 13.53 ms/token 是相对 plain decode 的 2.08×。
- **两条可攻击的方向**（本轮不实施）：
  1. **9.92 ms 的进入代价**：它来自"每一步都走多 query 路径"。若能在只验证 1–2 个 token 时
     回退到单 query 路径（或在 graph 捕获上分离两种 shape），q=2/q=3 直接受益。
  2. **draft 链约 26 ms/轮**：占 q=6 整轮的 33%，且与 q 无关；缩短它对高 q 同样有效。
- 不建议再按"独立路由 ⇒ 专家重读 6×"的方向做优化假设：实测只有约 2.4 新专家/token。
- **继续往上加 draft token 需要同时设 draft query block**：只调 PORT_DSPARK_TOKENS 到 6/7
  会启动崩溃，见 §7。

## 7. dspark>5 的边界（尝试后取消）

按同样的扫描把 q 推到 7、8 时，**默认配置无法启动**，两次尝试均在同一个点崩溃：

| 配置 | draft query block | 结果 |
| --- | ---: | --- |
| dspark=6（q=7） | 6（默认 = num_speculative_tokens） | **2/2 启动失败** |
| dspark=7（q=8） | 7（同上） | **2/2 启动失败** |
| dspark=6（q=7） | 10 | **一次启动成功** |

失败签名完全一致：日志打完
`Distributed DSpark is replaying captured CUDA graphs for the rank-local draft step`
（`speculator.py:518`，draft 图重放预热）后立即
`CUDA error: an illegal memory access was encountered`，NCCL watchdog 终止进程。

这不是配置被拒：`vllm/config/speculative.py:1535-1541` 对 `deepseek_v41` 明确豁免了
MTP 的整除检查，`speculator.py:193-201` 还专门为"query block 超过 checkpoint block size"
打了一条 info 日志。所以能进到 capture/replay，是 draft backbone 在 query block
**不是训练值 5 的整数倍**时越界。

已确认的修复方向：`dspark_num_query_tokens`（`speculative.py:484`，"draft backbone runs the
full trained query block; only its prefix becomes a proposal"）把 query block 固定回 5 的
倍数即可。本轮未继续跑 q=7/q=8 的请求测量，因此**曲线仍只到 q=6**。

证据：`profiles/deepseek_v4_1/20260928_verify_sweep/q7/`（两次失败的完整日志）、
`q8/`（同样失败）、`q7q10/`（query=10 启动成功的日志）。

复现该边界要在 `--speculative-config` 里显式给出 `dspark_num_query_tokens`；
本机 `run_port_pp5.sh` 里有一个可选开关做这件事，但该文件同时带着本轮之外的
未提交改动，未随本提交入库。

## 8. 复现

```bash
# 曲线与模型（离线，秒级）
.venv/bin/python -m benchmarks.deepseek_v4_1.verify_cost_model --out-dir /tmp/vcm
.venv/bin/python -m pytest -q tests/benchmarks/test_deepseek_v41_verify_cost_model.py

# 重跑整条曲线（每次重启约 8 分钟，共 6 次）
QS="1 2 3 4 5 6" bash profiles/deepseek_v4_1/20260928_verify_sweep/sweep.sh
.venv/bin/python profiles/deepseek_v4_1/20260928_verify_sweep/analyze.py
```

原始证据：`profiles/deepseek_v4_1/20260928_verify_sweep/q{1..6}/`（server.log、step_timing、
request）、`q1_fourrank_logged/`（补 PP4 归属前的旧 q=1 证据）、
`docs/design/vllm170-dsv41f-decode-ledger-2026-09-18.md`（字节账本与带宽）。

## 9. 本轮唯一的模型代码改动（测量，非推理）

`vllm/v1/worker/gpu/model_runner.py`：末级 PP rank 在**没有 speculator** 时也要闭合
step-timing 的事件对。此前关闭 DSpark 时 PP4 一行都不打印（`drafter_end()` 是解析与写日志的
触发点），q=1 的 target 和因此缺一个 rank（旧证据 `q1_fourrank_logged/`：4 rank 合计 20.83 ms）。
改动只在 step timing 打开时生效，不改变推理行为。
