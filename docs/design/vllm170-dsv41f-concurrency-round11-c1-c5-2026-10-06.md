# DSV41F 并发 1–5 调度与饱和复核（2026-10-06）

本文记录 2026-10-06 的测量现场。下述服务状态均为历史记录；后续已按用户要求停止 DeepSeek。

本轮停止动态 query 长度方向，只复核固定生产配置下的并发时序。测量期间没有修改推理代码，也没有重启服务。服务健康检查为 200，实际进程为 PID 347443，PP rank 0–4 使用物理 GPU 2–6。

## 测量口径

使用 `/mnt/data/apps/vllma100/vllm170/benchmarks/deepseek_v4_1/concurrency_probe.py`，并新增只读探针保存 GPU 100 ms 采样和 `/metrics` 快照。两组结果分开解释：

* 1K 随机 prompt、64 输出、temperature 1.0，用于观察并发扩展的上限，不用于接受率结论。正向和反向两轮结果在 `artifacts/dsv41f-concurrency-review-20261006/concurrency-c1-c5-gpu.json` 和 `concurrency-c5-c1-gpu.json`。
* 32K 固定真实 token prompt、热 prefix、greedy、128 输出，用于保持接受轨迹接近不变。结果在 `concurrency-greedy32k-hot-c1-c5.json`；额外的 C6 只是队列容量控制，在 `concurrency-greedy32k-c6-control.json`。

实际 `/proc/347443/cmdline` 只包含 `--additional-config {"ced_prefill":true,"dsv41_sm80_moe_tilelang":true}`，没有 `--engram-config` 和后续那份 additional-config。这是用户给出的 shell 命令在空行处断开后的有效 argv。因此下面的结论适用于当前 PP/DSpark 调度和 target 计算形状；不能宣称 Engram-on 的端到端结果。

## 并发曲线

32K 热 prefix、greedy 的整波吞吐如下。`decode tok/s` 是 probe 的联合 decode 窗口，`output tok/s` 是从整波开始到最后一个请求结束的端到端口径。

| 并发 | 整波秒 | output tok/s | decode tok/s | TTFT（秒） | drafts | accepted |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 2.686 | 47.66 | 61.39 | 0.61 | 50 | 80 |
| 2 | 4.063 | 63.01 | 71.46 | 0.50 / 0.80 | 100 | 160 |
| 3 | 6.064 | 63.32 | 68.69 | 0.51 / 0.82 / 1.13 | 150 | 240 |
| 4 | 7.441 | 68.80 | 73.62 | 0.53 / 0.83 / 1.17 / 1.49 | 201 | 314 |
| 5 | 9.476 | 67.54 | 71.05 | 0.53 / 0.83 / 1.14 / 1.46 / 1.81 | 251 | 394 |
| 6（控制） | 10.991 | 69.87 | 74.29 | 0.73 / 1.07 / 1.42 / 1.79 / 2.25 / 2.26 | 301 | 468 |

C1 是单请求基线。C2 后吞吐增长变慢，C3 几乎没有超过 C2，C4 有小幅增加，C5 回落 1.8%；C6 只比 C5 高 3.5%。随机 1K prompt 的两轮均值也显示同一形状：C1–C5 为 18.57、22.53、29.04、34.61、37.43 output tok/s，C4→C5 的增益只有约 8%。随机 prompt 接受率低，只作为形状交叉检查。

固定 prompt 的 `accepted / drafts` 为 1.60、1.60、1.60、1.56、1.57、1.56，跨并发基本不变；因此 C1–C6 的吞吐曲线没有被接受长度变化主导。

## 调度时序

源码中的 PP batch queue 是按设计工作的。`EngineCore` 把 `batch_queue_size` 设为 `max_concurrent_batches`，队列未满时用 `execute_model(..., non_block=True)` 连续提交 batch，只有队列满或没有新请求时才等待结果（`vllm/v1/engine/core.py:225-257,700-762`）。PP5 配合 V2 runner 和默认 async scheduling 时，`max_concurrent_batches` 为 `pp_size + 1`，即 6（`vllm/config/vllm.py:571-582`）；DSpark 也在默认 async scheduling 的允许列表中（`vllm/config/vllm.py:1292-1303,1313-1362`）。当前进程日志确认使用 V2 Model Runner。

C5 的实时快照给出直接证据：

* `num_requests_running` 在约 0.8 秒内由 0 变成 5，随后保持 5 到请求开始退出；`num_requests_waiting` 全程为 0。
* 五个请求累计 `request_queue_time_seconds` 只有 **0.000137 s**。C6 也只有 **0.000160 s**。主机请求队列不是瓶颈。
* C5 的 TTFT 依次约 0.53、0.83、1.14、1.46、1.81 s；这对应 PP 填充和请求在不同 step 被纳入 batch 的顺序。请求已经被调度后，完成时间集中在 9.2–9.4 s，未见某个请求长期饿死。
* C5 的 GPU 采样中，2 秒以后五个 rank 的利用率中位区间约为 PP0 100%、PP1 91%、PP2 94%、PP3 91%、PP4 87%；请求结束时 running 数由 5、4、2、1 依次排空。这是流水填充、稳定运行、排空三个阶段，不是 host scheduler 等待。

因此当前没有证据表明“并发 1–5 的请求调度没有合并”或“请求卡在 engine queue”。主要问题是进入稳定流水后，吞吐已经接近 PP0 的周期上限；PP4 等后级仍有可见气泡。

## 是否计算饱和

GPU2（PP0）在 C5 稳态采样持续 100%，但五卡平均功耗约为 107、106、84、102、100 W（稳态两秒后），远低于 250 W power limit。1K 随机控制的 memory-controller utilization 也只有约 9–14%。所以“GPU busy”成立，“算力或 HBM 带宽打满”不成立。更符合当前证据的是短 kernel/权重访问/launch 与 PP 依赖共同形成的延迟受限流水，PP0 先到周期上限，后级被依赖等待。

Round 9 已经在同输入同路由下观测过 PP0 的长 target span（C1 5.707 ms，固定 C4 27.442 ms），Round 3 也测过约 22 ms 的 metadata 回收窗口；后台回收和 fence 方案此前没有稳定端到端收益。本轮只确认该现象在并发 1–5 的稳态仍表现为 PP0 瓶颈，不重复把旧的回收候选当作新热点。

## 下一步实验顺序

1. **一次诊断重启做低扰动逐步账本。** 保持固定 greedy 32K prompt，`C=1,2,3,4,5,6` 各一波。以 `VLLM_DEEPSEEK_V4_STEP_TIMING_EVERY=5` 记录每个 PP rank 的 target、draft、prepare、period；同时在 CPU 侧每 5 step 记录 `step_id、num_reqs、num_tokens、padded_tokens、uniform_token_count、full_graph`。不要打开细粒度 phase profiler，因为它会同步 CUDA Event 并改变 PP 时序。目标是把 PP0 的周期拆成 Graph replay、metadata、PP recv/send、sampling/draft 四项。
2. **先验证 PP 分区，再改 kernel。** 用同一账本比较现有 `8,8,8,8,8` 与只移动一层的候选分区，候选必须由各 rank 的 target+draft 稳态周期决定；每个候选都要通过 C1 固定输出、C4/C5/C6 热 prefix 吞吐和五卡功耗/利用率门槛。没有 rank 周期账本前，不直接指定把层从 PP0 移到某个 rank。
3. **确认 5→6 的收益边界。** C6 仅比 C5 高 3.5%，说明多一个 in-flight slot 的收益已经很小；若逐步账本显示 C5 已有 6 个 queue slot 但只有 5 个活跃请求，应优先做请求到达/PP 填充实验，而不是扩大 `max-num-seqs`。
4. **验收门槛。** 任何调度或分区改动必须在相同 prompt、相同 seed、相同接受长度下，至少 C4/C5 两次交替重复；要求整波 output tok/s 提升 ≥5%，PP0 与最慢 rank 的周期差缩小，且固定 C1 输出 IDs、长度、finish reason、`[DONE]` 全部一致。未同时满足时保留现状。

本轮没有生产代码修改和默认参数修改。诊断数据及原始 client/GPU/metrics 记录见 `artifacts/dsv41f-concurrency-review-20261006/`。
