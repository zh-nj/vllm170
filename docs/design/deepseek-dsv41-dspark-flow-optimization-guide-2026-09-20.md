# DeepSeek-V4.1-Flash DSpark 流程提速指引

日期：2026-09-20

目标读者：后续接手的 DeepSeek；本轮由 Codex 实际推进实现、测试并记录进度。

## 第五轮进度：PP1–PP4 DSpark 分阶段计量完成

本轮把“PP 模式 DSpark”固定为以下调用边界：target 仍在 PP0–PP4 五个 rank
上执行；DSpark 只占用 PP1、PP2、PP3 三个中间 draft stage 和 PP4 的 draft
head。一次分布式 draft 的 relay 环为
`PP4(head) → PP1(stage 0) → PP2(stage 1) → PP3(stage 2) → PP4(head)`。
PP0 只执行 target，不把 PP0 的 target 时间或等待时间归入 DSpark。

第五轮固定 PP5/TP1、GPU `2,3,4,5,6`、32K prompt、q5、greedy、无 prefix
cache、`max_model_len=262144`、`max_num_batched_tokens=2048`，并保持
`dsv41_dspark_compact_relay=true`。采样请求只用于打开 profiling 窗口；phase
profiler 会在记录点使用 CUDA event，并在 flush 时同步设备，所以 profiling
吞吐不能与第四轮的低开销 step timing 直接比较。原始 JSONL 和 gate 文件保留在：

```text
artifacts/dspark_flow_20260920/round5_profile/
artifacts/dspark_flow_20260920/round5_full/
artifacts/dspark_flow_20260920/round5_meta/
```

### 外围阶段的结果

先用 eager 运行确认调用边界，再用 CUDA Graph FULL 只保留外围阶段。eager 中
PP1、PP2、PP3 的 `draft.stage.recv_wait` 稳定中位数约为 237.2、173.3、
110.7 ms；这些数字主要是流水线尚未生产上游 package 时的 pipeline bubble，
不是单跳通信耗时，因此不作为优化目标。

FULL 窗口中，各 rank 的稳定中位数如下（单位 ms；仅用于相位归因，包含
profiling 事件开销）：

| PP rank | target graph replay | target metadata 外围 | DSpark 阶段 | DSpark 阶段值 |
| --- | ---: | ---: | --- | ---: |
| PP0 | 8.182 | 4.058 | 无 DSpark | — |
| PP1 | 10.598 | 6.423 | `prepare` / `send` | 3.765 / 2.430 |
| PP2 | 10.838 | 5.821 | `prepare` / `send` | 3.791 / 2.548 |
| PP3 | 11.278 | 5.504 | `prepare` / `send` | 3.680 / 2.068 |
| PP4 | 11.286 | 5.561 | `head.recv_wait` / `markov` / `head.send` | 24.933 / 4.402 / 2.077 |

`draft.stage.recv_wait` 仍然覆盖“收到前一跳 metadata 和 tensor”之前的整段
等待，不能和 `send` 相加当作通信总耗时。真正位于 DSpark critical path 的
稳定信号是 PP4 head 回收 PP3 结果的等待，以及 PP1–PP3 每跳约 2.1–2.5 ms
的同步发送；这两个边界需要继续拆成 CPU metadata、device isend 和 handle
wait，才足以决定是否值得做 pack 或 overlap。

### target metadata 的子阶段结果

`target.attn_metadata` 在每个 PP rank 都构建四个 SWA KV group 和一个压缩 MLA
group。四个 SWA builder 的 `decode_indices` 中位数约 0.43–0.45 ms，`validity`
约 0.18–0.20 ms，`token_map` 约 0.10–0.11 ms，`result` 约 0.10 ms，
`tail_clear` 约 0.11 ms，其余 `split`、`prefill_common` 和
`tile_scheduler` 均为亚 0.05 ms 量级。单个 SWA 子 kernel 没有超过 0.5 ms；
把 nested stage 全部打开后，外层 `target.attn_metadata` 被放大到约 8.5 ms
(PP0) 和 14–15 ms (PP1–PP4)，说明这些绝对值含有额外 event/flush 开销。

因此当前没有证据支持重写某一个 SWA kernel。可验证的候选是四次 builder 调用
之间的 host/object orchestration 和 descriptor 复用，但不能直接把四组 block
table、slot mapping 或 causal 标志合并：每个 group 的物理 KV 地址不同。后续
应先用只包 builder 外层的轻量窗口测量 CPU 构造比例，再决定是否引入 common
descriptor；本轮没有合入未经证明的 metadata cache。

### relay 清单和下一处热点

compact relay 的四跳 tensor message 数是
`PP4→PP1: 8`、`PP1→PP2: 13`、`PP2→PP3: 13`、`PP3→PP4: 10`，合计 44；
关闭 R1 时为 8、17、17、10，合计 52。B=1 synthetic wire fixture 的总 tensor
bytes 仅从 954752 降到 954512，说明 R1 主要减少小消息调度，不能解释 PP4
head 仍有约 25 ms 的回等。

下一步只做 transport boundary 的可归因计量：在 PP1–PP4 的实际 package 上记录
字段数、按 dtype 的 bytes、CPU metadata 序列化、device isend 发起和同步 wait，
并与 PP4 `head.recv_wait`、整步 period 一起保存。`isend_tensor_dict` 虽会自持
source tensor 和 handles，但发送方下一步可能重用同一 staging 地址；在没有双缓冲
和 step-id 生命周期证明前，不把同步 `send_tensor_dict` 改成异步。若拆分后
device wait 只占很小比例，则转向 target metadata host orchestration；若小消息
和 CPU metadata 占主导，再评估 DSpark 专用 flat relay package，继续不改共享
PP 通信协议。

为下一次独占硬件窗口，`speculator.py` 的两个 send phase 已加入 opt-in wire
inventory：profile JSONL 的 `extra` 会记录 `tensor_keys`、`tensor_count`、
`tensor_bytes`、按 dtype 的字节数以及非 tensor header 字段。这个遍历只在
`VLLM_DEEPSEEK_V4_PHASE_PROFILE=1` 且存在 control file 时执行；普通 serving、
CUDA Graph capture 和 R1 默认关闭路径均不改变。CPU 契约新增的 inventory 测试
与原 relay 测试一起通过，避免下一轮再次依靠 synthetic fixture 推断实际 package。

本轮服务在 profiling 请求完成后已停止，26663 端口和 GPU 2–6 均无残留进程。
CED 移植已完成且不属于本轮变量；不再做 CED 兼容矩阵或联合路径归因。

## 第六轮进度：读取 PP1–PP4 的真实 relay package

在加入 wire inventory 后重新启动同一 PP5/TP1 服务，只执行两次 32K/64 请求，
并在请求完成后停止。`round6_wire/manifest.json` 保存了配置、原始 profile 和
选择规则；这里的纯 decode 样本取 `step_kind=decode`、`token_count=6`，丢弃前两
个匹配 step 以避开首轮 JIT。真实 package 的稳定形态如下：

| relay hop | tensor 数 | tensor bytes | `send` p50 (ms) | `prepare` p50 (ms) | `recv_wait` p50 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| PP4 → PP1 | 8 | 61,648 | 2.080 | — | — |
| PP1 → PP2 | 13 | 318,128 | 3.145 | 3.231 | 72.026 |
| PP2 → PP3 | 13 | 318,128 | 2.629 | 3.161 | 58.290 |
| PP3 → PP4 | 10 | 256,608 | 2.291 | 3.125 | 44.057 |

每一跳还有相同的五个非 tensor header：`scheduler_step_id`、`dummy_run`、
`req_ids`、`num_reqs`、`skip_attn`。纯 decode 每轮合计 44 个 tensor message、
954,512 bytes；PP1→PP2 和 PP2→PP3 的 13-tensor 包已确认 R1 删除了四个
sampling/query tensor，PP3→PP4 的 10-tensor 包仍保留 head 所需的四项结果。
这与 CPU/NCCL fixture 的消息数量一致，但 bytes 现在来自真实模型的 CUDA
package，不再只是形状推断。

PP4 的 `draft.head.recv_wait` p50 为 24.501 ms。PP1–PP3 的 `recv_wait` 为
72.026/58.290/44.057 ms，随流水线位置递减，继续证明它们主要是上游 stage
和 target 调度造成的 bubble，不能直接当作 NCCL 单跳耗时。三个 stage 的
`prepare` 都约 3.1–3.2 ms，而 send 约 2.3–3.1 ms；因此下一处可归因热点是
**PP4 head 的 relay 回等和发送方同步 wait 的组成**，不是把 `recv_wait` 全部
拿去做通信压缩。

当前尚未合入 flat package 或异步发送。`parallel_state.py` 的 `isend_tensor_dict`
会自持 source tensor 和 handles，但现有 staging 地址在下一步可能被 graph
重用；没有双缓冲、完成 event 和 step-id 证明前，异步改动会有 silent overwrite
风险。下一轮先在相同 inventory 窗口中增加 CPU metadata pickle、device isend
发起和 handle wait 的独立计时；如果大头落在多个小 tensor 的 launch/metadata，
再设计 DSpark 专用 flat buffer，并以 PP4 `head.recv_wait` 和整步 period 做
A/B/A 验收。

## 第七轮进度：flat-state 仅在 transport-only 基准中成立

基于第六轮的真实字段和 shape，`dspark_relay_bench.py` 增加了
`--candidate flat-state`：把 `hidden_states/residual` 合成一个 bf16 tensor，
把 `post_mix/res_mix/pre_mix` 合成一个 fp32 tensor。四跳 tensor message 从
44 减到 35，总 bytes 保持 954,512，不改变 compact relay 已删除的字段。

四进程真实 NCCL 环固定 GPU `2,3,4,5`、`NCCL_P2P_DISABLE=1`、1 MiB buffer、
8 channels，每格 50 轮、预热 10 轮、3 次重复。结果如下：

| batch | baseline 合并中位数 (ms) | flat-state 中位数 (ms) | 变化 |
| --- | ---: | ---: | ---: |
| B=1 | 14.354 | 12.370 | -13.82% |
| B=8 | 13.849 | 11.796 | -14.82% |

原始结果保存在 `round7_flat_transport/relay-nccl.json`。这个基准只包含 transport
package 构造和四跳发送；它没有包含 FULL CUDA Graph 为固定地址做的 staging，
因此只能说明“少发 9 个 tensor”值得进入一次整模型 B 组，不能证明端到端收益。

## 第八轮进度：整模型 flat relay 无正收益，运行时候选已撤销

运行时候选为两个持久化 relay buffer 逐字段执行 `copy_`，接收端再恢复 view；
只在纯 decode 生效，prefill/dummy 保持原 package。固定配置与第六轮相同，输出
长度改为 256，每个实例跑 3 次。A1 与 B 的结果是：

| 组 | 三次 post-first-chunk tok/s | 中位数 | 相对 A1 |
| --- | --- | ---: | ---: |
| A1：flat off | 73.253 / 75.056 / 74.503 | 74.503 | — |
| B：flat on | 67.090 / 68.804 / 66.675 | 67.090 | -9.95% |

A1 与 B 的三次 token IDs 完全一致；engine counters 也完全一致：132 个 draft
round、660 个 draft tokens、633 个 accepted tokens。负收益不是接受率或输出
变化导致。A1 的稳态 period 主要落在约 76.6–80.2 ms，B 主要落在约
81.9–88.7 ms。transport-only 省下的 launch 数被发送前新增的逐字段 GPU copy
抵消，且 FULL graph 路径原本已经有 staging copy。

收到新的停止规则前 A2 已经完成，中位数为 64.179 tok/s，实例间出现明显系统
漂移，因此不把 A2 用作挽救或否定 B 的依据。此后候选在 B 已明确无正收益时
立即停止，不再启动 A2；只有 B 有正收益或结论落在噪声带内时才做 A2 回归。

运行时 `dsv41_dspark_flat_relay`、pack/unpack 和额外常驻 buffer 已从源码撤销，
普通 serving 不保留该死路径。transport-only benchmark 与原始 A1/B/A2 日志
继续保存在 `round7_flat_transport/` 和 `round8_flat_relay/`，作为“不能只看
message count，必须计入 staging/device copy”的反例。服务已停止，26663 端口
和 GPU 2–6 已释放。

## 第九轮进度：同步发送的真实组成和 staging 微基准

第六轮把 `send` 拆成了一个总阶段；本轮在相同 PP5/TP1、GPU 2–6、32K
prompt、q5、greedy 配置下，进一步记录 `launch`、metadata handle wait 和
tensor handle wait。采样仍只用于打开 profiling 窗口，不能把它的吞吐量当作
生产吞吐量。稳定纯 decode 样本（丢弃前两步，n=23）的中位数如下；GPU 列是
CUDA event，CPU 列是 host wall time，单位 ms：

| relay 边界 | send GPU/CPU | launch GPU/CPU | metadata wait GPU/CPU | tensor wait GPU/CPU | recv wait GPU/CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| PP1 → PP2 | 3.541 / 3.988 | 2.805 / 3.057 | 0.047 / 0.107 | 0.157 / 0.207 | 35.973 / 38.818 |
| PP2 → PP3 | 3.434 / 3.819 | 2.720 / 2.890 | 0.044 / 0.101 | 0.155 / 0.204 | 29.071 / 36.502 |
| PP3 → PP4 | 3.002 / 3.295 | 2.319 / 2.409 | 0.046 / 0.104 | 0.123 / 0.171 | 25.589 / 33.184 |
| PP4 head send | 0.120 / 2.932 | 0.003 / 2.071 | 0.003 / 0.107 | 0.100 / 0.161 | head recv 18.850 / 23.279 |

真实 device handle wait 只有约 0.1–0.2 ms；主要可归因时间在 host 发起、metadata
处理和流水线等待。`round9_send_breakdown/` 保存了 gate、profile JSONL 和
`pack-microbench.json`。后者在 GPU1 上对 B=5、hidden=5120、hc=4 的实际形状
做了正确性相同的打包计量：五次 eager copy 的 CUDA event 中位数为 108.544 us，
`torch.cat(..., out=...)` 为 17.920 us；固定地址 staging 的五次 graph copy 为
8.4992 us，交错 contiguous backing view 为 9.216 us。host enqueue 分别为
104.644、32.814、37.567 和 38.317 us。

机器没有安装 `cupti-python`，因此本轮使用 CUDA event fallback；microbenchmark
工具也报告 cold-L2 未启用，因为没有向计时闭包传入输入参数。这些数字只用于
阈值和热点排序，不能冒充 CUPTI kernel 级计量。结论是继续优化 host
orchestration 或只在能保持 kernel ABI 的前提下消除 copy，不能因为 handle wait
很小就贸然改成异步发送。

## 第十轮进度：packed staging 候选在 graph capture 被 ABI 拒绝

为消除 FULL graph 外部的五字段 staging copy，新增了临时 opt-in 候选
`dsv41_dspark_packed_staging_relay`：hidden/residual 和三组 mHC mix 共享两个
contiguous wire backing，接收端用预分配 buffer 直接填充。CPU 契约测试 20 passed，
并覆盖了 backing view 的精确 round-trip；随后按固定配置做了 A1/B。

| 组 | 结果 |
| --- | --- |
| A1：packed off | 72.074 / 74.164 / 73.423 tok/s，中位数 73.423 |
| B：packed on | CUDA Graph capture 失败，未进入 serving 请求 |

B 在 PP2、PP3 的 `mhc_post_tilelang` capture 中报：
`input a strides[2] violates packed ABI constraint; expected: 4, got: 1`。这是
`as_strided` relay view 不满足 mHC TileLang 的 packed ABI，而不是可接受的性能
负收益；继续修改布局再加载完整模型没有必要。B 失败后已立即停止服务，没有运行
A2。候选实现已从运行时代码撤销，失败证据和 A1 gate 保留在
`artifacts/dspark_flow_20260920/round10_packed_staging/`。后续若重新尝试，必须
先在 mHC kernel 的真实 stride contract 上做独立 GPU microtest，再进入模型加载。

## 第三轮进度：完成真实 NCCL relay 和探索性整模型 A/B/A

本轮先关闭既有 26663 服务，释放 GPU 2–6 后重新启动三次固定配置。三次服务都
使用同一份源码和模型：PP5/TP1、
GPU `2,3,4,5,6`、32K raw token prompt、q5、greedy、256 个输出 token、
CUDA graph FULL、完整 prefill/decode 路径、无 prefix
cache；`marlin_full_k=true` 和 `dsv41_strict_bf16_gemm=true` 固定开启，唯一
变量是 `dsv41_dspark_compact_relay`。每次请求的 delta token IDs、usage 和
Prometheus counters 都单独保存。服务已在 A2/off 完成后关闭，GPU 回到约 19 MiB。
这组结果用于缩小候选范围，不能替代第一轮 R1 的最终验收。

### NCCL transport 验证

`dspark_relay_bench.py --backend nccl` 现在使用真实 CUDA tensor 加 NCCL、Gloo
metadata 的 `GroupCoordinator.send_tensor_dict/recv_tensor_dict`，不是把 CPU/Gloo
结果冒充 NCCL。四个独立进程形成 head→stage1→stage2→stage3→head 环，数据仍是
synthetic fixture，不加载模型、不执行 draft/target kernel。

10 个 correctness 场景（B=1、2、8、6、1，各 off/on）和 18 个 A/B/A 时延格全部
通过。四跳 tensor 消息从 52 降到 44；B=1 每轮少 240 bytes，B=8 每轮少 1920
bytes。NCCL（`NCCL_P2P_DISABLE=1`、1 MiB buffer、8 channels）往返中位数如下：

| B | A1/off | B/on | A2/off |
| --- | ---: | ---: | ---: |
| 1/repeat0 | 19.408 | 14.357 | 16.457 |
| 1/repeat1 | 16.578 | 15.511 | 16.534 |
| 1/repeat2 | 16.714 | 14.679 | 16.691 |
| 8/repeat0 | 15.684 | 13.844 | 16.215 |
| 8/repeat1 | 17.282 | 14.467 | 15.549 |
| 8/repeat2 | 15.624 | 13.787 | 16.875 |

B 格多数低于对照，但 A1/A2 本身漂移明显；这是 transport-only 的候选信号，
不能换算成端到端收益，也不足以修改 NCCL 或默认开关。原始结果在
`artifacts/dspark_flow_20260920/round3/relay-nccl.json`。

### 探索性整模型结果

三次独立服务启动后的 gate 结果为客户端交付速率（首个 token-bearing chunk
之后的 delta token IDs），不是 CUDA step time：

| 配置 | 三次 post-first-chunk tok/s | TTFT (s) | draft / accepted |
| --- | --- | --- | --- |
| A1/off | 73.538, 73.988, 73.991 | 6.760, 6.440, 6.423 | 660 / 633 |
| B/on | 75.725, 75.364, 74.539 | 6.755, 6.417, 6.472 | 660 / 633 |
| A2/off | 71.995, 72.995, 70.326 | 6.685, 6.447, 6.435 | 660 / 633 |

每格均为 96,000 prompt tokens、768 generation tokens，三次请求各 256 tokens；
A1/B/A2 的 prompt SHA、每个请求的 256 个 token IDs 均完全一致。B 的中位数
为 75.364 tok/s，对照 A1 为 73.988、A2 为 71.995；但 A1→A2 的漂移约 2.7%，
且只有一个独立 B 窗口，故当前结论是“有正向信号，尚未达到改默认值的稳定证据”。
保持 `dsv41_dspark_compact_relay` 默认关闭，下一次独占窗口再重复一组完整
A/B/A；若收益仍超过 A1/A2 噪声并保持 token/counter/graph 一致，才考虑启用默认值。

本轮还将 `dspark_flow_gate.py` 的 schema 提升为 2：每次 gate 自动读取
`/metrics`，保存带完整 label 的 engine counters 和 before/after delta，覆盖
`prompt_tokens_total`、`generation_tokens_total`、draft/accepted totals 及
accepted-per-position。旧三份请求 JSON 保留 schema 1，并与同目录的 `.metrics.prom`
快照配对；后续运行会直接把 counters 写入 JSON。完整清单、源码 hash、启动固定项
和停止结论见 `artifacts/dspark_flow_20260920/round3/manifest.json`。

本轮代码验证为 **41 passed**（新增 relay/计量契约、既有 DSV4F DSpark 回归），
另通过 `git diff --check`、`bash -n benchmarks/deepseek_v4_1/run_port_pp5.sh`
和两个 benchmark 的 Python 编译检查；没有把服务请求或 NCCL transport-only
结果计入 CPU 单测数字。

## 第四轮进度：R1 硬件 off/on/off 验收完成

为回应第一轮“尚未验证”的状态，本轮在同一源码、同一模型和同一 PP5/TP1 配置下
重新完成 A1/off → B/on → A2/off。固定项仍为 32K prompt、q5、greedy、256
输出 token、CUDA graph FULL、完整 prefill/decode、无 prefix cache、
`marlin_full_k=true` 和 `dsv41_strict_bf16_gemm=true`；唯一变量是
`dsv41_dspark_compact_relay`。每次服务均在请求完成后关闭，26663 端口和 GPU
2–6 已释放。

### 正确性和计量结果

三组各执行 3 个请求，prompt SHA 和每个请求的 256 个 token ID 全部一致。每个
配置的 engine counter delta 也一致：96,000 prompt tokens、768 generation
tokens、660 draft tokens、132 draft rounds、633 accepted tokens，accepted
per-position 为 `[132, 132, 132, 120, 117]`。Gate JSON、原始服务日志和完整
`/metrics` delta 保存在
`artifacts/dspark_flow_20260920/round4/`，汇总见同目录的 `manifest.json`。

客户端首个 token-bearing chunk 后的中位速率如下；这是 emitted token 的交付
速率，不是把 SSE chunk 数当作 token 数：

| 配置 | 三次 post-first-chunk tok/s | 中位数 |
| --- | --- | ---: |
| A1/off | 69.865, 69.933, 72.472 | 69.933 |
| B/on | 70.244, 73.006, 74.677 | 73.006 |
| A2/off | 69.386, 70.883, 72.471 | 70.883 |

两组 off 的中位数合并为 70.408 tok/s，B/on 高 3.69%。A1/A2 中位数差为
1.36%，而 B/on 的三个样本均没有改变输出或接受计数；因此这次已经超过当前
off 噪声，并满足 R1 的硬件正确性和 request-level 改善验收。

### 低开销 step timing

服务设置 `VLLM_DEEPSEEK_V4_STEP_TIMING_EVERY=16`，只取稳定纯 decode 窗口，
排除首轮 Triton JIT 和 32K prefill。PP4（draft head）的 period 中位数为
83.485 ms（A1/off）、80.515 ms（B/on）、81.975 ms（A2/off）；PP0–PP3
分别为 83.43/80.08/81.93、83.64/80.20/81.91、83.90/80.92/82.64、
84.03/81.87/82.46 ms（按 A1/B/A2 排列）。五个 rank 的 B/on 都低于两组
off，且 PP4 draft p50 从 20.65/20.55 ms 降为 18.60 ms。该结果与 relay
删除中间 sampling/query tensor 后的预期相符。

R1 现已通过硬件验收，但开关仍保持窄范围 opt-in、默认关闭；是否把默认值切为
开启属于后续 rollout 决策，不与本轮验证混在一起。下一步转入 target metadata
和 prepare/relay wait 的静态边界审查，每个候选继续单独做 A/B/A。

### 下一候选的静态边界

当前调用链已经收窄到三个可独立计量的区段：

- `speculator.py::_prepare_stage_inputs` 按 `draft_kv_cache_group_ids` 循环调用
  Triton `prepare_dflash_inputs`，随后分别构造 context-KV 和 draft attention
  metadata。不能因为函数形状相同就直接合并循环；
  `nvidia/dspark.py::get_draft_kv_cache_layer_names` 为每个本地 draft layer
  返回独立的 SWA cache prefix，必须先记录每个 PP rank 的实际 group 映射。
- `participate_distributed` 的 `draft.stage.recv_wait`、`draft.stage.prepare`
  和 `draft.stage.send` 是 relay 候选的完整边界。R1 只减少中间 package 的
  tensor entries，没有改变 staging、graph replay 或 head result copy。
- `parallel_state.py::isend_tensor_dict` 先发 CPU metadata，再逐 tensor 发 device
  payload；同步 `send_tensor_dict` 会等待这一轮所有 handles。后续若要做异步
  overlap，必须单独证明 source buffer 的 stream 生命周期和 step-id 顺序，不能
  把 R1 的字段删除当成异步发送。

下一轮先从现有 phase profile 和 step timing 中记录每个 PP rank 的 group 数、
descriptor/package bytes、`recv_wait`、`prepare`、`send` 和 head recv wait；若
`prepare` 或 wait 的变化不超过 A1/A2 噪声，停止该候选，不进入共享通信基础设施
重构。

## 第二轮进度：R1 减少临时 view，并完成真实 CPU 通信验证

本轮使用独立 CPU/Gloo fixture，未向其他模型服务发送请求、修改启动参数或占用
其 GPU。继续使用第一轮独立 worktree，验证后合入三个无冲突的实现/测试文件。

### 已合入的增量

R1 原本先创建四个 sampling/query tensor view，再从中间包移除。本轮改为：
V4.1 纯 decode 中间 stage 根本不创建这些 view，同时用 `pop(key, None)` 丢弃
可能存在的入站副本；最终 stage、prefill、dummy、开关关闭及 DSV4F 路径仍创建
并发送完整字段。开关名称和默认关闭状态不变。

三段 pipeline 每轮少创建八个无用 view，其中 sampling 三项的 view 访问次数
由 9 降至 3，已在真实 stage 编排的 CPU 契约中计数；另两个为 query input view。
这减少 CPU tensor 包装工作，不等于少执行了八次 GPU kernel，也没有测得独立 ms 收益。

新增 `benchmarks/deepseek_v4_1/dspark_relay_bench.py`：

- 四个独立进程，head→stage1→stage2→stage3→head，使用真实
  `GroupCoordinator.send_tensor_dict/recv_tensor_dict` 与 Gloo。
- 数据为 synthetic CPU tensors，默认匹配当前模型 hidden=5120、hc_mult=4、q5
  的形状/类型。验证序列为 B=1→2→8→6→1，每个 B 都验证 off/on，检查每一跳的
  tensor、控制字段和 scheduler step；测试另外核对 fixture 与 runtime 的包字段。
- 不加载权重，不执行 draft/target kernel，阶段输出由预建 fixture 提供。
  因此这是 transport-only 验证，既不是整条 DSpark 算法验证，也不能外推 NCCL 性能。

### 通信结果与停止结论

10 组跨进程 correctness 场景通过。完整四跳的 tensor 消息 **52→44**；
B=1 少传 240 bytes，B=8 少传 1920 bytes，符合第一轮发送清单。

CPU/Gloo A/B/A，每格 50 轮、预热 10 轮，表中为每格往返延迟中位数（ms）：

| B | 重复 | A1/off | B/on | A2/off |
| --- | --- | --- | --- | --- |
| 1 | 0 | 13.664 | 11.067 | 11.014 |
| 1 | 1 | 11.428 | 14.346 | 36.962 |
| 1 | 2 | 40.301 | 7.876 | 7.967 |
| 8 | 0 | 13.243 | 12.206 | 15.679 |
| 8 | 1 | 96.005 | 101.844 | 96.705 |
| 8 | 2 | 96.532 | 13.292 | 14.080 |

**没有建立可靠加速证据。** 五组的 A1/A2 明显漂移；唯一漂移较小的 B=8/repeat1，
候选延迟反而高于对照。不可选取快的 B 单格或合并中位数宣称收益。当前宿主机同时
有其他工作，无法将波动归因于具体进程，本轮不重复扩大 CPU 实验。
R1 保持默认关闭，等待独立 NCCL/整模型窗口；没有把 Gloo 结果当作修改 NCCL
参数、重新分层或重启其他模型实例的理由。

复现命令（在独立 worktree 内，未访问模型服务）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python benchmarks/deepseek_v4_1/dspark_relay_bench.py \
  --rounds 50 --repeats 3 \
  --out /tmp/dsv41-dspark-flow-evidence-20260920/relay-gloo.json
```

### metadata 方向的进一步收窄

已追到 `nvidia/dspark.py::local_stage_indices/get_draft_kv_cache_layer_names`、
`DraftModelSpeculator.set_attn(active_layer_names=...)` 及
`DFlashSpeculator.set_attn`：在当前 `[1,2,3]` 布局中，每个 draft rank 只有一个
本地 draft layer/SWA cache，draft 输入准备没有此前设想的“四个本地 draft group”
可合并。ledger §9 四个 SWA builder 属于 target metadata。

因此暂不实施“draft 多 group 共用输入准备”的改动，也不引用 target 的 3–6 ms
作为它的预期收益。下一次真实 profile 应先分别记录 target group 数、draft group
数和其 builder 耗时；若仍是单 draft group，去重该循环不会省掉 kernel launch。
这不排除 target metadata 的优化，但它与 draft metadata 必须继续分别计量。

### 验证和后续接手

第二轮仍为 **40 passed**；增强了 sampling view 次数和 benchmark schema 对照。
保持第一轮相同的三份 pytest suite、CPU 模式，Ruff 和差异空白检查通过。
最新三文件与测试 worktree 的 SHA256 一致。

证据保存在 `artifacts/dspark_flow_20260920/`：
`round2.patch`（相对于第一轮）、`round2-manifest.json`、`round2-cpu.xml/log`、
`relay-gloo.json/log` 和 `relay-gloo-summary.json`。第一轮文件保留，不覆盖其历史结果。
后续合入基线需依次应用第一轮和第二轮 patch，且先检查当前工作区的其他修改。

下一步先测当前 R1 的 NCCL 与整模型 A/B，而不是叠加第三个未测开关。
如果 CPU 调度/metadata 仍主导且 R1 不能降低完整 step，则记录无收益并停止该候选；
继续 target metadata 前先落实模型专用边界和独立归因。

## 实施进度：2026-09-20，第一轮已完成代码和 CPU 验证

范围严格限定 **DSV41F（target `deepseek_v41`、40 层）的 distributed DSpark / TP1**。
基线为 `e52bf3dcc4`。本轮在独立 worktree
`/tmp/vllm170-dsv41-dspark-flow-20260920` 开发验证，再以最小补丁合入当前工作区；
未提交。本轮没有修改 prefill 移植接口或共享 PP 通信，未发送服务请求、重启模型
或运行 GPU workload。

| 项目 | 当前状态 | 证据边界 |
| --- | --- | --- |
| R1：中间 draft relay 去除冗余 sampling tensors | 已实现，硬件验收通过，默认关闭 | CPU relay/发送列表、NCCL fixture、整模型 off/on/off、token IDs、engine counters 和低开销 step timing 均通过；默认切换另行决策 |
| M1：按 API delta token IDs 计量 | 已实现独立 flow gate | 离线 SSE 回归通过；真实服务请求尚未执行 |
| 接收端持久 buffer 复用 | 暂缓，无此实现合入 | 长 prefill buffer 的驻留显存、增长/缩小及 stream 生命周期尚未验证 |
| metadata 去重、准备工作重叠、context KV graph | 未改代码 | 仍为后续候选，需区分 target 与 draft 的计时归属 |

### R1：只删除下游重新生成、仅 head 消费的数据

沿 `participate_distributed → _prepare_distributed_stage → _prepare_stage_inputs`
检查：每个 stage 均从 request 控制张量、`idx_mapping` 和本地 block table 重新生成
query/sampling buffers，不读取入站包中的下列四项：

`query_input_ids`、`sample_indices`、`sample_pos`、`sample_idx_mapping`。

PP1→PP2、PP2→PP3 可以不发送这四项；PP3→head 仍发送本地重建的完整结果。
hidden/residual/mHC、完整 context、request controls、step id 和 req id 校验全部保留。
不会提前采样、跳过 target verify 或修改 CUDA graph staging。

- `vllm/models/deepseek_v4_1/dspark_relay.py`：开关和模型/TP/distributed 范围检查。
- `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`：仅初始化开关和中间发送前
  剔除四项两处修改，没有改共享 PP 通信实现。
- 开关为 `additional_config.dsv41_dspark_compact_relay=true`，必须使用 JSON 布尔值。
  默认 false；DSV4F (`deepseek_v4`)、其他模型、其他层数、TP>1 和非 distributed
  DSpark 不开启。prefill、混合批次、dummy 均保留旧发送包。
- 标准三段 draft、q5 下，每轮少 **8 次 tensor send / 对应 recv**。
  B=1 少 240 bytes，B=2 少 480 bytes，B=3 少 720 bytes；通过真实
  `_split_tensor_dict` 枚举发送列表验证。节省的是小消息调度，不是大量链路带宽。
  不据此推算或声称节省了多少 ms、提高了多少 tok/s。

没有增加 cache、持久 GPU buffer、异步 handle 或新消息布局；当前任意 batch
大小都不沿用上一步的接收状态。后续若改 `_prepare_stage_inputs` 使其开始读取
这四个入站字段，必须同步修改此优化和契约测试。

### M1：独立计量入口，避免影响其他 serving 脚本

`benchmarks/deepseek_v4_1/dspark_flow_gate.py` 使用 `/v1/completions` 的
`return_token_ids=true`。当前 completion protocol 明确这些是每个 chunk 的增量
token IDs；脚本计入空文本 token，并校验其总数等于 `usage.completion_tokens`。

主要指标为 `(总 token 数 - 首 chunk token 数) / (末 token chunk 时间 - 首 chunk 时间)`，
字段名 `post_first_chunk_tok_per_s`。同时报告包含 TTFT/响应收尾的
`request_tok_per_s`，保留每个 chunk 的时间、完整 token IDs、usage、prompt SHA。
仅一个 token-bearing chunk 时区间吞吐为 null，不能声称测得稳态速度。
缺 token IDs、计数不匹配、缺 `[DONE]`、未达到固定输出长度直接报错，不降级回 chunk 计数。
该指标仍是客户端交付速率，网络合包会影响时间，不能当成 CUDA step time。

### 本轮验证和交接入口

在上述 worktree 内运行，`.venv` 与已编译扩展只链接到主工作区，未安装或重编译依赖：

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest \
  tests/v1/spec_decode/test_dsv41_compact_relay.py \
  tests/models/test_deepseek_v4_dspark.py \
  tests/benchmarks/test_dsv41_dspark_flow_gate.py \
  -q -p no:cacheprovider -o junit_family=xunit1 \
  --junitxml=/tmp/dsv41-dspark-flow-evidence-20260920/cpu-contracts.xml
```

结果：**40 passed**（14 项新 relay 契约、21 项既有 DSV4F DSpark 回归、5 项计量测试）。
relay 测试运行真实 stage 编排、`_prepare_stage_inputs` 参数传递和发送字典拆分，
仅用 CPU 替身替换模型/输入 kernel/graph replay；覆盖 eager/graph 分支、请求重排、
2→1→3 的行数变化、prefill/dummy 回退和 head 入参逐 tensor 一致性。
这不是 CUDA Graph、真实 GPU kernel 或模型质量验证。

Ruff check 和 `git diff --check` 通过；独立 gate 的 `--help` 通过。
合入后五个实现/测试文件 SHA256 均与已测 worktree 一致。
补丁、manifest、JUnit 和日志位于 `artifacts/dspark_flow_20260920/`，其中
`compact-relay.patch` 仅包含本轮代码/测试，不包含其他工作流修改。

旧 `tests/v1/spec_decode/test_dspark_pp.py` 在该基线引用了不存在的
`_collect_q1_drafter_audit_records` / `_get_q1_drafter_audit_eligible_rows`，收集失败。
本轮未修改或删除旧测试，新增独立 V4.1 relay suite；不能报告整个 spec_decode suite 通过。

R1 的独占硬件验收已经完成。后续重跑仍需固定同一源码/模型/配置，只改变上述
additional-config 字段，将结果写入新的 artifact，不能覆盖历史设置；同时继续
记录启动参数、源码及工作区 hash、graph hit、accepted/draft 和 step timing。
本轮实际使用的客户端入口为：

```bash
.venv/bin/python benchmarks/deepseek_v4_1/dspark_flow_gate.py \
  --base-url http://127.0.0.1:26663 --model dsv41f-port \
  --prompt-tokens 32000 --max-tokens 256 --repeats 3 --seed 0 \
  --label R1-A1-off --out artifacts/dspark_flow_20260920/R1-A1-off.json
```

分别使用 B/on、A2/off 标签和文件名；同时对照每个请求的 token IDs。若输出、
接受率或 graph 命中变化，先定位正确性，不把近似 prefill、模型质量或其他开关
的输出变化归因于 R1。

本指引以
[`vllm170-dsv41f-dspark-flow-ledger-2026-09-18.md`](vllm170-dsv41f-dspark-flow-ledger-2026-09-18.md)
和当前 `port/dsv41f-hybrid` 源码为基线。它只处理 DSpark 流程中每一步的
延迟和 CPU/GPU 空转；近似采样、模型结构变更、MoE kernel 重写和默认 draft
width 调整另行评审，不要把它们混入本轮速度结论。

## CED 状态和本指引范围

CED 移植已经完成，本指引不再安排 CED 兼容矩阵、接口等待或联合路径归因。
DSpark 流程线直接按现有完整 prefill/decode 合约推进；工作区中属于其他移植线的
文件仍需原样保留，但不作为本轮优化的变量、阻塞条件或性能解释。

## 当前建议的推进顺序

DSpark 流程线现在直接做不改变语义的工作：

1. 固定并修正 DSpark 计量脚本，补齐 engine counters、实际 emitted token
   数和低开销 step timing；这些改动保持现有 model runner contract 不变。
2. 对 `speculator.py`、`sparse_swa.py`、`parallel_state.py` 做调用关系和
   buffer 生命周期审查，输出 descriptor 字段、消息 bytes、同步点和潜在
   overlap 的静态清单，不先提交共享基础设施重构。
3. 为 metadata/relay 候选写 CPU 契约、失效条件和回归矩阵，覆盖完整
   prefill/decode、请求重排和 graph bucket 生命周期。
4. 在独立 worktree 逐项实现候选，完成契约测试后合入不冲突的小补丁。
5. R1 已通过最终 off/on/off 验收；现在推进 target metadata、准备工作重叠或
   context KV graph，每项都单独建立基线和停止规则，且不改变 R1 的变量。

本轮交付实际优化代码、计量和测试结果；其他移植线的 model runner、prefill
metadata 和 artifacts 只作为工作区边界保护，不参与 DSpark 速度结论。

## 先固定事实和验收口径

原 ledger 的 PP5/TP1 实例是 5 张 CMP170HX（SM80），`NCCL_P2P_DISABLE=1`，
PP0--PP4 各一张卡，draft stage 在 PP1--PP3，draft head 在 PP4。32K prompt、
q5、greedy 的最新历史结果是：draft graph forward 约 0.55--0.59 ms/层，
step 从 104.5 ms 降至 76.9 ms，约 75.19 emitted tokens/s。这个结果是已完成
的回归基线，不等于本轮任何新改动已经有效。

旧的阶段 profiler 会在每条记录后同步 CUDA，把约 76.9 ms 的 step 放大到
235--257 ms；它只能用于相位归因，不能用来加总成端到端延迟。新的低开销
计时通过 `VLLM_DEEPSEEK_V4_STEP_TIMING_EVERY=N` 定期解析 CUDA event，另行
记录完整 step、各 target slice、draft head prepare 和等待时间。

### P0：先修正计量，再动实现

1. `benchmarks/deepseek_v4_1/dspark_gate.py` 当前导入
   `decode_gate.run_once`，其吞吐仍按 SSE chunk 数除以时间。DSpark 一次
   step 可以把多个 token 合并到一个 chunk，因此不能用它给候选排序。
2. `dspark_stream_gate.py` 的 `usage.completion_tokens` 比 chunk 计数可靠，
   但 `(tokens - 1) / (last_chunk - first_chunk)` 默认第一个 chunk 只有一个
   token。实现者应记录每个 chunk 的实际 emitted token 数，或者同时报告：
   - whole-request `completion_tokens / (response_end - request_start)`；
   - engine 内部 accepted、draft、emitted 计数和 event step time；
   - 首 token 后的稳定 step 吞吐，明确是否包含 TTFT、EOS 和尾部 drain。
3. 不要用文本 delta 的空格或 UTF-8 边界自行猜 token。优先使用 engine 的
   spec-decode counters；若 endpoint 只能给 usage，则把它标成 request-level
   估计，并用足够长的输出降低首尾偏差。固定 prompt、max tokens、seed、
   temperature、request 数和 graph capture sizes，候选之间只改变一个变量。
4. 每次运行保存完整配置、commit SHA、GPU 顺序、模型路径、prompt token 数、
   emitted/accepted/draft 计数和原始响应。推荐目录格式：

   ```text
   artifacts/dspark_flow_20260920/<candidate>/
     config.json  metrics.json  counters.prom  response.jsonl  timing.log
   ```

   任何“更快”的结论都必须能从这些文件重算，不能只保留终端上的 tok/s。

## 优先级一：削减每步 metadata 和输入准备的 CPU 工作

原低开销记录中，target graph slice 约 6.9--10.1 ms，但 PP0 的 target
prepare 约 3.41 ms，PP4 约 6.05--6.32 ms；四个 SWA KV group builder 每个
rank、每步都会重复构造。2026-09-22 的独立 microtest 又证实 builder 外围
包含 GPU ragged 构造，不能将它全部归为 host orchestration。这里仍有可计量的
候选，前提是先把 CPU 构造、ragged GPU 工作和五个 rank 的关键路径分开。
这里来自 ledger §9 的 target metadata，不是 `_prepare_distributed_stage` 的
draft metadata，PP0 本来就没有 draft block。两者必须分别计量，不能将这组数字
当作 draft 输入去重的可回收时间。

重点检查 `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`：

- `_prepare_stage_inputs()` 当前遍历 draft KV groups，并为每个 group 调用
  `prepare_dflash_inputs`。把真正与 group 无关的 request、position、query
  和长度派生结果做成一次性的 step descriptor；只保留每个 group 必需的
  block table、slot mapping、window/causal 标志和物理 KV 索引。
- `_prepare_distributed_stage()` 中的 ragged `num_target_tokens`、
  `num_query_tokens`、`seq_lens_cpu_upper_bound` 和 context slot 必须继续按
  当前 request batch 计算，不能用 q5 的固定值替代。context KV 写入的 slot
  顺序必须与每个 group 的 block table 一致。
- `sparse_swa.py` 已经有 token-to-request cache 和 graph persistent buffer；
  不要重复造一个只缓存 window size 的全局表。causal/noncausal、短 extend、
  prefill tail 和不同 KV group 的物理 slot 都会改变索引。
- cache key 必须至少包含 batch request 顺序、每个 request 的有效 token 数、
  seq-lens、causal/short-extend 模式、KV group 配置和 block-table generation。
  scheduler 重排、抢占、释放 block 或 request 数变化时失效；不能以“本步
  仍是 B=1”作为唯一复用条件。

建议的最小实现顺序：先加一个不可变的 common descriptor，测量每一步构造
次数和耗时；确认结果后，再把 group-specific 的数组改成持久化 tensor 或
预分配 CPU buffer。每一步都要比较 descriptor 内容和旧路径的 slot、indptr、
indices，而不是只比较最终速度。若 descriptor 需要读取新的 prefill metadata，
先固定字段和生命周期，再把它作为独立候选；DSpark 的第一版只接受现有完整
decode/verify batch。

## 优先级二：压缩 PP relay 的控制和 tensor 传输

`participate_distributed()` 目前是 receive、校验、prepare、graph/eager、
再向下一 stage `send_tensor_dict` 的串行流程；`parallel_state.py` 的发送
实现会保留 isend buffer 和 handle 直到完成。目标不是全局修改 PP 通信协议，
而是在 DSpark 消息上减少 Python dictionary、tensor 数量、临时分配和无效等待。

建议 DeepSeek 按以下顺序验证：

1. 为每个 stage 记录 message count、按 dtype 的总 bytes、CPU pack/copy 时间、
   `isend`/wait 时间和 receiver 到达时间。先证明瓶颈是消息调度还是链路带宽。
2. 将稳定的 step header（`scheduler_step_id`、req id 顺序、`num_reqs`、
   `num_query_per_req`、有效 token 数和 dummy 标记）编码成一个固定布局，
   将同 dtype 的小 tensor 合并到持久化 buffer。可变的 hidden/residual、
   sample indices 和 context output 仍需按其生命周期传输。
3. 用双缓冲或 ring buffer 复用地址，只有在上一个 handle/event 完成后才重用。
   禁止发送方在 receiver 仍可能读时覆写；在 CUDA graph replay 地址固定的
   阶段，buffer identity 是契约的一部分。
4. 保留 `_validate_distributed_package()` 的 step id、dummy marker、req id
   顺序和 request 数校验。任何“省 header”方案必须在校验前仍能检测错步、
   错序和错误 batch，不能以吞吐换 silent corruption。

如果 pack 后只减少了 Python 调用，却增加了一个大同步或额外 device copy，
应回退。验收必须同时报告 relay CPU 时间、通信等待、head recv wait 和整个
step；只看 send 本身会把 pipeline skew 误当收益。任何 prefill/decode 模式都必须
让 header 中的 request order、有效行数和 context generation 与实际输入保持一致，
不能把未定义的裁剪行语义压进 DSpark relay header。

## 优先级三：让可独立的准备工作提前，但不破坏依赖

draft proposal 依赖 target sampling 产生的 hidden、sample indices 和当前
step 控制信息，不能把“下一轮 draft”错误地提前到本轮 verify 完成之前。
可尝试的是把**同一步已经确定的控制和纯 CPU 准备**提前传播，并与上游 stage
正在进行的上一段 GPU work 重叠：

- 在 `_propose_distributed()` 产生 step header 后，尽早发送不依赖新 hidden 的
  request order、长度、graph bucket 和 descriptor generation；
- 各 stage 收到 header 后准备 CPU descriptor 和持久化地址，等 hidden/residual
  到达再做最后一次 copy/dispatch；
- 对每个候选记录 header-ready、data-ready、dispatch-ready、graph-start 四个
  event，确认 overlap 是真实重叠而不是把等待移到别的相位。

必须继续保持 `_validate_distributed_package()` 的同一步校验，且 request 被
重排、取消、抢占或改变 prefill/decode 状态时丢弃旧 descriptor。原始计量使用
`NCCL_P2P_DISABLE=1`；2026-09-22 已验证 P2P 可用，但同一 PCIe switch 上
提前发送仍可能增加链路竞争；若通信 bytes 或
head recv wait 上升，应停止该方向。

## 优先级四：谨慎处理 context KV 和 graph descriptor

历史上尝试把 context KV 全部 graph capture 的方案失败过：draft query rows
是 `5B`，verify context rows 可以是 `6B`，warmup 还可能是 `7B`（例如 8 请求
的 padding）。不能从 q 或 batch size 推断唯一 descriptor，也不能只把当前
`6` 改成一个更大的固定值就宣布完成。

若要重试，先选定一种明确契约：

- 为 context rows 建独立 capture descriptor，明确有效行数、padding 行和
  `-1`/dummy slot 的语义；或
- 使用固定上限 buffer，并让 ragged mapping 显式携带每行的 request/position，
  确保 padding 行不会写真实 KV，也不会让 sample index 指向上一轮数据。

先做 B=1 的 q5→target6、带 padded7 warmup、请求数增长到 8、短 prefill tail
和 request reorder 的 correctness 测试，再测 graph hit rate、CPU prepare、
context KV event 和端到端 step。任何 stale mapping、旧 sample row、错误 slot
或 graph miss 都比 1 ms 的潜在收益严重；没有独立 descriptor 设计时不要重开
这条实验线，也不要把设计审查结果写成真实 graph/context 收益。

## 明确暂不优先的方向

- target graph slice 已在约 7--10 ms，先不要重写 target kernel 或追逐更大的
  graph bucket；先确保实际 `B*(q+1)` 和 `B*q` bucket 命中。`[8]` 不能覆盖
  q5 的 target verify6，且会静默回 eager。
- Markov head 已降到约 1.3 ms，继续把五个因果、token-conditioned sample
  过程强行并行化会改变语义。只有拆分出不依赖当前 sampled token 的工作后才
  讨论 overlap。
- 不要用 draft-like 的 128-expert/top3 微基准推导 target MoE 的普适下界，
  也不要在本流程任务中重新开启 MoE tile/format 矩阵。
- 不要以概率采样、q7/q5 改动或模型质量变化抵消流程延迟；这些必须有单独的
  正确性和质量验收。

## 必须覆盖的回归矩阵

在任何优化合入前，至少运行以下 CPU/单测契约；涉及 CUDA graph 的项目再用
固定硬件运行最小 GPU 矩阵：

| 场景 | 要证明的性质 |
| --- | --- |
| B=1，q5，target6 | draft/target graph 命中正确 bucket，sample row 不错位 |
| padded7 warmup 后实际 target6 | padding 不污染 context KV、sample indices 和 residual |
| request 数 1→2→8 | descriptor、block table、indptr/indices 按 batch 失效/重建 |
| q=1、q=5，短 prefill tail 1/2/5/6 | prefill/decode 分类和 causal 标志不被 cache 复用污染 |
| request reorder、抢占、取消后复用 | 旧 generation、旧 req id 和旧 slot 不会被接受 |
| PP1/PP2/PP3 relay 顺序异常 | step id、dummy、req 顺序校验仍能 fail closed |
| 2 个连续 step | 双缓冲/handle 生命周期不读写交叉，第二步不带第一步数据 |

输出 token 必须与 target-only greedy 基线逐 token 相同；出现第一个差异时，
保存该 step 的 descriptor、slot、indices、header 和 logits 摘要，先修正确性，
不要继续比较速度。已有的五个 reference prompt 只说明有限覆盖，不能替代上面
这组 batch/生命周期测试。

## 推荐的执行和停止规则

每轮只选一个候选：先在同一 commit 做 A1/B，预热、prompt、max tokens、
graph sizes、环境变量和 GPU 排序完全相同；每个实例内用低开销 event timing
和正确的 emitted-token 计量重复至少三次。B 已经明确没有正向收益时立即停止，
不再加载 A2；B 有正收益或结果落在噪声带内时，再做 A2/off 排除时序漂移。
报告 median、p10/p90、step time、tokens/step、accepted/draft、CPU prepare、
relay wait、graph hit/miss，以及显存峰值。

只有在端到端稳定 step 或 request-level throughput 有可重复的实际改善，且
没有 correctness、graph miss、显存或通信回归时，才进入下一候选。一个候选
在 A1/B 中没有超过测量噪声、只改善 phase profiler、或只减少某个 send 而
总 step 不变，应停止并记录为无效方向。不要为了重新加载 20 分钟的完整模型
而扩大实验矩阵；优先复用已经运行的实例和离线 descriptor/单元测试，把硬件
运行留给通过静态审查的候选。

新证据追加到本 ledger 的后续日期小节，保留旧数字和“计量错误已修正”的
历史说明。每个候选写清：假设、改动文件、依赖不变量、A/B/A 原始 artifact、
首个收益相位、端到端结论和停止原因。这样 DeepSeek 可以按优先级逐项实现，
也能在收益不成立时快速回退，而不会把 pipeline 等待、SSE chunk 或 profiler
同步误报成流程速度。

## 2026-09-21：真实 stride contract 已确认，接收 buffer 候选停止

针对 `mhc_post_tilelang` 单独增加了
`benchmarks/kernels/mhc_post_tilelang_stride_contract.py`。测试先构造完整宽度
张量，再按 fast prefill 相同的 `index_select(0, keep)` 取行，最后分别调用低层
预分配输出 kernel 和现有会分配输出的 wrapper，并与 PyTorch reference 比较。
GPU1 上 `T=1,4,8,128,1024` 五组全部通过；实际 stride 固定为：

- `x[T,5120]` BF16：`[5120,1]`；
- `residual[T,4,5120]` BF16：`[20480,5120,1]`；
- `post_mix[T,4,1]` FP32：`[4,1,1]`；
- `res_mix[T,4,4]` FP32：`[16,4,1]`。

所有选中张量都是 contiguous。低层 kernel 中位数为 55--93 us，wrapper 为
86--122 us；工具提示环境没有 `cupti-python`，所以这里使用 CUDA event fallback，
不能把差值解释成完整 allocator 收益。任意 `as_strided` 的 packed view 不满足
这个 contract；此前 graph capture 的失败正是 `strides[2] expected: 4, got: 1`。

随后只对五个状态 tensor 做了 transport-only NCCL A1/B，不加载模型，也不跑 A2。
baseline 模拟“接收新 tensor 后 copy 到现有 staging”，B 直接接收进 staging 并
跳过 copy。第一轮 B=1/8 的中位数为 `16.432→16.310 ms`（-0.7%）和
`17.265→15.876 ms`（-8.0%）；独立 batch=8 A1/B 重复为
`14.871→16.088 ms`（+8.2%）。方向不稳定，不能形成可复现的正收益，因而不把
预分配接收 buffer 合入运行时代码，也不继续 host metadata 优化。候选实现已经
撤回；保留的基准和结果为：

```text
benchmarks/deepseek_v4_1/dspark_recv_buffer_bench.py
artifacts/mhc_post_tilelang_20260921/stride-contract.json
artifacts/mhc_post_tilelang_20260921/recv-buffer-nccl-with-copy.json
artifacts/mhc_post_tilelang_20260921/recv-buffer-nccl-with-copy-r2.json
artifacts/mhc_post_tilelang_20260921/assessment.md
```

本轮没有启动或保留服务；现有工作区中的 CED 移植、fast prefill 默认值和
prefix caching 配置继续按此前结论保留，不作为本轮变量。

## 2026-09-22：P2P 已可用，但对当前 PP5 DSpark 无端到端正收益

GPU2–6 的内容级 peer copy、远端 SM 读和远端 SM 写已通过。9 月 21 日第一次
P2P-on 整模型尝试只完成两次 32K/256 请求，第三次发生 CUDA launch failure；
服务日志没有保留，且该次未固定本轮的严格 kernel 配置，不能用它比较吞吐。
CUDA 故障处理后，本轮重新完成了固定配置 A1/B，各三次请求，无服务崩溃。

固定配置为 PP5/TP1、GPU2–6、32K prompt、q5 greedy、256 输出 token、
FULL graph、fast prefill 开、prefix caching 关、text-only、
`max_model_len=262144`、`max_num_batched_tokens=1024`、KV cache 700 MB，
并同时开启 `marlin_full_k`、`dsv41_strict_bf16_gemm`、
`dsv41_strict_prefill_tail` 和 compact relay。两组只切换
`NCCL_P2P_DISABLE=1→0`。NCCL 初始化日志分别确认 PP 路径为
`SHM/direct/direct` 和 `P2P/CUMEM`，不是只凭 `nvidia-smi topo` 判断。

| 计量 | A1：P2P 关 | B：P2P 开 | B 相对 A1 |
| --- | ---: | ---: | ---: |
| 三次 post-first-chunk tok/s | 63.677 / 63.224 / 59.235 | 60.468 / 62.354 / 60.984 | 中位数 -3.54% |
| PP4 稳态 period，16-step 窗口 p50 的中位数 | 81.66 ms | 84.61 ms | +3.61% |
| PP4 target / draft / prep 窗口中位数 | 11.19 / 17.16 / 7.19 ms | 11.37 / 18.26 / 7.34 ms | — |
| 独立 compact relay 四跳，150 轮 pooled p50 | 14.994 ms | 14.553 ms | -0.441 ms（-2.94%） |

PP4 period 排除首个包含 prefill 的 16-step 窗口，保留每组三个请求共九个纯
decode 窗口；每个请求三个窗口的 period 中位数，A1 为
81.66/81.49/87.88 ms，B 为 85.46/81.83/85.11 ms。客户端与服务端的
三次请求均显示时序漂移，不能把小幅差值归因为链路本身。六次请求的 256 个
token ID 逐项一致，engine delta 也一致：150 个 draft round、750 个 draft
token、624 个 accepted token、768 个 emitted token。与 9 月 20 日不同配置的
132-round 基线相比，当前输出从第 55 个 token 开始分歧；因此只比较本轮 A1/B。

独立 relay 使用实际 head→PP1→PP2→PP3→head 卡序 `6,3,4,5`，每轮四跳共
44 个 tensor message、954,512 bytes；每次计量前后均校验完整 package。
P2P 对它只节省约 0.44 ms，无法解释或消除约 82–85 ms 的整步开销。
五个 rank 的纯 decode target slice 中位数约为
`8.24+9.50+9.71+11.08+11.19=49.72 ms`（A1）和
`8.21+9.50+9.70+11.48+11.37=50.26 ms`（B）。这些是各 rank 的窗口
中位数之和，用于估计串行 verify 路径，不能当作同步量到的四卡通信时间；
它也明显高于 target-only 的约 22.48 ms graph replay。当前主导项仍是 q6
target verify，随后是约 17–18 ms 的分布式 draft 和每 rank 的 metadata/输入
准备。不要把 PP1–PP3 的 `recv_wait` 流水线空泡相加为通信耗时。
本轮每 150 个 draft round 交付 768 个 token，平均 5.12 token/round；按
A1 的窗口中位数估算，49.72 ms target slice 约占 81.66 ms period 的 61%。
即使把其余阶段全部理想化为零，当前接受率下仅 target verify 的粗略上界约为
`5.12/0.04972≈103 tok/s`。这进一步说明单独压低 relay 传输不足以把
DSpark 提到远高于当前约 63 tok/s 的水平。

B 没有端到端正收益，遵守停止规则不跑 A2，也不将启动脚本的
`NCCL_P2P_DISABLE` 默认值改为 0。独立 64 MiB all-reduce 的 P2P 带宽收益
只说明大消息集合通信能受益，不能外推到此处的小消息 relay。下一步先拆
q6 verify 中 target MoE/attention 的真实占比和四组 SWA metadata 的 host
构造成本，按一次约 50 ms 的 target 串行路径排序候选；任何改动都必须在相同
token IDs、accepted/round 计数下验证整步和 request-level 收益。

原始记录和重算摘要保存在
`artifacts/dspark_flow_20260922/p2p_ab/` 的 `manifest.json`、两份 gate JSON、
两份服务日志和两份 transport-only JSON/日志。服务已关闭，26663 端口和
GPU2–6 无计算进程。

### SWA metadata 外围并非纯 host 耗时

当前 SM80 `TRITON_MLA_SPARSE_DSV41` 使用
`DeepseekV41AmpereMLAAttention → DeepseekV41ROCMAiterMLAAttention`，其 SWA
metadata builder 在 `super().build()` 之后，还调用
`build_ragged_indices_from_dense()` 生成 indptr、打包 dense indices，然后
`_copy_ragged_to_graph_buffers()` 复制进 graph 稳定地址。旧 phase profile 的
`target.attn_metadata.swa.*` 子阶段没有单独覆盖这段 wrapper 工作；因而把
builder 总 span 与已命名子阶段之间的差值全部称为 Python/host 开销证据不足。

`benchmarks/deepseek_v4_1/swa_ragged_metadata_bench.py` 在空闲 CMP GPU1 上用
实际 q6 行数、SWA width=128、int32、四个独立 group，先校验 indptr、有效
index 前缀和 graph buffer 指针，再用 20 次预热、100 次 CUDA event 计量：

| eager wrapper 边界 | GPU event p50 | host enqueue p50 |
| --- | ---: | ---: |
| 一组 `T=6,width=128` | 0.425 ms | 0.452 ms |
| 四组串行 | 1.541 ms | 1.569 ms |

这里保留了生产 wrapper 的动态分配、`cumsum`、Triton pack 和输出 copy；
没有安装 `cupti-python`，使用 CUDA event fallback。它是单卡孤立计量，
不能直接从五个 rank 的时长相加得到端到端收益，也不能拿它解释全部
6–8 ms target prepare。它将下一轮 metadata 审查缩小为两件事：先在真实
服务中分别计量 `super().build()` 与 ragged wrapper，再判断是否值得把
indptr/pack 直接写入持久化 buffer；在证明实际 step 收益前不改运行时代码。
原始样本和设置在 `artifacts/dspark_flow_20260922/p2p_ab/swa-ragged-q6.json`。

## 2026-09-22：先归因 target-only 42→32 tok/s 退化

这轮先暂停扩大 DSpark 实验矩阵，用同一份 32K prompt、64 个输出 token、
10 次重复和显式 `max_num_batched_tokens=2048` 追查 target-only 基线。所有组的
prompt SHA256 都是 `4b9344c728afc904...`，并关闭 step timing，排除了 profiler
本身约 0.5 ms/token 的影响。

首先用当前代码恢复 9 月 18 日的运行参数：PP5/TP1、FULL graph、1M 最大上下文、
1.3 GB KV、fast prefill 关、prefix caching 关，不传 additional config。结果为
`23.604 ms/token = 42.376 tok/s`，与历史 artifact 的
`23.798 ms/token = 42.038 tok/s` 重合。因此 42→32 不是 9 月 18 日至当前的代码
版本退化，也不是 2048 budget 变化；退化位于后来启用的精确性配置。

| 配置增量 | ms/token | tok/s | 相对上一组 |
| --- | ---: | ---: | ---: |
| 当前代码、历史运行参数、无 strict/full-K | 23.604 | 42.376 | 基线 |
| 只开 `dsv41_strict_bf16_gemm` | 28.846 | 34.669 | +5.242 ms |
| 再开 `marlin_full_k` | 30.590 | 32.691 | +1.744 ms |
| 9 月 22 日完整配置（修复前） | 30.139 | 33.182 | -0.451 ms |

`dsv41_strict_bf16_gemm` 单独解释完整配置退化的 80.2%。它不是只约束 prefill：
worker 启动时持久覆盖 `mm/addmm/matmul/linear/bmm/softmax/mean`，并让 mHC 的
decode split-K 固定回 1。`marlin_full_k` 再解释剩余主要部分；fast prefill、
strict prefill tail、较短 max model len 和较小 KV budget 合计反而净省约
0.45 ms。两项精确性开关的增量之和为 6.986 ms，已覆盖从 23.604 到
30.139 ms 的全部退化，不能再把这 6.5 ms 归到 PP 通信。

### SM80 strict GEMM 缺少调优表

上游 persistent batch-invariant matmul 只有 Ada、Hopper 和 Blackwell 的形状表，
SM80 对所有 BF16 形状都回退到通用 `128x128x64, 8 warps, 3 stages`。CMP 170HX
上的 M=1 独立计量表明，大量时间来自遮掉 127 行的无效工作。保持
`BLOCK_K=64` 不变并只调整 M/N tile、warp 和 stage 后，结果与原 strict kernel
逐元素一致：

| M=1 GEMM（K=5120） | 通用 strict | SM80 配置 | 倍率 |
| --- | ---: | ---: | ---: |
| N=129280，lm_head | 3215.7 us | 1262.9 us | 2.55x |
| N=5120 | 262.7 us | 87.2 us | 3.01x |
| N=2304 | 139.1 us | 63.8 us | 2.18x |
| N=1536 | 126.8 us | 50.7 us | 2.50x |
| N=512 | 123.7 us | 47.2 us | 2.62x |
| N=384 | 124.8 us | 54.3 us | 2.30x |

实现为 SM80 增加上述六个 M=1 bucket；M>1 仍使用原通用配置，并保持同一个
`BLOCK_K=64`，所以每行的 K reduction order 不变。生产 `wo_a` 的
`B=8,M=1,K=4096,N=1024` BMM 也增加 SM80 M=1 配置，裸 kernel 从
89.2 降到 47.9 us。相关改动位于
`vllm/model_executor/determinism/batch_invariant_configs.py`、
`vllm/model_executor/determinism/batch_invariant.py` 和已有 determinism 测试。

相同历史参数、同时开启 strict/full-K 的整模型 A/B 为
`30.590→28.010 ms/token`、`32.691→35.703 tok/s`，吞吐提升 9.2%。修复前后
64 个 token ID 逐项一致，token 序列 SHA256 都是
`489e906d657d3e8f68f0200486776210480bd66ceb618b65ee5d0d94e7e44643`。
因此该补丁保留 strict 的输出契约，但只追回 2.58 ms；相对无 strict/full-K
基线仍有 4.41 ms，需要继续拆分 full-K、mHC 固定 split 和尚未调优的 strict
算子，不能宣称已经恢复 42 tok/s。

在实际配置（2048、fast prefill 开、prefix caching 关、262K/700 MB、
strict prefill tail 和 compact relay 保持）上，修复前后为
`30.139→27.877 ms/token`、`33.182→35.874 tok/s`：延迟下降 7.5%，吞吐提升
8.1%。实际配置的 token 序列也与修复前 strict/full-K 组逐项一致。这个结果
将可用 target-only 基线从约 32--33 提回约 35.9 tok/s，但仍低于没有两项精确性
开关的约 42.4 tok/s；下一轮应以剩余约 4.27 ms/token 为分母继续拆解，不能把
35.9 当作结案值。

Marlin 的 `thread_k/thread_n/num_threads=128/128/256` 在 M=1 的两个大投影上
可省约 15--17 us，但 M=281 没有对应 kernel 实例，不能满足 full-K 的跨 batch
形状契约。这一候选按停止规则淘汰，没有继续跑 A2，也没有修改 Marlin 运行
代码。

完整 `test_matmul_batch_invariant.py` 在空闲 SM80 上为 38 passed；新增测试覆盖
BMM 的 M=1 调优 bucket 与 M=8 的逐元素一致性，已有 compressor FP32 测试覆盖
M=1 到 M=1024 的跨 bucket 一致性。ruff check、ruff format 和 `diff --check`
均通过。原始 gate、服务日志、microtest JSON 和 token-ID 对照位于
`artifacts/dspark_flow_20260922/target_only_match/`。
