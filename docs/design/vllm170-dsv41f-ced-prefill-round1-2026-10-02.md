# DSV41F CED prefill：热点测量、Engram 重叠与大行量化融合

本轮按单请求冷 32K prefill 验收，CED 开启、DSpark 关闭。已完成一次私有
诊断服务加载、Engram 同进程端到端筛选、已有 SiLU/INT8 融合的大行算子
实验和正式服务恢复。没有接入新的推理优化：Engram 最大 TTFT 改善只有
0.24%；大行量化融合的算子收益折算到 PP0 约 4.9 ms/request，尚无服务
A/B 证据。本轮没有修改推理源码，开始时的 21 个未提交文件逐个校验保留。

所有原始数据和可执行脚本位于
`artifacts/dsv41f-ced-prefill-round1-20261002/`。`baseline-state.json` 保存
HEAD、既有修改 SHA256 和原服务实际 argv；`launch.json` 保存诊断启动与
源码摘要；`production-launch.json` 保存恢复启动；`analysis.json` 保存
测量汇总。HEAD 为 `4df3a185db133e63934d5739cc18582a215be609`。

## 负载与计时边界

- DeepSeek-V4.1-Flash，PP5/TP1/DP1，分区 `8,8,8,8,8`，SM80 CMP170HX。
  PP0/PP1/PP3 HBM 时钟为 1728 MHz，PP2/PP4 为 1458 MHz。
- TileLang 0.1.15、Torch 2.13.0+cu130、Triton 3.7.1；FP8 KV，2.7 GB/rank，
  `max-num-batched-tokens=2048`、`max-num-seqs=8`。
- `ced_prefill`、已有 SM80 MoE TileLang 和 fused SWA metadata 开启；
  target prepare-ahead 关闭。实际 argv 没有 speculative config。
- prefix caching 保持生产配置开启，每次冷请求使用独立 `cache_salt`，
  从 metrics 验证命中增量为零；不以“prefix caching 开着”推断请求热冷。
- 输入复用 `20260928_verify_sweep/q6/request/prompt_ids.json`，长度 32768，
  共 65 种 token，主体为周期性重复片段。本轮结论限于这个固定负载，
  不把它视为所有真实长文本的专家/Engram 分布。
- 输出为 greedy、ignore EOS 的 16 token，验证 token IDs、usage 和完整结束。
  TTFT 是客户端收到首个 token-ID chunk 的时间，包含调度、encoder、
  CED suffix replay、PP 交接和采样；`输入长度 / TTFT` 是服务 prefill 指标。
  decode 恢复验证另使用已有 256-token 与并发 128-token 参考。

正式服务初始冷 32K TTFT 中位数为 **3.9719 s**，约 **8250 token/s**。
冷 2K 两次为 0.7225/0.7255 s。JIT/预热请求单独保存，不纳入这些中位数。

## CED 的实际执行与热点

诊断扩展只在私有 loopback 26662 服务启用。它用 CUDA Event 在纯 eager
prefill 的少量父级阶段记录事件，请求完成后通过一次 RPC 同步、读取。
没有在每层读取 event 或插入测量用的 stream synchronize。正式计时的
Engram A/B 关闭事件记录，但各 arm 使用相同 Python hook。

一次冷 32K profile 的 TTFT 为 **4.0175 s**，以下是每个 rank 的整请求
累计 stream 区间，单位 ms。它们是包含 CPU enqueue 间隙的区间，不能
视为纯 kernel 时间；子区间与父区间也不能相加。

| rank | model.total | layer.total | attn.total | moe.total | Engram prepare | Engram lookup | CED source publish |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PP0 | 3316.68 | 3263.19 | 1152.10 | 1675.04 | 46.91 | 46.30 | — |
| PP1 | 3185.50 | 3129.44 | 1147.21 | 1616.79 | 46.02 | 45.23 | — |
| PP2 | 1649.45 | 1624.37 | 557.98 | 911.88 | — | — | 2.01 |
| PP3 | 84.69 | 76.18 | 28.17 | 41.10 | — | — | — |
| PP4 | 84.30 | 78.06 | 28.58 | 42.38 | — | — | — |

PP0/PP1 的 MoE 分别占本 rank 前向区间 **50.5%/50.8%**，attention 占
**34.7%/36.0%**。Engram prepare 各约 1.4%，lookup 是其子区间。
`CED source publish` 只包围 `publish_decoder_kv`，不包含之前的 mHC 和
之后的 export clone；仅凭 2.01 ms 不能把整个 CED 数据交接标成 2.01 ms。
当前父级测量也不能进一步区分 W13、W2、router、mHC 和 GPU 等待。

每个 rank 都有 22 个 prefill model 调用。PP0/PP1 各执行 176 个 layer
调用；PP2 为 92 个（4 层完整 encoder × 22 chunk，加最后 4 层 replay）；
PP3/PP4 各只有最后 8 层 replay。这个行进方式符合 CED 的源码边界。

**五个 rank 的 model 区间之和为 8320.62 ms，超过客户端 TTFT 4017.45 ms。**
因此 prefill 有跨 rank 的区间重叠，不能复用 decode 的 PP 串行求和口径，
也不能用 8.32 s 当 prefill 延迟或以它为端到端收益分母。目前尚未测出完整
时间线，不能从这些累计区间确定每段等待的关键路径贡献。

实测 32K 分块为 **6×256 + 15×2048 + 1×512**。PP0 六个启动小 chunk
累计 315.13 ms、PP1 316.02 ms，里面包含处理 1536 个 token 的必要工作，
不能把整个 315 ms 当作可消除的固定成本。

`Scheduler._adaptive_long_prefill_pp_startup_tokens` 等于 threshold 乘
`max_concurrent_batches`；V2+async+PP5 的后者为 6，故启动窗口是 1536。
running 和 waiting 分支都会应用该上限。这是为在途 PP 批次接纳后来并发
请求保留的窗口，不是 CED decoder 重复执行。本轮保留该策略，没有把只
对 solo 请求可能有利的设置推广到有并发的生产服务。

## 实验一：Engram lookup 与 encoder 重叠

`prefill_worker.py` 的 baseline 调原实现；候选把 lookup 放到专用 CUDA
stream，在 `embed` 消费前等待 ready event。`async_full` 使用原完整 grid；
`async_background` 使用已有 background grid。只对 pure prefill、TP1、CPU
offload、rows>1、非 graph capture 启用。持久 staged buffer、ready event、
未消费 lookup 的收尾等待均保留，避免下一请求覆写未完成输出。

同一进程对三个 arm 分别预热后，以 baseline/full/background 和反序各
测一轮；所有输出 IDs 与 baseline 一致，冷 KV 命中均为零。数据保留于
`experiment.json`，每个 arm 两次，属于筛选而非统计显著的最终 A/B。

| arm | TTFT 1 s | TTFT 2 s | median s | 相对 baseline |
| --- | ---: | ---: | ---: | ---: |
| baseline | 3.99103 | 4.02563 | 4.00833 | — |
| async_full | 4.00765 | 4.00000 | 4.00383 | -0.112% |
| async_background | 3.98873 | 4.00887 | 3.99880 | -0.238% |

两个候选都未通过事先设定的 1% 筛选门槛，没有执行更长的 A/B 或接入。
已测 lookup 合计约 91.53 ms，分散在 PP0/PP1，单 rank 的可重叠空间小于
旧非 CED profile 暗示的量级。并发 lookup 还可能争用内存访问资源，但本轮
没有硬件计数器或分项候选 profile 支持该归因，不能把它作为已证实原因。

## 实验二：推广已有精确 SiLU＋INT8 量化融合到大 prefill

生产融合当前只覆盖 `num_routed <= 128`。CED suffix 的 128 个 token
对应 768 路由行；普通 256/512/2048-token chunk 对应 1536/3072/12288 行，
均落回 CUDA activation 与 Triton per-token INT8 两段路径。

本轮在诊断服务停止、没有 serving GPU 工作时，使用 PP0 物理 GPU UUID
`GPU-db10ad12-cd93-e577-b5e7-a59f3d6c177f` 筛选**已有冻结 256-thread
TileLang kernel**，没有改 kernel 或运行时上限。实际模型 clamp 为 10。
对所有有限 BF16 gate 的 SiLU rounding certificate 通过；四个大行形状的
INT8 输出与 FP32 scale 均逐位一致，包含生产 q/scale 复用 BF16 workspace
的布局，以及 graph replay 覆写测试。

计时复用 MoE harness，CUDA Event+Graph、三轮交错 A/B、每轮 15 个样本、
16 次调用/graph，分配、编译、参考计算和 copy 均在计时外。沿用此卡不支持
CUPTI 的既有记录，本轮直接使用 Event fallback；warm-buffer 结果不能当
cold-HBM 上限。

| 路由行 | separate us | fused us | 延迟下降 | 节省 us |
| ---: | ---: | ---: | ---: | ---: |
| 768 | 16.128 | 12.224 | 24.2% | 3.904 |
| 1536 | 22.144 | 15.680 | 29.2% | 6.464 |
| 3072 | 46.208 | 34.368 | 25.6% | 11.840 |
| 12288 | 167.552 | 130.112 | 22.3% | 37.440 |

按实测 chunk 计数，每个完整 encoder layer 的 GPU 节省预测为
`6×6.464 + 15×37.440 + 1×11.840 = 612.224 us`，PP0 八层合计
**4.898 ms**，约为 profile TTFT 的 **0.122%**。这是算子投影，不包括
CPU launch/allocator 收益，也没有处理完整 PP 关键路径。没有为这个量级
再加载一次候选服务，因此**没有该候选的端到端增益结论**，不扩大生产门。

原始计时为 `activation-screen.json`；执行时脚本快照另存
`activation_screen_measured.py.txt`，当前可复用脚本为 `activation_screen.py`。
后续修改仅绑定 loop closure 参数及去除错误的 generic clock 元数据，不改
算子或计时函数。原始 metadata 的 `clock_and_power` 通过 NVML index 0
读取了显示卡，不能使用；实际 GPU 由 Torch UUID、PCI 84:00.0 和显式
`CUDA_VISIBLE_DEVICES` 验证。后续脚本不再输出这个字段。

## 下一轮的测量范围

下一轮优先把 PP0/PP1 的 MoE 拆成 W13、W2、activation/quant、router 和
shared expert，配合每个 chunk 的时间线，再讨论 packed-word 等已有优化
如何推广。旧 prefill 文档已记录多次 tile/stage/grid 负结果，不重做大扫参。
旧文档里的独立路由或集中路由也不能替代本轮真实输入的直接观测。

另一处待量化的位置是
`DeepseekV41AmpereMLAAttention._combine_prefill_indices`：eager 模式在
合并索引前调用 `current_stream().synchronize()`。源码说明它用来限制
移除 Torch 隐式同步后的临时张量峰值。它落在当前 `attn.total` 内，但
现有 profile 没有单独记录其 host wait。本轮未删除或减少这个 fence；
后续需同时记录等待、allocator 峰值和 workspace 生命周期，才能设计
可验证的缓冲区复用/同步调整。

## 正式服务恢复与验证

正式 `dsv41f-manual.service` 已恢复，MainPID 4004338；诊断 unit inactive，
26662 无 listener，正式 26661 health=200，dev RPC 返回 404。实际进程
argv/env、模型路径、V2 runner、五个服务拥有的物理 GPU、源码与原有修改
摘要均通过校验。没有 DSpark、worker extension、dev mode、PYTHONPATH
override 或 phase/step profile。原有 CED/MoE/SWA 开关保留。

| 恢复验证 | 结果 |
| --- | --- |
| 冷 32K，排除预热后两次 | 3.96809 / 3.98041 s，中位 3.97425 s，8245 token/s |
| 32K 同盐 prefix hit | 命中 32640 token，保留 128-token suffix，TTFT 0.42045 s，16 个输出 IDs 与冷请求相同 |
| 129 / 2049 / 2050-token 冷请求 | 正常完成；部分形状发生 JIT，仅作边界交付检查 |
| 冷 64K 一次探查 | 7.60160 s，8621 token/s，正常完成；不是稳态多次中位数 |
| 单请求 256-token decode | 50.34 token/s，输出 IDs 与前轮参考逐个一致 |
| 并发两请求、各 128 token | 两路均与前轮参考逐个一致 |
| speculative metrics | 增量为零 |

上述恢复数据来自新进程，是基线恢复校验，不是任何候选的因果增益。
原始结果见 `production-validation.json`、`production-prefill.json`。

测试首跑隐藏 CUDA 后出现 12 个 attention/ROCm 兼容模块导入失败，另一个
调度测试的 OPT fixture 指定 8192 长度，超过 fixture config 的 2048 上限。
保留首跑日志；让 CPU 算术测试可见闲置 CMP GPU，并仅在 pytest 设置
`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` 后，该次运行为 **96 passed、1 failed**。
剩余 PP3 调度测试需要三张可见 GPU 作配置校验，按三卡可见条件单独重跑
为 **1 passed**，结果见 `adaptive-startup-test.log`。共 97 个选中用例分别
通过，156 个未选中用例未执行。这些运行条件不写入正式服务配置。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 .venv/bin/python -m pytest \
  tests/models/test_deepseek_v41_ced.py \
  tests/models/test_deepseek_v41_fast_prefill.py \
  tests/v1/core/test_scheduler.py \
  -k 'ced or fast_prefill or adaptive_long_prefill' -q

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216,GPU-db10ad12-cd93-e577-b5e7-a59f3d6c177f,GPU-77253a69-8a00-4c58-ad6c-d8f9b7a28311 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_USE_V2_MODEL_RUNNER=1 \
  .venv/bin/python -m pytest \
  tests/v1/core/test_scheduler.py::test_adaptive_long_prefill_caps_pp_inflight_startup -q
```

本轮没有增加模型影响改动，不将固定 token-ID 交付校验宣称为完整模型
质量评测。新增测量脚本通过 ruff/check 与 format，`git diff --check` 通过。
