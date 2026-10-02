# DSV41F target 第四轮：单请求 token-map 推广

日期：2026-10-01。基线为上一轮已部署的 q1–48 attention 8 warp 和 M1/M6 W13
packed-word；本轮保留这些工作树改动，只新增 V4.1 metadata 优化。
原始证据：`artifacts/dsv41f-target-opt-round4-20261001/`。

后续第六轮用同进程三组对照重新验证了token map及SWA融合，见
[metadata端到端验收](vllm170-dsv41f-metadata-e2e-2026-10-02.md)。
本文件中的跨重启结果保持原口径，不能与新campaign直接相减。

## 1. 先纠正热点测量口径

复用上一轮 trace 做离线分解，没有再次扫描 q。发现启用所有 nested phase scope
时，插桩整轮周期中位数为 **278.589 ms**，上一轮无插桩约68 ms/轮。
`target.attn_metadata` 的 CPU span 19–23 ms 不代表关闭插桩后的真实成本。

原因有源码和 trace 两类证据：

- `phase_profile_stage` 在 host scope 两端记录 CUDA Event，metadata 的 Event
  区间包含等待宿主 enqueue 的空隙，不能解释为 metadata GPU kernel 的纯执行时间。
- 不启动 kernel 的 `split`、`prefill_common`、`tile_scheduler` 等 scope 也有
  0.1–0.2 ms 级 host span。大量 nested instrumentation 会抬高整个 builder span。
- metadata 位于 `phase_profile_step` 之前；最外层 scope 关闭时会同步和 flush。
  每次约50–51条记录，逐条取 metadata、创建目录并打开 / append 文件。
  PP1–PP4 的第一个记录到 outer metadata 记录的 timestamp 跨度约12.5 ms，
  此段还不包含最后一次写入和首次写入前的同步，属于写入阶段时间的下界。
- stage 本身不会每次同步设备；同步发生在 outer scope / step 的 flush，
  不能称为“每个 nested scope 都同步”。

这些结果解释了旧 trace 为什么不能用来分配当前的真实关键路径占比。
它们不推翻前轮的独立算子正确性、隔离 A/B 和 target Graph 边界结果。
本轮不修改 profiler，不把“降低测量干扰”计为推理收益。
离线产物为 `rounds-candidate/`、`metadata-flush-write-spans.json` 和两个
`*-stage-summary.json`。

## 2. 找到并补齐推广缺口

SWA builder 已有单请求快路径：从持久 int32 buffer 取当前 slice，**每次清零**，
并发布到 `CommonAttentionMetadata` cache。V4.1 的 compressed MLA 和 compressor
却仍调用通用 `token_to_req_indices`，每步执行 GPU subtraction、arange、
repeat_interleave 与 copy，即使唯一请求的所有 token 都属于 request0。

本轮将同一机制接入：

- `vllm/models/deepseek_v4_1/sparse_mla.py`：compressed MLA C1/C2。
- `vllm/models/deepseek_v4_1/compressor.py`：compressor 的 ring slot mapping 输入。

两个 builder 的加载期 gate 都要求：现有
`VLLM_DEEPSEEK_V4_PERSIST_SINGLE_REQ_TOKEN_MAP=1`、CUDA、**精确 SM80.0**。
每次 build 再要求 `num_reqs == 1`，与 `max_num_seqs` 无关。
其他硬件、开关关闭和多请求继续调用原通用算法。
没有修改 generic metadata、V4.0、MLA/indexer kernel、MoE 或 FP8 dense 路径。

每次 zero 是必要条件：同一个 buffer 会被多请求批次写入1、2等 request ID，
单请求不能认为“最初分配时为零”就跳过刷新。返回值仍来自持久 buffer，
不引入新分配或改变 Graph 的地址稳定性。该机制也适用于单请求的其他 query
长度和 prefill；本轮量化 q5/q6，未把它们的数字外推成长 prefill 收益。

## 3. 关闭 phase hook 后测量真实 builder

先用小型 prototype 证明机制，再用正式 benchmark 对实际 `builder.build` 的
快路径开关做三个交替轮次，每 arm / 轮40次。所有输入和输出检查在计时外。
设备：空闲 CMP170HX SM80，UUID `GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216`，
PCI02:00.0。没有 CUPTI/NCU，也未改 GPU 时钟。

输入为合成32K上下文、最大2048 token capacity、max-num-seqs8、block-table width8192。
每个 builder 构建独立的 common metadata，模拟独立 KV group。
此 benchmark **不包含 indexer、runner、PP 接收或采样**，不是完整 target 耗时。
eager 的 host enqueue 和等待完成时间用 `perf_counter_ns`；另捕获 builder GPU
操作，使用 CUDA Event 测 Graph replay，从而分开 host 成本与 GPU 工作。
小 metadata 测量不用于带宽上限推导，没有权重冷缓存负载。

| q6 单请求组合 | host 基线→本轮 | host 降幅 | 隔离 Graph GPU 基线→本轮 |
| --- | ---: | ---: | ---: |
| MLA C1/C2 + compressor，机制对照 | 0.682→0.189 ms | 72.37% | 0.0676→0.0225 ms |
| MLA C2 + compressor + 四组 SWA，代表性 source rank | 1.600→1.229 ms | 23.19% | 0.0911→0.0625 ms |
| MLA C1/C2 + 四组 SWA，代表性双 MLA rank | 1.549→1.197 ms | 22.71% | 0.0891→0.0604 ms |

后两组分别省 **0.371 / 0.352 ms host**；GPU 约省0.029 ms。
说明本轮主要削减 Python/Torch 调度与小 kernel launch 的固定成本。
不能把0.37 ms乘以五个 rank 当作整轮节省，因为 metadata 与 upstream 前向可重叠。

q5 的两组含 SWA 组合：1.605→1.244 / 1.521→1.175 ms host。
双请求 q12 对照：2.240→2.223 / 2.132→2.144 ms host，方向混合，无收益结论；
代码确实保持原通用路径。所有组合的 request map 与独立 expected map 精确一致，
Graph replay 后也一致。完整三轮样本见 `token-map-builders.json`。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python -m benchmarks.deepseek_v4_1.token_map_metadata_bench \
  --samples 40 --out /tmp/dsv41f-token-map-builders.json
```

## 4. 正确性与已有测试缺口

复用 `tests/kernels/attention/test_dsv4_sparse_prefill_sm80.py`，不新建镜像测试。
已有单请求与真实 multi→single→multi transition 测试现在可以覆盖 V41 两个
builder。将 transition 测试扩展到 CPU 和 CUDA，检查刷新值、dtype、cache 和
持久 pointer，防止把上个批次的 request ID 带入本轮。

同时修复旧 fixture：补 `additional_config={}`；把已不存在的
`_use_paged_decode_swa` 断言改为实际 V41 的 `_use_direct_ragged_swa`。
这是 fixture 对当前代码的校准，未改变对应 runtime 行为。
上轮排除的3项测试现在均通过。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python -m pytest \
  tests/kernels/attention/test_dsv4_sparse_prefill_sm80.py -q
```

结果：**52 passed / 14 warnings / 22.87 s**，没有 deselect。
四个本轮 Python 文件的 pre-commit 全部适用 hook 通过，包含 mypy；
`git diff --check` 通过。MoE runtime 未改动，沿用上一轮的247项通过证据，
没有重复运行不相关的硬件测试。

## 5. 生产验收

生产 A/B 使用原 `dsv41f-manual.service`，只重启一次，不改变 unit 参数。
PP5/TP1、DSpark5、FULL Graph、KV、CED、prefix cache、上一轮 kernel 优化均保留。
同一32768-token prompt，greedy、ignore_eos、生成256；64-token warmup 后各三次。
不打开 phase `ENABLE`，指标为首 chunk 后255个 token 的吞吐，排除 TTFT。

| 版本 | 三次无插桩 tok/s | 中位数 |
| --- | --- | ---: |
| 基线 | 82.986 / 86.735 / 87.752 | 86.735 |
| 本轮 | 86.243 / 83.804 / 87.090 | 86.243 |

中位数变化 **−0.57%**，`measurement_inconclusive`，没有整轮提速结论。
六次输出256个 ID、接受计数和逐位置接受完全相同：44轮、draft220 / accepted211，
逐位置44/44/44/40/39；prompt SHA 与上轮相同。
仅三次重复、前后顺序 A/B、没有重启后基线回测，不能判定确定回退。
宿主 memory PSI 在验收期间约27%，只是环境观测，不作为变化的因果解释。

另做一次关闭 phase 的双请求并发32K/128输出回归，两次128个 ID 均与上轮已部署
版本的并发结果精确相同。该请求没有 FULL Graph 命中计数或独立接受率，
不作 q12 target 耗时和吞吐归因。见 `concurrent-output.json`。

新 MainPID3190020、PP0–PP4 worker3194963–3194967，GPU2–6归属正确；
health200、实际模型路径正确、running/waiting均0，phase关闭。
unit前后逐字相同，部署前冻结的源码 SHA 在测量后仍相同；
空闲测量卡恢复6 MiB / 0%利用率。见 `final-state.json`。
本轮一次加载约6分钟，没有再次 q 扫描或多轮重载。
旧进程停止再次触发 unit 的20秒超时和 SIGKILL 清理；新服务随后正常启动，
GPU上已无旧 worker PID。停止阶段日志保存在 `restart-journal.log`，
不能把这次旧进程清理记录当作新 metadata 路径的运行错误。

**metadata 局部成本下降成立，整轮吞吐提升仍未证明。** 新代码已在生产进程加载；
没有把局部23%的降幅记作端到端收益。工作树尚未提交。

## 6. 排除与保留的下一步

只读盘点还发现 indexer 在 adaptive verification 下存在 variable flatten 分支，
但生产 `enable_adaptive_verification=false`，均匀 query 已有专用 kernel；
本轮排除该候选，没有为不可达路径跑额外实验。

四组 SWA 仍分别生成 validity：`is_valid_token.copy_(slot_mapping >= 0)`。
slot mapping 来自不同 KV group，尚未证明 mask 逐步相同，因此本轮不跨组共享。
后续可考虑组内融合，必须保留 live PAD / rejected suffix 的语义。
当前证据只确认局部 metadata 固定成本，尚不能解释旧拟合截距33.87 ms的全部来源。

本轮两份 runtime 原文件位于 `before/`，`restore_before.py` 校验 candidate SHA、
unit、请求空闲和 phase 关闭后，只恢复这两份文件并重启。
恢复 helper 未实际执行，stop 超时会在复制前中止。
它保留上一轮8 warp / packed-word 及其他工作树改动。
