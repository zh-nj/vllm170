# DSV41F prefill 第五轮：将混合 MoE 推广到 DSpark-5

日期：2026-10-03。基线提交 `b250c9690d`，生产要求保持
**DSpark=5、CED 开启**。原始证据保存在
`artifacts/dsv41f-ced-prefill-round5-20261003/`。

本轮补齐第四轮确认的推广缺口：已有 BM128 pairs / BM64 tails
在 DSpark 配置下被静态门控拒绝，即使 production additional-config
保留 hybrid=true，也不实际执行。现已适配 DSpark-5 的实际
M=2044，并完成同实例端到端对照。

## 同口径端到端收益

两类输入均为 32768 token：既有周期 prompt 和异质代码语料。
每次请求使用新 cache salt，assert prefix hit=0；greedy 固定输出
16 / 64 token，全部输出 IDs 与既有参考一致。

同一私有 PP5 实例运行两条路径，配置、权重、缓存、DSpark 和 CED
相同。baseline 只在调用原 `Experts.apply` 时暂时禁用该 expert 的
hybrid scratch，candidate 直接使用新生产 dispatch。未替换 align 或
native GEMM；两条路径的诊断 wrapper 相同。计时阶段关闭 Event profiling。
各路径、各输入预热一次，随后按 AB/BA/AB/BA 运行四组对照。

| 冷 32K 输入 | 原路径 TTFT p50 | 混合路径 TTFT p50 | 降幅 | 四组配对降幅范围 |
| --- | ---: | ---: | ---: | ---: |
| 周期 prompt | 4.07794 s | 3.76633 s | 7.64% | 7.49%–7.76% |
| 异质代码 | 5.19788 s | 4.93962 s | 4.97% | 4.76%–5.58% |

两类输入均满足预设的每组改善、p50 改善超过 2% 的验收条件。
上述 TTFT 是客户端至首个含 token-ID 的 SSE chunk 的时间，包含请求
处理、prefill 和首 token 交付；不是 GPU kernel 时间。该收益范围仅对应
本轮两个固定冷输入和单请求配置，不外推其他长度、并发或模型。
证据：`experiment.json` 的 `ab`、`summary`，以及 `experiment.log`。

## 实际形状和生产路径命中

运行时确认单请求完整 prefill chunk 为 M=2044，而非 M=2048。
DSpark-5 的 parallel drafting 预留 4 个新 slot，scheduler 的 2048
预算因此只剩 2044。32K 请求包含 15 个完整 chunk；完整 chunk 的
encoder MoE 调用在 PP0/PP1/PP2 上分别为 120/120/60 次，共 300 次。
hybrid W13 和 W2 各命中 300 次，baseline 各为 0。PP3/PP4 无该路径命中。

M=256 的起始小块和 prompt tail 继续使用原路径。额外检查
129/2049/2050 token 请求，两条路径输出 IDs 相同且无 hybrid 调用。
证据：`experiment.json` 的 `warmups[*].inventory` 和 `fallbacks_exact`。

## 热点分解与机制

验收之后对代码语料分别做一次 CUDA Event 分解，并捕获各 encoder
layer 的首个 M=2044 chunk 路由。两条路径的 20 层专家 IDs、权重、
histogram 和 stride 完全一致，因此 tile 减少没有来自路由变化。

对每个专家设 `b=ceil(count/64)`，将相邻 BM64 块配对为 BM128，奇数
尾块保留 BM64。逻辑 M tile 数由 `b` 变为 `floor(b/2)+(b%2)`，
padding 行数仍为 `64*b`，不扩大稀疏专家的尾块。

| 20 层首个完整 chunk 的路由总计 | 原 BM64 | 混合分块 |
| --- | ---: | ---: |
| 有效 expert slot | 245280 | 245280 |
| 逻辑 M tile | 8129 | 6714 |
| padded rows | 520256 | 520256 |

逻辑 M tile 减少 17.41%。W13/W2 同时复用分块元数据，BM128 部分使用
已有 persistent native GEMM；收益同时包含分块和调度变化。
逻辑 tile 数不能当作实测 HBM 字节，本轮没有据此归因具体 cache miss
或 HBM stall。

| 代码语料 M=2044 的 300 次调用，跨 rank Event span 之和 | 原路径 | 混合路径 |
| --- | ---: | ---: |
| W13 GEMM | 3483.487 ms | 3128.708 ms |
| W2 GEMM，输入为 12264 routed slots | 1845.667 ms | 1669.234 ms |
| align，含混合 metadata 构建 | 5.994 ms | 9.799 ms |

目标 W13+W2 span 之和下降约 9.97%，metadata 增加约 3.80 ms。
这些是同 rank 同 stream 的不重叠 GEMM span；跨 PP rank 的相加表示
工作量，不能作为 TTFT 的严格串行分解，也不能与 inclusive parent
stage 相加。Event instrumentation 会改变调度和吞吐，验收收益使用
前面的未插桩 AB/BA。证据：`profile-summary.json`，可通过
`analyze_profiles.py` 从现有数据重算，无需再跑 GPU。

## 适配实现与回退

修改 `mxfp4_int8_sm80.py`，保留全局 opt-in
`dsv41_sm80_moe_hybrid_prefill` 默认 false；既有本机 production 配置
显式 true。本轮没有修改启动脚本或 DSpark 默认值。

配置门仅接受已验证的两种情况：无 speculation 时 M=2048；DSpark
5 token、query tokens 为默认或 5、draft ranks `[1,2,3]`、greedy、无
adaptive verification、无动态 token 数映射、预留 4 slot 时 M=2044。
继续要求 Deepseek V4.1 Flash、SM80、H5120/I2304/E384/top-k6、BF16
SILU、无 bias/LoRA、PP5/TP1/DP1、V2 runner、CED on、scheduler 2048/8
及既有未分片条件。

hybrid scratch 按实际 M×6 构建，并在权重加载后、Graph capture 前
预热；padding sentinel、有效 slot 上限、GEMM output capacity 均保持
一致。没有改变 workspace lane 或 Graph 地址生命周期。
执行门继续要求 pure eager prefill、无 decode/mixed batch、无 ubatching、
Graph NONE、BM64/BN128、精确 M。decode、draft E128/top-k3、其他
speculation、形状和拓扑继续回退。已有 `HybridRoutes` 和 native
CUTLASS kernel 均未修改。

回退方式是将 additional-config 中
`dsv41_sm80_moe_hybrid_prefill` 置 false 并按封存 production 命令重启；
DSpark=5 和 CED 无需关闭。

## 正确性与验证命令

扩展已有 kernel suite，增加 DSpark 配置门控和不支持配置回退，并将
完整 `Experts.apply` 测试参数化为 M=2048/2044。测试涵盖
quantize → W13 → SILU → W2 → reduce、output/workspace alias、热专家
到分散路由变化；输出和 W2 slots 均 BF16 逐位一致。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python -m pytest \
  tests/kernels/quantization/test_mxfp4_int8_sm80.py \
  -k hybrid_prefill -q

.venv/bin/pre-commit run --files \
  vllm/model_executor/layers/fused_moe/experts/mxfp4_int8_sm80.py \
  tests/kernels/quantization/test_mxfp4_int8_sm80.py \
  docs/design/vllm170-dsv41f-ced-prefill-round5-2026-10-03.md
```

定向 suite 为 **34 passed, 247 deselected**，实际 SM80 CMP170HX 空闲卡
验证，没有占用服务卡。源码、测试的全部 pre-commit hooks 已通过。
私有完整模型的两类输入各 20 个 encoder layer `Experts.apply` 原始
输出均有限且逐位一致。长 decode 256 token 及双请求各 128 token
在两条路径下共 1024 个 IDs 与参考相同；decode 的 DSpark draft/接受
计数也一致。证据：`integration-tests.log`、`pre-commit.log`、
`raw_experts_check`、`private-preservation.json`。

本轮只进行一次私有模型加载和一次正式生产恢复，不扫描 q，不启用
FP8 dense TileLang，不实施尚无硬件证据的小尾块方案。

## 正式生产交付

停止私有 `dsv41f-ced-prefill.service` 后，确认无遗留 GPU 计算进程、
26661/26662 listener，再恢复原 `dsv41f-manual.service` 命令。
保持模型 `/media/z/data6/models/DeepSeek-V4.1-Flash`、served name
`dsv41f`、端口26661、物理 GPU2–6、PP5/TP1/DP1、DSpark-5、draft
ranks `[1,2,3]`、greedy、draft Graph、CED、prefix cache 及原 KV/scheduler
参数。生产 hybrid flag 原已为 true，本轮适配后首次在此 DSpark 配置命中。

| 无诊断扩展的正式服务检查 | 结果 |
| --- | --- |
| 两次32K/256 decode | 各256 IDs 与既有参考一致 |
| 预热后单请求客户端 decode 吞吐 | 97.525 tok/s，首个含 token-ID chunk 后计时 |
| 对应 speculation 计数 | 44轮，提出220 token，接受211 token |
| 双请求各128 token | 两路全部 IDs 一致 |
| 冷周期32K/16 | TTFT 3.82898 s，prefix hit=0，16 IDs 一致 |
| 同盐 prefix32K/16 | TTFT 0.54158 s，命中32512 token，16 IDs 一致 |
| 冷代码32K/64 | TTFT 4.91326 s，prefix hit=0，64 IDs 一致 |

交付共 864 个输出 IDs 一致。上述生产时间是 fresh-process 验证记录，
不作为新增 paired A/B；收益百分比使用前面的同进程对照。保留 CED
和 DSpark 的重算后缀，未改变 prefix cache 策略。

最终 MainPID=3458144，五个 worker 为3460728–3460732，目标 GPU
计算进程全部属于正式 unit；GPU1 验证卡无计算进程。health=200，
dev RPC=404，无 worker extension、dev mode、PYTHONPATH 或性能插桩；
17 个封存运行源码摘要与当前文件一致。私有 unit 已移除，26662 无
listener，running/waiting 请求数均为0。

证据：`production-launch.json`、`production-validation.json`、
`production-validation.log`、`production.log`、`final-state.json`。
生产复核可用 `verify_production.py`，进程归属与清理可用
`finalize_state.py`，都使用 `.venv/bin/python`。本轮交付源码、测试和
本文件三个改动，后续第六轮继续复用该适配。
