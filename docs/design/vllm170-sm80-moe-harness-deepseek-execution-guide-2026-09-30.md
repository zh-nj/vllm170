# 给 DeepSeek 的执行指引：仅 DSV41F SM80 MoE grouped GEMM 验收设施

日期：2026-09-30。这是一份待执行任务书，不是实现完成或 GPU 测试通过的报告。

后续实际 kernel 优化可参考
[DSV41F SM80 TileLang 重写与离线 autotune 方案](vllm170-dsv41f-sm80-tilelang-gemm-plan-2026-09-30.md)。
先完成本文验收设施，再按该方案验证候选；该链接不改变本文第一批任务的完成边界。

## 1. 任务与完成边界

你在 `/mnt/data/apps/vllma100/vllm170` 工作。请完成 **仅面向 DSV4.1-Flash
（dsv41f）SM80 routed MoE 的第一批 grouped GEMM
测试评价设施**：独立数值参考、受生产接口约束的路由用例、正确的字节统计、统一
CUDA Graph 测量和可复现结果。目的是为后续专家 kernel 优化建立可信验收基线。

**用户的硬约束：仅优化 dsv41f 推理中的 SM80 MoE grouped GEMM，不得影响其他模型推理。**

- DSV4F、Qwen 及其他模型，其他 GPU 架构、其他量化/MoE 后端均不在范围内。
- dsv41f 的 shared expert、attention、router 选专家算法、DSpark 采样算法均不优化；
  routing 数据仅用于复现 grouped GEMM 的输入与验证写回。
- top-k3 用例仅代表 dsv41f 服务配套的 DSpark 专家调用，不授权修改全局 draft 路径。
- 本轮只改离线设施，不改生产默认值、全局环境变量、通用 backend 优先级或公共 dispatch。
- 后续若优化共享 CUDA/Triton 实现，必须先列出全部调用者，采用独立入口或精确模型门控，
  联合确认 dsv41f 模型身份、SM80 目标设备和指定 MoE 后端；仅凭 shape、top-k 或
  GPU capability 匹配不足以隔离模型。不能把通用 major==8 当作 dsv41f 身份。
- 后续生产改动必须增加非目标模型/后端的分派与默认配置回归，证明旧路径仍被选择；
  共享实现发生变化时，还需相应数值回归。不能仅靠“dsv41f 跑通”宣称其他模型不受影响。

请实际修改并验证这些设施，不要再只输出可行性分析。用户把本任务的具体执行交给你。
不要在本轮重写 CUDA kernel、切换 MoE 后端、改变默认 tile/量化语义、启动自进化、
修改服务启动参数或重启现有模型服务。DeepGEMM 源码作为只读参考。
提交、推送和 PR 不属于本任务；保留可审查 diff 和结果。

正常完成条件是：CPU 参考检查、小规模 GPU 正确性、受控生产形状基线均有证据，
JSONL 能区分通过、失败、跳过和未校准。若没有空闲 GPU，先完成代码与可运行的
CPU 检查，明确报告 GPU 验收待执行；不能把这种状态标为任务完全通过。

完整 routed MLP、模型 A/B、BF16 legacy 对照和新 kernel 属于后续阶段；
本轮不要以扩大这些范围代替完成 W13/W2/reduce 的基础验收。

## 2. 开始前必须核对

阅读根目录 `AGENTS.md`、`.agents/skills/kernel-microbenchmark/SKILL.md`，以及：

- 本仓库 `docs/design/vllm170-sm80-moe-grouped-gemm-harness-review-2026-09-30.md`。
- 来源说明 `/mnt/data/apps/DeepGEMM/docs/sm80_test_harness_port.zh.md`。

来源说明有已确认的错误，以本地源码契约为准，不按其中占位容差和任意布局扫描照搬。
审计时两个 HEAD 为：

```text
vllm170: f9c6a54b064cd1178828bdf7fe73f283908223c4
DeepGEMM: 057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7
```

先执行并保存到本轮 artifacts：

```bash
pwd
git status --short
git rev-parse HEAD
git -C /mnt/data/apps/DeepGEMM rev-parse HEAD
command -v uv
test -x .venv/bin/python
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
```

HEAD 不同先核对受影响接口，无须为了匹配旧报告 reset。记录初始脏文件；已有评审
文档可能尚未提交，不得覆盖、清理或把它归为你的新增实现。
所有 Python 命令用 `.venv/bin/python`；环境管理用 `uv`，不用系统 python3 或裸 pip。
已有环境可用就复用，不重装 torch/CUDA。缺失依赖先定位最小需求并记录实际动作。

## 3. 修改文件与源码地图

| 文件/符号 | 本轮动作 |
| --- | --- |
| `benchmarks/kernels/deepseek_v41_mxfp4_int8_sm80.py` | 主扫描入口，扩展 experts/topk/路由模式/参考校验/JSONL；保留现有入口兼容性 |
| `benchmarks/kernels/deepseek_v41_moe_bandwidth_probe.py` | 修正 W13/W2/联合字节口径，标明纯读对照的边界 |
| `benchmarks/kernels/deepseek_v41_moe_harness.py` | 可新增一个离线 helper 模块，集中 case、独立参考、指标、计时和结果 schema；保持职责清楚 |
| `tests/kernels/quantization/test_mxfp4_int8_sm80.py` | 扩展现有文件，增加有实际数值断言的非零 GEMM/路由/pack 测试 |
| `benchmarks/kernels/deepseek_v41_mxfp4_int8_accuracy.py` | 参考其原始 MXFP4 与 remap 质量分解；非必要不改 |
| `benchmarks/kernels/utils.py::Bench/ArgPool` | 复用或借鉴计时设施；不要为了本任务改动全部 benchmark 公共行为 |
| `vllm/model_executor/layers/fused_moe/experts/mxfp4_int8_sm80.py` | 只读：生产 shape、tile 选择、slot 行索引、完整 apply 次序 |
| `vllm/model_executor/layers/quantization/utils/mxfp4_int8_sm80.py` | 只读：pack、shift、A8 量化、近似 remap 契约 |
| `csrc/libtorch_stable/moe/mxfp4_int8_sm80/` | 只读：dispatch、解码、INT32 累加、输出缩放、reduce |

新增 helper 不导入 `deep_gemm`。不要让 pytest 通过 `sys.path` 修改去依赖 benchmark
脚本。测试中保留少量可手工核对的独立标量/稀疏期望值，benchmark 使用自己的完整
向量化参考；这种交叉验证比共享同一份有潜在共同错误的实现更有价值。
如确需共用测试支持代码，先查 `tests/kernels/quant_utils.py` 等现有位置；
不要把离线验收代码塞入生产热路径。

借鉴 DeepGEMM 的设计不等于原样导入包：`deep_gemm/__init__.py` 会加载 `_C`。
若复制实质代码，保留来源 commit，并按照源 `LICENSE` 携带完整版权与许可声明；
不要只写一句来源链接就认为保留了许可文本。

## 4. P0：先写明数值与接口契约

先在结果说明中写出以下契约，再写测试。每个测试说明它守护的错误是什么，
优先使用最低成本层级，避免新增仅验证函数存在或参数透传的测试。

### 4.1 支持范围

- H=5120，I=2304，TP1；W13 的 N=4608/K=5120，W2 的 N=5120/K=2304。
- kernel A 为 INT8，A scale 和 channel scale 为 FP32，输出 BF16。
- 对外生产 MoE 接收 BF16 激活并内部量化；不要把 benchmark 直接 A8 输入误称为完整 MoE。
- top-k 仅 3 和 6；本轮不增加 dtype/masked/EP/TP 轴。
- 暴露的 tile 为 `(16,64)、(16,128)、(64,128)、(128,128)`，以实际 dispatch 再确认。
- pack 层可以测 N 尾部、K=32/64 等；它们不因此成为固定 shape CUDA GEMM 的合法输入。
- 非法组合显式报错；不能静默过滤后显示“全部通过”。

### 4.2 独立解包实际 INT8 权重

输入为生产格式的 `b_mxfp4 / b_shift2 / b_channel_scale`。直接解包这三者，
不能拿 checkpoint 的 MXFP4 精确反量化当作同语义 kernel 参考。

应对照但不要直接调用生产 `_pack_codes_coalesced` 的逆过程：

```text
packed shape: [E, K/32, ceil(N/8), 128]
每个 (K/32, N/8) 小块可视为 [32 lanes, 4 bytes]
lane = (n % 8) * 4 + thread_in_group
thread_in_group ∈ [0,3]
byte_permutation = [0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15]
lane 中第 j 字节对应 k_pair = byte_permutation[thread_in_group*4+j]
低 nibble 对应 k=32*kb+2*k_pair，高 nibble 对应后一个 k

shift = (b_shift2[e, kb//4, n//8, n%8] >> (2*(kb%4))) & 3
mag = [0,1,2,3,4,6,8,12][code & 7]
signed = -mag if code & 8 else mag
W_int8[e,n,k] = signed * (1 << shift)
```

这张表已经带有 E2M1 的 x2 因子；这里不要再除以 2，最终缩放由生产 channel_scale
承担。解码范围为 [-96,96]，可以用小标量实现验证所有 nibble/shift 组合。

至少准备一组不经过生产 pack 生成的手工 packed 字节 fixture，覆盖 lane 置换、
高低 nibble、正负零和四个 shift。只做“生产 pack→自己的 unpack→原值”会留下
双方同错的可能；该 round-trip 可以补充，不能作为唯一锚点。

### 4.3 GEMM、写回和 reduce

参考从原始 `routing_ids` 建立逐专家分组，不使用生产 `sorted_ids/expert_ids` 作为
唯一真值，以便发现 align 元数据错误。

```text
s = token * topk + slot_in_token
e = routing_ids[token, slot_in_token]
W13 source_row = s // topk
W2  source_row = s
acc_int32[s,n] = sum_k A_q[source_row,k] * W_int8[e,n,k]
v_fp32 = (float32(acc_int32) * A_scale[source_row]) * channel_scale[e,n]
routed[s,n] = BF16_RNE(v_fp32)
output[token,n] = BF16_RNE(FP32 路由加权累加 routed[token*topk+j,n])
```

W13/W2 的 GEMM 输出都按原始 slot 顺序；routing weight 在最终 reduce 应用，
不得提前乘权重或在 W2 和 reduce 重复乘。核实 reduce 的编译 FMA/累加次序后再要求
任意浮点数据逐位一致；先以精确可表示的整数/二次幂数据建立严格锚点。

参考累加可以在小 fixture 使用 CPU INT64；真实形状按活跃专家/输出列分块，用 FP64
计算整数 dot 后模拟 INT32→FP32→两次缩放→BF16。先证明输入范围内不会 INT32 溢出。
不要用未经约束的 FP32/TF32 GEMM冒充整数精确参考，也不要提前把 scale 乘到权重上
再 GEMM，这会改变舍入位置。参考绝不进入计时区，避免一次反量化全 E 个专家到 FP64。

### 4.4 正确性指标与阈值

- 同时记录 finite、max_abs、rel_l2、calc_diff，记录最差专家/行与首个失配坐标。
- `calc_diff = ||x-y||²/(||x||²+||y||²)`，与 rel_l2 不等价；不能照抄 `1e-2/2e-2`。
- 两者全零定义误差 0；参考全零而候选非零必须失败；NaN/Inf 先失败，不能跳过。
- JSON 使用合法有限数字/null 加状态，不输出非标准 JSON 的 NaN/Infinity。
- 同契约、可精确表示的 fixture 要严格通过；随机样本如不一致，先定位整数结果、
  FP32 缩放、BF16 舍入、reduce 的差别，不允许通过放宽阈值掩盖错误。
- 量化质量阈值未校准时标 `uncalibrated`。它与 kernel correctness 是独立字段，
  不能用近似 remap 的误差预算容忍 kernel 算错。
- 保留原有 tile 等价性检查和 `exact_rate_all == 1.0` 语义断言。

P0 出口：手工锚点与 CPU 参考有证据；明确区分 kernel 正确性与量化质量。

## 5. P1：扩展用例与现有测试

先做少量高信号用例，避免 experts×topk×tokens×tile×route 全乘积爆炸。

| 类别 | 用例 | 必须捕获的错误 |
| --- | --- | --- |
| 小规模 GPU smoke | E=8，top-k=3/6，M=1/4，生产 N/K，tile=16x128 | 非零 GEMM、scale、正负值、W13/W2 源行混淆 |
| 路由打乱 | 非排序专家 ID、每个 slot 不同输入/权重；跨 token 重复命中专家 | 按 expert 排序后没有散回原 slot、误用 token 行 |
| 热点与空专家 | 多个 token 都命中同一组 top-k 专家，其他专家空闲 | padding、重复命中流量、空专家输出污染 |
| 边界 | 热点路由下 M=15/16/17、63/64/65、127/128/129 | 每专家 m_e 跨 block_m 边界；按各 tile 对应三点定向测 |
| reduce | 非均匀、非单位和的 routing weights，零权重 slot | 重复乘权重、隐式 renormalize、漏 slot |
| pack 单测 | N=7/8/9/11，K=32/64/128 等合法 pack 参数 | N 尾组和 shift 尾组；不调用不支持该形状的 GEMM |
| 写回完整性 | valid output 预填 sentinel，再检查所有有效输出 | 部分 slot 未写、输出仍沿用前一次调用 |

测试延伸 `test_mxfp4_int8_sm80.py`，每条只守护一个行为。
原 `test_dspark_topk3_cuda_dispatch` 没有验证非零 GEMM 结果；不要只扩充其零数据规模。
正常路由要求 `E >= topk` 且单 token 内专家不重复，跨 token 重复保留。
总 M=0 只做已确认支持的 helper/参数行为检查，不擅自向 CUDA 发空 grid。

建议新增 CLI 契约如下，**这些是待实现参数，不能现在直接运行后宣称命令已有**。
这些轴用于 dsv41f 专家 kernel 的受控测试，不构成其他模型支持承诺：

```text
--experts INT [INT ...]            默认沿用 384
--topk {3,6} [{3,6} ...]           默认沿用 6
--route-pattern {uniform,hot,sparse} [...]  固定 seed；hot 用固定共享专家集合
--seed INT
--check-reference                  correctness 失败时以非零退出，不再测该 case 性能
--cache-mode {warm-fixed,rotating-experts}
--output-jsonl PATH                新格式；旧 --output 的 JSON 用途保持兼容
```

原 `--rotate-experts` 保留为兼容入口。第一批 rotating 模式可以继续只支持 M=1，
其余组合显式拒绝，不必为了泛化重写全部 capture。新参数最终名称如有调整，
必须同步更新本文命令和结果说明。
`--experts/--topk` 必须传到所有 shape、align、tile 校验、reduce 与 rotation 代码，
不要只改生成路由的一处；保持其他脚本引用的 `make_gemm_call` 等接口可用。

## 6. P2：先修正流量，再做计时

### 6.1 字节与工作量

每个 stage 独立计算 weight、shift2、channel_scale 的字节数。W13、W2 的实际体积
不同，不把两者之和用于任一单独 stage，也不把总字节除以 2 估算 W13。

```text
active_experts = unique(routing_ids)
m_e[e] = count(routing_ids == e)
active_weight_bytes(stage) = sum(该 stage 每个活跃专家的三种张量字节)
m_tiles = sum_e ceil(m_e[e]/block_m)
padded_rows = sum_e ceil(m_e[e]/block_m)*block_m
useful_flops(stage) = 2 * (M*topk) * N_stage * K_stage
active_weight_gb_s = active_weight_bytes / p50_us / 1000
useful_tflops = useful_flops / p50_us / 1e6
```

将 `active_weight_gb_s` 标为权重逻辑有效带宽，不称实测 DRAM/HBM 带宽。
如增加 `logical_io_bytes`，必须逐项给公式及读写次数；不完整时该字段置 null。
张量容量/分配体积单列，与活跃专家 footprint 区分；共享 storage 的 view 不重复统计
物理分配，普通 `count_bytes` 只提供逻辑张量大小。

必做一次 CPU 算术验算：固定命中专家，M 从 1 增至 4，active_weight_bytes 应保持不变，
useful_flops 随 slot 数增加；跨 block_m 时 m_tiles 增加。检查 W13+W2 字节严格可加。
旧错误字段若改名或废弃，升级 schema 并说明映射，不让旧结果与新结果默默合并。

bandwidth probe 里的连续 scratch 纯读可保留，命名为“相同逻辑字节量的连续读对照”。
不要称它与生产 GEMM同地址。slab/pattern/decode 三种读对照本轮可只保留原入口，
不强制同时重构；要宣称共享地址对照时，必须让它们显式接收同一批实际张量和 IDs。

### 6.2 计时与 baseline

- 使用 Event 包围 CUDA Graph replay；导入、编译、随机数据、分配、参考计算、日志
  全部在计时区外，capture 前完成 warmup 并同步。
- 同一 case 使用相同的权重、A、routing IDs/weights 和 seed 比较 tile。
  baseline 记录生产 selector 当前选出的 tile，而不是所有 M 固定用 16x128。
- tile 变化可能改变 sorted metadata，须从同一 routing IDs 重新 align，不能重用错误块长。
- warm-fixed 重放相同工作集；rotating-experts 记录访问序列及其活跃权重并集。
  rotation 周期耗时归一化时，字节分子也按该周期每次调用的同口径工作量归一化。
- 轮换专家不自动等于 cold-cache；记录 working_set_bytes/rotation_period。
- baseline/candidate 交替 A/B/B/A，至少 3 个独立 round，保留每个样本；
  p90 是 Graph 批均摊延迟分位，不标作请求尾延迟。
- 不根据一次 p50 或 `GB/s >= baseline` 判定通过。先测 baseline 对 baseline 的波动；
  落在波动范围内的差别标 inconclusive，不扩测直到出现“想要的收益”。
- 保留 eager 作为诊断可以，但不要用 eager 时间排序生产 Graph 候选。

### 6.3 JSONL 最小字段

每个 case/stage/variant 一行，至少包含：

```text
schema_version, run_id, case_id, stage, variant, baseline_id
git_head, dirty_diff_hash, command, seed, gpu_uuid, pci_bus_id
torch/cuda/triton/driver_version, relevant_env
M, N, K, E, topk, block_m, block_n, source_rows_are_slots
route_pattern, route_hash, active_experts, expert_token_histogram
padded_rows, m_tiles, cache_mode, rotation_period, working_set_bytes
correctness_status, reference_kind, threshold_id
finite, max_abs, rel_l2, calc_diff, worst_expert, first_mismatch
active_weight_bytes, logical_io_bytes, useful_flops
round, graph_calls, samples_us, p50_us, p90_us, baseline_ratio
measurement_status, skip_or_failure_reason
```

输入构造、dtype/shape/seed 之外如还依赖路由文件或真实权重片段，记录它们的 hash。
没有模型 checkpoint 的纯 synthetic 验收不能报告“模型精度通过”。
结束时原子写出 summary，保留失败 case；异常退出不能留下一个看似全 PASS 的汇总。

## 7. 验证顺序与可执行命令模板

先创建一个唯一的 `artifacts/moe-grouped-gemm-harness-<时间戳>/` 目录存放
环境、命令、CPU/GPU 日志、JSONL、diff 和 summary。不要覆盖以前结果。

### 7.1 CPU 与静态检查

根据已完成参数先检查 `--help`，然后执行：

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/kernels/quantization/test_mxfp4_int8_sm80.py -q
.venv/bin/python -m compileall -q benchmarks/kernels/deepseek_v41_moe_harness.py benchmarks/kernels/deepseek_v41_mxfp4_int8_sm80.py benchmarks/kernels/deepseek_v41_moe_bandwidth_probe.py
git diff --check
```

运行项目规定的 pre-commit 文件级检查，依赖通过 `uv` 管理。记录通过/失败/skipped
数量；若 CUDA 隐藏后发生 import/平台探测错误，先定位现有测试环境约束，不能改生产
初始化来迎合测试。CPU 模式不通过不等于 CUDA kernel 已失败，反之也一样。
离线 helper 应能在无 CUDA 条件下导入并执行纯指标/解包自检；将 pack/指标边界验证
纳入现有测试或明确的 self-check，不只做 compileall。

### 7.2 选择设备

只在确认空闲并具有足够显存的 SM80/CMP 设备上进行 isolated GPU 测试。
通过 UUID 绑定，避免 PCI/NVML/CUDA 索引混淆；不要靠停服务获取空闲设备。

```bash
# 将 GPU-<UUID> 替换成已经核实为空闲的实际 UUID。
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-<UUID>
```

先估算 packed 权重、scale、输出、参考解包和 Graph/workspace 峰值，再分配 E=384。
本任务不需要完整模型加载。共享忙卡的延迟可以记作诊断，不能纳入性能 baseline。

### 7.3 小规模 GPU 正确性

下面新增 CLI 实现完成后才执行；输出路径的 `<run>` 用实际唯一目录替换。

```bash
timeout 15m .venv/bin/python -m pytest tests/kernels/quantization/test_mxfp4_int8_sm80.py -q
timeout 15m .venv/bin/python benchmarks/kernels/deepseek_v41_mxfp4_int8_sm80.py --experts 8 --topk 3 6 --tokens 1 4 --route-pattern uniform hot --seed 4242 --block-m 16 --block-n 128 --check-reference --cache-mode warm-fixed --warmup 3 --repeats 10 --graph-calls 4 --output-jsonl artifacts/<run>/gpu-smoke.jsonl
```

这个少量 samples 的 smoke 用于发现错误，不作为稳定性能结论。
再按 P1 对应的 block_m 逐组跑三个边界点，不对所有 tile/所有 M 做大乘积。
只要发现数值错误就停止该组性能测量，保存最小失败输入与索引信息，回到参考契约定位。

### 7.4 生产形状基线

smoke 和边界通过、显存预算允许后，先跑 top-k6/M1/E384 的固定与轮换专家两组。
以下给固定工作集模板；轮换组仅更换 cache-mode 和输出名：

```bash
timeout 20m .venv/bin/python benchmarks/kernels/deepseek_v41_mxfp4_int8_sm80.py --experts 384 --topk 6 --tokens 1 --route-pattern uniform --seed 4242 --block-m 16 --block-n 128 --check-reference --cache-mode warm-fixed --warmup 20 --repeats 100 --graph-calls 40 --output-jsonl artifacts/<run>/production-m1-warm.jsonl
```

每组至少记录 3 round，使用独立输出文件或实现显式 round 字段；不要覆盖前一轮。
随后做 top-k3 和一个从当前运行配置/既有记录确定的代表性多 token case，
记录为什么选它，不假设旧服务参数仍然生效。计时不含 FP64 参考成本。
本阶段先建立生产 baseline，无须扫 16x64 去寻找收益，更不能据旧 microbenchmark
结果把默认 tile 改掉。

## 8. 停止、回退和异常分类

| 情况 | 必须采取的动作 |
| --- | --- |
| 没有空闲 GPU/预算不足 | 完成 CPU 与代码部分，记录 GPU 未验证；不杀服务、不自动换后端 |
| CUDA op 缺失或动态库 ABI 不符 | 记录解释器、扩展路径和错误；定位构建状态，不能修改 dispatch 绕过检查 |
| 需要重编译/替换服务正在使用的扩展 | 本轮暂停该验证并报告依赖；本任务没有授权在线替换服务构件 |
| kernel 结果不符 | 保存 seed/case/路由/首个失配，停止计时；先区分参考、pack、索引、缩放/舍入 |
| OOM | 保存估算与实际峰值，只释放本测试进程资源；降低 E/参考 chunk 后仍需标注覆盖缩减 |
| IMA、device assert、超时 | 结束本测试进程，保留错误，不在已污染 CUDA context 中继续采样；不 reset GPU |
| baseline 自比波动过大/存在竞争进程 | 标 inconclusive，查设备占用；不通过加大 sweep 粉饰噪声 |
| HEAD 或脏文件执行中被其他人改变 | 重新确认变更归属和结果 provenance，避免混用不同代码的样本 |

命令 timeout 只约束本测试命令。不要用宽泛 pkill，不停止未知 PID。
所有失败应有明确阶段和原因；dependency_unavailable、numerical_failure、
resource_unavailable、measurement_inconclusive 分开报告。

## 9. 交付验收清单

最终提交给用户一份简短总结及下列可定位文件：

1. **代码 diff**：只涉及上述设施与必要测试，列出每个文件的目的；不改变生产行为。
   单列范围审计：生产模型/后端分派、公共默认值、全局配置均未修改，其他模型无新依赖。
2. **命令与环境记录**：HEAD/脏 diff、GPU UUID、版本、seed、完整命令、退出码。
3. **正确性证据**：手工 packed 锚点、非零 W13/W2/reduce、路由边界、scale 与尾组测试。
4. **机器可读结果**：JSONL schema、原始样本、误差、公式、失败或跳过原因。
5. **baseline 表**：stage、M/E/topk、route/cache、tile、p50/p90、有效字节、useful FLOPS；
   明确哪些是合成权重结果、哪些性能数字受噪声影响。
6. **执行报告**：`docs/design/vllm170-sm80-moe-harness-implementation-<日期>.md`，
   包含已完成/未完成、实测结论和最多三项有证据支持的下一步 kernel 假设。

验收时逐项回答：

- 如果所有 tile 一起算错，独立参考能否发现？
- 如果 W2 误用 token 行，或路由 scatter/reduce 错位，非零测试能否发现？
- 相同专家跨 token 复用时，字节数是否仍合理？
- 固定专家与轮换专家的计时是否被明确区分？
- 一条失败/跳过/未校准结果会不会被汇总成 PASS？
- 是否严格限于 dsv41f SM80 grouped GEMM，没有改变其他模型分派、默认参数或执行路径？

只有证据满足的项目才写“通过”。本轮完成设施与基线之后即停止；
完整 MoE、模型质量评估与服务 A/B 是后续 kernel 候选晋升的必要条件，不能由
此次隔离基准替代，也不在本任务中擅自启动。
