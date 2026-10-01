# TileKernels SwiGLU+INT8：DSV41F 生产接入

本文保留融合初次接入轮次的配置与测量。用户随后放弃 FP8 dense 的 TileLang
路线；M6 W2 接入及 packed-word 后续优化见
[最新 MoE 优化报告](vllm170-dsv41f-target-moe-w2-word-load-opt-2026-10-01.md)。

日期：2026-10-01。承接
[离线原型轮次](vllm170-tilekernels-swiglu-int8-round-2026-10-01.md)与
[吸收复核](vllm170-tilekernels-absorption-review-2026-10-01.md)。
基于 HEAD `ae32948b167dbbb1cc87dbe4f965919f9705dc1c` 的工作区代码，
TileLang 0.1.15、Torch 2.13.0+cu130、Triton 3.7.1。

本轮将固定 I2304 的 activation+A2 量化融合移入运行模块，沿用现有
`additional_config.dsv41_sm80_moe_tilelang` 显式开关。**已开启生产 PP5、完成模型加载、
Graph 捕获和 HTTP 回归，服务保持 active/running，health/models 均 200。**
247 项测试通过，3 组探针 token 与基线逐个一致；端到端吞吐变化仍不足以越过基线噪声。

## 生产形状与接入边界

生产为 `dsv41f-manual.service`、26661 端口、PP5/TP1，启动请求 FULL CUDA Graph，
使用 `/media/z/data6/models/DeepSeek-V4.1-Flash`。保留原 q5 greedy DSpark、
KV cache、CED、并行划分与设备选择，只将原 JSON 开关 false 改为 true。
开关在通用代码中仍默认关闭。

与离线 M=1 原型相比，这次接入补齐两个决定可达性与数值语义的条件：

| 条件 | 实际值 | 本轮处理 |
| --- | --- | --- |
| target 验证行数 | q5 对应 M=6、top-k6，即 36 routed rows | 融合独立于 M=1 GEMM 门禁，1–128 routed rows 可用 |
| checkpoint `text_config.swiglu_limit` | 10.0 | gate 仅截上界，up 截双边，再按 CUDA BF16 舍入执行 SiLU/乘法 |
| draft 专家 | E128/top-k3 | 保留原 activation+quant 与 CUTLASS |
| 未支持的 clamp/大批次 | 其他 clamp 或 >128 routed rows | 保留原 activation+quant |

模型父类从 `config.swiglu_limit` 传递到 `FusedMoE`，再由
`ApplyMoEActivationConfig.from_configs` 合并 quant/model 配置。
没有把 multimodal wrapper 根配置误当作 text 配置，也没有假定 checkpoint 不使用 clamp。
grouped GEMM 候选仍只用于原先满足门禁的 M=1；M=6 的两个 GEMM 都是 CUTLASS。

## 实现与 TileKernels 吸收点

运行模块：
[dsv41_sm80_activation.py](../../vllm/model_executor/layers/fused_moe/experts/dsv41_sm80_activation.py)。
调用入口：
[mxfp4_int8_sm80.py](../../vllm/model_executor/layers/fused_moe/experts/mxfp4_int8_sm80.py)。
benchmark 从运行模块导入 builder，生产模块不依赖 benchmark 或 autotuner。

- 吸收一次读取中完成 activation、amax、scale、量化的组织方法。
- 吸收 producer 直接写消费者所需输出的接口：q bytes 与 FP32 scale 均为调用方 buffer。
- 吸收原始 bytes/scale 校验、快速数学必须有对应数值证据的验证方法。

冻结调度为每行一个 CTA、256 threads、block K4096，固定 I2304，行数为动态参数。
未采用 TileKernels 的 FP8/packed FP4 输出、pow2 scale、1024-thread persistent grid。
当前输出仍为数值 INT8、整行 FP32 scale，amax/127、RNE、[-127,127]、零行 scale=1。

保留 SiLU 结果 BF16 舍入，再乘 up 并再次 BF16 舍入；amax 基于第二次舍入结果。
快速指数使用 `__expf`，SiLU 的除法与量化两处除法均显式使用 `div.full.f32`。
全局 fast math 关闭。此前 `rcp.approx.ftz` 的 subnormal 差异与 `__fdividef`
在极负 gate 的差异不再出现在冻结版本中。

加载时按 worker/device/clamp 缓存编译并预热，forward/Graph capture 不再编译。
复用 workspace13 中 BF16 intermediate 的 storage：前 `rows*2304` bytes 写 INT8，
随后写 `rows` 个 FP32 scale；两者不重叠，scale offset 按四字节对齐。
Gate/up 与 W2 routed slots 各在独立区域，不新增持久 workspace，也不提前乘 routing weight。

## 正确性与检查

在独立空闲 CMP170HX（SM80）上运行完整现有测试文件：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python -m pytest tests/kernels/quantization/test_mxfp4_int8_sm80.py -q
```

加载前 **247 passed，99.75s**；设备/同步 API 清理后的最终源码再次
**247 passed，100.40s**。包括：

- fast/standard、clamp=None/10 两两组合，全部 65280 个有限 BF16 gate 的中间舍入校验。
- 零值、极端值、subnormal、RNE ties，q bytes 与 FP32 scale bits 对现有 CUDA+Triton 一致。
- 1/6/18/36/40/128 rows，调用方 buffer、Graph 重放、变化输入及输出毒化。
- 实际 `Experts.apply` M1/M2/M6、动态路由、全部 routed slots 与最终 BF16 output 逐位一致。
- workspace13/output 别名；M22/top-k3/不支持 clamp 回退；forward/capture 禁止再次编译。
- 默认关闭、身份、设备、并行配置门禁。

scoped ruff-check、ruff-format、mypy-3.12 与 `git diff --check` 通过。
完整 pre-commit 曾指出新增主机侧 `torch.cuda.device/synchronize` 调用，已按仓库规范
改为 `torch.accelerator.device_index/synchronize`；六个变更文件的全部适用 hook
（含 mypy-3.10、SPDX、forbidden imports、CUDA API、markdownlint）及 mypy-3.12 均通过。
部署时源码另存 `deployed-source/`，最终源码存 `source/`；seal 验证两者仅有这两项
设备/同步 API 替换，内核公式、builder 与 forward 逐字一致。最终源码的 GPU 回归另存
`validation/pytest-final.log`。已加载服务无需为这项加载期设备上下文替换再次捕获 Graph。
本轮没有新增 sanitizer 通过声明；先前 CMP170HX 的 GPU debugging 限制仍不能视为 memcheck 证据。

## 本轮 M=6、clamp=10 GPU 测量

使用真实运行入口开关 false/true，没有 monkeypatch activation 或 quant。
合成 E384 packed 权重、非单位 channel scale、非均匀 routing weights，
完整 apply 包含 A1 quant、metadata、W13、activation+A2、W2、reduce。
双方 M=6 均使用 CUTLASS GEMM，差异归于融合与 workspace 输出路径。
Graph 输出先毒化再验证，固定组额外完成三次输入/路由变化校验。

| 边界 | 原路径 µs | 融合 µs | 延迟变化 | 基线轮间 spread | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| 36 rows activation+A2 | 4.6720 | 2.4800 | -46.92% | 0.00%（舍入后） | 可测 |
| 完整 apply，fixed | 796.0832 | 792.7808 | -0.415% | 0.01% | 可测 |
| 完整 apply，rotating | 799.3600 | 800.4096 | +0.131% | 0.99% | inconclusive |

以上均 q/scale 或 routed/final output 逐位一致，均为 **CUDA Event + CUDA Graph**，
不是 CUPTI。完整 apply 每 Graph 40 次调用、3 轮、每轮 15 个样本，交替测量前预热 500ms；
阶段测量每 Graph 64 次调用、3 轮、每轮 25 个样本、预热 1000ms。
权重/随机生成/编译/分配不在计时区。rotating 在 40 组不同 workspace/专家路由间轮换，
不是每次物理 flush 全部 L2 的冷缓存测量。

M=6 的完整专家段收益小于前轮 M=1 结果。不能把阶段 -46.92% 或 M=1 的收益
换成线上 tokens/s 收益；轮换结果也没有证明稳定整段提升。

## 服务切换与 HTTP 对照

启动前 `/health` 与 `/v1/models` 均 200，running/waiting 均零。
原 MainPID 150939；保存原 transient unit、启动配置与关闭开关的回退脚本。
首次复用旧启动器时，其 Bash JSON 默认值写法在传参后多加 `}`，CLI 在加载前退出。
已修复本轮启动/回退副本的默认值赋值方式，再用相同配置启动；未触发模型内核错误。
修正后 MainPID 为 1183086，日志确认 additional_config 开关 true。

基线已在原服务上完成：3 个 1024→64 greedy prompt 各重复一次；
另用固定 seeds 做两轮各 3 个 1024→128、concurrency1 的 decode 测量（每轮预热1次）。
候选就绪后使用完全相同命令与 seeds，计时窗未打开 phase profiling。

基线两轮同 seed 的 decode throughput 变化为 +5.883% / +0.710% / -0.337%；
最差同 seed 自比波动约 5.88%。本轮仅进行一次完整模型重新加载及小规模 HTTP 对照，
不通过多次昂贵重启追逐低于这一级噪声的变化。

修正后启动器记录 380s 就绪轮询时间。target 与各 rank-local draft Graph 捕获完成；
运行时 attention 能力将 FULL 配置调整为 FULL_DECODE_ONLY，profile 中 target decode
执行标记为 `full`。没有为计时关闭 Graph。

验收结果：

- health/models 均 200，模型名 `dsv41f`；MainPID 1183086、五个 PP worker 保持运行。
- 原/新 transient unit 的有效配置逐行比较，唯一差异为上述 JSON 布尔值。
- 3 组探针各重复一次，`repeat_same=true`，与基线的 generated token IDs 全部一致。
- 两轮 decode 均无 HTTP 错误；每个请求 usage=1024/128、steps=[68,54,73]，
  accept_len=[1.882353,2.370370,1.753425]，与基线逐请求一致。
- sweep JSON 没有保留 token IDs，不能仅凭相同 steps/accept_len 声称其输出逐 token 一致。
  精确 token 结论来自 probe；这是小规模数值回归，不是通用模型质量评测。

| 同 seed 对照 | baseline A tok/s | on A tok/s | baseline B tok/s | on B tok/s |
| --- | ---: | ---: | ---: | ---: |
| 2026101079 | 26.8472 | 28.3867 | 28.4266 | 28.4523 |
| 2026102056 | 34.3500 | 35.8767 | 34.5939 | 35.1364 |
| 2026103033 | 25.1294 | 26.5290 | 25.0446 | 25.9691 |

A/B 两轮配对提升中位分别为 +5.57% / +1.57%；六组配对中位 +4.07%。
这些均未越过基线最大同 seed 自比变化 5.88%，且没有跨重启消除状态差异，
因此 **端到端性能仍为 inconclusive**。生产开启基于本轮用户明确要求及正确性验收，
不代表已证明 tokens/s 加速，也没有将默认关闭的开关改为全局默认启用。

计时后单独用一个 262→16 请求开启短 phase profile 窗口，结束后移除控制文件。
真实 checkpoint 的 prefill 尾步记录了 20 个 `activation_w2_quant` 调用：
PP0 8 个、PP1 8 个、PP2 4 个，对应 layer0–19、eager、外层 token_count=6。
layer20–39 保留原 activation：CED 从 layer20 开始重放 sliding_window=128 的尾部，
实际 MoE 输入不是外层 6 行；128×top-k6 已超过 128 routed rows 的融合门禁。
因此不能将外层 profile token_count 不加区分地解释为所有层的 MoE 行数。

profile 同时确认 target decode 为 `full`、token_count=6；Graph 重放不会进入 Python
内部 stage context，故没有逐 kernel 的 decode hit 计数，不把这次 eager hit 当成
Graph 内节点追踪。Graph 正确性来自 GPU apply 测试与实际 HTTP 回归，
decode 选择由已加载的门禁与形状推导。profile 的高开销阶段时间不参与性能表。

最终 profiling control 不存在，running/waiting 均零，测试卡回到 6MiB 空闲状态。

## 后续方向与证据

本轮已吸收融合组织、调用方输出 buffer 和数值校验方法。
后续优先研究真实 M=6 工作集的权重读取/GEMM，而不是继续扩大 activation 的百分比：
本轮整段约 0.8ms，activation+A2 原阶段仅约 4.7µs。
FP8 dense 主线仍需以 TileKernels `cast_back + common.py` 的 scale/layout 作为参考，
单独验证 MXFP8 W8A16 与 E8M0 0/255 的契约；这次接入不代表 dense linear 已改写。

[本轮证据目录](../../artifacts/tilekernels-swiglu-production-20261001/)包含
冻结运行源码、SHA256、tracked source patch、完整测试日志、两组 microbenchmark、
加载前服务配置、启动/回退脚本和 HTTP 原始结果。

M=6 测量复现：

```bash
PYTHONPATH=/mnt/data/apps/vllma100/vllm170 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python artifacts/tilekernels-swiglu-production-20261001/bench_apply_m6.py
```

阶段测量将末尾脚本换为同目录 `bench_activation36.py`。运行前确认测试 UUID 空闲。
回退使用同目录 `service/restore-before.sh`（关闭开关并完整加载），不能对已捕获 Graph 热切换。
