# TileKernels M6 W2：运行层接入与生产复验

本文保留 M6 W2 初次接入轮次的配置与测量；后续 packed-word 优化及其生产验证见
[最新 MoE 优化报告](vllm170-dsv41f-target-moe-w2-word-load-opt-2026-10-01.md)。

日期：2026-10-01。承接
[M6 grouped GEMM 离线轮次](vllm170-tilekernels-m6-gemm-round-2026-10-01.md)。
用户已明确放弃 FP8 dense 的 TileLang 路线；本轮只推进 MoE W2。
基于 vLLM HEAD `ae32948b167dbbb1cc87dbe4f965919f9705dc1c` 的工作区修改，
TileLang 0.1.15、Torch 2.13.0+cu130、Triton 3.7.1。

**M6 W2/BN128 已接入生产 PP5，完整现有测试文件 247 项通过；
六次 HTTP 请求的 generated token IDs 与切换前完全一致，服务 health=200。**
真实权重抽样下完整 `Experts.apply` 的耗时降低 0.98%–1.67%，
此数字不是整个模型的 tokens/s 收益；本轮 HTTP 性能仍为 inconclusive。

## 运行层边界

[dsv41_sm80_tilelang.py](../../vllm/model_executor/layers/fused_moe/experts/dsv41_sm80_tilelang.py)
新增独立 M6 W2 编译/预热入口；
[mxfp4_int8_sm80.py](../../vllm/model_executor/layers/fused_moe/experts/mxfp4_int8_sm80.py)
只在 M=6、top-k6、local/global E384、BM16/BN128 时选择它。
沿用原来默认关闭的 `additional_config.dsv41_sm80_moe_tilelang` 开关，以及
DSV41F/H5120/I2304/BF16/SM80/单 TP/EP 的加载门禁。
draft E128/top-k3、其他 M、BN64 保留 CUTLASS。

冻结配置为 BM16/BN128/BK128、128 threads、pipeline=3。
沿用 PRMT 解码、动态 align metadata 和明确的 padding zero-fill。
TileLang 0.1.15 的 packed-byte 解码继续使用 `T.unroll(8)`，
避免 vectorizer 将每个 packed 输入广播成八个相同 lane。
M6 最多 36 个专家 tile，sorted IDs 容量 576；不复用 M1 固定 metadata。
编译按设备缓存，加载时验证 packed/shift/channel-scale 布局并预热；
forward/Graph capture 不编译、不调参，不新增持久 workspace。

本轮吸收离线分阶段验证与调用方输出 buffer 的组织方式，只推广 W2 的确定配置。
M6 W13 继续 CUTLASS：上轮 hot 路由退化最高 26.98%，不能推广整个 grouped GEMM。
原 M1 GEMM 与已部署的 SwiGLU+INT8 融合保持其选择条件与数值语义。

## checkpoint 抽样与计量边界

使用独立空闲 CMP170HX：UUID `GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216`，
PCI `02:00.0`，SM80；未在生产五张 PP 卡上运行 microbenchmark。
从 `/media/z/data6/models/DeepSeek-V4.1-Flash` 读取 layer20 的专家 137–172，
每个专家的 W1/W3/W2 packed MXFP4 权重与 E8M0 scale，共 216 个 payload、
676,823,040 bytes。保存逐 tensor 的 shard、绝对偏移、dtype、shape、SHA256。
复用生产 remap：headroom_bits=3、reuse_weight_storage=True；W13 按 W1/W3 拼接。
其余 E384 槽显式填零，实验路由仅选择这 36 个真实专家。

hidden 为 seed942 合成 BF16，routing weights 非均匀，clamp=10；
**没有捕获真实线上 MoE hidden、专家 ID 或 routing weights**。
已有 rowdump 的 attention top-k 不是 MoE 专家路由，不能代替它。

两组都使用已部署的运行层 activation fusion，只有 W2 不同：
基线清空实例的 M6 W2 callable，候选走真实运行层 callable。
分别断言 eager/capture host 调用计数，不把计数当作 Graph replay kernel hit 数。
计时包括 A1 quant、routing cast/align、W13、activation+A2、W2 与 reduce 的
完整 `Experts.apply` GPU 工作，编译/分配/参考计算均在计时外。
采用 **CUDA Event + CUDA Graph**，不是 CUPTI；
三轮交错 A/B，各 15 个 samples，每个 Graph 40 次 apply，切臂前预热 500ms。

| 路由 | CUTLASS W2 完整路径 µs | 运行层 TL W2 完整路径 µs | 耗时降低 | 基线轮间波动 |
| --- | ---: | ---: | ---: | ---: |
| hot | 174.3104 | 171.3920 | 1.67% | 0.03% |
| spread | 796.2112 | 785.2544 | 1.38% | 0.02% |
| rotating40 | 796.6720 | 788.8640 | 0.98% | 0.59% |

hot 为六个 token 共享六个专家，spread 为 36 个不同专家。
rotating40 改变路由顺序与 workspace，但始终使用上述同一组 36 个真实专家，
不能称作 40 层不同权重工作集或物理 cold-L2 flush。
hot/spread 最终输出及所有 W2 routed slots 均逐 bit 一致，捕获后两次改变
hidden/专家 ID 并毒化输出，Graph 重放仍一致；rotating40 也逐 bit 一致。
两组 host W2 调用计数 hot/spread 为 49/49，rotating40 为 200/200。
rotating 的收益较小，不能外推为稳定端到端提升。

复现命令：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python -m benchmarks.kernels.deepseek_v41_moe_tilelang_m6 \
  --runtime-apply \
  --sampled-checkpoint /media/z/data6/models/DeepSeek-V4.1-Flash \
  --out-dir artifacts/tilekernels-m6-w2-runtime-20261001/checkpoint36
```

第一份 synthetic diagnostic 运行期间驱动文件发生过后续编辑，已注明不能用作
本轮晋级依据。上表只引用随后冻结源码的 checkpoint36 运行，
脚本 SHA256 与 `source-manifest.json` 已核对一致。

## 正确性与静态检查

扩展现有
[test_mxfp4_int8_sm80.py](../../tests/kernels/quantization/test_mxfp4_int8_sm80.py)，
未添加一次性 benchmark 到 tests。
GPU suite **247 passed，101.34s**，聚焦接入检查 **6 passed**。

- M6/top-k6 确认进入独立 W2；M7/top-k3/BN64 回退。
- M6 不支持的 activation clamp 仍可使用数值独立的 W2 分支，activation 按原路径执行。
- clamp=None/10、fast/standard 环境下比较 routed slots 与最终 BF16 output。
- 捕获 hot 路由后切到 36 个不同专家，更改 hidden 并毒化输出，重放逐 bit 一致。
- 复用 output/workspace13 storage；forward/capture 禁止再次编译。
- 模型/设备/配置门禁关闭时不准备新 W2。

四个变更 Python 文件的全部适用 pre-commit hooks 与 mypy-3.12 通过，
`git diff --check` 通过。测试与候选实现的 SHA256、部署源码副本、日志留档。
本轮没有新增 sanitizer 或 CUPTI 通过声明。

## 生产切换与回退

服务为 `dsv41f-manual.service`、26661 端口、PP5/TP1、q5 greedy DSpark。
保持已有设备、模型、CED、KV cache、Graph 与 additional_config 参数。
切换前 MainPID 1183086，五个 PP worker 占用既有五张 GPU；health=200，
running/waiting=0。已保存 transient unit、进程/GPU 状态及日志偏移。
切换前固定三个 1024→128 greedy token prompt，各重复两次，token IDs 一致。

重载成功，启动器记录 370s 就绪轮询时间。新 MainPID 1686211，
PP0–4 worker 为 1688944/1688945/1688946/1688947/1688948，仍占用原五张 GPU。
切换前后 transient unit 全文逐行相同，本轮没有修改 serving 参数。
日志确认 target 仍使用 `MarlinMxfp8LinearKernel`，并明确启用 M6 W2/BN128。
target 和各 rank-local draft Graph 捕获完成；attention 能力仍将请求的 FULL
调整为 FULL_DECODE_ONLY，与此前行为一致。

切换后复用原固定 prompts/seeds，预热一次后运行两轮各三次 1024→128 请求。
六对请求的 prompt hash、generated token IDs、usage 与 stream chunk token-count
序列均相同；切换前后各自两轮的 token IDs 也相同。
这是小规模模型数值回归，不是完整通用模型质量评测。
decode 吞吐按首个内容 chunk 到最后一个 chunk 的时间计量，扣除首 chunk 的所有 token。
计时窗未开启 phase profiling；不将 TTFT、cached prefill 或 startup 时间算入 decode。

| 固定 prompt | before A tok/s | after A tok/s | before B tok/s | after B tok/s |
| --- | ---: | ---: | ---: | ---: |
| 0 | 32.9657 | 32.5414 | 32.2035 | 32.8620 |
| 1 | 31.5854 | 32.5447 | 32.4748 | 32.4219 |
| 2 | 65.7353 | 72.3056 | 68.5994 | 70.1602 |

六组配对 throughput 变化中位数 **+2.16%**，
基线同 prompt 两轮最大变化 **4.36%**，且只进行了单次模型重载。
**端到端性能为 inconclusive**，不能用 prompt2 第一轮的 +9.99% 声称整模型加速。
保留本轮生产接入基于用户先前的部署授权、窄分支 GPU 验证和 HTTP 精确回归，
没有将通用开关改成全局默认启用。

本轮未增加 Graph replay 内的逐 kernel 计数或另开高开销 profiler。
q5 target 的 M6 选择由运行层门禁、已加载日志及形状推导，
不把 host 编译/预热日志当作 decode GPU kernel hit 计数。
CUDA Graph 的动态输入正确性由现有 GPU suite 和真实权重抽样验证支持。
先前观察到的 CED prefill 后半层实际 M128 仍按原条件回退，不将外层 M6 套用到所有层。

最终 active/running、health/models=200、running/waiting=0；profile 控制文件不存在。

回退脚本为
`artifacts/tilekernels-m6-w2-runtime-20261001/service/rollback-w2.sh`：
先核对当前两个运行文件仍匹配本轮 SHA256，再仅还原本轮修改前的运行文件并重载。
不关闭已生产部署的 SwiGLU fusion；后续修改导致 hash 不匹配时脚本停止，避免覆盖新工作。
测试/benchmark 不属于紧急服务回退脚本的改写范围。

## 证据

[本轮证据目录](../../artifacts/tilekernels-m6-w2-runtime-20261001/)保存
`checkpoint36/apply-ab.json` 原始 samples、checkpoint provenance、
`source-manifest.json`、冻结部署源码、生成 CUDA、完整 GPU 测试/静态检查日志、
原服务配置、重载/回退脚本及 HTTP 请求的 generated token IDs。
FP8 dense 撤销记录与实验源码仅保留在
`artifacts/tilekernels-fp8-dense-20261001/ABANDONED.md`。
