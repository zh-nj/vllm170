# TileKernels 下一轮：M=6 grouped GEMM 与 MXFP8 dense 契约

日期：2026-10-01。承接
[SwiGLU+INT8 生产接入](vllm170-tilekernels-swiglu-production-2026-10-01.md)。
后续更新：用户已放弃 FP8 dense 的 TileLang 路线；下文 dense 部分仅为历史研究。
M6 W2 的真实权重抽样与运行层接入见
[后续轮次](vllm170-tilekernels-m6-w2-runtime-2026-10-01.md)。
vLLM HEAD `ae32948b167dbbb1cc87dbe4f965919f9705dc1c`；
TileKernels HEAD `66258df6175d2f630ffecb04c5ab66bff8a2ae6a`；
TileLang 0.1.15、Torch 2.13.0+cu130、Triton 3.7.1。

**本轮完成 M=6 的阶段与完整专家路径验证。W2 BN128 是可继续吸收的候选，
完整路径耗时降低 0.85%–2.22%，输出逐位一致；W13 不能直接扩展 M=1 分支。**
新候选仍为离线实现，未修改生产运行模块、重启模型或进行本轮 HTTP 性能 A/B。
已接入的 SwiGLU+INT8 生产服务保持 active，health 200。

## 测量边界与复现

使用空闲 CMP170HX：UUID `GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216`，
PCI `02:00.0`，SM80；与生产 PP5 的五张卡分开。
计时为 **CUDA Event + CUDA Graph**，不是 CUPTI。
每个结果包含三轮交错 A/B，原始 samples、p50/p90、基线自身波动保留在 JSON。
编译、随机数、分配与参考计算均在计时外。

驱动：
[deepseek_v41_moe_tilelang_m6.py](../../benchmarks/kernels/deepseek_v41_moe_tilelang_m6.py)。
复用现有通用 TileLang PRMT grouped-GEMM builder 与 MoE harness。
只考察 BM16/BK128/128 threads，BN64/128；W13 pipeline=2、W2=3。
CUTLASS 使用生产默认 BN128，未用旧 M=1 的 BN64 数字作基线。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python -m benchmarks.kernels.deepseek_v41_moe_tilelang_m6 \
  --out-dir artifacts/tilekernels-m6-gemm-20261001

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python -m benchmarks.kernels.deepseek_v41_moe_tilelang_m6 \
  --apply-only --out-dir artifacts/tilekernels-m6-gemm-20261001
```

E384、M6、top-k6，共 36 routed rows。W13 为 N4608/K5120，
W2 为 N5120/K2304。权重为 seed940/941 的合成 packed MXFP4/shift2，
非均匀 channel scale；阶段激活为随机 INT8 和非均匀 per-row FP32 scale。
不是从生产 checkpoint 抽取的专家权重或真实路由分布。

路由覆盖：

- hot：六个 token 共享同一组六个专家，六个 active experts。
- spread：36 个不同专家，36 个 active experts。
- rotating：完整路径连续 40 个不同 workspace 的调用，每步专家编号偏移 7，
  每步 36 个 active experts。工作集超过 L2；没有强制 cold-L2 flush。

## 单阶段结果

阶段计时包含一个 GEMM，不包含 align、activation 或 reduce；
两组使用相同预计算 metadata。Graph 每次包含 16 次调用，每轮九个 samples。
每种候选先与 FP64 整数点积/BF16 epilogue oracle 逐位一致，再计时。
Graph 中更改激活与 expert IDs 两次，毒化输出后与 CUTLASS 逐位核对。

| 阶段 | 路由 | TL BN | CUTLASS µs | TileLang µs | 耗时变化 | TL useful TOPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| W13 | hot | 64 | 94.912 | 99.520 | +4.86% | 17.07 |
| W13 | hot | 128 | 93.696 | 118.976 | +26.98% | 14.28 |
| W13 | spread | 64 | 503.616 | 502.464 | -0.23%，未越过波动 | 3.38 |
| W13 | spread | 128 | 503.616 | 484.800 | -3.74% | 3.50 |
| W2 | hot | 64 | 59.264 | 56.256 | -5.08% | 15.10 |
| W2 | hot | 128 | 59.136 | 58.496 | -1.08% | 14.52 |
| W2 | spread | 64 | 261.056 | 269.312 | +3.16% | 3.15 |
| W2 | spread | 128 | 261.952 | 257.600 | -1.66% | 3.30 |

useful TOPS = `2 * 36 * N * K / latency_us / 1e6`，
不把 BM16 padding 的无效行计入有效计算量。JSON 另存每个 active expert 的
packed weight/shift/channel-scale 逻辑字节除以时间：605–891 GB/s。
这些是逻辑读取量，不能解释为测得的 DRAM 带宽或每个 CTA 的总读取量。

W13 BN128 在两端路由间从退化 26.98% 变为收益 3.74%。
W2 BN64 同样发生方向反转。没有证据支持按 M=6 整体替换两个 GEMM，
也没有验证基于 active-expert 数量的新调度阈值。

## 完整 Experts.apply：只替换 W2 BN128

离线进程使用真实 `Mxfp4Int8Sm80Experts.apply`，两组均启用已生产接入的
clamp10 SwiGLU+INT8；只对候选实例的 W2 routed-output 地址派发 TileLang。
W13、align、量化、routing cast 与 reduce 都保持相同。
计时范围包含完整 apply 的 GPU 工作，每个 Graph 共 40 次调用，三轮各 15 samples。
不是模型吞吐或单独 W2 时间。

| 路由 | CUTLASS W2 完整路径 µs | TL W2 完整路径 µs | 耗时降低 | 基线自身波动 |
| --- | ---: | ---: | ---: | ---: |
| hot | 174.0800 | 170.2144 | 2.22% | 0.10% |
| spread | 793.1136 | 781.5680 | 1.46% | 0.01% |
| rotating40 | 795.9808 | 789.2224 | 0.85% | 0.31% |

三组均逐位核对最终输出和所有 W2 routed slots。
hot/spread 的 Graph 各进行两次 hidden/IDs mutation，hot 捕获后可切换到
36 个不同专家；输出与 slots 毒化后重放仍一致。
rotating 覆盖 40 组路由，未另做捕获后的 mutation。

派发必须有调用证据：每个 eager pair 断言 baseline/candidate W2 各被调用一次。
完整预热/捕获阶段，hot/spread 两组各记录 49 次，rotating 两组各记录 200 次。
计数是 host 调用记录，不是 Graph replay 的 GPU kernel hit counter。

第一版离线派发误把 C++ op 的第六个参数 `output` 当成第七个参数
`routed_out`，地址登记未命中，两组实际都调用 CUTLASS。
该批结果已隔离在 `invalid-apply-pointer-control/`，不能作为候选性能证据。
修正索引、清空每组地址登记并加入上述断言后重新运行；本表只引用修正版。
独立阶段测试未使用这段派发，不受影响。

**吸收方向：只考虑 M6 的 W2 BN128。** 运行模块目前仍是固定 M1/BN64 builder，
不能把离线 builder 的验证当成运行模块已经支持 M6。
下一次运行层接入应保持独立 W2 选择、加载时编译/预热、动态 align metadata，
沿用 E384/top-k6/SM80/单 TP/EP/BF16 的目标门禁。
M6 的 sorted IDs 容量为 576、expert IDs 容量 36，不能复用 M1 的不可变 metadata。
接入后需要在现有 pytest 中验证分支命中、fallback、Graph 和加载后无编译，
再以实际专家权重/路由和服务回归决定是否部署。
本轮离线收益不足以单独证明全模型吞吐提升。

## 回到 FP8 dense：已核对的真实输入合同

生产 target 日志选择 `MarlinMxfp8LinearKernel`。
后续日志中还有 Emulation 选择，处于 draft 构造阶段，不能据此判断 target 已换后端。
DeepSeek 的 ordinary dense 走 MXFP8，而非普通 FP8 block128×128 scaled-MM：

1. `models/deepseek_v4_1/quant_config.py` 将 `[32,32]` E8M0 checkpoint 配置
   交给 `ModelOptLinearMethod(QuantSpec(kMxfp8Static, kMxfp8Dynamic))`。
2. `quantization/modelopt.py:KMxfp8Static` 的 scale loader 将
   `[N/32,K/32]` 原始 E8M0 bytes 按行 repeat32，成为连续 uint8 `[N,K/32]`。
   权重保持 E4M3 `[N,K]`，layer 的逻辑 block 变为 `[1,32]`。
3. Marlin 加载时把 FP8 权重打包/repack，并转置、排列 scale。
   新原型应保存 repack 前的原始权重/scale，不把 Marlin 内部排列交给 TileKernels。
4. 当前 Emulation 默认在加载时反量化一次为 BF16，执行时调用 BF16 linear。
   参考算法应先执行 `FP8 -> FP32 * exp2(byte-127) -> BF16`，再做 GEMM。
   不能使用未舍入的 FP32 权重 GEMM 作为唯一正确性标准。

| ordinary dense | N | K | 展开后的 scale |
| --- | ---: | ---: | --- |
| fused wq_a+wkv | 1792 | 5120 | [1792,160] |
| wq_b | 32768 | 1280 | [32768,40] |
| wo_b | 5120 | 8192 | [5120,256] |

`wo_a` 标记为 grouped BMM，应单独研究。
TileKernels 的 `quant/common.py` 可参照其布局集中描述，
`quant/cast_back_cuda.py:65-81` 可参照 scale 解码一次后通过 shared 广播。
适合吸收进 tile 内反量化；独立 cast_back 物化整块 BF16 权重不能直接视为 dense 优化。

### checkpoint scale 抽样

读取 layer0 四个 scale payload，总计 90,880 bytes；没有读取权重数据，
没有加载整模型或使用 GPU。主代理独立重读相同 byte ranges 并核对 SHA256。

| key | checkpoint scale shape | min/max byte | byte0 / byte255 |
| --- | --- | --- | --- |
| wq_a | [40,160] | 114/115 | 0/0 |
| wkv | [16,160] | 114/117 | 0/0 |
| wq_b | [1024,40] | 114/117 | 0/0 |
| wo_b | [160,256] | 114/116 | 0/0 |

此抽样不能证明整个 checkpoint 不含特殊 scale。
TileKernels `byte << 23` 的解码在 byte0 返回 0，而当前 vLLM 参考返回 2^-127；
byte255 的 vLLM `exp2` 路径为 inf，Torch E8M0 dtype 语义为 NaN。
原型必须声明参考语义并覆盖边界，不能照搬位移公式或假定特殊值不会出现。

下轮 FP8 原型可从 BF16 X、原始 FP8 W、展开 uint8 scale 入手：
先 N128/K128、M1/4/8，再实际 fused N1792/K5120、M1/6；
BM16/BN64/BK128，tile 内反量化到 BF16，SM80 BF16 MMA 与 full-K FP32 累加。
先核对 dequant bytes，再对 FP64 点积 oracle 和 BF16 linear，最后单独比较 Marlin
的误差与延迟。固定同一输入行跨 M 的输出稳定性是独立验收项。
本轮只完成合同与真实 scale 抽样，尚未实现这个 dense kernel。

## 检查与证据

新 benchmark 的全部适用 pre-commit hooks 和 mypy-3.12 通过。
GPU 正确性在阶段/完整路径 benchmark 中执行，生产代码本轮没有修改。
上轮生产 pytest 的 247 项结果属于上轮接入证据，不作为本轮新候选的测试数量。

证据目录：`artifacts/tilekernels-m6-gemm-20261001/`。
`stage-ab.json`、`apply-ab.json` 保留原始计量；四份 `.cu` 保留候选代码；
stage/apply 分别记录执行版本的脚本快照与 SHA256。
`dense-scale-inventory.json` 记录 dtype、索引、绝对/相对偏移与 payload hash。
`manifest.json` 绑定本轮源码、日志、检查结果与生产健康快照。
