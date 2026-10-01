# TileKernels 吸收下一轮：SwiGLU + INT8 A2 融合

本文保留离线原型轮次的测量。后续接入与验收见
[生产接入报告](vllm170-tilekernels-swiglu-production-2026-10-01.md)。

## 结论

本轮实现了固定 `I=2304`、动态 rows、调用方提供输出的离线 TileLang 原型。
最终候选为 **256 threads、快速指数、全范围除法、两次 BF16 舍入**。
SM80 上，6 routed rows 的 Graph 耗时从 4.448 us 降至 2.320 us，
比仓库已有 Triton 融合原型的 2.896 us 再下降 **19.89%**。

在两边均使用当前 TileLang W13/W2 GEMM 的真实 `Experts.apply` 调用链中，
固定专家和轮换专家分别下降 **1.29% / 1.53%**，输出位级一致。
这是局部 MoE Graph 结果；本轮没有整模型吞吐测试，也没有接入生产路径或改变开关。

## 从 TileKernels 吸收了什么

参考仓库为 `/mnt/data/apps/vllma100/TileKernels`，HEAD
`66258df6175d2f630ffecb04c5ab66bff8a2ae6a`，检查时工作区干净。

- `tile_kernels/quant/swiglu_forward_cuda.py` 将 activation、amax、scale 和量化
  放在同一 producer 中，避免中间张量往返。本轮保留现有 numerical INT8 契约，
  没有移植其 packed FP4 或 group-scale 布局。
- 同一文件的 `use_fast_certificate` 仅覆盖 BF16→E4M3、power-of-two scale。
  本轮吸收“先验证舍入决策再使用快速公式”的方法，重新建立适合现有 BF16→INT8
  路径的证据，没有直接复用其证书。
- 用多种线程数检验低 rows 下的代价。1024 threads 改善普通公式，但快速公式的
  最佳候选是 256 threads；persistent/high-thread 组织不能直接决定本项目参数。

融合本身不是新的发现。仓库已有
`benchmarks/kernels/deepseek_v41_activation_int8_sm80.py` Triton 原型，
[历史 decode 报告](vllm170-dsv41-pp5-decode-profile-2026-09-12.md)
记录过 4.656→2.688 us 的局部收益，整模型解释力不足而没有保留生产接入。
本轮将该原型作为直接 A/B 对照；新的价值是更快且有更明确数值验证的实现。

## 数值契约与修复

输入为连续 BF16 `[rows, 4608]`，前后两半分别为 gate/up。
输出为 numerical INT8 `[rows, 2304]` 和 FP32 `[rows]` scale。

现有 CUDA activation 的语义是：

1. `silu(gate)` 先舍入至 BF16。
2. 将该 BF16 值转 FP32，与 up 相乘，再舍入至 BF16。
3. 对第二次舍入后的值求整行 amax，scale 为 `amax/127`；scale=0 时用 1。
4. 以 round-to-nearest-even 转 INT32，clamp 至 `[-127, 127]`，存为 INT8。

依据为 `csrc/libtorch_stable/activation_kernels.cu` 的 scalar/packed compute
和 silu 实现，以及现有 `_quantize_per_token_kernel`。加载的 CUDA activation
编译规则没有 `--use_fast_math`，不能从 MXFP4 GEMM 的编译选项推断 activation。

原型验证发现了两个会被常规随机输入漏掉的问题：

- 用 `rcp.approx.ftz.f32` 做 INT8 normalization 时，极小 scale 会导致错误。
  有限 BF16 编码、四种 up 的检查中出现 **763 个 q byte 差异**，scale 本身一致。
  当前 Triton 3.7.1 的实际 PTX 在 scale 和 normalization 两处均使用
  `div.full.f32`；原型改为相同指令后消除了差异。
- 直接组合 `__expf` 与 `__fdividef`，在 gate=-87.5/-88/-88.5 上把有限 SiLU
  输出变成负零。先验证过尾部回退方案，再验证 `__expf` 配合 `div.full.f32`；
  后者通过全部有限 BF16 gate 编码，最终不需要尾部分支。

最终快速公式不是全局 `--use_fast_math`。它只替换指数，并显式保留全范围除法、
两次 BF16 round 和 INT8 round/clamp。

点对点证据是：把全部 **65,280 个有限 BF16 gate** 与 up=1 送入当前加载的
CUDA activation，比较 BF16 输出 bit pattern；普通和最终快速公式均为 0 差异。
这避免后续 INT8 量化掩盖 activation 差异。第一次 BF16 结果一致之后，使用同一
FP32 multiplication 和第二次 BF16 round 对其他有限 up 保持相同计算；另有
四种 up、随机幅度和舍入边界检查。没有穷举所有 gate/up 组合，也不声称覆盖
NaN/Inf 的完整语义。证据绑定当前 GPU、编译器与加载的二进制。

## 测量范围与结果

测试设备为闲置 CMP170HX，SM80，PCI `00000000:02:00.0`，UUID
`GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216`。直接 A/B 后记录的 SM/HBM clocks
为 1485/1728 MHz。Python 3.12.12、Torch 2.13.0+cu130、CUDA 13.0、
TileLang 0.1.15、Triton 3.7.1。vLLM HEAD 为
`ae32948b167dbbb1cc87dbe4f965919f9705dc1c`。

编译、autotune、分配、数据生成、reference 和日志均在计时外。
各路径写入调用方提供的 q/scale，计时内没有额外 `copy_`。
这修正了旧 benchmark wrapper 会额外计入两次输出复制的比较边界。

主指标为 CUDA Event 包围的 Graph replay，缓存为 warm。每次 Graph 含 64 次
activation+quant 调用；直接 A/B 为 5 轮、每轮 25 samples，交替顺序，并预热
1000 ms。完整 MoE 为 3 轮、每轮 15 samples，每 Graph 40 次 apply，预热 500 ms。
均保存原始 samples，并在计时前 poison 输出后 replay 验证。

| 6 rows activation+quant | Graph P50 (us) | 相对两算子路径耗时下降 |
| --- | ---: | ---: |
| 现有 CUDA activation + Triton quant | 4.448 | — |
| 旧 Triton 融合 | 2.896 | 34.89% |
| 最终 TileLang，256 threads | **2.320** | **47.84%** |

直接与旧 Triton 比较，baseline 自身 P50 spread 为 0.55%，候选下降 19.89%。
与两算子路径比较，baseline spread 为 0%。它们是两次独立直接 A/B，
没有用不同时钟状态下的跨实验绝对耗时做排名。

形状检查及独立 3 轮 microbenchmark 中，最终快速 256-thread 版的 P50 为：

| rows | 1 | 6 | 18 | 40 | 128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TileLang (us) | 2.320 | 2.336 | 2.384 | 2.432 | 3.104 |

| 真实 Experts.apply，同一 TileLang W13/W2 | 原路径 (us) | 融合 (us) | 下降 | baseline spread |
| --- | ---: | ---: | ---: | ---: |
| 固定 6 experts | 168.6272 | 166.4512 | 1.29% | 0.02% |
| 40 组轮换、不同 experts | 168.7040 | 166.1184 | 1.53% | 0.03% |

完整调用链使用 synthetic E384 packed weights、非均匀 channel scale 与路由权重，
包括 A13 quant、W13、activation/A2 quant、W2 和 reduce。仅在离线脚本中替换
候选 activation，并把已产生的 q/scale 交给真实 W2；生产源码未修改。
最终 output 与 workspace13 别名，检查全部 W2 routed slots 和 final output。
固定专家 Graph 还原地更改 hidden states 与 expert ids，三次 replay 均一致。
这不等于真实 checkpoint 或 PP5 整模型验证。

按技能默认尝试了 FlashInfer CUPTI，但环境缺少 `cupti-python`，实际使用其
CUDA Graph fallback。初始 cold-L2 请求还因 closure 未提供输入 tensor 而被禁用；
该轮 artifacts 保留警告，不作为 cold-cache 证据。最终辅助计时明确使用 warm
cache，并保存 fallback 警告。本文没有报告 CUPTI 或冷缓存测量结果。

## 验证与文件

生产路径和已有默认关闭开关均未改变，服务未重启。现有 integration 文档的
未提交修改保留。本轮新增离线原型和正确性测试，测试位于既有 quantization suite，
性能测量位于 `benchmarks/kernels/`。

最终验证：

- 正确性覆盖普通/快速公式、256/1024 threads、五种 rows、随机/小值/大值/
  RNE tie/±zero；有限 BF16 gate 编码和四种 up；Graph 原地改输入并覆盖 poison 输出。
- 最终 pytest：**110 passed、112 deselected**（44.94 s），结果见
  `artifacts/tilekernels-swiglu-int8-20261001/final-tests.log`。
- 真实 `Experts.apply` 固定/轮换专家、输出别名、Graph 输入/路由变更：位级一致。
- scoped pre-commit `ruff-check`、`ruff-format` 和 `git diff --check` 通过。
- 最终 CUDA source 的独立 SM80 cubin/ptxas 检查：35 registers、0 stack、
  0 spill stores/loads，见 `ptxas.log`。

可复现命令：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python -m benchmarks.kernels.deepseek_v41_moe_tilelang_activation \
  --output-dir artifacts/tilekernels-swiglu-int8-20261001/final-micro \
  --threads 256 1024 --rows 6 1 18 40 128

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python -m pytest tests/kernels/quantization/test_mxfp4_int8_sm80.py \
  -k tilelang_swiglu -q

PYTHONPATH=. CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python artifacts/tilekernels-swiglu-int8-20261001/bench_direct.py

PYTHONPATH=. CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python artifacts/tilekernels-swiglu-int8-20261001/bench_apply.py
```

关键文件在 `artifacts/tilekernels-swiglu-int8-20261001/`：

- `final-micro/metadata.json`、`microbench.json` 和生成 CUDA。
- `direct-ab.json`、`direct-metadata.json`：两组直接 A/B 的原始 samples 和 clocks。
- `apply-ab.json`、`apply-metadata.json`、`bench_apply.py`：实际调用链、原始 samples。
- `diagnosis/quant-*.ptx`、`diff-False.json`：实际 Triton 除法与最初失败。
- `math-certificate.json`：未修正快速除法的三个失败编码。
- `final-micro/metadata.json` 内 math certificates：最终公式零差异。
- `manifest.json`：当前源码、脚本、二进制、结果文件的 hashes。

## 后续接入边界

数值与局部性能支持保留该原型，但整模型收益还未验证，不应直接调整生产默认。
后续接入可限定在已有默认关闭的 TileLang decode 开关内，并在加载时编译，
forward/capture 只调用缓存 kernel。

必须检查 `self.activation_config.clamp_limit`，而不是只检查 SILU 枚举或 HF config：
它可能来自 quant config 的 `gemm1_clamp_limit`，也可能来自 MoE `swiglu_limit`。
带 clamp、其他 activation 或非目标形状必须使用原路径。不能提前乘 routing
weight，也不能把这里的 whole-row INT8 scale 当作 FP8/group-scale layout。
本轮没有继续推进 FP8 dense MM；以上优化仅作用于当前 MoE 的 INT8 A2 producer。
