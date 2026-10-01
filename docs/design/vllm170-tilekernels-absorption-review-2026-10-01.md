# TileKernels 吸收复核与下一轮优先级

后续状态：用户已放弃 FP8 dense 的 TileLang 路线，下文关于其优先级的判断保留为
该轮研究记录。已落地的 M6 W2 与 activation 量化融合，以及后续 packed-word
优化，见[最新 MoE 优化报告](vllm170-dsv41f-target-moe-w2-word-load-opt-2026-10-01.md)。

日期：2026-10-01。复核对象：vLLM `ae32948b16`，TileKernels
`66258df6175d2f630ffecb04c5ab66bff8a2ae6a`；实际环境为 TileLang 0.1.15、
Torch 2.13.0+cu130、apache-tvm-ffi 0.1.11。
开始时工作区已有 integration 文档的 3 行参数修订，本轮保留。

**结论：这一轮吸收方向正确，固定形状的候选可继续保留；收益口径、运行验证归档
和未来 FP8 的边界契约还需收紧。下一轮最值得吸收的是融合中的舍入与 scale 决策、
消费者所需布局直接输出、量化数值验证；FP8 dense 的实现参考仍以
`cast_back + common.py` 为核心。**

本轮未发现最新改动在已门控的 M1/E384/top-k6 路径中的确定性正确性缺陷。
这项判断来自代码复核、现有 A/B 样本和下面的 CPU/离线检查，并不等同于重新完成
全部 GPU 回归或模型级验收。

## 1. 已吸收内容与复核结论

| 变更 | 实际落地 | 复核 |
| --- | --- | --- |
| `ef49429923` 升级 | CUDA dependency pin 0.1.15，TVM-FFI 仍为 0.1.11 | 实际版本吻合；本轮原生导入全部 32 个 `*_cuda.py`，没有 shim |
| `946c3bfbda` PRMT 循环 | `T.Parallel(N,KB,4)` 内 `T.unroll(8)`；W2 改 128 threads/3 stages | 旧 `(j,nib)` 与新 `byte=2*j+nib` 索引等价；本轮当前代码 lower 和 nvcc 均通过 |
| `43571bf677` 契约 | 集中 packed-K/shift-word 索引与 shape，加载前检查 dtype/device/连续性 | 两阶段 codegen 与 946 逐字节相同；六项负例在编译前失败 |
| 可选接入 | 加载时 compile/warm，forward 用已准备的 kernel | 默认关闭、严格 bool、模型/形状/设备门禁、M2/top-k3 回退，调用链清楚 |

代码入口：
[候选内核](../../vllm/model_executor/layers/fused_moe/experts/dsv41_sm80_tilelang.py)、
[接入与门禁](../../vllm/model_executor/layers/fused_moe/experts/mxfp4_int8_sm80.py)。
加载检查在候选 `:229`；门禁在接入 `:98`、`:204`，运行分支在 `:362-410`。
测试 `test_mxfp4_int8_sm80.py:1674` 覆盖真实 apply、workspace 别名、动态 Graph 路由、
M2/top-k3 回退和 capture/forward 禁止再次编译。

435 的改动是当前 packed MXFP4/2-bit shift 格式的契约整理。
它没有实现可变粒度的 MXFP8 接口：`_k_blocks` 是整除，N/K 仍固定，
加载检查中 N/K 仍有字面常量。`sh_s` 是 `[N/8,8]` 的 shift-byte staging，
每 byte 包含四个 K block 的 2-bit shift。不能把它描述成已经具备通用
`[row,k_word]` MXFP8 scale tile。

## 2. 需要纠正或补齐的证据

### P1：GPU 验证记录需绑定最新源码

`artifacts/tilelang-integration-20261001/manifest.json` 的候选源码 hash 是
`ea19989d...`，当前是 `281569d8...`。旧目录的 W2 使用 64 threads，
最新代码使用 128 threads/3 stages。因此保存的 112 passed 和旧 `.cu`
不能直接作为最新 schedule 的完整回归记录。

最新 `artifacts/tilelang-015-remeasure-20261001/` 确实有全路径 A/B 原始样本、
`bitwise_exact=true` 和复现脚本，summary 声明对应 946+435。
但该目录没有完整 pytest stdout、当时源码快照/hash、环境与 GPU 占用快照；
第一轮仅有 summary，原始样本保留的是第二轮。
提交说明所述“112 passed 不变”当前无法从最新归档独立还原。
本轮补齐了 codegen 和负例证据，仍未补齐最新 GPU 完整回归。

### P2：性能数据应以当前原始样本为准

本轮重新计算第二轮的 median-of-round-medians：

| 工作集 | CUTLASS µs | TileLang µs | 延迟降低 | 基线轮间 spread |
| --- | ---: | ---: | ---: | ---: |
| fixed | 171.6480 | 169.4976 | 1.2528% | 0.1790% |
| rotating | 171.2640 | 169.1648 | 1.2257% | 0.1794% |

与 JSON 中四舍五入的 1.25%/1.23% 一致。
`ae32948b16` 提交说明中的“baseline spread 0.04% or better”不适用于第二轮；
summary 自身已经记录 `0.0018`，即约 0.18%。

旧 8.05%/7.85% 来自不同工具链/schedule/会话，不能用于当前实现。
“基线位移只能来自机器状态”也是尚未控制变量的推断：需要同时冻结源码、
编译参数、库版本、权重/路由工作集和设备状态后才能归因。
当前证据只支持合成权重、M1/E384/top-k6、`Experts.apply()` 的专家段收益，
没有模型 tokens/s、质量或 shared/routed 并发 A/B。

### P2：编译可用性、bank 布局和文档时间线

- resurvey `:21` 的“SM80 上全部可用”应改为“所列配置能 lower/编译”。
  同文 `:158-160` 仍写没有运行任何 kernel，却在 `:35-49` 添加了部分 mHC 上机结果；
  建议明确初始离线验证与后补上机验证的范围。
- contract-transfer `:73` 把 SM120 K-major 排列称为“避免 bank conflict”过强。
  性能取决于 SM80 消费者的 lane 映射和 load 宽度。例如 128 行、uint32 words、
  warp lane 连续取 row 0..31 时，公式的 bank 为 `4*lane mod 32`，会出现四路冲突；
  硬件原有消费者可能采用不同映射。应只作为候选排列，不能预先认定 bank 无冲突。
- survey `:200` 只列 `testing/numeric.py`，没有展开
  `testing/quant.py:7-27` 的可表示量化等级 rank；“无 ULP 指标”字面上只描述该文件，
  不应据此遗漏整个库已有的量化级差判据。

## 3. 下一轮值得吸收的优点

### A. 优先：SwiGLU 与现有 INT8 A2 量化融合

TileKernels 的 `quant/swiglu_forward_cuda.py` 把 activation、group amax、scale、
量化写入放在一次读取/launch 内（`:192-297`）；persistent loop 还理解 expert prefix sum
和 padding（`:124-190`）。这是当前 MoE `activation` → `w2_quant`
两个连续阶段的具体参考，接入位置为 `mxfp4_int8_sm80.py:430-434`。

当前格式是数值 INT8、整行 FP32 scale、amax/127、round-to-nearest-even，
全零行 scale=1（`quantization/utils/mxfp4_int8_sm80.py:33-62`）。
TileKernels 的 `torch.int8` 表示 packed FP4（`common.py:32-33`），公开 SwiGLU 仅支持
FP8 block32/128 或未量化输出（`swiglu_forward_kernel.py:41`），不能直接套格式。
其 1024 threads、persistent SMS grid、prefetch 参数也不能直接当作 M1 的最优调度。

**融合必须保留两处 BF16 舍入**：现有 CUDA `silu_kernel`/`packed_silu_kernel`
先把 SiLU 输出转回输入 dtype，然后乘 up，再转回 dtype
（`csrc/libtorch_stable/activation_kernels.cu:44,80-83,165-177`）。
INT8 amax 应基于第二次舍入后的值。不能使用一条 FP32 SwiGLU 表达式直接量化。
本轮 CPU 诊断在 6×2304 输入上，省掉中间 SiLU 舍入导致 3741 个 BF16 值、
236 个后续 INT8 值不同；该实验说明舍入顺序敏感，不是 CUDA `expf` 等价证明。

可落地版本：固定 I2304、调用方提供 INT8/scale 输出，复用现有 workspace 规则，
不提前乘 routing weight。先与现有 activation+quant 的 q bytes/scale 逐位比对，
再验完整 apply/Graph。收益首先来自少一次 launch 与少一次 intermediate 写回/重读，
无需改变两个 GEMM 的算术。

### B. FP8 dense 主线：scale 契约与 tile 内反量化

`quant/common.py:94-119,188-248,285-316` 将逻辑量化块、存储 dtype、stride、
packed/column-major 索引集中定义；`cast_back_cuda.py:65-81` 把 scale 解码一次放入
shared，然后按 block 广播。这两部分最适合成为 MXFP8 W8A16 候选的参考。

本轮用 `[17,160]` uint8 scale → int32 view → TileKernels canonicalization 验证：
恢复为 `[17,160]`、stride `[160,1]`，数据相等且同一指针，确实无拷贝。
适配仍需 K/128 对齐、byte storage offset/stride 满足 dtype-view 要求。
行主序 MXFP8 权重 scale、FP32 block128×128 普通 FP8 scale、
FlashInfer F8_128x4 activation scale 必须有不同的显式布局描述。

边界语义不能照抄：

- byte0：TileKernels shift23 解码为 0；vLLM `dequant_mxfp8_to_bf16` 为 2^-127。
  本轮 FP32 CPU 核对全部 256 bytes，仅 byte0 与这个 vLLM 参考不同。
  `448 * 2^-127` 转 BF16 仍为 `2.6331e-36`，并非必然被输出舍入消掉。
- byte255：两者的上述算法都给 inf，但当前 Torch 的 E8M0 dtype 将 byte255 解成 NaN。
  “两份算法相同”不等于“等于完整 E8M0 格式语义”。未来包装层应规定并验证
  有效 scale 范围，或显式处理特殊值；在 checkpoint 上是否出现这些 bytes 本轮未调查。

实现应从 checkpoint 的原始 E4M3 `[N,K]` 和 scale 入手，在 tile 内产生 BF16/FP16
给 SM80 MMA，保留 FP32 累加，再写 BF16。独立 `cast_back` 会物化完整权重，
适合作参考而非线性层的热路径。
候选通过 `Mxfp8LinearKernel` 的加载/执行接口接入；普通 FP8 与 MXFP8 分开验证。
当前 Marlin 的 `use_full_k` 配置仍在 `kernels/linear/mxfp8/marlin.py:36-64`。
新候选需先规定 full-K/跨 M 的累加顺序与误差合同，再决定分流；比较时保留原始
权重、bias 和每层真实 N/K，不能只测一组近似尺寸。

### C. 优先吸收验证模式：bytes、scale、量化等级与快速数学回退

`testing/numeric.py:5-30` 对 dtype/shape/stride/device 和原始 bytes 做精确检查，
`testing/quant.py:7-27` 对 BF16/E4M3/E2M1 给可表示等级 rank。
建议测试顺序为：scale/layout 精确 → quantized bytes 精确 → dequant 误差与 rank 诊断。
rank 把正负零视为同一级，不能替代 signed-zero/NaN/inf 的 byte 与有限性检查，
也不能把“相邻一级”自动作为当前要求逐位一致路径的验收条件。

`swiglu_forward_cuda.py:232-269` 更有价值的优化模式是：
BF16→E4M3、pow2 scale 时先用快速 `__expf/__fdividef`，检查量化决策边界，
对不安全 warp 回退原计算。它比直接打开全局 fast math 更值得研究。
但证书只覆盖特定 E4M3/SF 路径，不能移到 INT8 或 BF16 输出后继续声称成立。
后续若采用，应补临界量化 midpoint、scale 跳档、gate=-16 附近和特殊值测试。

benchmark 可借鉴 JSONL baseline、相对+绝对回归阈值
（`testing/pytest/benchmark.py:39-40,260-298`），但默认 5% 阈值无法判断本轮约 1.2%
收益是否在后续丢失。应按设备噪声为各工作集配置阈值，保存原始样本和环境。
其默认 CUPTI、warmup=0（`:780-800`）测量的是相应 kernel 时间口径，
完整 apply/Graph Event、冷加载/编译时间应另列，并在正式采样前显式预热。

### D. 有条件吸收：producer 直接写消费者布局、Norm 多输出

`norm_forward_cuda.py:179-239` 同时输出 BF16、FP8、scale；
`norm_forward_kernel.py:23-29,80-102` 支持外部输出 buffer 和 stride 校验。
其 MegaMoE scale 行映射（`:77-85`）说明 producer 可以直接写消费者布局，
省去专门 transpose kernel。

当前 vLLM 已有同一 MegaMoE 映射
（`vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py:91-108`）；
DSV41F 的 mHC 已融合 Norm（`deepseek_v4_1/nvidia/model.py:382-395`），
Q/KV Norm+Q MXFP8 的融合也已存在（`deepseek_v4_1/common/ops/query_quant.py:46-79`）。
这些不应再次列为新增实现。下一步只针对仍然分离的真实调用链，借鉴双输出和
最终布局直接写入的接口；避免重复 Norm，也避免生产量化张量却没有消费它。

设备资源参数应显式来自目标设备：TileKernels `config.py:39-47` 的 SM 数和
`common.py:13-16` 的向量宽度缓存不含 device key；混合架构进程不能照搬该缓存方式。

### E. 后续：router 的完整语义测试与表示范围证明

MoE router 除了 XOR tie-break，还保留 bias-free score 供选后归一化，分开处理
logical/physical expert 映射和 mask
（`moe/moe_topk_gate_forward_cuda.py:189-288`）。
`tests/moe/test_moe_topk_gate_forward.py:148-277` 的 bias、mask、固定/随机路由、
duplicate mapping、零 logits tie 是可吸收的测试集合。
采用稳定 tie-break 前必须核对现有 router 的选择与归一化顺序；它可能改变输出。
本轮不把 router 改写排在已有专家 GEMM/activation 融合之前。

`per_block_cast_lossless_cuda.py:20-30,119-143` 还体现一种有用做法：
改变 scale 分组时先推导可表示范围，再通过运行断言拒绝“lossless”前提不成立的块。
它只支持 E2M1→E4M3，并不是 INT8 converter；可以借鉴其范围检查方式，
为当前近似 MXFP4→INT8 转换补充 exponent-span 和误差分布统计，不能把当前近似
后端改称无损。本轮不扩展其权重转换或改变质量合同。

## 4. 建议给下一轮实施的边界

1. 先整理最新源码/环境与 GPU 回归归档；当前 default-off 继续保留。
2. 单独实现固定 I2304 的 SwiGLU+INT8 A2 原型，只吸收融合组织与验证方法。
   q/scale、routed output、最终 output、Graph 动态数据全部通过现有合同后再计时。
3. 原始 FP8 dense 主线先做 MXFP8 W8A16 小原型，明确 scale0/255 与 layout，
   以 `cast_back` 和当前 layer 输入/权重为参考，测试实际 M/N/K 和跨 M 的同一输入行。
   若正确性未过，不继续调 tile；若全路径无稳定收益，保留现有后端。
4. producer 布局直写、Norm 双输出仅在调用链和 profiling 证明有重复读写后落地；
   router、Engram、反向训练算子排后。

没有新增未经测量的加速百分比。本轮没有执行 GPU kernel、重启服务或改变线上开关。

## 5. 本轮可复现证据

[证据目录](../../artifacts/tilekernels-review-20261001)包含脚本、CPU JSON/log、
重构前后两阶段 CUDA、当前 cubin 与 ptxas log。

- CPU 现有测试：`CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest
  tests/kernels/quantization/test_mxfp4_int8_sm80.py -q`，76 passed、36 skipped、14 warnings。
- CPU/离线复核：`CUDA_VISIBLE_DEVICES='' .venv/bin/python
  artifacts/tilekernels-review-20261001/review_cpu.py`。负例用 meta tensors，
  compile 函数被阻断，验证合法输入能抵达 compile，错误输入在此前失败。
- 重构 codegen：W13 15985 bytes、SHA256 `54a28092f440259c...`；
  W2 19870 bytes、`e001dadd34f190a5...`，均与 946 逐字节一致。
- 当前两阶段离线 nvcc sm_80 编译 exit0：W13 75 registers、W2 71 registers；
  两者 stack/spill stores/spill loads 均为 0。编译有 CUDA/TileLang 模板弃用警告。
  这是本轮列明参数的外部 nvcc 结果，不是运行性能或 occupancy 验收。

编译复现（将 `w13` 换成 `w2` 可检查另一个阶段）：

```bash
nvcc -std=c++20 --expt-relaxed-constexpr -arch=sm_80 -DENABLE_BF16 \
  -I .venv/lib/python3.12/site-packages/tilelang/src \
  -I .venv/lib/python3.12/site-packages/tilelang/3rdparty/cutlass/include \
  --ptxas-options=-v -cubin \
  artifacts/tilekernels-review-20261001/w13-current.cu \
  -o artifacts/tilekernels-review-20261001/w13.cubin
```

上游对照：[TileKernels README](https://github.com/deepseek-ai/TileKernels)
声明 CUDA 支持目标为 SM90/SM100；本报告的 SM80 结论限定为逐项检查结果。
[TileLang v0.1.15 发布说明](https://github.com/tile-ai/tilelang/releases/tag/v0.1.15)
确认 launch/thread、reducer、缓存等升级边界；实际判断以本地安装源码为准。
复制 TileKernels 实现时保留其 MIT 版权和许可。
