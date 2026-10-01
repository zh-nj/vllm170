# DSV41F target 验证：MoE W2 packed-word 优化

日期：2026-10-01。承接上一轮 sparse decode 优化。
测量时工作区 HEAD 为 `ae32948b167dbbb1cc87dbe4f965919f9705dc1c`；
下述性能数据来自包含此前生产优化的工作区。

**已将 q6 MoE W2 的 packed-word 读取优化接入原生产服务。**
同一配置、同一 32K prompt，target Graph 五个 rank 中位数之和
从 **39.928 降到 38.932 ms，减少 0.996 ms / 2.49%**。
这是未跨 rank 对齐的统计量，不是整轮 PP wall time。
三次无插桩 256-token 请求的中位速度为 **86.06 → 89.64 tok/s**，
本次观察 +4.16%；六次请求的输出 ID 和接受量完全一致。
服务 health=200，phase 控制文件已移除。

## 热点判断与改动

本轮先梳理 q6 MoE 的实际调用链：A1 quant、routing cast/align、CUDA W13、
已部署的 TileLang activation+A2、TileLang M6 W2、CUDA reduce。
没有重新扫描 q，也没有重试已放弃的 FP8 dense TileLang。

此次提交同时纳入此前已部署、尚未提交的 MoE 运行层依赖：M6 W2 编译、
预热和 dispatch，以及 SwiGLU+INT8 A2 融合。后者保留 SiLU 和乘法的两次
BF16 舍入、FP32 amax/127 scale 与 INT8 RNE/clamp；复用原 activation workspace。
沿用默认关闭的 DSV41F opt-in，仅在 E384/top-k6、SILU、clamp=None/10、
routed rows≤128 时选择融合路径，其他配置保持现有 CUDA/Triton fallback。
运行层及测试所需的离线基准一并纳入，attention 与代价模型改动单独留在工作区。

[dsv41_sm80_tilelang.py](../../vllm/model_executor/layers/fused_moe/experts/dsv41_sm80_tilelang.py)
原先通过四个 uint8 cast、位移和 OR 拼出一个 packed word，再交给 PRMT 解码。
候选在共享内存上用 `T.address_of` 和一个 CUDA helper 直接读取 uint32。
只在 M6 W2 使用它：既有门禁仍限定 DSV41F、E384/top-k6、H5120/I2304、
SM80、BF16 和 BM16/BN128。BM16/BN128/BK128、128 threads、pipeline=3
保持原配置，INT32 GEMM、BF16 epilogue、activation、routing 和全局拷贝保持原逻辑。

共享数组的基址对齐，行跨度为 128 bytes，索引为
`channel_in_group * 16 + t * 4`，因此地址总是四字节对齐；
最大索引 124，读取覆盖到 127，仍在该行内。
没有新增持久 workspace、运行时搜索或 forward 中的编译。

生成代码检查确认两个 M1 GEMM 的 CUDA 源码与改动前完全相同。
接入后的 M6 CUDA 源码与离线候选完全相同，SHA256 为
`3fb672cfe54eb2595bc9eef84d4c2a03b9b62dc0afc67f565aee069c0b7798be`。
原生产 TileLang cache 中的新 `device_kernel.cu` 也匹配此 SHA256。

## 反汇编修正了最初的机制解释

从 TileLang TVM FFI 的 `executable.so` 中抽出 SM80 cubin，保留二进制和哈希，
使用 `cuobjdump` 检查实际 SASS。编译器原本已经合并部分字节读取，
所以不能把源码层的“四次 uint8 → 一次 uint32”解释为 LDS 数量减少四倍。

| 静态 SASS / 资源 | 原 W2 | packed-word W2 |
| --- | ---: | ---: |
| 总指令 | 2080 | 1888 |
| PRMT | 493 | 205 |
| SHF | 167 | 263 |
| LDS（含编译器生成的 predicated-off 指令） | 71 | 71 |
| LDS.U8 | 12 | 12 |
| LDGSTS.E.BYPASS.128 | 15 | 15 |
| IMMA.16832.S8.S8 | 48 | 48 |
| registers/thread | 105 | 106 |
| stack / local bytes | 0 / 0 | 0 / 0 |

这些是整段 kernel 的静态计数，包括 pipeline prologue/epilogue，
不是动态执行次数或硬件 stall 计数。实际收益与消除 packed-byte 重组指令相容；
不宣称证明了某一个唯一的硬件 stall 原因。
packed 权重字节量和全局读取路径没有减少，因此这轮也证明：
target 的可优化成本还包含解码指令效率，不能全部归结为 HBM 字节量。

## 离线 A/B：先单核，再完整 MoE

独立空闲 GPU：`GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216`，PCI 02:00.0，
SM80、74 SM、32 MiB L2。没有在生产 PP 卡上运行微基准。
从真实 checkpoint 读取 layer20 的专家 137–172，记录逐 tensor 的
shard、绝对偏移、dtype、shape、字节数和 SHA256，并走生产 remap。
其他 E384 槽填零，路由只选这些真实专家。
INT8 A / BF16 hidden / 专家路由为合成输入，**没有捕获线上专家 ID**。

单核比较采用 CUDA Events、单次调用的 Graph、三轮交错 A/B。
每个 arm 预热 500 ms，20 samples/轮；每个 sample 在计时区间外
写入 64 MiB buffer 清洗 L2。编译、分配和 reference 在计时区间外。
没有安装 `cupti-python`，这些结果不是 CUPTI measurement。

| 专家路由 | 不同专家数 | 原 W2 µs | 新 W2 µs | 耗时减少 |
| --- | ---: | ---: | ---: | ---: |
| hot | 6 | 64.512 | 57.344 | 11.11% |
| mid | 21 | 162.816 | 144.384 | 11.32% |
| spread | 36 | 263.168 | 232.448 | 11.67% |

记录了每个 arm 前后的 NVML 时钟：HBM 为 1728 MHz，SM 为 1410–1485 MHz。
hot 两臂均为 1485 MHz；mid/spread 的候选臂时钟略低，仍然更快。
早期 pipeline 探索和首轮 word 探索的绝对耗时不同，不跨 campaign 相减。
这里引用带 500 ms 预热和时钟记录的确认轮次。

完整 `Experts.apply` 两臂均使用已部署的 TileLang W2 和同一 activation，
只替换候选实例的 W2 callable，不拿旧 CUTLASS 路径重复计入收益。
三轮交错、每轮 15 samples、每个 Graph 40 次 apply、预热 500 ms；
计时包含 A1、cast/align、W13、activation+A2、W2 和 reduce。

| 路由 | 原完整 MoE µs | 新完整 MoE µs | 耗时减少 | 基线轮间波动 |
| --- | ---: | ---: | ---: | ---: |
| hot | 172.2624 | 164.8384 | 4.31% | 0.01% |
| spread | 792.5760 | 756.8640 | 4.51% | 0.33% |
| rotating40 | 792.6784 | 763.3152 | 3.70% | 0.40% |

rotating40 轮换路由顺序和 workspace，但使用同一组 36 个专家权重，
不是 40 层不同权重或独立 cold-L2 测量。
所有最终输出和 W2 routed slots 均逐 bit 相同；hot/spread 捕获后
两次改变 hidden 和专家 ID、毒化输出再 replay，结果仍逐 bit 一致。

## 同口径生产验证

只重启了一次 `dsv41f-manual.service`，耗时约六分钟；重启前后 unit 逐字一致。
PP5/TP1、q6（num_speculative_tokens=5）、FULL Graph、CED、prefix cache、
模型和设备顺序保持原配置。当前 MainPID 为 2484411。

每个版本先运行一个无插桩 128-token 请求，随后另一个 128-token 请求在收到
首 token 后开启 phase hooks，结束后关闭。两版 128-token 输出 ID 完全一致。
没有把 prefill、JIT 或 checkpoint load 纳入 Graph decode 测量。

| rank | 原 Graph p50 ms | 新 Graph p50 ms | 减少 ms | 样本数 A/B |
| --- | ---: | ---: | ---: | ---: |
| PP0 | 6.124 | 6.023 | 0.101 | 21 / 20 |
| PP1 | 7.560 | 7.386 | 0.174 | 21 / 21 |
| PP2 | 8.203 | 7.997 | 0.206 | 21 / 21 |
| PP3 | 8.839 | 8.599 | 0.241 | 21 / 21 |
| PP4 | 9.202 | 8.927 | 0.274 | 21 / 21 |
| rank 中位数之和 | 39.928 | 38.932 | 0.996 | — |

没有跨 rank 对齐 step ID，不称作 PP wall interval，也不外推新的 q 回归截距。
此前的 33.87 ms 本来就是 q≥2 回归线的外推截距，包含既有模型工作和路径差异，
不是一个独立的进入操作；本轮仅证明其中一类 Graph 内工作可以提速。

无插桩 32K/256，temperature=0、ignore_eos=true，每版 warmup 后三次请求：

| 版本 | 三次 decode tok/s | 中位数 |
| --- | --- | ---: |
| 原版本 | 84.463 / 86.058 / 87.840 | 86.058 |
| 新版本 | 89.928 / 87.467 / 89.640 | 89.640 |

六次 256-token 输出 ID 完全一致，draft rounds 均为 44，accepted 均为 211/220；
每次 generation/prompt 计数差均为 256/32768，计量 gate 通过。
中位速度 +4.16% 是这一对 A/B 的观察，没有时间顺序交叉复测或统计置信区间。
单对 128-token 无插桩 decode 时长反而从 1.401 增到 1.548 s。
本轮观察到主机 memory PSI some/full 在约 24%–37% 波动，因而不将全部端到端变化
归因到单个 kernel，也不把插桩 period 当作无插桩生产速度。

## 本轮筛掉的方向与测量限制

- 单 CTA Triton route align：语义、padding、无效 ID、超过 BM16 的组和 Graph
  poison 检查通过。明确 warm L2 的复测约 CUDA 7.27–7.78 µs、候选
  5.53–5.63 µs，正常路由每层仅省约 2 µs；没有接入运行层。
- W2 pipeline=2：早期三种路由仅降低 1.27%–1.65%；pipeline=4 退化
  3.96%–5.06%，256 threads 退化 15.19%–21.69%。保持 pipeline=3/128 threads，
  不和 packed-word 变更混在一起。
- W13 TileLang：已有 hot 场景最高 26.98% 退化，本轮不重复推广。
- metadata：单请求 token map 仍必须处理旧多请求内容；跨 KV group 的
  block table/slot mapping 不同，已有缓存也只在 CommonAttentionMetadata 对象内。
  本轮没有在缺少独立计量的情况下改动这些语义。
- Nsight Compute 一次尝试被明确拒绝：`ERR_NVCMPGPU`，CMP 不受支持。
  保留 `w2-ncu.log`，该次受工具干扰的 timings 全部排除。
- 早期 align 探索请求 cold L2，但 fallback 没有输入参数，提示 cold 未生效。
  原记录保留为 `align-warm-exploration.json`；随后改为明确 warm L2 并复测。
- 直接对 TVM FFI `.so` 使用 cuobjdump 看不到 device code；首次抽取 cubin
  漏了 program headers，诊断为无效 ELF。补齐后成功反汇编；不使用失败时的零计数。

## 验证与复现

21 项现有 pytest 通过，覆盖 opt-in 门禁、完整 Experts.apply、M1/M6、
其他形状 fallback、clamp 和 Graph 动态输入。没有添加只检查 helper 文本的测试。
提交前追加运行完整 `test_mxfp4_int8_sm80.py`，247 passed，耗时 96.18 s。
相关文件 pre-commit 全部通过，ruff/format/diff check 通过。
首次 pre-commit 发现 benchmark 新增的 `torch.cuda.synchronize`，改为
`torch.accelerator.synchronize` 后重新通过。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python -m benchmarks.kernels.deepseek_v41_moe_w2_pipeline \
  --word-only --apply \
  --model /media/z/data6/models/DeepSeek-V4.1-Flash \
  --samples 20 --output-dir /tmp/dsv41f-w2-word-ab

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python -m pytest tests/kernels/quantization/test_mxfp4_int8_sm80.py \
  -k 'dsv41_tilelang_experts_apply_and_changing_graph_routes or dsv41_sm80_tilelang_gate or dsv41_tilelang_load' -q
```

新 benchmark 在接入后的源码上仍能构造 scalar 与 word 两臂。
`repro-code/validation.json` 已确认它们的 CUDA 源码分别与原版本、新运行层完全一致。

完整证据在 `artifacts/dsv41f-target-opt-round2-20261001/`：
`comparison.json`、`w2-word-confirm/{results,apply-ab,checkpoint-provenance}.json`、
`baseline*/`、`candidate*/`、`generated-code-check.json`、
`production-word-cache.json`、`cubin-provenance.json`、`*.sass`、`sass-opcodes.json`、
`source-{before,after}.json`、unit 快照、pytest 和 pre-commit 日志。
运行层回滚点为 `before/vllm/model_executor/layers/fused_moe/experts/dsv41_sm80_tilelang.py`；
应仅回退本轮 hunk，保留此前的 M6 W2、activation 和 sparse-decode 优化。
