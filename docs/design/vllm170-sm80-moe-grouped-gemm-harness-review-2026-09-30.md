# SM80 MoE grouped GEMM：DeepGEMM 测试设施借鉴评审

日期：2026-09-30。范围：研究与源码审计，未运行 GPU 基准、未改变服务或内核。

用户随后明确的执行范围：**仅 DSV4.1-Flash（dsv41f）推理中的 SM80 routed MoE
grouped GEMM，不得影响其他模型推理。** 扫描与后续候选均受此限制；通用 helper
不意味着扩展生产适用范围。具体任务见
[DeepSeek 执行指引](vllm170-sm80-moe-harness-deepseek-execution-guide-2026-09-30.md)。

审计版本：vllm170 `f9c6a54b064cd1178828bdf7fe73f283908223c4`；
DeepGEMM `057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7`。
评审输入：`/mnt/data/apps/DeepGEMM/docs/sm80_test_harness_port.zh.md`。
以下行号均指上述本地快照。性能数字仅引用既有报告，不代表本轮复测。

## 判断

值得做，且适合安排在下一轮 grouped GEMM 优化之前。主要收益是让候选能在同一组
路由、量化语义和计时边界下比较，从而判断它究竟改善了什么。

原说明“借鉴设计、复用本仓库测量底座”的方向正确；但参考生成语义、误差指标、
路由字节量和支持范围有实质问题，不能直接把第 5 节当作实现规格。
优先级应是：独立正确性参考 > 路由分布覆盖 > 测量边界与缓存控制 > 扫描规模。

## 原说明需要修正的内容

### 1. DeepGEMM 的参考在量化前生成

`/mnt/data/apps/DeepGEMM/tests/generators.py:357-373` 先生成 BF16 的 A/B，
在 363 行计算 `ref_d`，之后才转换为 FP8/FP4。
grouped contiguous 路径也是 387-407 行生成参考、416-417 行量化。
因此第 4 节关于“参考操作数已量化”的解释与源代码相反；其容差包含量化误差。

本仓库 `benchmarks/kernels/deepseek_v41_mxfp4_int8_accuracy.py:82` 的
`dequant_reference()` 是原始 MXFP4 的精确反量化；97 行的 `remap_int8_route()`
才表示近似重映射后参与运算的权重。二者不能混用。

建议建立三层验收：

| 层级 | 比较对象 | 捕获的问题 |
| --- | --- | --- |
| Kernel 正确性 | 生产格式独立解包出的 INT8 权重、实际 A8 输入和 scale，逐专家参考 GEMM，对照候选输出 | packed layout、shift、slot/token 索引、累加、缩放、写回错误 |
| 量化质量 | 重映射 W4r/A8 路径对照原始 MXFP4 精确反量化 W4e/A16 路径 | 权重重映射与激活量化损失 |
| MoE/模型行为 | W13→激活→A8→W2→路由加权 reduce，以及真实模型请求 | BF16 中间舍入、融合边界、路由权重、端到端输出回归 |

Kernel 参考必须覆盖 W13 的 token 行与 W2 的 routed slot 行两个索引契约，
并模拟生产 epilogue 的缩放/舍入顺序。小尺寸可用独立整数累加参考；
真实 K 的浮点参考应控制 TF32、累加精度与舍入，不能把其误差误算为 kernel 错误。
不能只复用候选自己的 remap/pack 代码生成期望值，否则共同错误仍可能漏检。

### 2. `calc_diff` 与 `rel_l2` 不等价

`deep_gemm/testing/numeric.py:5-11` 的指标为：

```text
calc_diff(x,y) = ||x-y||² / (||x||² + ||y||²)
rel_l2(x,y)    = ||x-y|| / ||y||
```

当 x 接近 y 时，前者约等于后者的平方除以 2。阈值不能互相复制；
原说明里的 `1e-2` / `2e-2` 只能算未校准占位值。
统一报告 finite 检查、max_abs、rel_l2、calc_diff，并按专家/有效行统计局部错误，
避免全局均值淹没一个专家算错的问题。显式定义全零参考和 NaN/Inf 的处理。
校准按运算阶段、参考层级、K、量化参数和数据分布进行，不能只按 decode/prefill。

`assert_bitwise_equal` 可用于重排、pack、确定性回归；不能替换
`test_mxfp4_int8_sm80.py:71` 的 `exact_rate_all == 1.0`，后者验证重映射数值语义，
并非比较两个待测字节张量。跨算法的 GEMM 也不应无条件要求逐位一致。

### 3. `count_bytes` 统计存储体积，不能推导实际 HBM 流量

`deep_gemm/testing/numeric.py:14-21` 只累加 `numel * element_size`，不会识别
路由命中的专家、重复使用、存储别名、padding 或缓存命中。

现有 `deepseek_v41_moe_bandwidth_probe.py:110-136` 按
`(W13 + W2 每专家字节) * tokens * topk` 计算；路由 ID 取模后可以重复，
因此推广到多 token 时不能继续解释为“每个专家只读一次”。
该文件 216-222 行还把 W13+W2 总字节用于单独 W13/W2 的带宽，
并以总字节的一半估计 W13；实际 W13 和 W2 的权重体积约为 2:1。
这些口径应先修正，再提升到共用设施。

建议分别输出：

- `resident_bytes`：模型或测试权重的分配体积。
- `active_weight_bytes`：本次命中专家的 packed weight、shift2、channel scale，去重计数。
- `logical_io_bytes`：明确定义的理想输入/输出与元数据流量；GB/s 标为估算有效带宽。
- `m_tiles`：`sum_e ceil(m_e / block_m)`；作为调度/重读风险的解释量，不冒充实测 HBM 字节。
- `useful_flops`：单 GEMM 为 `2 * sum_e(m_e) * N * K`；padding 膨胀单独报告。

延迟和相对生产基线的加速比应作为验收依据。不能仅比较 GB/s：改变表示或融合后，
更快的实现可能搬运更少字节，GB/s 数值反而下降。

### 4. 扫描空间应服从生产接口

`vllm/model_executor/layers/fused_moe/experts/mxfp4_int8_sm80.py:122-163` 限制
BF16 激活、H=5120、I=2304、top-k=3/6，激活格式为 Standard。
CUDA dispatch 在 `csrc/libtorch_stable/moe/mxfp4_int8_sm80/mxfp4_int8_moe_entry.cu:496-499`
也只实例化 top-k 3/6；263 行要求 K 能被 128 整除。

因此不能直接加任意 dtype/topk/masked 的乘积，更不能把 DeepGEMM 的 128 行对齐
硬编码为这里的 MoE 对齐。当前 routed metadata 的 padding 由 block_m 决定。
非整除形状可以用于 pack 层测试，但不自动成为生产 GEMM 的合法正例。
masked backend 若作为未来候选，需要单独 adapter 和转换成本核算。

最有价值的新增轴是每专家实际 token 数 `m_e`：同样的 M/E/topk，均匀路由与热点
集中路由可以有完全不同的有效专家数、padding 和 grid。
现有 tile 基准 `deepseek_v41_mxfp4_int8_sm80.py:281-288` 主要是均匀随机分数 top-k
与相等路由权重，尚不足以覆盖这些行为。

### 5. 现有读带宽对照有价值，但不构成严格性能上界

`deepseek_v41_moe_slab_read_control.py:81-96` 自行分配 packed slab；只读权重，
不含 shift2、scale、激活和输出。它复现专家 slab 的组织方式，并没有与 GEMM
共享同一批实际分配地址，不能直接称为全路径“同地址、同流量”。

可复用的对照阶梯已经包括：

1. slab 连续读：`deepseek_v41_moe_slab_read_control.py`。
2. GEMM 形状的跨步读：`deepseek_v41_moe_pattern_control.py`。
3. 读 + nibble decode：`deepseek_v41_moe_decode_control.py`。
4. 生产 grouped GEMM：`deepseek_v41_mxfp4_int8_sm80.py`。

把它们接到共享输入与路由后，比单纯增加一个带宽 helper 更有价值。
差值用于提出瓶颈假设，不能直接当作“可消除开销”或承诺的加速空间；
控制 kernel 的调度、归约和指令结构也与生产不同。

### 6. Graph 计时还需要缓存和统计契约

本仓库已有 `Bench`/`ArgPool` 和专用 Event 包围 Graph replay。
DeepGEMM `bench.py:9-16` 在 warmup 前只清一次 L2，并非每个样本冷缓存。
本仓库重复固定指针的 Graph 同样不能默认称为冷缓存基准。

保留 warm-fixed 与 rotating-experts/working-set 两种明确模式；必要时加入缓存驱逐
对照，并记录工作集大小。仅在整个包含多个调用的 Graph 前清一次缓存，也不能让
图内每次调用都变冷。编译、分配、数据生成和参考校验放在计时区外。
记录 GPU UUID/PCI、时钟/功耗状态、其他 GPU 任务、seed、版本、命令和环境变量；
候选与基线交替运行，保存原始样本和 p50/p90。
每个 Graph 批量耗时除以调用数后的 p90 应标为“批均摊延迟分位”，不是单调用尾延迟。

既有 `vllm170-dsv41f-decode-ledger-2026-09-18.md:34-48` 记录 CMP170HX 的
CUPTI/NCU 不可用。本轮未重测工具状态；因此方案不依赖 CUPTI，使用 Event + Graph。
性能 CI 需要固定、空闲设备及噪声带；一般 CI 保留正确性，不能直接写 `GB/s >= baseline`。

### 7. 低依赖成立于抽取函数之后，不能原样导入

`deep_gemm/__init__.py:16` 无条件加载 `_C`，`utils/__init__.py:1-3` 还通过
`layout.py:1-9` 引入扩展符号。因此 `from deep_gemm.testing import ...` 不等于
只执行纯 Python helper；必须把需要的函数抽取到本地并重写导入。
`runtime.hpp:49-55` 的 alignment 是可写状态，128 是默认值，不能用常量无损替代
整个 generators 的 getter/setter 契约。

legacy 的 `m_grouped_gemm.py:6-7`、`tune_options.py:2` 仍依赖 `_C`，
且 `tests/test_legacy.py:42` 使用 `bench_kineto`。若试用该 BF16 Triton 基线，
还需要解除导入依赖、适配 routed 数据和 alignment，并替换计时；
“不需要写 CUDA”不能解释为“原样导入即可无扩展运行”。
它只能先作为算法性能对照，不能直接作为 MXFP4/INT8 的同语义正确性参考。

## 推荐的最小落地范围

### 第一批：建立可相信的量尺

- 扩展现有 `deepseek_v41_mxfp4_int8_sm80.py`，增加专家数、top-k 3/6、路由模式、
  独立参考校验和统一结果 schema；继续复用现有 `make_gemm_call`。
- 把必要的 case、reference、timing/report helper 放在 `benchmarks/kernels/`；
  不为每种试验再建一个脚本，也不先建任意 dtype/layout 的通用框架。
- 优先让 W13、W2、reduce 的非零输入逐一通过参考；现有 top-k3 测试
  `test_mxfp4_int8_sm80.py:116-203` 主要验证 dispatch 和手工注入 routed 输出后的 reduce，
  没有验收非零 GEMM 结果，是明确的补测切口。
- 在现有 quantization 测试文件中补路由索引、pack、scale 与尾块行为；
  不让 pytest 依赖 benchmark 脚本。性能基准继续放 `benchmarks/kernels/`。
- 先校准误差与噪声，输出 JSONL，再设验收阈值；没有证据的阈值保留为未校准状态。

### 第二批：围绕生产负载扩面

| 维度 | 首轮内容 |
| --- | --- |
| 阶段 | W13、W2、reduce、完整 routed expert MLP |
| 形状 | W13: N=4608/K=5120；W2: N=5120/K=2304 |
| token | M=1、小批 decode、DSpark 验证块、代表性 prefill；具体频率用运行记录确定 |
| 专家 | 小 E 正确性集合 + E=384 生产规模；top-k=3/6 分开 |
| 分布 | 均匀、多个 token 共享热点专家、稀疏散布、空专家、打乱 slot 顺序、真实路由回放 |
| tile 边界 | 构造 m_e=15/16/17、63/64/65、127/128/129，而非仅枚举全局 M |
| 数值 | 零/非零、正负值、非单位 scale、指数跨度、非均匀路由权重 |
| 缓存 | 固定专家热复用、轮换专家工作集，按同模式成对比较 |

正常路由保持单 token 内 top-k 专家不重复；同一专家跨 token 重复是重要负载。
特殊重复 slot 可另列接口压力测试，不能混入生产路由统计。
小集合做笛卡尔积，生产规模用定向 case，避免无意义组合和全量 BF16 参考显存膨胀。

### 第三批：候选晋升

按“独立正确性 → 隔离 GEMM → 完整 routed MoE → 无插桩模型 A/B”推进。
完整 MoE 必须计入 routing cast/align、两次量化、激活和加权 reduce；
候选若引入 pack/gather/scatter，部署路径中每次付出的成本也要计入。
比较显存工作区、初始化重排成本、TTFT/ITL/吞吐和输出质量，不能只报单 kernel 加速。

本仓库已有反例：`vllm170-dsv41-pp5-decode-profile-2026-09-12.md:446-455`
记载 16x64 孤立 GEMM 改善，却使当时端到端吞吐从 36.270 降到 35.284 tok/s。
这支持保留最终服务门禁；旧数值不能直接作为当前服务 baseline。

## 借鉴清单与投入判断

| DeepGEMM 要素 | 判断 |
| --- | --- |
| case 生成、有效行 mask、参考与容差随 case 绑定 | 高价值，按本仓库路由契约重写 |
| calc_diff / bitwise mismatch 诊断 | 小而有用，作为补充指标；不机械替换现有断言 |
| count_bytes | 可作存储体积工具，价值低于正确的路由流量模型 |
| bench / bench_kineto 整体复制 | 首轮不需要；现有 Event/Graph 足够承载测量 |
| FP8/FP4 枚举器和对齐 shim 整体搬运 | 不建议；其架构约束与生产格式不同 |
| BF16 legacy grouped GEMM | 可作为后续算法基线候选；需独立适配、验证和显存预算 |

第一批值得立即安排。原文“1.5–3 人日”可作小范围设施雏形的估算，
不能包含全部独立解包参考、真实路由采集、全链路 GPU 校准和稳定性能 CI 的承诺。
本轮没有给出加速收益估计；设施价值在于让下一轮 kernel 重写的结论可复现、可证伪。
