# DSV41F SM80 MoE grouped GEMM：验收设施实现报告

> 本文保留初版实现与测量记录。后续审查发现 scale、状态汇总、轮换验收、
> 复现和数值容差等缺口；当前实现与验收边界以
> [审查修复说明](vllm170-sm80-moe-harness-fixes-2026-09-30.md) 为准。
> 本文历史 PASS 与瓶颈推断不能扩展为非单位 scale、全部轮换或 Graph replay 已通过。

日期：2026-09-30。范围：实现并运行第一批验收设施，不重写 kernel、不切换后端、
不改生产默认值。

任务书：[vllm170-sm80-moe-harness-deepseek-execution-guide-2026-09-30.md](vllm170-sm80-moe-harness-deepseek-execution-guide-2026-09-30.md)
（本轮读到 16:36 版；16:45 该文件仅新增第 5-7 行指向
[vllm170-dsv41f-sm80-tilelang-gemm-plan-2026-09-30.md](vllm170-dsv41f-sm80-tilelang-gemm-plan-2026-09-30.md)
的下一阶段链接，并注明"不改变本文第一批任务的完成边界"。该 tilelang 文档与
评审、任务书两份文档都不是本轮产物，本轮未读取其内容以外的任何修改、未写入。）
评审输入：[vllm170-sm80-moe-grouped-gemm-harness-review-2026-09-30.md](vllm170-sm80-moe-grouped-gemm-harness-review-2026-09-30.md)

审计版本：vllm170 `f9c6a54b064cd1178828bdf7fe73f283908223c4`（未变）；
DeepGEMM `057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7`（未变，只读参考）。

结果目录：`artifacts/moe-grouped-gemm-harness-20260930-164205/`

## 1. 完成状态

本轮 CPU 与 GPU 两部分都已完成，**GPU 验收已实际执行**，不是"待执行"。

| 项目 | 状态 |
| --- | --- |
| P0 数值与接口契约 | 完成，见本文第 4 节 |
| P0 独立参考（解包/分组 GEMM/reduce/指标/schema） | 完成 |
| P0 打包格式手工锚点 | 完成，CPU 与 GPU 两侧独立实现都对手写字节一致 |
| P1 `--experts/--topk/--route-pattern/--seed/--check-reference/--cache-mode/--output-jsonl` | 完成并实跑 |
| P1 非零 GEMM / 路由 / 边界 / reduce / 写回测试 | 完成，49 项测试在真机 SM80 上全过 |
| P2 字节口径修正与 schema 升级 | 完成，见第 7.2 节 |
| P2 生产形状基线（三次 round） | 完成 |
| **未做**：完整 routed MLP、模型 A/B、BF16 legacy 对照、新 kernel | 按任务书要求留给后续 |

## 2. 改动文件

| 文件 | 目的 |
| --- | --- |
| `benchmarks/kernels/deepseek_v41_moe_harness.py`（新增） | 离线 helper：打包权重解码、路由构造与去重字节核算、独立参考、误差指标、Event+Graph 计时、JSONL schema、CPU 自检。不导入 `deep_gemm`，纯指标/解包部分在无 CUDA 下可导入运行 |
| `benchmarks/kernels/deepseek_v41_mxfp4_int8_sm80.py`（改） | 主扫描入口：新增专家数/top-k/路由模式/seed/参考校验/缓存模式/JSONL；保留 `make_gemm_call`、`--output`、`--rotate-experts` 兼容 |
| `benchmarks/kernels/deepseek_v41_moe_bandwidth_probe.py`（改） | 修正 W13/W2/联合字节口径，改用去重活跃专家集，schema 升级到 2 并写明旧字段映射 |
| `tests/kernels/quantization/test_mxfp4_int8_sm80.py`（改） | 新增打包格式手工锚点、pack 尾组、非零 W13/W2、路由打乱、热点复用、空专家、block_m 边界、写回完整性、reduce 语义；把 tile 表测试参数化 |

`git diff --stat`：3 个文件，+1654 / −268。改动全部落在
`benchmarks/kernels/` 与 `tests/kernels/quantization/`，**没有任何生产文件、
默认值、全局环境变量、backend 优先级或公共 dispatch 被修改**（
`git diff --name-only` 过滤结果为空）。新增 helper 不导入 `deep_gemm`，
也没有让 pytest 通过 `sys.path` 依赖 benchmark 脚本：测试自带一套独立的标量
编解码器，与 harness 的向量化实现互为交叉验证。

## 3. 环境与命令

设备选择：`nvidia-smi` CLI 在本机不可用（无法与驱动通信），但 NVML 可用，因此
用 `pynvml` 枚举。7 张卡中只有 1 张空闲，按 UUID 绑定：

```text
index 1  GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216  PCI 00000000:02:00.0
         NVIDIA Graphics Device (CMP 170HX)  cc 8.0  70 SM  63.4 GiB  32 MiB L2
         630 MiB used, 0% util, no compute processes
index 2-6  PP5 生产服务 (PID 260465-260469, 各 60-63 GiB) —— 未触碰
index 0    RTX 3080 Ti Laptop (cc 8.6，不在范围)
```

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216
```

驱动 610.43.03，torch 2.13.0+cu130，Python 3.12.12。全部命令、退出码与环境记录见
`artifacts/moe-grouped-gemm-harness-20260930-164205/{env,gpu,cpu}/`，包括
`env/gpu-selection.txt`（含生产进程 PID，证明未占用）、`gpu/*.command`、
`cpu/cpu-gate.log`、`cpu/pre-commit.log`。

实测该卡 FP64 matmul 9.64 TFLOPS、FP32(TF32) 52.0 TFLOPS，所以 M=2048 的独立
参考只需约 90 ms，参考始终留在计时区外。

## 4. 数值与接口契约（P0）

从 `mxfp4_int8_moe_entry.cu` / `mxfp4_moe_cutlass_core.cuh` 与生产 Python 层确认的
契约（评审文档与我的独立审计一致）：

- 形状：W13 `N=4608, K=5120`；W2 `N=5120, K=2304`；训练/服务仅 top-k 3 与 6；
  tile 只有 `16x64 / 16x128 / 64x128 / 128x128` 四种可由 entry 到达。
- **`out` 参数不参与计算**：`params_C` 的 `ref_C` 是 `nullptr`，
  `IdentityInt32OutputOp::is_source_needed()` 返回 false。两个 GEMM 的唯一目的地都是
  **`routed_out`**，按 slot 行索引；`out` 只被 `run_reduce` 写。W13 与 W2 的输出
  都是 slot 序，这一点是本轮独立参考能对齐的关键。
- A 源行：W13 是 `slot / topk`（token 行），W2 是 `slot`。`a_scale[source_row]`。
- 权重：`W_int8[e,n,k] = ±mag[code&7] << shift`，`mag=(0,1,2,3,4,6,8,12)`，
  `shift` 是每个 `(n, K/32)` 块的 2 bit 字段，取值域 `[-96, 96]`。
  与 `channel_scale` 相乘即为 GEMM 实际乘的浮点权重；E2M1 的 ×0.5 已折进
  `channel_scale`，不能再除一次。
- epilogue 顺序：`rne_bf16( float32(acc_int32) * a_scale[src] * channel_scale[e,n] )`，
  **先乘 a 再乘 channel**，只舍入一次。
- 累加是 **INT32 饱和 MMA**。`|acc| ≤ 127 * 96 * 5120 = 62,423,040 < 2^31−1`，
  低 34 倍，因此饱和不可达；这一界已写进 harness 自检与 pytest。
- reduce：`out[token,n] = rne_bf16( Σ_k fp32(routed[token*topk+k, n]) * topk_weights[slot] )`，
  k 升序 FP32 累加、末尾一次 BF16 舍入；`topk_weights` 用**扁平** slot 索引。
  路由权重只在这里出现，不在任一 GEMM 里。
- 混合精度契约：kernel A 为 INT8、A scale 与 channel scale 为 FP32、输出 BF16；
  **对外生产 MoE 接收 BF16 激活并内部量化**，本轮的 benchmark 直接喂 A8，
  不能称为"完整 MoE"。

## 5. 正确性证据

### 5.1 手工打包锚点（不经过生产 pack）

`deepseek_v41_moe_harness.py` 与测试文件各自用**字面字节**构造 K=128、N=8 的
packed fixture，并各自独立推导 24 个 `(n, k) → int8` 期望值，覆盖 lane 置换
（n=0 的值分别经 lane 0/1/2/3 写入）、一个字节的高低两个 nibble、正零、负零、
以及四个 shift 字段（`n=0` 的四个 K 块 shift 为 0/1/2/3，`n=1` 为 3/3/0/0）。
两份实现都对手写字节一致，`min = −96` 达到可表示下界；此外还有
`pack → decode` 的 round-trip（N=7/8/9/11 × K=32/64/128）作为补充锚点。

round-trip 只做"生产 pack → 自己 unpack → 原值"会留下双方同错的可能，所以它不是
唯一锚点；真正独立的是上面这份手写字节。

### 5.2 CPU 参考检查

- harness 自检 37 项全过（`cpu/harness-self-check.json`），含指标定义、
  `nonzero-vs-zero-reference`、NaN 判失败、去重字节核算、跨 block_m 的
  `m_tiles`、INT32 界、reduce 语义。
- `CUDA_VISIBLE_DEVICES='' pytest tests/kernels/quantization/test_mxfp4_int8_sm80.py`
  → **25 passed, 24 skipped**（24 项是 SM80 专属，CPU 下正确跳过）。
- 任务书第 6.1 节要求的 CPU 算术验算已纳入自检：固定命中专家、M 从 1 增至 4 时
  `active_weight_bytes` 不变而 `useful_flops` 随 slot 数线性增长；
  `m_tiles` 在 15/16/17、32/33 处按 block_m 正确跨档；W13+W2 字节严格可加。

### 5.3 真机 SM80 GPU 正确性

`pytest` 在 `GPU-f385e0d6…` 上：**49 passed, 0 skipped**（26.9 s）。覆盖：

- 非零 W13（top-k 3/6 × M 1/4）与 W2（top-k 3/6），权重稀疏编码、期望值用标量
  手算；
- **W2 的 A 行契约**：`a_q[slot,0]` 严格递增，并断言"用 `slot//topk` 读会得到不同
  值"，因此该用例真的在区分 slot 行与 token 行，而不是自证；
- 路由打乱：8 个专家各带一个互不相同的解码值，任一专家错位都会暴露；
- 热点：9 个 token 全部命中 {0,1,2}，专家 0 被复用 9 次；
- 空专家：专家 0/1 权重全零，其 slot 必须精确为 0，其它 expert 必须非零；
- block_m 边界：m_e = 15/16/17（block_m 16）、63/64/65（64）、127/128/129（128），
  并断言 `num_post_pad % block_m == 0`；
- 写回完整性：`routed_out` 尾部预填 sentinel，验证所有有效 slot 都被写、
  填充 slot 一个字节都没动；
- reduce：非均匀、和不为 1 的权重，含一个零权重 slot；断言等于逐 slot 加权和、
  且**不等于**重新归一化后的结果。

`--check-reference` 的端到端扫描（`gpu/*.jsonl`，共 **144 行，72 条参考检查全 pass**）：

| 用例 | W13 | W2 | reduce |
| --- | --- | --- | --- |
| E8 top-k3/6 × M1/4 × uniform/hot | 逐位相等（mismatch 0） | 逐位相等 | 98.7–99.4% 逐位相等 |
| E384 k6 M1 warm-fixed ×3 round | 逐位相等 | 逐位相等 | 98.8% 逐位相等 |
| E384 k6 M1 rotating ×3 round | 逐位相等 | 逐位相等 | 98.6% 逐位相等 |
| E384 k3 M1 ×7 round | 逐位相等 | 逐位相等 | 99.2% 逐位相等 |
| **E384 k6 M2048** ×3 round | 逐位相等 | 逐位相等 | 98.8% 逐位相等 |

**两个 GEMM stage 在全部 72 次检查里逐位相等**，包括 M=2048 的生产 prefill 形状。
这同时验证了打包布局/移位/符号、W13 token 行与 W2 slot 行的区别、写回位置、
两次缩放顺序与 BF16 舍入。

### 5.4 reduce 的残差是什么（阈值校准依据）

reduce 不可能要求逐位相等：`1/topk` 在 FP32 下不可表示，且 kernel 把
`acc + value*weight` 收缩成 FFMA，与"先乘后加"的累加器相差约一个 FP32 ulp。
只有当精确和落在 BF16 舍入边界的一个 ulp 之内时，两者才会给出相邻的两个 BF16 值。

因此阈值不是拍出来的绝对 `max_abs`，而是一个**结构性判据**：

```text
(|candidate − reference| − spacing_bf16(max(|c|,|r|))) / term_scale  ≤  2**-20
term_scale = max_n Σ_k |slot_k × weight_k|
```

减掉一个 BF16 间距后剩下的就是两次 FP32 累加的差异，除以项的量级即与尺度无关。
`2**-20` 对 top-k ≤ 6 的 FP32 累加是宽松上界（理论量级 `n·2**-24`），而任何路由或
加权错误都会让结果移动 term_scale 的 O(1)，因此判据仍有约 10⁶ 的区分度。

实测（`gpu/reduce-rounding-diagnostic.json` 与 72 条参考检查）：残差最大
**4.0e-08**，即预算的 1/24；绝大多数元素逐位相等。**near-zero 元素上
`max_bf16_steps` 可以到 244**——这不是错误，而是 top-k 相消后结果远小于项量级，
该处 BF16 间距极小；把它们按绝对值除 term_scale 后全部在 1e-8 量级。

> 校准过程中先写过一版绝对 `max_abs` 阈值和"必须相邻"阈值，都被实测数据否掉了：
> 前者随激活量级漂移，后者在相消时失效。最终判据同时经得起这两种情形。
>
> 另外发现：`torch` 的 fp64→bf16 转换会先经过 fp32。用 FP64 参考再转 BF16 会在
> 并列点上双重舍入，制造出"内核算错"的假象。参考的默认累加精度因此是 FP32
> （与 kernel 结构一致），FP64 只作诊断且不参与判定。

## 6. 生产形状基线（P2）

`p50/p90` 为 Event 包围 CUDA Graph replay、除以图内调用数的批均摊分位；字节为
按活跃专家去重后的**逻辑**权重字节。`active_weight_gb_s` 是逻辑有效带宽，
**不是实测 DRAM/HBM 带宽**。

| 用例 | stage | tile | round p50 (µs) | 中位 | p90 | 权重字节 | 逻辑 GB/s | useful TFLOPS | m_tiles |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E384 k6 M1 warm | w13 | 16x128 | 119.53/119.81/119.78 | **119.78** | 155.26 | 71,995,392 | 602.3 | 2.37 | 6 |
| E384 k6 M1 warm | w2 | 16x128 | 57.40/57.37/57.40 | **57.40** | 57.75 | 36,065,280 | 628.4 | 2.47 | 6 |
| E384 k6 M1 warm | reduce | — | 2.69/2.69/2.69 | **2.69** | 2.71 | — | — | — | 6 |
| E384 k6 M1 rotating | w13 | 16x128 | 120.78/121.03/121.01 | **121.01** | 121.04 | 71,995,392 | 596.1 | 2.34 | 6 |
| E384 k6 M1 rotating | w2 | 16x128 | 57.99/57.98/58.65 | **57.99** | 58.65 | 36,065,280 | 622.0 | 2.44 | 6 |
| E384 k3 M1 warm | w13 | 16x128 | 85.25/85.30/66.56/85.25/80.44/85.27/76.34 | **85.25** | 85.86 | 35,997,696 | 422.3 | 1.66 | 3 |
| E384 k3 M1 warm | w2 | 16x128 | 31.05/31.16/31.08/30.85/30.82/30.85/30.85 | **30.85** | 31.18 | 18,032,640 | 580.7 | 2.28 | 3 |
| **E384 k6 M2048 warm** | w13 | 64x128 | 11384.83/11427.58/11487.59 | **11427.58** | 11529.93 | 4,607,705,088 | 404.7 | **50.93** | 384 |
| E384 k6 M2048 warm | w2 | 64x128 | 6068.94/6099.87/6111.28 | **6099.87** | 6162.33 | 2,308,177,920 | 380.3 | **47.77** | 384 |
| E384 k6 M2048 warm | reduce | — | 95.23/95.23/95.23 | **95.23** | 95.54 | — | — | — | 384 |

整段合计（每层每次调用）：M1 top-k6 **179.87 µs**；M1 top-k3 **118.07 µs**；
M2048 top-k6 **17.62 ms**。

M=2048 的选取理由：它是既有运行记录里服务的 `--max-num-batched-tokens 2048`
（`artifacts/dsv41-bench-20260929/REPORT.md:33`），此时生产 selector 按
`ceil(2048·6/384)=32` 选出 block_m 64，与 decode 的 16x128 是不同档；
本轮不假设旧服务参数仍然生效，只是复现该记录的口径。

### 6.1 缓存模式：固定与轮换只差 1%

| stage | warm-fixed 工作集 | rotating 工作集 | p50 变化 |
| --- | ---: | ---: | ---: |
| w13 | 103.1 MiB（周期内 6 个专家） | 4122.2 MiB（周期 40 次调用、240 个专家） | **+1.03%** |
| w2 | 103.1 MiB | 4122.2 MiB | **+1.03%** |

固定 6 个专家的 103 MiB 工作集**已经大于 32 MiB L2**，所以 warm-fixed 并没有带来
缓存收益，把工作集放大 40 倍也只慢 1%。这说明当前 decode GEMM 的瓶颈不是
L2 复用，而是 DRAM 延迟/占用率本身，与带宽探针的读对照结论一致。

轮换模式按周期归一化：`active_weight_bytes` 用并集字节除以周期，与延迟分母同口径。

### 6.2 噪声与不确定结论

同一条命令重复 3–7 次：

- 生产 decode（k6/M1）、M2048、k3 的 w2 与 reduce：spread **0.0–1.3%**，结论可用。
- **k3/M1 的 w13：spread 22.0%（66.56 … 85.30 µs，7 次）**，标
  `measurement_inconclusive`。这一档只有 3 个 slot、3 个 m_tile，样本太少，
  本轮**不**据此对该形状下任何结论，也不通过加大扫描把它盖过去。

任务书禁止"仅凭一次 p50 或 `GB/s >= baseline` 判通过"，因此本轮只建立 baseline，
没有候选、没有对比、没有加速结论。轮换 vs 固定那 1.03% 也没有超过判据所需的
噪声带以外多少，同样不下结论。

## 7. 修正与发现

### 7.1 一个先存的测试错误（不是本轮引入）

`test_static_tile_table_respects_sparse_expert_occupancy` 在审计 HEAD 上就是**失败**的：
它断言 `select_mxfp4_int8_sm80_tile(1024, 6, 256) == (16, 128)`，而现行规则是
"覆盖 `ceil(M·topk/E)` 的最小 block_m 候选"，`ceil(1024·6/256)=24` → `(64, 128)`。
该断言描述的是已经作废的 `_TILE_TABLE` 查表行为（`_TILE_TABLE` 现为死代码）。
本轮把该测试改为按现行规则参数化，并在第 5.2 节的计数里如实反映。
**没有为了让测试过而改生产 selector。**

### 7.2 带宽探针的字节口径错误（已修，schema 1→2）

原实现把 W13+W2 的总字节用于每一个 stage，并用"总字节的一半"估 W13。用本轮同一批
实测延迟回算旧口径的偏差：

| 字段 | schema 1 | schema 2（正确） | 偏差 |
| --- | ---: | ---: | ---: |
| w13 effective GB/s（E=384） | 796.1 | 530.4 | **+50.1%** |
| w2 effective GB/s（E=384） | 1866.1 | 622.8 | **+199.6%** |
| `w13_only_gb_s`（已删除） | 398.1 | 530.4 | **−25.0%** |

W13:W2 的实际体积比是 **1.9963 : 1**（11,999,232 : 6,010,880 字节/专家），所以
"各占一半"在方向上恰好抬高了 W13。schema 2 写明 1→2 的字段映射，旧结果不能与新
结果合并。

修正后的读对照（同逻辑字节、**不同地址**的全新连续缓冲）：

| E | GEMM 对 (w13+w2) | 连续纯读对照 | 对照/GEMM |
| ---: | ---: | ---: | ---: |
| 384 | 189.34 µs | 66.79 µs（1618 GB/s） | **2.83×** |
| 64 | 189.06 µs | 66.84 µs（1617 GB/s） | **2.83×** |

该对照不共享生产地址、也不共享访问模式，只能作为"DRAM 还能快多少"的上界提示，
不能读成"可消除开销"。

另外，E=384 与 E=64 的活跃专家数、m_tiles、字节数完全一样，但 w13 的 p50 是
135.7 与 121.0 µs（差 12%）。这与分配体积/页表局部性有关，本轮只记录现象，
不据此下结论。

### 7.3 修复了一个会把失败汇总成 PASS 的实现缺陷

`run_case` 里 `rows = []` 被第二次初始化，把已经算好并通过/失败的
参考检查行整个丢掉，JSONL 里只剩计时行、且计时行的 `correctness_status` 来自
`--check-reference` 这个开关而不是实际结论。这正是任务书第 9 节问的
"一条失败/跳过/未校准结果会不会被汇总成 PASS"。

已修：参考检查行保留在同一 JSONL，计时行只有在参考检查通过后才会产生（失败即
提前返回并不计时），并在计时行里带上 `reference_rows_in_this_run`。
另加 `--summary-json` 原子写出汇总、保留全部失败与跳过项、`all_passed` 为假时
进程以非零码退出。

### 7.4 上游无法合并（沿用既有结论）

DeepGEMM 不接受 SM80（issue #236），所以 SM80 路径只能走 fork。本轮完全没有需要
这条结论——设施全部落在 vllm170 自己的 `benchmarks/` 与 `tests/` 内，
不依赖 DeepGEMM 包（`deep_gemm/__init__.py` 会加载 `_C`，因此没有导入它）。

## 8. 验收清单逐项回答

| 问题 | 回答 | 证据 |
| --- | --- | --- |
| 所有 tile 一起算错，独立参考能否发现？ | **能**。参考从实际 packed 字节解码 INT8 权重，不调用生产 pack/remap 代码；手工字节锚点先证明解码语义，再与 kernel 输出比。原 tile 等价检查只做 tile 间逐位一致，确实看不见"一起错"，现在两种判据并存 | 第 5.1、5.3 节 |
| W2 误用 token 行、或路由 scatter/reduce 错位，非零测试能否发现？ | **能**。W2 用例让 `a_q[slot,0]` 严格递增并断言 token 行会给出不同值；scramble 用例给 8 个专家各一个不同解码值；热点与空专家用例分别覆盖跨 token 复用与未命中专家 | 第 5.3 节 |
| 相同专家跨 token 复用时字节数是否仍合理？ | **合理**。字节按去重活跃专家集计算：E8-k3-M4 的 12 个 slot 命中 6 个专家、直方图 `[1,1,2,3,3,2]`，字节等于 6 个专家；E8-k6-M4 的 24 slot / 8 专家同理。`useful_flops` 随 slot 数走，`m_tiles` 随 block_m 走 | 第 5.2、`gpu/routing-and-bytes-evidence.txt` |
| 固定专家与轮换专家的计时是否明确区分？ | **是**。`cache_mode` 是独立字段，轮换模式记 `rotation_period` 与并集 `working_set_bytes`，字节分子按同一周期归一化；两者工作集相差 40 倍而延迟只差 1.03% | 第 6.1 节 |
| 一条失败/跳过/未校准结果会不会被汇总成 PASS？ | **不会**。见第 7.3 节：该缺陷本轮被实际发现并修掉；未校准阈值返回 `uncalibrated` 而不是 `pass`，`classify` 在指标缺失时也返回 `fail`。当前 72 条参考检查全为 `pass`，且每条都带独立的 `correctness_status` 与 `skip_or_failure_reason` | 第 7.3 节 |
| 是否严格限于 dsv41f SM80 grouped GEMM，没有改其他模型分派、默认参数或执行路径？ | **是**。改动只有 `benchmarks/kernels/` 3 个文件与 `tests/kernels/quantization/` 1 个文件；`git diff --name-only` 无生产文件；未改 `vllm/`、未改 `csrc/`、未改 env 默认值、未改 backend 优先级。`git status` 中其余未跟踪文档不是本轮产物 | 第 2、3 节 |

## 9. 下一步 kernel 假设（最多三条，均有本轮证据）

1. **decode 的 W13 不是带宽受限，而是"每专家 tile 太少"的延迟问题。**
   证据：M1/topk6 时只有 6 个 m_tile（每专家 1 个 16 行 tile），
   w13 逻辑带宽 602 GB/s，而同一批字节的连续纯读对照是 1617 GB/s（2.83×）；
   同时把工作集从 103 MiB 放大到 4122 MiB 只慢 1.03%，说明 L2 复用不是瓶颈。
   候选方向是把 W13 的 decode 拆成更多、更小的并行工作单元（例如 block_n 64，
   即既有 `VLLM_DEEPSEEK_V4_SM80_MOE_DECODE_BLOCK_N` 开关），而不是减少字节。
   注意：`vllm170-dsv41-pp5-decode-profile-2026-09-12.md:446-455` 记录过 16x64
   孤立 GEMM 改善却让端到端吞吐下降，所以该候选必须过最终服务门禁。

2. **W13 与 W2 的时间比（119.8 : 57.4）与字节比（2.00 : 1）不一致，W2 的效率更高。**
   W2 达到 628 GB/s、W13 只有 602 GB/s，在 decode 形状下 W13 用了 2 倍字节但只
   拿到 2.09 倍时间。两者的区别是 K（5120 vs 2304）与 N（4608 vs 5120）。
   可能的解释是 W13 的 K 更长、每 CTA 的 k 迭代更多而 m_tile 只有 1 行。
   可验证的做法：在**不改变数值语义**的前提下单独扫 K 方向的流水线深度
   （`MXFP4_MOE_PIPELINE_STAGES` 当前为 3），并保持 16x128 不变，用本轮同一
   `--check-reference` 门禁验收。

3. **M=2048 prefill 已经从延迟受限转为算力/L2 受限，优化目标与 decode 不同。**
   证据：同一个 kernel 在 M2048/block_m 64 时达到 50.9 TFLOPS 但逻辑带宽掉到
   404.7 GB/s（decode 是 602 GB/s、2.37 TFLOPS）。此时 m_tiles=384、
   每专家约 32 行，一个 m_tile 覆盖整段专家，字节不再被重复读。
   若按第二条把 decode 的并行度提高，需要单独确认 prefill 档不被拉低——
   两者的 tile 已经在生产 selector 里分开（16 vs 64/128），应作为两个独立的
   验收门禁，而不是一个"平均带宽"指标。

三条都只是**假设**，本轮没有实现任何候选，也没有测量任何候选的收益。

## 10. 本轮之后的边界

按任务书，完整 routed MLP（含 activation 与第二次 A8 量化）、模型质量评估、
BF16 legacy 对照、服务 A/B 与新 kernel 都**不在本轮**，也没有启动。
本轮交付的是量尺：72 条独立参考检查（两个 GEMM stage 全部逐位相等）、
49 项真机测试、修正后的字节口径、以及一组带噪声带的生产 baseline。
下一阶段按任务书第 5-7 行指向的 TileLang 方案验证候选时，可以直接复用
`--check-reference`、`--output-jsonl` 与本文第 6 节的 baseline。
