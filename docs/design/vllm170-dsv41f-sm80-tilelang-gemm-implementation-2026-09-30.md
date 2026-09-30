# DSV41F SM80 grouped GEMM：TileLang 重写与离线 autotune 报告

日期：2026-09-30。执行对象：
[vllm170-dsv41f-sm80-tilelang-gemm-plan-2026-09-30.md](vllm170-dsv41f-sm80-tilelang-gemm-plan-2026-09-30.md)。
承接上一轮的验收设施
[vllm170-sm80-moe-harness-implementation-2026-09-30.md](vllm170-sm80-moe-harness-implementation-2026-09-30.md)。

审计版本：vllm170 `59d0aaeabe64664ad1b7fdd55dbf18fab64975ab`（本轮起点，
本轮改动尚未提交）。设备：`GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216`
（index 1，CMP 170HX，cc 8.0，70 SM，64 GiB，空闲）。TileLang 0.1.12，
torch 2.13.0+cu130，驱动 610.43.03。生产服务（PID 260465–260469）未触碰。

结果目录：`artifacts/dsv41f-tilelang-gemm-20260930-172532/`。

## 1. 结论

**结论：不接入。** TileLang 候选在全部已测形状上比生产 CUTLASS 慢 **11%–37%**，
差距稳定、可复现，58 个通过校验的候选配置里没有一个接近基线。按方案第 5 阶段与末段
的停止条件，本轮不进入 dsv41f 专用接入评审，生产路径零改动。

但本轮把差距**定位到一个具体、可行动的局部**：

| 事实 | 证据 |
| --- | --- |
| TileLang 生成的 INT8 MMA 在本机真实可用 | SASS 中 `IMMA.16832.S8.S8` ×64，PTX `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` |
| grouped kernel 复现生产契约逐位一致 | 58 个候选配置 + 14 组 stage-4 复测，全部 candidate/baseline 与独立参考**逐位相等** |
| 去掉 E2M1 解码的同构骨架**比 CUTLASS 快 2.2–2.6×** | W13：60.57–63.74 µs vs 155.08–155.55 µs；W2：33.28–33.89 µs vs 74.24 µs（同 tile、同进程交替测） |
| 慢的全部来源是解码 | 最佳 tile 下解码占 W13 109 µs、W2 49 µs，CUTLASS 的 `__byte_perm` 解码约 42 µs 与 13 µs |

也就是说：**TileLang 的访存/流水/MMA/epilogue 骨架是赢的，输在 TileLang 目前
只能把 4-bit→INT8 展开写成逐元素标量解码。** 这是本轮最有价值的结论，也是下一次
尝试的唯一起点。

## 2. 改动文件

| 文件 | 目的 |
| --- | --- |
| `benchmarks/kernels/deepseek_v41_moe_tilelang_gemm.py`（新增） | TileLang grouped GEMM 候选：packed B 在 kernel 内解码、INT8 m16n8k32 MMA、INT32 累加、按生产次序两次缩放与 BF16 舍入、slot 写回；含 4 种解码 lowering 与正确性驱动 |
| `benchmarks/kernels/deepseek_v41_moe_tilelang_bench.py`（新增） | 候选 vs 生产基线的 A/B/B/A 交替复测；固定/轮换工作集、held-out seed、路由模式、tile 边界；计时前先用独立参考校验双方 |
| `benchmarks/kernels/deepseek_v41_moe_tilelang_autotune.py`（新增） | 离线候选搜索：逐配置「编译 → 独立参考逐位校验 → 计时」，分别记录编译/正确性/计时失败 |
| `benchmarks/kernels/deepseek_v41_moe_harness.py`（改） | 修掉 `interleaved_ab` 的两处缺陷（见第 8 节）；无生产影响 |

`git diff --name-only` 过滤后仍为空：**没有 `vllm/`、`csrc/`、默认值、环境变量或
backend 分派被修改**；新文件全部落在 `benchmarks/kernels/`。没有导入 `deep_gemm`。

## 3. 阶段 1：最小编译 spike

对真实 CMP170HX 编译并运行了一个最小 INT8 GEMM（`artifacts/.../spike/`）：
A/B 为 INT8 行主序，`T.gemm(A_s, B_s, C, transpose_B=True)`，累加器 INT32，
epilogue 为 `float32(acc) * a_scale[row] * channel_scale[col]` 后一次 BF16 舍入。

```text
bitwise exact bf16: 65536/65536
ptx  mma ops: mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
sass mma ops: IMMA.16832.S8.S8   (64 条)
```

生成代码用 `tl::ptx_ldmatrix_x4` + `tl::mma_sync<kInt8,kInt8,kInt32,16,8,32,false,true>`，
`cp.async` 双缓冲。方案要求的三点（INT8 MMA、INT32 累加、BF16 epilogue）都在真机
上验证，生成代码、PTX、SASS 均已保存。spike 脚本一并归档。

## 4. 阶段 2：单配置 grouped kernel

### 4.1 复现的生产契约

- A：INT8 `[rows, K]`，W13 的 `rows=tokens` 且源行为 `slot // topk`，W2 的
  `rows=slots` 且源行为 `slot`；`a_scale[source_row]` 为 FP32。
- B：保持生产 packed 格式 `[E, K/32, ceil(N/8), 128]` + `[E, ceil(K/32/4), ceil(N/8), 8]`
  2-bit shift，**kernel 内解码**，不存在展开后的 INT8 权重副本，packed 字节每个
  tile 只读一次（方案第 3 节要求解码计入计时区）。
- 路由元数据沿用生产 `moe_align_block_size` 的 `sorted_token_ids / expert_ids /
  num_tokens_post_padded`；`slot >= slots` 的 padding 行不写。
- 累加 INT32；epilogue `rne_bf16(float32(acc) * a_scale[src] * channel_scale[e,n])`，
  **先乘 a 再乘 channel**，只舍入一次。
- 输出写调用者预分配的 `routed_out[slot, :]`（in-place，非 `out_idx`）。

解码索引由文档化布局重新推导（不复制生产 `__byte_perm` 技巧）：对
`k_pair = kk//2`，packed 字节为
`(n%8)*16 + ((k_pair%8)//2)*4 + (k_pair//8)*2 + (k_pair%2)`，低 nibble 为偶数 k；
其逆排列有闭式 `k_pair = (j//2)*8 + t*2 + (j%2)`，这正是 `luta` 用的仿射形式。

### 4.2 kernel 结构

`T.Kernel(n_tiles, max_m_tiles)`，每个 CTA 负责一个 (expert, m_tile, n_tile) 块；
`expert_ids[by]` 为每 m-tile 的专家。流水线显式标注
`order=[0,1,2,3,4], stage=[0,0,0,1,1]`：stage 0 是 A gather 与 packed B/shift 的
`cp.async`，stage 1 是解码与 MMA。**解码直接写 MMA 的 B fragment**（`gemm_sr`），
不经过 shared memory 往返。

tilelang 经 `vllm.tilelang_utils` 导入，pass config 与生产 tilelang kernel 一致
（`TL_DISABLE_WARP_SPECIALIZED`、`TL_DISABLE_TMA_LOWER`、
`TL_PTXAS_REGISTER_USAGE_LEVEL=10`）。最后一项做过对照：带与不带它的最佳 W13
配置是 173.11 vs 172.85 µs（0.15%，噪声内，比值同为 1.111），因此第 5 节的扫描
结论不需要重跑。

> 显式 schedule 是必需的：编译器自动推断会因为解码语句同时依赖两个 async copy
> 而拒绝（`A dependency on multiple async stages is not supported`）；手写
> `order/stage` 后同一 kernel 可以编译，语义等价。

### 4.3 正确性证据

- E8 / top-k6 / M1：W13 与 W2 均与独立参考**逐位相等**（4 种解码 lowering 全部）。
- E384 生产形状：stage-3 的 58 个候选、stage-4 的 14 组复测（含 held-out seed、
  hot/sparse 路由、top-k3、M15/M17 边界、M2048、以及 40 次调用的轮换周期全部 40 个
  case）**candidate 与 baseline 都与独立参考逐位相等**。
- 校验门确实会拦错：把 `--decode raw`（故意不解码）送进自动搜索，记录为
  `verification_status=numerical_failure`、`measurement_status=not_run`（**没有计时**），
  summary `all_passed=false`、`candidates=0`。

## 5. 阶段 3：离线搜索

### 5.1 为什么不是 `AutoTuner.from_kernel`

方案优先要求程序化 AutoTuner；本机 0.1.12 的源码检查显示它**无法表达本候选的
in-place 契约**：`profiler.assert_allclose` 与 `profiler.manual_assert_close`
都只比较 kernel 的**返回值**（`lib_outs` vs `ref_outs`，见
`tilelang/profiler/__init__.py:121-193`），而本候选写入调用者预分配的
`routed_out` 并返回 `None`，两条路径看到的都是空列表，manual checker 也拿不到
`input_tensors`。

方案第 3 节对这种情况给了明确出路：「若该 API 难以表达 in-place 契约，必须在
外层逐配置严格验证后再进入计时，记录每个配置的通过证据，不得用无校验的
autotune 排名直接决定上线。」本轮的 `..._autotune.py` 即按此实现：每个候选
先跑独立参考逐位校验，未通过的一律不计时、不排名。

### 5.2 搜索空间与结果（全部逐位通过）

W13 / M1 / k6 / E384（26 个配置，`block_n∈{64,128}`、`block_k=128`、
`stages∈{1,2,3}`、`threads∈{64,128,256}`、`decode∈{luta,lut,mxfp4}`）：

| 配置 | p50 µs | 相对基线 |
| --- | ---: | ---: |
| 生产 CUTLASS 16x128 | 155.8 | 1.00 |
| **w13-16x64x128-s3-t64-luta** | **173.0** | 1.11 |
| w13-16x64x128-s1-t64-luta | 173.3 | 1.11 |
| w13-16x128x128-s3-t64-luta | 181.9 | 1.17 |
| w13-16x64x128-s2-t128-luta | 187.5 | 1.20 |
| w13-16x64x128-s1-t256-luta | 229.1 | 1.47 |

W2 / M1 / k6 / E384（17 个配置）：基线 74.4 µs，最好
`w2-16x64x128-s2-t64-luta` 82.8 µs（1.11）。

M2048 / k6 / E384（8 个配置，`block_m=64`）：W13 基线 11235.8 → 最好 15221.5 µs
（1.35）；W2 基线 5949.2 → 最好 7950.1 µs（1.34）。

规律：`threads=64`、`block_k=128` 最好；`num_stages` 在 1/2/3 之间几乎没有差别
（±0.5%），说明这条流水线不是被 copy/compute 重叠限制的；`luta` 与 `lut` 只差
1–2%，即**每元素索引算术不是瓶颈**。58 个候选无一编译失败，也无一数值失败。

## 6. 阶段 4：独立复测

最佳配置放回 Event+Graph harness，A/B/B/A 交替、3 个独立 round、每个 round 保留
样本；两侧都先过独立参考。`inconclusive` 是 harness 自己的判据：差异落在基线自比
波动带内。

| 用例 | stage | 基线 p50 µs | 候选 p50 µs | 比值 | 基线自比波动 | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| E384 k6 M1 uniform seed4242 | w13 | 155.60 | 172.85 | 1.111 | 0.07% | ok |
| E384 k6 M1 uniform seed4242 | w2 | 74.34 | 82.84 | 1.114 | 0.00% | ok |
| **held-out** seed20260930 | w13 | 155.65 | 172.80 | 1.110 | 0.07% | ok |
| **held-out** seed20260930 | w2 | 68.10 | 75.98 | 1.116 | 0.00% | ok |
| hot 路由 seed7 | w13 | 155.24 | 173.36 | 1.117 | 0.10% | ok |
| sparse 路由 seed11 | w2 | 74.34 | 82.84 | 1.114 | 0.14% | ok |
| top-k3 draft | w13 | 85.66 | 101.63 | 1.187 | 0.00% | ok |
| top-k3 draft | w2 | 40.19 | 47.56 | 1.183 | 0.00% | ok |
| M17（跨 block_m 边界） | w13 | 1296.69 | 1571.53 | 1.212 | 26.0% | inconclusive |
| M15（跨 block_m 边界） | w2 | 595.10 | 717.88 | 1.206 | 28.4% | inconclusive |
| **轮换专家**（周期 40，240 专家） | w13 | 121.06 | 135.71 | 1.121 | 1.19% | ok |
| **轮换专家**（周期 40，240 专家） | w2 | 58.03 | 64.85 | 1.117 | — | ok |
| M2048 prefill | w13 | 11243.01 | 15344.95 | 1.365 | 0.27% | ok |
| M2048 prefill | w2 | 5948.06 | 7995.19 | 1.344 | 1.01% | ok |

要点：

- 结论对 seed、路由分布、cache 模式、top-k、M 档都稳健，**不是调优集过拟合**。
- 轮换模式按方案要求区分记录：`rotation_period=40`、`rotation_experts_in_union=240`，
  分子分母同口径归一化；40 个 case 全部逐位校验通过。
- M15/M17 两档基线自比波动 26–28%，按判据标 `inconclusive`，**不用它下结论**。
- M2048 的两侧数值与上一轮提交的基线一致（11243 vs 11427；5948 vs 6099），说明
  本轮的测量设施与上一轮可对齐。

## 7. 差距定位：慢的是解码，不是骨架

同一进程内交替测量，同一路由与权重（E384 k6 M1）：

| 变体 | W13 p50 µs | W2 p50 µs | 说明 |
| --- | ---: | ---: | --- |
| 生产 CUTLASS 16x128 | 155.75 | 57.60–74.44 | 基线 |
| TileLang 16x128x64 s2 t128，`mxfp4`（逐元素算术解码） | 319.80 | 114.33 | 0.47× |
| …… `lut`（(shift,byte)→两个 INT8 的 smem 查表） | 213.76 | 77.77 | 0.73× |
| …… `luta`（查表 + 仿射 packed 索引） | 211.15 | 77.06 | 0.74× |
| …… `raw`（**不解码**，同访存、故意算错） | 80.95 | 41.27 | **1.92×** |
| `raw`，最佳 tile 16x64x128 s1 t64 | 63.74 | 33.89 | **2.43× / 2.19×** |
| `raw`，最佳 tile 16x128x128 s1 t64 | **60.57** | **33.28** | **2.57× / 2.23×** |
| 最佳正确配置 `luta` 16x64x128 s1 t64 | 172.85 | 82.84 | 0.90× |

推算：最佳 tile 下骨架 63.74 µs（W13）/ 33.89 µs（W2），正确解码后 172.85 / 82.84 µs，
即**解码各占 109 µs 与 49 µs**；生产 CUTLASS 的解码（其源码注释记录，把
`lookup_mxfp4_x2_byte_perm` 换成裸拷贝可从 126.2 降到 84.4 µs、60.7 降到 47.5 µs）
约 42 µs 与 13 µs。两个 stage 的解码差距都是 2.6–3.7×，与 TileLang 逐元素模型
vs CUTLASS 的 word-level `__byte_perm`（每 4 字节两次 PRMT 出 8 个 INT8）之间的
指令数差距一致。

TileLang 侧已尝试并排除的路径：

- 逐元素算术（`_MAG_PACKED` 移位取表 + 符号选择 + 移位）：每值约 13 条 ALU，
  ALU 受限，最慢。
- shared LUT（一次查表出两个值）：降到每对值 2 次 LDS + 少量 ALU，但随机
  4 字节查表存在 bank conflict，LDS 吞吐成为新瓶颈；仿射索引只再省 1–2%。
- 更宽的 LUT（两字节→四值）需要 65536 项（≥128 KB shared），不可行。
- 寄存器内查表需要 `__shfl_sync` 逐值分发，与本机 `T.shfl_sync` 的逐元素形态
  同样受限于每值一条指令，预期不会好于 LUT。

## 8. 顺带发现并修掉的两个设施缺陷

1. **`interleaved_ab` 从未被调用过，且带一个必然报错的参数名**
   （`benchmarks/kernels/deepseek_v41_moe_harness.py`）：它给 `time_graph`
   传 `graph_calls=`，而该函数的形参是 `repeats`。上一轮的基准脚本只用了
   `time_graph`，所以这个 A/B 助手一直是死代码。本轮首次调用即触发
   `TypeError`，已修正为 `repeats=graph_calls`，并补上 `calls_per_repeat`
   （轮换专家模式一个 `call` 要跑整个周期，分母必须相应放大）。
2. **本轮自己的复测脚本在轮换模式下会漏检**：校验跑在 base case 的缓冲区上，
   而计时实际写的是轮换序列各自的缓冲区，于是"未初始化 vs 参考"被当成失败。
   已改为校验整个周期内**每一个** case（40/40 逐位通过），而不是只看一个。

## 9. 下一步假设（最多三条，均有本轮证据）

1. **只有把解码降到 word-level，TileLang 才可能赢。**
   证据：骨架 60.57–63.74 µs（2.4–2.6× 快于 CUTLASS），解码 49–109 µs
   （CUTLASS 约 13–42 µs）。若解码能到 CUTLASS 水平，W13 ≈ 103–106 µs
   （约 1.5×）、W2 ≈ 46–47 µs（约 1.6×）；即使只到 60 µs 也已经是明显收益。可行路径是在 TileLang 里用 `T.import_source` + `T.call_extern`
   注入等价于 `lookup_mxfp4_x2_byte_perm` 的设备函数，并把解码循环按
   fragment 字（4 字节）组织——`luta` 已经证明本 kernel 的 B fragment 布局可以
   由调用方按 4 字节字控制。**这是唯一值得继续投入的方向。**
2. **M2048 的 1.34–1.37× 差距不只是解码，需要单独定位。**
   证据：prefill 档解码只占总时间约四分之一（每个专家的权重要被 32 行复用），
   但候选仍慢 35%。应先做 M2048 的 `raw` 消融，区分"TileLang MMA/流水效率"与
   "解码"，再决定是否值得单独调这条路径。
3. **`num_stages` 完全不起作用这一现象值得单独查。**
   证据：W13/W2 在 stages=1/2/3 上的 p50 差异 ≤0.5%，说明 `cp.async` 与计算
   没有真正重叠，或瓶颈完全在 stage 1 内部（解码）。若走第 1 条，顺手验证
   去掉解码后 stages 是否重新变得敏感——这能判断 TileLang 的流水是否真的生效。

三条都只是假设，本轮没有实现任何一条。

## 10. 边界与未做事项

- **没有接入生产**：方案第 5 阶段要求前四步有净收益才做默认关闭的 dsv41f 专用接入；
  本轮是净损失，因此没有新增入口、没有改分派、没有改默认 tile/环境变量，
  也没有跑完整 routed MoE 或模型 A/B。
- 没有使用 TMA、WGMMA、TMEM、原生 FP4 路径（目标机是 SM80，方案明确禁止照搬）。
- 没有引入 split-K、浮点 atomic、跨 CTA 通信、W13+激活融合或权重重排。
- 没有覆盖 M=0、top-k 3/6 之外的值、EP/TP 轴；这些仍由生产 CUTLASS 承担。
- 结论只对 dsv41f SM80 routed MoE grouped GEMM：其他模型、量化后端、GPU 架构、
  router/shared expert/attention/DSpark 均未触及。
- 复测用的是合成 packed 权重与合成路由，不是模型精度评估；本报告任何数字都不能
  读作"模型精度通过"。

## 11. 验收清单逐项回答

| 问题 | 回答 |
| --- | --- |
| 最小编译 spike 是否在真机验证了 INT8 MMA/INT32/BF16？ | 是，SASS `IMMA.16832.S8.S8`，逐位一致 65536/65536 |
| 候选是否复现生产数值契约？ | 是，58 个候选 + 14 组复测全部与独立参考逐位相等 |
| 失效的 kernel 会不会被静默计时？ | 不会，`raw` 消融被记为 `numerical_failure` 且不计时，summary `all_passed=false` |
| autotune 排名是否在无校验的情况下决定结论？ | 不是，每个候选都先过独立参考；AutoTuner 因 in-place 契约不可表达而被源码级排除 |
| 固定专家与轮换专家是否分开计时？ | 是，`cache_mode`、`rotation_period=40`、`union=240` 专家，40/40 逐位校验 |
| 是否只动了离线设施？ | 是，改动只在 `benchmarks/kernels/`；`git diff --name-only` 无生产文件 |
| 有没有为了让候选好看而放宽判据？ | 没有；M15/M17 因基线自比波动 26–28% 被标 `inconclusive`，不用作结论 |
| 是否按方案停止条件结束？ | 是，最快配置仍慢 11%，保留全部证据并结束该候选 |
