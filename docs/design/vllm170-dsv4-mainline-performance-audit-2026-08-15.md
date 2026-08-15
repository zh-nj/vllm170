# DeepSeek-V4 主线性能提交吸收审计（CMP170HX / PP3）

日期：2026-08-15

目标提交：`03971054634558102a355922568e398c03469316`

部署：3 x CMP170HX、PP3/TP1、TRITON_MLA_SPARSE_DSV4、Marlin、DSpark-7

模型：`/mnt/data6/models/DeepSeek-V4-Flash-0731`

## 结论

列出的 18 个主线 PR 的 merge commit 都已经是当前提交的祖先。因此本轮不做
cherry-pick，而是核对每项优化在当前 PP3/SM80/异步 DSpark 路线中是否真正
生效，并对可疑路径做 CMP170HX A/B。

当前结论分为四类：

- 已生效并应保留：短上下文跳过 indexer、eager workspace 复用、adaptive
  top-k width、紧凑 KV packing、mHC broadcast、fused top-k bias。
- 已由本分支等价或更适合 SM80 的实现覆盖：caller-owned sparse prefill
  output、重新实测后晋升的 SM80 adaptive split decode/caller output、ROCm
  sparse prefill 中的 `num_warps=4`、预取与小 `BLOCK_K` 思路。
- 已进入代码但当前拓扑不生效：sequence parallel、FlashInfer q-head
  unpadding、MXFP4 indexer cache，以及异步 DSpark 下的 CPU-side empty-C128
  skip。empty-C128 的 GPU 融合原型已经完成实机否决，不进入运行时代码。
- 不应直接迁移：ROCm/XPU 专用 kernel 和 DSpark 模型实现。只吸收算法结构，
  不能把 gfx942/gfx950/XPU launch 配置搬到 SM80。

## 逐项审计

| PR | 主线优化 | 当前状态 | CMP170HX 决策 |
| --- | --- | --- | --- |
| #46789 | sequence parallel | 代码已在树中；只在 PP=1、EP、TP>1 且满足 MegaMoE/DP 条件时启用 | PP3/TP1 不生效；不为当前拓扑改路由 |
| #48957 | 跳过没有 C128 边界的 compressor launch | eager 同步路径可用；异步 DSpark 会清空精确 CPU start，因此当前 decode 不生效 | GPU q8 融合原型在 FULL Graph 端点退化 1.74%，已撤回；保留现状 |
| #49486 | candidate 数不超过 top-k 时跳过 Q/router/top-k | 已生效；C4 `topk=512` 时覆盖原始上下文不超过 2048 的 chunk | 保留；5K prefill 仅首 chunk 命中，不是长上下文 steady-state 解法 |
| #49236 | eager scratch/workspace 复用 | `DeepseekV4EagerScratchPool` 已覆盖 Q、indexer、global top-k、compressor scratch | 保留；后续扩展必须复用/overlay，不能增加长期大 buffer |
| #50298 | FlashMLA combined-indices caller output | 当前不是 FlashMLA；本分支 sparse prefill 已有 caller-output 等价路径 | 不迁移 FlashMLA 代码；继续维护现有 SM80 Triton 路径 |
| #50004 | adaptive C128 top-k width | `_get_c128a_active_topk_width` 已生效 | 保留；上下文容量专题再审计 dense decode buffer |
| #50312 | 避免 PP last-rank 448 MiB buffer/copy | 主线逻辑已在树中；DSpark 不需要 pre-HC hidden states；本分支另有 PP receive-buffer reuse | 已吸收；不增加替代常驻显存 |
| #48993 | compact MXFP4 indexer cache / group overlay | packed group overlay 已生效；启动器仍使用 FP8 indexer cache | overlay 保留；MXFP4 只作为容量 A/B，必须验 logits/top-k/acceptance |
| #48047 | FlashInfer >=0.6.14 去 q-head padding | 仅 FlashInfer SM100/SM120 路线 | 当前 Triton SM80 不适用 |
| #48660 | specialized DSV4 router | 当前配置满足 fast-path 条件 | 保持默认；增加运行时回退开关用于审计，不推广 generic |
| #47463 | fused_topk_bias | 当前 3 个 hash layer 使用；其余层通常由 specialized router 接管 | 保留；不单独重写 |
| #48137 | 去除首层 repeat/copy 的 mHC broadcast | 当前 NVIDIA SM80 路径已生效 | CMP170HX micro 1.24x–3.22x，保留 |
| #47718 | ROCm HCA prefill 两阶段 compressor | 当前 `_prefer_two_stage_compressor()` 只允许 ROCm | 算法候选；只有 profiler 证明 compressor 重回 P0 才写 SM80 原型 |
| #48519 | ROCm sparse prefill 优化 | SM80 后端已吸收 warps=4、预取、小 BLOCK_K 等同类策略 | 不复制 ROCm kernel；继续基于本分支 C4/C128 A/B |
| #48788 | gfx950 sparse decode reducer occupancy | 原有 partial/reducer 只对 gfx942/gfx950 开放；SM80 实际走未 split fallback | 已在 CMP 上重扫 reducer `BLOCK_H/warps`，以 SM80 shape 表重实现 |
| #46275 | gfx942 split sparse decode | 原有 adaptive split 未覆盖 SM80 | 已复用算法和 cost model、重扫 launch shape；PP3 32K `ms/round` 改善 9.34% |
| #47419 | AMD DSpark | 平台移植代码，不是 NVIDIA kernel 优化 | 不吸收实现；仅对照 DSpark 阶段契约和 acceptance 计量 |
| #47677 | XPU DSpark | 平台移植代码，含 XPU HCHead/MHCPost | 不吸收实现；仅核对 aux hidden-state 和 proposal 接口 |

## P0 已验证：异步 DSpark 下的 empty-C128 不晋升

主线 #48957 依赖 `CommonAttentionMetadata._num_computed_tokens_cpu` 得到每个
请求的精确起点，然后在本 chunk 不跨越 128-token 边界时跳过
`compress_norm_rope_store`。它仍然先执行 `save_partial_states`；早期审计中“两个
kernel 都跳过”的表述不准确，本轮已按原始提交 `37e370fe9` 校正。

当前异步推测解码中，GPU tensor 才是 authoritative state，runner 会把
`_num_computed_tokens_cpu` 设为 `None`。因此 `_get_c128_boundary()` 返回
`None`，host-side skip 被保守关闭。32K eager 详细 trace 中，20 个 C128 层的
`compressor.save_partial_states` 和 `compressor.compress_norm_rope_store` 都各有
140 次调用，说明 decode 没有得到该收益。

不能改用 `seq_lens_cpu_upper_bound`：异步 rejection 后它是乐观上界，
`mod 128` 边界不保持单调，误跳过会漏写 C128 cache。本轮实现了一个实验性
Triton 原型：仅在 Ampere、FULL Graph、C128、`max_query_len<=8` 时，每个请求用
一个 program 串行写入本轮 partial states，再直接读取 GPU `positions` 判断是否压缩。
它不做 host sync，也不申请 workspace。

局部正确性覆盖单请求、多请求、命中/不命中边界；state 和 KV cache 均逐字节
一致。进一步把两个版本各自捕获为 CUDA Graph，从 32K 起连续重放 64 个 q8 round、
共 512 token，每轮 state/cache 都逐字节一致。

但生产形态否决了这个结构。下表为 CMP170HX 单层 micro，`speedup = baseline / fused`：

| q | 边界 | eager baseline/fused | eager speedup | Graph baseline/fused | Graph speedup |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 否 | 89.364/54.129 us | 1.651x | 5.448/4.116 us | 1.323x |
| 1 | 是 | 223.160/182.333 us | 1.224x | 176.722/181.504 us | 0.974x |
| 8 | 否 | 84.285/50.381 us | 1.673x | 4.731/6.472 us | 0.731x |
| 8 | 是 | 190.720/186.163 us | 1.024x | 187.771/185.457 us | 1.012x |

eager 的 1.67x 来自减少 Python/driver launch 间隙，不能外推到生产 FULL Graph。
真实 target 是 q8；Graph 已消除大部分 host launch 成本，而融合 kernel 串行写 8 行
partial state，空边界时反而慢 1.74 us/层。

32K、512 输出 token、3 repeats 的 PP3 A/B/A 只改变该原型开关：

| 路线 | ms/round mean | accepted length mean | TTFT mean | PP0/PP1/PP2 workload peak |
| --- | ---: | ---: | ---: | ---: |
| baseline A1 | 39.266 ms | 6.035 | 5.854 s | 58272/57360/63044 MiB |
| fused B | 39.754 ms | 4.691 | 5.821 s | 58272/57360/63044 MiB |
| baseline A2 | 38.885 ms | 5.885 | 5.842 s | 58272/57360/63044 MiB |

B 相对 A1/A2 合并均值 39.076 ms 退化 1.74%。accepted length 的差异不能单独
证明语义回归，但逐轮 cache parity 已排除原型的确定性字节漂移；性能退化本身已足以
否决。实验 kernel、路由、环境开关和测试均已撤回，不保留死代码。

原始产物：

- `profiles/deepseek_v4/p0-c128-fused-decode/baseline-a1/20260815T135823Z`
- `profiles/deepseek_v4/p0-c128-fused-decode/candidate-b/20260815T140213Z`
- `profiles/deepseek_v4/p0-c128-fused-decode/baseline-a2/20260815T140457Z`

## P1 已晋升：SM80 adaptive split sparse decode

审计校正了一个关键现状：当前文件中虽然已有 adaptive split cost model、partial
kernel 和 reducer，但运行路由只允许 gfx942/gfx950。CMP170HX/SM80 一直进入单
kernel、未 split fallback，不能视作“已经生效”。本轮保留算法结构，在 SM80 上
重新扫描 q1/q2/q4/q8 的 partial `BLOCK_H/BLOCK_K/warps` 和 reducer
`BLOCK_H/warps`，并把约 2.01 MiB FP32 partials overlay 到既有 workspace。

固定 CUDA Graph micro 中，SWA128+C4 top-512 的 q1/q2/q4/q8 split 组合相对
fallback 分别为 7.30x、6.46x、4.28x、3.09x；q8 C128 与 SWA-only 分别为
2.97x、2.42x。最大绝对误差不超过 2.44e-4。让 reducer 直接写 caller output
进一步省 1.74-2.16 us/层，不增加常驻显存。

32K、512 输出 token、3 repeats 的组合候选为 35.525 ms/round；A1/A2 六次
基线均值为 39.184 ms/round，改善 9.34%。split-only 也改善 7.16%。候选的
PP0/PP1/PP2 工作负载峰值为 58274/57360/63036 MiB，对比基线
58272/57360/63044 MiB，仅 PP0 可见约 2 MiB workspace 增量。正确性覆盖
q1/q2/q4/q8、锁定 workspace 和 CUDA Graph replay。

CMP170HX 启动器已默认启用：

```text
VLLM_DEEPSEEK_V4_SPARSE_DECODE_SPLIT=1
VLLM_DEEPSEEK_V4_SPARSE_DECODE_CALLER_OUTPUT=1
```

库级默认和其他架构路由不变，两个开关均可独立回滚。完整 shape 表、A/B/A、
产物和测量限制见
`docs/design/vllm170-dsv4-sm80-sparse-decode-p1-2026-08-15.md`。

## Router 与 mHC 的 CMP170HX 审计

新增 `benchmarks/kernels/deepseek_v4_mainline_fusions.py`，用真实 DSV4 形状比较
主线 specialized router / generic fused router，以及 mHC broadcast / 旧
repeat+copy 路线。计时为 CUDA Event median，多 sample，并同时检查结果。

### Router micro

`speedup = generic / specialized`；小于 1 表示 generic 更快。

| M | specialized (us) | generic (us) | speedup | top-k set recall | max weight abs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 53.760 | 50.811 | 0.945x | 1.0 | 5.96e-8 |
| 8 | 50.606 | 49.132 | 0.971x | 1.0 | 1.04e-7 |
| 64 | 58.532 | 45.793 | 0.782x | 1.0 | 1.34e-7 |
| 1667 | 53.617 | 44.892 | 0.837x | 1.0 | 1.49e-7 |

generic 单 kernel 在本机快约 3%–22%，但这不足以推广。完整 PP3
A/B/A 使用相同 prompt、5K prefill、60 decode tokens、DSpark-7、
`max-num-batched-tokens=1715`，每组 3 次请求：

| 路线 | TTFT (s) | ms/round | decode (s) | acceptance |
| --- | ---: | ---: | ---: | ---: |
| specialized A1 | 1.21883 | 34.70163 | 0.78638 | 0.25271 |
| generic B | 1.23205 | 34.51940 | 0.98975 | 0.18087 |
| specialized A2 | 1.22215 | 34.50311 | 0.85131 | 0.20395 |

相对 A1/A2 均值，generic 的 `ms/round` 只改善 0.24%，TTFT 退化 0.95%；
本批短样本的 acceptance 相对低 20.79%，decode 时间高 20.87%。acceptance
样本量不足以证明普遍退化，但足以阻止推广，尤其主线也已有低精度路由导致
DSpark 分布漂移的公开报告。最终保持 specialized 默认；
`VLLM_DEEPSEEK_V4_SPECIALIZED_ROUTER=0` 仅用于可复现审计。

显存样本差异仅 6–12 MiB，属于采样噪声，没有容量收益。

### mHC broadcast micro

| M | broadcast (us) | repeat+expanded (us) | speedup | 最大差异 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 500.736 | 622.899 | 1.244x | 0 |
| 8 | 279.040 | 403.149 | 1.445x | 0 |
| 64 | 283.750 | 402.022 | 1.417x | 1.19e-7 |
| 1667 | 528.589 | 1703.731 | 3.223x | BF16 layer input 2.44e-4；其余 1.19e-7 |

CMP170HX/SM80 不走主线中可能改变累加语义的 DeepGEMM TF32 分支。本机结果
显示 broadcast 路线在 decode 到 prefill 形状均有收益，因此保持现状。

## 复现

微基准：

```bash
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  .venv/bin/python benchmarks/kernels/deepseek_v4_mainline_fusions.py \
  --component all --m 1 8 64 1667 \
  --warmup 3 --iterations 20 --samples 5 \
  --output /tmp/dsv4-mainline-fusions.jsonl
```

整机 router 对照只改变一个环境变量：

```bash
# A：部署默认
VLLM_DEEPSEEK_V4_SPECIALIZED_ROUTER=1 \
  benchmarks/deepseek_v4/profile_cmp170hx_pp3.sh

# B：仅审计回退
VLLM_DEEPSEEK_V4_SPECIALIZED_ROUTER=0 \
  benchmarks/deepseek_v4/profile_cmp170hx_pp3.sh
```

本次 A/B/A 原始产物：

- `profiles/deepseek_v4/mainline-router-ab/specialized/20260815T070712Z`
- `profiles/deepseek_v4/mainline-router-ab/generic/20260815T070935Z`
- `profiles/deepseek_v4/mainline-router-ab/specialized-repeat/20260815T071205Z`

## 后续顺序

1. P1：把紧凑 MXFP4 indexer cache 作为容量专题独立 A/B，验 top-k 集合、
   acceptance、长上下文正确性和每 PP rank 显存，不能只验 kernel latency。
2. P2 已晋升：`wo_a` BMM cache 成为唯一 weight owner，释放从未进入 forward 的
   packed FP8 linear 表示。5K PP3 A/B/A 中 PP0/PP1 workload peak 各下降
   486 MiB，TTFT 变化仅 +0.10%，且没有新增 buffer。完整证据见
   `docs/design/vllm170-dsv4-sm80-output-projection-p2-2026-08-15.md`。
3. P2：只有 compressor 再次成为 phase P0 时，研究两阶段 HCA compressor；
   默认不新增常驻 buffer。

所有 promotion 都必须使用相同模型、prompt hash、PP3 分区、DSpark width、
max-batched-tokens 和 cache 配置做 A/B/A，并同时报告 TTFT、ms/round、
acceptance、输出 hash 与各 rank 峰值显存。

### MXFP4 容量路线的 SM80 前置门

本轮已额外执行到编译门，不把“主线已有 68B row”误写成 SM80 可直接启用。现有
Triton MXFP4 Q 量化在 CMP170HX 上被 `ptxas` 明确拒绝：

```text
Instruction 'cvt with .e2m1x2' not supported on .target 'sm_80'
```

此外，当前无 DeepGEMM 的 SM80 Triton indexer 只实现 FP8 prefill/paged logits。
因此独立 backend 至少需要：软件 RTNE E2M1 Q/K 编码、MXFP4 prefill logits、
MXFP4 paged decode logits，随后才能解除模型选择器、metadata builder 和 indexer
fallback 的三道 fail-closed 检查。

当前 `15,15,13` 分区每 rank 有 7 个 C4 indexer 层。132B→68B row 在 131K
上下文约节省 14 MiB/rank，在 1M 约节省 112 MiB/rank。容量收益真实但不足以接受
全量 BF16 反量化临时 buffer 或未经真实 logits/top-k/acceptance 验证的捷径；留给
1M 上下文专题重新实现。
