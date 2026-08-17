# DeepSeek-V4 SM80 streaming indexer 迁移审计与实机结论

日期：2026-08-16

## 结论

SM70 streaming top-k 具有明确的算法借鉴价值，但其收益来自一组联合设计，
不是把外层 K-tile 循环搬到 SM80 即可复现：

1. Q、权重和 grouped GEMM workspace 只准备一次；
2. K 按 tile 打分，不物化完整 `[M, N]` FP32 logits；
3. 无效区 mask 与 grouped epilogue 一起完成；
4. 每个 tile 先做精确 local top-k，再合并历史候选；
5. workspace 预分配、预热、显存 preflight 和环境变量回滚形成完整服务路由。

SM70 CUDA/TileLang kernel、G4/G8、T1024 和 launch 参数不能直接迁移到
CMP170HX。当前 SM80 原型已经证明集合正确；修正 full-path 基准中多余的
score gather 后，隔离组合路径仍快 `2.44%`，但 262K 端到端 TTFT 按 32K
启动漂移归一后回退 `1.37%`。因此 streaming 生产路由保持关闭。

本轮同时确认了更低风险的 P0：full-logits 不应固定压到最小显存。将临时
logits 预算从 32 MiB 提到 96 MiB、将 query chunk 限制为 1024 rows，并把
logits 纳入跨层复用的 caller-owned workspace 后，262K TTFT 的归一净改善为
`1.62%`，三阶段显存峰值不高于 32 MiB 基线。该方案已经晋级为 CMP170HX
launcher 默认值；后续 streaming 研究继续聚焦 score/block-local selection
融合，而不是继续缩小 workspace。

## SM70 中值得保留的设计

源实现位于：

- `vllm/v1/attention/ops/tilelang_prefill_streaming_topk.py`；
- `vllm/model_executor/layers/sparse_attn_indexer.py` 的长上下文路由；
- `docs/superpowers/validation/2026-07-16-deepseek-v4-sm70-streaming-topk-p0.md`；
- `docs/superpowers/validation/2026-07-16-deepseek-v4-sm70-streaming-topk-p0-abc.md`。

对应源工作树为：

```text
/mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split
```

源路由在压缩 row length 至少 8192 时才允许进入。生产获胜结构是：

```text
prepare Q/weights once
  -> K tile grouped logits
  -> tile-local exact top-k
  -> merge best K with tile K
  -> row-relative final indices
```

两项源结论尤其重要：

- grouped streaming 相对旧 headwise 路径的隔离提升约 `2.43x-2.65x`，
  32K SM70/TP8 服务对照曾提升 `20.585%`；这些数字只用于筛选设计，
  不能外推到 SM80/PP3。
- direct candidate update 在小 rows 约快 `2%`，但在 2K/4K rows 分别
  回退约 `42%/44%`。因此应保留 tile-local selector 加 candidate merge，
  不能改成每个 score block 直接竞争全局 best K。

## 目标 SM80 基线问题

现有 prefill 路径按 query rows 分块控制峰值，但每个分块仍执行：

```text
FP8 K -> BF16 K
  -> 完整 [M, N] FP32 logits
  -> 独立 prefill top-k
```

在 `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=32` 时：

- 32K 原始上下文对应压缩 `N ~= 8192`，单次最多约 1024 rows；
- 262K 原始上下文对应压缩 `N ~= 65536`，单次最多约 128 rows；
- 同一 scheduler chunk 的 K 转换和扫描会随 query subchunk 重复。

Streaming exact top-k 只消除完整 logits 驻留和部分重复准备，不改变每个
query row 必须扫描全部历史 K 的事实。因此其计算仍为 `O(M*N)`；累计 prefill
仍接近二次增长。若要改变渐进复杂度，需要近似、分层索引或模型语义变化，
不能把 exact streaming 宣称为线性算法。

## SM80 streaming 实验原型

本轮基于已有 Triton correctness prototype 增加：

- Q/K/tile-logits/candidate buffer 在单次调用内复用；
- tile range 准备；首 tile 直接写候选，不再单独初始化；
- 复用 SM80 `KV_GROUP=8`，使 Q/weights 跨 8 个 K blocks 复用；
- tile-local radix top-k 与 CUB 512+512 candidate merge 在同一 CTA 完成；
- 长 row 按 512 query rows 分块，短 row 保留 32 MiB full-logits cap；
- 仅 SM80、FP8 cache、DCP=1、top-k=512、压缩长度至少 16384 可进入；
- 环境变量总开关默认关闭。

实验开关为：

```text
VLLM_INDEXER_PREFILL_STREAMING_TOPK=0
VLLM_INDEXER_PREFILL_STREAMING_MIN_LEN=16384
VLLM_INDEXER_PREFILL_STREAMING_TILE_SIZE=16384
VLLM_INDEXER_PREFILL_STREAMING_MAX_ROWS=512
```

### 正确性

GPU 测试覆盖多 tile、不同 row ranges、短于 top-k 的行以及 `-1/-inf`
padding。比较使用完整 logits 的数学 top-512 score 集合，避免 tie 顺序造成
伪失败。当前测试通过，说明候选集合语义正确；这不等价于性能晋级。

### CMP170HX 微基准

同一张 CMP170HX、FP8 Q/K、64 heads、head dim 128、top-k 512，最终
`tile=16384/rows=512`：

| 实现阶段 | full logits | streaming | 相对 full | streaming 临时显存 |
| --- | ---: | ---: | ---: | ---: |
| ungrouped + 独立 merge | 21.698 ms | 22.701 ms | +4.62% | 69.95 MB |
| KV_GROUP=8 + 独立 merge | 21.121 ms | 20.733 ms | -1.83% | 58.41 MB |
| KV_GROUP=8 + fused selector/merge | 21.084 ms | 20.568 ms | -2.44% | 57.36 MB |

最后一行修正了旧基准偏差：旧 full path 为了返回 score，额外执行了生产
服务中不存在的 gather。修正后 full path 使用 128-row chunk，streaming 使用
512-row chunk；临时分配分别为 89.50 MB 和 57.36 MB。它们是 microbench 的
调用期分配，不能据此推断服务常驻显存。

同一形状的 CUDA Event 分段为：

| 阶段 | full ms | streaming ms |
| --- | ---: | ---: |
| Q FP8 -> BF16 | 0.361 | 0.138 |
| K FP8 -> BF16 | 1.022 | 0.376 |
| score | 19.036 | 18.627 |
| bounds | - | 0.101 |
| select / select+merge | 1.161 | 1.511 |

score 占 streaming 约 `90.6%`，selector/merge 反而比 full selector 多
0.35 ms。继续优化 Python 外层或单纯缩 tile 不是主方向。

### PP3 服务 A/B

固定三张 CMP170HX、`VLLM_PP_LAYER_PARTITION=15,16,12`、
`max-num-batched-tokens=1715`、`max-num-seqs=1`、DSpark 7-token、
prefix cache 关闭。每个 arm 先预热，再执行三次正式请求，取 TTFT 中位数。

| 上下文 | current baseline | fused streaming | raw 变化 | 判定 |
| --- | ---: | ---: | ---: | --- |
| 32K | 6.246 s | 6.206 s | -0.63% | 路由关闭，作为启动漂移哨兵 |
| 262K | 71.101 s | 71.621 s | +0.73% | 拒绝晋级 |

32K 的压缩长度约 8192，低于实验门限 16384，实际执行路径与 baseline 相同。
用 32K 的 `-0.63%` 漂移归一后，262K 净变化约为 `+1.37%`。
262K prompt hash 完全一致；服务完成全部请求，无 OOM、Xid 或计算期
EngineCore 异常。日志末尾的 EngineDeadError 发生在主动 SIGINT 清理阶段。

GPU 总显存采样的每卡峰值没有增加：baseline 的 GPU1/2/3 分别为
59024/61606/60298 MiB，candidate 为 59028/61610/60298 MiB。4 MiB 差异低于
500 ms 采样粒度可解释范围。该指标包含模型、
CUDA Graph 和 KV cache，不用于宣称 streaming 常驻显存收益，只用于排除新增
常驻预算。

本机 Nsight Systems 与 PyTorch CUDA profiler 均返回
`CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED`。因此 kernel 归因使用 CUDA Event、
phase profiler 和 E2E 对照；不能把缺失的 CUPTI trace 解读为无 GPU 开销。

### 阶段定位

早期 Python/Triton candidate 的 262K phase-only 对照显示：

| critical PP rank stage | baseline | candidate |
| --- | ---: | ---: |
| `target.model_forward` | 59.463 s | 59.709 s |
| `indexer.select` | 13.011 s | 13.791 s |
| `indexer.logits` | 12.026 s | 0.789 s（短 row） |
| `indexer.topk` | 0.917 s | 0.040 s（短 row） |
| `indexer.streaming_topk` | - | 12.902 s |

这组 phase 数据来自 grouped/fused 合并前版本，只用于定位组合 pipeline 的
开销，不能代替最终 fused 版的 E2E 判定。后续两轮隔离优化缩小了 kernel
开销，但没有让 262K 服务晋级。

## 下一步优化边界

下一版 SM80 实现不应继续优化 Python 外层拼装，而应满足：

1. score CTA 不写完整 `[M, tile_k]` FP32 tile；
2. score epilogue 直接产出每个 score block 的有界候选；
3. CTA 内先做 block-local selection，再由 fused grouped merge 合并；
4. Q/weights/K scale 准备一次，K decode 按 tile 完成；
5. 保留 exact score-set debug compare、caller workspace、预热与回滚；
6. 继续以 32K 路由关闭无回退、262K PP3 TTFT 改善为晋级门。

候选实现优先级：

1. 原生 CUDA tensor-core score + block-local radix/select；
2. Triton score epilogue 输出 block candidates，再调用当前 fused selector；
3. 若 exact fused 版本仍无收益，再单独立项评估分层/近似索引，不混入本轮。

当前 fused tile selector/candidate merge 可作为第二种实现的正确性参照，
但在 score 与 block-local selection 融合前，实验路由必须保持默认关闭。

## 晋级的 full-logits row-wave 优化

### 配置扫描

固定 `M=1715,H=64,D=128,top-k=512`，在同一 CMP170HX 进程中轮转各
row chunk 七轮、每轮十次，避免跨进程 Triton autotune 和温度漂移。96 MiB
预算再加 1024-row 上限，相对原 32 MiB 配置得到：

| 压缩 N | 32 MiB rows / ms | 96 MiB + cap rows / ms | 改善 |
| ---: | ---: | ---: | ---: |
| 8192 | 1024 / 2.778 | 1024 / 2.778 | 0.00% |
| 16384 | 512 / 5.104 | 1024 / 5.029 | 1.47% |
| 32768 | 256 / 10.220 | 768 / 9.871 | 3.41% |
| 65536 | 128 / 20.859 | 384 / 19.736 | 5.39% |

1024-row 上限很重要：32K 压缩 N=8192 时，直接让 96 MiB 预算吃满全部
1715 rows 会越过最优 wave；预算和 row cap 必须同时配置。继续从 96 MiB
增加到更大的 logits buffer 仍有小幅 kernel 收益，但单位显存收益明显下降，
未进入服务候选。

### caller-owned workspace

只扩大 `torch.empty` 会让不同层/流保留不同 size class。实机 262K 后，未经
复用的 96 MiB 版本在 GPU1/2/3 分别达到 61084/63666/61966 MiB，比基线多
约 1.67-2.06 GiB，不能接受。

最终实现让 `fp8_mqa_logits_triton` 接收 caller output，并与 indexer K values、
K scales 一起由全局 workspace manager 的同一 ubatch slot 预留。dummy/profile
和真实 forward 使用相同公式，workspace 锁定后不会热路径增长。环境开关为：

```text
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=96
VLLM_SPARSE_INDEXER_MAX_QUERY_ROWS=1024
VLLM_INDEXER_PREFILL_CALLER_LOGITS=1
```

三项均可独立回滚；library 默认仍为 512 MiB、无 row cap、caller output 关闭，
CMP170HX launcher 才启用上述实测组合。

### 最终 PP3 A/B

固定 PP `15,16,12`、1M deployment、2.7 GB KV/rank、1715 batched tokens、
max-num-seqs=1、DSpark7、prefix cache 关闭。每个上下文先预热再正式运行三次，
取 TTFT 中位数：

| 上下文 | 32 MiB baseline | 96 MiB + row cap + caller | raw 变化 |
| --- | ---: | ---: | ---: |
| 32K | 6.246 s | 6.201 s | -0.72% |
| 262K | 71.101 s | 69.442 s | -2.33% |

32K 两侧执行相同的 1024-row 分块，用作启动漂移哨兵。按其 `-0.72%` 漂移
归一后，262K 净改善 `1.62%`。prompt hash 和 token 数完全一致，所有请求均
生成 8 tokens；无 OOM、Xid、非法访存或计算期 EngineCore 异常。

500 ms 的整卡显存峰值为：

| 配置 | GPU1 MiB | GPU2 MiB | GPU3 MiB |
| --- | ---: | ---: | ---: |
| 32 MiB baseline | 59024 | 61606 | 60298 |
| 96 MiB caller workspace | 58798 | 61380 | 60298 |

采样粒度不支持宣称节省了 226 MiB，但足以证明没有新增显存峰值。这里的结论
不是“显存越少越好”，而是将可用临时预算集中到一个可复用 workspace，换取
更好的 score wave，同时避免 allocator 按层放大。
