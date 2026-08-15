# DeepSeek-V4 SM80 sparse decode P1 优化（2026-08-15）

## 结论

当前 CUDA SM80 路线此前并没有启用 adaptive split sparse decode。实现中的
partial/reducer 两阶段只对 gfx942/gfx950 开放，CMP170HX 会落入单 kernel、未
split 的通用 fallback。不能据此直接照搬 ROCm launch 参数，本轮在三张
CMP170HX、PP3/TP1、DSpark-7 上重新完成 shape 扫描、正确性、CUDA Graph 和服务
A/B/A。

最终晋升两项 CMP170HX 部署开关：

```text
VLLM_DEEPSEEK_V4_SPARSE_DECODE_SPLIT=1
VLLM_DEEPSEEK_V4_SPARSE_DECODE_CALLER_OUTPUT=1
```

库级默认仍为关闭，其他架构和 ROCm 原有路由不变。32K 输入、512 输出 token、
三次重复的组合候选相对 A1/A2 六次基线把 `ms/round` 从 39.184 ms 降到
35.525 ms，改善 9.34%。PP0/PP1/PP2 工作负载峰值从
58272/57360/63044 MiB 变为 58274/57360/63036 MiB；唯一可解释的增长是 PP0
约 2 MiB 的 FP32 partial workspace，没有增加长期大 buffer。

## 实现

SM80 decode 复用已有两阶段算法：

1. 依据 query 数、head blocks、SWA 长度和额外稀疏历史长度选择 split 数；
2. partial kernel 分片扫描 UE8M0 KV cache，保留每个 split 的 FP32
   `m/l/acc`；
3. reducer 合并 split 并写出 BF16 attention output；
4. caller-output 路径让 reducer 直接写调用方稳定 buffer，省去临时 BF16 output
   和最终 copy。

SM80 的 partial buffers 通过 `WorkspaceManager.get_simultaneous()` overlay 到
已有 eager/prefill workspace。q1/split16 与 q8/split2 的最大需求相同，均为
2,105,344 bytes（2.008 MiB）。真实服务已在 workspace lock 和 FULL CUDA Graph
下通过；不会在热路径扩容。

当前 CMP170HX shape 表是本机扫描后的保守配置：

| Query 数 | Split | Partial `(BLOCK_H, BLOCK_K, warps)` | Reducer `(BLOCK_H, warps)` |
| ---: | ---: | --- | --- |
| 1 | 16 | `(8, 16, 4)` | `(1, 8)` |
| 2 | 8 | `(16, 32, 8)` | `(1, 4)` |
| 4 | 4 | `(16, 32, 8)` | `(1, 4)` |
| 8 | 2 | `(16, 32, 8)` | `(2, 4)` |

split 数仍由已有 cost model 自适应计算；表中是 128-token SWA 加 C4 top-512
生产形状的典型输出，不是写死的上下文阈值。q1 的两个 partial 候选在重复扫描中
会交换领先位置，因此保留更保守的 `(8,16,4)`；这不影响 q8 target verification
主路径。

## CUDA Graph micro

基准 `benchmarks/kernels/deepseek_v4_sparse_mla_decode.py` 使用生产
`quantize_and_insert_k_cache` 构造 UE8M0 cache，分别扫描 partial 和 reducer，
然后比较组合结果与未 split fallback。所有结果均为同一张 CMP170HX 上的 CUDA
Graph median；`speedup = fallback / split`。

| 形状 | Fallback | Split | Speedup | 最大绝对误差 | Workspace |
| --- | ---: | ---: | ---: | ---: | ---: |
| q1, SWA128+C4 512 | 253.491 us | 34.739 us | 7.30x | 1.22e-4 | 2.008 MiB |
| q2, SWA128+C4 512 | 263.876 us | 40.841 us | 6.46x | 1.22e-4 | 2.008 MiB |
| q4, SWA128+C4 512 | 263.927 us | 61.653 us | 4.28x | 1.22e-4 | 2.008 MiB |
| q8, SWA128+C4 512 | 264.162 us | 85.586 us | 3.09x | 1.22e-4 | 2.008 MiB |
| q8, SWA128+C128 256 | 163.031 us | 54.907 us | 2.97x | 1.22e-4 | 2.008 MiB |
| q8, SWA128 only | 57.856 us | 23.951 us | 2.42x | 2.44e-4 | 2.008 MiB |

q1/q2/q4 的表格保留首次独立扫描值；固化产物还包含第二次全 shape 扫描，频率
状态不同但结论一致。不得把某次自动选出的亚微秒 reducer 差异解释成跨机器配置。

caller-output 在固定 q8/C4 Graph 中为 85.586 us，保留最终 BF16 copy 时为
87.716 us，单层省 2.130 us、改善 2.49%。q1/q2/q4/q8 以及 C4/C128/SWA-only
六组扫描中，copy 成本均为 1.74-2.16 us。

固化产物：

- `profiles/deepseek_v4/sparse-mla-decode-sm80-20260815/*.jsonl`

## 三卡 PP3 服务门

所有组都使用同一模型、prompt hash、32K 输入、512 输出、
`max-num-batched-tokens=1715`、DSpark-7、禁用 prefix cache。每组重启服务并做
3 repeats；`ms/round` 已按实际 target verification round 归一化，因此不直接用
波动的 decode 总时长比较 kernel 优化。

| 组 | Caller output | SM80 split | ms/round | Mean accepted length | TTFT | 工作负载峰值 PP0/PP1/PP2 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline A1 | 0 | 0 | 39.002 ms | 4.647 | 5.830 s | 58272/57360/63044 MiB |
| Caller-only B | 1 | 0 | 40.238 ms | 5.341 | 5.814 s | 58272/57360/63032 MiB |
| Split+caller C | 1 | 1 | 35.525 ms | 5.311 | 5.832 s | 58274/57360/63036 MiB |
| Split-only D | 0 | 1 | 36.376 ms | 4.814 | 5.813 s | 58274/57360/63046 MiB |
| Baseline A2 | 0 | 0 | 39.366 ms | 4.946 | 5.819 s | 58272/57360/63044 MiB |

Caller-only 跨服务启动的反向波动与固定 Graph micro 不一致，不能单独归因；它不
构成 promotion 证据。晋升依据是 A1/C/A2 的组合闭环、固定 Graph 中 1.74-2.16 us
的 copy 消除，以及直接写入/旧 copy 的逐元素正确性。split-only D 相对六次基线
也改善 7.16%，说明主要收益确实来自 split，而不是输出轨迹。

原始服务产物：

- `profiles/deepseek_v4/p1-sparse-decode/baseline-a1/20260815T142617Z`
- `profiles/deepseek_v4/p1-sparse-decode/caller-b/20260815T142929Z`
- `profiles/deepseek_v4/p1-sparse-decode/split-c/20260815T143213Z`
- `profiles/deepseek_v4/p1-sparse-decode/split-only-d/20260815T143530Z`
- `profiles/deepseek_v4/p1-sparse-decode/baseline-a2/20260815T143813Z`

Graph phase 产物位于
`profiles/deepseek_v4/p1-sparse-decode/graph-candidate/20260815T144221Z`。三 rank、
workload 和 CUDA-event phase 均完整；CMP170HX 仍返回
`CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED`，且 FULL Graph replay 内部模块 range
不可见。因此该 run 只作为 Graph/phase 兼容性验证，不用它带同步的
131.4 ms/round 做性能结论。

## 正确性与回滚

聚焦测试参数化覆盖 q1/q2/q4/q8，使用官方 UE8M0 cache 写入，比较未 split
fallback 与 split+caller-output，随后锁定 workspace 并捕获/replay CUDA Graph：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
  tests/kernels/attention/test_dsv4_sparse_prefill_sm80.py \
  -k 'sparse_decode_split or sparse_prefill_caller' -q
```

结果为 `5 passed`。两个开关可以独立回滚：

```text
VLLM_DEEPSEEK_V4_SPARSE_DECODE_SPLIT=0
VLLM_DEEPSEEK_V4_SPARSE_DECODE_CALLER_OUTPUT=0
```

回滚 split 会恢复 SM80 未 split fallback；回滚 caller output 会恢复临时 BF16
结果和最终 copy。库级默认关闭保证未经验证的 NVIDIA 拓扑不被自动改变。
