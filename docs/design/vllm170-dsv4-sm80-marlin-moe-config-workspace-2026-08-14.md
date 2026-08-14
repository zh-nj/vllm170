# DeepSeek V4 SM80 Marlin 配置、workspace 与融合研究（2026-08-14）

## 结论

本轮在单张 CMP170HX 上完成 W13/W2 的真实 DSV4 shape 扫描，并在三张
CMP170HX 的 PP3/TP1、DSpark 7-token 部署路径完成 caller-owned FP32 reduce
workspace 的 A/B/A。结论是：

1. `M={1000,1666,1667,2000,4048}`、block-M 48 下，W13 与 W2 的自动选择都与
   强制 `thread_k=64, thread_n=128, blocks_per_sm=2` 等效；没有发现稳定快于 auto
   的 shape，因此生产继续使用 auto，不增加项目私有强制表。
2. W13/W2 共用一个 caller-owned FP32 `c_tmp` 安全且地址稳定。`M=1667` 的最大
   需求为 3,440,640 个 FP32 元素，即 13.125 MiB/rank；隔离基准中的热路径临时
   分配峰值由 14.0 MiB 降为 0。
3. PP3 A/B/A 的 caller pooled TTFT 中位数为 `1260.607 ms`，internal control 为
   `1260.645 ms`，差异 `-0.003%`。该改动是分配生命周期与 CUDA Graph 稳定性
   优化，不是已证实的 TTFT 优化。
4. W13+activation 和 W2+reduce 都不能直接塞入现有 Marlin epilogue。二者分别
   跨 output-column CTA 和 expert CTA；直接融合需要新的调度/归并协议。当前不以
   原子加或在 W2 load 中重复计算 activation 走捷径。
5. W2 两阶段 FP32 grouped-tail 隔离原型已完成。60/60 个真实 shape/config 行
   逐元素零误差，但最佳候选仍比生产 BF16 路径慢 `3.23%-3.61%`，routed workspace
   翻倍；该方向未通过 promotion gate，不接 serving，也不触发 W13 配对调度重构。

## 测量边界

- GPU：单张 CMP170HX，SM80，70 SM；最终服务验收使用物理 GPU 1/2/3。
- 模型形状：hidden `K=4096`、intermediate `N=2048`、256 experts、top-k 6、
  MXFP4 group size 32、BF16 activation、block-M 48。
- W13：`[M,4096] x [256,4096,4096]`，kernel top-k 6。
- W2：`[M*6,2048] x [256,2048,4096]`，kernel top-k 1。
- 路由：一个均衡路由和两个随机 top-k 路由种子。
- 计时：CUDA Event P50/P90；每个正式候选 3 次 warmup、每 sample 10 次、5 个
  samples。
- 正确性：每个 forced config 与 auto 输出逐元素比较，门限 `atol=1e-5`。

CMP 驱动不支持 CUPTI CUDA activity，因此本文不声称 NCU occupancy、寄存器或
memory-throughput 指标。shape 表来自隔离 CUDA Event，最终决策还经过 clean Graph
PP3 服务门禁。

## W13/W2 shape 扫描

基准位于 `benchmarks/kernels/deepseek_v4_marlin_moe.py`。正式命令为：

```bash
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  .venv/bin/python benchmarks/kernels/deepseek_v4_marlin_moe.py \
  --stage both --m 1000 1666 1667 2000 4048 \
  --routing-seeds -1 8 9 --warmup 3 --iterations 10 --samples 5 \
  --output profiles/deepseek_v4/marlin_moe_config_sweep_20260814T0658Z.jsonl
```

候选覆盖 Marlin large-batch selector 的三组 thread shape：`64x256`、`64x128`、
`128x64`，每组分别强制 1/2 blocks per SM，并与 auto 比较。以下延迟是三个路由
种子的 P50 中位数：

| stage | M | auto | `64x128,b2` | 相对 auto | 生产决策 |
| --- | ---: | ---: | ---: | ---: | --- |
| W13 | 1000 | 3867.6 us | 3863.2 us | `+0.11%` | auto |
| W13 | 1666 | 4205.9 us | 4205.5 us | `+0.01%` | auto |
| W13 | 1667 | 4206.5 us | 4207.4 us | `-0.02%` | auto |
| W13 | 2000 | 5363.8 us | 5360.9 us | `+0.05%` | auto |
| W13 | 4048 | 9454.0 us | 9453.4 us | `+0.01%` | auto |
| W2 | 1000 | 2088.4 us | 2087.9 us | `+0.02%` | auto |
| W2 | 1666 | 2287.4 us | 2288.1 us | `-0.03%` | auto |
| W2 | 1667 | 2286.2 us | 2285.2 us | `+0.04%` | auto |
| W2 | 2000 | 2904.2 us | 2904.4 us | `-0.01%` | auto |
| W2 | 4048 | 5119.4 us | 5120.3 us | `-0.02%` | auto |

这张表的配置 key 是：

```text
(SM80, MXFP4-group32, BF16, block-M48,
 stage, K, N, M bucket) -> auto
```

`64x128,b2` 与 auto 的差异始终在 `0.11%` 内，不足以建立强制配置。其他 forced
候选普遍回退；`128x64,b1` 最差仅约 auto 的 `0.52x-0.55x`。正式文件有 210 行，
旧的零误差门限把两行 `1.862645149e-9` 的 W2 舍入差异标成 rejected；改为
`atol=1e-5` 后的定向复测 7/7 通过。因此这不是数值错误，也不改变配置决策。

不要把一次低 warmup 的单点结果写入 shape 表。强制配置只有在多路由、多 sample
以及 clean Graph PP3 都稳定超过 auto 后才允许进入启动器。

## Caller-owned FP32 reduce workspace

### 实现

Marlin 在 `use_fp32_reduce && !use_atomic_add` 时需要 FP32 global-reduce 缓冲。
C++ op 现在接受尾部可选参数 `fp32_reduce_workspace`，并验证：

- CUDA tensor，且与输入位于同一 device；
- contiguous、FP32；
- 元素数不小于 C++ 原公式的需求。

未传 workspace 时保留原有 C++ 内部分配路径，作为兼容和回滚。需求公式保持为：

```text
min(size_n * sorted_token_ids.numel(),
    sm_count * 4 * block_size_m * max_thread_n)
```

block-M 8 的原有倍增规则也未改变。Python 侧把最大 W13/W2 需求追加到 modular
`workspace2` 尾部，以 BF16 元素计数预留，再 view 成 FP32。W13 和 W2 顺序执行，
所以可以复用同一段；不会与仍活跃的 intermediate cache 重叠，也不会从
`apply()` 内再次向 workspace manager 申请一块可能别名的区域。

默认开关为：

```bash
VLLM_MARLIN_MOE_CALLER_FP32_WORKSPACE=1
```

设为 0 时恢复旧 workspace shape，并让两次 C++ op 各自走内部 `new_empty`，可用
同一二进制做部署回滚和 A/B。

### 容量与隔离基准

对 `M=1667, topk=6, E=256, block-M48, size_n=4096, 70 SM`：

```text
workspace = 3,440,640 FP32 = 13.125 MiB/rank
```

W13 和 W2 的 caller/internal 各 14 个 config 均通过正确性。重复隔离测量为：

| 模式 | 热路径临时分配峰值 | W13 P50 | W2 P50 |
| --- | ---: | ---: | ---: |
| caller repeat | 0 MiB | 4494.23 us | 2127.46 us |
| internal repeat | 14.0 MiB | 4441.29 us | 2123.06 us |

0.2%-1.2% 的 kernel 延迟差异是运行波动；workspace 只改变缓冲所有权，不改变
Marlin 数学或 kernel launch shape，因此不把它报告为 kernel 加速。确定收益是
measurement window 内不再发生 `c_tmp` 分配，且 CUDA Graph 使用稳定地址。

### PP3 A/B/A

共同配置为 clean CUDA Graph、5,000 prompt tokens、8 output tokens、PP3/TP1、
DSpark 7、`max-num-batched-tokens=1715`、131K deployment length、5 repeats。

| 顺序 | run id | caller workspace | TTFT 中位数 | decode ms/round 中位数 |
| --- | --- | --- | ---: | ---: |
| A1 | `20260814T072907Z` | on | 1255.684 ms | 22.258 ms |
| B | `20260814T073502Z` | off | 1260.645 ms | 21.840 ms |
| A2 | `20260814T073713Z` | on | 1261.609 ms | 22.317 ms |

两个 caller run 合并后的 TTFT 中位数为 `1260.607 ms`，相对 internal control
为 `-0.003%`。请求轨迹的 DSpark acceptance 不同，decode 小差异不归因于
workspace。采样 workload peak 在三个 rank 都低 4 MiB，但该值受 allocator cache
和采样粒度影响，也不等同于 13.125 MiB 的逻辑 workspace 大小。

详细 eager run `20260814T072612Z` 还验证了三个 PP artifact、prefill/decode/DSpark
子阶段、phase timing 和 workspace schema 均完整；Graph phase run
`20260814T073114Z` 的 W13/W2 总耗时与改动前基本相同。最终保留默认启用，理由是
生命周期与地址稳定性，性能结论为中性。

## W13 + activation 融合研究

DSV4 的 W13 产生 packed `[all gates; all ups]`，activation 使用
`silu_and_mul_with_clamp`，把 `[M*6,4096]` 变为 `[M*6,2048]`。现有 Marlin 的
FP32 global reduction 和最终 BF16 写回已经在 GEMM kernel 内完成，但 W13 的 gate
半区和对应 up 半区属于不同 `slice_col`，通常由不同 output-column CTA 产生。
因此单个 CTA 在 `write_result` 时并不同时拥有一对 gate/up 值。

对 `M=1667`：

- W13 BF16 routed 输出：78.14 MiB；
- activation BF16 输出：39.07 MiB；
- 真正的 W13+activation 融合可少一次 W13 写回和一次 activation 读取，即
  156.28 MiB/层。

PP0 实测 activation 为 `103.504 us/op`，W13 为 `4312.917 us/op`；直接消除
activation launch 的联合范围上限只有 `2.34%`，它相对整个 MoE 更只有约 1.3%。
按最小 117.21 MiB 读写量估算，当前 activation 已约为 1.19 TB/s，说明它主要是
带宽阶段，不应仅为少一个 launch 重写 GEMM。

### 可研究的结构

只有以下结构值得原型化：让一组 CTA 对同一 token-row 的 gate/up column pair
共同所有，完成最后一个 K slice 后在一个新的配对归并阶段执行 clamp/SwiGLU，并
直接写 `[M*6,2048]`。这需要修改 Marlin 的 column 调度、锁和 C_tmp layout，不能
作为现有 `write_result` 的模板分支小改。

初始原型必须严格限定为 SM80、MXFP4 group32、BF16、block-M48、DSV4
`SWIGLUOAI_UNINTERLEAVE`、无 LoRA。promotion gate 为：

1. activation 输出逐元素满足当前 kernel 误差门限；
2. W13+activation 联合 CUDA Event 至少稳定改善 3%；
3. W13 本身不能因 gate/up 配对调度回退超过节省的 activation；
4. 5K、32K 的 PP3 clean Graph TTFT 都不回退，CUDA Graph 地址稳定；
5. LoRA 或其他 activation 自动回退到当前路径。

把 activation 延迟到 W2 的 A-load 中计算不推进。相同 activation 会被每个 W2
output-column tile 重算，且破坏当前连续 BF16/cp.async load 路径，计算和调度代价
远大于一次 100 us 左右的 standalone activation。

## W2 + reduce 融合研究

W2 的 Marlin `write_result` 已经乘 top-k 权重；外部 `moe_sum` 只把 token 的 6 个
BF16 expert rows 用 FP32 accumulator 汇总成 `[M,4096]`。当前 reducer 已使用
向量化 16-byte load，问题不在权重乘法或 Python bookkeeping。

对 `M=1667`：

- W2 routed 输出：78.14 MiB；
- dense 输出：13.02 MiB；
- 真融合可少一次 routed 写回和一次 reducer 读取，同样是 156.28 MiB/层。

PP0 实测 reduce 为 `83.763 us/op`，W2 为 `2334.174 us/op`；直接联合范围上限为
`3.47%`。当前 reducer 的最小 91.16 MiB 流量对应约 1.14 TB/s，也已经是高带宽
小 kernel。

但一个 token 的 6 个 experts 由按 expert 调度的不同 CTA 产生，单个 W2 CTA
不能直接完成 dense sum。候选结构按优先级为：

1. **两阶段 grouped tail**：W2 保留 FP32 partial，在稳定的 token-major workspace
   中写入；第二阶段直接读 FP32 partial、完成 top-k6 reduction 并写 BF16 dense
   output。先验证减少 BF16 round-trip 是否能抵消 FP32 workspace 流量。
2. **persistent token-owner reducer**：少量常驻 CTA 通过 completion counter 等待
   6 个 expert partial，然后归并。它需要严格的 stream/graph 内存序和锁复位协议，
   只有两阶段原型证明 1% 以上 E2E 潜力后才研究。

以下捷径明确拒绝：

- SM8x BF16 atomic add：现有 Marlin 已因架构和 shape 禁止此路由，且会改变数值；
- 直接对 dense BF16 output 做 FP32 atomic：expert contention、非确定归并顺序和
  CUDA Graph 清零成本都没有验收依据；
- 从 Marlin C_tmp 猜测 `locks_off/slice` 映射：缺少跨 CTA completion barrier，
  容易读取未完成 partial；
- 在 LoRA 路径强制融合：当前 activation 后和 moe_sum 前都是明确注入点，首版必须
  对 LoRA 回退。

### 两阶段 FP32 grouped-tail 实机原型

隔离原型位于
`benchmarks/kernels/deepseek_v4_marlin_w2_grouped_tail.py`，由两个阶段组成：

1. Marlin W2 的最后写回可选地把 token-major routed rows 写成 FP32，不再写 BF16
   routed buffer；默认空指针仍走原生产路径。
2. 独立 Triton tail 按固定 top-k6 顺序用 FP32 累加，再写 BF16 dense output。

为使比较只反映传输和 grouped-tail 成本，第一阶段保留生产 BF16 写回的舍入和
top-k 权重语义，再把值转换为 FP32；它不是一个声称提高模型精度的 pre-rounding
路径。FP32 routed output 仅供 benchmark 传入，serving caller 不传该参数。

正式命令为：

```bash
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  .venv/bin/python \
  benchmarks/kernels/deepseek_v4_marlin_w2_grouped_tail.py \
  --m 1666 1667 2000 4048 --routing-seeds -1 8 9 \
  --warmup 3 --iterations 10 --samples 5 \
  --output profiles/deepseek_v4/marlin_w2_fp32_grouped_tail_20260814T0845Z.jsonl
```

扫描包含 `BLOCK_N={1024,2048,4096}`、`num_warps={4,8}` 的五组合法组合，共
60 行。FP32 routed output 转回 BF16 与生产 routed output 全部逐元素相等，最终
dense output 也全部 `max_abs_error=0`、`mismatch_count=0`。最佳 tail 始终是
`BLOCK_N=1024,num_warps=8`：

| M | BF16 W2 | FP32 写回 W2 | BF16 tail | FP32 tail | 生产联合 | FP32 联合 | 回退 | routed BF16/FP32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1666 | 2312.91 us | 2300.93 us | 79.36 us | 141.11 us | 2358.89 us | 2436.92 us | 3.31% | 78.09/156.19 MiB |
| 1667 | 2288.44 us | 2304.41 us | 79.46 us | 141.31 us | 2359.60 us | 2439.07 us | 3.37% | 78.14/156.28 MiB |
| 2000 | 2901.71 us | 2920.55 us | 93.70 us | 167.83 us | 2985.57 us | 3082.14 us | 3.23% | 93.75/187.50 MiB |
| 4048 | 5128.70 us | 5167.82 us | 181.35 us | 330.65 us | 5301.25 us | 5492.43 us | 3.61% | 189.75/379.50 MiB |

W2 FP32 写回相对 BF16 写回的变化仅 `-0.52%` 到 `+0.76%`，说明 Marlin store
本身不是主要失败点。FP32 tail 读取双倍 routed bytes，在所有 M 都明显慢于已经
向量化的 BF16 `moe_sum`；即使继续调 tail launch shape，也没有足够空间跨过
生产联合路径。因此：

- 不为 serving 分配 FP32 routed workspace；
- 不增加 runtime fusion 环境变量或 CUDA Graph 分支；
- 不推进 persistent completion-counter 或 FP32 atomic 版本；
- 不以本结果触发 W13 paired-column 重构。

可选 FP32 routed 写回保留为隔离 benchmark ABI，便于复现实验；默认传 `None`，
生产 Marlin 的输出布局和数值路径不变。

最小 shape 还通过了独立 CUDA correctness test。CMP170HX 上尝试
compute-sanitizer 时驱动返回 `GPU debugging features are disabled`，工具自身记录
一个 unsupported error，未实际执行 memcheck；因此本文只声称逐元素输出、schema、
编译和运行时验证，不声称 sanitizer/NCU 通过。

## 下一轮执行顺序

1. 保留 auto shape，不向生产启动器加入 `thread_k/thread_n/blocks_per_sm` override。
2. caller-owned workspace 默认启用，继续在 32K/128K 和并发请求中观察 allocator
   峰值；若出现回归用环境变量回退。
3. W2 两阶段 FP32 grouped-tail 已因联合回退和 workspace 翻倍停止；不做 atomic、
   completion-counter 或 serving 集成。
4. W13 paired-column prototype 的前置 gate 未满足，也停止；后续优化重新聚焦占比
   85% 的 W13/W2 主 GEMM 或其他已被 profile 证明的一阶热点。
5. 后续候选仍需先过真实 shape microbenchmark，再过 PP3 5K/32K Graph；不得用
   SM70 tile 参数或 synthetic GEMM 单点替代部署验收。

当前没有 fusion 实验路由接入 serving。现有 auto selector、block-M48 P0 和
caller-owned workspace 构成已验收基线；FP32 grouped-tail 是已拒绝、可复现的
隔离研究资产。
