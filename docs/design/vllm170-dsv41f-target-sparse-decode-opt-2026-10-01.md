# DSV4.1-Flash q6 target：稀疏 attention 优化与生产验证

日期：2026-10-01。基础 HEAD：`ae32948b167dbbb1cc87dbe4f965919f9705dc1c`。

## 结果

**已部署一处 q6 sparse-decode launch 调优。生产 target Graph 的五个 rank
中位数之和从 42.168 ms 降至 39.834 ms，减少 2.335 ms（5.54%）。**
这是本轮同配置 A/B 的 rank-local CUDA Event 统计，不是 PP wall time。
每个 rank 各有 21 个 q6 FULL Graph 样本，五个 rank 均下降。

| 测量 | 4 warp 基线 | 8 warp 候选 | 耗时下降 |
| --- | ---: | ---: | ---: |
| C1 partial + reduce，冷 L2，µs | 165.888 | 102.400 | 38.27% |
| C2 partial + reduce，冷 L2，µs | 164.864 | 102.400 | 37.89% |
| 生产 target Graph，rank 中位数之和，ms | 42.168 | 39.834 | 5.54% |

微测使用合成 query/KV 和随机物理索引；生产测量使用真实模型请求。两种证据
分开保留。按模型 20 个 C1 层和 18 个 C2 层折算，微测预计节省 2.394 ms，
与生产观测的 2.335 ms 接近；这个折算不是图内组件的精确分账。

候选的三次 32K/256 无插桩请求分别为 **86.17 / 87.97 / 88.83 tok/s**，
中位数 87.97。三次输出与改动前三次请求的全部 256 个 token ID 完全相同，
候选的接受量一致性 gate 通过，每次 44 draft 轮、211/220 draft token 接受。
基线三次吞吐严重波动且接受量 gate 失败，因此不据此计算端到端提升百分比。

服务 `dsv41f-manual.service` 已加载候选，health 200，running/waiting 均为 0，
phase control 已移除。只重启一次，没有进行新一轮六个 q 扫描。

## 为什么选择这个热点

上一轮截距复核已经说明，33.87 ms 是回归参数，不能作为一个可以删掉的
“进入验证”动作。本轮从实际 q6 算子形状找可以测量和验证的改动。

调用链是 DSV41 Ampere attention → 复用的 AMD/ROCm Triton sparse 实现 →
`rocm_sparse_attn_decode` → `_rocm_sparse_attn_decode_ragged_triton` →
partial/reduce。名称中有 ROCm，但生产 SM80 确实使用这里的 CUDA 分支。

当前 CMP170HX 有 74 个 SM。q6、64 heads、512 head dimension、SWA128、
top512 的默认 heuristic 选择 3 splits，partial grid 为 `(6,3,4)`。
原 launch 保留了 gfx950 调优的 `BLOCK_H=16, BLOCK_K=32, num_warps=4`。
搜索现有 benchmark 的 8 个 partial 和 16 个 reduce 配置后，选定只改 warp 数。

| 编译器报告的 partial 资源 | 4 warp | 8 warp |
| --- | ---: | ---: |
| registers/thread | 255 | 255 |
| `n_spills` | 8 | 0 |
| shared memory，bytes | 88064 | 88064 |

8 warp 增加参与同一 CTA 的线程并消除编译器报告的寄存器溢出，grid、共享
内存大小保持相同。这给出了加速机制的直接线索；本轮没有做 Nsight counters
实验，不把全部收益唯一归因于溢出消除。

## 实现范围

运行时只修改 `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` 的 launch 选择：

- CUDA SM80 family 的 split decode；
- `num_queries == 6`、`num_heads == 64`、`num_splits == 3`；
- 有 compressed extra cache，ragged extra 索引总数为 `6 × 512`；
- 满足上述条件使用 8 warp，否则继续使用 4 warp。

head dimension 的 512/448/64 契约由现有校验保证。分块、split 数、softmax
处理顺序、reduce 配置、workspace 和缓存格式均未改变。这个公共算子的相同
形状调用都会命中该配置桶；服务验证针对当前 DSV41F PP5/q6。

FP8 dense 继续使用 Marlin。已有 MoE TileLang 与 activation 改动的三个源文件
SHA256 在重启前后保持一致。本轮没有修改这些推理文件，也没有改启动参数。

benchmark 扩展复用原 `DecodeCase`，增加独立 extra-cache block size 和 cache
rows 参数，支持 C1 block128 与 C2 block64。新增
`benchmarks/kernels/deepseek_v41_target_sparse_decode.py` 使用相同 kernel、
一次调用的 CUDA Graph、每个样本计时外 64 MiB L2 flush，保存 3×20 个原始样本。
CUDA Event 有约 1 µs 的量化，报告采用中位数。

target 的 causal SWA 是 128；另以 main length 133 检查非整块尾部，避免仅在
整齐的 128 长度上验证。本文微测表采用 133 的确认结果；128 的完整搜索也
保留在同一 artifact 目录。133 不是对 target 逻辑窗口的修改。

## 正确性

扩展已有 `test_sparse_decode_split_and_caller_output_match`，增加 q6/top512、
3 splits、C1/C2 cache layout、caller-output 开关。通过 public decode API
对比新 launch 与强制旧 4-warp launch，要求 `atol=0, rtol=0`；同时保留
与 unsplit reference 的既有容差比较、有限值、输出 buffer 和 Graph replay 检查。
原 q1/q2/q4/q8 的测试仍保留。

最终针对该行为的测试 **12 passed, 17 deselected**。在主窗口 128 的前一轮，
连同已有 paged-SWA 测试为 13 passed；主窗口 133 的最终运行是上述 12 个用例。
C1/C2 微测候选与旧配置均逐位一致。这是已测输入的等价证据，不是所有输入
上的数学证明。

模型验证共包含改动前后各两次 128-token 请求，以及各三次 256-token 请求。
所有对应输出 ID 完全相同。两次 profiled 请求的 streaming token 分组也一致，
各为 23 个有 token 的 chunk，五 rank 各 21 个 q6 FULL Graph 样本。
无插桩的两次 128-token 请求各有 23 个 chunk，但中间两个接受分组略有不同，
因此该短请求的吞吐差不能当作精确的同轮分解。

这是固定 32K prompt、temperature=0 的回归验证，尚不包含多请求压力或全面
模型质量评测。

## 生产 Graph A/B

保持同一个 transient unit 的定义，基线使用原已加载进程，候选一次重启后
使用新进程。`unit-before.txt` 与 `unit-after.txt` 逐字节相同。
PP5/TP1、每 rank 八层、DSpark=5/q6、greedy draft、FULL Graph、CED prefill、
1M max-model-len、KV block128、prefix caching、现有 MoE TileLang 配置保持一致。
当前 strict BF16 math 关闭，不能混用上一轮 strict component 测量的数值。

每个版本先发一个无插桩 128-token 请求，然后在第二个请求收到首个 token 后
才开启 phase hooks，结束后关闭。这样避免把 prefill/JIT 纳入 decode trace。

| rank | 原 Graph p50 ms | 候选 Graph p50 ms | 减少 ms |
| --- | ---: | ---: | ---: |
| PP0 | 6.418 | 6.120 | 0.298 |
| PP1 | 7.991 | 7.511 | 0.480 |
| PP2 | 8.720 | 8.199 | 0.521 |
| PP3 | 9.327 | 8.843 | 0.483 |
| PP4 | 9.712 | 9.160 | 0.552 |
| 中位数之和 | 42.168 | 39.834 | 2.335 |

PP0 只有六个 compressed attention 层，其余 rank 各八个，与较小的收益相容。
PP2/PP4 所在卡 HBM 上限为 1458 MHz，其余生产卡为 1728 MHz；未改变设备时钟。
Graph input staging p50 仍是每 rank 约 0.017–0.040 ms。

这些 event 的 step ID 在不同 rank 不同，统计器明确标注未跨 rank 对齐，
不称作真实 PP 串行 wall interval。只有一个版本 A/B，也不报告统计置信区间。
phase profiling 的整轮开销较大；使用相同采集方法比较 Graph 区间，不用它的
整轮速度替代无插桩生产速度。

## 端到端测量与剩余工作

改动前的三次 256-token 请求是 58.60 / 13.73 / 71.37 tok/s，工具以
`baseline_acceptance_equal=False` 失败，原始记录保留。其全局计数差甚至出现
generation=329、313，与单请求 256 不符，无法把这些计数当作准确的单请求
接受量。异步计数、并发流量及主机抖动的贡献未分别证实，不作单一归因。

同期主机 2 GiB swap 已满，memory PSI 的 some/full 持续约 25–29%；这是实际
观察到的测量限制。没有调整系统内存、其他服务或 GPU 配置。
无插桩 128-token 请求的 decode 速度为 83.20 → 85.31 tok/s，变化约 +2.5%，
只作为单对短请求观察。候选三次 256-token 请求稳定且 gate 通过，但基线不合格，
所以不能宣称 58.60 → 87.97 是本改动的端到端收益。

这轮确定并消除了一个实际 q6 kernel 配置热点，不能据此重算 q≥2 的历史截距。
剩余优化需要分别对待：MoE W1/W2 的真实生产路由工作量、indexer 在 FULL Graph
中的捕获宽度、metadata builder 的公共工作。前两项位于 Graph 内；metadata
主要影响宿主与整轮重叠，不能加进 Graph 的 39.834 ms 作为组件分账。
当前 PP5 不支持 `--enable-return-routed-experts`，下一轮不得用该开关捕获路由，
也不得由 target 曲线斜率代替真实 expert ID 观测。

## 复现与证据

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python -m benchmarks.kernels.deepseek_v41_target_sparse_decode \
  --quick --main-len 133 --ratio 1 --samples 20 --output /tmp/c1-q6.json
# ratio 2 测 C2；去掉 --quick 搜索全部配置。

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python -m pytest \
  tests/kernels/attention/test_dsv4_sparse_prefill_sm80.py \
  -k sparse_decode_split_and_caller_output_match -q
```

全部原始证据在 `artifacts/dsv41f-target-opt-20261001/`：

- `sparse-c{1,2}-q6.json`：主窗口 128 的全配置搜索；
- `sparse-c{1,2}-q6-confirm.json`：主窗口 133 的最终比较、资源与原始样本；
- `baseline-boundary/`、`candidate-boundary/`：请求、服务状态、原始 trace 和统计；
- `summarize_boundaries.py`：保留 scope 与样本数的统计脚本；
- `baseline/request.json`、`candidate/request.json`：三次 256-token 原始请求记录；
- `comparison.json`：跨版本精确 ID、接受量 gate、Graph 和 kernel 比较；
- `activation-before.json`、`activation-after.json`、`unit-{before,after}.txt`：
  服务、源码 SHA256、GPU 归属和内存压力；
- `production-eight-warp-cache.json`：重启后生成的生产 8-warp Triton 编译 metadata；
- `source-manifest.json`、`sparse-tests-confirm.log`、`precommit.log`：源码和检查证据。

四个本轮 Python 文件的 pre-commit 全部通过，包含 ruff、format、mypy 和
torch API 检查；`git diff --check` 通过。

提交前复核的 split decode 与 paged SWA 专项检查为 13 passed。合并运行 attention
文件与代价模型测试为 38 passed、3 failed；失败项均为文件前部已有的 single-request
metadata 测试：一项的测试配置缺少 `additional_config`，另外两项涉及 builder
缺少 `_get_token_to_req_indices` 接口。
将 HEAD 中的原始测试文件复制到独立路径后，三项均重现相同失败；本次未修改这些
测试或 metadata builder，不能将完整测试文件标为通过。日志保存在
`artifacts/dsv41f-target-remaining-commits-20261001/` 的 `focused-tests.log`、
`tests.log` 和 `baseline-metadata-tests.log`。
