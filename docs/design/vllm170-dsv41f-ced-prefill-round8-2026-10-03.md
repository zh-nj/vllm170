# DSV41F prefill 第八轮：流程折叠与有界并行

日期：2026-10-03。基线提交 `a33bd2b5cb`，保持 DSpark=5、CED 开启、
PP5/TP1/DP1、2048/8 scheduler，以及第六轮的 M2044
BM128 pairs / BM64 tails / BM16 small tails。
原始记录位于 `artifacts/dsv41f-ced-prefill-round8-20261003/`。

本轮实现并验证了三个离线候选，没有改推理源码、运行开关或 native
extension，也没有重启生产服务。路由元数据的四次调用可以融合为一次，
局部 GPU 和 CPU 提交成本明显下降；完整 eager apply 的代码路由收益为
0.58%/1.42%，没有同时通过预设筛选条件，未加载候选服务。
共享专家并行的两种安排也未通过筛选。本轮没有端到端提速结论。

## 从实际流程选择机会

源码中的 M2044 串行顺序为：router gate → shared MLP → expert select →
BM64 align → route partition/copies → W13 INT8 quant/GEMM → clamped
SILU → W2 INT8 quant/GEMM → weighted reduce → shared/routed combine。

`SharedExperts` 的默认 aux-stream 门限为256，因此 M2044 不会自动
并行 shared MLP。第六轮的 route partition 用一次扫描，再分别复制
BM128/BM64/BM16 三份列表。以下研究的是这些环节的组织方式，
保留已验收的计算、分块和全局 routed slot 语义。

| 环节 | 能否折叠、融合或并行 | 本轮处理及约束 |
| --- | --- | --- |
| shared MLP 与 routed MoE | 数据上可以并行，资源上可能争用 | 测全程并行、只与准备阶段并行；没有扩大生产 aux 门限 |
| route partition + 三份 copies | 可以合成一次 GPU 调用 | 实现单次扫描/复制候选，验证 slot、expert、count 和完整 apply |
| 大行 SILU + W2 quant | 已有 kernel 可以融合 | 第一轮已验证但请求投影收益约4.9ms/rank；本轮不重复该实验 |
| native BM64 align + partition | 可以研究更深的接口融合 | align 包含跨 CTA 计数与排序依赖，不能直接在当前排序 kernel 中读取未完成的其他 CTA 输出；本轮不改 native |
| attention combine 前的 host fence | 可以研究缓冲区复用与限制排队深度 | 旧证据的约2.2s host wait 是等待已排队工作；约8ms GPU gap 才接近可研究的空隙，且仍不是纯 fence 成本 |
| PP 启动阶段的小 prefill chunks | 可研究合并计算或提前准备 | 同时承担流水线填充和并发请求入队；不能仅按 launch 数删除 startup window |

流程入口见 `moe_runner.py:_apply_quant_method/_forward_impl`，共享专家
调度见 `runner/shared_experts.py`，metadata 见
`experts/dsv41_sm80_hybrid_prefill.py`，实际 apply 见
`experts/mxfp4_int8_sm80.py`。shared 模型是
`vllm/models/deepseek_v4/nvidia/model.py` 中的 `DeepseekV4MLP`。

第五轮的 DSpark-on 代码语料 profile 中，shared W13/activation/W2
三个叶子阶段的 GPU 工作量之和为 PP0/PP1/PP2 的
207.16/190.58/100.05ms；M2044 调用数为120/120/60。
这些是 per-rank 工作量，不能跨 rank 相加为 TTFT。外层 hook 和
`moe.shared_experts` 包含重复或重叠区间，未纳入上述和。
该 profile 早于 BM16 上线；它用于定位流程机会，不作为当前基线耗时。
重新提取结果保存为 `existing-flow-evidence.json`。

## 共享专家并行：依赖独立并不保证更快

只使用空闲物理 GPU1 CMP170HX，UUID
`GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216`，SM80、74 SM。
正式 PP5 持续占用物理 GPU2–6。

读取 checkpoint layer0/14 的真实 shared W1/W3/W2 与 scale，按
生产 Marlin 格式准备；激活使用 BF16 合成输入。shared activation
沿用 clamp=10、alpha=1、beta=0。两个线性层的 workspace 独立，
也与 routed workspace 分离。输入只读，输出在 aux stream 完成事件
后才被主 stream 消费。fork/join 的 Event 开销包含在对照中。

routed 部分复用第五轮实际 M2044 corpus IDs/weights，以及六热专家
合成路由；packed expert weights 和两个 stage 的输入为合成数据。
W13/W2 在这个筛选中是 GEMM pair，没有经过完整 activation/quant
数据链、reduce、attention 或 PP，所以它不是完整推理模拟。
CUDA Event + Graph，图内4次调用，3轮交替对照，各7个样本，地址与
权重复用；Graph replay 后 shared 和全部 routed slots 逐位一致。

| 路由 | 串行 µs | shared 全程并行 µs | 耗时变化 |
| --- | ---: | ---: | ---: |
| corpus layer0 | 15440.13 | 15752.96 | +2.03% |
| corpus layer14 | 13360.90 | 13746.43 | +2.89% |
| 六热专家 layer0 | 7034.88 | 6980.61 | −0.77% |
| 六热专家 layer14 | 7051.01 | 6991.62 | −0.84% |

证据：`shared-screen.json` 和对应源码快照。代码路由的退化超过基线
自身波动，不能扩大 `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`。
广泛专家分布下资源争用、热路由下较易重叠，是结果支持的解释；没有
HBM counter，不能确定全部退化都来自带宽。

随后缩短重叠范围：aux stream 跑 shared，主 stream 仅跑 router、
top-k、align、W13 quant 和 metadata，进入 routed W13 GEMM 前汇合。
gate 用实际 checkpoint weight/bias 与生产 pinned GEMM；top-k 结果
不改变捕获的 GEMM 路由，layer0 的 hash routing 也没有在该代理中复现。

首次准备阶段对照把串行 gate 放在 shared 后，复核源码后纠正为
gate → shared → select/align/quant。旧数据和源码保存在
`prepare-screen-before-gate-order.*`，不采用旧数字做筛选结论。
最终数值见 `prepare-screen.json`；结果是 corpus 两层均在基线波动内，
六热专家 layer14 改善约0.86%，仍不足以支持生产实验或全局推广。

## 元数据融合：四次调用折叠为一次

离线 `FusedMetadataRoutes` 继承已验收的 `SmallTailRoutes`，原始
BM64 align 和全部 scratch capacity 不变。每个 CTA 读取相同的 expert
列表，通过 prefix scan 得到 pairs/tails/tiny 的目标位置，直接把
原 sorted slots 写入 BM128/BM64/BM16 目标列表；每 CTA 处理8个源块。
CTA0 更新三份 count，整个 kernel 完成后才开始 GEMM。

这里用重复的小列表扫描换掉 intermediate indices 和三次 copy launch。
没有 host count read、atomics 或新 stream，也没有改变专家权重读取量。
同专家相邻块仍优先配对，只有第17个 slot 为 sentinel 的奇数尾块
转给 BM16。W2 保留原 routed slot 索引，native 使用全局有效 slot
上限12264；不能改成某份分组列表的局部 count。

元数据单独计时包括 update 的完整 GPU 工作，排除 initial align。
CUDA Graph 内64次调用，3轮交错，各7个样本。另用有界 eager 循环
每轮100次 update，计提交区间并在末尾同步；提交区间可能包含 runtime
backpressure，不将它称为纯 CPU 指令时间。

| 路由 | metadata GPU µs，原→融合 | GPU 降幅 | eager 提交 µs，原→融合 |
| --- | ---: | ---: | ---: |
| corpus layer0 | 11.776 → 5.984 | 49.2% | 115.51 → 40.92 |
| corpus layer14 | 10.112 → 5.232 | 48.3% | 113.26 → 42.01 |
| 六热专家 | 10.016 → 4.368 | 56.4% | 111.42 → 40.40 |

证据：`metadata-cost.json`。提交时间节省约71–75µs/调用；PP0 的
120个 bulk 调用对应约9ms 提交工作量，但这些工作可被 GPU 遮住，
不能直接换算为9ms TTFT 改善。

将 update 与一个完整 GEMM stage 一起计入后，结果如下。基线是
第六轮已上线的 `SmallTailRoutes`，两条路径的分块和计算完全相同。

| 路由 / stage | 原路径 µs | 融合 metadata µs | 改善 |
| --- | ---: | ---: | ---: |
| corpus layer0 W13 | 9208.58 | 9204.22 | 0.05% |
| corpus layer14 W13 | 7813.63 | 7807.23 | 0.08%，低于波动 |
| corpus layer0 W2 | 4884.22 | 4878.34 | 0.12% |
| corpus layer14 W2 | 4147.20 | 4142.85 | 0.10% |
| 六热专家 W13 | 3657.98 | 3650.05 | 0.22% |
| 六热专家 W2 | 1990.91 | 1986.82 | 0.21% |

证据：`fused-metadata.json`。Graph 隔离掉 host launch 间隙，显示
局部约5µs 节省并不会改变毫秒级 GEMM 的主要成本。
这里各 stage 都计一次 update，而真实 apply 只在 W13 前更新一次；
不能把 W13/W2 的两份 update 节省相加为真实 apply 收益。

## 完整 eager apply：确认 CPU 收益有没有被遮住

为补齐 Graph 无法回答的 host 影响，只在 GPU1 测一次有界完整
生产 `Mxfp4Int8Sm80Experts.apply` 对照。包括 initial align、一次
metadata update、W13 quant/GEMM、clamped SILU、W2 quant/GEMM 和
reduce，使用预分配且 aliased 的 workspace/output。两臂沿用 M2044
pure eager gate，只把 route 对象从 `SmallTailRoutes` 换成融合候选。

使用实际 corpus IDs/weights，hidden/packed weights 为合成；scale
缩小避免无意义的饱和，clamp 与实际模型一致。每个样本前排空 stream，
因此包含 host 未及时提交下一个 kernel 的空隙；这比生产中可能已有
GPU 工作遮盖的状态更容易暴露 CPU 收益，仍不替代服务验证。
3轮 AB/BA/AB，各7个样本，记录 wall、Event 和 enqueue 原始值。

在测量前规定：两个 corpus 完整 apply 的 wall 改善都必须超过1%
并超过各自基线 spread，才加载私有服务。该条件不是端到端验收标准。

| 路由 | 原 apply wall µs | 融合 apply wall µs | 改善 | 基线 spread |
| --- | ---: | ---: | ---: | ---: |
| corpus layer0 | 14684.48 | 14598.58 | 0.58% | 0.57% |
| corpus layer14 | 12661.63 | 12481.88 | 1.42% | 1.05% |
| 六热专家 | 6350.29 | 6256.08 | 1.48% | 0.21% |

所有输出和 W2 slots 有限且 BF16 逐位一致，原始数据在
`eager-apply-cost.json`。筛选 `service_screen_passed=false`。
因此没有为该量级执行一次私有加载加一次生产恢复，也没有为微小差异
扩大样本或追逐离群值。结论是完整局部 eager 路径存在小收益线索，
生产 TTFT 收益未知；不能宣称“端到端无收益”，也不接入生产默认。

## 正确性与保留范围

融合候选的18个独立参考覆盖 W13/W2 的
M=1/16/17/63/64/65/127/128/129，包括非恒定 activation/channel
scale、所有有效 slots、expert 身份、输出 sentinel。实际 M2044
layer0/14 及六热专家两 stage 均与现有分块输出逐位一致。

扩展已有 pytest，不新增一次性 test benchmark 文件：metadata
hot→spread→hot Graph replay 检查 count 没有残留；完整生产 apply
测试覆盖 aliased output/workspace、两 GEMM、activation、quant、
reduce 及变化路由。GPU1 测试结果 **38 passed、247 deselected**。
相关 pre-commit hooks 与 `git diff --check` 通过。

保留两个 kernel benchmark 和现有 suite 的小扩展，供后续组合候选
使用；`FusedMetadataRoutes` 仅在 benchmark 中定义，没有生产开关。
第七轮报告和第六轮提交说明保持原样。安装及 build native 均未替换。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
PYTHONPATH=/mnt/data/apps/vllma100/vllm170 \
.venv/bin/python benchmarks/kernels/deepseek_v41_moe_hybrid_prefill.py \
  --fused-metadata \
  --expected-uuid GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
  --out artifacts/dsv41f-ced-prefill-round8-20261003/reproduce-metadata.json

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
VLLM_DSV41_ROUTER_GEMM_ALIGN=split VLLM_MARLIN_M_ALIGN=1 \
PYTHONPATH=/mnt/data/apps/vllma100/vllm170 \
.venv/bin/python benchmarks/kernels/deepseek_v41_moe_shared_overlap.py \
  --prepare-only \
  --expected-uuid GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
  --out artifacts/dsv41f-ced-prefill-round8-20261003/reproduce-overlap.json

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python -m pytest \
  tests/kernels/quantization/test_mxfp4_int8_sm80.py -q -k hybrid_prefill
```

元数据 host 消融和完整 eager apply 的可执行脚本、源码快照、全部
原始样本及失败日志也保存在本轮 artifacts 中。没有 q 扫描、native
编译或新增 profiler 会话；新增硬件测量均限制在 GPU1。

## 生产复查与后续优先级

正式服务保持 MainPID 4007178，五个 worker 4010330–4010334，
模型路径 `/media/z/data6/models/DeepSeek-V4.1-Flash`。DSpark=5、CED、
hybrid 和 BM16 开关开启；无诊断 worker、dev mode 或 phase/step 插桩。
本轮冷 corpus32K/64 和同盐 prefix32K/16 共80个输出 IDs 与既有参考
一致，DSpark draft/接受计数为正，冷请求 prefix hit=0，同盐命中32512。
这些是交付检查，不从历史 TTFT 相减宣称收益。

已检查正式 unit、实际 argv/env、GPU 归属、worker 映射的 native inode、
16份运行源码及 native SHA256；封存清单中的离线 benchmark 正在修改，
因此单独排除它，未声称17份全部不变。health=200、dev RPC=404，
26662无 listener，running/waiting 请求数均为0，GPU1 无残留计算进程。
证据：`production-check.json`、`final-state.json`。

后续流程研究优先区分两类成本：必须等待的 GPU 数据依赖，以及可被
已有 GPU 工作遮住的 host 提交工作。当前小粒度 metadata 融合可保留
为组合候选，但不继续单独扩大量测。更大机会在 PP/CED chunk 的工作
组织和 routed GEMM 的权重侧成本。

若研究 chunk 折叠，先从现有 chunk 时间线建立流水线模型：启动窗口
按 long-prefill threshold × max-concurrent-batches 保留并发入队机会，
合并 chunk 同时会改变专家分布、权重重读、PP 填充和请求公平性。
必须同时覆盖 solo cold TTFT 与中途到达的第二请求，不能只看 solo。
若研究 attention fence，先给临时张量和 lane2 workspace 建立明确的
复用/完成边界；GPU3 容量紧，不能通过增加 live buffers 换取未证明
的 overlap。共享专家是 MXFP8 Marlin、routed 是 MXFP4 INT8，不能简单
作为“第七个专家”塞进同一 grouped GEMM 而保持当前数值合同。
