# DSV41F 普通 decode：第二轮图内热点测量与 router 实验

日期：2026-10-02。基线提交 `4df3a185db133e63934d5739cc18582a215be609`；
本轮前 20 个 dirty 路径保留。原始证据目录：
`artifacts/dsv41f-decode-q1-round2-20261002/`。

## 1. 本轮范围和实际路径

延续上一轮：关闭 DSpark，单请求普通 decode 优先，PP5/TP1/DP1、五张
SM80 CMP170HX，模型 `/media/z/data6/models/DeepSeek-V4.1-Flash`，
CED/prefix cache、KV FP8/2.7 GB、max-seqs=8、max-batched-tokens=2048。
保留已有 M1 MoE、8-warp attention、packed-word 和 metadata 优化。
不重启 FP8 dense TileLang 路线。

同一份 32768 prompt IDs，greedy 固定预算、ignore EOS；客户端指标沿用
SSE 首个含 token 的 chunk 之后的 delta token IDs / 首尾 chunk 时间。
该指标不包含 TTFT，也不是 GPU kernel event 时间。

先做有界 source/artifact 审计，再停本服务进行一个隔离 router 实验，随后加载
诊断实例测实际 FULL q1 Graph。诊断公开范围仅 `127.0.0.1:26662`。
五 rank 均断言无 speculative config、无 speculator、decode query length=1、
V2 runner、PP5/TP1。正式生产恢复使用原无 DSpark 参数。
本轮没有修改推理源码或新增默认优化。

## 2. Router split-K：微基准更快，但本轮不接入

使用 checkpoint 第 2、20、39 层真实 BF16 router 权重，每层三个合成 activation
种子，共九组 `BF16[1,5120] × BF16[384,5120] -> FP32`。
比较既有 serial split-K 和三种 parallel split-K；后者 16 个 split、BK64、
BN16/32/64，FP32 partial 按固定顺序归约。四种候选共 36 次比较。
独立 FP64 reference、原 cuBLAS reference 和候选自身 eager/Graph replay 分开验证。

所有候选满足 FP64 容差、合成 logits 的 raw top-6 索引相同，且各自 Graph replay
与 eager bitwise 一致；但四种候选均有 0/9 个输入与原 cuBLAS bitwise 一致，
最大绝对差 `4.76837158203125e-6`。raw top-6 相同不等于真实带 scoring/bias 的
路由及全模型输出已验证。

在单张空闲 CMP170HX 上做 CUDA Graph、graph 外 64 MiB flush、3×20 次交替测量；
此时 PP 服务停止，没有并发 kernel benchmark。

| 实现 | 中位数 µs | 相对 cuBLAS |
| --- | ---: | ---: |
| 原 cuBLAS | 15.360 | 基线 |
| serial16 | 49.152 | 更慢 |
| parallel16 | 14.336 | −6.67% |
| parallel32 | 13.312 | −13.33% |
| parallel64 | 14.336 | −6.67% |

最好候选每层仅省约 2.048 µs，直接折算 40 层约 **0.08192 ms/token**，
约占上一轮 19.6 ms 客户端周期的 **0.42%**。这只是按隔离差值估算的量级，
不是严格的生产上界或 serving 收益；共享 stream 干扰可能改变差值。
综合很小的收益空间与数值差异，候选在 serving 前拒绝；未改变 router 或部署候选。
不把其他版本/配置的旧 router 延迟当作本轮热点。
证据：`router_probe.py`、`router-probe.json`、`router-probe.log`。

## 3. 已有候选的实际适用范围

q1 sparse split 选择先用 BK32，实际 partial 用 BK16 的不一致确实存在。
但真实 DSV41F 仅层 0、1 为纯 SWA；38 层走 extra top512，后者 splits 仍为 16。
实际 SWA page size=32，compressed page size 为 64/128，不套用 synthetic page128。
该调整只影响 PP0 的两层，乐观全模型空间约 40 µs（0.2%），没有追加 sweep。

mHC split=5（首 broadcast）/10（正常 K20480）已经在当前 delayed-pre 路径中。
模型每 rank 八层，走 delayed-pre 和 standalone-post；不误认为走 V4 的 fused
post-pre，也不把已经存在的 split=10 当作新优化。旧 isolated µs 或旧版本性能
不能替代当前服务证据。

MoE 的 shared expert 已有 auxiliary-stream overlap。它与 routed 的 inclusive
时间不能直接相加；细分测量必须保留 stream 和起止时间，确认 critical path。
M1 W13 packed-word 已启用，W2 的 M1 scalar-load 选择此前没有新的正收益证据，
本轮不重复那个实验。

## 4. 图内事件的校准与失败记录

第一份诊断在每 rank 插入 264–267 对 CUDA Event，并在每步读出。
32-token 请求 IDs 一致，但 root Graph 合计约 **28.757 ms**，客户端仅
**15.641 tok/s**，而同实例无插桩中位数 **50.978 tok/s**。
这种测量扰动过大，不能作为生产占比。

第一份 module hook 还遗漏了 `moe.total`，其 remainder 包含没有归类的 MoE，
不能把 remainder 命名为新的“other hotspot”。全部失败数据与源文件保存在
`fine-profile/`，不删掉失败记录，不混用新旧诊断。

为排除该偏差，追加一次诊断加载：root、coarse、mHC、attention、MoE 分别捕获
同一个 q1 descriptor，保留原 Graph manager 作为 baseline；其他 batch descriptor
继续使用原 Graph。保留原 hidden/intermediate 输出 buffer 地址和 workspace 生命周期。
每种 probe 只插入相关事件；不在 Graph replay 后执行每步同步、JSON 或写日志。
请求结束且 running/waiting=0 后读取最后一次 replay，记录 stage、layer、parent、
stream 和相对 root 的起止时间。

coarse 的 attention/MoE/Engram 兄弟区间位于同一 stream，需断言 40/40/2 个区间；
MoE wrapper 按 module 名字 suffix 匹配，避免多模态 wrapper 的额外 prefix 漏测。
短请求按轮换顺序采样，每 probe 六个 32K/16 请求；长期校准为 baseline/coarse
两组 AB/BA、32K/256。全部输出 IDs 必须与上一轮对应 reference 一致。
细分 probe 与 coarse 是不同捕获图，禁止相加成生产账本。

## 5. 校准后的实际图内热点

每 rank 事件对数为：root-only **1**，coarse **18/18/17/17/17**，
mHC **32**，attention **51/53/51/53/53**，MoE **49**。
所有 warmup、四次正式校准以及 30 次短 probe 请求的 IDs 都与原 reference 一致。
request 完成后才读 event；没有 per-step event 同步。

| 同实例 32K/256、两组 AB/BA 校准 | 无插桩 baseline | coarse probe |
| --- | ---: | ---: |
| 客户端中位数 tok/s | 51.022986 | 47.235538 |
| 客户端中位数 ms/token | 19.599077 | 21.170598 |

coarse 客户端吞吐下降 **7.423%**。它远小于第一份失败探针，但仍有可测扰动，
不能称为无插桩性能，也不能用来接受任何生产优化。

每个样本先求五 rank 的 event 之和，再对六个相同位置的 last replay 求中位数。
root-only 模型区间 **18.141696 ms**，coarse 模型区间 **19.450880 ms**，
后者比前者高 **7.216%**。以下兄弟区间无嵌套、同 stream、不重叠：

| coarse Graph 内组件 | 五 rank 对齐后中位数 ms | 诊断 Graph 占比 |
| --- | ---: | ---: |
| attention | 8.901120 | 45.76% |
| MoE | 8.280576 | 42.57% |
| Engram 两层 | 0.236544 | 1.22% |
| 其余未细分区间，含 mHC、embedding/final norm 等 | 2.033152 | 10.45% |
| model root | 19.450880 | 100% |

中位数不是可加统计量，因此表中分项中位数之和可能与 root 中位数有微小差异。
这些百分比描述 **coarse 诊断图**；7.2% 的扰动使其不能成为精确的生产占比。
但 attention 与 MoE 合计约 **88.3%**，足以确定优先级；只优化 metadata/router
不足以解释或消除当前主要成本。

| 各 rank root-only 中位数 ms | PP0 | PP1 | PP2 | PP3 | PP4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 实际模型 Graph | 3.436 | 3.595 | 3.777 | 3.516 | 3.818 |

PP2/PP4 对应 HBM clock 较低的两张卡，且层形状/复用边界也有差异；未做交换卡
实验，不把该相关性直接归因为带宽。Graph root 不含 staging、宿主准备、PP
handoff、logits 或 sampling，不与上一轮包含交接的 forward span 直接相减。

## 6. 细分结果及不能误判的部分

| 细分 probe | probe root ms | 相对 root-only |
| --- | ---: | ---: |
| mHC | 20.251648 | +11.63% |
| attention | 20.875776 | +15.07% |
| MoE | 21.791744 | +20.12% |

细分图的扰动更大；下列数值只用于定位，**不作为生产耗时预测或精确收益预算**。

attention probe 中，`attn.prepare` 为 **4.802 ms**，output projection 为
**4.476 ms**，其中 wo_a 为 **2.396 ms**。`attn.prepare` 的实际函数是
`_prepare_and_attn()`，包括 Q 投影/RoPE/KV insert、compression/indexer 及 sparse
attention；它不是宿主 metadata，也不是 sparse partial 单算子的净耗时。
indexer logits/top-k 分别为 **0.239/0.557 ms**，嵌套在 prepare 中，不能再相加。
本轮没有把 prepare 中剩余时间全部归因给 partial kernel。

wo_a 虽耗时较大，但当前 Ampere 类设置 `is_bmm=True`；
`_get_cached_wo_a_bf16()` 和 `rocm_inv_rope_einsum()` 使用 BF16 cache 与 grouped
`torch.einsum`。`text_config` 为 groups=8、rank=1024、heads=64、head-dim=512，
运行权重 `[8,1024,4096]`，每层 **67,108,864 B**，40 层为 **2.684 GB**。
不能按 checkpoint FP8 的 33.55 MB/layer 算成低带宽。

本次 wo_a probe 约 59.904 µs/layer，按这些运行字节估算约 **1120 GB/s**；
只是诊断有效带宽，使用过去的 1207 GB/s slab 数据会得到约 92.8%，不当作本轮
同口径硬件上限。它与历史实测接近带宽上限的结论一致。
已有 einsum→BMM 和多路 MXFP8 Marlin 负结果见
`vllm170-dsv41-pp5-decode-profile-2026-09-12.md` 的 output projection 记录；
本轮不重做，也不恢复放弃的 FP8 dense TileLang。

MoE probe 中 W13 为 **3.971 ms**、W2 为 **2.686 ms**；先按样本求二者之和、
再求中位数为 **6.654464 ms**。其余 router/quant/reduce/routing 区间也带探针开销，
不能将 MoE probe 的 11.111 ms 套到 coarse 的 8.281 ms 上。
路由专家的 W13/W2 是下一轮最有价值的具体研究对象。

shared expert inclusive 为 **2.174 ms**。跨 stream 记录确认六个样本的
**240 次层调用均在 routed 结束前完成**，与 router 重叠约 1.595 ms，
与 routed 的直接重叠约 0.003 ms，其余处于路由准备等间隙。
该关系只描述当前细分诊断图，已足以排除把 shared 的 2.174 ms 再当新增串行项；
不是所有 batch/shape 都完全遮住的证明。证据为 `moe-overlap.json`。

mHC 四类 annotated pre/post 对齐后合计 **2.413056 ms**，但该探针把 root 增加
约 2.11 ms；还遗漏 rank-final post 等未 annotation 的调用。因此不将 2.413 ms
当作净 mHC 生产成本，更不能与 coarse remainder 相加。

## 7. 下一轮的有界实验顺序

第一优先是 **M1 routed MoE 的生产对照**：当前 q1 能触发已启用的 TileLang M1，
过去只有 isolated apply 正收益，DSpark q6 的可达性结论不能替代普通 q1。
下轮先在同一次诊断加载中预先捕获原 CUTLASS fallback 与当前 TileLang M1 图，
关闭所有 timing probe，做交替 32K/256、完整 token IDs 和双请求回归。
仅比较 GEMM backend，保留相同 quant、activation、routing、workspace 和 shared
overlap；不能把同时关闭 activation fusion 的结果混成 GEMM 收益。
该候选仍保持 DSV41F/SM80/E384/top6/H5120/I2304、M1 的原有身份与形状门，
其他 batch 继续 fallback。先确定实际端到端收益，再决定是否调新参数或扩大适用范围。

第二优先才是 prepare 内 q1 partial/reduce 与 Q 投影的分离，使用少量、单层或
单一算子事件先校准，避免重复本轮多探针的测量膨胀。38 层 top512 主路径优先，
两层纯 SWA 的 split 调整不作新的大扫描。

本轮确实实现并验证了 diagnostic 图捕获/切换、deferred readout 和 router 原型，
但没有接受新的生产优化；不把诊断工具变快、isolated kernel 变快或恢复后的
吞吐漂移写成模型提速。

## 8. 恢复和验证

正式服务 `dsv41f-manual.service`，MainPID **3635473**，公开 `0.0.0.0:26661`，
DSpark 关闭。实际 argv 无 speculative config、worker extension；无 dev mode 或
诊断 PYTHONPATH。phase/step timing 均关闭，target prepare-ahead 显式为 false。
诊断 unit inactive，26662 关闭；health=200、dev collective RPC=404。

三次 32K/256 正式核验分别为 **50.946/50.418/50.418 tok/s**，中位数
**50.418063 tok/s、19.834161 ms/token**。这只描述重启后当前服务性能，
不是优化 A/B；与先前约 51 tok/s 的差异没有归因给任何代码收益或回退。
单请求 256 个、双请求各 128 个 output IDs 全部与上一轮对应 reference 一致，
speculation counters 无增长。五张 GPU UUID/PID 属于正确 service cgroup，
实际模型路径、监听 PID、V2 runner 和运行源码 SHA256 均核对。

CPU 原 PP suite **31 passed**；新增 artifact controller、harness、analysis 和
production verifier 的 ruff/format/py_compile 通过；本文 pre-commit hooks 通过。
本轮前 20 个 dirty 路径 SHA256 全部保留，没有自动提交。

```bash
# 只读取归档数据，不发起 GPU 实验：
.venv/bin/python artifacts/dsv41f-decode-q1-round2-20261002/analyze_graph.py

# 原 PP gate/fallback 回归：
.venv/bin/python -m pytest tests/v1/worker/test_pp_utils.py -q
```

精确启动 argv、runtime source hashes 和诊断脚本 hashes 保存在 `launch.json`、
`production-launch.json`；完整请求和 token IDs 为 `graph-run.json` 与
`production-validation.json`。第一次失真诊断单独归档到 `fine-profile/`。
