# Qwen3.8 Flash-Next AutoRound：关闭 MTP 的 decode 调查

## 范围和现场

目标模型是 `/media/z/data6/models/Qwen3.8-Flash-Next-W4A16-AutoRound`。
本轮只研究关闭 MTP 后的 decode，使用物理 GPU 2/3、PCI bus 顺序、
CMP170HX SM80、PP2/TP1 和 `26,22` 分层。CPU PLE offload、BF16 激活、
INC AutoGPTQ INT4 对称 group-128 routed experts、Marlin WNA16 保持一致。
已有 QSA merged projection 和 fused prepare 开启。

目标仓库 HEAD 为 `e2e327dc53b93acb85c8b56b526d369eb5461421`，存在保留的
本地修改。源头 `/mnt/data/apps/vllm` HEAD 为
`31ebe18bb63303be83137a59f668ca9812ac3579`；另审查已有的 `origin/main`
对象 `0e468adb43d60484b5a6546e0124b686fd76baf3`，未 fetch 或切换分支。
精确代码哈希、命令和现场记录在本轮 artifacts 中。

用户要求 DeepSeek 保持关闭；本轮未恢复 `dsv41f-manual.service`。

## 已完成的 MTP-off 基线

每种负载一次 warmup、五次正式测量，每个请求输出 256 tokens。
使用记录的精确输入 token IDs、temperature 0、seed 42、独立 cache salt。
所有请求要求找回输入 40% 位置的验证代码并继续输出正常文本。
30 个正式请求均通过；MTP drafts/accepted tokens 和 prefix hits 增量为零。

| 输入长度 | 并发 | 单请求 decode 中位数 tok/s | 全部请求同时 decode 窗口 tok/s | TTFT 中位数 s |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 1 | 80.03 | 80.03 | 0.320 |
| 32768 | 1 | 78.53 | 78.53 | 3.515 |
| 1024 | 4 | 62.59 | 254.04 | 0.745 |

Decode 从首个含 token 的 SSE 帧之后计时；并发窗口取四个请求 decode
时间的交集。HTTP/SSE 时间不是单个 CUDA kernel 时间。
保留了四并发首个正式 wave 的慢样本，未按性能筛选样本。
此前 MTP3 测速只作为历史记录，不是本轮基线。

## 分阶段诊断

在 FULL decode graph 中插入 external CUDA Events，记录 PP 两段，另记录
词表投影。每个负载生成 40 tokens，分析去掉首尾各五次回放后的数据。
事件会增加开销；各段 GPU 时间和嵌套区间不直接相加成端到端延迟。
首轮编译图未覆盖全部密集投影，因此 `model.total` 的未归属时间不能
解释为调度或通信时间。

1K、单请求的第一轮诊断如下，单位 ms，均为每段全部层的累计中位数：

| 阶段 | PP0，26 层 | PP1，22 层 |
| --- | ---: | ---: |
| routed MoE，含准备和收尾 | 2.987 | 2.597 |
| shared experts，含门控 | 1.402 | 1.229 |
| Marlin W13 GEMM | 0.646 | 0.571 |
| Marlin W2 GEMM | 0.438 | 0.388 |
| Marlin expert 对齐 | 0.284 | 0.251 |
| GDN fused core/norm | 0.395 | 0.329 |
| QSA indexer | 0.337 | 0.343 |
| LM head，图外独立计时 | — | 0.933 |

CPU PLE embedding gather 在 PP0 约 0.011 ms，不是这轮首先优化的热点。
32K indexer 增至约 0.532/0.525 ms；四并发 routed MoE 增至约
4.512/3.587 ms。MoE、共享专家、密集投影和词表投影应优先研究。

补充诊断覆盖了编译图的 extern GEMM、opaque op 内的 functional linear
和 native projection。1K/M1 下，GDN QKVZ+BA 输入投影的最外层区间
为 1.781/1.443 ms，输出投影为 0.937/0.764 ms，确认它们是明显热点。
HC up 两支在第一轮密集投影诊断约为 0.394+0.364 / 0.332+0.311 ms。
当前 M4 GDN input/output 和 M1/M4 HC down 的 native SM80 路径实际可达；
geometry 记录包含输入 stride，未发现需要放宽 packed-row gate 的证据。

更密集的事件诊断有明显扰动。相同 stage 的重复嵌套观测只取最外层一次；
共享专家等 CUDA stream 可以重叠，逻辑 parent 树不是 GPU 时间分区。
原始记录中部分 `model.total` 独占差值为负，因此不使用该字段计算
剩余热点或阶段占比；表中数据也不相加成无事件插桩的 decode 延迟。

## 源头提交和候选

- `e4294144a8ca4e57da712934dca33f89bef8fe78`，非 speculative GDN 回到
  standard path。本地候选限制为 Qwen4Exp、INC、SM80、TP1、BF16、
  hidden 2560、head dimensions 128、MTP 关闭，库默认关闭。
  新路径必须重新编译；只改 Python 属性而沿用已有 AOT graph 不能构成 A/B。
- `6cbb3c154ef1449d2b3c9131a237f36faa695734`，移除 capture 时 metadata
  CPU 同步，主要影响 capture，不能据此宣称 steady decode 加速。
- `dff1bde84dd6e34a49c150116d0f212507280910`，FP8 KV 分支；当前 BF16 KV
  不符合条件。
- `55b80221ed571adc1d6bfcc3edf4b48fa23e9884`，HC up strided input；
  源头的设备 gate 未覆盖 SM80，不直接移植。
- `48d8880d098c567c64150203110b6a6a4221d44f`，PLE side-stream 生命周期；
  当前同步 UVA gather 路径不同。

本地另一候选是 M1/top10/E512/block8 的 expert 对齐，避免通用路径
扫描 512 experts。原型按专家排序，保留 token index，处理重复、越界
和全无效输入。55 组独立参考和 graph 回放校验精确一致。
物理 GPU1 的 bounded microbenchmark 中，回放中位数为 8.47 → 4.73 μs；
该原型结果只说明局部潜力；完整模型的同实例 A/B 结果见下一节。

## 配对实验和推广结果

GDN 与 alignment 各测三种负载。每种负载先分别 warmup 两臂，再做
四组交替 AB/BA；每个请求输出 256 tokens，全部正式样本保留。
下表的收益为逐组 decode 窗口比值的中位数；并发采用共同 decode 窗口。

| 候选 | 1K/M1 | 32K/M1 | 1K/C4 | 决定 |
| --- | ---: | ---: | ---: | --- |
| 上游 GDN standard | -0.75% | -0.65% | +1.17% | 默认关闭，不能作为通用提速 |
| M1 expert 对齐 | +0.88% | +0.99% | +0.44% | 推广单请求收益；并发收益不成立 |

Alignment 的八组单请求配对全部为正：1K 各组 +0.95/+1.05/+0.81/+0.70%，
32K 各组 +1.13/+0.58/+1.31/+0.85%。两臂单请求中位数为
80.17 → 80.88 和 78.49 → 79.19 tok/s。
C4 四组为 +0.60/+1.34/-2.50/+0.27%，已有 M4 路径不变，不将波动解释为
稳定并发加速。两个 A/B 实验共 96 个正式请求均通过代码检索、正常输出
和固定长度检查。连同基线，126 个正式请求通过，MTP 和 prefix hits
增量均为零。

基线自己重复生成时也没有完整 token-ID 一致性，因此未以生成 token
完全一致作为候选正确性证据。这些是有固定语义检查的 serving 测量，
不是 GSM8K 等模型准确率评估。Alignment 的整数元数据使用独立参考
精确校验；真实加载的 INT4 权重另在 PP0/PP1 各检查八组 disjoint
专家路由，MoE 输出全部逐位一致。

Marlin 参数实验对两个 PP rank、M1/M4、W13/W2，分别试了默认配置及
六个 `thread_k/thread_n/blocks_per_sm` 计划。每个图轮换八组不重叠专家，
M1/M4 的 routed 权重工作集约为 197/786 MB，超过 L2；使用实际加载的
权重和 serving 同样的 native `moe_sum`。候选先比较现有 backend 输出，
再做九轮交替 CUDA Event 计时。`64x256` 计划不受支持，其余计划均未
胜过自动配置。两段默认完整 MoE 调用约为 M1 41.5–42.2 μs、M4
106.3–108.7 μs；这些是局部合成路由测量，不能替代正常 serving 数据。
没有推广 Marlin 参数改动。

生产代码加入了精确 shape/device gate 的 Triton M1 对齐，库默认
`VLLM_MOE_ALIGN_M1_E512_SM80=0`。AutoRound launcher 在 `MTP_TOKENS=0`
时默认设为 `1`；其他 shape、设备、expert map、dtype 或 padding 设置
保留此前路径。该整数重排可供相同契约的负载显式启用，但其他模型和
拓扑的性能需要另行验证。

生产 public API 的 bounded kernel benchmark，在同一 SM80 GPU1 上
采用 64 次调用/graph、两秒预热和 11 轮交替计时，中位数
5.544 → 2.670 μs，局部耗时减少约 52%。它和原型的单调用 graph 回放
边界不同，不能把两张 microbenchmark 表混作同一测量。
48 层每步省约 `48 × (5.544 - 2.670) μs = 0.138 ms`，
相对约 12.5 ms 的单请求 decode step，量级约 1.1%；这只是局部耗时
外推，实际收益以 +0.88%/+0.99% 的 serving 配对测量为准。

复现生产内核测量：

```bash
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python benchmarks/kernels/benchmark_moe_align_block_size.py \
  --decode-m1-e512 --output-json /tmp/qwen38-align.json
```

回退：设置 `VLLM_MOE_ALIGN_M1_E512_SM80=0` 后重启 Qwen。GDN standard
继续默认 `0`，Marlin 自动参数保持原值。

## 启动和证据边界

第一次 `.95` 启动成功，模型权重占用 PP0/PP1 为 39.01/32.66 GiB。
第二次相同设置在实际图捕获完成后、首次 NCCL P2P communicator
初始化时 OOM，尚未执行候选实验。失败日志完整保留，退出后 GPU
2/3 回到 6 MiB，没有遗留计算进程。
第三次诊断启动设为 `.90`，增加约 3.2 GiB 余量；后续 A/B 两臂
使用同一实例、同一 KV 配置，不能直接与第一次启动跨实例作收益比较。
262144 是启动长度配置，完整 256K 请求尚未验证。

Launcher 默认显存利用率改为 `.90`。本轮第三次诊断启动的 KV capacity
估计为 1,097,450 tokens、4.19 个最大长度请求；这是启动估计，不是实测
四并发 256K 容量。最终服务使用普通 API，关闭 development RPC，
移除 worker extension 和诊断 PYTHONPATH。

## 原始记录

本轮目录：`artifacts/qwen38-autoround-decode-20261006/`，由 Git ignore
排除，包含 `preflight.json`、`launch*.json`、`server*.log`、
`baseline.jsonl`、`profile.jsonl`、`analysis.json`、逐秒 GPU 采样、
exact-token prompts、诊断 worker、候选代码和测试日志。
`align-micro.json` / `align_micro.py` 保留局部对齐实验；
首次原型 scan 编译错误也保留，修正 combine function 后通过。
`ab-gdn*.json*`、`ab-align*.json*` 保存配对实验；`profile-dense-v2.jsonl`
保存密集投影和 geometry；`tune_marlin.json` 保存真实权重和参数实验。
`align-public-micro-final.json` 保存生产 API 的预热后 kernel 计时。
`launch-final.json` 保存最终普通服务的完整启动命令和运行时代码哈希。

## 验证

CPU 配置/路径测试 25 passed；GPU M1/M4 对齐测试 3 passed，覆盖 flag
两态、重复/无效专家、padding 和 graph 回放时更改输入。
本轮修改文件的 pre-commit 检查通过；最初的格式和行宽修正有独立日志。
最终普通服务 `qwen38-autoround-manual.service` 在 `0.0.0.0:26661` 健康运行，
MTP 关闭，alignment 开启，GDN standard 关闭。`/health` 和模型身份检查
通过；development RPC 返回 404，启动参数没有 worker extension。
默认思考、自动工具调用和四并发算术六项检查全部通过。
1K 和 128K 精确 token 输入，各输出 256 tokens，代码检索、正常文本、
固定长度和无 MTP/prefix-hit 增量检查通过。这两次上线探针不是重复测速，
不能用它们与旧实例比较性能。

现场核对了 unit/cgroup、API/Core/两个 GPU workers 和物理 GPU 归属。
GPU2/3 总占用为 58449/49551 MiB，GPU1/4/5/6 为 6 MiB；运行时代码哈希
与 `launch-final.json` 一致。DeepSeek unit 保持 inactive、MainPID=0。
最终证据为 `ready-final.json`、`smoke-summary.json`、`smoke-results.json`、
`final-serving-probes.json`、`runtime-final.json`、`metrics-final.txt` 和
`server-final.log`。本轮未提交、推送或发布 PR。
