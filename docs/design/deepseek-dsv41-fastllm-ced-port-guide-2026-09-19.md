# 给 DeepSeek：移植 FastLLM 的 DSV4.1 快速 prefill

## 任务与交付范围

将 `/mnt/data/apps/fastllm` 已有的 decoder SWA bounded replay 移植到本项目
`/mnt/data/apps/vllma100/vllm170`，服务 DeepSeek-V4.1-Flash 的 SM80、TP1、PP5、
`8,8,8,8,8` 分层，以及分布式 DSpark。PP5 启动器默认开启近似 prefill 优化，
可用 `PORT_FAST_PREFILL=0` 显式关闭；直接构造 engine 配置时仍需显式设置开关。

主代理负责架构、实质改动、集成和最终判断；遵守目标仓库 AGENTS.md。
本文件是实现任务书；生成此指引的会话没有执行移植，也没有授权立即停服或重启现有服务。
先完成源码、CPU 验证和可审阅结果，再根据执行会话中的授权安排独占硬件验证。

交付代码、相应测试、启动器开关与简短验证报告。不要把源码存在、开关开启或成功返回文本
当作裁剪生效的证据；必须证明每层实际计算的有效 token 数。

## 固定证据基线

实现前记录实时 HEAD 和 dirty files；下列是此次指引核查的快照，不要求回退工作区：

| 对象 | 核查版本 / 路径 | 用途 |
| --- | --- | --- |
| FastLLM | `cb465fe483aaee2e6d5214ba23f9e60f79ec8cc8` | 移植行为的直接依据 |
| FastLLM 首次快速 prefill 提交 | `f2727f7a` | 理解增量；还需包含后续暂存复用等修正 |
| 本项目 | `2c5e578bd1` | 核查时目标 HEAD，已有其他工作会继续推进 |
| 模型示例 | `/media/z/data6/models/DeepSeek-V4.1-Flash/inference/` | 模型算子、权重、位置和默认完整前向语义 |
| 源头 vLLM | `/mnt/data/apps/vllm` 的 `v0.30.0rc2`，`fa6ff060667f2cd96f142af021ba9e35165beb78` | 上游完整计算基线 |
| 源头 main 快照 | `c3b48446349569512749db7f6e2164aa8a33437d` | 区分 prefix-hit SWA replay 与本次优化 |

源头目录工作区可能仍停在旧版本，用 `git show <sha>:<path>` 读取正确对象，不要切换或
覆盖它的工作区。源头包名是 `vllm/models/deepseek_v41`，本项目是 `deepseek_v4_1`。
目标当时有未跟踪的 `benchmarks/deepseek_v4_1/budget_gate.py`，不属于本任务。

模型示例与上述源头 vLLM 的默认 prefill 都跑完整 40 层；不能根据模型卡的 8B prefill
描述，假定公开代码已经裁剪。decode 同样需要完整 40 层，本次不改变该行为。

## 必须移植的精确语义

FastLLM 并非“前 20 层计算、后 20 层跳过”，也不是本项目已有的 CPU 混推 CED：

1. `tail_layer = max(kv_source_layer_ids) + 1`。当前权重 source 为 `[2,8,14,20]`，
   因此 `tail_layer=21`。以下全部使用零基层号。
2. 每个文本 prefill chunk 的层 **0–20** 完整处理所有 token，包括第 20 层完整的
   attention、FFN 和全量 global KV / indexer K 生产。
3. 进入层 **21** 前，按请求分别保留本 chunk 的最后 `min(chunk_tokens, sliding_window)`
   行。当前 `sliding_window=128`。层 **21–39** 只计算保留行。
4. 这是**每个 chunk 都执行**的尾部计算；不是只在最后一个 chunk 执行，也不跨 chunk
   收集 encoder suffix 后一次性 replay。
5. global KV / indexer K 包含完整历史；只裁剪查询侧 hidden、mHC 状态和 query 索引行。
6. 被裁剪请求的局部 SWA 不读截断点之前的旧窗口。global attention 的上下文长度、
   RoPE 和最终 KV 写槽仍使用真实绝对位置，不能整体把 positions 改为从零开始。
7. 请求逻辑进度推进完整 chunk 长度；最后一行输出必须对应原请求最后一行。

单个长度为 N>128 的 chunk，其每层有效行数应为：

| PP rank | 层 | 有效行数 |
| --- | --- | --- |
| PP0 | 0–7 | N |
| PP1 | 8–15 | N |
| PP2 | 16–20 | N |
| PP2 | 21–23 | 128 |
| PP3 | 24–31 | 128 |
| PP4 | 32–39 | 128 |

短片段与 decode 行在混合批次中保留全部。验证阶段必须区分有效行数、padding 行数、
通信行数，不能用外层张量 shape 单独证明计算减少。

## FastLLM 源码阅读清单

行号基于上述快照，优先按符号定位：

| 文件 / 位置 | 应读内容 |
| --- | --- |
| `tools/fastllm_pytools/util.py:849,1624` | `--fast_prefill` / `--fast-prefill`，默认关闭，CLI 覆盖内部环境变量 |
| `src/models/deepseekv41.cpp:1244` | `decoderSwaTailLayer` 与 source 边界校验 |
| `src/models/deepseekv41.cpp:3208` | `ForwardSegments` 的启用条件与原始/裁剪 segment 分离 |
| `src/models/deepseekv41.cpp:3776` | 裁剪 hidden、preMix、top-k、candidate、DSpark main hidden |
| `src/models/deepseekv41.cpp:3966` | 从 KV source 获取完整共享缓存 |
| `src/models/deepseekv41.cpp:4139` | 局部 SWA 起点重置与绝对 KV 写入分离 |
| `src/models/deepseekv41.cpp:4689,4735` | 按请求取末行、按完整输入推进逻辑长度 |
| `src/models/deepseekv41_dspark.cpp:1364` | `DsparkAdvance` 使用 `mainHiddenStartPos` |
| `test/ops/deepseekV41BoundedReplayRegression.cpp` | 分块/混合/恢复/量化 KV/DSpark 测试，以及微型模型的覆盖局限 |
| `docs/deepseek_v41.md:280` | 明确的近似性、默认值和适用边界 |

注意：FastLLM 的微型 fixture 仅一个裁剪后的层。最后一行与完整前向相近或相等，
不能证明真实 19 层裁剪后的等价性。它没有消除多层 SWA 截断误差。

## 目标项目设计要求

### 开关、回退和近似性

增加独立 `additional_config.dsv41_fast_prefill` 布尔开关；直接构造
`VllmConfig` 时默认 false，`run_port_pp5.sh` 则通过默认开启的
`PORT_FAST_PREFILL=1` 注入该开关，并保留 `PORT_FAST_PREFILL=0` 的显式回退。
合并已有 additional-config，不能覆盖其他选项。
新开关与旧 `additional_config.ced_prefill` 同时开启时给出清楚的冲突错误。

不开启时维持现有计算路径。开启后打印一次明确日志：近似模式、边界层、窗口宽度。
不要沿用 CPU 混推 CED 要求“第 19 层及全部 decoder 位于最后一个 rank”的限制。

第一版以 CUDA SM80、TP1、PP5、V2 runner 为验证范围。未支持的部署配置应在启动前
明确报错；请求级不满足条件时完整计算，并记录可诊断的回退原因。

请求/批次回退要求：

- 无长 prefill 片段：完整计算。
- 图像嵌入、图像 mask 或不能确认纯文本的输入：完整计算。
- DSpark verification、要求全部候选位置的计算：完整计算；允许第一版整批回退。
- 请求 prompt logprobs（含 0）：该请求完整计算；允许第一版整批回退。
- dummy/profile/warmup/capture：先完整计算，不能凭空构造可用于真实请求的 suffix。

这是近似优化：局部历史截断会改变后续层 hidden、KV、logits，可能改变生成 token。
报告质量变化，不得承诺与关闭时 token 一致，也不得将近似误差伪装成仅浮点舍入。

### 每请求行映射与 attention metadata

先定义可 CPU 测试的元数据契约：原始 query_start_loc、每请求绝对起点、原始行数、
保留行索引、裁剪后 query_start_loc、scatter 目标行、局部 SWA 下界。
映射由真实请求边界构造，不能对整个合并 batch 简单取最后 128 行。

同步处理 positions、input_ids、hidden、residual、post_mix、res_mix、pre_mix、
top-k、candidate，以及相关辅助状态；只处理实际存活的张量，避免无意义的额外复制。

重建目标后段的 metadata 和 slot mapping；global seq_lens 保留完整上下文。
裁剪请求的 SWA gather 只能读取保留窗口，混合批次中未裁剪请求保持原 SWA 语义。
必须沿 Ampere 路径追到 `amd/rocm.py` 及实际 gather/attention kernel，证明只改
某个长度字段确实限制了读范围，而不是靠字段名字推断。

主要入口：

- `vllm/models/deepseek_v4_1/nvidia/model_state.py`
- `vllm/models/deepseek_v4_1/nvidia/model.py` 的 `forward`
- `vllm/models/deepseek_v4_1/nvidia/vl_model.py` 的参数传递
- `vllm/models/deepseek_v4_1/ampere/{ampere_sparse,prefill_metadata}.py`
- `vllm/models/deepseek_v4_1/amd/rocm.py`
- `vllm/v1/worker/gpu/model_runner.py` 的 graph dispatch 和结果消费

旧 `ced.py` 可以参考 metadata 工具，但不能直接打开、删除分层 assert 后宣称移植完成。
现有 `requires_eager_prefill` 的实际消费者也要重新核查，不能假定它已接通。
第一版裁剪 prefill 可以 eager；各 PP rank 的 dispatch 必须一致，decode 图保持原路径。

### PP5：完整 KV 与尾部查询分开处理

这是移植的核心。PP2 在层 20 产生全量 KV；PP3/PP4 仍需要写入这批全量 KV 的本地副本，
不能因为 hidden 变成 128 行就只传播 128 行 compressed latent 或 indexer K。

核查 `make_empty_intermediate_tensors`、`_import_pp_attention_state`、
`_append_pp_attention_state`、`IncrementalPPKV` 和 runner 的收发缓冲协议。当前
`_import_pp_attention_state` 使用完整 positions 和 source slot mapping 写入缓存；
top-k/candidate 却是 query 侧数据。两者不能共用裁剪后的行数。

可优先评估保留现有完整 PP 传输形状、rank 内 gather 尾部计算后 scatter 回原位置的方案，
减少对通信协议的改动；但必须保证未计算行绝不被下游 attention、sampler、DSpark 当作
有效 hidden 消费。每个下游 rank 先完整导入 global KV，再只计算选中查询行。
如果采用紧凑 PP 消息，必须显式传递完整 cache 更新与查询行映射，并验证收发双方协议。

不能改成把后 20 层塞进最后一张卡来绕过 PP5 问题，也不引入 CPU expert offload。

### 分布式 DSpark 与 prefix caching

阅读 `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` 的 `_propose_distributed`、
context KV 包、`_context_slot_mappings`、`_publish_ced_draft_context`，以及模型侧
`nvidia/dspark.py`。旧 CED context 依赖“末 rank 一次 suffix replay”，不能原样套用。

DSpark context 必须只提交真实算出的 aux hidden，携带对应绝对 positions/slots 和请求
边界。scatter 出来的占位行不能写入 draft KV。短末 chunk、跨 chunk 的窗口连续性、
verify/accept/reject 和多请求隔离都需覆盖，不能简单把 num_target_tokens 改小。
保持本项目现有 aux hidden 的提取语义，不顺便替换成 FastLLM 的 target-layer 编号约定。

FastLLM 支持缓存恢复，vLLM 的 paged prefix sharing 与本优化保持独立配置。Prefix hit
只缩短当前请求需要执行的 query 区间，fast prefill 仍按实际 chunk 的行映射建立尾部
计划；两者可以同时开启。验证时仍需分别记录 prefix hit、尾部有效行数和输出质量，不能
把缓存命中本身当作 fast prefill 生效证据。

## 验证顺序与判据

先写出模块用途、输入输出契约、预防的故障和最便宜的测试层级，再扩展附近测试。
Python 使用目标 `.venv/bin/python` 和 uv，禁止系统 Python / bare pip。

1. **CPU 单元/契约测试**：窗口边界 127/128/129；多个 chunk，尤其末 chunk 1/2；
   混合 prefill/decode；请求顺序变换；prompt logprobs、图像、verify 回退；
   原始/裁剪行映射、绝对位置、请求进度、生命周期清理。
2. **多层计算语义测试**：至少两个、最好多个裁剪后的层。与独立实现的 FastLLM 式
   近似计算 oracle 比较；分别量化与全量 oracle 的差异，不要求其为零。
3. **PP/DSpark 合约测试**：后 rank 完整 cache 更新不丢行，selected query 正确；
   draft context 不摄入占位值，verify 不裁剪。扩展
   `tests/models/test_deepseek_v41_ced.py`、`tests/models/test_deepseek_v4_dspark.py`、
   `tests/v1/spec_decode/test_dspark_pp.py` 等合适的现有 suite，避免测试重写实现。
4. **硬件验证**：先确认目标进程/端口/GPU 占用，在可独占测试窗口进行。先 DSpark off，
   再 distributed DSpark on；单请求，再混合/并发；检查 finite logits、缓存位置和输出。
5. **真正生效证据**：逐层计数或 profiler 证明 N>128 时层 0–20 为 N 行，21–39 为
   128 行；多请求分别求和。记录实际 kernel 计算范围，排除仅通信/输出 shape 变小。
6. **性能与质量**：固定权重、量化、chunk、KV、采样和输入，用同一时间窗 A/B；长
   prompt 至少覆盖 4K、32K、64K，报告 TTFT、prefill 吞吐、峰值显存和 decode/DSpark
   接受情况。质量用固定任务集及可复核参考，不以一句“能回答”验收。报告 logits 差异、
   token 分歧与任务正确率；没有事先确定的质量容忍标准时，不宣称可默认上线。

测试命令只列出实际执行的命令和结果。硬件不可用时交付源码与 CPU 证据，清楚标注
未验证项；不要扩大成后端切换、tile 搜索或反复长时间加载模型的实验。

## 最终交付清单

- 独立且默认关闭的开关，启动器正确合并配置；清楚的启用/回退诊断。
- PP5 保持原分层；完整 global KV 与尾部查询的传播契约有测试。
- 单 token decode 和 DSpark verify 全层全候选；prefill 才裁剪。
- 多层近似误差报告；没有将微型单层测试当作真实模型等价证明。
- 明确列出与 FastLLM 的行为差异，尤其 prefix caching、图像、chunk 和 DSpark。
- 可复现的逐层行数证据、A/B 性能及模型质量结果；未运行项目单独列出。
- 保留并避开其他任务的修改；不提交、不推送、不修改模型权重，除非执行会话另有要求。
