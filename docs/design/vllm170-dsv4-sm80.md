# vllm170：DeepSeek V4 Flash 在 170HX/SM80 上的推理路线

## 目标与边界

`vllm170` 从官方 `v0.27.0` 标签（`4bdc8a788d`）派生，目标部署为 5 张
170HX 中 3 张运行 DeepSeek V4 Flash，另外 2 张保留给其他模型。由于卡间互联
带宽有限，DSV4F 固定采用 `PP=3, TP=1`，不把 MoE expert 或 attention head 横跨
显卡切分。

本分支遵守以下原则：

1. 保留 vLLM 的模型注册、attention backend、MoE oracle 和 modular MoE 边界，
   不使用启动时 monkey patch；新增模型或设备后端按同样边界扩展。
2. 官方 `v0.27.0` 中已有的 DSV4、mHC、输入 GEMM 多流、eager scratch、辅助
   hidden state、MXFP4 MoE 和 sequence-parallel 优化继续作为主体。
3. PP+DSpark 和 SM80 sparse-MLA 是本部署的正确性路径；MXFP4→INT8 是近似
   实验后端，必须显式选择，验证前不进入 `auto` 优先级。
4. 服务器恢复前只宣称静态分析、CPU 单元测试和构建接口检查，不宣称 170HX
   性能、CUDA kernel 正确性或长上下文稳定性。

## 上游与参考项目结论

### 官方 vLLM

`v0.27.0` 已包含完整的 DSV4 NVIDIA 模型与 DSpark 框架，且 target model 已实现
`SupportsPP`。缺口不是重新实现 DSV4，而是：

- DSpark 原先禁止 PP，draft 并未按“只在末级完整运行”处理；
- SM80 不能使用 DeepGEMM/FlashMLA 所需的新架构 kernel；
- Triton 在 SM89 以下拒绝原生 E4M3 转换；
- 官方 MXFP4 自动后端没有针对 SM80 INT8 Tensor Core 的实验路径。

本分支没有另写一套 DSV4 主循环，而是直接保留 `v0.27.0` 已合入的官方性能主干：

| 已包含的官方提交 | 保留并复用的设计思想 |
| --- | --- |
| `df71917cf1` | eager break 的临时量统一进入 scratch/workspace 复用，避免循环分配 |
| `904fae8be1` | PP intermediate tensor 按真实 mHC 形状直接传递，消除 448 MiB 重复分配/拷贝 |
| `f70caef48b` | request 到 token 的映射由 metadata builder 缓存，不在各 kernel 前重复构造 |
| `b0cb1da1bd`、`37e370fe93` | 无有效工作时跳过 router/top-k/空 c128 launch，而不是启动空 kernel |
| `b2f9e4caa4`、`e18fe932ca` | top-k 宽度与 prefill chunk 根据有效长度规划，避免按最大上下文做无效工作 |
| `837eae6458`、`442c421e79` | 去掉 full-kernel、repeat 与 copy，保持 mHC/MLA 数据在消费布局中 |
| `38a466e7b6` | sequence-parallel 边界已进入正式模型接口；本部署 TP1 不启用，但不破坏它 |

这些提交形成了新增代码的约束：SM80 attention 复用现有 metadata/workspace，INT8 MoE
接入 modular experts/oracle，PP 只补 token 状态，不回退到第三方旧版 model runner。

基于 `v0.27.0` 之后的官方提交，当前处理如下：

| 提交 | 思路 | vllm170 处理 |
| --- | --- | --- |
| `d40c3e3c00` | DSpark warmup 在 SWA-only 时不应访问 sparse index buffer | 已移植正确性修复 |
| `79c865b838` | 缩小 DSV4 eager CUDA graph 区域 | 暂不移植，待实机比较 graph 边界与分配 |
| `5789897aa4` | DSpark Markov projection 只算 top-k | 暂不移植，先建立 PP3 正确性基线 |
| `1044d19998` | ROCm MXFP4 TP8 shard 分配 | 与 CUDA PP3/TP1 无关 |

`a311916a29` 将 autoregressive drafter 的多步 decode 合进单个 CUDA graph；DSpark
继承的是 DFlash 并行 drafter，不走该 autoregressive speculator，因此不应为了“看起来
更新”而搬运。可以借鉴其原则：只有 attention backend 能在图内原地推进 metadata 时，
才融合多步；当前先验证 DSpark 自己的 graph 边界。

后续同步官方主线时，先按功能目录 cherry-pick 或重写小补丁，不合并第三方大提交。

### `dsv4-flash-a100`

本地 `haosdent/vllm:dsv4-flash-a100` 是第三方分支，不是官方 vLLM 提交或官方
分支。静态参考点是该分支提交 `12810046c799cbe874967e19b1c0fa134ab7b209`；它
基于早于 `v0.27.0` 的父提交，以一个约 111 文件的大提交混合 attention、kernel 和
通用性能改动。可借鉴的核心是 SM80 FP8 字节编解码、Triton MQA logits 和纯 Triton
sparse-MLA；不整体搬运。

### SGLang MXFP4→INT8

`deepseek-v4-a100-sglang-v0516` 的静态参考点是
`dd8cdb87ace3b053fcf4f20d898e048a4b33dc8a`。它的方法可以借鉴，但不是无损格式
转换：

- checkpoint 的 E2M1 code 与每 32 元素 UE8M0 scale，被重排为 E2M1 code、
  每 block 2-bit shift 和每输出通道 FP32 scale；
- activation 采用每 token、对称 `[-127, 127]` INT8，RNE 舍入；
- CUTLASS SM80 主循环在读取权重 fragment 时展开 E2M1 和 shift，执行 INT8 MMA，
  INT32 accumulator 在 epilogue 乘 activation/weight scale；
- W13 输出后执行 SwiGLU，W2 先产生 routed slot，再融合 top-k 权重归并。

该方案利用 INT8 Tensor Core，但 scale 折叠会产生近似误差。SGLang 现有结果主要是
抽样矩阵与 synthetic routing microbenchmark，不能替代 DSV4F PP3 端到端比较。因此
vllm170 将其实现为 `mxfp4_int8_sm80`，而 Marlin 仍是未指定 backend 时的安全候选。

## 模块化实现

### PP3 + DSpark

PP relay 的静态参考点是 `deepseek-v4-cmp170hx` 提交
`2d10b1672d90b6c6b4fbbf85728e09236e23d72e` 中拆分保存的补丁；本分支按
`v0.27.0` 当前接口重新实现，并增加配置期 partition 验证，而不是直接套用补丁。

选择“target PP3，draft PP1 只驻留最后一级”的方案：

1. target 三个 stage 正常传递 hidden states；最后一级拥有 target norm/lm_head。
2. DSpark draft parallel config 强制 `PP=1`，避免 draft 再被错误切成三段。
3. draft 依赖的 target auxiliary layer 必须全部位于最后一个 target partition；配置
   阶段验证 `dspark_target_layer_ids`，否则失败并提示调整 partition。
4. 最后一级完成 sampling 与 DSpark propose 后，通过 PP 独立 broadcast communicator
   依次广播固定宽度 sampled tokens、sample/reject 计数和 draft tokens。
5. 其他 stage 延迟 `pp_size` 个 step 消费，并用 request slot generation 过滤已经结束
   或重用的请求。

固定宽度 sampled-token padding 很关键：prefill 可能只产生 1 个 token，而接收端为
`num_speculative_tokens + 1`；NCCL 两端元素数量不同会挂死。

备选方案没有采用：

- 每个 PP stage 放一份 draft：显存浪费，并且中间 stage 没有完整 target 输出；
- 把 auxiliary hidden state 跨 stage 广播到独立 draft rank：通信更大、调度更复杂；
- 将 draft 本身 PP3：小 draft 会被通信延迟支配，且破坏 DSpark 当前完整模型假设。

建议 43 层模型先使用：

```bash
export VLLM_PP_LAYER_PARTITION=15,15,13
```

如果 checkpoint 的 `dspark_target_layer_ids` 不全在 `[30, 43)`，启动会失败；此时先
根据实际 layer IDs 重排三个 partition，不能跳过验证。

### SM80 sparse-MLA

SM8x 由 `TRITON_MLA_SPARSE_DSV4` backend 显式承接，并复用官方 DSV4 ROCm 路径中
不依赖 AITER 的纯 Triton sparse prefill/decode、ragged indices 和 inverse-RoPE
实现。设备选择与 kernel 实现分开，未来可把通用部分改名到 `common`，无需改模型。

关键算法选择：

- FP8 写入：SM80 上使用位级 E4M3FN、RNE、satfinite encoder；SM89+ 与 ROCm
  继续原生转换。
- FP8 读取：decode kernel 使用 256 项 BF16 LUT，无法传 LUT 的小 kernel 使用 ALU
  解码；避免 SM80 Triton 生成非法 FP8 convert。
- indexer logits：DeepGEMM 可用时保持官方路径，否则用 Triton MQA/paged-MQA；
  SM80 只支持 FP8 indexer cache，MXFP4 indexer cache直接失败。
- sparse prefill：沿用 ragged gather，避免按 dense top-k 最大宽度浪费计算。
- sparse decode：按序列长度做 split-K，再稳定归并 softmax 状态；当前启发式来自
  AMD/第三方静态结果，170HX 上必须重测。

### MXFP4→INT8 modular MoE

实现分为四层：

| 层 | 位置 | 职责 |
| --- | --- | --- |
| 格式转换 | `quantization/utils/mxfp4_int8_sm80.py` | Torch oracle、Triton GPU repack、误差统计、per-token INT8 RNE |
| 算子选择 | `fused_moe/oracle/mxfp4.py` | 显式 backend 映射、失败关闭、格式转换 |
| experts | `experts/mxfp4_int8_sm80.py` | routing 对齐、W13、SwiGLU、W2、加权 reduce |
| SM80 kernel | `csrc/libtorch_stable/moe/mxfp4_int8_sm80/` | CUTLASS INT8 MMA、scale epilogue、slot reduce |

首版有意只实例化实际 DSV4F TP1 形状：`H=4096, I=2048, E=256, topk=6`，且
只支持 BF16/SwiGLU、无 bias、无 LoRA、无 TP/EP/DP/PCP/SP。PP 不改变单层 MoE
布局，因此 PP3 可用。其他模型必须新增 shape policy/实例化，不能默默落入此 kernel。
modular runner 会让最终输出与 `workspace13` 复用底层存储；本后端因此把 routed W2
slots 放在 `workspace2`，只让已经完成 A2 量化的 activation 暂存与最终输出复用，避免
reduce 写回时覆盖后续 token 尚未读取的 routed slot。

`headroom_bits=3` 是静态起点：它让最大 exponent block 保留 3-bit 左移空间，减少
INT8 动态范围浪费；但 exponent span 大的 channel 会发生 code 重新量化。加载时不做
全权重统计，以免增加启动峰值显存，实机验证脚本必须逐 shard 输出 overflow、exact
rate、RMSE 和最大相对误差。

## 初始启动配置

先以 Marlin 建立正确性基线，再切换实验 INT8 backend：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
VLLM_PP_LAYER_PARTITION=15,15,13 \
vllm serve /models/deepseek-v4-flash \
  --pipeline-parallel-size 3 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 8 \
  --tokenizer-mode deepseek_v4 \
  --attention-backend TRITON_MLA_SPARSE_DSV4 \
  --moe-backend marlin \
  --speculative-config '{
    "method":"dspark",
    "num_speculative_tokens":5
  }'
```

DeepSeek-V4-Flash-0731 的 DSpark 权重就在 target checkpoint 内，不应再填写独立
`model` 路径。`num_speculative_tokens=5` 与该 checkpoint 的 `dspark_block_size` 一致；
更改 checkpoint 后必须重新核对，而不能机械沿用。

只有 Marlin 基线通过后，才将 `--moe-backend marlin` 改为：

```text
--moe-backend mxfp4_int8_sm80
```

如果实际 checkpoint、draft config 或 layer 数不同，以上路径只是模板；启动前先核对
`num_hidden_layers`、`dspark_target_layer_ids`、MoE shape 与量化 scale dtype。

## 服务器恢复后的分阶段验证

### 0. 硬件与构建门槛

- 记录 5 张卡的 device name、compute capability、显存、PCIe 拓扑、P2P 能力和单向/
  双向带宽；确认 170HX 对外报告 SM8x。
- 用 `TORCH_CUDA_ARCH_LIST=8.0` 构建，确认 `_moe_C` 包含两个
  `dsv4_mxfp4_int8_sm80_*` op；检查 CUTLASS dynamic shared-memory 设置。
- 先单卡运行 FP8 encoder/LUT、MQA logits、sparse attention 和 MXFP4 INT8 GEMM 的
  reference parity，禁止直接进入 1M context。

### 1. PP3 正确性

- 不开 DSpark，比较单卡短序列与 PP3 的 greedy token、logits 和 KV cache 更新。
- 分别测试默认 partition 与 `15,15,13`，记录 stage 显存、prefill/decode 时间与 idle。
- 覆盖 chunked prefill、请求完成/取消/slot 重用、不同 batch 中混合 prefill/decode。
- 检查独立 broadcast communicator 的 collective 顺序；使用 1、5、7 draft token
  验证没有 NCCL size mismatch 或 step 错位。

### 2. DSpark

- `enforce_eager` 先做 token parity，再打开 CUDA graph。
- 记录 acceptance rate、acceptance length、draft latency、target verify latency和总 TPS；
  对比无 speculative、5-token、7-token。
- 验证所有 auxiliary hidden IDs 位于最后 stage，并检查 draft embedding/lm_head 权重
  的加载、共享与显存占用。
- A/B 官方后续的 top-k Markov projection（`5789897aa4`），只在 parity 通过后采用。

### 3. attention/indexer

- 序列长度：1K、8K、32K、128K，内存允许后再到 1M；batch 覆盖 1、8、32。
- 对比 Torch/高精度 reference 的 top-k overlap、attention 输出最大误差和 token parity。
- A/B LUT 与 ALU decode、paged-Q 预解码开关、prefill KV grouping、decode split 数。
- 调整 `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` 的 row chunk 为 128/256/512 MiB，观察
  峰值显存、kernel 次数和长上下文 latency。
- A/B 主线缩小 eager graph 区域（`79c865b838`），记录 graph 显存和 decode 抖动。

### 4. MoE backend

- 先逐层/逐 shard 扫描全部 MoE layer 的 exponent span 与重排误差，不能只抽样。
- kernel microbench 使用真实 routing histogram；比较 Marlin、MXFP4 direct/参考实现、
  MXFP4→INT8 的 W13、activation、W2、reduce 分段延迟。
- tile 至少比较 `16x64`、`16x128`、`64x128`、`128x128`，按真实 batch 与 expert
  occupancy 重建表，不沿用 A100 结果作为结论。
- A/B `headroom_bits=2/3/4`；同时记录 overflow、RMSE、模型 ppl/任务分数和生成差异。
- 检查每 token quant 临时 tensor 是否被 CUDA graph 稳定复用；若 allocator 地址不稳，
  第二版把 INT8 activation/scale 纳入 workspace manager，而不是扩大 graph 边界。

### 5. 端到端准入条件

建议同时满足后才把实验 INT8 backend 用于长期服务：

- PP3 连续压测无 collective hang、请求串线或 cache 越界；
- greedy parity 与容许误差内的 sampling/eval 指标通过；
- 128K 稳定后才扩大到 1M；
- MXFP4→INT8 相比 Marlin 有端到端收益，而不是只有 synthetic microbench 收益；
- 记录可回退启动命令，Marlin 始终可用。

## 后续扩展约定

新增其他模型时保持以下目录关系：模型只描述 shape、routing 与 hidden-state contract；
设备差异放到模型的 platform adapter；attention kernel 通过 backend registry；MoE 通过
oracle + experts + quant utils + 独立 C++ 目录。任何近似量化后端先显式选择，完成实机
矩阵后才讨论加入 `auto`。这样 `vllm170` 可以周期性 rebase/cherry-pick 官方 vLLM，
而不是长期维护一个无法拆分的 111 文件补丁。
