# Qwen3.8 Flash Next AutoRound：MTP 接受率排查

本轮确认接受率高度依赖输出内容，尚未定位到导致低接受率的实现错误。
同一服务、同一 MTP3 配置下，英文说明约 46%，重复短语和顺序整数数组
各两次均为 100%。后两类证明当前路径能够连续接受三枚草稿，但不足以
证明所有输入、缓存状态或数值路径都正确。

完整权重比对、加载映射和 token/hidden/position/状态提交审计未发现明显
错配。既有同实例实验也不支持 FP8 草稿输出头或融合计算是低接受率的
主要原因。AutoRound 主模型数值分布与 BF16 MTP 的匹配程度仍是待验证
因素，不能将“量化导致低接受率”写成已确认根因。

## 运行边界和指标定义

目标为 `/media/z/data6/models/Qwen3.8-Flash-Next-W4A16-AutoRound`。
复用普通服务 `0.0.0.0:26661`，没有重启、注入 worker hook 或开启诊断 RPC。
物理 CMP170HX GPU 2/3，PCI 排序，PP2/TP1，分层 `26,22`，K=3；
CPU PLE、BF16 target head、FP8 draft head、融合多步草稿保持启用。
PP relay 修复启用；prepare-ahead、prefix metadata、metadata reuse、draft
top-k sharing 保持关闭。服务最大长度 262144、最大并发 4、batch tokens
4096、显存比例 0.90。DeepSeek 保持停止。

目标 HEAD 为 `e2e327dc53b93acb85c8b56b526d369eb5461421`，包含既有本地修改。
源头 `/mnt/data/apps/vllm` HEAD 为
`31ebe18bb63303be83137a59f668ca9812ac3579`，本地缓存 `origin/main` 为
`0e468adb43d60484b5a6546e0124b686fd76baf3`；本轮没有 fetch。

若验证轮数为 R、接受草稿总数为 A，则草稿接受率为 `A/(3R)`，平均每轮
输出长度指标为 `1+A/R`。第一、二、三位置计数分别代表接受前缀至少达到
一、二、三枚，分母都是 R；不是三个位置相互独立的命中率。
最后一轮可能接受超过 `max_tokens` 截止位置的草稿，指标会包含这些草稿，
客户端输出会被截断。

本轮请求 `temperature=0`，greedy 接受条件是草稿 ID 等于 target argmax；
在第一枚不匹配处停止接受，并提交 target 的 token。源码为
`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:540-595`。
验证流程的耗时不会直接改变这一相等判断。

## 同一服务的八次短诊断

四种请求各两次，输出上限 128，固定 seed=42、`ignore_eos=false`，模板
`enable_thinking=false`。英文任务完全相同，仅中性背景从精确 1024 token
扩为 32768 token；其他两类使用 1024 token。每次使用唯一 cache salt，
请求前检查无运行/等待请求，请求成功计数增量必须为 1，prefix hits 为 0，
draft tokens 为轮数的三倍，分位置接受计数之和必须等于总接受计数。

合并两次请求的整数计数后得到：

| 输出内容 | 输入长度 | 轮数 | 草稿接受率 | 平均长度指标 | 第一位置 | 第二位置 | 第三位置 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 英文解释流水线并行 | 1K | 108 | 45.68% | 2.370 | 75.00% | 42.59% | 19.44% |
| 同一英文问题 | 32K | 107 | 45.79% | 2.374 | 65.42% | 48.60% | 23.36% |
| 重复 `blue river` | 1K | 64 | 100.00% | 4.000 | 100.00% | 100.00% | 100.00% |
| 顺序整数 JSON 数组 | 1K | 64 | 100.00% | 4.000 | 100.00% | 100.00% | 100.00% |

英文 1K 的两次接受率为 46.30%/45.06%，32K 为 46.54%/45.06%。
在已接受第一枚的轮次中，第二枚的条件接受率为 56.79%/74.29%；已接受
两枚时，第三枚的条件接受率为 45.65%/48.08%。因此总率较低同时涉及
第一枚不匹配和后续递归草稿不匹配，不能仅归因于某一个位置。

所有请求均生成 128 token、以 length 结束，没有特殊输出 token。
重复短语和 JSON 的输出分别通过完整预期文本的前缀检查；JSON 在整数 45
处被长度上限截断。这是接受机制诊断，不能宣称完成了整项任务或一般模型
准确率评测。本轮没有用客户端时延衡量优化收益。

此前普通服务回归使用不同的 needle retrieval 加说明文本任务，每种负载
三个正式样本，合并计数为：

| 既有负载 | 草稿接受率 | 平均长度指标 | 第一位置 | 第二位置 | 第三位置 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1K/C1 | 49.46% | 2.484 | 72.58% | 47.74% | 28.06% |
| 32K/C1 | 35.76% | 2.073 | 61.99% | 31.27% | 14.02% |
| 1K/C4 | 50.01% | 2.500 | 72.17% | 49.72% | 28.15% |

原始数据是 `artifacts/qwen38-autoround-verification-20261007/restored.jsonl`。
新诊断中相同问题的 1K/32K 总率接近，说明既有 32K 的 35.76% 不能直接
归因于上下文长度。其逐请求输出也没有特殊 token，不支持强制 EOS 之后
继续生成无意义文本是低率的解释。

## 权重、配置和 tokenizer

对 AutoRound 与 NVFP4 checkpoint 做完整 CPU tensor 比对，包括全部 512
个 MTP 专家的 gate/up/down、全部非专家 MTP 权重、主 LM head 和 token
embedding。NVFP4 的 packed expert 按专家和 gate/up 分片映射到 AutoRound
独立专家序列化，比较形状、dtype 和数值。共比较 1567 个 tensor，每份
checkpoint 7,757,098,496 字节，零差异，耗时 18.91 秒。

其中 MTP 有 1565 个 tensor，全部 BF16，总计 5,214,301,696 字节。
主 head 和 embedding 各 1,271,398,400 字节，BF16 且逐值相同。
因此 AutoRound 文件中的 MTP 并非 INT4；这排除了“两份 checkpoint 的
MTP 权重本身不同”这一解释，不等价于运行时全部数值逐位一致。

加载源码审计覆盖以下边界：

| 边界 | 结论与源码 |
| --- | --- |
| checkpoint 路径转换 | `mtp.py:80-107` 将 `mtp.*` 映射到 draft `model.*`，单层运行索引为 0；48 是量化配置中的绝对层前缀 |
| draft 量化配置 | `mtp.py:110-145` 独立解析 draft 配置并重映射 ignored/excluded layers；checkpoint 的 MTP 规则为 bits16 |
| head/embedding | `tie_word_embeddings=false`；target/draft 分别实例化，从相同 checkpoint 数值加载；`mtp.py:432-446` 使用 AutoWeightsLoader |
| FP8 草稿 head | `mtp_fp8_lm_head.py:100-127` 对 SM80、TP1、BF16 head 在线转换为 FP8 Marlin；target head 保持 BF16 |
| target INT4 专家 | INC → AutoGPTQMoE → Marlin；`inc/schemes/inc_wna16_scheme.py:159-203`，`auto_gptq.py:800-814` |

AutoRound 是对称 INT4、group128、`auto_round:auto_gptq`。专家 gate/up
header 为 `qweight [320,640]`、`qzeros [20,80]`、`scales [20,640]`，down
为 `[80,2560]`、`[5,320]`、`[5,2560]`，匹配 TP1 的 fused expert 加载合同。
CUDA Marlin 对称路径 `use_zp=false`，运行时 `w1_zp/w2_zp=None`，使用
`uint4b8`。未发现把 GPTQ qzeros 当成 AWQ zero point 的源码错误。
专家分片在 `routed_experts.py:892-977,1089-1145`，未发现 gate/up 顺序或
专家索引错误。这仍是配置、header 和源码核查，未做全模型独立反量化参考。

一个未定量因素是 target scales：checkpoint 为 F16，模型参数为 BF16，
加载时会转换精度；qweight/qzeros 为 I32。主模型 INT4、这一 scales
转换及 BF16 MTP 的分布匹配都可能影响 top1，但本轮没有逐 token 的
target/draft margin 或单因素实验，不能确认贡献大小。

两份 `text_config` 仅 `ple_embedding_dtype` 不同：AutoRound 未指定，
NVFP4 为 `float8_e4m3fn`。PLE 属于 target，不属于 MTP。
tokenizer 文件的字节 hash 不同，但 248044 项词表 ID 完全一致，规范化
merges 完全一致，chat template 的 hash 完全一致。pre-tokenizer 的 Unicode
mark 正则及 trim 标记、decoder 标记存在差异；六个固定英文、中文、重复、
JSON、组合字符和空白样例编码一致，不能据此宣称所有 Unicode 输入等价。
当前 target/draft 直接共享同一 token ID 空间，没有跨 tokenizer 再编码步骤。

## 已有同实例数值开关实验

复核 `artifacts/qwen38-autoround-mtp-20261006/analysis.json` 的配对实验：

| 开关与负载 | 配对数 | 原接受率 | 改动后接受率 |
| --- | ---: | ---: | ---: |
| 草稿 head FP8 → BF16，1K/C1 | 4 | 48.53% | 49.35% |
| 草稿 head FP8 → BF16，32K/C1 | 4 | 35.89% | 35.92% |
| 草稿 head FP8 → BF16，1K/C4 | 4 | 50.19% | 50.51% |
| 草稿融合 → 非融合，1K/C1 | 8 | 49.27% | 49.04% |
| 草稿融合 → 非融合，128K/C1 | 4 | 40.25% | 42.20% |

这些率由各臂平均长度指标的中位数换算，不是合并计数。
BF16 head 没有将低接受率恢复到明显更高水平；1K 非融合也没有改善，
128K 波动伴随输出变化，已有报告未确认稳定收益。因此没有理由仅为了
接受率关闭当前 FP8 head 或融合路径。保留既有 K3 配置。

旧 NVFP4 测速记录使用不同 checkout、GPU 4/5、长度配置、prompt 和
128-token 输出；本轮检查时其原始 artifact 目录已不存在。旧数字属于历史
记录，不能与当前 AutoRound 做受控 A/B。本轮只比较了 checkpoint，
没有重新加载 NVFP4 服务。

## token、hidden、position 与拒绝后的状态

`model.py:654-669` 在 final HC mixer 时物化并保存 `[T,10240]` multi-stream
hidden；`gpu/model_runner.py:2255-2276` 将活动前缀传给 drafter。
`autoregressive/speculator.py:448-478` 使用位置 P 的 hidden 和 P+1 的 token
来预测 P+2，采样位置调整为 P+1，后续递归使用返回的 MTP hidden。
本轮没有发现 hidden 被错误换成 target 单流 `[T,2560]` 的证据。

重点检查了 rejected tail：draft prefill 保持 target 的完整 shape，但只复制
有效前缀的 token/position，未来尾部可能保留旧 buffer 内容。
这本身不构成已经确认的缓存错误。QSA cache 的逻辑写地址由 metadata
推导（`qsa_cache.py:253-257`），而非旧 RoPE position；K3、压缩比例 4
使用容量 8 的 ring（同文件 `815-825`），避免覆盖已提交的开放组成员。
compression 每行只读取以该行结束的组（`ops/qsa.py:631-659`），可见组
上界受当前 query position 限制（同文件 `166-172`）。单请求连续位置下，
未来被拒绝的位置在可被下一次 query 读取前会被重写。

GDN 在 `model_states/mamba_hybrid.py:338-370` 保存接受数量，conv/SSM
kernel 用 `num_accepted_tokens-1` 选择初始状态，分别见
`causal_conv1d.py:860-875`、`mamba_ssm.py:333-355`。源头 MRV2 也保留
完整 shape/有效前缀的设计。本轮没有把未来 speculative 写入误判为根因，
也没有直接修改复制范围或加入可能改变语义的 rejected mask。

上述源码推导和局部测试不能替代多请求重排、MROPE 等边界的端到端数值
验证；本轮短 probe 限于 C1 和文本 1D position。

## 验证、改动与后续判定边界

本轮唯一代码改动是补齐既有融合测试的 SimpleNamespace 替身方法绑定。
运行中的模型代码、配置和服务未改动。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m pytest \
  tests/v1/worker/test_gpu_autoregressive_speculator.py \
  tests/models/qwen4_exp/test_config.py \
  tests/models/qwen4_exp/test_mtp_fp8_lm_head.py -q

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m pytest \
  tests/models/qwen4_exp/test_qsa_reference.py::test_qsa_circular_buffer_survives_one_speculative_step \
  tests/models/qwen4_exp/test_qsa_reference.py::test_qsa_ring_capacity_covers_one_speculative_step \
  tests/models/qwen4_exp/test_qsa_reference.py::test_qsa_draft_metadata_update_is_graph_safe_and_padding_aware \
  tests/kernels/mamba/test_gdn_fused_mtp.py -q
```

第一组 52 项通过。第二组初跑 31 项通过、1 项失败；失败项
`test_fused_forward_uses_packed_entrypoint` 的替身没有绑定重构后的
`_forward_input_projection`，不是推理数值不匹配。补齐绑定后仅重跑失败项，
1 项通过。共 84 个不同用例通过；没有宣称完整模型精度或所有 GPU 路径
获得覆盖。

可确认的是文本内容与接受率存在明显关联，未找到全局 token 错位、不同
MTP checkpoint 参数、对称 zero-point 合同错误或当前 C1 的缓存污染路径。
要进一步区分“自然语言预测分歧”“target 量化数值变化”“只在特定状态下
触发的实现问题”，还需要同一轨迹中被拒绝的 draft ID、target top1/top2、
draft 在 target 中的排名和 logit gap，以及 teacher-forced 数值参考。
本轮计数器和 SSE 没有这些信息，不能凭计数推断实际拒绝 token 的性质。
既有 q1 audit 仅为 DSpark/DFlash 启用，没有在普通 MTP 服务启用。

后续受控实验应冻结完全相同的 token 输入和输出前缀，先收集少量拒绝
样本及 margin，验证 argmax/位置/状态，再决定是否需要主模型量化对照。
不通过放宽 greedy 接受条件提升数字，这会改变原采样语义。

本轮 artifacts 为 `artifacts/qwen38-autoround-acceptance-20261007/`：

| 文件 | 证据 |
| --- | --- |
| `probe.py`、`plan.json`、`requests.jsonl`、`probe.log` | 八次短请求、完整输入/输出 IDs、SSE 与计数器增量 |
| `checkpoint_compare.py`、`checkpoint-comparison.json` | 全部 MTP/head/embed tensor 比较及专家映射 |
| `analyze.py`、`summary.json`、`analyze.log` | 合并计数、条件率、配置/tokenizer 差异与固定样例 |
| `pytest-existing.log`、`pytest-state.log`、`pytest-fixture-recheck.log` | 原始通过、失败及修复后复查日志 |
| `validate_final.py`、`delivery-validation.json`、`precommit.log` | 普通服务、物理 GPU、诊断 RPC 关闭、源码 hash 与静态检查 |

最终 API `/health` 和 `/v1/models` 正常，诊断 RPC 为 404，PP worker 在
物理 GPU 2/3；局部测试退出释放 GPU 4，DeepSeek unit inactive/MainPID=0。

后续按用户命令完成了
[同输入 AutoRound / NVFP4 对照](qwen38-autoround-nvfp4-acceptance-comparison-2026-10-07.md)。
新测量没有发现 NVFP4 普遍提高接受率，替代了上述仅有历史 NVFP4 记录的
证据边界。该后续实验结束后保留 NVFP4 在 26661，AutoRound 停止。
