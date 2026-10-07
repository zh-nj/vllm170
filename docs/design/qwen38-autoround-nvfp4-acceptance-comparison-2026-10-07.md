# Qwen3.8 Flash Next：AutoRound / NVFP4 接受率对照

同一 checkout、物理 GPU 2/3、PP2/TP1、K3，重放相同 token 输入和采样
参数后，NVFP4 没有普遍改善接受率。此前检索加说明文本的 1K/32K 任务，
本轮 AutoRound 为 49.41%/39.30%，NVFP4 为 47.27%/33.77%。当前证据
不支持将 AutoRound INT4 认定为低接受率的主要原因。这也不等于排除了
两边共同的实现问题；尚缺相同输出轨迹的 logits 参考。

## 配置与测量合同

目标 HEAD 为 `e2e327dc53b93acb85c8b56b526d369eb5461421`，保留既有本地
修改。本轮没有修改推理代码。AutoRound 测量前核对已加载实例的相关源码
hash 与上一轮记录一致；NVFP4 测量结束再次核对相同源码 hash。

| 项目 | AutoRound | NVFP4 |
| --- | --- | --- |
| checkpoint | `Qwen3.8-Flash-Next-W4A16-AutoRound` | `Qwen3.8-Flash-Next-NVFP4` |
| 物理 GPU、分片、MTP | GPU 2/3，PCI 排序，PP2/TP1，26/22，K3 | 相同 |
| model dtype | 显式 BF16 | 配置解析为 BF16 |
| target 专家路径 | INC 对称 INT4/group128、Marlin | ModelOpt NVFP4/group16、Marlin |
| 最大长度、并发、batch tokens | 262144、4、4096 | 相同 |
| CUDA Graph 请求配置 | FULL_AND_PIECEWISE，1/2/4/8/12/16 | 相同 |
| prefix caching、CPU PLE、HC SM80 开关 | 启用 | 启用 |
| `VLLM_QWEN4_EXP_MTP_FP8_LM_HEAD` | 1 | 1 |
| 显存比例 | 0.90 | 0.95，按用户命令 |
| PP relay | INC 范围内 GPU scatter 开关启用 | 通用 relay，过滤无效 request index |
| prepare-ahead / prefix metadata / metadata reuse | 关闭 | 默认关闭 |

NVFP4 按用户提供的完整参数启动，没有追加 INC 量化选项或放宽优化资格。
`qwen_mtp_utils.py:34` 限定 INC，NVFP4 使用
`gpu/model_runner.py:1129-1142` 的通用 relay，同样过滤无效 request index。
两边 target 量化和 PLE dtype 也不同，因此这是完整 checkpoint 及其 serving
recipe 对照，不是只有 INT4 一个变量的消融。

NVFP4 日志确认 BF16、`modelopt_fp4`、Marlin NVFP4 MoE。
不加载权重的配置解析确认 target/draft 均 BF16、词表均 248320、draft 为
`qwen4_exp_mtp`、`tie_word_embeddings=false`。两边请求启用 MTP FP8 head，
但本轮未注入 worker 读取 head 对象或 packed dtype；启动日志没有该转换
确认行，不将环境开关单独视为完整运行时数值路径证明。

先在普通 AutoRound 服务完成新基线，再停止其 systemd cgroup，验证
26661 监听和 GPU 2/3 compute process 全部释放，两卡约 6 MiB。随后启动
独立 `qwen38-nvfp4-manual.service`。约 576 秒后加载、编译和图捕获完成，
健康检查就绪。未同时加载两份模型，DeepSeek 始终停止。

每份模型两次 shape 预热，然后 14 次正式 C1 文本请求：

- 固定英文说明，精确 1K/32K，各两次、输出 128 token。
- 重复 `blue river` 和顺序整数 JSON，各两次、输出 128 token。
- 既有 needle retrieval 加说明文本，精确 1K/32K，各三次、输出 256 token。

前两组直接使用上一轮 `requests.jsonl` 的 payload；needle 使用既有
`prompts/single-1024-1.json` 和 `single-32768-1.json`。逐条断言完整 token
IDs、prompt SHA256 相同。除 model 名称、唯一 cache salt 外，采样参数
完全一致：temperature=0、top_p=1、seed=42。短请求 `ignore_eos=false`，
needle 延续原实验 `ignore_eos=true`。

请求前确认无运行/等待请求，每次成功计数增量 1、prefix hits 0、草稿数
为验证轮数三倍、分位置接受数之和等于总接受数。两边最终抢占计数为零。
完整原始输入、输出 ID、SSE 和计数器保留在 artifacts。

## 本轮结果

合并同一负载全部正式请求的整数计数，接受率为 `A/(3R)`，平均长度指标
`L=1+A/R`；A 为接受草稿总数，R 为验证轮数。最后一轮指标可能包含超过
客户端长度上限的已接受草稿。

| 输出任务 | 输入 / 输出上限 | 请求数/模型 | AutoRound 接受率 | NVFP4 接受率 | 差值 pp | AutoRound L | NVFP4 L |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 固定英文说明 | 1K / 128 | 2 | 44.85% | 44.85% | 0.00 | 2.345 | 2.345 |
| 同一英文问题 | 32K / 128 | 2 | 38.98% | 42.77% | +3.79 | 2.169 | 2.283 |
| 重复短语 | 1K / 128 | 2 | 100.00% | 100.00% | 0.00 | 4.000 | 4.000 |
| 顺序整数 JSON | 1K / 128 | 2 | 100.00% | 98.96% | -1.04 | 4.000 | 3.969 |
| 检索 + 英文说明 | 1K / 256 | 3 | 49.41% | 47.27% | -2.13 | 2.482 | 2.418 |
| 检索 + 英文说明 | 32K / 256 | 3 | 39.30% | 33.77% | -5.53 | 2.179 | 2.013 |

分位置是连续接受前缀至少达到该位置的比例，分母都是验证轮数，不是
彼此独立的位置准确率：

| 任务 | AutoRound 第一/第二/第三位置 | NVFP4 第一/第二/第三位置 |
| --- | --- | --- |
| 英文说明 1K | 74.55 / 40.91 / 19.09% | 73.64 / 40.00 / 20.91% |
| 英文说明 32K | 64.41 / 33.05 / 19.49% | 70.80 / 39.82 / 17.70% |
| 检索说明 1K | 70.87 / 45.63 / 31.72% | 70.13 / 45.60 / 26.10% |
| 检索说明 32K | 65.34 / 35.23 / 17.33% | 59.32 / 29.13 / 12.86% |

NVFP4 后两枚草稿连续命中比例仍较低。其 32K needle 三次接受率为
38.38%/34.39%/29.17%，1K 为 43.84%/50.65%/47.62%。2–3 个短样本不足
以确定每个负载的小差异是否稳定，但本轮没有普遍恢复到 70–80%。

相同输入不保证生成输出相同。对应两臂样本的共同输出 token 前缀为：
英文 1K 34/5，英文 32K 34/86，needle 1K 51/27/27，needle 32K 18/30/22。
重复短语两次均完整 128 token 一致；JSON 因空格差异仅前三个 token 一致。
以上不是 teacher-forced 相同输出轨迹的 logits 比较，不能将总率差异
单独归因于权重量化。

28 次正式请求均生成规定长度，以 length 结束，无特殊 token。12 次
needle 均检索正确校验码。短语和 JSON 通过忽略空白差异的预期文本前缀
检查；128 token 截断不能证明完整 JSON 任务或一般模型准确率。
英文说明没有单独质量评分。

首帧之后的客户端 decode 速度中位数，needle 1K/32K 为 AutoRound
141.42/120.66、NVFP4 96.96/76.54 tok/s。先 A 后 B 独立加载、未锁频，
relay 范围和显存比例不同，速度仅为本轮 recipe 观察，不能作为某个
kernel 或量化格式的受控收益。

## 结论、交付和证据

本轮直接 NVFP4 测量取代了“只有历史 NVFP4 数字”的证据边界。以前约
68–79% 的记录来自不同 checkout、GPU、prompt、长度等条件，没有在
本轮固定任务上复现，不能用旧数字推断 AutoRound 实现损坏，也不能据此
认定所有自然语言低接受率都正常。

两份 checkpoint 的全部 MTP/head/embed 权重逐值相同，见
[完整接受率排查](qwen38-autoround-acceptance-investigation-2026-10-07.md)。
更换 target checkpoint 未系统性提高接受率，弱化了“AutoRound 特有量化
差异是主因”的解释。若继续查根因，应锁定共同输出轨迹，检查每个拒绝
token 的 draft ID、target top1/top2、logit gap 及 accepted state/position，
再区分预测能力、数值执行顺序和共同实现缺陷。

NVFP4 保留在 `0.0.0.0:26661` 按用户命令服务。API PID 2972468，PP0/PP1
PID 2976070/2976071，compute ownership 属于该 systemd cgroup 和物理
GPU 2/3。`/health`、`/v1/models` 正常，chat smoke 返回 `1+1=2`，最大长度
262144，开发 RPC `/collective_rpc` 为 404。AutoRound 和 DeepSeek 均
inactive、MainPID=0。

启动 KV 估计 1,042,908 token、3.98 个最大长度请求；本轮没有完整 256K
输入或四个满长度请求，不把估计当作实测容量。

artifacts：`artifacts/qwen38-quantization-acceptance-comparison-20261007/`。

| 文件 | 用途 |
| --- | --- |
| `frozen-plan.json`、`replay.py` | 精确输入和采样参数、预热和正式请求 |
| `autoround-requests.jsonl`、`nvfp4-requests.jsonl`、两份 client log | 原始输出、SSE 和接受计数 |
| `analyze.py`、`comparison-summary.json` | 合并计数、分位置/条件率、输出共同前缀、语义检查 |
| `autoround-runtime-before.json`、`switch-release.json`、`source-sha256.json` | 原实例配置、释放检查、源码边界 |
| `launch_nvfp4.py`、`launch-nvfp4.json`、`server-nvfp4.log`、`nvfp4-ready.json` | 完整启动命令与加载日志 |
| `config-resolution.json`、`config-resolution-recheck.log` | 不加载权重的 target/draft dtype 与配置解析 |
| `validate_final.py`、`delivery-validation.json`、`delivery-client.log` | 服务、GPU ownership、RPC 和 chat smoke |
| `autoround-metrics-final.txt`、`nvfp4-metrics-final.txt`、`precommit.log` | 请求/抢占状态和静态检查 |

Python 实验脚本通过 `py_compile`，文档通过 scoped pre-commit。未修改
kernel、模型或采样实现，没有重复上一轮 GPU 单元测试；本轮验证重点是
两份模型的真实 serving 请求和服务交付状态。
