# DeepSeek V4.1 Engram RAM / SSD 端到端对照

日期：2026-10-05。基线 commit：`a33bd2b5cb38590dd04a6228bfadb65a4b4db1d1`，含工作区已有 SSD 功能和速度优化修改。

## 结论

SSD 后端已完成完整 PP5 服务启动、graph capture 和四档长输入推理。两种后端的输出与 DSpark 接受计数一致。SSD 的主要代价在 prefill：长输入即使全部命中文件缓存，也比 RAM 慢；decode 的读数差距较小。其收益是减少不可回收的主内存常驻量。

后续核对发现，本轮沿用的运行配置关闭了此前已验收的 target prepare-ahead，且语料的 DSpark 接受率显著低于历史周期输入。因此下面的 RAM / SSD 对照有效，但不能把 RAM 读数当作已保留全部生产优化的 decode 基线；与历史约 150 tok/s 的差异见后文。

本轮只增加测速与分析，不修改推理路径、不提交、不部署正式服务。

## 固定条件与统计口径

- 5 × CMP170HX SM80，物理 PCI 2–6，PP5 / TP1，分层 8,8,8,8,8，单请求串行。
- 模型 `/media/z/data6/models/DeepSeek-V4.1-Flash`；输入精确为 1024 / 32768 / 65536 / 131072 tokens，输出固定 256 tokens，temperature=0、ignore_eos=true。
- DSpark=5、CED 开启，fast prefill 关闭；batching=2048 / max-seqs=8，KV=2,700,000,000 bytes/rank、FP8，max-model-len=1,048,576；hybrid prefill、BM16 tail、fused SWA metadata、TileLang MoE 开启，target prepare-ahead 关闭。
- FULL 请求在两组均按实际 attention 能力解析为 FULL_DECODE_ONLY；5 个 rank 的 target graph 和分布式 draft graph 均完成捕获。
- 运行时 PyTorch 2.13.0+cu130、TileLang 0.1.15、Triton 3.7.1、FlashInfer 0.6.16.post3；CUDA_HOME、nvcc 与 Triton ptxas 均固定 `/usr/local/cuda-13.3`，使用本轮独立编译缓存。沿用已安装的原生扩展，不重建。
- RAM：完整 FP8 weights + E8M0 scales 驻留 CPU pinned RAM，通过原有 UVA lookup。SSD：weights 在 `/media/z/data6/vllm-engram-ssd`，scales 驻留普通 CPU RAM（两表约 5.72 GiB），ssd_io_workers=8 自适应读取，cpu_cache_gb=0。
- 非周期 Python 代码语料，保存 token IDs 和 SHA256；两组逐档复用相同 IDs。每次请求使用新 cache_salt，所有 KV prefix hit 增量为 0。
- 每档每后端 3 次。首轮单列：RAM 首轮包含部分首次 JIT，不纳入稳态；SSD 首轮在请求前对两张 weight 文件调用 POSIX_FADV_DONTNEED。稳态采用后两次中位数；此时 SSD 服务 read_bytes 增量全部为 0，明确是热文件缓存。
- TTFT 从本机 HTTP 请求到首个含 token IDs 的 chunk；prefill_s 使用服务端 request_prefill_time_seconds 的单请求增量，即首次 scheduled 到 first token，包含 PP、CED、lookup 等流程成本，不是 GPU kernel 时间。prefill tok/s = prompt_tokens / prefill_s。
- decode tok/s = 首个 chunk 之后的实际 token 数 / 首到末 token chunk 的时间；本轮每个首 chunk 都有 1 token，因此分子为 255。没有把流式 chunk 数当 token 数。整请求耗时另列，不把 prefill 算进 decode。

## 稳态速度（热文件缓存、KV prefix 不命中）

| 输入 | RAM TTFT s | SSD TTFT s | RAM prefill tok/s | SSD prefill tok/s | RAM decode tok/s | SSD decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 0.679 | 0.761 | 1532.4 | 1363.3 | 46.42 | 43.74 |
| 32K | 4.842 | 7.116 | 6917.5 | 4678.0 | 47.16 | 44.07 |
| 64K | 9.113 | 13.252 | 7349.8 | 5015.1 | 47.40 | 46.31 |
| 128K | 18.883 | 26.770 | 7070.0 | 4954.4 | 54.36 | 51.10 |

| 输入 | SSD prefill 速度变化 | SSD decode 速度变化 | RAM 整请求 s | SSD 热缓存整请求 s | 整请求耗时增加 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1K | -11.0% | -5.8% | 6.173 | 6.593 | +6.8% |
| 32K | -32.4% | -6.6% | 10.250 | 12.903 | +25.9% |
| 64K | -31.8% | -2.3% | 14.497 | 18.760 | +29.4% |
| 128K | -29.9% | -6.0% | 23.579 | 31.762 | +34.7% |

## SSD 首轮权重页冷缓存

这里只回收本轮生成的 weight 文件缓存，scales 仍驻留 RAM，不回收系统缓存或其他模型数据。pread 使用普通文件 I/O；冷状态是建议回收，并由服务进程实际 read_bytes 增量确认。读数覆盖整个服务请求，不是逐文件 I/O 插桩。decode 在同一请求的 prefill 后开始，因此这一列不能解释为独立的全冷 decode。

| 输入 | SSD TTFT s | SSD prefill tok/s | SSD decode tok/s | 整请求 s | 服务物理读取 GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1K | 1.344 | 768.1 | 37.23 | 8.193 | 0.203 |
| 32K | 13.114 | 2520.6 | 39.26 | 19.609 | 2.756 |
| 64K | 23.796 | 2777.2 | 39.87 | 30.192 | 4.927 |
| 128K | 49.376 | 2673.4 | 45.80 | 54.944 | 8.426 |

## 一致性与解释

24 次请求均返回完整 usage、256 个 delta token IDs、finish_reason=length 和 DONE；每档所有 RAM / SSD 请求的 256 IDs 逐位一致，总计 3072 对 RAM / SSD 输出 IDs。各档 DSpark drafts / drafted tokens / accepted tokens 也完全一致：

| 输入 | draft 轮数 | drafted tokens | accepted tokens |
| --- | ---: | ---: | ---: |
| 1K | 90 | 450 | 165 |
| 32K | 90 | 450 | 166 |
| 64K | 86 | 430 | 174 |
| 128K | 79 | 395 | 181 |

因此本轮两后端的耗时差不是输出长度、接受率或 KV prefix 命中造成的。只验证了这些代码语料上的 greedy 等价，不代表通用模型质量评测。

热 SSD 请求的物理读取量为 0，而长输入 prefill 仍下降约 30%–32%。这排除了这些热轮由 NVMe 读取带宽主导的解释；源码中的索引 D2H、CPU 去重与逐行 pread、packed bytes staging / H2D 仍执行。各部分的独立耗时未在本轮插桩测量，不能把总差额直接归给某一个阶段。

冷轮在 32K / 64K / 128K 多读约 2.76 / 4.93 / 8.43 GiB，其 TTFT 为 RAM 稳态约 2.6–2.7 倍；文件缓存能显著缓解这段代价。

decode 稳态读数低 2.3%–6.6%，但每组只有两次稳态样本，未做 A-B-A，也未锁 GPU 时钟。64K 的 RAM 46.03–48.78 与 SSD 45.58–47.04 tok/s 区间重叠，其 2.3% 差距不能作为显著回归证据。大幅 prefill 差距明确，小幅 decode 差距应作为本次观测值使用。

## 为什么历史约 150 tok/s、本轮只有约 47 tok/s

复核历史原始 token-ID chunks、spec counters 和实际启动命令后，确认既有运行配置退化，也有工作负载变化。两次都开启 DSpark-5；本轮 target / draft graphs 均完成捕获，没有证据支持 DSpark 被关闭或图整体失效。旧、新测速均使用首个 token-ID chunk 之后的实际 token 数，不是 chunk 数，也没有混入 TTFT。

| 32K 单请求稳态 | 历史 RAM 生产 | 本轮 RAM | 本轮热 SSD |
| --- | ---: | ---: | ---: |
| 输出 token 预算 | 512 | 256 | 256 |
| target prepare-ahead | 开启 | 关闭 | 关闭 |
| draft 轮数 | 88 | 90 | 90 |
| accepted / drafted | 423 / 440 | 166 / 450 | 166 / 450 |
| draft token 接受率 | 96.14% | 36.89% | 36.89% |
| 首 chunk 后实际输出 / 轮 | 5.807 | 2.833 | 2.833 |
| client 估算 ms / 轮 | 39.409 | 60.079 | 64.296 |
| decode tok/s | 147.348 | 47.163 | 44.073 |

轮周期使用 client decode span 除以 draft counter；本表每个请求的含 token-ID chunk 数都恰好为 draft 轮数加一。它包含完整交付路径，仍不是 target kernel 或精确 GPU step 计时。TPS 和周期分别取中位数，两个样本时存在很小的倒数统计差异。

**配置退化有明确的历史 A/B 证据。** 10 月 2 日同一加载实例的交替对照，只改变 target 接收顺序：开启为 145.935 tok/s、39.790 ms/轮，关闭为 100.149 tok/s、57.988 ms/轮，关闭多花 18.197 ms/轮。对应源码仍在 `gpu_worker.py`：关闭时先调用 `irecv_tensor_dict` 等上游 header，再进入 runner 准备；开启时通过 lazy receive 让独立的输入与 attention metadata 准备先推进，首次消费上游 tensors 时再等待。它减少 PP1–PP4 暴露在关键路径上的准备和等待，历史诊断的各 rank span 不能简单相加。

配置传播链也可复现：

1. 10 月 2 日 DSpark 优化验收后的 `production-launch.json` 为 `dsv41_sm80_target_prepare_ahead:true`。
2. 随后的无 DSpark decode 实验拒绝将该优化推广到 q1，运行配置显式设为 false；已验收的 DSpark-5 源码没有撤回。
3. 10 月 3 日恢复 DSpark 时，`production-dspark-launch.json` 只增加 speculative config，保留了 false。该次周期输入交付检查已只有 97.8969 tok/s、44 draft 轮、211/220 接受，说明这段下降早于 SSD 改动。
4. 第八轮 `final-state.json` 仍为 false；本次 RAM / SSD 基准复制这份配置，也保留了 false。本轮只核对两后端参数一致，没有核对历史已验收优化清单，这是测速基线的遗漏。

**剩余的大幅 TPS 变化来自每轮交付量。** 历史输入为 32768-token 周期词串，只有 65 种 token；本轮同长度代码输入有 2515 种 token，两个 prompt 不同。实际接受率从 96.14% 降到 36.89%，首 chunk 后交付量从 5.807 降到 2.833 token/轮，只剩 48.79%。这是不同请求的观测，不能称为同一请求的模型接受率回归。

按观测值，`147.35 × (2.833 / 5.807) × (39.409 / 60.079) ≈ 47.16 tok/s`。其中把新交付量放进历史关闭 prepare-ahead 的 57.988 ms 周期，得到 **48.86 tok/s**，已接近本轮 47.16。当前 RAM 周期比历史关闭样本长 3.61%；这段残差还包含不同 expert 路由、输出长度、prefill 实现、编译缓存和 NCCL 配置等条件差异，没有同实例消融，不能唯一归因。

**SSD 在本轮 32K 配对中的额外代价**是 4.217 ms/轮、TPS 低 6.55%，不是 150→47 的来源。这个数仍受前述小样本和跨实例时钟边界限制。

修复优先级应先恢复已验收的 `dsv41_sm80_target_prepare_ahead:true`，再在一次 RAM 加载内以旧周期输入和新代码输入分别做同实例开/关对照，保持 CUDA 13、输出长度和其余配置不变，并核验 IDs / spec counters。没有测当前语料开启后的速度，因此不能承诺它恢复到 150 tok/s，也不能把历史节省的 18.197 ms 直接扣到当前周期上作为实测。本次复核只读既有证据，没有再次加载模型或改运行配置。

可重算摘要与所有来源 SHA256：`artifacts/dsv41f-engram-e2e-20261005/decode-regression-analysis.json`。历史机制和同实例验收见 [target prepare-ahead 报告](vllm170-dsv41f-target-prepare-ahead-2026-10-02.md)，恢复 DSpark 的交付记录见 [prefill 第四轮](vllm170-dsv41f-ced-prefill-round4-2026-10-03.md)。

## 内存、加载与最终状态

服务 cgroup 的匿名内存：RAM 约 201.4 GiB，SSD 约 17.0 GiB，减少约 184.4 GiB。这不等于总 RSS / 总物理内存只用 17 GiB：SSD 最后一次请求时 cgroup file cache 约 191.9 GiB（包括加载 checkpoint 的缓存和访问工作集），可由 OS 回收；cpu_cache_gb 不限制 OS 文件缓存。本机 RAM 充足，热轮的重复语料工作集能保留，低内存、缓存受压情况下的稳态未测，应参考冷轮边界。

SSD 的两表数据总量约 188.83 GiB，首次完整转换的 weights 写入约从 11:18 到 11:27；PP0 / PP1 model loading 分别 702.77 / 708.44 秒。服务 11:27:51 就绪。这个首次生成成本未计入请求测速。来源 manifest 非空且检查了 safetensors 路径、inode、偏移、大小与时间；本轮未通过再次完整启动单独计时缓存复用。

本轮开始时正式服务 inactive、GPU 无 compute apps；两组使用临时 user systemd unit、26662 端口。测试完成已停止所有测试模型服务，释放 5 张卡，恢复初始未运行状态。保留可复用的 SSD 缓存，未删除旧的未校验缓存；磁盘剩余约 28 GiB。原有 round6–8 和 SSD 实现修改保留。

## 复现与证据

启动参数、工具链与 unit 记录在 artifacts/dsv41f-engram-e2e-20261005/ram-cuda13-launch.json、ssd-cuda13-launch.json、ssd-cuda13-start-command.json；两组除 Engram JSON 外的实际 CLI 参数逐项相等。

测速脚本：benchmarks/deepseek_v4_1/engram_e2e_gate.py；复用 dspark_flow_gate.py 的 token-ID 流验证，只增加可选 cache_salt 和必要的 prefix / prefill / decode metrics 采集。

```bash
.venv/bin/python benchmarks/deepseek_v4_1/engram_e2e_gate.py \
  --label ram-cuda13 \
  --unit dsv41f-engram-e2e-ram-cuda13-20261005.service \
  --prompts artifacts/dsv41f-engram-e2e-20261005/prompts.json \
  --out /tmp/engram-ram-new-run.json

.venv/bin/python benchmarks/deepseek_v4_1/engram_e2e_gate.py \
  --label ssd-cuda13 \
  --unit dsv41f-engram-e2e-ssd-cuda13-20261005.service \
  --prompts artifacts/dsv41f-engram-e2e-20261005/prompts.json \
  --out /tmp/engram-ssd-new-run.json \
  --cold-ssd-dir /media/z/data6/vllm-engram-ssd
```

以上需要先按记录启动对应服务；本轮测试 unit 已停止。默认 lengths 为四档、max-tokens=256、repeats=3；已有输出文件被拒绝覆盖。

原始记录：artifacts/dsv41f-engram-e2e-20261005/ram-cuda13.json、ssd-cuda13.json；汇总与完整性断言结果：comparison.json；逐请求包含资源前后快照、token IDs、chunk 时间、usage、fresh salt 和 Prometheus 增量。

验证：tests/benchmarks/test_dsv41_dspark_flow_gate.py 六个用例通过；指定测速和测试文件 pre-commit 全部通过，git diff --check 通过。未重跑推理组件测试，因为本轮未修改推理实现；之前的 SSD 组件测试记录保留在实现文档。

排除记录：第一次未指定 CUDA_HOME 时 pip CUDA compiler / header 不一致导致 warmup 失败；临时 CUDA 12.8 编译的 RAM 测量保存在 ram-cuda12-excluded.json，明确不纳入以上任何表格。用户确认项目 CUDA 13 后，两组都以独立 13.3 编译缓存重新测量。
