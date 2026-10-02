# DSV41F target 第五轮：SWA metadata 组内融合

日期：2026-10-02。保留上一轮8 warp、W13 packed-word 和 compressed
MLA/compressor 单请求 token map；不改 FP8 dense。
原始产物：`artifacts/dsv41f-target-opt-round5-20261002/`。

后续验收：第六轮以同进程A/B和512-token复核确认端到端收益，见
[metadata端到端验收](vllm170-dsv41f-metadata-e2e-2026-10-02.md)。
下文“未启用生产”是第五轮结束时的历史状态。

## 1. 本轮边界

进场时 `dsv41f-manual.service` 已 inactive，临时 unit 文件已不存在，26661
没有监听，GPU2–6均无模型进程。主机启动时间为06:34:25。
历史 journal 的最后一次 stop 发生在10月1日23:49，早于本次启动；不能把停机
直接归因于重启。本轮保留停机状态，不重新加载模型、不扫 q、不比较重启前吞吐。

在空闲 CMP170HX SM80.0 上完成源码、算子和真实 builder 的隔离验证：
UUID `GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216`，PCI02:00.0，74 SM。
phase 未启用；采用 CUDA Events/Graphs，未使用 CMP 上不可用的 CUPTI/NCU。

**本轮确定了 metadata 的局部固定成本并实现候选，尚未取得生产验收和整轮收益。**
旧拟合截距33.87 ms仍不能用此局部数据完整解释，也不能与本轮数字相减。

## 2. 两处重复 launch

V4.1 SM80 的每组 SWA builder 原先依次执行：

1. 单请求 token map 清零，或多请求通用 map。
2. `slot_mapping >= 0` 生成临时 bool tensor。
3. 将 bool tensor copy 到持久 validity buffer。
4. 独立 kernel 构造 SWA lens 和 ragged indptr。
5. 另一个 kernel 重读输入、构造 ragged indices。

改动在 `vllm/models/deepseek_v4_1/amd/rocm.py`：

- `torch.ge(slot_mapping, 0, out=is_valid_token)` 直接写持久 buffer，省一次 copy
  launch 和临时输出。每步仍从 live slot 重新计算，保留 rejected suffix / PAD。
- 新小 batch kernel 在每个 token CTA 内重算小规模 prefix sum，直接得到自己的
 输出区间，并同时写 lens、indptr、indices，省独立 indptr launch。
  CTA 不读取其他 CTA 的输出，没有跨 CTA 等待；各行写入的区间不相交。

单请求每组由5个 GPU 操作降到3个。多请求 map 仍为原通用算法，省的是同样两处
SWA 操作。四组 SWA 的 slot/table 和输出 buffer 保持独立，不跨组共享 validity。

开关为 additional config 的 JSON boolean `dsv41_sm80_fused_swa_metadata`，
**默认 false**。builder 加载期还要求 CUDA device 和精确 SM80.0。
ragged 融合进一步要求 `1 <= num_decode_tokens <= 48`、`1 <= index_width <= 256`；
其他形状调用原两 kernel 算法。关闭开关时 validity 也回到原实现。
V4.0、SM90、ROCm 原路径未改。existing direct-ragged 开关继续生效。

同一个 V4.1 builder 同时处理 target causal 和 DSpark draft noncausal，
因此候选覆盖两种流程和各 SWA KV group；不修改 draft 的 query 数、路由或 attention。
`load_dflash_model` 保留 additional config，draft 不需要第二套配置。

## 3. 先分离两个候选，再测正式 builder

`probe_swa.py` 对基线 / validity-out / ragged-fused / 两项同时逐一对照。
q6 的 MLA C1/C2 + compressor + 四 SWA 组合结果：

| arm | host ms | 独立 Graph GPU µs |
| --- | ---: | ---: |
| 基线 | 1.378 | 62.464 |
| 仅 validity-out | 1.277 | 55.296 |
| 仅 ragged-fused | 1.185 | 56.320 |
| 两项同时 | 1.063 | 47.104 |

这一步已包含 PAD slot；两项改动各自有效，没有凭两项叠加结果猜测归因。

正式 benchmark 复用 `token_map_metadata_bench.py`，新增 `--variant swa-metadata`，
对实际 `builder.build` 的候选开关交替测三轮，每 arm / 轮40样本。
baseline 保留上轮单请求 token map 优化，candidate 只增加本轮 SWA 改动。
q5 使用 noncausal，q6/q12 使用 causal。
输入是合成32K上下文、max-seqs8、max-batched2048、block table width8192；
不加载模型，不包含 indexer、runner、PP 或采样。

| 输入 / 代表性组合 | host 基线→候选 ms | 降幅 | 隔离 Graph GPU 基线→候选 µs |
| --- | ---: | ---: | ---: |
| q6，MLA C2 + compressor + 四 SWA | 1.244→0.908 | 27.01% | 61.440→43.008 |
| q6，MLA C1/C2 + 四 SWA | 1.144→0.838 | 26.72% | 59.392→43.008 |
| q5 noncausal，source 组合 | 1.224→0.914 | 25.33% | 61.440→45.056 |
| q5 noncausal，双 MLA 组合 | 1.171→0.871 | 25.66% | 59.392→43.008 |
| q12 双请求，source 组合 | 2.168→1.911 | 11.86% | 155.648→147.456 |
| q12 双请求，双 MLA 组合 | 2.175→1.882 | 13.48% | 153.600→146.432 |

三轮 host 中位数都同向。没有 SWA 的 MLA C1/C2 + compressor 控制组没有改动，
host 变化方向混合，GPU 也有微小浮动，不能记作收益。
所有 map、SWA validity、lens、indptr 和有效 indices 均与基线精确一致，
Graph replay 后也一致；比较前 clone，避免持久输出相互覆盖导致伪一致。

q6 约省0.306–0.336 ms host、16–18 µs GPU，主要收益是少 launch。
这些组合只是代表性 KV roster，不是完整 rank metadata；不能乘五当整轮节省。
upstream 前向可以遮蔽下游 metadata，真实关键路径收益需模型测量。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python -m benchmarks.deepseek_v4_1.token_map_metadata_bench \
  --variant swa-metadata --samples 40 \
  --out /tmp/dsv41f-swa-metadata-builders.json
```

## 4. 融合形状边界与编译资源

`probe_ragged_boundary.py` 只测 ragged 构造，Graph 内连续64次调用，四个交替
轮次，每轮每 arm20样本，CUDA Event 时间除以64；排除编译、分配和验证。
这是热 metadata / L2 下的 Graph 结果，没有模型权重冷缓存压力。

| token 数 | causal 基线→融合 µs | noncausal 基线→融合 µs |
| --- | ---: | ---: |
| 1 | 4.272→2.224 | 4.368→2.400 |
| 5 | 4.416→2.480 | 4.528→2.512 |
| 6 | 4.400→2.512 | 4.512→2.528 |
| 12 | 3.504→2.016 | 3.552→2.032 |
| 48 | 3.696→2.464 | 3.760→2.496 |

因此上限48有实际性能依据。未固定 GPU clock，比较限于每一行的交替 A/B；
不能据绝对值解释 q12为何比q6快。kernel math 与所有有效输出精确一致。

编译资源见 `compiled/resources.json` 与对应 PTX/cubin：q6/q48 都是4 warp，
causal21寄存器、noncausal22寄存器，均零 spill；q48 shared8 B，q6 shared0。
本轮热点没有此前 partial attention 的寄存器溢出特征，收益来自融合两次 launch。

## 5. 回归与最终状态

复用现有 attention suite，先确认契约：输出是当前 live 输入的 lens/indptr/indices；
失败风险是 stale PAD/seq/table、空 ragged 行、buffer 越界和 Graph pointer 改变。
最便宜的验证是既有独立 dense→pack reference，加真实 GPU Graph 动态输入回归。

测试覆盖 causal/noncausal、短上下文、多请求、q1/q6/q12/q48、全 PAD、混合 PAD，
以及49 token / width512 的原算法 fallback。
Graph 捕获后改变 seq lens、validity、block table，对照原算法，检查指针和尾部 canary。
validity-out 额外复用 CPU/CUDA 的旧 mask→live→PAD 迁移测试。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python -m pytest \
  tests/kernels/attention/test_dsv4_sparse_prefill_sm80.py -q
```

完整 suite **65 passed / 14 warnings / 27.42 s**。
随后只增加配置 boolean 类型校验，并定向复测相关构造用例：1 passed，64 deselected。
三个 Python 文件的全部适用 pre-commit hook 通过，含 mypy；`git diff --check` 通过。
首轮 mypy 的 additional_config union 类型错误已修复，原失败日志保留。
正式 builder timing 在最终 boolean 校验加入前完成；该修改不涉及 kernel 或 build
计时区域，最终 source seal 与 benchmark 内 source SHA 分开保存，不声称逐字相同。

代码和 benchmark 已落地，开关默认关闭，没有修改或启动生产 unit。
生产模型输出、接受率、整轮 tok/s 尚未验收，候选不能计为已部署的端到端优化。
已有模型级 evidence 属于上一轮版本，不能代替本轮验收。
所有 GPU 实验已结束，空闲测量卡恢复6 MiB / 0%，phase control 不存在；
工作树未提交，上一轮改动保留。

未来启用需要在原 additional config 中合并字段，保留 CED 和 TileLang MoE：
`{"ced_prefill":true,"dsv41_sm80_moe_tilelang":true,"dsv41_sm80_fused_swa_metadata":true}`。
回退将最后一个字段设为 false，继续使用既有算法；`before/rocm.py` 另存本轮 runtime
原文件。该候选的接受条件仍是模型输出/接受精确回归，以及无 nested metadata
profiler 干扰的同进程或同配置生产测量。
