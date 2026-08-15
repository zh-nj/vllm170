# DeepSeek-V4 SM80 output projection P2

日期：2026-08-15

部署：3 x CMP170HX、PP3/TP1、Marlin、DSpark-7

## 结论

本轮晋升 `wo_a` BMM-only 低驻留路径。SM80 output projection 的 `wo_a`
forward 只读取加载期生成的 BF16 grouped-BMM weight；原先随后生成的 Marlin
packed FP8 weight 和展开 scale 不再被读取。现在完成 BMM cache 后释放源量化参数，
并跳过无用的 repack、scale permutation 和 workspace 创建。

这不是用显存换延迟：计算 kernel、输入、输出和 BF16 weight 均未改变。按模型真实
形状，每层静态驻留减少约 32.5 MiB，`15,15,13` 分区理论上分别减少约
487.5/487.5/422.5 MiB。生产启动器默认启用；设置
`VLLM_DEEPSEEK_V4_BMM_ONLY_WO_A=0` 可恢复原预处理。

## 调用链与所有权

CMP170HX 选择 `DeepseekV4AmpereAttention`，其 output projection 继承
`DeepseekV4ROCMAiterMLAAttention._o_proj`：

```text
inverse RoPE
  -> _get_cached_wo_a_bf16
  -> torch.einsum("tgd,grd->tgr")
  -> wo_b Marlin
```

`prepare_fp8_layer_for_marlin` 原先先创建 `bmm_weight_bf16`，然后仍将同一份
`wo_a` checkpoint weight repack 为普通 linear 所需的 Marlin 表示。该 module 在
SM80 路径不会调用普通 linear；所有工程内 consumer 审计也显示 Ampere forward
只读取 `bmm_weight_bf16`。因此释放仅由 DSV4 显式 BMM-only 标记触发，其他 FP8
linear 和默认 BMM cache 行为不变。原 parameter 名称保留为空 tensor，避免改变
module 参数结构。

局部 CUDA 测试覆盖两条分支：默认预处理仍保留 packed weight，BMM-only 分支释放
weight/scale、不创建 Marlin workspace，并逐元素验证保留的 BF16 grouped weight 与
量化参考完全一致。

## CMP170HX PP3 A/B/A

固定相同 prompt hash、5K prefill、60 decode tokens、DSpark-7、
`max-num-batched-tokens=1715`，每组 3 次请求。表中延迟取每组三次中位数；显存为
profile harness 记录的 workload peak。

| 路线 | TTFT | ms/round | decode | acceptance | PP0/PP1/PP2 显存 |
| --- | ---: | ---: | ---: | ---: | ---: |
| packed A1 | 1.22526 s | 32.291 ms | 0.88784 s | 0.17347 | 58080/57062/62846 MiB |
| BMM-only B | 1.22449 s | 31.699 ms | 0.67133 s | 0.30612 | 57594/56576/62310 MiB |
| packed A2 | 1.22127 s | 33.636 ms | 0.84287 s | 0.22857 | 58080/57062/62846 MiB |

B 相对 A1/A2 TTFT 中位数均值为 `+0.10%`，属于噪声。短输出的 acceptance 波动会
同时改变 decode time 和 ms/round，因此本轮不宣称 decode 加速。稳定结论是：

- PP0、PP1 workload peak 均减少 486 MiB；
- PP2 workload peak 减少 536 MiB；其中可由静态 layer 表示严格解释的部分约
  422.5 MiB，额外差值受 allocator/Graph pool 与 0.5 秒显存采样影响；
- 计算路径没有变化，BF16 BMM weight 的单测结果逐元素一致；
- 没有增加 persistent 或临时 workspace。

原始产物：

- `profiles/deepseek_v4/p2-wo-a-bmm-only/baseline-a1/20260815T153119Z`
- `profiles/deepseek_v4/p2-wo-a-bmm-only/candidate-b/20260815T153420Z`
- `profiles/deepseek_v4/p2-wo-a-bmm-only/baseline-a2/20260815T153625Z`

## `wo_b` Marlin launch 扫描

为避免把容量问题误解为单一 launch 问题，本轮同时开放 benchmark-only 的
`thread_k/thread_n/num_threads/sms` 参数，并扫描 CMP170HX 的 70 SM。生产调用未
传这些参数，仍使用 Marlin auto heuristic。

- 大 prefill 形状的 auto 路由已经选择 `k64/n256/256 threads`，70 SM 最优或接近
  最优；没有稳定的手工配置可晋升。
- M=64 的 `k64/n128/128 threads` 局部约快 8%–9%，但它不覆盖主要 5K prefill
  shape，且手工配置不能安全覆盖 M=1667 的尾块，故只保留扫描能力。
- 将非 atomic reduction 的中间累加从 FP32 改为 BF16，在部分形状仅有约 1%–3%
  收益且不稳定；mean absolute error 可由约 `3e-6` 上升到 `1e-3` 量级。该路线
  拒绝，不进入生产。
- 缓存 BF16 `wo_b` 虽能改善 prefill，却需 PP0/PP1 各 960 MiB，已在 P1 阶段因
  长上下文容量约束否决。

因此 `wo_b` 保持 FP32 reduction 和自动 launch。下一轮若继续优化，应研究不新增
每层 weight 副本的 fused/streamed 结构，而不是继续扩张常驻 BF16 cache。
