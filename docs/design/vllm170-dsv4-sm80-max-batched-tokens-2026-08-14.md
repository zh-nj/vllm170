# DeepSeek V4 SM80 prefill 调度上限对照（2026-08-14）

## 结论

三张 CMP170HX、PP3/TP1、精确 5,000-token prefill、DSpark 7-token 的部署
形状下，`max-num-batched-tokens` 不是越大越快：

- `2048` 可运行，实际 target 上限为 2,000，形成
  `2000 + 2000 + 1000` 三个微批；
- `4096` 实际 target 上限为 4,048，只形成 `4048 + 952` 两个微批，clean
  benchmark 的 TTFT 比 2,048 慢 `32.93%`；
- `8192` 在当前 PP2 显存边界上无法完成 mixed-token warmup，不能作为可用配置；
- `1715` 为本形状的推广配置：预留 48 个 speculative slots 后正好留下
  1,667 个 target slots，形成 `1667 + 1667 + 1666`，131K 部署长度的 TTFT
  比 2,048 改善 `4.08%`，decode 变化只有 `+0.27%`。

因此只把 CMP170HX 项目启动器的默认值改为 1,715；vLLM 通用默认和其他流量
形状不变。其他 prompt 长度、并发数或 speculative width 必须重新扫参，可通过
`DSV4_MAX_BATCHED_TOKENS` 回退或覆盖。

## 测量协议

共同条件为 DSV4F、三张 CMP170HX `1,2,3`、PP3/TP1、layer partition
`15,15,13`、FP8 KV、Marlin MXFP4 MoE、仅 `M>=1000` 使用 block-M 48、
DSpark width 7、精确 5,000 prompt tokens、60 completion tokens、固定 prompt
hash 和 seed。每个可运行配置先以同形状预热，再连续测量 5 次。

部署决策使用 `DSV4_PROFILE_EXECUTION=benchmark`：CUDA Graph 开启，不启动
Torch profiler，也不插入 phase CUDA-event synchronize。分模块归因另用 graph
诊断模式；诊断时间不能与 clean endpoint latency 混用。

## 2,048 / 4,096 / 8,192 对照

为了让三个候选先接受同一 5K 请求，控制组使用 `max_model_len=8192`、每 rank
1.6 GB KV、`max_num_seqs=8`。显存为请求有效窗口内的 MiB 峰值。

| 配置 | 实际 target 微批 | TTFT p50 | 相对 2,048 | DSpark ms/round p50 | 接受率 p50 | GPU1/2/3 峰值 | 结论 |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 2,048 | `2000 + 2000 + 1000` | 1328.393 ms | 基线 | 34.754 ms | 27.86% | 58026 / 57224 / 62876 | 可用 |
| 4,096 | `4048 + 952` | 1765.867 ms | `+32.93%` | 34.728 ms | 20.88% | 58818 / 58216 / 64212 | 性能拒绝 |
| 8,192 | 请求前失败 | - | - | - | - | 60704 / 59410 / 64456 | PP2 warmup OOM |
| 1,715 | `1667 + 1667 + 1666` | 1264.184 ms | `-4.83%` | 34.734 ms | 22.36% | 57974 / 57072 / 62834 | 候选通过 |

原始 run：

- 2,048：`20260814T063904Z`；
- 4,096：`20260814T064122Z`；
- 8,192：`20260814T055722Z`；
- 1,715：`20260814T064332Z`。

三组可运行配置的每轮 decode 时间只相差 0.026 ms，说明 scheduler 上限没有
改变 decode/small-M kernel 路径。60-token decode 总时长和 tokens/s 会随实际
生成轨迹的 DSpark 接受率改变，不能在不同 output hash 间直接用来判定 batch
配置；因此表中并列报告稳定的 `ms/round` 和接受率。

8,192 的失败不是请求超长。服务在 mixed-token warmup 的 PP2
`compress_norm_rope_store_triton` 中发生 CUDA OOM。把 KV 从 1.6 GB 降到
1.2 GB 会在 admission 阶段拒绝 8,192 长度，只提供 7,632-token 估算上限；取
1.32 GB 虽可通过 admission，仍在同一 compressor warmup OOM。因此当前显存
区间内不存在已验证可启动的 8,192 配置。

## 131K 部署长度门禁

最终推广重新使用启动器的 131,072 max model length 和每 rank 1.6 GB KV，唯一
变量为 scheduler 上限。该门禁使用 8 completion tokens 隔离 TTFT；完整
decode-60 验收见上一节：

| 配置 | TTFT 五次样本 | TTFT p50 | decode p50 | 有效窗口 GPU1/2/3 峰值 |
| ---: | --- | ---: | ---: | --- |
| 2,048 | 1309.480 / 1312.844 / 1320.852 / 1325.916 / 1324.530 ms | 1320.852 ms | 153.776 ms | 58124 / 57314 / 62990 MiB |
| 1,715 | 1266.985 / 1266.491 / 1265.859 / 1267.533 / 1270.485 ms | 1266.985 ms | 154.195 ms | 57980 / 57172 / 62930 MiB |

1,715 的 TTFT 改善为 `4.08%`；decode 差异 `+0.419 ms`，即 `+0.27%`，
没有形成可归因退化。三个 PP rank 的有效窗口峰值分别减少 144、142、60 MiB。
原始 run 为 `20260814T062830Z` 和 `20260814T063049Z`。

131K 下 4,096 + 1.6 GB KV 在 admission 阶段失败：服务至少需要 2.56 GiB KV，
当前只有 1.49 GiB，估算最大长度 10,184。把 KV 提到 2.8 GB 后又在 PP2 的同一
compressor warmup OOM。因此 4,096 不仅对 5K TTFT 更慢，也与当前 131K 显存
预算不兼容。8,192 的资源需求更高，不进入部署候选。

## 子模块归因

以下为 graph 诊断模式的 PP0 每请求 CUDA-event 总量；它们用于解释方向，不作为
clean endpoint 时间。2,048、4,096、1,715 分别来自
`20260814T061623Z`、`20260814T061914Z`、`20260814T062555Z`。

| PP0 stage | 2,048 | 4,096 | 1,715 |
| --- | ---: | ---: | ---: |
| attention metadata | 3.844 ms | 3.484 ms | 3.137 ms |
| model inputs | 0.065 ms | 0.044 ms | 0.057 ms |
| model forward | 761.928 ms | 732.663 ms | 729.133 ms |
| attention | 317.141 ms | 315.825 ms | 306.418 ms |
| FFN | 368.172 ms | 342.989 ms | 350.994 ms |
| Marlin MoE parent | 367.930 ms | 342.814 ms | 350.753 ms |
| Marlin W13 | 197.488 ms | 182.101 ms | 192.261 ms |
| Marlin W2 | 106.041 ms | 97.444 ms | 103.947 ms |

4,096 的 model、attention、MoE、W13、W2 计时都没有变慢，却出现约 442 ms 的
clean TTFT 回退；metadata 和 input 构造也更低。因此可排除“Marlin 或 attention
单模块变慢”作为主因。结合 4,096 只有两个 target 微批，而 PP 深度为 3，最强的
现有解释是未充分填满 PP3 后产生的 pipeline idle/synchronization bubble；这部分
位于嵌套 GPU 模块范围之外。该判断是基于时间分解的归因，不是 CUPTI timeline
结论，因为本机驱动返回 `CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED`。

1,715 则在不增大单块 kernel 压力的前提下恢复三个近似等长微批。诊断 run 中每次
请求的 PP0 block 时间约为 233.6、247.0、249.0 ms；Marlin 结构仍由 W13
约 55%、W2 约 30% 主导，没有用 kernel 退化换取调度收益。

## 决策与后续优化

1. CMP170HX PP3 启动器和 profiling harness 默认改为 1,715，并保留环境变量
   回滚；通用 scheduler 行为不修改。
2. 2,048 保留为其他流量形状的显式对照，不再是 5K/PP3 默认。
3. 4,096 在当前形状同时触发 TTFT 与 131K 容量拒绝，不推广。
4. 8,192 必须先降低 PP2 mixed-token warmup/capture 峰值，重点检查 compressor
   输出、并行 attention 临时量与 CUDA Graph residency；在服务能稳定启动前不做
   endpoint 性能宣称。
5. 后续调度研究应扫 `ceil(prompt_tokens / PP)` 附近的 target 容量，并显式加上
   DSpark 为所有 sequence 保留的 slots；不能只按越大的 batch 上限越好来选值。
