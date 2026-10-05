# Engram 主存与 SSD 优化模式重测

日期：2026-10-05。CUDA 13.3、PP5/TP1、DSpark=5、CED 开启、
`dsv41_sm80_target_prepare_ahead=true`。主存和 SSD 优化实例各加载一次，
固定相同的 1K/32K/64K/128K Python code prompts，输出 256，fresh KV salt，
每个长度三次，取后两次稳态中位数。SSD 表复用，不重建。

SSD 配置为 `ssd_batch_reads=true`、`ssd_prefetch=true`、scales in RAM、
8 个 IO workers；SSD 第一轮额外对 weight 文件做了 DONTNEED，稳态结果取后两轮。
两种模式的 token IDs、prefix hit 和 DSpark spec counters 均通过校验。

| 输入 | 主存 prefill tok/s | SSD 优化 prefill tok/s | SSD 相对主存 | 主存 decode tok/s | SSD 优化 decode tok/s | SSD 相对主存 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 1564.0 | 1544.0 | -1.3% | 66.18 | 63.87 | -3.5% |
| 32K | 6924.4 | 6289.3 | -9.2% | 67.67 | 66.35 | -1.9% |
| 64K | 7365.2 | 6994.4 | -5.0% | 70.39 | 68.90 | -2.1% |
| 128K | 7058.9 | 6631.6 | -6.1% | 75.98 | 74.68 | -1.7% |

SSD 首轮冷页结果明显慢于稳态：1K/32K/64K/128K prefill 分别为 878/2726/
3137/3211 tok/s；这些数据不用于 warm steady-state 对比。稳态结果显示，
target prepare-ahead 开启后，SSD 优化模式的 decode 退化收敛到约 1.7%–3.5%，
长上下文 prefill 退化约 5%–6%；32K 的 prefill 仍有约 9% 差距。

原始数据：`artifacts/dsv41f-engram-e2e-targettrue-20261005/ram.json`、
`ssd_opt.json`、`comparison.json`。测试服务已停止，正式服务保持 inactive，
GPU compute apps 为空。
