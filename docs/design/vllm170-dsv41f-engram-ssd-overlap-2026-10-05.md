# SSD Engram：批量读取与同轮预取集成复核

日期：2026-10-05。CUDA 13.3，PP5/TP1，DSpark=5，CED 开启，target
prepare-ahead 开启。结果和原始服务日志在
`artifacts/dsv41f-engram-ssd-overlap-20261005/`。

这轮把两个可控优化接入真实推理路径：大批暖页读取使用只读 mmap 的 NumPy
行 gather；同一轮在 PP 非首 rank 等待上游 hidden states 前提交 CPU SSD
读取，原有 GPU staging 仍在行真正消费前完成。读取的前8行先用 preadv 探针：
慢读继续旧的并发 preadv，快读只有在 scales 驻留 RAM、请求至少512行且
io_workers 大于1时走 mmap。小批请求和 `ssd_scale_in_ram=false` 没有切换到
mmap。

## 正确性和生命周期

`SSDTable.prefetch` 复制输入索引并返回 Future，最多一个 table 的后台任务；
`close` 先拒绝新任务、等待后台任务，再关闭 mmap 和文件。Embedding 把 CPU
读取和 GPU staging 拆成 begin/finish，pending Future 出错时清理状态，下一次
读取可恢复。PP hook 是 ModelState 的默认 no-op；DeepSeek V4.1 只在 SSD
prefetch 开启时生效，且只对非首 PP rank 的真实请求调用。FULL/eager 的原消费
边界负责 join 和 staging，避免覆盖仍在使用的 `staged_rows`。

通过：

* SSD CPU gate 37 passed；其中 batch warm/cold、scales 在 RAM/磁盘、1/8
  workers、重复和 invalid 行都逐字节比较。
* GPU Engram 9 passed；包括 prepared rows 的 FULL graph 刷新和 Future 异常
  恢复。
* 回归集合最终为 172 passed、1 skipped，输出保存在 `regression-tests.log`；
  skip 是需要第二张可见 GPU 的 collective 用例。
* ruff、format、mypy、shellcheck 及其余 pre-commit hooks 通过；第一次
  pre-commit 的 mypy 报告了两个新类型问题，修复后完整 hook 通过。

## 同实例 PP5 A/B

同一个已加载 SSD 实例交替运行 `baseline`（旧读取、无预取）、`batch`（批量
读取）、`both`（批量读取加预取）。固定 32768-token Python 代码 prompt 和
历史 q6 周期 prompt；每个模式两次稳态样本，fresh KV salt，256/512 输出，
并逐请求比较 token IDs 和 DSpark spec counters。两层 Engram 表均为 layer 1
和 14，均在 PP0/PP1；没有重载模型或重建 SSD 表。

| 输入 | 模式 | prefill tok/s | decode tok/s | decode ms/round |
| --- | --- | ---: | ---: | ---: |
| 32K 代码 | baseline | 4727 | 63.47 | 44.64 |
| 32K 代码 | batch | 6193 | 63.62 | 44.53 |
| 32K 代码 | both | 5796 | 65.67 | 43.15 |
| q6 周期 | baseline | 7726 | 135.95 | 42.71 |
| q6 周期 | batch | 7868 | 134.08 | 43.31 |
| q6 周期 | both | 8395 | 139.26 | 41.70 |

每个请求的 token IDs 和 spec counters 都一致。结果支持两个判断：批量读取对
长 prefill 的收益最稳定，decode 变化接近测量噪声；预取在这台 PP5 实例上能
把部分 CPU gather 隐藏在 PP 等待中，周期输入收益约2.4%，代码输入约3.4%，
但它仍只在两个 prompt、每模式两个样本上验证，不能据此把预取改成默认开启。
Profile 记录显示 PP1 确实先执行了 `begin_ssd_lookup`，随后在原 gather/stage
边界完成 join；PP2–PP4 只有 no-op hook，没有误读 Engram 表。

因此本轮不改变默认配置。通过 `run_port_pp5.sh` 新增显式开关：
`PORT_ENGRAM_BATCH_READS=1` 和 `PORT_ENGRAM_PREFETCH=1`。默认仍关闭，便于
现有 SSD 部署回退；batch 路径已经在慢探针时自动回到原并发 preadv。下一轮若
要推广预取，应在至少三个真实 prompt、冷/热页和并发请求下重复同实例 A/B，
并将 PP1 的 wait、gather、stage 分段和端到端 TTFT 一起作为门槛。

服务已停止，正式 `dsv41f-manual.service` 保持 inactive，GPU compute apps
为空。
