# SSD Engram：批量读取与提前预取的可行性

日期：2026-10-05。HEAD `a33bd2b5cb`，保留工作区已有 SSD、prefill 和默认
target prepare-ahead 修改。本轮只增加离线 benchmark 和分析，没有改推理路径、
启动完整模型、提交或部署。证据：`artifacts/dsv41f-engram-ssd-prefetch-20261005/`。

## 判断

有优化价值，优先研究 prefill。现有 reader 已支持最多8路并发，但调用方仍同步
等待全部读取；热文件缓存下，大批逐行 Python / preadv 工作也可以先批量化。
单纯增加线程不能同时解决这两个问题，直接换成串行 mmap 又会严重拖慢冷读取。

本轮在完整表上验证了 CPU 批量取行与同轮异步 CPU gather 的组件收益。
GPU 重叠使用独立合成 GEMM，没有复现真实 PP5 调度，不能把这些百分比当成
prefill TTFT 或 decode TPS 提升。

## 源码确认的重叠窗口

当前 FULL decode 的顺序为：

`attention metadata → model_inputs / lookback → PP receive + wait + copy →
prepare_full_cudagraph_inputs → hash / D2H / SSD gather / H2D / dequant → FULL replay`。

位置见 `model_runner.py:1926`、`1955`、`2009`；具体 hook 位于
`model_state.py:261` 和 `model.py:1105`。target prepare-ahead 已默认开启，
但 SSD gather 仍在 PP 接收完成之后，尚未覆盖这个独立准备环节。

- Engram 的层号是1、14；PP5的8/8/8/8/8分区对应 PP0、PP1。
- hash 需要 token IDs、positions、query boundaries、image/dead mask 和 lookback，
  不需要当前 hidden states。lookback 在 `model_state.prepare_inputs()` 生成。
- V2 的 `NgramHashState.use_slot_cache=False`，没有 V1 的滚动 slot-cache 写入。
  SSD 配置已要求 V2，因此同轮 hash 可在 PP 数据到达前准备。
- eager/CED prefill 在模型 forward 开头准备所有本地 Engram 行；PP1此时也已经
  接收了上游数据。真正注入在 layer14，之前还有本地 layer8–13。
- 当前先 gather 再进入 decoder loop，尚未利用这些前置层的计算窗口。

最小可研究改动是：在当前 batch 的 token IDs、metadata、lookback 就绪后生成
索引，提交 CPU gather；继续 PP receive/wait，在已有 GPU 行消费边界 join。
PP1可以覆盖部分上游 PP0 的计算/等待，PP0没有上游等待窗口。
这一点由源码依赖支持，没有做真实 PP overlap 测量。

prefill 的后续 prompt 已知，还可以有界预取下一 chunk；但需要绑定实际请求、
位置、lookback、head 分片和 chunk generation。decode 的下一轮输入取决于
采样、接受结果及 draft 交接，不能无条件跨轮预取未知 token。

## 完整表的有界组件实验

单独使用物理 PCI1 CMP170HX SM80，UUID
`GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216`。CUDA_HOME / Triton ptxas为
`/usr/local/cuda-13.3`，Torch `2.13.0+cu130`，没有其他 GPU compute apps。

读取现有 layer1完整表：384006168行、256-byte FP8 weight/行、8-byte scale/行，
weight文件约91.55GiB；scales复制到普通 RAM，约2.86GiB。文件只读打开，
没有转换或复制 weight 全表。使用实际24个 prime-sized head范围中的固定随机
索引，含一个无效行；不是模型实际代码语料的 hash trace。应用行缓存为0。

每场景5次，reader顺序交替。热轮先预热请求行，进程 read_bytes全为0。
冷轮只对请求对应的4KiB页调用 DONTNEED；mmap先移除本进程 PTE。
所有reader均使用生产的 POSIX_FADV_RANDOM，mmap另加 MADV_RANDOM。
冷轮各reader的物理读取完全一致：q6为585728bytes，2044为200716288bytes。
这说明冷对照包含真实 I/O，而非只换了系统缓存状态。

CPU gather包括去重、索引、原始行读取和请求顺序重排，不包含索引D2H、
pinned staging、H2D和GPU dequant。

| tokens × heads | 缓存 | 现有自适应 preadv ms | 批量 mmap 串行 ms | 批量 mmap 8路 ms |
| --- | --- | ---: | ---: | ---: |
| 6 × 24 | 热 | 0.527 | 0.152 | 0.575 |
| 6 × 24 | 冷 | 3.879 | 5.701 | 1.930 |
| 2044 × 24 | 热 | 197.611 | 44.223 | 33.422 |
| 2044 × 24 | 冷 | 778.649 | 4313.223 | 689.015 |

大批热行的取数从逐行 Python/syscall 路径转到 NumPy 批量 gather，串行降低
77.6%，8路降低83.1%。小批热行再加8线程反而增加9.2%，不能固定全场景8路。
大批冷行的串行 mmap为基线5.54倍，8路 mmap只改善11.5%；收益不能用热缓存
结果外推。没有采用任何 mmap 候选到推理实现。

## 异步读取与 GPU 工作重叠

固定相同 reader 和输入，比较 `CPU gather → GPU work → staging / dequant`
与 `提交 CPU gather → GPU work → join → staging / dequant`。
后台任务只执行 CPU gather，不在后台操作 CUDA。CPU返回自有 packed buffer，
GPU拷贝与 dequant仍在主线程/主stream执行，沿用 staging Event保护。

独立 GPU 工作为 BF16 1536×1536 GEMM CUDA Graph。q6约6ms，大批约100ms，
这是合成可覆盖预算，不是实测上游 PP 时长；GPU自动升频使标定和后续稳态
工作时间有差别。两个模式使用同一 graph、矩阵和调用数，交替5次。
组合耗时包含索引D2H、CPU gather、GPU工作、pinned staging、H2D、dequant和
末尾 synchronize；初始化、编译、热缓存预热与正确性检查不计时。

| tokens | reader | 串行组合 ms | 预取组合 ms | 组合耗时减少 |
| --- | --- | ---: | ---: | ---: |
| 6 | 现有 preadv | 6.352 | 5.831 | 8.2% |
| 6 | 批量 mmap 串行 | 5.971 | 5.833 | 2.3% |
| 2044 | 现有 preadv | 271.998 | 230.669 | 15.2% |
| 2044 | 批量 mmap 串行 | 128.798 | 87.079 | 32.4% |

全部20组配对同向，但这只证明在该组件中可以隐藏部分读取。q6批量 reader
预取只多节省约0.14ms，大批节省约41.72ms，prefill更值得优先研究。

后台读取也有额外成本：2044的现有 reader自身耗时中位数由186.94增加到
218.05ms，批量 reader由43.20增加到52.58ms。异步化没有消除CPU工作，不能
假设完整隐藏标定的 GPU 预算。已记录 graph host submission、join wait、
stage host和read耗时；这些计时不足以唯一归因于线程迁移、GIL或内存争用。

## 集成约束与下一步

1. 先保留 cold/warm自适应：大批热缓存走批量 gather，小批热缓存避免固定8线程；
   冷缓存保留有效的多路读取。不能用串行 mmap替换所有场景。
2. 将 SSD接口拆为 begin / finish，最多预取一个 chunk，CPU packed结果独立持有；
   缓存命中、淘汰、重复行、读取错误和取消均保留现有契约。
3. 同轮预取先放在 PP1接收前，与默认 target prepare-ahead组合验证；prefill再
   研究下一 chunk。没有真实预算时，不用代理 GPU 时长预测 TPS。
4. 提前读取CPU原始行可以与 GPU共存，但提前改写GPU `staged_rows`需要额外保护。
   现有 `_ssd_copy_done`记录在 lookup后，只证明lookup完成，不证明FULL graph
   已完成行消费。先在既有消费边界完成 GPU staging；需要更早GPU准备时，再加
   双缓冲及消费完成事件，不能仅复用当前单 Event。
5. 端到端验收需要一次 SSD加载内，同实例交替对照，固定 CUDA13、DSpark5、CED、
   target prepare-ahead开启、prompt / output预算、缓存和其他开关；分别检查周期
   输入与代码输入的IDs / spec counters、冷/热 prefill和decode速度。当前没有这项
   验收，不能宣称部署收益。

组件使用独立 `os.pread`产生完整请求的FP8/E8M0 byte参考，再独立解码BF16；
所有reader raw rows、40次GPU组合输出和独立GEMM结果均精确一致。
仅覆盖单表TP1；没有并发请求、跨TP、prefill chunk切换或取消的集成验收。

初次脚本未解析视觉模型的 text_config，启动即退出；失败日志保留。之后的pilot
遗漏生产RANDOM提示，冷mmap曾发生大量readahead，完整保留但排除于上表。
正式screen已修正读取策略，物理读取量逐arm核对一致。运行源码冻结为
`screen-source.py`且SHA匹配结果；随后将已等待的loop closure改为显式传入
table/indices，并按项目hook要求改用 accelerator.synchronize，完成兼容性检查。
读取、GPU和计时边界未改。推理源码SHA均与开始时一致。

复现（输出路径必须不存在；不应与模型服务并行）：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
CUDA_HOME=/usr/local/cuda-13.3 \
TRITON_PTXAS_PATH=/usr/local/cuda-13.3/bin/ptxas \
.venv/bin/python -m benchmarks.kernels.deepseek_v41_engram_ssd_prefetch \
  --directory /media/z/data6/vllm-engram-ssd \
  --model /media/z/data6/models/DeepSeek-V4.1-Flash \
  --repeats 5 --output /tmp/ssd-prefetch-screen-new.json
```

原始数据：`screen.json`；派生数字和来源：`analysis.json`；日志、执行源码及
运行前后清单保存在同一artifact目录。结束时 GPU进程释放，正式服务保持inactive。
