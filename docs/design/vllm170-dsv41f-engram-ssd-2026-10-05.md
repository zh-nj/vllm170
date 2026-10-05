# DeepSeek V4.1 Engram SSD storage

日期：2026-10-05。基线：`a33bd2b5cb38590dd04a6228bfadb65a4b4db1d1`。

本文保留功能实现和组件速度优化阶段的记录。后续已完成 CUDA 13 下完整 PP5
RAM / SSD 对照，含 1K / 32K / 64K / 128K 输入、256 输出、冷/热文件缓存和
逐 token 一致性验证，见
[端到端报告](vllm170-dsv41f-engram-ram-ssd-e2e-2026-10-05.md)。

## 使用

新增可选 SSD Engram 后端。未设置 `ssd_offload_path` 时沿用原有 GPU / CPU
驻留路径；SSD 不是默认配置，PP5 launcher 的 DSpark=5、CED 开启默认值保持不变。

直接启动时设置：

```bash
--engram-config '{"cpu_offload":true,"embedding_across_dp":false,"ssd_offload_path":"/media/z/data6/vllm-engram-ssd","cpu_cache_gb":2}'
```

现有 PP5 launcher 的完整用法：

```bash
PORT_ENGRAM_SSD_PATH=/media/z/data6/vllm-engram-ssd \
PORT_ENGRAM_CPU_CACHE_GB=2 \
bash benchmarks/deepseek_v4_1/run_port_pp5.sh
```

以上是使用示例，本轮没有执行完整模型启动。目录在实际构建 Engram 表时创建。
`cpu_cache_gb` 的单位是 GiB，预算作用于**每张本地表**，不是整台机器。
PP5/TP1 中两层 Engram 分别位于 PP0、PP1，各有一张表；设置 2 即合计至多
4 GiB 原始行缓存。设为 0 时不保留应用级行缓存，仍有固定容量的 pinned
CPU / GPU staging。速度优化版默认还将 scales 保存在普通 CPU RAM，两层合计约
5.72 GiB，这部分不计入 `cpu_cache_gb`。负数、NaN、无穷值以及无 SSD 路径的
非零缓存预算被拒绝。

SSD 模式的新选项：

| 配置 | 默认 | 作用 |
| --- | --- | --- |
| `ssd_scale_in_ram` | `true` | scales 驻留普通 CPU RAM，权重仍在 SSD；设为 false 减少内存占用 |
| `ssd_io_workers` | `8` | 每张本地表最多 8 路读取；允许 1–32，1 为串行 |

launcher 对应 `PORT_ENGRAM_SCALE_IN_RAM=1`、`PORT_ENGRAM_IO_WORKERS=8`。
低内存、串行读取回退：

```bash
PORT_ENGRAM_SSD_PATH=/media/z/data6/vllm-engram-ssd \
PORT_ENGRAM_CPU_CACHE_GB=0 \
PORT_ENGRAM_SCALE_IN_RAM=0 \
PORT_ENGRAM_IO_WORKERS=1 \
bash benchmarks/deepseek_v4_1/run_port_pp5.sh
```

## 存储与加载

原先配置已有 `ssd_offload_path`、`cpu_cache_gb` 字段，但当前 Engram 实现没有消费
它们。本轮把配置接入实际表加载与 lookup，保留完整 hash-head 的 TP 分片规则。

表以原始 FP8 E4M3 weight 和每 32 元素的 E8M0 scale 字节存储，两类文件分开。
没有把整表展开为 BF16，也没有分配整表大小的 CPU pinned 内存或 HBM。
加载器仍注册原来的 weight loader，参数改为零行占位；文件存储由独立对象持有。
一般按参数元素数计算的模型权重账本不包含 SSD 文件与 CPU 行缓存。

首次加载以至多 64 MiB 的块流式转换选中的 checkpoint 分片。完成的块执行
`fdatasync`，并建议 OS 回收目标文件 page cache；源 safetensors 映射沿用现有
page-out helper。数据先写临时文件，fsync 后原子替换，最后发布完成 manifest；
转换期间通过文件锁协调同一分片的加载者。

复用同时检查文件长度、格式版本、分片形状和 checkpoint 来源：路径、device、inode、
size、mtime、ctime、tensor 文件偏移。来源必须能识别为连续 CPU safetensors
映射；否则每次重新转换。这要求 checkpoint 在加载期间保持不变，不是逐字节内容
哈希验证。旧的裸 `.u8` 缓存没有来源记录，不直接复用，也没有删除或改写旧目录。
同一个运行时对象不支持原地重新加载权重。

当前 DSV41F/TP1 两张完整表的存储量约 188.84 GiB。首次启动需要相应 SSD
可用空间和转换时间；失效文件重写时旧完整文件与新临时文件会暂时共存。
本轮只写入真实表的小分片，随后删除临时转换文件，没有生成两张完整表。

## 每轮 lookup

1. 在 GPU 上沿用现有 hash 运算，复制本地 head 的索引到 CPU。
2. 去重有效行，按首次出现顺序访问 FIFO cache。缓存命中不改变插入顺序；
   未命中使用 `preadv` 读取 weight，默认从 RAM 读取 scale；关闭 scales 驻留时
   scale 也从 SSD 读取。无效索引和 padded head 返回零。
3. 将请求顺序的 packed bytes 复制到固定容量 pinned staging，再异步 H2D。
4. 使用原有 lookup kernel 解码 FP8/E8M0，输出 staged BF16 rows，继续原有
   WKV / gating 流程。重复请求只读取一次，单批超过缓存容量仍能返回完整结果。

缓存索引按实际缓存容量增长，没有分配与整张表行数等大的索引数组。
staging 容量由 `max_num_batched_tokens × 本地 head 数` 决定。CUDA event
确保下一次 CPU 改写 staging 前，上一次传输和 lookup 已完成。

这里使用普通文件 I/O，未接入 GDS、O_DIRECT 或异步 SSD 预取。OS 仍可能保留
可回收的文件 page cache；`cpu_cache_gb` 只限制 packed row 缓存，不是进程 RSS
上限，也不包含 scales 驻留内存、Python 缓存索引、staging 和 OS page cache。每轮多了索引 D2H、
CPU 随机读取和 packed rows H2D，性能取决于缓存命中率与 SSD；不能据此声称提速。

## Graph、CED 与 DSpark

SSD I/O 在 FULL CUDA Graph 外执行，通过现有
`DeepseekV41ModelState.prepare_full_cudagraph_inputs` hook 提前准备 embedding，
并刷新固定地址的 hash 和 image token mask。graph 内只消费已准备的 rows。
语言模型与视觉 wrapper 均转发 prepared 标志。无 Engram 的 PP rank 不执行
SSD lookup；DSpark draft 模型不增加 Engram 表。

SSD 模式强制 prefill eager，保留 CED 流程；decode 可使用 FULL graph。
没有 SSD 配置时，hash 和 gather 的原有执行位置不变。

| 条件 | 行为 |
| --- | --- |
| CUDA DeepSeek V4.1 + Model Runner V2 | 接入 SSD 后端 |
| FULL / FULL_DECODE_ONLY / NONE | 支持；未指定时选 FULL_DECODE_ONLY |
| CED、DSpark、视觉 wrapper | 接入；组件与原有回归测试通过 |
| TP head 分片 / sequence-parallel 行重排 | 保留；TP=1/2/4/8 的模拟分片组件测试通过 |
| PIECEWISE / FULL_AND_PIECEWISE | 配置阶段拒绝 |
| ubatching / Model Runner V1 | 配置阶段拒绝 |
| 跨 DP 的 Engram 分片，DP>1 | 配置阶段拒绝；各 DP 副本独立存储不受此限制 |
| Qwen PLE | SSD 配置拒绝，不改变原有 PLE 存储实现 |

## 验证记录

在 CMP170HX SM80 上使用 PCI 1 卡隔离运行组件测试。未与正式模型服务并行。
本轮开始和结束检查时 `dsv41f-manual.service` 都为 inactive / MainPID=0；
未启停正式服务。原有 round6–8 工作区修改保留，未提交本轮修改。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
.venv/bin/python -m pytest \
  tests/kernels/test_engram.py tests/config/test_config_utils.py -q

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
.venv/bin/python -m pytest \
  tests/models/test_deepseek_v41_ced.py \
  tests/models/test_deepseek_v4_dspark.py \
  tests/models/test_deepseek_v4_vision.py -q
```

- Engram / config：112 passed，1 skipped。skip 是需要第二张可见 GPU 的 collective
  用例；模拟 TP shard 和行重排用例已运行。
- CED / DSpark / vision：74 passed。它们验证原有流程回归，不是完整 SSD 模型推理。
- 最后简化 SSD lookup 分支、加强 image mask 刷新断言后，SSD / head shards
  子集再次运行：21 passed。FULL Graph 在不同 token 和图像 mask 下复用，
  rows、hash、mask 都更新；FIFO eviction、零缓存、跨 stream staging 均有覆盖。
- 实际 checkpoint 层 1、14 各取 65536 行，dim=256；token 数 1 / 6 / 128 / 2044，
  8 组 SSD lookup 均有限且与 resident GPU lookup 的 BF16 输出逐位相同。
  这是人工单 head 布局的分片验证，不等于完整表路由或模型输出评估。
- 实际 `EngineArgs.create_engine_config()`：PP5 / DSpark5 / CED，默认、FULL、NONE
  三种 graph 配置解析通过，没有构建模型或生成完整 SSD 表。
- launcher 的默认参数、含空格及引号的 SSD 路径 JSON 转义、非法缓存预算检查通过。
- 指定修改文件的 pre-commit、ruff、format、mypy、shellcheck 与 diff check 通过。

保留证据目录：`artifacts/dsv41f-engram-ssd-20261005/`，包括测试日志、
`real-table-check.json`、`configuration-check.json`、`launcher-check.json` 和交付清单。

尚无完整 PP5 SSD 服务启动、greedy token-id 对照、TTFT 或吞吐量数据。
这次交付确认存储、加载、数值与 graph staging 的组件契约；生产效果应另做
同配置的 RAM / SSD 对照，避免把分片验证当作端到端收益。

## SSD 速度优化轮

继续在已有 SSD 后端上优化，磁盘文件格式、来源校验、hash、FP8/E8M0 数值、
TP head 分片、CED 和 FULL Graph prepared-row 契约保持一致。没有引入整表地址
索引；`row_id × width` 已能计算地址。

### 测量与热点

使用 `benchmarks/kernels/deepseek_v41_engram_ssd.py`。取真实层 1 的 1572864 行，
人工划为 24 个 head，每个 65536 行。SSD 临时数据约 396 MiB，测完删除。
测试在 PCI 1 CMP170HX、TP1 上进行，包含 GPU 索引 D2H、CPU gather、SSD 读取、
H2D、原有 dequant kernel 和最后的 device synchronize；不是纯 kernel 时间，
也不是完整模型端到端时间。

基线保存本轮开始时的 `SSDTable` 源码；旧与新 reader 使用相同文件、索引和
GPU lookup wrapper。交替 A/B 顺序，先确认与直接解码真实 checkpoint rows 的
独立参考逐位一致。cold-file 只对该临时文件调用 `POSIX_FADV_DONTNEED`，不清空
整机 page cache；用 `/proc/self/io` 的 `read_bytes` 确认物理读取发生。
warm-file 和应用行缓存命中样本的 read_bytes 都为 0。

确认三个热点：

- 未命中每行原先串行读取 weight / scale，两次独立读取及随机页访问。
- 固定多线程在 cold-file 有收益，但 warm-file 明显回退。2044 token 的 warm-file
  首轮固定 8 路为 673.84 ms，比旧路径 364.70 ms 更慢；开销来自线程调度与 Python
  循环。已经取消固定并行策略。
- 缓存全命中的大批次仍有大量 Python 查询与逐行搬运。cProfile 确认主时间落在
  `gather`，包括 generator / dictionary 查找和 NumPy 行复制，而非 SSD。

### 落地

1. SSD scales 默认驻留普通 CPU RAM；每层约 2.86 GiB，总计约 5.72 GiB。
   按原始 E8M0 字节保存，未展开，不占 pinned 整表空间。可显式关闭。
2. 先从 RAM cache 批量复制已有行，再读取未缓存行；这一批的行数据会保留到返回，
   不因前面的 FIFO 插入而重新读取已在批次开始时缓存的行。
3. 少于 64 个未缓存行串行读取。较大请求先串行读取 8 行，平均读取耗时不足
   20 µs/行则保持串行；存在等待时才用至多 8 路 reader。线程池按需创建并复用。
   所有 future（包括异常情况下）完成后才离开 gather，成功后才更新 FIFO 状态。
4. 全命中请求直接批量 gather；大于等于 512 行且均有效时用 CPU
   `torch.index_select` 写输出，避免逐行循环和中间复制。小批次与无效行保留 NumPy
   路径。零缓存时跳过 FIFO 更新循环。缓存索引仍按缓存容量增长。

上述优化进入所有 SSD Engram 表的共同 lookup，包括 eager prefill、CED、普通
decode、DSpark target 和 FULL Graph 外的输入准备。非 SSD 路径不增加 RAM scales
或 I/O 线程。SSD 模式默认开启新的 reader 行为，整个 SSD 后端仍是 opt-in。

### 同口径结果

无应用行缓存，5 次交替 A/B 中位数（`zero-cache-final.json`）：

| 场景 | token/head | 原路径 ms | 优化后 ms | 时间减少 |
| --- | --- | --- | --- | --- |
| cold-file | 6 / 24 | 13.961 | 3.458 | 75.2% |
| warm-file | 6 / 24 | 1.359 | 0.991 | 27.1% |
| warm-file | 2044 / 24 | 295.075 | 171.643 | 41.8% |

q6 cold-file 的实际 read_bytes 从 1163264 减至 589824，约减半。
有 16 MiB 行缓存，2044-token 全命中请求的单独 7 次交替 A/B 中位数
（`cache-hit-final.json`）：89.090 → 31.241 ms，时间减少 64.9%，两边 read_bytes 均为 0。

另外的 7 次 A/B（`final-ab.json`）覆盖 q1 / q6 cold-file、warm-file、缓存命中，
以及 128-token cold-file。q1 warm-file 的收益仅 1.2%，不作为明显提速证据；
q1 缓存命中同样接近测量噪声。改善主要在冷读和较大批次。

可复跑：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
.venv/bin/python benchmarks/kernels/deepseek_v41_engram_ssd.py \
  --baseline artifacts/dsv41f-engram-ssd-speed-20261005/engram_ssd_before.py \
  --variants before auto_ram --cpu-cache-gb 0 \
  --case cold_file:6 --case warm_file:6 --case warm_file:2044 \
  --repeats 5 \
  --output artifacts/dsv41f-engram-ssd-speed-20261005/recheck.json
```

正确性扩展到 RAM / SSD scales、串行 / 并行读取、跨 stream staging、超过缓存容量
的批次、FIFO 插入导致批内淘汰、重复行、全命中批量拷贝和读取异常不破坏缓存。
新增字段覆盖边界检查；launcher 覆盖默认值与低内存回退。回归与 lint 结果保存在
`artifacts/dsv41f-engram-ssd-speed-20261005/`。

本轮五个既有套件合并运行：201 passed，1 skipped，122.59 s；包括 Engram、
配置、CED、DSpark、视觉回归。skip 仍为需要第二张可见 GPU 的 collective。
指定修改文件的 pre-commit、ruff、format、mypy、shellcheck 与 diff check 通过。

这些数据来自小于完整表的真实分片；完整表的工作集、CPU/SSD 争用以及 PP 调度
可能改变收益。正式服务保持原来的 inactive 状态，未部署或做全模型 token-id / TTFT /
吞吐 A/B，因此没有端到端提速结论。
