# DSV41F PP5 并发退化排查

日期：2026-10-05。目标是解释当前 PP5/DSpark5 服务在多请求下吞吐不随并发增长的原因。
第一轮做配置核对和在线测量；第二轮纠正测量口径，用固定请求集合做准入节拍对照，
随后取得用户许可，保存原服务 argv/env，在 GPU2–6 启动仅绑定 loopback 的诊断实例。
本轮没有修改模型推理源码。诊断结果和服务恢复状态见下文。

## 现网配置核对

26661 的实际父进程是 1680143，物理 GPU 映射为 GPU2→PP0、GPU3→PP1、GPU4→PP2、
GPU5→PP3、GPU6→PP4，health=200。实际 `/proc/1680143/cmdline` 与用户给出的文本
不一致：

* `--engram-config` 没有进入进程 argv。模型的 Engram 默认仍是 `cpu_offload=true`，
  但这个隐式行为不能作为部署契约。
* 进程 argv 只有第一个 `--additional-config`，实际只有
  `ced_prefill=true,dsv41_sm80_moe_tilelang=true`。
  `dsv41_sm80_fused_swa_metadata`、`dsv41_sm80_moe_hybrid_prefill` 和
  `dsv41_sm80_moe_small_tail_prefill` 都没有打开。
* `dsv41_sm80_target_prepare_ahead` 没有出现在 argv；当前源码默认值是 true，
  但启动命令没有把这个关键性能开关固定下来。

用户命令中续行后的空行使后面的 `--engram-config` 和第二个 `--additional-config`
没有进入同一条 shell 命令。这与参数解析器对重复 JSON 参数的处理是两回事；不能根据
当前 argv 推断重复参数会合并。生产命令应只传一个完整 JSON 对象。

主机总内存约252 GiB，可用约24 GiB，swap 2 GiB 几乎耗尽，整机 memory PSI 约20%，
PP0/PP1 各约97 GiB RSS。但第二轮实测服务 cgroup 的 some/full PSI 均为0，累计等待
计数在四个对照臂内完全不变；EngineCore 和五个 worker 的 major fault 增量均为0，
PP0/PP1 的进程 swap 为0。**撤回第一版“整机内存压力放大当前服务退化”的归因**：
整机指标不能证明这些请求在等待内存，也不能据此推荐切换 SSD 来解决并发问题。

## 在线并发证据

以下为第一轮历史观察，不是第二轮的受控对照。旧探针对不同并发波次改变 seed，
且丢弃带标签的服务计数器；因此“同一请求集合、接受计数严格成倍”的说法撤回。
端到端总输出/墙钟的数字仍有效，旧合并 decode 口径不再使用。

| 并发 | 墙钟 s | 合并输出 tok/s | 各请求 post-first-chunk tok/s |
| ---: | ---: | ---: | --- |
| 1 | 2.270 | 112.8 | 145.6 |
| 2 | 3.553 | 144.1 | 82.4 / 88.9 |
| 4 | 6.593 | 155.3 | 49.1 / 42.7 / 48.1 / 50.8 |

四路合并墙钟吞吐只达到单路的1.38倍，而每路 post-first-chunk 吞吐降到约单路的
三分之一。32K/512
四路复测墙钟11.629 s，合并176.1 tok/s，各路 post-first-chunk 47.3–50.4 tok/s。

短请求的 1K/128 八路复测也没有出现线性扩展：墙钟9.379 s、合并输出109.2 tok/s，
首请求 TTFT 0.713 s，其余请求约2.90 s。这说明该波次的输出延迟不均衡，不能仅凭客户端
时间判定内部 batch、GPU 饱和位置或所有请求的 PP 相位。

GPU 采样显示四路期间 GPU2（PP0）平均利用率94.9%，GPU3–6 为73.3–79.3%；单路
GPU2 约81%。**撤回由此直接判定 PP0 计算饱和的结论**：NCCL 等待也会计入 GPU 利用率。
源码中的 M=6B 指的是某个纯 decode 调度批次里的 B 个请求，不是全局并发请求数。
异步 PP 对每个请求设置 `next_decode_eligible_step=current_step+pp_size`，不同准入时刻
可以形成不同相位，4路并发不保证每步 M24。只有实际批次记录才能判定 MoE dispatch。
现有 GPU1 M2/M4/M8 隔离测试也不是 DSpark 的 M12/M24/M48，不能据此排除这些形状
的优化价值。

此前同口径
SSD 单请求结果相对主存为 prefill -1.3%/-9.2%/-5.0%/-6.1%（1K/32K/64K/128K），
decode -3.5%/-1.9%/-2.1%/-1.7%。SSD 是否改善并发需要重启后的 c1/c4 A/B，不能用
单请求数字替代。

## 启动路径的已有修改与边界

`benchmarks/deepseek_v4_1/run_port_pp5.sh` 的已有修改输出一个 additional-config 对象，
默认补齐 fused metadata、hybrid/small-tail 和 target prepare。`moe_tilelang=true`
仍需调用方提供；当前实现对重复 additional-config 对象保留最后一个，并非逐字段合并。
下面是期望的完整配置示例，不能作为 launcher 自动生成全部字段的证据：

```json
{
  "ced_prefill": true,
  "dsv41_sm80_moe_tilelang": true,
  "dsv41_sm80_fused_swa_metadata": true,
  "dsv41_sm80_moe_hybrid_prefill": true,
  "dsv41_sm80_moe_small_tail_prefill": true,
  "dsv41_sm80_target_prepare_ahead": true
}
```

每项都可用 `PORT_FUSED_SWA_METADATA=0`、`PORT_MOE_HYBRID_PREFILL=0` 或
`PORT_MOE_SMALL_TAIL_PREFILL=0` 回退；调用方显式传入同名 JSON 字段时优先保留调用方值。
`--engram-config` 也由 launcher 始终生成，避免依赖模型默认值。

此前同进程验收显示 fused SWA metadata 相对 token-map 基线约省6.17%/轮；target
prepare-ahead 已在单路和双路完整 decode 中取得约45.7%和36.1%收益。当前服务虽然
依赖源码默认打开后者，但缺少前者，且没有把配置写入 argv；下一次重启必须使用合并后的
单 JSON 命令重新验收。

## 第二轮：同配置准入节拍 ABBA

测量脚本 `benchmarks/deepseek_v4_1/concurrency_probe.py` 已修复：保留带标签的指标，
使用所有请求的 `min(first_token)` 到 `max(last_token)` 作为合并 decode 窗口，分子扣除
每个请求的完整首个输出 chunk；各波次复用相同 seed 和 prompt IDs。记录原始输出 IDs，
验证 usage、结束标志和实际 token 数。客户端 chunk 数不称为 draft 轮数。
两项回归测试加在已有 `tests/benchmarks/test_dsv41_dspark_flow_gate.py`，共8项通过。

在原26661服务上用固定32K/512、4路、greedy、共享已热 prefix，做0/100/100/0 ms对照。
100ms表示相邻请求的到达间隔，最后一个请求延后300ms，端到端墙钟包含这段延迟。

| 到达间隔 ms | 端到端墙钟 s | 端到端 tok/s | 合并 decode tok/s | 输出迭代记录 | 请求 draft 轮数合计 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 11.306 | 181.14 | 191.28 | 268 | 353 |
| 100 | 8.856 | 231.26 | 242.73 | 179 | 353 |
| 100 | 8.839 | 231.70 | 242.89 | 179 | 353 |
| 0 | 11.143 | 183.80 | 191.43 | 268 | 353 |

中位数端到端吞吐提高26.86%，合并 decode 提高26.89%。16个完成请求的512个输出 IDs
逐项相同；每个对照臂 prefix hits=130048，preemption=0，draft rounds=353。
accepted draft tokens 为1691/1695/1695/1691，相差4个 token，无法解释约27%的吞吐差。
服务 cgroup PSI 不变，所有相关进程 major fault 增量为0。

`iteration_tokens_total_count` 来自输出记录，histogram 观察 computed prompt tokens
加 emitted tokens（`vllm/v1/metrics/loggers.py`），**不是 GPU 实际 query token 数**。
因此268→179只能证明输出迭代组织发生变化，实际 batch/cohort 由诊断实例直接捕获。
这是一项客户端到达方式实验，尚不是生产调度优化；不能建议所有生产请求固定 sleep100ms。
工作负载是一个周期性32K prompt、热prefix、4路greedy，不代表非共享长prompt或生产采样。

证据：`artifacts/dsv41f-concurrency-investigation-20261005/round2/arrival-abba.json`，
`arrival-summary.json`。原服务完整参数和源文件哈希保存于 `service-before.json`；完整环境
保存在600权限的私有文件，不纳入提交。

## 第二轮：诊断实例复现与分解

取得用户对一次诊断重启及恢复的明确许可后，原服务按保存的 PID/start-time 身份退出。
诊断实例使用相同 argv/env，只修改 host/port 为127.0.0.1:26662，增加 worker extension
及 dev-mode。源码哈希不变，五个 worker 仍使用物理GPU2–6，CPU affinity仍为
14–27,42–55。RPC 直接确认所有 rank 的 target prepare-ahead=true、token-map=true、
fused SWA metadata=false。dev-mode 不绑定外部地址。

先做无插桩 ABBA，得到合并 decode：191.72/240.25/241.07/192.86 tok/s。
同时到达中位192.29，100ms间隔中位240.66，提高25.15%；输出迭代计数仍为
268/179/179/268，draft rounds仍为353。说明差异在重启后同配置中复现。

随后使用每步CUDA Event、CPU monotonic 时间和 scheduler step ID，记录空调度和实际
InputBatch；所有事件在完成请求后统一解析，计时期间不逐步同步。

| 插桩工作负载 | 实际纯decode批次 | 纯decode执行数 | 空调度数 | 合并decode tok/s |
| --- | --- | ---: | ---: | ---: |
| 单路 | M6/B1，FULL graph，补齐后仍M6 | 89 | 357 | 144.30 |
| 四路同时到达 | 四个独立M6/B1 cohort | 356 | 90 | 166.20 |
| 四路100ms间隔 | M6/B1 90次，M18/B3 89次，FULL graph，补齐后仍M6/M18 | 179 | 277 | 239.86 |

**插桩对准入有扰动**：同时到达的无插桩输出记录为268，插桩变为360（356个decode，
4个prefill），吞吐降低约14%；不能把四个单请求cohort当成原服务无插桩批次的直接观测。
100ms间隔仍稳定为B1+B3，输出记录181（含2个prefill），吞吐与无插桩240.66相差约0.3%。
因此 B1+B3 的分解可以用于解释此诊断运行；M12或M24作为原服务同时到达形状的说法
仍未直接证实。空调度数增加而运行更快，说明删除空step不是根据本证据得到的优化方向。

单路 target event span 的五个rank中位分别为5.65/5.63/6.40/6.47/7.01ms；
PP0 period约40.86ms。四个M6 cohort时，PP1–PP4仍约5.67/6.41/6.47/7.01ms，
但PP0的“forward”变为23.03ms。CPU时间可以进一步分开：

| PP0 CPU区间中位 ms | 单路M6 | 四个M6 cohort | B1+B3中的M6 | B1+B3中的M18 |
| --- | ---: | ---: | ---: | ---: |
| forward_start至model_runner.execute_model返回 | 0.084 | 0.150 | 0.098 | 0.101 |
| 返回至sample_tokens内forward_end | 0.583 | 22.886 | 0.638 | 0.667 |

这把主要等待定位到**模型提交之后、target计时闭合之前**。源码中该窗口包含
`gpu_worker.Worker.execute_model` 的 `isend_tensor_dict` 及随后 `sample_tokens` 的调度
和进入，不能把22.9ms都算成模型计算，也尚不能把它全部归给某一个NCCL调用。
非首rank的GPU inputs时间包含上游等待；draft事件包含分布式链上的等待、相互重叠，
不能把五rank的inputs或draft中位直接相加当整轮成本。

B1+B3中，PP0–PP4的M18 target span约为10.38/10.18/11.40/12.78/13.29ms，
比执行三次独立M6便宜。即使M18不能命中当前MoE M6专用dispatch，减少target/draft
调用次数的收益仍占优。这将下一项研究从“多请求必然退回慢GEMM”转向“准入形成的cohort
数量、每cohort的固定成本，以及PP发送阶段的等待”。

### 同一加载实例的fused metadata ABBA

通过已有诊断RPC切换graph外metadata，检查持久buffer地址不变，先分别warmup：

| metadata模式 | 合并decode tok/s | 端到端tok/s | 输出迭代记录 |
| --- | ---: | ---: | ---: |
| token-map基线 | 192.23 | 184.40 | 268 |
| fused SWA | 194.06 | 185.75 | 268 |
| fused SWA | 196.24 | 188.05 | 268 |
| token-map基线 | 192.33 | 184.53 | 268 |

中位decode提高约1.5%，端到端约1.3%。两次配对不足以证明普遍稳定收益；至少说明该
缺失开关不能解释本工作负载约25–27%的差距。fused插桩运行仍是四个M6 cohort，吞吐
161.77，反而低于baseline插桩166.20。PP0 metadata host中位4.10→3.37ms，但后提交
等待22.89→24.72ms，显示局部metadata节省并不保证被整轮保留；这些插桩值不用于
替代上表的无插桩端到端A/B。

所有12个正式诊断对照臂（准入4、metadata4、插桩4）的输出IDs均与单路参考逐项相同，
所有preemption增量为0。证据：`round2/diagnosis.json`，其中保留各rank的原始事件、
scheduler step、request IDs及batch形状。`diagnose.py`和worker extension只在artifact中，
没有接入生产模型路径。

## 后续优化的具体顺序

1. 将shape/cohort捕获与CUDA事件分离：先用低开销的CPU调度记录确认同时到达的实际
   批次，避免计时改变准入。记录prefill完成时刻与每请求eligible step，而不是从并发数
   推算M。
2. 对已定位的PP0后提交窗口，分别计时 `_reap_completed_isends`、`isend_object`、
   tensor `isend` 和worker RPC间隙，区分宿主等待、发送反压和peer的draft依赖。
3. 根据结果选择准入合批、减少每cohort的固定成本或发送路径优化。任何PP环形队列变更
   必须保留空step推进和 `T+pp_size` 的消费语义；不能在worker内部把一个M24调度批次
   私自切为四个M6调用。
4. 使用非周期性代码prompt、非共享prefix和生产temperature/top_p复核，再决定默认。
   不把固定100ms客户端延迟写进生产；不把本轮热prefix结果推广为冷32K长prefill结论。
   内存/SSD并发比较排在这些已观测等待之后，整机PSI不作为切换依据。

本轮没有修改生产调度、MoE dispatch或Engram模式，未提交旧工作区的其它修改。

## 服务恢复与交付验证

诊断实例已停止，26662无listener。原26661服务改由用户级 `dsv41f-manual.service`
托管，ActiveState=active、SubState=running，MainPID=2365143；原argv和原环境的每个
键值均逐项核对一致，没有保留worker extension或dev-mode。五个worker为2367806–
2367810，PP0–PP4的GPU UUID与原服务完全一致。health=200，开发RPC返回404。
475 GiB权重的两次加载使维护时间超出最初10–15分钟估计，恢复后再完成实际推理核对。

恢复后的32K/512 greedy请求完整输出512个token，与诊断前参考IDs逐项一致，
合并decode为142.02 tok/s；该次包含冷prefix prefill，端到端8.05s不能与热prefix的
四路端到端值比较。`restored-state.json`记录完整验收结果，`restored-smoke.json`
保留原始请求输出、usage、计数器与客户端时间。

验证命令：

```bash
.venv/bin/python -m pytest tests/benchmarks/test_dsv41_dspark_flow_gate.py -q
.venv/bin/pre-commit run --files \
  benchmarks/deepseek_v4_1/concurrency_probe.py \
  tests/benchmarks/test_dsv41_dspark_flow_gate.py \
  docs/design/vllm170-dsv41f-concurrency-investigation-2026-10-05.md
git diff --check
```

结果：8 passed；本轮文件的pre-commit全部通过；diff检查通过。完整诊断数据和恢复
manifest在 `artifacts/dsv41f-concurrency-investigation-20261005/round2/`，不纳入源码提交。

## 第三轮：发送回收阻塞与两个候选的否决

沿用 GPU2–6、PP5/TP1、DSpark5、target prepare-ahead=true 的原服务配置，
保存完整 argv/env 和源码哈希后启动 loopback 诊断实例。本轮没有改模型或通信推理源码；
仅在诊断 worker extension 中包裹 CPU 调用，关闭原有逐步 CUDA Event 记录。
并发探针补传请求 `seed`，此前虽记录 seed，却没有传给服务。此次全部请求为 greedy，
输出仍与第二轮参考一致。原始数据保存在同目录的 `round3/`。

### 主要等待位于发送回收，而非新 tensor send 提交

第一组 CPU 记录直接捕获四个独立 M6/B1 cohort，每个89次 decode，总计356次。
PP0 的调用中位数如下；这是 CPU 调用墙钟，不能与上一轮 CUDA Event 区间直接相加。

| PP0 调用 | 中位 ms | 记录数 |
| --- | ---: | ---: |
| `GroupCoordinator._reap_completed_isends` | 22.182 | 356 |
| `isend_object` metadata 提交 | 0.333 | 356 |
| 单个 CUDA tensor `torch.distributed.isend` 提交 | 0.198 | 712 |
| 整个 `isend_tensor_dict` | 23.090 | 356 |

另一组直接捕获 M6/B1 179次、M12/B2 89次，回收中位分别29.833/20.421ms。
100ms到达间隔形成 B1+B3 时，同一回收函数仅0.043/0.051ms，合并decode为237.47 tok/s。
因此第二轮发现的约22.9ms窗口，现在已进一步定位到旧发送条目的回收。
它随实际流水线节拍变化，并非每次发送固有的固定开销。

CPU 记录仍会扰动 cohort：首个诊断加载中，关闭记录的四路波次也出现166–234 tok/s范围；
第二次加载的有记录基线捕获 B1+B3，而延后回收有记录波次捕获 B1+B1+B2。
不能把不同记录方式、不同 cohort 的吞吐直接相减作为优化收益。

### 回收中的慢项是 metadata handle 的阻塞等待

原回收逻辑先检查所有 tensor handle，完成后调用 metadata `_RetainedHandle.wait()`，
再释放队列条目。源码假设：接收方已经读取metadata才会发起tensor接收，故此wait近乎瞬时。
细分记录说明这个性能假设不成立：

* CUDA tensor handle 完成状态查询中位约0.003–0.008ms；已记录最大值低于0.1ms。
* 基线 B1+B3 波次中，metadata wait 捕获10.81/15.54ms长尾。
* 延后回收的 B1+B1+B2 波次中，metadata wait 最长48.51/29.48ms；
  两种形状共等待1106.60ms，占回收总墙钟1118.10ms的98.97%。
  该比例属于延后回收的这个记录波次，不能直接当原22.182ms基线的逐项占比。
* 最后一次基线波次同时记录线程CPU时间：14.084ms墙钟仅消耗0.060ms线程CPU，
  12.310ms墙钟仅消耗0.038ms线程CPU。主要是阻塞或线程未运行，不是CPU忙算。
* 在增加metadata完成查询的基线记录里，查询均为false，包括wait随后立即返回的条目。
  本实现不能据此获得可用的非阻塞完成信号。

这证明发送回收确实会在Gloo metadata wait上阻塞；尚未区分Gloo进度、传输确认与
操作系统调度分别贡献多少。CUDA tensor完成不等于宿主metadata wait已经无需等待。
本轮没有取得无记录运行中每次回收的耗时，故不宣称原生产每个慢step均等待同样的22ms。

### 候选一：批量 CUDA sends，真实 PP 路径挂起，否决

诊断候选保持metadata协议，使用 `batch_isend_irecv` 批量提交target CUDA sends，
接收侧保持原 `irecv_tensor_dict`。第一次候选warmup没有完成，客户端读取超时，
随后出现sample RPC超时和EngineCore退出。已停止整个诊断cgroup，确认五张GPU清空，
重新加载原发送实现。具体死锁机制未验证，不能仅凭批量发送API推断其与当前PP协议兼容。
候选只存在于忽略的诊断文件，未改推理源码，也没有生产开启。

### 候选二：每8次decode send回收一次，无稳定收益，否决

仅在FULL graph、纯decode的target发送中延后回收；保留所有handle和source tensor引用，
队列达到16项或进入prefill时调用原回收。关闭细分记录做同进程ABBA：

| 模式 | 合并decode tok/s | 端到端tok/s | 输出迭代记录 | 请求draft轮数合计 | 接受draft token |
| --- | ---: | ---: | ---: | ---: | ---: |
| 基线 | 159.01 | 153.49 | 356 | 352 | 1692 |
| 延后回收 | 168.58 | 162.70 | 356 | 352 | 1692 |
| 延后回收 | 191.43 | 183.80 | 268 | 353 | 1691 |
| 基线 | 195.33 | 187.50 | 268 | 353 | 1691 |

每臂均为32K/512、四路greedy、热共享prefix，prefix hits=130048，preemption=0。
“输出迭代记录”仍是服务输出指标，不等于直接记录的GPU batch形状。
中位decode仅177.17→180.00 tok/s（+1.60%），端到端170.50→173.25（+1.61%）。
两个迭代记录数匹配的对照分别约+6.0%和-2.0%，方向不一致；不能取最好的一臂宣称收益。
单路对照146.45→146.55 tok/s，100ms到达间隔对照242.77→243.70 tok/s，基本持平。

延后回收把多数调用降到约0.006ms，但集中回收时仍会阻塞30–49ms，整个记录波次共等
超过1秒，说明减少调用次数没有消除等待。这个候选不接入生产。
分离draft NCCL communicator完成了创建和warmup，但未做吞吐对照；不宣称收益或退化。

### 收敛后的下一项研究与验收边界

值得研究的是让metadata完成等待离开target提交线程，或在明确签名下复用metadata协议。
若使用后台回收，必须保留引用直到真正完成，并解决 `_RetainedHandle.wait()` 的并发单次
调用保证；现有 `_waited` 布尔值不提供线程互斥。也不能直接删掉metadata wait或根据
Gloo `is_completed()` 释放buffer。只有无记录端到端收益、完整输出IDs一致、单路及大batch
控制均通过，才有理由向相似发送路径推广。简单增加reap间隔和只合并CUDA发送已被排除。

### 验证与恢复

两次诊断加载的84个成功请求均完整输出512个token，IDs逐项等于第二轮参考，
SHA256为 `b7177bc77e14fdd77c2f9201cd9d01a508f90b5789e4aec11062c1e69869fa1c`；
失败候选的未完成请求不计入。全部成功波次preemption=0。
并发探针8项测试通过；compact relay/prepare-ahead相关CPU测试15项通过。
`tests/distributed/test_comm_ops.py` 的选择性测试因环境缺少可选依赖ray而收集失败，
不能计为通过。探针及已有相关测试的pre-commit通过。

恢复记录与最终实际推理验收见 `round3/restored-state.json` 和 `round3/restored-smoke.json`。
26661原服务已恢复，父进程2636079、PP0–PP4 worker为2638739–2638743；argv/env、
源码哈希及逐rank GPU UUID均与停服前一致，health=200，开发RPC返回404，26662无监听。
恢复后的32K/512 greedy请求输出IDs逐项等于参考，decode为144.10 tok/s；包含冷prefix
prefill的端到端为7.98s，不能与热prefix四路波次直接比较。未部署任何诊断候选。
由于批量发送候选挂起后额外加载一次，本轮维护时间超过最初估计；没有把失败候选留在服务中。

## 第四轮：后台metadata回收触发输出错位，否决

本轮验证第三轮收敛出的后台回收方向。仍使用原GPU2–6、PP5/TP1、DSpark5配置，
target prepare-ahead保持开启。原父进程2636079的argv/env和9个运行源码哈希单独保存。
诊断实现全部位于忽略的 `artifacts/dsv41f-concurrency-investigation-20261005/round4/`，
没有修改推理源码。诊断只绑定127.0.0.1:26662，结束后恢复26661原服务。

### 实现与停服前验证

后台线程仅调用Gloo metadata handle的wait；CUDA handle完成查询、pending FIFO操作及
source tensor引用释放仍在worker主线程。已完成CUDA发送的条目移入retirement队列，
metadata future真正完成后才释放，队列上限16，达到上限时对最老future施加背压。
metadata handle用锁保证底层wait仅调用一次；异常记录后传回前台，失败条目不提前释放。
切回基线前先同步GPU、关闭后台线程并清空旧发送条目，避免跨模式遗留未完成任务。

停服前完成：

* 原型7个CPU用例：非阻塞poll、source引用保留、CUDA未完成时不退休、双等待者、
  异常不重试及向前台传递、metadata-only引用保留、队列背压上限。
* 真实Gloo双进程64条16KiB级payload：顺序及内容完全一致，max_retired=16，
  背压46次。这个实验没有CUDA，只证明Gloo后台等待和CPU生命周期可行。
* 现有 `test_comm_ops.py` 选择性CPU测试5项通过，25项未选；
  compact relay/prepare-ahead相关CPU测试15项通过。

此前无法收集通信测试的依赖缺口已解决：使用uv为项目虚拟环境单独安装
ray 2.56.0和msgpack 1.2.3，不更新其他依赖。单节点5卡足够PP5，源码仍选择mp backend，
诊断日志也确认使用五个multiproc worker，未切换Ray执行器。

### 完成A/B时触发输出验收失败，未完成ABBA

固定32K prompt、512输出、greedy、热共享prefix，关闭细粒度记录。
每臂保留完整输出IDs，并与单路基线逐项比较。计划ABBA，首个候选B臂触发失败即停止，
没有继续采集后两个对照臂，不能从本轮计算稳定中位收益。

| 波次 | 模式 | 合并decode tok/s | 输出迭代记录 | 接受draft token | 输出IDs |
| --- | --- | ---: | ---: | ---: | --- |
| 单路warmup | 基线 | 140.17 | 89 | 425 | 参考 |
| 四路warmup | 基线 | 237.53 | 179 | 1695 | 全部一致 |
| 单路候选warmup | 后台回收 | 144.87 | 89 | 423 | 一致 |
| 四路候选warmup | 后台回收 | 189.27 | 268 | 1691 | 全部一致 |
| A | 基线 | 191.71 | 268 | 1691 | 全部一致 |
| B | 后台回收 | 184.25 | 274 | 1685 | 1个请求错位 |
| 切回后四路 | 基线 | 165.06 | 356 | 1692 | 全部一致 |

唯一完成的A/B中，候选decode下降3.89%；warmup的批次组织不同，不作为收益对照。
表中输出迭代记录是服务输出指标，未直接捕获这几个无记录波次的GPU batch形状。
全部波次均完整返回指定token数。错误不等于HTTP异常、丢SSE chunk或输出长度不足。
后台队列在实际推理中最多3项，没有触发16项背压。

错误请求是B臂的request index 1：首token与参考相同，从index 1开始错位；
候选index 1–505逐项等于参考index 7–511，即连续505个位置后移6个token。
例如参考开头为 `[1487,23648,10814,6235,4854,12093,23635,9264,1202,...]`，
候选为 `[1487,9264,1202,...]`。511个同位置IDs不同；该请求仍返回512个token，
而前几个SSE chunk大小为1、1、6、3。不能以“模型数值噪声”解释或放行这个整段错位。

诊断内共完成22个请求，其中21个与参考一致、1个不一致。单路和首个四路候选通过，
不能替代多波次正确性验收。候选立即切回基线，后续四路的4个请求再次全部一致。
拒绝原型并保留原始错误输出，不部署、不向其他路径推广。

### 源码收敛：引用存活与缓冲区复用是两件事

`gpu/cudagraph_utils.py:625-649` 中，非末rank FULL graph把每次forward结果复制到
同一个 `self.intermediate_tensors`，重放后返回这个静态buffer的切片。
`gpu_worker.py:1289` fire-and-forget发送后不等待CUDA handles。
GroupCoordinator的self-retention保留引用，并不能阻止后续graph向同一个存储写入。
这是明确的源码事实；本轮没有捕获GPU中间值，尚未证明它就是本次错位的具体原因。

下一项优先验证静态PP输出的复用依赖，以及采样/异步输出的step和request映射。
可研究在下次重放写入之前建立CUDA发送完成事件依赖，或使用带代次的独立输出slot。
本地PyTorch `ProcessGroupNCCL.hpp` 明确说明默认WorkNCCL.wait给当前CUDA stream建立
依赖；是否保持宿主异步还必须核对blocking-wait配置，并在完整推理路径验收。
不能只把source引用交给Future就认为GPU读写顺序已安全。

因此第三轮看到的metadata阻塞不能当作已证明可直接删除的纯开销；改变宿主推进节拍
可能暴露此前被等待掩盖的复用/输出问题。后台CPU生命周期测试通过并不等于GPU路径正确。
由于在正确性检查处已停止，本轮没有额外追测候选热点迁移，也不宣称某个GPU热点占比。

### 本轮验证命令与恢复记录

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest \
  artifacts/dsv41f-concurrency-investigation-20261005/round4/test_metadata_reaper.py -q
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='' PYTHONPATH=$PWD \
  .venv/bin/python artifacts/dsv41f-concurrency-investigation-20261005/round4/gloo_reaper_probe.py
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest \
  tests/distributed/test_comm_ops.py -q \
  -k 'isend_object_posts or isend_tensor_dict or send_tensor_dict_sync or async_intermediate_tensors'
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest \
  tests/v1/spec_decode/test_dsv41_compact_relay.py -q -k 'send_relay_package or prepare_ahead'
```

结果为7+5+15项通过及真实Gloo验证通过。诊断数据见 `round4/diagnosis.json`，
回退复测见 `baseline-after-rejection.json`，恢复验收见 `restored-state.json` 和
`restored-smoke.json`。候选没有进入原服务argv或推理源码。
恢复父进程2968503、PP0–PP4 worker为2971183–2971187，argv/env、9个运行源码哈希
和逐rank GPU UUID均与停服前一致；health=200，开发RPC返回404，26662无监听。
恢复后单路/四路32K/512的全部5个请求IDs逐项等于参考，decode分别141.77/232.70 tok/s。
四路是一次热prefix运行，不能将其与此前慢cohort的数字相比并宣称优化收益。
测试时的worker代码保存在 `lite_worker_tested.py`；诊断入口现已禁止重新开启被否决候选。

## 第五轮：补CUDA发送依赖后输出通过，但后台回收没有端到端收益

本轮验证第四轮留下的静态PP输出复用风险。继续使用GPU2–6、PP5/TP1、DSpark5、
主存Engram和target prepare-ahead。停服前父进程2968503的完整argv/env、逐rank
GPU映射及10个运行源码哈希单独保存；新增保护范围包括 `gpu/cudagraph_utils.py`。
原型和原始测量位于忽略的 `artifacts/dsv41f-concurrency-investigation-20261005/round5/`，
没有改动推理源码。一次诊断加载后在同一个实例内切换三种模式。

### 单独验证CUDA依赖，再组合后台回收

三个模式分别为原基线、`fence_only` 和 `fence_reap`。后两者在target execute阶段
发布tensor sends后，对返回的CUDA handles调用默认无timeout的 `wait()`，给当前
producer stream建立发送完成依赖，再允许该stream后续工作覆盖静态输出。
不等待metadata handle、不改变sample阶段的DSpark同步relay。
`fence_reap`另加第四轮的有界后台metadata回收，保留FIFO、异常传递和source引用。

安装的PyTorch `ProcessGroupNCCL.hpp:286-312,353-368` 明确规定NCCL在独立stream
执行，默认wait/synchronize建立当前stream依赖。原服务及诊断环境均未设置
`TORCH_NCCL_BLOCKING_WAIT` / `NCCL_BLOCKING_WAIT`；初始化还检查CUDA backend
是NCCL，实际target发送只含CUDA tensors。这个依赖针对GPU读写顺序，不能把
wait的宿主调用时间当作GPU发送时间。

停服前5个CPU合同用例通过：只等待设备handles、跳过空tensor、metadata不等待、
基线及draft路径保持原行为、遇到CPU tensor在任何wait之前拒绝。诊断确认PP0–PP3
每臂实际执行了fence，PP4不执行，所有臂的静态graph输出storage地址保持不变。
诊断初始化关闭原step-timing收集；短记录只在PP0测CPU调用及实际batch形状。

固定32K prompt、512输出、greedy、热共享prefix；所有请求与原单路参考IDs逐项比较。
共完成57个请求，全部返回512个token并逐项一致，没有preemption。其中基线22个、
仅fence 13个、fence加后台回收22个。第四轮的跳6token现象本轮没有复现，但不同
运行的22个组合候选通过不足以证明上一轮错误的具体根因或更广泛的正确性。

### 无记录ABBA及流程对照

| 波次 | 模式 | decode tok/s | e2e tok/s | 输出迭代记录 | 接受draft token |
| --- | --- | ---: | ---: | ---: | ---: |
| A | 基线 | 189.27 | 181.85 | 268 | 1691 |
| B | fence+后台回收 | 190.10 | 181.61 | 268 | 1691 |
| B | fence+后台回收 | 147.66 | 143.15 | 356 | 1692 |
| A | 基线 | 192.31 | 183.88 | 268 | 1691 |
| 单路对照 | 基线 | 145.32 | 129.77 | 89 | 423 |
| 单路对照 | fence+后台回收 | 145.38 | 130.74 | 89 | 423 |
| 四路100ms间隔对照 | 基线 | 241.07 | 229.99 | 179 | 1695 |
| 四路100ms间隔对照 | fence+后台回收 | 240.65 | 229.32 | 179 | 1695 |

ABBA中位数：decode **190.79 → 168.88 tok/s，-11.48%**；
e2e **182.87 → 162.38 tok/s，-11.20%**。第二个B臂输出迭代增加到356；
这些无记录波次没有直接GPU batch记录，不能把记录数直接换成某个M形状。
同为268记录的首个A/B仅decode +0.44%、e2e -0.13%，没有可采用收益。
单路decode +0.04%；100ms间隔四路decode -0.17%、e2e -0.29%。
100ms仍只是cohort对照实验，不是给生产请求添加等待的建议。

仅fence的两个无记录四路检查分别190.22/190.51 tok/s，均为268记录；
它们是正确性/可运行性检查，没有完成独立ABBA，不能与基线238.36的warmup
（179记录）相比并计算该依赖自身的速度损失。

### 为何没有收益：原“回收热点”随cohort消失

短记录中观察到的实际GPU batch与CPU调用中位数如下。表内耗时全部是宿主墙钟，
不与GPU kernel/target span相加，也不跨不同形状计算候选的独立成本。

| 模式 | 实际decode batch次数 | decode tok/s | PP0 reap ms | PP0 CUDA fence调用 ms |
| --- | --- | ---: | --- | --- |
| 基线 | M6/B1=90，M18/B3=89 | 242.64 | M6=0.055，M18=0.060 | — |
| 仅fence | M6/B1=179，M12/B2=89 | 189.97 | M6=26.160，M12=19.586 | 0.048/0.048 |
| fence+后台回收 | M6/B1=90，M18/B3=89 | 242.68 | M6=0.083，M18=0.092 | 0.022/0.022 |

基线与组合候选的两个快波次实际形状相同，decode只差0.015%。此时原回收本来
只耗约0.06ms，新增后台任务和fence没有可回收的几十毫秒固定开销。
两个快波次的PP0 runner_execute宿主中位数也基本相同：M6为29.122/28.869ms，
M18为57.218/57.415ms，说明主流程节拍未缩短；这里没有进一步GPU分解，
不能把这些宿主时间全部归到权重计算或某一个等待。

慢的仅fence波次确实再次出现约20–26ms回收等待，但批次组织也不同，不能用它
与M6+M18快波次相减宣称后台回收省下这段时间。本轮无记录ABBA及独立间隔对照
都未显示收益。因此“回收CPU调用很长”不等于“把它异步化就缩短同等端到端时间”。
这段等待与PP/cohort节拍相关；是否属于关键路径需在相同实际形状下继续判断。
组合候选所有rank retirement队列最多2项、背压0次，队列达到上限不是回退解释。

采样回传ring及AsyncOutput的只读复核也没有找到直接原因：PP receive每步新分配
tensor，消费有CUDA event和request generation检查；普通采样输出新分配后再D2H。
DSpark staging是静态复用，但现有relay显式等待所有发送handles。
`draft_tokens[input_batch.idx_mapping]`是高级索引，新建gather结果，不能误当成静态view。
这些源码边界不构成“所有异步路径已证明正确”的结论。

本轮否决fence+后台回收的性能推广，也不把未独立验证收益的仅fence改为默认。
下一优先项回到服务内部的request admission/cohort及PP step组织，在不改变T+5
sampled-token合同、不给客户端增加固定延迟的前提下，提高实际验证batch合并率。
若继续研究异步PP发送，应区分独占发送快照与复用静态buffer：前者可避免把发送
完成依赖直接放在下一轮producer stream上，但拷贝成本及端到端收益仍须单独验证。

### 交付与恢复

验证命令：

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest \
  artifacts/dsv41f-concurrency-investigation-20261005/round5/test_send_fence.py -q
.venv/bin/ruff check artifacts/dsv41f-concurrency-investigation-20261005/round5
PYTHONPATH=$PWD .venv/bin/python \
  artifacts/dsv41f-concurrency-investigation-20261005/round5/diagnose_round5.py
```

原始测量为 `round5/diagnosis.json`，汇总与拒绝理由为 `round5/analysis.json`，
实验合同为 `experiment-plan.json`，实际worker代码及哈希保存为
`lite_worker_tested.py` / `tested-code-sha256.json`。诊断入口已禁止重新开启被否决候选。
诊断完成后停止26662并恢复26661原argv/env；恢复验收与真实推理结果分别记录在
`round5/restored-state.json` / `round5/restored-smoke.json`。

恢复父进程3230660，PP0–PP4为3233438–3233442；完整argv/env、10个运行源码哈希、
逐rank GPU UUID均一致，health=200，开发RPC返回404，26662无监听。
恢复后单路及四路32K/512的全部5个请求再次与参考IDs逐项一致，decode分别
143.81/240.61 tok/s。单路首次prefix仍冷，e2e仅64.62 tok/s；四路热prefix的e2e为
229.14 tok/s。这是恢复验收，不是部署收益，不能与不同cohort/冷暖状态作提升比较。
本轮新增CPU合同测试5项通过，诊断代码ruff及format检查通过，diff whitespace检查通过。

## Round 6：PP decode 对齐候选，输出验收失败后回退

本轮确认了长期固定的 PP decode 分组机制，并得到诊断实例上的合批收益。
但正式入口四路未复现吞吐提升，且补充的可区分输出验收失败，候选已撤回，
不作为成功优化推广。此前相同续写的 prompt 检查不足以确认普遍正确性。

### 根因与调度合同

本轮保留原 DSpark5、CED、RAM Engram、target prepare-ahead、PP5/TP1、
GPU2–6 配置。原服务完整 argv/env、11 个运行源码哈希及逐 rank GPU UUID
保存到 `round6/service-before.json`；私有环境文件权限为 0600。

`Scheduler.schedule()` 每次递增 `current_step`，未到
`request.next_decode_eligible_step` 的请求不允许 decode。AsyncScheduler 在最后
一块 prefill 或 decode 后设置 `next_decode_eligible_step = current_step + 5`。
因此，分步完成 prefill 的请求会永久保留不同的 `step % 5` 相位；即使其他条件
都相同，也无法自然合并。上一轮基线完成 prefill 的 step 为 4986/4987，首次
decode 为 4991/4992，之后全部间隔 5 步。慢波次有 3 个相位，快波次有 2 个。
原始复核见 `round6/historical-cadence.json`。

本轮只修改下一次 decode 的可调度步号。至少两个请求处于 running 时，将
`current_step + 5` 向上对齐到 5 的整数倍；部分 prefill 与单路维持原行为。
同组进入共同相位后，后续每次仍间隔 5 步，不会反复额外等待。新请求加入或
恢复时可能再次对齐，额外延后为 0–4 个 scheduler step，**不是 0–4ms 保证**。
混合较长 prefill 的非空 step 可能更慢，尚未测其延迟分布。

采样输出仍在 T+5 的 ring slot 消费。空 step 必须经过原 worker 的
`update_pp_decode_requests()`，不能跳过；它在零 token 的提前返回前执行。
保留原 sends、metadata 回收、placeholder/KV 更新及 engine 的空步 1ms yield。
调度步数不要求减少，目的是合并有 GPU forward 的批次。

### 同实例对照与可观测实际形状

诊断实例使用原 AsyncScheduler 的薄包装，只在正常更新后按文件控制开关对齐。
同一个已加载实例交替基线/候选；32K 固定 token prompt、greedy、热 prefix，
每路 512 输出。无逐 step 记录的两轮 ABBA 如下，均以客户端实际输出计时：

| 轮次 | 臂 | 模式 | decode tok/s | e2e tok/s | 输出迭代记录 |
| --- | --- | --- | ---: | ---: | ---: |
| 第一轮 | A | 基线 | 244.12 | 230.86 | 179 |
| 第一轮 | B | 对齐 | 237.60 | 225.04 | 94 |
| 第一轮 | B | 对齐 | 265.07 | 249.43 | 92 |
| 第一轮 | A | 基线 | 244.56 | 231.42 | 179 |
| 第二轮 | A | 基线 | 240.04 | 225.85 | 179 |
| 第二轮 | B | 对齐 | 262.87 | 247.20 | 91 |
| 第二轮 | B | 对齐 | 278.40 | 261.31 | 92 |
| 第二轮 | A | 基线 | 244.52 | 231.30 | 179 |

第一轮中位数 decode +2.86%、e2e +2.63%；第二轮 +11.70%/+11.23%。
合并全部四个 A 与四个 B，中位数 **244.32 → 263.97 tok/s（+8.04%）**，
e2e **231.08 → 248.32 tok/s（+7.46%）**。候选单波次范围 237.60–278.40，
不能宣称每次请求必然提速。第一轮首个 B 确实比 A 慢，且多一个 prefill step；
尚无对应 GPU 阶段分解，不能把整段损失确定归因于 prefill。

原计划要求首轮两个指标均超过 5% 才复测。首轮未达到该门槛，但存在波次差异，
且短记录已确认实际合批，所以增加一轮有限 ABBA 来判断可重复性；保留此偏离，
不能把第二轮或合并统计写成原先预注册的通过结果。

短记录直接捕获 PP0 worker 送入 forward 的实际 batch，结果如下：

| 模式 | 实际 decode batch 次数 | decode tok/s |
| --- | --- | ---: |
| 基线 | M6/B1=90，M18/B3=89 | 243.60 |
| 对齐 | M6/B1=2，M18/B3=1，M24/B4=88 | 263.04 |

基线最后 prefill 在 step 3650/3651，首次 decode 3655/3656；候选最后 prefill
5008/5009，先在 5013 单路、5015 三路，随后合并到同一个相位。
这里的 M 为实际 query token 数，padded token 数也分别为 6/18/24。
表中的输出迭代记录来自 metrics，包含输出交付；仅凭它不能判断实际 M/B。
无记录 ABBA 的形状另有 scheduler CPU 计数，与 worker 短记录相互印证，
仍应区分“scheduled batch”与“实际 forward batch”。

### 相似条件推广与正确性边界

32K/256 的两路、八路各做一组容量对照；32K/512 做单路及 100ms 到达间隔对照：

| 对照 | 基线 decode/e2e tok/s | 对齐 decode/e2e tok/s | 范围 |
| --- | ---: | ---: | --- |
| 两路 | 158.72 / 139.34 | 199.96 / 170.09 | 单组容量检查 |
| 八路 | 258.61 / 242.58 | 323.45 / 298.61 | 单组容量检查 |
| 单路 | 145.66 / 130.26 | 143.91 / 128.42 | 原调度不变；约 1% 波动 |
| 四路，100ms 间隔 | 240.82 / 229.65 | 244.30 / 232.29 | 单组到达对照 |

两路稳态 scheduled B2/M12=44，八路 B8/M48=44；这两个容量检查未做独立 ABBA，
不把约 25% 的单组提升作为已确认的普遍收益。八路最后首 token 等待
1.674 → 1.892s，也不能承诺对齐会改善 TTFT。100ms 间隔是实验条件，
生产没有给客户端增加这个等待。

两轮固定 prompt 共 80 个请求（512/256 输出），全部 token ID 与原单路参考
逐项一致，计数/finish reason 完整，无 KV preemption。另用 4 个不同 prompt，
每个建立独立单路参考，再以 20ms 到达间隔执行四路 ABBA，共 20 个请求，
128 输出逐项一致。这是本地 serving 等价校验，不是完整模型质量评测。
复核时确认这 4 个不同 prompt 的参考续写恰好相同，因此该组不能证明请求之间
没有输出串位；生产验收另用生成结果不同的自然语言请求补上这一检查。

CPU 侧复用现有 PipelinedEngine 测试模拟真实 PP5、6 项 in-flight 队列，验证
分步 admission 合批、最小 T+5 间隔、不同 draft 接受数、token 顺序与位置；
并扩展已有 KV 压力/在途 preemption 测试。完整 AsyncScheduler suite 为
**30 passed**，pre-commit（含 ruff/format/mypy）通过。

### 候选实现与生产验收失败

候选修改 `vllm/v1/core/sched/async_scheduler.py`，设计开关
`dsv41_sm80_pp_decode_coalesce` 默认关闭。启用需同时满足 DSV41F 40 层、
SM80、PP5/TP1/DP1、V2 async runner、DSpark5、关闭 adaptive verification、
`max_num_seqs <= 8`。其他拓扑回退原行为，不据此声明其他模型/GPU 有收益。
未通过最终验收，当前推理源码与测试均已恢复到本轮开始时；候选源码、测试、
哈希作为可复核的实验材料保存在 `round6/`。

候选 26661 验收采用原环境与完整 argv，只在已有 additional-config 对象中增加
`"dsv41_sm80_pp_decode_coalesce":true`；11 个运行文件只允许 AsyncScheduler
哈希变化。诊断用文件轮询、batch 记录、开发 RPC 与自定义 scheduler/worker
均未装入生产。候选状态为 `round6/production-candidate-rejected.json` /
`candidate-production-state.json`。

| 正式入口验收 | decode tok/s | e2e tok/s | 输出迭代记录 |
| --- | ---: | ---: | ---: |
| 单路，32K/512，首次冷 prefix | 140.95 | 63.55 | 89 |
| 四路，32K/512，第 1 波 | 241.75 | 229.68 | 93 |
| 四路，32K/512，第 2 波 | 243.49 | 230.84 | 93 |
| 八路，32K/256 | 328.11 | 304.11 | 50 |

正式四路中位数 242.62/230.26，约等于上一轮原生产恢复检查的
240.61/229.14；这些跨重启结果不能算成同实例 A/B 收益。
第一波存在 `_fp8_paged_mqa_logits_kernel` 首次 JIT，第二波仍未复现诊断提升，
所以 JIT 不能完整解释这一差距。合批减少了输出迭代，但没有稳定缩短生产
四路总 decode 时间；诊断候选本身也有 237.60–278.40 的大幅波动，当前不能
从端到端结果确定是哪段 GPU/CPU 等待抵消收益。各波结束后的 SM 频率近似一致，
这也不足以排除波内变化。不得选择诊断中的较快 B 波次宣称生产收益。

以上 21 个固定/不同尾部周期 prompt 请求输出均一致，但这些续写不能覆盖一般
自然语言请求。进一步用 4 个短自然语言 chat prompt（43/43/45/44 token），
分别建立 128 输出的单路参考，确认四个参考结果各不相同，再以 20ms 到达间隔
执行四路。四路均不等于各自参考，首次差异位置分别为 **32/33/84/0**，
输出计数仍完整为 128。它们也不等于其他请求的完整参考，不能据此确定发生了
简单的请求串位；批次改变的数值差异、既有并发路径缺陷与候选本身都尚待区分。
记录为 `round6/production-distinct-outputs.json`。

触发既定 token ID 停止条件后，立即停止候选性能实验、撤回本轮推理/测试改动、
删除新增 serving 开关并按原 argv/env 和 11 个原源码哈希恢复服务。
下一项只做原调度上的同组请求功能对照，判断差异是否由候选引入；在此之前
不将该候选推进生产。

### 原调度功能对照与最终恢复

恢复原调度后，4 个单路参考均与候选期间逐项一致；四路并发与自己的单路
参考比较为 **[一致、不一致、一致、不一致]**，首次差异位置为
—/33/—/0。第一项与第三项在原调度一致、候选不一致，但此时实际 batch 也不同，
不足以单独证明新增状态错误。第二项原调度与候选的并发输出完全相同，
却都不同于单路，确认“并发必须与单路逐 token 相等”不能不加判断地当作
当前模型的通用正确性标准。记录为 `baseline-distinct-outputs.json` /
`baseline-candidate-functional-comparison.json`。

进一步在原调度用相同 4 个 prompt，设置 `max_tokens=1`、greedy、返回
top-5 logprob，分别单路与四路执行，8 个请求全部正常闭合。前三项首 token
一致，第四项仍不同：

| 第四请求首 token | ID 2581 的 logprob | ID 39932 的 logprob | 选择 |
| --- | ---: | ---: | --- |
| 原调度单路 | -0.698448 | -0.698448 | 2581 |
| 原调度并发 | -0.980548 | -0.480548 | 39932 |

单路前两名精确并列，并发相差 0.5；该差异在只生成首 token 时已存在。
这把至少第四请求的差异定位到初次前向/采样结果，无需多轮 DSpark 验证链来
解释，并支持进一步检查批次相关数值与 argmax 边界。不能据此断定其他三个
差异也只有这一原因，更不能宣称已证明全部并发路径没有状态缺陷。
这一对照同时改变了输出上限和 logprobs 开关，未做 GPU batch/阶段记录，
不能用其耗时评价性能。原始记录为 `baseline-first-token-control.json`。

最终恢复的固定 32K/512 单路及四路共 5 个请求全部与原参考逐项一致，decode
分别 **142.00 / 234.27 tok/s**；单路仍为首次冷 prefix。恢复后的原调度 suite
**22 passed**，本轮新增候选代码及候选测试均已撤回。只保留分析、原始数据与
被拒候选的源码副本；未修改其他已有工作区内容。

最终服务为原 `dsv41f-manual.service`、26661、GPU2–6、DSpark5、prepare-ahead
原默认。父进程 3568053，PP0–PP4 为 3571241–3571245；原完整 argv/env、
11 个运行源码哈希与逐 rank GPU UUID 均恢复一致，health=200、开发 RPC=404、
26662 无监听。最终验收为 `round6/restored-state.json`，真实推理为
`baseline-restored-smoke.json`。

本轮结论是保留“按 PP 相位合批”的研究方向，但**不接入这个候选**：四路生产
收益未稳定复现，数值等价与状态正确性的证据还需区分。下一轮应针对真实生产
batch 测量稳态周期与 target/draft 等待，并在相同 token 前缀上比较 logits/
接受结果，避免把大批次导致的近并列 token 翻转直接判成串位，也避免用相同
续写的周期 prompt 掩盖真正的 token 丢失。

原始实验与汇总：`round6/diagnosis.json`、`replication.json`、
`mixed-validation.json`、`analysis.json`。CPU/静态验证命令：

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest \
  tests/v1/core/test_async_scheduler.py -q
.venv/bin/pre-commit run --files \
  vllm/v1/core/sched/async_scheduler.py tests/v1/core/test_async_scheduler.py
```

## Round 7：空步 yield 干预及实际 batch 路径复核

本轮没有可接入的性能优化。将纯 PP cooldown 空步的 `time.sleep(0.001)`
改为 `time.sleep(0)`，显著减少宿主等待，却没有缩短实际 GPU 稳态周期或带来
同 batch 的端到端收益。生产保留原休眠策略，本轮未修改推理源码。

### 实验合同与成本

继续使用原 PP5/TP1/DP1、GPU2–6、DSpark5、CED、RAM Engram、FULL graph、
target prepare-ahead 配置。原父进程 3568053、PP0–PP4 3571241–3571245 的
argv/env、11 个源码哈希与逐 rank GPU UUID 保存为 `round7/service-before.json`。
私有环境文件权限为 0600；重启在之前已授权的诊断范围内。

候选只作用于：没有 waiting 请求，所有 running 请求都是 RUNNING 且
`next_decode_eligible_step > current_step`，本次没有执行模型但仍有未完成请求。
没有 KV/EC connector。仍执行原 `step_fn()`、输出交付、`post_step()`、worker
空步和 T+5 sampled-token ring，`sleep(0)` 仍释放 GIL。其他等待保留 1ms。
没有复用 Round 6 已否决的 decode 相位对齐，也没有开启 metadata 后台回收。

同一个诊断实例通过文件控制交替 A/B，性能波次关闭逐 step CUDA/CPU 记录，
只保留 scheduler 计数；两个模式经过同一个薄包装。32K 固定 token prompt、
greedy、prefix 缓存盐不变。单路/四路 warmup 后，各一轮 ABBA（512 输出），
随后单路两个 CUDA event 短记录；另有八路 256 输出容量对照。共 **43 个请求**，
全部 ID 与既有原单路参考逐项一致，输出计数/finish reason 完整、KV preemption
增量均为 0。这是固定 prompt 的 serving 等价检查，不是全面模型质量评测。

### 端到端：必须对齐实际调度形状

decode 为客户端首块之后交付的 token 数除以全波次 decode 区间，e2e 为全部
输出 token 数除以波次墙钟时间；两者均为聚合 tok/s，不是单请求速度。
表中分组为 scheduler 计数，M 是 query token 数；GPU 短记录另核实单路实际
forward 是 M6/B1、padded M6、FULL graph。

| 并发 | ABBA 臂 | decode tok/s | e2e tok/s | scheduled decode 次数 |
| ---: | --- | ---: | ---: | --- |
| 1 | A，1ms | 140.00 | 126.14 | B1/M6=89 |
| 1 | B，0ms | 142.10 | 126.60 | B1/M6=89 |
| 1 | B，0ms | 142.36 | 127.49 | B1/M6=89 |
| 1 | A，1ms | 142.63 | 127.58 | B1/M6=89 |
| 4 | A，1ms | 244.33 | 230.76 | B1/M6=90，B3/M18=89 |
| 4 | B，0ms | 244.37 | 231.17 | B1/M6=90，B3/M18=89 |
| 4 | B，0ms | 226.15 | 211.30 | B2/M12=178，B1/M6=2 |
| 4 | A，1ms | 150.97 | 146.09 | B1/M6=356 |

单路中位数 **141.32 → 142.23 decode（+0.65%）**、
**126.86 → 127.05 e2e（+0.15%）**，未达到 3% 的最小收益门槛。
四路只有前两个 A/B 的分组相同：decode **+0.016%**、e2e **+0.179%**。
这是一个匹配对，没有独立复测，不构成精确估计；足以否决“大幅休眠开销”假设。
后两个臂分别是 2+2 与四个单路，不能混入休眠收益中位数。

自动脚本当时仅按吞吐中位数判定四路 useful=true，未检查分组等价，所以又执行
一对八路容量检查。这是实验门槛的缺口，保留原始标志并在 `analysis.json`
明确覆盖为 rejected，不能把原始统计当成通过。八路 A/B 分别为
**240.43/226.47** 与 **344.29/316.21** decode/e2e；稳态分组由 1+5+2
变成 1+7。这不是可归因的休眠收益，停止扩大测试。下一轮接受门槛需同时检查
实际 batch、prefill/缓存条件与数值合同，不能只比较总吞吐。

### CUDA event：宿主节省被 GPU 工作覆盖

单路短记录各 89 次 decode，去掉前 5 次与后 2 次，逐 rank 各 82 个稳态样本。
使用现有 StepTiming CUDA event，结束且服务空闲后才同步/读取，没有逐步
`cuda.synchronize()`。表中为 GPU 时间线区间的中位数（ms），包含 stream
空闲、enqueue 和 PP 等待；不能解释为隔离的 kernel 时间。

| rank | target A/B | draft 区间 A/B | prepare 区间 A/B | forward 周期 A/B |
| --- | ---: | ---: | ---: | ---: |
| PP0 | 5.690 / 5.680 | 0.003 / 0.003 | 2.702 / 2.674 | 40.982 / 40.872 |
| PP1 | 5.627 / 5.627 | 22.657 / 22.610 | 8.429 / 8.396 | 40.999 / 40.887 |
| PP2 | 6.397 / 6.392 | 16.866 / 16.833 | 14.117 / 14.037 | 41.063 / 40.927 |
| PP3 | 6.479 / 6.482 | 10.848 / 10.785 | 20.461 / 20.326 | 41.112 / 40.927 |
| PP4 | 7.010 / 7.012 | 4.114 / 4.082 | 28.631 / 28.647 | 41.033 / 40.971 |

每个模式记录 356 个有 sleep 的 engine 空步。`step_fn()` 返回之后，输出交付、
`post_step()` 与 sleep 的宿主尾段累计 **391.83 → 30.68ms**，尾段中位数
**1.103 → 0.082ms**。这个区间并非 sleep syscall 的独立精确时间，也不能加到
GPU 成本上。虽然宿主累计减少约 361ms，PP0 forward 周期只减少 **0.110ms**，
无记录单路 e2e 仅 +0.15%。原 `step_with_batch_queue()` 保留 6 项 in-flight
队列，队列满后等待 worker/Future；空步休眠主要与在途 GPU 工作重叠。
该测试排除了本配置中的显著收益，不外推其他模型/拓扑。

本轮还修正一个分解口径：PP1–PP3 的 draft 区间不是独立 draft 计算。
`model_runner.py` 在非末级 target 结束后开始 drafter 计时，再调用
`spec_decode/dspark/speculator.py:participate_distributed()`；该函数先等待从
head/上级收到 package。PP4 只有 target 前向及采样完成后才发起 head propose。
因此 PP1 的 22.66ms 包含等待后续 target rank 与 draft package 的时间，
其他参与者区间也相互重叠；**不能将各 rank 的 draft 或 prepare 相加作为串行
额外成本**。PP4 target 结束至 drafter 开始约 1.127 → 1.116ms，但这里也只是
事件区间，不是已隔离的采样算子热点。要研究 draft 融合/并行，应进一步区分
stage recv、prepare、state wait、graph replay 与 head Markov，而不是直接削减
这个有重叠的总数。

### 相似形状的算子路径

只读复核了服务可达的 MoE、indexer 与 sparse MLA 分支：

| target M（q6） | MXFP4 W13/W2 GEMM | routed 行数 | activation + A2 quant |
| ---: | --- | ---: | --- |
| 6（B1） | 已准备的 TileLang M6 专用路径 | 36 | 融合 TileLang |
| 12（B2） | 通用 CUTLASS grouped | 72 | 融合 TileLang |
| 18（B3） | 通用 CUTLASS grouped | 108 | 融合 TileLang |
| 24（B4） | 通用 CUTLASS grouped | 144 | 分开执行 |
| 48（B8） | 通用 CUTLASS grouped | 288 | 分开执行 |

以上需同时满足 DSV41F E384/top-k6、SM80、对应方法已准备及 activation/clamp
条件。`mxfp4_int8_sm80.py` 的融合 dispatcher 当前限制 routed 行数 ≤128；
kernel 本身使用动态行数，但没有证明扩展门槛能带来 e2e 收益，本轮未扩大。
这是一处可检验的推广机会：B4/B8 可以研究 activation/quant 融合，仍须经过
独立数值、graph、实际 serving A/B。

单路 M6 的 W13 自身编译 BN64，外层 M6 分支检查 `(block_m, block_n)=(16,128)`。
编译 callable 自行生成 grid，不接收外层 BN 参数，不能将这两项直接判为错配。
服务加载日志已确认 M6 W13/BN64 + W2/BN128 准备完成；其他上述形状回退 CUTLASS。
专家真实 padding 为各专家 `ceil(count_e/16)*16` 之和，仅有 M 不能推断专家数。

indexer 的 DSpark exact top-k 分支只接受总 decode token 数等于验证宽度 6 或 8；
本配置单路 M6 可达，并发 M12/M18/M24/M48 走 persistent top-k。sparse MLA
partial 在符合 index 长度条件且 M≤48 时使用 8 warp，所以上述并发形状没有跨过
49-token 的 4-warp 回退点。这些是源码路径，不是本轮已测算子差距。
FP8 dense 仍使用当前原后端，遵守此前放弃其 TileLang 重写的决定。

### 验证、拒绝与恢复

候选只存在于 ignored `round7/engine_yield.py`、`yield_scheduler.py`、
`lite_worker.py` 等实验材料，实际测试源码哈希保存为 `tested-code-sha256.json`。
5 个 CPU 合同测试覆盖原休眠、纯 cooldown、可调度/waiting 回退、非空前向及
输出/step/hook 恰好执行一次，全部通过。artifact ruff、format 均通过。

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest \
  artifacts/dsv41f-concurrency-investigation-20261005/round7/test_engine_yield.py -q
.venv/bin/ruff check artifacts/dsv41f-concurrency-investigation-20261005/round7
.venv/bin/ruff format --check artifacts/dsv41f-concurrency-investigation-20261005/round7
.venv/bin/python artifacts/dsv41f-concurrency-investigation-20261005/round7/analyze_round7.py
```

原始 ABBA、scheduler 计数、GPU/engine 短记录见 `round7/diagnosis.json`，
可复算拒绝结论见 `round7/analysis.json`。诊断实例已停止，原生产按原完整
argv/env 恢复，11 个运行源码哈希、五个 rank 的 GPU UUID 映射均与实验前一致。
父进程 **3852556**，PP0–PP4 为 **3855284–3855288**，health=200、开发
RPC=404、26662 无监听；DSpark5 与 prepare-ahead 原默认保留。

恢复后固定 32K/512 的单路与四路共 5 个真实请求全部 ID 一致、计数及闭合完整。
单路 decode/e2e 为 **140.74/63.97 tok/s**，四路为 **238.22/226.27 tok/s**。
单路是恢复后的首次冷 prefix，不能与热 prefix e2e 比较，也不将这些跨重启结果
写成候选收益。恢复验收见 `round7/restored-state.json`、`restored-smoke.json`。
文档 pre-commit、artifact ruff/format 与 diff whitespace 检查通过；保留其他
已有工作区修改，没有提交或接入本轮候选。

## Round 8：融合推广收益估算与并发 MoE 分块实验

本轮补测 Round 7 发现的 activation/quant 融合门槛，并对并发 decode 可达的
CUTLASS grouped GEMM 分块做完整 MoE 对照。**没有修改或重启生产推理服务**：
融合仅有低于 0.3% 的周期收益估算，两个分块候选都未胜出，不进行推广。
保留可复用的 `benchmarks/kernels/deepseek_v41_moe_batch_tiles.py` 和原始证据。

### 范围与最小实验

在空闲物理 GPU1 上测试：UUID `GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216`，
SM80，项目 Torch **2.13.0+cu130 / CUDA 13.0**，TileLang **0.1.15**。
生产 GPU2–6 的进程维持原状，没有在 serving 卡上运行 microbenchmark。
先复用已有 activation screen，再只增加一个 grouped GEMM 分块实验；
未重扫 q、上下文长度或重启五卡模型。

实验合同保存为 `round8/experiment-plan.json`。预先约定：如果融合的服务收益
估算低于 1%，不为扩大门槛单独重启；可做一次有界 grouped GEMM tile 对照。
两组均用 CUDA Events + CUDA Graph，此 CMP 卡不支持 CUPTI，不能当成 CUPTI
或硬件 counter 分析。编译、分配、权重读取、参考计算与 cache flush 均在计时外。

### 融合：算子更快，但覆盖范围与绝对成本都小

复用生产冻结的 256-thread、clamp=10 融合核，未改 kernel，也未改 128 行
dispatcher 上限。对 128、132、144、180、216、252、288 行做三轮交错 A/B，
每轮 15 samples、16 calls/Graph、200ms 预热稳定活动。
沿用真实 workspace：INT8 q 与 FP32 scales 位于原 BF16 activation 存储中。
全部有限 BF16 gate 的 rounding certificate 为 0 mismatch；所有测试行数的
INT8 q、FP32 scale 都逐位相等，poison 后 Graph replay 完整覆盖输出。

| routed 行数 | separate µs | frozen fused µs | 每层节省 µs |
| ---: | ---: | ---: | ---: |
| 132 | 5.824 | 3.584 | 2.240 |
| 144（B4/q6） | 5.952 | 3.584 | 2.368 |
| 180（B5/q6） | 6.336 | 4.096 | 2.240 |
| 216（B6/q6） | 6.720 | 4.160 | 2.560 |
| 252（B7/q6） | 7.104 | 4.800 | 2.304 |
| 288（B8/q6） | 7.488 | 4.800 | 2.688 |

这是 warm-buffer 隔离计时，不能用作 cold-HBM 带宽上限。按每个 target layer
一次 activation、40 层计算，B4/B8 只节省 **0.09472/0.10752ms**；相对于
上一轮单路约 41ms 的周期，约 **0.23%/0.26%**。这是跨形状的收益估算，
没有测出真实并发周期的新收益，也不是绝对上界。

可达性比算子百分比更关键：上一轮四路的实际 1+3、2+2、四个单路分组，
M 最大只有 18、routed 行数最大 108，**已经走融合**。将门槛提高到 288
无法为这些波次增加覆盖。八路 1+5+2 或 1+7 的大组会获益，但按上述 isolated
节省估算，每次完整 target 仍只有约 0.09ms。因此保持门槛，不为约 36–40%
的算子降时宣称同等端到端提速，也不消耗两次六分钟的模型加载来测微小差异。
原始数据为 `activation-screen.json`，决策及估算为 `activation-decision.json`。

### 分块：完整 Experts.apply 对照

下一项验证原 BM16/BN128 是否适合 M12/M18。读取 checkpoint layer20 专家
137–172，按生产 remap，保留 tensor shard/offset/hash；其余 E384 槽填零且
不会被路由选中。hidden 与路由为合成数据，不是线上专家捕获。
三个路由分别触及 6、21、36 个专家；没有以路由数代替实际观测的线上 D。

只比较当前 C++ dispatcher 可调用的 **16×128、16×64、64×128**，每个候选
只改变一个 tile 维度。BM32 虽有 selector 类型声明，实际 dispatch 没有入口，
不能误称已经支持或直接传入 32。各臂使用同一当前 activation gate，M12/M18
均走融合；没有把关闭融合的旧 baseline 重算为新收益。

测量包括 A1 quant、cast/align、W13、activation/A2、W2 与 reduce。
output 与 workspace13 按 modular runner 方式别名。每个 Graph 一次完整 apply，
每个 sample 前写 64MiB buffer 清洗 L2，flush 不计时；三轮顺序正反交替，
每臂每轮 15 samples、200ms 稳定活动。所有输出及 W2 routed slots 逐位一致，
Graph poison 覆写、改变 hidden 和专家 ID 后 replay 也通过。

| M | 路由 / 活跃专家 | 16×128 µs | 16×64 µs | 64×128 µs |
| ---: | --- | ---: | ---: | ---: |
| 12 | hot / 6 | 183.30 | 199.68 | 346.11 |
| 12 | mid / 21 | 498.69 | 499.71 | 988.16 |
| 12 | spread / 36 | 800.77 | 825.34 | 1580.03 |
| 18 | hot / 6 | 305.15 | 316.42 | 352.26 |
| 18 | mid / 21 | 502.78 | 512.00 | 995.33 |
| 18 | spread / 36 | 804.86 | 832.51 | 1590.27 |

16×64 没有任何测试点胜出：hot 慢 **8.94%/3.69%**；mid 的小差异有部分处在
轮间波动范围，不能称改善；spread 慢 **3.07%/3.44%**。此前 M1 的 BN64
收益不能直接迁移到这些并发形状，M6 的生产 W13/BN64 本轮也没有被替换。
64×128 慢 **15–98%**，直接否决。原始 samples、逐 tile metadata、checkpoint
来源及生成数据范围保存为 `round8/batch-tiles/`，不是 serving 性能分解。

更明确的热点证据是每个专家的 occupancy：hot 路由中，M12→M18 的每专家
行数 12→18，跨过 BM16 门槛，M tile 数 **6→12**、padded rows **96→192**，
完整 MoE 从 **183.30→305.15µs（+66.5%）**。mid/spread 的 M tile 数分别
保持 21/36，完整 MoE 只增长约 **0.82%/0.51%**。这直接显示不能按 M 的
线性增长推断 MoE 延迟，也不能仅凭增加专家数推断字节增量。

BM64 在 hot M18 下虽然把 tile 数 12 降为 6，却将 padding 提高到 384 行；
mid/spread 下 tile 数不减，padding 变为原来的四倍，实测更慢。这与额外 padded
计算相容；没有 counters，不断言唯一的 stall 原因或精确 HBM 重读字节数。
进一步研究应保留真实 expert-count 分布，关注 BM16 边界和细粒度分块，
不再盲目扩大整个 batch 的 tile。尚未证明这些合成 occupancy 就是生产热点。

### 交付与生产状态

本轮没有 serving 候选，所以没有执行新的端到端 A/B 或模型质量评测；没有
把 isolated 节省换算为已确认的 tok/s 收益。完整 apply 的数值/Graph 验收及
两组算子数据保留在 `round8/analysis.json`。benchmark pre-commit（包括
ruff/format/mypy）、screen ruff/format 与 diff whitespace 检查通过。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python -m benchmarks.kernels.deepseek_v41_moe_batch_tiles \
  --model /media/z/data6/models/DeepSeek-V4.1-Flash \
  --output-dir artifacts/dsv41f-concurrency-investigation-20261005/round8/batch-tiles \
  --tokens 12 18 --patterns hot mid spread --rounds 3 --samples 15
```

生产父进程仍为 **3852556**、PP0–PP4 仍为 **3855284–3855288**，GPU2–6
映射、原 argv/env 和 11 个运行源码哈希一致，health=200。空闲 GPU1 的测试
进程已退出、无残留 GPU owner；DSpark5、prepare-ahead、原融合门槛和原分块
保留。当前验收记录为 `round8/production-state.json`，被测脚本副本及源码
SHA256 保留于 round8，其他已有工作区改动没有修改或提交。

## Round 9：真实 decode 路由与 target 计时边界（10 月 6 日续验收）

本轮用真实服务路由补上 Round 8 的合成分布缺口。不同提示组成的 M18
batch 中，跨越 BM16 边界增加的 tile 仅占 **0.094%**。当前证据不支持把
放大 M 分块作为主要优化方向。同时，固定 C1/C4 的输入和路由完全一致，
但 C4 的 PP0 target span 多出约 **21.7 ms**；该 span 包含模型计算后的
发送发起和执行器调度区间，不能当作纯 kernel 耗时。

### 捕获方法与验收

当前 `--enable-return-routed-experts` 在 PP>1 下被配置检查拒绝。本轮保留
该 flag 为 false，使用独立诊断 Worker，在 KV 初始化后、Graph 捕获前，
仅对 `runner.model` 调用 `bind_routed_experts_capturer()`。诊断缓冲区没有
挂到 `runner.routed_experts_capturer`，没有进入 routed-experts 返回及
scheduler 存储路径。DSpark draft 是独立的 `speculator.model`，不绑定。

每个 router 的 callback 把 top-k ID 复制到 GPU 缓冲区；这些复制操作会
进入 FULL Graph 并在每次 replay 更新。随后在 target `forward_end` 后、
draft 开始前保存 GPU 快照，待服务空闲再写本地 JSON。每卡常驻缓冲区为
`2048×40×6×4` 字节，即 **1.875 MiB**。窗口最多保存 512 帧，只分析
FULL Graph、无 prefill、padded tokens≤48 的 decode 帧。

先在空闲的物理 GPU1 检查 M=6/12/18/24/48：改变输入后 Graph replay
更新记录，先前快照保留，未写区域保持 sentinel；5 个用例通过。诊断
服务使用原生产的 GPU2–6、PP5/TP1、CUDA 13、DSpark5、CED、RAM Engram
及 prepare-ahead，仅改为 loopback 26662 和诊断 Worker。

测量没有进行 q 扫描。固定 32K/256 的 warmup、记录关闭前后 C4、记录
开启 C1/C4 共 14 个请求，输出 ID 全部与既有参考一致。另复用 4 个已有
不同提示，各输出 128 token；四份输出不同、长度及协议检查通过。这组
不同提示不是生产流量的统计样本，也不是跨 batch 贪心输出一致性评测。

共保存 **313 帧/rank、12,520 个层观测**。逐帧核对 5 个 rank 的 step、
请求顺序、实际/padded token 数、query 起止、输入 ID 和位置，全都一致；
无丢帧，全部专家 ID 在 0–383 内且每个 token 的 top-6 无重复。每个 rank
的记录随输入变化，排除了只读取 Graph warmup 数据的情况。

### 真实分布：BM16 重读并非这组不同提示的主要成本

`D` 是本次验证 batch 每层涉及的不同专家数；
`tiles16=Σ_e ceil(count_e/16)`，由本地 GEMM 的 row 分块计算。
这是 tile 几何统计，没有测量实际 HBM transaction 或缓存命中率。

| 捕获窗口与实际形状 | batch 数 | 平均 D/层 | 平均 BM16 tiles/层 | 额外 tile 占比 |
| --- | ---: | ---: | ---: | ---: |
| 固定 C1，M6/B1 | 45 | 19.761 | 19.761 | 0 |
| 固定 C4，M6/B1 | 180 | 19.761 | 19.761 | 0 |
| 不同提示 C4，M6/B1 | 44 | 19.926 | 19.926 | 0 |
| 不同提示 C4，M18/B3 | 43 | 50.865 | 50.912 | **0.094%** |
| 不同提示 C4，M12/B2 | 1 | 39.025 | 39.025 | 0 |

固定 C4 本次形成 4 个单请求 batch，不能用来证明 BM16 在真实 M18
相同提示 batch 的表现。不同提示的 43 个 M18 batch 则足以检验该形状：
1,720 个层观测中，4.24% 的层至少有一个专家超过 16 行，但总共只增加
82 个 tile。假设 BM32 可用，tile 数也只从平均 50.912 减少到 50.865；
绝大部分专家的 M padding 会变大。Round 8 中 BM64 的实测退化与该几何
结果相符，仍保留生产 BM16。**没有实现或宣称 BM32 的端到端收益。**

这组固定提示的 q6 单请求实际 D=19.76，明显小于独立路由假设的 34.6。
六个 query 合计增加约 13.76 个专家，即相对首个 query 的 6 个专家平均
约 2.75 个新专家/query。这里直接观测到了连续 query 路由重合，不再
需要从延迟斜率反推 D；数值只代表本次提示和配置。

M18 的三个请求分别验证时，D 之和平均为 61.681；合并后是 50.865，
跨请求共享减少约 **17.54%** 的不同专家访问数。batch 增大确实存在
权重共享，但不能假设三个请求共享同一组专家。平均每个活跃专家只有
`108/50.865≈2.12` 行，当前边际成本主要来自更多不同专家。

### 接受长度是另一条独立轴

从同一请求连续验证帧的 query 首位置差，统计每轮实际位置前进量：

| 窗口 | 请求数 | 可比较的轮间位置差 | 平均前进 token |
| --- | ---: | ---: | ---: |
| 固定 C1 | 1 | 44 | 5.795 |
| 固定 C4 | 4 | 176 | 5.795 |
| 不同提示 C4 | 4 | 171 | **2.965** |

位置差是接受长度的代理指标，不是 scheduler accepted-token ledger。
不同提示窗口的 175 次单请求验证中，单个请求分别需要 32/43/44/56 次
验证；低接受长度增加了 target/draft 轮次。因此不能把不同提示窗口与
固定提示的 tok/s 差异全部归因给并发、专家分布或某个 kernel。

### PP0 target span 的边界问题

固定 C4 的全部 180 帧，按输入 ID、位置和全部 40 层路由逐帧与 C1
匹配，全部相同；模型仍是 M6 FULL Graph。各 rank 的 target span 中位数：

| rank | 固定 C1 ms | 固定 C4 ms |
| --- | ---: | ---: |
| PP0 | 5.707 | **27.442** |
| PP1 | 5.648 | 5.780 |
| PP2 | 6.427 | 6.428 |
| PP3 | 6.492 | 6.489 |
| PP4 | 7.042 | 7.035 |

逐 batch 的 5 rank span 和中位数为 31.354→53.197 ms。额外时间几乎
全部在 PP0；这不是专家数增加或跨过 BM16 边界的结果，但尚未测得其中
纯 Graph、发送依赖和 CPU/执行器调度各占多少。

源码时序如下：

```text
runner.execute_model: forward_start
  -> FULL Graph input staging + graph replay
  -> output/state bookkeeping
Worker.execute_model: intermediate tensors isend（非末级）
executor 调用/排队 sample_tokens
runner.sample_tokens: forward_end
  -> 同一步 sampling / distributed draft / sampled-token receive
```

`forward_start` 位于 `gpu/model_runner.py:1992`；Graph replay 位于
`:2021`；`gpu_worker.py:1289` 的发送发起在 runner 返回后、sample 之前；
`forward_end` 则到 `gpu/model_runner.py:2118` 才记录。两个 Event 之间
可能包含模型返回后的 CPU 间隔和设备依赖等待。非首 rank 的输入 PP recv
在 forward_start 前，同一步 draft/采样在 forward_end 后；不能把这些
阶段直接全部归入 target，也不能把不同 rank 的 draft span 相加。

现有 `target.model_graph_replay` phase Event 可测更窄的 replay 区间；
它仍在 Graph 外，包含 launch 等开销。若后续需要新的流程归因，应同时
记录 replay、Worker 发送、execute→sample 间隔和现有 target span。
本轮没有新的逐调用记录，**未证明某一处等待独占这 21.7 ms**。

这也不是第一次观察到该长窗口：Round 2 已分出模型提交之后约 22.9 ms
的宿主间隔；Round 3 在四个 M6 cohort 下直接测得 PP0 metadata 回收
中位 **22.182 ms**，而新的 tensor send 提交约 0.198 ms。Round 9 新增
的是同输入、同路由的证据，不能把旧现象重新当成未研究的热点。Round 4
后台回收出现正确性失败，Round 5 加 fence 后的同形状对照没有收益；
因此不重做已否决的异步回收，也不把长 CPU wait 直接换算为可省时间。

### 低接受长度方向的源码门槛

不同提示的平均位置前进量只有 2.965，后续值得评估能否减少无效验证。
检查当前 checkpoint 索引，存在 `mtp.2.confidence_head.proj.weight`；
不能以“checkpoint 没有 confidence head”为理由排除该方向。但当前
adaptive verification **不能直接打开**：

* `DeepseekV41AmpereMLASparseBackend` 继承 ROCm sparse backend；其
  `get_builder_cls()` 返回 `DeepseekV4SparseMLAMetadataBuilder`。
* `deepseek_v4_1/sparse_mla.py:148` 将该 builder 的 Graph 支持声明为
  `AttentionCGSupport.UNIFORM_BATCH`。
* `gpu/spec_decode/adaptive_verification.py:478-489` 要求 target 全部
  builder 支持 `ALWAYS`，因为它要捕获 varlen decode Graph；否则显式拒绝。

这是一条支持缺口，不是已有收益。若继续此方向，必须先验证裁剪后的 device
query 边界与 CPU metadata、ragged SWA/indexer 和 PP 状态的合同，再做
相同提示的有限对照；仅把 Graph 支持声明改为 `ALWAYS` 不能证明兼容。
当前生产继续保持 `enable_adaptive_verification=false`。

### 本轮取舍、性能边界与产物

记录关闭前/开启/关闭后，固定 C4 客户端 decode 分别为
220.59/139.04/167.60 tok/s。关闭窗口没有实际 batch 形状记录，不能
保证相同 PP 相位；这些数值不能用于计算捕获开销或优化收益。不同提示
记录窗口为 105.75 tok/s，接受长度及上下文也不同，不与固定窗口做 A/B。

本轮没有生产提速候选，保留原分块、融合门槛、DSpark5、CED 和
prepare-ahead。后续优先审查低接受长度下的无效验证，以及真实 varlen
Graph 支持；任何流程候选仍要在相同 batch 形状及接受长度下确认端到端
收益。PP0 长窗口沿用前几轮已验证的解释边界，不重复异步回收实验；
不继续盲调大 M tile。

数据和复现脚本位于
`artifacts/dsv41f-concurrency-investigation-20261005/round9/`：

* `capture-check.json`：5 个 Graph 记录更新/快照验收用例。
* `routes-{1,2,3}-rank{0..4}.json`：真实路由、CPU batch 和 GPU Event。
* `analysis.json`：真实 D、tile 几何、batch 形状及层统计。
* `fixed-route-equivalence.json`：180 帧输入/位置/40 层路由一致。
* `position-advance.json`、`phase-span-comparison.json`：接受长度代理及计时。
* `service-before.json`、私有环境文件：精确恢复清单；环境文件权限 0600。

离线复核命令不需要占用 GPU：

```bash
.venv/bin/python artifacts/dsv41f-concurrency-investigation-20261005/round9/analyze_round9.py
.venv/bin/python artifacts/dsv41f-concurrency-investigation-20261005/round9/check_route_records.py
```

中断前恢复实例加载完成后收到 SIGTERM；10 月 6 日继续时服务已停止。
本次按原始 argv/env 和 13 个运行源码哈希重新恢复，最终服务及输出验收
结果记录于 `restored-state.json`、`restored-smoke.json`，见下方恢复记录。

### 10 月 6 日恢复验收

原生产服务已恢复：父进程 **138537**，PP0–PP4 为
**140947/140948/140949/140950/140951**。原始 argv/env、13 个运行源码
哈希及 GPU2–6 映射一致；health=200，诊断 RPC=404，26662 无 listener。
私有环境快照权限为 0600。DSpark5、CED、RAM Engram、prepare-ahead
和原推理配置保留。

固定 32K/256 的 5 个请求全部与已有输出 ID 参考一致：

| 并发 | decode tok/s | 端到端 tok/s | prefix 状态 |
| --- | ---: | ---: | --- |
| C1 | **143.25** | 41.61 | 重启后的首轮 prefill |
| C4 | **229.66** | 210.14 | 同提示热 prefix |

这是恢复验收，不是优化 A/B。本轮没有改推理源码，也没有确认新的提速。
路由 Graph 验收、离线跨 rank/跨窗口断言、诊断固定输出及生产恢复输出
检查通过；诊断脚本 ruff/format 和工作区 diff whitespace 检查通过。

## Round 10：验证前缀、动态长度及 Graph 候选缺口（2026-10-06）

Round 1–9 的已验证探针、MoE tile benchmark 和分析已提交为
`c0ef46101b`。本轮复用 Round 9 的真实路由，先检查低接受率下缩短
target 验证的机会，再做一个候选的 A→B→A。没有进行 q 扫描。
证据保存在 `artifacts/dsv41f-concurrency-investigation-20261005/round10/`。

### 自适应验证不能直接用于当前 PP5

`VllmConfig._validate_adaptive_verification()` 在 PP>1 时明确拒绝启用。
confidence 由末级产生，成本曲线及 confidence 尚未广播给所有 PP rank；
各 rank 独立裁剪会形成不同的 token 总数、query 边界和传输形状。
SM80 indexer 的 CPU/GPU query mismatch 支持声明及 SWA/MLA 的
`UNIFORM_BATCH` 声明也是限制，但放宽这些声明不能解决 PP 一致性。

共享 token→request 映射本身已经读取 GPU `query_start_loc`；CPU 边界
只提供映射总长度，因此本轮没有新增一个重复的 GPU 映射实现。
没有启用 `enable_adaptive_verification`，也没有修改 attention 支持声明。

### 原轨迹前缀账本：机会依赖接受率和 batch

对 Round 9 的全部 40 层路由统计保留每个请求前 q 个 query 后的 D。
相邻记录中同一请求的位置增量 A 作为输出数量代理，截短的数量记为
`min(A,q)`。只使用存在下一条记录的请求；这是固定原轨迹的前缀机会
账本，截短后未来 draft anchor 和路由会改变，**不是新配置的速度预测**。

| 原窗口/形状 | q | D/层 | D/q6 | 截短输出代理/q6 |
| --- | ---: | ---: | ---: | ---: |
| 固定提示 M6/B1，45 batch | 3 | 12.386 | 62.68% | 51.76% |
| 固定提示 M6/B1，45 batch | 5 | 17.520 | 88.66% | 84.71% |
| 不同提示 M18/B3，43 batch | 3 | 33.819 | 66.49% | 80.52% |
| 不同提示 M18/B3，43 batch | 4 | 40.328 | 79.28% | 91.40% |
| 不同提示 M18/B3，43 batch | 5 | 45.837 | 90.12% | 97.71% |
| 不同提示 M18/B3，43 batch | 6 | 50.865 | 100% | 100% |

全部不同提示的 171 个位置增量平均 2.965；截成 q3/q5 后数量保留
75.54%/95.46%，但只看 B3 窗口是 80.52%/97.71%。因此单请求仍保留
q6、多请求试 q5，比全局改成 q3 有更小的输出损失。不能将 batch 大小
当作低接受率的充分条件；同提示并发或实际流量可能有不同接受率。

### 空闲 GPU1：完整 MoE 前缀实验

使用物理 GPU1（SM80，UUID `f385e0d6-452b-ad54-922e-ef7f4deb6216`）、
CUDA 13.0、Torch 2.13.0+cu130、TileLang 0.1.15。按 layer20 的 D
选固定 B1 的中位帧，以及不同提示 B3 的最小 D 和中位 D 帧。
从 checkpoint 仅读取三帧所需的 **81 个专家**，保留原专家 ID；其余
E384 槽位为零且不参与路由。hidden 和 routing weights 为合成输入。

计时边界是完整 `Experts.apply`，包括量化、路由排列、两次 GEMM、
激活及加权归约。使用生产可达的 CUTLASS 路径及既有激活融合；每张
Graph 一次调用，64 MiB L2 flush 在事件区间外，3 轮交替顺序、每轮
15 个样本。这里**没有生产的 query/token padding、attention、draft
或 PP 等待**。

| 捕获 step / B / layer20 D(q6) | q6 µs | q5 µs / 降幅 | q4 µs / 降幅 | q3 µs / 降幅 |
| --- | ---: | ---: | ---: | ---: |
| 542 / B1 / 15 | 374.784 | 335.872 / 10.38% | 296.960 / 20.77% | 284.672 / 24.04% |
| 1267 / B3 / 38 | 849.920 | 755.712 / 11.08% | 701.440 / 17.47% | 576.512 / 32.17% |
| 1357 / B3 / 46 | 1019.904 | 896.000 / 12.15% | 793.600 / 22.19% | 673.792 / 33.94% |

12 个形状的保留 token 输出及 W2 routed slots 均与未截短 q6 bitwise
一致。改变 hidden/路由后，对输出填 NaN 再 Graph replay，仍逐项一致；
计时前恢复原始捕获路由。没有把单层冷 L2 时间乘以 40 当作 target
或端到端时间。benchmark 为
`benchmarks/kernels/deepseek_v41_moe_verify_prefix.py`，checkpoint 读取
复用原 loader，并增加可选 `expert_ids`；其原 36 专家默认值保留。

### 通用修复：不同 query 长度需要独立 Graph 候选区间

新增真实稀疏 capture grid 的回归暴露了 `_init_candidates()` 的缺口。
q5/q6 同时捕获时，旧代码在所有 query 长度之间共用一个
`current_range_start`：对 B5/q6 的 M30，候选只含 q5 的 M35，检查
`uniform_token_count` 后被拒绝，尽管 q6 的 M36 Graph 已经捕获。
于是该请求退回 eager。

候选区间现在按 `(uniform_token_count,max_query_len)` 独立推进；每个
候选仍经过原来的请求数、query 长度和 LoRA 兼容性检查，FULL/PW
优先级不变。B5/q6 的 M30 现在匹配 q6/M36。该修复用于 MRv2 的
所有动态 speculative schedule，不局限于 DSV41F。

离线对照旧源码，固定长度下 FULL_DECODE_ONLY、FULL_AND_PIECEWISE、
FULL 的完整候选字典分别有 49/97/97 个 key，**新旧逐项一致**。
回归同时覆盖原生产网格和动态 schedule 的默认网格、B1–B8、q5/q6
及不均匀 batch 的 fallback。

需区别两个网格：原 q6 默认网格含 12；动态 schedule 默认网格删除
12、增加 5/10/20。因此原网格下 B3/q5 可匹配 M15，但候选默认网格
下会匹配 **M20/B4**，比原 B3/q6 的 M18 多算两行。该补齐是源码和
CPU dispatch 确认的事实；其延迟贡献仍需单独干预才能量化。

### 端到端对照与恢复

一次诊断重启，使用原 GPU2–6、PP5/TP1、CUDA 13、CED、RAM Engram
及 prepare-ahead。A_before 是旧固定 q6 服务；B 仅使用已修复的通用
Graph 候选构造，并在原 speculative config 中增加
`num_speculative_tokens_per_batch_size=[[1,1,5],[2,8,4]]`；A_after 删除
该 schedule，恢复原 argv/env，保留候选构造修复。
没有 diagnostic Worker、runtime profiling hook 或开发 RPC。

每个 arm 先做相同 32K/256 C1 固定参考检查，再对相同四个不同提示
做两次 C4，每请求输出 128 token。共 27 个请求。decode 使用共享探针
的客户端 decode 区间口径：合计排除每请求首 chunk 的 token，除以
所有请求首 token 到末 token 的联合窗口；它不是 GPU kernel 时间。
端到端包含 prefill，以 512 个输出 token 除以整波请求时间。

| arm | C4 decode tok/s，两次 | C4 端到端 tok/s，两次 | 实际验证 draft K | 接受 draft + bonus / 请求验证 |
| --- | ---: | ---: | ---: | ---: |
| A_before，固定 q6 | 106.61 / 109.32 | 100.50 / 103.60 | 5.000 / 5.000 | 2.977 / 2.977 |
| B，默认图网格动态 q5/q6 | 91.55 / 99.04 | 87.42 / 94.61 | 4.281 / 4.220 | 3.066 / 3.042 |
| A_after，恢复固定 q6 | 107.15 / 109.80 | 101.60 / 104.52 | 5.000 / 5.000 | 2.977 / 2.977 |

A 的四次 decode 均值 **108.22 tok/s**，B 均值 **95.30 tok/s**，
下降 **11.94%**。A_after 相对 A_before 的均值仅增加 **0.47%**，
支持这是候选退化而非单向基线漂移。该小样本不代表生产流量分布。
**拒绝默认网格的动态 q5/q6 配置，原固定 q6 保留。**

计数器来自前后 `/metrics` 差分：`num_draft_tokens_total/num_drafts_total`
是实际送给 target 验证的 K；`1+num_accepted_tokens_total/num_drafts_total`
按请求的候选组计，不是全局 scheduler step 的 token 数。B 的平均 K
确实变小，接受数量反而略升，因此原轨迹 `min(A,q)` 不能代替新的
端到端测量：不同 draft anchor 和调度顺序会改变后续接受分布。

不能从单层 MoE 的 11%–12% 节省推导整轮提速，具体差距有以下证据边界：

* benchmark 没有图补齐；候选默认网格的 B3/q5 会由 15 补到 20，
  而原 B3/q6 是 18。本轮没有采集 B 的逐步 batch/Graph 记录，不能
  将全部 11.94% 退化都归因于这个补齐。
* DSpark `num_speculative_tokens` 仍为 5，`num_query_per_req` 和 Markov
  head 的循环读取配置最大长度；propose 不读取
  `SchedulerOutput.num_spec_tokens_to_schedule`。少验证一个 draft
  不等于 draft 的 GPU 链少计算一个 query。draft 时间未单独计时。
* async scheduler 将选定 K 写入下一步 placeholder；PP 中已有请求
  可能携带不同 K。在 cohort 合并时可能出现不均匀 batch，当前
  FULL_DECODE_ONLY 会退回 eager。这是代码上的可达路径，本轮未测
  其发生频率或时间贡献。

三个 arm 的固定 C1 decode 为 **146.53 / 142.58 / 145.53 tok/s**，
3 份 256-token 输出均与原参考 ID 完全一致。不同提示通过共享
`one_request()` 的 token ID、usage、长度、finish reason 和 `[DONE]`
检查；诊断控制器本轮只持久化了汇总，没有保存其原始输出 ID，
因此不能用于不同提示的输出等价评测。保存缺口已在后续使用的控制器
中补上，实际运行版本另存为 `trial-e2e-tested.py.txt`，不追补本轮证据。

服务中断约 **12 分 50 秒**。最终父进程 **347443**，PP0–PP4 为
**350356/350358/350360/350362/350364**；原 argv/env 和物理 GPU
映射一致，health=200、开发 RPC=404、26662 无 listener。13 个运行
源码哈希中，仅 `gpu/cudagraph_utils.py` 是本轮保留的已验证修复，
其余与重启前一致。固定长度的候选表没有变化，**不宣称该修复给
当前固定 q6 带来速度增益**。

验证：定向 dynamic-SD、Graph manager、adaptive-verification 四个
CPU suite 共 **46 passed**；补充两个稀疏网格后，Graph suite
**8 passed**。固定候选表新旧对照、B5/q6 缺口复现、GPU 前缀及
poison/replay 检查通过。误纳入的 Qwen 模型级 Graph 集成 suite
已取消，不计入这些通过结果。

本轮完成通用候选修复并保留原生产配置。后续若继续验证长度优化，
先使用显式图网格控制 B3/q5 的补齐，再量化真实 cohort 的 uniform /
eager 比例；不能直接推广默认动态 schedule，也不能绕过 PP adaptive
verification 的一致性检查。
