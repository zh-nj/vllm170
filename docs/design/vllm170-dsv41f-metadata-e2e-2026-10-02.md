# DSV41F 第六轮：metadata 同进程端到端验收

日期：2026-10-02。保留8 warp sparse partial、W13 packed-word 和原 MoE
开关；FP8 dense 继续使用已有实现。本轮不修改推理源码，验收第四轮 token map
和第五轮 SWA metadata 融合，并在通过后显式启用生产开关。

原始产物：`artifacts/dsv41f-target-opt-round6-20261002/`。
HEAD：`81175371741318fda60762f62052994be7c0bf31`，已有未提交改动保留；
六个 runtime 文件的 SHA256 见 `launch.json` 和 `production-launch.json`。

## 1. 结论与证据范围

第五轮 SWA 融合取得了模型级端到端收益。第一次32K/256同进程三组对照中，
相对保留第四轮 token map 的基线，吞吐中位数提高8.51%；随后32K/512独立
复核提高6.17%，四个配对全部同向，约省3.76 ms/轮。接受生产开关的依据
采用较稳定的512-token复核，不把第一次短请求的8.51%当作长期保证。

本轮比较边界是单请求、相同32K token ID prompt、greedy、固定 DSpark5、
PP5/TP1、FULL Graph、CMP170HX SM80。没有测新的上下文长度、并发吞吐或
其他 GPU；双请求只验输出。没有用局部 kernel 加速推算整轮加速。

第四轮 token map 在同一加载实例中也观察到收益；这比之前跨重启的
86.735→86.243 tok/s 更能控制配置和加载状态，不能把两次实验直接相减。
本轮只重新确认这项候选，未重新消融更早的8 warp和packed-word改动。

## 2. 对照方法：一次加载，运行期间切 metadata 属性

进场时生产 unit inactive、26661无监听，GPU2–6没有模型进程。
07:49启动一次 `dsv41f-metadata-ab.service`，07:55 ready。配置取第四轮
`unit-before.txt`，保留模型、权重、PP分层8/8/8/8/8、DSpark5、CED、KV FP8、
2.7 GB KV/rank、max-seqs8、max-batched2048、1M上限和原 cache 路径。

临时诊断差异为loopback端口26662、worker extension、dev RPC。
phase profiler和周期 step timing均关闭。extension仅在running/waiting=0时，
同步设备后修改实际builder属性；遍历target和draft，不替换模型/Graph，
切换前后验证持久buffer指针不变。控制文件在artifact内，不引入推理源码。

| arm | MLA/compressor 单请求 map | SWA validity-out / ragged融合 |
| --- | --- | --- |
| A：pre_metadata | 关闭 | 关闭 |
| B：token_map | 开启 | 关闭 |
| C：fused_swa | 开启 | 开启 |

三个arm都保留此前SWA单请求map路径。A不是最早版本的完整回退。
诊断模型初始化时显式启用C；两个新优化均在Graph外构造metadata，
同一持久buffer由Graph消费。热切换只用于这个受控、空闲的实验实例。

prompt：`profiles/deepseek_v4_1/20260928_verify_sweep/q6/request/prompt_ids.json`，
32768 token，SHA256：
`89a3cca89df5fa7bc4f300260fce5ede41ea0659c9c8b6676353db890e01f6e1`。
复用 `dspark_flow_gate.run_once`，统计所有delta token IDs，包括非文本token。
主吞吐是首个带token的chunk之后的输出tokens/秒，排除TTFT及首chunk全部tokens。
表中ms/轮为同一client decode span除以Prometheus draft-round增量，
不是CUDA Event周期；span和counter的首尾边界并非完全重合，只作同协议对照。

每arm先完整256-token warmup，然后六个平衡顺序：
ABC、CBA、BCA、ACB、CAB、BAC，共18个无插桩请求。
逐请求比较token IDs和接受计数；A/B/C均与第四轮历史candidate输出精确一致。
每请求44 draft rounds、220 draft tokens、211 accepted tokens，口径保持不变。
不在性能请求中打开Event或nested profiler。

## 3. 端到端结果和独立复核

第一次三组对照，每arm六次请求：

| arm | tok/s中位数 | client估算ms/轮中位数 | 相邻增益 |
| --- | ---: | ---: | ---: |
| A | 86.619 | 66.908 | — |
| B | 91.773 | 63.153 | +5.95% |
| C | 99.585 | 58.234 | +8.51% |

C/A中位数比提高14.97%。六个block中B/A和C/B都同向，但短请求有明显波动：
C/B配对提高4.77%、7.99%、8.63%、9.54%、10.29%、20.89%。
该范围说明第一次样本不足以精确确定收益幅度。

因此只追加一次小范围B/C复核，不重启模型、不扫q：每arm完整512-token
warmup，四个配对顺序BC、CB、BC、CB，共8个512-token请求。
一次/秒采样运行时GPU时钟、功率、利用率；请求边界记录PSI和vmstat。

| arm | tok/s中位数 | client估算ms/轮中位数 |
| --- | ---: | ---: |
| B | 89.683 | 64.749 |
| C | 95.218 | 60.984 |

C/B提高6.17%，每轮减少3.765 ms；四个配对分别提高6.44%、6.49%、5.91%、
6.50%。512-token输出IDs和所有spec counters精确一致：88 rounds、440 draft
tokens、423 accepted tokens，因输出长度不同，不与256-token接受计数混比。

两arm在各卡的显存频率均相同（1728/1728/1458/1728/1458 MHz）；SM频率
分布相近，PP0 1500 MHz、PP1主要1485、PP3 1140，其他两卡主要1140并有短暂
上浮。不是锁频实验，但未观察到候选系统性升频。
宿主仍有memory pressure和少量swap/major faults。B/C平衡顺序和四次稳定
配对减小了趋势影响，没有证据支持把某一arm的偶发换页认定为全部收益来源，
也不声称宿主噪声完全排除。数据见 `confirm.json`。

另外，三arm各做一次双请求32K/128输出回归，两请求IDs在三个arm间一致。
这不是并发性能验收。18个主请求、3个诊断请求、6个并发请求和8个复核请求
共核验10240个输出token IDs；warmup不计入这个数。

## 4. 收益出现在哪里：宿主准备影响PP推进

性能campaign结束后，三个arm各追加一个独立256-token Event诊断请求。
只在request尾同步一次读取事件；每rank收集同一批46行，45行是q6/FULL，
按共同index对齐。以下取index3–40，排除首段和结束边界，共38个step。
这些诊断仍会扰动性能，不用来替代前面的无插桩吞吐。

| 范围 | A ms | B ms | C ms |
| --- | ---: | ---: | ---: |
| 对齐每step后，五rank forward span之和的中位数 | 35.444 | 35.612 | 35.479 |
| PP0→PP4宿主forward-start标记间隔中位数 | 37.264 | 36.747 | 32.088 |
| PP4 drafter Event span中位数 | 15.574 | 15.867 | 14.609 |
| PP4 Event period中位数 | 67.133 | 67.445 | 61.008 |

**forward没有可解释整轮提速的下降。** Event forward覆盖Graph输入staging、
replay及其到sample_tokens中的结束标记，PP发送/等待边界也会影响它；
它不是纯Graph kernel时间。已经逐step对齐再求和，没有把rank中位数之和
冒充对齐重放，也没有把这个span之和当成完整target墙钟。

各rank target metadata宿主耗时：

| rank | A ms | B ms | C ms | B→C减少 ms |
| --- | ---: | ---: | ---: | ---: |
| PP0 | 2.597 | 2.286 | 1.854 | 0.432 |
| PP1 | 5.072 | 4.491 | 3.672 | 0.819 |
| PP2 | 4.226 | 3.919 | 3.156 | 0.764 |
| PP3 | 4.106 | 3.733 | 2.995 | 0.738 |
| PP4 | 4.020 | 3.779 | 2.961 | 0.818 |

metadata的CUDA Event间隔几乎等于宿主marker间隔。它包含宿主enqueue空隙，
不能称为GPU metadata kernel耗时；第五轮隔离Graph中整个代表性组合仅数十µs。
这次主要减少Python/ATen/Triton调度和临时tensor操作，对serving宿主准备的
节省比隔离builder的0.306–0.336 ms大。并行worker、完整builder roster和宿主
压力都与隔离环境不同；尚未逐项确定放大原因，不给它们编造精确比例。

在相同宿主单调时钟上，C把PP4的forward-start相对PP0提前约4.66 ms，
head的draft span另减少约1.26 ms，与插桩period下降方向一致。
这些是不同边界的观测，不能机械相加作为无插桩3.76 ms的精确分解。
中间rank drafter span包含等待下游target/head/relay，彼此重叠，禁止求和。
PP4 input Event从0.291升至1.016 ms，提示某些等待挪到更早抵达的接收边界；
因此单阶段没有收益或增长也不等于整轮退化。

源码解释了下游metadata为什么并非必然全部被前向遮住：
`GPUWorker.execute_model`先调用`irecv_tensor_dict`，再进入runner准备metadata；
`irecv_tensor_dict`内部先同步`recv_object`取得PP tensor header。
上游`isend_tensor_dict`在runner返回后发出header。因此下游准备不是所有rank
从同一时刻开始。本次host标记明确呈现逐rank推进，减少准备开销影响后续
enqueue。这里既有真实overlap，也有依赖造成的暴露成本；上一版简单用
upstream forward长度证明metadata全隐藏的说法不适用于这次实际执行路径。

真实KV roster另外包含SWA四组、共享KV group中的MLA/indexer，以及部分rank
的compressor；PP2的共享group有两个独立MLA builder。Common map cache原本
已经复用持久buffer，没有发现“隐藏第二次copy”来解释全部收益。
SWA融合通过同一个builder同时作用于target causal和draft noncausal；仅测
target metadata、忽略draft会漏掉已有优化的传播范围。

旧q扫描的33.87 ms拟合截距属于另一个campaign，包含权重执行、attention、
metadata、PP依赖等固定成本，不是一个独立“入口kernel”。本轮确认其中存在
可优化的metadata/推进成本，但没有重扫q，不能据本轮q6测量重新给出截距。

## 5. 生产启用与回退

同进程验收后停止诊断unit，确认26662无监听、所有诊断worker退出和模型GPU
释放，再按原命令启动 `dsv41f-manual.service` / 26661。只合并新开关：

```json
{"ced_prefill":true,"dsv41_sm80_moe_tilelang":true,"dsv41_sm80_fused_swa_metadata":true}
```

生产不启用dev RPC、worker extension或artifact PYTHONPATH；phase和step timing
显式关闭。原model/PP/DSpark/CED/KV参数保持。生产source SHA与诊断实例一致。
启动manifest、日志、正式请求和最终unit/GPU核验保存在round6目录。

08:07:48开始正式加载，08:14服务ready，移除诊断扩展后完成独立验收：
一次512-token warmup，三次32K/512正式请求，随后双请求32K/128，再回到
单请求32K/512。四次单请求的output IDs及88/440/423 spec counters均与同进程
参考精确一致；两个并发请求IDs与此前参考一致，验证单→双→单状态迁移。

三次正式吞吐为96.966、95.606、96.298 tok/s，中位数**96.298 tok/s**，
client估算period中位数60.300 ms/轮。它与同进程C的95.218 tok/s相近；
没有在正式服务做B回退，不把跨加载的这1.13%浮动记作额外优化收益。
部署采用的6.17%仍来自同一实例的B/C对照。

最终 `dsv41f-manual.service` active/running，MainPID440582，EngineCore442629，
PP0–4 worker443655–443659，分别占用物理GPU2–6；实际模型路径一致。
health200，running/waiting均0。26661监听恢复，26662无监听，诊断unit已回收。
`/collective_rpc`返回404；实际进程没有dev-mode、worker extension或PYTHONPATH，
phase=0、step-timing-every=0。六个runtime文件SHA与实验冻结值一致，
详见`production-gate.json`和`final-state.json`。

开关仍默认false，仅本生产SM80配置显式启用。ragged融合继续限制1–48 token、
index-width≤256；其他形状调用原算法。validity-out仍使用live slot逐步更新，
无跨组mask共享。V4.0、其他硬件保持原路径。
回退将`dsv41_sm80_fused_swa_metadata`设为false后重新加载；原unit snapshot
和运行文件副本已保存。`start_production.py`保留完整可复现命令。

## 6. 验证和下一处热点

本轮未改runtime，source seal和第五轮最终代码一致，不重复运行已有65项GPU
kernel回归；第五轮独立reference、PAD/live迁移、Graph动态输入和fallback
验证继续有效。本轮新增的是模型级output/acceptance及实际端到端验收。
文档执行适用pre-commit hooks和diff检查；未提交、未推送。

剩余约35.5 ms的forward span仍没有下降，metadata减launch已经有整轮价值。
下一处有源码和marker支持的候选是把不依赖intermediate tensor header的
target准备提前到同步PP header receive之前；应先确认runner状态和Graph buffer
生命周期，再以同进程/相同输出的端到端对照验收。已有DSpark prepare-ahead
默认开启，不能把它再次记作新优化。旧compact relay没有收益并曾退化，本轮
不重新采用；减少包字节或局部launch不是充分的生产收益证据。
