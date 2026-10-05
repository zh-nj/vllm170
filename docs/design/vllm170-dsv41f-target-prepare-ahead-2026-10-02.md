# DSV41F 第七轮：PP header 接收后移

日期：2026-10-02。承接第六轮已启用的SWA metadata融合，优化target准备顺序。
本轮保留既有8 warp、packed-word、TileLang MoE与FP8 dense实现。
HEAD为`81175371741318fda60762f62052994be7c0bf31`，前几轮未提交改动保留。
原始产物：`artifacts/dsv41f-target-opt-round7-20261002/`。

2026-10-05 更新：根据用户要求，target prepare-ahead 改为默认开启。
直接 `vllm serve` 缺省该配置项时启用，PP5 launcher 也显式写入 true；
显式 `dsv41_sm80_target_prepare_ahead:false` 仍可回退。原有 DSV41F、
SM80、V2、PP5/TP1/DP1、DSpark5 和 adaptive verification 关闭的门控保持。
以下 false 默认值是 10 月 2 日验收时的历史记录；本次只修改默认值，
不新增端到端速度结论。

## 1. 端到端结果

同一加载实例中只改变target接收顺序，关闭所有phase/step timing插桩。
固定32768-token历史prompt、greedy、生成512 token、PP5/TP1、DSpark5。
每arm先完整warmup，再按AB、BA、AB、BA执行，共8个性能请求。
输出token IDs和所有spec counter增量与重启前生产参考精确一致：
88 draft rounds、440 draft tokens、423 accepted tokens。

| 单请求 | 关闭 | 开启 | 改善 |
| --- | ---: | ---: | ---: |
| post-first-chunk tok/s中位数 | 100.149 | 145.935 | +45.72% |
| client估算ms/轮中位数 | 57.988 | 39.790 | 减少18.197 ms |

四个block配对提升42.63%、48.08%、47.12%、42.79%，全部同向。
这是完整decode交付速度，包含target、sampling、draft和轮间等待，排除TTFT
及首个带token的chunk全部tokens；不是target kernel吞吐。
ms/轮是同一client span除以draft-round counter，首尾边界并非完全重合，
只用于相同接受计数和相同请求协议的对照。

随后用双请求检查推广价值：相同32K prompt、各512 token，两请求同时发起；
两个arm先各warmup一对，再做AB、BA，共4个性能pair、8个completion。
合并decode窗口从两个请求最早首token chunk到最晚末token chunk，分子为
两个请求各自排除首chunk后的token总数；这不是两条单请求TPS简单相加。
client start由外层wrapper记录，和`run_once`内部起点有微秒级偏差。

| 双请求合并client decode窗口 | 关闭 | 开启 |
| --- | ---: | ---: |
| tok/s中位数 | 125.611 | 170.964 |
| 两次原始样本tok/s | 126.027 / 125.196 | 170.249 / 171.678 |

提高36.10%，两个配对分别提高35.09%、37.13%。每pair的输出IDs及counter
一致：176 rounds、880 draft tokens、846 accepted tokens。
额外双请求32K/128回归也一致。没有测不同prompt混合、4/8请求性能或高温采样。
单请求、诊断和并发正式样本共核验9216个输出IDs，warmup不计入这个数。

端到端收益是接受候选和显式启用生产开关的依据，不以isolated kernel节省推算。
数据见`ab.json`、`concurrent-ab.json`和`analysis.json`。

## 2. 代码改动与数据依赖

原顺序在`vllm/v1/worker/gpu_worker.py`：

1. `irecv_tensor_dict`先同步接收上游Gloo tensor header，再创建/复用buffer并
   发起tensor receive。
2. 进入V2 runner，更新上一轮采样状态，准备输入和attention metadata。
3. runner读取`intermediate_tensors.tensors`，等待通信并copy/stage输入。
4. 前向、向下游发送、采样和draft。

已有`AsyncIntermediateTensors`只延后第3步的wait，没有延后第1步。
独立准备仍在header等待后执行。修改只延后启动receive：

- 新增可选`comm_start_callback`，首次公开tensor访问或显式`wait_for_comm`
  时才执行接收；随后仍按handle.wait→postprocess→complete callback完成。
- worker将原`irecv_tensor_dict`调用封装成partial。候选把partial放到lazy
  容器；关闭候选仍立即调用partial，保持原顺序。
- persistent receive buffer的acquire和generation不改；finish callback仍
  只在通信完成后触发一次，release和DSpark ownership保持原规则。
- runner前半段只读取scheduler、request states、上一轮采样广播、KV block
  table和seq lengths，不读取本轮上游hidden states。第一次tensor访问位于
  attention metadata完成后的input staging，此时仍等待全部通信依赖。

没有修改`GroupCoordinator`、payload、消息顺序、模型数学、采样、Graph内容
或tensor shape。所有消费方仍看到同一个接收dictionary和原持久tensor。
切片保持原tensor view，multi-group metadata和live/PAD规则保持。

开关为additional config的JSON boolean：
`dsv41_sm80_target_prepare_ahead`，默认false，拒绝字符串等非bool值。
加载期要求V2、实际CUDA SM80.0、model_type=deepseek_v41、H5120、E384、40层、
PP5/TP1/DP1、DSpark5、adaptive verification关闭；其他条件返回false。
设备能力在device/runner初始化后检查，不在worker构造前创建其他GPU context。

因此当前开关覆盖所有非首PP rank（PP1–PP4）、单请求与多请求，以及相同runner
里的prefill；首rank没有上游receive，行为不受这项顺序变更影响。
双请求输出和性能、冷/缓存32K prefill及真实模型warmup均已有证据；其他并发
和多模态输入没有本轮端到端验收，不把代码可用范围当成性能证明。

## 3. 诊断：最大的变化是依赖与等待

性能请求结束后，两个arm各做一个独立32K/256诊断请求。step timing只记录
CUDA Event和host marker，在request尾同步一次；另外记录target receive
调用的host起止，依据`recv_buffers`参数区分draft通道。每个非首rank恰好
46次target receive，包含prefill与45个q6/FULL step。
以下共同index3–40对齐，排除首段和最后边界，共38步。

| 对齐范围，中位数 | 关闭 ms | 开启 ms |
| --- | ---: | ---: |
| 五rank forward Event span之和 | 34.935 | 31.247 |
| PP0→PP4 host forward-start标记间隔 | 31.404 | 6.075 |
| PP4 drafter Event span | 14.650 | 4.097 |
| PP4 Event period | 59.891 | 40.422 |

forward span包括Graph staging/replay及到sample_tokens的结束标记，并非纯
kernel时间；已按每step对齐后求和，不能把该值当成精确target墙钟。
head drafter span包含relay、接收等待、准备和projection；中间rank的draft
span互相重叠，禁止求和或说draft kernel直接加速了72%。

| rank | metadata host关闭→开启 ms | metadata Event关闭→开启 ms | input Event关闭→开启 ms |
| --- | ---: | ---: | ---: |
| PP0 | 1.732→1.681 | 1.730→1.679 | 0.102→0.101 |
| PP1 | 3.584→1.921 | 3.578→1.920 | 0.320→5.737 |
| PP2 | 3.188→1.760 | 3.185→1.759 | 0.233→11.547 |
| PP3 | 2.822→1.787 | 2.820→1.785 | 0.244→18.005 |
| PP4 | 2.883→2.851 | 2.880→0.055 | 1.237→27.944 |

PP4的metadata host几乎不变，Event间隔却从2.880 ms降到55 µs。这说明
Event span会受GPU backlog和宿主enqueue时机影响，不能把差值解释成新的
metadata kernel提速；本轮没有改这些kernel。
input span明显变大，因为header/tensor等待被移动到首次实际消费的边界。
它不是新增工作，整轮period和无插桩交付速度才决定收益。

target `irecv_tensor_dict`的host调用中位数（全部46次，非纯网络时间）：
PP1 13.795→0.727 ms，PP2 17.196→2.270，PP3 22.394→3.586，
PP4 27.952→15.043。开启后调用更晚进入，仍等待真正上游依赖，较多准备
已经先完成；不能把各rank减少的receive时间相加称为通信节省。

数据支持：从“等header再准备”改为“准备后等数据”，下游CPU推进提前，
forward和draft边界上的暴露等待减少。DSpark draft stage原本就默认开启
prepare-ahead；其本轮收益来自target准备时机变化，没有新增draft开关或
更改relay协议。draft准备需要head传来的接受/拒绝和context，不能把target
的lazy receive机械套到所有draft消息。

## 4. 时钟、冷启动及实验控制

一次加载，在空闲状态用loopback dev RPC切`_defer_pp_recv`，两个arm的所有
metadata优化保持开启。正式性能请求没有插桩；unit的权重、PP分层、cache、
CED、KV FP8/2.7 GB、max-seqs8、max-batched2048、1M上限均相同。
七个runtime文件SHA在`launch.json`冻结，真实gate和buffer reuse在五个rank
的control inventory中确认。诊断临时unit随后完整停止，不作为生产部署。

单请求性能期间每秒采样GPU时钟。显存频率在两个arm相同，依次为
1728/1728/1458/1728/1458 MHz；PP0 SM1500、PP1 SM1470也一致。
PP2–PP4在候选中更常保持1410/1455/1410 MHz，基线较多1140或中间频率。
这是自动时钟策略下的实际serving收益，未设置任何时钟参数；没有做锁频
消融，无法把纯调度贡献和伴随的升频收益完全分开。forward Event减少的
3.69 ms也可能包含这个效应，不能全部归因于header依赖。

初次cold warmup的断言失败发生在候选关闭时，原脚本在保存row前断言，
未留该row，具体哪个字段不一致无法直接判定。之后先做baseline probe，
512个IDs和全部88/440/423 counters与原生产参考一致；修正脚本为先保存
再断言，并重新完整warm两个arm和执行campaign，没有重载模型。

已有第六轮artifact解释了为什么cold/warm不能要求同一接受计数：其cold
32K/512 production warmup的IDs与稳态一致，但accepted为425，稳态423；
第六轮诊断实例cold 256 warmup为213，稳态211。它们发生在本轮代码之前。
不把本轮丢失的首个warm row伪称为已经确认相同机制，也不据此改采样代码。
所有纳入性能和诊断的本轮样本仍严格要求输出和接受计数一致。

初次还因cleanup已关闭开关而重跑遇到initial-inventory断言，现改为在idle
时显式设初始mode，支持同一加载实例重跑。失败日志与cold control inventory
保留，后续运行使用`ab-retry2.log`；没有隐去失败或剔除性能离群值。

## 5. 正式生产、回归与回退

停止并清空诊断unit后，以原生产配置重新加载`dsv41f-manual.service`，
新增`dsv41_sm80_target_prepare_ahead:true`，保留
`dsv41_sm80_fused_swa_metadata:true`及原有其他开关。
生产没有dev mode、worker extension或artifact PYTHONPATH；phase和step
timing均为0。启动命令和七个runtime SHA存于`production-launch.json`。

正式生产三次32K/512稳态请求为147.092、147.348、148.363 tok/s，中位数
147.348 tok/s，client估算39.409 ms/轮。每次512个输出IDs及完整spec counter
增量与重启前参考一致，均为88/440/423。随后并发两条32K/128输出也与旧顺序
参考一致；并发结束后再发32K/512，IDs与counter仍一致，147.120 tok/s。
生产绝对速度是部署确认，+45.72%的收益来自上面的同实例A/B。

生产cold 512 warmup的输出IDs与参考一致，accepted为425，稳态为423，
position4为79→77；本轮已完整保存该row，再次观察到上一轮存在的cold/warm
接受计数差异。其底层原因未证实；不把cold sample计入稳态性能对照。

最终状态保存于`final-state.json`：health 200，模型root为
`/media/z/data6/models/DeepSeek-V4.1-Flash`，主进程667595，unit为active/running，
PP0–PP4 PID为670080/670082/670084/670086/670087，均归属同一cgroup。
物理GPU UUID和PP rank逐一核对为PCI顺序的2–6号CMP170HX。
26661由该主进程监听，26662已关闭，诊断unit MainPID=0，
`/collective_rpc`返回404；running/waiting request均为0。
七个runtime SHA与诊断和生产启动seal一致，原先所有tracked dirty diff完整保留。

回归复用`tests/v1/worker/test_pp_utils.py`，验证首次访问才receive、
wait/postprocess/complete严格顺序、重复访问只完成一次、切片view与buffer
generation，以及配置缺省、关闭、精确目标和非目标fallback。另用CPU假的
PP group执行真实`Worker.execute_model`，确认关闭时receive→prepare、
开启时prepare→receive，两者均在forward前完成wait/postprocess。

```bash
.venv/bin/python -m pytest tests/v1/worker/test_pp_utils.py -q
.venv/bin/pre-commit run --files vllm/v1/worker/gpu_worker.py \
  tests/v1/worker/test_pp_utils.py \
  docs/design/vllm170-dsv41f-target-prepare-ahead-2026-10-02.md
git diff --check
```

CPU套件31 passed，14条已有Torch JIT deprecation warnings；pre-commit的
ruff、format、mypy及适用hooks通过。服务输出回归另见`production-gate.json`，
这不代替不同prompt或高温采样的模型评测。该轮只改变通信启动时机，未改数学。

回退时只把生产启动参数中的`dsv41_sm80_target_prepare_ahead`设为false，
按原流程重启；原metadata和kernel优化仍保留。代码默认false，非目标配置
自动沿原路径执行。原worker、测试文件和unit都保存在`before/`和
`unit-before.txt`，无需回退整个脏工作区。本轮提交范围为worker、回归测试和报告。

本轮完成了同一PP5路径上PP1–PP4及单/双请求的推广；没有把此前失败的compact
relay或FP8 dense TileLang重新纳入。余下aligned forward Event span之和约
31.25 ms，是后续细分kernel与Graph边界的候选范围，尚不能当成纯计算时间。
此前33.87 ms截距来自另一轮q扫描，不能拿本轮各rank span直接拼成同一个截距；
本轮新增证据确认了可消除的准备串行与暴露等待，并已用完整推理流程验收。
