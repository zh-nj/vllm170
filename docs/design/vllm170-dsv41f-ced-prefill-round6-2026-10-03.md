# DSV41F prefill 第六轮：BM16 小尾块与 persistent 调度

日期：2026-10-03。配置保持 DSpark=5、CED 开启，基线是第五轮
已上线的 M=2044 BM128 pairs / BM64 tails；包含尚未提交的第五轮适配。
原始记录：`artifacts/dsv41f-ced-prefill-round6-20261003/`。

## 路由证据与候选

第五轮代码语料的 20 个 encoder layer 首个 M=2044 chunk 共包含
3280 个奇数 BM64 尾块，其有效 slot 数为 1–16。将这些尾块改为 BM16，
其余 BM128 pairs 和 BM64 tails 保持不变，可将 padded rows 从
520256 降至362816，即减少30.26%；逻辑 M tile 数不减少。
这不是实测 HBM 字节，不能将 padding 降幅当作耗时降幅。
`route-costs.json` 保存同一批实际路由的计算。

`SmallTailRoutes` 从原 BM64 align 元数据出发，先配对同专家相邻块，
再检查奇数尾块的第17个 slot 是否为 sentinel。只有该位置已经 padding
时，才把前16个 slot 转给 BM16。始终使用全局有效 slot 数作为 native
GEMM 的上限，保留 W2 的原 routed slot 索引。metadata 构建在设备端，
没有为了读取尾块数增加 CPU 同步。

## 微测先发现的退化

先使用原 native dispatch，在物理 GPU1 CMP170HX 上比较当前 hybrid
与增加 BM16 tails 的候选。实际 corpus layer0/14 的路由 IDs、权重
来自第五轮 M=2044 捕获；激活和 expert packed weights 为合成数据。
另测六个热专家的合成 M=2044 路由，确保没有小尾块时也计入新增开销。

测量边界包括 route repartition、metadata copies 和一个完整 W13/W2
stage；不包括初始 BM64 align、权重构建、分配或编译。CUDA Event +
Graph，每次图含4次调用，3轮交替对照，各7个样本。权重和地址复用，
是 warm-cache 微测，不作为生产收益。

原 dispatch 在代码路由上改善17%–22%，但六热专家的 W13/W2
分别从3652.35/1981.70 µs 增至3709.18/2044.16 µs。即使 BM16
尾块数为0，仍多付约57/62 µs。证据：`uncapped-small-tail.json`。

源码解释该量级：native entry 用全局 `num_valid_tokens=12264` 和 E384
计算 BM16 capacity，`max_m=12264+384*15=18024`，即1127个 M tiles。
BM16 原使用非 persistent 二维 grid，W13/W2 的 N tiles 为36/40，
所以零有效尾块仍发出40572/45080个 CTA，之后在设备端提前返回。
仅缩小 Python scratch 会违反 native capacity 检查，不能解决该问题。

## 调度修正与再次验证

复用 kernel 已有的 persistent 分支：在设备端从 `num_post_pad` 读取
实际 tile 数，最多发出 `2*SM=148` 个常驻 CTA，循环处理实际 tiles。
没有新写 MMA 或 epilogue。native 门限定 top-k6、BM16/BN128、
有效 slot 数2044×6、E384；原 BM128 条件不变，其余 BM16 路径不变。
两种 GEMM stage 同时采用该修正。

| 路由 / stage | 当前 BM128/BM64 µs | BM16 + persistent µs | 变化 |
| --- | ---: | ---: | ---: |
| corpus layer0 W13 | 10703.87 | 9300.22 | −13.11% |
| corpus layer14 W13 | 9458.18 | 7848.19 | −17.02% |
| corpus layer0 W2 | 5713.41 | 4901.89 | −14.20% |
| corpus layer14 W2 | 5053.95 | 4148.22 | −17.92% |
| 六热专家 W13 | 3652.61 | 3662.08 | +0.26% |
| 六热专家 W2 | 1984.77 | 1999.87 | +0.76% |

persistent 在稀疏代码路由中比原 BM16 二维 grid 略慢，但将空列表
退化降至约9–15 µs，适合作为端到端验收候选。最终是否部署由冷输入
同实例对照决定。证据：`persistent-tail.json`。

## 正确性、构建与边界

微测的18个独立参考用例覆盖 W13/W2、M=1/16/17/63/64/65/127/128/129、
变化 activation/channel scale、slot 不丢失或重复及输出 sentinel。
实际 M=2044 路由的两种 stage 均与既有分块输出逐位一致。

扩展现有 kernel suite，增加 tiny metadata 的 hot→spread→hot Graph
replay，以及生产 `Experts.apply` 的 small-tail 分支。完整 apply 同时
检查 W13、SILU、W2、reduce、output/workspace alias，覆盖热路由、
少量新专家形成小尾块和分散路由；输出及全部 W2 slots 逐位一致。
**36 passed, 247 deselected**，测试只使用空闲 GPU1。

已遵循增量构建流程，仅编译 `mxfp4_int8_moe_entry.cu` 并链接 MoE
extension；实际该文件命令含 `compute_80/sm_80`，不能从全局
CMakeCache 的75推断其架构。先通过独立进程预加载 build 目录的
extension，保留生产安装的 .so；测试通过、旧服务停止后再原子替换。
源码、测试和 benchmark 的全部 pre-commit hooks 通过。

```bash
cmake --build cmake-build-release --target _moe_C_stable_libtorch -j 1

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python benchmarks/kernels/deepseek_v41_moe_hybrid_prefill.py \
  --small-tail --expected-uuid GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
  --out artifacts/dsv41f-ced-prefill-round6-20261003/reproduce.json
```

新开关 `dsv41_sm80_moe_small_tail_prefill` 默认 false，必须显式 JSON
true，且已有 hybrid gate 接受 DSpark-5/M2044 后才使用。继承第五轮
精确模型、硬件、PP/TP/DP、scheduler、pure eager prefill 门；不支持的
speculation、M=2048、decode、draft、mixed batch、Graph 和 ubatch
继续原路径。scratch 在加载阶段构建并预热，未改变 workspace lane。
关闭此开关即可保留第五轮 BM128/BM64 的收益。

## 端到端验收方法

同一私有 PP5 实例运行两条生产路径：baseline 将 route scratch
临时切换到预热的 `HybridRoutes`，candidate 使用 `SmallTailRoutes`。
两条路径都调用原生产 `Experts.apply`，共同使用同一 native extension；
native 调度新增条件只作用于 candidate 的 BM16 调用。计时关闭 Event
profiling，每个请求使用唯一盐并 assert prefix hit=0。

预设验收为：异质代码32K的四组配对全部改善且 TTFT p50 改善超过2%；
周期32K的 p50 退化不超过1%，任一配对退化不超过2%。同时检查两类
输入各20层原始 apply 输出有限且逐位一致、输出 IDs、decode、双请求
和 prefix cache。私有对照后恢复正式 DSpark-5/CED 服务。

## 同实例结果

四组 AB/BA/AB/BA 均已完成，`accepted=true`。基线为第五轮的
BM128/BM64，而非更早的纯 BM64。所有计时请求输出 IDs 与参考一致。

| 冷32K输入 | 当前 hybrid TTFT p50 | 小尾块 TTFT p50 | 改善 | 四组配对改善范围 |
| --- | ---: | ---: | ---: | ---: |
| 周期 prompt | 3.76554 s | 3.73383 s | 0.84% | 0.54%–1.07% |
| 异质代码 | 4.93523 s | 4.56146 s | 7.57% | 7.38%–7.79% |

每类输入每个路径的完整 M=2044 chunk 共300次 encoder MoE 调用，
PP0/PP1/PP2为120/120/60次；baseline 使用 `HybridRoutes`，candidate
使用 `SmallTailRoutes`，W13/W2命中次数相等。PP3/PP4无该路径命中。
两类输入各20个 encoder layer 的原始 apply 输出均有限且 BF16
逐位一致；129/2049/2050 token 请求无小尾块路径命中，输出一致。
证据：`experiment.json`、`experiment.log`。

TTFT 为客户端到首个含 token-ID 的 SSE chunk 的时间，包含主机处理、
prefill 与首 token 交付。微测的13%–18%改善只针对 GEMM stage；完整
流程还包含 attention、dense/shared expert、Engram、PP 和采样，因此
代码输入的端到端收益为7.57%。周期输入观测改善仅0.84%；第四轮
M=2048 路由中其小尾块仅419个，是收益差异的解释线索。本轮没有重新
捕获周期输入的 M=2044 路由，不能严格量化路由机会与新增 launch 成本
各占多少；不将0.84%外推为普遍改善。
没有用不同模型加载实例的历史 TTFT 相减，也没有与第五轮百分比相加。

随后完成长 decode 与并发保存检查：baseline/candidate 的两次256 token
及双请求各128 token，共1024个输出 IDs 与参考一致，单请求 DSpark
的 draft/接受计数也完全相同。证据：`private-preservation.json`。

## 正式生产交付

正式 `dsv41f-manual.service` 已恢复，显式开启
`dsv41_sm80_moe_small_tail_prefill=true` 和既有 hybrid flag。保持
DSpark=5、CED、draft ranks `[1,2,3]`、greedy、draft Graph、prefix cache、
模型路径、物理GPU2–6、PP5/TP1/DP1、2048/8 scheduler 和原 KV 参数。
生产无诊断扩展、dev mode、PYTHONPATH 或性能插桩。

| 正式服务验证 | 结果 |
| --- | --- |
| 两次32K/256单请求 | 各256个 IDs 与既有参考一致 |
| 预热后单请求 decode 客户端吞吐 | 99.956 tok/s，首个含 token-ID chunk 后计时 |
| 同一请求 speculation 计数 | 44轮，提出220 token，接受211 token |
| 双请求各128 token | 全部 IDs 一致 |
| 冷周期32K/16 | TTFT 3.73153 s，prefix hit=0，IDs 一致 |
| 同盐 prefix32K/16 | TTFT 0.55308 s，命中32512 token，IDs 一致 |
| 冷代码32K/64 | TTFT 4.52652 s，prefix hit=0，IDs 一致 |

共864个输出 IDs 一致。正式服务时间是 fresh-process 交付检查，
不从不同实例的 decode 吞吐或 TTFT 相减来宣称新收益；优化幅度以
前面的同实例 AB/BA 为准。

最终 MainPID=4007178，五个 worker 为4010330–4010334。目标 GPU
计算进程全部属于正式 unit；GPU1验证卡无计算进程。17份封存运行源码
及 native extension 摘要匹配；逐个检查 worker 的 `/proc/PID/maps`，
确认映射当前已验收 .so 的 inode，而非仅验证磁盘文件。
native SHA256 为 `f9b55f65e8feab78eb5394bfc467fd5990399e86b3b0cb6e2f688778df2ecd1b`。
health=200，dev RPC=404，私有 unit 已移除，26662无 listener，
running/waiting 请求数均为0。

证据：`production-launch.json`、`production-validation.json`、
`production-validation.log`、`production.log`、`final-state.json`。
本轮只有一次私有模型加载、一次正式生产恢复；构建先在独立进程验证，
没有覆盖运行中的 native 文件。未扫描 q，也没有追加 profiler 运行。

回退时仅将小尾块 flag 设为 false 并按封存参数重启，保留第五轮
BM128/BM64、DSpark=5和CED。完全回退 native 的旧 .so、源码及构建
目录产物已分别保存为 `*.installed-before`、`*.before`、`*.build-before`。
本轮代码、测试、benchmark 和设计文档与第五轮适配共同组成提交范围。
