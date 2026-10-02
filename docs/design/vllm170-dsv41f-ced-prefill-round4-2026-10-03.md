# DSV41F prefill 第四轮：恢复 DSpark 默认并复核推广条件

日期：2026-10-03。基线提交 `dabf780273` 已提交此前整个工作区的
26 个文件；本轮用户明确要求 **DSpark 默认开启**，取代此前为普通
decode / prefill 实验而关闭 DSpark 的运行配置。

原始记录：`artifacts/dsv41f-ced-prefill-round4-20261003/`。

## 默认配置与实际服务

`benchmarks/deepseek_v4_1/run_port_pp5.sh` 已默认
`PORT_DSPARK_TOKENS=5`、`PORT_DSPARK_EAGER=0`、`PORT_CED_PREFILL=1`。
本轮以替代可执行文件捕获启动 argv，无模型加载，确认默认包含
DSpark-5、分布式 draft ranks `[1,2,3]`、draft Graph 和 CED。
无需为已有默认值再改启动脚本。证据为 `launcher-default-argv.json`。

实际 `dsv41f-manual.service` 是此前实验恢复的 transient unit，显式
不传 `--speculative-config`。本轮停止该 unit，确认旧计算进程和
26661/26662 listener 已清理，再使用原封存参数添加：

```json
{
  "method": "dspark",
  "num_speculative_tokens": 5,
  "draft_sample_method": "greedy",
  "moe_backend": "mxfp4_int8_sm80",
  "enforce_eager": false,
  "enable_adaptive_verification": false,
  "dspark_pipeline_stage_ranks": [1, 2, 3]
}
```

原模型路径、served name、端口26661、PP5/TP1/DP1、分层8/8/8/8/8、
物理 GPU2–6、CED、prefix cache、KV FP8/2.7 GB、2048/8 scheduler、
1M 上限、生产缓存和其他 additional-config 保持原值。
`production-dspark-launch.json` 保存新旧完整命令和源码摘要。
后续正常恢复使用 DSpark 开启配置；关闭只用于明确要求的对照实验。

## 恢复后的交付检查

实际进程 argv、engine 初始化日志和接受计数均确认 DSpark 已启用。
MainPID为2361309，五个 worker 为2364033–2364037；五个目标 GPU
计算进程均属于该 unit，空闲验证卡没有计算进程。health=200，
dev RPC=404，无 worker extension、dev mode、PYTHONPATH 或性能插桩。
封存的17个源码摘要与当前文件一致。本轮只进行一次生产重载。

| 检查 | 结果 |
| --- | --- |
| 两次32K/256单请求 | 各256个输出 IDs 与此前 DSpark-off 参考一致 |
| 预热后单请求客户端吞吐 | 97.8969 tok/s；首个含 token-ID 的 chunk 之后计时 |
| 同一请求的 speculation 计数 | 44轮 draft，提出220个 token，接受211个 |
| 双请求各128 token | 两路 IDs 均与此前参考一致 |
| 冷周期32K/16 | TTFT 4.0695 s，零 prefix hit，16 IDs 一致 |
| 同盐 prefix32K/16 | 命中32512 token，TTFT 0.5477 s，16 IDs 一致 |
| 冷代码语料32K/64 | TTFT 5.1742 s，零 prefix hit，64 IDs 一致 |

上述检查共864个输出 IDs 一致。恢复后的时间是 fresh-process 交付记录，
不是同进程 paired A/B，不由此宣称新的优化收益或精确归因 TTFT 差值。
证据为 `production-dspark-validation.json` 和两份 validation log。

Prefix检查最初沿用了无 DSpark 时32640的期望值，因此在命中32512时
触发断言；这是验证假设错误，服务请求正常完成且 IDs 一致。
源码的 CED重算下限先保留128 token，DSpark又经过 `use_eagle()` 的
最后 cache block 退出策略，额外退掉128 token，得到
`32768 - 128 - 128 = 32512`。相关代码为
`scheduler.py:add_request`、`kv_cache_manager.py:get_computed_blocks`、
`single_type_kv_cache_manager.py:find_longest_cache_hit`。
初版脚本/日志另存，使用原响应完成确认，没有重复该 prefix 请求；
后续脚本已修正期望值，没有改推理缓存策略。

## 上轮混合 prefill 的推广缺口

第三轮的 TTFT 收益只验证了 **CED on、DSpark off、M=2048**。
当前源码有两个独立限制，不能直接把收益归入新生产配置：

1. `dsv41_sm80_hybrid_prefill_enabled` 要求
   `config.speculative_config is None`。DSpark 开启后不构建混合路由 scratch，
   即使 additional-config 中保留 hybrid=true，也会回退到原 CUTLASS 路径。
2. forward dispatch 要求 `m == 2048`。DSpark-5 是 parallel drafting，
   `max_num_new_slots_for_drafting = 5 - 1 = 4`；scheduler 对每个请求的
   input budget 预留这些槽位。因此单请求 eager prefill 的预算上限为
   **2044**，初始 PP 小 chunk 和 prompt tail 还可能更短。

第二点是代码推导，不是新采集的逐层 shape trace。V2 runner 对没有匹配
Graph 的 batch 使用原 token count，而当前 Graph capture 上限小于2044，
所以不能假设大 prefill 会自动补回2048。证据位置：
`config/speculative.py:max_num_new_slots_for_drafting`、
`v1/core/sched/scheduler.py:schedule`、
`v1/worker/gpu/cudagraph_utils.py:CudaGraphManager.dispatch`、
`v1/worker/gpu/model_runner.py:prepare_inputs`。

也不能只删除静态限制或把 dispatch 数字改为2044：`HybridRoutes.gemm`
使用实例的 `self.slots` 作为有效 slot 数。若仍按2048×6构建实例，
2044×6路由中的 padding sentinel 会被误判为有效 slot。
推广必须让实际有效 slot 数、padding sentinel、输出容量与 native GEMM
检查保持一致，并保留 draft E128/top-k3、decode Graph、混合 batch 的回退。
本轮没有修改这些推理门，也没有声称 DSpark 配置已获得 hybrid 收益。

## 小尾块方向的有限研究

原计划比较 BM128 pairs/BM64 tails 与增加 BM16 小尾块的方案。
使用第三轮已经捕获的实际路由重算：只将奇数 BM64 尾块中有效行数
不超过16的块改为 BM16，128配对和其余64尾块不变。

| 首个完整 chunk 的20个 encoder layer | 周期输入 | 异质代码输入 |
| --- | ---: | ---: |
| 可缩小的尾块 | 419 | 3284 |
| 原 hybrid padded rows | 297408 | 520768 |
| 加入 BM16 后的 padded rows | 277296 | 363136 |
| padded rows 减少 | 6.76% | 30.27% |

逻辑 weight tile 数不减少；候选还会增加 metadata copy 和每个 GEMM stage
的一次调用。padding 减少不等于相同百分比的耗时收益。
已有 CUTLASS prefill BM64/BM128 无寄存器 spill，不能机械套用此前
attention 的4→8 warp优化。

用户更改默认配置后，该离线实验暂停；未运行 GPU benchmark，未部署
BM16 候选。未运行草稿保存在 `small-tail-benchmark-unrun.py.txt`，
成本重算保存在 `route-costs.json`。原 benchmark 文件已恢复到提交版本。

下一轮的可执行方向是先验证当前 DSpark 的 eager prefill 输入形状，
正确推广已成功的混合分块，再比较小尾块；验收仍需同进程交替端到端
对照和输出 IDs 检查，不重新扫描 q，也不恢复 FP8 dense TileLang。
