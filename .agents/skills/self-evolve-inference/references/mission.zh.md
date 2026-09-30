# DSH 优化任务协议

用户在 DSH 会话中明确指示“使用 `self-evolve-inference` skill 优化本项目
dsv41f 的某个方向/解决某个问题”时，当前会话就是任务负责人。skill 需要读取
当前 stable、现场笔记和项目代码，实际设计候选、提交实验、解释结果，并最终在
同一个会话返回报告。

必须由用户明确要求使用本技能或“自进化”才能启用。普通“优化推理”“排查慢请求”
不触发本技能，不启动维生程序；`agents/openai.yaml` 已关闭隐式调用。DSH 即便
不识别该元数据也必须遵守此入口约束。已授权任务中的“继续”沿用原目标和预算，
恢复事件只能解释已有结果，不能创建新任务或自动进入 continuous。

没有明确要求持续时，使用 bounded 模式。它可以在同一个问题上进行多轮小实验，
但问题解决、当前证据下不可解决、预算耗尽、正确性/安全门失败或 supervisor
blocked 后必须停止。不得把“还有别的优化方向”当成继续理由。

只有用户明确说“持续自动优化”“持续迭代”“一直寻找下一个收益点”等开放指令时，
才使用 continuous 模式。该模式允许一轮结束后选择下一个有排序的方向，但仍需
有预算、停滞阈值、失败门和状态报告；用户说停止时立即停止在当前安全边界。

## 任务文件和 lifecycle

为每个任务生成不重复的 ASCII mission ID，例如
`dsv41f-decode-20260929-a1`，并在下面建立文件：

```text
/mnt/data/state/inference-evolution-dsv41f/notes/missions/<mission-id>/
├── brief.md          # 用户目标、模式、预算、接受条件、停止条件
├── checkpoint.md     # 下一步和故障后恢复入口
├── trajectory.md     # 本任务的 experiment/promotion/rollback 顺序
├── report.md         # 最终报告；任务未结束前标记 in-progress
└── raw-index.json    # 原始 result、benchmark、日志和 commit 路径
```

在首次受管实验前从 `installation.json` 读取 pinned Python/package，写入本次
启动与恢复参数，再执行。只读分析和 notes-only 恢复不需要启动维生程序：

```bash
ROOT=/mnt/data/state/inference-evolution-dsv41f
PINNED=$(jq -r .python "$ROOT/installation.json")
PACKAGE=$(jq -r .package "$ROOT/installation.json")
MISSION="$ROOT/notes/missions/dsv41f-decode-20260929-a1"
mkdir -p "$MISSION"
jq '{startup_timeout:900,recovery_timeout:900,rollback_release:.stable,
     probe_timeout:20,failure_threshold:3,stable_restarts:1,
     restart_policy:"restore_stable_then_block"}' \
  "$ROOT/state.json" > "$MISSION/lifeguard.json"
"$PINNED" "$PACKAGE/lifecycle.py" \
  --root "$ROOT" \
  --lifeguard "$MISSION/lifeguard.json" --ttl 3600 start
```

结束时只在 `state.json` 的 `active` 为 null 且 `phase` 为 `idle` 或 `blocked` 时
执行：

```bash
"$PINNED" "$PACKAGE/lifecycle.py" --root "$ROOT" stop
```

以上时间是示例，应按本轮负载、基线实测冷启动时间和剩余预算填写。`rollback_release`
只能在空实例 bootstrap 时为 null；已有 stable 必须指向当前 stable。`--ttl` 是
本次任务剩余秒数，默认 3600、最大 86400；不是每轮自动刷新的预算。到期会停止
新工作，必要时回滚未提升候选，在恢复完成或 blocked 后正常退出，保留 stable
服务。恢复可超过 TTL，但受 `recovery_timeout` 限制；正常任务仍应主动及时 stop。

这个 stop 只关闭 CPU lifeguard，不停止 stable server；如果有 active 实验、切换、
回滚或恢复任务，helper 会拒绝关闭。continuous 模式跨迭代保持 supervisor active，
直到用户停止或 stop 条件触发。不要用 `kill` 或直接停 server unit 替代 lifecycle。

`phase=ready` 时先根据证据提交 promote 或带明确 lifeguard 的 rollback，再 stop。
若 DSH 会话在重启期间丢失，重新进入原会话读取 checkpoint/result，沿用原预算，
不要启动第二个模型或复制实验。截止时间已过时写出 stopped 报告。

## brief 与每轮边界

`brief.md` 至少包含：

```markdown
# Mission <mission-id>
- user_request: 原文或不改变含义的短摘录
- mode: bounded | continuous
- target: dsv41f / PP5 / GPU UUID 或当前 stable
- direction: 一个可证伪问题
- acceptance: correctness、性能和稳定性门禁
- budget: wall time、实验次数、GPU 资源和可接受重启次数
- stop: solved、infeasible、budget、stagnation、gate-failed、blocked、user-stop
```

用户没有提供数值预算时，先采用一个保守的单轮预算并在 brief 中写明假设，
不以无限重试替代预算。continuous 模式必须额外写 `max_iterations` 或
`no_improvement_limit`；缺失时不能进入开放循环。

每一轮只对应一个 hypothesis 和一个 immutable candidate/request ID。开始前写
checkpoint，记录 baseline、候选差异、工作负载、正确性 gate、性能 gate、观察窗口、
rollback release 以及完整 lifeguard 参数，并在 request 中原样保存。本轮的
`rollback_release` 必须等于 baseline。启动成功不等于解决，单个 kernel 或短请求改善
不等于 dsv41f 端到端改善。

提交后在同一个 DSH 会话等待 durable result，不重复提交：

```bash
"$PINNED" "$PACKAGE/controller.py" --root "$ROOT" register "$RELEASE_JSON"
"$PINNED" "$PACKAGE/controller.py" --root "$ROOT" submit "$REQUEST_JSON"
"$PINNED" "$PACKAGE/controller.py" --root "$ROOT" wait "$EXPERIMENT_ID" --timeout 3600
```

`wait` 超时只表示等待预算用尽，请读取同一个 request/result 路径并继续原任务，
不得生成第二个 ID 代替它。读取 result、日志和 benchmark 后，skill 写入本任务
`report.md`，在安全边界执行 lifecycle stop，再把报告返回当前 DSH 会话。

## 报告和停止

最终 `report.md` 和 DSH 回复采用以下结构：

```markdown
# Mission report: <mission-id>
- status: solved | infeasible-for-now | stopped | blocked
- stop_reason: 精确停止条件
- target/baseline: 模型、commit、release、GPU、PP/TP、环境

## Experiments
| id | hypothesis | change | correctness | performance | result |
## Evidence
- raw result/log/benchmark paths
- measured facts and variance
## Interpretation
## What remains unknown
## Decision
```

`infeasible-for-now` 只表示当前代码、硬件、预算或门禁下未能证明可行，不能写成
数学上不可能。continuous 模式每轮追加 trajectory，并报告累计收益、剩余预算和下
一候选方向；没有达到 stop 条件不得声称最终解决。

controller 派送的 outbox recovery prompt 是 notes-only：只解释已有 result/state，
绝不把恢复通知当作用户新的 continuous 授权。用户重新明确提出任务后才创建新的
mission ID；续接旧 mission 时保留原模式与预算，不重放 uncertain 请求。
