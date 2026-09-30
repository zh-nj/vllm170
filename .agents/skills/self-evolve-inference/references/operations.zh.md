# 自进化设施操作手册

本文档是 `self-evolve-inference` skill 的中文操作说明。它描述当前仓库中已经实现的 CPU 外部监督器、DSH 恢复通道和笔记约定。监督器负责进程生命周期与恢复；skill 负责提出一个有边界的假设、设计实验、读取结果和维护知识。两者共享文件状态，但不共享推理服务的虚拟环境。

生产接管的实时记录和命令见 [references/production.zh.md](production.zh.md)，状态以 `installation.json`、`adoption.json` 和 `state.json` 为准。本文不把 standby 或 CPU fixture 写成已接管。

## 1. 设施边界

### 1.1 监督器做什么

外部监督器使用 Python 标准库运行在独立的 uv 环境中。它通过用户级 systemd 管理自己登记的 server、job 和 resume transient unit，并持久化以下动作：

- 逐个登记不可变 release，执行声明的 build 和 check 命令。
- 对 release 的声明文件建立 `seal.json`，切换服务前重新校验 seal。
- 启动服务后先检查 `/v1/models` 中的模型 ID，再发送一个 `max_tokens=1` 的本地 completion，并要求 `completion_tokens > 0`。
- build 失败时保留正在运行的 stable；切换后启动、检查或观察失败时恢复 stable。
- 将请求、阶段、结果、故障现场和 DSH outbox 原子写入磁盘；控制器被杀掉后根据持久阶段恢复或回滚，不重复不确定的 DSH 请求。
- 只在独立 promotion 请求中把通过观察的 candidate 变为 stable。`ready` 不等于自动 promotion。

它不会替推理模型提出优化方向，不会自动判断 benchmark 是否有意义，不会重启主机、重置 GPU，也不会为恢复另起一个 GPU 模型实例。

### 1.2 skill 做什么

skill 在实验前读取现场和稳定 release，选一个可证伪假设，写入 checkpoint，准备候选 release 和请求；实验完成后读取 `result.json`、事件日志和 benchmark 证据，把事实、解释和下一步分别写入四类笔记。恢复通知是 notes-only：只解释现有结果、更新 `current.md` 和 `trajectory.md`，不得借恢复通知再提交实验。

### 1.3 服务和资源边界

| 资源 | 所有者 | 约束 |
| --- | --- | --- |
| CPU supervisor | 外部安装目录的 pinned Python 和用户 systemd | 与 checkout、vLLM `.venv` 分开 |
| 推理 server | controller 创建的带描述 transient unit | 只控制当前 root 的 hashed unit；发现端口被外部进程占用即拒绝 |
| 模型权重 | 运营环境的只读外部路径 | 不由 release 自动复制；在 lock/evidence 中记录路径、版本和摘要 |
| CUDA、驱动、GPU | 主机和驱动管理 | 不由 supervisor 编译、重置或替换；共享它们必须记录版本和 GPU 映射 |
| DSH | `dsh_bridge.py` 和独立 headless profile | 只做已完成实验的恢复解释；不作为第二个 supervisor |

CPU fixture、fake server 和 `verify.py` 只验证事务、systemd 归属、失败回滚和文件持久化。它们不产生 GPU 性能、模型正确性或真实生产可用性证据。

## 2. 仓库文件和职责

skill 目录保持短小，运行代码在 `tools/inference_evolution`：

```text
.agents/skills/self-evolve-inference/
├── SKILL.md                         # skill 入口、触发条件和最小工作规则
├── agents/openai.yaml               # 展示名称、默认提示和隐式调用策略
└── references/
    ├── supervisor.md                # 英文字段约束、安装和验证参考
    ├── operations.zh.md             # 本中文操作手册
    ├── mission.zh.md                # 明确启用、任务预算、生命周期和报告
    └── production.zh.md             # 生产接管、失败恢复和实机验证

tools/inference_evolution/
├── controller.py                    # Store、Systemd、Controller 和 CLI
├── lifecycle.py                     # skill 启停、事务互斥、有限任务期限
├── dsh_bridge.py                    # 调用 headless DSH、记录 session/receipt
├── dsh.patch.yml                    # 本地 dsv41f 路由和 notes-only overlay 模板
├── install.py                       # 复制 pinned package、建 uv 环境和 user unit
├── setup-supervisor.sh              # 使用仓库 .venv 调用 install.py 的薄包装
├── snapshot.py                       # 独立快照源码、uv/Python、pth 依赖和 native 产物
├── adopt.py                          # 记录旧进程并执行可恢复的生产切换/fallback
├── smoke.py                          # 对真实本地服务执行 completion/chat smoke
├── verify.py                        # 新实例 CPU/systemd 故障注入验收
└── tests/
    ├── test_controller.py           # 不分配 GPU 的事务单元测试
    └── fake_server.py               # CPU-only OpenAI 形状假服务
```

`controller.py` 的主要职责如下：

- `Store.initialize/register/submit` 校验配置、release 和不可变请求，并使用原子 JSON 与锁文件。
- `Systemd` 只接受描述为 `Inference evolution <root>` 的 controller-owned unit；server、job、resume 使用不同 slot 和 control group。
- `Controller.start_release` 重新校验 seal、停止自己拥有的 server、检查端口归属、启动并执行真实 token probe。
- `Controller.execute` 实现 build → seal → switch → check → observe → ready；候选失败进入回滚。
- `Controller.rollback/recover/tick` 实现恢复、运行期健康门、稳定版重启预算、blocked 和 outbox 派送。
- controller CLI 只有 `init`、`register`、`submit`、`wait`、`run`、`status`、`logs`、`pause`、`continue`。首次接管使用独立的 `adopt.py`，见 [生产接管说明](production.zh.md)。

## 3. 外部安装目录和状态目录

安装器把可运行代码放在 checkout 之外，并按内容哈希建立版本目录。典型安装目录如下；`<version>` 是安装器计算的 16 位 SHA-256 前缀：

```text
/home/z/.local/share/inference-evolution/
├── .venv/bin/python
└── versions/<version>/
    ├── controller.py
    ├── dsh_bridge.py
    ├── dsh.patch.yml
    ├── lifecycle.py
    ├── adopt.py
    ├── snapshot.py
    ├── smoke.py
    └── skills/self-evolve-inference/
        ├── SKILL.md
        ├── agents/openai.yaml
        └── references/...
```

用户级 unit 的名字来自安装参数，例如 `inference-evolution-dsv41f.service`。它只启动外部 pinned controller。controller 为一个 root 计算出 `inference-evo-<sha256(abs-root)[:12]>-{server,job,resume}.service`，这些 transient unit 的描述必须匹配该 root。

状态 root（例如 `/mnt/data/state/inference-evolution-dsv41f`）的持久布局如下：

```text
<root>/
├── installation.json       # root、pinned python、package、unit、version
├── config.json             # poll/resume 配置和历史探活默认值
├── control.json            # paused、origin_session、expires_at（任务期限）
├── state.json              # stable/current/phase/active、当前 lifeguard 等
├── events.jsonl            # fsync 的状态和事件时间线
├── recovery.log            # 稳定版恢复 server 的输出
├── adoption.json           # 首次接管阶段和原命令环境；私有，不共享全文
├── adoption-*.log          # 接管、检查及原命令回退日志
├── runtime/<id>/           # 每个 release 的可变 JIT 缓存
├── releases/<id>/
│   ├── release.json        # 进程、环境、probe、build/check、seal 声明
│   ├── seal.json           # manifest hash 与 sealed 文件摘要
│   └── <source、env、native-artifacts、lock 等 capsule 内容>
├── requests/<id>.json      # 已接受且不可改写的 experiment/promote/rollback
├── experiments/<id>/
│   ├── proposal.json       # experiment 请求副本
│   ├── result.json         # status、error、current、stable、完成时间
│   ├── build-*.log         # build job 输出
│   ├── check-*.log         # check job 输出
│   ├── server.log          # candidate server 输出
│   ├── resume.log          # DSH 恢复 turn 输出
│   └── incident-*.json     # 失败时的 state 和 server unit 现场
├── notes/
│   ├── current.md
│   ├── trajectory.md
│   ├── lessons.md
│   ├── directions.md
│   └── checkpoints/<experiment>.md
├── outbox/<experiment>.json # DSH pending/delivered/uncertain/superseded
├── dsh.json                # headless argv 和 cwd（若安装 --dsh）
├── dsh.patch.yml           # root notes overlay 和 pinned skill path
├── dsh-session.json        # 成功会话后复用的 session ID
└── dsh-receipts/*.json     # 每次 DSH 派送的 exit/status/final 证据
```

`state.json` 中的 `previous_stable` 只保留上一个 stable 供人工诊断。回滚目标始终是当前 `stable`，不会自动跳到 `previous_stable`。

## 4. 四类现场笔记

笔记是 `events.jsonl`、`result.json` 和 benchmark 事实的解释层。

### `notes/current.md`：现在发生什么

记录当前状态、正在处理的 experiment、稳定 release、最近故障、下一步和限制：

```text
# 现场
- 时间：2026-09-29T15:20+08:00
- state：phase=ready, stable=release-0017, current=candidate-0042
- 证据：experiments/exp-0042/result.json；events.jsonl 行号或时间
- 当前动作：等待人工读取结果，尚未 promotion
- 下一步：比较 baseline/candidate 的同一 workload 和方差
- 限制：CPU fixture 未验证 GPU；DSH 只允许 notes-only
```

### `notes/trajectory.md`：按时间记录决策

每个条目链接不可变 experiment/request ID：

```text
## 2026-09-29 — exp-0042
- 假设：降低 decode 数据搬运会减少 ITL
- checkpoint：experiments/exp-0042/proposal.json
- 阶段：build → switching → checking → observing → ready
- 决策：暂不 promotion，先复核正确性和尾延迟
- 结果：experiments/exp-0042/result.json
```

### `notes/lessons.md`：可迁移经验和适用条件

每条经验包含 evidence、scope、confidence、counterexample 和可复现入口：

```text
## L-0042：仅在 PP5、同一模型和同一 batch 上观察到
- evidence：benchmark.json、candidate server.log、baseline 对照
- scope：dsv41f / GPU UUID 列表 / PP5 / max-num-seqs=8
- conclusion：候选 P50 ITL 下降 4.1%
- confidence：中；只跑了 3 个重复
- counterexample：长 prompt prefill 尚未测量，不能推广到 TTFT
- reproduce：experiments/exp-0042/check-0.log
```

### `notes/directions.md`：候选方向和廉价证伪

按优先级记录下一个假设、预期收益、最便宜证伪方法和停止条件：

```text
## P1：检查 decode workspace 的分配抖动
- expected benefit：减少长尾 ITL
- cheapest falsification：固定 prompt/seq，先跑 CPU-free 静态检查和短 A/B
- acceptance：正确性不变，P99 至少改善 3%，显存余量不下降
- stop：任一 correctness gate 失败，或两次重复无改善
- prerequisites：candidate capsule、相同 GPU 顺序、稳定版可回滚
```

## 5. 安装、定位 pinned 运行时和查看状态

首次安装使用仓库的 `.venv` 包装脚本；它只负责调用外部安装器：

```text
tools/inference_evolution/setup-supervisor.sh \
  --root /mnt/data/state/inference-evolution-validation \
  --instance validation \
  --dsh /mnt/data/apps/deepseek-harness
```

安装只写入 unit，不启动也不 enable。DSH 任务开始时由 skill 用
`lifecycle.py start` 启动临时 CPU lifeguard，任务结束后用同一 helper stop。

安装器拒绝非空 root、已有 unit 和 checkout 内的 prefix/root，以避免覆盖旧状态。生产 root 目前的安装记录必须通过 `installation.json` 定位，不要猜测 package 版本：

```text
ROOT=/mnt/data/state/inference-evolution-dsv41f
INSTALLATION="$ROOT/installation.json"
EVOLUTION_PYTHON=$(jq -r .python "$INSTALLATION")
EVOLUTION_PACKAGE=$(jq -r .package "$INSTALLATION")
EVOLUTION_UNIT=$(jq -r .unit "$INSTALLATION")

"$EVOLUTION_PYTHON" "$EVOLUTION_PACKAGE/controller.py" \
  --root "$ROOT" status
"$EVOLUTION_PYTHON" "$EVOLUTION_PACKAGE/controller.py" \
  --root "$ROOT" logs --lines 50
systemctl --user status "$EVOLUTION_UNIT" --no-pager
journalctl --user -u "$EVOLUTION_UNIT" -n 80 --no-pager
```

若没有 `jq`，使用已知的 pinned Python 读取 JSON，而不是系统 Python；`installation.json` 中的 `python` 是最终执行 controller 的解释器：

```text
PINNED=/home/z/.local/share/inference-evolution/.venv/bin/python
EVOLUTION_PYTHON=$($PINNED -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$INSTALLATION")
EVOLUTION_PACKAGE=$($PINNED -c 'import json,sys; print(json.load(open(sys.argv[1]))["package"])' "$INSTALLATION")
```

用户明确启用本技能后，按 [任务协议](mission.zh.md) 写好 `MISSION` 目录和本轮
lifeguard 文件，再启动、暂停或恢复 unit：

```text
"$EVOLUTION_PYTHON" "$EVOLUTION_PACKAGE/lifecycle.py" \
  --root "$ROOT" --lifeguard "$MISSION/lifeguard.json" --ttl 3600 start
"$EVOLUTION_PYTHON" "$EVOLUTION_PACKAGE/controller.py" --root "$ROOT" pause
"$EVOLUTION_PYTHON" "$EVOLUTION_PACKAGE/controller.py" --root "$ROOT" continue
"$EVOLUTION_PYTHON" "$EVOLUTION_PACKAGE/lifecycle.py" --root "$ROOT" stop
```

`pause` 不会停止 watchdog，也不会停止已拥有的 server unit；它阻止新的实验和恢复 turn，活动实验在下一个边界回滚。`continue` 不会清除 `blocked`。停止 supervisor unit 本身也不会停止它单独拥有的 server unit，停止生产服务前必须查看 `status` 中的归属并使用明确的 recovery 流程。

## 6. Release 和实验请求

### 6.1 先建立独立 capsule

候选目录必须是自己的 `releases/<id>/` 子树，`cwd` 必须位于该子树中。源码、虚拟环境、native/编译产物和 lock/evidence 应复制到 capsule 的独立路径；不要在两个 release 间硬链接可变产物，不要修改正在运行服务加载的 `.so`。seal 遇到指向 capsule 外部的 symlink 会拒绝。

当前 controller 不会自动复制 Python 环境、CUDA 库或模型。新的 snapshot 设施负责复制 vLLM 源码、uv 环境、Python 基础运行时、外部 `.pth` 指向的依赖和编译产物，并把 JIT cache 放进独立目录；它仍然不会复制共享模型权重、CUDA toolkit、驱动、GPU 固件或 GPU 状态。操作者必须在 `dependencies.lock` 或单独验收记录中记录这些外部输入的路径、版本/摘要、tokenizer 和读权限。共享它们不等于 seal 已覆盖它们。

`tools/inference_evolution/snapshot.py` 是生产 capsule 的快照器：分析 editable vLLM/uv 布局，复制源码、解释器、环境和 native artifacts，重写 capsule 内的 editable path，并把外部 `.pth` 依赖列入快照清单；`tools/inference_evolution/adopt.py` 是生产切换器：捕获旧服务精确 PID/starttime/boot ID、argv、环境和 fallback，持有 daemon lock 执行阶段化切换，失败或中断时回原命令；`tools/inference_evolution/smoke.py` 执行真实 completion/chat smoke。接管命令、实际 release ID 和验收证据见 [生产接管记录](production.zh.md)。

### 6.2 注册候选

以下是字段完整的示意 JSON；命令必须是绝对可执行文件的 argv 数组，路径按实际 capsule 修改：

```text
{
  "id": "candidate-0042",
  "cwd": "/mnt/data/state/inference-evolution-dsv41f/releases/candidate-0042/source",
  "argv": [
    "/mnt/data/state/inference-evolution-dsv41f/releases/candidate-0042/env/bin/python",
    "-m", "vllm.entrypoints.openai.api_server",
    "--model", "/media/z/data6/models/DeepSeek-V4.1-Flash",
    "--served-model-name", "dsv41f", "--host", "127.0.0.1", "--port", "26661"
  ],
  "env": {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "2,3,4,5,6",
    "HF_HUB_OFFLINE": "1",
    "VLLM_CACHE_ROOT": "/mnt/data/state/inference-evolution-dsv41f/cache/candidate-0042"
  },
  "probe": {
    "base_url": "http://127.0.0.1:26661",
    "model": "dsv41f",
    "prompt": "Reply with one short token."
  },
  "build": [["/mnt/data/state/inference-evolution-dsv41f/releases/candidate-0042/env/bin/python", "-m", "compileall", "-q", "."]],
  "checks": [["/mnt/data/state/inference-evolution-dsv41f/releases/candidate-0042/checks/correctness"], ["/mnt/data/state/inference-evolution-dsv41f/releases/candidate-0042/checks/benchmark"]],
  "seal": ["source", "env", "native-artifacts", "dependencies.lock"],
  "startup_timeout": 900,
  "job_timeout": 1800
}
```

先把目录和 spec 写好，再用 pinned CLI 注册；注册不会启动服务：

```text
mkdir -p "$ROOT/releases/candidate-0042"
"$EVOLUTION_PYTHON" "$EVOLUTION_PACKAGE/controller.py" \
  --root "$ROOT" register "$ROOT/releases/candidate-0042/release.json"
```

注册后提交一个不可修改的 experiment 请求。`baseline` 必须等于 status 中的当前 `stable`；空实例首次 bootstrap 才使用 `null`。`checkpoint` 要能让恢复代理知道证据、边界和下一步：

```text
{
  "id": "exp-0042",
  "kind": "experiment",
  "baseline": "release-0017",
  "candidate": "candidate-0042",
  "hypothesis": "降低已测得的 decode 数据搬运成本",
  "checkpoint": "记录 workload、GPU/PP 配置、正确性门、性能门和回滚 release；结果只解释本次范围",
  "observation_seconds": 120,
  "lifeguard": {
    "startup_timeout": 900,
    "recovery_timeout": 900,
    "rollback_release": "release-0017",
    "probe_timeout": 20,
    "failure_threshold": 3,
    "stable_restarts": 1,
    "restart_policy": "restore_stable_then_block"
  }
}
```

```text
"$EVOLUTION_PYTHON" "$EVOLUTION_PACKAGE/controller.py" \
  --root "$ROOT" submit "$ROOT/drafts/exp-0042-input.json"
```

提交被 fsync 后，controller 会按 build、seal、切换、token probe、check、observation 顺序推进。完成后 `experiments/exp-0042/result.json` 的 `status` 可能为 `ready`、`rolled_back`、`build_failed`、`rejected` 或 `blocked`。读取 machine result 和日志，再由 skill 写笔记；不要把 `ready` 当成已接受的生产版本。

### 6.3 promotion 和 rollback 是独立请求

candidate 通过观察并仍在服务时，promotion 仍要单独提交，并留下明确接受理由：

```text
{
  "id": "promote-0042",
  "kind": "promote",
  "experiment": "exp-0042",
  "decision": "三次重复的 correctness 通过，P99 ITL 改善 4.1%；TTFT 未覆盖，接受范围限定为 decode workload"
}
```

promotion 成功后，`stable` 变为 candidate，旧 stable 写入 `previous_stable`。它只用于人工诊断；rollback 恢复的是当时的当前 `stable`，不是 `previous_stable`：

```text
{
  "id": "rollback-0042",
  "kind": "rollback",
  "reason": "candidate 在观察窗口出现 token probe 失败，恢复已封存 stable",
  "lifeguard": {
    "startup_timeout": 900,
    "recovery_timeout": 900,
    "rollback_release": "release-0017",
    "probe_timeout": 20,
    "failure_threshold": 3,
    "stable_restarts": 1,
    "restart_policy": "restore_stable_then_block"
  }
}
```

```text
"$EVOLUTION_PYTHON" "$EVOLUTION_PACKAGE/controller.py" \
  --root "$ROOT" submit "$ROOT/drafts/promote-0042.json"
"$EVOLUTION_PYTHON" "$EVOLUTION_PACKAGE/controller.py" \
  --root "$ROOT" submit "$ROOT/drafts/rollback-0042.json"
```

输入 JSON 应先完整写入 `drafts/` 等非队列目录，再调用 submit；不要手工写 `requests/`，controller 会扫描该目录。一个 release ID 或 request ID 一旦写入就不可改写；相同内容重复提交是幂等的，改内容会被拒绝。每次新的候选使用新的 release ID，不能在已有 seal 上重新 build。

## 7. systemd、日志和故障处理

### 7.1 查看归属和日志

安装 unit 和 transient unit 是两层对象：

```text
systemctl --user status "$EVOLUTION_UNIT" --no-pager
journalctl --user -u "$EVOLUTION_UNIT" -f
"$EVOLUTION_PYTHON" "$EVOLUTION_PACKAGE/controller.py" --root "$ROOT" status
"$EVOLUTION_PYTHON" "$EVOLUTION_PACKAGE/controller.py" --root "$ROOT" logs --lines 100
```

`status` 的 `server` 字段包含 controller-owned transient unit 属性。实验的 build/check/server 输出在对应 `experiments/<id>/`；稳定版恢复输出在 `recovery.log`。不要只看 `active (exited)` 判断模型健康，必须结合 `state.json`、server unit、`/v1/models` 和真实 token probe。

### 7.2 blocked

`blocked` 表示稳定版 seal 缺失或损坏、稳定版恢复失败、稳定版重启预算耗尽，或其他无法安全自动恢复的故障。blocked 时 controller 不再消费实验，也不会假装已恢复。操作顺序是：

1. 保留 `state.json`、`events.jsonl`、incident、result 和所有服务日志。
2. 停止新提交，检查 GPU 进程树、端口、unit 描述和模型路径的实际归属。
3. 用人工控制的 recovery 修复 capsule、环境或服务，完成真实模型和 token probe。
4. 只有证据完整且状态恢复后，才清理 blocked；实现清理命令时必须补测试和更新本文档。

`continue` 只清除 pause，不清除 blocked，也不自动选择 `previous_stable`。

### 7.3 uncertain outbox

`outbox/<id>.json` 为 `uncertain` 时，DSH 进程可能已经接受请求但本地进程在 receipt 写完前退出。不要重放同一 prompt，也不要提交第二个实验。读取对应 result、state、current note 和 receipt；让同一个已保存 session 的 notes-only recovery turn 解释结果。旧 pending 通知可被标记为 `superseded`，原始 result 永远保留。

### 7.4 DSH 任务与 lifeguard 生命周期

DSH 用户会话是优化任务负责人。默认 bounded 任务在 solved、infeasible-for-now、
预算耗尽、门禁失败或 blocked 后停止；只有用户明确要求持续自动优化时才进入
continuous 模式。任务目录、报告模板和续接规则见 `references/mission.zh.md`。

用户明确启用自进化技能后，skill 在首次受管实验前用 `lifecycle.py start`
和 `--lifeguard FILE --ttl SECONDS` 启动 CPU supervisor；任务结束
且 `state.active=null`、`phase` 为 `idle` 或 `blocked` 后用同一脚本 `stop`。stop
只关闭 lifeguard，不停止 stable server。任何实验切换、回滚、恢复或 build job 活动
时 helper 都拒绝 stop。continuous 模式在迭代之间保持 supervisor，直到用户停止或
任务 stop 条件触发。

每个 experiment 和人工 rollback request 都必须携带 `lifeguard`：
`startup_timeout`、`recovery_timeout`、`rollback_release`、`probe_timeout`、
`failure_threshold`、`stable_restarts` 和 `restart_policy`。controller 在服务
切换前将其 fsync 到 `state.json`，运行期自动 rollback 复制当前参数；没有明确
参数的重启决定会被拒绝。promotion 不重启服务，不需要该字段。

### 7.5 pause、重启和运行期故障

pause 仍让健康 watchdog 运行；正在执行的实验会在边界看到 pause 并回滚。controller 重启时先读取 durable phase：健康稳定版上的 build 中断会标为 `build_failed`，切换或恢复阶段中断会尝试恢复已封存 stable，不会重新执行未知 build 或假定 check 已通过。运行期 probe 连续失败时会先使用有限的 stable restart budget；预算耗尽则停止 server 并进入 blocked。

## 8. DSH 恢复通道

带 `--dsh` 安装会把 headless DSH 的 argv 写入 `dsh.json`，把 root 的 `notes` 作为可写 overlay，并把安装 package 的 skill 作为唯一自定义 skill 路径。`dsh_bridge.py` 通常由 controller 派送 outbox 调用，也可在明确授权的验收中调用；它验证 JSON `turn_end`、`final` 和退出码，并把 receipt 写入 `dsh-receipts/`。

恢复提示只允许读取一个 result、state 和 current note，更新 current/trajectory 后结束。它不能读历史会话来拼接上下文，不能修改 service，不能注册/提交实验。localhost API 需要 `EVOLUTION_LOCAL_API_KEY`（未认证本地服务可用安装环境注入的 dummy `local`）；真实凭据不得写进 tracked 文件。

## 9. 生产接管记录

首次接管命令、失败恢复、PID 身份保护、真实 completion/chat smoke 和 `verify.py --adoption` 验收见 [references/production.zh.md](production.zh.md)。接管完成后以该记录、`adoption.json`、`installation.json` 和 `state.json` 的实时证据为准；不要把 snapshot 已生成、CPU fixture 通过或 fallback_running 误写成 stable 已接管。

`previous_stable` 只供人工诊断，rollback 始终回当前 `stable`。`pause` 仍让 watchdog 运行，`continue` 不会清除 `blocked`；这些运行时约束以本手册和 `references/supervisor.md` 为准。
