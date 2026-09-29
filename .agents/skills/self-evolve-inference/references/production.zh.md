# 生产接管与恢复

## 接管边界

首次接管由操作员授权并执行 `adopt.py`。日常实验继续使用 `controller.py`
队列；DSH 的自动恢复会话只维护笔记，不自行开始下一轮实验。

本项目在 `evolution/dsv41f` 上开发候选改动；生产服务从外部 capsule 运行，
不会跟随 checkout 分支或 HEAD 自动改变。Git 提交、release ID 和请求 ID 应一起
写入实验记录。controller 的 promotion 只更新运行时 stable，不会自动合并 Git
分支；反过来，合并代码也不代表生产配置已经通过验证。

`snapshot.py` 适配本项目 Linux/uv CPython 3.12 editable 环境。它复制 Git HEAD
源码、当前 `vllm/`（包括编译产物）、虚拟环境、Python 基础安装及显式指定的
外部 `.pth` 依赖，修正 editable 路径、脚本 shebang、原生库 RPATH，然后封存。
因此应先确认推理源码干净，不能把未审查的工作区源码冒充已提交版本。
需要 `cp`、`git`、`tar`、`patchelf`。不得硬链接可变产物。

JIT 缓存写入 `runtime/<release>/`，不属于不可变构建产物；模型权重、CUDA
toolkit、GPU 驱动和系统库仍共享，实验期间不得修改它们。该快照不是整机镜像。
第一次迁移可能重新编译图和加载 Engram，需预留磁盘和启动时间。

## 首次接管操作

先从状态目录读取 `installation.json`，得到独立解释器、安装包和 supervisor
unit。不要在生产机重复运行安装器覆盖已有实例，它会拒绝覆盖。

```bash
EVOLUTION_ROOT=/mnt/data/state/inference-evolution-dsv41f
EVOLUTION_PYTHON=$(jq -r .python "$EVOLUTION_ROOT/installation.json")
EVOLUTION_PACKAGE=$(jq -r .package "$EVOLUTION_ROOT/installation.json")
EVOLUTION_UNIT=$(jq -r .unit "$EVOLUTION_ROOT/installation.json")
```

先检查实际端口 PID、完整进程树、模型、argv、环境和 GPU UUID。保留当前
completion/chat 输出。使用新的 release ID 创建快照：

```bash
"$EVOLUTION_PYTHON" "$EVOLUTION_PACKAGE/snapshot.py" \
  --root "$EVOLUTION_ROOT" --repo /mnt/data/apps/vllma100/vllm170 \
  --id NEW_RELEASE_ID --pid ACTUAL_API_PID \
  --external /mnt/data/apps/vllma100/DeepSelect
```

快照 ID 不允许复用。创建后从快照 cwd 检查 import 来源和原生库路径，再执行
真实探活。`seal.json` 已生成后不能修改 release；修复应创建新 ID。

停止空载 supervisor 不会停止原 GPU 服务。随后准备接管日志：

```bash
systemctl --user stop "$EVOLUTION_UNIT"
"$EVOLUTION_PYTHON" "$EVOLUTION_PACKAGE/adopt.py" --root "$EVOLUTION_ROOT" \
  prepare --release NEW_RELEASE_ID --pid ACTUAL_API_PID
```

`prepare` 要求空实例、无历史队列、PID 确实拥有目标监听端口、真实 token 探活
成功且封存完整。它记录原进程的 PID、启动时间、boot ID、子进程和原始 argv/env。
`adoption.json` 含完整原始环境，必须保持私有，不能贴进共享日志或 DSH 提示词。

使用独立 CPU systemd 任务运行 `adopt.py ... run`，以便 LLM 下线后继续执行。
例如（每次接管使用独立 unit 名）：

```bash
systemd-run --user --unit=inference-evolution-adopt-dsv41f \
  --property=Type=exec --property=Restart=on-failure --property=RestartSec=5 \
  --property=UMask=0077 \
  --property="StandardOutput=append:$EVOLUTION_ROOT/adoption-run.log" \
  --property="StandardError=append:$EVOLUTION_ROOT/adoption-run.log" \
  "$EVOLUTION_PYTHON" "$EVOLUTION_PACKAGE/adopt.py" --root "$EVOLUTION_ROOT" run
```

接管任务独占 `.daemon.lock`。它先持久化阶段，向已登记身份的原 API 发 TERM，
等待子进程退出，必要时只 KILL 同一身份的残留进程。使用 pidfd 避免 PID 复用；
不杀终端 cgroup，不杀整个进程组。

随后由 supervisor 的独立 `server` unit 启动快照。模型列表、真实 token、
release 声明的检查和 60 秒观察均通过，再验证封存并写入 stable，启动 watchdog。
首次接管接受的是原生产配置的运行基线，不代表性能提高。

## 失败与中断

| `adoption.json` 阶段 | 含义和处理 |
| --- | --- |
| `prepared` | 已保存现场，尚未切换 |
| `stopping_original` / `starting_capsule` | 正在切换；不要同时启动第二个控制器 |
| `adopted` | 已通过接管门禁并提交 stable；查看 `state.json` 和服务实时状态 |
| `restoring_original` | 恢复保存的原启动命令 |
| `fallback_running` | 原命令恢复并探活成功，但不是独立封存基线；状态保持 blocked |
| `blocked` | 接管前检查失败或自动恢复失败；保留日志后人工处理 |

任务崩溃后再次运行同一日志：若 stable 已提交则完成日志，否则只恢复原命令，
不会重放未完成的候选切换。回退启动日志为 `adoption-fallback.log`。
fallback 依赖原工作区和原环境，因此在接管完成前不得清理它们。

`continue` 不会清除 blocked。不要手工把 `stable` 改成一个未验证 release。
先确认实际监听者和 GPU 所属进程，停止 controller 后修复明确故障；保留原日志，
用新实例或经过审查的状态迁移再接管。不要删除运行中的 release 或 runtime 缓存。

## 验证与维护

CPU 接管验证必须使用新的独立实例；分别安装两个空实例后运行：

```bash
.venv/bin/python tools/inference_evolution/verify.py \
  --root /mnt/data/state/inference-evolution-adopt-success --adoption success
.venv/bin/python tools/inference_evolution/verify.py \
  --root /mnt/data/state/inference-evolution-adopt-failure --adoption failure
```

它们在随机 loopback 端口使用 CPU fake server，验证原进程退出、独立 unit
接管，以及失败恢复原命令；结束后停止测试服务。不能当作 GPU 接管证据。

supervisor 更新需发布新的外部版本目录，保留旧包和配置备份；只在 idle、无活动
实验及 DSH 写入时停止 controller，切换 unit、resume argv 和 skill overlay 到
同一包版本，daemon-reload 后启动。不要覆盖正在运行的版本目录。
`installation.json` 是定位入口，不要把旧版本号写死在日常脚本里。

本机 `loginctl show-user z -p Linger` 为 yes，可在退出桌面后维持用户服务。
整机重启后的 GPU 恢复仍依赖模型盘、驱动和原环境可用；未做整机重启验收。
watchdog 的 `stable_restarts=1` 是当前稳定版本累计的恢复预算，重启 controller
不会重置它；新版本 promotion 后清零。预算耗尽会停止服务并 blocked，避免无限
重载。保留实验结果和当前/前一 stable；清理其他 release 前先核对 state、队列、
adoption 和实际进程引用，不能只按目录年龄删除。当前未提供自动磁盘 GC。

## 本次接管记录

工作分支：`evolution/dsv41f`；合并基线：`8c285117f5`。
原 main 已保留为 `backup/main-before-evolution-20260929`。
生产状态根：`/mnt/data/state/inference-evolution-dsv41f`。
生产接管已完成（2026-09-29 Asia/Shanghai）：`adoption.json` 为 `adopted`，release 为 `dsv41f-baseline-20260929-r2`；`state.json` 为 `phase=idle`、`stable=current`，supervisor unit 已 active。

验收证据：`/mnt/data/state/inference-evolution-dsv41f/production-acceptance.json` 和仓库 `artifacts/inference-evolution-production-20260929/production-acceptance.json`。真实 completion/chat 输出与接管前相同，5 个 GPU worker UUID 与原服务相同，快照 worker 未映射旧工作区库；启动耗时约 6 分钟，包含权重、Engram 和 CUDA Graph 初始化。

未验证：GPU 故障注入回滚、整机重启恢复、完整 1M 上下文、性能提升。当前 stable 代表可恢复的已验收运行基线，不代表优化收益。
