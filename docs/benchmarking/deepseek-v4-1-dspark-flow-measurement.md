# DeepSeek-V4.1 DSpark 流程耗时计量

给 DeepSeek agent 的执行入口。运行目录为仓库根目录，Python 均使用 `.venv/bin/python`。
默认 PP5：target PP0–PP4，draft PP1/PP2/PP3，head PP4，同一主机。
脚本不停止或重启服务，不改模型或推理代码。

## 最便宜：已有 trace 离线分析

```bash
.venv/bin/python -m benchmarks.deepseek_v4_1.dspark_phase_measure \
  profiles/deepseek_v4_1/20260927_dspark_hotspots_head_eb4d/trace \
  --out-dir /tmp/dsv41-dspark-offline
```

得到 `analysis.json`、`stages.csv`、`report.md`，不占 GPU。
这条示例分析的是历史 `eb4d`、fast-prefill 开启的 trace，不能代表当前 CED 服务。
只可分析同一实例、同一次采样窗口的各 rank 文件，不能合并重启前后或 A/B trace。

## 已有服务：请求级速度和接受率

```bash
RUN=$(mktemp -d "$PWD/profiles/dspark-measure.XXXXXX")
.venv/bin/python -m benchmarks.deepseek_v4_1.dspark_phase_collect \
  --base-url http://127.0.0.1:26662 --out-dir "$RUN/request-only" \
  --prompt-tokens 32768 --max-tokens 256 --repeats 3
```

自动执行一次 32K/64 预热和三次 32K/256，保存 prompt IDs/SHA、完整输出 IDs、
流式到达时刻、TTFT、decode tok/s、逐请求 engine counter 差值。
已有固定 prompt 时添加 `--prompt-ids /absolute/path/prompt_ids.json`，数量必须匹配。
`--repeats 1 --max-tokens 64` 可用于连通性检查，不用于宣称稳定性能收益。

**API 只能给出请求级结果，不能从未插桩的存量服务获取内部阶段时间线。**
当前没有热注入 host-only profiler 的正式接口。下面复用已有 CUDA-event 插桩，
按 scope 限定采样；已有实例未加载 profile 环境时，需要在下一次启动时配置。

## 新采样：一次启动，重复使用 control 文件

先确认服务进程、监听端口和五卡占用；不要启动第二个 PP5 实例，也不要盲目执行全局 kill。
停止原服务后，在同一个 shell 运行以下配置。每次启动使用新目录；control 起初不存在。

```bash
RUN=$(mktemp -d "$PWD/profiles/dspark-measure.XXXXXX")
TRACE="$RUN/trace"
CONTROL="$RUN/profile_on"
mkdir -p "$TRACE"
git rev-parse HEAD > "$RUN/head.txt"
git status --short > "$RUN/status.txt"
git diff > "$RUN/source.patch"
export VLLM_DEEPSEEK_V4_PHASE_PROFILE=1
export VLLM_DEEPSEEK_V4_PHASE_PROFILE_PATH="$TRACE/phase-{pp_rank}-{pid}.jsonl"
export VLLM_DEEPSEEK_V4_PHASE_PROFILE_CONTROL_PATH="$CONTROL"
export VLLM_DEEPSEEK_V4_PHASE_PROFILE_FLOWS=dspark
export VLLM_DEEPSEEK_V4_PHASE_PROFILE_RECORD_FUNCTION=0
export VLLM_DEEPSEEK_V4_PHASE_PROFILE_STAGES='draft.propose_total,draft.stage.recv_wait,draft.stage.prepare_send,draft.stage.prepare,draft.stage.prepare_inputs,draft.context_kv,draft.stage.metadata,draft.stage.state_launch,draft.stage.state_wait,draft.stage.forward,draft.stage.send,draft.head.context_hidden,draft.head.send,draft.head.recv_wait,draft.head.result_copy,draft.markov_total'
ATTEMPTS=1 HOST_FREE_TARGET_GIB=145 \
  bash benchmarks/deepseek_v4_1/start_pp5_with_retry.sh "$RUN/server.log" \
  PORT_PORT=26662 PORT_MM_IMAGES=0 PORT_MAX_BATCHED=2048 PORT_MAX_SEQS=8 \
  PORT_KV_BYTES=1300000000 PORT_FAST_PREFILL=0 PORT_DSPARK_EAGER=0 \
  PORT_PREFIX_CACHING=0 PORT_CUDAGRAPH_MODE=FULL \
  'PORT_EXTRA_ARGS=--additional-config={"marlin_full_k":true,"dsv41_strict_bf16_gemm":true,"dsv41_strict_prefill_tail":true,"ced_prefill":true}'
```

启动成功后执行：

```bash
.venv/bin/python -m benchmarks.deepseek_v4_1.dspark_phase_collect \
  --base-url http://127.0.0.1:26662 --out-dir "$RUN/measurement" \
  --control "$CONTROL" --trace-dir "$TRACE" \
  --prompt-tokens 32768 --max-tokens 256 --repeats 3
```

流程：无 profile 预热 → 三次无 profile 基线 → 创建 control → 一次相同请求分段采样
→ `finally` 删除 control → 比较全部输出 IDs/接受统计 → 离线分析 decode 行。
control 不在启动或 graph capture 时创建；不用旧实验 `runtime_probe.py` 注入代码。
客户端异常会关闭 control；若进程被 SIGKILL，手动 `unlink "$CONTROL"`。

需要独占请求窗口：engine counters 是实例总量，其他请求会污染差值。baseline 与 profile
的接受统计或 token IDs 不一致时，脚本报错并保留 `request.json`，先排查而不报告速度收益。
该比较验证测量没有改变本次输出，不代替 target-only/语义评测。

profile 前必须保持 trace 目录无 JSONL。同实例再次采样可在 control 关闭、请求已结束后，
将上一批 JSONL **移动到独立归档目录**，保持服务配置的 trace 路径不变，使用新 `--out-dir`。
不要重用旧输出目录或删除原始证据。服务端 profile 环境在 import 时读取，客户端 export
不能给已运行服务启用插桩。

## 热点不够细时再加 scope

上面的首轮只测阶段。重启成本较高，若明确需要全细分，可在启动前设置：

```bash
export VLLM_DEEPSEEK_V4_PHASE_PROFILE_STAGES=''
```

空字符串表示所有已有 scope，仍只记录 `dspark` flow。可看到：

| 环节 | 主要 scope |
| --- | --- |
| head context | `draft.head.context_hidden` 及现有子项 |
| prepare | `draft.stage.prepare_inputs`、`draft.context_kv`、`draft.stage.metadata` |
| context KV | `draft.context_kv.input_gemm`、`.norm`、`.insert` |
| metadata | `target.attn_metadata.common`、`.builder.*`、`.swa.noncausal.*`、`.swa.direct_ragged` |
| relay | `draft.stage.prepare_send`、`draft.stage.send`、`draft.head.send` 下的 `.launch`、`.metadata_wait`、`.tensor_wait` |
| 状态接收 | `draft.stage.state_launch`、`draft.stage.state_wait` |
| query / head | `draft.stage.forward`、`draft.head.result_copy`、`draft.markov_total` |

`flow=dspark` 下的 `target.attn_metadata.*` 是 draft metadata。FULL graph 内部的算子
不会因打开这些 scope 自动变成逐层计量：这里的 forward/Markov 是 graph 边界。
需要图内 kernel 分析时另做针对性测量，不要为了表格完整改成 eager 后当作生产耗时。

## 读表与停止规则

- Host 是 Python scope 墙钟；CUDA 是同 stream event span，含 host 发射空洞，非纯 kernel 时间。
- 父子 scope 不可相加，三张卡的 prepare 有重叠；看 `prepare_host_union_ms` 和逐轮关键路径。
- `P1_P2_state_receiver_late_ms`：PP1 发起 state send 到 PP2 开始接收的正延迟。
  新路径使用 `state_launch` 起点，旧路径回退至 `state_wait` 起点；结束边界采用 `state_wait`
  的结束（host 已提交必要等待），不把异步 launch 结束当作张量在 GPU 上已完成。
- 新路径的 `state_launch` 嵌在 `metadata` 和 `prepare` parent 中。parent 变长可能是接收
  工作移入，不能据此断言 metadata kernel 变慢。
- `recv_wait` 包含上游工作和接收 readiness，不能当作纯 Gloo/NCCL 通信时间。
- `step_id` 是 profiler 本地计数，各 rank 不保证同号。脚本以 PP4 proposal 的同主机
  host 时间区间为边界，只接受每个 rank 唯一、完整落入该区间且未被复用的 prepare。
  `rank_step_mapping` 保留对应关系；不完整或有歧义的轮次列入 `excluded_steps`。
  请求流结束时流水线可能仍有尾部工作，control 打开边界可能截获上一请求的部分 rank。
  因此不能按同号或 ordinal 强行对齐。每项保留分母 `n`；多实例 PID 或重复 draft scope
  直接报错。阶段汇总含边界行，跨 rank 指标仅含完整轮次；该方法不适用于多机时钟。
- profile 开启时 `_send_relay_package` 使用 `isend_tensor_dict` 后分项等待，关闭时使用
  `send_tensor_dict`；当前后者内部也是 isend 后等待，但没有分项插桩。
  relay 分项反映采样路径，不能直接认作生产路径的逐项成本。
- profile 请求的 tok/s 仅为扰动观察；性能结论只使用 control 关闭的基线。
  `decode_ms_per_round` 的首包后时间和 draft counter 边界略有不同，是请求归一化指标。
- TTFT 包含排队/prefill/首轮采样，不能称作纯 prefill kernel 时间。

一轮交付：`request.json`、`phase/report.md`、`phase/analysis.json`、`phase/stages.csv`、
原始 trace、启动日志、HEAD 和 diff。先指出一个具体 PP/stage 的卡点，再进入单变量 A/B；
正常 A/B 即可，不默认 A/B/A，不改变 chunk=2048、CED、fast-prefill、q5 等无关变量。
启动失败保留日志并停止盲目重试，报告首个真实错误；没有 trace 时不能编造内部耗时。
