# DeepSeek-V4.1 benchmarks

DSpark 分段计量入口：
[低成本执行指南](../../docs/benchmarking/deepseek-v4-1-dspark-flow-measurement.md)。

- `dspark_phase_collect.py`：复用运行中的服务，预热、固定请求、接受率/IDs 比较及可选 profile 窗口。
- `dspark_phase_measure.py`：离线生成每 rank/stage 耗时和 PP5 关键路径表（draft 链）。
- `dspark_round_timeline.py`：离线把 `dspark` 与 `decode` 两个 flow 合成整轮预算
  （head target 前向、拒绝采样、draft 链、上游 target 跨度与残差等待）。
- `dspark_flow_gate.py`：单独请求级流式 token IDs/engine counters 计量。
- `step_timing_log.py`：把 server 日志的低扰动 step-timing 行解析成每 rank 的
  target/draft/prep/period 预算，剔除非 decode 窗口。
- `verify_cost_model.py`：target 验证 q=1..6 的字节/带宽模型与实测曲线拟合
  （斜率、进入验证路径的固定代价、隐含新专家数）；结论见
  [q 扫描与代价模型](../../docs/design/vllm170-dsv41f-verify-q1-q6-2026-09-28.md)。

从仓库根目录用 `.venv/bin/python -m benchmarks.deepseek_v4_1.<模块名> --help` 查看参数。
分段采样要求服务启动时加载 profile 环境；客户端不能给未插桩的服务热启用内部计时。
