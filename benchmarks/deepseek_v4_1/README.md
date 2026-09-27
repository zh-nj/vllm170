# DeepSeek-V4.1 benchmarks

DSpark 分段计量入口：
[低成本执行指南](../../docs/benchmarking/deepseek-v4-1-dspark-flow-measurement.md)。

- `dspark_phase_collect.py`：复用运行中的服务，预热、固定请求、接受率/IDs 比较及可选 profile 窗口。
- `dspark_phase_measure.py`：离线生成每 rank/stage 耗时和 PP5 关键路径表。
- `dspark_flow_gate.py`：单独请求级流式 token IDs/engine counters 计量。

从仓库根目录用 `.venv/bin/python -m benchmarks.deepseek_v4_1.<模块名> --help` 查看参数。
分段采样要求服务启动时加载 profile 环境；客户端不能给未插桩的服务热启用内部计时。
