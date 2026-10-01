# DSV41F SM80 MoE 评价设施审查修复

范围：修复 [代码审查](vllm170-sm80-moe-harness-code-review-2026-09-30.md) 的 R1–R8。
只修改离线 benchmark/helper、对应 pytest 和文档，不修改生产 kernel、模型分派、
环境默认值或服务配置。本次没有实施新的 TileLang GEMM 优化。

## 修改

| 审查项 | 修复与验收行为 |
| --- | --- |
| R1 scale | 默认 `--scale-mode varied`，分别变化 token/slot 与 expert/channel scale；保留 `unit` 模式。GEMM 统一按整数 dot→FP32→A scale→channel scale→BF16 精确比较。GPU sparse fixture 加入非单位 scale 和 BF16 舍入边界；CPU 负例拒绝漏乘 scale、slot 错用 token、错 expert。 |
| R2 状态 | 默认启用 reference；只有所有行 correctness=pass、measurement=ok/not_applicable 且 run_status=complete 才能通过。每行立即写 JSONL，summary 原子更新；异常记录 aborted 和 error，保留已完成行。`--no-check-reference` 仅供诊断，summary 不通过并返回非零。 |
| R3 实际路由 | 对每个 tile、每条实际计时路由做独立 eager 和 replay 验证；每条路由使用独立输出，避免后一次调用掩盖前次错误。replay 前填 NaN sentinel；错误/no-op 必须失败。记录每条 route hash 和完整 sequence SHA256。轮换标为 sequential-rotation，reduce 明确标为固定输入。 |
| R4 复现 | 权重和两组 A 使用从 seed/shape/stage 派生的独立 generator，不依赖扫描顺序。记录权重、A、scale、routing weights 内容 SHA256；路由另存 hash。每轮 UUID 唯一，显式 `--round`。 |
| R5 字节 | 每调用按专家去重后求字节，再对周期取均值；工作集按 W13/W2 分别统计专家并集。两模式的总 graph calls 相同。权重工作集和输出 buffer 数/字节分开记录。 |
| R6 reduce | 以逐输出 `sum(abs(slot * weight))` 定义局部误差预算，零项明确处理。允许最终 BF16 相邻舍入差及 FP32 累加误差；calc_diff 只作局部模式的诊断项，避免近相消时误拒。小输出被清零负例失败，正确零结果和可解释相消差异通过。 |
| R7 身份 | dirty hash 包含 staged/untracked；另存 driver/helper、相关分派/对齐源码、CUDA 源码和扩展二进制 SHA256。哈希不能代替提交，但能区分实际运行输入与实现。 |
| R8 额外分配 | 仅以专家数、shape 和 element_size 求逻辑字节，不发出 index_select，不复制权重。越界 ID 报错。 |

reduce 阈值改为 `reduce-local-fp32-bound-v2`：最多 6 项时，对比 FFMA 与分别
乘加路径采用 `gamma_6 + gamma_12` 的保守界，局部余量为 `2**-19`，另计一次
最终 BF16 spacing。该界针对有限、正常 FP32 运算；本轮 synthetic fixture 满足
条件。历史全局阈值的经验数据不作为此局部规则的校准证据。

## 使用与兼容

JSONL schema 升为 **2**。保留 `--output` 的 legacy eager/graph_event 结构，
其中 eager 明确是第一条路由的 warm-fixed 诊断计时。`make_gemm_call` 接口保持兼容。
helper `time_graph` 新增可选 `validate_graph`，不要求其他离线调用方改变参数。

完整验收使用默认 reference 和 varied scale；例如：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python benchmarks/kernels/deepseek_v41_mxfp4_int8_sm80.py \
  --experts 8 --topk 3 6 --tokens 1 --block-m 16 --block-n 64 128 \
  --cache-mode rotating-experts --scale-mode varied --round 1 \
  --warmup 1 --repeats 1 --graph-calls 3 --graph-samples 3 \
  --output-jsonl /tmp/dsv41f-sm80-validation.jsonl \
  --summary-json /tmp/dsv41f-sm80-validation-summary.json --quiet
```

设备 UUID 是本机此次空闲卡；跨机器执行时替换为已确认空闲的 SM80 UUID。
JSONL 为追加写入，必须按 run_id 筛选；summary 只描述当前一次执行。
同路径开始新运行时先将 summary 标为 running，不能读取上次成功作为本轮结论。

## 验证与边界

最终结果：CPU **49 passed / 31 skipped**；空闲 SM80 上 **80 passed**；helper
自检 `all_ok=true`；相关文件 pre-commit 全部通过。

| CLI smoke | 结果 |
| --- | --- |
| warm-fixed，varied scale，M=1/4 | 72 行，exit 0，all_passed=true |
| rotating，varied scale，M=1 | 68 行，exit 0，all_passed=true |
| warm-fixed，unit scale，M=1 | 36 行，exit 0，all_passed=true |
| 显式关闭 reference | 12 行，exit 1，all_passed=false，符合预期 |

另外确认：warm/rotating 的 M=1 输入 hashes 在独立进程间一致，四轮 run ID
互异，所有已验证计时图的 `calls_per_graph=3`。

结果和精确命令保存在 `artifacts/moe-harness-fix-20260930/`；包含 CPU/GPU pytest、
helper 自检、pre-commit、warm/rotating/unit/unchecked CLI smoke 与跨进程输入比较。
本轮使用空闲 SM80 UUID `GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216`，
未停止或重启 PP5 服务。

这些短测证明修复后的评价路径可执行，不用于排名 kernel 性能。专家数为 8，
覆盖 top-k 3/6、M=1/4、16x64/16x128 tile 与轮换 wrap；未重建 E384 三轮完整基线，
没有加载模型、没有模型质量或吞吐提升结论。最终测量前应按 schema 2 重建同输入
基线，保留旧 artifacts，不能把 schema 1 与 schema 2 的轮换数字直接合并。

历史实现报告中的“单位 scale 覆盖缩放顺序”“固定工作集超过 L2 即无复用”、
以及仅凭连续读取速度判定 DRAM 延迟/occupancy 瓶颈等表述不构成已验证结论。
kernel 优化仍需要匹配输入、缓存模式与调用数的对照测量。
