# DSV41F SM80 TileLang grouped GEMM：PRMT 实现与复测

日期：2026-09-30。基于 `32f3db5e7d6b375914bc15cb5b1ca5a7273f489e` 的工作区修改。
承接 [代码审查](vllm170-dsv41f-sm80-tilelang-gemm-code-review-2026-09-30.md)。

## 结果与范围

已实现 TileLang 内嵌 CUDA PRMT 字级解码，并修复影响搜索可信度的设施问题。
候选保持离线：没有改动 `vllm/`、`csrc/`、环境默认值、backend 选择或模型服务。
此次不是完整模型推理提速证明；不接入生产。性能出口以如下固定/轮换复测共同判断，
不能仅采用搜索阶段的最快数字。

设备为唯一空闲的 CMP170HX（SM80，70 SM），UUID
`GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216`。生产 PP5 的五个 GPU 进程保持运行。
TileLang 0.1.12；版本、源码/扩展 SHA256、参数及 GPU 环境见每个 JSONL header。

## 实现

[候选内核](../../benchmarks/kernels/deepseek_v41_moe_tilelang_gemm.py)新增 `--decode prmt`：
从原 packed tile 取四个字节，CUDA helper 通过 `__byte_perm` 进行 magnitude/sign 解码，
得到八个 INT8，再按原布局写入 MMA fragment。字节提取采用无分支形式。
权重仍在 kernel 内解码；没有在计时外展开全量权重，也没有更换格式或近似语义。

INT8×INT8→INT32、activation scale 后 channel scale、单次 BF16 舍入、slot 写回保持不变。
非单位 scale 的 oracle 与候选/基线逐位一致；BF16 检查用原始 16-bit 位模式，区分正负零并拒绝非有限输出。

`num_stages` 现在实际控制 copy 与 consumer 的流水距离。1/2/3 不再都固定成双缓冲；
通过生成源码 SHA256 识别重复配置。本轮 E8 的 BM16/32 × stages1/2 × W13/W2 产生
8 个不同源码，全部通过。深度 3 的独立测试也通过，搜索未显示其更优。

最终 W13 配置为 BM16/BN64/BK128、stages2、128 threads；W2 为相同 tile、64 threads。
W13 独立 nvcc 重编译（SM80、register-usage-level=10）的资源与指令检查：
76 registers、0 stack/spill，SASS 包含 PRMT 和 `IMMA.16832.S8.S8`；
PTX 包含 `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`。
详见 `code/prmt-final.*` 和 `code/ptxas-final.log`。这些是编译证据，不是运行时指令占比。

## 复测

E384、M1、top-k6；seed=20260930 未参与搜索（搜索 seed=4242），变尺度。
正式采样前双方 Graph 交替预热至少 500 ms，预热不计入结果，也不修改 GPU 时钟。
每组 A/B 交替 3 轮，每轮 7 个原始 Event 样本；每个 Graph 都包含 **40 次实际 GEMM 调用**。
固定模式重复同一专家组，轮换模式完整覆盖 40 组路由、240 个不同专家。
双方先经过独立 oracle，再将全部输出填 NaN，重放实际计时 Graph，并逐位检查整个周期。
不把编译、输入构造、reference、校验或 hash 计算计入延迟。

| stage | 工作集 | CUTLASS µs | PRMT µs | 延迟降低 | 基线轮间 spread | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| w13 | warm-fixed | 119.45 | 111.64 | +6.54% | 0.02% | ok |
| w13 | rotating-experts | 119.91 | 111.85 | +6.73% | 0.19% | ok |
| w2 | warm-fixed | 57.27 | 56.63 | +1.12% | 0.00% | ok |
| w2 | rotating-experts | 57.60 | 56.78 | +1.42% | 0.04% | ok |

“延迟降低”是 `1 - candidate/baseline`，负数代表变慢。
spread 为基线各轮 p50 的极差/中位数，不能视作统计置信区间。
首次短预热复测出现 W13 基线从约 155 µs 降至 120–129 µs 的轮间漂移，
对应 20%–30% spread，被标为 inconclusive；原始结果保存在 `initial-ab/`。
因此增加交替 Graph 预热，并重新执行完整四组 A/B，没有删除或挑选不利样本。

JSONL 保留各轮双方的全部样本及 p50/p90。W13/W2 分开排名、分开匹配生产 baseline，
没有跨 stage 相除。以上仅为单卡 grouped GEMM，不等同于 MoE 全路径或模型 tokens/s。

第一轮对照（固定 BM16/BN64/BK128、64 threads、stages2）中，LUT → 初版 PRMT：
W13 173.67 → 153.50 µs，W2 83.15 → 73.57 µs。
第二轮去掉提取分支并搜索 BN/threads，W13 搜索最优 143.16 µs，W2 73.27 µs。
搜索时 baseline 会变化，因此接入判断只看上述交替复测，不能拿不同 run 的最优值相除。

## 设施修复及验收

- 改 block_M 时重新 align；所有候选共享相同 A、权重、scale、原始路由。
  baseline 独立采用生产选择器的 tile，并实际校验 eager 和 Graph 输出。
- 输入 seed 在生成权重前应用，采用局部 generator；默认变尺度，unit 仅作为显式诊断项。
  轮换复用原始 A/scale，记录实际路由序列 hash、A hash、专家直方图。
- `--no-check` 只产生诊断结果，不进入合格排名，summary=false、退出 1。
  raw 错误消融记为 numerical_failure，measurement=not_run、退出 1。
- 编译/运行/计时失败保留状态和原因；异常保留部分 JSONL 和失败 summary。
  CUDA fatal 结束本进程，禁止在污染 context 中继续搜索。
- interleaved A/B 在计时前校验实际 Graph，保存双方每轮原始样本。
  固定/轮换按实际调用数配平；失败或未校验不会被 summary 报成通过。
- 新增 `tile-boundary` 路由：M15/16/17 对应每个选中专家恰好 15/16/17 个 slot，
  修复此前用总 token 数假称每专家 tile 边界的问题。

验证结果：

- `tests/kernels/quantization/test_mxfp4_int8_sm80.py`：SM80 **94 passed**。
  包含新增 CPU 失败门禁/按位比较测试，以及 W13/W2 × M15/M17 的真实 PRMT + Graph 测试。
- `run_gates.py`：BM16/32 × stages1/2 × W13/W2 的 8 个候选全部通过，
  同 stage 输入 hash 一致；raw 与 no-check 均退出 1、合格候选数 0。
- 两轮搜索各 8 个候选通过参考和 Graph 校验；四组 A/B 均通过 eager 与实际 Graph 校验。
- 增加预热设施后的 CPU 回归：59 passed、35 skipped（仅 GPU 用例跳过）。
- Python 静态检查、相关 pre-commit hooks、`git diff --check` 通过。

编译器仍会对基于 `sorted_ids` 的 scatter 给出静态 data-race 警告：它不能证明不同 slot
唯一。测试使用合法无重复路由；没有关闭该检查，也没有将“校验通过”等同于任意非法
metadata 都安全。

## 复现与证据

结果目录：[artifacts/tilelang-opt-20260930](../../artifacts/tilelang-opt-20260930)。
机器汇总：[summary.json](../../artifacts/tilelang-opt-20260930/summary.json)。
固定的测试矩阵与完整子进程命令分别保存在 `run_gates.py` / `gate-results.json`、
`run_ab.py` / `ab-results.json`。例如：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python artifacts/tilelang-opt-20260930/run_ab.py
```

运行时应先确认该设备仍空闲；此处 UUID 是本次验证设备，不是部署默认值。
源码与工作区 diff 快照保存在同一目录。历史原始 JSONL 没有改写。
旧 SUMMARY 的 M2048 错配已修正：W13 ratio=1.3547、W2=1.3363，原 0.708 不成立；
原 Markdown 另存 `historical-summary-original.md`。
历史报告中 raw 延迟差=独立解码成本、TileLang 只能逐元素解码的结论已撤回。

本轮没有验证 E384 大 M 的性能矩阵、完整 routed MoE、PP5 模型 A/B 或端到端吞吐。
候选继续留在 benchmarks 中，不能据本报告替换生产 CUTLASS。
