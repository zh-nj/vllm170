# DSV41F SM80 MoE 评价设施代码审查

日期：2026-09-30。主要审查提交 `59d0aaeabe`，父提交 `f9c6a54b06`。
审查期间 HEAD 增至 `32f3db5e7d`；补查该提交对共用 helper 的修改。
新 TileLang kernel 本体及其性能结论不在本轮完整审查范围。

## 结论

**设施已具备有价值的数值参考和测试基础，但尚不能作为新 GEMM 候选的最终验收门禁。**
下面 R1–R3 是优先修复项：scale 覆盖缺失、汇总非严格通过、轮换输入没有实际验收。
R4–R7 影响复现、带宽统计和数值门禁可靠性。R8 会制造不必要的大额 GPU 分配。

本轮未修改任何实现或测试，仅新增本审查报告和 CPU 复现产物。
现有改动范围保持在 benchmarks/tests/docs；未发现生产 `vllm/`、`csrc/`、
模型分派、环境默认值或服务配置改动。因此“没有改变其他模型推理路径”在代码范围上成立。

## 审查发现

### R1 · P1：所有 GEMM fixture 的 scale 都为 1，无法拦截缩放缺失或索引错误

位置：`deepseek_v41_mxfp4_int8_sm80.py:273-307,475-494`；
`tests/kernels/quantization/test_mxfp4_int8_sm80.py:682-698` 及后续 W2 用例。

benchmark 的 A scale 与 channel scale 都通过 `torch.ones` 生成；新增非零 W13/W2
pytest 同样使用全 1 scale。因此删掉 epilogue 的一个甚至两个乘法，或者把 W2
scale 错用 token 行，新 kernel 仍能通过这组输入。当前报告关于“验证了两次缩放
顺序”的结论超出测试证据。

还存在后续扩展陷阱：`verify_stages()` 以 `unit_scales` 决定是否要求精确一致。
换成非单位 scale 后，`run_case` 会把 GEMM 交给 `REDUCE_THRESHOLD`，而 GEMM
Comparison 没有 term-scale/step-excess，结果会报 metric missing，不能直接补随机 scale
就算完成。

修复验收：为 W13 token、W2 slot、expert/channel 分别设置可辨认的非单位 scale；
加入非二次幂与 BF16 舍入边界用例。保持 FP64 整数 dot→FP32→先乘 A scale→再乘
channel scale→BF16 的参考次序，明确 GEMM 自身门禁，不借用 reduce 容差。
故意漏乘/错索引的候选必须失败。

### R2 · P1：summary 会把未检查结果汇总为通过，异常也不能完整保存本轮状态

位置：`deepseek_v41_moe_harness.py:1196-1223`；
主入口 `deepseek_v41_mxfp4_int8_sm80.py:553-620`。

`all_passed` 只判断 rows 非空且没有 `correctness_status == fail`/reason。
实际默认 CLI 不加 `--check-reference` 时，计时行是 `not_checked`、reason=None，
summary 仍会 `all_passed=true`。CPU 复现还确认 `skipped/uncalibrated` 无 reason，
以及 correctness=pass/measurement=fail 也会得到 true。状态不应依赖调用者恰好填写 reason。

另外 `run_case` 的所有行先缓存在内存，返回后才写 JSONL；tile 等价断言、capture、
OOM 等异常跳过 summary 路径。复用输出路径时，旧 summary 可能继续显示成功。
`run_case` 只把 `fail` 视为失败，uncalibrated 状态也不能可靠阻止计时与成功退出。

修复验收：以状态白名单定义严格通过，区分 correctness 与 measurement，并显式区分
参考行的“无需计时”与待测用例的 skipped。保留 failed/unchecked/uncalibrated/
inconclusive 计数。每 case 尽早落盘，异常最终写 aborted summary；新 run 不沿用旧成功。
给主入口加失败注入测试，验证返回码、JSONL 和 summary 一致。

### R3 · P1：rotating 模式验收的并非实际被计时的路由，也未验证 Graph replay 输出

位置：`deepseek_v41_mxfp4_int8_sm80.py:769-841,843-848,909-912,1024-1088`。

独立参考和 tile 等价检查只运行随机 `routing_ids`；之后才生成按 0..E 循环的
`rotation_routing_sequence`。计时使用这个新序列，但继续携带原 route_hash 和 PASS。
原检查的 6 个专家通过，无法证明轮换的 240 个专家及其 metadata 均正确。
`--route-pattern` 在轮换计时时也被该序列覆盖，结果仍标为用户指定的模式。

真实产物 `production-m1-rotating-r1.jsonl` 的参考行与计时行都报告
`route_hash=dedffb8b82bf99d8`；它对应原随机 IDs，并非完整轮换序列。
计时图只执行与计时，没有核对 replay 后输出；因此捕获错误/漏写也可能被先前 eager
写好的缓冲内容掩盖。

修复验收：对实际将使用的 route 序列/metadata 逐项检查，记录 sequence hash；
capture/replay 前将输出设为 sentinel，再校验 replay 输出。为轮换中途才出错的
专家、错误捕获 metadata、no-op replay 增加负例，均应阻止 PASS 和性能晋升。

### R4 · P2：`--seed` 仅控制路由，权重与激活不可复现

位置：`deepseek_v41_mxfp4_int8_sm80.py:543-560,273-307,475-478`。

原脚本的 `torch.manual_seed(4242)` 在改写中被删除，当前 seed 只传给
`make_routing_ids`。权重、shift 和 A 使用全局 RNG `.random_()`，没有 generator。
相同 seed/命令的不同 round 实际改变了数值输入，却被用于 baseline 自比噪声和
固定/轮换比较。CPU 使用原 builder 函数、缩小 geometry 的复现确认两次输入不同。

修复验收：使用显式、从 seed 和 case/stage 推导的 generator；至少保证相同 case
不随扫描顺序改变权重/A。保存权重、A、routing weights 和 IDs 的 hash，独立进程
重跑应一致。修复后重建 baseline，不混用旧 round。

### R5 · P2：轮换字节按并集除周期，专家复用时严重少算；工作集也未按 stage 区分

位置：`deepseek_v41_mxfp4_int8_sm80.py:683-709,854-865,915`。

代码用 `bytes(union(active experts across calls))/period` 当每调用字节。
这只有每步专家集互不重叠时才成立。合法的 E=8/top-k6/40 calls，每次访问 6 个专家，
应按 6 个专家统计；当前只算 8/40=0.2 个专家，少约 30 倍。CPU 小 geometry
复现为 reported=3532 B，expected=105984 B。
E=384/k6/40 无 wrap，既有该组 active_weight_bytes 恰好不受这项错误影响；不能
据此宣称所有参数组合正确。

`working_set_bytes` 则把 W13+W2 相加复制给所有 stage，但它们分开计时。
例如 top-k3 的 W2 活跃权重实际约 17.2 MiB，联合约 51.5 MiB；使用联合数判断
是否超过 32 MiB L2 会给出错误解释。轮换模式还捕获 40×40=1600 次调用，固定模式
仅 40 次；做缓存比较时必须对齐采样批长或解释其影响。

修复验收：每调用字节用 `sum_i bytes(unique(ids_i))/period`；working set 用并集，
但分别给 W13/W2 记录。加入 wrap、部分重叠、完全重复专家集的 CPU 算术测试。

### R6 · P2：reduce 的全局 term_scale 能掩盖局部大错，也会拒绝正确全零结果

位置：`deepseek_v41_moe_harness.py:561-579,634-652,750-775`。

term_scale 在所有 token/列上取全局最大；一个大输出会给其他小输出分配过大的误差
预算，而全局 calc_diff 同样被大值主导。CPU 反例：reference 为
`[100139008, 0.25]`，candidate 把第二列完全置零，仍判 PASS。
该例使用合法 BF16 routed 值和正的 routing weights，不依赖 NaN/Inf 或非法输入。

反方向，全零 routed 与全零正确 output 的 term_scale=0，比较结果完全相等仍因
`step_excess_metric_missing` 判 FAIL。

修复验收：使用逐输出元素的绝对项和定义局部误差预算，并明确处理零项。
如使用理论舍入界，核对其针对 FMA/非 FMA 的适用条件；先用可手算 fixture 校准。
全零正确结果应通过，局部输出被抹零应失败，真实相消/舍入差异仍可解释。

### R7 · P2：代码 fingerprint 漏掉新增 helper，现有产物无法唯一对应最终参考实现

位置：`deepseek_v41_moe_harness.py:1122-1131`；
主入口 `deepseek_v41_mxfp4_int8_sm80.py:543-548`。

`dirty_diff_hash` 只哈希 `git diff`，不含 staged 或 untracked 文件。
这次 helper 正是 untracked 文件，原始产物的 `env/git-status-final.txt` 明确记录它
尚未跟踪；144 行均只有父 HEAD `f9c6a54b06` 和同一个 diff_hash。
所以改变最关键的 reference/threshold 代码不会改变这些字段。
`run_id` 也只由 HEAD+diff_hash 拼接，所有不同 round 共用同一个 ID，round 字段为 null。

不能因此断言旧测试没有执行；已有日志说明执行过。但这些字段不足以证明运行的是
最终提交的哪一版 helper，也无法独立复现 threshold 校准来源。

修复验收：记录所有实际参与执行文件的内容 SHA256（含新增/staged），扩展二进制身份，
唯一 run ID、round ID 和输入 hashes。为关键新增 helper 保存源码快照或 commit。
先后改动 untracked/helper、staged diff 时，manifest 应能区分。

### R8 · P2：统计字节时会把所有命中权重复制一遍

位置：`deepseek_v41_moe_harness.py:338-343`。

仅为求 numel，代码对 packed/shift/scale 各做 `index_select`，在 GPU 分配实际副本。
E384/prefill 会额外复制约 4.5 GB 的 W13 packed 权重，轮换并集同样产生大分配。
在显存有余量时表现为额外准备耗时；接近预算时会让本来可运行的 benchmark OOM，
还会改变 allocator 状态。CPU 拦截 index_select 确认三次实际复制。

修复验收：当前每专家等 shape，直接以活跃专家数×每专家元素数×element_size
计算，校验 ID 范围即可。统计函数不应发 GPU kernel 或分配权重规模的张量。

## 测试和产物核对

本轮重新执行（隐藏 CUDA，不占用 GPU）：

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/kernels/quantization/test_mxfp4_int8_sm80.py -q
CUDA_VISIBLE_DEVICES='' .venv/bin/python benchmarks/kernels/deepseek_v41_moe_harness.py --self-check
```

结果：**25 passed / 24 skipped**；self-check `all_ok=true`。
这说明已有 CPU 用例通过，但没有覆盖上面复现出的状态、轮换和容差漏洞。

读取旧 GPU 产物确认：pytest 日志写有 **49 passed**；JSONL 共 144 行，72 行 reference，
其中 48 行 GEMM 全部 exact、24 行 reduce 判通过。**本轮没有重跑 GPU**。
对这些结果的可信边界应写成“已记录的单位 scale、固定 eager 参考 case 通过”，
不能扩展成全部 scale、全部轮换路径、Graph replay 与真实模型质量均通过。

后续提交 `32f3db5e7d` 修正了 `interleaved_ab()` 调用 `time_graph` 时将
`graph_calls` 错当关键字的问题，并补入 calls_per_repeat。旧提交该 helper 无法调用，
当前已修，不列为仍待修缺陷。旧 baseline 主入口未调用它，因此旧结果不是 A/B/B/A 实测。

实现报告还有应降级为假设的解释：固定工作集超过 L2 不等于完全没有复用；连续读
更快不证明 GEMM 确定受 DRAM 延迟/occupancy 限制；有用 TFLOPS 随 M 增加不证明
prefill 已算力/L2 受限。单独 stage 的中位时间相加是估算，不是联合图或完整 MoE 实测。

## 复现与交付位置

`artifacts/moe-harness-review-20260930/` 包含：

- `manifest.json`：审查 HEAD、文件 hashes、CPU 测试命令和结果。
- `self-check.json`：重新运行的 helper 自检。
- `reproduce.py` / `reproductions.json`：R2/R4/R5/R6/R8 的 CPU 复现。

复现脚本从当前 driver AST 提取未改动函数，避免导入生产 CUDA 环境；仅把合成
geometry 缩小以降低 CPU 内存需求。输出写入 `/tmp/dsv41f-harness-review-20260930`。
执行前创建该临时目录，使用仓库 `.venv/bin/python` 并隐藏 CUDA。

建议 DeepSeek 先修 R1–R3，再补 R4–R8 的精确回归。GPU 只需重跑受影响的
非单位 scale、实际 rotating replay 和成对 baseline；不需要重新加载模型。
保留现有结果作为历史证据，修订报告的覆盖声明，不覆盖旧 artifacts。
