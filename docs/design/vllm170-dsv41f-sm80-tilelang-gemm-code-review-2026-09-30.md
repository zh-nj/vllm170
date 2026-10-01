# DSV41F SM80 TileLang grouped GEMM 审查

日期：2026-09-30。审查提交 `32f3db5e7d`，父提交 `59d0aaeabe`。
输入为实现报告、三个 TileLang 脚本、归档 CUDA/PTX/SASS、JSONL 与 SUMMARY.md。
本轮仅增加审查文档和 CPU 复现材料；不修改候选、不运行 GPU、不接入生产。
此前通用 harness 的未提交修复保持原状，不将其测试结果算作 TileLang 验收。

## 结论

**认可本候选“不接入”的决定；不认可“只差解码、流水已排除、完整生产契约已验收”的表述。**

记录支持：在已测单位 scale、指定路由的 eager 检查中候选和 CUTLASS 均与参考相等；
有判别力的 stage-4 对照中，候选 p50 慢约 11%–37%，没有晋升证据。拒绝接入不需要
先证明全部输入正确，因此这些审查问题不会变成接入理由。

问题主要影响搜索设施的后续可复用性，以及据此决定下一步优化方向的可靠性。
以下优先级针对继续使用该设施；未发现本提交改变其他模型生产推理路径。

## T1 · P1：混合 block_M 搜索复用错误 metadata，baseline 名称也可能与实际 tile 不符

位置：`deepseek_v41_moe_tilelang_autotune.py:274-292,307-330`；
`deepseek_v41_moe_tilelang_bench.py:214-235,245-260`。

每 stage 只用 `args.block_m[0]` 创建一次 case/align；config_grid 却支持多个 block_M。
measure_config 为每个 config 编译 kernel 后，candidate_call 原样传入这个 case 的 align。
当请求 `--block-m 16 64`，64 的候选仍收到按 16 填充的 sorted_ids/expert_ids/容量。
可能在 wrapper shape 检查时报错，也可能产生错误索引，不能视为合法搜索。

baseline 也用这个 case.block_m 执行，却用 `select_production_tile` 的 default_m 命名。
只要 args.block_m[0] 与生产选择不同，输出就会错误标注 baseline。
本轮历史搜索分别固定 16 或 64，且与对应默认一致，所以此缺陷不直接推翻旧性能比。

修复要求：按 block_M 创建/缓存独立 metadata；生产 baseline 独立使用实际 selector。
加入跨 16/64 的控制流回归，检查传入 kernel 的容量和 metadata，不依赖候选自己报错。
CPU 提取原 main 的复现中，两个 block_M 的请求仅出现一次 build_case(16)。

## T2 · P1：不检查的候选仍可进入 validated；baseline PASS 是直接填写的

位置：`deepseek_v41_moe_tilelang_autotune.py:185-203,286-300,354-393`。

`--no-check` 将 verification_status 设为 skipped，但随后只要 measurement_status=ok，
就加入 validated；summary 仍能 all_passed=true、exit 0。CPU 执行原 main 并替换
GPU/编译调用，已复现这一状态。搜索 header 同时无条件宣称每个候选先通过参考。

baseline 直接 time_graph，然后硬编码 verification_status=pass；没有比较其输出。
stage-4 bench 确实另行检查 baseline，不能用它替代搜索阶段每行声称的已验证事实。
此外，若部分配置失败、部分成功，main 仍可能因 validated 非空而返回 0。

修复要求：diagnostic/no-check 与可晋升结果分离；只有检查通过才进入 validated。
baseline 自身也应先验证。退出码、summary、每行状态必须使用同一契约。
历史 58 条正确候选记录都标有实际 bitwise check，故不能说它们实际跳过了检查；
问题是工具在已支持的参数路径上无法保证自己声明的门禁。

## T3 · P2：num_stages 搜索未产生不同实现，不能据此排除流水瓶颈

位置：`deepseek_v41_moe_tilelang_gemm.py:199-208`；实现报告 `:146-148,237-241`。

T.Pipelined 同时接受 num_stages 和固定的 `order=[0,1,2,3,4]`、`stage=[0,0,0,1,1]`。
归档的 `w13-16x64x128-s1/s2/s3-t64-luta.cu` SHA256 **完全相同**：
`1b961c4aa63758d0a65f84d5f37b995bbb8b78b215095d2e75b89b5354bb86c2`。
全部 26 份归档 W13 CUDA 源码仅有 **9 个不同内容**，多组 stages 的变化没有改变代码。

生成代码里已经有 cp_async、交替 buffer 和 wait；不能仅凭参数不同但耗时相同，就
推断 copy/compute 重叠不重要，或推断 cp.async 没有重叠。
58 是通过的测量记录数，按 shape+完整 config 字段去重为 56，不是 58 个独立机器实现。

修复要求：调参前确认生成代码/资源确实发生变化；分别使用自动或明确变化的手工
pipeline schedule，按源码/二进制内容去重。撤回本轮由 stages 耗时推出的瓶颈判断。
这不要求继续搜索；停止本候选时应保留正确的搜索覆盖描述。

## T4 · P2：raw 差值不是可直接相加的独立解码成本，CUTLASS 对照也未配平

位置：`deepseek_v41_moe_tilelang_gemm.py:170-188,212,336-342`；
实现报告 `:182-212,228-241`；`mxfp4_cutlass_core.cuh:835-840`。

raw 不消费 shift2，不构造/访问 shared LUT，并改变 packed→fragment 的映射。
这些改动会改变有效访存、依赖、shared 使用与编译器优化机会；报告没有归档 raw
对应的最终 SASS/资源或 profiler 证据，无法确认“同访存、同骨架”这一严格前提。
172.85−63.74 和 82.84−33.89 是不同程序的总延迟差，可以提示解码相关路径值得关注，
不能解释为可从正常执行中独立剥离的 109/49 µs 部件。

CUTLASS 的 42/13 µs 来自源码中的历史注释，其原始基线 126.2/60.7 µs 与本次
155/74 µs 不同；本轮没有匹配 shape/路由/cache/时钟/二进制的 CUTLASS raw 对照。
因此“TileLang 解码贵 2.6–3.7 倍”与“替换解码就能到 1.5–1.6 倍加速”均未成立。
LDS 吞吐/bank conflict 是合理待检假设，不能在没有计数器证据时写成已定位的唯一瓶颈。

报告自己的下一步还承认 M2048 需要 raw 消融，和总体“只差一处”不相容。
“TileLang 只能逐元素解码”也超出本轮实现所证明的范围；同一报告已提出 extern
设备函数路线，却尚未实施。应写成“本轮实现的解码方式成本较高，原因待验证”。

## T5 · P2：非单位 scale、Graph replay 与真正每专家 tile 边界没有被验收

位置：`deepseek_v41_moe_tilelang_bench.py:94-102,469-517`；
`deepseek_v41_moe_tilelang_gemm.py:440-459,489-493`；
`stage4/run_matrix.sh:55-65`。

三个入口仍强制 A scale/channel scale 全 1。bench 中“非单位 scale 会破坏 bitwise
期望”的注释错误：只要按同一个 FP32 epilogue 次序，非单位 scale 也可精确验收。
已有 eager 相等不能证明 scale 顺序、索引或漏乘行为正确；源码表达式正确与测试覆盖
完整是两件事。通用 harness 新增的 varied scale 不会自动流入这些独立 builder。

修复后的 40 周期确实逐 case 比较 eager 输出，但随后 interleaved_ab 新 capture graph，
没有 sentinel/reset/replay 后比较。搜索入口也只检查 capture 前 eager。应准确称为
“40/40 eager 路由通过”，不能推广为 Graph replay 输出验收完成。

M15/M17 是总 token 数。按本轮 seed4242、E384、k6、uniform 重建 CPU 路由后，两组
每专家最大行数均为 **2**，没有专家达到 16；未覆盖宣称的每专家 m_e=15/16/17 边界。
M=1 的 hot/sparse 也各只有一个 token，不构成多 token 热专家重复的压力验证。

修复要求：使用非单位且可辨识的 scale、漏乘/错索引负例；验证实际捕获图的输出；
显式构造每专家 15/16/17 行及多 token 热路由。已停止的候选可先收窄报告，无需为
拒绝接入而重跑整个 GPU 矩阵；恢复优化前再补这些门禁。

## T6 · P2：计时异常路径出现二次 KeyError，无法保证完整失败记录

位置：`deepseek_v41_moe_tilelang_autotune.py:196-201,331-354`。

time_graph 异常在外层捕获后构造 harness_failure 行，却没有 measurement_status。
紧接着读取 `record["measurement_status"]`，CPU 注入 capture failure 已复现
`KeyError: 'measurement_status'`，从而没有最终 summary。
这与报告“计时失败各有状态、全部记录”的声明不符。

修复要求：错误状态结构统一，并在 finally 写 aborted summary；fatal CUDA 错误
仍停止进程，不在同一 context 中继续搜索。加入编译/运行/计时失败的控制流回归。

## T7 · P2：SUMMARY 把 W2 候选除以 W13 基线，制造不存在的性能胜出

位置：`artifacts/dsv41f-tilelang-gemm-20260930-172532/SUMMARY.md:51`。

`tune-m2048.jsonl` 一行把 W2 最好 7950.1312 µs 与 W13 基线 11235.8398 µs 配对，
得到 0.708。正确按 stage 配对为：

| stage | baseline µs | best candidate µs | candidate/baseline |
| --- | ---: | ---: | ---: |
| W13 | 11235.8398 | 15221.5042 | 1.3547 |
| W2 | 5949.1840 | 7950.1312 | 1.3363 |

实现报告正文的 M2048 配对是正确的，不接入决定没有依赖这条错行。
SUMMARY 也将故意算错且关闭检查的 raw 消融列在“all candidates verified”标题下，
应分开标注 diagnostic；Markdown SUMMARY 本身不是机器可读门禁，JSONL 才是。
另有计数笔误：W2 round2 原始数据是 20 条，正文写 17。

修复要求：按 stage+shape+cache+baseline identity 分组汇总，禁止跨 stage 求比值。
正文、SUMMARY、机器输出由同一来源生成并交叉验证。

## T8 · P2：权重 seed、实际轮换身份和原始计时样本不足以复现结论

位置：`deepseek_v41_moe_tilelang_bench.py:94-98,189-201,438-446,509-538`；
`deepseek_v41_moe_harness.py::interleaved_ab`（提交版本）；autotune header `:256-264`。

权重在 manual_seed 之前构造；相同 seed 的独立进程权重不可复现。A/B 同进程共享
权重这一点成立，不能因此宣称跨 seed/进程固定了全部输入。CPU 缩小 shape 后直接
执行未修改 builder，重复调用的 packed 权重不同。

轮换 case 内部固定 seed=0，row 的 case_id/route_hash 却仍来自未计时的 base case。
header 只有父提交 HEAD、环境和部分 arguments，没有本轮新增脚本内容哈希、输入
哈希或完整编译 pass config 身份。归档 generated CUDA 能证明一部分代码，不能把
所有 JSONL 唯一对应到最终源码版本。

stage-4 interleaved_ab 内有 Timing.samples_us，但返回时仅保留每轮 p50/p90；
复测 JSONL 没有报告所称“每轮保留的原始样本”。3 round 是同进程、同一对图的
AB/BA/AB 顺序，不是三个独立进程试验。归一化分母本身正确；但轮换每图为
20×40=800 次调用、fixed 为 20 次，不宜直接据此比较两缓存模式的绝对速度。

修复要求：复用新 harness 的 seed、内容哈希、实际 sequence 身份与逐样本输出；
保存生成代码/编译资源/pass configs，保留实验命令。明确同进程 round 和跨进程重复。

## 已核实的正面结果

- spike JSON 有 65536/65536 相等记录；归档 PTX/SASS 确实含 INT8 m16n8k32 / IMMA.16832。
- 四份搜索 JSONL 共 4+26+20+8=58 条有 correctness 的成功记录；故意错误的 gate-check-raw
  被记为 numerical_failure，未计时。这一默认检查路径有效。
- stage4 主矩阵 14 文件，加 1 个 ptxas10 补测；40 周期两边都记录 cases_checked=40。
- M15/M17 的性能被标 inconclusive；其余主矩阵没有胜出。保留当前 CUTLASS 符合停止条件。
- 提交只涉及 benchmarks 和设计报告，没有生产注册、模型分派或默认后端修改。
- 自写严格校验搜索是方案允许的 fallback；无需为“没有直接用 AutoTuner”判失败。
  但不应把当前 out_idx=[] 的使用限制扩写成 AutoTuner 永远无法表达 in-place 契约。

## 本轮证据与后续

`artifacts/tilelang-review-20260930/audit_cpu.py` 和 `audit.json` 保存 CPU 控制流复现、
路由直方图、源码哈希去重、58 条搜索重算与按 stage 的正确比值。CPU doubles 只证明
main 的状态/metadata 组织行为，不是新 GPU 正确性或性能证据。
本轮未重跑 GPU，未改候选实现，也未覆盖先前通用 harness 修复。

建议先修正文和 SUMMARY 的错误归因/计数，并将独立 TileLang runner 接到已修复的
统一验收契约。若继续研究，只能把 word-level 解码列为待检候选；不能以本报告的
差值推算作为收益保证，也不能声称已经排除了流水与大 M 路径。
