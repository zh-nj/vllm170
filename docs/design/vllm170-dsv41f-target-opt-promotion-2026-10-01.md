# DSV41F target 第三轮：推广 8 warp 和 packed-word

日期：2026-10-01。基线提交：`81175371741318fda60762f62052994be7c0bf31`。
本轮基线已包含上一轮 q6 attention 8 warp、M6 W2 packed-word 和 activation 融合。
本轮继续推广这两种机制，已接入原生产服务。

**本轮确认 target Graph 边界耗时下降；整轮吞吐提升尚未得到证明。**

| 同口径指标 | 基线 | 本轮 | 变化 |
| --- | ---: | ---: | ---: |
| 单请求 q6，五 rank 的 Graph 中位数之和 | 38.895 ms | 36.818 ms | −5.34% |
| 双请求 q12，五 rank 的 Graph 中位数之和 | 59.378 ms | 55.781 ms | −6.06% |
| 单请求无插桩 decode 吞吐中位数 | 85.99 tok/s | 84.95 tok/s | −1.21%，收益未确认 |

前两行是 **rank-local 中位数求和，没有跨 rank 对齐，不是 PP 整轮 wall time**。
不能与 9 月 28 日扫描的 target span 或 `33.87 + 2.42*q` 混用。
单请求输出 256 个 token 的 ID 和接受计数完全相同；双请求输出 ID 也相同。
双请求未收集独立的接受计数，不能补作接受率结论。

原始证据目录：`artifacts/dsv41f-target-opt-round3-20261001/`。
`comparison.json` 汇总生产 A/B；`source-candidate.json`、`source-after.json`
固定部署与复核的八个 Python 文件 SHA256。测量后源码未发生漂移。

## 1. 推广路径与边界

| 路径 | 本轮处理 | 依据 / 保留条件 |
| --- | --- | --- |
| target top-512 sparse decode | 从 q6 扩展到查询总数 1–48、H64、splits ≤16 的 8 warp | 自动 splits、C1/C2、并发形状均有检查 |
| draft SWA-only sparse decode | 相同查询范围，SWA 窗口 ≤133，使用 8 warp | q5 实测 kernel −21.62%，4/8 warp 输出逐位一致 |
| sparse prefill | 保留现有实现 | 独立的 prefill kernel 已使用 8 warp，不能重复计入收益 |
| M1 W13 | 将四字节重组替换为对齐 uint32 读取 | 真实 checkpoint 权重样本，约 −10% |
| M6 W13 | 新接入 BN64 packed-word kernel | 单算子及完整 `Experts.apply` 均有收益 |
| M6 W2 | 保留上一轮 packed-word | 本轮 full-MoE A/B 两侧使用相同 W2 |
| M1 W2 | 保留原四字节重组 | 62.464 →62.464 µs，没有实测收益 |
| draft / prefill CUTLASS MoE | 保留现有实现 | CUTLASS 已有对齐 uint32 读取，没有待消除的同类重组 |
| FP8 dense linear | 保留现有 Marlin 路径 | 用户已放弃 FP8 dense TileLang，本轮没有重启这条路线 |

attention 只改 partial kernel 的 launch warp 数，保留分片、reduce、workspace、
caller-output 和数学表达式。仍经过原 `use_sm80_split` 条件：CUDA、
`is_device_capability_family(80)`、split 开关。实测硬件是 SM80 CMP170HX；
没有其他 Ampere 型号的性能证据。extra 路径要求每 query 512 个 index；
无 extra 时要求直接分页窗口或索引上界 ≤133。
q>48、H≠64、显式 splits 32/64、extra 长度 64 等保留 4 warp。
q1 的 BK16 / reduce 8 warp、q>1 的 BK32 及现有 reduce 配置均不改动。

MoE 继续默认关闭 `additional_config.dsv41_sm80_moe_tilelang`，生产原 unit 已显式开启。
模型身份、E384/top-k6、H5120/I2304、BF16、SM80.0、TP/EP1 等门保持不变。
M6 W13 额外要求 `m=6`、local/global E384、top-k6、调度 BM16/BN128；
候选自身的 W13 tile 为 **BM16/BN64/BK128、128 threads、2 stages**。
W2 保持 BM16/BN128/BK128、128 threads、3 stages。
加载期编译和 warmup，forward / Graph replay 不进行编译。
M2–M5、M12、draft E128/top-k3 等继续走 CUTLASS。

代码入口：

- `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`：partial launch 条件。
- `vllm/model_executor/layers/fused_moe/experts/dsv41_sm80_tilelang.py`：word 读取、M6 W13 编译与 warmup。
- `vllm/model_executor/layers/fused_moe/experts/mxfp4_int8_sm80.py`：加载和 dispatch。
- `csrc/libtorch_stable/moe/mxfp4_int8_sm80/mxfp4_cutlass_core.cuh`：原有 uint32 packed 读取，未修改。

## 2. attention：相同溢出机制在其他 query 形状重现

空闲卡 UUID `GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216`，PCI `02:00.0`，
SM80、74 SM、32 MiB L2。使用 CUDA Event + 单次调用 Graph；三个交替轮次，
每 arm 等待 500 ms，每轮 20 样本，64 MiB L2 flush 在 Event 区间之外。
原始样本、前后时钟和编译资源保留在三个 `sparse-promotion*.json`。
没有 CUPTI/NCU 数据，不能据此报告动态 stall 或指令执行计数。

先测 q1/2/4/6/8/12/24 × C1/C2 的 14 组固定 splits=3，再测 16 组实际自动
splits / SWA-only，最后补 q3/5/18/36/48 和 SWA-only q5。
main 长度 133 覆盖 ragged 尾部；main block128，extra C1 block128/C2 block64；
索引来自合成 32K 上下文，4/8 warp 对照输出均逐位一致。
这是有边界的代表性矩阵，没有穷举每个查询数、窗口长度及 GPU 型号。

| 形状（C1，除另行标注） | splits | 4 warp partial+reduce | 8 warp partial+reduce |
| --- | ---: | ---: | ---: |
| top512 q1 | 16 | 31.744 µs | 28.672 µs |
| top512 q2 | 8 | 68.608 µs | 37.888 µs |
| top512 q3 | 6 | 80.896 µs | 46.080 µs |
| top512 q4，C2 | 4 | 96.256 µs | 63.488 µs |
| top512 q5 | 6 | 134.144 µs | 80.896 µs |
| top512 q6 | 3 | 133.120 µs | 80.896 µs |
| top512 q8 | 2 | 180.224 µs | 111.616 µs |
| top512 q12 | 3 | 232.448 µs | 148.480 µs |
| top512 q18 | 1 | 313.344 µs | 199.680 µs |
| top512 q24 | 3 | 424.960 µs | 282.624 µs |
| top512 q36 | 1 | 569.344 µs | 376.832 µs |
| top512 q48 | 3 | 842.752 µs | 571.392 µs |
| SWA-only q1 | 5 | 21.504 µs | 19.456 µs |
| SWA-only q5 | 5 | 37.888 µs | 29.696 µs |
| SWA-only q6 | 3 | 35.840 µs | 27.648 µs |
| SWA-only q12 | 3 | 61.440 µs | 47.104 µs |

top512 BK32：编译报告从 **255 registers / 8 spills** 变为
**255 registers / 0 spills**，shared bytes 同为 88064。
SWA-only BK32 则变为 **193 registers / 0 spills**。
q1 BK16 两者原本都没有 spill，registers 从 255 降至 171，也有小幅收益。
这些结果支持推广 warp 配置，但不能把整个 decode 的瓶颈都归因于 spill。

生产 draft 的逻辑 query 数为 5：`speculator.py` 在 sample-from-anchor 模式下
使用 `num_speculative_steps`，dispatch 的 `uniform_token_count` 也是 5。
`draft.stage.forward` 用 target `input_batch.input_ids` 打标签，所以其 trace
`token_count=6` 不能解释为 draft 实际算了 6 个 query。
SWA-only q5 自动 splits 是 **5**，top512 q5 自动 splits 是 **6**；
必须调用实际 `_decode_num_splits`，不能从 q6 的 splits3 直接外推。

## 3. packed-word：推广到 W13，而不是扩大 BN

W13 packed shared row 为 128 bytes，读取位置 `channel*16 + t*4`，
每次地址按 4 字节对齐。生成 CUDA 的 shared base 对齐 1024，packed buffer
偏移 128。uint32 load 消除四个 byte 的拼接，没有改变全局权重字节、dtype 或 MMA。

从真实 checkpoint layer20 读取 experts137–172，其余槽位置零，路由只选这些真实
experts；输入为合成 INT8。checkpoint provenance 与生成代码均保留。
M1 W13 两个 expert shift 对照：103.424/104.448 →93.184 µs；
M1 W2 两组都为 62.464 µs，未推广 word 读取。

最终 M6 W13 确认实验（`m6-w13-confirm/results.json`）：

| 路由 | CUTLASS | BN64 scalar | BN64 word | word 相对 CUTLASS |
| --- | ---: | ---: | ---: | ---: |
| hot | 97.792 µs | 104.448 µs | 93.184 µs | −4.71% |
| mid | 314.368 µs | 312.320 µs | 278.528 µs | −11.40% |
| spread | 513.024 µs | 517.120 µs | 461.824 µs | −9.98% |

旧 BN64 scalar 的 hot 回退在消除字节重组后被逆转。这不能用来替旧 W13 BN128
的失败背书；本轮接入的是 BN64。

完整 `Experts.apply` 只替换 W13，两侧使用相同现有 W2 和 activation，输出及
中间 slots 逐位一致。hot / spread 另各有两次动态输入与路由 Graph 变更检查。

| 路由 | 原完整 MoE | W13 word 完整 MoE | 降幅 | 基线三轮自比波动 |
| --- | ---: | ---: | ---: | ---: |
| hot | 164.813 µs | 156.186 µs | 5.23% | 0.05% |
| spread | 758.707 µs | 699.622 µs | 7.79% | 0.50% |
| rotating40 | 758.758 µs | 699.776 µs | 7.77% | 0.60% |

每个 Graph 40 次 apply。rotating40 轮换同一份 36-expert 权重的路由顺序及 workspace，
不是 40 层不同权重；不能当作真实 40 层 HBM 重读测量。

本轮 M1/M6 W13 的静态 SASS 均为：

| 项 | scalar | word |
| --- | ---: | ---: |
| PRMT | 166 | 70 |
| 总指令 | 891 | 827 |
| IMMA | 16 | 16 |
| registers | 75 | 76 |
| stack / local | 0 / 0 | 0 / 0 |

这是 W13 BN64 的计数，不能与上一轮 W2 的 493→205 / 2080→1888 直接相加。
计数来自匹配生成 CUDA SHA 的嵌入 cubin，包括 prologue / epilogue；不是动态计数。
resource 的 `SHARED:0` 仅指静态 shared，kernel 仍有动态 shared buffer。
生产编译缓存匹配同一 M6 W13 word CUDA SHA（`production-w13-cache.json`），
属于编译 / capture 证据，不是独立的 Graph replay kernel 命中计数器。

## 4. 一次生产重启后的同口径验证

原 unit `dsv41f-manual.service`、端口26661、PP5/TP1、物理 GPU2–6 不变。
DSpark5 → target q6，FULL Graph，KV FP8，CED、prefix caching、max-num-seqs8
及 MoE opt-in 都保留。`unit-before.txt` 与 `unit-after.txt` 逐字相同。

停止旧服务时触发 unit 的 20 秒超时，systemd 记录了旧 worker / ZMQ 清理的
SIGKILL 与 timeout；随后新服务成功启动。22:31:37 启动，22:37:54 ready。
最终 MainPID2904271，PP0–PP4 为2906935–2906939，GPU 上没有旧 worker PID。
最终 health200、实际模型目录正确、running/waiting 均0、phase `ENABLE` 已删除。
空闲测量卡回到6 MiB / 0%利用率。见 `final-state.json` 和 `gpu-process-final.csv`。

无插桩请求复用原 q6 扫描的同一 32768-token prompt：
SHA256 `89a3cca89df5fa7bc4f300260fce5ede41ea0659c9c8b6676353db890e01f6e1`。
greedy、ignore_eos、生成256；先64-token warmup，再各测三次。
指标为流式 **255 个首 chunk 之后的 token / 后续时间**，排除 TTFT。

| 版本 | 三次 tok/s | 中位数 |
| --- | --- | ---: |
| 基线 | 85.988 / 87.422 / 82.738 | 85.988 |
| 本轮 | 83.739 / 84.946 / 85.377 | 84.946 |

六次输出 ID、spec metrics 完全相同：每次44轮，draft220 / accepted211，
逐位置接受44/44/44/40/39；generation增量256，prompt增量32768。
本轮吞吐没有提高；只有三次重复、前后顺序 A/B、没有重启后基线回测，
不能把 −1.21% 判成确定回退，也不能宣称端到端收益。
实验时宿主 memory PSI 约24–28%是环境观测，未通过干预证明它是原因。

阶段测量各使用一次128-token warmup，再一次128-token profile，phase 只截取
对应 PID 文件的新增字节。q6 每 rank21个 decode Graph 样本：

| rank | 基线 Graph 中位数 | 本轮 Graph 中位数 |
| --- | ---: | ---: |
| PP0 | 5.973 ms | 5.634 ms |
| PP1 | 7.382 ms | 7.037 ms |
| PP2 | 8.001 ms | 7.582 ms |
| PP3 | 8.617 ms | 8.117 ms |
| PP4 | 8.922 ms | 8.448 ms |
| 中位数之和 | 38.895 ms | 36.818 ms |

每 rank 都下降。注意基线 q6 attention 已是8 warp，所以 q6 的本轮新增 target
收益主要对应 M6 W13；该 A/B 同时含 draft warp 推广，没有单独开关因果隔离。

并发实验：同一 prompt 的两次并发128-token请求，先 warm pair，再 profile pair。
过滤 `flow=decode,execution=full,token_count=12,target.model_graph_replay`，
每 rank23个样本。总和59.378→55.781 ms；q12 不走 M6 W13，
该 target 形状新增候选是 sparse attention 8 warp。
四个请求的 ID 在各版本内和前后版本之间一致。

draft stage 的 PP1/2/3 GPU 中位数为：
0.681/0.665/0.606 →0.689/0.662/0.595 ms，
求和1.952→1.946 ms（约−0.31%），各 rank 方向不一致。
**SWA-only kernel 提速成立，但整条 draft forward 提速未确认。**

## 5. 当前仍需区分的热点

本轮 `target.attn_metadata` CPU 中位数（PP0→PP4）：
基线9.94/20.20/19.10/19.81/19.27 ms，
本轮11.50/22.79/19.05/18.94/18.79 ms。
这些数来自插桩路径，不能相加进 Graph 耗时，也不能直接解释无插桩剩余延迟。
`phase_profile_stage` 把 CUDA Event 放在整个 host scope 两端，metadata 的 GPU Event
区间还可能包括等待宿主 enqueue 的空隙；它不是 metadata GPU kernel 的纯执行时间。
profile 每步 flush 又会同步设备，前后 metadata 变化不能归因于本轮 kernel 修改。

下一轮优先用现有源码和 trace 分清 metadata 的 host 工作、PP 等待、采样 / draft
串行等待，再针对真实关键路径做最小干预。当前证据不支持再次扫 q 或为了吞吐
反复重载整模型。rank Graph 的下降与整轮 throughput 尚未同步，正是需要解决的
剩余问题，而不是可以省略的验收项。

## 6. 验证、复现与恢复

硬件 pytest：**295 passed / 3 deselected / 48 warnings**。
包括完整 MoE apply、动态路由 / 输入、caller-output alias、fallback、禁止 forward
编译，以及 attention 4/8 warp 逐位对照、unsplit reference、poison replay。
3个排除项为此前原 HEAD 已存在的 metadata fixture / builder 失败：
fixture 缺 `additional_config`，builder 缺 `_get_token_to_req_indices`。
本轮没有修复或声称它们通过。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python -m pytest \
  tests/kernels/quantization/test_mxfp4_int8_sm80.py \
  tests/kernels/attention/test_dsv4_sparse_prefill_sm80.py \
  -k 'not test_v41_single_request_fast_paths_ignore_max_num_seqs and not test_persistent_token_map_handles_real_batch_transitions and not test_single_request_v41_builders_reuse_persistent_token_map' -q

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python -m benchmarks.kernels.deepseek_v41_moe_w2_pipeline \
  --word-only --w13 --apply \
  --model /media/z/data6/models/DeepSeek-V4.1-Flash \
  --samples 20 --output-dir /tmp/dsv41f-w13-word

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python -m benchmarks.kernels.deepseek_v41_target_sparse_decode \
  --quick --queries 5 --splits 0 --main-len 133 --extra-len 0 \
  --samples 20 --output /tmp/dsv41f-swa5.json
```

八个 Python 文件 pre-commit 全部适用 hook 通过，benchmark 最终修正另过一次
pre-commit；`git diff --check` 通过。首次 SWA benchmark 的 capture tuple/Graph
适配错误已修正，失败日志保留；首次离线 ELF 提取遗漏 program header 已修正，
未影响运行时代码。完整 serving 验证仅为上述固定 prompt 的回归，未作广泛模型评测。

`before/` 保存本轮前三个 runtime 文件；`restore_before.py` 有源码 SHA、unit、
phase 关闭和请求空闲校验，只恢复这三个文件并重启原服务。此脚本未执行，也没有
恢复耗时 / 成功率验证；调用应从仓库根目录使用 `.venv/bin/python`，随后等待
加载并检查 health、实际模型和 GPU 归属。若 stop 返回超时，脚本会在复制前中止。
其他测试、benchmark、文档及无关源码不会由该脚本恢复。

本轮工作树未提交，未创建上游 PR。AI 辅助完成代码与分析，生产性能结论以本文件
列明的证据层级为限。旧的仅 M1 接入 / 不可达报告是对应历史版本的结果；当前
M6 W13 的可达性与性能应以本轮及上一轮部署记录为准。
