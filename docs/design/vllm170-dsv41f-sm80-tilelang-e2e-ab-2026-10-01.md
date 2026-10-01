# DSV41F SM80 TileLang grouped GEMM：端到端 A/B（控制参数 `dsv41_sm80_moe_tilelang`）

日期：2026-10-01。对象：GPT 已接入的
[可选接入](vllm170-dsv41f-sm80-tilelang-integration-2026-10-01.md)
（`additional_config: {"dsv41_sm80_moe_tilelang": true}`，默认关闭）。

测试命令 = 用户给定的生产命令，**只改这一个 JSON 布尔值**，其余逐字一致。
每次换值都完整重启服务（PP5，端口 26661，CUDA_VISIBLE_DEVICES=2,3,4,5,6）。

结果目录：`artifacts/dsv41f-tilelang-e2e-20261001/`（脚本、四组原始 JSON、phase profile、
服务日志、启动耗时）。测试期间停掉并**已恢复**原 `dsv41f-manual.service`。

## 1. 结论

**这个开关在当前生产配置下没有可测的端到端收益，而且按代码路径根本不会被选中。**

1. **候选路径不可达**：接入分支要求 `m == 1`、top-k6、E=384、BM16；生产配置开着
   DSpark 5 个 draft token，target 每步验证 `1+5 = 6` 个 token，因此 MoE 的 `m` 恒为 6。
   四组 phase profile 里 decode 步的 `token_count` 全部是 6，没有一个是 1。
2. **draft 也不够格**：DSpark draft 是 `dspark_n_routed_experts=128`、
   `dspark_num_experts_per_tok=3`，连 top-k6/E=384 的门都过不了。
3. **就算够格，端到端也只有约 1%**：专家段本身只占 decode 关键路径的一成左右
   （见第 5 节的估算），段内 -8% 折算到端到端约 -0.9%。
4. **本次 A/B 的噪声比这个上限大一个量级**：两次**完全相同**的基线（都关开关）
   之间的 decode 吞吐差 **12.8%**，而开关相对第一次基线的差只有 **5.9%**。
   按既定判据（差异必须超过基线自比波动）本次测量只能判 `inconclusive`。

因此：**开关可以保留（安全、默认关闭、无副作用），但不能声称有端到端收益；
不应据此把它当作性能提升上线，也不应把它写进"已优化"的结论。**

## 2. 方法与噪声控制

| 项 | 内容 |
| --- | --- |
| 四组 arm | `off1`(08:42) → `on`(08:57) → `off2`(09:15) → `off3`(09:33)，各组都是完整重启 |
| 差异 | 仅 `--additional-config` 的 `dsv41_sm80_moe_tilelang`：`off*`=false，`on`=true |
| 负载 | greedy 探针（3×1024→64）、decode c1（3×1024→256）、decode c4（8×1024→256）、prefill 16K（3×16384→32）、短 profile 窗 |
| 配对 | 每组 arm 用**同一批 prompt seed**；decode c1 三个 prompt 的步数与投机接受长度在四组中完全一致，即 token 流逐位相同，可做逐 prompt 配对 |
| 对照 | prefill 16K 是"开关不可能影响"的对照负载 |
| 启动耗时 | 710 / 731 / 943 / 741 秒（同为加载同一 checkpoint，属正常波动） |

服务在 `on` arm 确实生效：`additional_config` 中出现该布尔值，加载期编译了
**23 个 TileLang `grouped_gemm` kernel**，并打印
`DSV41F SM80 TileLang PRMT enabled for M=1, E=384, top-k6; other shapes use CUTLASS.`
所以下面的"无差异"不是"开关没生效"，而是"生效了但用不上"。

## 3. 机制核查：候选分支为什么不会被选中

**代码门**（`vllm/model_executor/layers/fused_moe/experts/mxfp4_int8_sm80.py:362-383`）：

```python
use_persistent_decode_metadata = (
    m == 1 and topk == _TARGET_TOPK
    and num_experts == global_num_experts == _DECODE_NUM_EXPERTS
    and block_m == _DECODE_BLOCK_M
)
decode_gemms = self._tilelang_decode_gemms if use_persistent_decode_metadata else None
```

只有 `use_persistent_decode_metadata` 为真时才会走候选；`_DECODE_BLOCK_M=16`、
`_DECODE_NUM_EXPERTS=384`、`_TARGET_TOPK=6`。

**模型配置**（`/media/z/data6/models/DeepSeek-V4.1-Flash/config.json`）：

| 字段 | 值 | 含义 |
| --- | --- | --- |
| `n_routed_experts` / `num_experts_per_tok` | 384 / 6 | target 满足 E/top-k 门 |
| `dspark_block_size` | 5 | 每步 draft 5 个 token → target 每步 `m = 1 + 5 = 6` |
| `dspark_n_routed_experts` / `dspark_num_experts_per_tok` | 128 / 3 | draft 连门都过不了 |
| `num_hidden_layers` | 40 | PP5 → 每 rank 8 层 |

**phase profile 实测**（`profile/{arm}-pp*.jsonl`，四组一致）：

```text
flow=decode   steps=206/207   token_count = {6}         ← 从来没有 1
flow=dspark   steps=172       token_count = {6: 164, 256: 8}
flow=prefill  steps=40        token_count = {256}
```

decode 步的 `token_count` 就是 `input_batch.num_tokens`
（`vllm/v1/worker/gpu/model_runner.py:1994`），也就是 MoE 看到的 `m`。
四组 arm、400+ 个 decode 步，没有一步是 `m=1`，因此候选分支在 decode 路径上
**一次都不会被选中**；被选中的是 CUTLASS。

> 唯一可能触发的是"prefill 尾块恰好 1 个 token"这种 eager 步（长 prefill 分块 +
> CED 短尾）。本次负载的 prompt 长度都是 256 的整数倍，没有出现该形状；
> 即便出现，每次请求也只影响一次 1-token 的 MoE 调用。

## 4. 端到端数据

### 4.1 decode c1（token 流逐位相同，可配对）

| prompt seed | 步数 / 接受长度 | off1 | **on** | off2 | off3 |
| --- | --- | ---: | ---: | ---: | ---: |
| 770978 | 51 / 5.02 | 67.50 | **71.37** | 55.35 | 76.16 |
| 771955 | 112 / 2.29 | 30.38 | **32.17** | 28.34 | 35.06 |
| 772932 | 136 / 1.88 | 24.62 | **27.16** | 23.57 | 27.68 |

（单位 tok/s，越大越好。）

相对第一次基线的比值：

| arm | 比值（三个 prompt） | 中位 |
| --- | --- | ---: |
| `on` | 1.057 / 1.059 / 1.103 | **1.059** |
| `off2`（同样关闭） | 0.820 / 0.933 / 0.957 | 0.933 |
| `off3`（同样关闭） | 1.128 / 1.154 / 1.124 | **1.128** |

**两次完全相同的基线之间相差 12.8%，是开关差异（5.9%）的两倍多。**
`on` 比 `off2` 快、比 `off3` 慢，与时间顺序也不单调。结论只能是
`measurement_inconclusive`：这个量级的差异由重启后的环境状态决定，不是开关。

### 4.2 对照负载：prefill（开关不可能影响）

| prompt seed | off1 | on | off2 | off3 |
| --- | ---: | ---: | ---: | ---: |
| 770980 | 3.928 | 4.371 | 4.502 | 3.828 |
| 771957 | 2.854 | 2.789 | 2.854 | 2.842 |
| 772934 | 3.128 | 2.989 | 3.123 | 3.026 |

（16K 输入 TTFT，秒。）四组最大偏差 ~4%，且不随开关变化；prefill 的 MoE 阶段耗时
在 profile 里也逐项一致（W13 2911.7 vs 2919.9 µs，W2 两次都是 1496.1 µs）。

### 4.3 c4 聚合吞吐（仅参考，不可配对）

同一 prompt 在 c4 下四组的 token 流互不相同（请求到达/合批顺序不同导致 batch 组成不同，
进而数值与贪心选择分歧），因此只能看聚合：off1 58.7 / **on 58.9** / off2 51.2 / off3 49.4 tok/s。
同样地，`on` 与 `off1` 几乎相同，而两个靠后的基线低 ~15%。

## 5. 上限估算：就算路径被选中，端到端也只有约 1%

- GPT 的段内测量（同一 `Experts.apply()`，E384/M1/top-k6）：
  CUTLASS **200.40 µs** → TileLang **184.27 µs**，段内 **−8.05%**。
- 本次实测 decode 步延迟：`decode_span / steps = 3.778 s / 51 = 74.1 ms/步`，
  接受长度 5.02 token/步 → **14.8 ms/接受 token**。
- 单 rank 每 token 的专家段：8 层 × 200 µs ≈ **1.6 ms** → 约占关键路径 **11%**。
- 因此端到端上限 ≈ `11% × 8.05%` ≈ **0.9%**，这已经是"路径被选中"的前提下的最好情况；
  生产实际跑的 `m=6` 形状连这个都没有。

0.9% 远低于本次 A/B 观察到的 13% 重启噪声，也低于 §4.1 里
prompt 之间的差异（同一 arm 内 24–76 tok/s）。也就是说：**这个优化在当前
部署形态下，用端到端吞吐根本无法验收**，必须在专家段（已有 8% 的稳定结论）
或固定形状的内核级 A/B 上验收。

## 6. 正确性

- 贪心探针：三组 arm 的 3 个固定 prompt 生成 token **逐位相同**（off1 = on = off2 = off3），
  组内重复请求也逐位相同。
- decode c1：四组的步数与投机接受长度完全一致（51/5.02、112/2.29、136/1.88），
  说明 token 流没有分歧 → 开关没有改变数值结果。
- 服务加载期无报错；TileLang kernel 编译发生在加载阶段，forward/capture 期间无编译。

## 7. 建议

1. **保留开关但不要声称收益**：它默认关闭、无副作用、离线段内确有 -8%；
   但生产配置下不可达，端到端不可测。
2. **若要让它真正生效**，需要把门从 `m == 1` 扩展到实际 decode 形状
   `m = 1 + num_speculative_tokens`（当前 kernel 还把 `slots=6`、`max_m=96`、
   `block_m=16` 写死），并让 draft（E=128/top-k3）另行处理或不处理。
3. **改完后的验收方式**：先做段内 A/B（8% 那个量级，判据清晰），再做端到端时
   必须用噪声控制协议——同一 arm 多 prompt 多轮、arm 之间按时间交错、
   并以"两次相同基线的差"作为判据；单次重启对比无法分辨 1% 以下的差异。
4. **顺带记录一个环境事实**：本机 decode 吞吐在服务重启之间存在 ~13% 的漂移
   （prefill 只有 ~4%），任何 decode 侧的端到端 A/B 都必须先测这个噪声带。

## 8. 环境、证据与回滚

- 测试对象为工作区（未提交）代码：
  `vllm/model_executor/layers/fused_moe/experts/dsv41_sm80_tilelang.py`（新增）+
  `mxfp4_int8_sm80.py`（修改）。与线上 release
  `dsv41f-thr256-cedfix-20260930-r1` 的 `vllm/` 逐文件对比，差异**只有**这两处。
- 本次测试**没有修改任何源码**；只做了服务重启与测量。
- 四组 arm 的服务日志、bench JSON、phase profile、启动耗时都在
  `artifacts/dsv41f-tilelang-e2e-20261001/`；`arm.sh` 可复现单组 arm。
- 原 `dsv41f-manual.service`（release `dsv41f-thr256-cedfix-20260930-r1`，
  `additional_config={"ced_prefill":true}`）已按 `ab/MANUAL_DEPLOY.md` 的命令恢复为 active。
