# DeepSeek-V4.1 PP5 DSpark / 1M 上下文容量审计（2026-09-14）

## 当前结论

固定分层 `8,9,9,9,5` 已停止继续尝试，不作为生产方案。原因不是单一的
KV 配额，而是 9 层 stage 在权重加载时已经接近 63 GiB 上限，显存分布也明显
失衡；在此基础上再为 1M 上下文和 DSpark 预留 workspace/KV 没有可靠余量。

本机可用推理卡是 `CUDA_VISIBLE_DEVICES=1,2,3,4,5`（每卡约 63.07 GiB）；GPU0
为 3080 Ti，不能加入 PP5。Engram 仍按原设计完整驻留主存，每个拥有 Engram 的
rank 约 94.42 GiB，GPU 只保留约 24.75 MiB staging。

当前已通过真实五卡门禁的默认组合为 target `8,8,8,8,8`、DSpark decoder
分别位于 PP1/PP2/PP3、heads 位于 PP4、`max-model-len=1,048,576`、
`max-num-seqs=8`、FP8 KV `1,300,000,000` bytes。该 KV 容量实际为
1,078,799 tokens，因此支持一条 1M 请求；它不表示八条 1M 请求可同时驻留，
八并发用于较短请求。

## 已完成的容量证据

### 目标模型、无 DSpark

使用均衡的 8 层分配做 1M 探测，`kv-cache-memory-bytes=1,000,000,000` 时，
EngineCore 明确报告：服务 1,048,576 token 至少需要约 1.18 GiB KV，而可用仅
约 0.93 GiB，估算最大长度约 788,224。也就是说，原来的 400 MiB/1 GiB 预算
不能宣称支持 1M；实际配置必须经过启动日志的 token 容量检查。

### 固定 `8,9,9,9,5`、启用 DSpark

这套分层的目标层范围为 `0..7 / 8..16 / 17..25 / 26..34 / 35..39`，DSpark
辅助层 37/38/39 全部在 PP4。实际加载暴露了三个独立问题：

1. `mxfp4_int8_sm80` 的 9 层 stage 在 MXFP4 重排时只剩约 26 MiB，申请约
   180 MiB 临时 buffer 即 OOM；把专家转换块缩小后，PP2/PP3 的最终模型仍约
   59.8 GiB。
2. MXFP8 Marlin 线性层重排在 PP1 仍出现约 2.264 GB 临时申请，而该 rank
   仅剩约 1.97–2.20 GB，加载不能稳定完成。
3. DSpark draft 的 MoE shape 为 `topk=3`，原有 `mxfp4_int8_sm80` 只接受 V4.1
   target 的 `topk=6`。本轮已经把 CUDA dispatch 扩展为编译期专用的 `topk=3/6`
   两条路径，并在 CMP170HX 上通过 W13、W2 和加权 reduce 测试；这只解决 draft
   kernel 兼容性，不解决完整 draft 堆在 PP4 的容量问题。

上述失败发生在真正的 KV 分配和 1M 请求之前，继续调小 KV 或反复启动不能解决
根因；本轮不再运行该分层。

## 多并发约束

1M 上下文与并发数线性争用 KV、PP 在途 batch 和 graph workspace。当前缓存的
1M 最大并发系数为 1.0288，不能同时承载多条 1M 请求。服务已通过五轮 8 个短请求
并发（40/40 精确算术正确）和一轮 8 个随机标记并发（8/8 精确匹配）；canonical
request-slot 映射修复后未再出现请求串位。4×32K 不同口令也 4/4 正确，说明
`max-num-seqs=8` 下的实际多请求 prefill/decode、draft KV 和采样槽隔离可工作。
取消路径又连续执行三轮“中断 2K-token 流式生成、等待
Running/Waiting/KV 清零、立即提交 8 路不同算术请求”，三轮均 8/8 精确，共
24/24；服务日志没有 worker 异常。因此同步 ring 的请求槽回收与取消后复用已经
通过当前门禁，但这不等于未来异步、多 package ring 可以省略 generation 隔离。

## Draft 权重构成与建议放置

直接读取 checkpoint safetensors 得到的原始权重体积如下。数值不包含加载后的
backend workspace 和 allocator 碎片。

| draft stage | 原始体积 | 主要组成 |
| --- | ---: | --- |
| `mtp.0` | 2.470 GiB | experts 2.241 GiB、attention 120.9 MiB、main projection 75.1 MiB |
| `mtp.1` | 2.397 GiB | experts 2.241 GiB、attention 120.9 MiB |
| `mtp.2` | 2.520 GiB | experts 2.241 GiB、attention 120.9 MiB、heads 126.3 MiB |

目标层继续固定为均衡的 `8,8,8,8,8`。建议的附加放置是：

| target PP rank | target 层 | DSpark 职责 |
| --- | ---: | --- |
| PP0 | 8 | 无 draft 权重，只参加控制同步 |
| PP1 | 8 | `mtp.0` decoder block、私有 target embedding 副本 |
| PP2 | 8 | `mtp.1` decoder block |
| PP3 | 8 | `mtp.2` decoder block |
| PP4 | 8 | `main_proj/main_norm`、norm、LM/Markov/confidence heads |

`main_proj` 应放在 PP4：三个 target auxiliary hidden 都由 PP4 本地的 target 层
37/38/39 产生，先投影成 `[T,5120]` 再发送。对于当前 2048-token prefill chunk，
BF16 `main_x` 是 20 MiB，而发送未投影的三路 concat 是 60 MiB。target embedding
只在 PP0，而 query ids 必须先由拥有 draft KV/block table 的 PP1 构造；第一版因此
在 PP1 保留约 1.23 GiB 的私有 embedding 副本，避免为了不足 0.4 MiB 的 decode
query embedding 引入 PP1→PP0→PP1 的反向通信。该副本计入 PP1 容量门禁。

## 分布式 draft 协议

不能复用 target PP 的普通 hidden relay 来假装已经支持 DSpark。proposal 发生在
target 采样之后，需要一条独立、固定顺序的 draft ring：

1. PP4 完成 target sample，广播本轮 `num_sampled/num_rejected`、请求槽和 generation
   epoch。PP0 只同步，PP1--PP3 进入 draft stage。
2. PP4 计算 context `main_x` 并把 target 采样状态发送给 PP1；PP1 根据自己的 draft
   KV/block table 生成 query ids/positions，再用私有 embedding 构造初始 hidden。
   每条消息携带 scheduler step id、请求 ID、有效请求数和有效 token 数。
3. PP1、PP2、PP3 依次为各自 draft layer 写入本地 sliding-window KV，并执行 query
   forward；HC residual、mix state 和 `main_x` 沿 ring 传递。每层只能访问自己的
   KV cache group。
4. PP3 把最终 hidden/mix state 返回 PP4；PP4 collapse、norm、LM/Markov sampling，
   再沿现有 PP relay 广播 target sample 和 draft tokens。
5. 每一跳比较 scheduler step、请求 ID 序列和请求数，任何漂移直接 fail closed，
   禁止写入 draft KV。PP4 发送 canonical request-slot 映射，PP1--PP3 不再用各自
   进程的本地 slot 编号索引采样状态；这修复了并发请求可能读到相邻请求 token、
   temperature 或 seed 的风险。当前同步 ring 不允许同一 rank 上多个 draft ring
   package 重叠，因此尚不需要跨 batch FIFO；后续异步化时必须按
   `(scheduler_step_id, req_idx, generation)` 分配独立 slot。

第一版必须 eager 执行独立通信，只在单请求和并发 2/4/8 的语义、KV 隔离、取消
路径都通过后，才为每个本地 draft layer 捕获分段 CUDA Graph。NCCL send/recv 不放
进单卡 graph；固定 buffer 地址用于后续 graph replay。

## 实施门禁

- 启动器固定均衡 target，默认启用 DSpark-5；设置 `DSV41_DSPARK_TOKENS=0` 可切回
  target-only。任何非 `8,8,8,8,8` 的 DSpark 分层均 fail closed，不会再次隐式
  选择 `8,9,9,9,5`。
- 已完成 `mxfp4_int8_sm80` draft `topk=3` CUDA dispatch、rank-local draft model
  ownership、PP1 私有 embedding 加载和 correctness-first 同步 ring。当前 eager
  动态 tensor-dict ring 已通过单路、8 路并发、4×32K 和取消后槽复用门禁；后续若
  改为异步 ring 或分段 CUDA Graph，必须重新执行这些门禁，不能沿用当前结论。
- V4.1 SM80 sparse-prefill gather 已吸收 DSV4F 的 byte-budget 方案：warmup 按
  1M 最坏长度和 512 MiB 上限预留一行，运行时按 batch 实际最大长度决定每次
  gather 的请求数。它把 1M 配置 PP2 的空闲显存从约 507 MiB 提升到 3,459 MiB，
  同时保留 1,078,799-token KV 容量。
- 1M 默认配置实际完成单路 32,763-token 检索，24.171 秒，答案正确，运行最小
  显存余量 2,197 MiB。动态实际长度后，4×32K wall 从保守 1-row gather 的
  128.681 秒降到 94.327 秒（改善 26.7%），4/4 答案正确，最小余量 2,575 MiB。
  尚未实际执行接近 1M token 的完整 prefill，因此当前 1M 结论是容量可分配，
  不是百万 token 端到端吞吐验收。

## 2026-09-15 target graph IMA 排查与修复

先关闭 DSpark 做同配置对照后，32K prefill 能完成，但第一个 target
`next_n=1` graph replay 触发 asynchronous illegal memory access；强制 eager
则 32K/16 完整通过。这把问题限定在 target graph 生命周期，而不是 DSpark ring、
q6 或 Engram。

`VLLM_DEBUG_WORKSPACE=1` 给出了直接证据：profile/capture 前 PP0--PP4 的 lane-0
共享 workspace 分别稳定在 `769/769/1425/1425/1425 MiB`，但真实 32K prefill
又从 sparse indexer 路径把它们扩大到
`796.72/796.72/1481.44/1481.44/1481.44 MiB`。workspace manager 扩容会释放旧
buffer，而 decode graph 仍持有 capture 时的旧地址，因此首轮 replay 访问已释放
显存。

根因是 SM80 Triton indexer 新增的 BF16 K dequant caller buffer 使用了
`kv_cache.shape`：显存 profile 发生在 KV cache 绑定前，只能看到空 placeholder；
运行时绑定后的物理 cache 容量为 1,078,799 token，比逻辑
`max_model_len=1,048,576` 大，导致 graph capture 后首次 prefill 再次扩大共享
workspace。修复后 caller buffer 只使用 profile/runtime 都一致的逻辑上界；当多并发
K gather 的合计行数超过该上界时，回退到该次调用的独立 BF16 输出，不扩大或替换
graph 共用的全局 workspace。

修复后的 target-only、请求 `FULL`（后端能力解析为 `FULL_DECODE_ONLY`）、PP5、
`max-num-batched-tokens=2048` 已连续完成四次精确 32K/16 请求，所有请求 HTTP 200，
workspace debug 日志在启动预留后没有任何 resize。最终记录为 TTFT `6.730 s`、
decode `25.778 token/s`；短语义门禁按指令只返回“巴黎”。固定 seed 下输出哈希仍会
漂移，这与既有 DSV4 稀疏模型结论一致，不能把单次哈希作为语义金标准。

紧邻的 DSpark-q5 对照保持 draft eager、target graph：32K/16 在 `10.769 s` 内完成，
32K/64 在 `14.575 s` 内完成，均未再出现 workspace resize、IMA、worker 轮动或请求
不结束。32K/64 的 draft 接受率为 `17.71%`、mean accepted length `1.886`，decode
为 `11.281 token/s`；同为 32K/16 时 DSpark 为 `9.164 token/s`，target-only 为
`25.778 token/s`。因此本轮只确认可靠性错误已修复；DSpark 接受率和同步 ring 开销
仍是下一轮独立性能问题，不能把这组结果表述为 DSpark 已达到生产收益门槛。

## 已修复的 decode 回退：单请求快路径不再被 `max_num_seqs=8` 关闭

上面的 target-only `25.778 token/s` 不能当作正常水平。同一 32K/64 口径下，
`max_num_seqs=1` 的记录为 `35.65–37.20 token/s`（`moe_tile_ab/baseline_16x128_32k.jsonl`），
即当时 decode 慢了约三成。差异来自为支持 DSpark 多并发而把启动器默认值从
`max_num_seqs=1` 改成 `8`：本项目的单请求快路径原本在**构造时**用
`max_num_seqs == 1` 判定，改成 8 后它们在运行期不再生效。

| 快路径 | 门控 | 改为 8 时的退化（已修复） |
| --- | --- | --- |
| SWA builder token-map | `sparse_swa.py` | 每步重建 `token_to_req_indices` |
| SWA validity | `sparse_swa.py` | 每步重算 `slot_mapping >= 0` |
| direct-ragged tail 免清零 | `amd/rocm.py` | 恢复每步 `decode_swa_lens[num:] = 0` |
| paged decode SWA | `amd/rocm.py` | 回到逐 token gather 索引 |
| sparse-MLA / compressor token-map | `sparse_mla.py`、`compressor.py` | 每层每步重建映射 |

退化后的动态路径是 `repeat_interleave + copy_ + zero_` 的逐层逐 token 执行。

修法是删除这 5 处构造期门控，让快路径只由运行期条件决定，而这些条件是本来就存在的：
token-map 与 validity 判断 `num_reqs == 1`，paged decode 额外要求
`num_decode_tokens == 1 and not non_causal`，direct-ragged 免清零沿用
`_use_direct_ragged_swa`。删除后 8 并发容量保留，单请求快路径恢复，多请求批次仍走
动态路径（由 `num_reqs` 分支保证），并有回归测试锁定这一点。

需要一次默认 `max_num_seqs=8` 的 32K/64 复测来量化最终收益，本轮未运行。
prefill 侧本来就没有回退：两组 32K TTFT 基本一致（`6.44–6.79 s` 对 `6.47–7.29 s`）。
