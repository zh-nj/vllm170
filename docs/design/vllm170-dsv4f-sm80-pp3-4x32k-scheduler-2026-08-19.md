# DeepSeek-V4-Flash SM80 PP3 4x32K 并发调度与稳定性修复

## 结论

三张 CMP170HX、PP3/TP1、四并发精确 32,768-token 输入、每路 256-token
输出、DSpark-5 的速度坍塌由两个独立问题叠加产生：

1. PP 异步启动期只有首个请求进入 scheduler，原 adaptive long-prefill
   只限制它的第一块；下一步立即恢复 1,715-token 全预算，形成 FCFS 长
   prefill 垄断。四路 TTFT 因而拉开约 22 秒，早到请求的 decode 又被后续
   prefill 阻塞。
2. DSV4 packed KV cache 的多个层视图共享一个 block slab。旧清零器从每个
   层视图的偏移地址开始清除一个完整 block stride，既覆盖下一块，也可能在
   高 block ID 回收时越过 backing 尾部。连续第二轮 4x32K 可稳定触发
   `KVBlockZeroer` 中的 CUDA illegal memory access。

修复后，关闭 PP 接收缓冲复用的同一服务连续完成三轮 4x32K/16 和两轮
4x32K/256，未出现 CUDA 错误。两轮完整负载均值为 5,848.1 aggregate
prefill tok/s、253.6 mixed aggregate decode tok/s、326.2 tail aggregate
decode tok/s，8/8 语义探针通过，无跨请求串扰。重启并恢复生产默认接收缓冲
复用后，又连续通过两轮短输出和一轮完整输出；启用 production prefix cache
后再连续通过两轮完整输出。

## 测量口径

- `aggregate prefill`：四路 prompt token 总数除以首请求开始到末请求首 token
  的区间。
- `mixed aggregate decode`：服务发出第一个 token 后直到全部请求结束期间，
  实际返回 token 的聚合速率。它包含 prefill/decode 干扰，是本问题的主指标。
- `tail aggregate decode`：所有请求都已有首 token 后的聚合速率，只表示
  decode-only 尾段，不能替代 mixed 指标。
- 语义质量：每个请求检索自己的唯一口令；同时检查是否输出其他请求口令。

共同配置：模型 `/media/z/data6/models/DeepSeek-V4-Flash-0731`，物理 GPU
`1,2,3`，PP3/TP1，分层 `15,15,13`，FP8 KV，Marlin MoE，
`TRITON_MLA_SPARSE_DSV4`，`max-model-len=131072`，
`max-num-batched-tokens=1715`，`max-num-seqs=4`，DSpark greedy MTP5，
关闭 prefix cache。用于定位的主对照统一设置
`VLLM_PP_REUSE_RECV_BUFFER=0`，排除 PP 接收缓冲复用干扰。

## 调度坍塌

基线五次完整负载中位数如下：

| 配置 | prefill tok/s | mixed decode tok/s | tail decode tok/s | TTFT spread | wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原调度，五次中位数 | 4,658.4 | 41.3 | 315.2 | 22.10 s | 30.85 s |
| 修复后，两次均值 | 5,848.1 | 253.6 | 326.2 | 0.974 s | 25.52 s |

修复后 aggregate prefill 提升 25.5%，mixed aggregate decode 提升约
6.1 倍，wall time 缩短 17.3%。tail decode 没有同量级变化，进一步说明
坍塌发生在并发 prefill/decode 调度，而不是 decode kernel 本身。基线采样时
三张卡 SM 利用率持续为 96%-100%，也排除了“GPU 空闲，增大全局 batch 即可”
的解释。

`long-prefill-token-threshold=424` 使四个并发块占用 `424 * 4 = 1696`
个 target slots，适配 1,715 的 PP3 预算。原 adaptive 策略虽然会在检测到
多请求竞争后持续限块，但 PP 异步 scheduler 的最初几步尚未看到其他请求。
修复因此在 PP 启动深度内保留 424-token 上限；四个 in-flight batch 均已注入
后，真正单请求再恢复完整 token budget。非 PP 路径和无竞争稳态不变。

三次 scheduler 修复后的早期性能复测分别得到：

| prefill tok/s | mixed decode tok/s | tail decode tok/s | TTFT spread |
| ---: | ---: | ---: | ---: |
| 5,827.9 | 201.5 | 242.4 | 0.98 s |
| 5,837.3 | 197.5 | 234.7 | 0.97 s |
| 5,760.5 | 203.7 | 264.8 | 1.26 s |

## packed KV 清零越界

仅修调度后，第一轮请求通常成功，但同一进程再次运行 4x32K 会崩溃。使用
`CUDA_LAUNCH_BLOCKING=1` 后，真实故障点从后续 attention/TileLang 的异步
报错收敛到：

```text
gpu/model_runner.py:update_requests
  -> KVBlockZeroer.zero_block_ids
  -> _zero_kv_blocks_kernel
  -> Triton Error [CUDA]: illegal memory access
```

packed KV 的层视图满足：

```text
view.data_ptr = backing.data_ptr + layer_offset
view.stride(block_dim) = shared_block_slab_stride
```

旧逻辑将每个不同的 `view.data_ptr` 当作独立 segment。对 block `b` 清零时，
它实际写入
`backing + layer_offset + b * slab_stride .. + slab_stride`。只要
`layer_offset > 0`，就越过当前 block；最后几个 block 会越过 backing。
第一轮是否命中边界取决于分配到的 block ID，块回收后的第二轮更容易取得高
block ID，因此表现为“并发复用后才崩”。

修复对 blocks-first 且 block stride 大于单视图 block 数据的 cache：

- 以 `(storage base, block stride)` 对共享 backing 去重；
- 从 storage base 清零整个 block slab，而不是从层偏移重复清零；
- 保留 virtual block ratio，其他连续或非 blocks-first layout 沿用原路径。

GPU 回归测试构造两个有不同 storage offset、共享 12-element block stride 的
视图。它断言清零器只生成一个 storage-base segment，并且清除 block 2 后
block 0、1、3 完整保持不变。该测试在旧寻址下会覆盖 block 3。

## 连续实机验证

修复后的同一非 blocking 服务没有重启，按顺序运行五轮：

| 轮次 | 输出/路 | prefill tok/s | mixed decode tok/s | tail decode tok/s | TTFT spread | 语义 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 16 | 5,835.3 | 60.5 | 362.2 | 0.962 s | 4/4，无串扰 |
| 2 | 16 | 5,799.1 | 45.9 | 370.9 | 1.245 s | 4/4，无串扰 |
| 3 | 16 | 5,857.4 | 61.7 | 444.7 | 0.956 s | 4/4，无串扰 |
| 4 | 256 | 5,859.6 | 227.6 | 282.2 | 0.973 s | 4/4，无串扰 |
| 5 | 256 | 5,836.5 | 279.6 | 370.3 | 0.975 s | 4/4，无串扰 |

短输出轮次的 mixed decode 被约一秒的首 token 到达差异主导，只作为块回收
稳定性压力，不用于与完整 decode 吞吐横比。两轮 256-token 的 MTP draft 接受率
分别为 64.08% 和 75.30%，平均接受长度分别为 4.20 和 4.76 token/round。

恢复启动器默认的 `VLLM_PP_REUSE_RECV_BUFFER=1` 并重启后，同一服务继续
完成两轮 4x32K/16 和一轮 4x32K/256。最后完整轮为 prefill 5,860.8 tok/s、
mixed decode 207.0 tok/s、tail decode 255.4 tok/s、TTFT spread 0.968 秒、
wall 26.34 秒；4/4 语义通过且无串扰。该轮 MTP 接受率为 56.79%，平均接受
长度 3.84，因此 decode 与前两轮的差异不能全归因于 PP buffer 开关。三轮均
无 CUDA 错误，生产默认组合通过门禁。

最后以 `DSV4_MODE=production` 启用 prefix cache，再连续运行两轮完整负载：

| 轮次 | prefill tok/s | mixed decode tok/s | tail decode tok/s | TTFT spread | 语义 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 5,842.4 | 210.5 | 253.6 | 0.972 s | 4/4，无串扰 |
| 2 | 5,796.1 | 272.5 | 401.8 | 1.227 s | 4/4，无串扰 |

第二轮 MTP 接受率为 80.88%、平均接受长度 5.04，解释了较高的 decode 结果。
两轮服务日志均无 illegal memory access、Xid 或 worker error；生产命令的完整
cache/buffer 组合通过。

## 生产启动形状

四路 32K 是目标流量时，应显式使用四序列上限和 adaptive 424；不要只把
`max-num-batched-tokens` 降到 2,048/4,096，也不要用更大的 prefill chunk
重新引入 FCFS 阶梯：

```bash
CUDA_VISIBLE_DEVICES=1,2,3 \
DSV4_MODE=production \
DSV4_MAX_NUM_SEQS=4 \
DSV4_MAX_BATCHED_TOKENS=1715 \
DSV4_DSPARK_TOKENS=5 \
benchmarks/deepseek_v4/run_cmp170hx_pp3.sh \
  --enable-adaptive-long-prefill \
  --long-prefill-token-threshold 424 \
  --kernel-config '{"enable_jit_warmup":false}'
```

生产命令保留启动器已验收的 `VLLM_PP_REUSE_RECV_BUFFER=1`。本轮先使用 `0`
隔离根因，再用默认 `1` 完成 benchmark 和 production prefix-cache 模式的
重复负载门禁；三种路径都通过。

`enable_jit_warmup=false` 是当前工作区另一条未提交 warmup 修改下的启动规避项，
不是本次 scheduler/KV 修复的必要条件。它会使首轮 32K 请求编译若干 TileLang/
Triton shape；正式接流量前应先执行同形状预热。待该 warmup 修改独立验收后，
应恢复启动器默认预热并移除此覆盖。

## 验证命令

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/worker/test_kv_block_zeroer.py -q
.venv/bin/python -m pytest tests/v1/core/test_scheduler.py -q \
  -k adaptive_long_prefill
.venv/bin/python -m pytest \
  tests/benchmarks/test_deepseek_v4_concurrent_longctx.py -q
```

结果分别为 5/5、2/2、2/2 通过。实机 JSON 与服务日志位于 gitignored 的
`profiles/deepseek_v4/pp3-4x32k-scheduler-20260819/`。
