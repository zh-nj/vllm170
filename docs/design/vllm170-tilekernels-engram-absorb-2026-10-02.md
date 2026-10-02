# TileKernels Engram 吸收实验：FP32 权重预乘与提前 mask

本轮完成候选实现、CMP170HX 实测及正确性边界检查。FP32 预乘有微小的
gate 收益，但不是当前 target 验证热点；提前 mask 对屏蔽行有效，文本 prefill
会回退。两者均不接入生产，本轮没有推理代码改动，也没有模型重载。

前序源码研究见 [Engram 对照研究](vllm170-tilekernels-engram-gating-review-2026-10-02.md)。
本轮基准为 `benchmarks/kernels/deepseek_v41_engram_gate.py`，
完整数据、代码生成证据及诊断位于
`artifacts/tilekernels-engram-absorb-20261002/`。

## 实验边界

- vLLM HEAD：`4df3a185db133e63934d5739cc18582a215be609`。
- TileKernels 参考：`66258df6175d2f630ffecb04c5ab66bff8a2ae6a`。
- GPU：SM80 CMP170HX，74 SM、约 64 GiB、32 MiB L2；PCI `02:00.0`，
  UUID `f385e0d6-452b-ad54-922e-ef7f4deb6216`。生产使用其他五张卡。
- 实验前、实验中服务的运行/排队请求均为 0，没有并行 serving 测量。
- Torch `2.13.0+cu130`，Triton `3.7.1`。
- 读取真实 checkpoint 的层 1、14 的 BF16 `[4,5120]` q/k，随机 BF16
  hidden/KV。没有读取完整 Engram 表，没有重写 FP8 WKV。
- 测量仅含 post-WKV gate，不含 embedding lookup、WKV、metadata、PP 或 draft。

比较原 kernel、FP32 预乘、提前 mask、二者组合。预乘在计时外生成
`q.float() * k.float()`；这是静态预计算的最有利条件，尚未实现模型加载钩子。
所有候选保留 vLLM 的正零 signed-sqrt 规则、独立 hidden/key RMS 分母及输出布局。
另外增加不启用两个改动的同源 control，排查候选复制、mask 读位置等干扰。

计时采用交错顺序的 CUDA Graph/Event，每个配置 24 个样本。warm graph
包含 128 次调用，避免微秒 kernel 的宿主提交间隔主导结果。cold graph 只有
一次调用，每次在 Event 外清理 64 MiB L2；此口径包含单次 graph replay 的
固定成本，不能当作纯 kernel 时间。cold 值呈约 0.512–1.024 µs 的阶梯，
不用于判定亚微秒收益。没有使用本卡不可用的 CUPTI/NCU。

## 测量结果

下表为两层各自中位数之和，单位 µs，来自第一轮 `gate-ab.json`。
这是 gate 成本估计，不是 PP5 服务的端到端测量。

| token 数 | mask | 原 gate | FP32 预乘 | 提前 mask | 预乘加提前 mask |
| ---: | --- | ---: | ---: | ---: | ---: |
| 6 | 全 active | 8.160 | 7.748 | 8.180 | 8.408 |
| 12 | 全 active | 7.544 | 7.432 | 7.648 | 7.896 |
| 256 | 全 active | 55.272 | 53.988 | 58.880 | 54.716 |
| 2048 | 全 active | 369.796 | 352.832 | 383.488 | 363.660 |
| 256 | 半 active | 55.224 | 53.960 | 39.252 | 38.896 |
| 2048 | 半 active | 369.172 | 352.864 | 289.568 | 285.616 |
| 256 | 全 inactive | 55.108 | 53.700 | 17.816 | 18.096 |
| 2048 | 全 inactive | 366.364 | 352.916 | 214.604 | 215.168 |

同源 control 对照 `control-ab.json`：q6 两层原 gate 为 **8.152 µs**，
control 为 **8.220 µs**，FP32 预乘为 **7.740 µs**。
预乘相对原 gate 节省 **0.412 µs**，相对 control 节省 **0.480 µs**。
2048-token 全 active 的原 gate 为 **370.692 µs**，预乘为 **353.096 µs**，
节省 **17.596 µs，约 4.75%**。

不同形状/层之间存在 GPU 频率变化，例如先测的层 1 q6 比后测的层 14 q6
更慢。只比较同一形状内交错测量的候选，不根据跨形状绝对值拟合 q 曲线。

## 收益为什么小

预乘消除了一次逐元素 FP32 乘法，确实降低指令数。同源连续布局 control
与预乘生成代码对比（`code/manifest.json`）：

| 项目 | control | FP32 预乘 |
| --- | ---: | ---: |
| PTX `mul.f32` | 71 | 39 |
| SASS `FMUL` | 59 | 35 |
| SASS `FFMA` | 119 | 119 |
| SASS 总指令 | 576 | 480 |
| 寄存器/线程 | 60 | 64 |
| spill | 0 | 0 |
| shared memory | 32 B | 32 B |

静态指令数不是动态执行次数。减少乘法没有减少权重字节：两个 BF16 数
总共 4 B，预乘后一个 FP32 数也是 4 B。原 kernel 已融合 RMSNorm、门控、
残差写回，也没有寄存器溢出；此次不存在消除中间张量或 spill 的大额收益。

以既有同生产配置的 32K/512 结果 `round7/production-gate.json` 中
**39.4088 ms/轮** 为量级参照，0.412 µs 对应潜在整轮收益 **0.00105%**。
即使把两层 gate 完全消除，warm 测量对应的上限也仅 **0.0207%**。
这是估算，不是本轮新测的端到端提速。没有为这个量级重启 PP5 做 A/B。

若假设 32K 输入恰好执行 16 次完整 2048-token tile，预乘节省约
**0.28 ms**；这也只是按调用次数估算，不是 CED 实际 TTFT 测量。

提前 mask 在 inactive CTA 中跳过 key/value/q/k 与三次归约，解释了屏蔽行
收益；全 active CTA 多了动态分支。256-token 文本 gate 回退约 **6.53%**，
2048-token 回退约 **3.70%**。当前 text decode 的 mask 全 active，不能按
全屏蔽或半屏蔽结果推广。

## 正确性与适用边界

`gate-ab.json` 的 24 组及 `control-ab.json` 的 4 组连续布局随机输入，
各候选与原 kernel 的 BF16 输出均逐位一致。原 kernel 同时与独立的
PyTorch 公式按现有 Engram 测试的 `rtol=atol=1e-2` 比较。
这些结果不构成所有布局或数值边界的逐位一致保证。

`gpu-contract.json` 进一步检查：

- key 全零且 value 为 `2.015625` 时，原 kernel 及候选输出都是 `2.015625`，
  保留零点 clamp 的正幅度；没有沿用 TileKernels Torch reference 的 `sign(0)`。
- stride 为 2 的 hidden/KV/q/k/mask/output 首次运行逐位一致；padding 的
  无效 KV 行设为 NaN 后仍被正确屏蔽。
- 更新 strided hidden 并把 mask 改为全 active 后，FP32 预乘与组合候选各有
  1 个 BF16 元素相差 1 ULP（绝对差 `5.960464477539063e-08`）；control 和
  单独提前 mask 无差异。相同候选的 eager 和 graph 输出逐位一致，排除了
  graph replay 缓存旧 mask/hidden 的解释。
- `strided-code/manifest.json` 显示原/control 的 `sizePerThread=1`，预乘改为
  `sizePerThread=4`，归约使用的线程数据布局发生变化。这支持归约舍入差异
  的解释，尚未逐个 FP32 中间量定位；不能把 CPU 预乘精确性推成 GPU 全布局精确。
- 全 inactive 的 `hidden=-0.0`，提前返回保留负零，原算术路径产生正零，
  20,480 个输出位模式不同，但数值相等。
- 全 inactive 的 NaN KV，原输出仍为 NaN，提前返回输出 hidden。
  这是构造的非有限输入诊断，不是生产故障证据。

现有公共 Engram forward 测试：**6 passed**，含 SP shard、padding、mask、
零 token；没有修改推理代码，所以没有新增模型输出或吞吐评估声明。

## 接入决定与复现

本轮保留有界 benchmark 和实验数据，**不启用运行时候选**。FP32 预乘仅列为
后续较大热点融合中的参考；若重新采用，必须处理加载/重载后的缓存刷新，
并约束或验证归约布局。提前 mask 只值得在实际 image 工作负载中重新评估，
还需要明确零位模式及非有限值契约，不能推广到全 active 文本路径。

最终确认 production PID 未变，服务 health 200、模型路径正确，五张服务卡
仍属原 cgroup，实验卡无残留 compute 进程。先前 17 个 dirty 路径内容保持。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  .venv/bin/python benchmarks/kernels/deepseek_v41_engram_gate.py \
  --expected-uuid f385e0d6-452b-ad54-922e-ef7f4deb6216 \
  --tokens 6 12 256 2048 --samples 24 \
  --output artifacts/tilekernels-engram-absorb-20261002/reproduce.json

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  .venv/bin/python -m pytest tests/kernels/test_engram.py \
  -k fused_engram_post_wkv_matches_reference -q
```
