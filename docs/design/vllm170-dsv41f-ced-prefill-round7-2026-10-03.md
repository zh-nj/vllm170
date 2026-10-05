# DSV41F prefill 第七轮：BM32 中等尾块筛选与回退

日期：2026-10-03。第五、六轮代码已提交为 `a33bd2b5cb`。
配置保持 DSpark=5、CED 开启、PP5/TP1/DP1；本轮基线为已上线的
M=2044 BM128 pairs / BM64 tails / BM16 small tails。
原始证据位于 `artifacts/dsv41f-ced-prefill-round7-20261003/`。

结论：BM32 候选正确，但阶段收益小且部分低于基线波动，未进入服务
A/B，已恢复全部实验源码与 build extension。生产实例和安装的 native
extension 从未替换。这里没有证明端到端无收益，也没有新增提速主张。

## 目标、代码路径与静态成本

将有效 slot 为17–32的奇数 BM64 尾块转给 BM32，保留现有 BM16。
同专家相邻 BM64 块仍优先配成 BM128；不会拆开已经成对的块。

第五轮实际 DSpark5 语料首个 M=2044 chunk 的20层路由完全一致。
其中奇数尾块5299个：1–16 slot 为3280个，17–32为1106个，
33–64为913个。现有填充后行数362816，候选327424，减少35392行
即9.75%；有效 slot 总数245280不变。以上是逻辑行数，不是 HBM 实测。
逐层统计见 `route-costs.json`。

虽然 native selector 和 CUTLASS 类型已定义32×128，entry 的实际
dispatch 只接受16×64、16×128、64×128、128×128。因此实验补上
32×128 dispatch，并在 top-k6、M2044、E384、BN128 的精确条件下
复用已有 persistent loop，最多148 CTA。否则 BM32 会按全局容量
发出756×36/40个 CTA，包括大量空 tile。

benchmark 中暂增 `MediumTailRoutes`，设备端检查第17和33个 slot 的
sentinel，分成128/64/16/32四份元数据。传给 native 的有效 slot 上限
始终是全局12264；不读取 GPU count 到 CPU，也不改变 W2 slot 索引。
生产 `Experts.apply` 没有接入 BM32。

## 最小硬件实验

只使用空闲物理 GPU1 CMP170HX，UUID 为
`GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216`，SM80、74 SM。
正式 PP5 持续占用 GPU2–6；没有服务重启、q 扫描或 profiler 会话。

仅增量编译 `mxfp4_int8_moe_entry.cu`，链接 build 目录 extension，
在独立进程中预加载。安装给生产的 `.so` 没有覆盖。
基线和候选使用同一份实际路由 IDs/weights，激活及 packed expert
weights 为合成数据；另测六热专家合成路由。

计时边界包含 route repartition、metadata copies 和一个完整
W13/W2 stage，排除初始 align、分配、权重构建和编译。
CUDA Event + Graph，图内4次调用，3轮交替对照，每轮7个样本。
权重和地址复用；这是 warm-cache GPU 微测，未包含 CED eager 路径的
全部 host launch 成本，不能替代服务 TTFT。

| 路由 / stage | 已上线基线 µs | BM32 候选 µs | 耗时变化 | 判断 |
| --- | ---: | ---: | ---: | --- |
| corpus layer0 W13 | 9206.27 | 9053.44 | −1.66% | 可测 |
| corpus layer14 W13 | 7814.14 | 7772.93 | −0.53% | 小于1.04%基线波动 |
| corpus layer0 W2 | 4883.46 | 4818.94 | −1.32% | 可测 |
| corpus layer14 W2 | 4147.46 | 4095.74 | −1.25% | 可测 |
| 六热专家 W13 | 3659.78 | 3662.34 | +0.07% | 波动内 |
| 六热专家 W2 | 1993.47 | 1992.45 | −0.05% | 波动内 |

结果见 `medium-tail.json`。六热专家没有 BM32 尾块，适用于观察额外
空列表开销，不能替代真实 periodic 或 corpus 端到端请求。

## 消融：为什么填充减少没有等比例提速

对同样两个 layer 的路由，分别测元数据构建、预建元数据后的整个
GEMM stage，以及原 BM64 尾块列表与候选 BM64/BM32 两列表。
消融仍使用3轮、7样本，见 `ablation.json`。

元数据构建由 layer0 的10.320→12.912 µs、layer14 的
10.144→12.512 µs，每次多2.37–2.59 µs。整个 GEMM stage 在排除
元数据后收益仍只有约0.37%–1.38%；其中 W13 layer0 低于波动。
只计尾块列表时，W13 layer0 改善2.34%，W13 layer14 的0.91%低于
波动；W2 layer0 改善3.19%，W2 layer14 反而1196.800→1202.944 µs，
慢0.51%。不同计时边界会改变 cache 和 launch 上下文，不能把这些
分段耗时直接相加，解释主实验的全部差额。

源码可确定的原因是：BM64→BM32 保留每个 expert tail 的一个 M tile，
逻辑 M/N tile 数没有减少。`Mxfp4PackedBThreadblockMmaSkeleton` 的
`kPackedBStageBytes` 和 `kShiftStageBytes` 只依赖 N/K；在这两个
tile 中，N/K、warp 数及 warp-N 形状相同，因此每个 tile 的 packed
权重与 shift 搬运量不随 M 减半，B fragment 解码工作也未因 M 缩小而
消失。它只减少部分 A staging 和 MMA 工作，同时增加一个 kernel
列表和元数据复制。

这支持“缩小 M 没有减少权重侧工作”的解释；没有 HBM counter 或
指令级消融，不将其写成全部耗时都是带宽或解码的证明。后续优先
研究 packed 权重搬运/解码或尾块调度，停止继续按 padding 降幅外推
端到端收益。

## 正确性、回退与复现

24个独立参考用例覆盖 W13/W2 的
M=1/16/17/31/32/33/63/64/65/127/128/129，变化 scale、全部有效
slot 和输出 sentinel；全部 BF16 逐位一致。实际 M2044 layer0/14
及六热专家的两个 stage 也与已上线基线逐位一致。

临时扩展现有 metadata Graph replay 用例，验证 BM64→BM32→BM64
路由切换，没有漏 slot、重复 slot 或旧 count 残留。包含既有 gate
和生产 apply 基线测试的结果为37 passed、247 deselected。
这不构成 BM32 生产 apply 或完整模型验证。

已保存 `candidate.patch`、三个 `*.measured.txt` 源码快照、native
候选哈希、编译日志、实验命令及结果。随后恢复本轮实验涉及的
C++、benchmark、test 文件和 build `.so`，逐一确认与提交
`a33bd2b5cb` 的基线哈希一致。未保留未验证的推理开关或 BM32 dispatch。

本机复现入口如下；先应用封存的 `candidate.patch`，使用独立进程
加载候选 build extension，避免覆盖生产安装的 extension。

```bash
cmake --build cmake-build-release --target _moe_C_stable_libtorch -j 1

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
PYTHONPATH=/mnt/data/apps/vllma100/vllm170 \
.venv/bin/python artifacts/dsv41f-ced-prefill-round6-20261003/load_candidate.py \
  --medium-tail --expected-uuid GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
  --out artifacts/dsv41f-ced-prefill-round7-20261003/reproduce.json
```

正式服务保持原 MainPID 4007178，DSpark=5、CED 和第六轮 BM16 路径
开启。已检查17份 runtime source 哈希、native 哈希、5个 worker 的
映射 inode、GPU归属、health 200、private RPC 404、无等待请求。
追加的冷语料和 prefix 请求用于交付复查，不作为候选收益 A/B；结果
保存在 `production-check.json`。新一轮没有可宣称的端到端加速。
