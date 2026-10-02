# TileKernels Engram gating 与 DSV41F 推理路径对照

日期：2026-10-02。vLLM HEAD `4df3a185db133e63934d5739cc18582a215be609`，
TileKernels 为指定本地目录 `/mnt/data/apps/vllma100/TileKernels`，HEAD
`66258df6175d2f630ffecb04c5ab66bff8a2ae6a`，工作区干净。
环境为 Torch 2.13.0+cu130、TileLang 0.1.15。
原始证据存于 `artifacts/tilekernels-engram-review-20261002/`。

## 1. 结论与候选优先级

有可取之处，但不能把 README 的“fused RMSNorm”当成当前 vLLM 缺失的一次融合。
当前 post-WKV 已在一个 Triton kernel 中完成双 RMS 统计、加权 dot、
signed-sqrt、sigmoid、共享 value 注入，没有物化 normalized x/k，
没有保存反向中间量。Engram 只在 target 的第 1、14 层出现。

| TileKernels 做法 | 当前 vLLM 对照 | 判断 |
| --- | --- | --- |
| RMSNorm/dot/gate/residual 单 kernel | 已有相同融合边界 | 保留现有融合，不重复记收益 |
| 两份 BF16 norm weight 合成 FP32 product | 当前每次分别读取 q/k 并相乘 | 优先小候选：在加载完成后预计算，保持 FP32 |
| masked image 行先直接 passthrough | 当前先算 norm/dot/gate，再将 gate 置零 | 值得用于 image prefill；文本 q6 没有该收益 |
| 单 warp、vec8、cp.async 双缓冲、persistent grid | 当前每 token/head 一个 8-warp CTA | 编译可行；先评估 decode grid，再考虑适量 prefill，对 q6 没有性能证据 |
| `save_for_backward=False` | 当前本来不写 dot/gate/rstd buffers | 是移植时必须选的模式，不是当前新增优化 |
| `inplace=True` | 当前分配独立输出，之后进入 mHC pre | 先审计 alias/lifetime；不会消除输出写回，不优先改 |
| backward + weight-gradient reduction | 当前 q/k `requires_grad=False`，服务无 backward | 训练可参考，不能减少本推理链的工作 |
| save/no-save/inplace/mask 对照测试 | 当前已有 gate/布局/SP 测试 | 吸收测试思路，补 q6/零点/Graph 边界；不可盲用其 reference |

目前最小可研究改动是静态 FP32 `q_weight * k_weight`，而不是整包替换。
它只减少 post-WKV 里的乘法和加载指令组织，权重读取字节数没有降低。
本轮没有 gate 延迟实测、生产 A/B 或推理代码改动，不宣称端到端提速。

## 2. 实际执行路径与计算合同

入口为 [DeepseekV4DecoderLayer.forward](../../vllm/models/deepseek_v4_1/nvidia/model.py#L400)：

```text
n-gram hash -> host/UVA table lookup + FP8 dequant -> staged BF16 rows
  -> ReplicatedLinear WKV -> fused Engram post-WKV
  -> mHC pre -> attention
```

注入位于上一 sublayer 的 mHC post 与本层的 mHC pre 之间，mHC 必须看到
注入后的完整 hc stream。不能把注入移动到 mHC pre 之后。
当前 lookup 来自 `ParallelEngramEmbedding.lookup`：GPU 通过 UVA 读取
registered/pinned host table，`_engram_lookup_kernel` 同时 gather、解码 FP8、
乘 UE8M0 scale 并写 BF16。不是旧报告中那条 CPU gather/dequant + H2D 路径。
`VLLM_DEEPSEEK_V4_ENGRAM_GPU_DEQUANT` 当前只有 env 声明，无执行使用点；
启动参数里的该值不能证明旧路径仍被启用。

依据真实 checkpoint header：H=5120，hc=4，24 个 hash columns，head_dim=256。
每层 q/k 分别是 `[4,5120]` BF16；WKV weight 是 `[25600,6144]` F8_E4M3，
157,286,400 bytes，scale 是 `[800,192]` F8_E8M0，153,600 bytes。
生产日志选择 `MarlinMxfp8LinearKernel`；本轮只研究 projection 之后的 gate，
没有重启已放弃的 FP8 dense TileLang 路线。

对每个 token t、hc copy h，令 x 为 hidden，k 为 WKV 的对应 key，v 为共享 value：

```text
rx = rsqrt(mean(x*x) + 1e-20)
rk = rsqrt(mean(k*k) + 1e-20)
s  = sum(x * q_weight * k_weight * k) * rx * rk / sqrt(5120)
g  = sigmoid(signed_sqrt(clamp_min(abs(s), 1e-6)))
y  = BF16(x + g*v)
```

norm 统计、乘法和归约使用 FP32；最终输出为 BF16。
norm denominator 依赖未乘 norm weight 的 x/k，所以不能简单把 q/k 融进 WKV
投影后丢掉原始 key，再用加权 key 的 RMS 代替；会改变公式。

兼容性差异：

- 当前 `token_mask=True` 表示激活，TileKernels `image_token_mask=True` 表示
  禁用，接入时必须反转。当前 image sentinel 已在 model 中取反传给 gate。
  TileKernels 在读取 key/value/weight 前跳过 image 行；当前仅在 gate 算完后
  置零。对现有一-token/head CTA 可参考均匀早分支，仍须为 out-of-place 写回 x，
  但可省 masked 行的 key/value/weight 读取、RMS 和 dot。纯文本无此节省。
- 当前 kernel 接受显式 stride、`num_kv_tokens` 和 SP 尾部无效行；TileKernels
  固定 `[N,hc,H]`、`[N,hc+1,H]` 布局，没有独立 valid-KV-row 上界。
  同形 contiguous TP1 view 是候选前提，不能覆盖所有原 fallback。
- 当前 source 对零的符号用 `s < 0`；TileKernels CUDA 用 `copysign`。
  负零这一标量边界的语义也应明确，不能只测普通随机输入。

## 3. 静态权重预乘：可借鉴，收益边界很窄

TileKernels 的 [fused weight kernel](../../../TileKernels/tile_kernels/engram/engram_fused_weight_cuda.py#L15)
将两份 BF16 weight 转 FP32 后逐元素相乘，输出 FP32 product。
但高层 `EngramGateFn.forward` 每次调用都会运行这个独立 kernel，
仍默认 `save_for_backward=True`。它没有推理期固定权重的缓存策略。
不能直接使用该 autograd wrapper 获得推理收益。

当前权重固定且 `requires_grad=False`。合适的借鉴方式是在 q/k 加载完成后
建立 FP32 buffer，之后 post-WKV 直接读取；权重重载、设备迁移时必须重新建立，
初始化未加载的空参数不能用于预乘，也不应在 Graph replay 中追加该 kernel。

每层权重字节对比：

| 形式 | bytes |
| --- | ---: |
| 原 q BF16 + k BF16 | 40,960 + 40,960 = 81,920 |
| FP32 fused product | 81,920 |

两层总 buffer 为 160 KiB。保留原始参数时这是新增空间，不是读取量减半。
数学表达式的逐维乘法由三次变为两次；q6 两层合计少做 245,760 次逐元素
乘法，但仍有 RMS 统计、归约、gate 与读写。该算术量不是延迟或速度收益证明。
当前 TTIR 确认原 kernel 仍依次执行 `hidden*q`、`*k_weight`、`*key`。

CPU probe 只读取两个 shard 的 header 和各 80 KiB q/k 原始数据，没有加载
157 MB WKV weight 或 94 GiB table。真实两层权重、随机 BF16 x/kv，
N=1/6/12/256，共 8 个 case、11,264,000 个输出元素：

- FP32 预乘的逐维 product、dot、最终 BF16 output 均逐元素一致。
- 若 fused product 改为 BF16，38,823/40,960 个 weight product 被舍入，
  320,914 个输出元素变化。因此不能为省带宽把 product 压回 BF16。
- 样例中 product 无非有限值或非零 FP32 subnormal。
- 相同有限输入上，8 个 case 的 masked 行直接 passthrough 与当前 gate 置零
  公式数值一致。该比较不保证 signed-zero 的位模式相同。

这验证的是 CPU 表达式与给定数据，不证明 GPU 编译后的逐位等价。
Triton/NVCC 的 FMA、归约树及 exp/rsqrt 实现仍须与原 GPU kernel 对照。
也不是实际 serving hidden states 或模型质量评测。

## 4. 单 warp 双缓冲：SM80 可编译，不等于 q6 合适

TileKernels forward 固定 32 threads、vec8，H5120 选择 blk_d=1024。
缓存全部 x 于 shared memory，key/value 用两份 1024-element BF16 buffer，
每 thread 以小向量累加三个 FP32 reduction，再做 warp reduction；
输出阶段复用 x shared，预取 value 和下个 token。
相比一次持有完整 padded row，这种组织可以限制寄存器向量的活跃范围。

shared memory 为 `5120*2 + 1024*4 = 14,336` bytes/CTA。
persistent grid 按 SM shared memory 估计，而没有按 N 截断：

```text
blocks_per_sm = min(smem_per_sm // 14336, 16)
num_persistent_blocks = num_sms * blocks_per_sm // hc
grid = [hc, num_persistent_blocks]
```

以离线使用的 164 KiB SM shared memory 资源假设，blocks_per_sm=11，
总 grid 约为 `11*num_sms`；q6 只有 24 个 CTA 真正处理 token/head。
其余 CTA 的 token 范围为空，并不会产生有用并行。因此 decode 应先评估
按 N 限制 grid 或沿原一-token/head grid 改造，而不是直接搬 persistent grid。
大 N 的持续负载与 q6 的启动开销需分开测。

实际已有缓存与本轮离线编译结果：

| 项目 | 当前 Triton 缓存 | TileKernels 离线候选 |
| --- | ---: | ---: |
| target | sm80 | sm80 |
| threads/CTA | 256 | 32 |
| registers/thread | 60 | 90 |
| stack/local 或 spill | stack/local=0 | stack/spill load/store=0 |
| shared memory | metadata dynamic=32 bytes | 14,336 bytes |

寄存器数不能单独比较快慢：线程数、CTA 数和 shared memory 都不同。
当前 gate 已是 8 warp 且没有 spill，不能再套用 sparse attention 那次
4→8 warp 消除 spill 的解释。这里也没有 packed-word 重组热点的证据。

离线检查用 `get_tir` + `tilelang.engine.lower`，显式 sm80，N 为 dynamic，
no-save、image mask、out-of-place、PDL=false；随后按 TileLang 默认 C++20/
include flags 用 nvcc 编 cubin。CUDA device 不可见，未初始化 CUDA、未发 kernel。
shared-memory 查询以 164 KiB、num_sms=1 的资源 stub 替代，后者只用于编译检查，
不是物理设备清单或生产 grid 建议。
首个手写 nvcc 命令遗漏 template include，失败日志保留；修正为库的
`default_compile_options` 后通过，未修改上游 kernel。

官方 README 支持 SM90/SM100，源码注明只针对 H4096/7168 调优。
本轮补足了 H5120 的 SM80 编译证据，但没有补足执行正确性或性能。

## 5. Reference、backward 与测试吸收

发现一处独立的 reference 边界问题：

- TileKernels CUDA 使用 `copysign(sqrt(clamp(abs(s))), s)`。
- `tile_kernels/torch/engram.py:207` 使用 `sqrt(clamp(abs(s))) * sign(s)`。
- s=0 时前者保留 0.001 的幅度，gate≈0.50024998；后者为 0，gate=0.5。

CPU 构造 x=1、key=0、value=2.015625，H5120/hc4：CUDA 源码表达式与当前
vLLM 公式输出 BF16 2.015625，而 TileKernels reference 输出 2.0，
20,480 个元素全部不同。该结果是源码公式的 CPU 复现，不是 GPU kernel 执行。
普通随机 case 的两份 reference 也有 459 个 BF16 输出元素差异，来自不同
乘法/归约表达式；不能据“随机输入一般很接近”省略数值合同检查。

forward tests 值得吸收 save/no-save 一致、inplace 一致、mask token 原样返回
等覆盖，但默认 CUDA hidden 列表没有5120，default/core N 为512/8001，
full level才加0/1，仍没有6/12。当前 vLLM gate suite 有 SP/padding/stride
语义与 H5120/N1 样例，移植时应扩展同一套测试至 q6、多请求、动态 Graph replay、
零点与 clamp 两侧，不另建只重复实现的测试套件。
另一个 mask 语义细节：对构造的 masked NaN KV，直接 passthrough 仍返回 x，
但当前 `x + 0*NaN` 公式产生 NaN。CPU 例中差异覆盖20,480个元素。
这是无效输入的合同诊断，不是发现生产故障；移植时应明确 masked 数据的要求，
避免把 early exit 当成所有特殊值也完全等价的机械改动。

backward 将 grad_x、grad_kv、weight partial 在一个 kernel 中计算，
partial 为 `[num_persistent_blocks,hc,H]` FP32，避免各 token 对 weight 全局原子加。
随后 weight reduction 归约 partial，并融合链式乘法和 `main_grad` 累加：

```text
grad_q += sum(partial) * k_weight
grad_k += sum(partial) * q_weight
```

这是训练端“分块 partial → 固定归约 → 融合 epilogue”的参考模式，不是当前
target verification 的调用路径。不能把免原子加或 backward 吞吐换算成 decode 提速。

## 6. 下一步可执行边界与交付

若继续实验，先量当前两层 post-WKV 的独立 CUDA Graph 延迟，再只改变 FP32
静态权重 product。使用实际 H5120/hc4/N1、6、12及代表 prefill shape；
保持 mask/SP/PAD fallback，原 output buffer 和归约顺序不动。
正确性与编译资源通过后才比较延迟；gating 节省若低于噪声或占比极小，停止该候选，
分析是 launch floor、相同字节量还是原 kernel 已有效融合。
只在可测收益值得一次有限 serving A/B 时进一步接入，而不先重载模型扫 q。
image prefill 可独立评估 masked-row 早返回，不改变 hash、lookup 和 WKV 的
计算顺序；不能用该条件收益声称文本 target q6 也会加速。

当前第七轮 target 诊断没有独立 Engram gate timing。09-11/09-18 的 Engram
计量来自不同代码、eager/FULL、prompt和服务配置，不能拿其百分比给当前
39 ms/轮的生产估算收益。当前这组 TileKernels kernel 不覆盖 table lookup、
WKV projection、PP 或 DSpark draft 链，不能用于解释或消除全部固定截距。

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python \
  artifacts/tilekernels-engram-review-20261002/review_cpu.py
CUDA_VISIBLE_DEVICES='' timeout 90s .venv/bin/python \
  artifacts/tilekernels-engram-review-20261002/compile_sm80.py
```

交付为此报告、CPU contract JSON/log、TIR/CUDA/cubin、编译命令与 ptxas log，
以及原 Triton cache 的资源证据。未改推理代码、未更改生产开关、未加载模型，
TileKernels 工作区保持干净。研究报告尚未提交。
后续若复制其实现，保留 TileKernels MIT 的版权和许可声明。
