# DSV41F SM80 grouped GEMM：TileLang 重写与离线 autotune

日期：2026-09-30。本文是给 DeepSeek 的下一阶段设计指引，承接
[验收设施任务书](vllm170-sm80-moe-harness-deepseek-execution-guide-2026-09-30.md)。
本轮只核查环境元数据、源码和官方资料；未编译或运行候选 GPU kernel，未证明提速。

## 可行性判断

可以使用 TileLang 重写，并使用其 AutoTuner 离线搜索实现参数。
它适合降低 grouped 索引、解码、共享内存流水和 INT8 MMA 候选的试验成本。
autotune 能寻找给定算法的参数组合，不能自动消除错误的数据布局或保证优于当前 CUTLASS。

本机 `.venv` 安装版本为 0.1.12，`requirements/cuda.txt` 也固定为 0.1.12。
已读安装包源码确认：

- `tilelang/cuda/op/gemm/gemm_mma.py` 有 MMA GEMM lowering。
- `cuda/intrinsics/macro/mma_macro_generator.py` 有 INT8/INT32 类型和 m16n8k32 路径。
- `autotuner/tuner.py::AutoTuner.from_kernel/set_profile_args` 支持自定义 supply、
  reference/manual check，以及 `event/cupti/cudagraph` 后端。

这是“接口与代码路径存在”的证据。第一道实验仍应验证当前版本在实际 CMP170HX 上
能够编译正确的 INT8→INT32 GEMM，检查生成 CUDA/PTX 的 MMA 指令和资源用量。
不要因为官网有 SM90/SM100 示例就套用 TMA、WGMMA、TMEM 或原生 FP4 路径。

官方交叉参考：

- [v0.1.12 GEMM 测试，包含 INT8 输入与 INT32 累加](https://github.com/tile-ai/tilelang/blob/v0.1.12/testing/python/kernel/test_tilelang_kernel_gemm.py)。
- [AutoTuner API](https://www.tilelang.com/autoapi/tilelang/autotuner/tuner/index.html)。
- [GEMM autotune 示例](https://github.com/tile-ai/tilelang/blob/main/examples/gemm/example_gemm_autotune.py)。

在线文档/main 可能对应较新版本，实际参数与回调签名以本机固定版本为准。

## 范围与第一版算法

**仅 dsv41f 的 SM80 routed MoE grouped GEMM。** 其他模型、量化后端、GPU 架构、
router 选专家算法、shared expert、attention、DSpark 采样不改。
top-k3 仅覆盖 dsv41f 配套 draft 的专家调用，top-k6 覆盖 target。

第一版保留现有权重格式、A8 量化、channel scale、BF16 舍入、slot 顺序和 reduce：

```text
production packed MXFP4-remapped bytes + shift2
    → kernel 内解码成 INT8 B tile
routed A8 gather → A tile
    → INT8 × INT8 → INT32 accumulate
    → 原顺序 FP32 token/channel scale
    → BF16 routed slot output
```

这里计算的是已近似重映射后的权重，不是改回 checkpoint 的精确 MXFP4 语义。
先分别实现 W13 与 W2，保持 reduce 不变；复用原有 routing metadata。
最小编译 spike 可以使用提前解码的 INT8 B 验证 MMA，但正式性能候选必须计入解码，
不能把展开全量 INT8/BF16 权重后的访存和显存成本藏在计时区外。

第一版不要引入 split-K、浮点 atomic reduce、跨 CTA 通信、W13+激活融合或全量权重
重排。它们增加数值、workspace 和调度变量，应在基础候选可测后分别立项。

## 按执行阶段拆分 kernel 族

| 负载 | 优先尝试 | 后续可能的不同算法 |
| --- | --- | --- |
| M=1 / 极稀疏 decode | 小 M tile 的 INT8 tensor-core grouped GEMM | 若 padding/调度证据支持，再试 GEMV/SIMT 整数点积候选 |
| 小批 decode / draft verification | 独立的 grouped GEMM 配置 | 多专家 CTA 分配、适度流水 |
| prefill / 较大每专家 m_e | 较大 M tile、权重复用与流水 | 后续再评估更复杂调度 |

W13 和 W2 的 N/K、A 行映射不同，必须分别调优。
不能用大矩阵 dense GEMM 的默认 128x128 配置代表单 token 的稀疏专家调用。
也不能因为 M=1 就断言 GEMV 会更快；应让同一验收设施比较两种算法。

## Autotune 使用方式

采用两级流程：AutoTuner 做受限候选筛选，已有 Event+Graph harness 做严格复测和晋升。
优先使用程序化 `AutoTuner.from_kernel()`，便于显式控制输入、参数、失败记录和版本。

### 搜索维度

以下只是待编译验证的候选集合，不是已证明合法或最优的配置：

| 参数 | 起始候选 |
| --- | --- |
| block_M | decode 从 16 开始；其他阶段再加入 32/64/128 |
| block_N | 64、128 |
| block_K | 64、128 |
| num_stages | 2、3；0 用作正确性/流水消融 |
| threads | 128、256，按 warp layout 与 tile 约束筛选 |
| swizzle / warp policy | 第一轮固定；有合格基线后逐项增加 |

先固定 block_M、线程数和流水，只扫描 BN/BK 的四个组合；通过后扩到每类最多
16–32 个有理由的组合。不是把表中所有维度一次全乘积。
共享内存估算需包含 packed B、decoded B、A、shift/scale 暂存和流水双/多缓冲，
不能照搬纯 dense GEMM 公式。最终以编译资源信息为准，淘汰超资源配置并保留原因。

### 正确处理输入与路由元数据

通过 `supply_prog` 提供真实构造的 A、packed B、scale、IDs、padding metadata 和
输出，不能让通用随机 tensor supplier 随机生成指针索引或专家编号。
所有候选都要使用同一 logical routing case。

**改变 block_M 必须重新执行 align。** tuner 的 tensor cache 即便 shape 相同也
不理解“这个 metadata 属于另一个 block_M”。最稳妥的初版：

1. 外层按 block_M 建独立 tuner，生成对应 metadata。
2. 每个 tuner 内仅调 BN/BK/stages/threads 等不改变 metadata 的参数。
3. 缓存只在输入 shape 与语义都不变的组内启用；所有输入保持有效生命周期。

原始 routing IDs 始终用于独立参考，不能把候选自己的 aligned metadata 当成真值。
候选输出采用调用者预分配张量的生产契约；校验所有输出和必要的 sentinel。
不要因为 in-place 输出不在默认 `out_idx` 返回列表就漏检。

### 校验与计时

- 设置 `backend="cudagraph"`，不依赖此机器历史上不可用的 CUPTI。
- 默认 rtol/atol=1e-2、`max_mismatched_ratio=0.01` 不适用本项目；
  设置 `max_mismatched_ratio=0`，误差由前置验收设施的分层契约决定。
- 本机 tuner 只有在 `skip_check=False` 且 `ref_prog` 非空时才进入 reference/manual
  check 分支。因此仅设置 `manual_check_prog` 或保留默认 skip_check=False 不能保证校验。
- 本机 tuner 还会计时 `ref_prog`。不要直接传入包含 CPU 循环、CPU/GPU 传输的慢速
  独立解包 oracle，然后要求它能 CUDA Graph capture。
  可事先用独立 oracle 计算期望输出，接入已验证兼容的回调；若该 API 难以表达
  in-place 契约，必须在外层逐配置严格验证后再进入计时，记录每个配置的通过证据，
  不得用无校验的 autotune 排名直接决定上线。
- 每配置记录编译失败、正确性失败和计时失败；IMA/device assert 后停止对应 CUDA
  进程，不能让 autotuner 在污染 context 中继续挑“最快”。
- AutoTuner 初选可能偏向固定输入热缓存。将排名前 2–3 个候选放回同一 harness，
  交替复测固定专家和轮换专家两种模式，并用未参与调优的 route seeds/边界验证。
- 先测生产 baseline 的自比噪声，只有收益超过噪声且覆盖关键用例时再考虑晋升。

## 编译与生产接入

autotune 全部离线运行，不放在请求路径、模型 forward 或 CUDA Graph capture 内。
生产加载冻结的版本化配置表；先编译所需有限 shape/bucket 并 warmup，再 capture。
记录模型、backend、SM、GPU 资源、stage、N/K、M bucket、E/topk、量化/布局版本、
TileLang/CUDA/源码版本。原始路由直方图保留在调优 artifacts。

不要为查表在每个 decode 中 `.item()` 或将 routing IDs 复制到 CPU 计算分布。
运行时只用已有 host metadata/静态维度选表；离线要验证表对多个路由分布都稳健。
同一 Graph replay 中不切换已捕获的 kernel/config/workspace；新形状按既有规则选择
不同 graph 或 fallback。初版未覆盖的 shape 继续走现有 CUTLASS。

仅在 dsv41f 模型身份、目标 SM80 与指定 MoE 后端都匹配、显式启用候选时调用新入口。
非目标路径不 import/编译/tune 新 kernel，不改变通用后端优先级，不修改现有模型的
环境默认值。不要以“同 H/I/topk”代替模型门控。
准备阶段不修改任何生产路径；接入时另行审查调用者和非目标模型的分派回归。

## 阶段出口

1. **最小编译 spike**：真实设备验证 INT8 MMA、布局、INT32 累加与 BF16 epilogue；
   保存生成代码。失败则先定位该局部，不立即升级整个服务的 TileLang/TVM。
2. **单配置 grouped kernel**：W13/W2 各一个正确版本，保留原 packed 格式和 slot 写回。
3. **离线 autotune**：分组搜索、严格检查、保存配置/原始样本/失败原因。
4. **独立复测**：固定/轮换工作集、不同路由分布、tile 边界和 held-out seeds。
5. **接入评审**：只有前四步有净收益，才进行默认关闭的 dsv41f 专用接入；再做完整
   routed MoE 与无插桩模型 A/B，验证 shared/routed 并发和显存没有抵消收益。

最快配置没有显著超过当前 CUTLASS，或只在单一热缓存用例胜出时，应保留实验证据，
结束该候选。不能因 TileLang 实现更短或 autotune 给出了 best_config 就替换生产实现。
