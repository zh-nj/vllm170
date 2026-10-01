# SM80 FP8 dense linear：TileKernels 可复用性研究

日期：2026-10-01。目标沿用当前 CMP170HX/SM80 优化上下文；研究涵盖普通 FP8 与 MXFP8，二者接口不能混用。

## 结论

TileLang 可以表达 FP8 存储、块内反量化、BF16/FP16 MMA 的 W8A16 linear。是否超过现有 Marlin 尚未证明。
本地 TileKernels 没有 dense/grouped GEMM 主循环，不能提供直接替换 FP8 linear 的现成算子。
它的主要价值是反量化、scale 布局、量化融合和数值参考；建议选择性移植，不把整个库引入当前服务环境。

## 核查范围与证据

- TileKernels：`/mnt/data/apps/vllma100/TileKernels`，HEAD `66258df6175d2f630ffecb04c5ab66bff8a2ae6a`，核查开始时工作树干净。
- vLLM：HEAD `32f3db5e7d`，已有 MoE TileLang 等未提交修改；本研究没有改变这些文件。
- 当前虚拟环境：Python 3.12.12、torch 2.13.0+cu130、TileLang 0.1.12；`requirements/cuda.txt:20` 固定 TileLang 版本。
- TileKernels README/pyproject 要求 Python ≥3.12、torch ≥2.13、TileLang ≥0.1.15；CUDA 支持范围列为 SM90/SM100、Toolkit ≥13.1。
- 在本地 `tile_kernels/` 搜索 `T.gemm/wgmma/mma/tcgen`、matmul、deep_gemm、einsum：没有 GEMM 主循环；命中的是 PyTorch 参考 einsum 和 DeepGEMM scale 格式转换说明。
- 官方说明交叉核对：[TileKernels](https://github.com/deepseek-ai/TileKernels)。本报告以本地指定 HEAD 为准。

直接导入检查使用 vLLM `.venv/bin/python`，`CUDA_VISIBLE_DEVICES=''`，通过 `sys.path` 指向本地源码，没有安装包：

```text
tilelang 0.1.12
tilelang.cuda exists; tilelang.cuda.language absent
ImportError: cannot import name 'language' from 'tilelang.cuda'
  tile_kernels/quant/norm_forward_cuda.py:4
```

这是当前实际入口的阻碍，不代表改一行 import 后所有算子就能使用；新版 reducer 等 DSL API 和 SM80 codegen 仍需逐项核实。

## 算子清单

以下路径相对于 TileKernels 根目录。

| 优先级 | 算子 / 文件 | 可借鉴内容 | 当前目标的限制 |
| --- | --- | --- | --- |
| P0 | `quant/cast_back_cuda.py:16`、`cast_back_kernel.py:9` | FP8/FP4 + scale → BF16/FP32；shared staging、scale 复用、分块加载 | 独立反量化会物化整张 BF16 权重；应提取到 GEMM 内部，或仅作参考基线 |
| P0 | `quant/common.py:19,94,188,285` | 显式量化配置、scale stride/布局、UE8M0 解码、索引映射 | 公共 API 的 CUDA packed scale 为 int32，内部再 view uint8；与 vLLM MXFP8 uint8 输入不能直接互换 |
| P0 | `torch/cast.py:46`、`tests/quant/test_cast_back.py:112`、`testing/numeric.py:5` | 独立反量化参考；dtype/shape/stride/原始字节一致性检查 | 比较同一量化权重，不以原始未量化权重混淆 kernel 误差与量化误差 |
| P1 | `quant/per_token_cast_cuda.py`、`per_block_cast_cuda.py`、`per_channel_cast_cuda.py` | 分组 amax、幂二 scale、布局处理、小/大 batch 分派 | W8A16 不需要再次把 activation 量化为 FP8；只用于确有量化契约的相邻路径 |
| P1 | `quant/norm_forward_cuda.py:11` | residual + RMSNorm + per-32-group quant，可同时产出 BF16 | 新 API；hidden ≤8192；量化模式只接受 `(1,32)`，不能直接替代任意 block quant |
| P1 | `quant/swiglu_forward_cuda.py` | SwiGLU、clamp、路由权重、量化融合；寄存器内复用 | 必须匹配 gate/up 排布、clamp、路由权重应用次序；当前 INT8 MoE 不能直接消费其 FP8 输出 |
| P2 | `quant/batched_transpose_weight_sf_kernel.py:11` | 成批整理 scale；支持 caller 提供输出缓冲 | 处理的是 FP32 幂二 scale，输出 DeepGEMM UE8M0 布局，不是 weight transpose 或 GEMM；SM80 首版不需要 |
| P2 | `quant/per_block_cast_lossless_kernel.py:8` | FP4 E2M1 → FP8 E4M3 的重分组和边界断言 | 不是 FP8 → INT8；要求尺寸对齐，并在 kernel 检查指数范围；不能视为普遍无损转换 |
| 旁支 | `engram/engram_gate_fwd_cuda.py:13` | RMSNorm/dot/gate 融合、双缓冲、按 shared memory 估计常驻 CTA | 不是线性 GEMM；源码只注明 4096/7168 调优，5120 的收益未验证 |
| 旁支 | `mhc/pre_big_fuse_cuda.py:12`、`mhc/post_cuda.py` | mHC 归约与后处理融合 | `gemm_out_mul` 是输入，不包含前置 GEMM；vLLM 已有同类实现，应比较差异而非整体替换 |
| 旁支 | `moe/moe_topk_gate_forward_cuda.py:22`、`moe/topk_gate_cuda.py` | warp top-k、归一化、bias/mask/专家映射 | 是路由，不是专家矩阵乘；需保留 DSV41 路由语义 |
| 旁支 | `transform/rope_cuda.py` | RoPE 向量化与布局 | 不解决 FP8 dense linear 主循环，优先级低 |

“可借鉴”表示源码层面的移植候选；本轮未对 TileKernels 算子作 SM80 编译或 GPU 性能验收。

## 三个不能混淆的数据契约

1. 普通 FP8：E4M3 权重 + tensor/channel/2D-block scale；需要按实际量化键解释。
2. vLLM MXFP8：`weight[N,K]` E4M3 + `weight_scale[N,K/32]` uint8 E8M0；见 `vllm/model_executor/kernels/linear/mxfp8/Mxfp8LinearKernel.py:13`。
3. TileKernels CUDA packed UE8M0：公共张量以 int32 打包四个 scale byte；还可能是列主序、带 padding 的布局，内部通过 uint8 视图访问。

MXFP8 的候选计算应明确为：

```text
X[M,K] BF16 + W[N,K] FP8 + S[N,K/32] uint8
  -> 每个 K tile 内解码 W 与 group scale
  -> BF16 MMA / FP32 accumulation
  -> BF16 output[M,N]
```

scale 随 K 变化时不能全部移到最终 epilogue。跨多个 32 元素组的 tile 必须逐组施加 scale。
FP8 的 E4M3 编码也不能像先前 MXFP4 PRMT INT8 路线那样直接当作有符号整数；改为 INT8 将引入新的转换/精度契约。

### UE8M0 边界差异

TileKernels `common.py:304` 用 `reinterpret(uint32(sf) << 23, float32)`。
vLLM `mxfp8_utils.py:230` 的参考用 `exp2(sf - 127)`。
本轮 CPU 对照结果：scale byte 1、126、127、254 相同，但 byte 0 分别得到 **0** 和 **2^-127**。
因此不能把这个 bit trick 无条件替代 vLLM 参考；需要确定生产数据合法域，并检查 0/255、零值、极值和 subnormal。
这不是对 TileKernels 自身契约正确性的否定，而是两个项目契约有差异。

## SM80 与初始探针的证据边界

当前 TileLang 的 FP8 MMA dispatcher 使用 `cute::SM89_*`，BF16/INT8 dispatcher 有 SM80 路径；参见安装包 `tilelang/src/tl_templates/cuda/instruction/mma.h:197,237`。
直接 FP8×FP8、FP32 累加的临时探针能生成代码，但执行触发 SM89 专用 device assert；不能以“编译通过”视为 SM80 原生 FP8 支持。
FP8 输入直接搭配 BF16 accumulator 的尝试则是 unsupported configuration；这不等于 FP32 累加后 cast 为 BF16 output 不受支持。

前序临时反量化 + FP16/BF16 GEMM 探针能够执行，部分常量 scale 的随机输入通过 assert_close。
它们不是 MXFP8 group-32 全契约验证，也没有真实权重、非平凡 scale 与生产形状矩阵覆盖。
临时 `(N=4096,K=5120)` 测试未完成 PCI/UUID/设备占用核实、Graph 测量与缓存工作集控制，不能称为已验证生产形状或用于断言“TileLang 比 Marlin 慢若干倍”。
本报告不采纳其性能数字，也不将其作为停止 TileLang 研究的依据。

## 建议的下一步

1. 固定一个真实目标层的量化方法、raw 权重/scale、N/K、输入 dtype、Marlin 运行配置；优先从已有 `wq_b` / `wo_b` / `fused_wqa_wkv` 微基准中选一个，而非泛化替换全部 FP8。
2. 在现有 TileLang 0.1.12 中写独立 W8A16 候选，参考 `cast_back` 和 `common` 的数据访问；保留 vLLM uint8 scale 契约，关闭 PDL/TMA 专用路径。不升级生产环境来迁就整个包。
3. 解码 tile 在 shared/register 内消费，避免常驻 BF16 权重副本；FP32 累加后写 BF16。SM80 的 GEMM 主循环、权重布局和调度仍需自己实现/调优，TileKernels 没有替我们解决。
4. 正确性覆盖非单位 scale、E4M3 符号/零/subnormal、scale 边界、M=1/8/128/1024 及不整齐 M；若要求 batch invariance，验证相同 token 跨 M 的结果。先固定 K reduction，不能默认接受 split-K 重排。
5. 后续硬件实验先确认 GPU UUID、PCI 顺序与空闲状态；同一设备用真实 N/K、已解码参考与现有 Marlin 交替测量。分别记录 Graph 与 eager、热缓存与轮换权重工作集；包括解码/scale/输出成本，保存原始样本。
6. 只有完整 linear 延迟收益超过噪声且数值满足契约，再考虑默认关闭的模型/架构限定接入。对于 MXFP8，正确入口是 `init_mxfp8_linear_kernel` / `Mxfp8LinearKernel`，不能只在通用 FP8 selector 注册。

本轮产出为源码审计、无 GPU 的导入检查和 scale 边界 CPU 对照；没有安装依赖、改变服务或运行 TileKernels GPU 测试。
如后续复制实现，保留 TileKernels MIT 版权与许可声明。
