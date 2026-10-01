# DSV41F SM80 TileLang grouped GEMM 可选接入

日期：2026-10-01。承接 [PRMT 离线实现与复测](vllm170-dsv41f-sm80-tilelang-prmt-optimization-2026-09-30.md)。

## 接入方式与隔离范围

已在现有 `Mxfp4Int8Sm80Experts` 中增加默认关闭的候选分支。
启用时，在现有启动配置的 `additional_config` 中加入 JSON 布尔值：

```json
{"dsv41_sm80_moe_tilelang": true}
```

对应 CLI：`--additional-config '{"dsv41_sm80_moe_tilelang":true}'`。
若已有其他 additional-config 选项，应合并到原 JSON，不能用此示例覆盖原选项。
继续使用现有 `mxfp4_int8_sm80` MoE backend；此开关不改变后端选择优先级。

只有以下条件同时满足才编译、预热并进入候选：

- 配置模型身份是 `deepseek_v41`，HF 配置与专家实例均符合 Flash 的 H=5120、I=2304、E=384。
- 专家实例 top-k6、BF16、SILU，无 bias/LoRA；TP/EP/DP/PCP/SP 均为 1。
- 当前设备为精确 SM80（8.0）；SM86 等其他设备继续走原有实现。
- 显式开关为 `true`；目标模型使用字符串、整数或 null 会报配置错误，避免误启用。
- 每次调用必须是 M=1、局部/全局 E=384、top-k6，并使用已有 BM16 的持久路由 metadata。

相同尺寸的 `deepseek_v4`、Qwen 等非目标模型不能启用；它们也不导入新模块。
M>1、DSpark top-k3 及其他形状保持 CUTLASS。没有调整默认环境变量、shared expert、
router、attention、A8 量化、激活或 reduce。

## 生命周期

运行时内核位于
[vllm/model_executor/layers/fused_moe/experts/dsv41_sm80_tilelang.py](../../vllm/model_executor/layers/fused_moe/experts/dsv41_sm80_tilelang.py)，
不依赖 benchmarks、参考实现或 autotuner。参数冻结为 BM16/BN64/BK128、双缓冲，
W13 使用 128 threads，W2 使用 64 threads。

`process_weights_after_loading()` 在完成 packed 权重转换后检查形状、dtype、连续性和设备，
再编译和预热两个 kernel。编译对象按 worker/device 缓存，不持有模型权重副本。
预热只分配少量临时 activation/output；不展开全量权重，不增加持久模型 workspace。
forward 和 Graph capture 仅调用已准备好的 kernel，没有 JIT 搜索或首次编译。
显式启用后若编译失败，会在加载阶段报错；不会捕获并忽略可能污染 CUDA context 的运行错误。

已有持久 metadata 中的 padding sentinel 为 6。运行时显式屏蔽无效行的 A/scale 读取，
让对应 shared tile 零填充，输出仍只写有效 slot。初次移植的生成 CUDA
与离线候选逐字一致且已有 TileLang 自动边界保护；最终版本增加上述显式 guard，
不再仅依赖自动保护。曾试过把 padding 映射到合法第 0 行，但这样使原本跳过的 A 读取
实际执行，导致整段专家性能回退约 7%；该版本已经撤回，测量保留在
`apply-ab-row-clamp.json`。最终采用显式 masked load，并重新验证正确性和整段性能。

## 验证

设备为独立空闲 CMP170HX：`GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216`。
生产 PP5 的五个 GPU 进程保持运行，未重启、未启用线上服务开关。

测试通过现有 [test_mxfp4_int8_sm80.py](../../tests/kernels/quantization/test_mxfp4_int8_sm80.py)
扩展完成。实际实例化 `Mxfp4Int8Sm80Experts`，经权重加载入口启用，然后调用
`Experts.apply()`，覆盖 A8→W13→SILU→A8→W2→routing-weighted reduce：

- E384 的随机 packed/shift、非单位 channel scale、非均匀 routing weights。
- 最终 BF16 output 与全部 routed slots 对 CUTLASS 逐位一致。
- output 与 workspace13 按 modular runner 的规则别名，验证不会踩 routed W2 缓冲。
- Graph 捕获后两次原地更新 hidden states 与 expert IDs，重放结果逐位一致。
- M2、top-k3 确认不调用候选且仍逐位一致。
- 在首次权重准备后禁止编译函数再次被调用，验证 forward/capture 无编译。
- 默认关闭、同形状非目标模型、错误布尔值、非 SM80、并行配置和 LoRA/bias 隔离。

测试结果：

- 完整 SM80 相关测试：112 passed（显式行映射中间版本）。
- 最终 masked-load 版本：18 个接入相关用例通过，包含真实 full-path eager/Graph、
  动态路由、M2/top-k3 回退和身份门控；其余 94 个未受此次局部内核调整影响的用例已在完整运行通过。
- 通用 MXFP4 kernel selection：24 passed。
- 默认关闭/SM86 不导入候选：2 passed（也包含在上述 18 个接入用例中）。
- 最终相关 pre-commit（含 mypy）、ruff、`git diff --check` 通过。

最终专家全路径 A/B（显式 masked-load 版本）：

| 工作集 | CUTLASS µs | TileLang µs | 延迟降低 | 基线轮间 spread |
| --- | ---: | ---: | ---: | ---: |
| fixed | 200.40 | 184.27 | 8.05% | 0.01% |
| rotating | 200.29 | 184.58 | 7.85% | 0.00% |

两组均先验证全路径 eager 和全部 Graph 输出，再采样。该结果取代初次移植时的
约 9.6%–9.7% 数字；原始版本和被撤回的行映射版本数据均保留，未混用配置。

compute-sanitizer 实际尝试失败，设备报告：
`GPU debugging features are disabled`。因此不宣称 memcheck 通过；保留完整日志。
pytest 在该次运行通过不代表 sanitizer 工作正常。显式 padding guard、生成代码审查和
真实 eager/Graph 校验是当前证据。

## 整段专家 A/B 边界

使用真实 `Experts.apply()`，包含量化、激活、metadata 选择和 reduce；E384、M1、top-k6。
固定和 40 组轮换专家各做 3 轮，每轮 7 个 Event 样本，每个 Graph 40 次实际调用，
正式采样前交替预热 500 ms。双方校验 eager 及实际捕获 Graph 的全部输出后计时。
输入为合成权重与 activation，不是 checkpoint 模型级评测；当前未运行完整 PP5 模型
生成质量、吞吐或 shared/routed 并发 A/B。不能把专家段收益等同于 tokens/s 收益。

## 回退与证据

将 `dsv41_sm80_moe_tilelang` 设为 `false` 或移除该字段，并重新加载模型，即回到
CUTLASS；已捕获 Graph 不支持热切换。其他形状自动回退，无需新配置。

[结果目录](../../artifacts/tilelang-integration-20261001)保存最终源码、hash、工作区 patch、
完整测试日志、原始 A/B 样本和可复现的 `bench_apply.py`。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216 \
.venv/bin/python -m pytest tests/kernels/quantization/test_mxfp4_int8_sm80.py -q
```

运行前重新确认该 UUID 的设备仍空闲；这是测试设备，不是服务默认 GPU 配置。
