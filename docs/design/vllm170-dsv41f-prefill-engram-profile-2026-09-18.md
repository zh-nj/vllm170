# DSV4.1-Flash @ 5×CMP170HX：FlashMLA 结论 + prefill/engram 环节计量（2026-09-18）

目标：① 集成 FlashMLA 注意力后端；② 保持 2048 chunk，梳理 prefill/engram 各环节耗时，
以 prefill 10000 tok/s 为目标优化。

基线（本轮实测，63,053 token 单请求，`max_num_batched_tokens=2048`，31 个 chunk）：
**11.7–12.0 s ≈ 5,300 tok/s**（profile 开启时 13.1–13.6 s，插桩本身约 +12%）。

---

## 1. FlashMLA：SM80 上无法集成（有证据）

| 检查 | 结果 |
| --- | --- |
| `csrc/kernels/sm80/` | **只有 README**，没有任何 kernel 源码；`flash_mla/cuda.*.so` 是 sm_80-only 构建（`cuobjdump` 只列 `sm_80.cubin`） |
| `import flash_mla` / `get_mla_metadata` | ✓ 可用（arch-generic helper 能在 Ampere 上跑） |
| `flash_mla_sparse_fwd`（V4.1 稀疏 prefill 内核） | ✗ `RuntimeError: [FlashMLA] sparse_prefill_fwd is not implemented for sm_80 in this build` |
| 上游 V4.1 kernel | SM90/SM100：wgmma、TMA、mbarrier、thread-block cluster，sm_80 上连编译都过不去 |

仓库自带的 `docs/20260915-sm80-dsv41-attention-analysis.md`（同一台机器上测的）结论一致：
**FlashMLA 在 sm_80 上"没有可移植的东西"**，并给了替代方案——Triton 的 read-once
prefill tiling（`bh=64/bk=64/st=2/nw=8`，1.51×）。

**该替代方案已经在本仓库生效**：`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`
里有完整的 sm_80 门控（`head_dim==512` + bf16 KV + `shared_memory_per_block_optin >= 139264`
+ 未导出 `VLLM_DEEPSEEK_V4_SPARSE_PREFILL_BLOCK_K`），`run_port_pp5.sh` 没有导出该变量
→ PP5 启动就是 read-once tiling。本轮未改这块（已是最优可得配置）。

**结论**：第一步能做的不是"接 FlashMLA kernel"（不存在），而是（a）确认等价优化在位 ✓、
（b）把注意力/engram 的耗时量出来，喂给第二步。

---

## 2. 计量工具（本轮新增）

`vllm/models/deepseek_v4/phase_profile.py`（CUDA event 级 phase profiler）本来就被
`model_runner.py` 导入并注册，所以只要开环境变量就能测到 MoE/attention/indexer 的既有 stage；
本轮补上了缺失的 engram 与层归属：

| 新增 stage | 位置 |
| --- | --- |
| `engram.hash` | `deepseek_v4_1/nvidia/model.py`（n-gram hash + 滚动槽查表） |
| `engram.prepare` / `engram.lookup` | `common/engram.py`（**宿主 94 GiB 表的 PCIe gather**） |
| `engram.embed` / `engram.post_wkv` | `common/engram.py`（wkv 投影 + 融合注入 kernel） |
| `layer.total` + `phase_profile_layer(prefix)` | `nvidia/model.py`（每层归属，`layer` 字段带层名） |

用法（control 文件做开关，避免全程插桩）：

```bash
VLLM_DEEPSEEK_V4_PHASE_PROFILE=1 \
VLLM_DEEPSEEK_V4_PHASE_PROFILE_PATH='/tmp/dsv41_prof/rank{pp_rank}.jsonl' \
VLLM_DEEPSEEK_V4_PHASE_PROFILE_CONTROL_PATH=/tmp/dsv41_prof/ON \
VLLM_DEEPSEEK_V4_PHASE_PROFILE_FLOWS=prefill,dspark \
PORT_MAX_BATCHED=2048 bash benchmarks/deepseek_v4_1/run_port_pp5.sh
# 测完 rm /tmp/dsv41_prof/ON
```

---

## 3. prefill 环节账本（63k 请求 / 31 chunk，每 rank 每 step 平均 ms）

| 环节 | ms/step | 占 step（377 ms） | 说明 |
| --- | ---: | ---: | --- |
| **MoE 路由专家** | **158** | **42%** | w13_gemm 102 + w2_gemm 53 + 其余 3 |
| 其余 layer 内（attention kernel、hc/norm、PP-KV 导出等） | ~83 | 22% | 未单独插桩，由 `layer.total − 已知子项` 得出 |
| 步间空隙（非 GPU） | 63 | 17% | 引擎/PP 交接/采样，见 §4 |
| indexer.logits | 18 | 5% | 只在 ratio=1 的层（rank3/4 高） |
| 共享专家 | 11.5 | 3% | 2048 token 下未与路由专家重叠 |
| target.attn_metadata | 10 | 3% | metadata 构建 |
| attention wo_a / inv_rope / ragged | 9 | 2% | |
| engram（rank0 有层 1 / rank1 有层 14） | 18 / 2.6 | — | 整个请求 572 ms / 82 ms |

engram 明细（整个 63k 请求，rank0 / rank1）：
`prepare+lookup` 567/77 ms，`embed` 余下；即**宿主表 gather 就是 engram 的全部成本**，
每 step 约 18 ms（rank0）——占该 rank step 的 5% 左右，不是瓶颈。

dspark 流在 prefill 期间可忽略：整请求合计 `target.sampling_total` 63 ms + `target.logits` 41 ms
+ `target.rejection` 15 ms。

---

## 4. 那 63 ms/step 的空隙是什么（已排除的假设）

- **不是 PP 收发**：`target.pp_recv_wait` + `pp_recv_copy` 合计 4 ms/请求。
- **不是 DSpark draft**：prefill 期间 dspark 流总量 ~0.13 s/请求。
- **是引擎/宿主每步开销**：31 step × 63 ms ≈ 1.95 s ≈ 总墙钟的 15–17%。
  step 周期（由 profiler 时间戳差算出）345 ms，其中 GPU forward 283 ms、metadata 10 ms。
- 由此得到一条硬结论：**在 2048 token/step 下，prefill 上限 = 2048 / (GPU 每步 + 固定每步开销)；
  减少每步固定开销（更少步数或更低宿主开销）与降低 GPU 每步耗时同等重要。**

---

## 5. MoE 为什么是瓶颈，以及还剩多少空间

MoE 每 step 每 rank 要读 **8 层 × 6.79 GB = 54 GB** 的 mxfp4 专家权重（2048 token × topk6 =
12,288 个专家命中摊到 384 个专家上，每个约 32 token → **每个专家权重每步都要读一遍**）。
所以 MoE 时间 ≈ 权重字节 / 有效带宽，与 token 数几乎无关：**这是"每 chunk 固定成本"**。

实测（`benchmarks/kernels/deepseek_v41_mxfp4_int8_sm80.py --tokens 2048`，单层）：

| block_m × block_n | w13 (ms) | w2 (ms) | 合计 | 有效带宽 |
| --- | ---: | ---: | ---: | ---: |
| **64 × 128（当前生产 tile）** | **11.67** | **6.19** | **17.85** | **380 GB/s** |
| 16 × 128 | 13.58 | 7.07 | 20.64 | 329 GB/s |
| 16 × 64 | 14.24 | 7.69 | 21.93 | 310 GB/s |
| 128 × 128 | 15.79 | 8.50 | 24.29 | 280 GB/s |

结论：**tile 已经是四个候选里最好的**；瓶颈是内核本身的带宽效率（380 / 1240 GB/s ≈ 31%）。
内核结构：`threadblock 256×128×64`、`warp 128×32×64`、`kStages = 3`、cp.async +
运行时 mxfp4 解包（含 `b_shift2` 修正）。要提速只能动内核（更多 stage / 更大 K tile /
更宽 load），这需要重编 `_moe_C`（`cmake-build-release/`，nvcc 13.3 在位）。

### 到 10000 tok/s 的预算

10000 tok/s @ 2048 chunk ⇒ **每 step ≤ 205 ms**（现在 ~377 ms）。

| 项 | 现在 | 需要 | 手段 |
| --- | ---: | ---: | --- |
| MoE | 158 | ~70–80 | 内核带宽 380 → 700+ GB/s（kStages/K-tile/load 宽度） |
| 其余 layer 内 | 83 | ~45 | attention kernel + hc/norm 插桩后再定位 |
| 步间空隙 | 63 | ~20 | 引擎/宿主每步开销 |
| metadata + 其他 | 21 | ~15 | |

即：**这是一个"MoE 内核带宽 ×2 + 每步固定开销减半"的组合目标**，单靠调参到不了。

---

## 6. 本轮已试且无效 / 待试

- ✗ `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=4096` +
  `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=4096`（想让 11.5 ms 共享专家与路由专家重叠）：
  12.0 s vs 基线 11.7 s，**无收益**（多流路径大概率未对当前 MoE 后端生效，或直接争抢带宽）。
- ✗ MoE tile 换档：当前档已最优（§5）。
- ✗ FlashMLA sm_80 kernel：不存在（§1）。
- 待做（下一轮，按收益排序）：
  1. MoE 内核带宽：`kStages` / K-tile / load 宽度扫描（改 `mxfp4_moe_cutlass_core.cuh` + 重编 `_moe_C`，
     用 §5 的 benchmark 快速迭代，不需要起服务）。
  2. attention/hc/norm 的剩余 ~83 ms/step：给 `sparse_mla.attention` 加 stage 后定位。
  3. 每步 63 ms 宿主开销：查 engine→5 worker 的每步 RPC/序列化路径。
  4. engram（rank0 18 ms/step）：宿主表 gather 用后台流与主计算重叠（`lookup(background=True)` 已存在但未启用）。
