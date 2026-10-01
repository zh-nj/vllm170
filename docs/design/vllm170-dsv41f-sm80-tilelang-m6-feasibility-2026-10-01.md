# `use_persistent_decode_metadata` 能否扩展到 m=6？

日期：2026-10-01。承接
[端到端 A/B](vllm170-dsv41f-sm80-tilelang-e2e-ab-2026-10-01.md)。
测量设备：空闲 CMP170HX `GPU-f385e0d6-452b-ad54-922e-ef7f4deb6216`。
原始数据：`artifacts/dsv41f-tilelang-e2e-20261001/m6-{hot,uniform-sweep}.json`。

## 结论

**不要按现在的方式扩展。** 两部分各自失败，而且失败原因不同：

1. **元数据技巧**：机械上可以扩展、结果也正确，但它在 m=6 不再是免费的——
   36 个 slot 各占一个 16 行 tile，一旦多个 slot 命中同一专家（spec 验证的
   连续 token 恰恰如此），tile 数就按 `slots / 不同专家数` 膨胀，实测 **3.5–3.7× 变慢**。
2. **TileLang kernel 本身在 m=6 就不快**：即使用**真 align 元数据**，7 个配置
   全部比 CUTLASS 慢 **1.56–1.92×**（m=1 时它是快 1.08×）。
   所以今天扩门只会让 decode 更慢。

顺带得到一个有用的事实：**align kernel 很便宜**（m=6 只有 5.6–7.3 µs，
占该层 GEMM 的 1.1–2.1%），它不是瓶颈；m=1 快路径省下的是这笔小钱，
不是大头。

## m=1 的技巧为什么在 m=6 退化

`make_mxfp4_int8_sm80_decode_metadata` 造的是常量
`sorted_ids = [0, 16×sentinel(6), 1, 16×6, …]`（6 个 tile，每个 tile 第 0 行有效），
`expert_ids = routing_ids.view(-1)`，`num_post_pad = 96`。

它在 m=1 精确成立，因为**一个 token 的 top-6 专家互不重复**：6 个 slot = 6 个专家
= 6 个 tile，恰好就是 align 的最小值，连 sentinel 都是同一个数。

同样的构造在 m=6 就是 36 个 tile、每 tile 一个 slot。正确性没问题（每个 tile 只含
一个专家的行），但 tile 数不再是"不同专家数"，而 align 的真正价值正是把同专家的
slot 合并进同一个 tile——**tile 数同时决定权重流量和解码量**（每个 tile 都要把
`block_n × K` 的 packed 权重解一遍）。

## 实测（m=6, topk6, E=384, block_m=16, 36 slots）

| 路由 | 活跃专家 | align tile | 持久化 tile | stage | CUTLASS 真 align | CUTLASS 持久化 | 惩罚 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| hot（6 token 命中同一批专家） | 8 | 8 | 36 | w13 | 190.4 µs | 666.4 µs | **3.50×** |
| hot | 8 | 8 | 36 | w2 | 92.6 µs | 342.8 µs | **3.70×** |
| uniform | 36 | 36 | 36 | w13 | 517.1 µs | 36 tile（相同） | 1.00× |
| uniform | 36 | 36 | 36 | w2 | 265.7 µs | 36 tile（相同） | 1.00× |

- 四种组合（两个 stage × 两种 layout）**全部与独立参考逐位一致**，
  即上面的差异纯粹是性能悬崖，不是正确性问题。
- uniform 下持久化 layout 只是省掉 align（5.6 µs，占 1.1–2.1%）；
  但 decode 的真实流量是"同一序列连续 6 个 token"，专家重合度高，落在 hot 一侧。
- align 实测：m=6 时 5.6–7.3 µs；m=1 时同量级（`align_over_cutlass` 2–6%）。

## TileLang kernel 在 m=6 的表现（真 align 元数据）

同一个 grouped GEMM kernel，16×{64,128}、BK128、stages 1–2、threads 64/128/256、
`luta`（PRMT 风格）解码，与 CUTLASS 同元数据对比：

| stage | CUTLASS | TileLang 最好 | 比值 |
| --- | ---: | ---: | ---: |
| w13 | 517.1 µs | 815.8 µs（16x128x128-s2-t128） | 1.58× |
| w2 | 265.7 µs | 415.8 µs（16x128x128-s2-t128） | 1.56× |

七个配置全部 `exact=True`（与独立参考逐位一致），但没有一个接近 CUTLASS——
与 m=1 时"TileLang 快 8%"的结论相反。m=6 时每个 tile 仍然只有 1 行有效，
但 tile 数是 m=1 的 6 倍，解码量同样放大 6 倍，TileLang 逐元素/PRMT 解码
相对 CUTLASS 的劣势被同比放大。

## 建议

1. **保持 m=1 门不动**：它在自己的形状上正确、便宜、无副作用。
2. **不要为了 m=6 复用它现成的 kernel**：那会慢 1.6×。
3. 若确实想吃下 decode 验证形状（`m = 1 + num_spec = 6`）：
   - **保留真 align**（5.6 µs，1–2%），不要用单 slot 布局；
   - 先解决 kernel 在 m=6 的效率问题（tile 内只有 1 行有效 → 应先把
     同专家的 slot 合并，或改用 m 维度真正有效的 tile 形状），
     验收放在 **stage 段内**（CUTLASS 517+266 µs/层这个量级）；
   - 端到端上限仍是 ~0.9%：m=6 时路由 MoE 约 (517+266) µs/层 × 8 层
     ≈ 6.3 ms/步，占 74 ms 步延迟的 ~8.5%，段内 10% ≈ 端到端 0.9%，
     低于实测 13% 的重启噪声。
