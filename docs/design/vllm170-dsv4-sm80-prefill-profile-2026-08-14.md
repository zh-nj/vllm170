# DeepSeek V4 SM80 prefill phase analysis (2026-08-14)

## Measurement boundary

The accepted run is
`profiles/deepseek_v4/20260814T025725Z`: three CMP170HX GPUs, PP3/TP1 with
`15,15,13` target layers, Marlin MXFP4 MoE, Triton sparse MLA, FP8 KV cache,
and DSpark width 7. The server tokenizer produced exactly 5,000 prompt tokens;
the scheduler split them into 2,000, 2,000, and 1,000-token chunks.

Detailed mode uses eager execution and nested CUDA Events. It measures GPU
scope latency and deliberately synchronizes at each step. The numbers below
rank optimization targets; they are not production CUDA Graph throughput.
CMP170HX does not expose CUPTI CUDA activities on the current driver, so this
report does not claim per-kernel NCU metrics.

All percentages use the same PP0 `target.model_forward` parent. PP ranks and
nested levels are never added together.

## Critical-rank decomposition

| Sibling module | PP0 total | Parent share | Interpretation |
| --- | ---: | ---: | --- |
| FFN/MoE | 359.637 ms | 48.8% | Entire runtime is in the internal Marlin route + W13/SwiGLU/W2/reduce path. |
| Attention | 301.782 ms | 41.0% | C4 layers dominate the excess over C128 layers. |
| mHC attention transition | 35.901 ms | 4.9% | Per-layer fused post/pre or initial broadcast path. |
| mHC FFN transition | 36.999 ms | 5.0% | Per-layer fused attention-to-FFN transition. |
| Unattributed layer time | 0.855 ms | 0.1% | Event boundaries cover 99.9% of layer time. |

The three PP target totals were 736.266 ms, 703.825 ms, and 653.442 ms. PP0 is
the critical rank, but moving a target layer to PP2 is not currently feasible;
the residency A/B below explains why.

## Attention decomposition

| Attention child | PP0 total | Model share | Optimization meaning |
| --- | ---: | ---: | --- |
| Q/indexer/compressor/sparse-MLA region | 194.448 ms | 26.4% | Main attention target; includes overlapping child ranges. |
| Sparse MLA wrapper | 110.863 ms | 15.1% | The underlying sparse attention kernel alone is 104.439 ms. |
| Output projection | 88.196 ms | 12.0% | Large enough to justify an isolated grouped projection study. |
| Input GEMM overlap | 17.061 ms | 2.3% | Already overlapped; not a first-order target. |
| Q/K RMSNorm | 1.129 ms | 0.2% | No optimization priority. |

Within the indexer, query preparation costs 22.262 ms (3.0% of the model),
compress/cache 15.862 ms on PP0 (2.2%), and the complete select parent 7.310
ms (1.0%). Logits is 5.567 ms and top-k only 0.748 ms. Because select contains
logits and top-k, those values are descriptive children and must not be added.

Sparse-prefill data movement is small at this length: compressed gather is
1.428 ms, SWA gather 3.569 ms, and index combine 0.816 ms. Byte-budget gather
and future no-gather work remain capacity facilities, not the 5K speed target.

## Chunk scaling

| Chunk | Tokens | PP0 | PP1 | PP2 | Critical rank |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2,000 | 275.351 ms | 265.740 ms | 277.345 ms | PP2 |
| 1 | 2,000 | 292.051 ms | 278.572 ms | 237.835 ms | PP0 |
| 2 | 1,000 | 168.865 ms | 159.513 ms | 138.262 ms | PP0 |

The second 2K chunk is slower than the first because the accumulated context
crosses from short-context cache handling into full C4 indexer selection and a
larger sparse-attention history. The 1K tail still costs 57.8% of the slower
2K PP0 chunk, showing fixed launch/mHC overhead and imperfect small-row scaling.

## Layer-family evidence

The following values are work averages per layer and chunk, not PP wall time:

| Compression ratio | Layers | Layer total | Attention | FFN |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 13.779 ms | 4.937 ms | 7.388 ms |
| 4 | 21 | 17.505 ms | 8.199 ms | 7.685 ms |
| 128 | 20 | 15.074 ms | 5.513 ms | 7.944 ms |

C4 attention is 48.7% slower per layer/chunk than C128 attention, while FFN is
similar. The hottest individual layers are therefore C4 layers; layer 30 is
the maximum at 19.450 ms per chunk, followed by layers 34 and 32. The full
43-layer component table is in `phase_prefill.md` under the accepted run.

## Optimization order

1. **P0: Marlin MXFP4 MoE runtime.** It owns 48.8% of critical-rank model time.
   Instrument and benchmark the DSV4 shape (`H=4096`, `I=2048`, `E=256`,
   `topk=6`) around routing/alignment, W13, activation, W2, and weighted reduce.
   A 20% MoE improvement has an upper-bound model effect near 9.8%.
   The follow-up P0 decomposition and block-M result are recorded in
   `vllm170-dsv4-sm80-marlin-moe-profile-2026-08-14.md`.
2. **P1: C4 attention, first sparse MLA then the remaining Q/compressor path.**
   Sparse MLA alone is 15.1% and C4 is the measured expensive family. A 20%
   sparse-MLA improvement maps to about 3.0% of model time.
3. **P1: attention output projection.** Its 12.0% share is larger than the
   complete measured indexer select path. Study grouped projection layout and
   fusion independently before changing the indexer.
4. **P2: mHC small-row efficiency.** The two transitions total 9.9% and scale
   poorly on the 1K tail. Consider launch reduction only after MoE/attention.
5. **Length-dependent: streaming indexer.** At 5K, logits plus top-k is below
   1% of model time, so streaming top-k is not a short-prefill performance
   priority. Its value remains long-context `O(M*N)` memory and latency control;
   route it only after 32K+ sweeps justify the threshold.

The experimental `mxfp4_int8_sm80` backend is not promoted by this hotspot
result: earlier comparable short-prefill measurements were slower than
Marlin. Any replacement must beat Marlin in an isolated correctness-gated A/B
before a service run.

The P1 follow-up is complete in
`vllm170-dsv4-sm80-attention-p1-2026-08-14.md`. Only direct-output sparse
prefill and `BLOCK_K=32` pass the latency-and-residency promotion gate; the
five-repeat Graph TTFT improves by 2.80% with zero sampled peak-memory increase.
BF16 `wo_b`, BF16 indexer `wq_b`, and compressor state-save warp changes remain
unpromoted after their capacity, endpoint, and isolated A/B results.

## PP partition residency A/B

`VLLM_PP_LAYER_PARTITION=14,15,14` keeps DSpark target layers 40–42 on PP2 and
would improve model-only balance in theory. It failed to initialize in
`profiles/deepseek_v4/20260814T030427Z/server.log`. PP2 held 61.47 GiB and had
1.58 GiB free when DSpark's Marlin MXFP4 repack requested another 2.00 GiB.

Retain `15,15,13`. Revisit balancing only after reducing PP2 persistent or
temporary draft-model residency by at least the observed allocation gap with
additional safety headroom. The profile harness now notices an early server
exit and stops immediately instead of waiting for the health timeout.
