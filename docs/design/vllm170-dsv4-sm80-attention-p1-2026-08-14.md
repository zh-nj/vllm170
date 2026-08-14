# DeepSeek V4 SM80 attention P1 optimization (2026-08-14)

## Scope and measurement boundary

This study covers the two P1 targets identified by the 5K prefill profile:

1. C4 sparse MLA plus its query/indexer/compressor path;
2. inverse-RoPE, `wo_a`, and `wo_b` attention output projection.

The deployment shape is three CMP170HX GPUs, PP3/TP1 with layer partition
`15,15,13`, DSpark width 7, `max_num_batched_tokens=1715`, and target chunks
`1667 + 1667 + 1666`. All service comparisons use the same exact 5,000-token
prompt, 64 output tokens, CUDA Graph execution, five repeats, and disabled
prefix caching.

The accepted control is `profiles/deepseek_v4/20260814T094433Z`. It disables
the sparse P1 serving changes while retaining the already accepted P0 work:

```text
VLLM_DEEPSEEK_V4_SPARSE_PREFILL_CALLER_OUTPUT=0
VLLM_DEEPSEEK_V4_SPARSE_PREFILL_BLOCK_K=0
```

The accepted zero-residency candidate is
`profiles/deepseek_v4/20260814T104308Z`:

```text
VLLM_DEEPSEEK_V4_SPARSE_PREFILL_CALLER_OUTPUT=1
VLLM_DEEPSEEK_V4_SPARSE_PREFILL_BLOCK_K=32
```

CMP170HX returns `CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED`. The report therefore
uses isolated CUDA Events and nested phase CUDA Events, and does not claim NCU
or per-kernel hardware-counter evidence.

## C4 sparse MLA

The original SM80 prefill path allocated a temporary BF16 tensor with the same
shape as the caller output and copied it after the Triton kernel. The new path
can write directly into the stable caller buffer. The same kernel now exposes
a controlled K tile of 16 or 32 while zero retains the previous default.

`profiles/deepseek_v4/sparse_mla_prefill_cmp170hx_20260814.jsonl` contains 90
correctness-gated measurements over the production M values, three accumulated
context positions, C4/C128 index widths, and launch candidates. Direct output
plus `BLOCK_K=32`, `BLOCK_H=16`, and four warps is the only promoted shape:

| Family | Old copy path to promoted path | Minimum observed gain |
| --- | ---: | ---: |
| C4 | 1.2851x median speedup | 17.0% for the K-tile change alone |
| C128 | 1.2504x median speedup | 3.0% for the K-tile change alone |

`BLOCK_H=8` and eight warps regressed across the sweep and are rejected. The
serving implementation retains independent environment rollback for direct
output and K tiling.

In the matched PP3 A/B, `sparse_mla.attention` improved by 21.54%, 20.53%, and
18.80% on PP0, PP1, and PP2. Grouping layer records by compression family gives:

| Family | Control mean/layer/chunk | Candidate | Change |
| --- | ---: | ---: | ---: |
| C4 sparse MLA | 3.530 ms | 2.793 ms | -20.87% |
| C128 sparse MLA | 1.247 ms | 1.028 ms | -17.61% |
| C4 whole attention | 8.411 ms | 7.656 ms | -8.98% |
| C128 whole attention | 5.563 ms | 5.363 ms | -3.60% |

The whole-attention C4 premium fell from 51.2% to 42.7%. C4 remains the hotter
family because it performs real top-512 sparse selection and attends a much
wider combined history than C128. The next C4 kernel study should therefore
target its head/K mapping rather than the already small gather and index-combine
stages.

## Attention output projection

The phase split showed approximately 0.23 ms in inverse RoPE, 0.68 ms in
`wo_a`, and 0.97 ms in `wo_b` per layer/chunk. `wo_a` already retains a BF16
dequantized weight and uses grouped einsum. The remaining opportunity was the
8192-by-4096 block-FP8 `wo_b` Marlin projection.

`profiles/deepseek_v4/output_projection_wo_b_cmp170hx_20260814.jsonl` compares
the packed Marlin path against a load-time cached BF16 matrix. All eight shapes
pass `atol=0.1, rtol=0.02` against the same dequantized weight.

| M | Marlin | BF16 GEMM | Faster path |
| ---: | ---: | ---: | ---: |
| 1 | 58.47 us | 61.34 us | Marlin, 4.9% |
| 6 | 58.98 us | 59.70 us | Marlin, 1.2% |
| 64 | 90.11 us | 80.79 us | BF16, 11.5% |
| 1000 | 752.95 us | 502.07 us | BF16, 50.0% |
| 1667 | 1036.08 us | 771.79 us | BF16, 34.2% |
| 2000 | 1223.58 us | 843.37 us | BF16, 45.1% |

Each cached weight costs exactly 64 MiB. Caching all 43 target layers caused a
64 MiB allocation warning during Graph startup with only about 31 MiB free on
PP2. Restricting the cache to layers 0 through 29 still added exactly 960 MiB
to both PP0 and PP1. That experimental configuration reduced output projection
by about 12.2% and produced a combined 4.23% TTFT improvement, but it spends
persistent memory needed by the planned long-context capacity work.

The result is therefore rejected independent of its latency. All linear BF16
cache attributes, routing code, environment variables, and production launcher
settings were removed. `wo_b` remains on Marlin on all 43 layers. The isolated
benchmark is retained as evidence for a future fused or low-residency design;
it is not a serving option.

## Query and compressor follow-up

After the promoted sparse changes, the C4-only path still reports approximately
1.59-1.82 ms for `indexer.query_prepare` and 1.17-1.32 ms for
`indexer.compress_cache`. These ranges run on parallel streams and must not be
added; their maximum plus the join affects the parent.

The indexer `wq_b` shape is M-by-1024 times 1024-by-8192. A cached BF16 matrix
would cost 16 MiB/layer. The isolated sweep in
`profiles/deepseek_v4/indexer_wq_b_cmp170hx_20260814.jsonl` found BF16 1.61x
faster at M=1667 and 2.44x faster at M=6. The experimental route reduced the
detailed `indexer.query_prepare` range by 17.6%-24.4% across PP ranks.

It is not a production default. Against the otherwise accepted candidate,
`profiles/deepseek_v4/20260814T095811Z` changed five-repeat TTFT from
1575.65 ms to 1579.63 ms and DSpark latency from 76.35 to 76.50 ms/round, while
adding about 112 MiB/rank. The per-chunk model range improved by 0.3%-0.7%, but
the endpoint result did not pass the promotion gate and the residency conflicts
with the capacity objective. The route and environment switch were removed;
only the isolated benchmark remains.

The `save_partial_states` warp sweep is recorded in
`profiles/deepseek_v4/compressor_save_cmp170hx_20260814.jsonl`. For real C4 MLA,
C4 indexer, and C128 MLA shapes, one/two/four/eight warps all land near 33-38 us;
four warps is best or tied. This is far below the 0.7-1.3 ms nested phase range,
which includes upstream GEMM and cross-stream dependency waiting. Tuning this
kernel cannot remove that wait. The next compressor experiment should measure
the fused input projection-to-state-store dependency and event placement, not
change the state-save warp count.

## End-to-end promotion result

| Metric | Control | Candidate | Change |
| --- | ---: | ---: | ---: |
| TTFT, five-repeat median | 1645.18 ms | 1599.19 ms | -2.80% |
| DSpark ms/round, median | 78.42 ms | 77.04 ms | -1.76% observed |
| PP0 model range/chunk | 245.32 ms | 237.34 ms | -3.26% |
| PP1 model range/chunk | 234.43 ms | 226.25 ms | -3.49% |
| PP2 model range/chunk | 210.72 ms | 206.32 ms | -2.09% |
| PP0 Q/index/compress/MLA | 4.362 ms | 3.864 ms | -11.42% |
| PP0/PP1/PP2 peak memory | 57976/57168/63536 MiB | same | 0 MiB |

The DSpark result demonstrates no decode regression but is not attributed to
prefill-only changes; acceptance length varies between repeated requests. The
stable claims are the exact-token TTFT, nested prefill stages, isolated operator
benchmarks, and explicit memory budget.

## Correctness and rollback

The focused validation set is:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -q \
  tests/models/test_deepseek_v4_prefill_workspace_budget.py \
  tests/benchmarks/test_deepseek_v4_profile_workload.py \
  tests/kernels/attention/test_dsv4_sparse_prefill_sm80.py
```

The sparse test checks the old copy path against direct output exactly and the
K=32 candidate within its BF16 tolerance. The only promoted P1 changes can be
disabled independently:

```text
VLLM_DEEPSEEK_V4_SPARSE_PREFILL_CALLER_OUTPUT=0
VLLM_DEEPSEEK_V4_SPARSE_PREFILL_BLOCK_K=0
```
