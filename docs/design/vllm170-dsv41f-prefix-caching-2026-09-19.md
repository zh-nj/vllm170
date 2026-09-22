# Prefix caching on the DeepSeek-V4.1-Flash PP5 port

Date: 2026-09-19. Scope: make `--enable-prefix-caching` actually take effect on the
port instance, and establish what "in effect" means for this model.

Instance: PP5 / TP1 on 5x CMP 170HX (SM80), `mxfp4_int8_sm80` MoE,
`TRITON_MLA_SPARSE_DSV41` attention, FP8 KV, `block_size=128`, `max_num_batched_tokens=1024`,
`max_num_seqs=8`, FULL target cuda graphs, DSpark width 5 with
`dspark_pipeline_stage_ranks=[1,2,3]`, vision on (`PORT_MM_IMAGES=2`).

## Verdict

It was off, and off in the strong sense: the launcher hard-coded
`--no-enable-prefix-caching`, which does not merely skip block reuse, it turns block
hashing off altogether. `vllm:prefix_cache_queries_total` sat at exactly 0, so no
prompt could ever hit, no matter how often it was sent.

It is available as an explicit mode (`PORT_PREFIX_CACHING=1`) and works end to
end alongside the independently configured fast-prefill mode: exact
block-aligned hits, ~2x lower TTFT on a repeated 1.5-2.5K prompt, ~11x on a repeated
32K prompt, 59-69% engine hit rate in normal serving, byte-identical reference text
against the caching-off server, and working multimodal reuse. Decode throughput is
unchanged within the noise of this box.

Three behaviours showed up while verifying it. All three are root-caused and
fixed, with the evidence below:

- **The prefill result is sensitive to batch shape.** A historical sweep reported
  discrete logprob changes. Its `shape-{n}` salt did not verify cache-hit
  counters and recorded no mode provenance, so its 1.789114952-nat spread is a
  historical measurement rather than clean zero-reuse evidence -- but the effect
  is real, and it had three independent causes: Marlin's M-tuned tiling, the
  `out_dtype` overload of `aten::mm` that the batch-invariant overrides did not
  cover, and the MHC kernels choosing a split and a tiling from the row count.
  With all three repaired, a 35-length sweep across the old boundary is a single
  value (spread 0.0) and cold/warm now match: 0 of 16 pairs diverge over 128
  greedy tokens, against 8 of 8 realistic prompts before.
- **A prompt that ends 1..32 tokens past a 128-block boundary reconciled to zero
  reuse**, even though its blocks were cached. Root-caused to the sparse-retention tail
  in `SlidingWindowManager.reachable_block_mask` and **fixed**; the whole band now
  reuses the aligned prefix.

The three repairs are opt-in, as a group, through
`--additional-config '{"marlin_full_k":true,"dsv41_strict_bf16_gemm":true}'`;
with the defaults the engine behaves as before.

## Why it was off

`--no-enable-prefix-caching` arrived with the first launcher commit (`f5107b8f4f`) and
was copied into `run_cmp170hx_pp5.sh` unchanged. Nothing in the PP5 path needs it:

- The only component that requires prefix caching disabled is CED
  (`vllm/models/deepseek_v4_1/ced.py:33` raises `ValueError` when
  `cache_config.enable_prefix_caching`), and this launcher does not enable CED.
- `--block-size 128` is compatible with every group the model declares (checked below).

So the flag was inherited, not required.

## Why turning it on is legal for this model

The model declares four prefix-cacheable layouts plus one scratch layout, which is
exactly the shape the hybrid coordinator is built for:

| Group | Spec | block_size | prefix_cacheable |
| --- | --- | --- | --- |
| compressed KV (kv-source layers) | `MLAAttentionSpec`, `tokens_per_state` = `compress_ratio` | 128 | yes |
| indexer K | `MLAAttentionSpec` | 128 | yes |
| SWA | `SlidingWindowMLASpec`, `sliding_window=128` | **32** | yes |
| cross-rank replica | `MLAAttentionSpec` | 128 | yes |
| compressor state ring | `CircularBufferSpec` | 8 | **no** |

- `compress_ratios = [0,0,2*18,1*20,0,0,0]`, so `tokens_per_state` is 1 or 2 and every
  prefix-cache boundary (a multiple of the 128-token scheduler block) lands on a
  compression-group boundary. The engine asserts this itself
  (`kv_cache_utils` `prefix_alignments`), and the config loads.
- `CircularBufferSpec.prefix_cacheable is False` keeps the compressor's open-group ring
  per-request scratch, so no partially compressed group is ever shared.
- DSpark is `use_eagle() == True` with `use_multi_module_mtp() == False`, so
  `num_prefill_lookahead = 1`, and `deepseek_v41` gets the positional eagle-group
  fallback: the group holding the drafter's attention layer is flagged, so the
  lookahead tail is handled by the same mechanism every spec-decode model uses.
- The cross-rank KV replicas (`DeepseekV4PPCompressedCache`) are written from the PP
  message through the ordinary `slot_mapping`, at the block ids the scheduler assigned
  for the same tokens, so a hit reuses the source block and its replica together.

## Recipe

```bash
PORT_PREFIX_CACHING=1 PORT_MM_IMAGES=2 PORT_KV_BYTES=700000000 PORT_MAX_MODEL_LEN=262144 \
PORT_CUDAGRAPH_SIZES='[1,2,4,6,8,12,16,24,32,40]' \
VLLM_DSV41_PREFILL_WORKSPACE_TOKENS=32768 \
  benchmarks/deepseek_v4_1/start_pp5_patient.sh <log> ...
```

`PORT_PREFIX_CACHING` defaults to 0 independently of fast prefill. Enable it
explicitly with `PORT_PREFIX_CACHING=1`; fast prefill may remain enabled.
Nothing else changed, and the KV capacity is identical either way (499,062 tokens,
same `kv-cache-memory-bytes`, same graph capture list) -- so a comparison isolates the
flag.

## Verification

All numbers below are from the same box on the same day, with only
`PORT_PREFIX_CACHING` toggled.

| Check | Caching off | Caching on |
| --- | --- | --- |
| `vllm:prefix_cache_queries_total` | 0 | 755,899 cumulative |
| `vllm:prefix_cache_hits_total` | 0 | 448,512 cumulative |
| engine "Prefix cache hit rate" | 0.0% | 59.3-69.1% |
| 1500-token prompt, TTFT cold -> repeat | 1.05 -> 1.03 s | 0.99 -> 0.50 s |
| 2500-token prompt, TTFT cold -> repeat | 1.14 -> 1.13 s | 1.09 -> 0.52 s |
| 32K prompt, TTFT repeat 1/2/3 | 6.14 / 6.10 / 6.14 s | 6.28 / 0.57 / 0.59 s |
| 5 reference prompts, greedy token ids | baseline | **5/5 identical** |
| two-image reference example | PASS | **PASS**, 3 repeats, 640 cached tokens each |
| DSpark acceptance / accepted length | 95.45% / 5.773 | **95.45% / 5.773** |
| accepted-per-position | [66,66,66,60,57] | **[66,66,66,60,57]** |
| decode, 32K+q5, ms per 5.8-token step | 85.1 / 85.3 / 87.1 (median 85.3) | 84.8 / 87.2 / 91.5, 76.4 / 82.8 / 87.5 / 89.8 / 90.8 (median 87.2) |

Hit lengths are exact and block-aligned, not approximate:

| prompt | tail | reused | cap `((n-1)//128)*128` |
| --- | --- | --- | --- |
| 1500 | 92 | 1408 | 1408 |
| 2500 (crosses the 1024 chunk boundary) | 68 | 2432 | 2432 |
| 1216 | 64 | 1152 | 1152 |
| 1313 | 33 | 1280 | 1280 |

The cap is `num_tokens - 1`, floored to the hit alignment: a request always recomputes
its last prompt token to obtain logits (`KVCacheManager.get_computed_blocks`).

The A-B-A on decode throughput is the honest answer for this box: caching-off medians
85.3 ms/step against 87.2 ms/step caching-on, while run-to-run spread within a single
configuration is 76.4-91.5 ms (about 7%). The two distributions overlap, so there is no
measurable decode cost to the flag, and no measurable gain either -- the benefit is
entirely in prefill (TTFT).

## Finding 1: prefill shape dependence, root-caused and repaired

### Evidence and corrected scope

The historical 80-length sweep in `/tmp/final_shape.json` has a spread of
1.7891149520874023 nats. That artifact does not contain cache-hit counters or
server configuration. Its original script reused `shape-{n}` salts across runs;
zero reuse cannot be established from that file alone.

The corrected probe generates a UUID for each run, gives each request a distinct
salt, checks prefix-cache query/hit counter deltas, and checks per-request cached
usage when available. It retains unrounded values and repeats each length.
`/tmp/dsv41_corrected_baseline.json` records six requests on the restored default
service: two each at lengths 1301, 1405, 1533. All measured hit deltas were zero,
repetitions were exactly equal, and the three logprobs were -10.063070297241211,
-10.57004165649414 and -11.28186321258545. The three-length spread is 1.218792915,
which must not be compared as the same denominator as the older 80-length sweep.

Two indexing corrections matter:

- `/tmp/dsv41_rowdump_log` records a 1020-token first chunk, not 1024, for these
  requests. The old table's `n-1024` tail labels were four rows too short; the
  corresponding second chunks were 281, 385 and 513 rows. The corrected probe
  does not guess scheduler M. `--chunk1` only annotates an explicit assumption.
- `prompt_logprobs[1300]` is the probability of token 1300, predicted by hidden
  state at position 1299. The old dump selected position 1300. Consequently its
  boundary evidence is for an adjacent row, not that exact API logprob. The old
  claim that index 1024 was the first row of chunk 2 is also withdrawn.

CPU reinspection of `/tmp/dsv41row.rank0.M{281,385,513}.pt` confirmed:

| layer 0 stage | M281 vs M385 | M281 vs M513 |
| --- | ---: | ---: |
| mhc_pre_attn | 0 | 0 |
| attn_out | 0.0078125 | 0.0078125 |
| ffn_out | 0.015625 | 0.015625 |

These are single-row comparisons. `self.attn` includes input projections, Q/KV
normalization, RoPE/cache insertion, sparse SWA attention, `wo_a`, and `wo_b`.
The dump did not establish equality of all visible K/V inputs. Layer 0 has
`compress_ratio=0`, so its own compressor/indexer/top-k cannot cause this first
boundary; this does not rule out additional independent sources in later layers
or the MoE. The earlier cuBLAS random-matrix reproduction establishes shape
sensitivity of that primitive, not a causal explanation of every engine plateau.

### Marlin isolation and candidate repair

On the otherwise idle CMP 170HX at PCI 02:00.0, the real checkpoint's layer 0
weights were loaded with the same E8M0 byte interpretation and 32-row scale
expansion as the model loader. The first projection concatenates wq_a then wkv.
Holding the inputs fixed and comparing all common rows, not just row 0, gives:

| projection | N,K | default maximum cross-M difference |
| --- | --- | ---: |
| fused_wqa_wkv | 1792,5120 | 0.03125 |
| wq_b | 32768,1280 | 0.015625 |
| wo_b | 5120,8192 | 0.03125 |

All same-M repetitions were identical. Pinning thread K/N alone still left
cross-M differences. Marlin derives `parallel`, `global_mn_tiles`, the stream-K
tail, and `iters` from M, so fixed tile height alone is insufficient.

The candidate `marlin_full_k` path fixes the thread K/N configuration and launches
one CTA for each output tile. Every CTA traverses all K, removing cross-CTA
partial-sum reassociation. It does not divide prefill into repeated 16-row GEMM
launches. On the three shapes above, the tested common rows were bit-identical
for M=1..1024, including 281/385/513 and tile boundaries. The existing pytest suite
also verifies a row permutation and compares to a double-precision reference.
Full engine invariance remains a separate acceptance condition.

Use `--additional-config '{"marlin_full_k":true,"dsv41_strict_bf16_gemm":true}'`
for the current engine trial. Both options default to false. This is a candidate,
not a declaration that all DeepSeek V4.1 operators are batch-invariant.

The older `VLLM_MARLIN_M_ALIGN` covers only M in 9..128. Its source comment is
historical evidence, not a measured prefill cost bound. In particular the quoted
1922 -> 7318 microseconds corresponds to +280.75%, not +177%, and its 56-row
example does not describe the current default launch geometry.

### Root cause: `aten::mm`'s `out_dtype` overload is not covered

With the full-K launch enabled, the divergence is no longer the batch in the way
the earlier framing assumed. Comparing chunk shapes 281, 385 and 513 rows of the
same three prompts, where 281 and 385 are the control:

| rank | layers | M281 vs M385 | M281 vs M513 |
| --- | --- | ---: | ---: |
| 0 | 0-7 | identical | identical |
| 1 | 8-15 | identical | identical |
| 2 | 16-19 | identical | identical |
| 2 | 20 `attn_out` | identical | 1.562e-02 |
| 2 | 20-23 `ffn_out` | identical | up to 5.6e-02 |
| 3, 4 | 24-39 | identical | inherited over the pipeline boundary |

Two prompts differing only in trailing tokens are bit-identical below the
boundary and differ above it, so the trigger is a discrete threshold, not
accumulated noise. A length sweep puts it between chunk 2 = 509 and chunk 2 = 513
rows. Layer 20 is the first layer with `compress_ratio = 1`:

| layers | `compress_ratio` |
| --- | ---: |
| 0-1 | 0 (sliding window only, no compressor or indexer) |
| 2-19 | 2 |
| 20-39 | 1 |
| 40-42 | 0 (MTP) |

So the 16 ratio-2 layers below layer 20 see the same 513-row batch and stay
bit-identical. Whatever moves is specific to the ratio-1 geometry.

`_ProbeRowDump` narrowed it to a single stage inside layer 20. Ordered by the
attention's causal sequence, for the round-robin dump with `layers.20` selected:

| stage inside layer 20 | M281 vs M385 | M281 vs M513 |
| --- | ---: | ---: |
| `mhc_pre_attn` (input) | 0 | 0 |
| `input_proj.qr_kv` (fused_wqa_wkv, MXFP8 Marlin) | 0 | 0 |
| `input_proj.kv_score` | 0 | **9.686e-07** |
| `compressor.latent` | 0 | 1.907e-06 |
| `indexer.index_q`, `indexer.index_weights` | 0 | 0 |
| `indexer.topk_indices` | 0 | 0 |
| `attn_raw` | 0 | 4.883e-04 |
| `o_proj.out`, `attn_out`, `ffn_out` | 0 | 7.813e-03 |

`kv_score` is the compressor's pooling-score projection, and it is the one input
GEMM that is not a quantized linear:

```python
# vllm/models/deepseek_v4_1/attention.py
def compressor_kv_score() -> torch.Tensor:
    return torch.mm(
        hidden_states,
        compressor.fused_wkv_wgate.weight.T,
        out_dtype=torch.float32,
    )
```

Its width is `512` for `compress_ratio == 1` (no pooling gate) and `1024` for
`compress_ratio == 2` (kv plus gate), which is precisely what separates layer 20
from the layers below it. The top-k selection is *identical*, so this is not a
selection change; the drift spreads to all 128 rows of the dump window uniformly
(maximum 3.0e-06), which is what a reduction-order change looks like, and it
reaches the attention through the compressed K cache that `latent` writes.

Isolated on a spare CMP 170HX, `mm([M,5120], [5120,N], out_dtype=float32)` with a
bit-identical row 0, under `enable_batch_invariant_math()` (what
`dsv41_strict_bf16_gemm` calls):

| shape | `mm(out_dtype=f32)` rows that move | `mm()` / `matmul` |
| --- | --- | --- |
| N = 512 | M = 513, 514, 600, 1020, 1024 | none |
| N = 1024 | none | none |

The threshold is exactly `M > 512` and only for the fp32-output overload: the
SM80 branch of `enable_batch_invariant_mode()` registers `aten::mm`,
`aten::addmm`, `aten::matmul` and `aten::linear` *without* their `out_dtype`
overloads, so `aten::mm.dtype` keeps dispatching to cuBLAS. That is also why the
strict flag only ever halved the spread instead of removing it, and why
`CUBLAS_WORKSPACE_CONFIG=:16:8` does not help either (measured: the fp32 path then
moves at every M from 281 up, so the limit selects different kernels rather than
a fixed order). A plausible reading of the default behaviour is that cuBLAS turns
on split-K once the batch is wide enough to leave the device idle at 512 rows,
which would put the reassociation exactly at the observed boundary; that part is
inference from the threshold, not a captured kernel selection.

### Repair

`matmul_persistent` now honors an `out_dtype`, so the fp32 result is the fp32
accumulator itself rather than a rounded bf16 value, with the tiling still chosen
from the input dtype. `mm_batch_invariant` passes it through and the SM80 branch
registers the missing `aten::mm.dtype` overload. The fix is inert unless
`dsv41_strict_bf16_gemm` is on, so the default configuration is unchanged.

On the same isolated shape, the three production projections and the fp32 path
are bit-identical across M = 1..1024 including every crossing of 512, and the
result stays an fp32 accumulate: against an fp64 reference the error is two
orders of magnitude below the bf16 result the overload exists to avoid
(`tests/v1/determinism/test_matmul_batch_invariant.py`).

`aten::mm.out` and `aten::mm.dtype_out` remain uncovered; no call site in this
model uses them. The mode also makes unsupported dtypes raise rather than fall
back, which is pre-existing: an fp64 `matmul` under the flag fails with
`KeyError: torch.float64`, so the registration check runs in a child process.

Cost, measured on the shapes this model actually uses, cuBLAS against the
override (median of 30 CUDA-event timings after warmup):

| GEMM | M | cuBLAS | override |
| --- | ---: | ---: | ---: |
| lm_head, N = 129280 | 1 | 0.96 ms | 3.53 ms |
| lm_head, N = 129280 | 40 | 0.91 ms | 2.57 ms |
| lm_head, N = 129280 | 1533 | 12.01 ms | 24.73 ms |
| compressor, N = 512 | 513 | 0.07 ms | 0.52 ms |

The lm_head projection is bandwidth-bound on a 1.3 GB weight, so the override
mainly loses the small-M efficiency of `BLOCK_SIZE_M = 128`. That is the price of
exactness on this box; the tuned-config table in
`vllm/model_executor/determinism/batch_invariant_configs.py` is where it could be
recovered.

### Verification of the repair

Measured on the PP5 instance with `--additional-config
'{"marlin_full_k":true,"dsv41_strict_bf16_gemm":true}'`, one request per length,
each with a fresh UUID salt:

| check | before | after |
| --- | --- | --- |
| three lengths 1301/1405/1533, two repeats each | 2 distinct logprobs, spread 0.243736 | **1 value, spread 0.0** |
| the 35-length sweep across the old boundary | 2 distinct values | **1 value, spread 0.0** |
| layer 20 `input_proj.kv_score`, M281 vs M513 | 9.686e-07 | 0 |
| layer 20 `attn_out`, M281 vs M513 | 7.813e-03 | 0 |
| every stage, every layer, all five ranks | denominator diverged | 0 |

The only keys that still differ between the three shapes are the probe's own
`context_rows` and `support_indices`, which encode the context width and are
supposed to (232 = 1533 - 1301). The prefix-cache gate still passes with all five
cases reusing (mean TTFT 1.097 -> 0.489 s).

The absolute logprob moves (about -10.64 to -10.04 on that prompt) because the
reduction order changed from cuBLAS's to the persistent kernel's. That is
expected, and it is the same amplification the ledger already measured: a 1e-6
drift at layer 20 becomes 0.24 to 0.6 nats at the logits. Invariance across
shapes, not equality with the previous kernel's value, is the property being
established.

### Finding 3 (fixed): the MHC kernels tiled themselves by row count

After the `mm.dtype` repair the probe was flat, but cold and warm still produced
different tokens in three of the gate's five cases. The row dump says where:
layer 0 is bit-identical between a 480-row cold chunk and a 92-row warm tail,
and layer 1's `mhc_pre_attn` is the first thing that differs -- by 3.05e-05, one
bf16 ULP, at an element of magnitude 0.01 -- before the attention amplifies it.

Nothing upstream of that value differs, so the pre-norm offers two candidates,
and both of them read the row count:

- `_tilelang_hc_prenorm_num_splits` returns 1 above 128 rows and otherwise the
  largest legal split, which at the V4.1 geometry is 10 for 92 rows and 1 for
  480.
- `_tilelang_hc_prenorm_gemm` picks between three tilings of the same reduction:
  a block-M kernel at `>= 1024` rows, `n_thr=1024, tile_n=4` below 128 rows, and
  `n_thr=512, tile_n=12` in between.
- `mhc_post_tilelang` chooses a different fused kernel entirely at
  `num_tokens <= 16` (`use_small_fma`), with `n_splits` 4 or 8 instead of 1.

A prefix-cache hit recomputes at most 128 tokens of tail -- and a speculative
decode step far fewer -- so a warm request lands on the small side of every one
of those thresholds while the cold prefill it replaces lands on the large side.
That is also why the probe's 281..513-row sweep could not see it: that sweep
never goes below 128 rows.

The repair pins all three to the large-batch choice when
`batch_invariant_mode_enabled()` is set -- `n_splits = 1`, both special tilings
off, which is the path every in-range prefill chunk already takes. Without the
flag the heuristics are untouched. `tests/kernels/test_mhc_kernels.py` asserts
the split rule directly, through an `invariant=` argument, so it does not have
to install the process-global overrides.

Verified by dumping the same prompt served cold and warm, at the position that
predicts the first generated token:

| check | before | after |
| --- | --- | --- |
| every stage, every layer, rank 0 | layer 1 `mhc_pre_attn`, 3.05e-05 | 0 |
| every stage, every layer, all 5 ranks | 21/24/49/24/24 keys | 0 |

One key still differs, and has to: `layers.20.attn.prefill.support_indices`. Its
first 512 entries (the compressed top-k) are identical and its last 128 (the
sliding window) differ by exactly 388 = 480 - 92, because that part is addressed
relative to the query's own gathered row. Both point at the same KV rows, and
the only consumer of the indices, `rocm_sparse_attn_prefill`, returns a
bit-identical result.

End-to-end, on the same instance:

| check | before | after |
| --- | --- | --- |
| gate cases with any cold/warm token difference | 3 of 5 | **0 of 5** |
| gate mean TTFT cold -> warm | 1.097 -> 0.489 s | 1.096 -> 0.477 s |
| 16 cold/warm pairs, 128 greedy tokens each, tails 4..128 | 8 of 8 realistic prompts diverged | **0 of 16** |
| decode, 32000-token prompt | 89.6 ms/token | 80.9 ms/token |

The decode difference is across two restarts rather than within one window, and
this box spreads about 7% run to run, so read it as "not slower": the pinned
tiling is the one prefill already used, and the small-batch branch it replaces
was not the faster one here. Vision is unchanged (both images named, 3/3 repeats
identical, 640 cached tokens) and DSpark acceptance is 94.55% at 5.727 accepted
tokens, against 95.45% at 5.773 before the repairs -- the arithmetic path moved,
so a small move in acceptance is expected, not a defect.

### What remains open

`aten::mm.out` and `aten::mm.dtype_out` are still uncovered by the overrides,
though no call site in this model uses them, and the mode makes unsupported
dtypes raise instead of falling back (an fp64 `matmul` under the flag fails with
`KeyError: torch.float64`). The exactness above is measured for this model's
layers, not proven for every operator: each new batch-shaped heuristic in a
kernel launcher is a new instance of the same class of bug, and the row probe is
what finds them.

Two candidates were ruled out rather than left in the background, because both
are the only *literal* 512-row boundaries in the tree and either would be a
coincidence worth checking before trusting the explanation above:

- `split_indexer_prefill_chunks` sub-chunks the indexer's prefill rows at
  `VLLM_INDEXER_PREFILL_STREAMING_MAX_ROWS` (512), and switches top-k
  implementations with them. It is gated on
  `VLLM_INDEXER_PREFILL_STREAMING_TOPK`, which defaults off, and neither the
  worker environment nor the launcher sets any of that family.
- The same function caps a sub-chunk at `VLLM_SPARSE_INDEXER_MAX_QUERY_ROWS`,
  which defaults to 1024 -- above every chunk this deployment schedules
  (`max_num_batched_tokens` is 1024, and the first chunk is 1020).

Both were confirmed unset on the running instance. The `N = 512` in the
compressor is the cr-dependent parameter, and it is the stage the dump
convicts.

### Upstream comparison

The user-provided `/mnt/data/apps/vllm` contains current fetched refs. Its working
checkout is detached at `0b3ba88f165976e77ca5e6a7a3f5bba4562b80af`; the relevant
comparison is fetched `origin/main` at
`c3b4844634` (2026-09-19), read using `git show`, without changing that worktree.
The current upstream Marlin still computes M-dependent stream-K slices and has
no corresponding full-K/batch-invariant MXFP8 mode. Upstream's SM80 ATen overrides
are already represented locally and do not intercept custom Marlin GEMMs.

Upstream commit `ef5f7cd119` caches dequantized weights for the ROCm native MXFP8
backend; it is not a repair for the CUDA Marlin path. The fetched upstream MXFP8
emulation implementation matches this tree's implementation.

### Strict math and emulation: separate evidence

`dsv41_strict_bf16_gemm` invokes the full ATen batch-invariant registration helper:
mm/addmm/matmul/linear, bmm, softmax/log_softmax, mean, reduced-precision flags,
and preferred BLAS selection. It does not enable the global backend capability
checks or change the NCCL environment. Therefore its measured effect cannot be
attributed to BF16 GEMM alone. Explicit JSON booleans are required; worker logs
record the effective value per rank.

The older `/tmp/bimath_shape.json` has spread 0.749158859; `/tmp/strict_shape.json`
has spread 0.981816292. Neither stores sufficient mode/configuration provenance.
The reported decode 73 -> 77 ms/step and cold TTFT 0.95 -> 1.15 s are historical
observations, not a controlled estimate of this candidate's performance or proof
that the TTFT change is within noise.

The combined emulation-plus-strict trial returned a NaN serialization error.
That establishes a failed configuration, not the location or cause of its first
non-finite value. The real layer 0 scale bytes contain no 255 entries. Their
range is 114..117; a scale=255 hypothesis is unsupported for those projections.
No `nan_to_num` or output masking is an acceptable repair.

### Acceptance boundary

The prefix-cache gate checks cold A, cold B and warm A, including reuse and the
short-tail regression. It does not execute a second warm request and does not
require cold/warm token equality. Its PASS is a cache-reuse result, not a pass
for this issue. Some historical gate artifacts explicitly contain cold/warm
output differences despite PASS.

For this issue, require corrected cold counter evidence, same-position intermediate
comparisons, and exact cross-M equality of the repaired primitive; then separately
check engine logprobs and cold/warm greedy outputs on the same prompt set. Model
quality and latency also require engine measurements. Stabilizing one GEMM, a
smaller logprob spread, or five successful reuse cases does not close the issue.

## Finding 2 (fixed): a 1..32 token tail block reconciled to zero reuse

With genuinely unique prompts (`cold_hits == 0` proving the cold run was cold), reuse
depended on where the prompt ends inside its last 128-block:

```text
before the fix            after the fix
  tail   0 : 1280           tail   0 : 1280
  tail 1-32: 0              tail 1-32: 1280      <- full band restored
  tail  33+: 1408           tail  33+: 1408
```

It was not a race (a 3 s gap changed nothing), not content (six different contents at the
same length behaved identically), and not a failure to cache: a 1408-token *prefix* of a
failing prompt hit 1280 immediately afterwards. Only the request's own blocks were
unreachable to itself.

### Root cause

Two facts meet, and the first one is easy to miss:

1. `prefix_cache_retention_interval` **defaults to `0`, not `None`**
   (`config/cache.py:_get_prefix_cache_retention_interval`). Sparse retention means the
   SWA group stores only the `need`-block tail ending at each reachable boundary --
   for this model `need = cdiv(128 - 1, 32) + 1 = 5` blocks of 32, out of the whole
   prompt.
2. `SlidingWindowManager.find_longest_cache_hit` only matches a run of `need` blocks that
   *starts* on an `alignment_tokens` boundary and stays below
   `min(end, (num_prompt - 1) // block_size)`.

The store computes `end = aligned // 32 + shift`, where `aligned` is
`(num_prompt - 1) // 128 * 128` and `shift = 1` under EAGLE, so the retained tail is
`[end - 5, end)`. When the
prompt ends fewer than 33 tokens past the aligned boundary, block`end - 1` is not a full
block of that request -- `num_prompt // 32` is one short -- so the store skips it, only 4
of the 5 blocks exist, and the finder can never reach `need`. Every lookup of that prompt
reconciles to 0 and the whole prompt is re-prefilled.

`VLLM_KV_CACHE_DEBUG=1` shows it directly; `mask=5` is what the mask asked for, and only
four blocks exist:

```text
cache_blocks: group=0 SlidingWindowManager block_size=32 cached_before=29
              num_full_blocks=45 hashes=45 mask=5
prefix lookup: max=1423 hit=0 longest=1408 ... groups=[('MLAAttentionSpec',128,False,0),
              ('SlidingWindowMLASpec',32,True,0)]
```

### Fix

`SlidingWindowManager.reachable_block_mask` now also retains the previous *aligned* run
when the boundary tail cannot be matched:

```python
reach = min(end, end_block, boundary_tokens // block_size)
if tail_start + need > reach or tail_start >= end_block:
    fallback = (reach - need) // alignment_blocks * alignment_blocks
    for j in range(max(start_block, fallback), min(end_block, fallback + need)):
        mask[j - start_block] = True
```

Outside the broken band nothing changes: when the tail is matchable, `reach` already
covers it and the branch is skipped, so the retained set is byte-identical to before.

The band sweep after the fix (same probe, fresh content per length):

```text
tail   0: 1280   tail   1: 1280   tail   2: 1280   tail   4: 1280
tail   8: 1280   tail  16: 1280   tail  24: 1280   tail  31: 1280
tail  32: 1280   tail  33: 1408   tail  40: 1408   tail  64: 1408
tail  96: 1408   tail 127: 1408          no-reuse lengths: none
```

`prefix_cache_gate.py` carries the `short-tail` case, and it asserts a *bound* rather
than the exact value: the hit must land in `[(full - 1) * block_size, full * block_size]`
of the aligned prefix. Asserting the exact per-length value would only mirror the
implementation, and the exact value legitimately depends on how much of the tail the
retention policy could keep.

## Reproducing

```bash
# pass/fail gate, caching on
.venv/bin/python benchmarks/deepseek_v4_1/prefix_cache_gate.py --expect-hits

# negative control: must FAIL with "hits=N while caching is expected off"
.venv/bin/python benchmarks/deepseek_v4_1/prefix_cache_gate.py

# multimodal reuse
.venv/bin/python benchmarks/deepseek_v4_1/vision_reference_example.py --repeat 3

# cross-config output equality
.venv/bin/python benchmarks/deepseek_v4_1/dspark_output_dump.py --out /tmp/on.json   # on both servers

# prefill batch-shape sensitivity (Finding 1), zero reuse
.venv/bin/python benchmarks/deepseek_v4_1/prefix_cache_shape_probe.py

# the same effect isolated to cuBLAS on a spare GPU
python benchmarks/kernels/deepseek_v41_gemm_batch_invariance.py --shapes --map

# the opt-in mitigation (measured: spread 1.79 -> 0.75 nats, decode ~5% slower)
PORT_EXTRA_ARGS='--additional-config {"dsv41_strict_bf16_gemm":true}' \
  benchmarks/deepseek_v4_1/run_port_pp5.sh

# the KV cache group layout and per-group hit lengths (Finding 2)
VLLM_KV_CACHE_DEBUG=1 ... benchmarks/deepseek_v4_1/run_port_pp5.sh
```

Two traps the gate scripts now avoid, both of which produced wrong conclusions first:

- `--no-enable-prefix-caching` is not "caching is off but hashing still runs": queries
  stay 0, so an "off" measurement cannot be distinguished from "on but never hit"
  without reading the counters.
- `decode_gate.make_token_ids` builds its text from a 57-word cycle, so its prompts are
  periodic and two different seeds can tokenize to the *same* ids. That silently turns a
  "cold" request into a hit and made a whole length sweep meaningless. The gate now
  samples fresh words per call, and every case runs a second cold prefills under a
  distinct `cache_salt` to prove the first one was cold.

## Files

- `benchmarks/deepseek_v4_1/run_port_pp5.sh`: `PORT_PREFIX_CACHING` replaces the
  hard-coded `--no-enable-prefix-caching`.
- `benchmarks/deepseek_v4_1/prefix_cache_gate.py`: new. Metric deltas, exact hit
  lengths, TTFT, the `cache_salt` cold/cold control, and the pinned short-tail case.
- `benchmarks/deepseek_v4_1/vision_reference_example.py`: `--repeat N`, which turns the
  reference example into a multimodal reuse check.
- `benchmarks/deepseek_v4_1/prefix_cache_shape_probe.py`: new. The Finding 1 sweep at the
  engine level.
- `benchmarks/kernels/deepseek_v41_gemm_batch_invariance.py`: new. The same effect
  isolated to cuBLAS, plus the M-threshold map and the candidate knobs.
- `vllm/v1/core/single_type_kv_cache_manager.py`: the retention-tail fix, plus
  `VLLM_KV_CACHE_DEBUG`-gated store/lookup diagnostics that made Finding 2 legible.
- `vllm/v1/core/kv_cache_coordinator.py`: the same env var dumps the group layout once
  per engine and the per-group hit lengths per lookup.
- `vllm/envs.py`: `VLLM_KV_CACHE_DEBUG`, and `VLLM_DSV41_STRICT_BF16_GEMM` (default 0).
- `vllm/model_executor/determinism/batch_invariant.py`: `enable_batch_invariant_math`,
  the narrow subset of the batch-invariance mode that this model can use.
- `vllm/v1/worker/gpu_worker.py`: the knob is read from `additional_config` first,
  because an arbitrary `VLLM_*` name set on `vllm serve` does not reach the workers.
- `vllm/models/deepseek_v4_1/nvidia/model.py`: `_ProbeRowDump`, the layer/stage dump
  that localized the remaining dependence to the attention.
