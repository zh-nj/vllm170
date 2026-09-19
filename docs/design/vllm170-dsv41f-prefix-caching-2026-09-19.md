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

It is now on by default (`PORT_PREFIX_CACHING=1`), and it works end to end: exact
block-aligned hits, ~2x lower TTFT on a repeated 1.5-2.5K prompt, ~11x on a repeated
32K prompt, 59-69% engine hit rate in normal serving, byte-identical reference text
against the caching-off server, and working multimodal reuse. Decode throughput is
unchanged within the noise of this box.

Two behaviours showed up while verifying it, and both are documented below:

- **The bf16 GEMMs are not batch-invariant**: cuBLAS picks a different reduction order
  per `M`, worth exactly one bf16 ULP on a bit-identical row. Accumulated over a prefill
  chunk that is 0.026 nats at its first row and 1.79 nats by its 277th, which is enough
  for greedy decoding to diverge between a cached and an uncached serving. Proved below
  with zero reuse, and isolated to cuBLAS on a spare GPU. Not fixable at the
  prefix-cache layer; diagnosed, not fixed.
- **A prompt that ends 1..32 tokens past a 128-block boundary reconciled to zero
  reuse**, even though its blocks were cached. Root-caused to the sparse-retention tail
  in `SlidingWindowManager.reachable_block_mask` and **fixed**; the whole band now
  reuses the aligned prefix.

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

`PORT_PREFIX_CACHING` defaults to 1; `0` restores the old behaviour for A/B runs.
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

## Finding 1 (diagnosed, not fixed): the GEMMs are not batch-invariant

Cold and warm runs of the same prompt can produce different completions. It is not the
cache, and it is not fixable at the prefix-cache layer: a row's prefill result depends on
how many tokens are in the batch it is computed in. A prefix-cache hit is only the most
common way to change chunking -- the same divergence appears with caching off when
`max_num_batched_tokens` changes (1024 -> 1152 already changed 2 of 5 reference prompts
in an earlier measurement on this box).

### It is not the cache, and not non-causal

Every request in this sweep carries a unique `cache_salt`, so each is a full cold prefill
with zero reuse:

```
logprob of token 1300 vs the length of the chunk it is computed in
(first chunk fixed at 1024, so the second chunk is n - 1024):

  n=[1301,1337]  tail=[277,313]  logprob=-10.063070
  n=[1341,1393]  tail=[317,369]  logprob=-9.922690
  n=[1405,1465]  tail=[381,441]  logprob=-10.570042
  n=[1469,1529]  tail=[445,505]  logprob=-10.791334
  n=[1533,1565]  tail=[509,541]  logprob=-11.281863
  n=[1597,1617]  tail=[573,593]  logprob=-10.685512
```

Bit-identical over a range, then a discrete jump; 8 distinct values over 80 lengths,
spread 1.79 nats. Two controls say what kind of effect this is:

- **Not leakage.** The token at index 900, in the always-1024-token first chunk, is
  bit-identical (`-0.000127`) across every `n`.
- **Not content.** Replacing the tokens after the probe with different tokens *of the same
  length* leaves the probe bit-identical (`-9.867739`, four different tails).

Profiling the spread per probe index over chunk-2 lengths 16..597 shows the shape of it:
index 1024 (the first row of chunk 2) already moves, by only 0.026 nats; by index 1300 it
is 1.79. So the primitive effect is a small per-row perturbation, and the large output
changes are accumulation plus greedy amplification at near-tied tokens.

### Root cause: cuBLAS picks a different kernel per M

`benchmarks/kernels/deepseek_v41_gemm_batch_invariance.py` reduces it to a ten-line
reproduction on a spare GPU: hold row 0 of the activations bit-identical, vary `M`, and
compare row 0 of the product.

```
worst |row0 drift| across M in {277,320,384,448,512,597,900,1024}, bf16:
   5120 x 5120: 1.562e-02  M-DEPENDENT      2304 x 5120: 0.000e+00  invariant
   5120 x 2304: 1.562e-02  M-DEPENDENT      1536 x 5120: 0.000e+00  invariant
   5120 x  384: 1.562e-02  M-DEPENDENT       512 x 5120: 0.000e+00  invariant
   5120 x 1536: 1.562e-02  M-DEPENDENT       128 x 5120: 0.000e+00  invariant
   5120 x  512: 1.562e-02  M-DEPENDENT      2048 x 5120: 0.000e+00  invariant
```

Three things follow from the same probe:

- The drift is exactly one bf16 ULP (`|ref| ~ 3.5`, bf16 eps ~ 0.0137), so it is a
  reduction-order change, not a wrong result.
- The same `M` run twice is bit-identical, so it is the *selection* that depends on `M`,
  not run-to-run nondeterminism.
- `M` thresholds for the `5120x5120` shape sit near 256, 368, 384 and 436, which is the
  bucket structure the engine-level sweep sees after accumulation.

Only `K == 5120` -- the hidden size, i.e. every GEMM that reads the residual stream -- is
affected. The `K = 2304` expert down-projections and the rest are stable, and the MoE
grouped GEMM is CUTLASS with a full-K walk per tile, so it cannot depend on the grid.
MHC pre-norm is `n_splits = 1` for prefill (its split path is gated on DeepGEMM, which is
unavailable on SM80), and the indexer's `_select_prefill_kv_group` returns 1 below
`_KV_GROUP_MIN_N = 16384`.

Neither cheap knob removes it: `preferred_blas_library` is already plain cuBLAS, and
`CUBLAS_WORKSPACE_CONFIG=:16:8` only stabilises `N in {384, 512}`.

So the fix is a GEMM whose reduction order does not depend on `M` for the `K = 5120`
call sites during prefill. Upstream's `VLLM_BATCH_INVARIANT` does not cover these call
sites, and the cost has not been measured yet.

### Impact

Eight realistic 1200-token prompts built from this repo's docs, 128 greedy tokens each,
cold against cached:

```
diverged 8/8    first difference at token 0, 8, 11, 12, 24, 58, 88, 107
```

versus 5/5 byte-identical on the five reference prompts and 3/3 on the vision example, so
it needs a long free-form continuation and a near-tied token. It is still the reason a
cached and an uncached serving of the same long prompt should be expected to differ.

What *is* an invariant, and is asserted by the gate:

- two cold prefills of the same prompt with the same batch shape are bit-identical,
- two runs that both hit the cache are bit-identical,
- the hit length is exact.

## Finding 2 (fixed): a 1..32 token tail block reconciled to zero reuse

With genuinely unique prompts (`cold_hits == 0` proving the cold run was cold), reuse
depended on where the prompt ends inside its last 128-block:

```
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

The store computes `end = aligned // 32 + shift` with `aligned = (num_prompt - 1) // 128
* 128` and `shift = 1` under EAGLE, so the retained tail is `[end - 5, end)`. When the
prompt ends fewer than 33 tokens past the aligned boundary, block `end - 1` is not a full
block of that request -- `num_prompt // 32` is one short -- so the store skips it, only 4
of the 5 blocks exist, and the finder can never reach `need`. Every lookup of that prompt
reconciles to 0 and the whole prompt is re-prefilled.

`VLLM_KV_CACHE_DEBUG=1` shows it directly; `mask=5` is what the mask asked for, and only
four blocks exist:

```
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

```
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
- `vllm/envs.py`: `VLLM_KV_CACHE_DEBUG`.
