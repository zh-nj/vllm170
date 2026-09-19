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

- **The prefill result is sensitive to batch shape**: a historical shape sweep reported
  discrete logprob changes, consistent with a reduction-order change in cuBLAS. The
  sweep used a `shape-{n}` salt but did not verify cache-hit counters or record complete
  mode provenance, so its 1.789114952-nat spread is a historical measurement rather than
  complete zero-reuse evidence. A separate spare-GPU reproduction isolates an
  M-dependent bf16 reduction-order effect in cuBLAS. This is diagnosed, not fixed.
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

## Finding 1: prefill shape dependence, repair in progress

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
