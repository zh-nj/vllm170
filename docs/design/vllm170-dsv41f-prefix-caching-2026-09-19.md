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

Two behaviours showed up while verifying it, and both are documented below because
they are what a "did prefix caching change my outputs?" report will look like:

- **Prefill batch shape perturbs the logits** (~0.1 nat), which flips near-tied greedy
  tokens and can make a cached and an uncached completion diverge. This is a property
  of the model's kernels, not of the cache (proved below with zero reuse involved).
- **A prompt that ends 1..32 tokens past a 128-block boundary reconciles to zero
  reuse**, even though its blocks are cached. Deterministic and reproducible; it is in
  the upstream scheduler/coordinator, which this port does not touch.

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

## Finding 1: prefill batch shape perturbs the logits, and greedy decoding amplifies it

Cold and warm runs of the same prompt can produce different completions. It is not the
cache. Proof, with no reuse in the picture at all:

`ids[0:1500]` is a prefix of `ids[0:1628]`, so the token at index 1499 has a
byte-identical context in both. Chunked prefill computes it in a 476-token batch for
the first prompt and a 604-token batch for the second. Its own prompt logprob differs:

```
index 1499: -0.5303 (476-token chunk)  vs  -0.4345 (604-token chunk)
index 1300: -10.7913 (rank 8)          vs  -10.6855 (rank 9)
index 1100: -0.00417                   vs  -0.00701
```

The same magnitude (~0.1 nat) appears as the cold/warm difference at the last prompt
position, e.g.

```
cold top5 {'e': -1.678, 'k': -1.803, ...}
warm top5 {'k': -1.257, 'e': -1.882, ...}   # top-1 flips; the two are 0.125 apart cold
```

So: a cache hit necessarily recomputes its tail in a much smaller batch than the cold
prefill did (a hit of `n - (n mod 128)` leaves at most 129 tokens to compute), that
batch-shape change moves the logits by a few tenths of a nat, and on a near-tied token
greedy decoding flips, after which the two continuations diverge chaotically. This is
the same class of change as editing `max_num_batched_tokens`, and it is why the gate
does not assert byte-identical cold/warm output.

What *is* an invariant, and is asserted:

- two cold prefills of the same prompt with the same batch shape are bit-identical
  (they are given distinct `cache_salt` values so they cannot share blocks),
- two runs that both hit the cache are bit-identical,
- the hit length is exact.

Consequences worth knowing: on peaked text (the five reference prompts, the vision
example) cached and uncached runs agree byte for byte; on long free-form answers over
repetitive prompts they can diverge at a near-tied token. A user comparing cached and
uncached servings of the same long prompt should expect occasional divergence, and
should not read it as cache corruption.

## Finding 2: a 1..32 token tail block reconciles to zero reuse

With genuinely unique prompts (fresh random ids, `cold_hits == 0` proving the cold run
was cold), reuse depends on where the prompt ends inside its last full 128-block:

```
tail   0 : 1280 reused
tail 1-32: 0 reused          <- no reuse at all
tail  33+: 1408 reused
```

It is not a race (a 3 s gap changes nothing), not content (six different contents at
the same length behave identically), and not a failure to cache: a 1408-token *prefix*
of a failing prompt hits 1280 immediately afterwards, and a 1424-token prompt whose
prefix was cached by a separate 1408-token request hits 1280. Only the request's own
blocks are unreachable to itself.

The 32/33 boundary points at the SWA group, whose `block_size` is 32 while the
compressed/indexer groups use 128 and the scheduler block (the hit alignment) is 128.
Confirming the mechanism needs scheduler-side tracing; it lives in
`vllm/v1/core/` (untouched by this port -- `git status vllm/v1/core/` is clean), so it
is an upstream issue rather than a port regression.

Practical impact: prompt lengths that land in that band pay a full prefill on every
request. `prefix_cache_gate.py` pins the observation as a case with
`expected_hit_tokens=0` so a future fix flips it visibly instead of silently.

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
