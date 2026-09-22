# vllm170: DSV4.1 bounded decoder SWA replay (fast prefill) — design and evidence

Status: source + CPU contracts landed; device wiring and hardware verification
outstanding. The PP5 launcher enables `additional_config.dsv41_fast_prefill` by
default; set `PORT_FAST_PREFILL=0` to use the unbounded prefill path. The engine
switch itself remains opt-in when constructing `VllmConfig` directly.

Task book: `docs/design/deepseek-dsv41-fastllm-ced-port-guide-2026-09-19.md`.
Baseline: vllm170 `e52bf3dcc4` (+ this work, uncommitted), FastLLM
`cb465fe483aaee2e6d5214ba23f9e60f79ec8cc8` (clean tree), target instance
SM80 / TP1 / PP5 `8,8,8,8,8` / mxFP4-int8 MoE / DSpark width 5.

## What is being ported

FastLLM's approximate "decoder SWA bounded replay", extracted from source at the
pinned revision (not from its docs):

| FastLLM site | Fact |
| --- | --- |
| `src/models/deepseekv41.cpp:1244` | `decoderSwaTailLayer = max(kv_source_layer_ids) + 1`; asserts every later layer has `compress_ratio <= 1`. Enabled by env `FASTLLM_DSV41_DECODER_SWA_BOUNDED_REPLAY`, default off. |
| `:3208` | `boundedPrefill = tail >= 0 && inputEmbeds == nullptr && imageMask == nullptr`; requires `hasLongSegment` (some segment longer than the window); cleared when any segment's spec has `wantAllTokens`/`deferWindow` (verification computes every candidate position). |
| `:3776` | Per segment, `trimBatch` keeps `[end - min(seg.seqlen, window), end)` of **hidden and HC pre-mix**, and trims the per-query **top-k indices** and **candidate blocks** at the same offset. DSpark `mainHidden` is trimmed per target layer and `mainHiddenStartPos = startPos + skip`. `seg.startPos += skip; seg.seqlen -= skip`. |
| `:4130-4146` | For a clipped segment the local attention runs with `windowStart = 0` — "裁剪后边界之前的局部 KV 未计算，不能读取上一 chunk 的旧环形槽" — while "RoPE、indexer 和压缩 KV 的因果筛选已经按绝对位置完成". The window KV pointer is dropped entirely when `windowStart == 0`. |
| `deepseekv41_dspark.cpp:1364` | `DsparkAdvance` re-anchors on `mainHiddenStartPos`; when main hidden rows are missing it restarts accumulation (`filled = 0`), which "只影响接受率". |

So: layers `0..20` compute every token of the chunk; layers `21..39` compute only
the last `min(chunk, window)` rows of each request plus every decode row, at true
absolute positions, and their local (sliding-window) attention starts at the clip
boundary instead of reaching into history those layers never computed.

## Why this is sound in this tree (read, not yet run)

* **Everything the tail reads is produced at layers <= 20.** `kv_source_layer_ids
  = [2, 8, 14, 20]` own the compressors and compressed-KV caches
  (`attention.py:250-298`); the index sources `24/28/32/36` are *non-owning* and
  "share the kv source's paged K cache" (`attention.py:1145`), and
  `DeepseekV4Indexer._produce_k` asserts `owns_k` (`:1272`). The index K is
  `k_norm(wk(latent))` from the source-20 latent, written for every row of the
  chunk by the unclipped layer 20. The indexer's top-k therefore searches a
  complete cache, not a holed one.
* **The PP payloads that cross a boundary are full-chunk.** `hidden_states` and
  `pre_mix` are query-side (they shrink with the clip); `_PP_COMPRESSED_LATENT`,
  `_PP_NORMALIZED_INDEX_K`, `_PP_TOPK_INDICES`, `_PP_CANDIDATE_BLOCKS` are keyed
  by `positions` / `slot_mapping` (`model.py:871-1037`) and stay full width.
  `_pp_outgoing_topk` is False at every boundary that feeds a clipped rank
  (`_index_source_for(next_start) == next_start` for 24 and 32), so no query-side
  buffer crosses into the bounded region; candidate blocks are written by layer 20
  and forwarded verbatim by ranks 3 and 4.
* **The KV import is row-count agnostic.** `_import_pp_attention_state` writes the
  upstream source's latent into this rank's caches from `positions` and
  `compressed_metadata.slot_mapping` (`model.py:957-982`) — full chunk, whatever
  the local query row count is. Ranks 3 and 4 import before any layer runs, so the
  bounded tail cannot affect what lands in the cache.
* **The local window is the only thing the clip invalidates.** Each layer owns its
  SWA cache, written by `_fused_qnorm_rope_kv_insert` over that layer's query rows,
  and read through `prefill_gather_lens` (`rocm.py:838, 894-905`) into
  `combine_topk_swa_indices`. Two changes make the bounded read safe and exact:
  `gather_lens` clamped to the kept rows (so the gather only materialises rows this
  layer wrote) and the per-row window clamp below (so a row whose window would start
  before the block masks those columns instead of addressing anything).
* **Decode rows stay fully covered.** A decode row's window is the last <= 128
  positions, and every chunk's kept set contains its own last `min(chunk, window)`
  rows, so the union always covers that window. Short chunks are never truncated,
  so they keep reading their real history exactly as today.

## Changes so far

* `vllm/models/deepseek_v4_1/fast_prefill.py` (new): `fast_prefill_enabled` scope
  validation (v2 runner, TP1 without DBO, mutually exclusive with `ced_prefill`),
  `tail_layer` (FastLLM's derivation plus the ratio<=1 check),
  `plan_tail_rows` (the per-request keep set, clipped `query_start_loc`, the
  per-prefill gather clamp, decode rows kept in full, batch layout
  `[decodes][prefills]` validated), and the same invariants re-checked in
  `TailRows.__post_init__`.
* `vllm/models/deepseek_v4_1/ampere/prefill_metadata.py`: the combine kernel now
  clamps each row's window to the gathered block —
  `block_rows = query - query_start + gather_len - (query_end - query_start) + 1`.
  Provably a no-op for the unbounded gather
  (`gather_len = chunk + min(prefix, window - 1)`), and exact truncation at the
  clip boundary otherwise.
* `tests/models/test_deepseek_v41_fast_prefill.py` (new, 30 tests, CPU only):
  window boundary 127/128/129, one window per long chunk, short final chunk left
  unbounded, decode rows preserved with a 5-wide draft, request-order permutation,
  the sampled last row always kept, malformed plans and out-of-scope configs
  rejected, plus the mirrored kernel arithmetic proving the clamp is inert when
  unbounded and masks in-block when bounded.

## Outstanding

1. Device wiring: build the bounded metadata for the layers `>= tail_layer`
   (the cloned-group trick CED uses), merge it over the unbounded dict, swap it at
   layer 21, gather/scatter hidden + the four mHC states, and swap the per-layer
   `topk_indices_buffer` / candidate buffer references to gathered copies.

   Facts settled by reading the runner since the first draft of this list:
   * `build_attn_groups` (`attn_utils.py:87-127`) files KV-sharing layers into
     their target's kv cache group, so one `AttentionGroup` spans both sides of
     the tail boundary. The bounded build must therefore clone each group with
     `layer_names` filtered to `extract_layer_index(name) >= tail_layer` and call
     `build_attn_metadata` on the clones — the compressed group is keyed by its
     *source* layer (20), so a name filter alone would drop the entry that
     `forward_mqa` needs for the compressed read (`rocm.py:720-730`).
   * **`requires_eager_prefill` has no consumer anywhere in the tree** — grep finds
     it only in `model_state.py` where it is set and read to gate `ced_step`. The
     task book's warning is real: nothing forces prefill out of graph replay
     today, so a captured graph would freeze the row plan. The hook that does work
     is `skip_compiled` in `model_runner.execute_model` (`:1711-1741`), which feeds
     `need_eager` into `dispatch_cg_and_sync_dp`; `batch_req_state.is_prefilling_np`
     is already in scope there (`:1693-1702`). A bounded step must set it.
   * The bounded batch keeps each request's rows contiguous and in order, so the
     prefill region's row indices stay a valid prefix-indexed view of
     `topk_indices_buffer` / `candidate_block_buffer`: gating those buffers with
     `keep` and swapping the per-layer references reproduces FastLLM's in-sync
     `trimRows(segTopK)` / `trimRows(segCandidate)`.
   * Only `hidden_states` (and the aux states the drafter reads) have to be
     scattered back to full width; `residual`/`post_mix`/`res_mix` are consumed
     inside the bounded span (`model.py:1181-1221`).
2. Request-level fallbacks: multimodal features, `prompt_logprobs` (including 0),
   and verification batches (`num_draft_tokens > 0`) compute the full chunk.
3. DSpark draft context: publish only the bounded rows with their true positions
   and slots (`CEDDraftContext` shape, consumed by
   `speculator._publish_ced_draft_context`), never the scattered placeholders.
4. Launcher default `PORT_FAST_PREFILL=1`, with an explicit opt-out. Prefix
   caching is an independent switch and may be enabled at the same time.
5. Hardware: per-layer row counts proving `N>128 -> layers 0-20 N rows, 21-39 128
   rows` (the in-tree row probe already dumps per-layer hidden with positions), A/B
   TTFT/prefill throughput at 4K/32K/64K, peak memory, DSpark acceptance, and the
   reference prompt set. Not run yet — the PP5 instance is currently down
   (port 26663 health 000, GPUs idle), so a hardware window has to be scheduled.

This is an approximation. Token-identical output with the switch off is not
claimed; the switch-off path is unchanged by construction.

## Hardware findings (switch-on run, budget 2048, prefix caching off)

**1. Ranks below the boundary have an empty bounded set, and that is not an error.**
The first switch-on run died on its first request:

```text
ValueError: dsv41_fast_prefill found no attention layers at or after layer 21 on this rank
```

PP5 `8,8,8,8,8` puts layers 0-7 on rank 0 and 8-15 on rank 1, so neither holds a
layer at or after the boundary and there is nothing for them to bound. The guard
that was supposed to catch an incorrectly scoped rank raised instead, inside
`prepare_attn`, which hung the engine core (the request never completed and the
process stopped consuming the shared-memory broadcast). Fixed by asking the
question the guard meant to ask -- `_has_bounded_layers(groups)`, true when any
layer this rank holds is at or after `tail_layer` -- and returning no step when it
is false, so the chunk flows through those ranks unchanged, exactly as the
unbounded path does. Regression test:
`test_a_rank_below_the_boundary_has_nothing_to_bound`.

The general lesson for the remaining ranks: rank 2 (layers 16-23) straddles the
boundary and must build the bounded metadata; ranks 3 and 4 (24-31, 32-39) are
entirely bounded. Only 0 and 1 do nothing.

**2. The enable log needs a restart, not an edit.** The line
`logger.info_once("dsv41_fast_prefill: approximate bounded decoder SWA replay,
boundary layer %d, window %d")` was added to `FastPrefillMetadata.__init__`
after the workers of that run had already imported the module, so it did not fire
in that run; it fires on the next start. The switch's effect in that run is
evidenced by the per-layer row table instead, which is the stronger claim anyway.

**3. Unrelated pre-existing breakage, left alone.** `tests/v1/spec_decode/
test_dspark_pp.py` imports `_collect_q1_drafter_audit_records` and
`_get_q1_drafter_audit_eligible_rows` from the DSpark speculator; neither exists
there at `HEAD` (`git show HEAD:...` counts zero), so the module fails to collect
before any test runs. Both files match `HEAD` locally, so this predates the port
and belongs to whichever workstream owns that test.

**4. An out-of-scope edit from a delegated implementation was reverted.** The
implementer also added a distributed-DSpark receive-buffer reuse path
(`_remember_distributed_recv_buffers` / `_recv_distributed_package`, gated on
`model_type == "deepseek_v41"`, unrelated to this switch) plus a unit test for it.
That belongs to the DSpark flow workstream and would confound the A/B, so it was
saved to `artifacts/dspark_flow_20260920/out_of_scope/recv_buffer_reuse.patch`
and reverted.

**5. Open: the bounded path faults on a fully bounded rank for wide chunks.**
With the boundary guard fixed, the switch-on run starts and logs its banner --

```text
(Worker_PP0) INFO [fast_prefill.py:306] dsv41_fast_prefill: approximate bounded
decoder SWA replay, boundary layer 21, window 128
```

-- and short requests complete (the 1500-token smoke gate exits 0, `200 OK`s are
served), but wide prefills fail with

```text
File "vllm/models/deepseek_v4_1/amd/rocm.py", line 886, in _forward_prefill
    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
torch.AcceleratorError: CUDA error: invalid argument
```

on PP3, twice, in `self.model(**model_inputs)`. The 32K row-evidence request died
the same way: `prefill_gate.py` crashed computing its median (`int / None`), and
the recorder wrote no table at all, so no per-layer rows were captured.

CUDA errors surface at the first later launch, so `rocm.py:886` is the reporter,
not necessarily the fault: it is the first CUDA op after the context is poisoned.
The leading hypothesis, and the one gap between the ledger's wiring list and what
the implementation actually did, is that **the per-query buffers the tail reads
were never gathered**: `topk_indices_buffer` and `candidate_block_buffer` are
indexed by prefill-region row, and the bounded batch renumbers rows, so the tail
currently consumes the top-k/candidate entries of the *original* first R rows
(or of an earlier step's different layout) instead of the kept rows. FastLLM
gathers exactly these two buffers at the same offset as hidden and HC pre-mix
(`trimRows(segTopK)` / `trimRows(segCandidate)`, `deepseekv41.cpp:3790-3800`).
Next step is to gate both buffers with `keep` and swap the per-layer references
(`attn.topk_indices_buffer`, `attn.indexer.topk_indices_buffer`,
`attn.indexer.indexer_op.topk_indices_buffer`, and the candidate buffer), then
re-run the row evidence. A shape/tensor dump on the first bounded step would
confirm it in one request if the hypothesis is wrong.

The A baseline (`artifacts/fast_prefill_20260919/A/`) is unaffected: it was
recorded with the switch off, where none of this code runs.

**6. The buffer gather landed; the CUDA fault is still there and is not the gather.**
`FastPrefillStep.bind_tail_buffers` now gathers `topk_indices_buffer` and
`candidate_block_buffer` with `keep` and swaps the attributes of the bounded
layers' attention / indexer / indexer-op (identity-checked, restored in the
model's `finally`), with `test_binding_points_bounded_layers_at_gathered_rows_and_restores`
covering it. That was a real gap and is still required for correctness, but the
switch-on run faults identically: `torch.AcceleratorError: CUDA error: invalid
argument`, reported at `rocm.py:886`, on PP3, for wide chunks.

`rocm.py:886` is a division of a one-element int32 tensor, which cannot itself be
an invalid launch, so the error is sticky and belongs to an earlier launch on that
rank. Two facts narrow it: in `_sparse_indexer_and_attn` the indexer op runs
*before* `forward_mqa` (`attention.py:888` then `:904`), and every later step in
the same process then fails with `RuntimeError: Cannot start a receive while PP
receive buffer is model; expected free` (`pp_utils.py:129`) -- the failed forward
never released the PP receive buffer, which is why the row tables stop after ranks
0-2. Rank 4 holds nothing but bounded layers, so the same path is presumably
identical there.

Next step, already running: restart with `CUDA_LAUNCH_BLOCKING=1`, which turns the
asynchronous fault into a traceback at the real kernel, and send one 32K request.
The prime suspect is the indexer op with the bounded metadata (it consumes
`dense_mha_metadata_layer_name`'s slot mapping / seq_lens / block table, plus the
two gathered buffers); a synchronize between the indexer op and `forward_mqa`
would localise it without the env var if the workers turn out not to inherit it.

**7. Root cause of the CUDA fault: the bounded dict dropped the source layer's key.**
`CUDA_LAUNCH_BLOCKING=1` did not reach the workers (`vllm serve` does not pass it
on; the traceback is byte-identical with and without it), so the fault had to be
found by reading. It is in the metadata keys, not in the buffers:

* A non-owning index source **shares the kv source's paged K cache**
  (`attention.py:1145`), so the indexer looks its metadata up as
  `attn_metadata[self.k_cache.prefix]` -- a key that belongs to the **source
  layer below the boundary** (20), not to the layer running it (24/28/32/36).
* `_bounded_groups` filtered each group's `layer_names` to `>= tail_layer`, which
  removed exactly that key. The merged dict `{**original, **bounded}` then kept
  the *unbounded* entry under it, so the tail's indexer read metadata describing
  **2044 query rows** while its activations held **128** -- an invalid launch, and
  every later step in that process then failed on the unreleased PP receive
  buffer.

The fix keeps the full `layer_names` of any group that a bounded layer reads, and
skips only the groups no bounded layer touches, so the shared cache's key is
present in the bounded dict and describes the bounded batch. Covered by
`test_a_consumed_group_keeps_its_source_key_below_the_boundary`; the suite is 33
tests. This also explains why only wide chunks died: the mismatch is between
`num_prefill_tokens` from the metadata and the actual row count, which only
differs when the chunk is longer than the window.

`fast_prefill_rows.py` now reports a step that stops early as
`incomplete step, layers [...] never logged`, instead of misreading a crashed
forward as a boundary violation.

**8. Root cause, measured: the bounded dict kept the *unbounded* compressed entry.**
The shape probe (gated on `/tmp/dsv41_fastprefill_shapes_on`, writing
`/tmp/dsv41_fastprefill_shapes.rank*.tsv` and then synchronizing) captured it on
rank 2 with one 32K request:

```text
layers.16.attn  q=2044 ... npref_tok=2044  gather_len=1:2171 seq=1:6132 slots=2044 mpref_tok=2044 ratio=2
layers.23.attn  q=128  ... npref_tok=128   gather_len=1:128  seq=1:4088 slots=128  mpref_tok=2044 ratio=1
```

The SWA metadata for the bounded layer is exactly right (`q`, `npref_tok`,
`gather_len`, `slots` all 128), but the *compressed* metadata still describes
**2044** query rows. Layer 23's attention reads that entry through
`compressed_cache_prefix` and its indexer reads it through
`attn_metadata[self.k_cache.prefix]` -- the shared cache keyed by the kv source
layer 20 -- and the indexer op sizes its grid from the metadata while holding 128
rows of `q`. That is the invalid launch, and it is why the fault needed a chunk
longer than the window.

Finding 7's fix was insufficient: filtering *which groups* are cloned is not
enough when the group that owns the shared cache lists the source layer's name
only, so that group was skipped and the merge left the unbounded entry in place.
The fix is to stop filtering altogether: `_build` passes the caller's groups
through unfiltered (keeping all layer names), so every key a bounded layer can
read describes the bounded batch. The extra metadata builds cost one pass per
group per step.

`test_the_bounded_build_keeps_every_group_key` pins it: the build receives the
caller's groups object and the resulting dict still carries the source key below
the boundary. The suite is 33 tests. The probe stays in `rocm.py` for the next
investigation and is inert unless its trigger file exists.

**9. Verified on hardware: the bound holds, all five ranks, and the prefill is faster.**
After finding 8's fix, one 32K cold request at budget 2048 with the switch on:

* **0 tracebacks** in the whole run (previously 14, every step after the first).
* 32K TTFT **5.387 s / 6083 tok/s** against the switch-off baseline's 6.962 s /
  4707 tok/s -- **1.29x** on the widest measured prefill, and the first B-side
  number of the A/B.
* Bounded-rank bookkeeping is consistent: every rank logs
  `q == npref_tok == mpref_tok` (64 on the final short chunk), i.e. the SWA *and*
  compressed entries now describe the same rows as the activations.

Per-layer effective rows, from `record_layer_rows` (the evidence the task book
asks for), one 32K request:

| rank | layers | rows per 2044-row chunk | rows on the 64-row tail chunk |
| --- | --- | --- | --- |
| 0 | 0-7 | 2044 | 64 |
| 1 | 8-15 | 2044 | 64 |
| 2 | 16-20 / 21-23 | 2044 / **128** | 64 / 64 |
| 3 | 24-31 | **128** | 64 |
| 4 | 32-39 | **128** | 64 |

`fast_prefill_rows.py` reports `PASS: one row count below the tail, one smaller
above it, every rank`. The 64-row tail chunk is not truncated (it is shorter than
the window), so both sides agree there -- which is also what FastLLM does.

**10. 2026-09-20 takeover: the remaining source-metadata mismatch is fixed and
verified on the PP5 service.**

The previous section's group-key repair still left one case uncovered. A
compressed/indexer attention group can contain only the KV source layer (20),
while the bounded consumer is in the tail (21--39). Filtering groups by the
tail layer therefore skipped that source-only group and the merged metadata
dict retained its unbounded entry. The first switch-on run showed the exact
symptom: tail activations had `q=128`, SWA metadata had `npref_tok=128`, but
the compressed/indexer metadata still had `mpref_tok=2044`; the asynchronous
failure surfaced as `CUDA error: invalid argument` and left the PP receive
buffer busy.

The current implementation in `vllm/models/deepseek_v4_1/fast_prefill.py`
rebuilds every attention group with the bounded row plan and preserves every
layer name. That keeps both the source-layer key and the tail consumer key in
the bounded metadata. The model-side changes also keep CED and fast prefill
independent, export the original full-width PP positions after the compact
tail has run, and use the corrected metadata row count in the ROCm shape probe.
The fast-prefill unit suite is now **33 passed**.
The CED suite's affected state tests are **30 passed** when excluding four
older Engram fixture cases that construct a decoder layer without its existing
`prefix` field; those four failures are independent of this transplant.

The running PP5 service was launched by the existing patient launcher with
prefix caching disabled. Direct hardware checks returned HTTP 200 for both
requests:

* 2044 tokens: `/tmp/dsv41_fp_final_2044.json`, TTFT **1.104 s** and **1852
  tok/s**. The row probe shows PP2 layers 16--20 at 2044 rows and layers
  21--23 at 128; PP3 and PP4 also run 128 rows.
* 32768 tokens: `/tmp/dsv41_fp_final_32768.json`, TTFT **8.374 s** and
  **3913 tok/s**. This exercises multiple prefill chunks.

The shape probe files `/tmp/dsv41_fastprefill_shapes.rank*.tsv` show, on every
bounded layer, `q=128` and `mpref_tok=128`, while source layer 20 remains at
the full 2044-row width. On the clean row-table snapshot for these two
requests, `fast_prefill_rows.py --tail 21 --min-chunk 129` reports `PASS`.
Later DSpark verification batches appended to the same tables can be
full-width by design, so the checker must be run on a cleared snapshot for a
fast-prefill assertion. `/tmp/dsv41_pp5_final_fix.log` contains no traceback,
`invalid argument`, or PP receive-buffer error for these checks.

This establishes the DSV41F fast-prefill/CED-compatible execution path on the
current PP5 hardware. It is still the documented approximate replay path: no
accuracy comparison or DSpark end-to-end A/B claim is made here, and the
DSV4F path is untouched. Keep the current service alive for the DeepSeek
handoff; do not start a competing instance on port 26663.
