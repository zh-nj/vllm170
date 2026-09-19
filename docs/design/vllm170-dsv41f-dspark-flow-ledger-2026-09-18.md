# DeepSeek-V4.1-Flash DSpark flow ledger (PP5, 2x SM80)

Date: 2026-09-18. Scope: measure every stage of the DSpark flow on the PP5 port
instance, decide the hotspot, and optimize it.

Hardware: 5x CMP 170HX (SM80, 65,024 MiB each) on PCI 2-6, one PCIe switch, one
NUMA node. `NCCL_P2P_DISABLE=1` is mandatory on these cards (no P2P even inside
one switch).

Instance: PP5 / TP1, `mxfp4_int8_sm80` MoE, `TRITON_MLA_SPARSE_DSV41` attention,
FP8 KV, FULL target CUDA graphs, DSpark width 5 (`dspark_pipeline_stage_ranks`
`[1,2,3]` = draft blocks on PP1-PP3, head on PP4), greedy draft sampling.

## 1. The measurement was wrong before it was right

`benchmarks/deepseek_v4_1/decode_gate.py` divides the streaming span by the
number of SSE **chunks**. Without drafting the server emits one token per chunk,
so chunk == token and the gate is exact (that is why the 42.04 tok/s baseline in
`vllm170-dsv41f-decode-ledger-2026-09-18.md` is sound). With DSpark the server
coalesces every token of a step into one chunk, so the gate reported the **step
time as if it were the token time**:

| Configuration | chunk gate | token-accurate gate |
|---|---:|---:|
| DSpark off, 32K | 23.80 ms/step = 42.04 tok/s | identical (1.00 token/chunk) |
| DSpark on, 32K, q5 | 99.7 ms/chunk | 18.11 ms/token = **55.23 tok/s** |

`benchmarks/deepseek_v4_1/dspark_stream_gate.py` divides by
`usage.completion_tokens` instead and reports the chunk interval separately,
which is the step time when drafting is on. Cross-check against the engine's own
counter (`vllm:spec_decode_num_accepted_tokens_total` sampled at 1 Hz): 45.4
accepted tokens/s sustained, and accepted/draft = 4.565 of 5.565 emitted per
step, so 45.4 x 5.565/4.565 = 55.4 tok/s — the same number.

## 2. Stage ledger (32K prompt, q5, acceptance length 5.565/6, ms per step)

Measured with the CUDA-event phase profiler; the distributed draft path had no
instrumentation at all before this round (see §5).

| Stage | Rank | Mean |
|---|---|---:|
| `target.model_graph_replay` | PP0/1/2/3/4 | 6.94 / 7.02 / 7.90 / 8.39 / 7.96 |
| target model serial across PP | - | **35.1** |
| `target.attn_metadata` (per rank, overlaps) | PP0..PP4 | 9.5 - 17.0 |
| `target.sampling_total` (logits + reject + sample) | PP4 | 1.2 |
| **`draft.propose_total` (draft chain, head rank)** | PP4 | **86.0** |
| `draft.stage.forward` (one eager decoder layer, 5 query tokens) | PP1/2/3 | 13.15 / 13.42 / 13.33 |
| `draft.stage.prepare` (metadata + context KV + inputs) | PP1/2/3 | 6.81 / 6.73 / 6.63 |
| - `draft.stage.metadata` | PP1/2/3 | 4.36 / 4.35 / 4.27 |
| - `draft.context_kv` | PP1/2/3 | 1.49 / 1.44 / 1.42 |
| - `draft.stage.prepare_inputs` | PP1/2/3 | 0.38 / 0.36 / 0.36 |
| `draft.stage.send` (PP relay, one isend per tensor) | PP1/2/3 | 3.46 / 3.55 / 2.24 |
| `draft.head.recv_wait` (head waits for the last stage) | PP4 | 74.0 |
| `draft.markov_total` (5 sequential Markov steps) | PP4 | 6.31 |
| `draft.head.send` / `finish` / `context_hidden` / `result_copy` | PP4 | 2.11 / 0.84 / 1.16 / 0.30 |
| **measured step (unprofiled gate)** | - | **104.5** |

The stage ranks also report `draft.stage.recv_wait` of 249 / 189 / 124 ms. Those
are **pipeline skew, not cost**: a stage rank enters its receive as soon as its
own target share is done, and the head rank only sends after the whole target
pipeline (and its own sampling) finished. Only the head rank's 86 ms is on the
critical path.

## 3. Hotspot

**The distributed draft chain is 86 ms of a 104.5 ms step (82%).** The target
verification under CUDA graphs costs 35 ms. Inside the draft chain the ranking is
eager stage forwards (3 x 13.2 = 40 ms), per-stage metadata rebuild + context KV
(3 x 6.7 = 20 ms), relay sends (9 ms), Markov head (6.3 ms).

Every stage rank runs its draft block **eager**: the distributed path sets
`query_cudagraph_manager = None` in `init_cudagraph_manager` ("initial
validation"), and the launcher passed `enforce_eager: true`. A target layer that
costs 0.55 ms in a captured graph costs 10-13 ms eager, which is the whole gap.

## 4. The 20x pitfall: the verify batch needs its own captured size

The target's verify batch is `num_reqs x (num_spec + 1)` tokens, i.e. 6 for one
request at width 5. Capture sizes are rounded up to multiples of the manager's
`decode_query_len`, so a capture list that does not cover 6 makes the **entire**
verification step fall back to eager:

| `PORT_CUDAGRAPH_SIZES` | target verify | step |
|---|---|---:|
| `[8]` (6 not covered) | eager, 65.9-85.4 ms per rank, 10.3 ms per layer | 2096 ms |
| `[6,8,12,...]` | graph replay, 6.9-8.4 ms per rank | 571 ms |
| `[1,2,4,6,8,...]` (engine default) | graph replay | 592 ms |

Same for the draft manager, whose `decode_query_len` is `num_query_per_req` (5):
its graphs come from sizes rounded up to multiples of 5, so a list without
1/2/4 gives the draft no 5-token graph at all. And plain single-token decode
needs size 1 in the list: `[6,8,12,...]` measured 46.01 ms/step (21.73 tok/s)
against 24.12 ms/step (41.72 tok/s) for `[1,2,4,6,8,...]` in an otherwise
identical DSpark-off run.


## 5. Instrumentation added this round

The PP5 distributed DSpark path produced almost no phase records before:
`DSparkSpeculator.propose` overrides the decorated `DFlashSpeculator.propose`
without the decorator, `participate_distributed` was not decorated, the V4.1
draft model (`vllm/models/deepseek_v4_1/nvidia/dspark.py`) had no
`phase_profile_stage` calls, and the input-prepare/metadata section of
`_prepare_distributed_stage` was unwrapped.

Added: `@profile_parallel_draft_step` on `_propose_distributed` and
`participate_distributed`; stages `draft.stage.recv_wait`, `draft.stage.prepare`,
`draft.stage.prepare_inputs`, `draft.stage.metadata`, `draft.context_kv`,
`draft.stage.forward`, `draft.stage.send`, `draft.head.context_hidden`,
`draft.head.send`, `draft.head.recv_wait`, `draft.head.result_copy`,
`draft.head.finish`, `draft.head.reconstruct`, `draft.markov_total`; and the
model-side `draft.hidden_combine.main_proj/main_norm`,
`draft.context_kv.input_gemm/norm/insert`, `draft.logits.norm/lm_head`,
`draft.markov.embed/projection`, `draft.stage.embedding`.

## 6. Startup: the warmup OOM and what it costs

The mixed prefill+decode warmup issues the first collective on the PP
token-broadcast subgroup. NCCL allocates that communicator's device buffers on
first use, and the largest rank (PP1: 57.5 GiB of weights plus an Engram layer)
is down to a few hundred MiB at that moment, so startup dies with
`ncclUnhandledCudaError ... Cuda failure 2 'out of memory'`. It is marginal: the
same configuration started on the first attempt at 15:52 and failed ten times in
a row later the same afternoon.

Trimming NCCL's per-rank allocation fixes it. With DSpark on, only the 1 MiB
buffer setting has started reliably: the default (4 MiB) OOMs in warmup, and
2 MiB starts but dies on the first real serving step. The trim itself is free
for decode — measured DSpark off at 32K with the token-accurate gate, same
instance, same capture sizes, only `NCCL_BUFFSIZE` differing:

| NCCL setting | decode |
|---|---:|
| default 4 MiB, `NCCL_MAX_NCHANNELS=8` | 24.68 ms/step = 40.54 tok/s |
| 1 MiB, `NCCL_MAX_NCHANNELS=8` | **24.12 ms/step = 41.72 tok/s** |
| default buffers, round-5 config (2048 batched, 1M context) | 23.80 ms/step = 42.04 tok/s |

An earlier 46.01 ms/step (21.73 tok/s) reading in a 1 MiB run was **not** an NCCL
penalty: that run also captured `[6,8,12,...]`, which has no size 1, so plain
single-token decode fell off the captured graph (§4). The lesson generalises:
a capture-size list that misses the batch you actually run costs far more than
any transport knob.

`start_pp5_patient.sh` is the launcher that rides the startup flake out: it waits
for the device memory of a failed attempt to be released (not just host memory),
keeps each failed attempt's log, and does not kill a slow attempt at the 10
minute mark (a cold page cache needs ~20 min to load 48 shards).


## 7. Optimization: rank-local draft CUDA graphs

The hotspot of §3 is that every draft stage runs eager. Capturing the rank-local
draft step into FULL CUDA graphs fixes it. The port follows
`backup/wip-dspark-graph-20260917` (a validated WIP that the dsv41f port dropped):

- `DSparkSpeculator.init_cudagraph_manager` builds a `DFlashCudaGraphManager`
  with `decode_query_len = num_query_per_req` on the ranks that own a draft
  block or the head, instead of pinning `query_cudagraph_manager = None`.
- `capture()` captures `_draft_stage_forward` (the rank's draft blocks) on PP1-3
  and `_draft_head_forward` (state collapse + sequential Markov head) on PP4.
- A FULL graph can only read fixed addresses, so the relayed draft state moves
  through `_DSparkStageStaging` buffers: copied in before the replay, read out
  after it. The PP relay and `precompute_and_store_context_kv` stay eager.
- Knob: `PORT_DSPARK_EAGER` (0 = graphs, now the launcher default; 1 = the old
  eager path). `DFlashCudaGraphManager.capture` grew a `skip_attn` parameter so
  the head rank can capture without attention metadata.

Measured at 32K, q5, same instance, only this knob changed:

| Stage | eager draft (A) | draft graphs (B) |
|---|---:|---:|
| `draft.stage.forward` (per stage) | 13.15 / 13.42 / 13.33 ms | **0.58 / 0.59 / 0.55** |
| `draft.stage.prepare` (per stage) | 6.81 / 6.73 / 6.63 | 6.95 / 6.55 / 6.50 |
| `draft.stage.send` (per stage) | 3.46 / 3.55 / 2.24 | 3.32 / 3.14 / 1.98 |
| `draft.head.recv_wait` | 74.0 | 33.3 |
| `draft.markov_total` | 6.31 | **1.30** |
| `draft.propose_total` (head, whole chain) | **86.0** | **38.7** |
| measured dependency scope | 148.6 | 96.7 |
| **step (unprofiled)** | **104.5** | **76.9** |
| **throughput (unprofiled)** | **55.23 tok/s** | **75.19 tok/s** |

Correctness: 5/5 reference prompts return **byte-identical token ids** to
target-only greedy decoding (`dspark_output_dump.py` compared against
`ref_nodspark.json`). Greedy speculative verification is exact, so any staging or
capture bug would have shown up as a divergence; none did. Acceptance is
unchanged (5.565 tokens per step in both arms).

Note on reading the profiled columns: with the phase profiler armed, each record
ends in a synchronization, which serializes the PP pipeline and inflates the step
(76.9 ms -> ~235 ms per chunk step). The profiled numbers are for attribution
only; every timing claim above is from the unprofiled gate.

## 8. Status and open items

- DSpark flow fully instrumented; the ledger is reproducible with
  `phase_profile_analyze.py` over `logs/dspark_prof/`.
- At 32K, q5: **75.2 tok/s with drafting versus 40.5-42.0 tok/s without**, i.e.
  1.8x, with 5.565 of 6 tokens accepted and bit-identical greedy token ids.
- Remaining hotspots after the draft-graph change, in step order: the target
  verify graph replay (35.1 ms serial, now ~45% of the step), the per-stage
  `draft.stage.metadata` rebuild (3 x 4.3 ms, of which 2.6 ms is the SWA
  non-causal metadata builder), the relay hops (head `recv_wait` 33.3 ms plus
  8.4 ms of sends, one isend per tensor), `draft.context_kv` (3 x 1.4 ms), and
  the target's `attn_metadata` builds (9.5-16.5 ms per rank, partially
  overlapped). The draft metadata build cannot simply be skipped in the graph
  path: the captured graph reads the tensors that builder refreshes, and
  `attn_metadata`/`slot_mappings` are otherwise unused on that branch.
- Startup memory is still marginal with drafting on: see §6 for the NCCL trim,
  and note that the trim is not what makes any decode number slow.

## 9. Real per-step attribution (no profiler)

The phase profiler synchronizes once per record, which serializes the PP pipeline
(76.9 ms -> ~257 ms per step), so it can attribute but not time. Added
`VLLM_DEEPSEEK_V4_STEP_TIMING_EVERY=N`: the runner's own `step_timing` event
pairs (`forward_start/end` around the target, `drafter_start/end` around the
proposal) are now also closed on the real serving path, resolved every N steps
and logged as percentiles. One sync per N steps, so it runs during serving.

32K, q5, 74.8 ms/step, 5.69 tokens/step:

| Rank | prep (metadata + inputs + PP recv) | target (verify slice) | draft |
|---|---:|---:|---:|
| PP0 | 5-6 ms | 6.88 ms | - |
| PP1 | 5-6 | 6.97 | - |
| PP2 | 5-6 | 8.20 - 9.27 | - |
| PP3 | 5-6 | 8.64 - 10.10 | - |
| PP4 (head) | 5.65 - 6.54 | 8.12 - 10.02 | 17.2 - 21.4 |
| **period** | | | **74.9 - 79.1 ms** |

The head rank's own scopes cover ~34 ms of its 75-79 ms period; the remainder is
waiting on upstream ranks plus the engine loop. `prep` was previously invisible
(it sits outside the forward event pair); the period measured in-process agrees
with the client-side chunk interval (76 ms), which calibrates the instrument.

`prep` splits almost entirely into the target attention metadata build:

| rank | prep | of which attn_meta | of which inputs + PP recv |
|---|---:|---:|---:|
| PP0 | 3.41 ms | **3.23** | 0.10 |
| PP4 (head) | 6.05 - 6.32 | **5.59 - 5.64** | 0.25 - 0.65 |

Inside it, the phase profile shows the build running **four**
`DeepseekV4ROCMAiterSparseSWAMetadataBuilder` calls per step per rank (one per
SWA KV-cache group) plus one Indexer, one SparseMLA and (on PP0) one Compressor
build. Per call, the named sub-stages total only ~1.0 ms while the builder's own
span is 1.6-2.9 ms, and GPU-event time equals CPU time for every sub-stage --
i.e. the gap is host-side Python inside the builder, not kernel work and not a
hidden synchronization. At ~1.4 ms real per group build, four groups per step is
the largest remaining item that is not bound by the MoE or by the PP pipeline.


### Why the target verify has no further knob

The verify runs 6 tokens instead of 1, and the MoE dominates its layers. Measured
with `deepseek_v41_moe_bandwidth_probe.py` (128 experts, top-3, one rank):

| tokens | expert traffic | W13 | W2 | total | effective |
|---:|---:|---:|---:|---:|---:|
| 1 | 54.0 MB | 66.8 us | 31.2 us | 108.1 us | 500 GB/s |
| 6 | 324.2 MB | 280.1 us | 144.5 us | 434.2 us | 747 GB/s |

The byte traffic scales exactly with the token count, because each token of a
small batch activates 3 *distinct* experts: 6 tokens touch ~18 expert instances
instead of 3. That is the speculation tax, and it is the routing model rather
than a kernel defect. Against those same bytes streamed at 1.68 TB/s the MoE
runs at 747 GB/s, i.e. 44% -- the MXFP4->INT8 emulation share that the decode
ledger already established as this hardware's floor (SM80 has no native FP4
tensor core). Both terms are inherent: more tokens cost more expert traffic by
construction, and each byte costs ~2.2x a plain stream.

### Remaining "fullgraph" gap

Captured today: target forward (per rank), draft stage blocks (PP1-3), draft head
collapse + Markov head (PP4), all FULL graphs. Still eager: the PP relay, the
draft's `precompute_and_store_context_kv`, the SWA attention metadata build, and
target sampling.

The context-KV insert was attempted and **reverted**. It is not capturable as
things stand: its row count is the step's verify-token count, which is *not* a
function of the captured descriptor. The descriptor keys on the draft's query
count (`num_reqs * num_query_per_req`, 5 per request); the verify block is 6 per
request when serving but **7 per request** in the warmup batch
(`warmup.py::_run_decode_step` schedules `decode_query_len` per request, and the
capture-time assert fired with "56 context rows for 8 requests"). A captured
graph therefore cannot know how many context rows to insert for the step it is
replaying, and copying a prefix would misalign per-request rows. Folding it in
needs either a pipeline descriptor that carries the verify count or padding with
neutralized (-1) slot mappings; neither is worth the ~1-3 ms it would save. The
metadata build cannot be captured at all: it is CPU work that refreshes the very
tensors the graph reads.

### What is left

- Target verify 41 ms/step: MoE traffic model plus the SM80 MXFP4 emulation
  floor. A format with native SM80 tensor-core support is the only lever, as the
  decode ledger concluded.
- Draft chain 20 ms/step: a data-dependent serial chain (inputs -> metadata ->
  context KV -> replay -> send, four hops) that starts from this step's sampling
  result, so it cannot overlap the verify. Every item is 1-3 ms.
- `prep` 3.4-6.3 ms per rank, ~95% of it the target attention metadata build
  (§9): four SWA group builds per step per rank at ~1.4 ms each, host-bound.
  This is shared with non-drafting decode, so it is not a DSpark hotspot, but it
  is the largest item that is neither MoE-bound nor pipeline-bound.
- Engine loop and PP broadcast: the head rank's period is 75-79 ms while its own
  scopes cover ~34 ms (target 8-10 + draft 17-21 + prep 6); the rest is waiting
  on upstream ranks plus the engine loop.
- Capturing target sampling was tried in an earlier PP3 study on this project and
  rejected (target logits/rejection in graph: -0.26% end to end), so the remaining
  eager pieces are not free wins either.






