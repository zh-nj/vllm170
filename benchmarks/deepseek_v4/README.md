# DeepSeek V4 on three CMP170HX GPUs

This directory carries the launch, performance, and correctness harnesses
selectively absorbed from `deepseek-v4-cmp170hx` at commit `3dd2d88`. The
source and this project are Apache-2.0 licensed. Endpoint defaults were adapted
for this checkout; set `DSV4_BENCH_URL` and `DSV4_BENCH_MODEL` to override
them.

## Launch

```bash
benchmarks/deepseek_v4/run_cmp170hx_pp3.sh
```

For `DeepSeek-V4-Flash-Vision-Exp`, use the PP3 vision preset:

```bash
benchmarks/deepseek_v4/run_cmp170hx_pp3_vision.sh
```

It selects the local vision checkpoint when present, serves it as
`dsv4-vision`, and defaults DSpark to five speculative tokens because the
checkpoint declares `dspark_block_size=5` while exposing three reusable target
layers. The underlying PP3/SM80
settings remain identical to the text launcher, including physical GPUs
`1,2,3`, with the vision preset using partition `15,16,12`, FP8 KV cache, and
the Triton sparse-MLA path.
The vision preset requires `DSV4_DSPARK_TOKENS=5`.
The ViT, aligner, and image sentinel vectors reside only on PP0; later pipeline
ranks retain raw image-token IDs for `bias_vl` routing without replicating the
vision tower.

The defaults are the measured three-card configuration:

- physical GPUs `1,2,3`, PP3 with `15,16,12` layers and TP1;
- Triton sparse MLA, FP8 KV cache, block size 256;
- CUDA Graph execution and the measured CMP170HX DSpark default of 5
  speculative tokens; the official model README recommends width 7;
- 131,072-token context, 1,715 batched tokens, and 8 sequences;
- a 96 MiB caller-owned sparse-indexer logits budget, a 1,024-row cap, and
  1.6 GB KV cache per GPU;
- Marlin block-M 48 for MoE batches with at least 1,000 tokens; smaller decode
  batches retain the upstream shape heuristic.
- sparse-prefill caller output with K tile 32 and no additional persistent
  attention weights.
- BMM-only `wo_a` ownership: the grouped BF16 weight remains, while its unused
  packed FP8 linear duplicate is released after loading.
- direct PP activation receives into the V2 runner's persistent arena; the last
  stage also reuses that arena for DSpark auxiliary-hidden packing.

For a single-request 1,048,576-token deployment, use the capacity preset:

```bash
benchmarks/deepseek_v4/run_cmp170hx_pp3_1m.sh
```

It uses PP partition `15,16,12`, a 2.7 GB per-rank KV budget, and one active
sequence. The PP0/PP1 KV admission requirement is 2.46 GiB; the configured
budget exposes 1,071,031 cache tokens, or 1.02 concurrent full-length
requests. Keep `DSV4_MAX_NUM_SEQS=1` unless the KV and indexer workspace
budgets are recalculated for the higher concurrency.

For the validated long-context speed/correctness matrix, run the service with
`DSV4_MAX_MODEL_LEN=131200` and 2.7 GB KV memory, then execute one request at
each length:

```bash
for n in 1000 32000 128000; do
  .venv/bin/python benchmarks/deepseek_v4/bench_concurrent_longctx.py \
    --base-url http://127.0.0.1:8098 --model dsv4-vision \
    --concurrency 1 --prompt-tokens "$n" --max-tokens 64 \
    --seed "$((17000 + n / 1000))" --scenario general \
    --temperature 0 --top-p 1.0 --mode dspark --timeout 3600 \
    --output "/tmp/dsv4-vision-longctx-${n}.json"
done
```

The harness reports TTFT/prefill throughput, decode throughput after the first
token, DSpark acceptance, and exact needle recall/cross-talk checks.  The
measured 1K/32K/128K results and the post-run two-image correctness check are
recorded in
`docs/design/vllm170-dsv4-vision-sm80-pp3-dspark-2026-09-03.md`.

Do not add `--enforce-eager`: it disables the CUDA Graph decode path and causes
an order-of-magnitude regression on this machine. The launcher defaults to
`DSV4_MODE=benchmark`, which disables prefix caching. Use production mode for
normal serving:

```bash
DSV4_MODE=production benchmarks/deepseek_v4/run_cmp170hx_pp3.sh
```

Probabilistic DSpark must be enabled explicitly; request-side temperature alone
does not change the draft sampler:

```bash
DSV4_DSPARK_TOKENS=5 \
DSV4_DSPARK_DRAFT_SAMPLE_METHOD=probabilistic \
DSV4_MODE=production \
benchmarks/deepseek_v4/run_cmp170hx_pp3.sh
```

Send `temperature=1.0, top_p=1.0` for general sampling and
`temperature=1.0, top_p=0.95` for agentic traffic. The profile and long-context
harnesses enforce this scenario-to-`top_p` mapping. For tool-call validation,
start the service with `--enable-auto-tool-choice --tool-call-parser
deepseek_v4`, then run:

```bash
.venv/bin/python benchmarks/deepseek_v4/bench_probabilistic_agentic.py \
  --base-url http://127.0.0.1:8098 --model dsv4s --repeats 3 \
  --output /tmp/dsv4-probabilistic-agentic.json
```

`bench_concurrent_longctx.py` can also exercise the real chat-template
reasoning levels. For example, keep `top_p=1.0` fixed and run the same prompt
and seed twice to compare temperature under maximum reasoning:

```bash
.venv/bin/python benchmarks/deepseek_v4/bench_concurrent_longctx.py \
  --base-url http://127.0.0.1:8098 --model dsv4s \
  --concurrency 1 --prompt-tokens 32768 --max-tokens 128 \
  --seed 8400 --scenario general --top-p 1.0 \
  --reasoning-effort max --temperature 0 --mode dspark \
  --output /tmp/dsv4-max-temp0.json

.venv/bin/python benchmarks/deepseek_v4/bench_concurrent_longctx.py \
  --base-url http://127.0.0.1:8098 --model dsv4s \
  --concurrency 1 --prompt-tokens 32768 --max-tokens 128 \
  --seed 8400 --scenario general --top-p 1.0 \
  --reasoning-effort max --temperature 1 --mode dspark \
  --output /tmp/dsv4-max-temp1.json
```

When `--reasoning-effort` is present, the harness uses the chat endpoint and
counts the fully rendered chat prompt, so `--prompt-tokens` remains exact.

The CMP170HX PP3 matrix, quality evidence, stage profile, and rejected P0
candidates are documented in
`docs/design/vllm170-dsv4-sm80-dspark-probabilistic-2026-08-17.md`.
The single-GPU Qwen3.8-27B-NVFP4 4x32K scheduler study and production command
are documented in
`docs/design/vllm170-qwen38-nvfp4-sm80-4x32k-scheduler-2026-08-19.md`.
Set `VLLM_DEEPSEEK_V4_DSPARK_PROB_AUDIT=1` only for distribution diagnosis.
It logs exact p/q overlap and counterfactual proposals, and temporarily loads
the checkpoint confidence head. The audit synchronizes full-vocabulary data,
so its latency is not a serving benchmark.

For a one-shot DSpark implementation/reference comparison on CMP170HX, capture
one eager proposal and then replay the official DSpark layers and weights:

```bash
VLLM_DEEPSEEK_V4_DSPARK_REFERENCE_DUMP_PATH=/tmp/dsv4-dspark-reference.pt \
DSV4_DSPARK_TOKENS=5 \
DSV4_DSPARK_DRAFT_SAMPLE_METHOD=probabilistic \
DSV4_PROFILE_EXECUTION=detail \
DSV4_PROFILE_CONTEXT=1024 \
DSV4_PROFILE_TEMPERATURE=1 \
DSV4_PROFILE_TOKENS=8 \
DSV4_PROFILE_REPEATS=1 \
benchmarks/deepseek_v4/profile_cmp170hx_pp3.sh

CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  benchmarks/deepseek_v4/compare_dspark_official_reference.py \
  --model "${DSV4_MODEL_DIR:-/media/z/data6/models/DeepSeek-V4-Flash-0731}" \
  --capture /tmp/dsv4-dspark-reference.pt \
  --output /tmp/dsv4-dspark-official-comparison.json
```

The checkpoint calls the three DSpark parameter groups `mtp.0/1/2`; that is a
weight namespace, not an MTP runtime mode. The official TileLang linear uses an
SM89 FP8 MMA that cannot execute on CMP170HX, so the comparison tool keeps the
official DSpark architecture and weights but automatically uses portable
FP32-accumulating quantization math on SM80. It is a correctness reference, not
a kernel performance comparison.

The main overrides are `DSV4_MODEL_DIR`, `CUDA_VISIBLE_DEVICES`,
`DSV4_MOE_BACKEND`, `DSV4_DSPARK_TOKENS`,
`DSV4_DSPARK_DRAFT_SAMPLE_METHOD`, `DSV4_KV_CACHE_BYTES`, and
`DSV4_MAX_MODEL_LEN`. The 1,715-token scheduler limit was originally selected
for the measured 5,000-token, PP3, DSpark-7 workload. The measured DSpark-5
default reserves 32 speculative slots and leaves 1,683 target tokens per
scheduler step. Override `DSV4_MAX_BATCHED_TOKENS` for other traffic shapes. Set
`VLLM_MARLIN_MOE_BLOCK_SIZE=0` to restore Marlin's upstream shape heuristic.
`VLLM_MARLIN_MOE_BLOCK_SIZE_MIN_M` controls the minimum token batch affected by
an explicit block size.

The sparse-prefill gather workspace has a byte budget rather than assuming
that four maximum-length requests always fit. The default is 512 MiB; override
it with `VLLM_DEEPSEEK_V4_SPARSE_PREFILL_TEMP_MB`. Setting it to zero restores
the legacy four-request limit.

The sparse-indexer prefill gather workspace is also bounded by the scheduler's
request capacity. Deployments with fewer than 40 active sequences reserve at
most `max_num_seqs * max_model_len` history entries instead of 40 full-length
histories. The 1M single-sequence preset therefore removes a redundant 39x
indexer reservation without changing logits or top-k results.

The CMP170HX launcher intentionally uses temporary memory for a better
sparse-indexer score wave: `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=96` and
`VLLM_SPARSE_INDEXER_MAX_QUERY_ROWS=1024`. The logits tensor is a view of the
shared workspace when `VLLM_INDEXER_PREFILL_CALLER_LOGITS=1`; this avoids the
multi-layer allocator retention seen with independent 96 MiB allocations.
The measured 262K PP3 TTFT improved by 1.62% after 32K drift normalization,
without increasing sampled whole-GPU peak memory. Set the budget to `32`, the
row cap to `0`, and caller logits to `0` to restore the previous route.

The CMP170HX launcher enables `VLLM_PP_REUSE_RECV_BUFFER=1`. Set it to `0` to
restore allocating PP receives and DSpark's dedicated hidden-state buffer.
`VLLM_PP_REUSE_RECV_BUFFER_DEBUG=1` additionally validates generation and CUDA
stream ownership; it is intended for validation runs rather than production.
The library-wide default stays disabled because TP>1 and other PP topologies
have not passed this machine's real-hardware gate.

The promoted sparse kernel switches are
`VLLM_DEEPSEEK_V4_SPARSE_PREFILL_CALLER_OUTPUT=1` and
`VLLM_DEEPSEEK_V4_SPARSE_PREFILL_BLOCK_K=32`; zero restores the previous copy
path and backend tile. BF16 `wo_b` and indexer `wq_b` caches were measured but
removed from serving because they add 960 MiB to PP0/PP1 or about 112 MiB per
rank. Persistent attention memory is reserved for long-context capacity.
The `wo_a` path instead removes an unused representation: set
`VLLM_DEEPSEEK_V4_BMM_ONLY_WO_A=0` to restore the packed FP8 duplicate for an
audit run. The default PP3 partition saves roughly 487.5 MiB on PP0/PP1 and
422.5 MiB on PP2 without changing the projection kernel.

The CMP170HX launcher also enables the SM80 sparse-decode split path and direct
caller output with `VLLM_DEEPSEEK_V4_SPARSE_DECODE_SPLIT=1` and
`VLLM_DEEPSEEK_V4_SPARSE_DECODE_CALLER_OUTPUT=1`. The split count adapts to the
query count and indexed history length; its FP32 partials reuse the existing
workspace and require about 2.01 MiB at the measured q1/q8 production shapes.
Set either switch to `0` for an independent rollback. Other GPU families keep
their existing backend route, and the library-wide defaults remain disabled.

## PP3 phase profile

The canonical profile runs a same-context warmup, then opens one measurement
window for both the per-worker Torch profiler and CUDA-event phase records.
PP ranks are written independently. The harness records prompt/output hashes,
server token counts, speculative acceptance, ms/round, the live Git revision,
unconditional and reached-position-conditional acceptance, and the full
topology. It calibrates the generated prompt through the server's
`/tokenize` endpoint and fails unless `usage.prompt_tokens` exactly matches
`DSV4_PROFILE_CONTEXT`:

```bash
DSV4_DSPARK_TOKENS=5 \
DSV4_PROFILE_CONTEXT=32000 \
DSV4_PROFILE_TOKENS=64 \
DSV4_PROFILE_REPEATS=3 \
benchmarks/deepseek_v4/profile_cmp170hx_pp3.sh
```

`DSV4_PROFILE_EXECUTION=detail` is the default. It adds `--enforce-eager` so
the following nested stages can be timed rather than hidden inside graph replay:

- prefill: per-layer total, attention, FFN/MoE, both mHC transitions, embedding,
  stage reconstruction and final norm; attention further splits input GEMMs,
  indexer gather/logits/top-k/DCP merge, compressed/SWA gather, sparse MLA, and
  output projection; Marlin MoE further splits gate, routing, dispatch,
  workspace, token alignment, W13, activation, W2, weighted reduce, finalize,
  and combine, including actual padded-row utilization by DSV4 shape;
- decode: target stages, paged indexer logits, exact/persistent top-k,
  top-k metadata, sparse MLA, output projection, C1/C4/C128 layer families,
  routed Marlin experts, and the overlapped shared-expert W13/activation/W2
  branch plus its exposed join tail; attention metadata is also decomposed by
  KV-cache group and by SWA dense/ragged and MLA slot/C128 sub-stage;
- DSpark: target logits and rejection, hidden/input preparation, context-KV,
  draft backbone, base logits, and every Markov bias/sample step.

This detailed run is diagnostic and must not be quoted as production decode
throughput. Use `DSV4_PROFILE_EXECUTION=graph` for the deployed CUDA Graph path;
that mode reports graph replay and eager boundary totals but intentionally does
not claim unavailable graph-internal subdivisions.

For a lower-perturbation layer study, set
`VLLM_DEEPSEEK_V4_PHASE_PROFILE_LAYERS=model.layers.16`. Model-level and
metadata stages remain enabled, while layer-scoped CUDA events are limited to
the listed canonical layers (an `.attn` suffix is matched automatically).
`VLLM_DEEPSEEK_V4_PHASE_PROFILE_RECORD_FUNCTION=0` additionally disables Torch
range annotations while retaining the phase CUDA Events.

For a production-shaped DSpark boundary run, an exact stage whitelist avoids
inserting events throughout the captured draft and target graphs:

```bash
VLLM_DEEPSEEK_V4_PHASE_PROFILE_STAGES=\
target.attn_metadata,target.model_inputs,target.model_graph_replay,\
target.sampling_total,target.logits,target.rejection,target.sample,\
draft.propose_total,draft.hidden_prepare,draft.prepare_inputs,\
draft.context_kv,draft.dispatch,draft.metadata,draft.graph_replay \
DSV4_PROFILE_EXECUTION=graph \
DSV4_PROFILE_CONTEXT=32000 \
DSV4_PROFILE_REPEATS=1 \
benchmarks/deepseek_v4/profile_cmp170hx_pp3.sh
```

The DSpark report separates intermediate prefill proposals from steady decode
proposals. Its dependency-path table uses the critical worker for target input
and metadata preparation, but sums aligned PP0/PP1/PP2 target model ranges for
one q8 verification. It also emits a cross-rank work sum explicitly marked as
non-wall-time accounting.

The CMP170HX launcher enables
`VLLM_DEEPSEEK_V4_DIRECT_RAGGED_SWA=1` and
`VLLM_DEEPSEEK_V4_DIRECT_RAGGED_C128=1`. On SM8x these build causal/non-causal
SWA and C128 decode metadata directly in graph-stable ragged buffers, avoiding
the dense-to-ragged pack, final copy, and unused persistent dense decode buffers.
Each switch can be set to `0` for an independent rollback; other device families
retain the original builders.

Use `DSV4_PROFILE_EXECUTION=benchmark` for an endpoint-level comparison without
Torch profiler or CUDA-event synchronization. The harness still records exact
tokens, TTFT, decode latency, configuration metadata, and sampled per-GPU peak
memory. The 2,048/4,096/8,192 study and the promoted 1,715-token PP3 setting are
documented in
`docs/design/vllm170-dsv4-sm80-max-batched-tokens-2026-08-14.md`.

The script writes per-worker traces, profiler tables, a server log,
`result.jsonl`, raw `phase_trace/*.jsonl`, `phase_summary.json`, one Markdown
summary for each of prefill/decode/DSpark, and `profile_status.json` under
`profiles/deepseek_v4/<UTC timestamp>/`. Prefill chunks can overlap across PP
stages, so its report retains per-stage critical-rank attribution. A single
autoregressive target verification round instead traverses PP0, PP1, then PP2;
the decode report aligns those model ranges by round and reports their serial
sum alongside the slowest-stage balance diagnostic. Do not sum arbitrary
maxima or unaligned stages across ranks. Nested `total`, `module`, `stage`, and
`detail` ranges also overlap and must not be added across levels. Both prefill
and decode reports give model/layer coverage, per-layer attention/FFN/mHC
totals, and Marlin sub-stages. Decode additionally groups work by compression
ratio and separates the shared-expert auxiliary stream from its main-stream
join tail. Repeated workloads are all checked fail-closed and aggregated in
the same phase report.

The accepted CMP170HX Marlin P0 decomposition, block-M A/B and next optimization
queue are documented in
`docs/design/vllm170-dsv4-sm80-marlin-moe-profile-2026-08-14.md`.

The 32K DSpark-7 decode hierarchy, Graph/eager boundary, C4 and shared-expert
analysis, and memory-bounded optimization queue are documented in
`docs/design/vllm170-dsv4-sm80-decode-profile-2026-08-14.md`.

The DSpark-specific 32K/64K proposal split, draft metadata/context-KV detail,
non-final prefill analysis, and zero-large-residency optimization queue are
documented in
`docs/design/vllm170-dsv4-sm80-dspark-profile-2026-08-15.md`.

The follow-up W13/W2 execution-config sweep, caller-owned FP32 reduce workspace
A/B/A, and fusion feasibility study are documented in
`docs/design/vllm170-dsv4-sm80-marlin-moe-config-workspace-2026-08-14.md`.
Reproduce the isolated SM80 sweep with real DSV4 shapes using:

```bash
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  .venv/bin/python benchmarks/kernels/deepseek_v4_marlin_moe.py \
  --stage both --m 1000 1666 1667 2000 4048 \
  --routing-seeds -1 8 9 --warmup 3 --iterations 10 --samples 5 \
  --fp32-reduce-workspace caller --output /tmp/marlin-moe.jsonl
```

Use `--fp32-reduce-workspace internal` for the same-binary allocation control.
The production environment switch is
`VLLM_MARLIN_MOE_CALLER_FP32_WORKSPACE=0|1`; disabling it restores C++ internal
allocation. Execution configs remain on Marlin auto selection because no
forced shape beat it reproducibly.

The rejected W2 two-stage FP32 grouped-tail can be reproduced independently:

```bash
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  .venv/bin/python \
  benchmarks/kernels/deepseek_v4_marlin_w2_grouped_tail.py \
  --m 1666 1667 2000 4048 --routing-seeds -1 8 9 \
  --warmup 3 --iterations 10 --samples 5 --output /tmp/w2-fp32-tail.jsonl
```

This benchmark requests an otherwise-unused optional FP32 routed output from
Marlin and compares a top-k6 FP32 tail with the production BF16 `moe_sum`.
Serving never passes that output. CMP170HX measurements were exactly correct
but 3.23%-3.61% slower with twice the routed workspace, so this is a negative
result and not a deployment option.

The accepted attention P1 A/B, memory budget, rejected indexer-Q route, and
compressor warp study are documented in
`docs/design/vllm170-dsv4-sm80-attention-p1-2026-08-14.md`. Reproduce the
isolated sparse and FP8-linear sweeps with:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  benchmarks/kernels/deepseek_v4_sparse_mla_prefill.py \
  --output /tmp/sparse-mla.jsonl
CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  benchmarks/kernels/deepseek_v4_sparse_mla_decode.py \
  --queries 8 --main-len 128 --extra-len 512 --splits 2 \
  --output /tmp/sparse-mla-decode.jsonl
CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  benchmarks/kernels/deepseek_v4_output_projection.py \
  --m 1 6 64 256 1000 1666 1667 2000 \
  --output /tmp/wo-b.jsonl
CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  benchmarks/kernels/deepseek_v4_compressor_save.py \
  --output /tmp/compressor-save.jsonl
```

## Mainline DeepSeek-V4 performance audit

The per-PR route audit, CMP170HX router and mHC measurements, and the async
DSpark empty-C128 gap are documented in
`docs/design/vllm170-dsv4-mainline-performance-audit-2026-08-15.md`.
Reproduce the isolated mainline fusion comparison with:

```bash
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  .venv/bin/python benchmarks/kernels/deepseek_v4_mainline_fusions.py \
  --component all --m 1 8 64 1667 \
  --warmup 3 --iterations 20 --samples 5 \
  --output /tmp/dsv4-mainline-fusions.jsonl
```

The production default keeps the specialized DeepSeek-V4 router. Set
`VLLM_DEEPSEEK_V4_SPECIALIZED_ROUTER=0` only for a controlled generic-router
A/B; the switch is not a promoted deployment setting. Mainline's host-side
empty-C128 skip cannot safely use optimistic sequence lengths in async DSpark.
It remains a GPU-side exact-boundary optimization target.

The harness watches both the workload and server processes. If model loading
fails, it exits immediately with the server log instead of waiting for the
health-check timeout.

The current CMP170HX driver reports `CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED`.
The validator therefore accepts the reproducible workload and complete PP3
operator traces but marks `kernel_profile_accepted=false`. These traces are
useful for operator attribution, while the phase JSONL uses CUDA Events for
GPU stage latency without CUPTI. Phase timing is not a per-kernel NCU result.
Use CUDA Events in isolated kernel benchmarks for kernel-candidate latency and
peak-allocation measurements; if a future driver exposes CUDA activities,
`profile_status.json` will promote the capture automatically. Pass
`--require-cuda` directly to `profile_validate.py` for a fail-closed
kernel-profile gate.

Use Nsight Systems/Compute on isolated kernel microbenchmarks after the PP3
trace identifies a bottleneck. Late-attaching Nsight to already initialized
spawned workers is not a supported acceptance path on this host.

The checkpoint declares `dspark_block_size=5`, while its official README
recommends `num_speculative_tokens=7`. Width 5 is this CMP170HX launcher's
measured default because its earlier greedy 32K A/B improved decode throughput
by 2.15%; width 7 remains a supported production candidate, not merely an audit
shape. Probabilistic q5/q7 promotion must compare accepted length, ms/round and
decode throughput instead of acceptance percentage alone. The current 32K,
9-seed probabilistic A/B retained q5: q7 was throughput-neutral for general
sampling and 16.19% slower for agentic `top_p=0.95`. A captured run is
acceptable only when the
completion length matches exactly, timing fields are finite, speculative
counters conserve their meaning, and the profiler-owned process group exits.

The opt-in q1 acceptance audit records the production sampler decision, target
rank, drafter base logits, Markov bias, and final draft logits. Drafter-stage
records require eager execution because Python cannot inspect intermediates from
a replayed CUDA Graph. The audit skips non-final prefill chunks, remains bounded
and disabled by default, and synchronizes the GPU, so its latency is not a
production measurement:

```bash
VLLM_DEEPSEEK_V4_DSPARK_Q1_AUDIT=1 \
VLLM_DEEPSEEK_V4_DSPARK_Q1_AUDIT_SAMPLES=64 \
DSV4_PROFILE_EXECUTION=benchmark \
benchmarks/deepseek_v4/profile_cmp170hx_pp3.sh --enforce-eager
```

Pair `DSpark q1 drafter audit` and `DSpark q1 acceptance audit` records by
`position`. The final drafter proposal may have no target record when the
request reaches its output limit. ID differences at equal maximum logits are
ties, not numerical mismatches.

The opt-in real-logits audit compares either the persistent or exact selector
with an independent `torch.topk` reference for both q6 and q8. It validates the
index set, selected-score multiset, index range and uniqueness, short-row
padding, cutoff ties, and non-finite logits. It intentionally synchronizes and
therefore requires eager mode; it is a correctness probe, not a performance
run. Disable the production exact route when auditing the persistent selector:

```bash
VLLM_DEEPSEEK_V4_DSPARK_EXACT_TOPK=0 \
VLLM_DEEPSEEK_V4_DSPARK_TOPK_AUDIT=1 \
VLLM_DEEPSEEK_V4_DSPARK_TOPK_AUDIT_MIN_LEN=5000 \
VLLM_DEEPSEEK_V4_DSPARK_TOPK_AUDIT_SAMPLES=8 \
DSV4_DSPARK_TOKENS=7 \
benchmarks/deepseek_v4/profile_cmp170hx_pp3.sh --enforce-eager
```

The minimum is measured in compressed indexer tokens. It prevents warmup and
short-context requests from consuming the bounded per-worker sample budget.
When the 512th score is tied, different index sets can both be mathematically
valid; use `strict_score_misses`, `score_mismatches`, invalid/duplicate indices,
and padding errors as the correctness gate rather than raw index recall alone.
The old `VLLM_DEEPSEEK_V4_Q6_TOPK_AUDIT*` names remain compatibility aliases.

The project defaults `VLLM_DEEPSEEK_V4_DSPARK_EXACT_TOPK=1`; launchers do not
need to repeat it. The route is deliberately narrow: SM80, single-request
DSpark q6/q8, and top-k 512. Other decode shapes keep their existing
cooperative/persistent selector. Set the variable to `0` for an A/B rollback.
The real 32K audit and A/B/A evidence are recorded in
`docs/design/vllm170-dsv4-sm80-dspark-exact-topk-2026-08-16.md`.

## SM80 streaming-indexer experiment

The prototype benchmark compares full logits plus the production prefill
selector against tile-local exact top-k plus candidate merge:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
    benchmarks/kernels/deepseek_v4_streaming_indexer.py \
    --rows 1715 --keys 65536 --tile-size 8192 \
    --full-row-chunk 128 --streaming-row-chunk 1024
```

It fails if the selected score set differs from the mathematical full-logits
top-k, and reports CUDA-Event latency plus peak allocated bytes as JSON. The
serving route is experimental and defaults off. It is restricted to SM80,
FP8 indexer cache, DCP=1, top-k=512, and long compressed rows:

```bash
VLLM_INDEXER_PREFILL_STREAMING_TOPK=1
VLLM_INDEXER_PREFILL_STREAMING_MIN_LEN=16384
VLLM_INDEXER_PREFILL_STREAMING_TILE_SIZE=16384
VLLM_INDEXER_PREFILL_STREAMING_MAX_ROWS=512
```

The 2026-08-16 CMP170HX PP3 work reduced long-shape temporary memory and made
the corrected isolated 262K-shape kernel 2.44% faster than full logits. The final service
A/B still regressed normalized 262K TTFT by 1.37%, so production routing
remains disabled. See
`docs/design/vllm170-dsv4-sm80-streaming-indexer-2026-08-16.md`.

## Minimum performance suite

Discard the first request after startup because it includes JIT warmup. Keep
prefix caching disabled for these measurements:

```bash
.venv/bin/python benchmarks/deepseek_v4/bench_prefill.py cmp170hx
.venv/bin/python benchmarks/deepseek_v4/bench_longctx.py cmp170hx \
    12000,24000,50000,77000
.venv/bin/python benchmarks/deepseek_v4/bench_decode3.py cmp170hx
.venv/bin/python benchmarks/deepseek_v4/bench_decode_stream.py cmp170hx \
    2000,8000,32000,65000,100000
```

`bench_chunk_truth.py` validates that streaming code is using server-reported
completion tokens rather than SSE chunk counts. `bench_needle.py`,
`bench_conc_needle.py`, and `bench_ceiling.py` cover long-context correctness,
multi-request row isolation, and the context ceiling. `bench_concurrency.py`
measures multi-request throughput.

Historical accumulated-chat analyzers, the probe script, and the old
two-request decode-subtraction benchmark were intentionally not absorbed. They
do not contribute to the current three-card acceptance criteria or have been
superseded by server-reported token accounting.

## Measurement rules

- Use `usage.prompt_tokens` and `usage.completion_tokens` as ground truth.
- Set `ignore_eos` for comparable fixed-length decode runs.
- Do not infer decode rate by subtracting two independent request times.
- Report the full mixed-content run, not the fastest short window.
- Keep the model, hardware, PP partition, context, cache, and DSpark settings in
  the result record.
