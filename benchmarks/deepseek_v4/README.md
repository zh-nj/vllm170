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

The defaults are the measured three-card configuration:

- physical GPUs `1,2,3`, PP3 with `15,15,13` layers and TP1;
- Triton sparse MLA, FP8 KV cache, block size 256;
- CUDA Graph execution and DSpark with 7 speculative tokens;
- 131,072-token context, 1,715 batched tokens, and 8 sequences;
- a 32 MiB sparse-indexer logits budget and 1.6 GB KV cache per GPU.
- Marlin block-M 48 for MoE batches with at least 1,000 tokens; smaller decode
  batches retain the upstream shape heuristic.
- sparse-prefill caller output with K tile 32 and no additional persistent
  attention weights.

Do not add `--enforce-eager`: it disables the CUDA Graph decode path and causes
an order-of-magnitude regression on this machine. The launcher defaults to
`DSV4_MODE=benchmark`, which disables prefix caching. Use production mode for
normal serving:

```bash
DSV4_MODE=production benchmarks/deepseek_v4/run_cmp170hx_pp3.sh
```

The main overrides are `DSV4_MODEL_DIR`, `CUDA_VISIBLE_DEVICES`,
`DSV4_MOE_BACKEND`, `DSV4_DSPARK_TOKENS`, `DSV4_KV_CACHE_BYTES`, and
`DSV4_MAX_MODEL_LEN`. The 1,715-token scheduler limit is specific to the
measured 5,000-token, PP3, DSpark-7 workload: reserving 48 speculative slots
leaves 1,667 target tokens and produces balanced `1667 + 1667 + 1666` prefill
microbatches. Override `DSV4_MAX_BATCHED_TOKENS` for other traffic shapes. Set
`VLLM_MARLIN_MOE_BLOCK_SIZE=0` to restore Marlin's upstream shape heuristic.
`VLLM_MARLIN_MOE_BLOCK_SIZE_MIN_M` controls the minimum token batch affected by
an explicit block size.

The sparse-prefill gather workspace has a byte budget rather than assuming
that four maximum-length requests always fit. The default is 512 MiB; override
it with `VLLM_DEEPSEEK_V4_SPARSE_PREFILL_TEMP_MB`. Setting it to zero restores
the legacy four-request limit.

The promoted sparse kernel switches are
`VLLM_DEEPSEEK_V4_SPARSE_PREFILL_CALLER_OUTPUT=1` and
`VLLM_DEEPSEEK_V4_SPARSE_PREFILL_BLOCK_K=32`; zero restores the previous copy
path and backend tile. BF16 `wo_b` and indexer `wq_b` caches were measured but
removed from serving because they add 960 MiB to PP0/PP1 or about 112 MiB per
rank. Persistent attention memory is reserved for long-context capacity.

## PP3 phase profile

The canonical profile runs a same-context warmup, then opens one measurement
window for both the per-worker Torch profiler and CUDA-event phase records.
PP ranks are written independently. The harness records prompt/output hashes,
server token counts, speculative acceptance, ms/round, the live Git revision,
and the full topology. It calibrates the generated prompt through the server's
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
  benchmarks/kernels/deepseek_v4_output_projection.py \
  --m 1 6 64 256 1000 1666 1667 2000 \
  --output /tmp/wo-b.jsonl
CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  benchmarks/kernels/deepseek_v4_compressor_save.py \
  --output /tmp/compressor-save.jsonl
```

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

Use DSpark width 5 to audit the q6 target pass (`next_n=6`) and width 7 for the
production q8 configuration. A captured run is acceptable only when the
completion length matches exactly, timing fields are finite, speculative
counters conserve their meaning, and the profiler-owned process group exits.

The opt-in real-logits audit compares the persistent and exact q6 top-k sets.
It intentionally synchronizes and therefore requires eager mode; it is a
correctness probe, not a performance run:

```bash
VLLM_DEEPSEEK_V4_Q6_TOPK_AUDIT=1 \
VLLM_DEEPSEEK_V4_Q6_TOPK_AUDIT_MIN_LEN=5000 \
VLLM_DEEPSEEK_V4_Q6_TOPK_AUDIT_SAMPLES=8 \
DSV4_DSPARK_TOKENS=5 \
benchmarks/deepseek_v4/profile_cmp170hx_pp3.sh --enforce-eager
```

The minimum is measured in compressed indexer tokens. It prevents warmup and
short-context requests from consuming the bounded per-worker sample budget.

## SM80 streaming-indexer prototype

The prototype benchmark compares full logits plus the production prefill
selector against tile-local exact top-k plus candidate merge:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
    benchmarks/kernels/deepseek_v4_streaming_indexer.py \
    --rows 8 --keys 32768 --tile-size 1024
```

It fails if either selected indices or scores differ from the mathematical
full-logits top-k, and reports CUDA-Event latency plus peak allocated bytes as
JSON. This is a design probe, not a serving backend. Current CMP170HX results
show a large memory reduction for small row counts but prohibitive latency;
production routing remains disabled until score, local selection, and
candidate merge are fused in a native SM80 implementation.

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
