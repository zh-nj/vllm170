# Qwen3.8 Flash Next AutoRound: target verification investigation

This investigation isolates target verification after the PP relay fix and
the independent K=2 experiment. The configuration stays at K=3, PP2/TP1,
physical CMP170HX GPUs 2/3 in PCI order, layer partition `26,22`, BF16 target
head, FP8 draft head, fused draft steps and CPU PLE offload. Prepare-ahead,
prefix metadata and draft top-k sharing remain disabled. DeepSeek stays stopped.

The target checkout is `e2e327dc53b93acb85c8b56b526d369eb5461421` with local
changes. The source checkout `/mnt/data/apps/vllm` is
`31ebe18bb63303be83137a59f668ca9812ac3579`, with locally cached `origin/main`
at `0e468adb43d60484b5a6546e0124b686fd76baf3`. These are local source snapshots,
not a statement about the newest remote revision.

## Independent verification boundaries

The runtime uses MRV2 `MTPSpeculator` in `vllm/v1/worker/gpu/`. Each target
round computes the three draft positions and one bonus position for every
request before acceptance is decided. Target query length is 4; FULL graphs
for 1–4 requests contain 4/8/12/16 tokens. These token counts are not request
concurrency settings.

| Boundary | Source | Observable result |
| --- | --- | --- |
| Input/logits selection | `gpu/model_runner.py:1312-1433` | `cu_num_logits`, `logits_indices` select verification and bonus positions |
| Target forward | `qwen4_exp/nvidia/model.py:355-669` | 48 layers, intermediate hidden states, final mixer |
| Target LM head | `gpu/model_runner.py:1556-1607` | BF16 target logits, separate from draft FP8 head |
| Accept/reject | `gpu/spec_decode/rejection_sampler.py:789-842` | Greedy equality or probability-ratio acceptance; sampled tokens/counts |
| Token/state commit | `gpu/model_runner.py:1633-1662`, `model_states/mamba_hybrid.py:338-390` | History, computed-token counts, GDN/PLE state commit |
| PP sampled/draft relay | `gpu/pp_utils.py:292-466` | Deferred packed results; next draft follows verification |

Layers 0–25 on PP0 contain 20 GDN and 6 QSA layers; layers 26–47 on PP1
contain 16 GDN and 6 QSA layers. The model has 36 GDN and 12 QSA layers.
Every layer has attention HC, MLP HC, routed experts and shared experts.
`ple_layer_ids=[2]` uses one-based IDs, so PLE runs in zero-based layer 1
(the second layer) on PP0. PP0 materializes the intermediate tensor; PP1
applies the final HC mixer and target head.

## Measurement contract

Artifacts are in `artifacts/qwen38-autoround-verification-20261007/`.
The temporary diagnostic endpoint is loopback-only port 26662. Its deferred
worker extension installs no hooks until the first idle RPC. The initial
formal reference runs before that RPC, with no `sitecustomize.py` or CPU
trace injection. Ordinary serving is restored on `0.0.0.0:26661` after the
campaign, with development RPC, the worker extension and `PYTHONPATH` removed.

The frozen workloads are 1K/C1, 32K/C1 and 1K/C4, greedy sampling and 256
generated tokens. Every request uses a unique cache salt; prefix hits must
be zero. Needle retrieval, output length and absence of special output tokens
are asserted. C4 throughput is the common decode window across all requests.
Acceptance, mean emitted length and per-position accepted counts accompany
throughput. GPU clock and temperature observations qualify formal results;
the initial and restored reference records include clock, temperature and
power snapshots. Same-instance A/B records include clock, power and utilization
snapshots, but do not include temperatures. Clocks are not locked.
Formal A/B endpoint SM clock samples ranged from 1470 to 1485 MHz on GPU 2
and 1455 to 1470 MHz on GPU 3; these are endpoint observations, not continuous
clock control.

Ordinary phase hooks are stripped or disabled by FULL CUDA Graph/AOT.
The diagnostic graph is recaptured using the existing model, graph descriptors,
input buffers and state. Capture-safe globaltimer stamps bracket opaque
custom ops, external dense GEMMs and existing phase scopes. Weight pointers
identify per-layer projections. The original target graphs and all draft
graphs are retained. GPU events/globaltimer are used because CMP170HX does
not provide usable CUPTI/Nsight profiling here.

Only nested exclusive main-stream intervals plus uncovered gaps are summed.
Shared-expert secondary streams are recorded separately. CPU intervals,
parent/child scopes and NCCL completion waits are not added to GPU work;
uncovered intervals receive no speculative CPU or network attribution.
Instrumentation can alter C4 batching and adds timestamp kernels, so fine
graph measurements identify hotspots rather than measure serving speedup.
Any candidate requires a same-instance comparison with recording disabled.

## Verification measurements

The initial uninstrumented reference, before the first diagnostic RPC, gave:

| Workload | Decode tok/s | Accepted draft fraction | Mean emitted tokens per round |
| --- | ---: | ---: | ---: |
| 1K/C1 | 144.55 | 0.5000 | 2.5000 |
| 32K/C1 | 116.87 | 0.3522 | 2.0565 |
| 1K/C4 aggregate | 402.21 | 0.5166 | 2.5498 |

Each workload has one warmup and three formal waves. Acceptance varies with
the text and numerical execution order; these values do not establish a
universal optimal draft count. With K=3, mean emitted length is
`1 + 3 * accepted_draft_fraction`. The C1 cost index
`1000 * mean_emitted_length / decode_tok_s` is approximately 17.30/17.70 ms.
The corresponding C4 index uses aggregate throughput and is not a global
verification-round latency.

Coarse inclusive GPU interval medians, in milliseconds:

| Workload | PP0 target forward | PP1 target forward | PP1 target head | PP1 rejection | Rejection verify child | PP0/PP1 sampled postprocess |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K/C1 | 7.277 | 6.051 | 0.786 | 0.115 | 0.046 | 0.030 / 0.027 |
| 32K/C1 | 7.543 | 6.297 | 0.785 | 0.106 | 0.043 | 0.024 / 0.026 |
| 1K/C4 | 9.427 | 7.633 | 0.802 | 0.103 | 0.045 | 0.067 / 0.023 |

The verify child is already inside rejection; adding both would double count.
The head is outside target forward. Postprocess does not include every later
Mamba state update or PP completion wait. C4 mostly scheduled alternating
three-request and one-request rounds, with a two-request tail; the table is
not an M=16 kernel measurement. CPU metadata building overlaps GPU execution
and includes host submission/waits. Its duration is not added to GPU intervals
or interpreted as pure metadata GPU compute.

The coarse report contains 152 cross-rank round ledgers, all conserving the
measured window using exclusive intervals and uncovered gaps. The complete
coarse/fine/focus report has 21 windows, 42 rank profiles and 2138 rank-rounds,
with no parser or conservation warnings. Globaltimer quantization is about
1.024 microseconds. Timestamp kernels inflate short intervals. Full fine
instrumentation adds roughly half to target time and changes C4 scheduling
to two plus two requests. Separate GDN, MoE, HC, QSA and PLE captures reduce
that disturbance. All formal throughput comparisons disable recording.

### Target forward hotspots

The following are sums within each component's separate 1K/C1 focus capture,
reported as per-rank medians in milliseconds. They include instrumentation.
Rows from different captures are not an additive decomposition of the coarse
target total and should not be converted into exact percentages of it.

| Component | PP0 | PP1 | Interpretation |
| --- | ---: | ---: | --- |
| GDN input/output projections | 1.963 | 1.566 | Largest measured dense projection group |
| GDN fused recurrent core/norm | 0.494 | 0.393 | Smaller than its projections |
| MoE Marlin W13 | 1.072 | 0.890 | Routed expert matrix computation |
| MoE Marlin W2 | 0.763 | 0.629 | Routed expert matrix computation |
| MoE router/select | 0.286 | 0.249 | Independent of alignment |
| MoE alignment | 0.106 | 0.094 | Limited remaining speedup budget |
| MoE activation/reduction | 0.103 / 0.105 | 0.089 / 0.090 | Separate from the two matrix kernels |
| HC dense matrix operations | 1.126 | 0.983 | Attention/MLP HC and PP1 final mixer |
| QSA projections | 0.505 | 0.510 | Six QSA layers on each rank |
| QSA sparse attention | 0.157 | 0.160 | Separate from index selection |
| QSA indexer top-k | 0.034 | 0.035 | Grows with input length |
| PLE key/value projections | 0.053 / 0.015 | absent | Second layer, PP0 only |
| PLE ngram/CPU embedding/short-conv intervals | 0.022 / 0.009 / 0.008 | absent | GPU interval observations, not host table bandwidth |

At 32K/C1, QSA top-k reaches 0.216/0.219 ms; index scoring rises to
0.068/0.069 ms, while sparse attention is 0.170/0.171 ms. Long-context index
selection contributes to additional verification cost. GDN projections and
Marlin W13/W2 remain the leading measured groups. HC auxiliary norm, SiLU
and gate-mix are also individually recorded in `verification-summary.json`.

Shared experts run on private streams. The fully instrumented 1K/C1 shared
expert outer intervals are about 0.479/0.408 ms, with their dense children
recorded separately. They overlap main-stream work; neither parent/child
durations nor private-stream intervals are added to the main critical path.

Geometry records cover graph token shapes 4/8/12/16. Representative BF16 GDN
matrices are input `(16384, 2560)`, gate `(96, 2560)` and output
`(2560, 6144)`. HC uses input width 10240, down width 336 and up width 320.
Routing has 512 experts and top-k 10; Marlin uses INT4/group-128 experts and
block-8 alignment. This geometry qualifies future experiments; an M=1 decode
kernel result does not qualify an MTP verification optimization.

## Independent upstream metadata reuse

Source commit `dd3bc9c72faf410043b453e18e58553a1ad4d388`,
`[Perf][MRV2] Reuse Mamba/GDN metadata across KV cache groups (#58762)`,
matches the runtime: each rank has three GDN groups with identical MambaSpec
and exact builder type. Their block size is 1600, state shapes are
`((6, 10240), (48, 128, 128))`, dtypes BF16/FP32 and speculative blocks 3.
PLE and QSA are separate and retain their existing metadata paths.

The local implementation caches metadata per call by `(KVCacheSpec, builder
type)`. The first matching group builds batch metadata; later groups retain
that metadata and update their own physical state IDs. FULL graph state IDs
use each destination builder's own persistent buffers. Batch token indices,
accepted counts and query offsets can be shared; group state IDs cannot.
Aligned state IDs use the existing fused all-group preparation. Mixed prefill,
ordinary decode and padded speculative batches preserve the full-build result.

This is independently enabled by `QWEN_MTP_METADATA_REUSE=1` in the launcher,
or JSON boolean `qwen4_exp_mtp_metadata_reuse` in additional config. Both
defaults are off. The gate requires MRV2, the exact GDN builder, SM80,
Qwen4Exp H2560/48 layers/512 experts, INC/BF16, PP2/TP1/DP1/PCP1, K=3,
no adaptive verification, at most four sequences and FULL graphs.
Other Mamba/GDN models and derived builders retain their previous dispatcher.
K=2 remains excluded from this new feature.

Same-instance four alternating A/B pairs per workload, with recording off:

| Workload | Baseline tok/s | Reuse tok/s | Change of arm medians | Baseline/reuse mean emitted length |
| --- | ---: | ---: | ---: | ---: |
| 1K/C1 | 141.109 | 142.228 | +0.793% | 2.4735 / 2.4569 |
| 32K/C1 | 113.465 | 114.383 | +0.809% | 2.0238 / 2.0157 |
| 1K/C4 aggregate | 400.594 | 400.437 | -0.039% | 2.5197 / 2.4872 |

Individual paired changes were +2.31/-2.65/-0.09/-0.71% at 1K/C1,
+8.86/+2.07/-0.42/+1.98% at 32K/C1 and
+1.01/-7.74/+2.15/+1.92% at C4. The table compares arm medians; it is not
the median of those paired changes. The gain is not stable enough for default
promotion. Three full builds structurally become one build plus two state-ID
updates, but no completed candidate CPU timing window establishes a measured
host latency reduction. Overlap with target GPU work is a plausible reason
for limited throughput benefit, not a demonstrated exact attribution.

All 48 formal requests passed needle, output length, special-token and zero
prefix-hit checks. Free-form greedy outputs vary within both baseline and
candidate arms, so this is not a claim of whole-output bitwise identity.
Metadata equality and group-state independence are checked separately against
an independent full build in CPU tests.

Other audited source changes were not transplanted: pure-spec slicing already
has an equivalent independent local switch; QSA sparse GQA and HC fusion
commits require unsupported hardware or selection contracts. Existing SM80
fallbacks remain in use. The earlier negative MTP/dense/Marlin experiments
are recorded in the linked reports and were not repeated here.

## Isolated alignment prototype and failed serving experiment

`align_candidate.py` is an artifact-only Triton prototype for int32 contiguous
routes of M=8/12/16, top-k 10, 512 experts and block size 8. A single CTA sorts
composite expert/route keys, scans group boundaries and padded offsets, clears
outputs, synchronizes and scatters route IDs. It filters invalid experts and
does not support expert remapping. It is not installed in product dispatch.

An independent CPU reference checks group counts, expert IDs, padding and
the complete valid route multiset. Within-expert ordering may differ from the
atomic baseline, including across blocks, so these checks do not establish
real-weight MoE bitwise output identity. CUDA Event/Graph measurements on a
free physical CMP170HX GPU 4 exclude allocation, RNG and host transfers.
There are 18 shape/pattern cases and two measurement orders, with 256 calls
per graph replay, four warm replays and seven samples. Valid cases check both
implementations; invalid/empty cases validate the candidate contract.

Median microseconds across the two orders for representative valid patterns:

| M | Pattern | Baseline | Prototype |
| --- | --- | ---: | ---: |
| 8 | disjoint / overlap / random | 7.122 / 6.474 / 5.970 | 3.638 / 3.772 / 3.034 |
| 12 | disjoint / overlap / random | 5.774 / 5.804 / 6.008 | 3.058 / 3.064 / 3.060 |
| 16 | disjoint / overlap / random | 5.728 / 5.826 / 6.004 | 3.742 / 3.772 / 3.758 |

The isolated kernel improves about 35–50%. These tiny route arrays fit L2;
this is no claim about expert weight bandwidth or serving throughput. Applying
a few microseconds to 48 layers suggests only approximately 0.10–0.15 ms
per round, below 1% of the C1 cost index. That is a hypothesis for budgeting
experiments, not a measured end-to-end speedup.

The attempted C4 serving A/B did not produce formal candidate samples.
An initial utility RPC used a stale imported candidate-control module and
raised an assertion. After a diagnostic reload fixed that assertion, candidate
graph capture completed, but the first baseline warmup failed with
`EngineCore model_output=None` and `RuntimeError("unexpected error")` at
`vllm/v1/engine/core.py:689`. The logs do not identify a CUDA illegal access
or OOM; the engine failure cause remains unresolved. This is an unqualified
integration experiment. The diagnostic service was stopped, logs retained,
and ordinary Qwen serving restored without this prototype or diagnostic code.
No alignment switch was added to the launcher and no end-to-end gain is claimed.

## Validation, artifacts and retained recipe

The retained recipe is K=3, existing PP relay optimization on, FP8 draft head,
fused draft steps, and prepare-ahead/prefix-metadata/new metadata-reuse off.
The new metadata feature is available for controlled experiments; this round
does not justify another default performance change. Priorities from the
measured verification path are GDN dense projections, routed Marlin matrix
work and HC dense operations, followed by long-context QSA selection.
Acceptance checks and the bounded alignment prototype have much smaller
remaining budgets.

Validation commands:

```bash
.venv/bin/python -m pytest \
  tests/v1/attention/test_gdn_metadata_builder.py \
  tests/v1/worker/test_pp_utils.py \
  tests/v1/worker/test_mamba_hybrid_model_state.py -q
.venv/bin/pre-commit run --files \
  vllm/v1/attention/backends/gdn_attn.py \
  vllm/v1/worker/gpu/attn_utils.py \
  tests/v1/attention/test_gdn_metadata_builder.py \
  tests/v1/worker/test_pp_utils.py \
  scripts/serve_qwen38_flash_next_autoround_cmp170hx_pp2.sh
```

The tests passed: 127 tests, with 14 existing Torch JIT deprecation warnings.
Product-file pre-commit hooks passed. The metadata cases cover none/align cache
modes, FULL/non-FULL paths, regular and mixed prefill, and leading/trailing
speculative padding. Destination state buffers and unchanged source state IDs
are asserted alongside all public metadata fields.

The restored ordinary K=3 service passed all six thinking, automatic tool-call
and concurrent arithmetic checks. Its uninstrumented reference, with one
warmup and three formal waves per workload, was:

| Workload | Restored decode tok/s | Accepted draft fraction | Mean emitted tokens per round |
| --- | ---: | ---: | ---: |
| 1K/C1 | 141.62 | 0.4936 | 2.4808 |
| 32K/C1 | 116.61 | 0.3575 | 2.0726 |
| 1K/C4 aggregate | 391.75 | 0.5020 | 2.5061 |

These are independent reload samples, with output/acceptance and clock
variation, rather than a same-instance estimate of metadata-reuse benefit.
The restored C4 formal waves were 391.75/388.18/406.57 tok/s. All reference
requests passed the bounded output and cache checks.

Delivery validation confirmed `/health` and the AutoRound model identity,
the listener on `0.0.0.0:26661`, API PID 2749224 and PP workers
2753264/2754617. The two compute processes belong to the ordinary service
cgroup and physical GPU UUIDs for devices 2/3. No other compute process
remained. The worker extension and `PYTHONPATH` are absent, development mode
is off and `/collective_rpc` returns 404. The diagnostic and DeepSeek units
are inactive with PID zero. Shell syntax, launcher default/opt-in/invalid-value
checks, Python compilation, diff whitespace and final product/document
pre-commit checks passed.

Key artifacts:

| Files | Evidence |
| --- | --- |
| `preflight.json`, `current-inventory.json`, `launch-*.json` | Checkout, runtime topology, group identities and commands |
| `baseline.jsonl`, `baseline-summary.json` | Reference before diagnostic hooks |
| `coarse.jsonl`, `coarse-summary.json` | Head, acceptance, postprocess and conserved cross-rank ledgers |
| `fine.jsonl`, `focus-*.jsonl`, `*-geometry.json`, `verification-summary.json` | Independent target components, actual shapes and other streams |
| `ab-reuse.jsonl`, `ab-reuse-summary.json`, `ab-reuse-build.json` | Same-instance metadata A/B and candidate capture |
| `align_candidate.py`, `align_microbench.py`, `align-microbench.json` | Isolated prototype, CPU reference and raw timings |
| `ab-align-client*.log`, `ab-align-build.json`, `server-diagnostic-k3.log` | Failed serving attempt, not qualified throughput evidence |
| `pytest-all-final.log`, `precommit-final.log`, `launcher-switch-validation.json`, `pre-reuse/` | Tests, lint, switch validation and scoped source snapshots |
| `restored*.json*`, `smoke*.json`, `delivery-validation.json`, `delivery-client.log`, `server-restore-k3.log` | Ordinary serving regression and delivery checks |

Full 256K inputs, four simultaneous full 256K requests, randomized acceptance
and general model accuracy were not evaluated by this bounded campaign.
Earlier [MTP](qwen38-autoround-mtp-investigation-2026-10-06.md),
[phase](qwen38-autoround-mtp-phase-breakdown-2026-10-06.md) and
[K=2](qwen38-autoround-mtp-k2-investigation-2026-10-07.md) reports retain the
independent results that precede this target-verification work.
