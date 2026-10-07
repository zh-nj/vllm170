# Qwen3.8 Flash Next AutoRound: independent K=2 MTP experiment

Retain K=3 as the default. K=2 saves draft proposal time but did not improve
the median throughput of any tested workload against either K=3 reference.

This experiment follows the [MTP phase investigation](qwen38-autoround-mtp-phase-breakdown-2026-10-06.md).
It isolates draft count after removing the PP draft-relay synchronization.
The target is `/media/z/data6/models/Qwen3.8-Flash-Next-W4A16-AutoRound`,
CMP170HX SM80, physical GPUs 2/3 in PCI order, PP2/TP1, layer partition
`26,22`, BF16 activations, INC/AutoRound W4A16 experts, and CPU PLE offload.

## Source and configuration boundaries

The target checkout is `e2e327dc53b93acb85c8b56b526d369eb5461421` with local
changes. The inspected source checkout `/mnt/data/apps/vllm` is
`31ebe18bb63303be83137a59f668ca9812ac3579`; its locally cached `origin/main`
is `0e468adb43d60484b5a6546e0124b686fd76baf3`. No claim is made about newer
remote commits. The actual serving path is MRV2 `MTPSpeculator` and
`AutoRegressiveSpeculator`, under `vllm/v1/worker/gpu/`, rather than the older
`MultiModuleMTPSpeculator` in `gpu_model_runner.py`.

Draft count is fixed at engine construction. `DraftModelSpeculator` allocates
draft buffers with width K; `PPHandler` allocates a sampled width K+1 and
packs sampled tokens, K draft tokens and two counters. The target and draft
prefill graph managers use query length K+1, while the draft decode manager
captures the remaining K-1 steps with query length 1. Changing a Python
attribute on a live K=3 engine does not make a valid K=2 experiment.

| Component | K=3 | K=2 |
| --- | --- | --- |
| Target verification query length | 4 | 3 |
| Draft prefill query length | 4 | 3 |
| Draft decode forwards after step 0 | 2 | 1 |
| Draft LM head applications | 3 | 2 |
| PP packed relay width | 9 | 7 |
| Target uniform FULL graphs for 1–4 requests | 4/8/12/16 tokens | 3/6/9/12 tokens |

The original capture sizes `[1,2,4,8,12,16]` suffice for both configurations:
`CUDAGraphManager._init_candidates` rounds each candidate up to its uniform
query length. Adding explicit K=2 buckets is unnecessary and would also
change draft decode padding, introducing another independent variable.

The only product change in this round extends the opt-in fixed-shape relay
gate to K=2. All model, quantization, dtype, hardware, topology and maximum
sequence gates remain in place. Prepare-ahead and prefix-metadata candidates
remain K=3-only and disabled. The launcher continues to default to K=3.
There is no model-weight, LM-head, sampling, top-k-sharing or fused-step change.

The source audit found no additional upstream QSA or fused-draft change ready
to transplant into this SM80 path. Existing implementations already cover
merged QSA projections, fused preparation and multi-step capture. Newer cached
MoE weighted-sum tuning (`9cf087fd14bf`) is called by Humming indexed experts;
batched alignment tuning (`8a7f98c8c3ae`) affects a different alignment API.
The current service logs confirm the ordinary Marlin WNA16 MoE backend.

MTP still passes `enable_sm80_hc_down=False`, but the previous exact-shape
HC comparison bounded its potential saving to approximately 24 microseconds
per K=3 round. That is too small to justify another full checkpoint reload.
Earlier FP8 head plan retuning also remained below 1%, while INT4 draft-head
latency savings did not improve effective serving speed. Those independent
candidates are not repeated or combined with this experiment.

## Hypothesis and measurement

The fresh K=3 reference has median third-position accepted probabilities of
0.323, 0.150 and 0.294 per draft round at 1K/C1, 32K/C1 and 1K/C4.
If the first two probabilities remain unchanged, dropping the third draft
requires a round-cost reduction greater than approximately 12.45%, 7.03%
and 11.50%, respectively. This is a hypothesis rather than a prediction:
K=2 changes target and draft-prefill shapes and can change acceptance.

The experiment uses K=3 / K=2 / K=3 configuration order, with a full PP2
reload between configurations. Each configuration uses the same tokenized
1K/C1, 32K/C1 and 1K/C4 prompts, 256 generated tokens, greedy sampling,
one warmup and three measured waves. Every request gets a unique cache salt;
prefix-hit deltas must be zero. Record per-position acceptance counters,
GPU clock/temperature/power snapshots, output token IDs and stream timing.
C4 throughput is the common decode window across all requests.

K=2 formal timing runs before the first worker diagnostic RPC. Its extension
does not install wrappers or allocate trace buffers until that RPC, and its
directory contains no `sitecustomize.py`. CPU/GPU trace windows and newly
captured detail graphs run afterwards and are excluded from formal speed
comparisons. GPU traces use graph-compatible `%globaltimer` stamps; CPU
and GPU interval parents are not added together. NCCL completion intervals
include producer waiting and are not interpreted as wire transmission time.

## Uninstrumented configuration comparison

Values are median decode tok/s over three measured waves. C4 is aggregate
throughput over the common decode window, not per-request speed.

| Workload | K=3 before | K=2 | K=3 after | K=2 vs before | K=2 vs after |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1K/C1 | 146.00 | 137.10 | 140.24 | -6.10% | -2.24% |
| 32K/C1 | 119.91 | 112.62 | 117.21 | -6.08% | -3.92% |
| 1K/C4 | 405.62 | 403.46 | 411.99 | -0.53% | -2.07% |

K=3 before/after throughput drift is -3.95%, -2.25% and +1.57%, respectively.
The measurement is a bounded configuration A/B/A across reloads, not a
same-instance paired experiment or a statistically precise effect estimate.
In particular, C4 wave ranges overlap. The result rejects default promotion
of K=2; it does not prove that K=3 is universally optimal.

At 1K/C1, median accepted lengths are 2.545 / 2.277 / 2.438 for K=3 before /
K=2 / K=3 after. At 32K/C1 they are 2.133 / 1.903 / 2.073. The per-wave
`1000 * L / decode_tok_s` round-cost indices are 17.467 / 16.689 / 17.445 ms
and 17.792 / 16.969 / 17.687 ms. Thus K=3 round cost repeats closely even
while acceptance and throughput drift; the K=2 cost reduction does not
recover its lower accepted length in these samples. For C4 this expression
is only an aggregate cost index, not an actual PP batch-round duration.

Formal before/after request snapshots show GPU2 at 1485–1500 MHz and GPU3
at 1470–1485 MHz, with temperatures 35–50 C. Clocks were not locked; these
are endpoint snapshots rather than a continuous frequency trace. All 72
benchmark requests, including 54 formal requests and 18 warmup requests,
passed the needle, exact output-length, no-special-token and prefix-zero
checks. Draft-token/proposal ratios are exactly 3 for both K=3 references
and 2 for K=2.

## Independent phase measurements

K=2 inventory confirms target and draft-prefill FULL graphs at 3/6/9/12
tokens on both relevant ranks. Draft decode graphs remain at 1/2/4 requests;
there is no separate three-request draft decode graph. Relay is enabled on
both ranks, and prepare-ahead remains disabled.

GPU durations below are inclusive medians in milliseconds from the K=2
detail windows. They are diagnostic windows after formal timing, not a sum
of CPU/GPU costs or an uninstrumented throughput estimate.

| Phase | 1K/C1 | 32K/C1 |
| --- | ---: | ---: |
| PP0 target forward | 7.384 | 7.617 |
| PP1 target forward | 6.102 | 6.325 |
| Target BF16 LM head | 0.781 | 0.781 |
| Rejection, including nested stages | 0.112 | 0.112 |
| Draft step 0 forward | 0.530 | 0.565 |
| Draft step 0 FP8 LM head | 0.413 | 0.413 |
| Draft step 1 forward | 0.422 | 0.455 |
| Draft step 1 FP8 LM head | 0.413 | 0.413 |
| Whole draft proposal | 1.946 | 2.013 |

The prior K=3 detail windows measured a 2.665/2.770 ms proposal at these
input lengths. K=2 removes about 0.72/0.76 ms of draft work, but target
verification still consumes approximately 13.5/13.9 ms across PP0 and PP1.
These are separate-load diagnostic comparisons; the formal configuration
comparison determines the throughput decision.

CPU-only windows measure PP0 relay request updates at 0.329/0.330 ms and
target metadata construction at 4.113/4.149 ms. CPU metadata can overlap
previous GPU work, so those durations are not added to GPU forward time.
Nine windows contain 18 rank profiles and six GPU cross-rank ledgers with
448 conserved round intervals; the analyzer reports zero warnings.

GPU instrumentation changes C4 scheduling: the coarse GPU trace mostly
observes two requests per batch, while detail capture alternates three and
one. The CPU-only trace mostly observes four. Therefore C4 phase medians
are not treated as measurements of the same fixed batch shape or as proof
of end-to-end speedup. Only the uninstrumented C4 benchmark supports that
comparison.

## Evidence

Artifacts are under `artifacts/qwen38-autoround-mtp-k2-20261007/` (ignored):
`before-launch.json`, `launcher-validation.json`, `k2-hypothesis.json`,
`k3-before.jsonl`, `k2.jsonl`, `k3-after.jsonl`, the corresponding summaries,
launch commands, service logs, trace JSONL files and validation logs.

The relay tests exercise graph replay with mutable indices, freed slots,
non-contiguous packed sources, int32/int64 tokens and widths 1/2/3. Width 1
is a kernel correctness case, not a serving qualification. The targeted PP
and GDN suites passed 76 tests. The real CLI parser validated K=2 and K=3
with only the relay enabled, relay-disabled fallback, and invalid-switch
rejection. Pre-commit passed for the changed source and test files.

Validation commands:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m pytest \
  tests/v1/worker/test_pp_utils.py \
  tests/v1/attention/test_gdn_metadata_builder.py -q
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5 .venv/bin/python \
  artifacts/qwen38-autoround-mtp-k2-20261007/preflight.py
.venv/bin/pre-commit run --files \
  vllm/v1/worker/gpu/qwen_mtp_utils.py tests/v1/worker/test_pp_utils.py
```

The K=2 startup estimate is 1,051,472 KV tokens, or 4.01 full-length
requests. This is a startup estimate; four simultaneous 256K requests have
not been tested and the estimate is not a serving guarantee.

## Delivery and rollback

K=2 and restored K=3 each passed six functional smoke checks: default
thinking arithmetic, automatic tool call and four concurrent arithmetic
requests. This is bounded functional coverage, not a broad accuracy
evaluation or a claim of bitwise-identical free-text model outputs.

The ordinary service is restored with K=3, relay enabled, prepare-ahead and
prefix-metadata disabled, the FP8 draft head, original graph capture sizes,
physical GPUs 2/3 and `0.0.0.0:26661`. API PID is 2538283; its cgroup owns
PP0 2542778 and PP1 2543357 on the expected physical GPU UUIDs. Health and
model identity pass. Diagnostic RPC returns 404, port 26662 has no listener,
and no diagnostic worker or `PYTHONPATH` is installed. DeepSeek and the
diagnostic service remain inactive. Validation and source hashes are in
`delivery-validation.json`.

`MTP_TOKENS=2` remains an explicit experimental configuration with the same
relay guard. The default remains `MTP_TOKENS=3`; this round adds no default
head, attention, kernel or speculative-count change. The earlier positive
relay optimization stays enabled. Set `QWEN_MTP_PP_OPTIMIZATIONS=0` before
restart to recover the older relay copy, or `MTP_TOKENS=0` for decode-only
operation. Full 256K input and four simultaneous full 256K requests were
not tested in this round.
