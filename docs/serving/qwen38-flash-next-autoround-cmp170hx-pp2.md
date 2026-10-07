# Qwen3.8 Flash-Next AutoRound on two CMP170HX GPUs

The local `Qwen3.8-Flash-Next-W4A16-AutoRound` checkpoint uses the existing
Qwen4Exp implementation and INC quantization loader. On SM80 with TP1, its
symmetric INT4/group-128 routed experts select `AutoGPTQMoEMethod` and the
Marlin WNA16 backend. Attention, shared experts, hyper-connections, PLE and MTP
weights are stored in BF16 in the checkpoint. No model or quantization source
change is required for this format.

## Start

From `/mnt/data/apps/vllma100/vllm170`:

```bash
scripts/serve_qwen38_flash_next_autoround_cmp170hx_pp2.sh
```

The launcher follows the supplied NVFP4 serving configuration, with the model
path, served name and chat template changed to AutoRound. It explicitly selects
`--quantization inc --dtype bfloat16`. Defaults are physical GPUs `2,3` in PCI
bus order, PP2/TP1, the `26,22` layer partition, CPU PLE offload, MTP3,
`FULL_AND_PIECEWISE` graphs with capture sizes `[1,2,4,8,12,16]`, prefix caching,
262144 maximum model length, four sequences and 4096 max batched tokens.
GPU memory utilization now defaults to `0.90`, reserving space for NCCL
communicator initialization and runtime allocations.
Set `MTP_TOKENS=0` for the measured MTP-off decode recipe.
It retains the `qwen3_coder` tool parser, `qwen3` reasoning parser and thinking
template settings from the supplied command.

Check the checkpoint and print the command without starting the server:

```bash
CHECK_ONLY=1 scripts/serve_qwen38_flash_next_autoround_cmp170hx_pp2.sh
```

The check opens every index-referenced safetensors file and verifies its tensor
names and structure. Shard names follow the index rather than a contiguous
numeric sequence: this checkpoint uses `model_extra_tensors.safetensors` and
does not reference `model-00002-of-00017.safetensors`.

Settings can be overridden through environment variables. For example:

```bash
CUDA_VISIBLE_DEVICES=4,5 PORT=26662 MAX_MODEL_LEN=8192 MAX_NUM_SEQS=1 \
  scripts/serve_qwen38_flash_next_autoround_cmp170hx_pp2.sh
```

Select free GPUs and a free port before starting. Additional vLLM arguments
can be appended to the script invocation.

## Earlier MTP3 validation on 2026-10-06

The service restoration described in this historical section predates the
user's instruction to keep DeepSeek stopped. The current MTP investigation
keeps DeepSeek inactive and runs the AutoRound model on port 26661.

- All 17 indexed files and 224280 indexed tensors are present. Headers, tensor
  placement and physical payload lengths agree. File bytes total
  `181196007016`; the index's logical `metadata.total_size` is not the physical
  file size.
- The CPU PLE embedding alone contains `102400491520` bytes (95.37 GiB) of
  BF16 weights. Reserve additional host memory for loading and the runtime.
  MTP checkpoint weights are BF16 and total `5214301696` bytes.
- The launcher's complete argument list passed the real serving CLI parser,
  chat-template validation and engine configuration construction. Target
  expert dispatch selects Marlin, and draft expert/attention/projection
  configuration stays unquantized.
- Existing focused tests passed: AutoRound unit tests, 73 passed/9 skipped;
  Qwen4Exp configuration and draft FP8 LM-head tests, 25 passed. The upstream
  model-download test could not run with uncached models in the offline
  environment. Shell syntax and pre-commit checks passed for the launcher.
- The full launcher configuration loaded successfully on physical GPUs 2/3,
  including CPU PLE, PP2, MTP3 and CUDA Graph capture. The workers reported
  39.01/39.41 GiB for model loading; total GPU use after requests was
  59999/60965 MiB including KV caches and runtime allocations. `/health` and
  `/v1/models` returned 200 with the expected AutoRound model identity.
- All six bounded request checks passed: default thinking returned reasoning
  and the correct final answer; automatic tool choice returned
  `get_weather({"city":"上海市"})`; four simultaneous arithmetic requests
  returned the expected answers. MTP counters increased by 24 drafts,
  72 draft tokens and 66 accepted tokens. These are serving smoke checks,
  not a model accuracy evaluation or throughput benchmark.

Startup reported a GPU KV cache of 1184607 tokens and estimated concurrency
of 4.52 requests at 262144 tokens each. Full 256K inputs and four simultaneous
256K requests have not been tested; the startup estimate is not a measured
capacity result.

Before validation, `dsv41f-manual.service` owned GPUs 2 through 6 and port
26661, leaving approximately 24 GiB of available host memory. Stopping that
service released its GPU workers and raised available host memory to 228 GiB,
allowing the 95.37 GiB CPU PLE table to load. The AutoRound run was temporary.
After validation, it was stopped and GPUs 2/3 returned to 6 MiB with no
remaining compute processes. DeepSeek was restored from its original command
and environment snapshot. Its health and model endpoints returned 200, all
five GPU workers were present, and a generation check returned the correct
arithmetic answer.

Readiness artifacts are under
`/mnt/data/apps/vllm-profiles/qwen38-autoround-readiness-20261006/`:
`launch-check.txt`, `checkpoint-validation.json`, `config-validation.json` and
`config-validation.log` record preflight checks; `server.log`,
`runtime-validation.json`, `smoke-results.json`, `smoke-summary.json` and
`metrics-before.txt`/`metrics-after.txt` record the actual serving run.
`autoround-cleanup.json`, `deepseek-restore-launch.json` and
`deepseek-restore-validation.json` record cleanup and service restoration.

## Earlier MTP3 benchmark on 2026-10-06

The earlier MTP3 recipe was measured on physical GPUs 2/3. Requests used the local
chat template with thinking disabled, temperature 0, exact input token counts
and 128 output tokens. Each length had one warmup and three measured requests;
the table reports medians. A unique cache salt per request kept prefix cache
hits at zero while retaining the launcher's prefix caching setting.

| Input tokens | TTFT (s) | Prefill estimate (tok/s) | Per-request decode (tok/s) |
| --- | ---: | ---: | ---: |
| 1024 | 0.322 | 3180 | 95.7 |
| 32768 | 3.468 | 9448 | 75.9 |
| 131072 | 13.457 | 9740 | 75.6 |

For four concurrent 1024-token requests, measured over three waves, median
per-request decode was 79.8 tok/s. Aggregate throughput in the window where
all four requests were decoding was 327.0 tok/s. End-to-end output throughput,
including prefill, scheduling and HTTP, was 197.2 tok/s, with a measured range
of 89.3 to 208.5 tok/s. The first measured wave had higher TTFT; it remains in
the recorded samples.

TTFT measures receipt of the first SSE block containing output token IDs.
Prefill is estimated as input tokens divided by TTFT, including HTTP and the
first decode step. Decode excludes the entire first SSE block: MTP can deliver
multiple tokens in one block. A verification code at 40% of each input was
retrieved correctly in all 21 measured requests. Outputs continued with prose;
no special output tokens appeared in these fixed-length measurements.

Results, timing definitions, workload details and raw records are in
`/mnt/data/apps/vllm-profiles/qwen38-autoround-speed-20261006/`:
`report.md`, `summary.json`, `results.jsonl`, `metadata.json`, `validation.json`
and `bench_speed.py`. Initial needle-only trials and the client SSE line-limit
failure are retained in `preliminary/` and excluded from the reported results.
Full 256K inputs remain untested.

## Upstream optimization integration

The launcher also enables `VLLM_QWEN4_EXP_QSA_MERGED_PROJECTION=1` and
`VLLM_QWEN4_EXP_QSA_FUSED_PREPARE=1`. Runtime gates restrict these paths to
the qualified SM80/TP1 shapes and BF16 QSA recipe; merged projection additionally
requires unquantized projections and no LoRA. Either variable can be set to
`0` before restarting to recover the corresponding previous path.

Source commits, hardware qualification, validation and paired serving results
are recorded in
[the integration report](../design/qwen38-autoround-upstream-optimizations-2026-10-06.md).

## Optional MTP-off decode optimization

Setting `MTP_TOKENS=0` disables MTP and enables
`VLLM_MOE_ALIGN_M1_E512_SM80=1`.
The library default for this alignment is off. Its runtime gate requires SM80,
int32 contiguous route IDs of shape `(1, 10)`, 512 experts, block size 8,
no expert map, no extra sorted-ID padding and invalid-expert filtering.
Other cases retain their previous dispatch. The MTP3 launcher does not enable
this flag automatically.

Paired serving measurements on the same loaded PP2 instance, with four
alternating A/B pairs per workload, found:

| Input tokens | Concurrency | Baseline decode tok/s | Candidate decode tok/s | Median paired gain |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 1 | 80.17 | 80.88 | 0.88% |
| 32768 | 1 | 78.49 | 79.19 | 0.99% |

Four-concurrent-request results varied; this change has no demonstrated
concurrency benefit. Alignment metadata matched independent references, and
16 real-weight MoE comparisons across both PP ranks were bitwise identical.
Set `VLLM_MOE_ALIGN_M1_E512_SM80=0` before restart to roll back the alignment.

The upstream standard GDN path remains optional through
`VLLM_QWEN4_EXP_GDN_STANDARD_DECODE=1`, with exact model/hardware gates and a
default of `0`. It reduced single-request decode speed in this experiment.
Detailed hotspot measurements, accepted/rejected candidates and raw artifacts
are in [the decode investigation](../design/qwen38-autoround-decode-investigation-2026-10-06.md).

## Current MTP recipe

The launcher defaults to MTP3 with the existing FP8 draft LM head and automatic
fused multi-step decode. Top-k index sharing stays off. The main LM head stays
BF16. The current ordinary service uses physical GPUs 2/3, PP2/TP1, the 0.90
memory setting and port 26661, without the diagnostic worker or dev RPC.
DeepSeek remains stopped.

The launcher now defaults to `QWEN_MTP_PP_OPTIMIZATIONS=1`, enabling a fixed-shape
draft-token relay copy that avoids CPU synchronization from GPU boolean indexing.
Four paired A/B measurements per workload found median decode gains of 51.60%
at 1K/C1, 38.51% at 32K/C1, and 31.76% at 1K/C4. Set this variable to `0` before
restarting to recover the previous relay copy. The library default remains off,
with model, SM80, PP2/TP1 and MTP2/MTP3 qualification gates. MTP2 was added
as an independent draft-count experiment; MTP3 remains the default.

`QWEN_MTP_PREPARE_AHEAD` and `QWEN_MTP_PREFIX_METADATA` are separate experimental
switches, both defaulting to `0`. Their additional gains after the relay fix did
not repeat consistently, so they are not part of the default recipe. Independent
phase measurements, comparisons, tests and evidence are in
[the MTP phase report](../design/qwen38-autoround-mtp-phase-breakdown-2026-10-06.md).

The same-instance campaign used 256 output tokens and recorded 306 formal
requests, all passing the fixed needle checks. Initial MTP3 medians were
100.07 tok/s at 1K, 80.10 tok/s at 32K, and 332.09 aggregate tok/s for four
concurrent 1K requests. Acceptance and free-text output variation affect speed;
these measurements do not establish that K=3 is optimal for every workload.

Index reuse, BF16 draft head, separate-step decode and an INT4/g32 draft head
did not qualify for default promotion. INT4 reduced the draft proposal from
about 2.59 to 2.18 ms, but its paired serving throughput decreased. Keep the
FP8 head and fused path. All candidates, paired results and validation limits
are in [the MTP investigation](../design/qwen38-autoround-mtp-investigation-2026-10-06.md).

The ordinary service was revalidated after the relay change: three measured
waves gave medians of 142.86 tok/s at 1K/C1, 116.71 tok/s at 32K/C1, and
403.25 total tok/s at 1K/C4. Thinking, automatic tool choice, concurrent
arithmetic, and a single 128K-input needle check passed.

Set `MTP_TOKENS=0` before restart to return to the earlier decode-only recipe.
The current ordinary MTP3 startup KV estimate is 980560 tokens, or 3.74 full
256K requests;
four simultaneous full 256K inputs have not been validated.

## Independent K=2 validation on 2026-10-07

The default stays at MTP3. With the relay fix enabled in both configurations,
a K=3 / K=2 / K=3 comparison produced these median decode rates over three
measured waves per workload:

| Workload | K=3 before | K=2 | K=3 restored |
| --- | ---: | ---: | ---: |
| 1K/C1 | 146.00 | 137.10 | 140.24 |
| 32K/C1 | 119.91 | 112.62 | 117.21 |
| 1K/C4 aggregate | 405.62 | 403.46 | 411.99 |

K=2 reduces draft proposal time by about 0.7–0.8 ms, but does not improve
median effective throughput against either reference in these samples.
Keep `MTP_TOKENS=3`. `MTP_TOKENS=2` remains available for independent
experiments; only the relay gate extends to K=2. Prepare-ahead and
prefix-metadata remain K=3-only and default off. Original capture sizes
already produce K=2 FULL verification graphs at 3/6/9/12 tokens.

The restored ordinary service passed thinking, tool-call and concurrent
arithmetic checks on port 26661. Dev RPC is disabled and DeepSeek remains
inactive. The [K=2 investigation](../design/qwen38-autoround-mtp-k2-investigation-2026-10-07.md)
records acceptance, configuration drift, independent phase measurements,
tests and the remaining validation limits.

## Independent verification work on 2026-10-07

Target verification was split into PP0/PP1 forward, target head, acceptance,
sampled postprocess, metadata and individual GDN/MoE/HC/QSA/PLE components.
At 1K/C1 the target forward intervals were approximately 7.28/6.05 ms,
the target head 0.79 ms and rejection 0.11 ms. The leading measured target
components are GDN projections and routed Marlin W13/W2 computation.

`QWEN_MTP_METADATA_REUSE` is a separate experimental switch, defaulting to `0`.
It adapts upstream MRV2 reuse of batch metadata across matching GDN cache
groups, while retaining each group's physical state IDs and persistent buffers.
The gate requires the qualified SM80 AutoRound K=3 PP2/TP1 recipe and the exact
GDN builder. Same-instance arm medians changed by +0.79% at 1K/C1, +0.81%
at 32K/C1 and -0.04% at 1K/C4; paired results did not establish a stable
gain. Keep it off in ordinary serving. Set it back to `0` before restart
to roll back an experiment.

An M=8/12/16 alignment prototype improved isolated kernel latency, but its
serving experiment failed before formal candidate measurements. It remains
an artifact-only prototype. The
[verification report](../design/qwen38-autoround-verification-investigation-2026-10-07.md)
records the independent measurements, upstream qualification, failed attempt,
127 passing tests and retained recipe. The ordinary service uses port 26661
without diagnostic RPC; DeepSeek stays inactive.

The [acceptance investigation](../design/qwen38-autoround-acceptance-investigation-2026-10-07.md)
compares complete MTP checkpoint parameters and records eight short requests
against the ordinary service. Acceptance depends strongly on output content;
the current evidence does not establish a weight-loading or state-alignment bug.

The later [same-input NVFP4 comparison](../design/qwen38-autoround-nvfp4-acceptance-comparison-2026-10-07.md)
found no general acceptance increase. At the user's request, that comparison
left NVFP4 running on port 26661 and stopped AutoRound; the AutoRound launcher
above remains its documented recipe.
