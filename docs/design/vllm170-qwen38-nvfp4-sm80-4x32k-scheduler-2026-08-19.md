# Qwen3.8-27B-NVFP4 SM80 4x32K scheduler study

## Scope

This study uses one CMP 170HX (SM80) and
`/media/z/data6/models/Qwen3.8-27B-NVFP4` with exact 32,768-token prompts,
256 generated tokens, four concurrent requests, MTP5, FlashInfer attention,
Marlin NVFP4 linear kernels, FP8 KV cache, and FULL plus PIECEWISE CUDA graphs.
All quality probes ask each request to recover a distinct passphrase.

The main result is that the observed collapse was scheduler interference, not
unused GPU capacity. A single 8K prefill chunk already saturates this GPU. With
the long-prefill cap disabled, FCFS scheduling lets an early request repeatedly
consume the token budget while its decode is stalled by later 32K prefills.

## Measurement definitions

- Aggregate prefill is total prompt tokens divided by the interval from the
  first request start to the last request's first token.
- Mixed aggregate decode counts exact returned token IDs from the service's
  first emitted token until all requests finish. It includes prefill/decode
  interference and is the primary production metric for this workload.
- Tail aggregate decode counts only tokens emitted after every request has
  produced its first token. It measures the decode-only tail and must not be
  used to hide earlier stalls.
- Completion tokens divided by whole-request wall time is not decode speed;
  most of that interval is the 131,072-token aggregate prefill.

The benchmark now requests returned token IDs and records their SSE arrival
times. The previous calculation assigned all four requests' decode tokens to
the short interval after the last first token, even when earlier tokens were
emitted tens of seconds before that interval.

## Root cause

The baseline `max_num_batched_tokens=8192` configuration had no long-prefill
cap. Four nominally simultaneous requests therefore reached first token at
22.17, 45.06, 63.60, and 73.91 seconds. The early streams then waited behind
later prefill chunks, producing per-request perceived decode rates between
4.66 and 84.80 tokens/s. The baseline mixed aggregate decode is approximately
18.6 tokens/s when reconstructed from the old request timestamps.

The hybrid GDN/full-attention cache uses a 1,616-token page on this model. Four
contention-safe chunks therefore need at least 6,464 scheduled token slots.
Reducing the global budget does not expose idle compute:

| max batched | long cap | prefill tok/s | mixed decode tok/s | tail decode tok/s | TTFT range s | wall s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | disabled | 1,773.36 | ~18.6 | invalid old metric | 22.17-73.91 | 77.17 |
| 8,192 | adaptive 2,048 | 1,801.29 | 275.99 | 274.01 | 72.65-72.77 | 76.36 |
| 4,096 | adaptive 2,048 | 1,783.42 | 25.35 | 215.92 | 37.19-73.49 | 77.55 |
| 2,048 | adaptive 2,048 | 1,725.84 | 17.01 | 182.91 | 19.43-75.95 | 79.56 |

During the 4,096-token run, 30 consecutive one-second samples reported
99-100% SM utilization. The requests split into two TTFT groups because the
budget can advance only about two aligned long-prefill chunks per iteration.
At 2,048 tokens they formed a four-level staircase. A four-way first chunk
would require caps of at most 1,024 and 512 tokens respectively, which would
force finer sub-block execution throughout contention. These variants were
not selected because 8,192 already restores fairness without reducing prefill
capacity.

Setting the budget to exactly `1616 * 4 = 6464` is valid but leaves no
scheduler reserve after four aligned prefill chunks. MTP5 can require up to
`4 * (1 + 5) = 24` additional scheduled decode/draft tokens, before accounting
for prompt tails or arrival skew. vLLM consequently warns that 6,464 scheduled
tokens may be suboptimal with this speculative configuration. It is a useful
capacity lower bound, not the production recommendation; 8,192 retains 1,728
tokens of mixed-work slack.

Aggregate prefill should not be expected to scale above the single-request
rate here. The 8K single-request prefill already fills the scheduler budget
and saturates the GPU. The useful acceptance criterion is preserving that rate
while making aggregate decode exceed the single-request rate and bounding TTFT
spread.

## Scheduler change

`--enable-adaptive-long-prefill` makes the configured long-prefill threshold
contention-aware:

1. The first chunk is always capped, leaving one scheduler step for concurrent
   arrivals to join.
2. While more than one request is unfinished, every long prefill remains
   capped so all four requests advance together.
3. After contention clears, a lone request's later chunks use the full token
   budget, avoiding the single-request penalty of a permanent small cap.

With an 8,192-token budget and a 2,048-token threshold, the GDN alignment turns
the concurrent chunks into four 1,616-token units. The final run produced a
0.112-second TTFT spread. Two earlier repeats measured mixed aggregate decode
at 265.43 and 276.12 tokens/s, so the result is repeatable rather than a single
favorable sample.

Single-request performance remains close to the uncapped baseline:

| mode | prefill tok/s | decode tok/s | TTFT s | wall s |
| --- | ---: | ---: | ---: | ---: |
| uncapped baseline | 1,813.00 | 119.81 | 18.07 | 20.20 |
| adaptive 2,048 | 1,797.00 | 118.13 | 18.23 | 20.39 |

The adaptive single-request deltas are -0.88% prefill and -1.40% decode. The
final four-request run passed 4/4 passphrase probes with zero cross-request
leakage. MTP accepted 55.74% of drafted tokens with a mean accepted length of
3.79.

## Production command

Replace `<GPU_ID>` with the physical CMP 170HX selected for this instance:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<GPU_ID> \
.venv/bin/vllm serve /media/z/data6/models/Qwen3.8-27B-NVFP4 \
  --served-model-name qwen38-nvfp4 \
  --host 0.0.0.0 --port 8400 \
  --dtype half --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 4 \
  --long-prefill-token-threshold 2048 \
  --enable-adaptive-long-prefill \
  --gpu-memory-utilization 0.90 \
  --attention-backend FLASHINFER \
  --linear-backend marlin \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --compilation-config \
    '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[6,12,18,24]}' \
  --jit-monitor-mode error --jit-monitor-verbose
```

Runtime logs confirmed `MarlinNvFp4LinearKernel`, FlashInfer native SM80
decode with FP8 KV cache, MTP5, and all requested FULL and PIECEWISE graph
capture sizes. No inference-time JIT compilation was reported.

## Validation

```bash
.venv/bin/python -m pytest tests/v1/core/test_scheduler.py -q \
  -k 'adaptive_long_prefill or schedule_concurrent_partial_requests'
.venv/bin/python -m pytest \
  tests/benchmarks/test_deepseek_v4_concurrent_longctx.py -q
.venv/bin/pre-commit run ruff-format --files \
  benchmarks/deepseek_v4/bench_concurrent_longctx.py \
  tests/benchmarks/test_deepseek_v4_concurrent_longctx.py \
  tests/v1/core/test_scheduler.py tests/v1/core/utils.py \
  vllm/config/scheduler.py vllm/engine/arg_utils.py \
  vllm/v1/core/sched/scheduler.py
.venv/bin/pre-commit run ruff-check --files \
  benchmarks/deepseek_v4/bench_concurrent_longctx.py \
  tests/benchmarks/test_deepseek_v4_concurrent_longctx.py \
  tests/v1/core/test_scheduler.py tests/v1/core/utils.py \
  vllm/config/scheduler.py vllm/engine/arg_utils.py \
  vllm/v1/core/sched/scheduler.py
```

The targeted scheduler tests passed 3/3 and the benchmark tests passed 2/2.
The complete scheduler file passed 103 tests; 35 multimodal tests could not
resolve `llava-hf/llava-1.5-7b-hf` in the offline environment and failed before
constructing a scheduler.

## NVFP4 target with Qwen3.8 DSpark-7

`/media/z/data6/models/Qwen3.8-27B-DSpark` is a five-layer BF16 Qwen GQA
drafter with a seven-token block. Its published config uses the generic
remote-code architecture name `DSparkDraftModel`; vLLM previously interpreted
that name as the DeepSeek-V4 drafter and failed while looking for `hc_mult`.
The config compatibility fix maps only generic Qwen3 DSpark configs to the
native `Qwen3DSparkModel` loader. The NVFP4 target then loads through
`MarlinNvFp4LinearKernel`; target and draft attention both select FlashInfer.

The following greedy runs use the same scheduler settings as above, DSpark-7,
FP8 KV cache, exact binary 1K/32K/128K prompts, and 256 returned tokens. The
target-only comparison uses the automatically selected legacy runner while
DSpark requires the V2 runner, so the deltas are operational comparisons, not
an isolated drafter-only microbenchmark. The first 1K single-request DSpark
run also contains cold Triton JIT and is excluded from steady-state decisions.

| shape | mode | prefill tok/s | mixed decode tok/s | tail decode tok/s | wall s | mean accepted | quality |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4x1K | target only | 2,102.90 | 182.41 | 203.61 | 6.96 | - | 4/4 |
| 4x1K | DSpark-7 draft eager | 1,965.53 | 196.14 | 197.53 | 7.24 | 2.66 | 4/4 |
| 1x32K | target only | 1,858.83 | 51.51 | 51.51 | 22.58 | - | 1/1 |
| 1x32K | DSpark-7 draft eager | 1,801.03 | 40.00 | 40.00 | 24.57 | 1.83 | 1/1 |
| 1x32K | DSpark-7 FULL graph | 1,799.79 | 53.79 | 53.79 | 22.95 | 2.15 | 1/1 |
| 4x32K | target only | 1,849.42 | 138.78 | 165.42 | 76.94 | - | 4/4 |
| 4x32K | DSpark-7 draft eager | 1,794.91 | 117.02 | 118.60 | 81.40 | 1.89 | 4/4 |
| 4x32K | DSpark-7 FULL graph | 1,790.01 | 136.03 | 133.80 | 80.52 | 1.91 | 4/4 |
| 1x128K | target only | 1,250.34 | 44.08 | 44.08 | 110.61 | - | 1/1 |
| 1x128K | DSpark-7 draft eager | 1,223.70 | 39.20 | 39.20 | 113.62 | 1.95 | 1/1 |
| 4x128K | DSpark-7 draft eager | 1,191.29 | 69.38 | 96.16 | 450.26 | 2.19 | 4/4 |

The 4x128K DSpark run completed with a 435.52-440.10 second TTFT range and
peaked at approximately 99.4% KV-cache use. The requested target-only 4x128K
comparison was cancelled and is not reported as a completed result.

Greedy DSpark and target-only output hashes matched for all 1K and 32K probes.
At 128K the wording diverged after the correct answer, while both paths still
recovered the passphrase with no cross-request leakage. This is consistent with
the different target runner paths reaching a close-logit branch; it is not
evidence of a failed needle lookup.

FlashInfer non-causal draft FULL graph support is now wired for SM80. The
metadata builder owns stable CUDA-side indptr, index, and last-page buffers for
each uniform `(batch size, draft length)` capture shape. The graph wrapper does
not request `fixed_split_size`: FlashInfer rejects fixed split planning in CUDA
graph mode because its padded batch contract cannot be preserved. Split-KV
planning remains enabled for the graph-specific wrapper, while eager prefill
keeps its existing plan settings.

The real NVFP4 plus DSpark-7 service captured all four draft FULL graphs and all
four target FULL graphs for combined capture sizes
`[7, 8, 14, 16, 21, 24, 28, 32]`. Both 32K replay probes completed without a
graph or attention error. At 4x32K, where mean accepted length stayed nearly
constant at 1.89 versus 1.91, FULL graph reduced the measured round cost from
15.50 to 13.59 ms, increased mixed aggregate decode from 117.02 to 136.03 tok/s
(+16.25%), and increased tail aggregate decode from 118.60 to 133.80 tok/s
(+12.81%). Prefill remained flat within 0.3%, as expected. The larger 1x32K
decode gain also includes an acceptance-length change, but its round cost still
fell from 45.87 to 39.84 ms.

Validation for this path:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python -m pytest \
  tests/v1/attention/test_flashinfer_dcp_spec_reorder.py -q
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python -m pytest tests/kernels/attention/test_flashinfer.py -q \
  -k 'fast_plan_decode_multi_token_matches_reference or noncausal_prefill_cudagraph_matches_eager_with_padded_batch'
```

The unit suite passed 7/7 and the SM80 kernel selection passed 3/3: MTP3, MTP5,
and the non-causal replay comparison. The replay test captures a two-request
non-causal graph, replans it with one real request and one padded request, and
compares the real output with eager FlashInfer.

### FP8 target acceptance comparison

The same DSpark-7, scheduler, FP8 KV cache, FULL graph shapes, greedy seed 170,
exact 32K prompt, and 256-token output were also tested with
`Qwen3.8-27B-FP8`. The FP8 target loaded through
`MarlinFP8ScaledMMLinearKernel`; FlashInfer captured four target and four draft
FULL graphs.

| shape | target | prefill tok/s | aggregate decode tok/s | ms/round | mean accepted | draft-token acceptance | quality |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1x32K | NVFP4 | 1,799.79 | 53.79 | 39.84 | 2.151 | 16.45% | 1/1 |
| 1x32K | FP8 | 1,725.42 | 56.72 | 42.01 | 2.411 | 20.16% | 1/1 |
| 4x32K | NVFP4 | 1,790.01 | 136.03 | 13.59 | 1.905 | 12.93% | 4/4 |
| 4x32K | FP8 | 1,709.91 | 135.37 | 15.79 | 2.209 | 17.27% | 4/4 |

The target format materially changes proposal agreement. FP8 raises mean
accepted length by 12.08% at 1x32K and 15.94% at 4x32K; draft-token acceptance
rises by 3.71 and 4.34 percentage points. At 4x32K, position-wise acceptance
changes from `[72.25, 11.36, 2.79, 1.68, 1.30, 0.56, 0.56]%` for NVFP4 to
`[66.24, 20.00, 12.04, 8.17, 6.45, 4.09, 3.87]%` for FP8. The first proposal
is accepted less often, but FP8 preserves agreement much farther into the
seven-token block.

That acceptance improvement does not produce a 4x throughput gain on SM80.
FP8 weight-only Marlin increases the measured round cost by 16.17% and reduces
prefill by 4.48%, leaving aggregate decode 0.49% below NVFP4. All five needle
probes passed with zero cross-request leakage. The FP8 checkpoint lacks a KV
q scaling factor, and vLLM reported uncalibrated q/prob fallback values of 1.0;
this is an accuracy caveat for both semantic and acceptance interpretation.

Neither DSpark pairing is recommended for production yet. The FP8 target is a
better semantic match for the documented drafter, but mean accepted length is
still only 2.21 at 4x32K. The established NVFP4 MTP5 path reaches 3.79 and
approximately 265-276 mixed aggregate decode tok/s, versus 2.21 and 135.37
tok/s for FP8 plus FULL-graph DSpark-7.

The JIT warmup gap is now closed. V2 runner startup warms the runtime-dtype
rejection path and both DFlash and DSpark input preparation. The latter reuses
the initialized block-table layout and compiles block sizes 16, 32, 64, 128,
and 256 before the JIT monitor is activated. A fresh-cache CMP 170HX run with
the default `enable_jit_warmup=true` and `--jit-monitor-mode error` completed
both 1x1K and 4x1K requests without request-time compilation. Their measured
aggregate decode rates were 85.69 and 176.70 tok/s, respectively. Production
commands must not include
`--kernel-config '{"enable_jit_warmup":false}'`; a normal readiness probe is
still recommended, but same-shape traffic warmup is no longer required to avoid
these speculative-kernel JITs.
