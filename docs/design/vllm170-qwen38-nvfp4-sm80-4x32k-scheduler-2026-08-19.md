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
