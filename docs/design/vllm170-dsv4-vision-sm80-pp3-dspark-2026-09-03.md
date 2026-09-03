# DeepSeek-V4-Flash-Vision-Exp on SM80 PP3 with DSpark

This note records the implementation and validation plan for
`/media/z/data6/models/DeepSeek-V4-Flash-Vision-Exp` in this checkout.  The
target deployment is three CMP170HX cards, pipeline parallelism 3, tensor
parallelism 1, and the checkpoint's three-block DSpark draft.

## Research baseline

The implementation follows the merged upstream vLLM DeepSeek-V4 vision work
([PR #54566](https://github.com/vllm-project/vllm/pull/54566)) and cross-checks
the original out-of-vocabulary sentinel discussion ([issue #54561](https://github.com/vllm-project/vllm/issues/54561)),
SGLang's scheduler/span work ([PR #37253](https://github.com/sgl-project/sglang/pull/37253)),
and the CMP170HX PP reference ([kaka86mm/dsv4-vision-170hx](https://github.com/kaka86mm/dsv4-vision-170hx)).
The upstream PR is the source of truth for model semantics; the SM80 path
retains this repository's portable Triton and Marlin implementations where
the upstream GB200 path uses SM100-specific kernels.

## Checkpoint contract observed locally

- Text architecture: `DeepseekV4ForConditionalGeneration`, 43 target layers.
- Vision tower: 32 layers, hidden size 1024, patch size 14, downsample 3,
  maximum 384 vision tokens.
- The tokenizer reserves in-vocabulary image IDs 129257--129261.  Image
  placeholders are expanded into contiguous spans; two-image processing was
  verified with spans `(2, 311)` and `(315, 106)`.
- The checkpoint contains `mtp.0/1/2`; conversion selects
  `DSparkDraftModel` with three speculative tokens.  These names describe
  checkpoint weights, not a request to use generic MTP mode.

## Implementation decisions

1. Add the upstream-compatible vision processor, tower/aligner wrapper, and
   multimodal model registration under `vllm/models/deepseek_v4/`.
2. Keep image visibility bidirectional for the full image span in sparse SWA;
   text remains causal.  The direct Triton index builder is extended with the
   same visible-span rule and is covered by unit tests.
3. Apply `bias_vl` consistently in both hashed and non-hashed visual routing,
   including the C++ and Triton top-k paths.
4. On PP ranks after rank 0, preserve raw token IDs when the model requires
   them.  This is needed because every target MoE layer consumes `bias_vl` and
   sentinel IDs even though only rank 0 runs the vision encoder.
5. DSpark PP keeps its materialized draft embedding on the last rank when the
   target embedding is a `PPMissingLayer`; the placeholder is never treated as
   a shareable embedding, otherwise the draft would feed integer IDs into mHC.
6. CuTeDSL indexer kernels are selected only on SM90+.  SM80/CMP170HX uses the
   existing Triton path, including the SM80 split sparse-decode workspace.

## Reproducible PP3 launch

```bash
DSV4_MAX_MODEL_LEN=4096 \
DSV4_MAX_BATCHED_TOKENS=1024 \
DSV4_MAX_NUM_SEQS=2 \
DSV4_KV_CACHE_BYTES=800000000 \
DSV4_MODE=benchmark \
benchmarks/deepseek_v4/run_cmp170hx_pp3_vision.sh \
  --per-request-spec-decode-metrics summary
```

The preset selects physical GPUs `1,2,3` with `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
partition `15,15,13`, TP1, FP8 KV cache, sparse MLA Triton attention, Marlin
MoE, and DSpark probabilistic width 3.  Override `DSV4_DSPARK_TOKENS` only
with 3 or a larger multiple of 3.

## Validation ladder

Before a long-context benchmark, run the following in order:

```bash
.venv/bin/python -m compileall -q vllm tests
.venv/bin/python -m pytest tests/models/test_deepseek_v4_dspark.py -q
.venv/bin/python -m pytest tests/v1/attention/test_deepseek_v4_swa_visible.py -q
.venv/bin/python -m pytest tests/kernels/moe/test_topk_softplus_sqrt.py -q
```

Then confirm `/health` and `/v1/models`, send a text control request, and send
one request containing two base64 image URLs.  Record prompt token count,
image span count, output, latency, and `usage.prompt_tokens_details` (when
provided) so DSpark acceptance is not inferred from a successful HTTP status.
The short smoke run is a bring-up gate, not an OCRBench or throughput claim;
those require a separate fixed-seed evaluation matrix and long-context memory
budget.

## CMP170HX smoke result (2026-09-03)

The command above completed model loading, vision-encoder profiling, sparse-MLA
warmup, target CUDA-graph capture, DSpark draft capture, and API startup on
physical GPUs `1,2,3`.  The 800 MB per-rank KV allocation exposed 5,784 tokens
and 1.41x concurrency at the 4,096-token smoke limit.  Both `/health` and
`/v1/models` returned HTTP 200.

A two-image request used the checkpoint's `corn.jpeg` and `carrots.jpeg`.  It
returned HTTP 200 in 5.808 seconds, stopped normally after 418 completion tokens,
and answered: `第一张是玉米；第二张是胡萝卜；主要区别是玉米属于谷物（种子），胡萝卜属于根茎类蔬菜，且颜色、形态不同。`
The request contained 525 prompt tokens.  Per-request DSpark metrics reported
283 accepted draft tokens out of 405, 69.88% draft acceptance, and mean
acceptance length 3.096.  A text-only control also completed correctly, with
42/60 accepted draft tokens and mean acceptance length 3.10.  The server was
stopped after validation and the three CMP170HX devices were confirmed free of
vLLM worker processes.

The API currently returns `prompt_tokens_details: null`; therefore this run
uses the explicit `--per-request-spec-decode-metrics summary` response fields
and the matching Prometheus counters as acceptance evidence.

## Known limits and follow-up

- The current implementation is NVIDIA-oriented; AMD/ROCm remains a separate
  backend and is not covered by this PP3 preset.
- The local CMP170HX gate uses a short context and two-sequence budget to fit
  startup profiling.  Recalculate KV and sparse-indexer workspace budgets
  before enabling 32K/1M contexts or higher concurrency.
- Run OCRBench and a fixed image/text acceptance matrix after the smoke gate;
  compare against the upstream PR's reported GB200 numbers rather than using
  them as SM80 performance expectations.
