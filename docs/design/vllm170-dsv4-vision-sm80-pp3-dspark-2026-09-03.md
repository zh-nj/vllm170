# DeepSeek-V4-Flash-Vision-Exp on SM80 PP3 with DSpark

This note records the implementation and validation plan for
`/media/z/data6/models/DeepSeek-V4-Flash-Vision-Exp` in this checkout.  The
target deployment is three CMP170HX cards, pipeline parallelism 3, tensor
parallelism 1, and the checkpoint's five-token DSpark block.

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
- The checkpoint contains `mtp.0/1/2` target-layer blocks and declares
  `dspark_block_size=5`; conversion selects `DSparkDraftModel` with five
  speculative tokens. The three target blocks and five-token draft width are
  separate parts of the checkpoint contract.

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
7. Only PP rank 0 constructs and loads the ViT, aligner, and learned image
   sentinel vectors.  The V2 runner already executes the multimodal encoder
   only on the first PP rank, so replicas on ranks 1 and 2 consume memory
   without participating in inference.

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
partition `15,16,12`, TP1, FP8 KV cache, sparse MLA Triton attention, Marlin
MoE, and DSpark probabilistic width 5. The vision preset requires
`DSV4_DSPARK_TOKENS=5`.

## Validation ladder

Before a long-context benchmark, run the following in order:

```bash
.venv/bin/python -m compileall -q vllm tests
.venv/bin/python -m pytest tests/models/test_deepseek_v4_vision.py -q
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
physical GPUs `1,2,3`.  With the `15,16,12` partition and 800 MB per-rank KV
allocation, the engine exposed 5,085 tokens and 1.24x concurrency at the
4,096-token smoke limit.  Both `/health` and `/v1/models` returned HTTP 200.

A two-image request used the checkpoint's `corn.jpeg` and `carrots.jpeg`.  It
stopped normally after 44 completion tokens and answered `玉米` followed by
`胡萝卜`.  The request contained 522 prompt tokens.  Per-request DSpark
metrics reported 32 accepted draft tokens out of 55, 58.18% draft acceptance,
and mean acceptance length 3.909, with `num_spec_tokens=5`.  A text-only
control also completed with 21/50 accepted draft tokens and mean acceptance
length 3.10.  The service was stopped after the subsequent validation runs.

The vision-only-on-PP0 follow-up counted 267 BF16 checkpoint tensors under
`vision.*`, `aligner.*`, and `image_*`, totaling 932,786,176 bytes (0.869 GiB).
With otherwise identical launch settings, model-load memory changed from
52.39/54.79/53.53 GiB to 52.39/53.88/52.62 GiB for PP0/PP1/PP2.  Steady worker
allocations reported by `nvidia-smi` changed from 56,728/59,304/57,922 MiB to
56,734/58,316/56,954 MiB after image inference, reducing PP1 by 988 MiB and PP2
by 968 MiB while PP0 varied by 6 MiB.  A post-change two-image request returned
HTTP 200, answered `玉米，胡萝卜`, and reported `num_spec_tokens=5`; PP0 remains
the only rank logging ViT attention and encoder-cache initialization.

The API currently returns `prompt_tokens_details: null`; therefore this run
uses the explicit `--per-request-spec-decode-metrics summary` response fields
and the matching Prometheus counters as acceptance evidence.

## Long-context speed and semantic result (2026-09-03)

The service was restarted with `max_model_len=131200`, `max_num_seqs=1`,
`max_num_batched_tokens=1715`, and 2.7 GB KV memory per rank.  The benchmark
used `benchmarks/deepseek_v4/bench_concurrent_longctx.py` with one request,
`temperature=0`, `top_p=1`, DSpark width 5, and a fixed 64-token completion.
The prompt builder calibrated the server tokenizer to the exact requested
length.  TTFT is reported as prefill wall time; decode starts at the first
token and excludes prefill.

| Input tokens | Prefill time | Prefill tok/s | Decode time (63 tok) | Decode tok/s | Needle / cross-talk | DSpark acceptance | Mean accepted length |
| ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |
| 1,000 | 1.245 s | 803 | 0.964 s | 65.37 | 1/1, 0 | 31.20% | 2.56 |
| 32,000 | 6.077 s | 5,266 | 0.755 s | 83.48 | 1/1, 0 | 40.00% | 3.00 |
| 128,000 | 26.503 s | 4,830 | 0.736 s | 85.58 | 1/1, 0 | 43.81% | 3.19 |

`Needle=1/1` means the generated answer contains the unique passphrase
inserted into the prompt; cross-talk is zero.  The 128K row uses 128,000
prompt tokens (rather than 131,072) to leave room for the 64-token completion
under the 131,200-token service limit.  Raw JSON evidence is retained at
`/tmp/dsv4-vision-longctx-{1k,32k,128k}.json`.

After the long-context run, the official two-image sample was sent through
`/v1/chat/completions` with `enable_thinking=false`.  It returned HTTP 200,
`finish_reason=stop`, 438 prompt tokens, 12 completion tokens, and the exact
answer `第一张图片：玉米；第二张图片：胡萝卜`.  DSpark accepted 10/10 draft
tokens (width 5, mean accepted length 6.0).  The full response is saved at
`/tmp/dsv4-vision-longctx-image-no-think.json`.

## User-parameter revalidation (2026-09-03)

The same matrix was repeated with the user's production-style arguments:
physical GPUs `3,4,5`, `max_model_len=524288`, `max_num_seqs=4`, 2.7 GB KV
memory per rank, prefix caching, `enable_prompt_tokens_details`, per-request
metrics, adaptive long prefill (`threshold=424`),
`temperature=1.0/top_p=0.95`, default `enable_thinking=true` and
`reasoning_effort=max`, FULL_AND_PIECEWISE CUDA graphs, and DSpark width 5.
The vision checkpoint and `dsv4-vision` served name were kept unchanged.  The
command's automatic partition `[14,15,14]` first OOMed on PP2 during a 1 GiB
Marlin MXFP4 conversion allocation.  Adding the already validated
`VLLM_PP_LAYER_PARTITION=15,16,12` made all three workers start successfully;
the engine then exposed 761,042 KV tokens (1.45x at 524,288 tokens).

The long-context harness used one request per length, `scenario=agentic`,
`max_tokens=256`, and fixed seeds.  Prefill is TTFT; decode is measured for
the 255 tokens after the first token.

| Input tokens | Prefill time | Prefill tok/s | Decode time (255 tok) | Decode tok/s | Needle / cross-talk | DSpark acceptance | Mean accepted length |
| ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |
| 1,000 | 1.712 s | 584 | 2.443 s | 104.39 | 1/1, 0 | 60.95% | 4.05 |
| 32,000 | 6.216 s | 5,148 | 2.954 s | 86.31 | 1/1, 0 | 48.80% | 3.44 |
| 128,000 | 26.507 s | 4,829 | 3.024 s | 84.34 | 1/1, 0 | 50.14% | 3.51 |

The semantic gate again found the inserted passphrase for every request with
zero cross-talk.  Raw results are retained at
`/tmp/dsv4-vision-userparams-longctx-{1k,32k,128k}.json`.

With the same default reasoning settings, the two-image sample completed with
`finish_reason=stop`, 530 prompt tokens (including 418 multimodal tokens),
378 completion tokens, and content `第一张图片：玉米；第二张图片：胡萝卜`.
The response reported `prompt_tokens_details.multimodal_tokens.image=418`
and 73.39 output tokens/s.  It is saved at
`/tmp/dsv4-vision-userparams-image-512.json`.

## Text-checkpoint revalidation (2026-09-03)

The requested text checkpoint `/media/z/data6/models/DeepSeek-V4-Flash-0731`
was tested with the same user arguments, served as `dsv4f` on physical GPUs
`3,4,5`.  The unmodified command first failed during PP2 Marlin MXFP4
conversion because the automatic `[14,15,14]` layer split left only about
30 MiB free for a 1 GiB allocation.  No model or user sampling parameter was
changed; setting `VLLM_PP_LAYER_PARTITION=15,16,12` allowed all workers to load.
The service exposed 761,042 KV-cache tokens at the 524,288-token limit.

The benchmark used one request per length, `scenario=agentic`,
`temperature=1.0/top_p=0.95`, `reasoning_effort=max`, DSpark width 5, and
`max_tokens=256`.  Prefill is TTFT; decode is measured for the 255 tokens
after the first token.  `Needle=1/1` is a deterministic passphrase recall
gate and does not claim broad answer-quality coverage.

| Input tokens | Prefill time | Prefill tok/s | Decode time (255 tok) | Decode tok/s | Needle / cross-talk | DSpark acceptance | Mean accepted length |
| ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |
| 1,000 | 0.758 s | 1,319 | 3.436 s | 74.21 | 1/1, 0 | 36.67% | 2.83 |
| 32,000 | 6.244 s | 5,125 | 3.030 s | 84.16 | 1/1, 0 | 46.23% | 3.31 |
| 128,000 | 26.602 s | 4,812 | 3.120 s | 81.72 | 1/1, 0 | 45.57% | 3.28 |

All three requests recalled the inserted passphrase with zero cross-talk.  The
fixed 256-token cap and `ignore_eos` benchmark setting intentionally expose
throughput and speculative-decoding behavior; generated text can therefore be
repetitive after the answer.  Raw evidence is retained at
`/tmp/dsv4f-userparams-longctx-{1k,32k,128k}.json`.

## Known limits and follow-up

- The current implementation is NVIDIA-oriented; AMD/ROCm remains a separate
  backend and is not covered by this PP3 preset.
- The local CMP170HX gate uses a short context and two-sequence budget to fit
  startup profiling.  Recalculate KV and sparse-indexer workspace budgets
  before enabling 32K/1M contexts or higher concurrency.
- The first image request JIT-compiles image-visibility, SWA-index, and
  TileLang HC-prenorm kernels; extend multimodal warmup before measuring cold
  latency.
- Run OCRBench and a fixed image/text acceptance matrix after the smoke gate;
  compare against the upstream PR's reported GB200 numbers rather than using
  them as SM80 performance expectations.
