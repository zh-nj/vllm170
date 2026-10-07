# Qwen3.8 AutoRound: upstream inference optimizations on SM80

## Source and target

The target is `/mnt/data/apps/vllma100/vllm170` at
`e2e327dc53b93acb85c8b56b526d369eb5461421`, with the local SM80 optimizations
preserved. The reference checkout `/mnt/data/apps/vllm` has detached HEAD
`31ebe18bb63303be83137a59f668ca9812ac3579` (`v0.31.0rc4`). Its cached
`origin/main` is `0e468adb43d60484b5a6546e0124b686fd76baf3`; the two newest QSA
optimizations below are present there, rather than in the detached checkout.
Files were read at fixed commit IDs. No fetch or wholesale model replacement
was performed.

The serving target is
`/media/z/data6/models/Qwen3.8-Flash-Next-W4A16-AutoRound`: INC/Marlin routed
experts, BF16 attention and indexer projections, physical CMP170HX GPUs 2/3,
SM80, PP2/TP1, partition `26,22`, CPU PLE, MTP3, and
`FULL_AND_PIECEWISE` CUDA Graphs.

## Adopted commits

| Source commit | Adaptation |
| --- | --- |
| `20b52e9f5b5793d56586b91df9b93cbe30a24040` | Skip causal-conv metadata unused by PLE; reuse prefill offsets; avoid token permutations and request gathers for pure speculative decode; bound mixed-batch repeat-interleave with the CPU-known token count. Generic Mamba/short-conv consumers retain their metadata. |
| `a2be4d3cc7b978a6720dab32d28ae310775acb16` | Derive QSA key/position views through properties so profiling teardown releases the KV storage. Preserve this checkout's physical-cache layout adaptation. |
| `042ab0305cc4215e2c6fb215f1d6f2e65dd7d4ac` | Merge QKVG and replicated indexer Q/K weight shards into one projection. Preserve checkpoint shard loading and the existing output projection. Enable for measured SM80/TP1 shapes with unquantized projections and no LoRA. |
| `c9578f10ab3eaf252742960db3fd08f11cf6d25a` | Fuse indexer preparation, main Q/K norm and RoPE, gate extraction and BF16 KV write. Use the kernel as amended by `042ab0305c`, including token strides for merged projection views. Accept caller-owned main Q/gate outputs for the existing opaque attention transaction. |
| `7a877ae34dcf4a7d46de08eb16db4699e1e0047c` | Reuse one logits allocation across QSA scoring chunks. Keep the local 128 MiB budget, compact selection and SM80 prefill scoring kernel. |

The projection gate requires SM80, TP1, hidden size 2560, QKVG output width
13312, indexer width 640, unquantized methods for both projections and no LoRA.
The preparation gate requires SM80, TP1, hidden size 2560, 24 main Q heads,
two KV heads, head dimension 256 and the existing fused norm/RoPE and indexer
eligibility checks. The QSA owner still requires BF16 activations and KV cache.
Profiling and padded gate rows are initialized; live query/key cache updates
remain inside the opaque attention transaction.

Both optimizations default on within those gates. Restart with either flag
set to `0` to restore its previous path:

```bash
VLLM_QWEN4_EXP_QSA_MERGED_PROJECTION=0 \
VLLM_QWEN4_EXP_QSA_FUSED_PREPARE=0 \
  scripts/serve_qwen38_flash_next_autoround_cmp170hx_pp2.sh
```

The new environment variables participate in `envs.compile_factors()`, so
the rollback paths use distinct compiled-cache keys.

## Candidates excluded

`d6cce94fd4` (HC down + SiLU) explicitly requires capability >= 90. The HC
skinny-GEMM planning changes and `55b80221ed` target Hopper/Blackwell/SM121,
without SM80 configurations. Existing local SM80 HC and output-projection
kernels are retained.

The local CPU PLE gather runs synchronously inside its custom op and has no
upstream side-stream prefetch buffer, so `48d8880d09` does not apply directly.
AutoRound already loads the BF16 PLE table correctly; `4f52fa35ef` is not
needed for this checkpoint. FP8 KV-cache, Quark exports and framework-wide
Marlin workspace/layout migrations are outside this measured BF16/INC recipe.

## Kernel qualification

Physical GPU 1 was idle before these bounded tests. CUDA Events measured
CUDA Graph replay using retained input/output buffers, alternating baseline
and candidate order. These are hot-cache diagnostic measurements, separate
from serving performance.

Merged BF16 projection `(N,K)=(13952,2560)` was compared with two projections
of widths 13312 and 640. Median time fell from 67.3 to 57.4 us at M=1,
72.5 to 61.6 us at M=4, 73.4 to 62.7 us at M=8, and 79.0 to 64.7 us at M=16.
At M=4096 it fell from 1782.4 to 1712.6 us. Main QKVG results were bitwise
identical. Indexer relative L2 error was at most 0.00270 across the tested
shapes, reflecting BF16 GEMM rounding differences.

Fused preparation passed independent comparisons of Q, gates, raw/compressed
indexer state and paged main KV state, including MRoPE, negative slots, merged
row strides, mixed requests and a 4097-token prefill. Preparation time at
M=4097 fell from 390.1 to 169.8 us; small decode shapes took roughly half the
previous time.

The first preparation harness did not retain all captured graph inputs and
hit an illegal access; its second version shared a mutable compressed cache
with the correctness reference. Both harness defects were fixed before the
accepted run. These preliminary logs are retained and are not attributed to
the production kernel. The accepted run is `prepare-qualification-v3.log`.

## Validation artifacts

`artifacts/qwen38-upstream-optimizations-20261006/` contains the source patches,
file hashes, implementation diff, kernel qualification scripts/results and
test/lint logs. The Qwen suite passed 131 tests, and both additional generic
short-conv regression cases passed (133 distinct passing cases). Broader ReplaySSM/linear-attention suites could
not construct their uncached Hugging Face model configurations; the failures
occurred before metadata construction.

Real serving A/B uses the identical 28 recorded prompts from the earlier
AutoRound speed run, with fresh cache salts, one warmup and three measurements
at 1K/32K/128K, plus four concurrent 1K requests. Both variants include the
PLE, storage-release and workspace changes; only the two QSA flags differ.
The A/B therefore measures the incremental QSA changes. Raw records are under
`/mnt/data/apps/vllm-profiles/qwen38-autoround-upstream-20261006/`.

## Serving A/B results

| Input / concurrency | Baseline TTFT (s) | Candidate TTFT (s) | Baseline decode (tok/s/request) | Candidate decode (tok/s/request) | Decode change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024 / 1 | 0.321 | 0.308 | 96.8 | 96.2 | -0.6% |
| 32768 / 1 | 3.469 | 3.447 | 73.1 | 74.6 | +2.0% |
| 131072 / 1 | 13.505 | 13.381 | 71.2 | 75.9 | +6.6% |
| 1024 / 4 | 0.768 | 0.738 | 76.4 | 75.4 | -1.3% |

Four-request aggregate throughput in the shared decode window changed from
330.7 to 332.9 tok/s (+0.65%). End-to-end output throughput changed from
205.4 to 208.8 tok/s. Every formal wave remains in the results, including the
baseline's first slow concurrency wave and the candidate's second slow wave.
Three repetitions are insufficient to establish a stable gain for every shape.
MTP acceptance and generated prose also vary: the 128K aggregate draft-token
acceptance changed from 0.252 to 0.275, so its decode improvement cannot be
attributed entirely to kernel execution time.

Both variants passed 28/28 verification-code retrievals (21 measured requests,
seven warmups), used identical input token IDs, generated 128 ordinary tokens
per request and had zero prefix hits. Six complete output-token sequences
matched exactly across the 28 paired requests; the remaining prose differed.
These are bounded semantic checks, not a full accuracy evaluation. Candidate
smoke checks passed 6/6: default thinking arithmetic, automatic weather tool
choice, and four simultaneous arithmetic requests.

No CUDA error, illegal access, OOM or request failure occurred during serving.
After the explicit SIGTERM shutdown, the candidate logged an output-handler
EngineDeadError during teardown. It follows the shutdown trigger and is
recorded separately from serving errors. Both temporary units released their
GPU processes; original-service restoration is recorded in the same artifact
folder.

## Final service state

The temporary Qwen unit was stopped and collected. `dsv41f-manual.service`
was restored with the original argv and captured environment, verified against
its running process. Its health and model endpoints returned 200, all five GPU
workers belong to the original unit cgroup, and a real arithmetic request
returned `42`. Physical GPU 1 returned to 6 MiB with no compute process.
`cleanup-validation.json` and `deepseek-restore-validation.json` record these
checks. No commit, push or upstream PR was made.
