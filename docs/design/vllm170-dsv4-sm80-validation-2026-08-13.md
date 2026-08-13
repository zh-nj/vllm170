# DeepSeek V4 Flash on three CMP170HX GPUs

## Result

The development in this checkout was validated on three physical CMP170HX
GPUs with the local `DeepSeek-V4-Flash-0731` checkpoint. The selected three-card
baseline is PP3, Marlin, Triton sparse MLA, CUDA Graph, FP8 KV cache, and DSpark
with seven speculative tokens.

The measured configuration reached about 5,260 prompt tokens/s at a 59,236
token prompt and 95.7 aggregate decode tokens/s across three simultaneous
requests. It therefore meets the comparison targets of 5,000 prefill tokens/s
and 60 aggregate decode tokens/s on this machine.

## Hardware and build

- Three NVIDIA CMP170HX devices, compute capability 8.0 and 64 GiB each.
- NVIDIA driver 610.43.03, CUDA toolkit 13.3, and PyTorch 2.13.0+cu130.
- The cards do not expose peer-to-peer access, so PP uses
  `NCCL_P2P_DISABLE=1`.
- The requested checkpoint path `/mnt/data6/models/DeepSeek-V4-Flash-0731`
  resolves on this host to
  `/media/z/data6/models/DeepSeek-V4-Flash-0731`.
- The editable build targets SM80. Direct CUDA allocation and write tests
  succeeded on all three cards.

## Selective comparison with `deepseek-v4-cmp170hx`

The comparison checkout was inspected at commit `3dd2d88`. Its headline
numbers were mainly collected with four cards and are not treated as a direct
three-card baseline. Only mechanisms that are applicable to the current vLLM
tree and that survived local validation were retained.

| Reference mechanism | Decision | Current result |
| --- | --- | --- |
| PP layer partitioning | Retain | PP3 uses `15,15,13`; the DSpark auxiliary layers remain on the last stage. |
| DSpark speculative decoding | Retain | Seven draft tokens outperform the five-token control in the three-run mean. |
| CUDA Graph decode | Retain | Eager mode caused a large decode regression and is not used by the launcher. |
| FP8 KV, block size 256 | Retain | Explicit 1.6 GB KV allocation provides 161,246 cache tokens. |
| Sparse-indexer logits cap | Retain and adapt | The old row-chunk setting maps to `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=32`. |
| Persistent/filtered top-k patches | Do not copy | Equivalent or newer implementations are already present upstream. |
| Older DSpark PP patches | Do not duplicate | The required PP behavior is already present in this checkout. |
| Marlin FP8 dequant environment switch | Do not copy | The current source has a different Marlin lifecycle; its validated cache fix is retained instead. |
| MXFP4 INT8 SM80 MoE | Keep experimental only | It was slower in the comparable short-prefill run and is not the launcher default. |
| Historical accumulated-chat analyzers | Do not copy | They do not contribute to the current three-card acceptance boundary. |
| Two-request decode subtraction | Do not copy | Server-reported completion-token accounting supersedes it. |
| Deployment tarballs and destructive reset tools | Do not copy | They are machine-specific and not part of runtime correctness. |

The selected launcher and benchmark harnesses are in
`benchmarks/deepseek_v4/`. The launcher is:

```bash
benchmarks/deepseek_v4/run_cmp170hx_pp3.sh
```

Its relevant defaults are:

```text
CUDA_VISIBLE_DEVICES=1,2,3
VLLM_PP_LAYER_PARTITION=15,15,13
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=32
pipeline parallel = 3, tensor parallel = 1
MoE backend = marlin
DSpark speculative tokens = 7
max model length = 131072
max batched tokens = 2048, max sequences = 8
FP8 KV cache = 1600000000 bytes per worker, block size = 256
attention backend = TRITON_MLA_SPARSE_DSV4
prefix caching = disabled in benchmark mode
```

## Runtime fixes validated on SM80

Real model loading and request execution exposed five SM80 boundaries:

1. MHC pre-broadcast unconditionally called DeepGEMM. It now selects the
   TileLang prenorm GEMM when DeepGEMM is unsupported.
2. FP8 Marlin packed grouped `wo_a` BMM weights before the sparse-MLA fallback
   could read them. Preprocessing now caches a BF16 grouped view before packing.
3. The head-dimension-512 compressor selected a CuTeDSL SM90 implementation on
   SM80. It now uses the Triton paged-cache implementation below SM90.
4. Sparse SWA metadata construction called FlashMLA tile scheduling on SM80.
   SM80 now leaves those scheduler entries unset and follows the Triton path.
5. The SM80 sparse-indexer fallback ignored `skip_kv_gather` on query
   subchunks. With the 32 MiB logits budget this repeated a gather into the same
   shared workspace and produced an illegal memory access at long context. The
   fallback now gathers once per KV chunk and reuses it for subsequent query
   subchunks, matching the generic indexer contract.

The fifth fix was reproduced with both Marlin and the experimental MXFP4
backend, so the failure was shared sparse-indexer behavior rather than evidence
against either MoE implementation. After the fix, the exact failing Marlin
workload completed at both 38,467 and 59,236 prompt tokens.

## Real-hardware performance

All measurements below use the selected launcher settings. The first request
after startup is discarded when it includes JIT compilation.

### Prefill

| Actual prompt tokens | Prompt throughput (tokens/s) |
| ---: | ---: |
| 9,236 | 4,376.5 |
| 18,467 | 4,997.0 |
| 38,467 | 5,280.8 |
| 59,236 | 5,258.8 |

The shorter sweep measured 1,979.7, 2,379.6, 3,079.1, and 3,902.6 tokens/s at
784, 1,553, 3,091, and 5,399 actual prompt tokens respectively.

### Decode and DSpark control

Three repeated multi-request runs with DSpark 7 produced aggregate decode
rates of 96.1, 97.4, and 93.5 tokens/s, a mean of 95.7 tokens/s. The otherwise
identical DSpark 5 control produced 98.7, 89.0, and 87.6 tokens/s, a lower and
more variable mean of 91.8 tokens/s. DSpark 7 is therefore retained as the
three-card default.

The aggregate number is the sum across three simultaneous requests. It is not
single-stream inter-token latency and should not be compared as such.

### Long-context correctness

Needle retrieval passed at 38,505, 59,274, and 76,967 actual prompt tokens.
The corresponding prompt rates were about 5,192, 5,210, and 4,377 tokens/s.
This validates content retrieval beyond the 59K performance sweep rather than
only proving that allocation succeeds.

### Experimental MXFP4 INT8 SM80 backend

Before the shared indexer fix, its short-prefill sweep measured about 1,653,
2,219, 2,917, and 3,767 tokens/s, below Marlin at the corresponding sizes. Its
long-context crash from that run is invalid as an implementation comparison
because the same fallback defect was subsequently reproduced with Marlin. The
backend remains available for experiments, but it was not selected and no
post-fix claim is made for its long-context stability or speed.

## Verification boundary

Focused unit and GPU tests cover DSpark configuration and PP padding, Marlin
grouped-BMM caching, TileLang MHC fallback, Triton compressor selection, SM80
sparse-SWA scheduling, and query-subchunk KV-gather reuse. The selected suites
completed with 33 tests passing: 26 implementation-focused tests plus seven
sparse-indexer chunking and reuse cases.

A broader 236-case run of the touched test files found 199 passes, 29 skips,
and eight failures outside the selected SM80 path. Six existing MXFP4 Triton
cases emit the native `.e2m1x2` conversion that `ptxas` rejects for SM80, and
two existing CuTeDSL FP8-store cases fail NVVM compilation for SM80. Those
tests do not currently carry hardware skip guards. They are not counted as
passing and are not exercised by the selected Marlin launcher.

The current acceptance boundary is three-card PP, 131,072 configured maximum
context, verified needle retrieval through about 77K actual tokens, batch
capacity of eight, FP8 KV cache, Triton sparse MLA, Marlin, CUDA Graph, and
DSpark 7. A production concurrency saturation curve and correctness near the
full 131K ceiling have not yet been established. Shutdown can emit a non-fatal
Python `resource_tracker` warning; the validation server left no GPU compute
process behind.
