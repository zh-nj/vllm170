# DeepSeek-V4 CMP170HX PP buffer selective absorption

## Result

This change selectively absorbs the pipeline-buffer and structured-output
correctness facilities from `vllm-170hx-dsv4f-pp-dspark` without copying its
unrelated sparse-attention implementation.

For the deployed three-CMP170HX topology, V2 now receives pipeline activations
directly into the graph-stable model-runner arena. The arena has explicit
`receiving -> model -> dspark -> free` ownership, a monotonically increasing
generation, and an optional CUDA-stream check. On the last stage, DSpark uses
the same storage for auxiliary-hidden packing after target forward completes.
Mismatched receive metadata falls back to a fresh allocation.

The library-wide switch defaults to off. The measured CMP170HX launcher enables
it because that path is PP3/TP1 and passed the real-hardware gate below:

```bash
VLLM_PP_REUSE_RECV_BUFFER=0|1
VLLM_PP_REUSE_RECV_BUFFER_DEBUG=0|1
```

## A/B/A protocol

All three runs used the mounted
`/media/z/data6/models/DeepSeek-V4-Flash-0731` checkpoint, physical GPUs
`1,2,3`, PP3/TP1, partition `15,15,13`, Marlin MXFP4 MoE, Triton sparse MLA,
FP8 KV cache, CUDA Graph, DSpark-7, `max-num-batched-tokens=1715`, exactly 5,000
server prompt tokens, 60 completion tokens, and three repeats. B also enabled
the stream ownership debug gate.

| Run | switch | TTFT mean | DSpark ms/round mean | PP0 peak | PP1 peak | PP2 peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | 0 | 1.2172 s | 34.5079 ms | 58,080 MiB | 57,224 MiB | 62,922 MiB |
| B | 1 | 1.2200 s | 34.6882 ms | 58,080 MiB | 57,062 MiB | 62,854 MiB |
| A2 | 0 | 1.2198 s | 34.9247 ms | 58,080 MiB | 57,224 MiB | 62,922 MiB |

B relative to the mean of A1 and A2 is `+0.12%` TTFT and `-0.08%`
milliseconds per target round. The optimization therefore has no measured
latency regression. Sampled workload peaks fall by 162 MiB on PP1 and 68 MiB
on PP2; the 500 ms sampler makes these conservative observations rather than
an allocator-exact byte count. PP0 does not receive pipeline activations and is
unchanged.

End-to-end decode tokens/s is not used for attribution because the stochastic
DSpark trajectories had different acceptance rates. Every run passed the
workload gate with 5,000 prompt and 60 completion tokens.

Artifacts:

- `profiles/deepseek_v4/pp-recv-ab/baseline/20260815T040046Z`
- `profiles/deepseek_v4/pp-recv-ab/optimized/20260815T040257Z`
- `profiles/deepseek_v4/pp-recv-ab/baseline-repeat/20260815T040533Z`

## Structured-output correctness

V2 structured-output drafts are now carried with the exact producer scheduler
step instead of a mutable process-wide handoff. The scheduler caches bounded
out-of-order arrivals by `(request_id, producer_step_id)`, validates only the
matching generation, and pads missing or mismatched drafts with `-1` for
target-only decoding. XGrammar stops validating immediately after EOS. V1 PP
drains older batches before building a deferred grammar mask.

The DSpark draft `ParallelConfig` is also constructed with PP1 from the start,
so derived `world_size` no longer retains the target PP3 value. Side-stream PP
broadcasts use an owning snapshot of draft tokens.

## Validation

```text
59 passed, 15 warnings
3 DSpark workspace tests passed
pre-commit on all changed Python files: passed
distributed test_comm_ops collection: blocked locally by missing optional ray
direct preallocated-buffer shape/dtype smoke: passed
real PP3 A/B/A: workload_accepted=true for all 9 requests
```

The existing Ray-backed communication test was extended to assert exact storage
reuse for matching metadata and allocation fallback for a wrong shape. CI or an
environment with Ray can collect that file; the actual PP3 run exercises the
same receive path on NCCL and is the promotion evidence for this machine.
