# DeepSeek V4 on three CMP170HX GPUs

This directory carries the launch, performance, and correctness harnesses
selectively absorbed from `deepseek-v4-cmp170hx` at commit `3dd2d88`. The
source and this project are Apache-2.0 licensed. Endpoint defaults were adapted
for this checkout; set `DSV4_BENCH_URL` and `DSV4_BENCH_MODEL` to override
them.

## Launch

```bash
benchmarks/deepseek_v4/run_cmp170hx_pp3.sh
```

The defaults are the measured three-card configuration:

- physical GPUs `1,2,3`, PP3 with `15,15,13` layers and TP1;
- Triton sparse MLA, FP8 KV cache, block size 256;
- CUDA Graph execution and DSpark with 7 speculative tokens;
- 131,072-token context, 2,048 batched tokens, and 8 sequences;
- a 32 MiB sparse-indexer logits budget and 1.6 GB KV cache per GPU.

Do not add `--enforce-eager`: it disables the CUDA Graph decode path and causes
an order-of-magnitude regression on this machine. The launcher defaults to
`DSV4_MODE=benchmark`, which disables prefix caching. Use production mode for
normal serving:

```bash
DSV4_MODE=production benchmarks/deepseek_v4/run_cmp170hx_pp3.sh
```

The main overrides are `DSV4_MODEL_DIR`, `CUDA_VISIBLE_DEVICES`,
`DSV4_MOE_BACKEND`, `DSV4_DSPARK_TOKENS`, `DSV4_KV_CACHE_BYTES`, and
`DSV4_MAX_MODEL_LEN`.

## Minimum performance suite

Discard the first request after startup because it includes JIT warmup. Keep
prefix caching disabled for these measurements:

```bash
.venv/bin/python benchmarks/deepseek_v4/bench_prefill.py cmp170hx
.venv/bin/python benchmarks/deepseek_v4/bench_longctx.py cmp170hx \
    12000,24000,50000,77000
.venv/bin/python benchmarks/deepseek_v4/bench_decode3.py cmp170hx
.venv/bin/python benchmarks/deepseek_v4/bench_decode_stream.py cmp170hx \
    2000,8000,32000,65000,100000
```

`bench_chunk_truth.py` validates that streaming code is using server-reported
completion tokens rather than SSE chunk counts. `bench_needle.py`,
`bench_conc_needle.py`, and `bench_ceiling.py` cover long-context correctness,
multi-request row isolation, and the context ceiling. `bench_concurrency.py`
measures multi-request throughput.

Historical accumulated-chat analyzers, the probe script, and the old
two-request decode-subtraction benchmark were intentionally not absorbed. They
do not contribute to the current three-card acceptance criteria or have been
superseded by server-reported token accounting.

## Measurement rules

- Use `usage.prompt_tokens` and `usage.completion_tokens` as ground truth.
- Set `ignore_eos` for comparable fixed-length decode runs.
- Do not infer decode rate by subtracting two independent request times.
- Report the full mixed-content run, not the fastest short window.
- Keep the model, hardware, PP partition, context, cache, and DSpark settings in
  the result record.
