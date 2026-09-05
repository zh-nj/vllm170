# GLM-5.3 Flash AutoRound with DFlash2 on three CMP170HX GPUs

This deployment keeps the official vLLM GLM-5.3 implementation from PR
`#53906` as the model base and adds the local SM80 pipeline-parallel and
DFlash2 paths required by the three CMP170HX cards.

The source baseline is the official `#53906` squash commit
`98ed0856f31fa3aaf5e27464e2b4ef5a8ee6b2f5`, plus its packaging follow-up
`#55214` (`78300cdabffbaaffe7625ed34412ec2aec75535c`). Local overlays retain DFlash2,
carry its auxiliary hidden states across PP stages, and provide an SM80 sparse
MLA backend with software E4M3 encoding for the GLM indexer cache.

Start the service after all 34 target shards and the safetensors index have
finished downloading:

```bash
scripts/serve_glm53_autoround_dflash2_cmp170hx_pp3.sh
```

The default uses physical GPUs `2,3,5`, PP3 with a measured `17,15,13` layer
split, an 8K context, one sequence, 2048 max batched tokens, FP8 target KV cache,
`FULL` CUDA Graph capture, and seven speculative tokens. The SM80 sparse MLA
backend supports uniform decode batches, so vLLM resolves the requested mode to
`FULL_DECODE_ONLY`: decode uses the captured full graph while prefill remains
eager. The final target stage is intentionally lighter because DFlash2 is
constructed there and owns a full-vocabulary `lm_head`; this leaves room for the
larger prefill activation workspace. Seven draft tokens plus DFlash's anchor
query match the checkpoint's trained block size of eight. The DFlash cache
remains BF16 on SM80.

## Prefill measurement and optimization record

The current measurement ladder uses a single 7680-token request, five repeats,
server-side prefill time, and CUDA-event phase timings. With
`max-num-batched-tokens=2048`, prefill throughput for 7680 tokens was
3465 tokens/s before the sparse-MLA change and 3712 tokens/s after it. The
phase profile attributes the improvement to the sparse MLA attention kernel:
325.7 ms became 225.8 ms per profiled request (-30.7%); the critical target
model total fell from 1424.6 ms to 1333.0 ms (-6.4%).

The next remaining hotspots are the FFN/MoE path (656.9 ms on PP1), including
Marlin W13 378.6 ms and W2 203.2 ms, followed by KDA input projection and core
(118.8 ms and 130.5 ms on PP0). Marlin block-size microbenchmarks passed
correctness for GLM's `E=288`, `top-k=8`, `K=4096`, `N=2048` shape, but live
end-to-end A/B retained the existing block size: block 48 reached 2.0439 s
server prefill, block 64 reached 2.0832 s, and block 32 reached 2.1027 s.
Therefore no Marlin block-size override is enabled in the launcher. Raw phase
and kernel artifacts are kept under
`/mnt/data/apps/vllm-profiles/glm53-prefill-20260905/`.

An isolated KDA core-kernel check at the GLM prefill shape (`T=2041`,
`H=64`, `D=128`) measured about 1.68 ms with chunk32 versus 2.11 ms with the
default chunk64. Since the complete KDA core accounts for 130.5 ms of the
1333.0 ms target forward, its end-to-end upper bound is below 2% before
convolution and state-management overhead; chunk32 is therefore not enabled
as a production override.

## Decode measurement and optimization target

For target-only decode, set `ENABLE_DFLASH=0`; the production service was
restored with this setting and the existing `FULL` CUDA Graph. Three repeats
of 64 generated tokens gave 50.72 tok/s at a 128-token context and 47.51 tok/s
at a 1024-token context, with server decode medians of 1.242 s and 1.329 s,
respectively. At 4096 and 7680 tokens, decode reached 44.56 and 44.65 tok/s;
the long-context plateau is consistent with sparse attention reaching its
fixed top-k width. The raw rows and the complete path decomposition are
recorded in
`/mnt/data/apps/vllm-profiles/glm53-decode-20260905/decode-path-report.md`.

The FULL-graph phase profile measured a serial PP3 target pass of 19.859 ms
(PP0/PP1/PP2: 7.347/6.409/6.104 ms). Layer-level profiling requires a separate
eager diagnostic because replay hides the individual model calls. That
diagnostic ranked FFN at 47.6% of target model work, attention at 32.0%, the
sparse indexer at 8.5%, routed/modular MoE at 20.1%/13.5%, sparse MLA at 4.6%,
and KDA core at 4.1%. Focused screening did not find a production-worthy
decode change: replacing one-group grouped top-k was 2.37x slower in graph
replay; increasing KDA recurrent warps regressed; the best mHC tile differed
from the current tile by only 0.2%; and a `max_model_len=2048` short-graph A/B
improved 128/1024 decode by only 0.4%/0.9%. All candidates passed numerical
correctness before timing. No decode kernel override is enabled; future work
must demonstrate a stable multi-percent serving gain in routed MoE or the
long-context sparse attention/indexer chain.

All operational settings can be overridden without editing the script. For
example:

`ENABLE_DFLASH=0` starts the target-only path for decode profiling; it skips
DFlash model validation and does not pass `--speculative-config`. The default
remains `ENABLE_DFLASH=1` so the separate DFlash path can be restored without
changing the launcher. DFlash is intentionally out of the current decode
optimization scope.

```bash
CUDA_VISIBLE_DEVICES=1,3,4 PORT=8100 MAX_MODEL_LEN=4096 \
  DFLASH_NUM_SPECULATIVE_TOKENS=3 \
  scripts/serve_glm53_autoround_dflash2_cmp170hx_pp3.sh
```

The launcher validates every index-referenced safetensors header and declared
payload length, including all 34 numbered target shards and the AutoRound extra
tensor files. With DFlash enabled it also validates the DFlash2 weights; with
`ENABLE_DFLASH=0` only the target model and chat template are required. It does
not stop an existing service or modify the active GPU processes.

After startup, check the endpoint with:

```bash
curl http://127.0.0.1:8099/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "glm53-flash-autoround-dflash2",
    "messages": [{"role": "user", "content": "用一句话介绍北京。"}],
    "max_tokens": 32,
    "temperature": 0
  }'
```
