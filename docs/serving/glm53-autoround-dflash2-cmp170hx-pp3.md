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

The default uses physical GPUs `2,3,5`, PP3 with a `16,15,14` layer split, an
8K context, one sequence, 128 max batched tokens, FP8 target KV cache, eager
execution, and seven speculative tokens. The final target stage is intentionally
lighter because DFlash2 is constructed there and owns a full-vocabulary
`lm_head`; the 128-token default also keeps dummy-run activation workspace below
the 64-GiB card limit. Seven draft tokens plus DFlash's anchor query match the
checkpoint's trained block size of eight. The DFlash cache remains BF16 on SM80.

All operational settings can be overridden without editing the script. For
example:

```bash
CUDA_VISIBLE_DEVICES=1,3,4 PORT=8100 MAX_MODEL_LEN=4096 \
  DFLASH_NUM_SPECULATIVE_TOKENS=3 \
  scripts/serve_glm53_autoround_dflash2_cmp170hx_pp3.sh
```

The launcher validates every index-referenced safetensors header and declared
payload length, including all 34 numbered target shards and the AutoRound extra
tensor files. It fails before starting vLLM if any target weight, the DFlash2
weights, or the chat template is missing or incomplete. It does not stop an
existing service or modify the active GPU processes.

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
