# GLM-5.3 Flash EXL3 on four CMP170HX GPUs

This checkout includes the native `Glm5Next` model, selective EXL3 routed-MoE
loading, the SM80 sparse-MLA fallback, and the hybrid KDA/MLA/indexer cache
layout needed by GLM-5.3 Flash.

The initial supported deployment is tensor parallel size 1 and pipeline
parallel size 4. It uses physical GPUs 2, 3, 4, and 5, which are the four
same-NUMA CMP170HX devices on this host. P2P is disabled because communication
between these cards uses the host PCIe path.

## Build the EXL3 extension

Install the reviewed ExLlamaV3 version into the project environment, then build
for SM80. Keep the build cache on data6:

```bash
UV_CACHE_DIR=/media/z/data6/.cache/uv \
  uv pip install --python .venv/bin/python --no-deps exllamav3==0.0.43

mkdir -p /media/z/data6/.cache/vllm170/exllamav3_ext
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2 \
TORCH_CUDA_ARCH_LIST=8.0 \
TORCH_EXTENSIONS_DIR=/media/z/data6/.cache/torch_extensions \
MAX_JOBS=8 \
  .venv/bin/python scripts/build_exllamav3_ext.py \
    --source stock \
    --build-dir /media/z/data6/.cache/vllm170/exllamav3_ext
```

## Serve

```bash
scripts/serve_glm53_exl3_cmp170hx_pp4.sh
```

The conservative baseline listens on `127.0.0.1:8099`, exposes the model as
`glm53-flash-exl3`, starts with an 8K context, and uses eager execution. Change
settings through environment variables or append normal `vllm serve` options:

```bash
MAX_MODEL_LEN=32768 PORT=8100 \
  scripts/serve_glm53_exl3_cmp170hx_pp4.sh --max-num-seqs 2
```

Check the endpoint with:

```bash
curl http://127.0.0.1:8099/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "glm53-flash-exl3",
    "messages": [{"role": "user", "content": "用一句话介绍北京。"}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

## DFlash2 with FP8 target KV cache

The dedicated launcher combines the EXL3 target with the local DFlash2 draft
checkpoint and stores the target sparse-MLA cache as E4M3 FP8:

```bash
scripts/serve_glm53_exl3_dflash2_cmp170hx_pp4.sh
```

It uses DFlash2 `k=2`, probabilistic draft sampling, and the Triton attention
backend. The small DFlash2 sliding-attention cache remains BF16 because native
FP8 conversion in `TRITON_ATTN` requires SM89; the large target sparse-MLA cache
uses the SM80 software E4M3 decode path. Override either choice explicitly with
`KV_CACHE_DTYPE` or `DFLASH_KV_CACHE_DTYPE`.

The launchers do not require the reference fork's optional Marlin sidecars or
CUDA graphs. Those can be enabled separately after the baseline configuration
passes correctness checks.
