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

The launcher deliberately does not enable the reference fork's optional
DFlash2 draft model, Marlin sidecars, online FP8 conversion, or CUDA graphs.
Those are throughput optimizations with additional artifacts and should be
enabled only after the stock EXL3 PP4 path has passed correctness checks.
