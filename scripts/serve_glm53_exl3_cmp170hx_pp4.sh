#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
VLLM_BIN=${VLLM_BIN:-${REPO_ROOT}/.venv/bin/vllm}
MODEL_DIR=${MODEL_DIR:-/media/z/data6/models/GLM-5.3-Flash-EXL3-TR3-4bpw}
EXL3_EXTENSION_DIR=${EXL3_EXTENSION_DIR:-/media/z/data6/.cache/vllm170/exllamav3_ext}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-${REPO_ROOT}/chat_templates/glm53-enable-thinking-switch.jinja}

if [[ ! -x "${VLLM_BIN}" ]]; then
    echo "vLLM executable not found: ${VLLM_BIN}" >&2
    exit 2
fi
if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    echo "Invalid GLM-5.3 model directory: ${MODEL_DIR}" >&2
    exit 2
fi
if [[ ! -f "${CHAT_TEMPLATE}" ]]; then
    echo "Chat template not found: ${CHAT_TEMPLATE}" >&2
    exit 2
fi
if [[ ! -f "${EXL3_EXTENSION_DIR}/exllamav3_ext.so" ]]; then
    echo "EXL3 extension not found: ${EXL3_EXTENSION_DIR}/exllamav3_ext.so" >&2
    echo "Build it with scripts/build_exllamav3_ext.py first." >&2
    exit 2
fi
for shard in $(seq 1 120); do
    printf -v shard_name '%05d' "${shard}"
    path=${MODEL_DIR}/model-${shard_name}-of-00120.safetensors
    if [[ ! -s "${path}" ]]; then
        echo "Missing model shard: ${path}" >&2
        exit 2
    fi
done

export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,4,5}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_PP_LAYER_PARTITION=${VLLM_PP_LAYER_PARTITION:-13,11,11,10}
# Keep PP receive buffers independent while the SM80 path is validated.
export VLLM_PP_REUSE_RECV_BUFFER=${VLLM_PP_REUSE_RECV_BUFFER:-0}
export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-1}
export VLLM_KV_CACHE_LAYOUT=${VLLM_KV_CACHE_LAYOUT:-LBHNC}
export VLLM_EXL3_EXTENSION_DIR=${VLLM_EXL3_EXTENSION_DIR:-${EXL3_EXTENSION_DIR}}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/media/z/data6/.cache/torch_extensions}
export PYTHONPATH="${REPO_ROOT}:${EXL3_EXTENSION_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8099}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-glm53-flash-exl3}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-256}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.94}

exec "${VLLM_BIN}" serve "${MODEL_DIR}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 4 \
    --block-size 256 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --kv-cache-dtype auto \
    --attention-backend TRITON_MLA_SPARSE_SM80 \
    --language-model-only \
    --skip-mm-profiling \
    --enforce-eager \
    --disable-cascade-attn \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --generation-config vllm \
    --chat-template "${CHAT_TEMPLATE}" \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --enable-auto-tool-choice \
    "$@"
