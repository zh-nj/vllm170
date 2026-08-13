#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
VLLM_BIN=${VLLM_BIN:-${REPO_ROOT}/.venv/bin/vllm}

if [[ -z "${DSV4_MODEL_DIR:-}" ]]; then
    for candidate in \
        /mnt/data6/models/DeepSeek-V4-Flash-0731 \
        /media/z/data6/models/DeepSeek-V4-Flash-0731; do
        if [[ -d "${candidate}" ]]; then
            DSV4_MODEL_DIR=${candidate}
            break
        fi
    done
fi

if [[ ! -x "${VLLM_BIN}" ]]; then
    echo "vLLM executable not found: ${VLLM_BIN}" >&2
    exit 2
fi
if [[ -z "${DSV4_MODEL_DIR:-}" || ! -d "${DSV4_MODEL_DIR}" ]]; then
    echo "Set DSV4_MODEL_DIR to the DeepSeek-V4-Flash-0731 directory." >&2
    exit 2
fi

export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_PP_LAYER_PARTITION=${VLLM_PP_LAYER_PARTITION:-15,15,13}
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-32}

DSV4_HOST=${DSV4_HOST:-127.0.0.1}
DSV4_PORT=${DSV4_PORT:-8098}
DSV4_SERVED_MODEL=${DSV4_SERVED_MODEL:-dsv4s}
DSV4_MOE_BACKEND=${DSV4_MOE_BACKEND:-marlin}
DSV4_DSPARK_TOKENS=${DSV4_DSPARK_TOKENS:-7}
DSV4_KV_CACHE_BYTES=${DSV4_KV_CACHE_BYTES:-1600000000}
DSV4_MAX_MODEL_LEN=${DSV4_MAX_MODEL_LEN:-131072}
DSV4_MAX_BATCHED_TOKENS=${DSV4_MAX_BATCHED_TOKENS:-2048}
DSV4_MAX_NUM_SEQS=${DSV4_MAX_NUM_SEQS:-8}
DSV4_MODE=${DSV4_MODE:-benchmark}

case "${DSV4_MODE}" in
    benchmark)
        prefix_cache_flag=--no-enable-prefix-caching
        ;;
    production)
        prefix_cache_flag=--enable-prefix-caching
        ;;
    *)
        echo "DSV4_MODE must be benchmark or production, got: ${DSV4_MODE}" >&2
        exit 2
        ;;
esac

exec "${VLLM_BIN}" serve "${DSV4_MODEL_DIR}" \
    --served-model-name "${DSV4_SERVED_MODEL}" \
    --host "${DSV4_HOST}" \
    --port "${DSV4_PORT}" \
    --pipeline-parallel-size 3 \
    --tensor-parallel-size 1 \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --kv-cache-memory-bytes "${DSV4_KV_CACHE_BYTES}" \
    --max-model-len "${DSV4_MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${DSV4_MAX_BATCHED_TOKENS}" \
    --max-num-seqs "${DSV4_MAX_NUM_SEQS}" \
    --tokenizer-mode deepseek_v4 \
    --attention-backend TRITON_MLA_SPARSE_DSV4 \
    --moe-backend "${DSV4_MOE_BACKEND}" \
    --speculative-config \
        "{\"method\":\"dspark\",\"num_speculative_tokens\":${DSV4_DSPARK_TOKENS}}" \
    "${prefix_cache_flag}" \
    --no-enable-flashinfer-autotune \
    --reasoning-parser deepseek_v4 \
    "$@"
