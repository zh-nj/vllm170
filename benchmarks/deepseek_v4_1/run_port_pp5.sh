#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Port launcher: DeepSeek-V4.1-Flash on 5x CMP 170HX (SM80), PP5, TP1.
#
# Model implementation comes from the cpu-hybrid branch (CED, PP-KV replicas,
# SM80 fused index construction); the CPU/GPU hybrid expert path is NOT used,
# so the MoE runs entirely on the GPU. The project's cross-rank KV-sharing
# mechanism carries each kv-source layer's compressed latent, normalized
# indexer K, top-k indices and candidate blocks inside the ordinary PP
# message, which is what makes an 8,8,8,8,8 split legal. Distributed DSpark
# (draft blocks on PP1..PP3, heads on PP4) is kept.
#
# Stopping the server: killing only the `vllm serve` parent leaves the five
# `VLLM::Worker_PP*` processes holding ~62 GiB per card, and the next launch
# then dies in WorkerProc init. Kill them too, e.g.
#   pkill -f "[v]llm serve"; pkill -f "VLLM::Worker"
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
VLLM_BIN=${VLLM_BIN:-${REPO_ROOT}/.venv/bin/vllm}

export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3,4,5}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_PP_LAYER_PARTITION=${VLLM_PP_LAYER_PARTITION:-8,8,8,8,8}
# This host has ~252 GiB RAM; the branch's default 64 GiB spare-RAM guard
# would reject the two resident Engram tables (2 x 94.42 GiB) outright.
export VLLM_DSV41_HOST_HEADROOM_GIB=${VLLM_DSV41_HOST_HEADROOM_GIB:-12}
# The PP2 rank (layers 16-23 plus its DSpark draft stage) keeps only ~1.1 GiB of
# device memory free during serving, and the default allocator cannot satisfy a
# 128 MiB request from its fragmented free pool: at 2048 prefill tokens that
# raised "memory allocation failed with OOM on device 2 while trying to allocate
# 134217728 bytes (free: 118226944)".
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True does NOT rescue 2048 (same
# OOM, and it raises the resident pool by ~450 MiB per rank), so the chunk size
# below stays at 1024 with the stock allocator. Override PORT_MAX_BATCHED only
# after PP2 gains real headroom (fewer layers, smaller KV, or no draft stage).

PORT_PORT=${PORT_PORT:-26663}
PORT_MODEL_DIR=${PORT_MODEL_DIR:-/media/z/data6/models/DeepSeek-V4.1-Flash}
PORT_MAX_MODEL_LEN=${PORT_MAX_MODEL_LEN:-1048576}
PORT_KV_BYTES=${PORT_KV_BYTES:-1300000000}
PORT_MAX_BATCHED=${PORT_MAX_BATCHED:-1024}
PORT_MAX_SEQS=${PORT_MAX_SEQS:-8}
PORT_DSPARK_TOKENS=${PORT_DSPARK_TOKENS:-5}
PORT_MOE_BACKEND=${PORT_MOE_BACKEND:-mxfp4_int8_sm80}
PORT_EXTRA_ARGS=${PORT_EXTRA_ARGS:-}

spec_config=$(printf '{"method":"dspark","num_speculative_tokens":%s,"draft_sample_method":"greedy","moe_backend":"%s","enforce_eager":true,"enable_adaptive_verification":false,"dspark_pipeline_stage_ranks":[1,2,3]}' \
    "${PORT_DSPARK_TOKENS}" "${PORT_MOE_BACKEND}")

exec "${VLLM_BIN}" serve "${PORT_MODEL_DIR}" \
    --served-model-name dsv41f-port \
    --host 0.0.0.0 \
    --port "${PORT_PORT}" \
    --pipeline-parallel-size 5 \
    --tensor-parallel-size 1 \
    --tokenizer-mode deepseek_v41 \
    --attention-backend TRITON_MLA_SPARSE_DSV41 \
    --moe-backend "${PORT_MOE_BACKEND}" \
    --engram-config '{"cpu_offload":true,"embedding_across_dp":false}' \
    --limit-mm-per-prompt '{"image":0}' \
    --kv-cache-dtype fp8 \
    --block-size 128 \
    --kv-cache-memory-bytes "${PORT_KV_BYTES}" \
    --max-model-len "${PORT_MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${PORT_MAX_BATCHED}" \
    --max-num-seqs "${PORT_MAX_SEQS}" \
    --speculative-config "${spec_config}" \
    --gpu-memory-utilization 0.98 \
    --compilation-config '{"cudagraph_mode":"FULL"}' \
    --no-enable-prefix-caching \
    --no-enable-flashinfer-autotune \
    --reasoning-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v41 \
    ${PORT_EXTRA_ARGS} \
    "$@"
