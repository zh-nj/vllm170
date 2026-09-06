#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,4,5}
export PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-4}
export VLLM_PP_LAYER_PARTITION=${VLLM_PP_LAYER_PARTITION:-16,14,13,2}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-524288}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2048}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
# The global GLM+DFlash scheduler needs 155 blocks for 512K tokens shared by
# up to four concurrent requests. BlockPool permanently holds one null block,
# so allocate 156 blocks per rank.
export NUM_GPU_BLOCKS_OVERRIDE=${NUM_GPU_BLOCKS_OVERRIDE:-156}
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8}
export DFLASH_KV_CACHE_DTYPE=${DFLASH_KV_CACHE_DTYPE:-auto}
# Capture target verification and the DFlash drafter for every supported
# request count. Set VLLM_DFLASH_EAGER_TARGET=1 only for diagnostics.
export COMPILATION_CONFIG=${COMPILATION_CONFIG:-'{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[8,16,24,32]}'}
export VLLM_DFLASH_EAGER_TARGET=${VLLM_DFLASH_EAGER_TARGET:-0}
export ENABLE_MULTIMODAL=${ENABLE_MULTIMODAL:-1}
export LIMIT_MM_PER_PROMPT=${LIMIT_MM_PER_PROMPT:-'{"image":4,"video":1}'}
export MM_PROCESSOR_KWARGS=${MM_PROCESSOR_KWARGS:-'{"max_image_tokens":2048}'}
export MM_PROCESSOR_CACHE_GB=${MM_PROCESSOR_CACHE_GB:-1}

exec "${SCRIPT_DIR}/serve_glm53_autoround_dflash2_cmp170hx_pp3.sh" "$@"
