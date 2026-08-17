#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export VLLM_PP_LAYER_PARTITION=${VLLM_PP_LAYER_PARTITION:-15,16,12}
export DSV4_MAX_MODEL_LEN=${DSV4_MAX_MODEL_LEN:-1048576}
export DSV4_KV_CACHE_BYTES=${DSV4_KV_CACHE_BYTES:-2700000000}
export DSV4_MAX_NUM_SEQS=${DSV4_MAX_NUM_SEQS:-1}

exec "${SCRIPT_DIR}/run_cmp170hx_pp3.sh" "$@"
