#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export DFLASH_MODEL_DIR=${DFLASH_MODEL_DIR:-/media/z/data6/models/GLM-5.3-Flash-DFlash2}
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8}
# SM80 TRITON_ATTN decodes the small draft cache in BF16; the large target MLA
# cache uses the requested E4M3 FP8 storage.
export DFLASH_KV_CACHE_DTYPE=${DFLASH_KV_CACHE_DTYPE:-auto}

exec "${SCRIPT_DIR}/serve_glm53_exl3_cmp170hx_pp4.sh" "$@"
