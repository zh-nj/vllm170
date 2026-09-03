#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ -z "${DSV4_MODEL_DIR:-}" ]]; then
    for candidate in \
        /mnt/data6/models/DeepSeek-V4-Flash-Vision-Exp \
        /media/z/data6/models/DeepSeek-V4-Flash-Vision-Exp; do
        if [[ -d "${candidate}" ]]; then
            DSV4_MODEL_DIR=${candidate}
            break
        fi
    done
fi

export DSV4_MODEL_DIR
export DSV4_SERVED_MODEL=${DSV4_SERVED_MODEL:-dsv4-vision}
export DSV4_DSPARK_TOKENS=${DSV4_DSPARK_TOKENS:-3}
export DSV4_DSPARK_DRAFT_SAMPLE_METHOD=${DSV4_DSPARK_DRAFT_SAMPLE_METHOD:-probabilistic}

if ((DSV4_DSPARK_TOKENS > 3 && DSV4_DSPARK_TOKENS % 3 != 0)); then
    echo "Vision checkpoint DSpark width must be 3 or a multiple of 3." >&2
    exit 2
fi

exec "${SCRIPT_DIR}/run_cmp170hx_pp3.sh" "$@"
