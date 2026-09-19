#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Sample per-GPU utilization while a client run is in flight.
#
# Distinguishes a step that is GPU-work-bound (all ranks busy for most of the
# step) from one that is stalled in host-side work or PP transport (ranks idle,
# short bursts). Usage: gpu_busy_sampler.sh <out.csv> <seconds> [interval]
set -uo pipefail

OUT=${1:?usage: gpu_busy_sampler.sh <out.csv> <seconds> [interval]}
SECONDS_TO_RUN=${2:?seconds}
INTERVAL=${3:-0.5}

echo "t_s,index,util_gpu,util_mem,power_w,mem_used_mib" >"${OUT}"
start=$(date +%s.%N)
while :; do
    now=$(date +%s.%N)
    elapsed=$(echo "${now} - ${start}" | bc -l)
    if (($(echo "${elapsed} >= ${SECONDS_TO_RUN}" | bc -l))); then
        break
    fi
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,power.draw,memory.used \
        --format=csv,noheader,nounits |
        while IFS=, read -r idx gu mu pw mem; do
            echo "${elapsed},${idx},${gu},${mu},${pw},${mem}"
        done >>"${OUT}"
    sleep "${INTERVAL}"
done
