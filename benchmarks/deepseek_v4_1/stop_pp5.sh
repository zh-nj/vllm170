#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Stop the PP5 server without matching this script's own command line.
#
# `pkill -f "VLLM::"` run from a shell whose own command line contains that
# pattern kills the caller too, so the patterns are assembled here and the
# process list is never passed through a matching argv.
set -uo pipefail

PAT_SERVE=$(printf 'vllm%s' ' serve')
PAT_WORKER=$(printf 'VLLM%s' '::')

for _ in 1 2 3; do
    pids=$(pgrep -f "${PAT_SERVE}" || true)
    pids="${pids} $(pgrep -f "${PAT_WORKER}" || true)"
    pids=$(echo "${pids}" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u || true)
    [ -z "${pids}" ] && break
    # shellcheck disable=SC2086
    kill -9 ${pids} 2>/dev/null || true
    sleep 3
done

echo "remaining: $(pgrep -c -f "${PAT_WORKER}" || echo 0)"
for _ in $(seq 1 40); do
    free_gib=$(free -g | awk '/Mem:|内存/{print $4}')
    [ "${free_gib:-0}" -ge 150 ] && break
    sleep 3
done
echo "host_free=${free_gib:-?}GiB"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
