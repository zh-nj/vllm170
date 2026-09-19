#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Start the PP5 server, riding out the two startup flakes seen on this box.
#
# Extends start_pp5_with_retry.sh with the two waits that the flakes actually
# need, both measured on this box:
#
# 1. Device memory: a failed warmup leaves the five VLLM::Worker processes alive
#    holding ~60 GiB each. They must be gone *and* the driver must have released
#    the memory before the next attempt, or that attempt loads into a card that
#    is already nearly full and dies with the same
#    `ncclUnhandledCudaError ... Cuda failure 2 'out of memory'`.
# 2. Time: with a cold page cache the 48 shards take ~20 min to load, which is
#    longer than the 10 min the older helper waits before killing the attempt.
#
# Failed attempts keep their log as <log>.attempt<N>.
#
# Usage: start_pp5_patient.sh <log-path> [ENV=VALUE ...]
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOG=${1:?usage: start_pp5_patient.sh <log-path> [ENV=VALUE ...]}
shift

HOST_FREE_TARGET_GIB=${HOST_FREE_TARGET_GIB:-200}
ATTEMPTS=${ATTEMPTS:-6}
READY_TIMEOUT_S=${READY_TIMEOUT_S:-1500}

# One launcher at a time. Every attempt starts by killing all VLLM:: workers, so
# a second launcher left running by an earlier experiment silently kills the
# instance the first one is serving from (and vice versa) -- which is exactly
# what a killed-but-looping launcher did to a healthy server here.
LAUNCHER_LOCK=${LAUNCHER_LOCK:-/tmp/dsv41_pp5_launcher.lock}
exec 9>"${LAUNCHER_LOCK}"
if ! flock -n 9; then
    echo "another PP5 launcher holds ${LAUNCHER_LOCK}; refusing to start" >&2
    exit 3
fi

PAT_WORKER=$(printf 'VLLM%s' '::')
PAT_SERVE=$(printf 'vllm%s' ' serve')

wait_for_idle() {
    # Two Engram host tables are 2 x 94.4 GiB and are released by the kernel
    # only after the previous instance is really gone, so give the kernel a
    # generous window: launching into a starved host fails *after* a full 5
    # minute load, which costs far more than waiting here.
    for _ in $(seq 1 120); do
        local busy
        busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
            awk '$1 > 2000' | wc -l)
        # Column 7 is `available`, not `free`: the checkpoint leaves tens of GiB
        # of reclaimable page cache, and the Engram tables can use it, so gating
        # on `free` alone waits forever on an otherwise idle box.
        local free_gib
        free_gib=$(free -g | awk '/Mem:|内存/{print $7}')
        [ "${busy}" -eq 0 ] && [ "${free_gib:-0}" -ge "${HOST_FREE_TARGET_GIB}" ] && return 0
        sleep 5
    done
    return 1
}

for attempt in $(seq 1 "${ATTEMPTS}"); do
    pids=$(pgrep -f "${PAT_WORKER}" || true)
    pids="${pids} $(pgrep -f "${PAT_SERVE}" || true)"
    pids=$(echo "${pids}" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u || true)
    # shellcheck disable=SC2086
    [ -n "${pids}" ] && kill -9 ${pids} 2>/dev/null
    killall -9 VLLM::EngineCore 2>/dev/null || true

    if ! wait_for_idle; then
        # Do not launch into a starved host: the Engram tables would not fit and
        # the run dies after a full load. Retry the wait instead.
        echo "attempt=${attempt} host/gpu never went idle; retrying" >&2
        continue
    fi
    echo "attempt=${attempt} host_avail=$(free -g | awk '/Mem:|内存/{print $7}')GiB gpu_busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l)" >&2

    if [[ -s "${LOG}" ]]; then
        mv "${LOG}" "${LOG}.prev"
    fi
    setsid nohup env "$@" bash "${SCRIPT_DIR}/run_port_pp5.sh" >"${LOG}" 2>&1 </dev/null &

    for _ in $(seq 1 $((READY_TIMEOUT_S / 10))); do
        if grep -qE "Application startup complete" "${LOG}" 2>/dev/null; then
            echo "STARTED attempt=${attempt}"
            exit 0
        fi
        if grep -qE "Engine core initialization failed|Worker proc .* died|EngineDeadError" "${LOG}" 2>/dev/null; then
            break
        fi
        sleep 10
    done
    echo "attempt=${attempt} failed" >&2
    if [[ -s "${LOG}" ]]; then
        mv "${LOG}" "${LOG}.attempt${attempt}"
    fi
done

echo "FAILED after ${ATTEMPTS} attempts" >&2
exit 1
