#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Start the PP5 server, riding out the two startup flakes seen on this box.
#
# 1. Host memory must be free before starting. The two Engram host tables are
#    2 x 94.4 GiB; after the previous instance is killed the kernel needs
#    ~20-30 s to reclaim them. Starting too early makes the new instance's table
#    build (plus the warmup JIT compilations) fight for the last few GiB, and
#    the run dies in warmup with
#    `ncclUnhandledCudaError ... Cuda failure 2 'out of memory'`.
# 2. Even on a clean host, roughly one warmup in two used to fail the same way
#    at the `mixed tokens=16` step or hang with
#    `No available shared memory broadcast block found in 60 seconds`, so both
#    are treated as failure signals and the launch is retried.
#
# Usage: start_pp5_with_retry.sh <log-path> [ENV=VALUE ...]
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOG=${1:?usage: start_pp5_with_retry.sh <log-path> [ENV=VALUE ...]}
shift

HOST_FREE_TARGET_GIB=${HOST_FREE_TARGET_GIB:-150}
ATTEMPTS=${ATTEMPTS:-3}

for attempt in $(seq 1 "${ATTEMPTS}"); do
    for p in $(pgrep -f "VLLM::" || true); do kill -9 "$p" 2>/dev/null || true; done
    for _ in $(seq 1 30); do
        free_gib=$(free -g | awk '/Mem:|内存/{print $4}')
        [ "${free_gib:-0}" -ge "${HOST_FREE_TARGET_GIB}" ] && break
        sleep 4
    done
    echo "attempt=${attempt} host_free=${free_gib:-?}GiB" >&2

    rm -f "${LOG}"
    setsid nohup env "$@" bash "${SCRIPT_DIR}/run_port_pp5.sh" >"${LOG}" 2>&1 </dev/null &

    for _ in $(seq 1 60); do
        if grep -qE "Application startup complete" "${LOG}" 2>/dev/null; then
            echo "STARTED attempt=${attempt}"
            exit 0
        fi
        if grep -qE "Engine core initialization failed|Worker failed to start" "${LOG}" 2>/dev/null; then
            break
        fi
        if [ "$(grep -c 'No available shared memory broadcast block' "${LOG}" 2>/dev/null)" -ge 2 ]; then
            break
        fi
        sleep 10
    done
    echo "attempt=${attempt} failed" >&2
done

echo "FAILED after ${ATTEMPTS} attempts" >&2
exit 1
