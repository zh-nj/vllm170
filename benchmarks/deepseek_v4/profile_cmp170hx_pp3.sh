#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}
PROFILE_ROOT=${DSV4_PROFILE_ROOT:-${REPO_ROOT}/profiles/deepseek_v4}
PROFILE_CONTEXT=${DSV4_PROFILE_CONTEXT:-32000}
PROFILE_TOKENS=${DSV4_PROFILE_TOKENS:-64}
PROFILE_REPEATS=${DSV4_PROFILE_REPEATS:-1}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=${PROFILE_ROOT}/${STAMP}
TRACE_DIR=${RUN_DIR}/torch_trace
PHASE_DIR=${RUN_DIR}/phase_trace
PHASE_CONTROL=${RUN_DIR}/phase_profile.active
PROFILE_EXECUTION=${DSV4_PROFILE_EXECUTION:-detail}
GPU_SAMPLE_INTERVAL=${DSV4_PROFILE_GPU_SAMPLE_INTERVAL:-0.5}

export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3}
export DSV4_DSPARK_TOKENS=${DSV4_DSPARK_TOKENS:-5}
export DSV4_DSPARK_DRAFT_SAMPLE_METHOD=${DSV4_DSPARK_DRAFT_SAMPLE_METHOD:-greedy}
export DSV4_MAX_BATCHED_TOKENS=${DSV4_MAX_BATCHED_TOKENS:-1715}
export DSV4_MAX_NUM_SEQS=${DSV4_MAX_NUM_SEQS:-8}
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-96}
export VLLM_SPARSE_INDEXER_MAX_QUERY_ROWS=${VLLM_SPARSE_INDEXER_MAX_QUERY_ROWS:-1024}
export VLLM_INDEXER_PREFILL_CALLER_LOGITS=${VLLM_INDEXER_PREFILL_CALLER_LOGITS:-1}
export VLLM_MARLIN_MOE_BLOCK_SIZE=${VLLM_MARLIN_MOE_BLOCK_SIZE:-48}
export VLLM_MARLIN_MOE_BLOCK_SIZE_MIN_M=${VLLM_MARLIN_MOE_BLOCK_SIZE_MIN_M:-1000}
export VLLM_DEEPSEEK_V4_SPARSE_PREFILL_CALLER_OUTPUT=${VLLM_DEEPSEEK_V4_SPARSE_PREFILL_CALLER_OUTPUT:-1}
export VLLM_DEEPSEEK_V4_SPARSE_PREFILL_BLOCK_K=${VLLM_DEEPSEEK_V4_SPARSE_PREFILL_BLOCK_K:-32}
export VLLM_DEEPSEEK_V4_SPARSE_DECODE_CALLER_OUTPUT=${VLLM_DEEPSEEK_V4_SPARSE_DECODE_CALLER_OUTPUT:-1}
export VLLM_DEEPSEEK_V4_SPARSE_DECODE_SPLIT=${VLLM_DEEPSEEK_V4_SPARSE_DECODE_SPLIT:-1}
export VLLM_DEEPSEEK_V4_DIRECT_RAGGED_SWA=${VLLM_DEEPSEEK_V4_DIRECT_RAGGED_SWA:-1}
export VLLM_DEEPSEEK_V4_DIRECT_RAGGED_C128=${VLLM_DEEPSEEK_V4_DIRECT_RAGGED_C128:-1}
export VLLM_DEEPSEEK_V4_DSPARK_KV_ONLY_INSERT=${VLLM_DEEPSEEK_V4_DSPARK_KV_ONLY_INSERT:-1}
export VLLM_DEEPSEEK_V4_MHC_PRE_BROADCAST=${VLLM_DEEPSEEK_V4_MHC_PRE_BROADCAST:-1}

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 2
fi
mkdir -p "${TRACE_DIR}" "${PHASE_DIR}"

case "${PROFILE_EXECUTION}" in
    detail)
        profile_execution_args=(--enforce-eager)
        profiler_enabled=1
        ;;
    graph)
        profile_execution_args=()
        profiler_enabled=1
        ;;
    benchmark)
        profile_execution_args=()
        profiler_enabled=0
        ;;
    *)
        echo "DSV4_PROFILE_EXECUTION must be detail, graph, or benchmark" >&2
        exit 2
        ;;
esac

export VLLM_DEEPSEEK_V4_PHASE_PROFILE=${profiler_enabled}
export VLLM_DEEPSEEK_V4_PHASE_PROFILE_PATH="${PHASE_DIR}/phase-{pp_rank}-{pid}.jsonl"
export VLLM_DEEPSEEK_V4_PHASE_PROFILE_CONTROL_PATH="${PHASE_CONTROL}"

terminate_server_group() {
    local pgid=$1
    kill -0 -- "-${pgid}" 2>/dev/null || return 0
    kill -TERM -- "-${pgid}" 2>/dev/null || true
    for _ in {1..20}; do
        kill -0 -- "-${pgid}" 2>/dev/null || return 0
        sleep 0.25
    done
    kill -KILL -- "-${pgid}" 2>/dev/null || true
}

cleanup() {
    rm -f "${PHASE_CONTROL}"
    if [[ -n "${SERVER_WATCHDOG_PID:-}" ]] \
        && kill -0 "${SERVER_WATCHDOG_PID}" 2>/dev/null; then
        kill -TERM "${SERVER_WATCHDOG_PID}" 2>/dev/null || true
        wait "${SERVER_WATCHDOG_PID}" 2>/dev/null || true
    fi
    if [[ -n "${GPU_MEMORY_PID:-}" ]] && kill -0 "${GPU_MEMORY_PID}" 2>/dev/null; then
        kill -TERM "${GPU_MEMORY_PID}" 2>/dev/null || true
        wait "${GPU_MEMORY_PID}" 2>/dev/null || true
    fi
    if [[ -n "${WORKLOAD_PID:-}" ]] && kill -0 "${WORKLOAD_PID}" 2>/dev/null; then
        kill -TERM "${WORKLOAD_PID}" 2>/dev/null || true
        wait "${WORKLOAD_PID}" 2>/dev/null || true
    fi
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 -- "-${SERVER_PID}" 2>/dev/null; then
        terminate_server_group "${SERVER_PID}"
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

profiler_config=$(
    printf '{"profiler":"torch","torch_profiler_dir":"%s",' "${TRACE_DIR}"
    printf '"torch_profiler_with_stack":false,'
    printf '"ignore_frontend":true,"detailed_trace_annotation":true}'
)

profiler_args=()
workload_args=()
if ((profiler_enabled)); then
    profiler_args+=(--profiler-config "${profiler_config}")
else
    workload_args+=(--skip-profiler)
fi

setsid "${SCRIPT_DIR}/run_cmp170hx_pp3.sh" \
    "${profile_execution_args[@]}" \
    "${profiler_args[@]}" \
    "$@" \
    >"${RUN_DIR}/server.log" 2>&1 &
SERVER_PID=$!

(
    while kill -0 "${SERVER_PID}" 2>/dev/null; do
        if rg -q \
            'EngineCore failed to start|RuntimeError: Triton Error \[CUDA\]: out of memory|Worker proc .* died unexpectedly' \
            "${RUN_DIR}/server.log"; then
            terminate_server_group "${SERVER_PID}"
            break
        fi
        sleep 1
    done
) &
SERVER_WATCHDOG_PID=$!

if command -v nvidia-smi >/dev/null && [[ "${GPU_SAMPLE_INTERVAL}" != 0 ]]; then
    (
        printf '%s\n' \
            'timestamp,index,memory_used_mib,memory_total_mib,utilization_gpu_pct,profile_active'
        while kill -0 "${SERVER_PID}" 2>/dev/null; do
            profile_active=0
            [[ -f "${PHASE_CONTROL}" ]] && profile_active=1
            while IFS= read -r sample; do
                printf '%s,%s\n' "${sample}" "${profile_active}"
            done < <(
                nvidia-smi \
                    --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
                    --format=csv,noheader,nounits || true
            )
            sleep "${GPU_SAMPLE_INTERVAL}"
        done
    ) >"${RUN_DIR}/gpu_memory.csv" &
    GPU_MEMORY_PID=$!
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/profile_workload.py" \
    --base-url "http://${DSV4_HOST:-127.0.0.1}:${DSV4_PORT:-8098}" \
    --model "${DSV4_SERVED_MODEL:-dsv4s}" \
    --context "${PROFILE_CONTEXT}" \
    --max-tokens "${PROFILE_TOKENS}" \
    --scenario "${DSV4_PROFILE_SCENARIO:-general}" \
    --temperature "${DSV4_PROFILE_TEMPERATURE:-0}" \
    --repeats "${PROFILE_REPEATS}" \
    --phase-control "${PHASE_CONTROL}" \
    "${workload_args[@]}" \
    --output "${RUN_DIR}/result.jsonl" &
WORKLOAD_PID=$!

completed_pid=
if wait -n -p completed_pid "${SERVER_PID}" "${WORKLOAD_PID}"; then
    completed_status=0
else
    completed_status=$?
fi
if [[ "${completed_pid:-}" == "${SERVER_PID}" ]]; then
    echo "Profiling server exited before the workload completed; see ${RUN_DIR}/server.log" >&2
    if ((completed_status == 0)); then
        completed_status=1
    fi
    exit "${completed_status}"
fi
unset WORKLOAD_PID
if ((completed_status != 0)); then
    exit "${completed_status}"
fi

cleanup
unset SERVER_PID
trap - EXIT INT TERM

if ((profiler_enabled)); then
    analyzer_args=()
    if [[ "${PROFILE_EXECUTION}" == detail ]]; then
        analyzer_args+=(--require-complete)
    fi
    "${PYTHON_BIN}" "${SCRIPT_DIR}/phase_profile_analyze.py" \
        --phase-dir "${PHASE_DIR}" \
        --output-dir "${RUN_DIR}" \
        "${analyzer_args[@]}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/profile_validate.py" --run-dir "${RUN_DIR}"
else
    "${PYTHON_BIN}" "${SCRIPT_DIR}/profile_validate.py" \
        --run-dir "${RUN_DIR}" --workload-only
fi

if ((profiler_enabled)); then
    echo "Per-worker traces: ${TRACE_DIR}"
    echo "Phase summaries: ${RUN_DIR}/phase_{prefill,decode,dspark}.md"
fi
echo "Workload result: ${RUN_DIR}/result.jsonl"
echo "Profile status: ${RUN_DIR}/profile_status.json"
