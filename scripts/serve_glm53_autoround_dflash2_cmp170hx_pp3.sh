#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
VLLM_BIN=${VLLM_BIN:-${REPO_ROOT}/.venv/bin/vllm}
PYTHON_BIN=${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}
MODEL_DIR=${MODEL_DIR:-/media/z/data6/models/GLM-5.3-Flash-W4A16-AutoRound}
DFLASH_MODEL_DIR=${DFLASH_MODEL_DIR:-/media/z/data6/models/GLM-5.3-Flash-DFlash2}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-${MODEL_DIR}/chat_template.jinja}
ENABLE_DFLASH=${ENABLE_DFLASH:-1}
ENABLE_MULTIMODAL=${ENABLE_MULTIMODAL:-0}

for toggle in ENABLE_DFLASH ENABLE_MULTIMODAL; do
    value=${!toggle}
    case "${value}" in
        0|1) ;;
        *)
            echo "${toggle} must be 0 or 1: ${value}" >&2
            exit 2
            ;;
    esac
done

if [[ ! -x "${VLLM_BIN}" ]]; then
    echo "vLLM executable not found: ${VLLM_BIN}" >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 2
fi
for required in config.json model.safetensors.index.json; do
    if [[ ! -s "${MODEL_DIR}/${required}" ]]; then
        echo "GLM-5.3 download is incomplete: ${MODEL_DIR}/${required}" >&2
        exit 2
    fi
done
for shard in $(seq 1 34); do
    printf -v shard_name '%05d' "${shard}"
    path=${MODEL_DIR}/model-${shard_name}-of-00034.safetensors
    if [[ ! -s "${path}" ]]; then
        echo "GLM-5.3 download is incomplete: ${path}" >&2
        exit 2
    fi
done
if [[ "${ENABLE_DFLASH}" == "1" ]]; then
    for required in config.json model.safetensors; do
        if [[ ! -s "${DFLASH_MODEL_DIR}/${required}" ]]; then
            echo "Invalid GLM-5.3 DFlash2 directory: ${DFLASH_MODEL_DIR}/${required}" >&2
            exit 2
        fi
    done
fi
if [[ ! -s "${CHAT_TEMPLATE}" ]]; then
    echo "Chat template not found: ${CHAT_TEMPLATE}" >&2
    exit 2
fi
if ! "${PYTHON_BIN}" - "${MODEL_DIR}" "${DFLASH_MODEL_DIR}" "${ENABLE_DFLASH}" <<'PY'
import json
import sys
from pathlib import Path


def validate_safetensors(path: Path) -> None:
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"missing safetensors header: {path}")
        header_size = int.from_bytes(prefix, "little")
        if header_size <= 0 or 8 + header_size > size:
            raise ValueError(f"incomplete safetensors header: {path}")
        header = json.loads(handle.read(header_size))

    data_ends = [
        metadata["data_offsets"][1]
        for name, metadata in header.items()
        if name != "__metadata__"
    ]
    expected_size = 8 + header_size + max(data_ends, default=0)
    if size != expected_size:
        raise ValueError(
            f"incomplete safetensors file: {path} ({size} != {expected_size})"
        )


try:
    model_dir = Path(sys.argv[1])
    draft_dir = Path(sys.argv[2])
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    numbered_shards = {
        f"model-{shard:05d}-of-00034.safetensors" for shard in range(1, 35)
    }
    missing_shards = sorted(numbered_shards.difference(shards))
    if missing_shards:
        raise ValueError(f"target index is missing shards: {missing_shards}")
    for shard in shards:
        if Path(shard).name != shard:
            raise ValueError(f"target index contains an invalid path: {shard}")
        validate_safetensors(model_dir / shard)
    if sys.argv[3] == "1":
        validate_safetensors(draft_dir / "model.safetensors")
except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
    print(f"Model download validation failed: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
then
    exit 2
fi

export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,5}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-1}
# Keep the last target stage lighter because DFlash2 and its full-vocabulary
# lm_head are constructed there. This is the measured three-stage default;
# callers can provide a different PP size and matching layer partition.
export VLLM_PP_LAYER_PARTITION=${VLLM_PP_LAYER_PARTITION:-17,15,13}
export VLLM_PP_REUSE_RECV_BUFFER=${VLLM_PP_REUSE_RECV_BUFFER:-0}
export VLLM_KV_CACHE_LAYOUT=${VLLM_KV_CACHE_LAYOUT:-LBHNC}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8099}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-glm53-flash-autoround-dflash2}
PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-3}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
# Use a larger prefill chunk to amortize PP and scheduler overhead. The PP
# partition keeps the DFlash2 stage lighter for its larger activation workspace.
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2048}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.96}
NUM_GPU_BLOCKS_OVERRIDE=${NUM_GPU_BLOCKS_OVERRIDE:-}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8}
DFLASH_KV_CACHE_DTYPE=${DFLASH_KV_CACHE_DTYPE:-auto}
DFLASH_NUM_SPECULATIVE_TOKENS=${DFLASH_NUM_SPECULATIVE_TOKENS:-7}
COMPILATION_CONFIG=${COMPILATION_CONFIG:-'{"cudagraph_mode":"FULL"}'}
LIMIT_MM_PER_PROMPT=${LIMIT_MM_PER_PROMPT:-'{"image":1,"video":0}'}
MM_PROCESSOR_KWARGS=${MM_PROCESSOR_KWARGS:-'{"max_image_tokens":2048}'}
MM_PROCESSOR_CACHE_GB=${MM_PROCESSOR_CACHE_GB:-1}
MEDIA_IO_KWARGS=${MEDIA_IO_KWARGS:-'{}'}

SPECULATIVE_CONFIG=$(printf \
    '{"method":"dflash","model":"%s","num_speculative_tokens":%s,"draft_tensor_parallel_size":1,"draft_sample_method":"probabilistic","rejection_sample_method":"standard","attention_backend":"TRITON_ATTN","kv_cache_dtype":"%s"}' \
    "${DFLASH_MODEL_DIR}" \
    "${DFLASH_NUM_SPECULATIVE_TOKENS}" \
    "${DFLASH_KV_CACHE_DTYPE}")

SPECULATIVE_ARGS=()
if [[ "${ENABLE_DFLASH}" == "1" ]]; then
    SPECULATIVE_ARGS=(--speculative-config "${SPECULATIVE_CONFIG}")
fi

KV_CACHE_SIZE_ARGS=()
if [[ -n "${NUM_GPU_BLOCKS_OVERRIDE}" ]]; then
    if [[ ! "${NUM_GPU_BLOCKS_OVERRIDE}" =~ ^[1-9][0-9]*$ ]]; then
        echo "NUM_GPU_BLOCKS_OVERRIDE must be a positive integer" >&2
        exit 2
    fi
    KV_CACHE_SIZE_ARGS=(--num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE}")
fi

MULTIMODAL_ARGS=(--language-model-only --skip-mm-profiling)
if [[ "${ENABLE_MULTIMODAL}" == "1" ]]; then
    MULTIMODAL_ARGS=(
        --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}"
        --mm-processor-kwargs "${MM_PROCESSOR_KWARGS}"
        --mm-processor-cache-gb "${MM_PROCESSOR_CACHE_GB}"
        --media-io-kwargs "${MEDIA_IO_KWARGS}"
        --skip-mm-profiling
    )
fi

exec "${VLLM_BIN}" serve "${MODEL_DIR}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}" \
    --block-size 256 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    "${KV_CACHE_SIZE_ARGS[@]}" \
    --attention-backend TRITON_MLA_SPARSE_SM80 \
    "${MULTIMODAL_ARGS[@]}" \
    --compilation-config "${COMPILATION_CONFIG}" \
    --disable-cascade-attn \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --generation-config vllm \
    --chat-template "${CHAT_TEMPLATE}" \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --enable-auto-tool-choice \
    "${SPECULATIVE_ARGS[@]}" \
    "$@"
