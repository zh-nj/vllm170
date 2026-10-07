#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
VLLM_BIN=${VLLM_BIN:-${REPO_ROOT}/.venv/bin/vllm}
PYTHON_BIN=${REPO_ROOT}/.venv/bin/python
MODEL_DIR=${MODEL_DIR:-/media/z/data6/models/Qwen3.8-Flash-Next-W4A16-AutoRound}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-${MODEL_DIR}/chat_template.jinja}
CHECK_ONLY=${CHECK_ONLY:-0}
MTP_TOKENS=${MTP_TOKENS:-3}
QWEN_MTP_PP_OPTIMIZATIONS=${QWEN_MTP_PP_OPTIMIZATIONS:-1}
QWEN_MTP_PREPARE_AHEAD=${QWEN_MTP_PREPARE_AHEAD:-0}
QWEN_MTP_PREFIX_METADATA=${QWEN_MTP_PREFIX_METADATA:-0}
QWEN_MTP_METADATA_REUSE=${QWEN_MTP_METADATA_REUSE:-0}
COMPILATION_CONFIG=${COMPILATION_CONFIG:-'{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,12,16]}'}

case "${CHECK_ONLY}" in
    0|1) ;;
    *) echo "CHECK_ONLY must be 0 or 1" >&2; exit 2 ;;
esac
for setting in QWEN_MTP_PP_OPTIMIZATIONS QWEN_MTP_PREPARE_AHEAD QWEN_MTP_PREFIX_METADATA QWEN_MTP_METADATA_REUSE; do
    case "${!setting}" in
        0|1) ;;
        *) echo "${setting} must be 0 or 1" >&2; exit 2 ;;
    esac
done
json_bool() {
    if [[ "$1" == 1 ]]; then printf true; else printf false; fi
}
MTP_ADDITIONAL_CONFIG="{\"qwen4_exp_mtp_pp_optimizations\":$(json_bool "${QWEN_MTP_PP_OPTIMIZATIONS}"),\"qwen4_exp_mtp_prepare_ahead\":$(json_bool "${QWEN_MTP_PREPARE_AHEAD}"),\"qwen4_exp_mtp_prefix_metadata\":$(json_bool "${QWEN_MTP_PREFIX_METADATA}"),\"qwen4_exp_mtp_metadata_reuse\":$(json_bool "${QWEN_MTP_METADATA_REUSE}")}"
if [[ ! "${MTP_TOKENS}" =~ ^[0-9]+$ ]]; then
    echo "MTP_TOKENS must be a nonnegative integer (0 disables MTP)" >&2
    exit 2
fi
SPECULATIVE_ARGS=()
if (( MTP_TOKENS > 0 )); then
    SPECULATIVE_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}")
fi
if [[ ! -x "${VLLM_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing vLLM or Python executable in ${REPO_ROOT}/.venv/bin" >&2
    exit 2
fi
if [[ ! -s "${CHAT_TEMPLATE}" ]]; then
    echo "Chat template not found: ${CHAT_TEMPLATE}" >&2
    exit 2
fi

"${PYTHON_BIN}" - "${MODEL_DIR}" <<'PY'
import json
import sys
from pathlib import Path

from safetensors import safe_open

try:
    model_dir = Path(sys.argv[1])
    config = json.loads((model_dir / "config.json").read_text())
    quant = config["quantization_config"]
    expected = {
        "quant_method": "auto-round",
        "packing_format": "auto_round:auto_gptq",
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "data_type": "int",
    }
    if config["model_type"] != "qwen4_exp":
        raise ValueError("expected a Qwen4Exp checkpoint")
    for field, value in expected.items():
        if quant.get(field) != value:
            raise ValueError(f"expected {field}={value!r}, got {quant.get(field)!r}")
    weight_map = json.loads(
        (model_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]
    if not weight_map:
        raise ValueError("empty checkpoint index")
    shards = {}
    for name, shard in weight_map.items():
        if Path(shard).name != shard:
            raise ValueError(f"invalid checkpoint path: {shard}")
        shards.setdefault(shard, set()).add(name)
    for shard, names in sorted(shards.items()):
        with safe_open(str(model_dir / shard), framework="pt") as checkpoint:
            missing = names.difference(checkpoint.keys())
            if missing:
                raise ValueError(f"{shard} is missing tensor {min(missing)}")
    print(f"Validated {len(shards)} files and {len(weight_map)} indexed tensors")
except Exception as exc:
    print(f"Checkpoint validation failed: {exc}", file=sys.stderr)
    raise SystemExit(2)
PY

export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export VLLM_PP_LAYER_PARTITION=${VLLM_PP_LAYER_PARTITION:-26,22}
export VLLM_QWEN4_EXP_PLE_CPU_OFFLOAD=1
export VLLM_QWEN4_EXP_HC_SM80_GEMV=${VLLM_QWEN4_EXP_HC_SM80_GEMV:-1}
export VLLM_QWEN4_EXP_QSA_MERGED_PROJECTION=${VLLM_QWEN4_EXP_QSA_MERGED_PROJECTION:-1}
export VLLM_QWEN4_EXP_QSA_FUSED_PREPARE=${VLLM_QWEN4_EXP_QSA_FUSED_PREPARE:-1}
export VLLM_QWEN4_EXP_MTP_FP8_LM_HEAD=${VLLM_QWEN4_EXP_MTP_FP8_LM_HEAD:-1}
if (( MTP_TOKENS == 0 )); then
    export VLLM_MOE_ALIGN_M1_E512_SM80=${VLLM_MOE_ALIGN_M1_E512_SM80:-1}
fi
export VLLM_SKIP_P2P_CHECK=${VLLM_SKIP_P2P_CHECK:-1}

COMMAND=(
    "${VLLM_BIN}" serve "${MODEL_DIR}"
    --served-model-name "${SERVED_MODEL_NAME:-Qwen3.8-Flash-Next-W4A16-AutoRound}"
    --host "${HOST:-0.0.0.0}"
    --port "${PORT:-26661}"
    --trust-remote-code
    --quantization inc
    --dtype bfloat16
    --max-model-len "${MAX_MODEL_LEN:-262144}"
    --pipeline-parallel-size 2
    --tensor-parallel-size 1
    --disable-custom-all-reduce
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
    --enable-prefix-caching
    --max-num-seqs "${MAX_NUM_SEQS:-4}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-4096}"
    "${SPECULATIVE_ARGS[@]}"
    --additional-config "${MTP_ADDITIONAL_CONFIG}"
    --compilation-config "${COMPILATION_CONFIG}"
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --exclude-tools-when-tool-choice-none
    --reasoning-parser qwen3
    --chat-template "${CHAT_TEMPLATE}"
    --chat-template-content-format openai
    --default-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"xhigh"}'
    "$@"
)

cd -- "${REPO_ROOT}"
if [[ "${CHECK_ONLY}" == 1 ]]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi
exec "${COMMAND[@]}"
