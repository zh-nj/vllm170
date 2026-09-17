#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
VLLM_BIN=${VLLM_BIN:-${REPO_ROOT}/.venv/bin/vllm}
DSV41_MODEL_DIR=${DSV41_MODEL_DIR:-/media/z/data6/models/DeepSeek-V4.1-Flash}

if [[ ! -x "${VLLM_BIN}" ]]; then
    echo "vLLM executable not found: ${VLLM_BIN}" >&2
    exit 2
fi
if [[ ! -d "${DSV41_MODEL_DIR}" ]]; then
    echo "DeepSeek-V4.1 model directory not found: ${DSV41_MODEL_DIR}" >&2
    exit 2
fi

export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3,4,5}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_PP_REUSE_RECV_BUFFER=${VLLM_PP_REUSE_RECV_BUFFER:-1}
export VLLM_DFLASH_EAGER_TARGET=${VLLM_DFLASH_EAGER_TARGET:-0}
export VLLM_DEEPSEEK_V4_DSPARK_KV_ONLY_INSERT=${VLLM_DEEPSEEK_V4_DSPARK_KV_ONLY_INSERT:-1}
export VLLM_DEEPSEEK_V4_ENGRAM_GPU_DEQUANT=${VLLM_DEEPSEEK_V4_ENGRAM_GPU_DEQUANT:-1}
export VLLM_DEEPSEEK_V4_DECODE_TOPK_CACHE=${VLLM_DEEPSEEK_V4_DECODE_TOPK_CACHE:-1}
export VLLM_DEEPSEEK_V4_PERSIST_SINGLE_REQ_TOKEN_MAP=${VLLM_DEEPSEEK_V4_PERSIST_SINGLE_REQ_TOKEN_MAP:-1}
export VLLM_DEEPSEEK_V4_PERSIST_SINGLE_REQ_VALIDITY=${VLLM_DEEPSEEK_V4_PERSIST_SINGLE_REQ_VALIDITY:-1}
export VLLM_DEEPSEEK_V4_SPARSE_DECODE_SPLIT=${VLLM_DEEPSEEK_V4_SPARSE_DECODE_SPLIT:-1}
export VLLM_DEEPSEEK_V4_SM80_DECODE_NUM_SPLITS=${VLLM_DEEPSEEK_V4_SM80_DECODE_NUM_SPLITS:-32}
export VLLM_DEEPSEEK_V4_SPARSE_DECODE_CALLER_OUTPUT=${VLLM_DEEPSEEK_V4_SPARSE_DECODE_CALLER_OUTPUT:-1}
export VLLM_DEEPSEEK_V4_DIRECT_RAGGED_SWA=${VLLM_DEEPSEEK_V4_DIRECT_RAGGED_SWA:-1}
export VLLM_DEEPSEEK_V4_PAGED_DECODE_SWA=${VLLM_DEEPSEEK_V4_PAGED_DECODE_SWA:-1}
export VLLM_DEEPSEEK_V4_SPARSE_PREFILL_TEMP_MB=${VLLM_DEEPSEEK_V4_SPARSE_PREFILL_TEMP_MB:-512}
# Pin the Marlin M-tile geometry for short batches. Without this, the dense
# MXFP8 GEMMs round differently depending on how many rows the prefill batch
# has, and this model's logits are close enough to ties that a single bf16 ULP
# of reassociation flips the argmax: a prefill of two short prompts answered EOS
# where the same prompt answered correctly on its own. Set to 0 to restore the
# previous behaviour. Costs about 2.6x on the dense GEMMs for 9<=M<=128 (long
# prefills are untouched).
export VLLM_MARLIN_M_ALIGN=${VLLM_MARLIN_M_ALIGN:-1}

# cuBLAS picks the router GEMM's kernel from the problem shape, so a token's
# router logits depend on how many tokens share the batch: for
# layers.1.ffn.gate (5120 -> 384) M in [14, 64] is one kernel and 65..128
# another, and a 5-request prefill of 14-token prompts is M=70. At width 8 that
# cost every prompt its correct answer (30/48); pinning the router to the M<=64
# geometry - the only one a single request is ever computed with - makes the
# co-batched answers identical to the solo ones again (48/48). "split" keeps the
# plain call for M<=64 and uses a bit-identical 16-way split-K kernel above it;
# "chunk" calls cuBLAS once per 64 rows instead, equally exact but slower on
# long prefills. Empty restores the previous behaviour.
export VLLM_DSV41_ROUTER_GEMM_ALIGN=${VLLM_DSV41_ROUTER_GEMM_ALIGN:-split}

DSV41_HOST=${DSV41_HOST:-0.0.0.0}
DSV41_PORT=${DSV41_PORT:-26662}
DSV41_SERVED_MODEL=${DSV41_SERVED_MODEL:-deepseek-v4.1-flash}
DSV41_KV_CACHE_BYTES=${DSV41_KV_CACHE_BYTES:-1300000000}
DSV41_MAX_MODEL_LEN=${DSV41_MAX_MODEL_LEN:-1048576}
DSV41_MAX_BATCHED_TOKENS=${DSV41_MAX_BATCHED_TOKENS:-2048}
DSV41_MAX_NUM_SEQS=${DSV41_MAX_NUM_SEQS:-8}
DSV41_LONG_PREFILL_TOKEN_THRESHOLD=${DSV41_LONG_PREFILL_TOKEN_THRESHOLD:-256}
DSV41_ENABLE_ADAPTIVE_LONG_PREFILL=${DSV41_ENABLE_ADAPTIVE_LONG_PREFILL:-1}
DSV41_MOE_BACKEND=${DSV41_MOE_BACKEND:-mxfp4_int8_sm80}
DSV41_DSPARK_TOKENS=${DSV41_DSPARK_TOKENS:-5}
DSV41_DSPARK_DRAFT_SAMPLE_METHOD=${DSV41_DSPARK_DRAFT_SAMPLE_METHOD:-greedy}
DSV41_DSPARK_MOE_BACKEND=${DSV41_DSPARK_MOE_BACKEND:-mxfp4_int8_sm80}
DSV41_DSPARK_ENFORCE_EAGER=${DSV41_DSPARK_ENFORCE_EAGER:-1}
# The two PP stages that own Engram each hold about 94.42 GiB. A cache budget
# at least that large switches the implementation to full regular host-RAM
# residency; existing converted SSD files are used as the load source.
DSV41_ENGRAM_CPU_CACHE_GB=${DSV41_ENGRAM_CPU_CACHE_GB:-96}
DSV41_ENGRAM_SSD_PATH=${DSV41_ENGRAM_SSD_PATH:-/media/z/data6/vllm-engram-cache}
printf -v DSV41_ENGRAM_CONFIG \
    '{"cpu_offload":true,"embedding_across_dp":false,"cpu_cache_gb":%s,"ssd_offload_path":"%s"}' \
    "${DSV41_ENGRAM_CPU_CACHE_GB}" "${DSV41_ENGRAM_SSD_PATH}"

if [[ ! "${DSV41_DSPARK_TOKENS}" =~ ^[0-9]+$ ]]; then
    echo "DSV41_DSPARK_TOKENS must be a non-negative integer." >&2
    exit 2
fi
case "${DSV41_DSPARK_DRAFT_SAMPLE_METHOD}" in
    greedy|probabilistic) ;;
    *)
        echo "DSV41_DSPARK_DRAFT_SAMPLE_METHOD must be greedy or probabilistic." >&2
        exit 2
        ;;
esac
case "${DSV41_DSPARK_ENFORCE_EAGER}" in
    0) dspark_enforce_eager=false ;;
    1) dspark_enforce_eager=true ;;
    *)
        echo "DSV41_DSPARK_ENFORCE_EAGER must be 0 or 1." >&2
        exit 2
        ;;
esac

DSV41_SPECULATIVE_ARGS=()
export VLLM_PP_LAYER_PARTITION=${VLLM_PP_LAYER_PARTITION:-8,8,8,8,8}
if ((DSV41_DSPARK_TOKENS > 0)); then
    if [[ "${VLLM_PP_LAYER_PARTITION}" != 8,8,8,8,8 ]]; then
        echo "Distributed DSpark requires balanced target partition 8,8,8,8,8." >&2
        exit 2
    fi
    printf -v DSV41_SPECULATIVE_CONFIG \
        '{"method":"dspark","num_speculative_tokens":%s,"draft_sample_method":"%s","moe_backend":"%s","enforce_eager":%s,"enable_adaptive_verification":false,"dspark_pipeline_stage_ranks":[1,2,3]}' \
        "${DSV41_DSPARK_TOKENS}" \
        "${DSV41_DSPARK_DRAFT_SAMPLE_METHOD}" \
        "${DSV41_DSPARK_MOE_BACKEND}" \
        "${dspark_enforce_eager}"
    DSV41_SPECULATIVE_ARGS+=(--speculative-config "${DSV41_SPECULATIVE_CONFIG}")
fi

DSV41_SCHEDULER_ARGS=(
    --long-prefill-token-threshold "${DSV41_LONG_PREFILL_TOKEN_THRESHOLD}"
)
case "${DSV41_ENABLE_ADAPTIVE_LONG_PREFILL}" in
    1) DSV41_SCHEDULER_ARGS+=(--enable-adaptive-long-prefill) ;;
    0) ;;
    *)
        echo "DSV41_ENABLE_ADAPTIVE_LONG_PREFILL must be 0 or 1" >&2
        exit 2
        ;;
esac

exec "${VLLM_BIN}" serve "${DSV41_MODEL_DIR}" \
    --served-model-name "${DSV41_SERVED_MODEL}" \
    --host "${DSV41_HOST}" \
    --port "${DSV41_PORT}" \
    --pipeline-parallel-size 5 \
    --tensor-parallel-size 1 \
    --tokenizer-mode deepseek_v41 \
    --attention-backend TRITON_MLA_SPARSE_DSV4 \
    --moe-backend "${DSV41_MOE_BACKEND}" \
    --engram-config "${DSV41_ENGRAM_CONFIG}" \
    --limit-mm-per-prompt '{"image":0}' \
    --kv-cache-dtype fp8 \
    --block-size 128 \
    --kv-cache-memory-bytes "${DSV41_KV_CACHE_BYTES}" \
    --max-model-len "${DSV41_MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${DSV41_MAX_BATCHED_TOKENS}" \
    --max-num-seqs "${DSV41_MAX_NUM_SEQS}" \
    "${DSV41_SPECULATIVE_ARGS[@]}" \
    "${DSV41_SCHEDULER_ARGS[@]}" \
    --gpu-memory-utilization 0.98 \
    --compilation-config '{"cudagraph_mode":"FULL"}' \
    --kernel-config '{"enable_jit_warmup":false}' \
    --no-enable-prefix-caching \
    --no-enable-flashinfer-autotune \
    --reasoning-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v41 \
    "$@"
