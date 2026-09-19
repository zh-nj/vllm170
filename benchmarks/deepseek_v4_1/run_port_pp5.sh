#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Port launcher: DeepSeek-V4.1-Flash on 5x CMP 170HX (SM80), PP5, TP1.
#
# Model implementation comes from the cpu-hybrid branch (CED, PP-KV replicas,
# SM80 fused index construction); the CPU/GPU hybrid expert path is NOT used,
# so the MoE runs entirely on the GPU. The project's cross-rank KV-sharing
# mechanism carries each kv-source layer's compressed latent, normalized
# indexer K, top-k indices and candidate blocks inside the ordinary PP
# message, which is what makes an 8,8,8,8,8 split legal. Distributed DSpark
# (draft blocks on PP1..PP3, heads on PP4) is kept.
#
# Stopping the server: killing only the `vllm serve` parent leaves the five
# `VLLM::Worker_PP*` processes holding ~62 GiB per card, and the next launch
# then dies in WorkerProc init. Kill them too, e.g.
#   pkill -f "[v]llm serve"; pkill -f "VLLM::Worker"
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
VLLM_BIN=${VLLM_BIN:-${REPO_ROOT}/.venv/bin/vllm}

export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
# The box has 6x CMP 170HX on PCI 1..6 plus a desktop RTX 3080 Ti on PCI 0.
# GPUs 2..6 sit on one NUMA node and are PIX-connected to each other, so the
# PP5 set is taken from there; PCI 1 is left free for burn-in/validation.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,4,5,6}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
# The mixed prefill+decode warmup issues the first collective on the PP
# token-broadcast subgroup; NCCL allocates that communicator's device buffers on
# first use, and the largest rank is down to a few hundred MiB at that moment, so
# the warmup dies with `ncclUnhandledCudaError ... out of memory` unless the
# per-rank allocation is trimmed. With DSpark on, only the 1 MiB buffer setting
# has started reliably: the default OOMs in warmup, and 2 MiB starts but dies on
# the first real step. The trim is free for decode (DSpark off, 32K, token
# accurate: 24.12 ms/step here versus 24.68 with default buffers and 23.80 for
# the 2048-token/1M-context baseline), so the knob is not the reason any decode
# number looks slow -- check PORT_CUDAGRAPH_SIZES first.
export NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-1048576}
export NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-8}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_PP_LAYER_PARTITION=${VLLM_PP_LAYER_PARTITION:-8,8,8,8,8}
# This host has ~252 GiB RAM; the branch's default 64 GiB spare-RAM guard
# would reject the two resident Engram tables (2 x 94.42 GiB) outright.
export VLLM_DSV41_HOST_HEADROOM_GIB=${VLLM_DSV41_HOST_HEADROOM_GIB:-12}
# The V4.1 prefill gather workspace is now sized from the batch being served;
# this only caps what the warmup/profiling reservation asks for, so the rank
# does not hold a full 1M-token bf16 gather (~2-4 GiB) while serving 6-32k
# prompts. Raise it if you intend to serve megatoken prefills.
export VLLM_DSV41_PREFILL_WORKSPACE_TOKENS=${VLLM_DSV41_PREFILL_WORKSPACE_TOKENS:-65536}
# The SM80 sparse-MLA decode split defaults to a heuristic whose search cap is
# 16. The older CMP launcher pinned 32 and measured +3.6% on the 32K/64 gate
# (35.25 -> 36.51 token/s), so the port launcher exposes the same knob; 0 keeps
# the heuristic.
export VLLM_DEEPSEEK_V4_SM80_DECODE_NUM_SPLITS=${VLLM_DEEPSEEK_V4_SM80_DECODE_NUM_SPLITS:-0}
# The PP2 rank (layers 16-23 plus its DSpark draft stage) keeps only ~1.1 GiB of
# device memory free during serving, and the default allocator cannot satisfy a
# 128 MiB request from its fragmented free pool: at 2048 prefill tokens that
# raised "memory allocation failed with OOM on device 2 while trying to allocate
# 134217728 bytes (free: 118226944)".
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True does NOT rescue 2048 (same
# OOM, and it raises the resident pool by ~450 MiB per rank), so the chunk size
# below stays at 1024 with the stock allocator. Override PORT_MAX_BATCHED only
# after PP2 gains real headroom (fewer layers, smaller KV, or no draft stage).

PORT_PORT=${PORT_PORT:-26663}
# Images per prompt, 0 = text only. The checkpoint carries the vision tower
# (267 tensors, ~0.87 GiB) and the model class builds it on the first PP rank
# only, so enabling this adds encoder inputs without moving weights off PP0.
PORT_MM_IMAGES=${PORT_MM_IMAGES:-2}
PORT_MODEL_DIR=${PORT_MODEL_DIR:-/media/z/data6/models/DeepSeek-V4.1-Flash}
PORT_MAX_MODEL_LEN=${PORT_MAX_MODEL_LEN:-1048576}
# Empty means "let the engine profile and size the KV cache itself" (see kv_args).
PORT_KV_BYTES=${PORT_KV_BYTES-1300000000}
PORT_MAX_BATCHED=${PORT_MAX_BATCHED:-1024}
PORT_MAX_SEQS=${PORT_MAX_SEQS:-8}
PORT_DSPARK_TOKENS=${PORT_DSPARK_TOKENS:-5}
PORT_MOE_BACKEND=${PORT_MOE_BACKEND:-mxfp4_int8_sm80}
# NONE runs decode eager. The phase profiler cannot see inside a captured
# graph, so a per-layer decode census has to turn the graph off.
PORT_CUDAGRAPH_MODE=${PORT_CUDAGRAPH_MODE:-FULL}
# Optional JSON list, e.g. "[1]". Restricting the captured sizes to the decode
# size keeps the capture to a single graph, which is what makes the in-graph
# phase profile unambiguous.
PORT_CUDAGRAPH_SIZES=${PORT_CUDAGRAPH_SIZES:-}
PORT_EXTRA_ARGS=${PORT_EXTRA_ARGS:-}
# Serving default. The mixed prefill+decode warmup needs raw free device memory
# for the NCCL broadcast it issues, which the caching allocator cannot supply
# once the whole budget is reserved, so this doubles as the lever for the
# documented startup OOM flake.
PORT_GPU_UTIL=${PORT_GPU_UTIL:-0.98}
# Draft execution. 0 captures the rank-local draft step into FULL CUDA graphs:
# each of the three stage ranks captures its share of the draft blocks, the head
# rank captures the draft-state collapse plus the sequential Markov head. Only
# the PP relay and the context-KV precompute stay eager. Measured at 32K, q5:
# step 104.5 -> 76.9 ms, 55.2 -> 75.2 tok/s (+36%), and the greedy token ids are
# identical to target-only decoding on all five reference prompts. 1 keeps the
# whole draft eager (the behavior before this change).
PORT_DSPARK_EAGER=${PORT_DSPARK_EAGER:-0}
if [[ "${PORT_DSPARK_EAGER}" == "0" ]]; then
    dspark_eager=false
else
    dspark_eager=true
fi

# Prefix caching. This launcher used to hard-code --no-enable-prefix-caching,
# which does not merely skip reuse: it turns block hashing off altogether, so
# vllm:prefix_cache_queries_total stays 0 and a repeated prompt can never hit.
# Nothing here needs it off (CED, which does, is not used by this launcher),
# and the SM80 sparse-SWA backend plus the cross-rank KV replicas both write
# through the ordinary slot_mapping, so a hit reuses the same blocks on every
# rank. 0 restores the old behavior for A/B measurement.
PORT_PREFIX_CACHING=${PORT_PREFIX_CACHING:-1}
pc_args=(--enable-prefix-caching)
if [[ "${PORT_PREFIX_CACHING}" == "0" ]]; then
    pc_args=(--no-enable-prefix-caching)
fi

spec_config=$(printf '{"method":"dspark","num_speculative_tokens":%s,"draft_sample_method":"greedy","moe_backend":"%s","enforce_eager":%s,"enable_adaptive_verification":false,"dspark_pipeline_stage_ranks":[1,2,3]}' \
    "${PORT_DSPARK_TOKENS}" "${PORT_MOE_BACKEND}" "${dspark_eager}")

# DSpark off means "no speculative config at all": passing num_speculative_tokens=0
# still builds the draft stages and reserves their weights and KV on PP1-PP3.
spec_args=()
if [[ -n "${PORT_DSPARK_TOKENS}" && "${PORT_DSPARK_TOKENS}" != "0" ]]; then
    spec_args=(--speculative-config "${spec_config}")
fi

# An empty PORT_KV_BYTES drops --kv-cache-memory-bytes, which is what makes the
# engine run its memory profiler and log the per-rank ledger
# (weights / peak activation / non-torch / KV) instead of skipping it.
kv_args=()
if [[ -n "${PORT_KV_BYTES}" ]]; then
    kv_args=(--kv-cache-memory-bytes "${PORT_KV_BYTES}")
fi

# Images need one override, and the arithmetic is why.
#
# Enabling a modality forces --disable-chunked-mm-input (bidirectional image
# spans cannot be chunked), which makes the engine require
#     max_tokens_per_mm_item <= max_num_batched_tokens,
# and for this checkpoint that item bound is vision_max_n_token + COMPRESS_PAD_TO
# - 1 = 1024 + 2 - 1 = 1025. Raising max_num_batched_tokens to cover 1025 is not
# free: measured on this box, 1024 -> 1152 changed the greedy token ids on 2 of 5
# reference prompts and dropped DSpark acceptance from 5.69 to 5.06 tokens/step,
# identically with and without vision -- the budget itself, not the images.
#
# Capping the per-image token budget at 1023 puts the bound exactly at the
# default 1024 token budget, so the text path keeps byte-identical output
# (verified 5/5 on the reference prompts) and images still get 1022 tokens plus
# the compressor-alignment pad. The override must be flat (``vision_max_n_token``)
# rather than nested (``vision_config.max_image_tokens``): nested dict overrides
# are applied after the config derives that attribute from vision_config, so the
# nested form is silently a no-op.
mm_args=()
if [[ "${PORT_MM_IMAGES}" != "0" ]]; then
    mm_args=(--hf-overrides '{"vision_max_n_token":1023}')
    if [[ "${PORT_MAX_BATCHED}" -lt 1024 ]]; then
        echo "PORT_MAX_BATCHED=${PORT_MAX_BATCHED} is below the 1024 an image" \
            "item needs; raise it or the engine will refuse to start." >&2
        exit 2
    fi
fi

exec "${VLLM_BIN}" serve "${PORT_MODEL_DIR}" \
    --served-model-name dsv41f-port \
    --host 0.0.0.0 \
    --port "${PORT_PORT}" \
    --pipeline-parallel-size 5 \
    --tensor-parallel-size 1 \
    --tokenizer-mode deepseek_v41 \
    --attention-backend TRITON_MLA_SPARSE_DSV41 \
    --moe-backend "${PORT_MOE_BACKEND}" \
    --engram-config '{"cpu_offload":true,"embedding_across_dp":false}' \
    --limit-mm-per-prompt "{\"image\":${PORT_MM_IMAGES}}" \
    --kv-cache-dtype fp8 \
    --block-size 128 \
    "${kv_args[@]}" \
    --max-model-len "${PORT_MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${PORT_MAX_BATCHED}" \
    --max-num-seqs "${PORT_MAX_SEQS}" \
    "${spec_args[@]}" \
    "${mm_args[@]}" \
    --gpu-memory-utilization "${PORT_GPU_UTIL}" \
    --compilation-config "$(printf '{\"cudagraph_mode\":\"%s\"%s}' "${PORT_CUDAGRAPH_MODE}" "${PORT_CUDAGRAPH_SIZES:+, \"cudagraph_capture_sizes\": ${PORT_CUDAGRAPH_SIZES}}")" \
    --no-enable-flashinfer-autotune \
    "${pc_args[@]}" \
    --reasoning-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v41 \
    ${PORT_EXTRA_ARGS} \
    "$@"
