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
# The default chunk is 2048. Earlier DSpark-enabled runs exhausted PP2's
# activation headroom at this size; PORT_MAX_BATCHED=1024 remains available
# for that configuration. The CED/DSpark-off path is measured separately.

PORT_PORT=${PORT_PORT:-26663}
# Images per prompt, 0 = text only. The checkpoint carries the vision tower
# (267 tensors, ~0.87 GiB) and the model class builds it on the first PP rank
# only, so enabling this adds encoder inputs without moving weights off PP0.
PORT_MM_IMAGES=${PORT_MM_IMAGES:-2}
PORT_MODEL_DIR=${PORT_MODEL_DIR:-/media/z/data6/models/DeepSeek-V4.1-Flash}
# Optional SSD Engram tables; the FIFO budget is per local table, in GiB.
PORT_ENGRAM_SSD_PATH=${PORT_ENGRAM_SSD_PATH:-}
PORT_ENGRAM_CPU_CACHE_GB=${PORT_ENGRAM_CPU_CACHE_GB:-0}
PORT_ENGRAM_SCALE_IN_RAM=${PORT_ENGRAM_SCALE_IN_RAM:-1}
PORT_ENGRAM_IO_WORKERS=${PORT_ENGRAM_IO_WORKERS:-8}
PORT_ENGRAM_BATCH_READS=${PORT_ENGRAM_BATCH_READS:-0}
PORT_ENGRAM_PREFETCH=${PORT_ENGRAM_PREFETCH:-0}
engram_config=$("${REPO_ROOT}/.venv/bin/python" - "${PORT_ENGRAM_SSD_PATH}" "${PORT_ENGRAM_CPU_CACHE_GB}" "${PORT_ENGRAM_SCALE_IN_RAM}" "${PORT_ENGRAM_IO_WORKERS}" "${PORT_ENGRAM_BATCH_READS}" "${PORT_ENGRAM_PREFETCH}" <<'PY'
import json
import math
import sys

path, raw_budget, raw_scale, raw_workers, raw_batch, raw_prefetch = sys.argv[1:]
budget = float(raw_budget)
if not math.isfinite(budget) or budget < 0 or (budget and not path):
    raise ValueError("Engram cache budget must be finite, non-negative and have an SSD path")
config = {"cpu_offload": True, "embedding_across_dp": False}
if path:
    workers = int(raw_workers)
    if (
        raw_scale not in ("0", "1")
        or raw_batch not in ("0", "1")
        or raw_prefetch not in ("0", "1")
        or not 1 <= workers <= 32
    ):
        raise ValueError("Engram flags must be 0/1 and IO workers must be 1..32")
    config.update(ssd_offload_path=path, cpu_cache_gb=budget,
                  ssd_scale_in_ram=raw_scale == "1", ssd_io_workers=workers,
                  ssd_batch_reads=raw_batch == "1",
                  ssd_prefetch=raw_prefetch == "1")
print(json.dumps(config))
PY
)
PORT_MAX_MODEL_LEN=${PORT_MAX_MODEL_LEN:-1048576}
# Empty means "let the engine profile and size the KV cache itself" (see kv_args).
PORT_KV_BYTES=${PORT_KV_BYTES-1300000000}
PORT_MAX_BATCHED=${PORT_MAX_BATCHED:-2048}
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
# Fast prefill has produced divergent output and incorrect EOS behavior on this
# PP5 DSpark deployment. Reject it even if an older caller sets its launcher
# switch; the additional-config path is checked below.
PORT_FAST_PREFILL=${PORT_FAST_PREFILL:-0}
if [[ "${PORT_FAST_PREFILL}" != "0" ]]; then
    echo "PORT_FAST_PREFILL must be 0; fast prefill is disabled for this deployment." >&2
    exit 2
fi
# CED is the validated prefill default. Set PORT_CED_PREFILL=0 for a baseline.
PORT_CED_PREFILL=${PORT_CED_PREFILL:-1}
if [[ "${PORT_CED_PREFILL}" != "0" && "${PORT_CED_PREFILL}" != "1" ]]; then
    echo "PORT_CED_PREFILL must be 0 or 1." >&2
    exit 2
fi
# These DSV41F SM80 paths are validated production configuration candidates.
# Keep them independently overridable so a caller can reproduce a rollback
# without replacing the complete additional-config object.
PORT_FUSED_SWA_METADATA=${PORT_FUSED_SWA_METADATA:-1}
PORT_MOE_HYBRID_PREFILL=${PORT_MOE_HYBRID_PREFILL:-1}
PORT_MOE_SMALL_TAIL_PREFILL=${PORT_MOE_SMALL_TAIL_PREFILL:-1}
for flag_name in PORT_FUSED_SWA_METADATA PORT_MOE_HYBRID_PREFILL \
    PORT_MOE_SMALL_TAIL_PREFILL; do
    flag_value=${!flag_name}
    if [[ "${flag_value}" != "0" && "${flag_value}" != "1" ]]; then
        echo "${flag_name} must be 0 or 1." >&2
        exit 2
    fi
done
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

# CED cache hits leave a full decoder replay suffix uncached.
PORT_PREFIX_CACHING=${PORT_PREFIX_CACHING:-1}
if [[ "${PORT_PREFIX_CACHING}" != "0" && "${PORT_PREFIX_CACHING}" != "1" ]]; then
    echo "PORT_PREFIX_CACHING must be 0 or 1." >&2
    exit 2
fi
pc_args=(--enable-prefix-caching)
if [[ "${PORT_PREFIX_CACHING}" == "0" ]]; then
    pc_args=(--no-enable-prefix-caching)
fi

# PORT_DSPARK_QUERY_TOKENS sets the draft backbone's query block separately from
# the number of proposed tokens (default: they are equal). The checkpoint was
# trained with dspark_block_size 5, and raising PORT_DSPARK_TOKENS alone makes the
# query block 6 (q=7) or 7 (q=8), which fails startup with an illegal memory
# access during the draft graph replay. Setting this to a multiple of 5 (10)
# starts cleanly, so exceeding the trained block needs it. See
# docs/design/vllm170-dsv41f-verify-q1-q6-2026-09-28.md section 9.
PORT_DSPARK_QUERY_TOKENS=${PORT_DSPARK_QUERY_TOKENS:-}
dspark_query_arg=""
if [[ -n "${PORT_DSPARK_QUERY_TOKENS}" ]]; then
    dspark_query_arg=",\"dspark_num_query_tokens\":${PORT_DSPARK_QUERY_TOKENS}"
fi
spec_config=$(printf '{"method":"dspark","num_speculative_tokens":%s,"draft_sample_method":"greedy","moe_backend":"%s","enforce_eager":%s,"enable_adaptive_verification":false,"dspark_pipeline_stage_ranks":[1,2,3]%s}' \
    "${PORT_DSPARK_TOKENS}" "${PORT_MOE_BACKEND}" "${dspark_eager}" "${dspark_query_arg}")

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
# - 1 = vision_max_n_token + 1. The bound has to stay under the *chunk* a
# request gets, not under the budget: the scheduler admits
# max_num_batched_tokens - max_num_new_slots_for_drafting tokens per request, so
# the cap is PORT_MAX_BATCHED - 1 - PORT_DSPARK_SLOTS. An item that does not fit
# is rolled back and deferred on every step, which stalls the request, and the
# engine's startup check only compares it against the full budget -- so the
# shipped flat cap of 1023 at a budget of 1024 left a four-token window where an
# image could wedge the request. At 2048 the derived cap is 2043 (images up to
# 2042 tokens) and at 1024 it is 1019 (images up to 1018). Raise
# PORT_VISION_MAX_N_TOKEN explicitly to cap images lower. The checkpoint
# declares 1024 image tokens (vision_config.max_image_tokens), so a larger cap
# only widens the resolution grid the processor may pick, on the vision side.
#
# The budget no longer has to be held down for the text path. It used to be:
# measured here, 1024 -> 1152 changed the greedy token ids on 2 of 5 reference
# prompts and dropped DSpark acceptance from 5.69 to 5.06 tokens/step. That was
# the unpinned shape-selected kernels (see
# docs/design/vllm170-dsv41f-prefix-caching-2026-09-19.md) plus a chunked-prefill
# tail being scored by the decode kernels (see
# docs/design/vllm170-dsv41f-budget-2026-09-19.md). With marlin_full_k,
# dsv41_strict_bf16_gemm and dsv41_strict_prefill_tail on, 1024 -> 4096 leaves all
# five reference prompts byte-identical and the acceptance counters unchanged.
#
# The override must be flat (``vision_max_n_token``) rather than nested
# (``vision_config.max_image_tokens``): nested dict overrides are applied after
# the config derives that attribute from vision_config, so the nested form is
# silently a no-op.
PORT_DSPARK_SLOTS=0
if [[ -n "${PORT_DSPARK_TOKENS}" && "${PORT_DSPARK_TOKENS}" != "0" ]]; then
    PORT_DSPARK_SLOTS=$((PORT_DSPARK_TOKENS - 1))
fi
PORT_VISION_MAX_N_TOKEN=${PORT_VISION_MAX_N_TOKEN:-$((PORT_MAX_BATCHED - 1 - PORT_DSPARK_SLOTS))}
mm_args=()
if [[ "${PORT_MM_IMAGES}" != "0" ]]; then
    mm_args=(--hf-overrides "{\"vision_max_n_token\":${PORT_VISION_MAX_N_TOKEN}}")
    if [[ "${PORT_VISION_MAX_N_TOKEN}" -lt 2 ]]; then
        echo "PORT_VISION_MAX_N_TOKEN=${PORT_VISION_MAX_N_TOKEN} leaves no room" \
            "for an image; raise PORT_MAX_BATCHED." >&2
        exit 2
    fi
fi

# PORT_EXTRA_ARGS carries a flag list (the documented way to pass
# --additional-config here), so split it into the argv array used below.
read -r -a extra_args <<< "${PORT_EXTRA_ARGS}"

# Merge serving defaults with caller config and reject fast prefill in either
# --additional-config form.
merged_extra_args=$("${REPO_ROOT}/.venv/bin/python" - \
    "${PORT_CED_PREFILL}" "${PORT_FUSED_SWA_METADATA}" \
    "${PORT_MOE_HYBRID_PREFILL}" "${PORT_MOE_SMALL_TAIL_PREFILL}" \
    "${extra_args[@]}" <<'PY'
import json
import sys

ced_enabled = sys.argv[1] == "1"
fused_swa_metadata = sys.argv[2] == "1"
hybrid_prefill = sys.argv[3] == "1"
small_tail_prefill = sys.argv[4] == "1"
words = sys.argv[5:]
kept = []
config = {}
index = 0
while index < len(words):
    word = words[index]
    if word == "--additional-config":
        index += 1
        if index == len(words):
            raise SystemExit("--additional-config requires a JSON object")
        config = json.loads(words[index])
    elif word.startswith("--additional-config="):
        config = json.loads(word.split("=", 1)[1])
    else:
        kept.append(word)
    index += 1
if not isinstance(config, dict):
    raise SystemExit("--additional-config must be a JSON object")
if config.get("dsv41_fast_prefill"):
    raise SystemExit("dsv41_fast_prefill is disabled for this deployment")
config.pop("dsv41_fast_prefill", None)
config.setdefault("dsv41_sm80_fused_swa_metadata", fused_swa_metadata)
config.setdefault("dsv41_sm80_moe_hybrid_prefill", hybrid_prefill)
config.setdefault("dsv41_sm80_moe_small_tail_prefill", small_tail_prefill)
config.setdefault("dsv41_sm80_target_prepare_ahead", True)
config["ced_prefill"] = ced_enabled
kept += ["--additional-config", json.dumps(config, separators=(",", ":"))]
print(" ".join(kept))
PY
)
read -r -a extra_args <<< "${merged_extra_args}"

for arg in "$@"; do
    if [[ "${arg}" == "--additional-config" || "${arg}" == --additional-config=* ]]; then
        echo "Pass --additional-config through PORT_EXTRA_ARGS so prefill switches can be checked." >&2
        exit 2
    fi
done

exec "${VLLM_BIN}" serve "${PORT_MODEL_DIR}" \
    --served-model-name dsv41f-port \
    --host 0.0.0.0 \
    --port "${PORT_PORT}" \
    --pipeline-parallel-size 5 \
    --tensor-parallel-size 1 \
    --tokenizer-mode deepseek_v41 \
    --attention-backend TRITON_MLA_SPARSE_DSV41 \
    --moe-backend "${PORT_MOE_BACKEND}" \
    --engram-config "${engram_config}" \
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
    "${extra_args[@]}" \
    "$@"
