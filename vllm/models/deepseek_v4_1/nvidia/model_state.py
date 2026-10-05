# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.distributed.utils import get_pp_indices
from vllm.logger import init_logger
from vllm.models.deepseek_v4_1.ced import CEDMetadata, CEDStep, ced_prefill_enabled
from vllm.models.deepseek_v4_1.fast_prefill import (
    FastPrefillMetadata,
    FastPrefillStep,
    fast_prefill_enabled,
    tail_layer,
)
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)

# Modalities whose prompt spans carry their own position grid; a bounded tail
# would have to keep every row of them.
_BOUNDED_UNSUPPORTED_MODALITIES = frozenset(("image", "video"))


@triton.jit
def _gather_lookback_kernel(
    lookback_ptr,
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    num_reqs,
    DEPTH: tl.constexpr,
    BLOCK_DEPTH: tl.constexpr,
):
    # One program per lookback row; rows past the batch are filled with -1.
    batch_idx = tl.program_id(0)
    in_batch = batch_idx < num_reqs
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx, mask=in_batch, other=0)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)

    offs = tl.arange(0, BLOCK_DEPTH)
    pos = num_computed - 1 - offs
    valid = in_batch & (offs < DEPTH) & (pos >= 0)
    ids = tl.load(
        all_token_ids_ptr + req_state_idx * all_token_ids_stride + pos,
        mask=valid,
        other=-1,
    )
    tl.store(lookback_ptr + batch_idx * DEPTH + offs, ids, mask=offs < DEPTH)


class DeepseekV41ModelState(DefaultModelState):
    """DefaultModelState plus the engram lookback window and CED prefill.

    The engram n-gram hash needs the ids of the ``depth`` tokens preceding
    each request's chunk start (see ``common/engram.py``). The runner keeps
    the full token history on device, so the window is gathered there every
    step: exact for prompt and generated tokens alike, whatever instance
    produced their KV.

    CED prefill (``ced.py``) keeps a bounded per-request suffix of encoder
    hidden states and replays the decoder layers on it once per request; the
    replay metadata is built here, before the forward, and handed to the
    model through ``model_inputs["ced_step"]``.

    Fast prefill (``fast_prefill.py``) bounds the same tail differently: the
    layers at or after ``tail_layer`` compute only the last
    ``min(chunk, sliding_window)`` rows of each prefill chunk, with the bounded
    attention metadata built here and handed to the model with
    ``model_inputs["fast_prefill_step"]``.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)
        self.fast_prefill_enabled = fast_prefill_enabled(vllm_config)
        # Keep the two prefill transforms independent.  Fast prefill needs an
        # eager model call, but it must not create or prepare the legacy CED
        # suffix/replay state (the two transforms are mutually exclusive).
        self.ced_prefill_enabled = ced_prefill_enabled(vllm_config)
        self.requires_eager_prefill = (
            self.ced_prefill_enabled
            or self.fast_prefill_enabled
            or bool(
                vllm_config.engram_config and vllm_config.engram_config.ssd_offload_path
            )
        )
        pp = get_pp_group()
        _, end_layer = get_pp_indices(
            vllm_config.model_config.hf_config.num_hidden_layers,
            pp.rank_in_group,
            pp.world_size,
        )
        self.ced_metadata: CEDMetadata | None = (
            CEDMetadata(vllm_config, device)
            if self.ced_prefill_enabled and end_layer > 20
            else None
        )
        self.fast_prefill_metadata: FastPrefillMetadata | None = (
            FastPrefillMetadata(
                vllm_config,
                device,
                tail_layer(vllm_config.model_config.hf_config),
            )
            if self.fast_prefill_enabled
            else None
        )
        self.ced_step: CEDStep | None = None
        self.fast_prefill_step: FastPrefillStep | None = None
        self.ced_disabled_requests: set[str] = set()
        self.fast_prefill_disabled_requests: set[str] = set()
        depth = model.token_lookback_depth
        self.lookback_token_ids: torch.Tensor | None = None
        if depth > 0:
            # Persistent so a captured graph can read it on replay.
            self.lookback_token_ids = torch.full(
                (self.max_num_reqs, depth), -1, dtype=torch.int32, device=device
            )

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if self.ced_prefill_enabled:
            model_inputs["ced_step"] = self.ced_step
        if self.fast_prefill_enabled:
            model_inputs["fast_prefill_step"] = self.fast_prefill_step
        window = self.lookback_token_ids
        if window is None:
            return model_inputs
        all_token_ids = req_states.all_token_ids.gpu
        depth = window.shape[1]
        _gather_lookback_kernel[(window.shape[0],)](
            window,
            input_batch.idx_mapping,
            req_states.num_computed_tokens.gpu,
            all_token_ids,
            all_token_ids.stride(0),
            input_batch.idx_mapping.shape[0],
            DEPTH=depth,
            BLOCK_DEPTH=triton.next_power_of_2(depth),
        )
        model_inputs["lookback_token_ids"] = window
        return model_inputs

    def add_request(self, req_index, new_req_data):
        ced_enabled = getattr(
            self, "ced_prefill_enabled", getattr(self, "ced_metadata", None) is not None
        )
        fast_enabled = getattr(self, "fast_prefill_enabled", False)
        params = new_req_data.sampling_params
        if params is not None and params.prompt_logprobs is not None:
            if ced_enabled:
                # Prompt logprobs (including 0) read every prompt row, so the
                # chunk cannot be bounded on any rank.
                self.ced_disabled_requests.add(new_req_data.req_id)
            if fast_enabled:
                self.fast_prefill_disabled_requests.add(new_req_data.req_id)
        if fast_enabled and any(
            feature.modality in _BOUNDED_UNSUPPORTED_MODALITIES
            for feature in getattr(new_req_data, "mm_features", None) or ()
        ):
            self.fast_prefill_disabled_requests.add(new_req_data.req_id)
        return super().add_request(req_index, new_req_data)

    def remove_request(self, req_id):
        self.ced_disabled_requests.discard(req_id)
        getattr(self, "fast_prefill_disabled_requests", set()).discard(req_id)
        ced_metadata = getattr(self, "ced_metadata", None)
        if ced_metadata is not None:
            ced_metadata.remove_request(req_id)
        model = getattr(self, "model", None)
        if model is not None:
            for module in model.modules():
                ced = getattr(module, "ced_prefill", None)
                if ced is not None:
                    ced.remove_request(req_id)
        return super().remove_request(req_id)

    def prepare_attn(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=False,
    ):
        metadata = super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture,
        )
        self.ced_step = None
        if (
            getattr(
                self,
                "ced_prefill_enabled",
                getattr(self, "ced_metadata", None) is not None,
            )
            and self.ced_metadata is not None
            and not all(
                req_id in self.ced_disabled_requests for req_id in input_batch.req_ids
            )
        ):
            self.ced_step = self.ced_metadata.prepare(
                input_batch,
                block_tables,
                slot_mappings,
                attn_groups,
                kv_cache_config,
                disabled_requests=self.ced_disabled_requests,
            )
        self.fast_prefill_step = None
        fast_prefill_metadata = getattr(self, "fast_prefill_metadata", None)
        if (
            fast_prefill_metadata is not None
            and not any(
                req_id in self.fast_prefill_disabled_requests
                for req_id in input_batch.req_ids
            )
            and getattr(input_batch, "num_draft_tokens", 0) == 0
        ):
            # A verification batch computes every candidate position, so it
            # keeps the full chunk.
            self.fast_prefill_step = fast_prefill_metadata.prepare(
                input_batch,
                block_tables,
                slot_mappings,
                attn_groups,
                kv_cache_config,
            )
        return metadata

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if self.lookback_token_ids is not None:
            # The captured graph reads this buffer; replays refill it in place.
            self.lookback_token_ids.fill_(-1)
            model_inputs["lookback_token_ids"] = self.lookback_token_ids
        return model_inputs

    def prepare_full_cudagraph_inputs(self, model_inputs, attn_metadata):
        prepared = self.model.prepare_engram_embeddings(
            model_inputs["input_ids"],
            model_inputs["positions"],
            model_inputs.get("lookback_token_ids"),
            attn_metadata,
        )
        model_inputs["engram_embeddings_prepared"] = prepared

    def prepare_pp_independent_inputs(self, model_inputs, attn_metadata):
        self.model.begin_engram_embeddings(
            model_inputs["input_ids"],
            model_inputs["positions"],
            model_inputs.get("lookback_token_ids"),
            attn_metadata,
        )
