# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-isolated CED prefill with bounded decoder SWA replay."""

from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.kernels.mhc.tilelang import (
    mhc_post_tilelang,
    mhc_pre_delayed_tilelang,
)
from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.utils import AttentionGroup

# CED replay rows can come from earlier chunks and therefore be wider than the
# current scheduler batch.  They travel as extra tensors in the ordinary PP
# message; the regular hidden/pre_mix tensors retain the current batch shape.
CED_REPLAY_HIDDEN = "dsv41_ced_replay_hidden"
CED_REPLAY_PRE_MIX = "dsv41_ced_replay_pre_mix"
CED_REPLAY_TOPK = "dsv41_ced_replay_topk"
CED_REPLAY_CANDIDATES = "dsv41_ced_replay_candidates"


def ced_prefill_enabled(config: VllmConfig) -> bool:
    extra = config.additional_config
    enabled = isinstance(extra, dict) and bool(extra.get("ced_prefill", False))
    if not enabled:
        return False
    if not config.use_v2_model_runner:
        raise ValueError("CED prefill requires the v2 model runner")
    parallel = config.parallel_config
    if parallel.tensor_parallel_size != 1 or parallel.use_ubatching:
        raise ValueError("CED prefill currently requires TP1 without microbatches")
    if (
        config.speculative_config is not None
        and config.speculative_config.method != "dspark"
    ):
        raise ValueError("CED prefill currently supports only DSpark speculation")
    return True


class SuffixBuffer:
    """Keep a suffix across arbitrary chunk boundaries without retaining the chunk."""

    def __init__(self, width: int):
        self.width = width
        self.request_id: str | None = None
        self.next_position = 0
        self.tensors: dict[str, torch.Tensor] = {}

    def append(self, request_id: str, start: int, **tensors) -> dict[str, torch.Tensor]:
        if self.request_id is None or start == 0:
            self.request_id = request_id
            self.next_position = start
            self.tensors = {}
        if request_id != self.request_id or start != self.next_position:
            raise ValueError(
                "CED prefill chunks must be contiguous after the prefix hit"
            )
        lengths = {value.shape[0] for value in tensors.values()}
        if len(lengths) != 1:
            raise ValueError("CED suffix tensors have different token counts")
        count = lengths.pop()
        for name, value in tensors.items():
            if count < self.width and name in self.tensors:
                value = torch.cat((self.tensors[name], value))
            self.tensors[name] = value[-self.width :].clone()
        self.next_position += count
        return self.tensors


@dataclass
class CEDRequest:
    request_id: str
    start: int
    offset: int
    num_tokens: int
    is_prefilling: bool
    use_ced: bool
    final: bool
    replay_count: int = 0


@dataclass
class CEDStep:
    request_id: str
    start: int
    num_tokens: int
    final: bool
    replay_metadata: dict[str, Any] | None = None
    positions: torch.Tensor | None = None
    input_ids: torch.Tensor | None = None
    replay_slots: torch.Tensor | None = None
    requests: list[CEDRequest] | None = None
    source_positions: torch.Tensor | None = None


@dataclass
class CEDDraftContext:
    request_id: str
    positions: torch.Tensor
    slots: torch.Tensor
    auxiliary: list[torch.Tensor]
    request_ids: tuple[str, ...] = ()
    request_lengths: tuple[int, ...] = ()


class CEDMetadata:
    def __init__(self, config: VllmConfig, device: torch.device):
        self.config = config
        self.device = device
        self.width = config.model_config.hf_config.sliding_window
        self.suffixes: dict[str, SuffixBuffer] = {}
        self.groups: list[list[AttentionGroup]] | None = None

    def remove_request(self, request_id):
        self.suffixes.pop(request_id, None)

    def prepare(
        self,
        batch,
        block_tables,
        slot_mappings,
        groups,
        kv_config,
        disabled_requests=frozenset(),
    ):
        if not batch.has_prefill:
            return None
        requests = []
        selected = []
        suffixes = []
        starts = getattr(batch, "query_start_loc_np", [0, batch.num_tokens])
        for index, request_id in enumerate(batch.req_ids):
            offset, end = int(starts[index]), int(starts[index + 1])
            start = int(batch.num_computed_prefill_tokens_np[index])
            prefilling = bool(batch.is_prefilling_np[index])
            use_ced = prefilling and request_id not in disabled_requests
            final = start + end - offset == int(batch.prefill_len_np[index])
            request = CEDRequest(
                request_id, start, offset, end - offset, prefilling, use_ced, final
            )
            values = dict(
                slots=slot_mappings[:, offset:end].T,
                positions=batch.positions[offset:end],
                input_ids=batch.input_ids[offset:end],
            )
            if use_ced:
                buffer = self.suffixes.setdefault(request_id, SuffixBuffer(self.width))
                values = buffer.append(request_id, start, **values)
                if final and values["positions"].shape[0] < min(
                    self.width, int(batch.prefill_len_np[index])
                ):
                    raise ValueError(
                        "CED prefix hits must leave the full sliding-window suffix "
                        "uncached for decoder replay"
                    )
            if not use_ced or final:
                request.replay_count = values["positions"].shape[0]
                selected.append(index)
                suffixes.append(values)
                self.suffixes.pop(request_id, None)
            requests.append(request)
        first = requests[0]
        step = CEDStep(
            first.request_id,
            first.start,
            batch.num_tokens,
            bool(selected),
            requests=requests,
            source_positions=batch.positions,
        )
        if not selected:
            return step
        if self.groups is None:
            self.groups = []
            for collection in groups:
                cloned = []
                for group in collection:
                    # Consumer layers also read source-layer cache metadata.
                    names = list(group.layer_names)
                    if not names:
                        continue
                    new = AttentionGroup(
                        group.backend,
                        names,
                        group.kv_cache_spec,
                        group.kv_cache_group_id,
                    )
                    new.create_metadata_builders(
                        self.config,
                        self.device,
                        kernel_block_size=group.get_metadata_builder().kernel_block_size,
                    )
                    cloned.append(new)
                self.groups.append(cloned)
        suffix = {
            name: torch.cat([values[name] for values in suffixes])
            for name in ("positions", "input_ids", "slots")
        }
        active = [requests[index] for index in selected]
        lengths = [request.replay_count for request in active]
        query_cpu = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0).int()
        replay_count = sum(lengths)
        scheduler = getattr(self.config, "scheduler_config", None)
        if scheduler is not None and replay_count > scheduler.max_num_batched_tokens:
            raise ValueError(
                f"CED replay needs {replay_count} rows but the worker capacity is "
                f"{scheduler.max_num_batched_tokens}; reduce max_num_seqs or "
                "serve prompt-logprob requests separately"
            )
        indices = torch.tensor(selected, dtype=torch.long, device=self.device)
        seq_lens = batch.seq_lens.index_select(0, indices)
        seq_cpu = batch.seq_lens_cpu_upper_bound[selected]
        step.replay_metadata = build_attn_metadata(
            attn_groups=self.groups,
            num_reqs=len(selected),
            num_tokens=replay_count,
            query_start_loc_gpu=query_cpu.to(self.device),
            query_start_loc_cpu=query_cpu,
            max_query_len=max(lengths),
            seq_lens=seq_lens,
            max_seq_len=int(seq_cpu.max()),
            block_tables=[table.index_select(0, indices) for table in block_tables],
            slot_mappings=suffix["slots"].T.contiguous(),
            kv_cache_config=kv_config,
            seq_lens_cpu_upper_bound=seq_cpu,
            positions=suffix["positions"],
            is_prefilling=torch.tensor([r.is_prefilling for r in active]),
        )
        for metadata in step.replay_metadata.values():
            if (
                isinstance(metadata, DeepseekSparseSWAMetadata)
                and metadata.prefill_gather_lens is not None
            ):
                bounds = [
                    r.replay_count if r.use_ced else int(seq_cpu[i])
                    for i, r in enumerate(active)
                    if r.is_prefilling
                ]
                metadata.prefill_gather_lens = torch.minimum(
                    metadata.prefill_gather_lens,
                    metadata.prefill_gather_lens.new_tensor(bounds),
                )
        step.positions = suffix["positions"]
        step.input_ids = suffix["input_ids"]
        step.replay_slots = suffix["slots"].T.contiguous()
        return step


def publish_decoder_kv(attn, normalized, positions):
    compressor = attn.compressor
    kv_score = torch.mm(
        normalized, compressor.fused_wkv_wgate.weight.T, out_dtype=torch.float32
    )
    latent = compressor(kv_score, positions)
    compressor.insert_cache(latent, positions, attn.rotary_emb)
    attn.indexer._produce_k(latent, positions, attn.indexer_rotary_emb)
    attn._last_compressed_latent = latent
    return latent


class CEDPrefill:
    def __init__(self, width: int):
        self.width = width
        self.suffixes: dict[str, SuffixBuffer] = {}
        self.encoder_tokens = 0
        self.decoder_tokens = 0
        self.replay_calls = 0
        self.draft_context: CEDDraftContext | None = None

    def remove_request(self, request_id):
        self.suffixes.pop(request_id, None)

    def take_draft_context(self):
        context, self.draft_context = self.draft_context, None
        return context

    def forward(
        self,
        model,
        step,
        hidden,
        residual,
        post_mix,
        res_mix,
        pre_mix,
        intermediate_tensors=None,
    ):
        """Run the CED source/replay on this PP rank's local decoder layers."""
        physical_tokens = hidden.shape[0]
        physical_shape = (physical_tokens, pre_mix.shape[-1], model.config.hidden_size)
        self.draft_context = None
        model_start = getattr(model, "start_layer", 0)
        model_end = getattr(model, "end_layer", 40)
        num_layers = getattr(model.config, "num_hidden_layers", 40)
        start = max(20, model_start)
        end = model_end
        source_owner = model_start <= 20 < model_end
        context = get_forward_context()
        full_latent = full_index_k = None

        if source_owner:
            encoder_hidden = (
                mhc_post_tilelang(hidden, residual, post_mix, res_mix)
                if residual is not None
                else hidden
            )
            layer20 = model.layers[20]
            from vllm.models.deepseek_v4_1.ampere.ampere_sparse import (
                DeepseekV41AmpereMLAAttention,
            )

            if not isinstance(layer20.attn, DeepseekV41AmpereMLAAttention):
                raise ValueError(
                    "CED replay currently requires the Ampere attention backend"
                )
            _, _, normalized, _ = mhc_pre_delayed_tilelang(
                encoder_hidden,
                layer20.hc_attn_fn,
                layer20.hc_attn_scale,
                layer20.hc_attn_base,
                layer20.rms_norm_eps,
                layer20.hc_eps,
                layer20.hc_eps,
                layer20.hc_post_alpha,
                layer20.hc_sinkhorn_iters,
                pre_mix=pre_mix,
                norm_weight=layer20.attn_norm.weight,
                norm_eps=layer20.attn_norm.variance_epsilon,
            )
            publish_decoder_kv(layer20.attn, normalized, step.source_positions)
            self.encoder_tokens += step.num_tokens
            if end < num_layers:
                latent = layer20.attn._last_compressed_latent
                assert latent is not None
                full_latent = latent.clone()
                if model._pp_outgoing_index:
                    assert layer20.attn.indexer is not None
                    normalized_k = layer20.attn.indexer._last_normalized_k
                    assert normalized_k is not None
                    full_index_k = normalized_k.clone()

            saved_hidden = []
            saved_mix = []
            assert step.requests is not None
            for request in step.requests:
                begin = request.offset
                request_end = begin + request.num_tokens
                values = dict(
                    hidden=encoder_hidden[begin:request_end],
                    pre_mix=pre_mix[begin:request_end],
                )
                if request.use_ced:
                    suffix = self.suffixes.setdefault(
                        request.request_id, SuffixBuffer(self.width)
                    )
                    values = suffix.append(request.request_id, request.start, **values)
                if request.replay_count:
                    saved_hidden.append(values["hidden"])
                    saved_mix.append(values["pre_mix"])
                    self.suffixes.pop(request.request_id, None)
            replay_hidden = torch.cat(saved_hidden) if step.final else None
            replay_mix = torch.cat(saved_mix) if step.final else None
        elif step.final:
            assert intermediate_tensors is not None
            replay_hidden = intermediate_tensors[CED_REPLAY_HIDDEN]
            replay_mix = intermediate_tensors[CED_REPLAY_PRE_MIX]
            if model._pp_incoming_topk:
                model.topk_indices_buffer[: replay_hidden.shape[0]].copy_(
                    intermediate_tensors[CED_REPLAY_TOPK]
                )
            if model._pp_incoming_candidates:
                model.candidate_block_buffer[: replay_hidden.shape[0]].copy_(
                    intermediate_tensors[CED_REPLAY_CANDIDATES]
                )
        else:
            replay_hidden = replay_mix = None

        auxiliary = []
        if step.final:
            assert replay_hidden is not None and replay_mix is not None
            original_metadata = context.attn_metadata
            original_slots = context.slot_mapping
            original_padding = context.is_padding
            try:
                context.attn_metadata = step.replay_metadata
                context.slot_mapping = {
                    name: metadata.slot_mapping
                    for name, metadata in step.replay_metadata.items()
                    if hasattr(metadata, "slot_mapping")
                }
                context.is_padding = None
                hidden, pre_mix = replay_hidden, replay_mix
                residual = post_mix = res_mix = None
                from vllm.distributed import get_pp_group
                from vllm.models.deepseek_v4_1.fast_prefill import record_layer_rows

                for index in range(start, end):
                    record_layer_rows(
                        get_pp_group().rank_in_group, index, hidden.shape[0], 20
                    )
                    layer = model.layers[index]
                    hidden, residual, post_mix, res_mix, pre_mix = layer(
                        hidden,
                        step.positions,
                        step.input_ids,
                        pre_mix,
                        post_mix,
                        res_mix,
                        residual,
                    )
                    if index + 1 in model.aux_hidden_state_layers:
                        auxiliary.append(
                            mhc_post_tilelang(hidden, residual, post_mix, res_mix).mean(
                                dim=1
                            )
                        )
                hidden = mhc_post_tilelang(hidden, residual, post_mix, res_mix)
                self.decoder_tokens += hidden.shape[0]
                self.replay_calls += 1
            finally:
                context.attn_metadata = original_metadata
                context.slot_mapping = original_slots
                context.is_padding = original_padding
                if source_owner and end < num_layers:
                    model.layers[20].attn._last_compressed_latent = full_latent
                    if model._pp_outgoing_index:
                        model.layers[20].attn.indexer._last_normalized_k = full_index_k

        if end < num_layers:
            tensors = {
                "hidden_states": hidden.new_zeros(physical_shape),
                "pre_mix": pre_mix.new_zeros((physical_tokens, physical_shape[1])),
            }
            if step.final:
                tensors[CED_REPLAY_HIDDEN] = hidden
                tensors[CED_REPLAY_PRE_MIX] = pre_mix
                if model._pp_outgoing_topk:
                    tensors[CED_REPLAY_TOPK] = model.topk_indices_buffer[
                        : hidden.shape[0]
                    ].clone()
                if model._pp_outgoing_candidates:
                    tensors[CED_REPLAY_CANDIDATES] = model.candidate_block_buffer[
                        : hidden.shape[0]
                    ].clone()
            model._append_pp_attention_state(
                tensors, step.source_positions, intermediate_tensors
            )
            return IntermediateTensors(tensors)

        output = torch.zeros(
            (physical_tokens, model.config.hidden_size),
            dtype=hidden.dtype,
            device=hidden.device,
        )
        auxiliary_output = [
            torch.zeros_like(output) for _ in model.aux_hidden_state_layers
        ]
        if model._mtp_hidden_buffer is not None:
            model._mtp_hidden_buffer[:physical_tokens].zero_()
        if not step.final:
            return (output, auxiliary_output) if auxiliary_output else output
        collapsed = model.norm(hc_collapse_triton(hidden, pre_mix))
        replay_offset = 0
        draft_slices = []
        draft_requests = []
        assert step.requests is not None
        for request in step.requests:
            if not request.replay_count:
                continue
            replay_end = replay_offset + request.replay_count
            returned = min(request.num_tokens, request.replay_count)
            dest_end = request.offset + request.num_tokens
            destination = slice(dest_end - returned, dest_end)
            source = slice(replay_end - returned, replay_end)
            output[destination].copy_(collapsed[source])
            if model._mtp_hidden_buffer is not None:
                model._mtp_hidden_buffer[destination].copy_(hidden[source].flatten(1))
            for padded, value in zip(auxiliary_output, auxiliary):
                padded[destination].copy_(value[source])
            if request.use_ced and request.final:
                draft_slices.append(slice(replay_offset, replay_end))
                draft_requests.append(request)
            replay_offset = replay_end
        if auxiliary_output:
            if len(auxiliary) != len(auxiliary_output):
                raise ValueError("CED auxiliary states must come from decoder layers")
            if draft_requests:
                self.draft_context = CEDDraftContext(
                    draft_requests[0].request_id,
                    torch.cat([step.positions[s] for s in draft_slices]),
                    torch.cat([step.replay_slots[:, s] for s in draft_slices], dim=1),
                    [
                        torch.cat([value[s] for s in draft_slices])
                        for value in auxiliary
                    ],
                    tuple(r.request_id for r in draft_requests),
                    tuple(r.replay_count for r in draft_requests),
                )
            return output, auxiliary_output
        return output
