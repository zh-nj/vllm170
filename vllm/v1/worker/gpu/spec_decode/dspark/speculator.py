# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator: semi-autoregressive parallel drafting.

DSpark drafts a block of ``num_speculative_tokens`` tokens in one parallel pass
(reusing the DFlash machinery: context-KV precompute + a query-block forward),
then injects intra-block dependency with a lightweight sequential Markov head.

Differences from DFlash:
  * Anchor-as-first-prediction: each request emits exactly ``N =
    num_speculative_tokens`` query tokens (anchor + N-1 noise), NOT ``1 + N``.
    Every query position is a prediction (the anchor predicts the first draft
    token), so we sample at all N positions and ``sample_pos = query_pos + 1``
    (standard next-token), whereas DFlash's masks sit AT the predicted position.
    This is the ``sample_from_anchor`` path in the shared prepare-inputs kernel.
    Speculators-format checkpoints instead use the DFlash ``1 + N`` fill-in
    layout (anchor is the bonus token).
  * Sequential Markov sampling: instead of DFlash's single parallel sample, we
    sample left-to-right, adding a prefix-dependent Markov bias derived from the
    previously sampled token at each step.

CUDA graphs (FULL, mirroring DFlash) cover the whole draft step: the parallel
backbone forward AND the sequential Markov sampling.
"""

import json
import os
from typing import Any

import numpy as np
import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.models.deepseek_v4.phase_profile import phase_profile_stage
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

logger = init_logger(__name__)

_Q1_MARKOV_SCALE_AUDIT = (
    ("000", 0.0),
    ("050", 0.5),
    ("075", 0.75),
    ("100", 1.0),
    ("125", 1.25),
    ("150", 1.5),
)


def _get_q1_drafter_audit_eligible_rows(input_batch: InputBatch) -> np.ndarray:
    completes_prefill = (
        input_batch.num_computed_prefill_tokens_np
        + input_batch.num_scheduled_tokens
        >= input_batch.prefill_len_np
    )
    return ~input_batch.is_prefilling_np | completes_prefill


def _collect_q1_drafter_audit_records(
    base_logits: torch.Tensor,
    markov_bias: torch.Tensor,
    final_logits: torch.Tensor,
    draft_ids: torch.Tensor,
    previous_ids: torch.Tensor,
    positions: torch.Tensor,
    temperatures: torch.Tensor,
    eligible_rows: torch.Tensor,
    max_rows: int,
) -> list[dict[str, bool | float | int]]:
    """Collect bounded greedy q1 records before target verification."""
    if max_rows <= 0:
        return []
    valid_rows = torch.nonzero(
        (temperatures == 0) & eligible_rows, as_tuple=False
    ).flatten()[:max_rows]
    if valid_rows.numel() == 0:
        return []

    base_native = base_logits.index_select(0, valid_rows)
    bias_native = markov_bias.index_select(0, valid_rows)
    final_native = final_logits.index_select(0, valid_rows)
    scale_top_ids = [
        (base_native + bias_native * scale).argmax(dim=-1).to(torch.int64)
        for _, scale in _Q1_MARKOV_SCALE_AUDIT
    ]
    base = base_native.to(torch.float32)
    bias = bias_native.to(torch.float32)
    final = final_native.to(torch.float32)
    selected = draft_ids.index_select(0, valid_rows).to(torch.int64)

    base_top_values, base_top_ids = torch.topk(base, k=2, dim=-1)
    bias_top_values, bias_top_ids = torch.topk(bias, k=2, dim=-1)
    final_top_values, final_top_ids = torch.topk(final, k=2, dim=-1)

    selected_base = base.gather(1, selected[:, None]).squeeze(1)
    selected_bias = bias.gather(1, selected[:, None]).squeeze(1)
    selected_final = final.gather(1, selected[:, None]).squeeze(1)
    selected_base_rank = 1 + (base > selected_base[:, None]).sum(dim=1)
    selected_base_rank_end = (base >= selected_base[:, None]).sum(dim=1)

    base_top1_ids = base_top_ids[:, 0]
    base_top1_bias = bias.gather(1, base_top1_ids[:, None]).squeeze(1)
    base_top1_final = final.gather(1, base_top1_ids[:, None]).squeeze(1)
    base_top1_final_rank = 1 + (final > base_top1_final[:, None]).sum(dim=1)
    base_top1_final_rank_end = (final >= base_top1_final[:, None]).sum(dim=1)

    integer_rows = torch.stack(
        (
            valid_rows.to(torch.int64),
            positions.index_select(0, valid_rows).to(torch.int64),
            previous_ids.index_select(0, valid_rows).to(torch.int64),
            selected,
            base_top_ids[:, 0].to(torch.int64),
            base_top_ids[:, 1].to(torch.int64),
            bias_top_ids[:, 0].to(torch.int64),
            bias_top_ids[:, 1].to(torch.int64),
            final_top_ids[:, 0].to(torch.int64),
            final_top_ids[:, 1].to(torch.int64),
            selected_base_rank.to(torch.int64),
            selected_base_rank_end.to(torch.int64),
            base_top1_final_rank.to(torch.int64),
            base_top1_final_rank_end.to(torch.int64),
            *scale_top_ids,
        ),
        dim=1,
    ).cpu()
    float_rows = torch.stack(
        (
            temperatures.index_select(0, valid_rows).to(torch.float32),
            selected_base,
            selected_bias,
            selected_final,
            base_top_values[:, 0],
            base_top_values[:, 1],
            base_top_values[:, 0] - base_top_values[:, 1],
            bias_top_values[:, 0],
            bias_top_values[:, 1],
            final_top_values[:, 0],
            final_top_values[:, 1],
            final_top_values[:, 0] - final_top_values[:, 1],
            base_top1_bias,
            base_top1_final,
        ),
        dim=1,
    ).cpu()

    records: list[dict[str, bool | float | int]] = []
    for integer_row, float_row in zip(
        integer_rows.tolist(), float_rows.tolist(), strict=True
    ):
        (
            request_index,
            position,
            previous_token_id,
            draft_token_id,
            base_top1_id,
            base_top2_id,
            bias_top1_id,
            bias_top2_id,
            final_top1_id,
            final_top2_id,
            draft_base_rank,
            draft_base_rank_end,
            base_top1_final_rank,
            base_top1_final_rank_end,
            *markov_scale_top1_ids,
        ) = integer_row
        (
            temperature,
            draft_base_logit,
            draft_bias,
            draft_final_logit,
            base_top1_logit,
            base_top2_logit,
            base_top2_margin,
            bias_top1_logit,
            bias_top2_logit,
            final_top1_logit,
            final_top2_logit,
            final_top2_margin,
            base_top1_bias,
            base_top1_final_logit,
        ) = float_row
        record: dict[str, bool | float | int] = {
            "request_index": request_index,
            "position": position,
            "predicted_position": position + 1,
            "previous_token_id": previous_token_id,
            "draft_token_id": draft_token_id,
            "base_top1_id": base_top1_id,
            "base_top2_id": base_top2_id,
            "bias_top1_id": bias_top1_id,
            "bias_top2_id": bias_top2_id,
            "final_top1_id": final_top1_id,
            "final_top2_id": final_top2_id,
            "draft_base_rank": draft_base_rank,
            "draft_base_rank_end": draft_base_rank_end,
            "base_top1_final_rank": base_top1_final_rank,
            "base_top1_final_rank_end": base_top1_final_rank_end,
            "temperature": temperature,
            "draft_base_logit": draft_base_logit,
            "draft_bias": draft_bias,
            "draft_final_logit": draft_final_logit,
            "base_top1_logit": base_top1_logit,
            "base_top2_logit": base_top2_logit,
            "base_top2_margin": base_top2_margin,
            "bias_top1_logit": bias_top1_logit,
            "bias_top2_logit": bias_top2_logit,
            "final_top1_logit": final_top1_logit,
            "final_top2_logit": final_top2_logit,
            "final_top2_margin": final_top2_margin,
            "base_top1_bias": base_top1_bias,
            "base_top1_final_logit": base_top1_final_logit,
            "markov_changed_argmax": base_top1_id != final_top1_id,
            "sample_matches_final_top1": draft_token_id == final_top1_id,
            "sample_matches_final_max": draft_final_logit == final_top1_logit,
        }
        record.update(
            {
                f"markov_scale_{label}_top1_id": top1_id
                for (label, _), top1_id in zip(
                    _Q1_MARKOV_SCALE_AUDIT,
                    markov_scale_top1_ids,
                    strict=True,
                )
            }
        )
        records.append(record)
    return records


class DSparkSpeculator(DFlashSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # Whether to sample from the anchor position. When True, uses anchor-as-first
        # (N slots, each position predicts the next token). When False, uses 1+N
        # fill-in block (anchor is a bonus token).
        self.sample_from_anchor = getattr(
            self.draft_model_config.hf_config, "sample_from_anchor", True
        )
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_steps
        else:
            self.num_query_per_req = 1 + self.num_speculative_steps

        draft_hidden = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens, draft_hidden, dtype=self.dtype, device=device
        )
        self._target_hidden_states_workspace: torch.Tensor | None = None

        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )

        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )

        # Reduced-vocab probabilistic drafting only; set in load_draft_model.
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None
        self._q1_drafter_audit_samples = 0
        self._q1_drafter_audit_sample_limit = (
            max(0, envs.VLLM_DEEPSEEK_V4_DSPARK_Q1_AUDIT_SAMPLES)
            if envs.VLLM_DEEPSEEK_V4_DSPARK_Q1_AUDIT
            else 0
        )
        self._q1_drafter_decode_mask = (
            torch.empty(self.max_num_reqs, dtype=torch.bool, device=device)
            if self._q1_drafter_audit_sample_limit > 0
            else None
        )

    def set_target_hidden_states_workspace(self, workspace: torch.Tensor) -> bool:
        if (
            workspace.is_contiguous()
            and workspace.dtype == self.dtype
            and workspace.device == self.device
        ):
            self._target_hidden_states_workspace = workspace.view(-1)
            self.hidden_states = None
            return True
        return False

    def _prepare_context_hidden_states(
        self,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_target_tokens: int,
    ) -> torch.Tensor:
        if self._target_hidden_states_workspace is None:
            return super()._prepare_context_hidden_states(
                last_hidden_states, aux_hidden_states, num_target_tokens
            )
        if not aux_hidden_states:
            return last_hidden_states[:num_target_tokens]

        num_rows = aux_hidden_states[0].shape[0]
        packed_width = sum(tensor.shape[-1] for tensor in aux_hidden_states)
        packed_numel = num_rows * packed_width
        workspace = self._target_hidden_states_workspace
        can_reuse_workspace = (
            workspace is not None
            and workspace.numel() >= packed_numel
            and all(
                tensor.untyped_storage().data_ptr()
                != workspace.untyped_storage().data_ptr()
                for tensor in aux_hidden_states
            )
        )
        if can_reuse_workspace:
            assert workspace is not None
            packed_hidden_states = workspace[:packed_numel].view(num_rows, packed_width)
            torch.cat(aux_hidden_states, dim=-1, out=packed_hidden_states)
        else:
            packed_hidden_states = torch.cat(aux_hidden_states, dim=-1)

        hidden_states = self.model.combine_hidden_states(packed_hidden_states)
        return hidden_states[:num_target_tokens]

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        # Reduced draft vocab: probabilistic rejection sampling indexes draft
        # logits by target id, so precompute the draft->target column map and a
        # scratch buffer to scatter logits into target vocab before sampling.
        if self.draft_logits is not None and model.draft_id_to_target_id is not None:
            d2t = model.draft_id_to_target_id
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            # -inf once; the per-step scatter overwrites the draft->target
            # columns. Kept separate from draft_logits to avoid aliasing.
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )
        return model

    def _audit_q1_drafter(
        self,
        base_logits: torch.Tensor,
        markov_bias: torch.Tensor,
        final_logits: torch.Tensor,
        draft_ids: torch.Tensor,
        previous_ids: torch.Tensor,
        positions: torch.Tensor,
        temperatures: torch.Tensor,
        eligible_rows: torch.Tensor,
    ) -> None:
        remaining = (
            self._q1_drafter_audit_sample_limit - self._q1_drafter_audit_samples
        )
        if remaining <= 0 or self._d2t_scatter_index is not None:
            return
        control_path = envs.VLLM_DEEPSEEK_V4_PHASE_PROFILE_CONTROL_PATH
        if control_path and not os.path.exists(control_path):
            return
        if torch.cuda.is_current_stream_capturing():
            return
        records = _collect_q1_drafter_audit_records(
            base_logits,
            markov_bias,
            final_logits,
            draft_ids,
            previous_ids,
            positions,
            temperatures,
            eligible_rows,
            remaining,
        )
        for record in records:
            record["sample_index"] = self._q1_drafter_audit_samples
            logger.info(
                "DSpark q1 drafter audit: %s",
                json.dumps(record, sort_keys=True),
            )
            self._q1_drafter_audit_samples += 1

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        # Sequential Markov sampling over the backbone's output hidden states.
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        # Per-(req, position) head hidden, ordered (req, step).
        with phase_profile_stage("draft.hidden_gather", head_hidden, flow="dspark"):
            sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        # Draft-vocab logits; sampled ids are remapped to target vocab below.
        with phase_profile_stage("draft.base_logits", sample_hidden, flow="dspark"):
            base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)

        # Anchor (bonus) token per request = the input id at query offset 0,
        # read via the precomputed persistent index (fixed buffer for capture).
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            # Sequential stage: Markov bias from the previously sampled token.
            with phase_profile_stage(
                "draft.markov_bias", prev, flow="dspark", extra={"draft_step": i}
            ):
                markov_embed = self.model.markov_embed(prev)
                bias = self.model.markov_bias(markov_embed)
                logits_i = base_logits[:, i] + bias
            with phase_profile_stage(
                "draft.markov_sample",
                logits_i,
                flow="dspark",
                extra={"draft_step": i},
            ):
                if self.draft_logits is not None:
                    # Probabilistic sampling operates in the target vocabulary.
                    if self._d2t_scatter_index is not None:
                        assert self._draft_scatter_buf is not None
                        buf = self._draft_scatter_buf[:num_reqs]
                        buf.index_copy_(
                            1, self._d2t_scatter_index, logits_i.to(buf.dtype)
                        )
                        logits_i = buf
                    draft_sampled_i = gumbel_sample(
                        logits_i,
                        idx_map[:, i],
                        self.temperature,
                        self.seeds,
                        sample_pos[:, i] - 1,
                        apply_temperature=True,
                        output_processed_logits=self.draft_logits,
                        output_processed_logits_col=self._step_cols[i],
                        use_fp64=self.use_fp64_gumbel,
                    )
                else:
                    draft_sampled_i = self.model.map_draft_to_target(
                        logits_i.argmax(dim=-1)
                    )
            if i == 0 and self._q1_drafter_audit_sample_limit > 0:
                request_temperatures = self.temperature.index_select(
                    0, idx_map[:, i].to(torch.int64)
                )
                decode_mask = self._q1_drafter_decode_mask
                assert decode_mask is not None
                self._audit_q1_drafter(
                    base_logits[:, i],
                    bias,
                    logits_i,
                    draft_sampled_i,
                    prev,
                    sample_pos[:, i] - 1,
                    request_temperatures,
                    decode_mask[:num_reqs],
                )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

    def propose(
        self, input_batch: InputBatch, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        if self._q1_drafter_audit_sample_limit > 0:
            decode_mask = self._q1_drafter_decode_mask
            assert decode_mask is not None
            decode_mask[: input_batch.num_reqs].copy_(
                torch.from_numpy(_get_q1_drafter_audit_eligible_rows(input_batch))
            )
        return super().propose(input_batch, *args, **kwargs)

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        # Full draft step (captured under CUDA graph): parallel backbone forward
        # then sequential Markov sampling over its hidden state outputs.
        with phase_profile_stage(
            "draft.backbone", self.input_buffers.input_ids, flow="dspark", level="stage"
        ):
            head_hidden = self._run_model(
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
        with phase_profile_stage(
            "draft.markov_total", head_hidden, flow="dspark", level="stage"
        ):
            self._sample_sequential(num_reqs, head_hidden)
