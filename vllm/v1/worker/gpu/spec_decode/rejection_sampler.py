# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
from collections.abc import Iterable, Iterator

import numpy as np
import torch

from vllm import envs
from vllm.config import SpeculativeConfig
from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.logger import init_logger
from vllm.profiler.phase_profile_hook import optional_phase_profile_stage
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    get_num_sampled_and_rejected,
)
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)

# Cap on the FP32 target-logits buffer materialized by apply_sampling_params.
# TODO(mgoin): Chunking is a workaround. The rejection kernels already upcast
# per vocab block on load and apply ops like temperature and gumbel, so folding
# sampling-param application into those kernels would remove this buffer and
# its traffic entirely.
MAX_CHUNK_BYTES = 2**30  # 1GB
_FP32_BYTES = 4

logger = init_logger(__name__)

_PROB_AUDIT_DRAFT_TEMPERATURE_RATIOS = (0.7, 0.8, 0.9, 1.1, 1.2, 1.3)
_PROB_AUDIT_DRAFT_TOP_K = (1, 2, 4, 8, 16, 32, 64)
_PROB_AUDIT_DRAFT_TOP_P = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)


def get_max_chunk_logits(vocab_size: int) -> int:
    """Largest number of logits rows one verification chunk may hold."""
    return max(1, MAX_CHUNK_BYTES // (vocab_size * _FP32_BYTES))


def _iter_request_chunks(
    cu_num_logits: np.ndarray, max_chunk_logits: int
) -> Iterator[tuple[int, int]]:
    """Yield maximally packed request ranges without splitting requests."""
    assert max_chunk_logits > 0
    num_reqs = cu_num_logits.size - 1
    start = 0
    while start < num_reqs:
        max_logit = int(cu_num_logits[start]) + max_chunk_logits
        end = int(np.searchsorted(cu_num_logits, max_logit, side="right") - 1)
        end = min(num_reqs, max(start + 1, end))
        yield start, end
        start = end


def _collect_q1_acceptance_audit_records(
    processed_logits: torch.Tensor,
    draft_sampled: torch.Tensor,
    sampled: torch.Tensor,
    num_sampled: torch.Tensor,
    pos: torch.Tensor,
    cu_num_logits: torch.Tensor,
    request_temperatures: torch.Tensor,
    max_rows: int,
) -> list[dict[str, bool | float | int]]:
    """Collect a bounded greedy q1 audit from rejection-sampler tensors."""
    if max_rows <= 0:
        return []
    num_draft_tokens = torch.diff(cu_num_logits).to(torch.int64) - 1
    valid_requests = torch.nonzero(
        (num_draft_tokens > 0) & (request_temperatures == 0),
        as_tuple=False,
    ).flatten()[:max_rows]
    if valid_requests.numel() == 0:
        return []

    starts = cu_num_logits[:-1].to(torch.int64).index_select(0, valid_requests)
    q1_logits = processed_logits.index_select(0, starts).to(torch.float32)
    top_values, top_ids = torch.topk(q1_logits, k=2, dim=-1)
    draft_ids = draft_sampled.index_select(0, starts + 1).to(torch.int64)
    sampler_target_ids = sampled.index_select(0, valid_requests)[:, 0].to(torch.int64)
    draft_logits = q1_logits.gather(1, draft_ids[:, None]).squeeze(1)
    draft_strict_ranks = 1 + (q1_logits > draft_logits[:, None]).sum(dim=1)
    draft_rank_ends = (q1_logits >= draft_logits[:, None]).sum(dim=1)
    sampler_target_logits = q1_logits.gather(1, sampler_target_ids[:, None]).squeeze(1)
    accepted_lengths = num_sampled.index_select(0, valid_requests).to(torch.int64) - 1

    integer_rows = torch.stack(
        (
            valid_requests.to(torch.int64),
            pos.index_select(0, starts).to(torch.int64),
            num_draft_tokens.index_select(0, valid_requests),
            accepted_lengths,
            draft_ids,
            sampler_target_ids,
            top_ids[:, 0].to(torch.int64),
            top_ids[:, 1].to(torch.int64),
            draft_strict_ranks.to(torch.int64),
            draft_rank_ends.to(torch.int64),
        ),
        dim=1,
    ).cpu()
    float_rows = torch.stack(
        (
            request_temperatures.index_select(0, valid_requests).to(torch.float32),
            sampler_target_logits,
            draft_logits,
            top_values[:, 0],
            top_values[:, 1],
            top_values[:, 0] - top_values[:, 1],
            sampler_target_logits - draft_logits,
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
            draft_count,
            accepted_length,
            draft_token_id,
            sampler_target_id,
            torch_top1_id,
            torch_top2_id,
            draft_strict_rank,
            draft_rank_end,
        ) = integer_row
        (
            temperature,
            sampler_target_logit,
            draft_logit,
            torch_top1_logit,
            torch_top2_logit,
            top2_margin,
            draft_gap,
        ) = float_row
        records.append(
            {
                "request_index": request_index,
                "position": position,
                "num_draft_tokens": draft_count,
                "accepted_length": accepted_length,
                "draft_token_id": draft_token_id,
                "sampler_target_id": sampler_target_id,
                "torch_top1_id": torch_top1_id,
                "torch_top2_id": torch_top2_id,
                "draft_strict_rank": draft_strict_rank,
                "draft_rank_end": draft_rank_end,
                "temperature": temperature,
                "sampler_target_logit": sampler_target_logit,
                "draft_logit": draft_logit,
                "torch_top1_logit": torch_top1_logit,
                "torch_top2_logit": torch_top2_logit,
                "top2_margin": top2_margin,
                "draft_gap": draft_gap,
                "accepted": draft_token_id == sampler_target_id,
                "sampler_argmax_matches_torch": (sampler_target_id == torch_top1_id),
            }
        )
    return records


def _collect_probabilistic_acceptance_audit_records(
    processed_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    draft_sampled: torch.Tensor,
    num_sampled: torch.Tensor,
    pos: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    max_rows: int,
    draft_confidence: torch.Tensor | None = None,
) -> list[dict[str, bool | float | int]]:
    """Collect theoretical overlap and observed probabilistic acceptance."""
    if max_rows <= 0:
        return []

    cu_num_logits_cpu = cu_num_logits.to(torch.int64).cpu().tolist()
    idx_mapping_cpu = idx_mapping.to(torch.int64).cpu().tolist()
    row_indices: list[int] = []
    request_indices: list[int] = []
    request_state_indices: list[int] = []
    draft_steps: list[int] = []
    for request_index, request_state_index in enumerate(idx_mapping_cpu):
        start = cu_num_logits_cpu[request_index]
        num_draft_tokens = cu_num_logits_cpu[request_index + 1] - start - 1
        for draft_step in range(num_draft_tokens):
            row_indices.append(start + draft_step)
            request_indices.append(request_index)
            request_state_indices.append(request_state_index)
            draft_steps.append(draft_step)
            if len(row_indices) >= max_rows:
                break
        if len(row_indices) >= max_rows:
            break
    if not row_indices:
        return []

    device = processed_logits.device
    row_tensor = torch.tensor(row_indices, dtype=torch.int64, device=device)
    request_tensor = torch.tensor(request_indices, dtype=torch.int64, device=device)
    state_tensor = torch.tensor(request_state_indices, dtype=torch.int64, device=device)
    step_tensor = torch.tensor(draft_steps, dtype=torch.int64, device=device)
    target = processed_logits.index_select(0, row_tensor).to(torch.float32)
    draft = draft_logits[state_tensor, step_tensor].to(torch.float32)
    target_probs = target.softmax(dim=-1)
    draft_probs = draft.softmax(dim=-1)
    target_support = torch.isfinite(target)
    overlap = torch.minimum(target_probs, draft_probs).sum(dim=-1)
    draft_mass_outside_target = draft_probs.masked_fill(target_support, 0.0).sum(dim=-1)
    counterfactual_overlaps = {
        ratio: torch.minimum(
            target_probs,
            (draft / ratio).softmax(dim=-1),
        ).sum(dim=-1)
        for ratio in _PROB_AUDIT_DRAFT_TEMPERATURE_RATIOS
    }

    draft_sorted, draft_sort_indices = draft.sort(dim=-1, descending=False)
    target_sorted_probs = target_probs.gather(1, draft_sort_indices)
    target_sorted_support = target_support.gather(1, draft_sort_indices)
    draft_sorted_cumulative = draft_sorted.softmax(dim=-1).cumsum(dim=-1)
    top_k_overlaps: dict[int, torch.Tensor] = {}
    for top_k in _PROB_AUDIT_DRAFT_TOP_K:
        keep = min(top_k, draft.shape[-1])
        top_k_logits = draft_sorted.clone()
        top_k_logits[:, :-keep] = float("-inf")
        top_k_probs = top_k_logits.softmax(dim=-1)
        top_k_overlaps[top_k] = torch.minimum(target_sorted_probs, top_k_probs).sum(
            dim=-1
        )

    top_p_overlaps: dict[float, torch.Tensor] = {}
    draft_top_p_mass_outside_target: torch.Tensor | None = None
    for top_p in _PROB_AUDIT_DRAFT_TOP_P:
        top_p_mask = draft_sorted_cumulative <= 1.0 - top_p
        top_p_mask[:, -1] = False
        top_p_logits = draft_sorted.masked_fill(top_p_mask, float("-inf"))
        top_p_probs = top_p_logits.softmax(dim=-1)
        top_p_overlaps[top_p] = torch.minimum(target_sorted_probs, top_p_probs).sum(
            dim=-1
        )
        if top_p == 0.95:
            draft_top_p_mass_outside_target = top_p_probs.masked_fill(
                target_sorted_support, 0.0
            ).sum(dim=-1)
    assert draft_top_p_mass_outside_target is not None

    target_support_draft_probs = draft_probs.masked_fill(~target_support, 0.0)
    target_support_draft_probs /= target_support_draft_probs.sum(
        dim=-1, keepdim=True
    ).clamp_min(torch.finfo(target_support_draft_probs.dtype).tiny)
    target_support_overlap = torch.minimum(
        target_probs, target_support_draft_probs
    ).sum(dim=-1)
    draft_ids = draft_sampled.index_select(0, row_tensor + 1).to(torch.int64)
    target_selected = target_probs.gather(1, draft_ids[:, None]).squeeze(1)
    draft_selected = draft_probs.gather(1, draft_ids[:, None]).squeeze(1)
    sampled_accept_probability = torch.minimum(
        torch.ones_like(target_selected), target_selected / draft_selected
    )
    accepted_lengths = num_sampled.index_select(0, request_tensor).to(torch.int64) - 1

    integer_rows = torch.stack(
        (
            request_tensor,
            state_tensor,
            step_tensor,
            pos.index_select(0, row_tensor).to(torch.int64),
            accepted_lengths,
            draft_ids,
            target.argmax(dim=-1).to(torch.int64),
            draft.argmax(dim=-1).to(torch.int64),
        ),
        dim=1,
    ).cpu()
    float_rows = torch.stack(
        (
            overlap,
            sampled_accept_probability,
            target_selected,
            draft_selected,
            draft_mass_outside_target,
            *(top_k_overlaps[top_k] for top_k in _PROB_AUDIT_DRAFT_TOP_K),
            *(top_p_overlaps[top_p] for top_p in _PROB_AUDIT_DRAFT_TOP_P),
            draft_top_p_mass_outside_target,
            target_support_overlap,
            *(
                counterfactual_overlaps[ratio]
                for ratio in _PROB_AUDIT_DRAFT_TEMPERATURE_RATIOS
            ),
        ),
        dim=1,
    ).cpu()
    confidence_rows = (
        draft_confidence[state_tensor, step_tensor].to(torch.float32).cpu().tolist()
        if draft_confidence is not None
        else [None] * len(row_indices)
    )

    records: list[dict[str, bool | float | int]] = []
    for integer_row, float_row, confidence_logit in zip(
        integer_rows.tolist(),
        float_rows.tolist(),
        confidence_rows,
        strict=True,
    ):
        (
            request_index,
            request_state_index,
            draft_step,
            position,
            accepted_length,
            draft_token_id,
            target_top1_id,
            draft_top1_id,
        ) = integer_row
        (
            expected_overlap,
            sampled_probability,
            target_probability,
            draft_probability,
            outside_mass,
        ) = float_row[:5]
        num_top_k = len(_PROB_AUDIT_DRAFT_TOP_K)
        num_top_p = len(_PROB_AUDIT_DRAFT_TOP_P)
        filtered_start = 5
        top_k_overlap_values = float_row[filtered_start : filtered_start + num_top_k]
        top_p_start = filtered_start + num_top_k
        top_p_overlap_values = float_row[top_p_start : top_p_start + num_top_p]
        draft_top_p_outside_mass = float_row[top_p_start + num_top_p]
        target_support_expected_overlap = float_row[top_p_start + num_top_p + 1]
        temperature_ratio_overlaps = float_row[top_p_start + num_top_p + 2 :]
        record: dict[str, bool | float | int] = {
            "request_index": request_index,
            "request_state_index": request_state_index,
            "draft_step": draft_step,
            "position": position,
            "accepted_length": accepted_length,
            "draft_token_id": draft_token_id,
            "target_top1_id": target_top1_id,
            "draft_top1_id": draft_top1_id,
            "reached": draft_step <= accepted_length,
            "accepted": draft_step < accepted_length,
            "top1_match": target_top1_id == draft_top1_id,
            "draft_in_target_support": target_probability > 0.0,
            "expected_overlap": expected_overlap,
            "sampled_accept_probability": sampled_probability,
            "target_probability": target_probability,
            "draft_probability": draft_probability,
            "draft_mass_outside_target_support": outside_mass,
            "draft_top_p_095_mass_outside_target_support": (draft_top_p_outside_mass),
            "target_support_oracle_expected_overlap": (target_support_expected_overlap),
        }
        for top_k, top_k_overlap in zip(
            _PROB_AUDIT_DRAFT_TOP_K, top_k_overlap_values, strict=True
        ):
            record[f"draft_top_k_{top_k}_expected_overlap"] = top_k_overlap
        for top_p, top_p_overlap in zip(
            _PROB_AUDIT_DRAFT_TOP_P, top_p_overlap_values, strict=True
        ):
            top_p_label = f"{top_p:.2f}".replace(".", "")
            record[f"draft_top_p_{top_p_label}_expected_overlap"] = top_p_overlap
        for ratio, ratio_overlap in zip(
            _PROB_AUDIT_DRAFT_TEMPERATURE_RATIOS,
            temperature_ratio_overlaps,
            strict=True,
        ):
            ratio_label = f"{ratio:.1f}".replace(".", "")
            record[f"draft_temperature_ratio_{ratio_label}_expected_overlap"] = (
                ratio_overlap
            )
        if confidence_logit is not None:
            record["confidence_logit"] = confidence_logit
            record["confidence_probability"] = torch.sigmoid(
                torch.tensor(confidence_logit)
            ).item()
        records.append(record)
    return records


@triton.jit
def _flatten_sampled_kernel(
    # [num_logits]
    flat_sampled_ptr,
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    for i in range(num_sampled):
        token_id = tl.load(sampled_ptr + req_idx * sampled_stride + i)
        tl.store(flat_sampled_ptr + start_idx + i, token_id)


class RejectionSampler:
    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig,
        device: torch.device,
    ):
        self.sampler = sampler
        self.device = device
        self.num_speculative_steps = spec_config.num_speculative_tokens
        self.has_draft_logits = spec_config.draft_sample_method == "probabilistic"
        self._q1_audit_samples = 0
        self._q1_audit_sample_limit = 0
        if spec_config.method == "dspark" and envs.VLLM_DEEPSEEK_V4_DSPARK_Q1_AUDIT:
            self._q1_audit_sample_limit = max(
                0, envs.VLLM_DEEPSEEK_V4_DSPARK_Q1_AUDIT_SAMPLES
            )
        self._prob_audit_samples = 0
        self._prob_audit_sample_limit = 0
        if (
            spec_config.method == "dspark"
            and spec_config.draft_sample_method == "probabilistic"
            and envs.VLLM_DEEPSEEK_V4_DSPARK_PROB_AUDIT
        ):
            self._prob_audit_sample_limit = max(
                0, envs.VLLM_DEEPSEEK_V4_DSPARK_PROB_AUDIT_SAMPLES
            )
        self.enable_adaptive_verification = spec_config.enable_adaptive_verification
        rejection_sample_method = spec_config.rejection_sample_method
        self.use_block_verification: bool = False
        self.synthetic_conditional_rates: torch.Tensor | None = None
        if rejection_sample_method == "synthetic":
            assert spec_config.synthetic_acceptance_rates is not None
            self.synthetic_conditional_rates = torch.tensor(
                unconditional_to_conditional_rates(
                    spec_config.synthetic_acceptance_rates
                ),
                dtype=torch.float32,
                device=device,
            )
        elif rejection_sample_method == "block":
            self.use_block_verification = True

    @torch.inference_mode()
    def warmup(
        self,
        *,
        vocab_size: int,
        dtype: torch.dtype,
        max_num_reqs: int,
    ) -> None:
        """Compile the runtime-dtype rejection sampling kernels."""
        num_logits_per_req = self.num_speculative_steps + 1
        num_logits = max_num_reqs * num_logits_per_req
        target_logits = torch.zeros(
            num_logits,
            vocab_size,
            dtype=dtype,
            device=self.device,
        )
        if vocab_size > 1:
            target_logits[:, 1] = 1
        draft_sampled = torch.zeros(num_logits, dtype=torch.int32, device=self.device)
        cu_num_logits = (
            torch.arange(max_num_reqs + 1, dtype=torch.int32, device=self.device)
            * num_logits_per_req
        )
        idx_mapping = torch.arange(max_num_reqs, dtype=torch.int32, device=self.device)
        expanded_idx_mapping = idx_mapping.repeat_interleave(num_logits_per_req)
        expanded_local_pos = torch.arange(
            num_logits_per_req, dtype=torch.int32, device=self.device
        ).repeat(max_num_reqs)
        pos = torch.arange(num_logits, dtype=torch.int64, device=self.device)
        temperature = torch.zeros(max_num_reqs, dtype=torch.float32, device=self.device)
        seed = torch.zeros(max_num_reqs, dtype=torch.int64, device=self.device)
        draft_logits = None
        if self.has_draft_logits:
            draft_logits = torch.zeros(
                max_num_reqs,
                self.num_speculative_steps,
                vocab_size,
                dtype=torch.float32,
                device=self.device,
            )

        rejection_sample(
            target_logits,
            draft_logits,
            draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            temperature,
            seed,
            self.num_speculative_steps,
            self.synthetic_conditional_rates,
            use_fp64=self.sampler.use_fp64_gumbel,
            use_block_verification=self.use_block_verification,
        )

    def _get_logprobs_tensors(
        self,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        logits: torch.Tensor,
        cu_num_logits: torch.Tensor,
        cu_num_logits_np: np.ndarray,
        max_num_logprobs: int,
    ) -> LogprobsTensors | None:
        if max_num_logprobs == NO_LOGPROBS:
            return None

        num_reqs = cu_num_logits.shape[0] - 1
        num_logits = logits.shape[0]
        flat_sampled = torch.zeros(
            num_logits, dtype=sampled.dtype, device=sampled.device
        )
        _flatten_sampled_kernel[(num_reqs,)](
            flat_sampled,
            sampled,
            sampled.stride(0),
            num_sampled,
            cu_num_logits,
            num_warps=1,
        )
        expanded_logits = num_logits != num_reqs
        cu_num_generated_tokens: list[int] | torch.Tensor | None = None
        if expanded_logits:
            if self.enable_adaptive_verification:
                # Adaptive verification keeps the true per-request boundaries
                # on device only; cu_num_logits_np holds the pre-compacted
                # layout.
                cu_num_generated_tokens = cu_num_logits.clone()
            else:
                cu_num_generated_tokens = cu_num_logits_np.tolist()
        return compute_topk_scores(
            logits,
            max_num_logprobs,
            flat_sampled,
            cu_num_generated_tokens,
            logits_mode=self.sampler.logprobs_mode
            in ("raw_logits", "processed_logits"),
        )

    def _verify(
        self,
        logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        draft_confidence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with optional_phase_profile_stage(
            "target.rejection.apply_sampling_params", logits
        ):
            processed_logits = self.sampler.apply_sampling_params(
                logits,
                expanded_idx_mapping,
                idx_mapping,
                idx_mapping_np,
                pos,
                draft_sampled,
                expanded_local_pos,
            )
        with optional_phase_profile_stage(
            "target.rejection.verify",
            processed_logits,
            extra={"has_draft_logits": draft_logits is not None},
        ):
            sampled, num_sampled = rejection_sample(
                processed_logits,
                draft_logits,
                draft_sampled,
                cu_num_logits,
                pos,
                idx_mapping,
                expanded_idx_mapping,
                expanded_local_pos,
                self.sampler.sampling_states.temperature.gpu,
                self.sampler.sampling_states.seeds.gpu,
                self.num_speculative_steps,
                self.synthetic_conditional_rates,
                use_fp64=self.sampler.use_fp64_gumbel,
                use_block_verification=self.use_block_verification,
            )
        self._audit_q1(
            processed_logits,
            draft_sampled,
            sampled,
            num_sampled,
            pos,
            cu_num_logits,
            idx_mapping,
        )
        self._audit_probabilistic(
            processed_logits,
            draft_logits,
            draft_sampled,
            num_sampled,
            pos,
            cu_num_logits,
            idx_mapping,
            draft_confidence,
        )
        return processed_logits, sampled, num_sampled

    def _audit_probabilistic(
        self,
        processed_logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        pos: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        draft_confidence: torch.Tensor | None,
    ) -> None:
        remaining = self._prob_audit_sample_limit - self._prob_audit_samples
        if remaining <= 0 or draft_logits is None:
            return
        control_path = envs.VLLM_DEEPSEEK_V4_PHASE_PROFILE_CONTROL_PATH
        if control_path and not os.path.exists(control_path):
            return
        if torch.cuda.is_current_stream_capturing():
            return
        records = _collect_probabilistic_acceptance_audit_records(
            processed_logits,
            draft_logits,
            draft_sampled,
            num_sampled,
            pos,
            cu_num_logits,
            idx_mapping,
            remaining,
            draft_confidence,
        )
        for record in records:
            record["sample_index"] = self._prob_audit_samples
            logger.info(
                "DSpark probabilistic acceptance audit: %s",
                json.dumps(record, sort_keys=True),
            )
            self._prob_audit_samples += 1

    def _audit_q1(
        self,
        processed_logits: torch.Tensor,
        draft_sampled: torch.Tensor,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        pos: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
    ) -> None:
        remaining = self._q1_audit_sample_limit - self._q1_audit_samples
        if remaining <= 0:
            return
        control_path = envs.VLLM_DEEPSEEK_V4_PHASE_PROFILE_CONTROL_PATH
        if control_path and not os.path.exists(control_path):
            return
        request_temperatures = (
            self.sampler.sampling_states.temperature.gpu.index_select(
                0, idx_mapping.to(torch.int64)
            )
        )
        records = _collect_q1_acceptance_audit_records(
            processed_logits,
            draft_sampled,
            sampled,
            num_sampled,
            pos,
            cu_num_logits,
            request_temperatures,
            remaining,
        )
        for record in records:
            record["sample_index"] = self._q1_audit_samples
            logger.info(
                "DSpark q1 acceptance audit: %s",
                json.dumps(record, sort_keys=True),
            )
            self._q1_audit_samples += 1

    def _verify_in_chunks(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        max_chunk_logits: int,
        max_num_logprobs: int,
        draft_confidence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, LogprobsTensors | None]:
        cu_num_logits_np = input_batch.cu_num_logits_np
        use_processed_logits = self.sampler.logprobs_mode in PROCESSED_LOGPROBS_MODES
        num_reqs = input_batch.num_reqs

        if logits.shape[0] <= max_chunk_logits:
            # One chunk covers the batch. Adaptive verification compacts the logits
            # without updating cu_num_logits_np (it keeps the pre-compacted layout),
            # so the stale sums must not pick chunk boundaries; its budget cap
            # guarantees the compacted batch always lands here.
            request_chunks: Iterable[tuple[int, int]] = ((0, num_reqs),)
        else:
            assert not self.enable_adaptive_verification
            request_chunks = _iter_request_chunks(cu_num_logits_np, max_chunk_logits)

        sampled_chunks: list[torch.Tensor] = []
        num_sampled_chunks: list[torch.Tensor] = []
        logprobs_chunks: list[LogprobsTensors] = []

        for start, end in request_chunks:
            lo = int(cu_num_logits_np[start])
            hi = int(cu_num_logits_np[end])
            chunk_cu_num_logits_np = cu_num_logits_np[start : end + 1] - lo
            chunk_cu_num_logits = input_batch.cu_num_logits[start : end + 1] - lo
            # draft_logits uses persistent request-state indices and stays global.
            processed_logits, sampled, num_sampled = self._verify(
                logits[lo:hi],
                draft_logits,
                draft_sampled[lo:hi],
                pos[lo:hi],
                chunk_cu_num_logits,
                input_batch.idx_mapping[start:end],
                input_batch.idx_mapping_np[start:end],
                input_batch.expanded_idx_mapping[lo:hi],
                input_batch.expanded_local_pos[lo:hi],
                draft_confidence,
            )
            chunk_logprobs = self._get_logprobs_tensors(
                sampled,
                num_sampled,
                processed_logits if use_processed_logits else logits[lo:hi],
                chunk_cu_num_logits,
                chunk_cu_num_logits_np,
                max_num_logprobs,
            )
            if chunk_logprobs is not None:
                logprobs_chunks.append(chunk_logprobs)
            del processed_logits
            sampled_chunks.append(sampled)
            num_sampled_chunks.append(num_sampled)

        if len(sampled_chunks) == 1:
            logprobs_tensors = logprobs_chunks[0] if logprobs_chunks else None
            return sampled_chunks[0], num_sampled_chunks[0], logprobs_tensors

        logprobs_tensors = None
        if logprobs_chunks:
            expanded_logits = logits.shape[0] != input_batch.num_reqs
            logprobs_tensors = LogprobsTensors.cat(
                logprobs_chunks,
                cu_num_generated_tokens=(
                    cu_num_logits_np.tolist() if expanded_logits else None
                ),
            )

        sampled = torch.cat(sampled_chunks)
        num_sampled = torch.cat(num_sampled_chunks)
        return sampled, num_sampled, logprobs_tensors

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None = None,
        draft_confidence: torch.Tensor | None = None,
    ) -> SamplerOutput:
        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.sampler.compute_nans else None

        with optional_phase_profile_stage("target.rejection.inputs", logits):
            draft_sampled = input_batch.input_ids[input_batch.logits_indices]
            pos = input_batch.positions[input_batch.logits_indices]

        max_num_logprobs = self.sampler.sampling_states.max_num_logprobs(
            input_batch.idx_mapping_np
        )
        chunk_logit_limit = get_max_chunk_logits(logits.shape[1])
        sampled, num_sampled, logprobs_tensors = self._verify_in_chunks(
            logits,
            input_batch,
            draft_logits,
            draft_sampled,
            pos,
            chunk_logit_limit,
            max_num_logprobs,
        )

        with optional_phase_profile_stage("target.rejection.counts", num_sampled):
            num_sampled, num_rejected = get_num_sampled_and_rejected(
                num_sampled,
                input_batch.seq_lens,
                input_batch.cu_num_logits,
                input_batch.idx_mapping,
                self.sampler.req_states.prefill_len.gpu,
            )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
