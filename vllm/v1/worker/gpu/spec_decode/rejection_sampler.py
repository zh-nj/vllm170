# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
from collections.abc import Iterator

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
    sampler_target_logits = q1_logits.gather(
        1, sampler_target_ids[:, None]
    ).squeeze(1)
    accepted_lengths = (
        num_sampled.index_select(0, valid_requests).to(torch.int64) - 1
    )

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
                "sampler_argmax_matches_torch": (
                    sampler_target_id == torch_top1_id
                ),
            }
        )
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
        self.num_speculative_steps = spec_config.num_speculative_tokens
        self._q1_audit_samples = 0
        self._q1_audit_sample_limit = 0
        if (
            spec_config.method == "dspark"
            and envs.VLLM_DEEPSEEK_V4_DSPARK_Q1_AUDIT
        ):
            self._q1_audit_sample_limit = max(
                0, envs.VLLM_DEEPSEEK_V4_DSPARK_Q1_AUDIT_SAMPLES
            )
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
        return compute_topk_scores(
            logits,
            max_num_logprobs,
            flat_sampled,
            cu_num_logits_np.tolist() if expanded_logits else None,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with optional_phase_profile_stage(
            "target.rejection.apply_sampling_params", logits
        ):
            processed_logits = self.sampler.apply_sampling_params(
                logits,
                expanded_idx_mapping,
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
        return processed_logits, sampled, num_sampled

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
    ) -> tuple[torch.Tensor, torch.Tensor, LogprobsTensors | None]:
        cu_num_logits_np = input_batch.cu_num_logits_np
        use_processed_logits = self.sampler.logprobs_mode in PROCESSED_LOGPROBS_MODES
        sampled_chunks: list[torch.Tensor] = []
        num_sampled_chunks: list[torch.Tensor] = []
        logprobs_chunks: list[LogprobsTensors] = []

        for start, end in _iter_request_chunks(cu_num_logits_np, max_chunk_logits):
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
        max_chunk_logits = max(1, MAX_CHUNK_BYTES // (logits.shape[1] * _FP32_BYTES))
        with optional_phase_profile_stage(
            "target.rejection.verify_total",
            logits,
            extra={
                "num_logits": logits.shape[0],
                "vocab_size": logits.shape[1],
                "has_draft_logits": draft_logits is not None,
            },
        ):
            sampled, num_sampled, logprobs_tensors = self._verify_in_chunks(
                logits,
                input_batch,
                draft_logits,
                draft_sampled,
                pos,
                max_chunk_logits,
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
