# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace
from typing import get_args

import numpy as np
import pytest
import torch

from vllm import envs
from vllm.config.model import PROCESSED_LOGPROBS_MODES, LogprobsMode
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    RejectionSampler,
    _collect_probabilistic_acceptance_audit_records,
    _collect_q1_acceptance_audit_records,
    _iter_request_chunks,
)


def test_iter_request_chunks_preserves_request_boundaries():
    cu_num_logits = np.array([0, 3, 4, 11, 13], dtype=np.int32)

    assert list(_iter_request_chunks(cu_num_logits, max_chunk_logits=5)) == [
        (0, 2),
        (2, 3),
        (3, 4),
    ]


def test_collect_q1_acceptance_audit_records() -> None:
    processed_logits = torch.tensor(
        [
            [0.0, 1.0, 4.0, 3.5],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [5.0, 2.0, 1.0, 4.75],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    draft_sampled = torch.tensor([0, 2, 3, 0, 3, 2])
    sampled = torch.tensor([[2, 3, 0], [0, 0, 0]])
    num_sampled = torch.tensor([3, 1], dtype=torch.int32)
    pos = torch.tensor([100, 101, 102, 200, 201, 202])
    cu_num_logits = torch.tensor([0, 3, 6], dtype=torch.int32)
    temperatures = torch.zeros(2)

    records = _collect_q1_acceptance_audit_records(
        processed_logits,
        draft_sampled,
        sampled,
        num_sampled,
        pos,
        cu_num_logits,
        temperatures,
        max_rows=2,
    )

    assert records == [
        {
            "request_index": 0,
            "position": 100,
            "num_draft_tokens": 2,
            "accepted_length": 2,
            "draft_token_id": 2,
            "sampler_target_id": 2,
            "torch_top1_id": 2,
            "torch_top2_id": 3,
            "draft_strict_rank": 1,
            "draft_rank_end": 1,
            "temperature": 0.0,
            "sampler_target_logit": 4.0,
            "draft_logit": 4.0,
            "torch_top1_logit": 4.0,
            "torch_top2_logit": 3.5,
            "top2_margin": 0.5,
            "draft_gap": 0.0,
            "accepted": True,
            "sampler_argmax_matches_torch": True,
        },
        {
            "request_index": 1,
            "position": 200,
            "num_draft_tokens": 2,
            "accepted_length": 0,
            "draft_token_id": 3,
            "sampler_target_id": 0,
            "torch_top1_id": 0,
            "torch_top2_id": 3,
            "draft_strict_rank": 2,
            "draft_rank_end": 2,
            "temperature": 0.0,
            "sampler_target_logit": 5.0,
            "draft_logit": 4.75,
            "torch_top1_logit": 5.0,
            "torch_top2_logit": 4.75,
            "top2_margin": 0.25,
            "draft_gap": 0.25,
            "accepted": False,
            "sampler_argmax_matches_torch": True,
        },
    ]


@pytest.mark.parametrize(
    ("method", "dspark_enabled", "dflash_enabled", "expected_limit"),
    [
        ("dspark", True, False, 17),
        ("dflash", False, True, 23),
        ("eagle", True, True, 0),
    ],
)
def test_q1_audit_method_configuration(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    dspark_enabled: bool,
    dflash_enabled: bool,
    expected_limit: int,
) -> None:
    monkeypatch.setattr(envs, "VLLM_DEEPSEEK_V4_DSPARK_Q1_AUDIT", dspark_enabled)
    monkeypatch.setattr(envs, "VLLM_DFLASH_Q1_AUDIT", dflash_enabled)
    monkeypatch.setattr(envs, "VLLM_DEEPSEEK_V4_DSPARK_Q1_AUDIT_SAMPLES", 17)
    monkeypatch.setattr(envs, "VLLM_DFLASH_Q1_AUDIT_SAMPLES", 23)

    sampler = SimpleNamespace(use_fp64_gumbel=False)
    spec_config = SimpleNamespace(
        method=method,
        num_speculative_tokens=7,
        draft_sample_method="probabilistic",
        rejection_sample_method="standard",
        enable_adaptive_verification=False,
    )
    rejection_sampler = RejectionSampler(
        sampler, spec_config, torch.device(current_platform.device_type)
    )

    assert rejection_sampler._audit_method == method
    assert rejection_sampler._q1_audit_sample_limit == expected_limit


def test_collect_probabilistic_acceptance_audit_records() -> None:
    processed_logits = torch.tensor(
        [
            [2.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, float("-inf"), 2.0],
            [0.0, 0.0, 0.0],
        ]
    )
    draft_logits = torch.tensor(
        [
            [[1.0, 2.0, 0.0], [1.0, 0.0, 2.0]],
            [[0.0, 2.0, 1.0], [0.0, 0.0, 0.0]],
        ]
    )
    draft_sampled = torch.tensor([9, 0, 1, 8, 1])
    num_sampled = torch.tensor([2, 1], dtype=torch.int32)
    pos = torch.tensor([100, 101, 102, 200, 201])
    cu_num_logits = torch.tensor([0, 3, 5], dtype=torch.int32)
    idx_mapping = torch.tensor([1, 0], dtype=torch.int32)
    draft_confidence = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

    records = _collect_probabilistic_acceptance_audit_records(
        processed_logits,
        draft_logits,
        draft_sampled,
        num_sampled,
        pos,
        cu_num_logits,
        idx_mapping,
        max_rows=3,
        draft_confidence=draft_confidence,
    )

    assert [(row["request_index"], row["draft_step"]) for row in records] == [
        (0, 0),
        (0, 1),
        (1, 0),
    ]
    assert records[0]["reached"] is True and records[0]["accepted"] is True
    assert records[1]["reached"] is True and records[1]["accepted"] is False
    assert records[2]["reached"] is True and records[2]["accepted"] is False
    assert records[0]["top1_match"] is False
    assert records[0]["confidence_logit"] == pytest.approx(0.3)
    assert records[0]["confidence_probability"] == pytest.approx(
        torch.sigmoid(torch.tensor(0.3)).item()
    )
    assert records[2]["draft_in_target_support"] is False
    assert records[2]["draft_mass_outside_target_support"] == pytest.approx(
        torch.softmax(draft_logits[0, 0], dim=-1)[1].item()
    )
    expected_overlap = torch.minimum(
        torch.softmax(processed_logits[0], dim=-1),
        torch.softmax(draft_logits[1, 0], dim=-1),
    ).sum()
    assert records[0]["expected_overlap"] == pytest.approx(expected_overlap.item())
    assert records[0]["draft_temperature_ratio_07_expected_overlap"] == pytest.approx(
        torch.minimum(
            torch.softmax(processed_logits[0], dim=-1),
            torch.softmax(draft_logits[1, 0] / 0.7, dim=-1),
        )
        .sum()
        .item()
    )
    assert records[0]["target_support_oracle_expected_overlap"] == pytest.approx(
        expected_overlap.item()
    )
    target_probs = torch.softmax(processed_logits[0], dim=-1)
    draft_top1_id = draft_logits[1, 0].argmax()
    assert records[0]["draft_top_k_1_expected_overlap"] == pytest.approx(
        target_probs[draft_top1_id].item()
    )
    assert records[0]["draft_top_k_64_expected_overlap"] == pytest.approx(
        expected_overlap.item()
    )
    assert "draft_top_p_050_expected_overlap" in records[0]
    assert "draft_top_p_099_expected_overlap" in records[0]
    assert (
        records[2]["target_support_oracle_expected_overlap"]
        > records[2]["expected_overlap"]
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("logprobs_mode", get_args(LogprobsMode))
def test_chunked_scores_match_full_batch(logprobs_mode: str):
    device = torch.device("cuda")
    cu_num_logits_np = np.array([0, 3, 4, 8, 10], dtype=np.int32)
    num_logits_per_req = np.diff(cu_num_logits_np)
    idx_mapping_np = np.array([7, 2, 9, 1], dtype=np.int32)
    input_batch = SimpleNamespace(
        num_reqs=4,
        cu_num_logits_np=cu_num_logits_np,
        cu_num_logits=torch.from_numpy(cu_num_logits_np).to(device),
        idx_mapping_np=idx_mapping_np,
        idx_mapping=torch.from_numpy(idx_mapping_np).to(device),
        expanded_idx_mapping=torch.from_numpy(
            np.repeat(idx_mapping_np, num_logits_per_req)
        ).to(device),
        expanded_local_pos=torch.from_numpy(
            np.concatenate(
                [np.arange(count, dtype=np.int32) for count in num_logits_per_req]
            )
        ).to(device),
    )
    rejection_sampler = object.__new__(RejectionSampler)
    rejection_sampler.sampler = SimpleNamespace(logprobs_mode=logprobs_mode)
    rejection_sampler.num_speculative_steps = 3
    rejection_sampler.enable_adaptive_verification = False

    def fake_verify(
        self,
        logits,
        _draft_logits,
        _draft_sampled,
        _pos,
        cu_num_logits,
        idx_mapping,
        *_mappings,
    ):
        num_sampled = torch.diff(cu_num_logits).to(torch.int32)
        sampled = (
            idx_mapping.to(torch.int64).unsqueeze(1) + torch.arange(4, device=device)
        ) % logits.shape[1]
        return logits.float() + 1, sampled, num_sampled

    rejection_sampler._verify = MethodType(fake_verify, rejection_sampler)
    logits = torch.arange(170, dtype=torch.float32, device=device).view(10, 17)

    sampled, num_sampled, chunked_logprobs = rejection_sampler._verify_in_chunks(
        logits,
        input_batch,
        draft_logits=None,
        draft_sampled=torch.arange(10, device=device),
        pos=torch.arange(10, device=device),
        max_chunk_logits=5,
        max_num_logprobs=2,
    )
    score_logits = logits + 1 if logprobs_mode in PROCESSED_LOGPROBS_MODES else logits
    full_logprobs = rejection_sampler._get_logprobs_tensors(
        sampled,
        num_sampled,
        score_logits,
        input_batch.cu_num_logits,
        input_batch.cu_num_logits_np,
        max_num_logprobs=2,
    )

    assert sampled[:, 0].tolist() == idx_mapping_np.tolist()
    assert num_sampled.tolist() == num_logits_per_req.tolist()
    assert chunked_logprobs is not None
    assert full_logprobs is not None
    assert torch.equal(
        chunked_logprobs.logprob_token_ids,
        full_logprobs.logprob_token_ids,
    )
    assert torch.equal(chunked_logprobs.logprobs, full_logprobs.logprobs)
    assert torch.equal(
        chunked_logprobs.selected_token_ranks,
        full_logprobs.selected_token_ranks,
    )
    assert (
        chunked_logprobs.cu_num_generated_tokens
        == full_logprobs.cu_num_generated_tokens
    )
