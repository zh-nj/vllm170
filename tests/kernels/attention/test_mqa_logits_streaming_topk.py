# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.mqa_logits_triton import (
    fp8_mqa_logits_streaming_topk_triton,
    fp8_mqa_logits_triton,
)


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(80),
    reason="SM8x Triton kernel required",
)
@torch.inference_mode()
def test_full_logits_uses_caller_output() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260816)
    M, N, H, D = 3, 1024, 64, 128
    q = torch.randn((M, H, D), device="cuda", generator=generator).to(
        torch.float8_e4m3fn
    )
    k = torch.randn((N, D), device="cuda", generator=generator).to(torch.float8_e4m3fn)
    scales = torch.rand((N,), device="cuda", generator=generator) + 0.5
    weights = torch.randn((M, H), device="cuda", generator=generator)
    starts = torch.zeros(M, dtype=torch.int32, device="cuda")
    ends = torch.full((M,), N, dtype=torch.int32, device="cuda")
    expected = fp8_mqa_logits_triton(q, (k, scales), weights, starts, ends)
    output = torch.empty_like(expected)
    q_bf16 = torch.empty_like(q, dtype=torch.bfloat16)
    k_bf16 = torch.empty_like(k, dtype=torch.bfloat16)

    actual = fp8_mqa_logits_triton(
        q,
        (k, scales),
        weights,
        starts,
        ends,
        logits_out=output,
        q_bf16_out=q_bf16,
        k_bf16_out=k_bf16,
    )

    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(q_bf16, q.to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(k_bf16, k.to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(80),
    reason="SM8x Triton kernel required",
)
@torch.inference_mode()
def test_streaming_topk_matches_full_logits() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260813)
    M, N, H, D, topk = 5, 2304, 64, 128, 512
    q = torch.randn((M, H, D), device="cuda", generator=generator).to(
        torch.float8_e4m3fn
    )
    k = torch.randn((N, D), device="cuda", generator=generator).to(torch.float8_e4m3fn)
    scales = torch.rand((N,), device="cuda", generator=generator) + 0.5
    weights = torch.randn((M, H), device="cuda", generator=generator)
    starts = torch.tensor([0, 37, 400, 900, 1500], dtype=torch.int32, device="cuda")
    ends = torch.tensor(
        [2304, 2200, 1800, 1500, 2200], dtype=torch.int32, device="cuda"
    )

    full_logits = fp8_mqa_logits_triton(
        q, (k, scales), weights, starts, ends, clean_logits=False
    )
    expected_scores, expected_absolute_indices = full_logits.topk(topk, dim=1)
    expected_indices = expected_absolute_indices - starts[:, None]
    scores, indices = fp8_mqa_logits_streaming_topk_triton(
        q,
        (k, scales),
        weights,
        starts,
        ends,
        topk,
        tile_size=1024,
    )

    assert torch.equal(
        indices.sort().values,
        expected_indices.to(torch.int32).sort().values,
    )
    torch.testing.assert_close(
        scores.sort(dim=1, descending=True).values,
        expected_scores,
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(80),
    reason="SM8x Triton kernel required",
)
@torch.inference_mode()
def test_streaming_topk_matches_full_logits_with_grouped_kv_tiles() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260816)
    M, N, H, D, topk = 32, 16384, 64, 128, 512
    q = torch.randn((M, H, D), device="cuda", generator=generator).to(
        torch.float8_e4m3fn
    )
    k = torch.randn((N, D), device="cuda", generator=generator).to(torch.float8_e4m3fn)
    scales = torch.rand((N,), device="cuda", generator=generator) + 0.5
    weights = torch.randn((M, H), device="cuda", generator=generator)
    starts = torch.zeros(M, dtype=torch.int32, device="cuda")
    ends = torch.full((M,), N, dtype=torch.int32, device="cuda")

    full_logits = fp8_mqa_logits_triton(
        q, (k, scales), weights, starts, ends, clean_logits=False
    )
    expected_scores = full_logits.topk(topk, dim=1).values
    scores, _ = fp8_mqa_logits_streaming_topk_triton(
        q,
        (k, scales),
        weights,
        starts,
        ends,
        topk,
        tile_size=8192,
    )

    torch.testing.assert_close(
        scores.sort(dim=1, descending=True).values,
        expected_scores,
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(80),
    reason="SM8x Triton kernel required",
)
@torch.inference_mode()
def test_streaming_topk_pads_rows_shorter_than_topk() -> None:
    generator = torch.Generator(device="cuda").manual_seed(17)
    M, N, H, D, topk = 2, 768, 64, 128, 512
    q = torch.randn((M, H, D), device="cuda", generator=generator).to(
        torch.float8_e4m3fn
    )
    k = torch.randn((N, D), device="cuda", generator=generator).to(torch.float8_e4m3fn)
    scales = torch.rand((N,), device="cuda", generator=generator) + 0.5
    weights = torch.randn((M, H), device="cuda", generator=generator)
    starts = torch.tensor([100, 500], dtype=torch.int32, device="cuda")
    ends = torch.tensor([400, 768], dtype=torch.int32, device="cuda")

    full_logits = fp8_mqa_logits_triton(
        q, (k, scales), weights, starts, ends, clean_logits=False
    )
    scores, indices = fp8_mqa_logits_streaming_topk_triton(
        q, (k, scales), weights, starts, ends, topk
    )

    for row, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        valid = end - start
        expected_scores, expected_indices = full_logits[row, start:end].topk(valid)
        assert torch.equal(
            indices[row, :valid].sort().values,
            expected_indices.to(torch.int32).sort().values,
        )
        assert torch.all(indices[row, valid:] == -1)
        torch.testing.assert_close(
            scores[row, :valid].sort(descending=True).values,
            expected_scores,
            rtol=0,
            atol=0,
        )
        assert torch.all(scores[row, valid:] == float("-inf"))
