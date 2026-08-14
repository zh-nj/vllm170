# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(80),
    reason="DeepSeek-V4 Triton sparse prefill tuning targets SM8x",
)


@torch.inference_mode()
def test_sparse_prefill_caller_output_and_block_k_match(monkeypatch) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_prefill

    torch.manual_seed(0)
    q = torch.randn(4, 8, 512, dtype=torch.bfloat16, device="cuda") * 0.125
    kv = torch.randn(8, 1, 512, dtype=torch.bfloat16, device="cuda") * 0.125
    indices = torch.tensor(
        [
            [0, 2, 4, -1, -1, -1],
            [1, 3, 5, 7, -1, -1],
            [0, 1, 2, 3, 4, -1],
            [2, 3, 4, 5, 6, 7],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    lengths = torch.tensor([3, 4, 5, 6], dtype=torch.int32, device="cuda")
    sink = torch.full((8,), -float("inf"), dtype=torch.float32, device="cuda")

    def run(caller_output: int, block_k: int) -> torch.Tensor:
        monkeypatch.setenv(
            "VLLM_DEEPSEEK_V4_SPARSE_PREFILL_CALLER_OUTPUT", str(caller_output)
        )
        monkeypatch.setenv("VLLM_DEEPSEEK_V4_SPARSE_PREFILL_BLOCK_K", str(block_k))
        output = torch.full_like(q, float("nan"))
        rocm_sparse_attn_prefill(
            q=q,
            kv=kv,
            indices=indices,
            topk_length=lengths,
            scale=512**-0.5,
            head_dim=512,
            nope_head_dim=448,
            rope_head_dim=64,
            attn_sink=sink,
            output=output,
        )
        return output

    copied = run(caller_output=0, block_k=16)
    direct = run(caller_output=1, block_k=16)
    tuned = run(caller_output=1, block_k=32)

    torch.testing.assert_close(direct, copied, atol=0, rtol=0)
    torch.testing.assert_close(tuned, copied, atol=2e-2, rtol=2e-2)
    assert torch.isfinite(tuned).all()
