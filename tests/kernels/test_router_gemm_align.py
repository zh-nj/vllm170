# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MoE router GEMM must not depend on how many tokens share the batch.

cuBLAS picks its kernel from the problem shape, so on SM80 a MoE router GEMM
(5120 -> 384, bf16 in, fp32 out) computes different logits for the same row at
M=14 and at M=70: thousands of entries differ by up to 3.4e-6. A single request
always runs in the M<=64 regime, so a request's logits change with concurrency
and a near-tie can flip.

VLLM_DSV41_ROUTER_GEMM_ALIGN pins every batch to the M<=64 regime. These tests
use a random weight, so they need no checkpoint:

  * the split-K kernel is M-invariant by construction -- the same rows come back
    bit-identical at every total M;
  * above the boundary the pinned path reproduces the M<=64 result bit-for-bit,
    which is what makes the pin a no-op for anything a solo request computes.

Run with: .venv/bin/python -m pytest tests/kernels/test_router_gemm_align.py -v
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.router.gate_linear import (
    _router_gemm_pinned,
    _router_gemm_split_k,
)
from vllm.platforms import current_platform

# The real DSV4.1 MoE router shape.
HIDDEN = 5120
EXPERTS = 384
# M values around every cuBLAS regime boundary that matters here.
SIZES = [14, 16, 64, 65, 70, 96, 112, 128, 192, 256, 512, 1024, 2048]
LEAD = 14


def _requires_sm80():
    if not current_platform.is_cuda():
        pytest.skip("the router pin is characterised for CUDA only")
    if not current_platform.is_device_capability((8, 0)):
        pytest.skip("the router pin applies to SM80 only")


def _inputs(m: int, seed: int = 7):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    weight = torch.randn(
        EXPERTS, HIDDEN, generator=gen, device="cuda", dtype=torch.bfloat16
    )
    x = torch.randn(m, HIDDEN, generator=gen, device="cuda", dtype=torch.bfloat16)
    return x, weight


@pytest.mark.parametrize("m", SIZES)
def test_split_kernel_is_batch_invariant(m: int):
    """The same rows must come back bit-identical at every total M."""
    _requires_sm80()
    x, weight = _inputs(max(SIZES))
    reference = _router_gemm_split_k(x[:LEAD], weight)
    assert torch.equal(_router_gemm_split_k(x[:m], weight)[:LEAD], reference)


@pytest.mark.parametrize("m", SIZES)
def test_pinned_path_reproduces_the_solo_regime(m: int):
    """Pinning rewrites M>64 batches into the M<=64 regime, not into new maths.

    The reference is the plain call at LEAD rows, i.e. exactly what a solo
    request computes today. A failure means the split structure no longer
    matches this cuBLAS's small-M kernel, so VLLM_DSV41_ROUTER_GEMM_ALIGN must
    be re-derived before it is enabled again.
    """
    _requires_sm80()
    x, weight = _inputs(max(SIZES))
    solo = torch.mm(x[:LEAD], weight.T, out_dtype=torch.float32)
    assert torch.equal(_router_gemm_pinned(x[:m], weight)[:LEAD], solo)


def test_pinned_path_is_inert_when_disabled(monkeypatch):
    """Without the env var the layer must call cuBLAS exactly as before."""
    _requires_sm80()
    monkeypatch.setenv("VLLM_DSV41_ROUTER_GEMM_ALIGN", "")
    x, weight = _inputs(70)
    expected = torch.mm(x, weight.T, out_dtype=torch.float32)
    assert torch.equal(_router_gemm_pinned(x, weight), expected)
