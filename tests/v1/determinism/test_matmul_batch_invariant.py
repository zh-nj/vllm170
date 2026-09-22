# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test batch-invariant matmul against torch.matmul for various shape combinations.

Tests correctness (matches torch.matmul) and batch invariance (result for one
item doesn't change based on other items in the batch).
"""

import subprocess
import sys

import pytest
import torch
from utils import skip_if_not_cuda, skip_unsupported

from vllm.model_executor.determinism.batch_invariant import (
    bmm_batch_invariant,
    matmul_batch_invariant,
    matmul_persistent,
)
from vllm.model_executor.determinism.batch_invariant_configs import (
    _BATCH_INVARIANT_MATMUL_TUNED_CONFIGS,
    _get_tuned_matmul_arch_family,
)
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


@skip_unsupported
@pytest.mark.parametrize(
    "a_shape,b_shape",
    [
        # 2D x 2D
        ((32, 64), (64, 16)),
        # 2D x 3D
        ((64, 16), (4, 16, 32)),
        # 3D x 2D
        ((4, 32, 64), (64, 16)),
        # 4D x 2D
        ((1, 4, 32, 64), (64, 16)),
        # 3D x 3D
        ((4, 32, 64), (4, 64, 16)),
        # 3D x 4D
        ((2, 32, 64), (1, 2, 64, 16)),
        # 4D x 3D (Gemma4 pattern)
        ((1, 2, 32, 64), (2, 64, 16)),
        # 4D x 4D
        ((1, 2, 32, 64), (4, 2, 64, 16)),
        # 2D x 4D
        ((32, 64), (1, 2, 64, 16)),
        # 2D x 5D
        ((32, 64), (1, 2, 2, 64, 16)),
        # 5D x 2D
        ((1, 2, 2, 32, 64), (64, 16)),
        # 5D x 5D
        ((1, 2, 4, 32, 64), (1, 2, 4, 64, 16)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_correctness(a_shape, b_shape, dtype):
    """
    Compare matmul_batch_invariant against torch.matmul for various shapes.
    """
    device = torch.device(DEVICE_TYPE)

    torch.manual_seed(42)
    a = torch.rand(a_shape, dtype=dtype, device=device)
    b = torch.rand(b_shape, dtype=dtype, device=device)

    # Standard implementation (CUDA ops)
    standard_output = torch.matmul(a, b)

    # Batch-invariant implementation (Triton)
    triton_output = matmul_batch_invariant(a, b)

    # Compare outputs
    # Use looser tolerance for bfloat16 due to its lower precision
    if dtype == torch.bfloat16:
        rtol, atol = 1e-1, 1e-1  # 10% relative tolerance for bfloat16
    else:
        rtol, atol = 1e-2, 1e-2  # 1% for float16/float32

    torch.testing.assert_close(
        triton_output,
        standard_output,
        rtol=rtol,
        atol=atol,
        msg=f"matmul mismatch for a ndim={a.ndim}, b ndim={b.ndim},",
    )


@skip_unsupported
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_batch_invariance(dtype):
    """
    Verify that the result for one item is bitwise identical regardless
    of what other items are in the batch.
    """

    device = torch.device(DEVICE_TYPE)

    torch.manual_seed(42)
    a_single = torch.rand((1, 64, 32), dtype=dtype, device=device)
    b = torch.rand((32, 128), dtype=dtype, device=device)

    standard_output = matmul_batch_invariant(a_single, b)

    a_batch = torch.rand((8, 64, 32), dtype=dtype, device=device)
    a_batch[3] = a_single[0]

    batch_output = matmul_batch_invariant(a_batch, b)
    batch_output_a = batch_output[3]

    assert torch.equal(standard_output[0], batch_output_a)


@skip_unsupported
@pytest.mark.parametrize("m", [8, 32, 256, 2048])
@pytest.mark.parametrize("transpose_b", [False, True], ids=["contiguous", "transposed"])
def test_matmul_batch_invariance_across_tuned_m_buckets(m, transpose_b):
    # Tuned M buckets must preserve each row's K-reduction order.
    capability = (
        current_platform.get_device_capability() if current_platform.is_cuda() else None
    )
    arch_family = _get_tuned_matmul_arch_family(capability)
    if arch_family not in _BATCH_INVARIANT_MATMUL_TUNED_CONFIGS:
        pytest.skip("No tuned persistent matmul config for this architecture")

    device = torch.device(DEVICE_TYPE)
    n = k = 2048
    torch.manual_seed(42)
    a = torch.rand((m, k), dtype=torch.bfloat16, device=device)
    if transpose_b:
        b = torch.rand((n, k), dtype=torch.bfloat16, device=device).t()
    else:
        b = torch.rand((k, n), dtype=torch.bfloat16, device=device)

    single_output = matmul_batch_invariant(a[:1], b)
    batch_output = matmul_batch_invariant(a, b)

    assert torch.equal(single_output[0], batch_output[0])


@skip_if_not_cuda
def test_bmm_batch_invariance_across_tuned_m1_bucket():
    capability = current_platform.get_device_capability()
    if _get_tuned_matmul_arch_family(capability) != "ampere":
        pytest.skip("The M=1 BMM config is tuned for SM80")

    device = torch.device(DEVICE_TYPE)
    batch, m, k, n = 8, 8, 4096, 1024
    torch.manual_seed(42)
    a = torch.rand((batch, m, k), dtype=torch.bfloat16, device=device)
    b = torch.rand((batch, k, n), dtype=torch.bfloat16, device=device)

    single_output = bmm_batch_invariant(a[:, :1], b)
    batch_output = bmm_batch_invariant(a, b)

    assert torch.equal(single_output[:, 0], batch_output[:, 0])


# An `out_dtype=` GEMM is how the LM head projects logits and how the DeepSeek
# V4.1 compressor scores its pooling gate, both of which read the residual
# stream (K = 5120). cuBLAS reassociates K for that width as M crosses its
# kernel-selection thresholds, so the fp32 path needs the same fixed reduction
# order as the bf16 one. `aten::mm` has a separate `dtype` overload, which the
# SM80 overrides used to miss.
COMPRESSOR_K, COMPRESSOR_N = 5120, 512
CROSSING_512 = (1, 281, 385, 509, 510, 511, 512, 513, 514, 600, 1024)


@skip_if_not_cuda
def test_matmul_persistent_fp32_output_is_batch_invariant():
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(0)
    b = torch.randn(COMPRESSOR_K, COMPRESSOR_N, dtype=torch.bfloat16, device=device)
    row0 = torch.randn(COMPRESSOR_K, dtype=torch.bfloat16, device=device)

    reference = None
    for m in CROSSING_512:
        a = torch.randn(m, COMPRESSOR_K, dtype=torch.bfloat16, device=device)
        a[0] = row0
        out = matmul_persistent(a, b, out_dtype=torch.float32)[0]
        if reference is None:
            reference = out.clone()
        else:
            assert torch.equal(out, reference), f"row 0 changed at M={m}"


@skip_if_not_cuda
def test_matmul_persistent_fp32_output_keeps_the_accumulator():
    """The wider output is the fp32 accumulator, not a rounded bf16 result."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(0)
    a = torch.randn(512, COMPRESSOR_K, dtype=torch.bfloat16, device=device)
    b = torch.randn(COMPRESSOR_K, COMPRESSOR_N, dtype=torch.bfloat16, device=device)

    out = matmul_persistent(a, b, out_dtype=torch.float32)

    assert out.dtype == torch.float32
    # fp64 reference without dispatching through the overridden matmul ops.
    reference = torch.sum(a[0].double().unsqueeze(1) * b.double(), dim=0)
    error = (out[0].double() - reference).abs().max()
    rounded = (out[0].bfloat16().double() - reference).abs().max()
    # Two orders of magnitude better than the bf16 result this overload exists
    # to avoid, which is what "it is the accumulator itself" means.
    assert error * 100 < rounded


# The mode is a one-shot, process-global dispatcher registration that makes
# unsupported dtypes raise, so the check runs in a child process instead of
# poisoning every later test in this session.
_MM_DTYPE_CHILD = """
import torch

from vllm.model_executor.determinism.batch_invariant import enable_batch_invariant_mode

enable_batch_invariant_mode()
K, N = 5120, 512
torch.manual_seed(0)
weight = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
row0 = torch.randn(K, dtype=torch.bfloat16, device="cuda")
reference = None
for m in (1, 281, 385, 509, 510, 511, 512, 513, 514, 600, 1024):
    activations = torch.randn(m, K, dtype=torch.bfloat16, device="cuda")
    activations[0] = row0
    out = torch.mm(activations, weight, out_dtype=torch.float32)[0]
    if reference is None:
        reference = out.clone()
    elif not torch.equal(out, reference):
        raise SystemExit(f"row 0 changed at M={m}")
print("batch invariant")
"""


@skip_if_not_cuda
def test_batch_invariant_mode_covers_the_mm_out_dtype_overload():
    if not current_platform.is_device_capability_family(80):
        pytest.skip("The SM80 overrides are the ones that were incomplete")

    result = subprocess.run(
        [sys.executable, "-c", _MM_DTYPE_CHILD],
        capture_output=True,
        text=True,
        timeout=900,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "batch invariant" in result.stdout
