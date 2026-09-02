# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="HC kernels require CUDA and Triton",
)

HC = 4
HIDDEN_SIZE = 2560
HYPER_HIDDEN_SIZE = HC * HIDDEN_SIZE
EPS = 1e-6


def test_grouped_gemma_rmsnorm() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    actual = grouped_gemma_rmsnorm(x, weight, EPS, HC)

    grouped = x.float().unflatten(-1, (HC, HIDDEN_SIZE))
    variance = grouped.square().mean(-1, keepdim=True)
    expected = grouped * torch.rsqrt(variance + EPS)
    expected = expected.flatten(-2) * (1.0 + weight.float())
    torch.testing.assert_close(actual, expected.to(torch.bfloat16))


def test_hc_gate_mix() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    actual = hc_gate_mix(x, gate, HC)
    expected = (
        torch.sigmoid(gate.float().unflatten(-1, (HC, HIDDEN_SIZE)))
        * x.float().unflatten(-1, (HC, HIDDEN_SIZE))
    ).mean(-2)

    torch.testing.assert_close(actual, expected.to(torch.bfloat16))


def test_hc_combine() -> None:
    torch.manual_seed(0)
    block_output = torch.randn(2, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    injection = torch.randn(2, HC, dtype=torch.bfloat16, device="cuda")

    actual = hc_combine(residual, block_output, injection, HC)
    injection_weight = 2.0 * torch.sigmoid(injection.float() / HC)
    expected = residual.float().unflatten(-1, (HC, HIDDEN_SIZE))
    expected = expected + block_output.float().unsqueeze(
        -2
    ) * injection_weight.unsqueeze(-1)

    torch.testing.assert_close(actual, expected.flatten(-2).to(torch.bfloat16))


def test_hc_combine_norm() -> None:
    torch.manual_seed(0)
    block_output = torch.randn(2, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    injection = torch.randn(2, HC, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    actual, actual_norm = hc_combine_norm(
        residual, block_output, injection, weight, EPS, HC
    )

    injection_weight = 2.0 * torch.sigmoid(injection.float() / HC)
    expected = residual.float().unflatten(-1, (HC, HIDDEN_SIZE))
    expected = expected + block_output.float().unsqueeze(
        -2
    ) * injection_weight.unsqueeze(-1)
    expected = expected.flatten(-2).to(residual.dtype)
    grouped = expected.float().unflatten(-1, (HC, HIDDEN_SIZE))
    variance = grouped.square().mean(-1, keepdim=True)
    expected_norm = grouped * torch.rsqrt(variance + EPS)
    expected_norm = expected_norm.flatten(-2) * (1.0 + weight.float())

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_norm, expected_norm.to(torch.bfloat16))


@pytest.mark.skipif(
    not current_platform.is_device_capability((8, 0))
    or not hasattr(torch.ops._C, "qwen4_exp_hc_sm80_down"),
    reason="Qwen4Exp HC down kernel requires an SM80 build",
)
@pytest.mark.parametrize("num_tokens", [1, 4])
def test_hc_sm80_down_cuda_graph(num_tokens: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260901)
    x = torch.empty((num_tokens, 10240), dtype=torch.bfloat16, device="cuda").uniform_(
        -0.1, 0.1, generator=generator
    )
    weight = torch.empty((336, 10240), dtype=torch.bfloat16, device="cuda").uniform_(
        -0.02, 0.02, generator=generator
    )
    output = torch.empty((num_tokens, 336), dtype=torch.bfloat16, device="cuda")
    expected = torch.nn.functional.linear(x, weight)

    for _ in range(3):
        torch.ops._C.qwen4_exp_hc_sm80_down(output, x, weight)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops._C.qwen4_exp_hc_sm80_down(output, x, weight)
    graph.replay()
    torch.accelerator.synchronize()

    torch.testing.assert_close(output, expected, rtol=0.016, atol=0.01)


@pytest.mark.skipif(
    not current_platform.is_device_capability((8, 0))
    or not hasattr(torch.ops._C, "qwen4_exp_gdn_sm80_input_projection"),
    reason="Qwen4Exp GDN input projection requires an SM80 build",
)
def test_gdn_sm80_input_projection_cuda_graph() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260902)
    x = torch.empty((4, 2560), dtype=torch.bfloat16, device="cuda").uniform_(
        -0.1, 0.1, generator=generator
    )
    qkvz_weight = torch.empty(
        (16384, 2560), dtype=torch.bfloat16, device="cuda"
    ).uniform_(-0.02, 0.02, generator=generator)
    ba_weight = torch.empty((96, 2560), dtype=torch.bfloat16, device="cuda").uniform_(
        -0.02, 0.02, generator=generator
    )
    qkvz_output = torch.empty((4, 16384), dtype=torch.bfloat16, device="cuda")
    ba_output = torch.empty((4, 96), dtype=torch.bfloat16, device="cuda")
    expected_qkvz = torch.nn.functional.linear(x, qkvz_weight)
    expected_ba = torch.nn.functional.linear(x, ba_weight)

    def run() -> None:
        torch.ops._C.qwen4_exp_gdn_sm80_input_projection(
            qkvz_output,
            ba_output,
            x,
            qkvz_weight,
            ba_weight,
        )

    for _ in range(3):
        run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.accelerator.synchronize()

    assert qkvz_output.is_contiguous()
    assert ba_output.is_contiguous()
    torch.testing.assert_close(qkvz_output, expected_qkvz, rtol=0.016, atol=0.01)
    torch.testing.assert_close(ba_output, expected_ba, rtol=0.016, atol=0.01)


@pytest.mark.skipif(
    not current_platform.is_device_capability((8, 0))
    or not hasattr(torch.ops._C, "qwen4_exp_sm80_out_projection"),
    reason="Qwen4Exp out projection requires an SM80 build",
)
def test_sm80_out_projection_cuda_graph() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260902)
    x = torch.empty((4, 6144), dtype=torch.bfloat16, device="cuda").uniform_(
        -0.1, 0.1, generator=generator
    )
    weight = torch.empty((2560, 6144), dtype=torch.bfloat16, device="cuda").uniform_(
        -0.02, 0.02, generator=generator
    )
    output = torch.empty((4, 2560), dtype=torch.bfloat16, device="cuda")
    expected = torch.nn.functional.linear(x, weight)

    def run() -> None:
        torch.ops._C.qwen4_exp_sm80_out_projection(output, x, weight)

    for _ in range(3):
        run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.accelerator.synchronize()

    torch.testing.assert_close(output, expected, rtol=0.016, atol=0.02)
