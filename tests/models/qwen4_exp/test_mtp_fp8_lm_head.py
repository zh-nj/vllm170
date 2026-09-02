# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.models.qwen4_exp.nvidia import mtp_fp8_lm_head


def _make_head() -> nn.Module:
    head = nn.Module()
    head.tp_size = 1
    head.params_dtype = torch.bfloat16
    head.bias = None
    head.weight = nn.Parameter(
        torch.empty((248320, 2560), dtype=torch.bfloat16, device="meta"),
        requires_grad=False,
    )
    head.quant_method = UnquantizedEmbeddingMethod()
    return head


def test_mtp_fp8_lm_head_eligibility_is_draft_only_sm80_tp1(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        mtp_fp8_lm_head.current_platform,
        "is_device_capability",
        lambda capability: capability == (8, 0),
    )
    head = _make_head()

    assert mtp_fp8_lm_head.can_use_qwen4_exp_mtp_fp8_lm_head(head, tied=False)
    assert not mtp_fp8_lm_head.can_use_qwen4_exp_mtp_fp8_lm_head(head, tied=True)

    head.tp_size = 2
    assert not mtp_fp8_lm_head.can_use_qwen4_exp_mtp_fp8_lm_head(head, tied=False)


def test_mtp_fp8_lm_head_rejects_other_capabilities(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        mtp_fp8_lm_head.current_platform,
        "is_device_capability",
        lambda capability: False,
    )
    assert not mtp_fp8_lm_head.can_use_qwen4_exp_mtp_fp8_lm_head(
        _make_head(), tied=False
    )


def test_mtp_fp8_lm_head_can_be_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        mtp_fp8_lm_head.current_platform,
        "is_device_capability",
        lambda capability: capability == (8, 0),
    )
    monkeypatch.setattr(mtp_fp8_lm_head.envs, "VLLM_QWEN4_EXP_MTP_FP8_LM_HEAD", False)

    assert not mtp_fp8_lm_head.can_use_qwen4_exp_mtp_fp8_lm_head(
        _make_head(), tied=False
    )


def test_enable_mtp_fp8_lm_head_marks_an_independent_head(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        mtp_fp8_lm_head.current_platform,
        "is_device_capability",
        lambda capability: capability == (8, 0),
    )
    head = _make_head()

    has_own_lm_head = mtp_fp8_lm_head.enable_qwen4_exp_mtp_fp8_lm_head(head, tied=False)

    assert has_own_lm_head
    assert isinstance(head.quant_method, mtp_fp8_lm_head.Qwen4ExpMTPFP8LMHeadMethod)


def test_mtp_fp8_lm_head_apply_uses_exact_shape(
    monkeypatch: pytest.MonkeyPatch,
):
    expected = torch.randn(2, 3)
    seen = {}

    def fake_apply(**kwargs):
        seen.update(kwargs)
        return expected

    monkeypatch.setattr(mtp_fp8_lm_head, "apply_fp8_marlin_linear", fake_apply)
    layer = nn.Module()
    layer.weight = nn.Parameter(torch.empty(1), requires_grad=False)
    layer.weight_scale = nn.Parameter(torch.empty(1), requires_grad=False)
    layer.workspace = torch.empty(1)
    x = torch.randn(2, 2560)

    output = mtp_fp8_lm_head.Qwen4ExpMTPFP8LMHeadMethod().apply(layer, x)

    assert output is expected
    assert seen["input"] is x
    assert seen["size_n"] == 248320
    assert seen["size_k"] == 2560
    assert seen["bias"] is None


def test_groupwise_fp8_quantization_tracks_each_k_group():
    weight = torch.cat(
        [
            torch.linspace(-1, 1, 128),
            torch.linspace(-32, 32, 128),
        ]
    ).reshape(1, 256)
    quantized, scales = mtp_fp8_lm_head._quantize_groupwise_fp8(weight)
    reconstructed = (
        quantized.float().reshape(1, 2, 128) * scales[:, :, None]
    ).reshape_as(weight)

    assert quantized.dtype == torch.float8_e4m3fn
    assert scales.shape == (1, 2)
    assert scales[0, 1] == pytest.approx(32 / 448)
    torch.testing.assert_close(reconstructed, weight, atol=0.5, rtol=0.05)
