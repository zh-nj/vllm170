# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

BENCHMARK = (
    Path(__file__).parents[2]
    / "benchmarks"
    / "deepseek_v4"
    / "compare_dspark_official_reference.py"
)
sys.path.insert(0, str(BENCHMARK.parent))
SPEC = importlib.util.spec_from_file_location("dsv4_dspark_reference", BENCHMARK)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


@pytest.mark.cpu_test
def test_dspark_reference_maps_historical_checkpoint_namespace() -> None:
    assert (
        benchmark._checkpoint_name(
            "model.mtp.1.self_attn.wo_a.weight_scale_inv"
        )
        == "mtp.1.attn.wo_a.scale"
    )
    assert benchmark._checkpoint_name("model.mtp.0.emb.weight") is None


@pytest.mark.cpu_test
def test_dspark_reference_logit_metrics_detect_distribution_match() -> None:
    reference = torch.tensor([[1.0, 3.0, 2.0] * 11])
    metrics = benchmark._logit_metrics(reference, reference.clone())

    assert len(metrics) == 1
    assert metrics[0]["distribution_overlap"] == pytest.approx(1.0)
    assert metrics[0]["top1_match"] is True
    assert metrics[0]["top32_intersection"] == 32
