# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path
from typing import Any

ANALYZER = (
    Path(__file__).parents[2]
    / "benchmarks"
    / "deepseek_v4"
    / "phase_profile_analyze.py"
)
SPEC = importlib.util.spec_from_file_location("dsv4_phase_profile_analyze", ANALYZER)
assert SPEC is not None and SPEC.loader is not None
phase_profile_analyze = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = phase_profile_analyze
SPEC.loader.exec_module(phase_profile_analyze)


def _row(flow: str, stage: str, pp_rank: int, elapsed_us: float) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "flow": flow,
        "stage": stage,
        "phase": f"dsv4.{flow}.{stage}",
        "level": "detail",
        "pp_rank": pp_rank,
        "elapsed_us": elapsed_us,
        "execution": "none",
    }


def test_analyzer_uses_max_pp_rank_instead_of_summing_ranks() -> None:
    records = [
        _row("decode", "indexer.topk", 0, 10),
        _row("decode", "indexer.topk", 0, 15),
        _row("decode", "indexer.topk", 1, 40),
        _row("decode", "indexer.topk", 2, 30),
    ]

    summary = phase_profile_analyze.analyze_phase_records(records)
    stage = summary["flows"]["decode"]["critical_stage_rank"][0]

    assert stage["pp_rank"] == 1
    assert stage["total_us"] == 40
    assert stage["total_us"] != 95


def test_analyzer_reports_cpu_scope_separately_from_cuda_events() -> None:
    record = _row("prefill", "target.attn_metadata", 0, 10)
    record["cpu_elapsed_us"] = 25

    summary = phase_profile_analyze.analyze_phase_records([record])
    stage = summary["flows"]["prefill"]["by_rank_stage"][0]

    assert stage["total_us"] == 10
    assert stage["cpu_total_us"] == 25


def test_analyzer_requires_distinct_prefill_decode_and_dspark_details() -> None:
    records: list[dict[str, Any]] = []
    for flow, stages in phase_profile_analyze.DETAIL_REQUIREMENTS.items():
        records.extend(_row(flow, stage, 0, 1) for stage in stages)

    summary = phase_profile_analyze.analyze_phase_records(records)

    assert all(
        summary["flows"][flow]["detail_complete"]
        for flow in ("prefill", "decode", "dspark")
    )


def test_prefill_reports_layer_and_model_coverage() -> None:
    records = [
        _row("prefill", "target.model_forward", 0, 100),
        _row("prefill", "layer.total", 0, 80),
        _row("prefill", "layer.mhc_attention_pre", 0, 10),
        _row("prefill", "layer.attention", 0, 30),
        _row("prefill", "layer.mhc_ffn_pre", 0, 10),
        _row("prefill", "layer.ffn", 0, 25),
    ]
    for row in records:
        row["step_id"] = 0
        row["token_count"] = 5000
        row["layer"] = "model.layers.0"

    summary = phase_profile_analyze.analyze_phase_records(records)
    coverage = summary["flows"]["prefill"]["module_coverage"][0]

    assert coverage["model_unattributed_us"] == 20
    assert coverage["layer_unattributed_us"] == 5
    assert coverage["model_coverage_pct"] == 80
    assert coverage["layer_coverage_pct"] == 93.75


def test_prefill_chunks_align_by_rank_ordinal_not_global_step_id() -> None:
    records = []
    for pp_rank, step_ids, elapsed in (
        (0, (0, 1), (100, 120)),
        (1, (0, 1), (90, 110)),
        (2, (0, 2), (105, 95)),
    ):
        for step_id, elapsed_us in zip(step_ids, elapsed):
            row = _row("prefill", "target.model_forward", pp_rank, elapsed_us)
            row["step_id"] = step_id
            row["token_count"] = 2000
            records.append(row)

    chunks = phase_profile_analyze.analyze_phase_records(records)["flows"]["prefill"][
        "chunks"
    ]

    assert chunks[1]["rank_total_us"] == {"0": 120, "1": 110, "2": 95}
    assert chunks[1]["critical_pp_rank"] == 0


def test_prefill_reports_additive_marlin_breakdown_and_shapes() -> None:
    records = [_row("prefill", "moe.fused_route_experts", 0, 100)]
    for stage, elapsed_us in (
        ("moe.router_gate", 5),
        ("moe.router_select", 7),
        ("moe.marlin.align", 8),
        ("moe.marlin.w13_gemm", 40),
        ("moe.marlin.activation", 4),
        ("moe.marlin.w2_gemm", 25),
        ("moe.marlin.reduce", 3),
    ):
        row = _row("prefill", stage, 0, elapsed_us)
        if stage.startswith("moe.marlin."):
            row["extra"] = {
                "M": 2000,
                "K": 4096,
                "N": 2048,
                "E": 256,
                "topk": 6,
                "block_size_m": 64,
            }
            if stage == "moe.marlin.align":
                row["extra"]["num_tokens_post_padded"] = 15360
        records.append(row)

    marlin = phase_profile_analyze.analyze_phase_records(records)["flows"]["prefill"][
        "marlin_moe"
    ]

    assert marlin["by_rank"][0]["accounted_us"] == 92
    assert marlin["by_rank"][0]["residual_us"] == 8
    assert marlin["by_rank"][0]["coverage_pct"] == 92
    w13 = next(
        row for row in marlin["by_shape_stage"] if row["stage"] == "moe.marlin.w13_gemm"
    )
    assert w13["M"] == 2000
    assert w13["block_size_m"] == 64
    assert w13["mean_us"] == 40
    align = next(
        row for row in marlin["by_shape_stage"] if row["stage"] == "moe.marlin.align"
    )
    assert align["mean_tokens_post_padded"] == 15360
    assert align["useful_token_pct"] == 78.125
