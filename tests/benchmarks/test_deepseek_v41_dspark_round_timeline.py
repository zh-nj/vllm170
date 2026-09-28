# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from benchmarks.deepseek_v4_1.dspark_round_timeline import (
    analyze_rounds,
    build_rounds,
    write_round_report,
)

FORWARD_MS = 9.0
SAMPLING_MS = 3.0
DRAFT_MS = 48.0
UPSTREAM_MS = 8.0


def _row(rank, stage, start_ms, duration_ms, *, flow, step=None, step_kind=None):
    return {
        "flow": flow,
        "step_kind": step_kind,
        "pp_rank": rank,
        "stage": stage,
        "step_id": step,
        "cpu_start_ns": int(start_ms * 1_000_000),
        "cpu_elapsed_us": duration_ms * 1_000,
        "elapsed_us": duration_ms * 1_000,
    }


def _target(rank, start_ms, duration_ms):
    return _row(rank, "target.model_graph_replay", start_ms, duration_ms, flow="decode")


def _round_rows(origin_ms, *, upstream=(0, 1, 2, 3)):
    """One round window: head forward, sampling, draft chain, upstream forwards."""
    rows = [
        _target(4, origin_ms, FORWARD_MS),
        _row(
            4,
            "target.sampling_total",
            origin_ms + FORWARD_MS,
            SAMPLING_MS,
            flow="dspark",
            step_kind="decode",
        ),
        _row(
            4,
            "draft.propose_total",
            origin_ms + FORWARD_MS + SAMPLING_MS,
            DRAFT_MS,
            flow="dspark",
            step_kind="decode",
        ),
        _row(
            4,
            "draft.head.recv_wait",
            origin_ms + FORWARD_MS + SAMPLING_MS + 2,
            DRAFT_MS - 4,
            flow="dspark",
            step_kind="decode",
        ),
    ]
    cursor = origin_ms + FORWARD_MS + SAMPLING_MS + DRAFT_MS
    for rank in upstream:
        rows.append(_target(rank, cursor, UPSTREAM_MS))
        rows.append(
            _row(
                rank,
                "draft.stage.prepare",
                cursor,
                1.0,
                flow="dspark",
                step_kind="decode",
            )
        )
        cursor += UPSTREAM_MS
    return rows


def _trace(tmp_path, rows):
    path = tmp_path / "phase-4-100.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return [path]


def test_round_budget_splits_the_head_period_into_the_loop_pieces(tmp_path):
    rows = _round_rows(0) + _round_rows(100) + _round_rows(200)

    result = analyze_rounds(_trace(tmp_path, rows))

    summary = result["budget"]["summary"]
    assert result["complete_rounds"] == 2
    assert result["rounds_with_all_upstream"] == 2
    assert summary["period_ms"]["p50"] == pytest.approx(100.0)
    assert summary["head_forward_ms"]["p50"] == pytest.approx(FORWARD_MS)
    assert summary["head_sampling_ms"]["p50"] == pytest.approx(SAMPLING_MS)
    assert summary["draft_chain_ms"]["p50"] == pytest.approx(DRAFT_MS)
    assert summary["upstream_target_span_ms"]["p50"] == pytest.approx(4 * UPSTREAM_MS)
    for rank in (0, 1, 2, 3):
        assert summary[f"target_P{rank}_ms"]["p50"] == pytest.approx(UPSTREAM_MS)
    # The head forward, its sampling, the draft chain and the upstream span are
    # disjoint pieces of the window, so only the remainder is unexplained.
    residual = 100.0 - FORWARD_MS - SAMPLING_MS - DRAFT_MS - 4 * UPSTREAM_MS
    assert summary["residual_ms"]["p50"] == pytest.approx(residual)
    assert summary["P4_draft.head.recv_wait_ms"]["p50"] == pytest.approx(DRAFT_MS - 4)


def test_timeline_offsets_are_relative_to_the_head_forward(tmp_path):
    rows = _round_rows(0) + _round_rows(100)

    timeline = analyze_rounds(_trace(tmp_path, rows))["timeline"]

    assert timeline["head_forward_start"]["p50"] == pytest.approx(0.0)
    assert timeline["head_forward_end"]["p50"] == pytest.approx(FORWARD_MS)
    assert timeline["head_sampling_start"]["p50"] == pytest.approx(FORWARD_MS)
    assert timeline["draft_chain_start"]["p50"] == pytest.approx(
        FORWARD_MS + SAMPLING_MS
    )
    assert timeline["target_P0_start"]["p50"] == pytest.approx(
        FORWARD_MS + SAMPLING_MS + DRAFT_MS
    )
    assert timeline["target_P3_end"]["p50"] == pytest.approx(
        FORWARD_MS + SAMPLING_MS + DRAFT_MS + 4 * UPSTREAM_MS
    )


def test_missing_upstream_forward_is_reported_not_silently_dropped(tmp_path):
    rows = (
        _round_rows(0, upstream=(0, 1, 2, 3))
        + _round_rows(100, upstream=(0, 1))
        + _round_rows(200, upstream=(0, 1, 2, 3))
    )

    result = analyze_rounds(_trace(tmp_path, rows))

    rounds = build_rounds(rows)
    assert rounds[0]["upstream_missing"] == []
    assert rounds[1]["upstream_missing"] == [2, 3]
    assert result["rounds_with_all_upstream"] == 1
    # The truncated round still reports the two ranks it does have, so the span
    # statistic is a lower bound; the count above is what qualifies it.
    assert result["budget"]["summary"]["upstream_target_span_ms"]["n"] == 2
    # Median of the complete 4-rank span and the truncated 2-rank one.
    assert result["budget"]["summary"]["upstream_target_span_ms"]["p50"] == (
        pytest.approx(24.0)
    )


def test_two_forwards_of_one_rank_in_a_window_are_rejected():
    rows = _round_rows(0) + _round_rows(100)
    rows.append(_target(0, 10, 2))

    with pytest.raises(ValueError, match="two PP0 target forwards"):
        build_rounds(rows)


def test_missing_flow_names_the_environment_variable(tmp_path):
    rows = [
        _row(
            4,
            "draft.propose_total",
            0,
            10,
            flow="dspark",
            step_kind="decode",
        )
    ]

    with pytest.raises(ValueError, match="PHASE_PROFILE_FLOWS"):
        analyze_rounds(_trace(tmp_path, rows))


def test_report_lists_budget_and_timeline(tmp_path):
    rows = _round_rows(0) + _round_rows(100)
    result = analyze_rounds(_trace(tmp_path, rows))
    out_dir = tmp_path / "round"

    write_round_report(result, out_dir)

    report = (out_dir / "report.md").read_text()
    assert "`draft_chain_ms`" in report
    assert "`target_P0_start`" in report
    assert json.loads((out_dir / "rounds.json").read_text())["complete_rounds"] == 1
