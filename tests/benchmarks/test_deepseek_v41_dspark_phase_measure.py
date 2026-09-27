# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pytest import approx

from benchmarks.deepseek_v4_1.dspark_phase_measure import analyze


def _row(rank, stage, start_ms, duration_ms, *, step=7):
    return {
        "flow": "dspark",
        "step_kind": "decode",
        "pp_rank": rank,
        "stage": stage,
        "step_id": step,
        "cpu_start_ns": start_ms * 1_000_000,
        "cpu_elapsed_us": duration_ms * 1_000,
        "elapsed_us": duration_ms * 1_000,
    }


def test_dspark_phase_measure_aligns_prepare_and_relay_intervals():
    rows = [
        _row(4, "draft.propose_total", 0, 10),
        _row(4, "draft.head.send", 0.5, 0.5),
        _row(4, "draft.head.recv_wait", 6, 2),
        _row(1, "draft.stage.prepare", 1, 3),
        _row(1, "draft.stage.prepare_send", 1, 0.5),
        _row(1, "draft.stage.recv_wait", 0.5, 0.5),
        _row(1, "draft.stage.send", 5, 0.5),
        _row(2, "draft.stage.prepare", 2.5, 2),
        _row(2, "draft.stage.recv_wait", 1, 0.5),
        _row(2, "draft.stage.state_launch", 5.8, 0.2),
        _row(2, "draft.stage.state_wait", 6, 0.5),
        _row(2, "draft.stage.send", 6, 0.5),
        _row(3, "draft.stage.prepare", 4, 2),
        _row(3, "draft.stage.recv_wait", 2, 0.5),
        _row(3, "draft.stage.state_wait", 6.8, 0.2),
        _row(3, "draft.stage.send", 7, 0.5),
    ]

    result = analyze(rows)

    assert result["aligned_host"]["P1_P2_state_send_to_recv_ms"]["p50"] == approx(1.5)
    assert result["aligned_rounds"] == 1
    assert result["aligned_host"]["prepare_host_union_ms"]["p50"] == 5
    assert result["aligned_host"]["P1_P2_state_receiver_late_ms"]["p50"] == approx(0.8)


def test_missing_or_time_shifted_rank_is_excluded_from_alignment():
    rows = [
        _row(4, "draft.propose_total", 0, 10),
        _row(1, "draft.stage.prepare", 1, 2),
        _row(2, "draft.stage.prepare", 2, 2),
    ]
    assert analyze(rows)["aligned_rounds"] == 0
    rows.append(_row(3, "draft.stage.prepare", 20, 2))
    result = analyze(rows)
    assert result["excluded_steps"] == [7]
    assert result["aligned_host"] == {}


def test_restarted_workers_or_duplicate_scopes_are_rejected():
    import pytest

    rows = [_row(4, "draft.propose_total", 0, 10)]
    with pytest.raises(ValueError, match="duplicate scope"):
        analyze(rows * 2)
    with pytest.raises(ValueError, match="multiple worker PIDs"):
        analyze([{**rows[0], "pid": 100}, {**rows[0], "pid": 200}])


def test_local_step_offsets_and_partial_request_boundary_use_time_containment():
    rows = [
        _row(4, "draft.propose_total", 0, 10, step=0),
        _row(2, "draft.stage.prepare", 2, 2, step=0),
        _row(3, "draft.stage.prepare", 3, 2, step=0),
        _row(4, "draft.propose_total", 20, 10, step=2),
        _row(1, "draft.stage.prepare", 21, 2, step=0),
        _row(2, "draft.stage.prepare", 22, 2, step=2),
        _row(3, "draft.stage.prepare", 23, 2, step=2),
        _row(2, "draft.stage.metadata", 22, 2, step=2),
        _row(2, "draft.stage.state_launch", 22, 0.5, step=2),
    ]

    result = analyze(rows)

    assert result["excluded_steps"] == [0]
    assert result["aligned_rounds"] == 1
    assert result["rank_step_mapping"] == {2: {1: 0, 2: 2, 3: 2, 4: 2}}
    assert result["aligned_host"]["prepare_host_union_ms"]["p50"] == 4
    assert result["aligned_host"]["P2_metadata_excluding_state_launch_ms"]["p50"] == 1.5


def test_ambiguous_prepare_intervals_are_excluded():
    rows = [
        _row(4, "draft.propose_total", 0, 10),
        _row(1, "draft.stage.prepare", 1, 2),
        _row(1, "draft.stage.prepare", 4, 2, step=8),
        _row(2, "draft.stage.prepare", 2, 2),
        _row(3, "draft.stage.prepare", 3, 2),
    ]
    assert analyze(rows)["aligned_rounds"] == 0


def test_bad_json_is_not_silently_dropped(tmp_path):
    import pytest

    from benchmarks.deepseek_v4_1.dspark_phase_measure import load_rows

    path = tmp_path / "truncated.jsonl"
    path.write_text('{"flow":')
    with pytest.raises(ValueError, match="invalid JSON"):
        load_rows([path], "dspark", "decode")
