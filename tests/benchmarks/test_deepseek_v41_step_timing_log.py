# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from benchmarks.deepseek_v4_1.step_timing_log import (
    decode_windows,
    parse_step_timing,
    summarize,
    write_report,
)

# The three shapes the collector has logged over time.
CURRENT = (
    "(Worker_PP4 pid=380050) INFO 09-19 07:38:12 [async_utils.py:119] step "
    "timing over 10 steps: target p50 7.90 ms (min 7.66), draft p50 20.96 ms "
    "(min 20.19), prep p50 6.03 ms (attn_meta 5.62, inputs 0.24), "
    "period p50 78.09 ms (other 43.19 ms)"
)
NO_DETAIL = (
    "(Worker_PP1 pid=266968) INFO 09-19 00:07:56 [async_utils.py:105] step "
    "timing over 10 steps: target p50 69.88 ms (min 5.42), draft p50 0.01 ms "
    "(min 0.00), prep p50 8.17 ms, period p50 467.61 ms (other 389.55 ms)"
)
COMBINED = (
    "(Worker_PP0 pid=233912) INFO 09-18 22:39:55 [async_utils.py:91] step "
    "timing over 10 steps: target p50 149.15 ms (min 36.78), draft p50 0.02 ms "
    "(min 0.00), combined p50 149.17 ms"
)
PREFILL = (
    "(Worker_PP2 pid=380048) INFO 09-19 07:38:06 [async_utils.py:119] step "
    "timing over 10 steps: target p50 140.44 ms (min 9.74), draft p50 0.00 ms "
    "(min 0.00), prep p50 6.95 ms (attn_meta 2.34, inputs 4.55), "
    "period p50 149.39 ms (other 1.99 ms)"
)


def test_each_field_keeps_its_own_minimum():
    rows = parse_step_timing(CURRENT)

    assert len(rows) == 1
    row = rows[0]
    assert row["rank"] == 4
    assert row["target"] == pytest.approx(7.90)
    assert row["target_min"] == pytest.approx(7.66)
    assert row["draft"] == pytest.approx(20.96)
    # The draft minimum must not overwrite the target one.
    assert row["draft_min"] == pytest.approx(20.19)
    assert row["attn_meta"] == pytest.approx(5.62)
    assert row["inputs"] == pytest.approx(0.24)
    assert row["other"] == pytest.approx(43.19)


def test_older_log_shapes_still_parse():
    without_detail = parse_step_timing(NO_DETAIL)[0]
    assert without_detail["prep"] == pytest.approx(8.17)
    assert "attn_meta" not in without_detail
    assert without_detail["other"] == pytest.approx(389.55)

    combined = parse_step_timing(COMBINED)[0]
    assert combined["target"] == pytest.approx(149.15)
    assert combined["combined"] == pytest.approx(149.17)
    assert "prep" not in combined


def test_unrelated_and_malformed_lines_are_ignored():
    text = "\n".join(
        [
            "INFO something else entirely",
            "(Worker_PP0 pid=1) INFO step timing over 10 steps: target p50",
            CURRENT,
        ]
    )
    assert len(parse_step_timing(text)) == 1


def test_prefill_windows_are_excluded_from_the_budget():
    windows = parse_step_timing("\n".join([CURRENT, PREFILL]))

    kept = decode_windows(windows, target_max_ms=50.0)
    assert [row["rank"] for row in kept] == [4]

    result = summarize(windows, target_max_ms=50.0)
    assert result["windows_total"] == 2
    assert result["windows_decode"] == 1
    assert set(result["ranks"]) == {"4"}
    assert result["decode_target_sum_ms"] == pytest.approx(7.90)


def test_medians_aggregate_per_rank_and_report_missing_fields(tmp_path):
    faster = CURRENT.replace("target p50 7.90", "target p50 6.90")
    # Same shape without the prep detail, but a decode-sized target so it is
    # kept and exercises a rank whose windows lack attn_meta/inputs.
    other_rank = NO_DETAIL.replace("target p50 69.88", "target p50 6.99")
    rows = parse_step_timing("\n".join([CURRENT, faster, other_rank]))

    result = summarize(rows, target_max_ms=50.0)

    assert result["ranks"]["4"]["windows"] == 2
    assert result["ranks"]["4"]["target"] == pytest.approx(7.40)
    assert result["ranks"]["1"]["target"] == pytest.approx(6.99)
    assert "combined" not in result["fields"]


def test_report_lists_every_rank(tmp_path):
    result = summarize(parse_step_timing(CURRENT), target_max_ms=50.0)

    write_report(result, tmp_path)

    report = (tmp_path / "report.md").read_text()
    assert "| 4 |" in report
    assert "7.90" in report
    assert (
        json.loads((tmp_path / "step_timing.json").read_text())["windows_decode"] == 1
    )
