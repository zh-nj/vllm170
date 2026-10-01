# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve sweep arithmetic without treating a byte inverse as route evidence."""

import pytest

from benchmarks.deepseek_v4_1.verify_cost_model import (
    MOE_GBPS_BY_SLOTS,
    NUM_EXPERTS_PER_TOK,
    SWEEP_TARGET_MS,
    compare,
    curve,
    expected_distinct_experts,
    expert_ms_per_new_expert,
    fit_dspark_on,
    fixed_replay_bytes_gb,
    implied_new_experts_per_token,
    moe_bandwidth_gbps,
)
from benchmarks.deepseek_v4_1.verify_intercept_analysis import fit_curve


def test_one_query_reads_exactly_its_topk_experts():
    assert expected_distinct_experts(1) == pytest.approx(NUM_EXPERTS_PER_TOK)


def test_independent_routing_expectation_and_conditional_slope_inverse():
    # 384(1 - (378/384)^6) = 34.6 distinct experts if the six queries route
    # independently. The slope inverse assumes only expert traffic is marginal.
    assert expected_distinct_experts(6) == pytest.approx(34.6, abs=0.1)
    assert 2.0 < implied_new_experts_per_token(2.425) < 3.0


def test_moe_bandwidth_is_clamped_and_grows_with_slots():
    assert moe_bandwidth_gbps(1) == MOE_GBPS_BY_SLOTS[6]
    assert moe_bandwidth_gbps(10_000) == MOE_GBPS_BY_SLOTS[48]
    rates = [moe_bandwidth_gbps(slots) for slots in (6, 12, 24, 36, 48)]
    assert rates == sorted(rates)
    assert expert_ms_per_new_expert(6) > expert_ms_per_new_expert(36)


def test_non_expert_bytes_do_not_depend_on_the_query_count():
    assert fixed_replay_bytes_gb() == pytest.approx(6.69, abs=0.01)


def test_measured_sweep_fit_preserves_the_extrapolated_q1_difference():
    fit = fit_dspark_on()

    assert fit["slope_ms_per_token"] == pytest.approx(2.42, abs=0.1)
    assert fit["max_residual_ms"] < 1.0
    # q=1 is DSpark off; the fitted DSpark-on line sits well above it.
    assert fit["fixed_penalty_ms"] == pytest.approx(9.9, abs=1.0)
    assert fit["q1_measured_ms"] == SWEEP_TARGET_MS[1]


def test_measured_slope_is_half_what_independent_routing_predicts():
    fit = fit_dspark_on()

    assert fit["independent_routing_slope_ms_per_token"] == pytest.approx(4.85, abs=0.1)
    assert (
        fit["independent_routing_slope_ms_per_token"] / fit["slope_ms_per_token"]
    ) == pytest.approx(2.0, abs=0.2)


def test_conditional_inverse_has_lower_effective_non_expert_bandwidth():
    result = compare()

    independent = result["ceiling_at_independent_routing"]
    sharing = result["ceiling_at_measured_sharing"]
    # These are model implications, not bandwidth floors or observed routes.
    assert sharing["implied_non_expert_gbps"] < 400
    assert independent["implied_non_expert_gbps"] > sharing["implied_non_expert_gbps"]


def test_curve_is_ordered_and_reproduces_the_throughput_ranking():
    rows = {row["q"]: row for row in curve()}

    assert sorted(rows) == [1, 2, 3, 4, 5, 6]
    per_token = [rows[q]["ms_per_output_token"] for q in (1, 2, 3, 4, 5, 6)]
    # One draft token is a net loss against plain decode, then it recovers.
    assert per_token[1] > per_token[0]
    assert per_token[3] < per_token[0]
    assert per_token[5] < per_token[3]
    assert rows[6]["ms_per_output_token"] == pytest.approx(13.53, abs=0.2)


def test_intercept_includes_baseline_and_extrapolated_difference():
    """The q=0 intercept must not be mistaken for a DSpark entry cost."""
    fit = fit_curve(SWEEP_TARGET_MS)
    assert fit["intercept_ms"] == pytest.approx(33.87)
    assert fit["excess_at_q1_ms"] == pytest.approx(9.915)
    assert fit["intercept_identity_ms"] == pytest.approx(fit["intercept_ms"])


def test_rank_fits_preserve_the_additive_sweep_statistic():
    first = {1: 3.0, 2: 5.0, 3: 7.0, 6: 13.0}
    second = {1: 4.0, 2: 6.0, 3: 7.0, 6: 10.0}
    total = fit_curve({q: first[q] + second[q] for q in first})
    fits = [fit_curve(first), fit_curve(second)]
    for field in ("intercept_ms", "slope_ms", "excess_at_q1_ms"):
        assert total[field] == pytest.approx(sum(fit[field] for fit in fits))


@pytest.mark.parametrize("values", ({2: 4.0, 3: 5.0}, {1: 2.0, 2: 4.0}))
def test_intercept_rejects_an_unidentified_baseline_or_slope(values):
    with pytest.raises(ValueError, match="Need DSpark-off"):
        fit_curve(values)
