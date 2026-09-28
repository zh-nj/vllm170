# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The verification cost model must keep its two claims honest.

The byte model and the measured sweep disagree, and the module exists to keep
both visible:

* the measured sweep is linear in q with a fixed penalty for entering the
  speculative path, and its slope implies far fewer new experts per token than
  independent routing would;
* weight bytes alone cannot explain the *level* of the step.

If a constant or a formula is edited so either claim stops holding, these tests
fail instead of shipping a model that no longer matches the measurement.
"""

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


def test_one_query_reads_exactly_its_topk_experts():
    assert expected_distinct_experts(1) == pytest.approx(NUM_EXPERTS_PER_TOK)


def test_independent_routing_is_the_upper_bound_that_gets_refuted():
    # 384(1 - (378/384)^6) = 34.6 distinct experts if the six queries route
    # independently; the measurement implies ~2.4 new experts per extra token.
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


def test_measured_sweep_is_linear_with_a_fixed_entry_penalty():
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


def test_weight_bytes_alone_cannot_close_the_books():
    result = compare()

    independent = result["ceiling_at_independent_routing"]
    sharing = result["ceiling_at_measured_sharing"]
    # Assuming the routing the measurement implies leaves an impossible rate
    # for the non-expert weights, so the step is not purely weight traffic.
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
