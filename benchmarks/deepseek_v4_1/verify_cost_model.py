#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cost model for one DeepSeek-V4.1 target verification step, q=1..6.

At small decode batch sizes, weight traffic is an important cost. This byte
model assumes weights split into two kinds:

* read once per step whatever q is: attention projections, shared expert,
  router/mHC (6.69 GB), plus the LM head in sampling;
* read once per *distinct expert*: the routed MoE (0.72 GB per expert across the
  40 layers), whose traffic grows with routing diversity rather than compute.

Two independent things are reported here, and they disagree:

* the **byte model**, which needs a routing assumption. Under independent routing
  each extra verified token would pull in ~6 new experts and cost ~5 ms;
* the **measured sweep** (one PP5 restart per q, same configuration, step timing),
  whose q>=2 least-squares slope is approximately 2.43 ms per extra token.

Converting that slope to ~2.4 new experts requires attributing the entire slope
to MoE at the assumed bandwidth. It is not a routing observation. The fitted
intercept is also not a measured DSpark entry cost. See the 2026-10-01 intercept
analysis for component probes and the limits of these assumptions.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

# Whole-model bytes read per decode step, from the checkpoint tensor headers
# (decode ledger 2026-09-18 section 4).
ATTENTION_PROJECTIONS_GB = 4.96
SHARED_EXPERT_GB = 1.42
ROUTER_AND_MHC_GB = 0.31
LM_HEAD_GB = 1.32
TOP6_OF_384_EXPERTS_GB = 4.32

N_ROUTED_EXPERTS = 384
NUM_EXPERTS_PER_TOK = 6
# Pure slab read of the same addresses, and the production grouped-GEMM rate at
# increasing numbers of distinct experts (decode ledger section 6).
SLAB_READ_GBPS = 1207.0
MOE_GBPS_BY_SLOTS = {6: 575.0, 12: 700.0, 24: 760.0, 36: 790.0, 48: 795.0}

# Measured in this campaign: one PP5 restart per q, identical configuration
# except PORT_DSPARK_TOKENS, 32768-token fixed prompt / 256 output, sum of the
# five ranks' low-perturbation target spans. q=1 is DSpark off.
SWEEP_TARGET_MS = {
    1: 26.38,
    2: 38.34,
    3: 41.21,
    4: 43.77,
    5: 46.92,
    6: 47.61,
}
SWEEP_PERIOD_MS = {1: 28.12, 2: 76.88, 3: 71.31, 4: 72.42, 5: 76.70, 6: 78.41}
SWEEP_TOKENS_PER_ROUND = {
    1: 1.0,
    2: 1.992,
    3: 3.000,
    4: 3.923,
    5: 4.904,
    6: 5.795,
}
SWEEP_DECODE_TPS = {
    1: 35.55,
    2: 25.40,
    3: 42.37,
    4: 54.78,
    5: 65.12,
    6: 74.81,
}
# The q=6 round was also measured with the profiler environment loaded but the
# step-timing stream off, on an earlier instance of the same configuration.
PRODUCTION_Q6_ROUND_MS = 71.386


def fixed_replay_bytes_gb() -> float:
    """Bytes every step reads once, independent of the query count."""
    return ATTENTION_PROJECTIONS_GB + SHARED_EXPERT_GB + ROUTER_AND_MHC_GB


def expected_distinct_experts(
    queries: float,
    *,
    topk: int = NUM_EXPERTS_PER_TOK,
    experts: int = N_ROUTED_EXPERTS,
) -> float:
    """Distinct experts per layer if each query routes independently."""
    return experts * (1 - ((experts - topk) / experts) ** queries)


def moe_bandwidth_gbps(distinct_experts: float) -> float:
    """Production grouped-GEMM rate, interpolated over the measured slots."""
    slots = sorted(MOE_GBPS_BY_SLOTS)
    if distinct_experts <= slots[0]:
        return MOE_GBPS_BY_SLOTS[slots[0]]
    if distinct_experts >= slots[-1]:
        return MOE_GBPS_BY_SLOTS[slots[-1]]
    for low, high in zip(slots, slots[1:]):
        if low <= distinct_experts <= high:
            fraction = (distinct_experts - low) / (high - low)
            return MOE_GBPS_BY_SLOTS[low] + fraction * (
                MOE_GBPS_BY_SLOTS[high] - MOE_GBPS_BY_SLOTS[low]
            )
    raise AssertionError("unreachable")


def expert_ms_per_new_expert(distinct_experts: float) -> float:
    """Marginal cost of one more distinct expert in a step."""
    per_expert_gb = TOP6_OF_384_EXPERTS_GB / NUM_EXPERTS_PER_TOK
    return 1000 * per_expert_gb / moe_bandwidth_gbps(distinct_experts)


def byte_model(
    queries: float,
    *,
    distinct_experts: float | None = None,
    non_expert_ms: float,
) -> dict[str, Any]:
    """Step time from bytes, given a routing assumption and a fitted fixed term."""
    distinct = (
        expected_distinct_experts(queries)
        if distinct_experts is None
        else distinct_experts
    )
    expert_gb = TOP6_OF_384_EXPERTS_GB * distinct / NUM_EXPERTS_PER_TOK
    expert_ms = 1000 * expert_gb / moe_bandwidth_gbps(distinct)
    total_gb = fixed_replay_bytes_gb() + expert_gb
    return {
        "queries": queries,
        "distinct_experts_per_layer": distinct,
        "expert_gb": expert_gb,
        "moe_gbps": moe_bandwidth_gbps(distinct),
        "expert_ms": expert_ms,
        "non_expert_ms": non_expert_ms,
        "total_ms": expert_ms + non_expert_ms,
        "total_gb": total_gb,
        "slab_floor_ms": 1000 * total_gb / SLAB_READ_GBPS,
        "percent_of_slab_read": 100
        * 1000
        * total_gb
        / SLAB_READ_GBPS
        / (expert_ms + non_expert_ms),
    }


def implied_new_experts_per_token(slope_ms: float, *, mid_q: float = 4.0) -> float:
    """Conditional byte-model inverse of a measured ms/token slope.

    This retains the original inverse using bytes/bandwidth per expert at a
    self-consistent midpoint. It neglects bandwidth's derivative with D and
    all non-MoE marginal work; it does not measure expert diversity.
    """
    k = 2.0
    for _ in range(50):
        distinct_mid = NUM_EXPERTS_PER_TOK + k * (mid_q - 1)
        k_next = slope_ms / expert_ms_per_new_expert(distinct_mid)
        if abs(k_next - k) < 1e-9:
            break
        k = k_next
    return k


def byte_model_ceiling(measured_ms: float, distinct_experts: float) -> dict[str, Any]:
    """What a pure weight-byte model leaves for the non-expert term.

    If the step really were only weight traffic, the expert bytes at this
    routing assumption would leave ``measured_ms - expert_ms`` for 6.69 GB of
    non-expert weights, which fixes an implied bandwidth for them.
    """
    expert_ms = (
        1000
        * (TOP6_OF_384_EXPERTS_GB * distinct_experts / NUM_EXPERTS_PER_TOK)
        / moe_bandwidth_gbps(distinct_experts)
    )
    non_expert_ms = measured_ms - expert_ms
    return {
        "distinct_experts": distinct_experts,
        "expert_ms": expert_ms,
        "non_expert_ms": non_expert_ms,
        "implied_non_expert_gbps": (
            1000 * fixed_replay_bytes_gb() / non_expert_ms
            if non_expert_ms > 0
            else float("inf")
        ),
    }


def fit_dspark_on(
    sweep: dict[int, float] | None = None, *, min_q: int = 2
) -> dict[str, Any]:
    """Least-squares fit of the DSpark-on points and a conditional byte inverse.

    Only q>=2 uses the speculative path, so q=1 is excluded from the fit and
    used instead as the reference the fitted line's q=1 intercept is compared
    against. The gap is an extrapolated difference, not a measured entry cost.
    The legacy ``fixed_penalty_ms`` key is retained for existing consumers.
    """
    sweep = SWEEP_TARGET_MS if sweep is None else sweep
    points = sorted((q, ms) for q, ms in sweep.items() if q >= min_q)
    qs = [q for q, _ in points]
    times = [ms for _, ms in points]
    mean_q = statistics.mean(qs)
    mean_t = statistics.mean(times)
    slope = sum((q - mean_q) * (t - mean_t) for q, t in points) / sum(
        (q - mean_q) ** 2 for q in qs
    )
    intercept = mean_t - slope * mean_q
    fitted = {q: intercept + slope * q for q in qs}
    residual = max(abs(fitted[q] - t) for q, t in points)
    q1_measured = sweep[min(sweep)]
    independent_slope = (
        byte_model(max(qs), non_expert_ms=0.0)["expert_ms"]
        - byte_model(min(qs), non_expert_ms=0.0)["expert_ms"]
    ) / (max(qs) - min(qs))
    return {
        "slope_ms_per_token": slope,
        "intercept_ms": intercept,
        "max_residual_ms": residual,
        "points": points,
        "fitted": fitted,
        "q1_measured_ms": q1_measured,
        "dspark_on_intercept_at_q1_ms": intercept + slope,
        "fixed_penalty_ms": intercept + slope - q1_measured,
        "implied_new_experts_per_token": implied_new_experts_per_token(slope),
        "independent_routing_slope_ms_per_token": independent_slope,
    }


def curve(*, min_q: int = 1) -> list[dict[str, Any]]:
    """Measured curve with the per-output-token economics."""
    rows = []
    for q in sorted(SWEEP_TARGET_MS):
        if q < min_q:
            continue
        tokens = SWEEP_TOKENS_PER_ROUND[q]
        round_ms = SWEEP_PERIOD_MS[q]
        rows.append(
            {
                "q": q,
                "target_ms": SWEEP_TARGET_MS[q],
                "period_ms": round_ms,
                "tokens_per_round": tokens,
                "decode_tps": SWEEP_DECODE_TPS[q],
                "ms_per_output_token": round_ms / tokens,
                "target_ms_per_output_token": SWEEP_TARGET_MS[q] / tokens,
            }
        )
    return rows


def compare() -> dict[str, Any]:
    fit = fit_dspark_on()
    k = fit["implied_new_experts_per_token"]
    # Retain the conditional inverse and historical keys for reproducibility.
    distinct_q6 = NUM_EXPERTS_PER_TOK + k * 5
    return {
        "fit": fit,
        "curve": curve(),
        "ceiling_at_measured_sharing": byte_model_ceiling(
            SWEEP_TARGET_MS[6], distinct_q6
        ),
        "ceiling_at_independent_routing": byte_model_ceiling(
            SWEEP_TARGET_MS[6], expected_distinct_experts(6)
        ),
        "byte_model_independent_routing": [
            byte_model(
                q,
                non_expert_ms=byte_model_ceiling(
                    SWEEP_TARGET_MS[6], expected_distinct_experts(6)
                )["non_expert_ms"],
            )
            for q in range(1, 7)
        ],
        "production_q6_round_ms": PRODUCTION_Q6_ROUND_MS,
        "instrumentation_inflation": SWEEP_PERIOD_MS[6] / PRODUCTION_Q6_ROUND_MS - 1,
    }


def write_report(result: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "verify_cost_model.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    fit = result["fit"]
    lines = [
        "# DeepSeek-V4.1 target verification cost, q=1..6",
        "",
        (
            "One PP5 restart per q, identical configuration except the number of "
            "draft tokens; sums of the five ranks' low-perturbation target spans."
        ),
        "",
        "## Measured",
        "",
        "| q | target ms | period ms | tokens/round | decode tok/s | ms/output token |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["curve"]:
        lines.append(
            f"| {row['q']} | {row['target_ms']:.2f} | {row['period_ms']:.2f} | "
            f"{row['tokens_per_round']:.3f} | {row['decode_tps']:.2f} | "
            f"{row['ms_per_output_token']:.2f} |"
        )
    lines += [
        "",
        (
            f"DSpark-on fit over q>=2: **{fit['intercept_ms']:.2f} + "
            f"{fit['slope_ms_per_token']:.2f}*q** ms (max residual "
            f"{fit['max_residual_ms']:.2f} ms)."
        ),
        (
            f"q=1 (DSpark off) measured {fit['q1_measured_ms']:.2f} ms; the fitted "
            f"line's q=1 value is {fit['dspark_on_intercept_at_q1_ms']:.2f} ms, so "
            f"the extrapolated difference is "
            f"**{fit['fixed_penalty_ms']:.2f} ms**. No DSpark-on q=1 was measured; "
            f"this does not identify a fixed entry operation."
        ),
        "",
        (
            f"One more distinct expert costs "
            f"{expert_ms_per_new_expert(NUM_EXPERTS_PER_TOK):.3f} ms at 6 slots and "
            f"{expert_ms_per_new_expert(24):.3f} ms at 24, so the measured "
            f"{fit['slope_ms_per_token']:.2f} ms/token has a conditional byte "
            f"inverse of "
            f"**{fit['implied_new_experts_per_token']:.2f} new experts per extra "
            f"verified token**, if all marginal work is MoE. Independent routing "
            f"under the original byte model would give "
            f"{fit['independent_routing_slope_ms_per_token']:.2f} ms/token and is "
            f"inconsistent with that model fit. Expert IDs were not observed."
        ),
        "",
        "## Consistency check: can weight bytes alone close the books?",
        "",
        (
            "| routing assumption | D at q=6 | expert ms | left for 6.69 GB of "
            "non-expert weights | implied non-expert GB/s |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("independent routing", "ceiling_at_independent_routing"),
        ("conditional slope inverse", "ceiling_at_measured_sharing"),
    ):
        row = result[key]
        lines.append(
            f"| {label} | {row['distinct_experts']:.1f} | {row['expert_ms']:.2f} | "
            f"{row['non_expert_ms']:.2f} | {row['implied_non_expert_gbps']:.0f} |"
        )
    lines += [
        "",
        (
            "The conditional inverse yields effective non-expert bandwidth below "
            "prior 400-1207 GB/s probes. Those probes are not a lower bandwidth bound "
            "for all kernels. This mismatch neither measures routing nor closes a "
            "component latency decomposition."
        ),
        "",
        "## Byte model under independent routing (fixed term fitted at q=6)",
        "",
        "| q | independent D | independent ms | measured ms |",
        "|---:|---:|---:|---:|",
    ]
    for row, measured in zip(result["byte_model_independent_routing"], result["curve"]):
        lines.append(
            f"| {row['queries']:.0f} | "
            f"{row['distinct_experts_per_layer']:.1f} | {row['total_ms']:.2f} | "
            f"{measured['target_ms']:.2f} |"
        )
    lines += [
        "",
        (
            f"Step-timing instrumentation inflates the q=6 period by "
            f"{result['instrumentation_inflation']:+.1%} "
            f"({SWEEP_PERIOD_MS[6]:.2f} ms vs "
            f"{result['production_q6_round_ms']:.3f} ms "
            "uninstrumented)."
        ),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare()
    write_report(result, args.out_dir)
    fit = result["fit"]
    print(
        json.dumps(
            {
                "slope_ms_per_token": round(fit["slope_ms_per_token"], 3),
                "fixed_penalty_ms": round(fit["fixed_penalty_ms"], 2),
                "implied_new_experts_per_token": round(
                    fit["implied_new_experts_per_token"], 2
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
