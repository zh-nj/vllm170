# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit the verification intercept using raw per-rank sweep artifacts.

The intercept and the extrapolated q=1 difference are regression parameters.
This module deliberately makes no conversion from latency to routed experts.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def fit_curve(values: dict[int, float]) -> dict:
    """Fit q>=2 and keep the observed DSpark-off q=1 separate.

    Args:
        values: Query count to measured cost, including DSpark-off q=1.

    Returns:
        Fit, residuals and the algebraic intercept decomposition, in ms.

    Raises:
        ValueError: A baseline or two distinct speculative widths are missing.
    """
    points = sorted((q, cost) for q, cost in values.items() if q >= 2)
    if 1 not in values or len(points) < 2:
        raise ValueError("Need DSpark-off q=1 and at least two q>=2 points")
    x_mean = statistics.fmean(q for q, _ in points)
    y_mean = statistics.fmean(cost for _, cost in points)
    slope = sum((q - x_mean) * (cost - y_mean) for q, cost in points) / sum(
        (q - x_mean) ** 2 for q, _ in points
    )
    intercept = y_mean - slope * x_mean
    excess = intercept + slope - values[1]
    return dict(
        intercept_ms=intercept,
        slope_ms=slope,
        baseline_ms=values[1],
        extrapolated_q1_ms=intercept + slope,
        excess_at_q1_ms=excess,
        residuals_ms={q: cost - intercept - slope * q for q, cost in points},
        intercept_identity_ms=values[1] + excess - slope,
        observations=values,
    )


def analyze_sweep(root: Path) -> dict:
    files, ranks, prompt_hashes = {}, {}, {}
    for q in range(1, 7):
        path = root / f"q{q}" / "step_timing" / "step_timing.json"
        raw = path.read_bytes()
        data = json.loads(raw)
        if set(data["ranks"]) != {str(rank) for rank in range(5)}:
            raise ValueError(f"Missing PP rank for q={q}")
        files[str(path)] = hashlib.sha256(raw).hexdigest()
        ranks[q] = data["ranks"]
        request = root / f"q{q}" / ("decode.json" if q == 1 else "request/request.json")
        prompt_hashes[q] = json.loads(request.read_text())["prompt_sha256"]
    if len(set(prompt_hashes.values())) != 1:
        raise ValueError("Sweep prompts differ")
    totals = {q: sum(row["target"] for row in rs.values()) for q, rs in ranks.items()}
    per_rank = {
        rank: fit_curve({q: rs[str(rank)]["target"] for q, rs in ranks.items()})
        for rank in range(5)
    }
    return dict(
        measurement="sum of rank-local window medians; not measured PP wall time",
        total=fit_curve(totals),
        per_rank=per_rank,
        prompt_sha256=prompt_hashes[1],
        source_hashes=files,
        windows_by_rank={
            q: {rank: row["windows"] for rank, row in rs.items()}
            for q, rs in ranks.items()
        },
        leave_one_width_out={
            omitted: fit_curve({q: cost for q, cost in totals.items() if q != omitted})
            for omitted in range(2, 7)
        },
        endpoint_fit=fit_curve({q: totals[q] for q in (1, 2, 6)}),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_sweep(args.sweep)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["total"], indent=2))


if __name__ == "__main__":
    main()
