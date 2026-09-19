#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Turn phase_profile JSONL into a PP-serial decode ledger.

Ranks of a pipeline execute one token after another, so a stage's share of the
critical path is the SUM of its per-step time across the ranks that run it.
Parent and child scopes both appear, so nesting is marked instead of adding a
column blindly.

Scopes that run before ``phase_profile_step`` is entered (attention metadata,
model inputs) carry no step id: the runner builds them ahead of the step
context, then the step flush writes them out. They are re-attributed to the
step whose flush wrote them, in file order.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

STEP_MARKER = "step.host_total"


def _attribute_step_ids(rows: list[dict]) -> int:
    """Assign step ids to rows written without one, in flush order."""
    # A flush writes the pending rows and then the step marker, so the marker's
    # step id is the step those rows belong to.
    marker_steps = [
        i for i, r in enumerate(rows) if r.get("stage") == STEP_MARKER
    ]
    if not marker_steps:
        return 0
    bounds = []
    prev = -1
    for idx in marker_steps:
        bounds.append((prev, idx, rows[idx].get("step_id")))
        prev = idx
    for start, end, step in bounds:
        for row in rows[start + 1 : end + 1]:
            if row.get("step_id") is None:
                row["step_id"] = step
    return len(bounds)


def load(paths: list[Path], flow: str) -> tuple[list[dict], int]:
    rows = []
    n_steps = 0
    for path in paths:
        file_rows = []
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    file_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        n_steps += _attribute_step_ids(file_rows)
        rows.extend(r for r in file_rows if r.get("flow") == flow)
    return rows, n_steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--flow", default="decode")
    ap.add_argument("--min-us", type=float, default=5.0)
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    rows, n_steps = load([Path(p) for p in args.paths], args.flow)
    if not rows:
        raise SystemExit("no rows for that flow")
    ranks = sorted({r.get("pp_rank") for r in rows})
    per_rank_steps = max(1, n_steps // max(1, len(ranks)))

    per_step: dict[tuple[str, int], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    cpu_per_step: dict[tuple[str, int], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        key = (row["stage"], row.get("pp_rank") or 0)
        step = row.get("step_id")
        per_step[key][step].append(row["elapsed_us"])
        cpu_per_step[key][step].append(row.get("cpu_elapsed_us", 0.0))

    stage_rank_ms: dict[str, dict[int, float]] = defaultdict(dict)
    stage_rank_cpu: dict[str, dict[int, float]] = defaultdict(dict)
    stage_rank_n: dict[str, dict[int, float]] = defaultdict(dict)
    for (stage, pp), by_step in per_step.items():
        totals = [sum(v) / 1000.0 for v in by_step.values()]
        stage_rank_ms[stage][pp] = sum(totals) / per_rank_steps
        stage_rank_n[stage][pp] = sum(len(v) for v in by_step.values()) / max(
            1, len(by_step)
        )
        cpu_totals = [
            sum(v) / 1000.0 for v in cpu_per_step[(stage, pp)].values()
        ]
        stage_rank_cpu[stage][pp] = sum(cpu_totals) / per_rank_steps

    ranked = sorted(stage_rank_ms.items(), key=lambda kv: -sum(kv[1].values()))
    print(
        f"flow={args.flow} ranks={ranks} steps/rank={per_rank_steps} "
        f"rows={len(rows)}"
    )
    print(
        f"{'stage':<52}{'PPserial':>10}{'calls':>7}{'cpu':>9}  per-rank ms/step"
    )
    csv_rows = []
    for stage, per_rank in ranked[: args.top]:
        total = sum(per_rank.values())
        if total * 1000 < args.min_us:
            continue
        cpu = sum(stage_rank_cpu[stage].values())
        calls = sum(stage_rank_n[stage].values())
        detail = " ".join(f"P{p}={ms:.3f}" for p, ms in sorted(per_rank.items()))
        print(
            f"{stage:<52}{total:>10.3f}{calls:>7.0f}{cpu:>9.3f}  {detail}"
        )
        csv_rows.append((stage, total, cpu, detail))

    if args.csv:
        with open(args.csv, "w") as fh:
            json.dump(
                [
                    {"stage": s, "pp_serial_ms": t, "cpu_ms": c, "per_rank": d}
                    for s, t, c, d in csv_rows
                ],
                fh,
                indent=2,
            )
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
