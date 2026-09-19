#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer census and attribution for an eager phase_profile decode run.

Answers two questions a graph-captured run cannot: which decoder layers a
decode step actually executes on each PP rank, and what each of them spends its
time on. ``layer.total`` is a parent scope, so its children are reported
separately and the residual is left explicit instead of being double counted.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

STEP_MARKER = "step.host_total"


def attribute_step_ids(rows: list[dict]) -> None:
    marker_steps = [i for i, r in enumerate(rows) if r.get("stage") == STEP_MARKER]
    prev = -1
    for idx in marker_steps:
        for row in rows[prev + 1 : idx + 1]:
            if row.get("step_id") is None:
                row["step_id"] = rows[idx].get("step_id")
        prev = idx


def load(paths: list[Path], flow: str) -> tuple[list[dict], int]:
    rows: list[dict] = []
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
        attribute_step_ids(file_rows)
        n_steps += sum(1 for r in file_rows if r.get("stage") == STEP_MARKER)
        rows.extend(r for r in file_rows if r.get("flow") == flow)
    return rows, n_steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--flow", default="decode")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    rows, n_steps = load([Path(p) for p in args.paths], args.flow)
    if not rows:
        raise SystemExit("no rows for that flow")
    ranks = sorted({r.get("pp_rank") for r in rows})
    per_rank_steps = max(1, n_steps // max(1, len(ranks)))

    # --- census: which layers appear, how often, on which rank ---------------
    layer_calls: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    layer_steps: dict[int, set] = defaultdict(set)
    for row in rows:
        layer = row.get("layer")
        if not layer or not layer.endswith(".total"):
            continue
        rank = row.get("pp_rank") or 0
        layer_calls[rank][layer] += 1
        layer_steps[rank].add(row.get("step_id"))

    print(f"flow={args.flow} steps/rank={per_rank_steps} ranks={ranks}")
    print("\n== layer census (layer.total records per rank) ==")
    for rank in ranks:
        names = sorted(
            layer_calls[rank],
            key=lambda n: int(n.split(".")[0]) if n.split(".")[0].isdigit() else -1,
        )
        idxs = [int(n.split(".")[0]) for n in names if n.split(".")[0].isdigit()]
        print(
            f"  PP{rank}: {len(idxs)} layers, range {min(idxs) if idxs else '-'}"
            f"..{max(idxs) if idxs else '-'}, "
            f"calls/step={layer_calls[rank][names[0]] // per_rank_steps if names else 0}"
        )
        print(f"        {idxs}")

    # --- attribution: stage totals per rank -------------------------------
    per_step: dict[tuple[str, int], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        per_step[(row["stage"], row.get("pp_rank") or 0)][row.get("step_id")].append(
            row["elapsed_us"]
        )
    totals: dict[str, dict[int, float]] = defaultdict(dict)
    for (stage, rank), by_step in per_step.items():
        totals[stage][rank] = sum(sum(v) / 1000.0 for v in by_step.values()) / (
            per_rank_steps
        )

    ranked = sorted(totals.items(), key=lambda kv: -sum(kv[1].values()))
    print(f"\n== per-stage PP-serial ms/step (eager decode) ==")
    print(f"{'stage':<50}{'PPserial':>10}  per-rank ms")
    for stage, per_rank in ranked[: args.top]:
        total = sum(per_rank.values())
        detail = " ".join(f"P{p}={ms:.2f}" for p, ms in sorted(per_rank.items()))
        print(f"{stage:<50}{total:>10.3f}  {detail}")

    # --- residual inside layer.total --------------------------------------
    print("\n== layer.total vs its declared children (PP-serial ms/step) ==")
    child_stages = [
        s
        for s in totals
        if s.startswith("layer.") and not s.endswith(".total")
    ]
    child_sum = sum(sum(totals[s].values()) for s in child_stages)
    total_sum = sum(totals.get("layer.total", {}).values())
    print(f"  layer.total            {total_sum:9.3f}")
    for stage in sorted(child_stages, key=lambda s: -sum(totals[s].values())):
        print(f"  {stage:<22} {sum(totals[stage].values()):9.3f}")
    print(f"  children (double counted with each other): {child_sum:.3f}")


if __name__ == "__main__":
    main()
