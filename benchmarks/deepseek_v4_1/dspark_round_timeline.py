#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the whole DSpark round timeline from one phase-profile trace.

``dspark_phase_measure`` reports the draft chain per rank. This module adds the
other half of the round: the target verify forward on PP0-PP4 and the rejection
sampling on the head rank, which the draft-only report never covered.

One round is anchored on the head rank's target forward: the window
``[head_forward(n), head_forward(n+1))`` holds the head forward and sampling of
step n, the whole distributed draft chain of step n, and the upstream ranks'
forwards of step n+1, which is every serial piece of the loop. The budget below
splits the head period into exactly those pieces.

Offline only: point it at the trace files of one profiled request.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks.deepseek_v4_1.dspark_phase_measure import (
    _interval,
    _paths,
    _summary,
    load_rows,
)

# The target forward is one opaque graph replay in FULL mode and one scope in
# eager/piecewise mode; accept either so the tool cannot silently report zero.
TARGET_FORWARD_STAGES = ("target.model_graph_replay", "target.model_forward")
DRAFT_HEAD_STAGES = (
    "draft.head.context_hidden",
    "draft.head.send",
    "draft.head.recv_wait",
    "draft.head.result_copy",
    "draft.markov_total",
)
DRAFT_STAGE_STAGES = (
    "draft.stage.recv_wait",
    "draft.stage.prepare_send",
    "draft.stage.prepare",
    "draft.stage.metadata",
    "draft.stage.state_launch",
    "draft.stage.state_wait",
    "draft.stage.forward",
    "draft.stage.send",
)
UPSTREAM_RANKS = (0, 1, 2, 3)


def _by_rank_stage(
    rows: list[dict[str, Any]],
) -> dict[tuple[int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["pp_rank"]), row["stage"])].append(row)
    for members in grouped.values():
        members.sort(key=lambda row: float(row["cpu_start_ns"]))
    return grouped


def _contained(
    members: list[dict[str, Any]] | None,
    lo: float,
    hi: float,
) -> list[dict[str, Any]]:
    """Rows that start inside the window and do not run past its end."""
    selected = []
    for row in members or ():
        start, end = _interval(row)
        if start >= lo and end <= hi:
            selected.append(row)
    return selected


def _target_forward(
    grouped: dict[tuple[int, str], list[dict[str, Any]]], rank: int
) -> list[dict[str, Any]]:
    for stage in TARGET_FORWARD_STAGES:
        members = grouped.get((rank, stage))
        if members:
            return members
    return []


def build_rounds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split the trace into rounds anchored on the head rank's target forward."""
    grouped = _by_rank_stage(rows)
    head_forward = _target_forward(grouped, 4)
    if not head_forward:
        raise ValueError(
            "no head-rank target forward scope; expected one of "
            f"{', '.join(TARGET_FORWARD_STAGES)} on PP4"
        )
    proposals = grouped.get((4, "draft.propose_total"))
    rounds: list[dict[str, Any]] = []
    for index, forward in enumerate(head_forward[:-1]):
        lo, _ = _interval(forward)
        hi = _interval(head_forward[index + 1])[0]
        upstream: dict[int, tuple[float, float]] = {}
        for rank in UPSTREAM_RANKS:
            for row in _contained(_target_forward(grouped, rank), lo, hi):
                start, end = _interval(row)
                if rank in upstream:
                    raise ValueError(
                        f"two PP{rank} target forwards inside round {index}"
                    )
                upstream[rank] = (start, end)
        draft_stages: dict[int, dict[str, tuple[float, float]]] = defaultdict(dict)
        for rank in (1, 2, 3, 4):
            for stage in DRAFT_HEAD_STAGES + DRAFT_STAGE_STAGES:
                for row in _contained(grouped.get((rank, stage)), lo, hi):
                    draft_stages[rank].setdefault(stage, _interval(row))
        sampling = _contained(grouped.get((4, "target.sampling_total")), lo, hi)
        draft = _contained(proposals, lo, hi)
        rounds.append(
            {
                "round": index,
                "period_ms": hi - lo,
                "head_forward": _interval(forward),
                "head_sampling": _interval(sampling[0]) if sampling else None,
                "draft_chain": _interval(draft[0]) if draft else None,
                "upstream": upstream,
                "upstream_missing": [
                    rank for rank in UPSTREAM_RANKS if rank not in upstream
                ],
                "draft_stages": draft_stages,
            }
        )
    return rounds


def _phase_budget(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    budget: dict[str, list[float]] = defaultdict(list)
    for row in rounds:
        budget["period_ms"].append(row["period_ms"])
        start, end = row["head_forward"]
        budget["head_forward_ms"].append(end - start)
        if row["head_sampling"]:
            budget["head_sampling_ms"].append(
                row["head_sampling"][1] - row["head_sampling"][0]
            )
        if row["draft_chain"]:
            budget["draft_chain_ms"].append(
                row["draft_chain"][1] - row["draft_chain"][0]
            )
        upstream_span = 0.0
        if row["upstream"]:
            upstream_start = min(value[0] for value in row["upstream"].values())
            upstream_end = max(value[1] for value in row["upstream"].values())
            upstream_span = upstream_end - upstream_start
            budget["upstream_target_span_ms"].append(upstream_span)
        for rank, (rank_start, rank_end) in sorted(row["upstream"].items()):
            budget[f"target_P{rank}_ms"].append(rank_end - rank_start)
        for rank, stages in row["draft_stages"].items():
            for stage, (stage_start, stage_end) in stages.items():
                budget[f"P{rank}_{stage}_ms"].append(stage_end - stage_start)
        accounted = (
            end
            - start
            + upstream_span
            + (budget["head_sampling_ms"][-1] if row["head_sampling"] else 0.0)
            + (budget["draft_chain_ms"][-1] if row["draft_chain"] else 0.0)
        )
        budget["residual_ms"].append(row["period_ms"] - accounted)
    return {
        "complete_rounds": len(rounds),
        "summary": {key: _summary(values) for key, values in sorted(budget.items())},
    }


def _timeline(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """Median offset of every endpoint from the head forward of its round."""
    offsets: dict[str, list[float]] = defaultdict(list)
    for row in rounds:
        origin = row["head_forward"][0]
        offsets["head_forward_start"].append(0.0)
        offsets["head_forward_end"].append(row["head_forward"][1] - origin)
        if row["head_sampling"]:
            offsets["head_sampling_start"].append(row["head_sampling"][0] - origin)
            offsets["head_sampling_end"].append(row["head_sampling"][1] - origin)
        if row["draft_chain"]:
            offsets["draft_chain_start"].append(row["draft_chain"][0] - origin)
            offsets["draft_chain_end"].append(row["draft_chain"][1] - origin)
        for rank, (start, end) in row["upstream"].items():
            offsets[f"target_P{rank}_start"].append(start - origin)
            offsets[f"target_P{rank}_end"].append(end - origin)
        for rank, stages in row["draft_stages"].items():
            for stage, (start, end) in stages.items():
                offsets[f"P{rank}_{stage}_start"].append(start - origin)
                offsets[f"P{rank}_{stage}_end"].append(end - origin)
    return {
        key: _summary(values)
        for key, values in sorted(
            offsets.items(), key=lambda item: _summary(item[1])["p50"]
        )
    }


def _load(trace: list[Path], flow: str, step_kind: str) -> list[dict[str, Any]]:
    try:
        return load_rows(trace, flow, step_kind)
    except ValueError as exc:
        raise ValueError(
            f"no '{flow}' phase rows in {[str(path) for path in _paths(trace)]}; "
            "the server must be started with VLLM_DEEPSEEK_V4_PHASE_PROFILE_FLOWS "
            f"that includes '{flow}'"
        ) from exc


def analyze_rounds(trace: list[Path]) -> dict[str, Any]:
    draft_rows = _load(trace, "dspark", "decode")
    # Decode-flow rows carry no step_kind: a target verify step is not a draft.
    target_rows = _load(trace, "decode", "")
    rows = draft_rows + target_rows
    rounds = build_rounds(rows)
    return {
        "rows": len(rows),
        "draft_rows": len(draft_rows),
        "target_rows": len(target_rows),
        "complete_rounds": len(rounds),
        "rounds_with_all_upstream": sum(
            1 for row in rounds if not row["upstream_missing"]
        ),
        "paths": [str(path) for path in _paths(trace)],
        "budget": _phase_budget(rounds),
        "timeline": _timeline(rounds),
    }


def write_round_report(result: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rounds.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    summary = result["budget"]["summary"]
    with (out_dir / "rounds.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "n", "p50_ms", "p95_ms", "mean_ms"])
        for key, value in summary.items():
            writer.writerow(
                [key, value["n"], value["p50"], value["p95"], value["mean"]]
            )
    lines = [
        "# DSpark whole-round timing",
        "",
        "Host scope time only. Parent and child rows are not additive; the three "
        "draft stages overlap and so do the upstream target forwards.",
        f"Rounds: {result['complete_rounds']} (complete, with a following round); "
        f"{result['rounds_with_all_upstream']} also carry all four upstream forwards",
        f"Rows: {result['rows']} "
        f"(draft {result['draft_rows']}, target {result['target_rows']})",
        "",
        "`residual_ms` is the head period minus the head forward, the head "
        "sampling, the draft chain and the upstream target span: broadcast, "
        "scheduler and pipeline gaps.",
        "",
        "## Round budget",
        "",
        "| Metric | n | p50 ms | p95 ms |",
        "|---|---:|---:|---:|",
    ]
    for key, value in summary.items():
        lines.append(
            f"| `{key}` | {value['n']} | {value['p50']:.3f} | {value['p95']:.3f} |"
        )
    lines += [
        "",
        "## Timeline from the head target forward (median endpoints)",
        "",
        "| Endpoint | p50 ms | p95 ms |",
        "|---|---:|---:|",
    ]
    for key, value in result["timeline"].items():
        lines.append(f"| `{key}` | {value['p50']:.3f} | {value['p95']:.3f} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_rounds(args.trace)
    write_round_report(result, args.out_dir)
    print(
        json.dumps(
            {
                "rounds": result["complete_rounds"],
                "budget": {
                    key: round(value["p50"], 3)
                    for key, value in result["budget"]["summary"].items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
