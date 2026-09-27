#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize DeepSeek-V4.1 DSpark phase-profile JSONL files.

The analyzer keeps host scope time, CUDA event time, and cross-rank host
intervals separate. It is intentionally offline: point it at trace files from
one profiled request and it does not need a running model or another GPU run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)]


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "p50": 0.0, "p95": 0.0, "mean": 0.0}
    return {
        "n": len(values),
        "p50": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "mean": statistics.mean(values),
    }


def _interval(row: dict[str, Any]) -> tuple[float, float]:
    start = float(row["cpu_start_ns"]) / 1e6
    return start, start + float(row.get("cpu_elapsed_us", 0.0)) / 1000.0


def _union(intervals: list[tuple[float, float]]) -> float:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def _paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for path in inputs:
        if path.is_dir():
            paths.extend(sorted(path.glob("*.jsonl")))
        else:
            paths.append(path)
    return paths


def load_rows(inputs: list[Path], flow: str, step_kind: str) -> list[dict[str, Any]]:
    rows = []
    for path in _paths(inputs):
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc
                if row.get("flow") != flow:
                    continue
                if step_kind and row.get("step_kind") != step_kind:
                    continue
                if row.get("pp_rank") is None or row.get("stage") is None:
                    continue
                rows.append(row)
    if not rows:
        raise ValueError("no matching phase rows")
    return rows


def _step_key(row: dict[str, Any]) -> tuple[str, int]:
    step = row.get("step_id")
    if step is not None:
        return "step", int(step)
    return "host", int(row["cpu_start_ns"])


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # step_id is local to a worker lifetime; never mix restarts in one report.
    pids: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        for field in ("cpu_start_ns", "cpu_elapsed_us", "elapsed_us"):
            if field not in row or not math.isfinite(float(row[field])):
                raise ValueError(f"missing or invalid {field}; need CUDA phase trace")
        if row.get("pid") is not None:
            pids[row["pp_rank"]].add(row["pid"])
    if any(len(values) > 1 for values in pids.values()):
        raise ValueError("multiple worker PIDs per rank; select one server run")
    grouped: dict[tuple[int, str, tuple[str, int]], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[(int(row["pp_rank"]), row["stage"], _step_key(row))].append(row)

    stage_rows: list[dict[str, Any]] = []
    for (rank, stage), members in _group_stage_rows(grouped).items():
        host = [
            sum(float(row.get("cpu_elapsed_us", 0.0)) for row in group) / 1000
            for group in members
        ]
        cuda = [
            sum(float(row.get("elapsed_us", 0.0)) for row in group) / 1000
            for group in members
        ]
        stage_rows.append(
            {
                "pp_rank": rank,
                "stage": stage,
                "samples": len(members),
                "denominator": (
                    "round" if members[0][0].get("step_id") is not None else "call"
                ),
                "calls_per_sample": statistics.mean(len(group) for group in members),
                "host_ms": _summary(host),
                "cuda_ms": _summary(cuda),
            }
        )
    stage_rows.sort(key=lambda row: (row["pp_rank"], row["stage"]))

    by_step: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for (rank, stage, key), members in grouped.items():
        if key[0] == "step":
            if stage.startswith("draft.") and len(members) != 1:
                raise ValueError(f"duplicate scope: PP{rank}, {stage}, {key}")
            if len(members) == 1:
                by_step[(rank, key[1])][stage] = members[0]

    # Cross-rank step IDs are local counters. Match only a unique prepare
    # interval contained in the head proposal on this single-host PP5 layout.
    head_steps = sorted(
        step
        for (rank, step), scopes in by_step.items()
        if rank == 4 and "draft.propose_total" in scopes
    )
    aligned: dict[str, list[float]] = defaultdict(list)
    mapping: dict[int, dict[int, int]] = {rank: {} for rank in (1, 2, 3, 4)}
    common_steps = []
    excluded = []
    used: set[tuple[int, int]] = set()
    for head_step in head_steps:
        lo, hi = _interval(by_step[(4, head_step)]["draft.propose_total"])
        selected = {4: head_step}
        for rank in (1, 2, 3):
            matches = [
                step
                for (candidate_rank, step), scopes in by_step.items()
                if candidate_rank == rank
                and "draft.stage.prepare" in scopes
                and lo
                <= _interval(scopes["draft.stage.prepare"])[0]
                <= _interval(scopes["draft.stage.prepare"])[1]
                <= hi
            ]
            if len(matches) == 1 and (rank, matches[0]) not in used:
                selected[rank] = matches[0]
        if len(selected) != 4:
            excluded.append(head_step)
            continue
        common_steps.append(head_step)
        for rank, step in selected.items():
            mapping[rank][head_step] = step
            used.add((rank, step))
    for head_step in common_steps:
        rank_steps = {rank: mapping[rank][head_step] for rank in (1, 2, 3, 4)}
        intervals = []
        for rank in (1, 2, 3):
            row = by_step.get((rank, rank_steps[rank]), {}).get("draft.stage.prepare")
            if row:
                intervals.append(_interval(row))
        if intervals:
            aligned["prepare_host_union_ms"].append(_union(intervals))
            aligned["prepare_host_work_sum_ms"].append(
                sum(end - start for start, end in intervals)
            )

        head = by_step[(4, head_step)]
        for rank in (1, 2, 3):
            scopes = by_step[(rank, rank_steps[rank])]
            metadata = scopes.get("draft.stage.metadata")
            launch = scopes.get("draft.stage.state_launch")
            if metadata:
                elapsed = float(metadata["cpu_elapsed_us"]) / 1000
                if launch:
                    elapsed -= float(launch["cpu_elapsed_us"]) / 1000
                aligned[f"P{rank}_metadata_excluding_state_launch_ms"].append(elapsed)
        if "draft.head.send" in head and "draft.head.recv_wait" in head:
            aligned["head_send_to_result_host_ms"].append(
                _interval(head["draft.head.recv_wait"])[1]
                - _interval(head["draft.head.send"])[0]
            )
        edges = (
            (4, 1, "draft.head.send", ("draft.stage.recv_wait",), "prepare"),
            (1, 2, "draft.stage.prepare_send", ("draft.stage.recv_wait",), "prepare"),
            (2, 3, "draft.stage.prepare_send", ("draft.stage.recv_wait",), "prepare"),
            (
                1,
                2,
                "draft.stage.send",
                ("draft.stage.state_launch", "draft.stage.state_wait"),
                "state",
            ),
            (2, 3, "draft.stage.send", ("draft.stage.state_wait",), "state"),
            (3, 4, "draft.stage.send", ("draft.head.recv_wait",), "state"),
        )
        for source, target, send, receive_options, label in edges:
            sender = by_step.get((source, rank_steps[source]), {}).get(send)
            receiver = next(
                (
                    by_step.get((target, rank_steps[target]), {}).get(stage)
                    for stage in receive_options
                    if by_step.get((target, rank_steps[target]), {}).get(stage)
                ),
                None,
            )
            if sender and receiver:
                # state_launch ends after posting, not after receiving tensors.
                completion = (
                    by_step[(target, rank_steps[target])].get(
                        "draft.stage.state_wait", receiver
                    )
                    if receive_options[0] == "draft.stage.state_launch"
                    else receiver
                )
                if _interval(completion)[1] < _interval(sender)[0]:
                    continue
                aligned[f"P{source}_P{target}_{label}_send_to_recv_ms"].append(
                    _interval(completion)[1] - _interval(sender)[0]
                )
                aligned[f"P{source}_P{target}_{label}_receiver_late_ms"].append(
                    max(0.0, _interval(receiver)[0] - _interval(sender)[0])
                )

    return {
        "rows": len(rows),
        "aligned_rounds": len(common_steps),
        "head_rounds": len(head_steps),
        "alignment": "unique prepare containment in head proposal (single-host)",
        "rank_step_mapping": {
            step: {rank: mapping[rank][step] for rank in mapping}
            for step in common_steps
        },
        "aligned_host_raw": dict(aligned),
        "excluded_steps": excluded,
        "stages": stage_rows,
        "aligned_host": {
            key: _summary(values) for key, values in sorted(aligned.items())
        },
    }


def _group_stage_rows(
    grouped: dict[tuple[int, str, tuple[str, int]], list[dict[str, Any]]],
) -> dict[tuple[int, str], list[list[dict[str, Any]]]]:
    result: dict[tuple[int, str], list[list[dict[str, Any]]]] = defaultdict(list)
    for (rank, stage, _), members in grouped.items():
        result[(rank, stage)].append(members)
    return result


def write_report(result: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with (out_dir / "stages.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "pp_rank",
                "stage",
                "samples",
                "calls_per_sample",
                "host_p50_ms",
                "host_p95_ms",
                "cuda_p50_ms",
                "cuda_p95_ms",
            ]
        )
        for row in result["stages"]:
            writer.writerow(
                [
                    row["pp_rank"],
                    row["stage"],
                    row["samples"],
                    row["calls_per_sample"],
                    row["host_ms"]["p50"],
                    row["host_ms"]["p95"],
                    row["cuda_ms"]["p50"],
                    row["cuda_ms"]["p95"],
                ]
            )
    lines = [
        "# DSpark phase timing",
        "",
        "CPU is host scope time. CUDA is the same-stream event span and can "
        "include host issue gaps.",
        "Parent and child rows are not additive.",
        "",
        f"Rows: {result['rows']}; aligned rounds: {result['aligned_rounds']}",
        f"Head rounds: {result['head_rounds']}; excluded: {result['excluded_steps']}",
        f"Alignment: {result['alignment']}",
        "Stage summaries include boundary rounds; aligned intervals use only "
        "complete rounds. See analysis.json for rank-local step mappings.",
        "",
        "| PP | Stage | Samples | Host p50 ms | Host p95 ms | "
        "CUDA p50 ms | CUDA p95 ms |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["stages"]:
        lines.append(
            f"| {row['pp_rank']} | `{row['stage']}` | {row['samples']} | "
            f"{row['host_ms']['p50']:.3f} | {row['host_ms']['p95']:.3f} | "
            f"{row['cuda_ms']['p50']:.3f} | {row['cuda_ms']['p95']:.3f} |"
        )
    lines += ["", "## Aligned host intervals", ""]
    for key, value in result["aligned_host"].items():
        lines.append(
            f"- `{key}`: n={value['n']}, p50={value['p50']:.3f} ms, "
            f"p95={value['p95']:.3f} ms"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trace", nargs="+", type=Path, help="JSONL file or trace directory"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--flow", default="dspark")
    parser.add_argument("--step-kind", default="decode")
    args = parser.parse_args()
    result = analyze(load_rows(args.trace, args.flow, args.step_kind))
    write_report(result, args.out_dir)
    print(
        json.dumps({"rows": result["rows"], "aligned_rounds": result["aligned_rounds"]})
    )


if __name__ == "__main__":
    main()
