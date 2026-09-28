#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize the streaming step-timing lines a PP5 server logs.

The phase profiler synchronizes per record, which serializes the pipeline, so it
can attribute cost but not time it. ``VLLM_DEEPSEEK_V4_STEP_TIMING_EVERY`` logs a
per-worker split of target forward, draft, preparation and the residual step
period instead, with one event resolution per N steps and no control file. This
module turns those log lines into the serving-time budget.

Windows that contain prefill steps are excluded by default: a prefill step's
target forward is an order of magnitude longer than a decode one, and a window
mixing the two would report neither.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import regex as re

_HEAD = re.compile(
    r"\(Worker_PP(?P<rank>\d+)[^)]*\).*?"
    r"step timing over (?P<steps>\d+) steps: (?P<body>.*)$"
)
# A top-level "name p50 value ms" field and the parenthesised detail that
# belongs to it, e.g. "target p50 7.82 ms (min 7.45)".
_ITEM = re.compile(
    r"(?P<name>[a-z_]+) p50 (?P<value>-?[\d.]+) ms"
    r"(?:\s*\((?P<inner>[^)]*)\))?"
)
# Bare "name value" pairs inside a parenthesised group, e.g. "attn_meta 5.64".
_INNER = re.compile(r"(?P<name>[a-z_]+) (?P<value>-?[\d.]+)")
# Fields the budget is built from; anything else is still recorded.
BUDGET_FIELDS = (
    "target",
    "target_min",
    "draft",
    "prep",
    "attn_meta",
    "inputs",
    "period",
    "other",
)


def parse_step_timing(text: str) -> list[dict[str, Any]]:
    """Return one record per logged window, in log order."""
    windows: list[dict[str, Any]] = []
    for line in text.splitlines():
        head = _HEAD.search(line)
        if head is None:
            continue
        row: dict[str, Any] = {
            "rank": int(head.group("rank")),
            "steps": int(head.group("steps")),
        }
        for match in _ITEM.finditer(head.group("body")):
            name = match.group("name")
            row[name] = float(match.group("value"))
            inner = match.group("inner")
            if inner is None:
                continue
            for item in _INNER.finditer(inner):
                key = item.group("name")
                # "min" qualifies the field it follows; every other name is a
                # field of its own (attn_meta, inputs, other).
                row[f"{name}_min" if key == "min" else key] = float(item.group("value"))
        if "target" in row:
            windows.append(row)
    return windows


def decode_windows(
    windows: list[dict[str, Any]], *, target_max_ms: float
) -> list[dict[str, Any]]:
    """Keep windows whose every step looks like decode, not prefill."""
    return [
        row
        for row in windows
        if row["target"] <= target_max_ms
        and row.get("target_min", row["target"]) <= target_max_ms
    ]


def _p50(values: list[float]) -> float:
    return statistics.median(values) if values else math.nan


def summarize(windows: list[dict[str, Any]], *, target_max_ms: float) -> dict[str, Any]:
    kept = decode_windows(windows, target_max_ms=target_max_ms)
    by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in kept:
        by_rank[row["rank"]].append(row)
    fields = [
        name
        for name in (*BUDGET_FIELDS, "combined")
        if any(name in row for row in kept)
    ]
    ranks = {
        str(rank): {
            "windows": len(rows),
            **{
                name: _p50([row[name] for row in rows if name in row])
                for name in fields
            },
        }
        for rank, rows in sorted(by_rank.items())
    }
    return {
        "windows_total": len(windows),
        "windows_decode": len(kept),
        "target_max_ms": target_max_ms,
        "fields": fields,
        "ranks": ranks,
        "decode_target_sum_ms": sum(
            ranks[rank]["target"]
            for rank in ranks
            if not math.isnan(ranks[rank]["target"])
        ),
    }


def _table(result: dict[str, Any]) -> list[str]:
    fields = result["fields"]
    lines = [
        "| PP | windows | " + " | ".join(fields) + " |",
        "|---:|---:|" + "---:|" * len(fields),
    ]
    for rank, row in result["ranks"].items():
        cells = " | ".join(
            "n/a" if math.isnan(row[name]) else f"{row[name]:.2f}" for name in fields
        )
        lines.append(f"| {rank} | {row['windows']} | {cells} |")
    return lines


def write_report(result: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "step_timing.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Streaming step timing (no phase profiler)",
        "",
        "Median over decode-only windows, per worker. `prep` is "
        "`prepare_start -> forward_start`, `period` is consecutive "
        "`forward_start` deltas and `other = period - target - draft - prep`.",
        "Parent and child values are not additive; each rank reports its own "
        "slice of the pipeline, and PP0 has no draft role.",
        "",
        f"Windows kept: {result['windows_decode']} of {result['windows_total']} "
        f"(target p50 and min at or below {result['target_max_ms']} ms)",
        "",
        *_table(result),
        "",
        "Sum of the per-rank target medians (a serial-chain upper bound, not a "
        f"measured interval): {result['decode_target_sum_ms']:.2f} ms",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="server log with step timing lines")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--target-max-ms",
        type=float,
        default=50.0,
        help="drop windows whose target p50 or min exceeds this (prefill)",
    )
    args = parser.parse_args()
    windows = parse_step_timing(args.log.read_text(errors="replace"))
    if not windows:
        raise SystemExit(
            "no step timing lines; start the server with "
            "VLLM_DEEPSEEK_V4_STEP_TIMING_EVERY>0"
        )
    result = summarize(windows, target_max_ms=args.target_max_ms)
    result["windows"] = windows
    write_report(result, args.out_dir)
    print(json.dumps({"windows_decode": result["windows_decode"]}))


if __name__ == "__main__":
    main()
