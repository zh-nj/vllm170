#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the bounded tail's per-layer row counts.

Reads the tables written by ``fast_prefill.record_layer_rows`` while the switch
is on and verifies the port's central claim, per rank and per step: every layer
below the tail processed one and the same row count (the whole chunk), and every
layer from the tail on processed one and the same, smaller row count (the bounded
suffix). A layer whose count falls outside its group means the boundary moved, a
rank disagreed, or something recomputed rows it should have skipped.

Usage:
    echo 1 > /tmp/dsv41_fastprefill_rows.step
    touch /tmp/dsv41_fastprefill_rows_on
    # send one long prompt request
    python benchmarks/deepseek_v4_1/fast_prefill_rows.py --tail 21
"""

import argparse
import glob
from collections import defaultdict

import regex as re


def read_tables(pattern):
    """rank -> occurrence -> layer -> rows.

    Every layer on a rank logs once per engine step, so the Nth line of each
    layer belongs to the same step. Grouping by occurrence rather than by the
    recorder's step id keeps a multi-chunk prompt -- whose chunks have different
    widths -- from looking like a disagreement within one step.
    """
    tables = {}
    for path in sorted(glob.glob(pattern)):
        match = re.search(r"rank(\d+)\.tsv$", path)
        rank = int(match.group(1)) if match else path
        per_layer = defaultdict(list)
        with open(path) as handle:
            for line in handle:
                fields = line.split()
                if len(fields) != 4:
                    continue
                _step, layer, rows, _tail = (int(value) for value in fields)
                per_layer[layer].append(rows)
        steps = defaultdict(dict)
        for layer, values in per_layer.items():
            for occurrence, rows in enumerate(values):
                steps[occurrence][layer] = rows
        tables[rank] = steps
    return tables


def check(tables, tail, min_chunk):
    """Return (rows, failures) for the per-rank tables.

    A rank holds layers on one side of the boundary or on both, so the check is
    per side: within a rank every layer of a side must show the same row count,
    and across ranks each side must show one row count for the whole pipeline.
    """
    report = []
    failures = []
    by_step = defaultdict(lambda: {"below": {}, "above": {}})
    for rank in sorted(tables):
        expected = set()
        for step in tables[rank].values():
            expected |= set(step)
        for step in sorted(tables[rank]):
            layers = tables[rank][step]
            missing = expected - set(layers)
            if missing:
                # A step that dies mid-forward logs only the layers it reached;
                # reporting that as a boundary violation would hide the crash.
                failures.append(
                    f"rank {rank} step {step}: incomplete step, layers "
                    f"{sorted(missing)} never logged (the forward stopped early)"
                )
                continue
            groups = {
                "below": {v for k, v in layers.items() if k < tail},
                "above": {v for k, v in layers.items() if k >= tail},
            }
            counts = {
                side: len([k for k in layers if (k >= tail) == (side == "above")])
                for side in groups
            }
            for side, values in groups.items():
                if len(values) > 1:
                    failures.append(
                        f"rank {rank} step {step}: {side}-tail layers ran "
                        f"different row counts {sorted(values)}"
                    )
                elif values:
                    by_step[step][side][rank] = next(iter(values))
            report.append(
                {
                    "rank": rank,
                    "step": step,
                    "below_layers": counts["below"],
                    "below_rows": sorted(groups["below"]),
                    "above_layers": counts["above"],
                    "above_rows": sorted(groups["above"]),
                }
            )
    for step in sorted(by_step):
        below = set(by_step[step]["below"].values())
        above = set(by_step[step]["above"].values())
        if len(below) > 1:
            failures.append(
                f"step {step}: ranks disagree below the tail: "
                f"{dict(sorted(by_step[step]['below'].items()))}"
            )
        if len(above) > 1:
            failures.append(
                f"step {step}: ranks disagree at or above the tail: "
                f"{dict(sorted(by_step[step]['above'].items()))}"
            )
        if not below or not above:
            failures.append(f"step {step}: only one side of the tail ran")
            continue
        whole, bounded = max(below), max(above)
        if whole > min_chunk and bounded >= whole:
            failures.append(
                f"step {step}: chunk of {whole} rows was not bounded "
                f"(tail layers ran {bounded})"
            )
    return report, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", default="/tmp/dsv41_fastprefill_rows.rank*.tsv")
    parser.add_argument("--tail", type=int, default=21)
    parser.add_argument(
        "--min-chunk",
        type=int,
        default=129,
        help="chunk width above which the tail must be smaller",
    )
    args = parser.parse_args()

    tables = read_tables(args.glob)
    if not tables:
        raise SystemExit(f"no recorder tables matched {args.glob}")
    report, failures = check(tables, args.tail, args.min_chunk)
    print(f"tail layer {args.tail}")
    print("| rank | step | layers < tail | rows | layers >= tail | rows |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in report:
        print(
            f"| {row['rank']} | {row['step']} | {row['below_layers']} | "
            f"{row['below_rows']} | {row['above_layers']} | {row['above_rows']} |"
        )
    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"* {failure}")
        raise SystemExit(1)
    print("\nPASS: one row count below the tail, one smaller above it, every rank")


if __name__ == "__main__":
    main()
