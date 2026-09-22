#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B report for the bounded decoder SWA replay (``dsv41_fast_prefill``).

Reads two candidate directories produced by the gates in this directory and
reports, side by side:

* prefill latency and rate per prompt length (``prefill_gate.py``);
* DSpark acceptance, accepted length and per-token time per seed
  (``dspark_gate.py``);
* greedy token divergence and logit movement between the candidates
  (``budget_gate.py``), including the first position that differs;
* every scalar the reference prompt set reports (``prefix_cache_gate.py``).

The switch is an approximation, so a token divergence is a *result to report*,
not a failure: the tool exits non-zero only for a missing measurement or for an
acceptance/latency regression beyond the thresholds passed on the command line.
"""

import argparse
import glob
import json
import os
import statistics


def _load(directory, name):
    path = os.path.join(directory, name)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def _seed_files(directory):
    found = {}
    for path in sorted(glob.glob(os.path.join(directory, "dspark_s*.json"))):
        seed = os.path.basename(path)[len("dspark_s") : -len(".json")]
        with open(path) as handle:
            found[seed] = json.load(handle)
    return found


def prefill_table(a, b):
    if a is None or b is None:
        return []
    left = {row["length"]: row for row in a["rows"]}
    right = {row["length"]: row for row in b["rows"]}
    rows = []
    for length in sorted(set(left) & set(right)):
        a_row, b_row = left[length], right[length]
        rows.append(
            {
                "length": length,
                "a_ttft_s": a_row["ttft_median_s"],
                "b_ttft_s": b_row["ttft_median_s"],
                "speedup": a_row["ttft_median_s"] / b_row["ttft_median_s"],
                "a_tokens_per_s": a_row["prefill_tokens_per_s"],
                "b_tokens_per_s": b_row["prefill_tokens_per_s"],
                "a_spread_s": max(a_row["ttft_s"]) - min(a_row["ttft_s"]),
                "b_spread_s": max(b_row["ttft_s"]) - min(b_row["ttft_s"]),
                "a_cache_hits": a_row.get("cache_hit_tokens", 0.0),
                "b_cache_hits": b_row.get("cache_hit_tokens", 0.0),
            }
        )
    return rows


def dspark_table(a_dir, b_dir):
    a_files, b_files = _seed_files(a_dir), _seed_files(b_dir)
    rows = []
    for seed in sorted(set(a_files) & set(b_files)):
        a_row, b_row = a_files[seed], b_files[seed]
        rows.append(
            {
                "seed": seed,
                "a_acceptance": a_row["acceptance_rate"] * 100,
                "b_acceptance": b_row["acceptance_rate"] * 100,
                "acceptance_delta": (
                    b_row["acceptance_rate"] - a_row["acceptance_rate"]
                )
                * 100,
                "a_accepted_length": a_row["accepted_length"],
                "b_accepted_length": b_row["accepted_length"],
                "a_ms_per_token": a_row["ms_per_token_mean"],
                "b_ms_per_token": b_row["ms_per_token_mean"],
                "a_ttft_s": a_row["ttft_s_mean"],
                "b_ttft_s": b_row["ttft_s_mean"],
                "a_drafts": a_row["drafts"],
                "b_drafts": b_row["drafts"],
                "a_accepted_per_pos": a_row["accepted_per_pos"],
                "b_accepted_per_pos": b_row["accepted_per_pos"],
            }
        )
    return rows


def _top1(entry):
    if not entry:
        return None
    first = entry[0]
    if isinstance(first, (list, tuple)):
        return int(first[0])
    return int(first)


def _logits(entry):
    values = []
    for item in entry or ():
        if isinstance(item, (list, tuple)) and len(item) > 1:
            try:
                values.append(float.fromhex(item[1]))
            except ValueError:
                return None
    return values


def text_table(a, b):
    if a is None or b is None:
        return []
    left = {case["length"]: case for case in a["cases"]}
    right = {case["length"]: case for case in b["cases"]}
    rows = []
    for length in sorted(set(left) & set(right)):
        a_case, b_case = left[length], right[length]
        a_ids = a_case["greedy"]["ids"]
        b_ids = b_case["greedy"]["ids"]
        divergence = next(
            (
                index
                for index, pair in enumerate(zip(a_ids, b_ids))
                if pair[0] != pair[1]
            ),
            None,
        )
        shared = min(len(a_ids), len(b_ids))
        moved = 0
        worst = 0.0
        for index in range(shared):
            a_logits = _logits(a_case["prompt_topk"][index])
            b_logits = _logits(b_case["prompt_topk"][index])
            if a_logits is None or b_logits is None:
                continue
            for left_value, right_value in zip(a_logits, b_logits):
                delta = abs(left_value - right_value)
                if delta:
                    moved += 1
                    worst = max(worst, delta)
            if divergence is not None and index > divergence:
                break
        rows.append(
            {
                "length": length,
                "greedy_matches": a_ids == b_ids,
                "first_divergence": divergence,
                "compared_tokens": shared,
                "positions_with_logit_movement": moved,
                "max_logit_delta": worst,
            }
        )
    return rows


def reference_diff(a, b):
    if a is None or b is None:
        return []
    left = {case["case"]: case for case in a["cases"]}
    right = {case["case"]: case for case in b["cases"]}
    changes = []
    for name in sorted(set(left) & set(right)):
        for key, value in left[name].items():
            other = right[name].get(key)
            if isinstance(value, (int, float, bool)) and value != other:
                changes.append({"case": name, "field": key, "a": value, "b": other})
    return changes


def _markdown(tables, thresholds):
    prefill, dspark, text, reference = tables
    out = ["# dsv41_fast_prefill A/B", ""]
    if prefill:
        out += [
            "| prompt | A TTFT s | B TTFT s | speedup | A tok/s | B tok/s |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in prefill:
            out.append(
                f"| {row['length']} | {row['a_ttft_s']:.3f} | "
                f"{row['b_ttft_s']:.3f} | {row['speedup']:.3f}x | "
                f"{row['a_tokens_per_s']:.0f} | {row['b_tokens_per_s']:.0f} |"
            )
        median = statistics.median(row["speedup"] for row in prefill)
        out += ["", f"median speedup {median:.3f}x", ""]
    if dspark:
        out += [
            "| seed | A accept | B accept | delta | A len | B len | "
            "A ms/tok | B ms/tok |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in dspark:
            out.append(
                f"| {row['seed']} | {row['a_acceptance']:.2f}% | "
                f"{row['b_acceptance']:.2f}% | {row['acceptance_delta']:+.2f} | "
                f"{row['a_accepted_length']:.3f} | {row['b_accepted_length']:.3f} | "
                f"{row['a_ms_per_token']:.1f} | {row['b_ms_per_token']:.1f} |"
            )
        out.append("")
    if text:
        out += [
            "| prompt | greedy same | first divergence | positions moved | "
            "max logit delta |",
            "| ---: | :---: | ---: | ---: | ---: |",
        ]
        for row in text:
            divergence = (
                "-" if row["first_divergence"] is None else row["first_divergence"]
            )
            out.append(
                f"| {row['length']} | {'yes' if row['greedy_matches'] else 'no'} | "
                f"{divergence} | {row['positions_with_logit_movement']} | "
                f"{row['max_logit_delta']:.3e} |"
            )
        out.append("")
    if reference:
        out.append("reference prompt scalars that moved:")
        out.append("")
        for change in reference:
            out.append(
                f"* `{change['case']}`: {change['field']} "
                f"{change['a']} -> {change['b']}"
            )
        out.append("")
    failures = []
    worst_acceptance = min((row["acceptance_delta"] for row in dspark), default=0.0)
    if dspark and worst_acceptance < -thresholds["max_acceptance_drop"]:
        failures.append(f"acceptance dropped {worst_acceptance:.2f} points")
    median = statistics.median(row["speedup"] for row in prefill) if prefill else 0.0
    if prefill and median < thresholds["min_speedup"]:
        failures.append(f"median prefill speedup {median:.3f}x below threshold")
    out.append("verdict: " + ("; ".join(failures) if failures else "within thresholds"))
    return "\n".join(out), failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", help="candidate directory with the switch off")
    parser.add_argument("b", help="candidate directory with the switch on")
    parser.add_argument("--min-speedup", type=float, default=1.0)
    parser.add_argument("--max-acceptance-drop", type=float, default=2.0)
    parser.add_argument("--out")
    args = parser.parse_args()

    thresholds = {
        "min_speedup": args.min_speedup,
        "max_acceptance_drop": args.max_acceptance_drop,
    }
    tables = (
        prefill_table(_load(args.a, "prefill.json"), _load(args.b, "prefill.json")),
        dspark_table(args.a, args.b),
        text_table(_load(args.a, "budget.json"), _load(args.b, "budget.json")),
        reference_diff(
            _load(args.a, "reference.json"), _load(args.b, "reference.json")
        ),
    )
    report, failures = _markdown(tables, thresholds)
    print(report)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(
                {
                    "prefill": tables[0],
                    "dspark": tables[1],
                    "text": tables[2],
                    "reference_changes": tables[3],
                    "thresholds": thresholds,
                    "failures": failures,
                },
                handle,
                indent=2,
            )
        print(f"\nwrote {args.out}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
