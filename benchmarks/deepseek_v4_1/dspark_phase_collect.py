#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collect unprofiled DSpark delivery metrics, then one opt-in phase request.

Run with -m from the repository root. The server must already be started with
phase-profile environment variables if --control is provided.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

from benchmarks.deepseek_v4_1.decode_gate import make_token_ids
from benchmarks.deepseek_v4_1.dspark_flow_gate import (
    metric_deltas,
    read_engine_metrics,
    run_once,
)
from benchmarks.deepseek_v4_1.dspark_phase_measure import (
    analyze,
    load_rows,
    write_report,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:26662")
    parser.add_argument("--model", default="dsv41f-port")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--prompt-ids", type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=8400)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--trace-dir", type=Path)
    args = parser.parse_args()
    if bool(args.control) != bool(args.trace_dir):
        parser.error("--control and --trace-dir must be supplied together")
    if min(args.repeats, args.prompt_tokens) < 1 or args.max_tokens < 2:
        parser.error("positive repeats/prompt and at least 2 output tokens required")
    if args.control and args.control.exists():
        parser.error("control already exists: profile must be off for warmup/baseline")
    if args.trace_dir and list(args.trace_dir.glob("*.jsonl")):
        parser.error("trace directory is not empty; use a fresh server trace path")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    prompt = (
        json.loads(args.prompt_ids.read_text())
        if args.prompt_ids
        else make_token_ids(args.base_url, args.prompt_tokens, args.seed)
    )
    if (
        not isinstance(prompt, list)
        or len(prompt) != args.prompt_tokens
        or not all(type(token) is int and token >= 0 for token in prompt)
    ):
        parser.error("prompt must be a list of prompt-tokens nonnegative token IDs")
    (args.out_dir / "prompt_ids.json").write_text(json.dumps(prompt) + "\n")
    result = dict(
        settings={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], text=True),
        prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
        baseline=[],
    )

    def save():
        (args.out_dir / "request.json").write_text(json.dumps(result, indent=2) + "\n")

    def request(count):
        before = read_engine_metrics(args.base_url, args.timeout)
        row = run_once(args.base_url, args.model, prompt, count, args.timeout)
        after = read_engine_metrics(args.base_url, args.timeout)
        row["metrics"] = metric_deltas(before, after)
        spec_metrics = {k: v for k, v in row["metrics"].items() if "spec_decode_" in k}
        row["spec_metrics"] = spec_metrics
        rounds = sum(
            v
            for k, v in spec_metrics.items()
            if k.split("{")[0] == "vllm:spec_decode_num_drafts_total"
        )
        if rounds <= 0:
            raise ValueError("no DSpark draft rounds; check service/counter isolation")
        row["decode_ms_per_round"] = row["post_first_chunk_s"] * 1000 / rounds
        return row

    # Warmup also establishes all counter labels before taking metric deltas.
    run_once(args.base_url, args.model, prompt, 64, args.timeout)
    for index in range(args.repeats):
        row = request(args.max_tokens)
        result["baseline"].append(row)
        save()
        print(
            f"baseline {index}: {row['post_first_chunk_tok_per_s']:.3f} tok/s",
            flush=True,
        )
    reference = result["baseline"][0]
    result["baseline_ids_equal"] = all(
        r["token_ids"] == reference["token_ids"] for r in result["baseline"]
    )
    result["baseline_acceptance_equal"] = all(
        r["spec_metrics"] == reference["spec_metrics"] for r in result["baseline"]
    )
    result["median_decode_tps"] = statistics.median(
        r["post_first_chunk_tok_per_s"] for r in result["baseline"]
    )
    save()
    if not result["baseline_ids_equal"] or not result["baseline_acceptance_equal"]:
        raise ValueError(
            "baseline output/acceptance varies; inspect request.json first"
        )
    if args.control:
        args.control.parent.mkdir(parents=True, exist_ok=True)
        with args.control.open("x"):
            pass
        try:
            result["profile"] = request(args.max_tokens)
        finally:
            args.control.unlink(missing_ok=True)
        result["profile_ids_equal"] = (
            result["profile"]["token_ids"] == reference["token_ids"]
        )
        result["profile_acceptance_equal"] = (
            result["profile"]["spec_metrics"] == reference["spec_metrics"]
        )
        save()
        if not result["profile_ids_equal"] or not result["profile_acceptance_equal"]:
            raise ValueError("profile changed output/acceptance; inspect request.json")
        summary = analyze(load_rows([args.trace_dir], "dspark", "decode"))
        write_report(summary, args.out_dir / "phase")
        if not summary["aligned_rounds"]:
            raise ValueError("no aligned PP5 rounds; inspect phase report exclusions")
    print(args.out_dir / "request.json", flush=True)


if __name__ == "__main__":
    main()
