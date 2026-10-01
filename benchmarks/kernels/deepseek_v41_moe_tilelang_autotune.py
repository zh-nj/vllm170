# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline candidate search for the TileLang DSV4.1-Flash SM80 grouped GEMM.

Runs entirely offline: no request path, no model forward, no CUDA-Graph
capture inside a serving process.

Why this is not ``tilelang.autotuner.AutoTuner``: the candidate writes into a
caller-preallocated ``routed_out`` (the production in-place contract) and
returns nothing, while both tuner check paths
(``profiler.assert_allclose`` and ``profiler.manual_assert_close``) compare
only the kernel's *return* values against ``ref_prog``'s, so neither can see
this output at all. The plan anticipates that case and requires per-config
strict validation outside the tuner with recorded evidence, which is what
``deepseek_v41_moe_harness.reference_stage`` provides here: a candidate is
timed only after it reproduces the independent reference bit for bit.

Every step is recorded: compile failure, correctness failure and timing
failure each get their own status, and a CUDA error aborts the run instead of
letting the search continue in a poisoned context.
"""

import argparse
import hashlib
import json
import sys
import time
import traceback
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deepseek_v41_moe_harness import (  # noqa: E402
    STAGE_SHAPES,
    W2_STAGE,
    W13_STAGE,
    environment,
    git_head,
    time_graph,
)
from deepseek_v41_moe_tilelang_bench import (  # noqa: E402
    Case,
    build_case,
    candidate_call,
    case_identity,
    first_mismatch,
    graph_validator,
    production_call,
    run_identity,
    select_production_tile,
)
from deepseek_v41_moe_tilelang_gemm import (  # noqa: E402
    GemmConfig,
    TileLangMoeGemm,
)

# ``raw`` is the deliberately wrong decode-free ablation: it exists so the
# verification gate can be shown to reject a kernel instead of timing it.
DECODE_MODES = ("mxfp4", "lut", "luta", "prmt", "raw")


def config_grid(
    *,
    stage: str,
    num_experts: int,
    topk: int,
    block_ms: Iterable[int],
    block_ns: Iterable[int],
    block_ks: Iterable[int],
    stages: Iterable[int],
    thread_counts: Iterable[int],
    decodes: Iterable[str],
    max_configs: int,
) -> list[GemmConfig]:
    """Candidate space: tile, pipeline depth, threads and decode lowering.

    The plan's first round fixes block_m, threads and the pipeline and sweeps
    only (block_n, block_k); later rounds widen one axis at a time. The grid
    here is the union, ordered so the cheapest ablations run first.
    """
    n, k, source_rows_are_slots = STAGE_SHAPES[stage]
    configs: list[GemmConfig] = []
    seen = set()
    for decode in decodes:
        for block_m in block_ms:
            for block_n in block_ns:
                for block_k in block_ks:
                    for num_stages in stages:
                        for threads in thread_counts:
                            # INT8 MMA needs k >= 32 per tile and at least one
                            # 16x8 warp tile per warp.
                            if k % block_k or n % block_n:
                                continue
                            if block_m * block_n < threads // 32 * 16 * 8:
                                continue
                            if block_m < 16 or block_n < 8:
                                continue
                            key = (
                                decode,
                                block_m,
                                block_n,
                                block_k,
                                num_stages,
                                threads,
                            )
                            if key in seen:
                                continue
                            seen.add(key)
                            configs.append(
                                GemmConfig(
                                    n=n,
                                    k=k,
                                    num_experts=num_experts,
                                    topk=topk,
                                    block_m=block_m,
                                    block_n=block_n,
                                    block_k=block_k,
                                    num_stages=num_stages,
                                    threads=threads,
                                    source_rows_are_slots=source_rows_are_slots,
                                    decode=decode,
                                )
                            )
    return configs[:max_configs]


def is_cuda_fatal(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        token in text
        for token in (
            "cuda error",
            "illegal memory access",
            "device-side assert",
            "misaligned address",
            "an illegal instruction",
            "unspecified launch failure",
        )
    )


def measure_config(
    *,
    case: Case,
    config: GemmConfig,
    reference: torch.Tensor,
    warmup: int,
    samples: int,
    graph_calls: int,
    check: bool,
) -> dict[str, Any]:
    """Compile, verify and time one candidate. Never times a wrong kernel."""
    record: dict[str, Any] = {
        "stage": case.stage,
        "config": config.label,
        "config_fields": asdict(config),
        "verification_status": "not_run",
        "measurement_status": "not_run",
        "promotable": False,
    }
    start = time.perf_counter()
    try:
        gemm = TileLangMoeGemm.create(config, tokens=case.tokens, slots=case.slot_count)
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        record["verification_status"] = "compile_failure"
        record["failure_reason"] = f"{type(exc).__name__}: {exc}"[:2000]
        record["wall_seconds"] = time.perf_counter() - start
        if is_cuda_fatal(exc):
            record["cuda_fatal"] = True
        return record
    record["generated_source_sha256"] = hashlib.sha256(gemm.source.encode()).hexdigest()
    record["input_sha256"] = case_identity(case)
    record["compile_seconds"] = round(gemm.compile_seconds, 3)
    # A fresh output buffer per config keeps the verification honest: a stale
    # buffer from an earlier config would hide a kernel that never writes.
    case.routed_out.fill_(float("nan"))
    call = candidate_call(gemm, case)
    try:
        call()
        torch.accelerator.synchronize()
    except Exception as exc:  # noqa: BLE001
        record["verification_status"] = "runtime_failure"
        record["failure_reason"] = f"{type(exc).__name__}: {exc}"[:2000]
        record["cuda_fatal"] = is_cuda_fatal(exc)
        record["wall_seconds"] = time.perf_counter() - start
        return record
    if check:
        checked = first_mismatch(case.routed_out, reference)
        record["correctness"] = checked
        record["verification_status"] = (
            "pass" if checked["bitwise_exact"] else "numerical_failure"
        )
        if not checked["bitwise_exact"]:
            record["wall_seconds"] = time.perf_counter() - start
            return record
        record["promotable"] = True
    else:
        record["verification_status"] = "skipped"
    try:
        timing = time_graph(
            call,
            warmup=warmup,
            repeats=graph_calls,
            samples=samples,
            validate_graph=graph_validator([case]) if check else None,
        )
    except Exception as exc:  # noqa: BLE001 - preserve timing failure status
        record["promotable"] = False
        record["measurement_status"] = "timing_failure"
        record["failure_reason"] = f"{type(exc).__name__}: {exc}"[:2000]
        record["cuda_fatal"] = is_cuda_fatal(exc)
        record["wall_seconds"] = time.perf_counter() - start
        return record
    record.update(timing.as_row())
    record["measurement_status"] = "ok"
    record["graph_replay_checked"] = check
    record["wall_seconds"] = time.perf_counter() - start
    return record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=[W13_STAGE, W2_STAGE, "both"], default=W13_STAGE
    )
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--route-pattern", default="uniform")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--scale-mode", choices=["varied", "unit"], default="varied")
    parser.add_argument("--block-m", type=int, nargs="+", default=[16])
    parser.add_argument("--block-n", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--block-k", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--num-stages", type=int, nargs="+", default=[2])
    parser.add_argument("--threads", type=int, nargs="+", default=[128])
    parser.add_argument(
        "--decode", nargs="+", choices=list(DECODE_MODES), default=["luta"]
    )
    parser.add_argument("--max-configs", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--graph-calls", type=int, default=20)
    parser.add_argument(
        "--check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="verify each candidate against the independent reference first",
    )
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--dump-dir", type=str, default="")
    parser.add_argument("--baseline-rounds", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 2
    capability = torch.cuda.get_device_capability(0)
    if capability != (8, 0):
        print(f"expected an SM80 device, got {capability}", file=sys.stderr)
        return 2
    device = torch.device("cuda")
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stages = [W13_STAGE, W2_STAGE] if args.stage == "both" else [args.stage]
    env = environment()
    header = {
        "schema": "tilelang-moe-autotune-v1",
        "git_head": git_head(),
        **run_identity(args),
        "env": env,
        "arguments": vars(args),
        "mode": "strict" if args.check else "diagnostic",
        "note": (
            "candidates are promotable only after deepseek_v41_moe_harness."
            "reference_stage passes bitwise"
        ),
    }
    failures: list[str] = []
    validated: list[dict[str, Any]] = []
    baselines = {}
    attempted = 0
    generated_sources = set()
    with out_path.open("w") as handle:
        handle.write(json.dumps(header, default=str) + "\n")
        try:
            for stage in stages:
                default_m, default_n = select_production_tile(
                    args.tokens, args.topk, args.experts
                )
                baseline_case = build_case(
                    stage=stage,
                    num_experts=args.experts,
                    topk=args.topk,
                    tokens=args.tokens,
                    pattern=args.route_pattern,
                    seed=args.seed,
                    scale_mode=args.scale_mode,
                    block_m=default_m,
                    block_n=default_n,
                    device=device,
                )
                if args.baseline_rounds:
                    baseline_case.routed_out.fill_(float("nan"))
                    production_call(baseline_case)()
                    torch.accelerator.synchronize()
                    baseline_check = first_mismatch(
                        baseline_case.routed_out, baseline_case.reference
                    )
                    if not baseline_check["bitwise_exact"]:
                        raise RuntimeError(
                            f"production baseline failed reference: {baseline_check}"
                        )
                    baseline_timing = time_graph(
                        production_call(baseline_case),
                        warmup=args.warmup,
                        repeats=args.graph_calls,
                        samples=args.samples * args.baseline_rounds,
                        validate_graph=graph_validator([baseline_case]),
                    )
                    baseline_row = {
                        "stage": stage,
                        "config": f"production-{default_m}x{default_n}",
                        "kind": "baseline",
                        "measurement_status": "ok",
                        "verification_status": "pass",
                        "correctness": baseline_check,
                        "graph_replay_checked": True,
                        "input_sha256": case_identity(baseline_case),
                        **baseline_timing.as_row(),
                    }
                    baselines[stage] = baseline_timing.p50_us
                    handle.write(json.dumps(baseline_row, default=str) + "\n")
                    handle.flush()
                    print(
                        f"baseline {stage} {default_m}x{default_n} "
                        f"p50={baseline_timing.p50_us:.2f}us"
                    )
                configs = config_grid(
                    stage=stage,
                    num_experts=args.experts,
                    topk=args.topk,
                    block_ms=args.block_m,
                    block_ns=args.block_n,
                    block_ks=args.block_k,
                    stages=args.num_stages,
                    thread_counts=args.threads,
                    decodes=args.decode,
                    max_configs=args.max_configs,
                )
                print(f"{stage}: {len(configs)} candidate configs")
                cases = {
                    block_m: build_case(
                        stage=stage,
                        num_experts=args.experts,
                        topk=args.topk,
                        tokens=args.tokens,
                        pattern=args.route_pattern,
                        seed=args.seed,
                        block_m=block_m,
                        block_n=default_n,
                        device=device,
                        weights=baseline_case.weights,
                        routing_ids=baseline_case.routing_ids,
                        a_q=baseline_case.a_q,
                        a_scale=baseline_case.a_scale,
                    )
                    for block_m in sorted({config.block_m for config in configs})
                }
                source_configs = {}
                for config in configs:
                    attempted += 1
                    case = cases[config.block_m]
                    try:
                        record = measure_config(
                            case=case,
                            config=config,
                            reference=case.reference,
                            warmup=args.warmup,
                            samples=args.samples,
                            graph_calls=args.graph_calls,
                            check=args.check,
                        )
                    except Exception as exc:  # noqa: BLE001
                        record = {
                            "stage": stage,
                            "config": config.label,
                            "verification_status": "harness_failure",
                            "measurement_status": "harness_failure",
                            "promotable": False,
                            "failure_reason": (
                                (
                                    f"{type(exc).__name__}: {exc}\n"
                                    f"{traceback.format_exc()}"
                                )[:4000]
                            ),
                        }
                        record["cuda_fatal"] = is_cuda_fatal(exc)
                    source_hash = record.get("generated_source_sha256")
                    if source_hash:
                        generated_sources.add(source_hash)
                        record["duplicate_source_of"] = source_configs.get(source_hash)
                        source_configs.setdefault(source_hash, config.label)
                    handle.write(json.dumps(record, default=str) + "\n")
                    handle.flush()
                    if record.get("cuda_fatal"):
                        print(
                            f"CUDA-fatal failure at {config.label}; stopping",
                            file=sys.stderr,
                        )
                        raise RuntimeError(record.get("failure_reason", "CUDA failure"))
                    if record["measurement_status"] == "ok" and record.get(
                        "promotable", False
                    ):
                        validated.append(record)
                        if args.dump_dir:
                            dump = Path(args.dump_dir)
                            dump.mkdir(parents=True, exist_ok=True)
                            (dump / f"{config.label}.cu").write_text(
                                TileLangMoeGemm.create(
                                    config, tokens=case.tokens, slots=case.slot_count
                                ).source
                            )
                        print(
                            f"  ok  {config.label:34s} p50={record['p50_us']:9.2f}us "
                            f"p90={record['p90_us']:9.2f}us"
                        )
                    else:
                        reason = record.get(
                            "failure_reason", record["verification_status"]
                        )
                        failures.append(f"{config.label}: {reason}")
                        print(f"  FAIL {config.label:34s} {reason}"[:200])
        except Exception as exc:  # noqa: BLE001 - summary survives fatal failures
            reason = f"{type(exc).__name__}: {exc}"
            failures.append(reason)
            handle.write(
                json.dumps(
                    {
                        "kind": "run_failure",
                        "failure_reason": reason,
                        "promotable": False,
                    }
                )
                + "\n"
            )
        summary = {
            "schema": "tilelang-moe-autotune-summary-v1",
            "attempted": attempted,
            "unique_generated_sources": len(generated_sources),
            "candidates": len(validated),
            "by_stage": {
                stage: {
                    "baseline_p50_us": baselines.get(stage),
                    "ranked": sorted(
                        (
                            {
                                "config": r["config"],
                                "p50_us": r["p50_us"],
                                "baseline_ratio": r["p50_us"] / baselines[stage]
                                if stage in baselines
                                else None,
                            }
                            for r in validated
                            if r["stage"] == stage
                        ),
                        key=lambda r: r["p50_us"],
                    ),
                }
                for stage in stages
            },
            "failures": len(failures),
            "failure_list": failures,
            "validated": sorted(
                (
                    {
                        "stage": row["stage"],
                        "config": row["config"],
                        "p50_us": row["p50_us"],
                        "p90_us": row["p90_us"],
                    }
                    for row in validated
                ),
                key=lambda row: (row["stage"], row["p50_us"]),
            ),
            "all_passed": bool(args.check) and bool(validated) and not failures,
            "mode": "strict" if args.check else "diagnostic",
        }
        handle.write(json.dumps({"kind": "summary", **summary}, default=str) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "validated"}, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
