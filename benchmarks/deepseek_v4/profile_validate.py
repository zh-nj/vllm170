# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate DeepSeek-V4 PP profiling artifacts without overstating CUDA data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import regex as re

_PP_RANK = re.compile(r"_pp(?P<rank>\d+)_")
_CUPTI_UNAVAILABLE = (
    "CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED",
    "CUPTI initialization failed - CUDA profiler activities will be missing",
)


def _jsonl_records(path: Path) -> list[dict[str, Any]]:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"no records in {path}")
    return [json.loads(line) for line in lines]


def _gpu_memory_summary(
    path: Path,
) -> tuple[int, dict[str, float], dict[str, float]]:
    if not path.exists():
        return 0, {}, {}
    rows = list(csv.DictReader(path.read_text().splitlines()))
    peak_by_gpu: dict[str, float] = {}
    peak_profile_by_gpu: dict[str, float] = {}
    for row in rows:
        gpu = row["index"].strip()
        used_mib = float(row["memory_used_mib"])
        peak_by_gpu[gpu] = max(peak_by_gpu.get(gpu, 0.0), used_mib)
        if row.get("profile_active", "0").strip() == "1":
            peak_profile_by_gpu[gpu] = max(peak_profile_by_gpu.get(gpu, 0.0), used_mib)
    return len(rows), peak_by_gpu, peak_profile_by_gpu


def inspect_profile_run(run_dir: Path) -> dict[str, Any]:
    results = _jsonl_records(run_dir / "result.jsonl")
    result = results[-1]
    trace_dir = run_dir / "torch_trace"
    traces = sorted(trace_dir.glob("*.pt.trace.json.gz"))
    tables = sorted(trace_dir.glob("profiler_out_*.txt"))
    expected_ranks = int(result["pp_size"])
    observed_ranks = sorted(
        {
            int(match.group("rank"))
            for path in traces
            if (match := _PP_RANK.search(path.name)) is not None
        }
    )
    expected_rank_ids = list(range(expected_ranks))
    artifacts_complete = (
        len(traces) == expected_ranks
        and len(tables) == expected_ranks
        and observed_ranks == expected_rank_ids
    )
    workload_accepted = all(
        row["completion_tokens"] == row["max_tokens"]
        and row["decode_s"] > 0
        and row["decode_tokens_per_s"] > 0
        and row["ttft_s"] > 0
        for row in results
    )
    profiler_enabled = all(row.get("profiler_enabled", True) for row in results)

    server_log = (run_dir / "server.log").read_text(errors="replace")
    cupti_unavailable = any(message in server_log for message in _CUPTI_UNAVAILABLE)
    cuda_tables = sum(
        "Self CUDA" in table.read_text(errors="replace")
        or "CUDA total" in table.read_text(errors="replace")
        for table in tables
    )
    cuda_activity_available = (
        artifacts_complete and not cupti_unavailable and cuda_tables == expected_ranks
    )
    if not profiler_enabled:
        profile_scope = "workload_only"
        limitation = "Profiler execution is disabled for the clean benchmark"
    elif cuda_activity_available:
        profile_scope = "operator_and_cuda_kernel"
        limitation = None
    elif cupti_unavailable:
        profile_scope = "operator_only"
        limitation = "CUPTI reports CMP device profiling is unsupported"
    else:
        profile_scope = "operator_only"
        limitation = "CUDA activity columns are absent from one or more PP ranks"

    phase_summary_path = run_dir / "phase_summary.json"
    phase_flows: dict[str, Any] = {}
    if phase_summary_path.exists():
        phase_flows = json.loads(phase_summary_path.read_text()).get("flows", {})
    phase_timing_accepted = bool(phase_flows) and all(
        phase_flows.get(flow, {}).get("record_count", 0) > 0
        for flow in ("prefill", "decode", "dspark")
    )
    phase_detail_complete = phase_timing_accepted and all(
        phase_flows[flow].get("detail_complete", False)
        for flow in ("prefill", "decode", "dspark")
    )
    phase_flow_summary = {
        flow: {
            "record_count": phase_flows.get(flow, {}).get("record_count", 0),
            "pp_ranks": phase_flows.get(flow, {}).get("pp_ranks", []),
            "detail_complete": phase_flows.get(flow, {}).get("detail_complete", False),
        }
        for flow in ("prefill", "decode", "dspark")
        if flow in phase_flows
    }
    (
        gpu_memory_sample_count,
        peak_gpu_memory_mib,
        peak_profile_gpu_memory_mib,
    ) = _gpu_memory_summary(run_dir / "gpu_memory.csv")

    return {
        "schema_version": 1,
        "workload_count": len(results),
        "workload_accepted": workload_accepted,
        "pp_artifacts_complete": artifacts_complete,
        "kernel_profile_accepted": (
            workload_accepted and artifacts_complete and cuda_activity_available
        ),
        "phase_timing_accepted": workload_accepted and phase_timing_accepted,
        "phase_detail_complete": phase_detail_complete,
        "phase_timing_scope": "cuda_event" if phase_timing_accepted else None,
        "phase_flows": phase_flow_summary,
        "profiler_enabled": profiler_enabled,
        "profile_scope": profile_scope,
        "limitation": limitation,
        "expected_pp_ranks": expected_rank_ids,
        "observed_pp_ranks": observed_ranks,
        "trace_count": len(traces),
        "table_count": len(tables),
        "cuda_table_count": cuda_tables,
        "cupti_unavailable": cupti_unavailable,
        "gpu_memory_sample_count": gpu_memory_sample_count,
        "peak_gpu_memory_mib": peak_gpu_memory_mib,
        "peak_workload_gpu_memory_mib": peak_profile_gpu_memory_mib,
        "peak_profile_gpu_memory_mib": peak_profile_gpu_memory_mib,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--workload-only", action="store_true")
    args = parser.parse_args()

    status = inspect_profile_run(args.run_dir)
    (args.run_dir / "profile_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    if not status["workload_accepted"]:
        raise SystemExit(2)
    if not args.workload_only and not status["pp_artifacts_complete"]:
        raise SystemExit(2)
    if (args.run_dir / "phase_summary.json").exists() and not status[
        "phase_timing_accepted"
    ]:
        raise SystemExit(4)
    if args.require_cuda and not status["kernel_profile_accepted"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
