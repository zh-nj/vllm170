# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
import sys
from pathlib import Path

PROFILE_VALIDATE = (
    Path(__file__).parents[2] / "benchmarks" / "deepseek_v4" / "profile_validate.py"
)
SPEC = importlib.util.spec_from_file_location("dsv4_profile_validate", PROFILE_VALIDATE)
assert SPEC is not None and SPEC.loader is not None
profile_validate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = profile_validate
SPEC.loader.exec_module(profile_validate)


def _make_run(tmp_path: Path, server_log: str, cuda_table: bool) -> Path:
    run_dir = tmp_path / "run"
    trace_dir = run_dir / "torch_trace"
    trace_dir.mkdir(parents=True)
    result = {
        "pp_size": 3,
        "completion_tokens": 16,
        "max_tokens": 16,
        "decode_s": 1.0,
        "decode_tokens_per_s": 15.0,
        "ttft_s": 0.5,
    }
    (run_dir / "result.jsonl").write_text(json.dumps(result) + "\n")
    (run_dir / "server.log").write_text(server_log)
    (run_dir / "gpu_memory.csv").write_text(
        "timestamp,index,memory_used_mib,memory_total_mib,utilization_gpu_pct,"
        "profile_active\n"
        "2026-08-14 00:00:00,1,60000,65024,90,0\n"
        "2026-08-14 00:00:01,1,61000,65024,95,1\n"
    )
    for rank in range(3):
        (trace_dir / f"dp0_pp{rank}_tp0_rank{rank}.pt.trace.json.gz").touch()
        header = "Self CPU CPU total"
        if cuda_table:
            header += " Self CUDA CUDA total"
        (trace_dir / f"profiler_out_{rank}.txt").write_text(header)
    return run_dir


def test_inspect_profile_run_accepts_complete_cuda_artifacts(tmp_path: Path) -> None:
    status = profile_validate.inspect_profile_run(
        _make_run(tmp_path, server_log="", cuda_table=True)
    )

    assert status["workload_accepted"]
    assert status["workload_count"] == 1
    assert status["pp_artifacts_complete"]
    assert status["kernel_profile_accepted"]
    assert status["profile_scope"] == "operator_and_cuda_kernel"
    assert status["gpu_memory_sample_count"] == 2
    assert status["peak_gpu_memory_mib"] == {"1": 61000.0}
    assert status["peak_profile_gpu_memory_mib"] == {"1": 61000.0}


def test_inspect_profile_run_marks_cmp_cupti_gap(tmp_path: Path) -> None:
    status = profile_validate.inspect_profile_run(
        _make_run(
            tmp_path,
            server_log="CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED",
            cuda_table=False,
        )
    )

    assert status["workload_accepted"]
    assert status["pp_artifacts_complete"]
    assert not status["kernel_profile_accepted"]
    assert status["profile_scope"] == "operator_only"


def test_inspect_profile_run_labels_clean_benchmark_scope(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, server_log="", cuda_table=False)
    result_path = run_dir / "result.jsonl"
    result = json.loads(result_path.read_text())
    result["profiler_enabled"] = False
    result_path.write_text(json.dumps(result) + "\n")

    status = profile_validate.inspect_profile_run(run_dir)

    assert not status["profiler_enabled"]
    assert status["profile_scope"] == "workload_only"
    assert status["peak_workload_gpu_memory_mib"] == {"1": 61000.0}
    assert "disabled" in status["limitation"]


def test_inspect_profile_run_requires_every_repeated_workload(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, server_log="", cuda_table=True)
    failed = {
        "pp_size": 3,
        "completion_tokens": 15,
        "max_tokens": 16,
        "decode_s": 1.0,
        "decode_tokens_per_s": 15.0,
        "ttft_s": 0.5,
    }
    with (run_dir / "result.jsonl").open("a") as output:
        output.write(json.dumps(failed) + "\n")

    status = profile_validate.inspect_profile_run(run_dir)

    assert status["workload_count"] == 2
    assert not status["workload_accepted"]


def test_inspect_profile_run_exposes_three_flow_event_status(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, server_log="", cuda_table=False)
    flow_summary = {
        "flows": {
            "prefill": {
                "record_count": 4,
                "pp_ranks": [0, 1, 2],
                "detail_complete": True,
            },
            "decode": {
                "record_count": 9,
                "pp_ranks": [0, 1, 2],
                "detail_complete": True,
            },
            "dspark": {
                "record_count": 3,
                "pp_ranks": [2],
                "detail_complete": True,
            },
        }
    }
    (run_dir / "phase_summary.json").write_text(json.dumps(flow_summary))

    status = profile_validate.inspect_profile_run(run_dir)

    assert status["phase_timing_accepted"]
    assert status["phase_detail_complete"]
    assert status["phase_timing_scope"] == "cuda_event"
    assert status["phase_flows"]["decode"] == {
        "record_count": 9,
        "pp_ranks": [0, 1, 2],
        "detail_complete": True,
    }
