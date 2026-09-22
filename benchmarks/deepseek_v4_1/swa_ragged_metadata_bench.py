# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure the eager dense-to-ragged SWA metadata path used by DSV41 SM80."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from vllm.models.deepseek_v4_1.amd.rocm import _copy_ragged_to_graph_buffers
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    build_ragged_indices_from_dense,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows, args.width, args.repeats) < 1 or args.warmup < 0:
        parser.error("rows, width and repeats must be positive; warmup >= 0")

    device = torch.device("cuda:0")
    lengths = torch.tensor(
        [args.width - (row % min(args.width, 7)) for row in range(args.rows)],
        device=device,
        dtype=torch.int32,
    )
    dense = [
        torch.arange(args.rows * args.width, device=device, dtype=torch.int32)
        .reshape(args.rows, args.width)
        .add_(group * args.rows * args.width)
        for group in range(4)
    ]
    index_buffers = [
        torch.empty(args.rows * args.width, device=device, dtype=torch.int32)
        for _ in range(4)
    ]
    indptr_buffers = [
        torch.empty(args.rows + 1, device=device, dtype=torch.int32) for _ in range(4)
    ]

    def run_group(group: int) -> tuple[torch.Tensor, torch.Tensor]:
        ragged, indptr = build_ragged_indices_from_dense(dense[group], lengths)
        return _copy_ragged_to_graph_buffers(
            ragged,
            indptr,
            index_buffers[group],
            indptr_buffers[group],
            args.rows,
            args.width,
        )

    expected_indptr = [0]
    expected_values = []
    for row, length in enumerate(lengths.cpu().tolist()):
        expected_values.extend(dense[0][row, :length].cpu().tolist())
        expected_indptr.append(expected_indptr[-1] + length)
    for group in range(4):
        indices, indptr = run_group(group)
        expected = [value + group * args.rows * args.width for value in expected_values]
        assert indptr.cpu().tolist() == expected_indptr
        assert indices[: expected_indptr[-1]].cpu().tolist() == expected
        assert indices.data_ptr() == index_buffers[group].data_ptr()
        assert indptr.data_ptr() == indptr_buffers[group].data_ptr()

    def measure(groups: range) -> dict[str, object]:
        for _ in range(args.warmup):
            for group in groups:
                run_group(group)
        torch.accelerator.synchronize()
        gpu_ms = []
        host_enqueue_ms = []
        for _ in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            host_start = time.perf_counter_ns()
            start.record()
            for group in groups:
                run_group(group)
            end.record()
            host_enqueue_ms.append((time.perf_counter_ns() - host_start) / 1e6)
            end.synchronize()
            gpu_ms.append(start.elapsed_time(end))
        return {
            "gpu_event_ms_median": statistics.median(gpu_ms),
            "gpu_event_ms_p10": sorted(gpu_ms)[len(gpu_ms) // 10],
            "gpu_event_ms_p90": sorted(gpu_ms)[9 * len(gpu_ms) // 10],
            "host_enqueue_ms_median": statistics.median(host_enqueue_ms),
            "gpu_event_ms": gpu_ms,
            "host_enqueue_ms": host_enqueue_ms,
        }

    result = {
        "device": torch.cuda.get_device_name(0),
        "rows": args.rows,
        "width": args.width,
        "dtype": "int32",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timing": "CUDA events plus perf_counter_ns; CUPTI unavailable",
        "boundary": "build_ragged_indices_from_dense + graph-buffer copy",
        "correctness": "exact prefix, indptr and persistent pointer identity",
        "one_group": measure(range(1)),
        "four_groups": measure(range(4)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"one group {result['one_group']['gpu_event_ms_median']:.3f} ms GPU, "
        f"four groups {result['four_groups']['gpu_event_ms_median']:.3f} ms GPU"
    )


if __name__ == "__main__":
    main()
