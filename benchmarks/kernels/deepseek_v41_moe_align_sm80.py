# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe single-CTA q6/E384 route alignment against the production CUDA op."""

import argparse
import json
import statistics
import warnings
from pathlib import Path

import torch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _align_q6_e384(ids_ptr, sorted_ptr, experts_ptr, padded_ptr):
    slot = tl.arange(0, 64)
    expert = tl.load(ids_ptr + slot, slot < 36, other=-1).to(tl.int32)
    valid = (slot < 36) & (expert >= 0) & (expert < 384)
    equal = (expert[:, None] == expert[None, :]) & valid[:, None] & valid[None, :]
    before = slot[None, :] < slot[:, None]
    rank = tl.sum((equal & before).to(tl.int32), 1)
    count = tl.sum(equal.to(tl.int32), 1)
    first = valid & (rank == 0)
    blocks = (count + 15) // 16
    preceding = (expert[None, :] < expert[:, None]) & first[None, :]
    offset = tl.sum(tl.where(preceding, blocks[None, :], 0), 1)
    total = tl.sum(tl.where(first, blocks * 16, 0), 0)
    index = tl.arange(0, 1024)
    tl.store(sorted_ptr + index, 36, index < 576)
    tl.store(experts_ptr + slot, -1, slot < 36)
    tl.debug_barrier()
    tl.store(sorted_ptr + offset * 16 + rank, slot, valid)
    tl.store(experts_ptr + offset + rank // 16, expert, valid & (rank % 16 == 0))
    tl.store(padded_ptr, total)


def candidate(ids, outputs, warps):
    return _align_q6_e384[(1,)](ids, *outputs, num_warps=warps)


def canonical(outputs):
    sorted_ids, experts, padded = (tensor.cpu() for tensor in outputs)
    length = padded.item()
    groups = []
    for tile in range(length // 16):
        groups.append(
            (
                experts[tile].item(),
                sorted(sorted_ids[tile * 16 : (tile + 1) * 16].tolist()),
            )
        )
    return length, groups


def measure(call):
    from flashinfer.testing import bench_gpu_time_with_cupti

    with warnings.catch_warnings(record=True) as caught:
        values = bench_gpu_time_with_cupti(
            call,
            dry_run_iters=10,
            repeat_iters=30,
            use_cuda_graph=True,
            cold_l2_cache=False,
        )
    messages = [str(w.message) for w in caught]
    return dict(
        median_us=statistics.median(values) * 1000,
        raw_us=[float(value) * 1000 for value in values],
        warnings=messages,
        backend="fallback"
        if any("falling back" in msg.lower() for msg in messages)
        else "cupti",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20261001)
    patterns = {
        "hot": torch.arange(6).repeat(6, 1),
        "mid": torch.arange(6).unsqueeze(0) + torch.arange(6).unsqueeze(1) * 3,
        "spread": torch.arange(36).view(6, 6),
        "one_expert": torch.zeros((6, 6), dtype=torch.int32),
        "random": torch.randint(-1, 385, (6, 6)),
        "invalid": torch.full((6, 6), -1, dtype=torch.int32),
    }
    records = []
    for pattern, raw in patterns.items():
        ids = raw.to(device="cuda", dtype=torch.int32)
        reference = moe_align_block_size(ids, 16, 384)
        expected = canonical(reference)
        calls = {"cuda": lambda ids=ids: moe_align_block_size(ids, 16, 384)}
        for warps in (4, 8):
            outputs = tuple(torch.empty_like(x) for x in reference)
            candidate(ids, outputs, warps)
            actual = canonical(outputs)

            # For groups spanning multiple tiles, compare the whole expert group.
            def merged(value):
                length, groups = value
                by_expert = {}
                for expert, rows in groups:
                    by_expert.setdefault(expert, []).extend(rows)
                return length, {key: sorted(rows) for key, rows in by_expert.items()}

            assert merged(actual) == merged(expected), (
                pattern,
                warps,
                actual,
                expected,
            )
            call = lambda ids=ids, outputs=outputs, warps=warps: candidate(
                ids, outputs, warps
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
            for tensor in outputs:
                tensor.fill_(-99)
            graph.replay()
            assert merged(canonical(outputs)) == merged(expected)
            calls[f"triton_{warps}"] = call
        timing = {name: measure(call) for name, call in calls.items()}
        records.append(dict(pattern=pattern, padded_rows=expected[0], timing=timing))
        print(
            pattern,
            {name: value["median_us"] for name, value in timing.items()},
            flush=True,
        )
    result = dict(
        gpu=str(torch.cuda.get_device_properties(0)),
        scope="q6/E384/BM16 alignment; synthetic routes; warm L2; graph",
        records=records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
