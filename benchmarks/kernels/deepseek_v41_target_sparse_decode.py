# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tune target sparse decode with cold L2 and production C1/C2 cache layouts."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch

from benchmarks.kernels.deepseek_v4_sparse_mla_decode import (
    PARTIAL_CONFIGS,
    REDUCE_CONFIGS,
    DecodeCase,
)
from benchmarks.kernels.deepseek_v41_moe_w2_pipeline import time_graphs
from benchmarks.kernels.deepseek_v41_verify_component_probe import capture
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _decode_num_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=6)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--main-len", type=int, default=128)
    parser.add_argument("--extra-len", type=int, default=512)
    parser.add_argument("--ratio", type=int, choices=(1, 2), default=1)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(20261001)
    splits = args.splits or _decode_num_splits(
        args.queries, 4, args.main_len, args.extra_len, 32
    )
    case = DecodeCase(
        args.queries,
        args.main_len,
        args.extra_len,
        splits,
        20261001,
        128,
        0,
        extra_block_size=128 // args.ratio,
        cache_rows=32768 // args.ratio,
    )
    case.main_indices.copy_(
        torch.arange(args.queries * args.main_len, device="cuda", dtype=torch.int32)
        .remainder(args.main_len)
        .add(32768 // args.ratio - args.main_len)
    )
    case.extra_indices.copy_(
        torch.randint(0, 32768 // args.ratio, case.extra_indices.shape, device="cuda")
    )
    block_k = 16 if args.queries == 1 else 32
    baseline_partial = (16, block_k, 4)
    baseline_reduce = (
        (1, 8) if args.queries == 1 else (2 if args.queries >= 8 else 1, 4)
    )
    case.split(baseline_partial, baseline_reduce)
    reference = case.out.clone()

    def run(partial_config, reduce_config):
        case.split(partial_config, reduce_config)
        return case.out

    configs = {
        f"partial_{h}_{k}_{w}": (config, baseline_reduce)
        for config in PARTIAL_CONFIGS
        for h, k, w in (config,)
    }
    configs.update(
        {
            f"reduce_{h}_{w}": (baseline_partial, config)
            for config in REDUCE_CONFIGS
            for h, w in (config,)
        }
    )
    if args.quick:
        configs = {
            name: configs[name]
            for name in (f"partial_16_{block_k}_4", f"partial_16_{block_k}_8")
        }
    graphs, comparisons = {}, {}
    for name, (partial_config, reduce_config) in configs.items():
        call = partial(run, partial_config, reduce_config)
        output = call().clone()
        torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)
        comparisons[name] = dict(
            exact=torch.equal(output, reference),
            max_abs=(output.float() - reference.float()).abs().max().item(),
            partial=partial_config,
            reduce=reduce_config,
            partial_registers=case.partial_kernel.n_regs,
            partial_spills=case.partial_kernel.n_spills,
            partial_shared_bytes=case.partial_kernel.metadata.shared,
        )
        graphs[name] = {args.queries: capture(call)}
    raw = time_graphs(
        {name: value[args.queries][0] for name, value in graphs.items()}, args.samples
    )
    results = {
        name: dict(
            **comparisons[name],
            **values,
        )
        for name, values in raw.items()
    }
    result = dict(
        queries=args.queries,
        splits=splits,
        ratio=args.ratio,
        main_len=args.main_len,
        extra_len=args.extra_len,
        scope="synthetic 32K-context, ragged SWA+compressed KV, cold L2",
        gpu=str(torch.cuda.get_device_properties()),
        timing="one-call graph, 3 rounds, 500ms settle, 64MiB flush outside events",
        results=results,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: (v["median_us"], v["exact"]) for k, v in results.items()}))


if __name__ == "__main__":
    with torch.inference_mode():
        main()
