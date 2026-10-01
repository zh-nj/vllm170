# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tune target sparse decode with cold L2 and production C1/C2 cache layouts."""

import argparse
import json
import statistics
from functools import partial
from pathlib import Path

import torch

from benchmarks.kernels.deepseek_v4_sparse_mla_decode import (
    PARTIAL_CONFIGS,
    REDUCE_CONFIGS,
    DecodeCase,
)
from benchmarks.kernels.deepseek_v41_verify_component_probe import capture, time_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=6)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--main-len", type=int, default=128)
    parser.add_argument("--ratio", type=int, choices=(1, 2), default=1)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(20261001)
    case = DecodeCase(
        args.queries,
        args.main_len,
        512,
        args.splits,
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
    baseline_partial, baseline_reduce = (16, 32, 4), (1, 4)
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
            name: configs[name] for name in ("partial_16_32_4", "partial_16_32_8")
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
    raw = time_cases(graphs, args.samples)
    results = {
        name: dict(
            **comparisons[name],
            median_us=statistics.median(sum(values[args.queries], [])),
            raw_us=values[args.queries],
        )
        for name, values in raw.items()
    }
    result = dict(
        queries=args.queries,
        splits=args.splits,
        ratio=args.ratio,
        main_len=args.main_len,
        scope="synthetic 32K-context, SWA128+top512, random indices, cold L2",
        gpu=str(torch.cuda.get_device_properties()),
        timing="one-call graph, 3 rounds, 64MiB flush outside each sample",
        results=results,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: (v["median_us"], v["exact"]) for k, v in results.items()}))


if __name__ == "__main__":
    with torch.inference_mode():
        main()
