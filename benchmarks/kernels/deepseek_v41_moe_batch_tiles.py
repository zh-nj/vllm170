# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare reachable CUTLASS tiles for small DSV41F verification batches.

Uses 36 checkpoint experts and synthetic hidden states/routes. Measures the
complete Experts.apply path with one call per Graph and cold L2 per sample.
The existing activation fusion gate is identical in all arms.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from benchmarks.kernels import deepseek_v41_moe_harness as harness
from benchmarks.kernels.deepseek_v41_moe_tilelang_m6 import load_sampled_weights
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.experts import (
    dsv41_sm80_activation,
)
from vllm.model_executor.layers.fused_moe.experts import (
    mxfp4_int8_sm80 as impl,
)

TILES = ((16, 128), (16, 64), (64, 128))


def require_exact(actual, expected):
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def make_call(experts, layer, weights, hidden, ids, routing_weights):
    tokens, topk = ids.shape
    ws13 = torch.empty((tokens * topk, 6912), device="cuda", dtype=torch.bfloat16)
    ws2 = torch.empty((tokens * topk, 5120), device="cuda", dtype=torch.bfloat16)
    output = ws13.flatten()[: tokens * 5120].view(tokens, 5120)

    def call():
        experts.apply(
            output,
            hidden,
            layer.w13_weight,
            layer.w2_weight,
            routing_weights,
            ids,
            MoEActivation.SILU,
            384,
            None,
            None,
            None,
            ws13,
            ws2,
            None,
            False,
        )

    return call, output, ws2


def cold_time(graphs, *, samples, rounds):
    flush = torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    assert flush.numel() > torch.cuda.get_device_properties(0).L2_cache_size
    result = {name: [] for name in graphs}
    pairs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(samples)
    ]
    for index in range(rounds):
        names = list(graphs)
        if index % 2:
            names.reverse()
        for name in names:
            graph = graphs[name]
            settle = time.monotonic()
            while time.monotonic() - settle < 0.2:
                graph.replay()
                torch.accelerator.synchronize()
            for start, end in pairs:
                flush.zero_()
                start.record()
                graph.replay()
                end.record()
            torch.accelerator.synchronize()
            result[name].append(
                [start.elapsed_time(end) * 1000 for start, end in pairs]
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[12, 18])
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=["hot", "mid", "spread"],
        default=["hot", "mid", "spread"],
    )
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if (
        args.samples < 1
        or args.rounds < 3
        or any(not 7 <= m <= 48 for m in args.tokens)
    ):
        raise ValueError("requires samples>0, rounds>=3, and 7<=tokens<=48")
    if torch.cuda.get_device_capability() != (8, 0):
        raise ValueError("this experiment targets SM80")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights = load_sampled_weights(args.model, torch.device("cuda"), args.output_dir)
    parallel = SimpleNamespace(
        tp_size=1, pcp_size=1, dp_size=1, ep_size=1, sp_size=1, use_ep=False
    )
    moe = SimpleNamespace(
        num_experts=384,
        num_local_experts=384,
        experts_per_token=6,
        hidden_dim=5120,
        intermediate_size_per_partition=2304,
        in_dtype=torch.bfloat16,
        activation=MoEActivation.SILU,
        has_bias=False,
        is_lora_enabled=False,
        moe_parallel_config=parallel,
        swiglu_limit=10.0,
        swiglu_alpha=None,
        swiglu_beta=None,
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
    )
    layer = SimpleNamespace(
        w13_weight=weights["w13"]["packed"],
        w2_weight=weights["w2"]["packed"],
        w13_mxfp4_int8_shift2=weights["w13"]["shift2"],
        w2_mxfp4_int8_shift2=weights["w2"]["shift2"],
    )
    quant = FusedMoEQuantConfig.make(
        w1_scale=weights["w13"]["channel_scale"],
        w2_scale=weights["w2"]["channel_scale"],
    )
    with patch.object(impl, "get_current_vllm_config_or_none", lambda: None):
        experts = impl.Mxfp4Int8Sm80Experts(moe, quant)
        experts.process_weights_after_loading(layer)
    # Match live activation while every selected M uses the CUTLASS GEMMs.
    experts._tilelang_activation_quant = dsv41_sm80_activation.prepare_activation_quant(
        torch.device("cuda:0"), 10.0
    )
    experts._tilelang_activation_clamp = 10.0
    generator = torch.Generator(device="cuda").manual_seed(2026100608)
    result = {
        "environment": harness.environment(),
        "git_head": harness.git_head(),
        "dirty_diff_sha256": harness.dirty_diff_hash(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timer": "CUDA Event + single-call Graph, 64MiB L2 flush outside timing",
        "scope": "full Experts.apply, checkpoint layer20 experts137-172;"
        " synthetic hidden/routes; identical frozen activation gate",
        "rows": [],
    }
    result["environment"].pop("clock_and_power", None)
    destination = args.output_dir / "batch-tiles.json"
    for tokens in args.tokens:
        for pattern in args.patterns:
            hidden = (
                torch.randn(
                    (tokens, 5120),
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=generator,
                )
                * 0.125
            )
            index = torch.arange(tokens, device="cuda", dtype=torch.int32)
            offset = (
                torch.zeros_like(index)
                if pattern == "hot"
                else ((index % 6) * (3 if pattern == "mid" else 6))
            )
            ids = 137 + offset[:, None] + torch.arange(6, device="cuda")[None, :]
            ids = ids.to(torch.int32).contiguous()
            rw = torch.tensor(
                [[0.07, 0.11, 0.13, 0.17, 0.23, 0.29]], device="cuda"
            ).repeat(tokens, 1)
            cases, graphs, work = {}, {}, {}
            for bm, bn in TILES:
                name = f"bm{bm}_bn{bn}"
                cases[name] = make_call(experts, layer, weights, hidden, ids, rw)
                call, output, slots = cases[name]
                with patch.object(
                    impl,
                    "select_mxfp4_int8_sm80_tile",
                    lambda m, topk, e, bm=bm, bn=bn: (bm, bn),
                ):
                    call()
                    if not graphs:
                        expected = output.clone(), slots.clone()
                        assert torch.isfinite(expected[0]).all()
                    else:
                        require_exact(output, expected[0])
                        require_exact(slots, expected[1])
                    graphs[name] = harness.capture_graph(call, warmup=3, graph_calls=1)
                accounting = harness.compute_route_work(
                    ids,
                    bm,
                    {
                        s: (w["packed"], w["shift2"], w["channel_scale"])
                        for s, w in weights.items()
                    },
                )
                work[name] = accounting.as_row()
            for name, graph in graphs.items():
                cases[name][1].fill_(float("nan"))
                cases[name][2].fill_(float("nan"))
                graph.replay()
                torch.accelerator.synchronize()
                require_exact(cases[name][1], expected[0])
                require_exact(cases[name][2], expected[1])
            # Data changes must reach captured routing, quantization, and GEMMs.
            hidden.mul_(0.73)
            ids.copy_(137 + (ids - 137 + 7) % 36)
            graphs["bm16_bn128"].replay()
            expected = cases["bm16_bn128"][1].clone(), cases["bm16_bn128"][2].clone()
            for name in list(graphs)[1:]:
                cases[name][1].fill_(float("nan"))
                cases[name][2].fill_(float("nan"))
                graphs[name].replay()
                torch.accelerator.synchronize()
                require_exact(cases[name][1], expected[0])
                require_exact(cases[name][2], expected[1])
            samples = cold_time(graphs, samples=args.samples, rounds=args.rounds)
            medians = {
                name: statistics.median(statistics.median(r) for r in rows)
                for name, rows in samples.items()
            }
            row = {
                "tokens": tokens,
                "pattern": pattern,
                "bitwise_exact": True,
                "work_before_route_rotation": work,
                "samples_us": samples,
                "median_us": medians,
            }
            result["rows"].append(row)
            destination.write_text(json.dumps(result, indent=2) + "\n")
            print(tokens, pattern, medians, flush=True)


if __name__ == "__main__":
    main()
