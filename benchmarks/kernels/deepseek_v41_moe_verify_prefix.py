# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure shortened DSV41F verification prefixes using captured layer20 routes.

This is an isolated full Experts.apply experiment. Hidden states and routing
weights are synthetic. It excludes attention, draft, PP waits and serving graph
padding, and cannot predict end-to-end throughput.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from benchmarks.kernels import deepseek_v41_moe_harness as harness
from benchmarks.kernels.deepseek_v41_moe_batch_tiles import (
    cold_time,
    make_call,
    require_exact,
)
from benchmarks.kernels.deepseek_v41_moe_tilelang_m6 import load_sampled_weights
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.experts import (
    dsv41_sm80_activation,
)
from vllm.model_executor.layers.fused_moe.experts import (
    mxfp4_int8_sm80 as impl,
)


def load_frame(path: Path, step_id: int) -> dict:
    capture = json.loads(path.read_text())
    if 20 not in capture["layers"] or capture["route_dropped"]:
        raise ValueError("requires a complete capture containing layer20")
    frame = next(row for row in capture["route_frames"] if row["step_id"] == step_id)
    query = frame["query_start_loc"]
    if query[0] != 0 or query[-1] != frame["tokens"]:
        raise ValueError("invalid captured query boundaries")
    if any(b - a != 6 for a, b in zip(query, query[1:])):
        raise ValueError("requires q6 capture")
    routes = torch.tensor(frame["routes"], dtype=torch.int32)
    routes = routes[: frame["tokens"], capture["layers"].index(20)]
    if routes.shape != (frame["tokens"], 6) or not torch.all(
        (routes >= 0) & (routes < 384)
    ):
        raise ValueError("invalid captured routes")
    return {
        "source": str(path.resolve()),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "step_id": step_id,
        "query_start_loc": query,
        "routes": routes,
    }


def make_experts(weights):
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
        moe_parallel_config=SimpleNamespace(
            tp_size=1, pcp_size=1, dp_size=1, ep_size=1, sp_size=1, use_ep=False
        ),
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
    experts._tilelang_activation_quant = dsv41_sm80_activation.prepare_activation_quant(
        torch.device("cuda:0"), 10.0
    )
    experts._tilelang_activation_clamp = 10.0
    return experts, layer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--captures", type=Path, nargs="+", required=True)
    parser.add_argument("--step-ids", type=int, nargs="+", required=True)
    parser.add_argument("--prefixes", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    prefixes = sorted(set(args.prefixes))
    if (
        len(args.captures) != len(args.step_ids)
        or not prefixes
        or prefixes[-1] != 6
        or prefixes[0] < 2
        or args.samples < 1
        or args.rounds < 3
    ):
        raise ValueError("requires paired captures/steps, 2<=q<=6, q6, rounds>=3")
    if torch.cuda.get_device_capability() != (8, 0):
        raise ValueError("this experiment targets SM80")
    frames = [
        load_frame(path, step)
        for path, step in zip(args.captures, args.step_ids, strict=True)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expert_ids = sorted(
        {expert for f in frames for expert in f["routes"].flatten().tolist()}
    )
    print(f"Loading {len(expert_ids)} real layer20 experts", flush=True)
    weights = load_sampled_weights(
        args.model,
        torch.device("cuda"),
        args.output_dir,
        expert_ids=expert_ids,
        routing_and_hidden="captured target routes; synthetic hidden/routing weights",
    )
    experts, layer = make_experts(weights)
    result = {
        "environment": harness.environment(),
        "git_head": harness.git_head(),
        "dirty_diff_sha256": harness.dirty_diff_hash(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": __doc__,
        "timer": "CUDA Event + single-call Graph; cold L2 outside timing",
        "prefix_correctness": "bitwise equal to retained rows of untrimmed q6",
        "rows": [],
    }
    generator = torch.Generator(device="cuda").manual_seed(2026100610)
    for frame in frames:
        query = frame["query_start_loc"]
        full_hidden = (
            torch.randn(
                (query[-1], 5120),
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            * 0.125
        )
        full_ids = frame["routes"].to("cuda")
        full_rw = torch.tensor(
            [[0.07, 0.11, 0.13, 0.17, 0.23, 0.29]], device="cuda"
        ).repeat(query[-1], 1)
        cases, graphs, indices, inputs, work = {}, {}, {}, {}, {}
        for q in reversed(prefixes):
            idx = torch.tensor(
                [token for start in query[:-1] for token in range(start, start + q)],
                device="cuda",
            )
            hidden, ids, rw = full_hidden[idx], full_ids[idx], full_rw[idx]
            name = f"q{q}"
            indices[name] = idx
            inputs[name] = hidden, ids
            cases[name] = make_call(experts, layer, weights, hidden, ids, rw)
            call, output, slots = cases[name]
            call()
            if q == 6:
                expected = output.clone(), slots.view(query[-1], 6, 5120).clone()
                assert torch.isfinite(expected[0]).all()
            require_exact(output, expected[0][idx])
            require_exact(slots.view(len(idx), 6, 5120), expected[1][idx])
            graphs[name] = harness.capture_graph(call, warmup=3, graph_calls=1)
            work[name] = harness.compute_route_work(
                ids,
                16,
                {
                    s: (w["packed"], w["shift2"], w["channel_scale"])
                    for s, w in weights.items()
                },
            ).as_row()
        # Replay changed inputs, then poison outputs to rule out stale Graph data.
        full_hidden.mul_(0.73)
        full_ids.copy_(full_ids.roll(1, dims=0))
        for name, (hidden, ids) in inputs.items():
            hidden.copy_(full_hidden[indices[name]])
            ids.copy_(full_ids[indices[name]])
        graphs["q6"].replay()
        expected = (
            cases["q6"][1].clone(),
            cases["q6"][2].view(query[-1], 6, 5120).clone(),
        )
        for name, graph in graphs.items():
            cases[name][1].fill_(float("nan"))
            cases[name][2].fill_(float("nan"))
            graph.replay()
            torch.accelerator.synchronize()
            require_exact(cases[name][1], expected[0][indices[name]])
            require_exact(
                cases[name][2].view(len(indices[name]), 6, 5120),
                expected[1][indices[name]],
            )
            # Restore captured routes before timing; hidden states stay synthetic.
            inputs[name][1].copy_(frame["routes"].to("cuda")[indices[name]])
        samples = cold_time(graphs, samples=args.samples, rounds=args.rounds)
        medians = {
            name: statistics.median(statistics.median(round_) for round_ in rounds)
            for name, rounds in samples.items()
        }
        result["rows"].append(
            {
                **{k: v for k, v in frame.items() if k != "routes"},
                "bitwise_exact": True,
                "work": work,
                "samples_us": samples,
                "median_us": medians,
                "ratio_to_q6": {
                    name: value / medians["q6"] for name, value in medians.items()
                },
            }
        )
        (args.output_dir / "verify-prefix.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(frame["step_id"], medians, flush=True)


if __name__ == "__main__":
    main()
