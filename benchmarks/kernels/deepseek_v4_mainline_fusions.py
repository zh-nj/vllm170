# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Audit DeepSeek-V4 mainline router and mHC fusions on CUDA hardware."""

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch

from vllm.model_executor.kernels.mhc.tilelang import (
    mhc_pre_broadcast_tilelang,
    mhc_pre_tilelang,
)
from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    vllm_topk_softplus_sqrt,
)
from vllm.platforms import current_platform

KernelFn = Callable[[], object]


def _measure(
    fn: KernelFn, warmup: int, iterations: int, samples: int
) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    timings = []
    for _ in range(samples):
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000.0 / iterations)
    return statistics.median(timings), sorted(timings)[int(0.9 * (samples - 1))]


def _topk_set_recall(actual: torch.Tensor, expected: torch.Tensor) -> float:
    matches = (actual.unsqueeze(-1) == expected.unsqueeze(-2)).any(dim=-1)
    return matches.float().mean().item()


def _router_rows(
    sizes: list[int],
    warmup: int,
    iterations: int,
    samples: int,
    generator: torch.Generator,
) -> list[dict[str, object]]:
    rows = []
    num_experts = 256
    topk = 6
    scaling = 1.5
    correction_bias = torch.full(
        (num_experts,), 8.08, dtype=torch.float32, device="cuda"
    )

    for size_m in sizes:
        logits = torch.randn(
            (size_m, num_experts),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )

        def specialized(logits=logits):
            return dsv4_topk(logits, correction_bias, torch.int32, scaling)

        def generic(size_m=size_m, logits=logits):
            weights = torch.empty((size_m, topk), dtype=torch.float32, device="cuda")
            indices = torch.empty((size_m, topk), dtype=torch.int32, device="cuda")
            token_expert_indices = torch.empty_like(indices)
            return vllm_topk_softplus_sqrt(
                weights,
                indices,
                token_expert_indices,
                logits,
                True,
                correction_bias,
                routed_scaling_factor=scaling,
            )

        specialized_weights, specialized_ids = specialized()
        generic_weights, generic_ids = generic()
        torch.accelerator.synchronize()
        specialized_us, specialized_p90_us = _measure(
            specialized, warmup, iterations, samples
        )
        generic_us, generic_p90_us = _measure(generic, warmup, iterations, samples)
        row: dict[str, object] = {
            "component": "router",
            "device": current_platform.get_device_name(),
            "m": size_m,
            "num_experts": num_experts,
            "topk": topk,
            "specialized_us": specialized_us,
            "specialized_p90_us": specialized_p90_us,
            "generic_us": generic_us,
            "generic_p90_us": generic_p90_us,
            "speedup": generic_us / specialized_us,
            "ordered_ids_equal": torch.equal(specialized_ids, generic_ids),
            "topk_set_recall": _topk_set_recall(specialized_ids, generic_ids),
            "weight_max_abs": (specialized_weights - generic_weights)
            .abs()
            .max()
            .item(),
            "weight_mean_abs": (specialized_weights - generic_weights)
            .abs()
            .mean()
            .item(),
        }
        rows.append(row)
    return rows


def _mhc_rows(
    sizes: list[int],
    warmup: int,
    iterations: int,
    samples: int,
    generator: torch.Generator,
) -> list[dict[str, object]]:
    rows = []
    hidden_size = 7168
    hc_mult = 4
    hc_mult3 = hc_mult * (2 + hc_mult)
    eps = 1e-6
    sinkhorn_repeat = 20
    fn = torch.randn(
        (hc_mult3, hc_mult, hidden_size),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    ).flatten(1, 2)
    fn.mul_(1e-4)
    fn_broadcast = fn.view(hc_mult3, hc_mult, hidden_size).sum(dim=1)
    hc_scale = torch.randn(
        (3,), dtype=torch.float32, device="cuda", generator=generator
    ).mul_(0.1)
    hc_base = torch.randn(
        (hc_mult3,), dtype=torch.float32, device="cuda", generator=generator
    ).mul_(0.1)
    norm_weight = torch.randn(
        (hidden_size,), dtype=torch.bfloat16, device="cuda", generator=generator
    )

    for size_m in sizes:
        residual = torch.randn(
            (size_m, hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )

        def broadcast(residual=residual):
            return mhc_pre_broadcast_tilelang(
                residual,
                fn,
                hc_scale,
                hc_base,
                eps,
                eps,
                eps,
                2.0,
                sinkhorn_repeat,
                norm_weight=norm_weight,
                norm_eps=eps,
                fn_broadcast=fn_broadcast,
            )

        def expanded(residual=residual):
            residual_expanded = residual[:, None, :].repeat(1, hc_mult, 1)
            post_mix, res_mix, layer_input = mhc_pre_tilelang(
                residual_expanded,
                fn,
                hc_scale,
                hc_base,
                eps,
                eps,
                eps,
                2.0,
                sinkhorn_repeat,
                norm_weight=norm_weight,
                norm_eps=eps,
            )
            return residual_expanded, post_mix, res_mix, layer_input

        broadcast_outputs = broadcast()
        expanded_outputs = expanded()
        torch.accelerator.synchronize()
        broadcast_us, broadcast_p90_us = _measure(
            broadcast, warmup, iterations, samples
        )
        expanded_us, expanded_p90_us = _measure(expanded, warmup, iterations, samples)

        max_abs = []
        mean_abs = []
        exact = []
        for actual, reference in zip(broadcast_outputs, expanded_outputs, strict=True):
            delta = (actual.float() - reference.float()).abs()
            max_abs.append(delta.max().item())
            mean_abs.append(delta.mean().item())
            exact.append(torch.equal(actual, reference))
        row = {
            "component": "mhc_broadcast",
            "device": current_platform.get_device_name(),
            "m": size_m,
            "hidden_size": hidden_size,
            "hc_mult": hc_mult,
            "broadcast_us": broadcast_us,
            "broadcast_p90_us": broadcast_p90_us,
            "expanded_us": expanded_us,
            "expanded_p90_us": expanded_p90_us,
            "speedup": expanded_us / broadcast_us,
            "output_exact": exact,
            "output_max_abs": max_abs,
            "output_mean_abs": mean_abs,
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("all", "router", "mhc"), default="all")
    parser.add_argument("--m", type=int, nargs="+", default=[1, 8, 64, 1667])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    rows: list[dict[str, object]] = []
    if args.component in ("all", "router"):
        rows.extend(
            _router_rows(
                args.m,
                args.warmup,
                args.iterations,
                args.samples,
                generator,
            )
        )
    if args.component in ("all", "mhc"):
        rows.extend(
            _mhc_rows(
                args.m,
                args.warmup,
                args.iterations,
                args.samples,
                generator,
            )
        )

    for row in rows:
        print(json.dumps(row, sort_keys=True), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
