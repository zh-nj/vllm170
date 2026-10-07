# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import itertools
import json
import statistics
import time
from pathlib import Path

import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.triton_utils import triton
from vllm.utils.torch_utils import set_random_seed


def get_topk_ids(num_tokens: int, num_experts: int, topk: int) -> torch.Tensor:
    return torch.stack(
        [
            torch.randperm(num_experts, dtype=torch.int32, device="cuda")[:topk]
            for _ in range(num_tokens)
        ]
    )


# test configurations
num_tokens_range = [1, 16, 256, 4096]
num_experts_range = [16, 64, 224, 256, 280, 512]
topk_range = [1, 2, 8]
ep_size_range = [1, 8]
configs = list(
    itertools.product(num_tokens_range, num_experts_range, topk_range, ep_size_range)
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "num_experts", "topk", "ep_size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["vllm"],
        line_names=["vLLM"],
        plot_name="moe-align-block-size-performance",
        args={},
    )
)
def benchmark(num_tokens, num_experts, topk, ep_size, provider):
    """Benchmark function for Triton."""
    block_size = 256
    set_random_seed(0)
    topk_ids = get_topk_ids(num_tokens, num_experts, topk)

    e_map = None
    if ep_size != 1:
        local_e = num_experts // ep_size
        e_ids = torch.randperm(num_experts, device="cuda", dtype=torch.int32)[:local_e]
        e_map = torch.full((num_experts,), -1, device="cuda", dtype=torch.int32)
        e_map[e_ids] = torch.arange(local_e, device="cuda", dtype=torch.int32)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "vllm":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: moe_align_block_size(
                topk_ids, block_size, num_experts, e_map, ignore_invalid_experts=True
            ),
            quantiles=quantiles,
        )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


def benchmark_m1_e512_decode(output: str | None) -> None:
    """Compare alignment dispatch in graphs with amortized host submission."""
    from vllm.platforms import current_platform

    assert current_platform.is_device_capability((8, 0))
    set_random_seed(0)
    ids = get_topk_ids(1, 512, 10)
    previous = envs.VLLM_MOE_ALIGN_M1_E512_SM80
    graphs, retained = [], []
    try:
        for enabled in (False, True):
            envs.VLLM_MOE_ALIGN_M1_E512_SM80 = enabled
            for _ in range(3):
                moe_align_block_size(ids, 8, 512, ignore_invalid_experts=True)
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(64):
                    retained.append(
                        moe_align_block_size(ids, 8, 512, ignore_invalid_experts=True)
                    )
            graphs.append(("candidate" if enabled else "baseline", graph))
    finally:
        envs.VLLM_MOE_ALIGN_M1_E512_SM80 = previous
    for _, graph in graphs:
        graph.replay()
    torch.accelerator.synchronize()
    for baseline, candidate in zip(retained[63], retained[-1]):
        torch.testing.assert_close(candidate, baseline, atol=0, rtol=0)
    warmup_end = time.monotonic() + 2
    while time.monotonic() < warmup_end:
        for _, graph in graphs:
            for _ in range(10):
                graph.replay()
        torch.accelerator.synchronize()
    samples = {name: [] for name, _ in graphs}
    for repetition in range(11):
        for name, graph in graphs if repetition % 2 == 0 else reversed(graphs):
            for _ in range(10):
                graph.replay()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(50):
                graph.replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / (50 * 64))
    result = {
        "shape": [1, 10],
        "num_experts": 512,
        "block_size": 8,
        "timing": "CUDA Events, 64 alignments per graph, alternating arms",
        "median_us": {
            name: statistics.median(times) for name, times in samples.items()
        },
        "samples_us": samples,
    }
    print(json.dumps(result, indent=2))
    if output:
        Path(output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decode-m1-e512", action="store_true", help="Run the bounded SM80 decode A/B."
    )
    parser.add_argument("--output-json", help="Save decode A/B samples.")
    parser.add_argument(
        "--num_experts",
        type=int,
        default=64,
        choices=[8, 16, 32, 64, 128, 256],
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=8,
        choices=[2, 4, 8],
        help="Top-k value for correctness check.",
    )
    args = parser.parse_args()

    if args.decode_m1_e512:
        benchmark_m1_e512_decode(args.output_json)
    else:
        benchmark.run(print_data=True, show_plots=True)
