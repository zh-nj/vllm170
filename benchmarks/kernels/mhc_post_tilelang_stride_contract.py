# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent GPU contract test for the TileLang mHC post kernel.

The fast-prefill tail reaches mHC post after row selection.  This test first
constructs full-width tensors and applies the same ``index_select(0, keep)``
operation as the model, then checks the resulting shapes and strides before
calling the kernel.  It also separates the production wrapper (which allocates
its output) from the underlying kernel with a preallocated output buffer.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang
from vllm.model_executor.kernels.mhc.tilelang_kernels import (
    mhc_post_tilelang as mhc_post_tilelang_kernel,
)


def _time(call, warmup: int, repeats: int) -> dict[str, float]:
    samples = [
        value * 1000.0
        for value in bench_gpu_time_with_cupti(
            call,
            dry_run_iters=warmup,
            repeat_iters=repeats,
            cold_l2_cache=True,
            use_cuda_graph=False,
        )
    ]
    return {
        "median_us": statistics.median(samples),
        "p90_us": sorted(samples)[min(len(samples) - 1, int(len(samples) * 0.9))],
    }


def _case(
    full_tokens: int,
    tokens: int,
    hidden_size: int,
    hc_mult: int,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(20260921 + tokens)
    x_full = torch.randn(
        (full_tokens, hidden_size), dtype=torch.bfloat16, device=device
    )
    residual_full = torch.randn(
        (full_tokens, hc_mult, hidden_size), dtype=torch.bfloat16, device=device
    )
    post_full = torch.randn(
        (full_tokens, hc_mult, 1), dtype=torch.float32, device=device
    )
    comb_full = torch.randn(
        (full_tokens, hc_mult, hc_mult), dtype=torch.float32, device=device
    )
    if tokens == full_tokens:
        keep = torch.arange(full_tokens, device=device, dtype=torch.long)
    else:
        # Non-unit row spacing models a compact tail selected from a larger
        # scheduler batch while keeping the same index_select contract.
        keep = (
            torch.linspace(
                0, full_tokens - 1, tokens, device=device, dtype=torch.float32
            )
            .round()
            .to(torch.long)
        )
        keep = torch.unique_consecutive(keep)
        if keep.numel() != tokens:
            raise AssertionError("keep construction did not preserve token count")

    x = x_full.index_select(0, keep)
    residual = residual_full.index_select(0, keep)
    post = post_full.index_select(0, keep)
    comb = comb_full.index_select(0, keep)
    post_2d = post.squeeze(-1)
    out = torch.empty_like(residual)
    reference = (
        x.float().unsqueeze(-2) * post
        + torch.bmm(comb.transpose(1, 2), residual.float())
    ).bfloat16()

    tensors = {
        "x": x,
        "residual": residual,
        "post": post,
        "post_2d": post_2d,
        "comb": comb,
        "out": out,
    }
    strides = {name: list(value.stride()) for name, value in tensors.items()}
    if not all(value.is_contiguous() for value in tensors.values()):
        raise AssertionError(f"index_select contract is not contiguous: {strides}")

    def kernel_call() -> None:
        mhc_post_tilelang_kernel(
            comb,
            residual,
            post_2d,
            x,
            out,
            hc_mult,
            hidden_size,
        )

    kernel_call()
    torch.accelerator.synchronize()
    torch.testing.assert_close(out, reference, atol=5e-2, rtol=1e-2)

    wrapper_result: list[torch.Tensor | None] = [None]

    def wrapper_call() -> None:
        # The wrapper allocates its output, matching the production call. Keep
        # the returned tensor live without adding a copy to the timed region.
        wrapper_result[0] = mhc_post_tilelang(x, residual, post, comb)

    wrapper_call()
    torch.accelerator.synchronize()
    assert wrapper_result[0] is not None
    torch.testing.assert_close(wrapper_result[0], reference, atol=5e-2, rtol=1e-2)
    return {
        "full_tokens": full_tokens,
        "tokens": tokens,
        "hidden_size": hidden_size,
        "hc_mult": hc_mult,
        "keep_first_last": [int(keep[0]), int(keep[-1])],
        "strides": strides,
        "kernel_preallocated": _time(kernel_call, warmup, repeats),
        "wrapper_with_alloc": _time(wrapper_call, warmup, repeats),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--hc-mult", type=int, default=4)
    parser.add_argument("--full-tokens", type=int, default=1024)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 8, 128, 1024])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.hidden_size <= 0 or args.hc_mult <= 0 or args.full_tokens <= 0:
        raise ValueError("hidden-size, hc-mult, and full-tokens must be positive")
    if any(tokens <= 0 or tokens > args.full_tokens for tokens in args.tokens):
        raise ValueError("tokens must be in [1, full-tokens]")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this microtest")

    torch.accelerator.set_device_index(args.device)
    device = torch.device("cuda", args.device)
    cases = [
        _case(
            args.full_tokens,
            tokens,
            args.hidden_size,
            args.hc_mult,
            args.warmup,
            args.repeats,
            device,
        )
        for tokens in args.tokens
    ]
    result = {
        "device": torch.cuda.get_device_name(args.device),
        "device_index": args.device,
        "torch": torch.__version__,
        "cases": cases,
    }
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
