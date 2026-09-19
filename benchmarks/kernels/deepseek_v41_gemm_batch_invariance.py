#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolate the batch-shape dependence of the SM80 GEMMs.

Why this exists: a prefix-cache hit recomputes its tail in a much smaller
prefill batch than the cold request used, so cached and uncached servings of the
same prompt can diverge. That is not a cache defect -- the same divergence
appears with prefix caching off when ``max_num_batched_tokens`` changes -- but
finding out *why* took a sweep of the engine's own logits. This probe reduces it
to a ten-line reproduction on a spare GPU.

The measurement: build ``[K, N]`` weights once, then multiply by ``[M, K]``
activations for a range of ``M`` while holding row 0 bit-identical. If row 0 of
the product moves, the library picked a different kernel or reduction order for
that ``M``, and every model with that GEMM inherits the instability.

Findings on a CMP 170HX (SM80), torch 2.x bf16:

- ``K == 5120`` shapes -- everything that reads the residual stream -- shift by
  exactly one bf16 ULP (1.56e-02 on |ref| ~ 3.5) as ``M`` crosses cuBLAS's
  kernel-selection thresholds near 256, 368, 384 and 436.
- Other ``K`` values are bit-stable over the same ``M`` range.
- The same ``M`` run twice is bit-identical, so it is the *selection*, not
  run-to-run nondeterminism.
- Neither ``preferred_blas_library`` (cublas / cublaslt) nor
  ``CUBLAS_WORKSPACE_CONFIG=:16:8`` removes it; the workspace limit only makes
  ``N in {384, 512}`` stable.

Anything exact under greedy decoding therefore needs a GEMM whose reduction
order does not depend on ``M``; upstream's ``VLLM_BATCH_INVARIANT`` does not
cover these call sites.

Usage:
    python benchmarks/kernels/deepseek_v41_gemm_batch_invariance.py [--mode MODE]
"""

from __future__ import annotations

import argparse
import os

import torch

# The hidden size of the DSV4.1 checkpoint, i.e. the K of every GEMM that reads
# the residual stream.
HIDDEN = 5120
DEFAULT_MS = (277, 320, 384, 448, 512, 597, 900, 1024)
SHAPES = (
    (HIDDEN, 5120),
    (HIDDEN, 2304),
    (HIDDEN, 384),
    (HIDDEN, 1536),
    (HIDDEN, 512),
    (2304, HIDDEN),
    (1536, HIDDEN),
    (512, HIDDEN),
    (128, HIDDEN),
    (2048, HIDDEN),
)


def apply_mode(mode: str) -> None:
    """Apply a candidate mitigation before any CUDA work happens."""
    if mode == "ws16":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    elif mode == "cublas":
        torch.backends.cuda.preferred_blas_library("cublas")
    elif mode == "cublaslt":
        torch.backends.cuda.preferred_blas_library("cublaslt")


def row0_spread(K: int, N: int, ms: tuple[int, ...], device: str = "cuda") -> float:
    """Largest drift of a bit-identical row across the batch sizes in ``ms``."""
    weights = torch.randn(K, N, dtype=torch.bfloat16, device=device) / (K**0.5)
    row0 = torch.randn(K, dtype=torch.bfloat16, device=device)
    reference = None
    worst = 0.0
    for m in ms:
        activations = torch.randn(m, K, dtype=torch.bfloat16, device=device)
        activations[0] = row0
        out = (activations @ weights).float()[0]
        if reference is None:
            reference = out.clone()
        else:
            worst = max(worst, (out - reference).abs().max().item())
    return worst


def threshold_map(K: int, N: int, lo: int, hi: int, step: int) -> list[tuple]:
    """Ranges of ``M`` over which a bit-identical row stays bit-identical."""
    weights = torch.randn(K, N, dtype=torch.bfloat16, device="cuda") / (K**0.5)
    row0 = torch.randn(K, dtype=torch.bfloat16, device="cuda")
    spans: list[tuple] = []
    previous = None
    start = lo
    for m in range(lo, hi + 1, step):
        activations = torch.randn(m, K, dtype=torch.bfloat16, device="cuda")
        activations[0] = row0
        out = (activations @ weights).float()[0]
        if previous is None:
            previous, start = out, m
            continue
        if not torch.equal(out, previous):
            spans.append((start, m - step))
            start, previous = m, out
    spans.append((start, hi))
    return spans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        default="default",
        choices=("default", "cublas", "cublaslt", "ws16"),
    )
    ap.add_argument("--shapes", action="store_true", help="sweep the shape table")
    ap.add_argument("--map", action="store_true", help="map the M thresholds")
    ap.add_argument("--lo", type=int, default=250)
    ap.add_argument("--hi", type=int, default=600)
    ap.add_argument("--step", type=int, default=2)
    args = ap.parse_args()

    apply_mode(args.mode)
    print(
        f"mode={args.mode} device={torch.cuda.get_device_name(0)} "
        f"lib={torch.backends.cuda.preferred_blas_library()} "
        f"workspace={os.environ.get('CUBLAS_WORKSPACE_CONFIG', '<unset>')}"
    )

    if args.map or not args.shapes:
        for K, N in ((HIDDEN, HIDDEN), (HIDDEN, 2304)):
            print(f"  thresholds for {K}x{N}, M in [{args.lo},{args.hi}]:")
            for a, b in threshold_map(K, N, args.lo, args.hi, args.step):
                print(f"     M=[{a:5d},{b:5d}]")

    if args.shapes:
        print("worst |row0 drift| across M:")
        affected = 0
        for K, N in SHAPES:
            worst = row0_spread(K, N, DEFAULT_MS)
            affected += bool(worst)
            print(
                f"  {K:5d} x {N:5d}: {worst:.3e}  "
                f"{'M-DEPENDENT' if worst else 'invariant'}"
            )
        print(f"affected: {affected}/{len(SHAPES)}")


if __name__ == "__main__":
    main()
