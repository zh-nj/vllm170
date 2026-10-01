# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline SwiGLU + per-token INT8 candidate for DSV41F on SM80.

The CUDA activation rounds SiLU to BF16 before multiplication, then rounds
the product again. Quantization consumes that rounded product, uses a whole
row's FP32 amax/127 scale, and rounds/clamps to [-127, 127]. Callers own outputs.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import warnings
from pathlib import Path

import torch

from vllm.model_executor.layers.fused_moe.experts.dsv41_sm80_activation import (
    _MATH_SOURCE,
    PASS_CONFIGS,
    build_swiglu_int8,
)
from vllm.tilelang_utils import T, tilelang


def build_swiglu_rounding_probe(*, fast_math: bool, clamp_limit: float | None = None):
    """Expose the BF16 result for finite-gate exhaustive certification."""
    count = T.dynamic("count")
    fast = "true" if fast_math else "false"
    clamp = "false" if clamp_limit is None else "true"
    func = f"dsv41_swiglu_input_rounded<{fast}, {clamp}>"
    limit = 0.0 if clamp_limit is None else clamp_limit

    @T.prim_func
    def kernel(
        gate_up: T.Tensor((count, 2), "bfloat16"),
        out: T.Tensor((count, 1), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(count, 256), threads=256) as block:
            T.import_source(_MATH_SOURCE)
            for col in T.Parallel(256):
                i = block * 256 + col
                if i < count:
                    out[i, 0] = T.call_extern(
                        "float32",
                        func,
                        T.Cast("float32", gate_up[i, 0]),
                        T.Cast("float32", gate_up[i, 1]),
                        T.float32(limit),
                    )

    return kernel


def certify_swiglu_math(*, fast_math: bool, clamp_limit: float | None = None):
    """Compare every finite BF16 gate with the currently loaded CUDA SiLU."""
    bits = torch.arange(65536, device="cuda", dtype=torch.int32)
    bits = bits[(bits & 0x7F80) != 0x7F80]
    gates = bits.to(torch.int16).view(torch.bfloat16)
    x = torch.stack([gates, torch.ones_like(gates)], dim=1)
    expected = torch.empty((gates.numel(), 1), device="cuda", dtype=torch.bfloat16)
    if clamp_limit is None:
        torch.ops._C.silu_and_mul(expected, x)
    else:
        torch.ops._C.silu_and_mul_with_clamp(expected, x, clamp_limit, 1.0, 0.0)
    actual = torch.empty_like(expected)
    fn = tilelang.compile(
        build_swiglu_rounding_probe(fast_math=fast_math, clamp_limit=clamp_limit),
        pass_configs=PASS_CONFIGS,
    )
    fn(x, actual)
    torch.accelerator.synchronize()
    mismatches = (actual.view(torch.int16) != expected.view(torch.int16)).sum().item()
    if mismatches:
        raise AssertionError(f"BF16 SiLU certificate failed: {mismatches} mismatches")
    return {
        "finite_gate_codes": gates.numel(),
        "bitwise_mismatches": mismatches,
        "fast_math": fast_math,
        "clamp_limit": clamp_limit,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 6, 18, 40, 128])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--graph-calls", type=int, default=64)
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        choices=[128, 256, 512, 1024],
        default=[128, 256, 512],
    )
    args = parser.parse_args()
    if any(rows <= 0 for rows in args.rows):
        raise ValueError("rows must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError("this experiment targets SM80")
    from flashinfer.testing import bench_gpu_time_with_cupti

    from benchmarks.kernels import deepseek_v41_moe_harness as harness
    from benchmarks.kernels.deepseek_v41_activation_int8_sm80 import (
        _silu_mul_quant_kernel,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp4_int8_sm80 import (
        _quantize_per_token_kernel,
    )

    metadata = harness.environment()
    metadata.pop("clock_and_power", None)
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    if not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    metadata.update(
        command=sys.argv,
        git_head=harness.git_head(),
        dirty_diff_sha256=harness.dirty_diff_hash(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        tilelang_version=tilelang.__version__,
        graph_scope="activation + quantization, caller-owned outputs, no copies",
    )
    clocks = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            uuid,
            (
                "--query-gpu=uuid,pci.bus_id,clocks.sm,"
                "clocks.mem,power.draw,temperature.gpu"
            ),
            "--format=csv",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    metadata["selected_gpu_clock_and_power"] = clocks
    kernels = {}
    metadata["math_certificates"] = [
        certify_swiglu_math(fast_math=fast_math) for fast_math in (False, True)
    ]
    variants = [(f"tilelang_{threads}", threads, False) for threads in args.threads]
    variants.extend(
        (f"tilelang_fast_{threads}", threads, True) for threads in args.threads
    )
    for name, threads, fast_math in variants:
        kernel = tilelang.compile(
            build_swiglu_int8(threads=threads, fast_math=fast_math),
            pass_configs=PASS_CONFIGS,
        )
        source = kernel.get_kernel_source()
        (args.output_dir / f"{name}.cu").write_text(source)
        metadata[f"cuda_sha256_{name}"] = hashlib.sha256(source.encode()).hexdigest()
        kernels[name] = kernel
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    def flashinfer_time(call):
        with warnings.catch_warnings(record=True) as caught:
            samples = bench_gpu_time_with_cupti(
                call,
                dry_run_iters=10,
                repeat_iters=30,
                use_cuda_graph=True,
                cold_l2_cache=False,
            )
        messages = [str(w.message) for w in caught]
        fallback = any("falling back" in m.lower() for m in messages)
        return {
            "backend": "flashinfer_fallback" if fallback else "cupti",
            "cache": "warm",
            "warnings": messages,
            "samples_us": [float(v) * 1000 for v in samples],
            "median_us": statistics.median(samples) * 1000,
        }

    gen = torch.Generator(device="cuda").manual_seed(20261002)
    records = []
    for rows in args.rows:
        x = torch.randn(
            (rows, 4608), device="cuda", dtype=torch.bfloat16, generator=gen
        )
        middle = torch.empty((rows, 2304), device="cuda", dtype=torch.bfloat16)
        outputs = {
            name: (
                torch.empty((rows, 2304), device="cuda", dtype=torch.int8),
                torch.empty((rows,), device="cuda", dtype=torch.float32),
            )
            for name in ("separate", "triton_fused", *kernels)
        }

        def separate(middle=middle, x=x, rows=rows, outputs=outputs):
            torch.ops._C.silu_and_mul(middle, x)
            _quantize_per_token_kernel[(rows,)](
                middle, *outputs["separate"], rows, 2304, 0.0, BLOCK_K=4096
            )

        def triton_fused(x=x, rows=rows, outputs=outputs):
            _silu_mul_quant_kernel[(rows,)](
                x, *outputs["triton_fused"], rows, 2304, BLOCK_K=4096
            )

        calls = {"separate": separate, "triton_fused": triton_fused}
        calls.update(
            {
                name: lambda name=name, x=x, outputs=outputs: kernels[name](
                    x, *outputs[name]
                )
                for name in kernels
            }
        )
        separate()
        expected = tuple(t.clone() for t in outputs["separate"])

        def validate(name, outputs=outputs, expected=expected):
            q, sf = outputs[name]
            torch.accelerator.synchronize()
            assert torch.equal(q, expected[0]), name
            assert torch.equal(sf.view(torch.int32), expected[1].view(torch.int32)), (
                name
            )

        def validator(name, outputs=outputs, validate=validate):
            def check(graph):
                outputs[name][0].fill_(-128)
                outputs[name][1].fill_(float("nan"))
                graph.replay()
                validate(name)

            return check

        for name, call in calls.items():
            call()
            validate(name)
        flashinfer_baseline = flashinfer_time(separate)
        # The old Triton prototype is a performance comparator on these inputs;
        # its approximate exp has not been certified over the full BF16 domain.
        for name in ("triton_fused", *kernels):
            timing = harness.interleaved_ab(
                separate,
                calls[name],
                rounds=args.rounds,
                warmup=10,
                graph_calls=args.graph_calls,
                samples=args.samples,
                settle_ms=150,
                baseline_validator=validator("separate"),
                candidate_validator=validator(name),
            )
            flashinfer_candidate = flashinfer_time(calls[name])
            row = {
                "rows": rows,
                "candidate": name,
                "bitwise_exact": True,
                "graph_ab": timing,
                "flashinfer_baseline": flashinfer_baseline,
                "flashinfer_candidate": flashinfer_candidate,
            }
            records.append(row)
            print(
                json.dumps(
                    {
                        "rows": rows,
                        "candidate": name,
                        "baseline_us": timing["baseline_median_p50_us"],
                        "candidate_us": timing["candidate_median_p50_us"],
                        "ratio": timing["baseline_ratio"],
                        "flashinfer_baseline": flashinfer_baseline["median_us"],
                        "flashinfer_candidate": flashinfer_candidate["median_us"],
                    }
                ),
                flush=True,
            )
            (args.output_dir / "microbench.json").write_text(
                json.dumps(records, indent=2)
            )


if __name__ == "__main__":
    main()
