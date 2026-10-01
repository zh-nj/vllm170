# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe pipeline geometry and packed-word loads in the live M6 W2 template."""

import argparse
import hashlib
import inspect
import json
import statistics
import time
from pathlib import Path

import torch

from benchmarks.kernels.deepseek_v41_moe_tilelang_m6 import (
    load_sampled_weights,
    probe_apply,
    require_exact,
)
from vllm.model_executor.layers.fused_moe.experts import dsv41_sm80_tilelang as runtime
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)


def compile_variant(stages, threads, destination, *, word_load=False):
    source = inspect.getsource(runtime._build_decode_gemm)
    source = source.replace("    threads = 128\n", f"    threads = {threads}\n")
    source = source.replace(
        "    num_stages = 3 if source_rows_are_slots else 2\n",
        f"    num_stages = {stages} if source_rows_are_slots else 2\n",
    )
    suffix = "-word" if word_load else ""
    has_word_switch = "    use_word_load = " in source
    if has_word_switch:
        source = source.replace(
            "    use_word_load = source_rows_are_slots "
            "and tokens == _TILELANG_W2_TOKENS\n",
            f"    use_word_load = {word_load}\n",
        )
    elif word_load:
        begin = source.index("                        word = (\n")
        end = source.index("                        decoded =", begin)
        source = (
            source[:begin]
            + "                        word = T.call_extern(\n"
            + '                            "uint32", "dsv41_load_u32",\n'
            + "                            T.address_of(p_s[kb_rel, n_group, base]),\n"
            + "                        )\n"
            + source[end:]
        )
    path = destination / f"template-s{stages}-t{threads}{suffix}.py"
    path.write_text(source)
    namespace = dict(vars(runtime))
    if word_load and not has_word_switch:
        namespace["_PRMT_DECODE_SOURCE"] += r"""
__device__ __forceinline__ uint32_t dsv41_load_u32(const uint8_t* data) {
  return *reinterpret_cast<const uint32_t*>(data);
}
"""
    exec(compile(source, str(path.resolve()), "exec"), namespace)
    start = time.perf_counter()
    kernel = runtime.tilelang.compile(
        namespace["_build_decode_gemm"](True, tokens=6, block_n=128),
        pass_configs=runtime.PASS_CONFIGS,
    )
    generated = kernel.get_kernel_source()
    (destination / f"kernel-s{stages}-t{threads}{suffix}.cu").write_text(generated)
    return kernel, dict(
        compile_s=time.perf_counter() - start,
        template_sha256=hashlib.sha256(source.encode()).hexdigest(),
        cuda_sha256=hashlib.sha256(generated.encode()).hexdigest(),
    )


def clock_state():
    import pynvml

    pynvml.nvmlInit()
    uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
    return dict(
        uuid=uuid,
        sm_mhz=pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
        memory_mhz=pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM),
    )


def time_graphs(graphs, samples):
    flush = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
    starts = [torch.Event(enable_timing=True) for _ in range(samples)]
    ends = [torch.Event(enable_timing=True) for _ in range(samples)]
    raw = {name: [] for name in graphs}
    clocks = {name: [] for name in graphs}
    for round_index in range(3):
        names = list(graphs)
        if round_index % 2:
            names.reverse()
        for name in names:
            graph = graphs[name]
            for _ in range(10):
                graph.replay()
            settle_start = time.perf_counter()
            while time.perf_counter() - settle_start < 0.5:
                graph.replay()
                torch.accelerator.synchronize()
            before = clock_state()
            for start, end in zip(starts, ends):
                flush.zero_()
                start.record()
                graph.replay()
                end.record()
            ends[-1].synchronize()
            raw[name].append([s.elapsed_time(e) * 1000 for s, e in zip(starts, ends)])
            clocks[name].append(dict(before=before, after=clock_state()))
    return {
        name: dict(
            median_us=statistics.median(sum(values, [])),
            raw_us=values,
            clocks=clocks[name],
        )
        for name, values in raw.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--word-only", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    dest = args.output_dir
    dest.mkdir(parents=True, exist_ok=True)
    if args.apply and not args.word_only:
        parser.error("--apply requires --word-only")
    all_weights = load_sampled_weights(
        args.model,
        torch.device("cuda"),
        dest,
        stages=("w13", "w2") if args.apply else ("w2",),
    )
    weights = all_weights["w2"]
    torch.manual_seed(20261001)
    a = torch.randint(-127, 128, (36, 2304), dtype=torch.int8, device="cuda")
    scale = torch.linspace(0.0007, 0.013, 36, device="cuda")
    kernels, compilation = {}, {}
    configs = [(3, 128, False), (2, 128, False), (4, 128, False), (3, 256, False)]
    if args.word_only:
        configs = [(3, 128, False), (3, 128, True)]
    for stages, threads, word_load in configs:
        name = f"s{stages}-t{threads}" + ("-word" if word_load else "")
        if name == "s3-t128" and "use_word_load" not in inspect.getsource(
            runtime._build_decode_gemm
        ):
            kernels[name] = runtime.compile_m6_w2(0)
            compilation[name] = {"live_runtime_kernel": True}
        else:
            kernels[name], compilation[name] = compile_variant(
                stages, threads, dest, word_load=word_load
            )
        print("compiled", name, flush=True)
    patterns = {
        "hot": torch.arange(6).repeat(6, 1),
        "mid": torch.arange(6).unsqueeze(0) + torch.arange(6).unsqueeze(1) * 3,
        "spread": torch.arange(36).view(6, 6),
    }
    results = []
    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
    for pattern, raw_ids in patterns.items():
        ids = raw_ids.to(device="cuda", dtype=torch.int32).add_(137)
        aligned = moe_align_block_size(ids, 16, 384)
        out = torch.empty((36, 5120), dtype=torch.bfloat16, device="cuda")
        inputs = (
            a,
            scale,
            weights["packed"],
            weights["shift2"],
            weights["channel_scale"],
            *aligned,
        )
        kernels["s3-t128"](*inputs, out)
        expected = out.clone()
        graphs = {}
        for name, kernel in kernels.items():
            kernel(*inputs, out)
            require_exact(out, expected)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                kernel(*inputs, out)
            out.fill_(float("nan"))
            graph.replay()
            require_exact(out, expected)
            graphs[name] = graph
        timings = time_graphs(graphs, args.samples)
        row = dict(
            pattern=pattern,
            active_experts=len(set(raw_ids.flatten().tolist())),
            exact=True,
            timing=timings,
        )
        results.append(row)
        print(
            pattern,
            {name: values["median_us"] for name, values in timings.items()},
            flush=True,
        )
        (dest / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()
    if args.apply:
        probe_apply(
            all_weights,
            kernels["s3-t128-word"],
            dest,
            runtime=True,
            sampled=True,
            baseline_runtime=True,
            baseline_runtime_kernel=kernels["s3-t128"],
        )
    metadata = dict(
        scope=(
            "live M6 W2 template; real layer20 expert weights; "
            "synthetic routes/INT8 A; cold L2 one-call graphs"
        ),
        gpu=str(torch.cuda.get_device_properties(0)),
        compilation=compilation,
        source_sha256=hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest(),
        timing=(
            "CUDA Events; three alternating rounds; 500ms settle per arm; "
            "64MiB flush outside events"
        ),
        cupti_python_available=False,
    )
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
