# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded SSD Engram A/B including D2H, file reads, H2D and dequantization.

Uses a real checkpoint prefix with an artificial 24-head layout. Cold samples
request file-page eviction and record process read_bytes; they do not drop the
machine's page cache. This is a component benchmark, not PP serving latency.
"""

import argparse
import cProfile
import hashlib
import importlib.util
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from vllm.models.deepseek_v4_1.common import engram as impl
from vllm.models.deepseek_v4_1.common import engram_ssd
from vllm.platforms import current_platform


def _read_bytes() -> int:
    for line in Path("/proc/self/io").read_text().splitlines():
        if line.startswith("read_bytes:"):
            return int(line.split()[1])
    raise RuntimeError("Process read_bytes not available")


def _baseline_class(path):
    spec = importlib.util.spec_from_file_location(
        "vllm.models.deepseek_v4_1.common.engram_ssd_before", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Before(module.SSDTable):
        def __init__(self, *args, **kwargs):
            kwargs.pop("scale_in_ram", None)
            kwargs.pop("io_workers", None)
            kwargs.pop("batch_reads", None)
            super().__init__(*args, **kwargs)

    return Before


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("/media/z/data6/models/DeepSeek-V4.1-Flash")
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--directory", type=Path, default=Path("/media/z/data6"))
    parser.add_argument("--rows-per-head", type=int, default=65536)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["before", "serial_ram", "auto_ram", "auto_disk"],
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cpu-cache-gb", type=float, default=16 / 1024)
    parser.add_argument(
        "--case", action="append", help="Limit to state:tokens, e.g. row_cache_hit:2044"
    )
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rows_per_head <= 0 or args.repeats <= 0:
        raise ValueError("rows and repeats must be positive")
    impl.get_tensor_model_parallel_world_size = lambda: 1
    impl.get_tensor_model_parallel_rank = lambda: 0
    torch.set_num_threads(1)
    heads, dim = 24, 256
    rows = heads * args.rows_per_head
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    key = "layers.1.engram.embed"
    before = _baseline_class(args.baseline)
    current = engram_ssd.SSDTable
    settings = {
        "before": (before, {}),
        "serial_ram": (current, {"ssd_scale_in_ram": True, "ssd_io_workers": 1}),
        "auto_ram": (current, {"ssd_scale_in_ram": True, "ssd_io_workers": 8}),
        "auto_disk": (current, {"ssd_scale_in_ram": False, "ssd_io_workers": 8}),
    }
    rng = np.random.default_rng(20261005)
    results = []
    with tempfile.TemporaryDirectory(
        dir=args.directory, prefix="engram-speed-"
    ) as directory:
        embeddings = {}
        with safe_open(
            args.model / index[key + ".weight"], framework="pt", device="cpu"
        ) as source:
            weight = source.get_tensor(key + ".weight")[:rows]
            scale = source.get_tensor(key + ".scale")[:rows]
            for name in args.variants:
                cls, extra = settings[name]
                engram_ssd.SSDTable = cls
                try:
                    with torch.device("cuda"):
                        layer = impl.ParallelEngramEmbedding(
                            rows,
                            dim,
                            (args.rows_per_head,) * heads,
                            cpu_offload=True,
                            ssd_offload_path=directory,
                            cache_key="bounded-real-checkpoint",
                            cpu_cache_gb=args.cpu_cache_gb,
                            max_tokens=2048,
                            **extra,
                        )
                    layer.weight.weight_loader(layer.weight, weight)
                    layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, scale)
                    embeddings[name] = layer
                finally:
                    engram_ssd.SSDTable = current

            # Independent requested-row reference, not another SSD reader.
            def reference(ids):
                raw_ids = ids.cpu().long().flatten()
                w = (
                    weight.view(torch.uint8)[raw_ids]
                    .cuda()
                    .view(torch.float8_e4m3fn)
                    .float()
                )
                s = scale.view(torch.uint8)[raw_ids].cuda().float()
                return (
                    (w * torch.exp2(s - 127).repeat_interleave(32, dim=1))
                    .to(torch.bfloat16)
                    .reshape(ids.shape[0], heads, dim)
                )

            for cache_state, tokens in (
                ("cold_file", 1),
                ("cold_file", 6),
                ("cold_file", 128),
                ("warm_file", 1),
                ("warm_file", 6),
                ("warm_file", 2044),
                ("row_cache_hit", 1),
                ("row_cache_hit", 6),
                ("row_cache_hit", 2044),
            ):
                if not args.cpu_cache_gb and cache_state == "row_cache_hit":
                    continue
                if args.case and f"{cache_state}:{tokens}" not in args.case:
                    continue
                ids = torch.from_numpy(
                    rng.integers(
                        0, args.rows_per_head, size=(tokens, heads), dtype=np.int32
                    )
                    + np.arange(heads, dtype=np.int32) * args.rows_per_head
                ).cuda()
                ref = reference(ids)
                outputs = {
                    name: torch.empty(
                        tokens, heads, dim, dtype=torch.bfloat16, device="cuda"
                    )
                    for name in args.variants
                }
                for name in args.variants:
                    embeddings[name].lookup(ids, outputs[name])
                    torch.accelerator.synchronize()
                    torch.testing.assert_close(outputs[name], ref, rtol=0, atol=0)
                samples = {name: [] for name in args.variants}
                io_bytes = {name: [] for name in args.variants}
                # Alternate order so all variants see both early and late samples.
                for repeat in range(args.repeats):
                    order = (
                        args.variants
                        if repeat % 2 == 0
                        else list(reversed(args.variants))
                    )
                    for name in order:
                        embedding = embeddings[name]
                        table = embedding.ssd_table
                        if cache_state != "row_cache_hit":
                            table.index.clear()
                            table.keys.fill(-1)
                            table.next_slot = 0
                        if cache_state == "cold_file":
                            for file in table._files.values():
                                os.posix_fadvise(
                                    file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED
                                )
                        else:
                            # Warm the file pages without retaining application rows.
                            if cache_state == "warm_file":
                                embedding.lookup(ids, outputs[name])
                                table.index.clear()
                                table.keys.fill(-1)
                                table.next_slot = 0
                        torch.accelerator.synchronize()
                        read_start = _read_bytes()
                        start = time.perf_counter_ns()
                        embedding.lookup(ids, outputs[name])
                        torch.accelerator.synchronize()
                        elapsed = (time.perf_counter_ns() - start) / 1e6
                        io_bytes[name].append(_read_bytes() - read_start)
                        samples[name].append(elapsed)
                result = {
                    "state": cache_state,
                    "tokens": tokens,
                    "heads": heads,
                    "samples_ms": samples,
                    "median_ms": {
                        name: statistics.median(values)
                        for name, values in samples.items()
                    },
                    "median_process_read_bytes": {
                        name: statistics.median(values)
                        for name, values in io_bytes.items()
                    },
                }
                results.append(result)
                print(json.dumps(result), flush=True)
            if args.profile:
                profile = cProfile.Profile()
                with profile:
                    embeddings[args.variants[-1]].lookup(
                        ids, outputs[args.variants[-1]]
                    )
                    torch.accelerator.synchronize()
                profile.dump_stats(args.profile)
            for layer in embeddings.values():
                layer.ssd_table.close()
    payload = {
        "scope": (
            "real checkpoint prefix, artificial 24-head TP1 layout; "
            "complete SSD lookup wall time, not full-model serving"
        ),
        "gpu_name": current_platform.get_device_name(0),
        "gpu_uuid": current_platform.get_device_uuid(0),
        "model": str(args.model),
        "rows": rows,
        "dim": dim,
        "temporary_ssd_bytes": rows * (dim + dim // 32),
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "repeats": args.repeats,
        "all_variants_bitwise_exact": True,
        "cpu_cache_gb": args.cpu_cache_gb,
        "temporary_tables_removed": True,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
