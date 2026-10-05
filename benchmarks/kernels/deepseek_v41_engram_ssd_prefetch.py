# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen batched reads and same-step prefetch on existing full SSD tables.

Files are opened read-only. The GPU overlap uses independent synthetic GEMMs,
not a PP serving run. Cold samples evict only the requested file pages.
"""

import argparse
import hashlib
import json
import mmap
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from benchmarks.kernels.deepseek_v41_engram_ssd import _read_bytes
from vllm.models.deepseek_v4_1.common import engram as impl
from vllm.models.deepseek_v4_1.common.engram_ssd import SSDTable


class MappedTable(SSDTable):
    """Offline alternative: gather file-backed rows in NumPy batches."""

    def bind_mapping(self):
        self.mapping = mmap.mmap(
            self._files["weight"].fileno(), 0, access=mmap.ACCESS_READ
        )
        self.mapping.madvise(mmap.MADV_RANDOM)
        self.mapped_rows = np.ndarray(
            (self.rows, self.dim), dtype=np.uint8, buffer=self.mapping
        )

    def _read_rows(self, unique, raw, misses):
        raw[misses, self.dim :] = self._scales[unique[misses]]

        def read(group):
            raw[group, : self.dim] = self.mapped_rows[unique[group]]

        if self._io_workers == 1 or len(misses) < 64:
            read(misses)
        else:
            if self._io_pool is None:
                self._io_pool = ThreadPoolExecutor(max_workers=self._io_workers)
            futures = [
                self._io_pool.submit(read, group)
                for group in np.array_split(misses, self._io_workers)
            ]
            errors = []
            for future in futures:
                try:
                    future.result()
                except Exception as error:
                    errors.append(error)
            if errors:
                raise errors[0]

    def close(self):
        super().close()
        if hasattr(self, "mapped_rows"):
            del self.mapped_rows
            self.mapping.close()


def _source_reference(table, ids):
    packed = np.zeros((ids.size, table.row_bytes), dtype=np.uint8)
    for index, row in enumerate(ids.ravel()):
        if 0 <= row < table.rows:
            payload = os.pread(
                table._files["weight"].fileno(), table.dim, int(row) * table.dim
            )
            if len(payload) != table.dim:
                raise OSError("Short reference read")
            packed[index, : table.dim] = np.frombuffer(payload, dtype=np.uint8)
            packed[index, table.dim :] = table._scales[row]
    return packed


def _timed_read(table, ids):
    start = time.perf_counter_ns()
    packed = table.gather(ids)
    return packed, (time.perf_counter_ns() - start) / 1e6


def _evict_requested(tables, ids):
    for table in tables:
        if isinstance(table, MappedTable):
            table.mapping.madvise(mmap.MADV_DONTNEED)
    table = tables[0]
    pages = np.unique(ids[ids >= 0] * table.dim // 4096)
    for page in pages:
        os.posix_fadvise(
            table._files["weight"].fileno(),
            int(page) * 4096,
            4096,
            os.POSIX_FADV_DONTNEED,
        )


def _make_work(target_ms):
    left = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)
    right = torch.randn_like(left)
    output = torch.empty_like(left)
    for _ in range(4):
        torch.mm(left, right, out=output)
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(16):
        torch.mm(left, right, out=output)
    end.record()
    end.synchronize()
    calls = max(1, round(target_ms / (start.elapsed_time(end) / 16)))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            torch.mm(left, right, out=output)
    graph.replay()
    torch.accelerator.synchronize()
    expected = output.clone()
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return graph, (left, right), output, expected, start.elapsed_time(end), calls


def _stage(embedding, packed, output):
    if embedding._ssd_copy_pending:
        embedding._ssd_copy_done.synchronize()
    rows = len(packed)
    np.copyto(embedding._ssd_host_weight[:rows].numpy(), packed[:, : embedding.dim])
    np.copyto(embedding._ssd_host_scales[:rows].numpy(), packed[:, embedding.dim :])
    embedding._ssd_device_weight[:rows].copy_(
        embedding._ssd_host_weight[:rows], non_blocking=True
    )
    embedding._ssd_device_scales[:rows].copy_(
        embedding._ssd_host_scales[:rows], non_blocking=True
    )
    grid = min((rows + 15) // 16, embedding._num_sms)
    impl._engram_lookup_kernel[(grid,)](
        embedding._ssd_device_weight,
        embedding._ssd_device_scales,
        embedding._ssd_row_ids,
        output,
        0,
        rows,
        rows,
        embedding.part_n_hash_cols,
        1,
        HEAD_START=0,
        LOCAL_HEADS=embedding.part_n_hash_cols,
        TOTAL_HEADS=embedding.part_n_hash_cols,
        DIM=embedding.dim,
        QUANT_BLOCK=embedding.block_size,
        BLOCK_R=16,
        GRID=grid,
    )
    embedding._ssd_copy_done.record()
    embedding._ssd_copy_pending = True


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache-id", default="8a41f086fe2be055ca98af465b3d2d9b")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.output.exists():
        parser.error("Positive repeats and a fresh output path are required")
    torch.set_num_threads(1)
    torch.manual_seed(20261005)
    manifest_path = args.directory / f"{args.cache_id}.weight.json"
    manifest = json.loads(manifest_path.read_text())
    rows, dim = manifest["shape"]
    source = manifest["source"]
    source_stat = Path(source["path"]).stat()
    assert source_stat.st_ino == source["inode"]
    assert source_stat.st_size == source["size"]
    assert source_stat.st_mtime_ns == source["mtime_ns"]
    scale_path = args.directory / f"{args.cache_id}.scale.u8"
    scales = np.fromfile(scale_path, dtype=np.uint8).reshape(rows, dim // 32)
    config = json.loads((args.model / "config.json").read_text())
    layout = impl.EngramLayout(SimpleNamespace(**config.get("text_config", config)))
    sizes = tuple(p for group in layout.primes[0] for p in group)
    assert sum(sizes) == rows
    impl.get_tensor_model_parallel_world_size = lambda: 1
    impl.get_tensor_model_parallel_rank = lambda: 0
    embedding = impl.ParallelEngramEmbedding(
        rows,
        dim,
        sizes,
        cpu_offload=True,
        ssd_offload_path=str(args.directory),
        max_tokens=2044,
    )
    tables = {}
    for name, cls, workers in (
        ("pread_adaptive", SSDTable, 8),
        ("mmap_batch_serial", MappedTable, 1),
        ("mmap_batch_8", MappedTable, 8),
    ):
        table = cls(
            str(args.directory),
            "readonly-benchmark",
            rows,
            dim,
            32,
            0,
            0,
            io_workers=workers,
        )
        for kind in ("weight", "scale"):
            path = args.directory / f"{args.cache_id}.{kind}.u8"
            file = table._owners.enter_context(open(path, "rb"))  # noqa: SIM115
            os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_RANDOM)
            table._files[kind] = file
        table._scales = scales
        if isinstance(table, MappedTable):
            table.bind_mapping()
        tables[name] = table
    results = {
        "scope": "Full-table component screening; synthetic GPU work, no PP service",
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_home": os.environ.get("CUDA_HOME"),
        "table_rows": rows,
        "table_bytes": rows * dim,
        "scales_ram_bytes": scales.nbytes,
        "repeats": args.repeats,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "read_policy": "POSIX_FADV_RANDOM; mmap also uses MADV_RANDOM",
        "cases": [],
        "overlap": [],
    }
    rng = np.random.default_rng(20261005)
    with ThreadPoolExecutor(max_workers=1) as prefetch:
        for tokens in (6, 2044):
            ids = np.stack([rng.integers(s, size=tokens) for s in sizes], axis=1)
            ids += np.cumsum((0, *sizes[:-1]))
            ids[-1, -1] = -1
            reference = _source_reference(tables["pread_adaptive"], ids)
            for table in tables.values():
                np.testing.assert_array_equal(table.gather(ids.ravel()), reference)
            for state in ("warm", "cold"):
                samples = {name: [] for name in tables}
                for repeat in range(args.repeats):
                    names = list(tables)
                    if repeat % 2:
                        names.reverse()
                    for name in names:
                        if state == "cold":
                            _evict_requested(list(tables.values()), ids)
                        else:
                            tables[name].gather(ids.ravel())
                        read_start = _read_bytes()
                        start = time.perf_counter_ns()
                        packed = tables[name].gather(ids.ravel())
                        wall_ms = (time.perf_counter_ns() - start) / 1e6
                        physical_bytes = _read_bytes() - read_start
                        np.testing.assert_array_equal(packed, reference)
                        samples[name].append(
                            {"ms": wall_ms, "read_bytes": physical_bytes}
                        )
                row = {
                    "tokens": tokens,
                    "heads": len(sizes),
                    "state": state,
                    "samples": samples,
                    "median_ms": {
                        n: statistics.median(s["ms"] for s in v)
                        for n, v in samples.items()
                    },
                }
                results["cases"].append(row)
                args.output.write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps(row), flush=True)

            graph, work_inputs, gpu_output, gpu_ref, work_ms, calls = _make_work(
                6 if tokens == 6 else 100
            )
            assert all(tensor.is_cuda for tensor in work_inputs)
            device_ids = torch.from_numpy(ids.astype(np.int32)).cuda()
            output = torch.empty(
                tokens, len(sizes), dim, dtype=torch.bfloat16, device="cuda"
            )
            raw = torch.from_numpy(reference.copy()).cuda()
            expected = raw[:, :dim].contiguous().view(
                torch.float8_e4m3fn
            ).float() * torch.exp2(raw[:, dim:].float() - 127).repeat_interleave(32, 1)
            expected = expected.to(torch.bfloat16).reshape_as(output)
            _stage(embedding, reference, output)
            torch.accelerator.synchronize()
            torch.testing.assert_close(output, expected, rtol=0, atol=0, equal_nan=True)
            for name in ("pread_adaptive", "mmap_batch_serial"):
                table = tables[name]
                samples = {"serial": [], "prefetch": []}
                details = {"serial": [], "prefetch": []}
                for repeat in range(args.repeats):
                    modes = (
                        ("serial", "prefetch")
                        if repeat % 2 == 0
                        else ("prefetch", "serial")
                    )
                    for mode in modes:
                        table.gather(ids.ravel())
                        torch.accelerator.synchronize()
                        start = time.perf_counter_ns()
                        local_ids = device_ids.cpu().numpy().astype(np.int64).ravel()

                        if mode == "prefetch":
                            pending = prefetch.submit(_timed_read, table, local_ids)
                        else:
                            packed, read_ms = _timed_read(table, local_ids)
                        launch_start = time.perf_counter_ns()
                        graph.replay()
                        launch_ms = (time.perf_counter_ns() - launch_start) / 1e6
                        join_ms = 0.0
                        if mode == "prefetch":
                            join_start = time.perf_counter_ns()
                            packed, read_ms = pending.result()
                            join_ms = (time.perf_counter_ns() - join_start) / 1e6
                        stage_start = time.perf_counter_ns()
                        _stage(embedding, packed, output)
                        stage_ms = (time.perf_counter_ns() - stage_start) / 1e6
                        torch.accelerator.synchronize()
                        wall_ms = (time.perf_counter_ns() - start) / 1e6
                        samples[mode].append(wall_ms)
                        details[mode].append(
                            {
                                "wall_ms": wall_ms,
                                "read_ms": read_ms,
                                "graph_host_ms": launch_ms,
                                "join_wait_ms": join_ms,
                                "stage_host_ms": stage_ms,
                            }
                        )
                        torch.testing.assert_close(
                            output, expected, rtol=0, atol=0, equal_nan=True
                        )
                        torch.testing.assert_close(gpu_output, gpu_ref, rtol=0, atol=0)
                row = {
                    "tokens": tokens,
                    "reader": name,
                    "state": "warm",
                    "gpu_work_ms": work_ms,
                    "gemm_calls": calls,
                    "samples_ms": samples,
                    "details": details,
                    "median_ms": {n: statistics.median(v) for n, v in samples.items()},
                }
                results["overlap"].append(row)
                args.output.write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps(row), flush=True)
    for table in tables.values():
        table.close()
    embedding.ssd_table.close()
    results["all_outputs_exact"] = True
    results["files_opened_readonly"] = True
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
