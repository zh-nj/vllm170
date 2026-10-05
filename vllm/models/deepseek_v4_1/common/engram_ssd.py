# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""File-backed FP8 Engram rows with a bounded host FIFO cache."""

import hashlib
import json
import mmap
import os
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, suppress
from itertools import repeat
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
_LOAD_BYTES = 64 * 1024**2
_PARALLEL_MIN_ROWS = 64
_IO_PROBE_ROWS = 8
_PARALLEL_READ_NS = 20_000
_BATCH_MIN_ROWS = 512


def _checkpoint_identity(tensor: torch.Tensor) -> dict | None:
    """Identify a safetensors mapping without reading the entire checkpoint."""
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        return None
    base = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()
    try:
        for line in Path("/proc/self/maps").read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[5].endswith(".safetensors"):
                continue
            first, last = (int(x, 16) for x in fields[0].split("-"))
            if first <= base < base + size <= last:
                path = Path(fields[5])
                stat = path.stat()
                return {
                    "path": str(path.resolve()),
                    "inode": stat.st_ino,
                    "device": stat.st_dev,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "ctime_ns": stat.st_ctime_ns,
                    "offset": int(fields[2], 16) + base - first,
                }
    except OSError:
        pass
    return None


class SSDTable:
    """Own converted files and cache only requested packed rows.

    Files are reused only after both provenance and completion are checked.
    Open descriptors retain their generation if another worker replaces a file.
    """

    def __init__(
        self,
        path: str,
        key: str,
        rows: int,
        dim: int,
        block: int,
        start: int,
        cpu_cache_gb: float,
        scale_in_ram: bool = True,
        io_workers: int = 8,
        batch_reads: bool = False,
    ):
        if not 1 <= io_workers <= 32:
            raise ValueError("Engram SSD io_workers must be between 1 and 32")
        self.path = Path(path).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)
        self.rows, self.dim, self.scale_dim = rows, dim, dim // block
        identity = f"v1:{key}:{rows}:{dim}:{block}:{start}"
        self.cache_id = hashlib.sha256(identity.encode()).hexdigest()[:32]
        self.row_bytes = dim + self.scale_dim
        self.capacity = min(rows, int(cpu_cache_gb * 1024**3) // self.row_bytes)
        self.cache = torch.empty(
            (self.capacity, self.row_bytes),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=self.capacity > 0,
        )
        self.keys = np.full(self.capacity, -1, dtype=np.int64)
        self.index: dict[int, int] = {}
        self.next_slot = self.hits = self.misses = 0
        self._cache = self.cache.numpy()
        self._files: dict[str, BinaryIO] = {}
        self._owners = ExitStack()
        self._lock = threading.Lock()
        self._scale_in_ram = scale_in_ram
        self._scales: np.ndarray | None = None
        self._io_workers = io_workers
        self._io_pool: ThreadPoolExecutor | None = None
        self.batch_reads = batch_reads
        self._mapped_weight: np.ndarray | None = None
        self._prefetch_pool: ThreadPoolExecutor | None = None
        self._prefetch_lock = threading.Lock()
        self._closed = False

    def load(self, kind: str, shard: torch.Tensor) -> None:
        """Persist one shard in bounded chunks, or reuse its verified file."""
        import fcntl

        width = self.dim if kind == "weight" else self.scale_dim
        if tuple(shard.shape) != (self.rows, width):
            raise ValueError(f"Engram SSD {kind} has an incompatible shard shape")
        source = _checkpoint_identity(shard)
        expected = {"version": 1, "source": source, "shape": [self.rows, width]}
        data = self.path / f"{self.cache_id}.{kind}.u8"
        manifest = self.path / f"{self.cache_id}.{kind}.json"
        with self._lock, open(data.with_suffix(".lock"), "a+b") as lock:
            if kind in self._files:
                raise RuntimeError("Engram SSD weights cannot be reloaded in place")
            fcntl.flock(lock, fcntl.LOCK_EX)
            reuse = False
            if source is not None and manifest.exists() and data.exists():
                with suppress(OSError, ValueError):
                    reuse = (
                        json.loads(manifest.read_text()) == expected
                        and data.stat().st_size == self.rows * width
                    )
            if not reuse:
                manifest.unlink(missing_ok=True)
                self._sync_directory(self.path)
                self._write(data, shard.view(torch.uint8))
                # Publish completion only after the data has been fsynced.
                self._write_manifest(manifest, expected)
            file = self._owners.enter_context(open(data, "rb", buffering=0))  # noqa: SIM115
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_RANDOM)
            self._files[kind] = file
            if kind == "scale" and self._scale_in_ram:
                scales = np.empty((self.rows, width), dtype=np.uint8)
                if scales.size:
                    self._read_into(file.fileno(), memoryview(scales).cast("B"), 0)
                self._scales = scales
                logger.info(
                    "Engram SSD scales in CPU RAM: %.2f GiB", scales.nbytes / 1024**3
                )
            logger.info(
                "Engram SSD %s: %.2f GiB, %s",
                kind,
                self.rows * width / 1024**3,
                "reused" if reuse else "written",
            )

    @staticmethod
    def _read_into(fd: int, payload: memoryview, offset: int) -> None:
        while payload:
            count = os.preadv(fd, [payload[:_LOAD_BYTES]], offset)
            if not count:
                raise OSError("Short read from Engram SSD table")
            payload = payload[count:]
            offset += count

    def _read_rows(self, unique, raw, misses):
        weight_fd = self._files["weight"].fileno()
        scale_fd = self._files["scale"].fileno()
        if self._scales is not None:
            raw[misses, self.dim :] = self._scales[unique[misses]]

        def read(group):
            for i in group:
                row = int(unique[i])
                self._read_into(
                    weight_fd, memoryview(raw[i, : self.dim]), row * self.dim
                )
                if self._scales is None:
                    self._read_into(
                        scale_fd,
                        memoryview(raw[i, self.dim :]),
                        row * self.scale_dim,
                    )

        workers = min(self._io_workers, len(misses))
        if workers <= 1 or len(misses) < _PARALLEL_MIN_ROWS:
            read(misses)
            return
        # Threads lose on page-cache hits; probe before paying dispatch costs.
        start = time.perf_counter_ns()
        read(misses[:_IO_PROBE_ROWS])
        elapsed = time.perf_counter_ns() - start
        remaining = misses[_IO_PROBE_ROWS:]
        if elapsed < _IO_PROBE_ROWS * _PARALLEL_READ_NS:
            if (
                not self.batch_reads
                or self._scales is None
                or len(misses) < _BATCH_MIN_ROWS
            ):
                read(remaining)
                return
            if self._mapped_weight is None:
                mapping = self._owners.enter_context(
                    mmap.mmap(weight_fd, 0, access=mmap.ACCESS_READ)
                )
                mapping.madvise(mmap.MADV_RANDOM)
                self._mapped_weight = np.ndarray(
                    (self.rows, self.dim), dtype=np.uint8, buffer=mapping
                )

            mapped_weight = self._mapped_weight

            def read(group):
                raw[group, : self.dim] = mapped_weight[unique[group]]

        if self._io_pool is None:
            self._io_pool = ThreadPoolExecutor(
                max_workers=self._io_workers, thread_name_prefix="engram-ssd"
            )
        futures = [
            self._io_pool.submit(read, group)
            for group in np.array_split(remaining, workers)
        ]
        error = None
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error

    def prefetch(self, indices: np.ndarray) -> Future[np.ndarray]:
        """Read owned request IDs in a background thread without GPU staging."""
        with self._prefetch_lock:
            if self._closed:
                raise RuntimeError("Engram SSD table is closed")
            if self._prefetch_pool is None:
                self._prefetch_pool = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="engram-prefetch"
                )
            return self._prefetch_pool.submit(self.gather, indices.copy())

    @staticmethod
    def _sync_directory(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _write_manifest(path: Path, contents: dict) -> None:
        fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
        try:
            with os.fdopen(fd, "w") as output:
                json.dump(contents, output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(name, path)
            SSDTable._sync_directory(path.parent)
        finally:
            Path(name).unlink(missing_ok=True)

    @staticmethod
    def _write(path: Path, shard: torch.Tensor) -> None:
        from .engram import _engram_checkpoint_pageout

        width = shard.shape[1]
        release = _engram_checkpoint_pageout(shard)
        fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
        try:
            with os.fdopen(fd, "wb", buffering=0) as output:
                rows_per_chunk = max(1, _LOAD_BYTES // width)
                for start in range(0, len(shard), rows_per_chunk):
                    chunk = shard[start : start + rows_per_chunk].cpu().contiguous()
                    payload = memoryview(chunk.numpy()).cast("B")
                    offset = output.tell()
                    while payload:
                        count = output.write(payload)
                        if not count:
                            raise OSError("Short write to Engram SSD table")
                        payload = payload[count:]
                    os.fdatasync(output.fileno())
                    if hasattr(os, "posix_fadvise"):
                        os.posix_fadvise(
                            output.fileno(),
                            offset,
                            chunk.numel(),
                            os.POSIX_FADV_DONTNEED,
                        )
                    if release is not None:
                        release(start * width, chunk.numel())
                os.fsync(output.fileno())
            os.replace(name, path)
            SSDTable._sync_directory(path.parent)
        finally:
            Path(name).unlink(missing_ok=True)

    def gather(self, indices: np.ndarray) -> np.ndarray:
        """Return raw rows in request order; invalid/padded heads are zero."""
        if self._files.keys() != {"weight", "scale"}:
            raise RuntimeError("Engram SSD weight and scale must both be loaded")
        valid = (indices >= 0) & (indices < self.rows)
        unique, first, inverse = np.unique(
            indices[valid], return_index=True, return_inverse=True
        )
        with self._lock:
            slots = np.fromiter(
                map(self.index.get, unique.tolist(), repeat(-1)),
                dtype=np.int64,
                count=len(unique),
            )
            cached = slots >= 0
            misses = np.flatnonzero(~cached)
            if not len(misses):
                self.hits += len(unique)
                gathered = np.zeros((indices.size, self.row_bytes), dtype=np.uint8)
                if indices.size >= 512 and valid.all():
                    torch.index_select(
                        self.cache,
                        0,
                        torch.from_numpy(slots[inverse]),
                        out=torch.from_numpy(gathered),
                    )
                else:
                    gathered[valid] = self._cache[slots[inverse]]
                return gathered
            raw = np.zeros((len(unique), self.row_bytes), dtype=np.uint8)
            raw[cached] = self._cache[slots[cached]]
            self._read_rows(unique, raw, misses)
            if not self.capacity:
                self.misses += len(unique)
            else:
                for i in np.argsort(first):
                    row = int(unique[i])
                    slot = self.index.get(row)
                    if slot is not None:
                        self.hits += 1
                        continue
                    self.misses += 1
                    slot = self.next_slot
                    old = int(self.keys[slot])
                    if old >= 0:
                        del self.index[old]
                    self.keys[slot] = row
                    self.index[row] = slot
                    self._cache[slot] = raw[i]
                    self.next_slot = (slot + 1) % self.capacity
        gathered = np.zeros((indices.size, self.row_bytes), dtype=np.uint8)
        gathered[valid] = raw[inverse]
        return gathered

    def close(self) -> None:
        with self._prefetch_lock:
            self._closed = True
            pool = self._prefetch_pool
            self._prefetch_pool = None
        if pool is not None:
            pool.shutdown(wait=True)
        with self._lock:
            if self._io_pool is not None:
                self._io_pool.shutdown(wait=True)
                self._io_pool = None
            self._mapped_weight = None
            self._owners.close()
            self._files.clear()
            self._scales = None

    def __del__(self):
        if hasattr(self, "_owners"):
            self.close()
