# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Break down the Engram host-table load path (disk read vs copy vs pageout).

The port's loader (``common/engram.py:_engram_head_shard_weight_loader``) walks
the checkpoint with safetensors' mmap and, per 64 MiB chunk, copies into a
deferred anonymous mapping, ``cudaHostRegister``s the chunk and
``MADV_PAGEOUT``s the consumed source pages. This probe measures each leg on
the real checkpoint so the observed ~120 s/rank can be attributed.

CUDA registration needs a device, so it is not measured here; the reported
legs are a lower bound for the full path.

Usage:
  .venv/bin/python benchmarks/deepseek_v4_1/engram_load_probe.py \
      --model /media/z/data6/models/DeepSeek-V4.1-Flash --gib 8 --drop
"""

import argparse
import ctypes
import mmap
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

GiB = 1024**3
MADV_PAGEOUT = 21
MADV_WILLNEED = 3
POSIX_FADV_DONTNEED = 4
CHUNK = 64 * 1024**2
PAGE = 4096
libc = ctypes.CDLL(None, use_errno=True)
libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.posix_fadvise.argtypes = [ctypes.c_int, ctypes.c_long, ctypes.c_long, ctypes.c_int]


def report(name, seconds, gib, note=""):
    print(f"{name:<50} {seconds:8.2f} s {gib / seconds:7.2f} GiB/s  {note}", flush=True)


def timed(name, gib, fn, note=""):
    start = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - start
    report(name, elapsed, gib, note)
    return elapsed


def drop_cache(fd, offset, length):
    libc.posix_fadvise(fd, offset, length, POSIX_FADV_DONTNEED)
    libc.posix_fadvise(fd, offset, length, POSIX_FADV_DONTNEED)


def page_touch(view, length):
    """Force one fault per 4 KiB page with numpy (negligible Python overhead)."""
    return view[:length:PAGE].sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--gib", type=float, default=8.0)
    parser.add_argument("--tensor", default="model-00047-of-00048.safetensors")
    parser.add_argument("--drop", action="store_true")
    args = parser.parse_args()

    path = os.path.join(args.model, args.tensor)
    length = int(args.gib * GiB)
    gib = length / GiB
    print(f"file  {path}\nwindow {gib:.2f} GiB\n" + "-" * 82, flush=True)
    fd = os.open(path, os.O_RDONLY)

    mapping = mmap.mmap(fd, length, prot=mmap.PROT_READ, flags=mmap.MAP_SHARED)
    src = np.frombuffer(mapping, dtype=np.uint8)
    src_ptr = src.ctypes.data

    dst_map = mmap.mmap(-1, length, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    dst = np.frombuffer(dst_map, dtype=np.uint8)
    dst_ptr = dst.ctypes.data

    if args.drop:
        drop_cache(fd, 0, length)
        print("source page cache dropped\n", flush=True)

    # 1. What safetensors does today: mmap + demand faults, single thread.
    timed("mmap demand read, cold (safetensors path)", gib, lambda: page_touch(src, length))
    timed("mmap demand read, page-cache warm         ", gib, lambda: page_touch(src, length))

    # 2. Large-block pread + memmove into a fresh anonymous mapping, 64 MiB chunks
    #    (the shape the loader already uses, minus fault-driven source reads).
    def pread_copy():
        buf = np.empty(CHUNK, dtype=np.uint8)
        offset = 0
        while offset < length:
            want = min(CHUNK, length - offset)
            got = os.preadv(fd, [memoryview(buf)[:want]], offset)
            if got <= 0:
                break
            ctypes.memmove(dst_ptr + offset, buf.ctypes.data, got)
            offset += got

    drop_cache(fd, 0, length)
    timed("pread(64 MiB)+memmove, cold source        ", gib, pread_copy)
    drop_cache(fd, 0, length)
    timed("pread(64 MiB)+memmove, cold source (2nd)  ", gib, pread_copy)

    # 3. Copy legs isolated, page-cache-warm source.
    timed(
        "memmove warm src -> pre-faulted dst        ",
        gib,
        lambda: ctypes.memmove(dst_ptr, src_ptr, length),
        "pure copy",
    )
    del dst
    del dst_map
    dst_map = mmap.mmap(-1, length, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    dst = np.frombuffer(dst_map, dtype=np.uint8)
    dst_ptr = dst.ctypes.data
    timed(
        "first touch of fresh anon dst (zero-fill)  ",
        gib,
        lambda: dst.fill(0),
        "what cudaHostRegister pays",
    )
    timed(
        "chunked copy warm->anon in 64 MiB chunks   ",
        gib,
        lambda: [
            ctypes.memmove(
                dst_ptr + o, src_ptr + o, min(CHUNK, length - o)
            )
            for o in range(0, length, CHUNK)
        ],
    )

    # 4. Reclaim leg.
    for offset in range(0, length, CHUNK):
        page_touch(src, min(CHUNK, length - offset))
    timed(
        "MADV_PAGEOUT source, 64 MiB chunks         ",
        gib,
        lambda: [
            libc.madvise(src_ptr + o, min(CHUNK, length - o), MADV_PAGEOUT)
            for o in range(0, length, CHUNK)
        ],
    )

    # 5. Parallel copy (pre-faulted dst, warm src).
    for workers in (2, 4, 8):
        def shard_copy(worker, workers=workers):
            per = (length + workers - 1) // workers
            start, stop = worker * per, min((worker + 1) * per, length)
            for offset in range(start, stop, CHUNK):
                size = min(CHUNK, stop - offset)
                ctypes.memmove(dst_ptr + offset, src_ptr + offset, size)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            timed(
                f"parallel memmove, {workers} threads".ljust(42),
                gib,
                lambda: list(pool.map(shard_copy, range(workers))),
                "warm src",
            )

    del src, dst, mapping, dst_map
    os.close(fd)
    print("-" * 82)


if __name__ == "__main__":
    main()
