# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-pass A/B transport test for DSpark preallocated relay receives.

This benchmark uses the real GroupCoordinator tensor-dictionary transport and
synthetic DSpark-shaped tensors. It runs baseline first and the preallocated
receive path second for each batch, with no A2 repeat. It does not load a model
and therefore measures transport behavior only.
"""

import argparse
import json
import statistics
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.deepseek_v4_1.dspark_relay_bench import (  # noqa: E402
    _check,
    make_hop_packages,
)
from vllm.distributed.parallel_state import GroupCoordinator  # noqa: E402


def _recv(
    group: GroupCoordinator,
    src: int,
    recv_buffers: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor | object] | None:
    if recv_buffers is None:
        return group.recv_tensor_dict(src=src)
    tensor_dict, handles, postprocess = group.irecv_tensor_dict(
        src=src, recv_buffers=recv_buffers
    )
    for handle in handles:
        handle.wait()
    for callback in postprocess:
        callback()
    return tensor_dict


def _make_recv_buffers(batch: int, device: torch.device) -> dict[str, torch.Tensor]:
    rows = batch * 5
    return {
        "hidden_states": torch.empty((rows, 5120), dtype=torch.bfloat16, device=device),
        "residual": torch.empty((rows, 4, 5120), dtype=torch.bfloat16, device=device),
        "post_mix": torch.empty((rows, 4, 1), dtype=torch.float32, device=device),
        "res_mix": torch.empty((rows, 4, 4), dtype=torch.float32, device=device),
        "pre_mix": torch.empty((rows, 4), dtype=torch.float32, device=device),
    }


def _worker(rank: int, rendezvous: str, args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4)
    device_group = dist.new_group(backend="nccl")
    group = GroupCoordinator.__new__(GroupCoordinator)
    group.rank_in_group = rank
    group.world_size = 4
    group.ranks = list(range(4))
    group.cpu_group = dist.group.WORLD
    group.device_group = device_group
    group.use_cpu_custom_send_recv = False
    group._pending_isends = deque()

    def roundtrip(
        hops: list[dict[str, object]],
        step: int,
        recv_buffers: dict[str, torch.Tensor] | None,
        staging_buffers: dict[str, torch.Tensor] | None,
        verify: bool = False,
    ) -> None:
        outgoing = dict(hops[rank])
        outgoing["scheduler_step_id"] = step
        if rank == 0:
            group.send_tensor_dict(outgoing, dst=1)
            received = _recv(group, 3, recv_buffers)
        else:
            received = _recv(group, rank - 1, recv_buffers)
            if received is None or received["scheduler_step_id"] != step:
                raise AssertionError("relay step mismatch")
            group.send_tensor_dict(outgoing, dst=(rank + 1) % 4)
        if received is not None and staging_buffers is not None:
            for key, destination in staging_buffers.items():
                source = received.get(key)
                if isinstance(source, torch.Tensor):
                    destination[: source.shape[0]].copy_(source)
        if verify:
            _check(received, hops[(rank - 1) % 4], step)

    report: dict[str, object] = {
        "backend": "nccl",
        "device": str(device),
        "model_inference": False,
        "candidate": "preallocated-recv",
        "settings": vars(args),
        "correctness": [],
        "runs": [],
    }
    try:
        step = 0
        for batch in args.batches:
            hops = make_hop_packages(batch, compact=False, device=device)
            for enabled in (False, True):
                recv_buffers = _make_recv_buffers(batch, device) if enabled else None
                roundtrip(hops, step, recv_buffers, None, verify=True)
                dist.barrier()
                torch.accelerator.synchronize()
                if rank == 0:
                    report["correctness"].append(
                        {"batch": batch, "candidate_enabled": enabled}
                    )
                step += 1

            for enabled in (False, True):
                recv_buffers = _make_recv_buffers(batch, device) if enabled else None
                staging_buffers = None if enabled else _make_recv_buffers(batch, device)
                for _ in range(args.warmup):
                    roundtrip(hops, step, recv_buffers, staging_buffers)
                    step += 1
                dist.barrier()
                torch.accelerator.synchronize()
                times = []
                for _ in range(args.rounds):
                    dist.barrier()
                    torch.accelerator.synchronize()
                    start = time.perf_counter_ns()
                    roundtrip(hops, step, recv_buffers, staging_buffers)
                    torch.accelerator.synchronize()
                    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                    elapsed = torch.tensor(elapsed_ms, dtype=torch.float64)
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                    times.append(float(elapsed.item()))
                    step += 1
                dist.barrier()
                if rank == 0:
                    report["runs"].append(
                        {
                            "batch": batch,
                            "candidate_enabled": enabled,
                            "median_ms": statistics.median(times),
                            "round_ms": times,
                        }
                    )
                    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
                    label = "B" if enabled else "A1"
                    print(
                        f"B={batch} {label}: {statistics.median(times):.3f} ms",
                        flush=True,
                    )
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if min(args.batches) < 1 or max(args.batches) > 8:
        parser.error("Use batch sizes 1..8")
    if min(args.rounds, args.warmup) < 0:
        parser.error("rounds and warmup must be nonnegative")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dsv41-recv-buffer-") as directory:
        mp.spawn(
            _worker,
            args=(f"file://{directory}/rendezvous", args),
            nprocs=4,
            join=True,
        )


if __name__ == "__main__":
    main()
