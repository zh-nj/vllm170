# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transport-only measurements for DSV41F DSpark relay candidates.

Runs four local processes (head and three draft stages) using the real vLLM
tensor-dictionary transport. Gloo uses CPU tensors. NCCL uses CUDA tensors and
Gloo metadata, matching the serving transport split. Synthetic tensors match
the relay shapes/dtypes; no model kernels run. Results do not predict serving
speedups.
"""

import argparse
import json
import statistics
import sys
import tempfile
import time
from collections import deque
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vllm.distributed.parallel_state import (  # noqa: E402
    GroupCoordinator,
    _split_tensor_dict,
)

SAMPLING_KEYS = (
    "query_input_ids",
    "sample_indices",
    "sample_pos",
    "sample_idx_mapping",
)
CONTEXT_KEYS = (
    "context_main_x",
    "num_sampled",
    "num_rejected",
    "last_sampled",
    "next_prefill_tokens",
    "temperature",
    "seeds",
)
STATE_BF16_KEYS = ("hidden_states", "residual")
STATE_FP32_KEYS = ("post_mix", "res_mix", "pre_mix")


def _pack_state_tensors(package):
    package["relay_hidden_residual"] = torch.cat(
        [package.pop(key).reshape(-1) for key in STATE_BF16_KEYS]
    )
    package["relay_mix"] = torch.cat(
        [package.pop(key).reshape(-1) for key in STATE_FP32_KEYS]
    )


def make_hop_packages(
    batch,
    compact,
    hidden=5120,
    hc_mult=4,
    width=5,
    device="cpu",
    flat_state=False,
):
    """Independent wire fixture; compare its schema with the runtime in tests."""
    q = batch * width
    package = {
        "scheduler_step_id": 0,
        "dummy_run": False,
        "req_ids": tuple(f"request-{i}" for i in reversed(range(batch))),
        "num_reqs": batch,
        "skip_attn": False,
        "idx_mapping": torch.arange(batch - 1, -1, -1, device=device),
        "context_main_x": torch.full(
            (batch * (width + 1), hidden),
            1.0,
            dtype=torch.bfloat16,
            device=device,
        ),
        "num_sampled": torch.ones(batch, dtype=torch.int32, device=device),
        "num_rejected": torch.zeros(batch, dtype=torch.int32, device=device),
        "last_sampled": torch.arange(8, dtype=torch.int64, device=device),
        "next_prefill_tokens": torch.zeros(8, dtype=torch.int32, device=device),
        "temperature": torch.zeros(8, dtype=torch.float32, device=device),
        "seeds": torch.arange(8, dtype=torch.int64, device=device),
    }
    hops = [dict(package)]
    for stage in range(3):
        package.update(
            hidden_states=torch.full(
                (q, hidden), stage + 2.0, dtype=torch.bfloat16, device=device
            ),
            residual=torch.full(
                (q, hc_mult, hidden),
                stage + 3.0,
                dtype=torch.bfloat16,
                device=device,
            ),
            post_mix=torch.full((q, hc_mult, 1), stage + 4.0, device=device),
            res_mix=torch.full((q, hc_mult, hc_mult), stage + 5.0, device=device),
            pre_mix=torch.full((q, hc_mult), stage + 6.0, device=device),
            query_input_ids=torch.arange(q, dtype=torch.int32, device=device),
            sample_indices=torch.arange(q, dtype=torch.int64, device=device),
            sample_pos=torch.arange(q, dtype=torch.int64, device=device) + 100,
            sample_idx_mapping=torch.arange(
                batch - 1, -1, -1, dtype=torch.int32, device=device
            ).repeat_interleave(width),
        )
        if compact and stage < 2:
            for key in SAMPLING_KEYS:
                package.pop(key)
        if flat_state:
            _pack_state_tensors(package)
        if stage == 2:
            for key in CONTEXT_KEYS:
                package.pop(key)
        hops.append(dict(package))
    return hops


def wire_inventory(hops):
    records = []
    for hop in hops:
        metadata, tensors = _split_tensor_dict(hop)
        records.append(
            {
                "tensor_messages": len(tensors),
                "tensor_bytes": sum(t.numel() * t.element_size() for t in tensors),
            }
        )
    return {
        "hops": records,
        "tensor_messages": sum(r["tensor_messages"] for r in records),
        "tensor_bytes": sum(r["tensor_bytes"] for r in records),
    }


def _check(actual, expected, step):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
        else:
            assert actual[key] == (step if key == "scheduler_step_id" else value), key


def _worker(rank, rendezvous, args):
    torch.set_num_threads(1)
    device = torch.device("cpu")
    if args.backend == "nccl":
        torch.accelerator.set_device_index(rank)
        device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=90),
    )
    device_group = dist.group.WORLD
    if args.backend == "nccl":
        device_group = dist.new_group(backend="nccl", timeout=timedelta(seconds=90))
    # Only the transport fields are needed. Avoid GroupCoordinator's device
    # discovery/communicator initialization, which belongs to model serving.
    group = GroupCoordinator.__new__(GroupCoordinator)
    group.rank_in_group = rank
    group.world_size = 4
    group.ranks = list(range(4))
    group.cpu_group = dist.group.WORLD
    group.device_group = device_group
    group.use_cpu_custom_send_recv = False
    group._pending_isends = deque()

    def roundtrip(hops, step, verify=False):
        outgoing = dict(hops[rank])
        outgoing["scheduler_step_id"] = step
        if rank == 0:
            group.send_tensor_dict(outgoing, dst=1)
            received = group.recv_tensor_dict(src=3)
        else:
            received = group.recv_tensor_dict(src=rank - 1)
            if received["scheduler_step_id"] != step:
                raise AssertionError("relay step mismatch")
            group.send_tensor_dict(outgoing, dst=(rank + 1) % 4)
        if verify:
            _check(received, hops[(rank - 1) % 4], step)

    report = {
        "backend": args.backend,
        "device": device.type,
        "model_inference": False,
        "candidate": args.candidate,
        "settings": vars(args),
        "correctness": [],
        "runs": [],
    }
    try:
        step = 0
        if args.transport_only:
            for batch in args.batches:
                hops = make_hop_packages(batch, compact=True, device=device)
                roundtrip(hops, step, verify=True)
                if rank == 0:
                    report["correctness"].append(
                        {"batch": batch, "step": step, "all_fields_equal": True}
                    )
                step += 1
                dist.barrier()
                for repeat in range(args.repeats):
                    for _ in range(args.warmup):
                        roundtrip(hops, step)
                        step += 1
                    dist.barrier()
                    times = []
                    for _ in range(args.rounds):
                        start = time.perf_counter_ns()
                        roundtrip(hops, step)
                        times.append((time.perf_counter_ns() - start) / 1e6)
                        step += 1
                    roundtrip(hops, step, verify=True)
                    if rank == 0:
                        report["correctness"].append(
                            {"batch": batch, "step": step, "all_fields_equal": True}
                        )
                    step += 1
                    dist.barrier()
                    if rank == 0:
                        record = {
                            "batch": batch,
                            "repeat": repeat,
                            "label": "compact-transport",
                            "median_ms": statistics.median(times),
                            "round_ms": times,
                            "inventory": wire_inventory(hops),
                        }
                        report["runs"].append(record)
                        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
                        print(
                            f"B={batch} repeat={repeat} transport: "
                            f"{record['median_ms']:.3f} ms",
                            flush=True,
                        )
            return
        for batch in (1, 2, 8, 6, 1):
            for candidate_enabled in (False, True):
                compact = args.candidate == "flat-state" or candidate_enabled
                roundtrip(
                    make_hop_packages(
                        batch,
                        compact,
                        device=device,
                        flat_state=(
                            args.candidate == "flat-state" and candidate_enabled
                        ),
                    ),
                    step,
                    verify=True,
                )
                dist.barrier()
                report["correctness"].append(
                    {
                        "batch": batch,
                        "compact": compact,
                        "candidate_enabled": candidate_enabled,
                        "step": step,
                        "all_fields_equal": True,
                    }
                )
                step += 1
        for batch in args.batches:
            variants = {
                enabled: make_hop_packages(
                    batch,
                    args.candidate == "flat-state" or enabled,
                    device=device,
                    flat_state=(args.candidate == "flat-state" and enabled),
                )
                for enabled in (False, True)
            }
            for repeat in range(args.repeats):
                for label, enabled in (
                    ("A1", False),
                    ("B", True),
                    ("A2", False),
                ):
                    hops = variants[enabled]
                    for _ in range(args.warmup):
                        roundtrip(hops, step)
                        step += 1
                    dist.barrier()
                    times = []
                    for _ in range(args.rounds):
                        start = time.perf_counter_ns()
                        roundtrip(hops, step)
                        elapsed = (time.perf_counter_ns() - start) / 1e6
                        step += 1
                        times.append(elapsed)
                    roundtrip(hops, step, verify=True)
                    step += 1
                    dist.barrier()
                    if rank == 0:
                        record = {
                            "batch": batch,
                            "repeat": repeat,
                            "label": label,
                            "candidate": args.candidate,
                            "candidate_enabled": enabled,
                            "compact": args.candidate == "flat-state" or enabled,
                            "median_ms": statistics.median(times),
                            "round_ms": times,
                            "inventory": wire_inventory(hops),
                        }
                        report["runs"].append(record)
                        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
                        print(
                            f"B={batch} repeat={repeat} {label}: "
                            f"{record['median_ms']:.3f} ms",
                            flush=True,
                        )
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument(
        "--candidate",
        choices=("compact", "flat-state"),
        default="compact",
    )
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--transport-only",
        action="store_true",
        help="Measure the compact relay alone, without candidate A/B/A runs.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if (
        min(args.rounds, args.repeats, *args.batches) < 1
        or max(args.batches) > 8
        or args.warmup < 0
    ):
        parser.error(
            "Use batch sizes 1..8, positive rounds/repeats, nonnegative warmup"
        )
    if args.backend == "nccl" and torch.accelerator.device_count() < 4:
        parser.error("NCCL mode needs at least four visible CUDA devices")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dsv41-relay-gloo-") as directory:
        mp.spawn(
            _worker, args=(f"file://{directory}/rendezvous", args), nprocs=4, join=True
        )


if __name__ == "__main__":
    main()
