# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare Engram backends with fixed code prompts and full-prefill evidence.

Run once per backend against a separately launched server. The first run saves
the exact prompt IDs; later runs reuse them. Every request has a fresh KV cache
salt. Decode throughput counts delta token IDs after the entire first chunk.
An optional SSD directory advises eviction of weight file pages before the
first request at each length; later requests keep the natural OS cache.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.deepseek_v4_1 import dspark_flow_gate as flow  # noqa: E402


def make_prompts(base, model, lengths, path):
    if path.exists():
        record = json.loads(path.read_text())
        for length in lengths:
            if len(record["prompts"][str(length)]) != length:
                raise ValueError("Saved prompt has the wrong token count")
        return record
    sources = [
        "vllm/v1/core/sched/scheduler.py",
        "vllm/v1/worker/gpu/model_runner.py",
        "vllm/forward_context.py",
        "vllm/model_executor/layers/fused_moe/modular_kernel.py",
        "vllm/models/deepseek_v4_1/ced.py",
        "vllm/models/deepseek_v4_1/ampere/ampere_sparse.py",
        "vllm/v1/core/kv_cache_manager.py",
    ]
    more = sorted(str(p) for p in Path("vllm/models").rglob("*.py"))
    sources += [p for p in more if p not in sources]
    bodies, hashes, size = [], {}, 0
    for name in sources:
        text = Path(name).read_text()
        hashes[name] = hashlib.sha256(text.encode()).hexdigest()
        bodies.append(f"\nFile {name}:\n{text}")
        size += len(text)
        if size >= max(lengths) * 10:
            break

    def tokenize(text):
        payload = {"model": model, "prompt": text, "add_special_tokens": False}
        with flow._post(f"{base}/tokenize", payload, timeout=120) as response:
            return json.loads(response.read())["tokens"]

    body = tokenize("\n".join(bodies))
    head = tokenize("Read the following Python code archive.\n")
    tail = tokenize(
        "\nDescribe the architecture and major components of this code archive "
        "in detail.\nAnswer:"
    )
    if len(body) < max(lengths) - len(head) - len(tail):
        raise ValueError("Source archive is too short")
    prompts = {str(n): head + body[: n - len(head) - len(tail)] + tail for n in lengths}
    record = {
        "kind": "nonperiodic_python_code_archive",
        "source_sha256": hashes,
        "prompts": prompts,
        "prompt_sha256": {
            key: hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            for key, ids in prompts.items()
        },
        "distinct_tokens": {key: len(set(ids)) for key, ids in prompts.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record) + "\n")
    return record


def resources(unit):
    group = subprocess.check_output(
        ["systemctl", "--user", "show", unit, "-p", "ControlGroup", "--value"],
        text=True,
    ).strip()
    if not group:
        raise ValueError(f"Unit has no control group: {unit}")
    root = Path("/sys/fs/cgroup") / group.lstrip("/")
    pids = {
        int(p) for path in root.rglob("cgroup.procs") for p in path.read_text().split()
    }
    processes = {}
    for pid in sorted(pids):
        try:
            proc = Path(f"/proc/{pid}")
            processes[str(pid)] = {
                "comm": (proc / "comm").read_text().strip(),
                "io": (proc / "io").read_text(),
                "status": "\n".join(
                    line
                    for line in (proc / "status").read_text().splitlines()
                    if line.startswith(("VmRSS:", "VmLck:", "VmSwap:"))
                ),
            }
        except (OSError, ProcessLookupError):
            continue
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "control_group": group,
        "processes": processes,
        "memory_current": (root / "memory.current").read_text().strip(),
        "memory_stat": (root / "memory.stat").read_text(),
        "io_stat": (root / "io.stat").read_text()
        if (root / "io.stat").exists()
        else None,
        "host_meminfo": Path("/proc/meminfo").read_text(),
        "gpus": subprocess.check_output(
            [
                "nvidia-smi",
                (
                    "--query-gpu=index,uuid,memory.used,temperature.gpu,power.draw,"
                    "clocks.current.sm,clocks.current.memory"
                ),
                "--format=csv,noheader",
            ],
            text=True,
        ),
    }


def counter(deltas, name):
    values = [value for key, value in deltas.items() if key.split("{", 1)[0] == name]
    if not values:
        raise ValueError(f"Missing metric: {name}")
    return sum(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:26662")
    parser.add_argument("--model", default="dsv41f-port")
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[1024, 32768, 65536, 131072]
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--cold-ssd-dir", type=Path)
    args = parser.parse_args()
    if args.repeats < 2 or min(args.lengths + [args.max_tokens]) <= 0:
        parser.error("Use positive token counts and at least two repeats")
    prompts = make_prompts(args.base_url, args.model, args.lengths, args.prompts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise ValueError("Output already exists; use a new file to preserve evidence")
    report = {
        "label": args.label,
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "prompt_sha256": prompts["prompt_sha256"],
        "rows": [],
    }
    # Warm kernels with a separate short prompt, without warming the measured
    # code archive's Engram rows or its KV prefix.
    with flow._post(
        f"{args.base_url}/tokenize",
        {"model": args.model, "prompt": "Explain what a GPU kernel does. " * 100},
        timeout=120,
    ) as response:
        warm_ids = json.loads(response.read())["tokens"][:512]
    report["warmup"] = flow.run_once(
        args.base_url, args.model, warm_ids, 32, 1800, cache_salt=uuid4().hex
    )
    for length in args.lengths:
        for repeat in range(args.repeats):
            time.sleep(2)
            evicted = []
            if repeat == 0 and args.cold_ssd_dir:
                for path in sorted(args.cold_ssd_dir.glob("*.weight.u8")):
                    with path.open("rb", buffering=0) as file:
                        os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                    evicted.append(str(path))
                if not evicted:
                    raise ValueError("No SSD weight files found for cold-page advice")
            before_resources = resources(args.unit)
            before = flow.read_engine_metrics(args.base_url, 120)
            salt = uuid4().hex
            row = flow.run_once(
                args.base_url,
                args.model,
                prompts["prompts"][str(length)],
                args.max_tokens,
                1800,
                cache_salt=salt,
            )
            after = flow.read_engine_metrics(args.base_url, 120)
            deltas = flow.metric_deltas(before, after)
            for _ in range(5):
                if counter(deltas, "vllm:request_prefill_time_seconds_count") == 1:
                    break
                time.sleep(1)
                after = flow.read_engine_metrics(args.base_url, 120)
                deltas = flow.metric_deltas(before, after)
            hits = counter(deltas, "vllm:prefix_cache_hits_total")
            if hits != 0:
                raise ValueError(f"Request reused {hits} prefix tokens")
            if counter(deltas, "vllm:request_prefill_time_seconds_count") != 1:
                raise ValueError("Expected metrics for exactly one finished request")
            row.update(
                {
                    "length": length,
                    "repeat": repeat,
                    "cache_salt": salt,
                    "prefix_cache_hits": hits,
                    "weight_page_eviction_advice": evicted,
                    "prefill_s": counter(
                        deltas, "vllm:request_prefill_time_seconds_sum"
                    ),
                    "metric_deltas": deltas,
                    "resources_before": before_resources,
                    "resources_after": resources(args.unit),
                }
            )
            row["prefill_tok_per_s"] = length / row["prefill_s"]
            report["rows"].append(row)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"{args.label} {length} repeat={repeat}: "
                f"TTFT={row['ttft_s']:.3f}s prefill={row['prefill_tok_per_s']:.1f} "
                f"decode={row['post_first_chunk_tok_per_s']:.2f} tok/s "
                f"total={row['request_s']:.3f}s prefix_hits={hits}",
                flush=True,
            )
    report["summary"] = []
    for length in args.lengths:
        rows = [row for row in report["rows"] if row["length"] == length]
        steady = rows[1:]
        report["summary"].append(
            {
                "length": length,
                "steady_repeats": len(steady),
                "identical_repeat_token_ids": all(
                    row["token_ids"] == rows[0]["token_ids"] for row in rows
                ),
                **{
                    key: statistics.median(row[key] for row in steady)
                    for key in (
                        "ttft_s",
                        "prefill_s",
                        "prefill_tok_per_s",
                        "post_first_chunk_s",
                        "post_first_chunk_tok_per_s",
                        "request_s",
                    )
                },
            }
        )
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
