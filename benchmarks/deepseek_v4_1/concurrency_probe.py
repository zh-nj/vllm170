#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small controlled concurrency probe for a running DSV41F service.

The probe sends one exact-token wave for each concurrency level and records
client timing, vLLM counters, GPU samples, and host memory pressure.  It is
intended for bounded diagnosis, not as a production load generator.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import aiohttp
import regex as re

_METRIC_LINE = re.compile(r"^(vllm:[^{]+(?:\{[^}]*\})?)\s+([-+0-9.eE]+)$")


def _read_metrics(body: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in body.splitlines():
        match = _METRIC_LINE.match(line)
        if match is None:
            continue
        sample, value = match.groups()
        if sample.split("{", 1)[0].endswith("_created"):
            continue
        with contextlib.suppress(ValueError):
            out[sample] = float(value)
    return out


async def metrics(session: aiohttp.ClientSession, base: str) -> dict[str, object]:
    async with session.get(f"{base}/metrics") as response:
        response.raise_for_status()
        body = await response.text()
    return {"wall": time.time(), "values": _read_metrics(body)}


def host_snapshot() -> dict[str, object]:
    def read(path: str) -> str:
        try:
            return Path(path).read_text().strip()
        except OSError:
            return ""

    mem = {}
    for line in read("/proc/meminfo").splitlines():
        key, _, value = line.partition(":")
        if key in {"MemTotal", "MemFree", "MemAvailable", "SwapTotal", "SwapFree"}:
            mem[key] = value.strip()
    psi = read("/proc/pressure/memory")
    vmstat = {}
    for line in read("/proc/vmstat").splitlines():
        key, _, value = line.partition(" ")
        if key in {
            "pgmajfault",
            "pgscan_direct",
            "allocstall_normal",
            "allocstall_movable",
        }:
            vmstat[key] = int(value)
    return {"meminfo": mem, "psi_memory": psi, "vmstat": vmstat}


def gpu_sample() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=index,power.draw,utilization.gpu,memory.used,clocks.sm",
        "--format=csv,noheader,nounits",
    ]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def summarize_requests(requests: list[dict], wall_s: float) -> dict[str, float]:
    """Measure delivered tokens over the union of client decode intervals."""
    start = min(row["first_token_s"] for row in requests)
    end = max(row["last_token_s"] for row in requests)
    span = end - start
    tokens = sum(row["post_first_chunk_tokens"] for row in requests)
    return {
        "aggregate_output_tok_s": sum(row["completion_tokens"] for row in requests)
        / wall_s,
        "aggregate_decode_tok_s": tokens / span if span > 0 else 0.0,
        "aggregate_decode_span_s": span,
        "aggregate_decode_tokens": tokens,
    }


async def one_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    input_len: int,
    output_len: int,
    seed: int,
    temperature: float,
    top_p: float,
    *,
    prompt_ids: list[int] | None = None,
    cache_salt: str | None = None,
    delay_s: float = 0,
) -> dict[str, object]:
    await asyncio.sleep(delay_s)
    if prompt_ids is None:
        rng = random.Random(seed)
        prompt = [rng.randrange(16, 100000) for _ in range(input_len)]
    else:
        prompt = prompt_ids
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_len,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "ignore_eos": True,
        "stream": True,
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
        "cache_salt": cache_salt or f"concurrency-probe-{seed}",
    }
    start = time.perf_counter()
    first = None
    last = None
    generated = 0
    chunks = 0
    usage = None
    done = False
    finish_reason = None
    first_chunk_tokens = 0
    token_ids = []
    token_chunks = []
    async with session.post(url, json=payload) as response:
        if response.status != 200:
            body = await response.text()
            raise RuntimeError(f"HTTP {response.status}: {body[:500]}")
        async for raw in response.content:
            line = raw.strip()
            if not line.startswith(b"data: "):
                continue
            data = line[6:]
            if data == b"[DONE]":
                done = True
                break
            obj = json.loads(data)
            if obj.get("error"):
                raise RuntimeError(f"Server returned an error: {obj['error']}")
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                if choice.get("index", 0) != 0:
                    raise RuntimeError("Expected one completion")
                ids = choice.get("token_ids")
                if ids is None:
                    raise RuntimeError("Server did not return delta token IDs")
                finish_reason = choice.get("finish_reason") or finish_reason
                if not ids:
                    continue
                now = time.perf_counter()
                if first is None:
                    first_chunk_tokens = len(ids)
                first = now if first is None else first
                last = now
                generated += len(ids)
                chunks += 1
                token_ids.extend(ids)
                token_chunks.append({"elapsed_s": now - start, "num_tokens": len(ids)})
    end = time.perf_counter()
    if first is None or last is None or not done or not usage:
        raise RuntimeError("Incomplete stream: token IDs, usage and [DONE] required")
    if (
        generated != output_len
        or generated != usage["completion_tokens"]
        or usage["prompt_tokens"] != len(prompt)
        or finish_reason != "length"
    ):
        raise RuntimeError("Stream token counts or finish reason differ from request")
    decode_span = last - first
    decode_tokens = generated - first_chunk_tokens
    return {
        "seed": seed,
        "prompt_tokens": int((usage or {}).get("prompt_tokens", input_len)),
        "completion_tokens": generated,
        "chunks": chunks,
        "tokens_per_client_chunk": generated / chunks,
        "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
        "token_ids_sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
        "token_ids": token_ids,
        "start_s": start,
        "first_token_s": first,
        "last_token_s": last,
        "first_chunk_tokens": first_chunk_tokens,
        "post_first_chunk_tokens": decode_tokens,
        "token_chunks": token_chunks,
        "ttft_s": first - start,
        "e2e_s": end - start,
        "decode_span_s": decode_span,
        "decode_tok_s": decode_tokens / decode_span if decode_span > 0 else None,
    }


async def wave(
    session: aiohttp.ClientSession,
    base: str,
    model: str,
    input_len: int,
    output_len: int,
    concurrency: int,
    seed_base: int,
    temperature: float,
    top_p: float,
    *,
    prompt_ids: list[int] | None = None,
    cache_salt: str | None = None,
    stagger_ms: float = 0,
) -> dict[str, object]:
    before = await metrics(session, base)
    host_before = host_snapshot()
    start = time.perf_counter()
    requests = await asyncio.gather(
        *[
            one_request(
                session,
                f"{base}/v1/completions",
                model,
                input_len,
                output_len,
                seed_base + i,
                temperature,
                top_p,
                prompt_ids=prompt_ids,
                cache_salt=cache_salt,
                delay_s=i * stagger_ms / 1000,
            )
            for i in range(concurrency)
        ]
    )
    end = time.perf_counter()
    after = await metrics(session, base)
    return {
        "input_len": input_len,
        "output_len": output_len,
        "concurrency": concurrency,
        "wave_wall_s": end - start,
        **summarize_requests(requests, end - start),
        "requests": requests,
        "metrics_before": before,
        "metrics_after": after,
        "metric_deltas": {
            key: value - before["values"][key]
            for key, value in after["values"].items()
            if key in before["values"]
        },
        "host_before": host_before,
        "host_after": host_snapshot(),
        "gpu_after": gpu_sample(),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:26661")
    parser.add_argument("--model", default="dsv41f")
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--cache-salt")
    parser.add_argument("--stagger-ms", type=float, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.input_len, args.output_len) <= 0 or args.stagger_ms < 0:
        parser.error("Token counts must be positive and stagger-ms non-negative")
    prompt_ids = None
    if args.prompt_file is not None:
        prompt_ids = json.loads(args.prompt_file.read_text())
        if not isinstance(prompt_ids, list) or not prompt_ids:
            parser.error("prompt-file must contain a non-empty token ID list")
        args.input_len = len(prompt_ids)
    connector = aiohttp.TCPConnector(limit=max(map(int, args.concurrency.split(","))))
    timeout = aiohttp.ClientTimeout(total=1800, sock_read=1800)
    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout, read_bufsize=2**26
    ) as session:
        rows = []
        for concurrency in (int(x) for x in args.concurrency.split(",")):
            print(f"[run] c={concurrency}", flush=True)
            row = await wave(
                session,
                args.base_url,
                args.model,
                args.input_len,
                args.output_len,
                concurrency,
                args.seed,
                args.temperature,
                args.top_p,
                prompt_ids=prompt_ids,
                cache_salt=args.cache_salt,
                stagger_ms=args.stagger_ms,
            )
            rows.append(row)
            print(
                f"  wall={row['wave_wall_s']:.3f}s "
                f"aggregate={row['aggregate_output_tok_s']:.2f} tok/s",
                flush=True,
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["out"] = str(config["out"])
    config["prompt_file"] = str(args.prompt_file) if args.prompt_file else None
    args.out.write_text(
        json.dumps({"schema_version": 2, "args": config, "waves": rows}, indent=2)
        + "\n"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
