#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure concurrent long-context capacity, speed, and needle recall."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from profile_workload import (
    SpecMetrics,
    conditional_acceptance_rates,
    fetch_metrics,
)

WORDS = (
    "system",
    "kernel",
    "memory",
    "buffer",
    "thread",
    "process",
    "socket",
    "packet",
    "register",
    "cache",
    "pointer",
    "allocate",
    "schedule",
    "interrupt",
    "virtual",
    "physical",
    "address",
    "translate",
    "compile",
    "execute",
    "branch",
    "predict",
    "pipeline",
    "vector",
    "matrix",
    "tensor",
    "gradient",
    "cluster",
    "network",
    "storage",
    "device",
    "driver",
    "module",
    "segment",
    "offset",
    "boundary",
)


@dataclass(frozen=True)
class RequestResult:
    index: int
    needle: str
    prompt_tokens: int
    completion_tokens: int
    started_at: float
    first_token_at: float
    finished_at: float
    ttft_s: float
    decode_s: float
    decode_tokens_per_s: float
    needle_found: bool
    cross_talk: tuple[str, ...]
    output_text: str
    output_text_sha256: str


@dataclass(frozen=True)
class BranchResult:
    index: int
    output_text: str
    output_text_sha256: str


def request_json(
    url: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def count_tokens(base_url: str, model: str, prompt: str) -> int:
    response = request_json(
        f"{base_url}/tokenize",
        {"model": model, "prompt": prompt},
        300,
    )
    count = response.get("count")
    if not isinstance(count, int) or count <= 0:
        raise RuntimeError(f"invalid /tokenize response: {response!r}")
    return count


def render_prompt(words: list[str], word_count: int, needle: str) -> str:
    insert_at = min(word_count, max(1, len(words) // 10))
    body = " ".join(words[:insert_at])
    body += f". IMPORTANT FACT: the secret passphrase is {needle}. "
    body += " ".join(words[insert_at:word_count])
    return (
        "Read the following notes carefully.\n\n"
        f"{body}\n\n"
        "Question: What is the secret passphrase mentioned in the notes?\n"
        "Answer: The secret passphrase is"
    )


def make_exact_prompt(
    base_url: str,
    model: str,
    target_tokens: int,
    seed: int,
    needle: str,
) -> str:
    rng = random.Random(seed)
    words = [rng.choice(WORDS) for _ in range(target_tokens)]
    word_count = max(1, int(target_tokens / 1.3))
    observations: list[tuple[int, int]] = []
    for _ in range(20):
        prompt = render_prompt(words, word_count, needle)
        actual_tokens = count_tokens(base_url, model, prompt)
        observations.append((word_count, actual_tokens))
        if actual_tokens == target_tokens:
            return prompt
        word_count += target_tokens - actual_tokens
        word_count = min(max(word_count, 1), len(words))
    raise RuntimeError(
        f"could not construct exact {target_tokens}-token prompt: {observations}"
    )


def run_stream(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    index: int,
    needle: str,
    all_needles: tuple[str, ...],
    barrier: threading.Barrier,
    timeout: float,
) -> RequestResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    barrier.wait()
    started_at = time.perf_counter()
    first_token_at: float | None = None
    output: list[str] = []
    usage: dict[str, int] = {}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage.update(event["usage"])
            for choice in event.get("choices", ()):
                text = choice.get("text", "")
                if text:
                    first_token_at = first_token_at or time.perf_counter()
                    output.append(text)
    finished_at = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError(f"request {index} produced no output")
    output_text = "".join(output)
    completion_tokens = int(usage.get("completion_tokens", 0))
    if completion_tokens != max_tokens:
        raise RuntimeError(
            f"request {index}: expected {max_tokens} completion tokens, "
            f"got {completion_tokens}"
        )
    decode_s = finished_at - first_token_at
    cross_talk = tuple(
        other for other in all_needles if other != needle and other in output_text
    )
    return RequestResult(
        index=index,
        needle=needle,
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=completion_tokens,
        started_at=started_at,
        first_token_at=first_token_at,
        finished_at=finished_at,
        ttft_s=first_token_at - started_at,
        decode_s=decode_s,
        decode_tokens_per_s=(completion_tokens - 1) / decode_s,
        needle_found=needle in output_text,
        cross_talk=cross_talk,
        output_text=output_text,
        output_text_sha256=hashlib.sha256(output_text.encode()).hexdigest(),
    )


def run_branch_stream(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    branches: int,
    seed: int,
    timeout: float,
) -> tuple[float, float, float, dict[str, int], list[BranchResult]]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "top_p": 1,
        "seed": seed,
        "ignore_eos": True,
        "n": branches,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started_at = time.perf_counter()
    first_token_at: float | None = None
    outputs: list[list[str]] = [[] for _ in range(branches)]
    usage: dict[str, int] = {}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage.update(event["usage"])
            for choice in event.get("choices", ()):
                text = choice.get("text", "")
                if not text:
                    continue
                first_token_at = first_token_at or time.perf_counter()
                outputs[int(choice["index"])].append(text)
    finished_at = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("branch request produced no output")
    expected_completion_tokens = max_tokens * branches
    completion_tokens = int(usage.get("completion_tokens", 0))
    if completion_tokens != expected_completion_tokens:
        raise RuntimeError(
            f"expected {expected_completion_tokens} completion tokens, "
            f"got {completion_tokens}"
        )
    results = []
    for index, fragments in enumerate(outputs):
        output_text = "".join(fragments)
        results.append(
            BranchResult(
                index=index,
                output_text=output_text,
                output_text_sha256=hashlib.sha256(output_text.encode()).hexdigest(),
            )
        )
    return started_at, first_token_at, finished_at, usage, results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8098")
    parser.add_argument("--model", default="dsv4s")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--branches", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=170)
    parser.add_argument("--mode", choices=("target", "dspark"), required=True)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if args.branches <= 0:
        raise ValueError("branches must be positive")
    if args.branches > 1:
        if args.concurrency != 1:
            raise ValueError("branch mode requires --concurrency 1")
        needle = f"orbital-falcon-branch-{args.seed}"
        prompt = make_exact_prompt(
            args.base_url,
            args.model,
            args.prompt_tokens,
            args.seed,
            needle,
        )
        metrics_before = fetch_metrics(args.base_url)
        (
            started_at,
            first_token_at,
            finished_at,
            usage,
            branch_results,
        ) = run_branch_stream(
            args.base_url,
            args.model,
            prompt,
            args.max_tokens,
            args.branches,
            args.seed,
            args.timeout,
        )
        metrics_after = fetch_metrics(args.base_url)
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        decode_s = finished_at - first_token_at
        summary = {
            "schema_version": 1,
            "mode": args.mode,
            "workload": "shared-prefix-branches",
            "branches": args.branches,
            "prompt_tokens": prompt_tokens,
            "max_tokens_per_branch": args.max_tokens,
            "completion_tokens": completion_tokens,
            "ttft_s": first_token_at - started_at,
            "decode_s": decode_s,
            "decode_tokens_per_s": (completion_tokens - args.branches) / decode_s,
            "needle_passes": sum(
                needle in result.output_text for result in branch_results
            ),
            "branches_output": [asdict(result) for result in branch_results],
        }
        if metrics_before is not None and metrics_after is not None:
            spec_metrics = metrics_after - metrics_before
            summary["spec_decode"] = asdict(spec_metrics)
            if spec_metrics.draft_tokens:
                summary["acceptance_rate"] = (
                    spec_metrics.accepted_tokens / spec_metrics.draft_tokens
                )
            if spec_metrics.drafts:
                summary["mean_accepted_length"] = (
                    1 + spec_metrics.accepted_tokens / spec_metrics.drafts
                )
                summary["ms_per_round"] = 1000 * decode_s / spec_metrics.drafts
                summary["per_position_acceptance_rates"] = [
                    accepted / spec_metrics.drafts
                    for accepted in spec_metrics.accepted_tokens_per_pos
                ]
                summary["conditional_per_position_acceptance_rates"] = (
                    conditional_acceptance_rates(
                        spec_metrics.drafts,
                        spec_metrics.accepted_tokens_per_pos,
                    )
                )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    needles = tuple(
        f"orbital-falcon-{index}-{args.seed}" for index in range(args.concurrency)
    )
    prompts = [
        make_exact_prompt(
            args.base_url,
            args.model,
            args.prompt_tokens,
            args.seed + index,
            needles[index],
        )
        for index in range(args.concurrency)
    ]
    metrics_before = fetch_metrics(args.base_url)
    barrier = threading.Barrier(args.concurrency)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = [
            executor.submit(
                run_stream,
                args.base_url,
                args.model,
                prompts[index],
                args.max_tokens,
                index,
                needles[index],
                needles,
                barrier,
                args.timeout,
            )
            for index in range(args.concurrency)
        ]
        results = [future.result() for future in futures]
    metrics_after = fetch_metrics(args.base_url)
    results.sort(key=lambda result: result.index)

    started_at = min(result.started_at for result in results)
    prefill_finished_at = max(result.first_token_at for result in results)
    finished_at = max(result.finished_at for result in results)
    prompt_tokens = sum(result.prompt_tokens for result in results)
    completion_tokens = sum(result.completion_tokens for result in results)
    prefill_s = prefill_finished_at - started_at
    decode_s = finished_at - prefill_finished_at
    wall_s = finished_at - started_at
    ttfts = [result.ttft_s for result in results]
    spec_metrics: SpecMetrics | None = None
    if metrics_before is not None and metrics_after is not None:
        spec_metrics = metrics_after - metrics_before

    summary: dict[str, Any] = {
        "schema_version": 1,
        "mode": args.mode,
        "concurrency": args.concurrency,
        "prompt_tokens_per_request": args.prompt_tokens,
        "max_tokens_per_request": args.max_tokens,
        "total_tokens_per_request": args.prompt_tokens + args.max_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prefill_s": prefill_s,
        "aggregate_prefill_tokens_per_s": prompt_tokens / prefill_s,
        "decode_s_after_all_first_tokens": decode_s,
        "aggregate_decode_tokens_per_s": (
            args.concurrency * (args.max_tokens - 1) / decode_s
        ),
        "service_completion_tokens_per_s": completion_tokens / wall_s,
        "ttft_min_s": min(ttfts),
        "ttft_mean_s": sum(ttfts) / len(ttfts),
        "ttft_max_s": max(ttfts),
        "wall_s": wall_s,
        "needle_passes": sum(result.needle_found for result in results),
        "cross_talk_failures": sum(bool(result.cross_talk) for result in results),
        "requests": [asdict(result) for result in results],
    }
    if spec_metrics is not None:
        summary["spec_decode"] = asdict(spec_metrics)
        if spec_metrics.draft_tokens:
            summary["acceptance_rate"] = (
                spec_metrics.accepted_tokens / spec_metrics.draft_tokens
            )
        if spec_metrics.drafts:
            summary["mean_accepted_length"] = (
                1 + spec_metrics.accepted_tokens / spec_metrics.drafts
            )
            summary["ms_per_round"] = 1000 * decode_s / spec_metrics.drafts
            summary["per_position_acceptance_rates"] = [
                accepted / spec_metrics.drafts
                for accepted in spec_metrics.accepted_tokens_per_pos
            ]
            summary["conditional_per_position_acceptance_rates"] = (
                conditional_acceptance_rates(
                    spec_metrics.drafts,
                    spec_metrics.accepted_tokens_per_pos,
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
