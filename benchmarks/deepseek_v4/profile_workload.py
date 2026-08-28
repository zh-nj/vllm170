# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run reproducible DeepSeek V4 profiling workloads against a server."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import regex as re

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
class SpecMetrics:
    drafts: int = 0
    draft_tokens: int = 0
    accepted_tokens: int = 0
    accepted_tokens_per_pos: tuple[int, ...] = ()

    def __sub__(self, other: SpecMetrics) -> SpecMetrics:
        num_positions = max(
            len(self.accepted_tokens_per_pos),
            len(other.accepted_tokens_per_pos),
        )
        return SpecMetrics(
            drafts=self.drafts - other.drafts,
            draft_tokens=self.draft_tokens - other.draft_tokens,
            accepted_tokens=self.accepted_tokens - other.accepted_tokens,
            accepted_tokens_per_pos=tuple(
                (
                    self.accepted_tokens_per_pos[pos]
                    if pos < len(self.accepted_tokens_per_pos)
                    else 0
                )
                - (
                    other.accepted_tokens_per_pos[pos]
                    if pos < len(other.accepted_tokens_per_pos)
                    else 0
                )
                for pos in range(num_positions)
            ),
        )


def conditional_acceptance_rates(
    drafts: int, accepted_tokens_per_pos: tuple[int, ...]
) -> list[float]:
    """Return acceptance at each position conditioned on reaching it."""
    denominator = drafts
    rates = []
    for accepted in accepted_tokens_per_pos:
        rates.append(accepted / denominator if denominator > 0 else 0.0)
        denominator = accepted
    return rates


def resolve_top_p(scenario: str, requested: float | None) -> float:
    """Resolve and validate the scenario-specific nucleus threshold."""
    expected = 0.95 if scenario == "agentic" else 1.0
    if requested is not None and not math.isclose(requested, expected):
        raise ValueError(
            f"{scenario} scenario requires top_p={expected}, got {requested}"
        )
    return expected


def parse_spec_metrics(payload: str) -> SpecMetrics | None:
    values = {"drafts": 0, "draft_tokens": 0, "accepted_tokens": 0}
    accepted_tokens_per_pos: dict[int, int] = {}
    found = False
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line.startswith("vllm:spec_decode"):
            continue
        parts = line.split(None, 1)
        metric_name = parts[0].split("{", 1)[0]
        if not metric_name.endswith("_total") or len(parts) != 2:
            continue
        try:
            value = int(float(parts[1]))
        except ValueError:
            continue
        found = True
        if "num_drafts" in metric_name:
            values["drafts"] += value
        elif "num_draft_tokens" in metric_name:
            values["draft_tokens"] += value
        elif "num_accepted_tokens_per_pos" in metric_name:
            position_match = re.search(r'position="(\d+)"', parts[0])
            if position_match is not None:
                position = int(position_match.group(1))
                accepted_tokens_per_pos[position] = (
                    accepted_tokens_per_pos.get(position, 0) + value
                )
        elif "num_accepted_tokens" in metric_name:
            values["accepted_tokens"] += value
    if not found:
        return None
    per_position = tuple(
        accepted_tokens_per_pos.get(position, 0)
        for position in range(max(accepted_tokens_per_pos, default=-1) + 1)
    )
    return SpecMetrics(**values, accepted_tokens_per_pos=per_position)


def make_prompt(approx_tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    return f"[profile {seed}] " + " ".join(
        rng.choice(WORDS) for _ in range(int(approx_tokens / 1.3))
    )


def _make_prompt_words(word_count: int, seed: int) -> str:
    rng = random.Random(seed)
    return f"[profile {seed}] " + " ".join(rng.choice(WORDS) for _ in range(word_count))


def _request(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 3600,
) -> bytes:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {} if data is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def count_prompt_tokens(base_url: str, model: str, prompt: str) -> int:
    response = json.loads(
        _request(
            f"{base_url}/tokenize",
            payload={"model": model, "prompt": prompt},
            timeout=60,
        )
    )
    count = response.get("count")
    if not isinstance(count, int) or count <= 0:
        raise RuntimeError(f"invalid /tokenize response: {response!r}")
    return count


def make_exact_prompt(
    base_url: str,
    model: str,
    target_tokens: int,
    seed: int,
) -> str:
    """Build a deterministic prompt with the server's exact token count."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    word_count = target_tokens
    observations: list[tuple[int, int]] = []
    for _ in range(8):
        prompt = _make_prompt_words(word_count, seed)
        actual_tokens = count_prompt_tokens(base_url, model, prompt)
        observations.append((word_count, actual_tokens))
        if actual_tokens == target_tokens:
            return prompt
        word_count += target_tokens - actual_tokens
        if word_count <= 0:
            break
    raise RuntimeError(
        f"could not construct exact {target_tokens}-token prompt: {observations}"
    )


def wait_until_ready(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            _request(f"{base_url}/health", timeout=5)
            return
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            time.sleep(2)
    raise TimeoutError(f"server did not become ready: {last_error}")


def fetch_metrics(base_url: str) -> SpecMetrics | None:
    try:
        payload = _request(f"{base_url}/metrics", timeout=10).decode()
    except (OSError, urllib.error.URLError):
        return None
    return parse_spec_metrics(payload)


def run_stream(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    seed: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    text_fragments: list[str] = []
    usage: dict[str, int] = {}
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if event.get("usage"):
                usage.update(event["usage"])
            for choice in event.get("choices", ()):
                text = choice.get("text", "")
                if text:
                    first_token_at = first_token_at or time.perf_counter()
                    text_fragments.append(text)
    finished = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("stream produced no output token")
    completion_tokens = int(usage.get("completion_tokens", 0))
    if completion_tokens != max_tokens:
        raise RuntimeError(
            f"expected {max_tokens} completion tokens, got {completion_tokens}"
        )
    decode_s = finished - first_token_at
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": completion_tokens,
        "ttft_s": first_token_at - started,
        "decode_s": decode_s,
        "decode_tokens_per_s": (completion_tokens - 1) / decode_s,
        "total_s": finished - started,
        "output_text_sha256": hashlib.sha256(
            "".join(text_fragments).encode()
        ).hexdigest(),
    }


def build_result(
    args: argparse.Namespace,
    workload: dict[str, Any],
    metrics: SpecMetrics | None,
    prompt: str,
    repeat_index: int,
) -> dict[str, Any]:
    repo_root = Path(__file__).parents[2]
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_ns": time.time_ns(),
        "phase": "decode",
        "git_head": git_head,
        "git_dirty": git_dirty,
        "model": args.model,
        "context_target": args.context,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "repeat_index": repeat_index,
        "repeat_count": args.repeats,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "pp_size": 3,
        "tp_size": 1,
        "dspark_tokens": int(os.getenv("DSV4_DSPARK_TOKENS", "0")),
        "dspark_draft_sample_method": os.getenv(
            "DSV4_DSPARK_DRAFT_SAMPLE_METHOD", "greedy"
        ),
        "scenario": args.scenario,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "dspark_exact_topk": bool(
            int(os.getenv("VLLM_DEEPSEEK_V4_DSPARK_EXACT_TOPK", "1"))
        ),
        "dspark_q1_audit": bool(
            int(os.getenv("VLLM_DEEPSEEK_V4_DSPARK_Q1_AUDIT", "0"))
        ),
        "dspark_q1_audit_samples": int(
            os.getenv("VLLM_DEEPSEEK_V4_DSPARK_Q1_AUDIT_SAMPLES", "64")
        ),
        "max_num_batched_tokens": int(os.getenv("DSV4_MAX_BATCHED_TOKENS", "1715")),
        "max_num_seqs": int(os.getenv("DSV4_MAX_NUM_SEQS", "8")),
        "indexer_max_logits_mb": int(
            os.getenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "96")
        ),
        "indexer_max_query_rows": int(
            os.getenv("VLLM_SPARSE_INDEXER_MAX_QUERY_ROWS", "1024")
        ),
        "indexer_prefill_caller_logits": bool(
            int(os.getenv("VLLM_INDEXER_PREFILL_CALLER_LOGITS", "1"))
        ),
        "marlin_moe_block_size": int(os.getenv("VLLM_MARLIN_MOE_BLOCK_SIZE", "48")),
        "marlin_moe_block_size_min_m": int(
            os.getenv("VLLM_MARLIN_MOE_BLOCK_SIZE_MIN_M", "1000")
        ),
        "marlin_moe_caller_fp32_workspace": bool(
            int(os.getenv("VLLM_MARLIN_MOE_CALLER_FP32_WORKSPACE", "1"))
        ),
        "sparse_prefill_caller_output": bool(
            int(os.getenv("VLLM_DEEPSEEK_V4_SPARSE_PREFILL_CALLER_OUTPUT", "1"))
        ),
        "sparse_prefill_block_k": int(
            os.getenv("VLLM_DEEPSEEK_V4_SPARSE_PREFILL_BLOCK_K", "32")
        ),
        "sparse_decode_caller_output": bool(
            int(os.getenv("VLLM_DEEPSEEK_V4_SPARSE_DECODE_CALLER_OUTPUT", "1"))
        ),
        "sparse_decode_split": bool(
            int(os.getenv("VLLM_DEEPSEEK_V4_SPARSE_DECODE_SPLIT", "1"))
        ),
        "direct_ragged_swa": bool(
            int(os.getenv("VLLM_DEEPSEEK_V4_DIRECT_RAGGED_SWA", "1"))
        ),
        "direct_ragged_c128": bool(
            int(os.getenv("VLLM_DEEPSEEK_V4_DIRECT_RAGGED_C128", "1"))
        ),
        "dspark_kv_only_insert": bool(
            int(os.getenv("VLLM_DEEPSEEK_V4_DSPARK_KV_ONLY_INSERT", "1"))
        ),
        "mhc_pre_broadcast": bool(
            int(os.getenv("VLLM_DEEPSEEK_V4_MHC_PRE_BROADCAST", "1"))
        ),
        "profiler_enabled": not args.skip_profiler,
        "phase_profile_flows": os.getenv(
            "VLLM_DEEPSEEK_V4_PHASE_PROFILE_FLOWS", "prefill,decode,dspark"
        ),
        "phase_profile_layers": os.getenv("VLLM_DEEPSEEK_V4_PHASE_PROFILE_LAYERS", ""),
        "phase_profile_stages": os.getenv("VLLM_DEEPSEEK_V4_PHASE_PROFILE_STAGES", ""),
        "phase_profile_record_function": bool(
            int(os.getenv("VLLM_DEEPSEEK_V4_PHASE_PROFILE_RECORD_FUNCTION", "1"))
        ),
        **workload,
    }
    if metrics is not None:
        result["spec_decode"] = asdict(metrics)
        if metrics.drafts > 0:
            result["ms_per_round"] = 1000 * workload["decode_s"] / metrics.drafts
            result["mean_accepted_length"] = (
                1 + metrics.accepted_tokens / metrics.drafts
            )
        if metrics.draft_tokens > 0:
            result["acceptance_rate"] = metrics.accepted_tokens / metrics.draft_tokens
        if metrics.drafts > 0:
            result["per_position_acceptance_rates"] = [
                accepted / metrics.drafts
                for accepted in metrics.accepted_tokens_per_pos
            ]
            result["conditional_per_position_acceptance_rates"] = (
                conditional_acceptance_rates(
                    metrics.drafts, metrics.accepted_tokens_per_pos
                )
            )
    for name in ("ttft_s", "decode_s", "decode_tokens_per_s", "total_s"):
        if not math.isfinite(result[name]) or result[name] <= 0:
            raise RuntimeError(f"invalid timing field: {name}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8098")
    parser.add_argument("--model", default="dsv4s")
    parser.add_argument("--context", type=int, default=32000)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--scenario", choices=("general", "agentic"), default="general")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--warmup-context", type=int)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--phase-control", type=Path)
    parser.add_argument("--skip-profiler", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if not 0.0 <= args.temperature <= 2.0:
        raise ValueError("temperature must be in [0, 2]")
    args.top_p = resolve_top_p(args.scenario, args.top_p)
    wait_until_ready(args.base_url, args.startup_timeout)
    warmup_context = args.warmup_context or args.context
    warmup_prompt = make_exact_prompt(args.base_url, args.model, warmup_context, seed=1)
    run_stream(
        args.base_url,
        args.model,
        warmup_prompt,
        8,
        seed=1,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    prompt = make_exact_prompt(args.base_url, args.model, args.context, args.seed)
    if args.phase_control is not None:
        args.phase_control.parent.mkdir(parents=True, exist_ok=True)
        args.phase_control.write_text("active\n")
    profiler_started = False
    results = []
    try:
        if not args.skip_profiler:
            _request(f"{args.base_url}/start_profile", payload={})
            profiler_started = True
        for repeat_index in range(args.repeats):
            before = fetch_metrics(args.base_url)
            workload = run_stream(
                args.base_url,
                args.model,
                prompt,
                args.max_tokens,
                seed=args.seed,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            if workload["prompt_tokens"] != args.context:
                raise RuntimeError(
                    "server prompt-token count changed after /tokenize alignment: "
                    f"expected {args.context}, got {workload['prompt_tokens']}"
                )
            after = fetch_metrics(args.base_url)
            delta = after - before if before is not None and after is not None else None
            results.append(build_result(args, workload, delta, prompt, repeat_index))
    finally:
        try:
            if profiler_started:
                _request(f"{args.base_url}/stop_profile", payload={})
        finally:
            if args.phase_control is not None:
                args.phase_control.unlink(missing_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as output:
        for result in results:
            output.write(json.dumps(result, sort_keys=True) + "\n")
    rendered: Any = results[0] if len(results) == 1 else results
    print(json.dumps(rendered, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
