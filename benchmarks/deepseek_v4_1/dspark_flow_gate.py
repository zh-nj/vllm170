# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSV41F flow gate using delta token IDs, including non-text tokens.

The first token-bearing chunk is excluded in full from post-first-chunk TPS.
This is a client delivery measurement, not a GPU step timer. Keep the existing
CED benchmark scripts unchanged; this gate is for isolated DSpark flow A/Bs.
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import regex as re

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.deepseek_v4_1.decode_gate import _post, make_token_ids  # noqa: E402

_METRIC_LINE = re.compile(
    r"^(?P<sample>vllm:[^{]+(?:\{[^}]*\})?)\s+"
    r"(?P<value>[-+0-9.eE]+)$"
)


def read_engine_metrics(base: str, timeout: float) -> dict[str, object]:
    """Read counters needed to interpret a DSpark gate run.

    The values are keyed by the complete Prometheus sample, including labels,
    so separate engines or draft positions cannot be accidentally combined.
    """
    with urllib.request.urlopen(f"{base}/metrics", timeout=timeout) as response:
        body = response.read().decode()
    values: dict[str, float] = {}
    for line in body.splitlines():
        match = _METRIC_LINE.match(line)
        if match is None:
            continue
        sample = match.group("sample")
        name = sample.split("{", 1)[0]
        if not (
            name.startswith("vllm:spec_decode_num_")
            or name
            in {
                "vllm:prompt_tokens_total",
                "vllm:generation_tokens_total",
            }
        ):
            continue
        if name.endswith("_created"):
            continue
        values[sample] = float(match.group("value"))
    return {
        "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "values": values,
    }


def metric_deltas(
    before: dict[str, object], after: dict[str, object]
) -> dict[str, float]:
    before_values = before["values"]
    after_values = after["values"]
    assert isinstance(before_values, dict) and isinstance(after_values, dict)
    return {
        key: float(after_values[key]) - float(before_values[key])
        for key in after_values.keys() & before_values.keys()
    }


def run_once(base, model, prompt_ids, max_tokens, timeout):
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    chunks, usage, done = [], None, False
    finish_reason = None
    with _post(f"{base}/v1/completions", payload, timeout=timeout) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            if line[6:] == "[DONE]":
                done = True
                break
            obj = json.loads(line[6:])
            if obj.get("error"):
                raise ValueError(f"Server returned an error: {obj['error']}")
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                ids = choice.get("token_ids")
                if ids is None:
                    raise ValueError("Server did not return delta token_ids")
                if choice.get("index", 0) != 0:
                    raise ValueError("This gate expects one completion")
                finish_reason = choice.get("finish_reason") or finish_reason
                if ids:
                    chunks.append(
                        {"elapsed_s": time.perf_counter() - start, "token_ids": ids}
                    )
    request_s = time.perf_counter() - start
    if not done or not chunks or not usage:
        raise ValueError("Incomplete stream: token IDs, usage and [DONE] are required")
    ids = [token for chunk in chunks for token in chunk["token_ids"]]
    if len(ids) != usage["completion_tokens"]:
        raise ValueError("Delta token count differs from usage.completion_tokens")
    if usage["prompt_tokens"] != len(prompt_ids):
        raise ValueError("Server prompt token count differs from the supplied prompt")
    if len(ids) != max_tokens or finish_reason != "length":
        raise ValueError("Incomplete benchmark output: expected the fixed token budget")
    first_chunk_tokens = len(chunks[0]["token_ids"])
    span = chunks[-1]["elapsed_s"] - chunks[0]["elapsed_s"]
    remaining = len(ids) - first_chunk_tokens
    return {
        "usage": usage,
        "token_ids": ids,
        "chunks": chunks,
        "finish_reason": finish_reason,
        "first_chunk_tokens": first_chunk_tokens,
        "ttft_s": chunks[0]["elapsed_s"],
        "request_s": request_s,
        "request_tok_per_s": len(ids) / request_s,
        "post_first_chunk_tokens": remaining,
        "post_first_chunk_s": span,
        "post_first_chunk_tok_per_s": remaining / span if span > 0 else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:26663")
    parser.add_argument("--model", default="dsv41f-port")
    parser.add_argument("--prompt-tokens", type=int, default=32000)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--label", required=True, help="For example A1/off or B/on")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.prompt_tokens, args.max_tokens, args.repeats) <= 0:
        parser.error("Token counts and repeats must be positive")
    prompt_ids = make_token_ids(args.base_url, args.prompt_tokens, args.seed)
    metrics_before = read_engine_metrics(args.base_url, args.timeout)
    report = {
        "schema_version": 2,
        "label": args.label,
        "metric": "delta_token_ids_after_first_chunk",
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "prompt_ids": prompt_ids,
        "prompt_sha256": hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest(),
        "engine_metrics_before": metrics_before,
        "rows": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for index in range(args.repeats):
        row = run_once(
            args.base_url, args.model, prompt_ids, args.max_tokens, args.timeout
        )
        report["rows"].append(row)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"[{index}] TTFT={row['ttft_s']:.3f}s "
            f"post-first-chunk TPS={row['post_first_chunk_tok_per_s']}"
        )
    rates = [row["post_first_chunk_tok_per_s"] for row in report["rows"]]
    report["median_post_first_chunk_tok_per_s"] = (
        statistics.median(rates) if all(rate is not None for rate in rates) else None
    )
    metrics_after = read_engine_metrics(args.base_url, args.timeout)
    report["engine_metrics_after"] = metrics_after
    report["engine_metric_deltas"] = metric_deltas(metrics_before, metrics_after)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
