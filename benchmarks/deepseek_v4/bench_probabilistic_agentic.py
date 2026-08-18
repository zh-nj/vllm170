#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate probabilistic DSpark on deterministic tool-selection tasks."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from profile_workload import fetch_metrics

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search technical documentation.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order by identifier.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
]


@dataclass(frozen=True)
class ToolCase:
    name: str
    prompt: str
    expected_tool: str
    expected_key: str
    expected_value: str


CASES = (
    ToolCase(
        "weather",
        "What is the current weather in Shanghai? Use the available tool.",
        "get_weather",
        "city",
        "Shanghai",
    ),
    ToolCase(
        "calculator",
        "Calculate (23 * 47) + 5 using the appropriate tool.",
        "calculator",
        "expression",
        "(23*47)+5",
    ),
    ToolCase(
        "documentation",
        "Find documentation about CMP170HX sparse MLA using the available tool.",
        "search_docs",
        "query",
        "CMP170HX sparse MLA",
    ),
    ToolCase(
        "order",
        "Check the status of order A-170 with the available tool.",
        "lookup_order",
        "order_id",
        "A-170",
    ),
)


def _normalized(value: Any) -> str:
    return "".join(str(value).lower().split())


def score_response(case: ToolCase, response: dict[str, Any]) -> dict[str, Any]:
    """Score parsed tool name and the required argument."""
    choice = response.get("choices", [{}])[0]
    message = choice.get("message") or {}
    tool_calls = message.get("tool_calls") or []
    tool_name = None
    arguments: dict[str, Any] = {}
    parse_error = None
    if tool_calls:
        function = tool_calls[0].get("function") or {}
        tool_name = function.get("name")
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except (TypeError, json.JSONDecodeError) as error:
            parse_error = str(error)
    expected = _normalized(case.expected_value)
    observed = _normalized(arguments.get(case.expected_key, ""))
    success = (
        tool_name == case.expected_tool
        and parse_error is None
        and (expected == observed or expected in observed or observed in expected)
    )
    return {
        "case": case.name,
        "success": success,
        "finish_reason": choice.get("finish_reason"),
        "tool_name": tool_name,
        "arguments": arguments,
        "parse_error": parse_error,
        "completion_tokens": int(response.get("usage", {}).get("completion_tokens", 0)),
    }


def request_case(
    base_url: str,
    model: str,
    case: ToolCase,
    seed: int,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": case.prompt}],
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 1.0,
        "top_p": 0.95,
        "seed": seed,
        "max_tokens": 256,
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8098")
    parser.add_argument("--model", default="dsv4s")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")

    metrics_before = fetch_metrics(args.base_url)
    started = time.perf_counter()
    results = []
    for repeat in range(args.repeats):
        for case_index, case in enumerate(CASES):
            response = request_case(
                args.base_url,
                args.model,
                case,
                args.seed + repeat * len(CASES) + case_index,
                args.timeout,
            )
            result = score_response(case, response)
            result["repeat"] = repeat
            results.append(result)
    elapsed = time.perf_counter() - started
    metrics_after = fetch_metrics(args.base_url)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "scenario": "agentic",
        "temperature": 1.0,
        "top_p": 0.95,
        "repeats": args.repeats,
        "requests": len(results),
        "successes": sum(result["success"] for result in results),
        "success_rate": sum(result["success"] for result in results) / len(results),
        "elapsed_s": elapsed,
        "results": results,
        "cases": [asdict(case) for case in CASES],
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
