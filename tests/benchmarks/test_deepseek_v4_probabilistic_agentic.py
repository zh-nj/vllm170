# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
import sys
from pathlib import Path

BENCHMARK = (
    Path(__file__).parents[2]
    / "benchmarks"
    / "deepseek_v4"
    / "bench_probabilistic_agentic.py"
)
sys.path.insert(0, str(BENCHMARK.parent))
SPEC = importlib.util.spec_from_file_location("dsv4_probabilistic_agentic", BENCHMARK)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_score_response_accepts_normalized_tool_argument() -> None:
    case = benchmark.CASES[1]
    response = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "calculator",
                                "arguments": json.dumps(
                                    {"expression": "(23 * 47) + 5"}
                                ),
                            }
                        }
                    ]
                },
            }
        ],
        "usage": {"completion_tokens": 12},
    }

    result = benchmark.score_response(case, response)

    assert result["success"] is True
    assert result["completion_tokens"] == 12


def test_score_response_rejects_wrong_tool() -> None:
    case = benchmark.CASES[0]
    response = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_docs",
                                "arguments": '{"city":"Shanghai"}',
                            }
                        }
                    ]
                },
            }
        ]
    }

    assert benchmark.score_response(case, response)["success"] is False
