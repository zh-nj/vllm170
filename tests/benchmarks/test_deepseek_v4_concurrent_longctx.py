# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
import sys
import threading
from pathlib import Path

import pytest

BENCHMARK = (
    Path(__file__).parents[2]
    / "benchmarks"
    / "deepseek_v4"
    / "bench_concurrent_longctx.py"
)
sys.path.insert(0, str(BENCHMARK.parent))
SPEC = importlib.util.spec_from_file_location("dsv4_concurrent_longctx", BENCHMARK)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


class FakeResponse:
    def __init__(self, lines: tuple[bytes, ...]) -> None:
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def __iter__(self):
        return iter(self.lines)


def test_count_tokens_applies_max_reasoning_template(monkeypatch) -> None:
    captured = {}

    def fake_request(_url, payload, _timeout):
        captured.update(payload)
        return {"count": 123}

    monkeypatch.setattr(benchmark, "request_json", fake_request)

    assert benchmark.count_tokens("http://server", "model", "prompt", "max") == 123
    assert captured["messages"] == [{"role": "user", "content": "prompt"}]
    assert captured["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "max",
    }


def test_run_stream_uses_chat_max_and_counts_reasoning_token(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        del timeout
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        return FakeResponse(
            (
                (
                    b'data: {"choices":[{"delta":{"reasoning":"think"},'
                    b'"token_ids":[11,12]}]}\n'
                ),
                (
                    b'data: {"usage":{"prompt_tokens":3,"completion_tokens":3},'
                    b'"choices":[{"delta":{"content":"answer"},'
                    b'"token_ids":[13]}]}\n'
                ),
                b"data: [DONE]\n",
            )
        )

    monkeypatch.setattr(benchmark.urllib.request, "urlopen", fake_urlopen)
    result = benchmark.run_stream(
        "http://server",
        "model",
        "prompt",
        max_tokens=3,
        index=0,
        needle="answer",
        all_needles=("answer",),
        barrier=threading.Barrier(1),
        timeout=30,
        seed=17,
        temperature=1.0,
        top_p=1.0,
        reasoning_effort="max",
    )

    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["payload"]["reasoning_effort"] == "max"
    assert captured["payload"]["return_token_ids"] is True
    assert captured["payload"]["messages"][0]["content"] == "prompt"
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 3
    assert result.decode_tokens_per_s == pytest.approx(2 / result.decode_s)
    assert result.output_text == "thinkanswer"
