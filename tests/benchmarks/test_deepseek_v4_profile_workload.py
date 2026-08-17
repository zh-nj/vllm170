# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
import sys
from pathlib import Path

PROFILE_WORKLOAD = (
    Path(__file__).parents[2] / "benchmarks" / "deepseek_v4" / "profile_workload.py"
)
SPEC = importlib.util.spec_from_file_location("dsv4_profile_workload", PROFILE_WORKLOAD)
assert SPEC is not None and SPEC.loader is not None
profile_workload = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = profile_workload
SPEC.loader.exec_module(profile_workload)


def test_parse_spec_metrics_collects_per_position_counters() -> None:
    payload = """
vllm:spec_decode_num_drafts_total{model_name="dsv4s"} 10
vllm:spec_decode_num_draft_tokens_total{model_name="dsv4s"} 50
vllm:spec_decode_num_accepted_tokens_total{model_name="dsv4s"} 37
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 10
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 8
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="1",position="1"} 1
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="3"} 2
"""

    assert profile_workload.parse_spec_metrics(payload) == (
        profile_workload.SpecMetrics(
            drafts=10,
            draft_tokens=50,
            accepted_tokens=37,
            accepted_tokens_per_pos=(10, 9, 0, 2),
        )
    )


def test_spec_metrics_delta_includes_per_position_counters() -> None:
    before = profile_workload.SpecMetrics(
        drafts=4,
        draft_tokens=20,
        accepted_tokens=12,
        accepted_tokens_per_pos=(4, 3, 2),
    )
    after = profile_workload.SpecMetrics(
        drafts=10,
        draft_tokens=50,
        accepted_tokens=35,
        accepted_tokens_per_pos=(10, 8, 6, 2),
    )

    assert after - before == profile_workload.SpecMetrics(
        drafts=6,
        draft_tokens=30,
        accepted_tokens=23,
        accepted_tokens_per_pos=(6, 5, 4, 2),
    )


def test_conditional_acceptance_rates_use_reached_position_as_denominator() -> None:
    assert profile_workload.conditional_acceptance_rates(10, (8, 6, 3, 0, 0)) == [
        0.8,
        0.75,
        0.5,
        0.0,
        0.0,
    ]


def test_parse_spec_metrics_requires_spec_counter() -> None:
    assert profile_workload.parse_spec_metrics("vllm:num_requests 3") is None


def test_prompt_is_seeded_and_sized() -> None:
    prompt = profile_workload.make_prompt(32000, 42)

    assert prompt == profile_workload.make_prompt(32000, 42)
    assert prompt != profile_workload.make_prompt(32000, 43)
    assert len(prompt.split()) > 20000


def test_exact_prompt_uses_server_token_count(monkeypatch) -> None:
    def fake_count(_base_url: str, _model: str, prompt: str) -> int:
        return len(prompt.split()) + 4

    monkeypatch.setattr(profile_workload, "count_prompt_tokens", fake_count)

    prompt = profile_workload.make_exact_prompt("http://server", "model", 100, 42)

    assert fake_count("http://server", "model", prompt) == 100


def test_run_stream_passes_request_seed(monkeypatch) -> None:
    captured_payload = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def __iter__(self):
            return iter(
                (
                    b'data: {"choices":[{"text":"a"}]}\n',
                    b'data: {"usage":{"prompt_tokens":3,'
                    b'"completion_tokens":2},"choices":[{"text":"b"}]}\n',
                    b"data: [DONE]\n",
                )
            )

    def fake_urlopen(request, timeout):
        del timeout
        captured_payload.update(json.loads(request.data))
        return FakeResponse()

    monkeypatch.setattr(profile_workload.urllib.request, "urlopen", fake_urlopen)

    result = profile_workload.run_stream(
        "http://server",
        "model",
        "prompt",
        max_tokens=2,
        seed=17,
    )

    assert captured_payload["seed"] == 17
    assert result["prompt_tokens"] == 3
    assert result["completion_tokens"] == 2
