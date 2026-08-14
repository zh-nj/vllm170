# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
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


def test_parse_spec_metrics_ignores_per_position_counters() -> None:
    payload = """
vllm:spec_decode_num_drafts_total{model_name="dsv4s"} 10
vllm:spec_decode_num_draft_tokens_total{model_name="dsv4s"} 50
vllm:spec_decode_num_accepted_tokens_total{model_name="dsv4s"} 37
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 10
"""

    assert profile_workload.parse_spec_metrics(payload) == (
        profile_workload.SpecMetrics(
            drafts=10,
            draft_tokens=50,
            accepted_tokens=37,
        )
    )


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
