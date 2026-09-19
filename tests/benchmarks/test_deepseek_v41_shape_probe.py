# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import io
import json
from pathlib import Path

import pytest

PATH = (
    Path(__file__).parents[2] / "benchmarks/deepseek_v4_1/prefix_cache_shape_probe.py"
)
SPEC = importlib.util.spec_from_file_location("shape_probe", PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_logprob_selects_target_token_instead_of_first_candidate(monkeypatch):
    response = {
        "choices": [
            {
                "prompt_logprobs": [
                    None,
                    {
                        "7": {"logprob": -0.1},
                        "9": {"logprob": -2.0},
                    },
                ]
            }
        ],
        "usage": {"prompt_tokens_details": {"cached_tokens": 0}},
    }
    monkeypatch.setattr(
        probe, "_post", lambda *a, **k: io.BytesIO(json.dumps(response).encode())
    )
    result = probe.probe_logprob("http://unused", [3, 9], 2, 1, "unique")
    assert result["logprob"] == -2.0
    assert result["predictor_position"] == 0
    assert result["target_token_id"] == 9


@pytest.mark.parametrize(
    ("query_delta", "hit_delta", "expected_ok"),
    [
        (0, 0, True),
        (0, 128, False),
        (0, -1, False),
        (7000, 0, False),
        (-1, 0, False),
    ],
)
def test_cold_evidence_requires_no_lookup_and_no_hits(
    monkeypatch, query_delta, hit_delta, expected_ok
):
    """``prompt_logprobs`` skips the cache lookup, so any query is a bug."""
    keys = [f"{name}|model=x" for name in probe.PREFIX_COUNTERS]
    metrics = iter(
        [
            {keys[0]: 200, keys[1]: 128},
            {keys[0]: 200 + query_delta, keys[1]: 128 + hit_delta},
        ]
    )
    monkeypatch.setattr(probe, "scrape", lambda _: next(metrics))
    monkeypatch.setattr(
        probe,
        "probe_logprob",
        lambda *a: {
            "logprob": -1.0,
            "predictor_position": 0,
            "target_token_id": 9,
            "usage_cached_tokens": None,
        },
    )
    row = probe._run_request("http://unused", [3, 9], 2, 1, "unique", 0, None)
    assert (row["status"] == "ok") == expected_ok
    assert row["cold_hits"] == hit_delta
    assert row["estimated_tail_chunk"] is None
    if query_delta == 7000:
        assert "prefix-cache lookup" in row["error"]


def test_cold_evidence_requires_metrics_before_sending_request(monkeypatch):
    monkeypatch.setattr(probe, "scrape", lambda _: {})
    monkeypatch.setattr(
        probe, "probe_logprob", lambda *a: pytest.fail("must not send request")
    )
    row = probe._run_request("http://unused", [3, 9], 2, 1, "unique", 0, None)
    assert row["status"] == "error"
    assert "missing prefix-cache counters" in row["error"]


def test_requested_length_must_include_target():
    with pytest.raises(ValueError, match="greater than probe"):
        probe._parse_lengths(["1300"], 1300, 1620, 4)
