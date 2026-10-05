# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import io
import json
from typing import Any

import pytest

from benchmarks.deepseek_v4_1 import dspark_flow_gate as gate
from benchmarks.deepseek_v4_1.concurrency_probe import (
    _read_metrics,
    summarize_requests,
)

pytestmark = pytest.mark.cpu_test


def _stream(monkeypatch, ids, usage_tokens, *, done=True):
    chunks: list[dict[str, Any]] = [
        {
            "choices": [
                {
                    "index": 0,
                    "text": "",
                    "token_ids": row,
                    "finish_reason": "length" if i == len(ids) - 1 else None,
                }
            ]
        }
        for i, row in enumerate(ids)
    ]
    chunks.append(
        {
            "choices": [],
            "usage": {"completion_tokens": usage_tokens, "prompt_tokens": 2},
        }
    )
    data = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
    if done:
        data += "data: [DONE]\n\n"

    def post(url, payload, timeout):
        assert payload["return_token_ids"] is True
        return io.BytesIO(data.encode())

    clock = iter(range(20))
    monkeypatch.setattr(gate, "_post", post)
    monkeypatch.setattr(gate.time, "perf_counter", lambda: next(clock))
    return gate.run_once("unused", "dsv41f", [1, 2], usage_tokens, 1)


def test_subtracts_all_first_chunk_tokens_and_counts_empty_text(monkeypatch):
    row = _stream(monkeypatch, [[10, 11, 12], [13, 14], [15]], 6)
    assert row["token_ids"] == [10, 11, 12, 13, 14, 15]
    assert row["first_chunk_tokens"] == 3
    assert row["post_first_chunk_tokens"] == 3
    assert row["post_first_chunk_tok_per_s"] == 1.5
    assert row["request_tok_per_s"] == 1.5  # six tokens / four seconds incl. TTFT


def test_one_chunk_has_no_decode_interval(monkeypatch):
    row = _stream(monkeypatch, [[10, 11]], 2)
    assert row["post_first_chunk_tok_per_s"] is None


@pytest.mark.parametrize(
    "ids,usage,done,match",
    [
        ([[10], [11]], 3, True, "Delta token count"),
        ([None], 1, True, "delta token_ids"),
        ([[10]], 1, False, "Incomplete stream"),
    ],
)
def test_incomplete_or_unverifiable_output_is_rejected(
    monkeypatch, ids, usage, done, match
):
    with pytest.raises(ValueError, match=match):
        _stream(monkeypatch, ids, usage, done=done)


def test_engine_metric_snapshot_keeps_labeled_counters(monkeypatch):
    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()

    body = b"""# HELP ignored
vllm:spec_decode_num_draft_tokens_total{engine=\"0\"} 10
vllm:spec_decode_num_accepted_tokens_total{engine=\"0\"} 8
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine=\"0\",position=\"0\"} 4
vllm:spec_decode_num_accepted_tokens_created{engine=\"0\"} 123
vllm:prompt_tokens_total{engine=\"0\"} 20
vllm:prefix_cache_hits_total{engine=\"0\"} 0
vllm:request_prefill_time_seconds_sum{engine=\"0\"} 2.5
vllm:request_prefill_time_seconds_count{engine=\"0\"} 1
"""
    monkeypatch.setattr(
        gate.urllib.request,
        "urlopen",
        lambda url, timeout: Response(body),
    )
    snapshot = gate.read_engine_metrics("http://unused", 1)
    assert snapshot["values"] == {
        'vllm:spec_decode_num_draft_tokens_total{engine="0"}': 10.0,
        'vllm:spec_decode_num_accepted_tokens_total{engine="0"}': 8.0,
        (
            "vllm:spec_decode_num_accepted_tokens_per_pos_total"
            '{engine="0",position="0"}'
        ): 4.0,
        'vllm:prompt_tokens_total{engine="0"}': 20.0,
        'vllm:prefix_cache_hits_total{engine="0"}': 0.0,
        'vllm:request_prefill_time_seconds_sum{engine="0"}': 2.5,
        'vllm:request_prefill_time_seconds_count{engine="0"}': 1.0,
    }

    after = {"values": {key: value + 1 for key, value in snapshot["values"].items()}}
    assert all(value == 1.0 for value in gate.metric_deltas(snapshot, after).values())


def test_concurrent_decode_includes_staggered_arrival_and_entire_first_chunks():
    """Separate request durations must not inflate aggregate throughput."""
    requests = [
        {
            "first_token_s": 1.0,
            "last_token_s": 3.0,
            "completion_tokens": 9,
            "post_first_chunk_tokens": 6,
        },
        {
            "first_token_s": 4.0,
            "last_token_s": 6.0,
            "completion_tokens": 9,
            "post_first_chunk_tokens": 6,
        },
    ]
    summary = summarize_requests(requests, wall_s=6.0)
    assert summary["aggregate_decode_span_s"] == 5.0
    assert summary["aggregate_decode_tok_s"] == 12 / 5
    assert summary["aggregate_output_tok_s"] == 18 / 6


def test_concurrency_probe_retains_labeled_draft_and_preemption_metrics():
    """Production metrics have engine/model labels; dropping them hides failures."""
    body = """vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 12
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} 48
vllm:num_preemptions_total{engine="0",model_name="m"} 1
vllm:num_preemptions_created{engine="0",model_name="m"} 123
"""
    values = _read_metrics(body)
    assert values['vllm:spec_decode_num_drafts_total{engine="0",model_name="m"}'] == 12
    assert values['vllm:num_preemptions_total{engine="0",model_name="m"}'] == 1
    assert not any("_created" in key for key in values)
