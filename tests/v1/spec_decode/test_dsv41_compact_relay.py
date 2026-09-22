# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real stage relay with CPU stand-ins for model/kernel work.

The contract is unchanged final head inputs and request controls, with fewer
intermediate tensor sends. GPU timing and model quality need a separate A/B.
"""

import inspect
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from benchmarks.deepseek_v4_1.dspark_relay_bench import (
    make_hop_packages,
    wire_inventory,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import _split_tensor_dict
from vllm.models.deepseek_v4_1.dspark_relay import compact_relay_enabled
from vllm.v1.worker.gpu.spec_decode.dspark import speculator as module

pytestmark = pytest.mark.cpu_test
SAMPLING_KEYS = {
    "query_input_ids",
    "sample_indices",
    "sample_pos",
    "sample_idx_mapping",
}


def _config(
    target="deepseek_v41",
    draft="deepseek_v41",
    layers=40,
    enabled=True,
):
    return SimpleNamespace(
        additional_config={"dsv41_dspark_compact_relay": enabled},
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type=target, num_hidden_layers=layers)
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        speculative_config=SimpleNamespace(
            method="dspark",
            use_distributed_dspark=lambda: True,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type=draft)
            ),
        ),
    )


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, True),
        ({"enabled": False}, False),
        ({"enabled": "true"}, False),
        ({"target": "deepseek_v4"}, False),
        ({"draft": "deepseek_v4"}, False),
        ({"target": "qwen3"}, False),
        ({"layers": 43}, False),
    ],
)
def test_compact_relay_is_opt_in_and_v41_flash_only(kwargs, expected):
    assert compact_relay_enabled(_config(**kwargs)) is expected


def test_compact_relay_requires_distributed_dspark_tp1():
    config = _config()
    config.additional_config = {}
    assert not compact_relay_enabled(config)
    config.additional_config = {"dsv41_dspark_compact_relay": True}
    config.parallel_config.tensor_parallel_size = 2
    assert not compact_relay_enabled(config)
    config.parallel_config.tensor_parallel_size = 1
    config.speculative_config.use_distributed_dspark = lambda: False
    assert not compact_relay_enabled(config)
    config.speculative_config = None
    assert not compact_relay_enabled(config)


def test_relay_profile_extra_reports_wire_inventory(monkeypatch):
    monkeypatch.setattr(module, "phase_profile_enabled", lambda flow: flow == "dspark")
    package = {
        "hidden_states": torch.empty(2, 4, dtype=torch.bfloat16),
        "sample_indices": torch.empty(3, dtype=torch.int64),
        "scheduler_step_id": 7,
    }

    extra = module._relay_profile_extra(package)

    assert extra is not None
    assert extra["tensor_count"] == 2
    assert extra["tensor_bytes"] == 2 * 4 * 2 + 3 * 8
    assert extra["tensor_keys"] == ["hidden_states", "sample_indices"]
    assert extra["non_tensor_keys"] == ["scheduler_step_id"]
    assert extra["tensor_bytes_by_dtype"] == {
        "torch.bfloat16": 16,
        "torch.int64": 24,
    }


def test_profiled_relay_send_splits_launch_and_handle_waits(monkeypatch):
    events: list[Any] = []

    class Handle:
        def __init__(self, name):
            self.name = name

        def wait(self):
            events.append(f"wait:{self.name}")

    handles = [Handle("metadata"), Handle("tensor-0"), Handle("tensor-1")]

    class Group:
        def __init__(self):
            self._pending_isends = []

        def isend_tensor_dict(self, package, dst):
            events.append(("isend", package, dst))
            self._pending_isends.append((handles, list(package.values())))
            return handles

        def send_tensor_dict(self, package, dst):
            raise AssertionError("profiled send must use isend_tensor_dict")

    @contextmanager
    def profile(stage, ref, **kwargs):
        events.append(f"enter:{stage}")
        yield

    group = Group()
    package = {"hidden_states": torch.ones(1)}
    monkeypatch.setattr(module, "phase_profile_enabled", lambda flow: True)
    monkeypatch.setattr(module, "phase_profile_stage", profile)

    module._send_relay_package(
        group,
        package,
        dst=2,
        ref=package["hidden_states"],
        stage="draft.stage.send",
    )

    assert events == [
        "enter:draft.stage.send.launch",
        ("isend", package, 2),
        "enter:draft.stage.send.metadata_wait",
        "wait:metadata",
        "enter:draft.stage.send.tensor_wait",
        "wait:tensor-0",
        "wait:tensor-1",
    ]
    assert group._pending_isends == []


def test_unprofiled_relay_send_keeps_synchronous_path(monkeypatch):
    calls = []
    group = SimpleNamespace(
        send_tensor_dict=lambda package, dst: calls.append((package, dst))
    )
    package = {"hidden_states": torch.ones(1)}
    monkeypatch.setattr(module, "phase_profile_enabled", lambda flow: False)

    module._send_relay_package(
        group,
        package,
        dst=2,
        ref=package["hidden_states"],
        stage="draft.stage.send",
    )

    assert calls == [(package, 2)]


def test_flat_state_transport_fixture_reduces_messages_without_changing_bytes():
    baseline = wire_inventory(make_hop_packages(1, compact=True))
    candidate = wire_inventory(make_hop_packages(1, compact=True, flat_state=True))

    assert baseline["tensor_messages"] == 44
    assert candidate["tensor_messages"] == 35
    assert candidate["tensor_bytes"] == baseline["tensor_bytes"] == 954512


def _relay(
    monkeypatch,
    *,
    compact,
    req_ids,
    has_prefill,
    dummy,
    graph,
    views=None,
):
    count, width = len(req_ids), 5
    rows = count * width
    batch = SimpleNamespace(
        req_ids=req_ids,
        num_reqs=count,
        num_tokens=count * 6,
        input_ids=torch.arange(count * 6),
        has_prefill=has_prefill,
    )
    package = dict(
        scheduler_step_id=-1 if dummy else 17,
        dummy_run=dummy,
        req_ids=tuple(req_ids),
        num_reqs=count,
        skip_attn=False,
        idx_mapping=torch.arange(count - 1, -1, -1),
        context_main_x=torch.ones(count * 6, 2),
        num_sampled=torch.ones(count, dtype=torch.int32),
        num_rejected=torch.zeros(count, dtype=torch.int32),
        last_sampled=torch.arange(count),
        next_prefill_tokens=torch.zeros(count),
        temperature=torch.zeros(count),
        seeds=torch.arange(count),
    )
    controls = {key: package[key] for key in package}
    messages = []
    prepare_signature = inspect.signature(module.prepare_dflash_inputs)

    class SampleBuffer:
        def __init__(self, tensor):
            self.tensor = tensor

        def copy_(self, value):
            self.tensor.copy_(value)

        def __getitem__(self, index):
            if views is not None:
                views.append(index)
            return self.tensor[index]

    def prepare_inputs(*args, **kwargs):
        values = prepare_signature.bind(*args, **kwargs).arguments
        # Use the actual _prepare_stage_inputs argument plumbing, including
        # request remapping; intermediate sampling fields are not its inputs.
        idx = values["request_idx_mapping"]
        torch.testing.assert_close(idx, controls["idx_mapping"])
        values["input_buffers"].input_ids.copy_(idx.repeat_interleave(width))
        values["sample_indices"].copy_(torch.arange(rows))
        values["sample_pos"].copy_(torch.arange(rows) + 101)
        values["sample_idx_mapping"].copy_(idx.repeat_interleave(width))

    monkeypatch.setattr(module, "prepare_dflash_inputs", prepare_inputs)
    monkeypatch.setattr(module, "set_forward_context", lambda *a, **k: nullcontext())
    monkeypatch.setattr(
        module,
        "dispatch_cg_and_sync_dp",
        lambda *a, **k: (
            SimpleNamespace(
                cg_mode=CUDAGraphMode.FULL if graph else CUDAGraphMode.NONE
            ),
            None,
        ),
    )

    for stage in range(3):
        spec = module.DSparkSpeculator.__new__(module.DSparkSpeculator)
        spec.distributed = True
        spec._compact_v41_relay = compact
        spec._distributed_step_id = controls["scheduler_step_id"]
        spec._stage_index, spec._stage_ranks, spec._head_rank = stage, (1, 2, 3), 4
        spec.num_query_per_req = spec.num_speculative_steps = width
        spec.input_buffers = SimpleNamespace(
            input_ids=torch.empty(rows, dtype=torch.int32), positions=torch.arange(rows)
        )
        spec.context_positions = torch.zeros(batch.num_tokens)
        spec._context_slot_mappings = torch.zeros(1, batch.num_tokens)
        for key in ("sample_indices", "sample_pos", "sample_idx_mapping"):
            setattr(
                spec,
                key,
                SampleBuffer(
                    torch.empty(
                        rows,
                        dtype=torch.int64
                        if key != "sample_idx_mapping"
                        else torch.int32,
                    )
                ),
            )
        spec.temperature = spec.seeds = torch.empty(count)
        spec.draft_kv_cache_group_ids = [0]
        spec.block_tables = SimpleNamespace(
            slot_mappings=torch.zeros(1, rows),
            input_block_tables=[torch.zeros(count, 1)],
            kernel_block_sizes=[16],
            cp_rank=0,
            cp_size=1,
            cp_interleave=1,
        )
        spec.parallel_drafting_token_id = 0
        spec.max_num_reqs, spec.max_num_tokens, spec.max_model_len = count, rows, 1024
        spec.sample_from_anchor = True
        spec.dp_size, spec.dp_rank = 1, 0
        spec.vllm_config = None
        spec.staging = module._DSparkStageStaging(rows, 2, 2, torch.float32, "cpu")
        spec._note_graph_replay = lambda: None
        spec.model = SimpleNamespace(
            prepare_pipeline_query=lambda ids: ids.float().unsqueeze(1).expand(-1, 2),
            forward_local_stage=lambda hidden, pos, ids, *state: (
                hidden + 1,
                torch.ones(rows, 2, 2),
                torch.ones(rows, 2, 1),
                torch.ones(rows, 2, 2),
                torch.ones(rows, 2),
            ),
        )
        spec.query_cudagraph_manager = SimpleNamespace(
            run_fullgraph=lambda desc, spec=spec: spec._draft_stage_forward(
                count, rows, None, None, None, CUDAGraphMode.FULL
            )
        )

        def prepare(batch, package, spec=spec):
            spec._prepare_stage_inputs(batch, package)
            return rows, None, None

        spec._prepare_distributed_stage = prepare
        spec._pp_group = SimpleNamespace(
            recv_tensor_dict=lambda src, package=package: package,
            send_tensor_dict=lambda msg, dst: messages.append(dict(msg)),
        )
        inspect.unwrap(module.DSparkSpeculator.participate_distributed)(spec, batch)
        package = dict(messages[-1])
        if stage < 2:
            for key, value in controls.items():
                assert package[key] is value
    return messages


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize(
    "has_prefill,dummy", [(False, False), (True, False), (False, True)]
)
def test_v41_relay_preserves_head_inputs_and_only_trims_decode(
    monkeypatch, record_property, graph, has_prefill, dummy
):
    # Shrinking/reordered batches exercise the row count and request identity
    # independently of the wire optimization (which retains no cross-step state).
    for req_ids in (["r0", "r1"], ["r1"], ["r1", "r2", "r0"]):
        common = dict(
            req_ids=req_ids, has_prefill=has_prefill, dummy=dummy, graph=graph
        )
        baseline_views: list[Any] = []
        candidate_views: list[Any] = []
        with monkeypatch.context() as patch:
            baseline = _relay(patch, compact=False, views=baseline_views, **common)
        with monkeypatch.context() as patch:
            candidate = _relay(patch, compact=True, views=candidate_views, **common)
        assert len(baseline_views) == 9
        assert len(candidate_views) == (3 if not has_prefill and not dummy else 9)
        if not has_prefill and not dummy:
            fixture_hops = make_hop_packages(
                len(req_ids), compact=True, hidden=2, hc_mult=2
            )
            for runtime, fixture in zip(candidate, fixture_hops[1:]):
                assert runtime.keys() == fixture.keys()
                for key in SAMPLING_KEYS & runtime.keys():
                    assert runtime[key].dtype == fixture[key].dtype
                    assert runtime[key].shape == fixture[key].shape
        for key, expected in baseline[-1].items():
            actual = candidate[-1][key]
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(actual, expected)
            else:
                assert actual == expected
        assert candidate[-1].keys() == baseline[-1].keys()
        before_tensors = [t for msg in baseline for t in _split_tensor_dict(msg)[1]]
        after_tensors = [t for msg in candidate for t in _split_tensor_dict(msg)[1]]
        expected_sends_saved = 8 if not has_prefill and not dummy else 0
        assert len(before_tensors) - len(after_tensors) == expected_sends_saved
        saved_bytes = sum(t.numel() * t.element_size() for t in before_tensors) - sum(
            t.numel() * t.element_size() for t in after_tensors
        )
        assert saved_bytes == (240 * len(req_ids) if expected_sends_saved else 0)
        record_property(f"B{len(req_ids)}_sends_saved", expected_sends_saved)
        record_property(f"B{len(req_ids)}_bytes_saved", saved_bytes)
        for before, after in zip(baseline[:2], candidate[:2]):
            removed = set(before) - set(after)
            assert removed == (
                SAMPLING_KEYS if not has_prefill and not dummy else set()
            )
