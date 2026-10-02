# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.sequence import IntermediateTensors
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu.pp_utils import (
    PPRecvBufferGuard,
    PPRecvBufferState,
    _snapshot_draft_tokens_for_pp,
    compute_need_sampled_mask,
    pack_speculative_pp_relay,
    pad_sampled_tokens,
    unpack_speculative_pp_relay,
)


def _batch(num_computed, prefill_len, num_scheduled):
    return Mock(
        num_reqs=len(num_computed),
        num_computed_tokens_np=np.array(num_computed, dtype=np.int32),
        prefill_len_np=np.array(prefill_len, dtype=np.int32),
        num_scheduled_tokens=np.array(num_scheduled, dtype=np.int32),
    )


def test_need_sampled_mask_excludes_non_final_prefill_chunks():
    batch = _batch(
        num_computed=[512, 1000],
        prefill_len=[4096, 1004],
        num_scheduled=[448, 4],
    )

    mask = compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [False, True]


def test_need_sampled_mask_returns_none_when_no_row_samples():
    batch = _batch(
        num_computed=[0, 512],
        prefill_len=[4096, 4096],
        num_scheduled=[448, 448],
    )

    assert compute_need_sampled_mask(batch) is None


def test_need_sampled_mask_keeps_decode_past_length_cap():
    """Decode rows must be relayed while the scheduler still runs them."""
    batch = _batch(
        num_computed=[14176],
        prefill_len=[12175],
        num_scheduled=[8],
    )

    mask = compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True]


def test_need_sampled_mask_handles_decode_before_prefill_chunk():
    batch = _batch(
        num_computed=[10, 512],
        prefill_len=[8, 4096],
        num_scheduled=[1, 448],
    )

    mask = compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True, False]


def test_pad_sampled_tokens_matches_fixed_collective_width():
    sampled = torch.tensor([[11], [22]], dtype=torch.int64)

    padded = pad_sampled_tokens(sampled, 4)

    assert padded.tolist() == [[11, -1, -1, -1], [22, -1, -1, -1]]
    assert padded.dtype == torch.int64


def test_pad_sampled_tokens_preserves_full_width_buffer():
    sampled = torch.tensor([[11, 12, 13]], dtype=torch.int64)

    assert pad_sampled_tokens(sampled, 3) is sampled


def test_pad_sampled_tokens_rejects_collective_count_mismatch():
    sampled = torch.tensor([[11, 12]], dtype=torch.int64)

    with pytest.raises(ValueError, match="exceeds PP buffer width"):
        pad_sampled_tokens(sampled, 1)


def test_snapshot_draft_tokens_for_pp_owns_storage():
    draft_tokens = torch.tensor([[11, 12, 13]], dtype=torch.int64)

    snapshot = _snapshot_draft_tokens_for_pp(draft_tokens)
    draft_tokens.fill_(-1)

    assert snapshot.tolist() == [[11, 12, 13]]
    assert snapshot.data_ptr() != draft_tokens.data_ptr()


def test_speculative_pp_relay_round_trip_uses_one_fixed_width_buffer():
    sampled = torch.tensor([[21], [31]], dtype=torch.int64)
    drafts = torch.tensor([[22, 23, 24], [32, 33, 34]], dtype=torch.int64)
    num_sampled = torch.tensor([2, 4], dtype=torch.int32)
    num_rejected = torch.tensor([2, 0], dtype=torch.int32)

    relay = pack_speculative_pp_relay(
        sampled, drafts, num_sampled, num_rejected, max_sample_len=4
    )
    unpacked = unpack_speculative_pp_relay(relay, max_sample_len=4)

    assert relay.shape == (2, 9)
    assert unpacked[0].tolist() == [[21, -1, -1, -1], [31, -1, -1, -1]]
    assert unpacked[1].tolist() == [2, 4]
    assert unpacked[2].tolist() == [2, 0]
    assert unpacked[3].tolist() == [[22, 23, 24], [32, 33, 34]]


def test_unpack_speculative_pp_relay_returns_contiguous_counts():
    relay = torch.arange(18, dtype=torch.int64).reshape(2, 9)

    _, num_sampled, num_rejected, _ = unpack_speculative_pp_relay(
        relay, max_sample_len=4
    )

    assert num_sampled.is_contiguous()
    assert num_rejected.is_contiguous()


@pytest.mark.parametrize("access", ["getitem", "tensors"])
def test_async_intermediate_tensors_lazy_wait_order(access):
    events = []
    tensor = torch.arange(4, dtype=torch.float32)

    class RecordingWork:
        def __init__(self):
            self.wait_calls = 0

        def wait(self):
            events.append("handle.wait")
            self.wait_calls += 1

    work = RecordingWork()

    def receive():
        events.append("receive")
        return {"x": tensor}, [work], [postprocess]

    def postprocess():
        events.append("postprocess")

    def complete():
        events.append("comm_complete_callback")

    it = gpu_worker.AsyncIntermediateTensors(
        {}, comm_start_callback=receive, comm_complete_callback=complete
    )

    assert it._comm_handles is None
    assert it._comm_waited is False
    assert events == []

    if access == "getitem":
        torch.testing.assert_close(it["x"], tensor)
    else:
        torch.testing.assert_close(it.tensors["x"], tensor)
    assert events == [
        "receive",
        "handle.wait",
        "postprocess",
        "comm_complete_callback",
    ]

    _ = it["x"]
    it.wait_for_comm()
    assert events == [
        "receive",
        "handle.wait",
        "postprocess",
        "comm_complete_callback",
    ]
    assert work.wait_calls == 1


def test_async_intermediate_tensors_slice_preserves_deferred_views():
    events = []
    tensor = torch.arange(12, dtype=torch.float32).reshape(6, 2)

    def receive():
        events.append("receive")
        return {"x": tensor}, [], []

    it = gpu_worker.AsyncIntermediateTensors({}, comm_start_callback=receive)
    sliced = it[1:4]

    assert events == ["receive"]
    torch.testing.assert_close(sliced["x"], tensor[1:4])
    assert sliced["x"].data_ptr() == tensor[1:4].data_ptr()


def test_async_intermediate_tensors_completes_pp_buffer_once():
    events = []
    tensor = torch.tensor([1])
    guard = PPRecvBufferGuard()
    generation = guard.begin_receive()
    callback_calls = 0

    def receive():
        events.append("receive")
        return {"x": tensor}, [], []

    def complete():
        nonlocal callback_calls
        callback_calls += 1
        events.append("comm_complete_callback")
        guard.finish_receive(generation)

    it = gpu_worker.AsyncIntermediateTensors(
        {}, comm_start_callback=receive, comm_complete_callback=complete
    )
    _ = it["x"]
    it.wait_for_comm()

    assert events == ["receive", "comm_complete_callback"]
    assert callback_calls == 1
    assert guard.state is PPRecvBufferState.MODEL


def _target_prepare_ahead_config(enabled=True):
    return SimpleNamespace(
        additional_config={"dsv41_sm80_target_prepare_ahead": enabled},
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="deepseek_v41",
                hidden_size=5120,
                n_routed_experts=384,
                num_hidden_layers=40,
            )
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=5,
            tensor_parallel_size=1,
            data_parallel_size=1,
        ),
        speculative_config=SimpleNamespace(
            method="dspark",
            num_speculative_tokens=5,
            enable_adaptive_verification=False,
        ),
    )


def _patch_target_prepare_ahead_platform(monkeypatch, *, cuda=True, sm80=True):
    monkeypatch.setattr(
        gpu_worker,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: cuda,
            is_device_capability=lambda capability, _: sm80 and capability == 80,
        ),
    )


@pytest.mark.parametrize(
    ("additional_config", "expected"),
    [
        ({}, False),
        ({"dsv41_sm80_target_prepare_ahead": False}, False),
        ({"dsv41_sm80_target_prepare_ahead": True}, True),
    ],
)
def test_dsv41_target_prepare_ahead_default_and_exact_target(
    monkeypatch, additional_config, expected
):
    _patch_target_prepare_ahead_platform(monkeypatch)
    config = _target_prepare_ahead_config()
    config.additional_config = additional_config

    assert (
        gpu_worker._dsv41_sm80_target_prepare_ahead(
            config, True, torch.device("cuda", 0)
        )
        is expected
    )


@pytest.mark.parametrize(
    ("target_name", "field", "value"),
    [
        ("hf", "model_type", "deepseek_v40"),
        ("hf", "hidden_size", 4096),
        ("parallel", "pipeline_parallel_size", 4),
        ("parallel", "tensor_parallel_size", 2),
        ("parallel", "data_parallel_size", 2),
        ("speculative", "method", "eagle"),
        ("speculative", "num_speculative_tokens", 4),
        ("speculative", "enable_adaptive_verification", True),
    ],
)
def test_dsv41_target_prepare_ahead_rejects_target_mismatch(
    monkeypatch, target_name, field, value
):
    _patch_target_prepare_ahead_platform(monkeypatch)
    config = _target_prepare_ahead_config()
    target = {
        "hf": config.model_config.hf_config,
        "parallel": config.parallel_config,
        "speculative": config.speculative_config,
    }[target_name]
    setattr(target, field, value)

    assert not gpu_worker._dsv41_sm80_target_prepare_ahead(
        config, True, torch.device("cuda", 0)
    )


@pytest.mark.parametrize(
    ("cuda", "sm80", "device"),
    [(False, True, torch.device("cpu")), (True, False, torch.device("cuda", 0))],
)
def test_dsv41_target_prepare_ahead_rejects_non_target_platform(
    monkeypatch, cuda, sm80, device
):
    _patch_target_prepare_ahead_platform(monkeypatch, cuda=cuda, sm80=sm80)
    config = _target_prepare_ahead_config()

    assert not gpu_worker._dsv41_sm80_target_prepare_ahead(config, True, device)


def test_dsv41_target_prepare_ahead_rejects_v1_and_missing_dspark(monkeypatch):
    _patch_target_prepare_ahead_platform(monkeypatch)
    config = _target_prepare_ahead_config()

    assert not gpu_worker._dsv41_sm80_target_prepare_ahead(
        config, False, torch.device("cuda", 0)
    )

    config.speculative_config = None
    assert not gpu_worker._dsv41_sm80_target_prepare_ahead(
        config, True, torch.device("cuda", 0)
    )


def test_dsv41_target_prepare_ahead_rejects_non_boolean_config(monkeypatch):
    _patch_target_prepare_ahead_platform(monkeypatch)
    config = _target_prepare_ahead_config("true")

    with pytest.raises(ValueError, match="must be a JSON boolean"):
        gpu_worker._dsv41_sm80_target_prepare_ahead(
            config, True, torch.device("cuda", 0)
        )


@pytest.mark.parametrize("defer_receive", [False, True])
def test_worker_execute_model_receive_prepare_order(monkeypatch, defer_receive):
    events = []

    class RecordingWork:
        def wait(self):
            events.append("wait")

    class FakePPGroup:
        is_first_rank = False
        is_last_rank = False

        def irecv_tensor_dict(self, **kwargs):
            events.append("receive")
            return (
                {"x": torch.tensor([1])},
                [RecordingWork()],
                [lambda: events.append("postprocess")],
            )

        def isend_tensor_dict(self, tensors, **kwargs):
            assert list(tensors) == ["x"]
            events.append("send")

    class FakeRunner:
        is_pooling_model = False

        def execute_model(self, scheduler_output, intermediate_tensors):
            events.append("prepare")
            assert intermediate_tensors is not None
            _ = intermediate_tensors.tensors
            events.append("forward")
            return IntermediateTensors({"x": torch.tensor([2])})

    worker = gpu_worker.Worker.__new__(gpu_worker.Worker)
    worker._reuse_pp_recv_buffer = False
    worker.use_v2_model_runner = True
    worker._defer_pp_recv = defer_receive
    worker.model_runner = FakeRunner()
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=False)
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2,
            distributed_executor_backend="mp",
        ),
    )
    worker.annotate_profile = lambda _: nullcontext()

    pp_group = FakePPGroup()
    monkeypatch.setattr(gpu_worker, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(gpu_worker, "get_tp_group", lambda: object())

    worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=1))

    assert events == (
        ["receive", "prepare", "wait", "postprocess", "forward", "send"]
        if not defer_receive
        else ["prepare", "receive", "wait", "postprocess", "forward", "send"]
    )
