# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for the bounded decoder SWA replay row plan."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v4_1 import fast_prefill
from vllm.models.deepseek_v4_1.fast_prefill import (
    FastPrefillMetadata,
    FastPrefillStep,
    TailRows,
    fast_prefill_enabled,
    plan_tail_rows,
    tail_layer,
)

WINDOW = 128


def _plan(chunks, decodes=(), window=WINDOW, num_decode_tokens=None):
    """Plan one batch: `decodes` are per-request row counts, then `chunks`."""
    starts = [0]
    for rows in (*decodes, *chunks):
        starts.append(starts[-1] + rows)
    prefilling = np.array([False] * len(decodes) + [True] * len(chunks), dtype=bool)
    if num_decode_tokens is None:
        num_decode_tokens = sum(decodes)
    return plan_tail_rows(
        np.asarray(starts, dtype=np.int64),
        prefilling,
        window,
        num_decode_tokens=num_decode_tokens,
    )


@pytest.mark.parametrize("rows", [1, 2, 127, 128])
def test_short_chunks_are_left_unbounded(rows):
    # FastLLM only bounds a segment longer than the window.
    assert _plan([rows]) is None


def test_a_long_chunk_keeps_its_last_window():
    plan = _plan([129])
    assert plan.num_tokens == WINDOW
    assert np.array_equal(plan.keep, np.arange(1, 129))
    assert plan.truncated.tolist() == [True]
    assert plan.gather_clamp.tolist() == [WINDOW]
    assert plan.query_start_loc.tolist() == [0, WINDOW]


def test_every_decode_row_survives_and_only_prefills_are_bounded():
    # Two decode requests (one plain, one with a 5-wide DSpark draft) and a
    # long prefill chunk: the tail layers must still verify every draft row.
    plan = _plan([300], decodes=(1, 6))
    expected = np.concatenate([np.arange(7), np.arange(179, 307)])
    assert np.array_equal(plan.keep, expected)
    assert plan.query_start_loc.tolist() == [0, 1, 7, 135]
    assert plan.gather_clamp.tolist() == [WINDOW]
    assert plan.truncated.tolist() == [False, False, True]
    assert plan.num_reqs == 3


def test_a_boundary_chunk_and_a_short_final_chunk_stay_unbounded():
    assert _plan([129]).num_tokens == WINDOW
    for rows in (1, 2, 127, 128):
        assert _plan([rows]) is None


def test_a_short_chunk_still_reads_its_real_history_window():
    # Not truncated -> -1 keeps the builder's own gather length (chunk + the
    # 127 positions before it), which the previous chunk's tail did write.
    plan = _plan([600, 40])
    assert plan.truncated.tolist() == [True, False]
    assert plan.gather_clamp.tolist() == [WINDOW, -1]
    assert plan.num_tokens == WINDOW + 40


def test_multiple_long_chunks_keep_one_window_each():
    plan = _plan([1020, 900])
    assert plan.num_tokens == 2 * WINDOW
    assert plan.query_start_loc.tolist() == [0, WINDOW, 2 * WINDOW]
    assert np.array_equal(plan.keep[:2], np.array([892, 893]))
    assert np.array_equal(plan.keep[-2:], np.array([1918, 1919]))


def test_request_order_does_not_change_the_row_plan():
    forward = _plan([1020, 300, 130])
    # Same requests, decodes still first, prefills in another order.
    other = _plan([130, 300, 1020])
    assert forward.num_tokens == other.num_tokens
    assert sorted(forward.gather_clamp.tolist()) == sorted(other.gather_clamp.tolist())
    assert forward.truncated.sum() == other.truncated.sum() == 3


def test_the_sampled_row_is_always_kept():
    # logits_indices point at each request's last row; the plan must contain it.
    for chunks in ([129], [1020, 3], [7, 1020], [1020, 900, 1]):
        plan = _plan(chunks)
        assert plan is not None
        ends = np.cumsum(chunks)
        for end in ends:
            assert end - 1 in set(plan.keep.tolist())


def test_the_batch_layout_must_be_decodes_then_prefills():
    with pytest.raises(ValueError, match="decodes.*prefills"):
        _plan([300], decodes=(6,), num_decode_tokens=5)
    with pytest.raises(ValueError, match="decodes.*prefills"):
        plan_tail_rows(
            np.array([0, 4, 10], dtype=np.int64),
            np.array([True, False]),
            WINDOW,
            num_decode_tokens=6,
        )


def test_malformed_requests_are_rejected():
    with pytest.raises(ValueError, match="num_reqs \\+ 1"):
        plan_tail_rows(
            np.array([0, 4]), np.array([True, True]), WINDOW, num_decode_tokens=0
        )
    with pytest.raises(ValueError, match="no scheduled rows"):
        plan_tail_rows(
            np.array([0, 4, 4]), np.array([True, True]), WINDOW, num_decode_tokens=0
        )
    with pytest.raises(ValueError, match="positive sliding window"):
        plan_tail_rows(np.array([0, 300]), np.array([True]), 0, num_decode_tokens=0)


def test_plan_rejects_inconsistent_index_arrays():
    good = _plan([300])
    with pytest.raises(ValueError, match="start at zero"):
        TailRows(
            keep=good.keep,
            query_start_loc=np.array([1, 129], dtype=np.int32),
            gather_clamp=good.gather_clamp,
            truncated=good.truncated,
        )
    with pytest.raises(ValueError, match="end at num_tokens"):
        TailRows(
            keep=good.keep,
            query_start_loc=np.array([0, 129], dtype=np.int32),
            gather_clamp=good.gather_clamp,
            truncated=good.truncated,
        )
    with pytest.raises(ValueError, match="ascend"):
        TailRows(
            keep=np.array([5, 5], dtype=np.int64),
            query_start_loc=np.array([0, 2], dtype=np.int32),
            gather_clamp=np.array([2], dtype=np.int32),
            truncated=np.array([True]),
        )


def test_plan_moves_to_the_device_with_the_expected_dtypes():
    plan = _plan([300])
    device_plan = plan.to_device("cpu")
    assert str(device_plan.keep.dtype) == "torch.int64"
    assert str(device_plan.query_start_loc.dtype) == "torch.int32"
    assert str(device_plan.gather_clamp.dtype) == "torch.int32"
    assert int(device_plan.query_start_loc[-1]) == plan.num_tokens


def _swa_window(prefix, chunk, gather, window, i):
    """Mirror of the Ampere combine kernel's local window for chunk row `i`.

    Returns the in-block offsets of the row's sliding window, so the caller can
    check that the window neither leaves the gathered block nor is shortened
    when the block already covers it.
    """
    position = prefix + i
    swa_len = min(max(position + 1, 0), window)
    swa_len = min(swa_len, i + gather - chunk + 1)
    own = i + gather - chunk
    return [own + column - swa_len + 1 for column in range(swa_len)], own


@pytest.mark.parametrize(
    "prefix,chunk", [(0, 300), (7, 129), (1020, 1020), (5000, 900)]
)
def test_unbounded_gather_keeps_the_full_local_window(prefix, chunk):
    # gather = chunk + min(prefix, window - 1) is what the metadata builder
    # produces without the clip; the clamp must never shorten those windows.
    gather = chunk + min(prefix, WINDOW - 1)
    for i in range(chunk):
        offsets, own = _swa_window(prefix, chunk, gather, WINDOW, i)
        assert offsets[0] >= 0
        assert offsets[-1] == own
        assert offsets == list(range(own - len(offsets) + 1, own + 1))


def test_bounded_gather_masks_instead_of_reaching_before_the_block():
    # A long chunk bounded to its last window rows: the first kept row's real
    # window reaches into rows the tail layers never computed at all.
    prefix, chunk, keep = 1020, 1020, WINDOW
    for i in range(chunk - keep, chunk):
        offsets, own = _swa_window(prefix, chunk, keep, WINDOW, i)
        assert offsets[0] >= 0, (i, offsets)
        assert offsets[-1] == own
        assert offsets == list(range(own - len(offsets) + 1, own + 1))
    # Without the clamp that first kept row would address before the block.
    first_kept = chunk - keep
    own = first_kept + keep - chunk
    unclamped_len = min(prefix + first_kept + 1, WINDOW)
    assert own - unclamped_len + 1 < 0


def test_short_chunk_windows_are_untouched_by_the_clamp():
    # A chunk no longer than the window is never truncated, so its window still
    # reaches into history that earlier chunks did write.
    for chunk in (1, 2, 127, 128):
        for prefix in (0, 1, 500):
            gather = chunk + min(prefix, WINDOW - 1)
            for i in range(chunk):
                offsets, own = _swa_window(prefix, chunk, gather, WINDOW, i)
                assert offsets[0] >= 0 and offsets[-1] == own


def test_tail_layer_is_one_past_the_last_kv_source():
    config = SimpleNamespace(
        kv_source_layer_ids=[2, 8, 14, 20],
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
        num_hidden_layers=40,
    )
    assert tail_layer(config) == 21


def test_tail_layer_rejects_a_late_wide_compression_layer():
    config = SimpleNamespace(
        kv_source_layer_ids=[2, 8, 14, 20],
        compress_ratios=[0, 0] + [2] * 18 + [1] * 10 + [2] + [1] * 9 + [0, 0, 0],
        num_hidden_layers=40,
    )
    with pytest.raises(ValueError, match="cannot bound layer 30"):
        tail_layer(config)


def test_tail_layer_needs_sources_and_ratios():
    with pytest.raises(ValueError, match="kv_source_layer_ids"):
        tail_layer(SimpleNamespace(kv_source_layer_ids=[]))
    with pytest.raises(ValueError, match="compress_ratios"):
        tail_layer(SimpleNamespace(kv_source_layer_ids=[2], compress_ratios=None))


def _vllm_config(**overrides):
    hf_config = SimpleNamespace(
        kv_source_layer_ids=[2, 8, 14, 20],
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
        num_hidden_layers=40,
    )
    values = dict(
        additional_config={"dsv41_fast_prefill": True},
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=False),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        model_config=SimpleNamespace(hf_config=hf_config),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_fast_prefill_is_opt_in():
    assert not fast_prefill_enabled(_vllm_config(additional_config={}))
    assert not fast_prefill_enabled(_vllm_config(additional_config=None))
    assert fast_prefill_enabled(_vllm_config())


def test_fast_prefill_allows_prefix_caching():
    assert fast_prefill_enabled(
        _vllm_config(cache_config=SimpleNamespace(enable_prefix_caching=True))
    )


@pytest.mark.parametrize(
    "overrides,match",
    [
        (
            {
                "additional_config": {
                    "dsv41_fast_prefill": True,
                    "ced_prefill": True,
                }
            },
            "ced_prefill",
        ),
        ({"use_v2_model_runner": False}, "v2 model runner"),
        (
            {
                "parallel_config": SimpleNamespace(
                    tensor_parallel_size=2, use_ubatching=False
                )
            },
            "TP1",
        ),
        (
            {
                "parallel_config": SimpleNamespace(
                    tensor_parallel_size=1, use_ubatching=True
                )
            },
            "TP1",
        ),
    ],
)
def test_unsupported_scope_fails_loudly(overrides, match):
    with pytest.raises(ValueError, match=match):
        fast_prefill_enabled(_vllm_config(**overrides))


def test_a_rank_below_the_boundary_has_nothing_to_bound():
    # PP rank 0 holds layers 0-7 only. Its bounded layer set is legitimately
    # empty, so it must report "nothing to bound" rather than raise: raising
    # here hung the engine core on the first request of the switch-on run.
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(sliding_window=128))
    )
    metadata = FastPrefillMetadata(config, "cpu", 21)
    below = [
        [SimpleNamespace(layer_names=["model.layers.0.attn", "model.layers.7.attn"])]
    ]
    above = [[SimpleNamespace(layer_names=["model.layers.24.attn"])]]
    mixed = [
        [SimpleNamespace(layer_names=["model.layers.20.attn", "model.layers.21.attn"])]
    ]
    assert not metadata._has_bounded_layers(below)
    assert metadata._has_bounded_layers(above)
    assert metadata._has_bounded_layers(mixed)


def _fake_bounded_model(rows=8):
    """A model whose bound buffers are the same objects the layers hold."""

    class Holder(SimpleNamespace):
        pass

    topk = torch.arange(rows * 2, dtype=torch.int32).reshape(rows, 2)
    candidates = torch.arange(rows * 3, dtype=torch.int32).reshape(rows, 3)
    # The model's layer list is indexed by global layer index, with placeholders
    # for the ranks' own layers; the code relies on that.
    layers: list[object] = [None] * 26
    for index in (24, 25):
        layer = Holder()
        attn = Holder()
        attn.topk_indices_buffer = topk
        indexer = Holder()
        indexer.topk_indices_buffer = topk
        op = Holder()
        op.topk_indices_buffer = topk
        op.candidate_blocks = candidates
        indexer.indexer_op = op
        attn.indexer = indexer
        layer.attn = attn
        layers[index] = layer
    below = Holder()
    below_attn = Holder()
    below_attn.topk_indices_buffer = topk
    below_attn.indexer = None
    below.attn = below_attn
    layers[20] = below  # a layer below the tail, which must not be touched
    model = Holder()
    model.layers = layers
    model.topk_indices_buffer = topk
    model.candidate_block_buffer = candidates
    return model, topk, candidates, layers


def test_binding_points_bounded_layers_at_gathered_rows_and_restores():
    # Regression: the tail reads top-k and candidate rows by prefill-region row
    # index, so they must be gathered with the same rows as hidden and pre_mix.
    # Without it the bounded layers consume another row's indices, which faulted
    # the indexer on PP3 the first time the switch ran on hardware.
    model, topk, candidates, layers = _fake_bounded_model()
    keep = torch.tensor([4, 5, 6, 7], dtype=torch.int64)
    plan = TailRows(
        keep=keep.numpy(),
        query_start_loc=np.array([0, 4], dtype=np.int32),
        gather_clamp=np.array([4], dtype=np.int32),
        truncated=np.array([True]),
    ).to_device("cpu")
    step = FastPrefillStep(
        tail_layer=21,
        plan=plan,
        metadata={},
        positions=torch.zeros(4, dtype=torch.int64),
        input_ids=torch.zeros(4, dtype=torch.int32),
        num_tokens=4,
    )

    swap = step.bind_tail_buffers(model)
    assert swap is not None
    bounded = layers[24].attn
    assert bounded.topk_indices_buffer.shape[0] == 4
    assert torch.equal(bounded.topk_indices_buffer, topk[keep])
    assert torch.equal(bounded.indexer.indexer_op.candidate_blocks, candidates[keep])
    assert torch.equal(bounded.indexer.topk_indices_buffer, topk[keep])
    # The layer below the boundary keeps reading the unbounded buffers.
    assert layers[20].attn.topk_indices_buffer is topk

    swap.restore()
    assert bounded.topk_indices_buffer is topk
    assert bounded.indexer.indexer_op.candidate_blocks is candidates


def test_the_bounded_build_keeps_every_group_key(monkeypatch):
    # Regression: the bounded metadata must still carry the keys of the caches a
    # bounded layer reads through a *source* layer below the boundary (the
    # compressed cache and the shared indexer K cache). Filtering the layer names
    # dropped them, so the tail read the unbounded entry -- 2044 query rows of
    # metadata against 128 rows of activations, measured with the shape probe as
    # `q=128 ... mpref_tok=2044` -- and faulted the indexer on PP3.
    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return {"model.layers.20.attn": object(), "model.layers.24.attn": object()}

    monkeypatch.setattr(fast_prefill, "build_attn_metadata", fake_build)
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(sliding_window=128))
    )
    metadata = fast_prefill.FastPrefillMetadata(config, "cpu", 21)

    def _group(names):
        return SimpleNamespace(
            backend=object(),
            layer_names=names,
            kv_cache_spec=object(),
            kv_cache_group_id=0,
            get_metadata_builder=lambda index=0: SimpleNamespace(kernel_block_size=16),
        )

    groups = [[_group(["model.layers.20.attn", "model.layers.24.attn"])]]
    batch = SimpleNamespace(
        seq_lens=torch.ones(2, dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.ones(2, dtype=torch.int32),
        positions=torch.zeros(8, dtype=torch.int64),
        is_prefilling_np=np.array([True, True]),
    )
    rows = TailRows(
        keep=np.arange(4, 8, dtype=np.int64),
        query_start_loc=np.array([0, 2, 4], dtype=np.int32),
        gather_clamp=np.array([2, -1], dtype=np.int32),
        truncated=np.array([True, False]),
    )
    step_metadata = metadata._build(
        rows,
        rows.to_device("cpu"),
        batch,
        [],
        torch.zeros((1, 8), dtype=torch.int32),
        SimpleNamespace(),
        groups,
    )
    assert captured["attn_groups"] is groups, "groups are rebuilt unfiltered"
    assert captured["num_tokens"] == rows.num_tokens
    assert "model.layers.20.attn" in step_metadata, (
        "the source key below the boundary must survive into the bounded dict"
    )
