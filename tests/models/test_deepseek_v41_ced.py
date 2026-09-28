# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CED suffix retention and separation of global KV from bounded SWA replay."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v4_1.ced import CEDMetadata, SuffixBuffer
from vllm.models.deepseek_v4_1.pp_kv import SharedKVTransferPlan
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


@pytest.mark.parametrize("chunks", [[1024] * 32, [128, 2], [31, 31, 31, 31, 9], [17]])
@pytest.mark.parametrize("prefix_hit", [0, 512])
def test_ced_suffix_survives_chunk_boundaries_and_reused_input_buffers(
    chunks, prefix_hit
):
    suffix = SuffixBuffer(128)
    consumed = prefix_hit
    for count in chunks:
        positions = torch.arange(consumed, consumed + count)
        saved = suffix.append("first", consumed, positions=positions)
        positions.zero_()
        consumed += count
        torch.testing.assert_close(
            saved["positions"], torch.arange(max(prefix_hit, consumed - 128), consumed)
        )
        assert saved["positions"].untyped_storage().nbytes() <= 128 * 8
    saved = suffix.append("second", 0, positions=torch.arange(3))
    torch.testing.assert_close(saved["positions"], torch.arange(3))


def test_ced_suffix_rejects_missing_encoder_history():
    suffix = SuffixBuffer(128)
    suffix.append("request", 512, positions=torch.arange(512, 640))
    with pytest.raises(ValueError, match="contiguous"):
        suffix.append("request", 1024, positions=torch.arange(1024, 1152))
    with pytest.raises(ValueError, match="contiguous"):
        suffix.append("other", 640, positions=torch.arange(640, 768))


@pytest.mark.parametrize("prefix_hit", [0, 512, 896, 1024])
def test_ced_replay_limits_swa_without_shortening_global_context(
    monkeypatch, prefix_hit
):
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(sliding_window=128))
    )
    state = CEDMetadata(config, torch.device("cpu"))
    state.groups = []
    captured = []

    def build(**kwargs):
        captured.append(kwargs)
        return {
            "decoder.swa": DeepseekSparseSWAMetadata(
                block_table=torch.zeros(1, 8, dtype=torch.int32),
                slot_mapping=kwargs["slot_mappings"][0],
                block_size=32,
                num_prefills=1,
                num_prefill_tokens=kwargs["num_tokens"],
                prefill_seq_lens=kwargs["seq_lens"],
                prefill_gather_lens=torch.tensor([255]),
            )
        }

    monkeypatch.setattr("vllm.models.deepseek_v4_1.ced.build_attn_metadata", build)
    chunks = [(prefix_hit, 1024 - prefix_hit), (1024, 2)]
    for start, count in chunks:
        if count == 0:
            continue
        positions = torch.arange(start, start + count)
        batch = SimpleNamespace(
            has_prefill=True,
            num_reqs=1,
            is_prefilling_np=np.array([True]),
            num_computed_prefill_tokens_np=np.array([start]),
            prefill_len_np=np.array([1026]),
            num_tokens=count,
            req_ids=["request"],
            positions=positions,
            input_ids=positions + 100,
            seq_lens=torch.tensor([start + count]),
            seq_lens_cpu_upper_bound=torch.tensor([start + count]),
        )
        if prefix_hit == 1024:
            with pytest.raises(ValueError, match="full sliding-window suffix"):
                state.prepare(batch, (), positions[None, :], [], None)
            return
        step = state.prepare(batch, (), positions[None, :], [], None)
        if start == prefix_hit:
            assert not step.final and not captured
    assert len(captured) == 1
    assert captured[0]["num_tokens"] == 128
    assert captured[0]["max_seq_len"] == 1026
    torch.testing.assert_close(step.positions, torch.arange(898, 1026))
    torch.testing.assert_close(step.input_ids, torch.arange(998, 1126))
    torch.testing.assert_close(captured[0]["slot_mappings"][0], step.positions)
    metadata = step.replay_metadata["decoder.swa"]
    assert metadata.prefill_gather_lens.item() == 128
    assert metadata.prefill_seq_lens.item() == 1026


@pytest.mark.parametrize("prompt_logprobs", [None, 0, 1])
@pytest.mark.parametrize("with_image", [False, True])
def test_ced_preserves_prompt_logprobs_by_running_the_full_prompt(
    monkeypatch, prompt_logprobs, with_image
):
    from unittest.mock import Mock

    from vllm.models.deepseek_v4_1.nvidia.model_state import DeepseekV41ModelState
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    full_metadata = object()
    monkeypatch.setattr(DefaultModelState, "add_request", lambda *args: None)
    monkeypatch.setattr(DefaultModelState, "remove_request", lambda *args: None)
    monkeypatch.setattr(DefaultModelState, "prepare_attn", lambda *args: full_metadata)
    state = object.__new__(DeepseekV41ModelState)
    state.requires_eager_prefill = True
    state.ced_disabled_requests = set()
    state.ced_metadata = Mock()
    request = SimpleNamespace(
        req_id="request",
        sampling_params=SimpleNamespace(prompt_logprobs=prompt_logprobs),
        mm_features=[object()] if with_image else [],
    )
    state.add_request(0, request)
    batch = SimpleNamespace(req_ids=[request.req_id])
    assert state.prepare_attn(batch, None, None, None, None, None) is full_metadata
    if prompt_logprobs is None:
        assert state.ced_step is state.ced_metadata.prepare.return_value
    else:
        state.ced_metadata.prepare.assert_not_called()
        assert state.ced_step is None
    state.remove_request(request.req_id)
    request.sampling_params.prompt_logprobs = None
    state.add_request(0, request)
    state.prepare_attn(batch, None, None, None, None, None)
    assert state.ced_step is state.ced_metadata.prepare.return_value


def test_dspark_ced_restores_the_entire_suffix_before_a_short_final_chunk(monkeypatch):
    from vllm.models.deepseek_v4_1.ced import CEDDraftContext, CEDPrefill
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    ced = CEDPrefill(128)
    positions = torch.arange(898, 1026)
    auxiliary = [torch.full((128, 4), float(layer)) for layer in (37, 38, 39)]
    slots = torch.stack([positions + group * 2048 for group in range(6)])
    ced.draft_context = CEDDraftContext("request", positions, slots, auxiliary)
    events = []

    def insert(states, saved_positions, layer_slots):
        torch.testing.assert_close(states, torch.cat(auxiliary, dim=-1))
        torch.testing.assert_close(saved_positions, positions)
        for actual, expected_group in zip(layer_slots, (5, 2, 5)):
            torch.testing.assert_close(actual, slots[expected_group])
        events.append("suffix")

    def ordinary(self, batch, *args, **kwargs):
        events.append("chunk")
        assert batch.num_tokens == 2
        return "draft"

    monkeypatch.setattr(DFlashSpeculator, "propose", ordinary)
    speculator = object.__new__(DSparkSpeculator)
    speculator.distributed = False
    speculator._ced_context_source = ced.take_draft_context
    speculator.draft_kv_cache_group_ids = [2, 5]
    speculator._layer_group_idx = [1, 0, 1]
    speculator.model = SimpleNamespace(
        combine_hidden_states=lambda value: value,
        precompute_and_store_context_kv=insert,
    )
    batch = SimpleNamespace(req_ids=["request"], num_tokens=2)
    placeholders = (None,) * 10

    assert speculator.propose(batch, *placeholders, dummy_run=True) == "draft"
    assert ced.draft_context is not None
    events.clear()
    assert speculator.propose(batch, *placeholders) == "draft"
    assert events == ["suffix", "chunk"]
    assert ced.draft_context is None
    events.clear()
    speculator.propose(batch, *placeholders)
    assert events == ["chunk"]


def test_v41_auxiliary_pp_support_requires_all_sources_on_the_last_stage(monkeypatch):
    from vllm.models.deepseek_v4_1.nvidia import model as model_module

    model = model_module.DeepseekV4Model.__new__(model_module.DeepseekV4Model)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(num_hidden_layers=40)
    model.start_layer = 14
    pp = SimpleNamespace(world_size=3, is_first_rank=False, is_last_rank=True)
    monkeypatch.setattr(model_module, "get_pp_group", lambda: pp)
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "8,6,26")
    model._set_aux_hidden_state_layers((37, 38, 39))
    assert model.supports_aux_hidden_states_over_pp
    assert model.aux_hidden_state_layers == (37, 38, 39)
    with pytest.raises(ValueError, match="last stage"):
        model._set_aux_hidden_state_layers((8, 37, 39))


@pytest.mark.parametrize("has_engram", [False, True])
@pytest.mark.parametrize("has_hashes", [False, True])
def test_v41_engram_injection_is_independent_of_pp_boundary(
    monkeypatch, has_engram, has_hashes
):
    """A PP boundary must preserve injection before the next attention pre-mix."""
    from vllm.models.deepseek_v4_1.nvidia import model as model_module

    inputs = []

    def pre(stream, *args, **kwargs):
        inputs.append(stream.clone())
        return None, None, stream.mean(dim=1), kwargs.get("pre_mix")

    def post(x, residual, *args):
        return residual + x.unsqueeze(1)

    class Engram:
        layer_hash_index = 1

        def __call__(self, stream, hashes, mask):
            return stream + hashes.unsqueeze(-1) * mask[:, None, None]

    engram = Engram()
    monkeypatch.setattr(model_module, "mhc_pre_delayed_tilelang", pre)
    monkeypatch.setattr(model_module, "mhc_post_tilelang", post)
    layer = SimpleNamespace(
        use_sequence_parallel=False,
        engram=engram if has_engram else None,
        hc_attn_fn=None,
        hc_attn_scale=None,
        hc_attn_base=None,
        hc_ffn_fn=None,
        hc_ffn_scale=None,
        hc_ffn_base=None,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_post_alpha=2.0,
        hc_sinkhorn_iters=20,
        attn_norm=SimpleNamespace(weight=None, variance_epsilon=1e-6),
        ffn_norm=SimpleNamespace(weight=None, variance_epsilon=1e-6),
        attn=lambda positions, x, unused: x,
        ffn=lambda x, ids: x,
    )
    residual = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    x = torch.full((2, 4), 0.25)
    stream = post(x, residual)
    hashes = torch.tensor([[[90.0], [3.0]], [[80.0], [7.0]]])
    mask = torch.tensor([True, False])
    kwargs = dict(
        positions=torch.arange(2),
        input_ids=None,
        pre_mix=torch.tensor([1.0]),
        engram_hashes=hashes if has_hashes else None,
        engram_mask=mask,
    )
    forward = model_module.DeepseekV4DecoderLayer._forward
    ordinary = forward(layer, x, residual=residual, **kwargs)
    boundary = forward(layer, stream, **kwargs)
    expected = (
        engram(stream, hashes[:, 1], mask) if has_engram and has_hashes else stream
    )
    torch.testing.assert_close(inputs[0], expected, atol=0, rtol=0)
    torch.testing.assert_close(inputs[2], expected, atol=0, rtol=0)
    for a, b in zip(ordinary, boundary):
        if a is not None:
            torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["auto", "portable", "sm120_decode"])
def test_heterogeneous_pp_selects_attention_for_the_worker_device(monkeypatch, mode):
    from vllm.models.deepseek_v4_1.nvidia import model
    from vllm.platforms.interface import DeviceCapability

    devices = []

    def capability(device=0):
        devices.append(device)
        return DeviceCapability(12 if device == 3 else 8, 0)

    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 3)
    monkeypatch.setattr(model.current_platform, "get_device_capability", capability)
    config = SimpleNamespace(
        attention_config=SimpleNamespace(backend=None),
        additional_config={"dsv41_attention": mode},
    )
    selected = model._select_dsv4_attn_cls(config)
    assert devices == [3]
    assert selected.__name__ == (
        "DeepseekV41SM120DecodeAttention"
        if mode == "sm120_decode"
        else "DeepseekV41AmpereMLAAttention"
    )


def test_sm120_decode_padding_keeps_valid_lengths_and_shared_metadata(monkeypatch):
    from vllm.models.deepseek_v4_1.ampere.ampere_sparse import (
        DeepseekV41SM120DecodeAttention,
    )
    from vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse import (
        DeepseekV4FlashInferSM120Attention,
    )

    attention = object.__new__(DeepseekV41SM120DecodeAttention)
    attention._decode_widths = (128, 512)
    indices = torch.arange(4 * 192).reshape(4, 1, 192).int()
    lengths = torch.tensor([128, 129, 130, 131], dtype=torch.int32)
    metadata = DeepseekSparseSWAMetadata(
        block_table=torch.zeros(1, 8, dtype=torch.int32),
        slot_mapping=torch.arange(4),
        block_size=64,
        num_decodes=1,
        num_decode_tokens=4,
        decode_swa_indices=indices,
        decode_swa_lens=lengths,
    )
    seen = []

    def native(self, q, kv_cache, swa, attn, swa_only, output):
        seen.append(swa)
        assert swa.decode_swa_lens is lengths
        torch.testing.assert_close(swa.decode_swa_indices[..., :192], indices)
        assert (swa.decode_swa_indices[..., 192:] == -1).all()

    monkeypatch.setattr(DeepseekV4FlashInferSM120Attention, "_forward_decode", native)
    attention._forward_decode(None, None, metadata, None, False, None)
    assert len(seen) == 1 and seen[0] is not metadata
    assert metadata.decode_swa_indices is indices


@pytest.mark.parametrize("hybrid", [False, True])
def test_ced_publishes_encoder_kv_with_a_compatible_prefill_subclass(
    monkeypatch, hybrid
):
    from vllm.models.deepseek_v4_1 import ced
    from vllm.models.deepseek_v4_1.ampere.ampere_sparse import (
        DeepseekV41AmpereMLAAttention,
        DeepseekV41SM120DecodeAttention,
    )

    attention_cls = (
        DeepseekV41SM120DecodeAttention if hybrid else DeepseekV41AmpereMLAAttention
    )
    attention = object.__new__(attention_cls)
    layer = SimpleNamespace(
        attn=attention,
        hc_attn_fn=None,
        hc_attn_scale=None,
        hc_attn_base=None,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_post_alpha=1.0,
        hc_sinkhorn_iters=1,
        attn_norm=SimpleNamespace(weight=torch.ones(2), variance_epsilon=1e-6),
    )
    model = SimpleNamespace(
        layers=[None] * 20 + [layer],
        config=SimpleNamespace(hidden_size=2),
        aux_hidden_state_layers=[],
        _mtp_hidden_buffer=None,
    )
    hidden = torch.zeros(3, 4, 2)
    published = []
    monkeypatch.setattr(ced, "mhc_post_tilelang", lambda *args: hidden)
    monkeypatch.setattr(
        ced,
        "mhc_pre_delayed_tilelang",
        lambda *args, **kwargs: (None, None, hidden.mean(1), None),
    )
    monkeypatch.setattr(ced, "get_forward_context", lambda: SimpleNamespace())
    monkeypatch.setattr(
        ced,
        "publish_decoder_kv",
        lambda attn, normalized, positions: published.append((attn, positions)),
    )
    replay = ced.CEDPrefill(128)
    step = ced.CEDStep(
        "request",
        0,
        3,
        False,
        requests=[ced.CEDRequest("request", 0, 0, 3, True, True, False)],
        source_positions=torch.arange(3),
    )
    output = replay.forward(model, step, hidden, None, None, None, torch.zeros(3, 4))
    assert output.shape == (3, 2) and not output.any()
    assert replay.encoder_tokens == 3 and replay.decoder_tokens == 0
    assert len(published) == 1 and published[0][0] is attention
    torch.testing.assert_close(published[0][1], torch.arange(3))


def _pp_kv_topology():
    return SimpleNamespace(
        num_hidden_layers=40,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20,
    )


@pytest.mark.parametrize(
    "start,end,incoming,outgoing",
    [
        (0, 4, None, 2),
        (4, 9, 2, 8),
        (9, 14, 8, None),
        (14, 40, None, None),
        (0, 2, None, None),
        (2, 8, None, None),
    ],
)
def test_transfer_plan_crosses_only_required_groups(start, end, incoming, outgoing):
    assert SharedKVTransferPlan.for_stage(
        _pp_kv_topology(), start, end
    ) == SharedKVTransferPlan(incoming, outgoing)


def test_transfer_plan_rejects_decoder_indexer_dependencies():
    with pytest.raises(ValueError, match="encoder group"):
        SharedKVTransferPlan.for_stage(_pp_kv_topology(), 20, 26)


@pytest.mark.parametrize("owns_vision", [True, False])
def test_checkpoint_stream_releases_tensors_before_finalizing_experts(owns_vision):
    """Interleaved text/vision shards must not retain the full checkpoint."""
    import weakref

    from vllm.model_executor.models.utils import WeightsMapper
    from vllm.models.deepseek_v4_1.nvidia.model import DeepseekV41LLMForCausalLM
    from vllm.models.deepseek_v4_1.nvidia.vl_model import DeepseekV41ForCausalLM

    finalized = []

    class LanguageModel(torch.nn.Module):
        load_weights = DeepseekV41LLMForCausalLM.load_weights

        def __init__(self):
            super().__init__()
            self.hf_to_vllm_mapper = WeightsMapper()
            self.blocks = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.zeros(1)) for _ in range(8)]
            )

        def process_weights_after_loading(self):
            assert [p.item() for p in self.blocks] == list(range(1, 9))
            finalized.append(True)

    model = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
    torch.nn.Module.__init__(model)
    model.language_model = LanguageModel()
    model.owns_vision = owns_vision
    model.vision = torch.nn.ParameterList(
        [torch.nn.Parameter(torch.zeros(1)) for _ in range(8)]
    )
    model.hf_to_vllm_mapper = WeightsMapper()
    live: list[weakref.ReferenceType[torch.Tensor]] = []
    expected = set()

    def weights():
        for i in range(8):
            for prefix in ("language_model.blocks", "vision"):
                assert not finalized
                assert sum(ref() is not None for ref in live) <= 2
                name = f"{prefix}.{i}"
                if owns_vision or prefix.startswith("language_model"):
                    expected.add(name)
                tensor = torch.tensor([float(i + 1)])
                live.append(weakref.ref(tensor))
                yield name, tensor
        assert not finalized

    assert model.load_weights(weights()) == expected
    expected_vision = list(range(1, 9)) if owns_vision else [0] * 8
    assert [p.item() for p in model.vision] == expected_vision
    assert finalized == [True]
    assert all(ref() is None for ref in live)
    model.process_weights_after_loading()
    assert finalized == [True]


def test_ced_interleaved_requests_keep_separate_suffixes_and_decode_rows(monkeypatch):
    """A mixed batch must select each request's own history and block table."""
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(sliding_window=128))
    )
    state = CEDMetadata(config, torch.device("cpu"))
    state.groups = []
    captured = []

    def build(**kwargs):
        captured.append(kwargs)
        return {}

    monkeypatch.setattr("vllm.models.deepseek_v4_1.ced.build_attn_metadata", build)

    def prepare(rows, disabled=frozenset()):
        positions = torch.cat(
            [torch.arange(start, start + n) for _, start, n, _, _ in rows]
        )
        counts = [row[2] for row in rows]
        batch = SimpleNamespace(
            has_prefill=True,
            num_reqs=len(rows),
            req_ids=[row[0] for row in rows],
            is_prefilling_np=np.array([row[4] for row in rows]),
            num_computed_prefill_tokens_np=np.array([row[1] for row in rows]),
            prefill_len_np=np.array([row[3] for row in rows]),
            query_start_loc_np=np.cumsum([0, *counts]),
            num_tokens=sum(counts),
            positions=positions,
            input_ids=positions + 100,
            seq_lens=torch.tensor([row[1] + row[2] for row in rows]),
            seq_lens_cpu_upper_bound=torch.tensor([row[1] + row[2] for row in rows]),
        )
        tables = [torch.arange(len(rows))[:, None] + 1000]
        return state.prepare(batch, tables, positions[None], [], None, disabled)

    first = prepare([("a", 0, 128, 130, True), ("b", 0, 130, 136, True)])
    assert not first.final
    mixed = prepare(
        [
            ("decode", 40, 4, 30, False),
            ("b", 130, 3, 136, True),
            ("a", 128, 2, 130, True),
        ]
    )
    assert [r.replay_count for r in mixed.requests] == [4, 0, 128]
    torch.testing.assert_close(
        mixed.positions, torch.cat([torch.arange(40, 44), torch.arange(2, 130)])
    )
    assert captured[-1]["query_start_loc_cpu"].tolist() == [0, 4, 132]
    assert captured[-1]["block_tables"][0].flatten().tolist() == [1000, 1002]
    assert captured[-1]["is_prefilling"].tolist() == [False, True]
    assert set(state.suffixes) == {"b"}
    final = prepare([("b", 133, 3, 136, True)])
    torch.testing.assert_close(final.positions, torch.arange(8, 136))
    assert not state.suffixes
    prepare([("cancelled", 0, 32, 256, True)])
    state.remove_request("cancelled")
    assert not state.suffixes
    mixed_logprobs = prepare(
        [("ced", 0, 96, 128, True), ("logprobs", 0, 4, 4, True)],
        disabled={"logprobs"},
    )
    assert [r.replay_count for r in mixed_logprobs.requests] == [0, 4]
    assert mixed_logprobs.positions.tolist() == [0, 1, 2, 3]
    assert set(state.suffixes) == {"ced"}
    state.remove_request("ced")


@pytest.mark.parametrize("final_chunk", [1, 2, 128])
@pytest.mark.parametrize("prefix_hit", [0, 896])
def test_ced_pp5_replay_matches_unsplit_decoder_and_keeps_full_kv(
    monkeypatch, final_chunk, prefix_hit
):
    """PP boundaries must carry the entire mHC suffix past short final chunks."""
    from vllm.models.deepseek_v4_1 import ced
    from vllm.models.deepseek_v4_1.ampere.ampere_sparse import (
        DeepseekV41AmpereMLAAttention,
    )

    context = SimpleNamespace(attn_metadata={}, slot_mapping={}, is_padding=None)
    monkeypatch.setattr(ced, "get_forward_context", lambda: context)
    monkeypatch.setattr(
        "vllm.models.deepseek_v4_1.fast_prefill.record_layer_rows", lambda *args: None
    )
    monkeypatch.setattr(
        "vllm.distributed.get_pp_group", lambda: SimpleNamespace(rank_in_group=0)
    )

    def post(hidden, residual, *unused):
        assert hidden.ndim == 2 and residual.ndim == 3
        return residual + hidden[:, None, :]

    monkeypatch.setattr(ced, "mhc_post_tilelang", post)
    monkeypatch.setattr(
        ced,
        "mhc_pre_delayed_tilelang",
        lambda stream, *args, **kwargs: (None, None, stream.mean(1), None),
    )

    def publish(attn, normalized, positions):
        attn._last_compressed_latent = positions[:, None].float().clone()
        attn.indexer._last_normalized_k = positions[:, None].float() + 100
        return attn._last_compressed_latent

    monkeypatch.setattr(ced, "publish_decoder_kv", publish)

    class Layer:
        def __init__(self, index):
            self.index = index
            self.attn = object.__new__(DeepseekV41AmpereMLAAttention)
            self.attn.indexer = SimpleNamespace()
            for name in (
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_attn_base",
                "rms_norm_eps",
                "hc_eps",
                "hc_post_alpha",
                "hc_sinkhorn_iters",
            ):
                setattr(self, name, None)
            self.attn_norm = SimpleNamespace(weight=None, variance_epsilon=1e-6)

        def __call__(self, hidden, positions, ids, pre_mix, pm, rm, residual):
            stream = post(hidden, residual) if residual is not None else hidden
            assert stream.shape == (positions.numel(), 4, 2)
            if self.index == 20:
                publish(self.attn, stream.mean(1), positions)
            # Nonconstant residual and mix make a missing mHC boundary observable.
            return stream.mean(1) * 0.01 + self.index, stream, None, None, pre_mix

    def model(start, end, aux_layers=()):
        value = SimpleNamespace(
            start_layer=start,
            end_layer=end,
            config=SimpleNamespace(hidden_size=2, num_hidden_layers=40),
            layers=[Layer(i) for i in range(40)],
            aux_hidden_state_layers=list(aux_layers),
            _mtp_hidden_buffer=None,
            _pp_incoming_topk=False,
            _pp_incoming_candidates=start > 20,
            _pp_outgoing_topk=False,
            _pp_outgoing_candidates=end < 40,
            _pp_outgoing_index=end < 40,
            candidate_block_buffer=torch.zeros(1024, 2),
            norm=lambda x: x,
        )

        def append(tensors, positions, incoming):
            if start <= 20:
                latent = value.layers[20].attn._last_compressed_latent
                index_k = value.layers[20].attn.indexer._last_normalized_k
            else:
                latent, index_k = incoming["full_kv"], incoming["full_index_k"]
            torch.testing.assert_close(latent[:, 0], positions.float())
            torch.testing.assert_close(index_k[:, 0], positions.float() + 100)
            tensors["full_kv"], tensors["full_index_k"] = latent, index_k

        value._append_pp_attention_state = append
        return value

    monkeypatch.setattr(ced, "hc_collapse_triton", lambda x, mix: x.mean(1))
    models = [model(16, 24), model(24, 32), model(32, 40, (37, 38, 39))]
    states = [ced.CEDPrefill(128) for _ in models]
    unsplit, reference = model(16, 40, (37, 38, 39)), ced.CEDPrefill(128)
    original_metadata = context.attn_metadata
    for start, count in [(prefix_hit, 1024 - prefix_hit), (1024, final_chunk)]:
        final = start == 1024
        positions = torch.arange(start, start + count)
        replay_positions = torch.arange(start + count - 128, start + count)
        replay_slots = torch.stack(
            [replay_positions + group * 2048 for group in range(6)]
        )
        stream = positions.float()[:, None, None].expand(-1, 4, 2).clone()
        mix = torch.ones(count, 4)
        step = ced.CEDStep(
            "request",
            start,
            count,
            final,
            replay_metadata={},
            positions=replay_positions,
            input_ids=replay_positions,
            replay_slots=replay_slots,
            requests=[
                ced.CEDRequest(
                    "request", start, 0, count, True, True, final, 128 if final else 0
                )
            ],
            source_positions=positions,
        )
        expected = reference.forward(unsplit, step, stream, None, None, None, mix)
        incoming = None
        for stage, state in zip(models, states):
            result = state.forward(
                stage,
                step,
                stream if incoming is None else incoming["hidden_states"],
                None,
                None,
                None,
                mix if incoming is None else incoming["pre_mix"],
                intermediate_tensors=incoming,
            )
            if stage.end_layer < 40:
                assert result["hidden_states"].shape == (count, 4, 2)
                assert result["pre_mix"].shape == (count, 4)
                if final:
                    assert result[ced.CED_REPLAY_HIDDEN].shape == (128, 4, 2)
                    assert result[ced.CED_REPLAY_CANDIDATES].shape == (128, 2)
                else:
                    assert ced.CED_REPLAY_HIDDEN not in result.tensors
                incoming = result
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        if final:
            assert reference.draft_context is not None
            assert states[-1].draft_context is not None
            actual_context = states[-1].draft_context
            expected_context = reference.draft_context
            assert actual_context.request_ids == ("request",)
            assert actual_context.request_lengths == (128,)
            assert expected_context.request_ids == actual_context.request_ids
            assert expected_context.request_lengths == actual_context.request_lengths
            torch.testing.assert_close(
                actual_context.positions, expected_context.positions
            )
            torch.testing.assert_close(actual_context.positions, replay_positions)
            torch.testing.assert_close(actual_context.slots, expected_context.slots)
            torch.testing.assert_close(actual_context.slots, replay_slots)
            assert len(actual_context.auxiliary) == 3
            for actual, expected in zip(
                actual_context.auxiliary, expected_context.auxiliary
            ):
                torch.testing.assert_close(actual, expected)
        assert context.attn_metadata is original_metadata
    assert [s.decoder_tokens for s in states] == [128, 128, 128]
    assert states[0].encoder_tokens == 1024 + final_chunk - prefix_hit
    assert all(not state.suffixes for state in states)

    if final_chunk != 2:
        return

    # Only a final CED prefill request may seed the distributed draft context.
    # Decode rows and requests that use the full fallback path still share the
    # replay forward, but must not contribute their rows to the CED payload.
    mixed_model = model(32, 40, (37, 38, 39))
    mixed_state = ced.CEDPrefill(128)
    replay_count = 128 + 2 + 2
    mixed_positions = torch.arange(replay_count)
    mixed_slots = torch.stack([mixed_positions + group * 4096 for group in range(6)])
    mixed_step = ced.CEDStep(
        "ced",
        1024,
        5,
        True,
        replay_metadata={},
        positions=mixed_positions,
        input_ids=mixed_positions + 100,
        replay_slots=mixed_slots,
        requests=[
            ced.CEDRequest("ced", 1024, 0, 1, True, True, True, 128),
            ced.CEDRequest("decode", 0, 1, 2, False, False, True, 2),
            ced.CEDRequest("fallback", 0, 3, 2, True, False, True, 2),
        ],
        source_positions=mixed_positions,
    )
    mixed_hidden = torch.zeros(replay_count, 4, 2)
    mixed_pre_mix = torch.zeros(replay_count, 4)
    mixed_state.forward(
        mixed_model,
        mixed_step,
        torch.zeros(5, 4, 2),
        None,
        None,
        None,
        torch.zeros(5, 4),
        intermediate_tensors={
            ced.CED_REPLAY_HIDDEN: mixed_hidden,
            ced.CED_REPLAY_PRE_MIX: mixed_pre_mix,
            ced.CED_REPLAY_CANDIDATES: torch.zeros(replay_count, 2),
        },
    )
    assert mixed_state.draft_context is not None
    mixed_context = mixed_state.draft_context
    assert mixed_context.request_ids == ("ced",)
    assert mixed_context.request_lengths == (128,)
    torch.testing.assert_close(mixed_context.positions, mixed_positions[:128])
    torch.testing.assert_close(mixed_context.slots, mixed_slots[:, :128])
    assert all(value.shape[0] == 128 for value in mixed_context.auxiliary)


def test_ced_replay_metadata_keeps_source_keys_in_independent_builders(monkeypatch):
    """Compact queries must not reuse or overwrite full-chunk source metadata."""
    from vllm.models.deepseek_v4_1 import ced

    made = []

    class Group:
        def __init__(self, backend, names, spec, group_id):
            self.backend, self.layer_names = backend, names
            self.kv_cache_spec, self.kv_cache_group_id = spec, group_id
            self.builder = SimpleNamespace(kernel_block_size=32)
            made.append(self)

        def get_metadata_builder(self):
            return self.builder

        def create_metadata_builders(self, *args, **kwargs):
            assert kwargs["kernel_block_size"] == 32

    group = Group(None, ["model.layers.14.cache", "model.layers.20.swa"], None, 0)
    monkeypatch.setattr(ced, "AttentionGroup", Group)
    captured = []

    def capture_metadata(**kwargs):
        captured.append(kwargs)
        return {}

    monkeypatch.setattr(ced, "build_attn_metadata", capture_metadata)
    state = ced.CEDMetadata(
        SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(sliding_window=128))
        ),
        torch.device("cpu"),
    )
    positions = torch.arange(256)
    batch = SimpleNamespace(
        has_prefill=True,
        num_tokens=256,
        req_ids=["request"],
        num_computed_prefill_tokens_np=[0],
        is_prefilling_np=[True],
        prefill_len_np=[256],
        positions=positions,
        input_ids=positions,
        seq_lens=torch.tensor([256]),
        seq_lens_cpu_upper_bound=torch.tensor([256]),
    )
    state.prepare(batch, (), positions[None], [[group]], None)
    replay_group = captured[0]["attn_groups"][0][0]
    assert replay_group.layer_names == group.layer_names
    assert replay_group.builder is not group.builder
    assert captured[0]["num_tokens"] == 128
    assert positions.numel() == 256
