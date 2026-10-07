# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import vllm.model_executor.layers.vocab_parallel_embedding as embedding_module
import vllm.model_executor.parameter as parameter_module
from tests.utils import large_gpu_test
from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.config import SpeculativeConfig, VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.model_loader.utils import device_loading_context
from vllm.models.qwen4_exp.common.ple import (
    PLEShardOverlap,
    compute_ple_shard_overlap,
    copy_ple_embedding_shard_,
)
from vllm.models.qwen4_exp.nvidia import ple_layer as ple_layer_module
from vllm.models.qwen4_exp.nvidia.ops.ple import (
    fused_ple_ngram_hash,
    fused_ple_prefill_short_conv,
    fused_ple_spec_short_conv,
    gather_pinned_ple_embedding,
)
from vllm.models.qwen4_exp.nvidia.ple_layer import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLEFp8EmbeddingMethod,
    Qwen4ExpPLELayer,
    _get_ple_embedding_quant_method,
)
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionMetadataBuilder,
    ShortConvAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import MambaSpec


@pytest.mark.parametrize(
    "builder_cls",
    [ShortConvAttentionMetadataBuilder, PleShortConvAttentionMetadataBuilder],
)
def test_only_ple_skips_unused_causal_convolution_metadata(builder_cls):
    """Other short-conv consumers still receive their prefill kernel metadata."""
    device = torch.device("cpu")
    builder = builder_cls(
        MambaSpec(block_size=16, shapes=((16, 64),), dtypes=(torch.float32,)),
        ["short_conv"],
        VllmConfig(),
        device,
    )
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[21], query_lens=[5]), 16, device
    )
    common.is_prefilling = torch.tensor([True])
    metadata = builder.build(0, common)
    assert metadata.num_prefills == 1
    assert metadata.query_start_loc_p.tolist() == [0, 5]
    if builder_cls is ShortConvAttentionMetadataBuilder:
        assert metadata.nums_dict[8]["tot"] == 1
        assert metadata.batch_ptr is not None
        assert metadata.token_chunk_offset_ptr is not None
    else:
        assert metadata.nums_dict is None
        assert metadata.batch_ptr is None
        assert metadata.token_chunk_offset_ptr is None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("full_graph", [False, True])
@pytest.mark.parametrize(
    "query_lens,drafts,spec_rows,non_spec_rows,token_order",
    [
        pytest.param([4, 2, 0], [3, 1, -1], [0, 1], [], None, id="spec-padding"),
        pytest.param(
            [3, 1, 2, 5, 0],
            [2, -1, 1, -1, -1],
            [0, 2],
            [1, 3],
            [0, 1, 2, 4, 5, 3, 6, 7, 8, 9, 10],
            id="mixed-padding",
        ),
        pytest.param([1, 5], [-1, -1], [], [0, 1], None, id="non-spec"),
    ],
)
def test_ple_metadata_preserves_request_and_token_order(
    device, full_graph, query_lens, drafts, spec_rows, non_spec_rows, token_order
):
    """Preserve PLE routing and offsets without unused convolution metadata."""
    if device == "cuda" and not current_platform.is_cuda():
        pytest.skip("CUDA required")
    device = torch.device(device)
    config = VllmConfig()
    config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=3
    )
    config.compilation_config.cudagraph_mode = (
        CUDAGraphMode.FULL if full_graph else CUDAGraphMode.NONE
    )
    config.compilation_config.max_cudagraph_capture_size = 32
    builder = PleShortConvAttentionMetadataBuilder(
        MambaSpec(block_size=16, shapes=((16, 64),), dtypes=(torch.float32,)),
        ["ple"],
        config,
        device,
    )
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[q + 16 for q in query_lens], query_lens=query_lens),
        16,
        device,
        arange_block_indices=True,
    )
    common.is_prefilling = torch.tensor(
        [q > 1 and d < 0 for q, d in zip(query_lens, drafts)]
    )
    # Token padding is independent of the sum of per-request query lengths.
    common.num_actual_tokens += 3
    accepted = torch.arange(1, len(query_lens) + 1, dtype=torch.int32, device=device)
    drafts_cpu = torch.tensor(drafts, dtype=torch.int32)
    metadata = builder.build(
        0,
        common,
        num_accepted_tokens=accepted,
        num_decode_draft_tokens_cpu=drafts_cpu,
    )

    def check(actual, expected):
        torch.testing.assert_close(
            actual, torch.tensor(expected, dtype=actual.dtype, device=device)
        )

    assert metadata.num_spec_decodes == len(spec_rows)
    assert metadata.num_decodes == sum(query_lens[i] == 1 for i in non_spec_rows)
    assert metadata.num_prefills == sum(query_lens[i] > 1 for i in non_spec_rows)
    if spec_rows:
        torch.testing.assert_close(
            metadata.spec_state_indices_tensor[: len(spec_rows)],
            common.block_table_tensor[spec_rows, 0],
        )
        torch.testing.assert_close(
            metadata.num_accepted_tokens[: len(spec_rows)], accepted[spec_rows]
        )
        check(
            metadata.spec_query_start_loc[: len(spec_rows) + 1],
            [0, *accumulate(query_lens[i] for i in spec_rows)],
        )
        if full_graph and not non_spec_rows:
            check(metadata.spec_state_indices_tensor[len(spec_rows) :], [NULL_BLOCK_ID])
            check(metadata.num_accepted_tokens[len(spec_rows) :], [1])
            check(
                metadata.spec_query_start_loc[len(spec_rows) + 1 :], [sum(query_lens)]
            )
            assert (
                metadata.spec_state_indices_tensor.data_ptr()
                == builder.spec_state_indices_tensor.data_ptr()
            )
    if non_spec_rows:
        torch.testing.assert_close(
            metadata.state_indices_tensor, common.block_table_tensor[non_spec_rows, 0]
        )
        prefill_lens = [query_lens[i] for i in non_spec_rows if query_lens[i] > 1]
        check(metadata.query_start_loc_p, [0, *accumulate(prefill_lens)])
        check(metadata.has_initial_states_p, [True] * len(prefill_lens))
    if token_order is not None:
        check(
            torch.cat((metadata.spec_token_indx, metadata.non_spec_token_indx)),
            token_order,
        )
    else:
        assert metadata.spec_token_indx is None
        assert metadata.non_spec_token_indx is None
    assert metadata.nums_dict is None
    assert metadata.batch_ptr is None
    assert metadata.token_chunk_offset_ptr is None


def _make_ngram_embedding_for_load_test() -> Qwen4ExpNGramEmbedding:
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(module)
    module.split_ngram_parts = 2
    module.register_buffer("layer_multipliers", torch.zeros(1, dtype=torch.long))
    module.register_buffer("ngram_heads_offsets", torch.zeros(1, dtype=torch.long))
    module.register_buffer("ngram_heads_vocab_sizes", torch.zeros(1, dtype=torch.long))
    module.ngram_embedding = SimpleNamespace(
        org_vocab_size=8,
        embedding_dim=2,
        weight=nn.Parameter(torch.full((4, 2), -1.0)),
        shard_indices=SimpleNamespace(
            org_vocab_start_index=2,
            org_vocab_end_index=6,
        ),
    )
    _set_test_embedding_weight_loader(module.ngram_embedding)
    return module


def _set_test_embedding_weight_loader(embedding) -> None:
    embedding.weight.weight_loader = partial(
        copy_ple_embedding_shard_,
        tp_start=embedding.shard_indices.org_vocab_start_index,
        tp_end=embedding.shard_indices.org_vocab_end_index,
    )


def _make_fp8_ngram_embedding_for_load_test() -> Qwen4ExpNGramEmbedding:
    module = _make_ngram_embedding_for_load_test()
    embedding = nn.Module()
    embedding.org_vocab_size = 8
    embedding.embedding_dim = 2
    embedding.shard_indices = SimpleNamespace(
        org_vocab_start_index=2,
        org_vocab_end_index=6,
    )
    embedding.register_parameter(
        "weight",
        nn.Parameter(
            torch.full((4, 2), -1.0).to(torch.float8_e4m3fn),
            requires_grad=False,
        ),
    )
    embedding.register_parameter(
        "weight_scale",
        nn.Parameter(torch.zeros(1, dtype=torch.bfloat16), requires_grad=False),
    )
    _set_test_embedding_weight_loader(embedding)
    module.ngram_embedding = embedding
    return module


def test_ple_shard_overlap_and_copy() -> None:
    overlap = compute_ple_shard_overlap(
        checkpoint_start=2, checkpoint_rows=5, tp_start=4, tp_end=8
    )
    assert overlap == PLEShardOverlap(source_start=2, destination_start=0, row_count=3)

    destination = torch.full((4, 2), -1.0)
    loaded = torch.arange(10, dtype=torch.float64).reshape(5, 2)
    copied = copy_ple_embedding_shard_(
        destination,
        loaded,
        checkpoint_start=2,
        tp_start=4,
        tp_end=8,
    )

    assert copied == 3
    torch.testing.assert_close(destination[:3], loaded[2:5].float())
    torch.testing.assert_close(destination[3], torch.tensor([-1.0, -1.0]))


def test_ple_shard_copy_is_a_noop_without_overlap() -> None:
    destination = torch.ones(4, 2)
    copied = copy_ple_embedding_shard_(
        destination,
        torch.zeros(2, 2),
        checkpoint_start=10,
        tp_start=4,
        tp_end=8,
    )

    assert copied == 0
    assert torch.equal(destination, torch.ones_like(destination))


def test_ngram_embedding_loads_shards_and_ignores_legacy_token_lookup() -> None:
    module = _make_ngram_embedding_for_load_test()
    shard_0 = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    shard_1 = torch.arange(8, 16, dtype=torch.float32).reshape(4, 2)

    loaded = module.load_weights(
        [
            ("ngram_embedding.shard_0.weight", shard_0),
            ("ngram_embedding.shard_1.weight", shard_1),
            ("token_lookup", torch.tensor([2, 1, 0])),
        ]
    )

    assert loaded == {"ngram_embedding.weight"}
    torch.testing.assert_close(
        module.ngram_embedding.weight,
        torch.cat((shard_0[2:4], shard_1[0:2])),
    )


def test_ngram_embedding_loads_fp8_shards_and_global_scale() -> None:
    module = _make_fp8_ngram_embedding_for_load_test()
    shard_0 = torch.arange(8, dtype=torch.float32).reshape(4, 2).to(torch.float8_e4m3fn)
    shard_1 = (
        torch.arange(8, 16, dtype=torch.float32).reshape(4, 2).to(torch.float8_e4m3fn)
    )
    weight_scale = torch.tensor([0.25], dtype=torch.bfloat16)

    loaded = module.load_weights(
        [
            ("ngram_embedding.shard_0.weight", shard_0),
            ("ngram_embedding.shard_1.weight", shard_1),
            ("ngram_embedding.weight_scale", weight_scale),
        ]
    )

    assert loaded == {"ngram_embedding.weight", "ngram_embedding.weight_scale"}
    assert module.ngram_embedding.weight.dtype == torch.float8_e4m3fn
    assert torch.equal(
        module.ngram_embedding.weight.float(),
        torch.cat((shard_0[2:4], shard_1[0:2])).float(),
    )
    assert torch.equal(module.ngram_embedding.weight_scale, weight_scale)


def _make_fp8_embedding_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> embedding_module.VocabParallelEmbedding:
    monkeypatch.setattr(embedding_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        embedding_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    method = Qwen4ExpPLEFp8EmbeddingMethod()
    layer = embedding_module.VocabParallelEmbedding(
        3,
        2,
        params_dtype=torch.bfloat16,
        padding_size=1,
        quant_method=method,
    )
    weight = torch.tensor([[1.0, 2.0], [4.0, 8.0], [16.0, 32.0]])
    layer.weight.data.copy_(weight.to(torch.float8_e4m3fn))
    layer.weight_scale.data.copy_(torch.tensor([0.25], dtype=torch.bfloat16))
    return layer


def test_ple_fp8_embedding_dequantizes_in_ple_layer(monkeypatch) -> None:
    layer = _make_fp8_embedding_layer(monkeypatch)
    quantized_output = layer(torch.tensor([2, 0]))
    ple_layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(ple_layer)
    ple_layer.ple_embedding = nn.Module()
    ple_layer.ple_embedding.ngram_embedding = layer

    output = ple_layer._dequantize_embeddings(
        quantized_output,
        torch.bfloat16,
    )

    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale.dtype == torch.bfloat16
    assert quantized_output.dtype == torch.float8_e4m3fn
    assert output.dtype == torch.bfloat16
    weight = torch.tensor([[1.0, 2.0], [4.0, 8.0], [16.0, 32.0]])
    torch.testing.assert_close(output, (weight[[2, 0]] * 0.25).bfloat16())


def test_ple_fp8_embedding_uses_int8_for_tp_reduce(monkeypatch) -> None:
    monkeypatch.setattr(embedding_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        embedding_module, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(
        embedding_module,
        "get_masked_input_and_mask",
        lambda *args: (
            torch.tensor([0, 0]),
            torch.tensor([False, True]),
        ),
    )
    reduced_dtypes = []

    def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        reduced_dtypes.append(tensor.dtype)
        return tensor.clone()

    monkeypatch.setattr(
        embedding_module,
        "tensor_model_parallel_all_reduce",
        all_reduce,
    )
    layer = embedding_module.VocabParallelEmbedding(
        4,
        2,
        params_dtype=torch.bfloat16,
        padding_size=1,
        quant_method=Qwen4ExpPLEFp8EmbeddingMethod(),
    )
    layer.weight.data.copy_(
        torch.tensor([[1.0, 2.0], [4.0, 8.0]]).to(torch.float8_e4m3fn)
    )

    output = layer(torch.tensor([0, 2]))

    assert reduced_dtypes == [torch.int8]
    assert output.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(output[0].float(), layer.weight[0].float())
    assert torch.count_nonzero(output[1].float()) == 0


def test_ple_fp8_embedding_respects_checkpoint_shard_exclusions() -> None:
    prefix = "model.layers.1.ple.ple_embedding.ngram_embedding"
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        ignored_layers=[],
        weight_block_size=[128, 128],
    )
    assert isinstance(
        _get_ple_embedding_quant_method(quant_config, prefix),
        Qwen4ExpPLEFp8EmbeddingMethod,
    )

    quant_config.ignored_layers = [f"{prefix}.shard_0"]
    assert _get_ple_embedding_quant_method(quant_config, prefix) is None


def test_ple_explicit_fp8_dtype_overrides_global_quantization() -> None:
    method = _get_ple_embedding_quant_method(
        None,
        "model.layers.1.ple.ple_embedding.ngram_embedding",
        "float8_e4m3fn",
    )

    assert isinstance(method, Qwen4ExpPLEFp8EmbeddingMethod)


def test_ple_cpu_offload_constructs_embedding_on_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_QWEN4_EXP_PLE_CPU_OFFLOAD", "1")
    monkeypatch.setattr(embedding_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        embedding_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        ple_layer_module,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=1)
        ),
    )
    config = SimpleNamespace(
        ngram_size=2,
        heads_per_ngram=1,
        eos_token_id=3,
        vocab_size=64,
        split_ngram_parts=2,
        seed=1234,
        ngram_vocab_size_base=5,
        make_ngram_vocab_size_divisible_by=1,
        ple_embedding_dtype="float8_e4m3fn",
    )

    with torch.device("meta"):
        module = Qwen4ExpNGramEmbedding(
            config,
            embedding_dim=4,
            ple_dense_layer_id=0,
            max_total_tokens=8,
            max_num_reqs=2,
            prefix="model.layers.1.ple.ple_embedding",
            layer_name="model.layers.1.ple",
            params_dtype=torch.bfloat16,
        )

    assert module.cpu_offload
    assert module.ngram_embedding.weight.device.type == "cpu"
    assert module.ngram_embedding.weight.dtype == torch.float8_e4m3fn
    assert module.ngram_embedding.weight.is_pinned() == is_pin_memory_available()
    assert module.ngram_embedding.weight_scale.device.type == "cpu"
    assert module.ngram_embedding.weight._vllm_skip_device_loading_context
    assert module.ngram_embedding.weight_scale._vllm_skip_device_loading_context
    assert module.layer_multipliers.device.type == "meta"

    with device_loading_context(module.ngram_embedding, torch.device("meta")):
        assert module.ngram_embedding.weight.device.type == "cpu"
        assert module.ngram_embedding.weight_scale.device.type == "cpu"


def test_ple_ngram_ids_custom_op_uses_current_request_layout(monkeypatch) -> None:
    class RuntimeNGramEmbedding(nn.Module):
        def compute_ngram_ids(
            self,
            input_ids: torch.Tensor,
            query_start_loc: torch.Tensor,
            ngram_context: torch.Tensor,
        ) -> torch.Tensor:
            del input_ids, ngram_context
            num_reqs = query_start_loc.numel() - 1
            return torch.full((4, 2), num_reqs, dtype=torch.long)

    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(layer)
    layer.ple_embedding = RuntimeNGramEmbedding()
    monkeypatch.setattr(
        ple_layer_module,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"ple": layer}),
    )
    input_ids = torch.arange(4)
    ngram_context = torch.zeros(2, 2, dtype=torch.long)
    output = torch.empty(4, 2, dtype=torch.long)

    ple_layer_module.qwen4_exp_compute_ple_ngram_ids(
        input_ids,
        torch.tensor([0, 4]),
        ngram_context,
        output,
        "ple",
    )
    assert torch.equal(output, torch.ones_like(output))

    ple_layer_module.qwen4_exp_compute_ple_ngram_ids(
        input_ids,
        torch.tensor([0, 2, 4]),
        ngram_context,
        output,
        "ple",
    )
    assert torch.equal(output, torch.full_like(output, 2))


def test_cpu_ple_embedding_custom_op_copies_lookup_to_output(monkeypatch) -> None:
    class RuntimeEmbedding(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            weight = torch.arange(24, dtype=torch.bfloat16).reshape(8, 3)
            self.weight = nn.Parameter(
                weight.to(torch.float8_e4m3fn), requires_grad=False
            )
            self.weight_scale = nn.Parameter(
                torch.tensor([0.5], dtype=torch.bfloat16), requires_grad=False
            )

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return F.embedding(input_ids, self.weight)

    class RuntimeNGramEmbedding(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ngram_embedding = RuntimeEmbedding()

    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(layer)
    layer.ple_embedding = RuntimeNGramEmbedding()
    monkeypatch.setattr(
        ple_layer_module,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"ple": layer}),
    )
    ngram_ids = torch.tensor([[1, 2], [3, 4]])
    output = torch.empty(2, 2, 3, dtype=torch.bfloat16)

    ple_layer_module.qwen4_exp_cpu_ple_embedding(
        ngram_ids,
        output,
        "ple",
    )

    expected = layer.ple_embedding.ngram_embedding(ngram_ids).to(torch.bfloat16) * 0.5
    torch.testing.assert_close(output, expected)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_pinned_ple_embedding_gathers_directly_from_host(
    weight_dtype: torch.dtype,
) -> None:
    values = torch.tensor(
        [
            [0.25, -0.5, 1.0],
            [2.0, 4.0, -8.0],
            [0.125, -0.25, 0.5],
            [16.0, -32.0, 64.0],
        ],
        dtype=torch.bfloat16,
    )
    weight = torch.empty(
        values.shape,
        dtype=weight_dtype,
        device="cpu",
        pin_memory=True,
    )
    weight.copy_(values.to(weight_dtype))
    input_ids = torch.tensor([[2, 3], [5, 6]], device="cuda", dtype=torch.long)
    output = torch.empty((2, 2, 3), device="cuda", dtype=torch.bfloat16)

    gather_pinned_ple_embedding(
        weight,
        input_ids,
        output,
        vocab_start=2,
        vocab_end=6,
        scale=0.5,
    )

    expected = torch.zeros_like(output, device="cpu")
    expected[0, 0] = weight[0].to(torch.bfloat16) * 0.5
    expected[0, 1] = weight[1].to(torch.bfloat16) * 0.5
    expected[1, 0] = weight[3].to(torch.bfloat16) * 0.5
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_pinned_ple_embedding_replays_in_cuda_graph() -> None:
    weight = torch.arange(
        32,
        dtype=torch.bfloat16,
    ).reshape(8, 4)
    pinned_weight = torch.empty_like(weight, pin_memory=True)
    pinned_weight.copy_(weight)
    input_ids = torch.tensor([[0, 1]], device="cuda", dtype=torch.long)
    output = torch.empty((1, 2, 4), device="cuda", dtype=torch.bfloat16)

    gather_pinned_ple_embedding(
        pinned_weight,
        input_ids,
        output,
        vocab_start=0,
        vocab_end=8,
        scale=1.0,
    )
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gather_pinned_ple_embedding(
            pinned_weight,
            input_ids,
            output,
            vocab_start=0,
            vocab_end=8,
            scale=1.0,
        )

    input_ids.copy_(torch.tensor([[6, 7]], device="cuda"))
    graph.replay()

    torch.testing.assert_close(output.cpu(), weight[6:8].unsqueeze(0))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_fused_ple_ngram_hash_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eos_token_id = 99
    input_ids = torch.tensor(
        [1, eos_token_id, 2, 3, 4, 5, 6, eos_token_id, 7],
        device="cuda",
        dtype=torch.long,
    )
    query_start_loc = torch.tensor([0, 5, 9], device="cuda", dtype=torch.long)
    ngram_context = torch.tensor(
        [[eos_token_id, 10], [11, 12]], device="cuda", dtype=torch.long
    )
    positions = torch.arange(input_ids.numel(), device="cuda")
    request_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
    token_offsets = positions - query_start_loc[request_indices]
    multipliers = torch.tensor(
        Qwen4ExpNGramEmbedding._make_layer_multipliers(
            ngram_size=3,
            unigram_vocab_size=248320,
            seed=1234,
            ple_dense_layer_id=1,
        ),
        device="cuda",
    )
    vocab_sizes = torch.tensor(
        [10007 + 2 * index for index in range(16)],
        device="cuda",
        dtype=torch.long,
    )
    head_offsets = F.pad(vocab_sizes.cumsum(0)[:-1], (1, 0))

    reference = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(reference)
    reference.ngram_size = 3
    reference.heads_per_ngram = 8
    reference.ngram_heads = 16
    reference.eos_token_id = eos_token_id
    reference.positions_buffer = positions
    reference.padded_buffer = torch.empty(
        (2, input_ids.numel()), device="cuda", dtype=torch.long
    )
    reference.layer_multipliers = multipliers
    reference.ngram_heads_vocab_sizes = vocab_sizes
    reference.ngram_heads_offsets = head_offsets
    monkeypatch.setattr(
        ple_layer_module, "can_use_fused_ple_ngram_hash", lambda *a: False
    )
    expected = reference.compute_ngram_ids(input_ids, query_start_loc, ngram_context)

    actual = fused_ple_ngram_hash(
        input_ids,
        ngram_context,
        request_indices,
        token_offsets,
        multipliers,
        vocab_sizes,
        head_offsets,
        eos_token_id,
    )

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("strided_state", [False, True])
def test_fused_ple_prefill_short_conv_matches_fallback(
    monkeypatch: pytest.MonkeyPatch,
    strided_state: bool,
) -> None:
    lengths = torch.tensor([2, 11, 19], device="cuda", dtype=torch.long)
    query_start_loc = F.pad(lengths.cumsum(0), (1, 0))
    request_indices = torch.repeat_interleave(torch.arange(3, device="cuda"), lengths)
    token_offsets = torch.arange(lengths.sum(), device="cuda")
    token_offsets -= query_start_loc[request_indices]
    x = torch.randn((int(lengths.sum()), 10240), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((10240, 4), device="cuda", dtype=torch.bfloat16)
    if strided_state:
        state = torch.randn(
            (4, 12, 10240), device="cuda", dtype=torch.bfloat16
        ).transpose(1, 2)
    else:
        state = torch.randn((4, 10240, 12), device="cuda", dtype=torch.bfloat16)
    state_indices = torch.tensor([1, 0, 3], device="cuda", dtype=torch.long)
    has_initial = torch.tensor([True, False, False], device="cuda")

    reference_layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(reference_layer)
    reference_layer.conv_state_len = 9
    reference_layer.short_conv_dilation = 3
    reference_state = state.clone()
    metadata = ple_layer_module.PleShortConvAttentionMetadata(
        num_prefills=3,
        num_prefill_tokens=x.shape[0],
        num_decodes=0,
        num_decode_tokens=0,
        num_reqs=3,
        has_initial_states_p=has_initial,
        query_start_loc_p=query_start_loc,
        num_computed_tokens_p=None,
        state_indices_tensor_p=state_indices,
        state_indices_tensor_d=None,
        query_start_loc_d=None,
        num_accepted_tokens=None,
        block_idx_last_scheduled_token=None,
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=None,
        block_idx_last_scheduled_token_prev_step=None,
        seq_lens=lengths,
        num_actual_tokens=x.shape[0],
        max_prefill_query_len=int(lengths.max()),
        query_start_loc=query_start_loc,
        state_indices_tensor=state_indices,
        non_spec_query_start_loc=query_start_loc,
    )
    monkeypatch.setattr(
        ple_layer_module, "can_use_fused_ple_prefill_short_conv", lambda *args: False
    )
    expected = reference_layer._short_conv_dilated_prefill_batched(
        x,
        metadata,
        reference_state,
        weight,
        state_indices,
        num_prefills=3,
        num_decode_tokens=0,
        num_prefill_tokens=x.shape[0],
    )

    actual_state = state.clone()
    actual = fused_ple_prefill_short_conv(
        x,
        actual_state,
        state_indices,
        has_initial,
        request_indices,
        token_offsets,
        query_start_loc,
        lengths,
        weight,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, reference_state, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("accepted_tokens", [1, 2, 3, 4])
@pytest.mark.parametrize("strided_state", [False, True])
def test_fused_ple_spec_short_conv_matches_fallback(
    monkeypatch: pytest.MonkeyPatch,
    accepted_tokens: int,
    strided_state: bool,
) -> None:
    x = torch.randn((4, 10240), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((10240, 4), device="cuda", dtype=torch.bfloat16)
    if strided_state:
        state = torch.randn(
            (4, 12, 10240), device="cuda", dtype=torch.bfloat16
        ).transpose(1, 2)
    else:
        state = torch.randn((4, 10240, 12), device="cuda", dtype=torch.bfloat16)
    state_indices = torch.tensor([2], device="cuda", dtype=torch.int32)
    query_start_loc = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
    accepted = torch.tensor([accepted_tokens], device="cuda", dtype=torch.int32)

    reference_layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(reference_layer)
    reference_layer.conv_state_len = 9
    reference_layer.short_conv_dilation = 3
    reference_state = state.clone()
    monkeypatch.setattr(
        ple_layer_module, "can_use_fused_ple_spec_short_conv", lambda *args: False
    )
    expected = reference_layer._short_conv_dilated_spec_batched(
        x,
        reference_state,
        weight,
        state_indices,
        query_start_loc,
        accepted,
        spec_query_len=4,
    )

    actual_state = state.clone()
    actual = fused_ple_spec_short_conv(
        x,
        actual_state,
        state_indices,
        query_start_loc,
        accepted,
        weight,
        spec_query_len=4,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, reference_state, rtol=0, atol=0)


@large_gpu_test(min_gb=8)
def test_fused_ple_spec_short_conv_large_state_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_index = 188
    state_slot_stride = 11_468_800
    state = torch.empty_strided(
        (state_index + 1, 10240, 12),
        (state_slot_stride, 1, 10240),
        device="cuda",
        dtype=torch.bfloat16,
    )
    target_state = torch.randn((10240, 12), device="cuda", dtype=torch.bfloat16)
    state[state_index].copy_(target_state)

    x = torch.randn((4, 10240), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((10240, 4), device="cuda", dtype=torch.bfloat16)
    state_indices = torch.tensor([state_index], device="cuda", dtype=torch.int32)
    query_start_loc = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
    accepted = torch.tensor([1], device="cuda", dtype=torch.int32)

    reference_layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(reference_layer)
    reference_layer.conv_state_len = 9
    reference_layer.short_conv_dilation = 3
    reference_state = torch.zeros((2, 10240, 12), device="cuda", dtype=torch.bfloat16)
    reference_state[1].copy_(target_state)
    monkeypatch.setattr(
        ple_layer_module, "can_use_fused_ple_spec_short_conv", lambda *args: False
    )
    expected = reference_layer._short_conv_dilated_spec_batched(
        x,
        reference_state,
        weight,
        torch.tensor([1], device="cuda", dtype=torch.int32),
        query_start_loc,
        accepted,
        spec_query_len=4,
    )

    actual = fused_ple_spec_short_conv(
        x,
        state,
        state_indices,
        query_start_loc,
        accepted,
        weight,
        spec_query_len=4,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state[state_index], reference_state[1], rtol=0, atol=0)


def test_dilated_ple_spec_state_rolls_back_before_next_forward() -> None:
    module = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(module)
    module.conv_state_len = 6
    module.short_conv_dilation = 2

    conv_weights = torch.tensor([[0.25, -0.5, 0.75, 1.0]])
    conv_state = torch.zeros(2, 1, 9)
    conv_state[1] = torch.arange(1, 10, dtype=torch.float32).reshape(1, 9)
    first_inputs = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    initial_state = conv_state[1:].clone()
    first_history = torch.cat(
        (initial_state[..., : module.conv_state_len], first_inputs.T.unsqueeze(0)),
        dim=-1,
    )

    graph_padded_inputs = F.pad(first_inputs, (0, 0, 0, 4))
    first_output = module._short_conv_dilated_spec_batched(
        graph_padded_inputs,
        conv_state,
        conv_weights,
        torch.tensor([1, 0]),
        torch.tensor([0, 4, 4]),
        torch.tensor([1, 0]),
        spec_query_len=4,
    )

    expected_first_output = F.silu(
        F.conv1d(
            first_history,
            conv_weights.unsqueeze(1),
            groups=1,
            dilation=module.short_conv_dilation,
        )
    ).transpose(1, 2)[0]
    expected_first_state = first_history[..., 1:10]
    torch.testing.assert_close(first_output[:4], expected_first_output)
    assert torch.count_nonzero(first_output[4:]) == 0
    assert torch.count_nonzero(conv_state[0]) == 0
    torch.testing.assert_close(conv_state[1:], expected_first_state)

    second_inputs = torch.tensor([[50.0], [60.0]])
    rollback_state = expected_first_state[..., 1:7]
    padded_second_inputs = F.pad(second_inputs.T.unsqueeze(0), (0, 2))
    second_history = torch.cat((rollback_state, padded_second_inputs), dim=-1)
    expected_second_state = expected_first_state.clone()
    expected_second_state[..., :7] = second_history[..., 1:8]

    second_output = module._short_conv_dilated_spec_batched(
        second_inputs,
        conv_state,
        conv_weights,
        torch.tensor([1]),
        torch.tensor([0, 2]),
        torch.tensor([2]),
        spec_query_len=4,
    )

    expected_second_output = F.silu(
        F.conv1d(
            second_history,
            conv_weights.unsqueeze(1),
            groups=1,
            dilation=module.short_conv_dilation,
        )
    ).transpose(1, 2)[0, :2]
    torch.testing.assert_close(second_output, expected_second_output)
    torch.testing.assert_close(conv_state[1:], expected_second_state)


def test_ple_short_conv_uses_fallback_when_profile_metadata_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(module)
    module.prefix = "model.layers.1.ple"
    inputs = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    expected = inputs + 1
    monkeypatch.setattr(module, "_short_conv_fallback", lambda _: expected)
    monkeypatch.setattr(
        ple_layer_module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"model.layers.0.self_attn": object()}),
    )

    output = module._short_conv(inputs)

    assert output is expected
