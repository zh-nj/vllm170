# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(80),
    reason="DeepSeek-V4 Triton sparse prefill tuning targets SM8x",
)


def test_direct_swa_ragged_controls_dense_decode_buffer(monkeypatch) -> None:
    import vllm.envs as envs
    from vllm.models.deepseek_v4.amd.rocm import (
        DeepseekV4ROCMAiterMLASparseMetadataBuilder,
        DeepseekV4ROCMAiterSparseSWAMetadataBuilder,
    )
    from vllm.models.deepseek_v4_1.amd.rocm import (
        DeepseekV4ROCMAiterSparseSWAMetadataBuilder as DeepseekV41SWABuilder,
    )

    builder = object.__new__(DeepseekV4ROCMAiterSparseSWAMetadataBuilder)
    monkeypatch.setattr(envs, "VLLM_DEEPSEEK_V4_DIRECT_RAGGED_SWA", False)
    assert builder._requires_dense_decode_swa_indices()

    monkeypatch.setattr(envs, "VLLM_DEEPSEEK_V4_DIRECT_RAGGED_SWA", True)
    assert not builder._requires_dense_decode_swa_indices()

    v41_builder = object.__new__(DeepseekV41SWABuilder)
    assert not v41_builder._requires_dense_decode_swa_indices()
    v41_builder._use_direct_ragged_swa = True
    assert not v41_builder._requires_decode_swa_lens_tail_clear()
    v41_builder._use_direct_ragged_swa = False
    assert v41_builder._requires_decode_swa_lens_tail_clear()

    c128_builder = object.__new__(DeepseekV4ROCMAiterMLASparseMetadataBuilder)
    monkeypatch.setattr(envs, "VLLM_DEEPSEEK_V4_DIRECT_RAGGED_C128", False)
    assert c128_builder._requires_dense_c128a_decode_indices()

    monkeypatch.setattr(envs, "VLLM_DEEPSEEK_V4_DIRECT_RAGGED_C128", True)
    assert not c128_builder._requires_dense_c128a_decode_indices()


def test_single_request_swa_builder_reuses_persistent_token_map() -> None:
    from vllm.models.deepseek_v4_1.amd.rocm import (
        DeepseekV4ROCMAiterSparseSWAMetadataBuilder,
    )

    builder = object.__new__(DeepseekV4ROCMAiterSparseSWAMetadataBuilder)
    builder._persist_single_req_token_map = True
    # Rows left behind by an earlier two-request batch. A single request must
    # not inherit them: every one of its tokens belongs to request 0.
    builder.token_to_req_indices = torch.tensor(
        [0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.int32
    )
    common = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=6,
        _token_to_req_indices_cache=None,
    )

    actual = builder._get_token_to_req_indices(common)

    assert actual.data_ptr() == builder.token_to_req_indices.data_ptr()
    assert common._token_to_req_indices_cache is actual
    torch.testing.assert_close(actual, torch.zeros(6, dtype=torch.int32))
    # The slice is refreshed in place, so the buffer's tail still holds the
    # previous batch and a longer follow-up request must refresh it again.
    common_long = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=8,
        _token_to_req_indices_cache=None,
    )
    actual_long = builder._get_token_to_req_indices(common_long)
    torch.testing.assert_close(actual_long, torch.zeros(8, dtype=torch.int32))

    # The general recompute path is unaffected and stays correct for the same
    # state transition when the fast path is switched off.
    builder._persist_single_req_token_map = False
    builder.token_to_req_indices = torch.tensor(
        [0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.int32
    )
    rebuilt = torch.zeros(8, dtype=torch.int32)

    def token_to_req_indices(buffer: torch.Tensor) -> torch.Tensor:
        buffer.zero_()
        return buffer[:8]

    common_slow = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=8,
        _token_to_req_indices_cache=None,
        query_start_loc_cpu=torch.tensor([0, 8], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 8], dtype=torch.int32),
        token_to_req_indices=token_to_req_indices,
    )
    torch.testing.assert_close(builder._get_token_to_req_indices(common_slow), rebuilt)


def test_single_request_swa_builder_recomputes_validity() -> None:
    """Validity tracks the live slot mapping, not the previous batch.

    A single request does not imply every slot is live: eviction and rejected
    suffixes both produce PAD slots, and a buffer left with False entries by an
    earlier batch must not mask a fully live batch (or vice versa).
    """
    from vllm.models.deepseek_v4_1.amd.rocm import (
        DeepseekV4ROCMAiterSparseSWAMetadataBuilder,
    )

    builder = object.__new__(DeepseekV4ROCMAiterSparseSWAMetadataBuilder)
    # Stale mask from a batch that had invalid slots.
    builder.is_valid_token = torch.tensor([True, False, True, False])
    common = SimpleNamespace(num_reqs=1)

    # A fully live single-request decode batch must clear the stale entries.
    live = torch.tensor([3, 4], dtype=torch.int64)
    actual = builder._get_is_valid_token(common, live, 2, 0)
    assert actual.data_ptr() == builder.is_valid_token.data_ptr()
    torch.testing.assert_close(actual, torch.ones(2, dtype=torch.bool))

    # The reverse direction: a single request whose slots are PAD must be
    # reported invalid rather than inheriting the previous all-live mask.
    padded = torch.tensor([7, 8, -1, -1], dtype=torch.int64)
    torch.testing.assert_close(
        builder._get_is_valid_token(common, padded, 4, 0),
        torch.tensor([True, True, False, False]),
    )


def test_v41_single_request_fast_paths_ignore_max_num_seqs(monkeypatch) -> None:
    """Single-request fast paths must be gated on batch size, not capacity.

    Configuring a larger `max_num_seqs` keeps multi-request capacity but must
    not disable the paths that only ever apply to a single-request batch.
    """
    import vllm.envs as envs
    from vllm.models.deepseek_v4_1.amd.rocm import (
        DeepseekV4ROCMAiterSparseSWAMetadataBuilder,
    )
    from vllm.models.deepseek_v4_1.compressor import CompressorMetadataBuilder
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

    monkeypatch.setattr(envs, "VLLM_DEEPSEEK_V4_PERSIST_SINGLE_REQ_TOKEN_MAP", True)
    monkeypatch.setattr(envs, "VLLM_DEEPSEEK_V4_DIRECT_RAGGED_SWA", True)

    def flags(max_num_seqs: int) -> tuple[bool, bool, bool]:
        spec = SlidingWindowMLASpec(
            block_size=32,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            sliding_window=128,
            cache_dtype_str="fp8_ds_mla",
            state_content_bytes=584,
            alignment=576,
            model_version="deepseek_v4",
        )
        vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=8,
            ),
            model_config=SimpleNamespace(
                max_model_len=1024,
                hf_config=SimpleNamespace(sliding_window=128),
            ),
            speculative_config=None,
            attention_config=None,
        )
        builder = DeepseekV4ROCMAiterSparseSWAMetadataBuilder(
            spec,
            ["model.layers.0.attn.swa_cache"],
            vllm_config,
            torch.device("cpu"),
        )
        return (
            builder._persist_single_req_token_map,
            builder._use_paged_decode_swa,
            builder._requires_decode_swa_lens_tail_clear(),
        )

    assert flags(1) == flags(8) == (True, True, False)
    assert not hasattr(
        DeepseekV4ROCMAiterSparseSWAMetadataBuilder(
            SlidingWindowMLASpec(
                block_size=32,
                num_kv_heads=1,
                head_size=512,
                dtype=torch.uint8,
                sliding_window=128,
                cache_dtype_str="fp8_ds_mla",
                state_content_bytes=584,
                alignment=576,
                model_version="deepseek_v4",
            ),
            ["model.layers.0.attn.swa_cache"],
            SimpleNamespace(
                scheduler_config=SimpleNamespace(
                    max_num_seqs=8, max_num_batched_tokens=8
                ),
                model_config=SimpleNamespace(
                    max_model_len=1024,
                    hf_config=SimpleNamespace(sliding_window=128),
                ),
                speculative_config=None,
                attention_config=None,
            ),
            torch.device("cpu"),
        ),
        "_persist_single_req_validity",
    )

    # The paths themselves still require an actually single-request batch.
    compressor = object.__new__(CompressorMetadataBuilder)
    compressor._persist_single_req_token_map = True
    compressor.token_to_req_indices = torch.zeros(4, dtype=torch.int32)
    rebuilt = torch.zeros(2, dtype=torch.int32)

    def token_to_req_indices(buffer: torch.Tensor) -> torch.Tensor:
        return rebuilt

    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=2,
        _token_to_req_indices_cache=None,
        token_to_req_indices=token_to_req_indices,
    )

    actual = compressor._get_token_to_req_indices(common)

    assert actual.data_ptr() == rebuilt.data_ptr()


def test_persistent_token_map_handles_real_batch_transitions() -> None:
    """Every map builder must survive a real multi -> single -> multi sequence.

    Unlike the synthetic-stub checks above, this drives the genuine
    `CommonAttentionMetadata.token_to_req_indices` recompute path, so the
    published buffer is really rewritten between the multi-request and
    single-request steps. Sequence of per-request query lengths:
    ``[1] * 8 -> [2] -> [8] -> [1] * 8`` must produce
    ``0..7 -> [0, 0] -> [0] * 8 -> 0..7``.
    """
    from vllm.models.deepseek_v4_1.amd.rocm import (
        DeepseekV4ROCMAiterSparseSWAMetadataBuilder,
    )
    from vllm.models.deepseek_v4_1.compressor import CompressorMetadataBuilder
    from vllm.models.deepseek_v4_1.sparse_mla import (
        DeepseekV4SparseMLAMetadataBuilder,
    )
    from vllm.v1.attention.backend import CommonAttentionMetadata

    def common(query_lens: list[int]) -> CommonAttentionMetadata:
        boundaries = [0]
        for length in query_lens:
            boundaries.append(boundaries[-1] + length)
        qsl = torch.tensor(boundaries, dtype=torch.int32)
        return CommonAttentionMetadata(
            query_start_loc=qsl,
            query_start_loc_cpu=qsl,
            seq_lens=torch.tensor(
                [max(length, 1)] * len(query_lens), dtype=torch.int32
            ),
            num_reqs=len(query_lens),
            num_actual_tokens=boundaries[-1],
            max_query_len=max(query_lens),
            max_seq_len=max(query_lens),
            block_table_tensor=torch.zeros(len(query_lens), 2, dtype=torch.int32),
            slot_mapping=torch.arange(boundaries[-1], dtype=torch.int64),
        )

    builders = (
        (DeepseekV4ROCMAiterSparseSWAMetadataBuilder, "token_to_req_indices"),
        (DeepseekV4SparseMLAMetadataBuilder, "req_id_per_token_buffer"),
        (CompressorMetadataBuilder, "token_to_req_indices"),
    )

    expected = [
        (list(range(8)), [1] * 8),
        ([0, 0], [2]),
        ([0] * 8, [8]),
        (list(range(8)), [1] * 8),
    ]

    for builder_cls, buffer_name in builders:
        for fast_path in (True, False):
            builder = object.__new__(builder_cls)
            builder._persist_single_req_token_map = fast_path
            setattr(builder, buffer_name, torch.zeros(16, dtype=torch.int32))
            for want, query_lens in expected:
                actual = builder._get_token_to_req_indices(common(query_lens))
                torch.testing.assert_close(
                    actual,
                    torch.tensor(want, dtype=torch.int32),
                    msg=f"{builder_cls.__name__} fast_path={fast_path} "
                    f"query_lens={query_lens}",
                )

        # The published pointer must stay the builder buffer's slice.
        builder = object.__new__(builder_cls)
        builder._persist_single_req_token_map = True
        buffer = torch.zeros(16, dtype=torch.int32)
        setattr(builder, buffer_name, buffer)
        # Multi-request step: the general path publishes a slice of the same
        # buffer, so compare the storage and the values, not object identity.
        metadata = common([1] * 8)
        published = builder._get_token_to_req_indices(metadata)
        assert published.data_ptr() == buffer.data_ptr()
        torch.testing.assert_close(
            metadata._token_to_req_indices_cache,
            torch.arange(8, dtype=torch.int32),
        )
        assert published.shape == (8,)
        assert published.dtype == torch.int32

        # Single-request step: the fast path publishes its own slice object.
        metadata = common([4])
        published = builder._get_token_to_req_indices(metadata)
        assert published.data_ptr() == buffer.data_ptr()
        assert metadata._token_to_req_indices_cache is published
        torch.testing.assert_close(published, torch.zeros(4, dtype=torch.int32))


def test_single_request_v41_builders_reuse_persistent_token_map() -> None:
    from vllm.models.deepseek_v4_1.compressor import CompressorMetadataBuilder
    from vllm.models.deepseek_v4_1.sparse_mla import (
        DeepseekV4SparseMLAMetadataBuilder,
    )

    for builder_cls, buffer_name in (
        (DeepseekV4SparseMLAMetadataBuilder, "req_id_per_token_buffer"),
        (CompressorMetadataBuilder, "token_to_req_indices"),
    ):
        builder = object.__new__(builder_cls)
        builder._persist_single_req_token_map = True
        # Left behind by a two-request batch; the single request must not
        # inherit the per-request row numbers.
        buffer = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        setattr(builder, buffer_name, buffer)
        common = SimpleNamespace(
            num_reqs=1,
            num_actual_tokens=4,
            _token_to_req_indices_cache=None,
        )

        actual = builder._get_token_to_req_indices(common)

        assert actual.data_ptr() == buffer.data_ptr()
        assert common._token_to_req_indices_cache is actual
        torch.testing.assert_close(actual, torch.zeros(4, dtype=torch.int32))

        # Back to a multi-request batch: the general path fills the buffer with
        # per-request rows again and must overwrite the zeroed slice.
        rebuilt = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        common_multi = SimpleNamespace(
            num_reqs=2,
            num_actual_tokens=4,
            _token_to_req_indices_cache=None,
            token_to_req_indices=lambda buffer, _rebuilt=rebuilt: _rebuilt,
        )
        torch.testing.assert_close(
            builder._get_token_to_req_indices(common_multi), rebuilt
        )


@torch.inference_mode()
def test_sparse_prefill_caller_output_and_block_k_match(monkeypatch) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_prefill

    torch.manual_seed(0)
    q = torch.randn(4, 8, 512, dtype=torch.bfloat16, device="cuda") * 0.125
    kv = torch.randn(8, 1, 512, dtype=torch.bfloat16, device="cuda") * 0.125
    indices = torch.tensor(
        [
            [0, 2, 4, -1, -1, -1],
            [1, 3, 5, 7, -1, -1],
            [0, 1, 2, 3, 4, -1],
            [2, 3, 4, 5, 6, 7],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    lengths = torch.tensor([3, 4, 5, 6], dtype=torch.int32, device="cuda")
    sink = torch.full((8,), -float("inf"), dtype=torch.float32, device="cuda")

    def run(caller_output: int, block_k: int) -> torch.Tensor:
        monkeypatch.setenv(
            "VLLM_DEEPSEEK_V4_SPARSE_PREFILL_CALLER_OUTPUT", str(caller_output)
        )
        monkeypatch.setenv("VLLM_DEEPSEEK_V4_SPARSE_PREFILL_BLOCK_K", str(block_k))
        output = torch.full_like(q, float("nan"))
        rocm_sparse_attn_prefill(
            q=q,
            kv=kv,
            indices=indices,
            topk_length=lengths,
            scale=512**-0.5,
            head_dim=512,
            nope_head_dim=448,
            rope_head_dim=64,
            attn_sink=sink,
            output=output,
        )
        return output

    copied = run(caller_output=0, block_k=16)
    direct = run(caller_output=1, block_k=16)
    tuned = run(caller_output=1, block_k=32)

    torch.testing.assert_close(direct, copied, atol=0, rtol=0)
    torch.testing.assert_close(tuned, copied, atol=2e-2, rtol=2e-2)
    assert torch.isfinite(tuned).all()


@torch.inference_mode()
def test_sparse_prefill_fp8_cache_matches_dequantized_reference(monkeypatch) -> None:
    """SM80 decodes E4M3 KV bytes through the software lookup-table path."""
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_prefill

    torch.manual_seed(17)
    head_dim = 256
    q = torch.randn(4, 8, head_dim, dtype=torch.bfloat16, device="cuda") * 0.125
    kv = torch.randn(8, 1, head_dim, dtype=torch.bfloat16, device="cuda") * 0.125
    kv_scale = torch.tensor([0.01], dtype=torch.float32, device="cuda")
    kv_fp8 = (kv / kv_scale).to(current_platform.fp8_dtype())
    kv_dequantized = (kv_fp8.to(torch.bfloat16) * kv_scale).to(torch.bfloat16)
    indices = torch.tensor(
        [
            [0, 2, 4, -1, -1, -1],
            [1, 3, 5, 7, -1, -1],
            [0, 1, 2, 3, 4, -1],
            [2, 3, 4, 5, 6, 7],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    lengths = torch.tensor([3, 4, 5, 6], dtype=torch.int32, device="cuda")
    monkeypatch.setenv("VLLM_DEEPSEEK_V4_SPARSE_PREFILL_CALLER_OUTPUT", "1")

    def run(cache: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
        output = torch.empty_like(q)
        rocm_sparse_attn_prefill(
            q=q,
            kv=cache,
            indices=indices,
            topk_length=lengths,
            scale=head_dim**-0.5,
            head_dim=head_dim,
            nope_head_dim=head_dim,
            rope_head_dim=0,
            attn_sink=None,
            output=output,
            kv_scale=scale,
        )
        return output

    reference = run(kv_dequantized, None)
    actual = run(kv_fp8, kv_scale)
    torch.testing.assert_close(actual, reference, atol=2e-2, rtol=2e-2)
    assert torch.isfinite(actual).all()


@torch.inference_mode()
def test_glm_sparse_prefill_tuning_matches_default(monkeypatch) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_prefill

    torch.manual_seed(29)
    num_queries, num_heads, head_dim = 4, 64, 512
    q = torch.randn(
        num_queries,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    kv_scale = torch.tensor(0.01, dtype=torch.float32, device="cuda")
    kv = (
        torch.randn(32, 1, head_dim, dtype=torch.bfloat16, device="cuda") / kv_scale
    ).to(current_platform.fp8_dtype())
    indices = torch.arange(num_queries * 16, dtype=torch.int32, device="cuda").reshape(
        num_queries, 16
    )
    lengths = torch.tensor([8, 12, 16, 16], dtype=torch.int32, device="cuda")
    monkeypatch.setenv("VLLM_DEEPSEEK_V4_SPARSE_PREFILL_CALLER_OUTPUT", "1")

    def run(**kernel_config: int) -> torch.Tensor:
        output = torch.empty_like(q)
        rocm_sparse_attn_prefill(
            q=q,
            kv=kv,
            indices=indices,
            topk_length=lengths,
            scale=head_dim**-0.5,
            head_dim=head_dim,
            nope_head_dim=head_dim,
            rope_head_dim=0,
            attn_sink=None,
            output=output,
            kv_scale=kv_scale,
            **kernel_config,
        )
        return output

    reference = run()
    tuned = run(block_h=32, block_k=64, num_warps=8)

    torch.testing.assert_close(tuned, reference, atol=2e-2, rtol=2e-2)
    assert torch.isfinite(tuned).all()


@torch.inference_mode()
@pytest.mark.parametrize("num_queries", [1, 2, 4, 8])
@pytest.mark.parametrize("caller_output", [False, True])
def test_sparse_decode_split_and_caller_output_match(
    monkeypatch, num_queries: int, caller_output: bool
) -> None:
    from vllm.models.deepseek_v4.common.ops import quantize_and_insert_k_cache
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as sparse_mod
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _rocm_sparse_attn_decode_ragged_triton,
        rocm_sparse_attn_decode,
    )
    from vllm.v1.worker import workspace

    num_heads, head_dim = 64, 512
    block_size, main_len, extra_len = 256, 128, 64
    num_rows = 1024
    torch.manual_seed(23)
    q = torch.randn(
        num_queries, num_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )

    def make_cache() -> torch.Tensor:
        cache = torch.zeros(4, block_size, 584, dtype=torch.uint8, device="cuda")
        values = torch.randn(num_rows, head_dim, dtype=torch.bfloat16, device="cuda")
        slots = torch.arange(num_rows, dtype=torch.int64, device="cuda")
        quantize_and_insert_k_cache(values, cache, slots, block_size)
        return cache

    main_cache = make_cache()
    extra_cache = make_cache()
    main_indices = torch.arange(
        num_queries * main_len, dtype=torch.int32, device="cuda"
    ).remainder_(num_rows)
    main_indptr = torch.arange(
        0,
        (num_queries + 1) * main_len,
        main_len,
        dtype=torch.int32,
        device="cuda",
    )
    extra_indices = torch.arange(
        num_queries * extra_len, dtype=torch.int32, device="cuda"
    ).remainder_(num_rows)
    extra_indptr = torch.arange(
        0,
        (num_queries + 1) * extra_len,
        extra_len,
        dtype=torch.int32,
        device="cuda",
    )
    sink = torch.randn(num_heads, dtype=torch.float32, device="cuda")
    scale = head_dim**-0.5

    monkeypatch.setenv("VLLM_DEEPSEEK_V4_SPARSE_DECODE_SPLIT", "0")
    reference = _rocm_sparse_attn_decode_ragged_triton(
        q=q,
        main_cache=main_cache,
        main_indices=main_indices,
        main_indptr=main_indptr,
        scale=scale,
        attn_sink=sink,
        nope_head_dim=448,
        rope_head_dim=64,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_indptr=extra_indptr,
    )

    manager = workspace.WorkspaceManager(torch.device("cuda"))
    monkeypatch.setattr(workspace, "_manager", manager)
    monkeypatch.setenv("VLLM_DEEPSEEK_V4_SPARSE_DECODE_SPLIT", "1")
    monkeypatch.setenv("VLLM_DEEPSEEK_V4_SM80_DECODE_NUM_SPLITS", "32")
    monkeypatch.setenv(
        "VLLM_DEEPSEEK_V4_SPARSE_DECODE_CALLER_OUTPUT", str(int(caller_output))
    )
    output = torch.full_like(q, float("nan"))
    swa_indices = torch.zeros(num_queries, 1, 1, dtype=torch.int32, device="cuda")
    main_lens = torch.full((num_queries,), main_len, dtype=torch.int32, device="cuda")
    extra_lens = torch.full((num_queries,), extra_len, dtype=torch.int32, device="cuda")
    caller_outputs: list[torch.Tensor | None] = []
    original_decode = sparse_mod._rocm_sparse_attn_decode_triton

    def capture_caller_output(*args, **kwargs):
        caller_outputs.append(kwargs.get("out"))
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(
        sparse_mod, "_rocm_sparse_attn_decode_triton", capture_caller_output
    )

    def candidate() -> None:
        rocm_sparse_attn_decode(
            q=q,
            kv_cache=extra_cache,
            swa_k_cache=main_cache,
            swa_only=False,
            topk_indices=None,
            topk_lens=extra_lens,
            swa_indices=swa_indices,
            swa_lens=main_lens,
            swa_ragged_indices=main_indices,
            swa_ragged_indptr=main_indptr,
            topk_ragged_indices=extra_indices,
            topk_ragged_indptr=extra_indptr,
            attn_sink=sink,
            scale=scale,
            head_dim=head_dim,
            nope_head_dim=448,
            rope_head_dim=64,
            output=output,
        )

    candidate()
    torch.accelerator.synchronize()
    assert (caller_outputs[-1] is output) is caller_output
    torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)
    assert torch.isfinite(output).all()
    manager.lock()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidate()
    output.fill_(float("nan"))
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)


@torch.inference_mode()
def test_sparse_decode_paged_swa_matches_ragged(monkeypatch) -> None:
    from vllm.models.deepseek_v4.common.ops import quantize_and_insert_k_cache
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _rocm_sparse_attn_decode_ragged_triton,
        _rocm_sparse_attn_decode_triton,
    )
    from vllm.v1.worker import workspace

    num_heads, head_dim = 16, 512
    block_size, seq_len, window_size = 256, 320, 128
    num_blocks = 4
    torch.manual_seed(29)
    q = torch.randn(1, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    cache = torch.zeros(num_blocks, block_size, 584, dtype=torch.uint8, device="cuda")
    values = torch.randn(
        num_blocks * block_size,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    slots = torch.arange(num_blocks * block_size, dtype=torch.int64, device="cuda")
    quantize_and_insert_k_cache(values, cache, slots, block_size)

    block_table = torch.tensor([[2, 0]], dtype=torch.int32, device="cuda")
    logical_positions = torch.arange(
        seq_len - window_size, seq_len, dtype=torch.int32, device="cuda"
    )
    page_indices = torch.div(logical_positions, block_size, rounding_mode="floor")
    ragged_indices = (
        block_table[0, page_indices] * block_size + logical_positions % block_size
    )
    ragged_indptr = torch.tensor([0, window_size], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    sink = torch.randn(num_heads, dtype=torch.float32, device="cuda")

    monkeypatch.setenv("VLLM_DEEPSEEK_V4_SPARSE_DECODE_SPLIT", "0")
    reference = _rocm_sparse_attn_decode_ragged_triton(
        q=q,
        main_cache=cache,
        main_indices=ragged_indices,
        main_indptr=ragged_indptr,
        scale=head_dim**-0.5,
        attn_sink=sink,
        nope_head_dim=448,
        rope_head_dim=64,
    )

    monkeypatch.setenv("VLLM_DEEPSEEK_V4_SPARSE_DECODE_SPLIT", "1")
    monkeypatch.setattr(
        workspace, "_manager", workspace.WorkspaceManager(torch.device("cuda"))
    )
    candidate = _rocm_sparse_attn_decode_triton(
        q=q,
        main_cache=cache,
        main_indices=torch.empty(1, 0, dtype=torch.int32, device="cuda"),
        main_lengths=seq_lens,
        scale=head_dim**-0.5,
        attn_sink=sink,
        nope_head_dim=448,
        rope_head_dim=64,
        main_block_table=block_table,
        main_seq_lens=seq_lens,
        main_window_size=window_size,
    )

    torch.testing.assert_close(candidate, reference, atol=2e-2, rtol=2e-2)
    assert torch.isfinite(candidate).all()


@pytest.mark.parametrize("non_causal", [False, True])
@torch.inference_mode()
def test_direct_swa_ragged_matches_dense_pack(non_causal: bool) -> None:
    from vllm.models.deepseek_v4.amd.rocm import (
        build_swa_ragged_indices_direct,
    )
    from vllm.models.deepseek_v4_1.amd.rocm import (
        build_swa_ragged_indices_direct as build_v41_swa_ragged_indices_direct,
    )
    from vllm.v1.attention.backends.mla.sparse_swa import (
        _compute_dspark_noncausal_swa_indices_kernel,
        _compute_swa_indices_and_lens_kernel,
    )
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        build_ragged_indices_from_dense,
    )

    num_tokens = 5
    window_size = 8
    index_width = 16 if non_causal else window_size
    block_size = 64
    query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([70, 4], dtype=torch.int32, device="cuda")
    token_to_req = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32, device="cuda")
    is_valid = torch.tensor([1, 1, 0, 1, 1], dtype=torch.bool, device="cuda")
    block_table = torch.tensor([[5, 7], [11, -1]], dtype=torch.int32, device="cuda")

    dense = torch.full(
        (num_tokens, 1, index_width), -99, dtype=torch.int32, device="cuda"
    )
    reference_lens = torch.full((num_tokens,), -99, dtype=torch.int32, device="cuda")
    if non_causal:
        _compute_dspark_noncausal_swa_indices_kernel[(num_tokens,)](
            dense,
            dense.stride(0),
            reference_lens,
            window_size,
            index_width,
            query_start_loc,
            seq_lens,
            token_to_req,
            is_valid,
            block_table,
            block_table.stride(0),
            block_size,
            token_offset=0,
            TRITON_BLOCK_SIZE=16,
        )
    else:
        _compute_swa_indices_and_lens_kernel[(num_tokens,)](
            dense,
            dense.stride(0),
            reference_lens,
            window_size,
            index_width,
            reference_lens,
            reference_lens,
            query_start_loc,
            seq_lens,
            token_to_req,
            is_valid,
            block_table,
            block_table.stride(0),
            block_size,
            token_offset=0,
            HAS_IMAGE=False,
            TRITON_BLOCK_SIZE=16,
        )
    reference_indices, reference_indptr = build_ragged_indices_from_dense(
        dense.reshape(num_tokens, -1), reference_lens
    )

    for build_direct in (
        build_swa_ragged_indices_direct,
        build_v41_swa_ragged_indices_direct,
    ):
        direct_lens = torch.full_like(reference_lens, -77)
        direct_indices_buffer = torch.full(
            (num_tokens * index_width,), -77, dtype=torch.int32, device="cuda"
        )
        direct_indptr_buffer = torch.full(
            (num_tokens + 1,), -77, dtype=torch.int32, device="cuda"
        )
        direct_indices, direct_indptr = build_direct(
            num_decode_tokens=num_tokens,
            window_size=window_size,
            index_width=index_width,
            non_causal=non_causal,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            token_to_req_indices=token_to_req,
            is_valid_token=is_valid,
            block_table=block_table,
            block_size=block_size,
            swa_lens=direct_lens,
            ragged_indices_buffer=direct_indices_buffer,
            ragged_indptr_buffer=direct_indptr_buffer,
        )

        torch.testing.assert_close(direct_lens, reference_lens)
        torch.testing.assert_close(direct_indptr, reference_indptr)
        nnz = int(reference_indptr[-1].item())
        torch.testing.assert_close(direct_indices[:nnz], reference_indices[:nnz])
        assert direct_indices.data_ptr() == direct_indices_buffer.data_ptr()
        assert direct_indptr.data_ptr() == direct_indptr_buffer.data_ptr()


@pytest.mark.parametrize("seq_len", [4, 70])
@torch.inference_mode()
def test_direct_single_token_swa_ragged_matches_dense_pack(seq_len: int) -> None:
    from vllm.models.deepseek_v4_1.amd.rocm import (
        build_swa_ragged_indices_direct,
    )
    from vllm.v1.attention.backends.mla.sparse_swa import (
        _compute_swa_indices_and_lens_kernel,
    )
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        build_ragged_indices_from_dense,
    )

    window_size = 8
    block_size = 64
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    token_to_req = torch.zeros(1, dtype=torch.int32, device="cuda")
    is_valid = torch.ones(1, dtype=torch.bool, device="cuda")
    block_table = torch.tensor([[5, 7]], dtype=torch.int32, device="cuda")

    dense = torch.full((1, 1, window_size), -99, dtype=torch.int32, device="cuda")
    reference_lens = torch.full((1,), -99, dtype=torch.int32, device="cuda")
    _compute_swa_indices_and_lens_kernel[(1,)](
        dense,
        dense.stride(0),
        reference_lens,
        window_size,
        window_size,
        reference_lens,
        reference_lens,
        query_start_loc,
        seq_lens,
        token_to_req,
        is_valid,
        block_table,
        block_table.stride(0),
        block_size,
        token_offset=0,
        HAS_IMAGE=False,
        TRITON_BLOCK_SIZE=16,
    )
    reference_indices, reference_indptr = build_ragged_indices_from_dense(
        dense.reshape(1, -1), reference_lens
    )

    direct_lens = torch.full_like(reference_lens, -77)
    direct_indices_buffer = torch.full(
        (window_size,), -77, dtype=torch.int32, device="cuda"
    )
    direct_indptr_buffer = torch.full((2,), -77, dtype=torch.int32, device="cuda")
    direct_indices, direct_indptr = build_swa_ragged_indices_direct(
        num_decode_tokens=1,
        window_size=window_size,
        index_width=window_size,
        non_causal=False,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        token_to_req_indices=token_to_req,
        is_valid_token=is_valid,
        block_table=block_table,
        block_size=block_size,
        swa_lens=direct_lens,
        ragged_indices_buffer=direct_indices_buffer,
        ragged_indptr_buffer=direct_indptr_buffer,
    )

    torch.testing.assert_close(direct_lens, reference_lens)
    torch.testing.assert_close(direct_indptr, reference_indptr)
    nnz = int(reference_indptr[-1].item())
    torch.testing.assert_close(direct_indices[:nnz], reference_indices[:nnz])


@torch.inference_mode()
def test_direct_c128_ragged_matches_dense_pack() -> None:
    from vllm.models.deepseek_v4.amd.rocm import (
        build_c128_decode_ragged_direct,
    )
    from vllm.models.deepseek_v4.sparse_mla import build_c128a_topk_metadata
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        build_ragged_indices_from_dense,
    )

    num_tokens = 4
    compress_ratio = 128
    active_width = 128
    max_width = 128
    block_size = 2
    positions = torch.tensor([127, 255, 383, 639], device="cuda")
    token_to_req = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device="cuda")
    block_table = torch.tensor(
        [[10, 11, 12], [20, 21, 22]], dtype=torch.int32, device="cuda"
    )
    slot_mapping = torch.tensor([0, -1, 0, 0], dtype=torch.int64, device="cuda")

    dense_buffer = torch.full(
        (num_tokens, max_width), -99, dtype=torch.int32, device="cuda"
    )
    reference_lens_buffer = torch.full(
        (num_tokens,), -99, dtype=torch.int32, device="cuda"
    )
    prefill_buffer = torch.empty_like(dense_buffer)
    dense, reference_lens, _ = build_c128a_topk_metadata(
        positions,
        compress_ratio,
        num_tokens,
        token_to_req,
        block_table,
        block_size,
        slot_mapping,
        dense_buffer,
        reference_lens_buffer,
        prefill_buffer,
        max_compressed_tokens=active_width,
    )
    reference_indices, reference_indptr = build_ragged_indices_from_dense(
        dense, reference_lens
    )

    direct_lens_buffer = torch.full_like(reference_lens_buffer, -77)
    direct_indices_buffer = torch.full(
        (num_tokens * max_width,), -77, dtype=torch.int32, device="cuda"
    )
    direct_indptr_buffer = torch.full(
        (num_tokens + 1,), -77, dtype=torch.int32, device="cuda"
    )
    direct_indices, direct_indptr, direct_lens = build_c128_decode_ragged_direct(
        positions=positions,
        token_to_req_indices=token_to_req,
        block_table=block_table,
        block_size=block_size,
        slot_mapping=slot_mapping,
        compress_ratio=compress_ratio,
        active_topk_width=active_width,
        max_topk_width=max_width,
        decode_lens_buffer=direct_lens_buffer,
        ragged_indices_buffer=direct_indices_buffer,
        ragged_indptr_buffer=direct_indptr_buffer,
    )

    torch.testing.assert_close(direct_lens, reference_lens)
    torch.testing.assert_close(direct_indptr, reference_indptr)
    nnz = int(reference_indptr[-1].item())
    torch.testing.assert_close(direct_indices[:nnz], reference_indices[:nnz])
    assert direct_indices.data_ptr() == direct_indices_buffer.data_ptr()
    assert direct_indptr.data_ptr() == direct_indptr_buffer.data_ptr()


@torch.inference_mode()
def test_sparse_prefill_direct_kv_matches_gathered() -> None:
    """Paged-cache KV reads must match the gathered rows, window included.

    The direct reader tags every support entry with the cache it belongs to
    (compressed context or sliding window) and unpacks the packed row in-kernel,
    so nothing is gathered. Values have to match the gathered path because both
    run the same UE8M0 dequant.
    """
    from vllm.models.deepseek_v4.common.ops import quantize_and_insert_k_cache
    from vllm.models.deepseek_v4_1.common.ops import dequantize_and_gather_k_cache
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        SPARSE_DIRECT_KV_REQ_STRIDE,
        SPARSE_DIRECT_KV_SWA_TAG_BASE,
        SPARSE_DIRECT_KV_TAG_BASE,
        rocm_sparse_attn_prefill,
    )

    torch.manual_seed(11)
    head_dim, nope_dim, rope_dim = 512, 448, 64
    block_size, num_blocks = 64, 4
    num_tokens = block_size * num_blocks
    num_queries, num_heads = 5, 4
    swa_len = 32

    q = (
        torch.randn(
            num_queries, num_heads, head_dim, dtype=torch.bfloat16, device="cuda"
        )
        * 0.125
    )

    def make_cache(seed: int) -> torch.Tensor:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        values = torch.randn(
            num_tokens, head_dim, dtype=torch.bfloat16, device="cuda", generator=gen
        )
        cache = torch.zeros(
            num_blocks, block_size, 584, dtype=torch.uint8, device="cuda"
        )
        slots = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
        quantize_and_insert_k_cache(values, cache, slots, block_size)
        return cache

    compressed_cache = make_cache(11)
    swa_cache = make_cache(23)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda")[None, :]
    lens = torch.tensor([num_tokens], dtype=torch.int32, device="cuda")
    swa_lens = torch.tensor([swa_len], dtype=torch.int32, device="cuda")

    # Reference layout: [compressed rows | sliding-window rows] gathered to bf16.
    gathered = torch.zeros(
        1, num_tokens + swa_len, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    dequantize_and_gather_k_cache(
        gathered[:, :num_tokens],
        compressed_cache,
        lens,
        None,
        block_table,
        block_size,
        0,
        use_fnuz=False,
    )
    dequantize_and_gather_k_cache(
        gathered[:, num_tokens:],
        swa_cache,
        swa_lens,
        None,
        block_table,
        block_size,
        0,
        use_fnuz=False,
    )
    kv_rows = gathered.transpose(0, 1).contiguous()

    compressed_positions = torch.arange(num_tokens, dtype=torch.int32, device="cuda")
    swa_positions = torch.arange(swa_len, dtype=torch.int32, device="cuda")
    dense = torch.cat(
        [
            compressed_positions.repeat(num_queries, 1),
            (num_tokens + swa_positions).repeat(num_queries, 1),
        ],
        dim=1,
    )
    tagged = torch.cat(
        [
            (compressed_positions + SPARSE_DIRECT_KV_TAG_BASE).repeat(num_queries, 1),
            (swa_positions + SPARSE_DIRECT_KV_SWA_TAG_BASE).repeat(num_queries, 1),
        ],
        dim=1,
    )
    widths = torch.full(
        (num_queries,), dense.shape[1], dtype=torch.int32, device="cuda"
    )

    def run(
        indices: torch.Tensor,
        kv: torch.Tensor,
        topk_length: torch.Tensor,
        direct: bool,
    ) -> torch.Tensor:
        output = torch.empty(
            num_queries, num_heads, head_dim, dtype=torch.bfloat16, device="cuda"
        )
        direct_kv = None
        if direct:
            direct_kv = (
                compressed_cache,
                block_table,
                lens,
                block_size,
                SPARSE_DIRECT_KV_TAG_BASE,
                SPARSE_DIRECT_KV_REQ_STRIDE,
                (
                    swa_cache,
                    block_table,
                    swa_lens,
                    block_size,
                    SPARSE_DIRECT_KV_SWA_TAG_BASE,
                ),
            )
        rocm_sparse_attn_prefill(
            q=q,
            kv=kv,
            indices=indices,
            topk_length=topk_length,
            scale=head_dim**-0.5,
            head_dim=head_dim,
            nope_head_dim=nope_dim,
            rope_head_dim=rope_dim,
            attn_sink=None,
            output=output,
            direct_kv=direct_kv,
        )
        return output

    # A single-key support normalizes to that key's value, so this isolates the
    # packed-row addressing and dequant, for both sources. It has to be exact.
    one = torch.ones((num_queries,), dtype=torch.int32, device="cuda")
    for position in (0, 1, block_size - 1, block_size, num_tokens - 1):
        idx = torch.full((num_queries, 1), position, dtype=torch.int32, device="cuda")
        want = run(idx, kv_rows, one, direct=False)
        got = run(idx + SPARSE_DIRECT_KV_TAG_BASE, kv_rows, one, direct=True)
        torch.testing.assert_close(got, want, atol=0.0, rtol=0.0)
    for position in (0, 7, swa_len - 1):
        idx = torch.full(
            (num_queries, 1), num_tokens + position, dtype=torch.int32, device="cuda"
        )
        want = run(idx, kv_rows, one, direct=False)
        got = run(
            torch.full(
                (num_queries, 1),
                position + SPARSE_DIRECT_KV_SWA_TAG_BASE,
                dtype=torch.int32,
                device="cuda",
            ),
            kv_rows,
            one,
            direct=True,
        )
        torch.testing.assert_close(got, want, atol=0.0, rtol=0.0)

    # With the whole support the two paths still agree, up to the online-softmax
    # reassociation a different KV tile width implies.
    expected = run(dense, kv_rows, widths, direct=False)
    actual = run(tagged, kv_rows, widths, direct=True)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    assert torch.isfinite(actual).all()


@torch.inference_mode()
def test_split_k_partials_reduce_to_the_single_pass_output() -> None:
    """Split-K prefill: per-split (m, l, acc) must reduce to the one-pass result.

    The split path exists so a prefill can walk its support in bounded blocks
    (bounded gather workspace); it is only usable if combining the raw partial
    states reproduces the single-pass output, including the attention sink.
    """
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        rocm_sparse_attn_prefill,
        rocm_sparse_attn_reduce_partials,
    )

    torch.manual_seed(5)
    head_dim, nope_dim, rope_dim = 512, 448, 64
    num_queries, num_heads, num_kv = 6, 4, 192
    scale = head_dim**-0.5

    q = (
        torch.randn(
            num_queries, num_heads, head_dim, dtype=torch.bfloat16, device="cuda"
        )
        * 0.125
    )
    kv = torch.randn(num_kv, head_dim, dtype=torch.bfloat16, device="cuda") * 0.25
    kv_arg = kv.unsqueeze(1).contiguous()
    sink = torch.randn(num_heads, dtype=torch.float32, device="cuda")

    # Every query attends to the whole context, so each split is dense.
    indices = torch.arange(num_kv, dtype=torch.int32, device="cuda")[None, :].repeat(
        num_queries, 1
    )
    lengths = torch.full((num_queries,), num_kv, dtype=torch.int32, device="cuda")

    def run(indices_arg, partial_out=None, attn_sink=None):
        out = torch.empty(
            num_queries, num_heads, head_dim, dtype=torch.bfloat16, device="cuda"
        )
        rocm_sparse_attn_prefill(
            q=q,
            kv=kv_arg,
            indices=indices_arg,
            topk_length=lengths,
            scale=scale,
            head_dim=head_dim,
            nope_head_dim=nope_dim,
            rope_head_dim=rope_dim,
            attn_sink=attn_sink,
            output=out,
            partial_out=partial_out,
        )
        return out

    expected = run(indices, attn_sink=sink)

    # Two overlapping-in-coverage splits, plus the sink applied only by the reduce.
    splits = (slice(0, 96), slice(96, num_kv))
    part_m = torch.empty(
        num_queries, len(splits), num_heads, dtype=torch.float32, device="cuda"
    )
    part_l = torch.empty_like(part_m)
    part_acc = torch.empty(
        num_queries,
        len(splits),
        num_heads,
        head_dim,
        dtype=torch.float32,
        device="cuda",
    )
    for split_id, span in enumerate(splits):
        split_indices = torch.full_like(indices, -1)
        split_indices[:, : span.stop - span.start] = indices[:, span]
        run(
            split_indices,
            partial_out=(part_m, part_l, part_acc, split_id),
        )

    actual = torch.empty_like(expected)
    rocm_sparse_attn_reduce_partials(
        part_m, part_l, part_acc, sink, actual, num_heads, head_dim
    )
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    assert torch.isfinite(actual).all()
