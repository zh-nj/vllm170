# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(80),
    reason="DeepSeek-V4 Triton sparse prefill tuning targets SM8x",
)


def test_direct_swa_ragged_controls_dense_decode_buffer(monkeypatch) -> None:
    from vllm.models.deepseek_v4.amd.rocm import (
        DeepseekV4ROCMAiterMLASparseMetadataBuilder,
        DeepseekV4ROCMAiterSparseSWAMetadataBuilder,
    )

    builder = object.__new__(DeepseekV4ROCMAiterSparseSWAMetadataBuilder)
    monkeypatch.setenv("VLLM_DEEPSEEK_V4_DIRECT_RAGGED_SWA", "0")
    assert builder._requires_dense_decode_swa_indices()

    monkeypatch.setenv("VLLM_DEEPSEEK_V4_DIRECT_RAGGED_SWA", "1")
    assert not builder._requires_dense_decode_swa_indices()

    c128_builder = object.__new__(DeepseekV4ROCMAiterMLASparseMetadataBuilder)
    monkeypatch.setenv("VLLM_DEEPSEEK_V4_DIRECT_RAGGED_C128", "0")
    assert c128_builder._requires_dense_c128a_decode_indices()

    monkeypatch.setenv("VLLM_DEEPSEEK_V4_DIRECT_RAGGED_C128", "1")
    assert not c128_builder._requires_dense_c128a_decode_indices()


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
@pytest.mark.parametrize("num_queries", [1, 2, 4, 8])
def test_sparse_decode_split_and_caller_output_match(
    monkeypatch, num_queries: int
) -> None:
    from vllm.models.deepseek_v4.common.ops import quantize_and_insert_k_cache
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
    monkeypatch.setenv("VLLM_DEEPSEEK_V4_SPARSE_DECODE_CALLER_OUTPUT", "1")
    output = torch.full_like(q, float("nan"))
    swa_indices = torch.zeros(num_queries, 1, 1, dtype=torch.int32, device="cuda")
    main_lens = torch.full((num_queries,), main_len, dtype=torch.int32, device="cuda")
    extra_lens = torch.full((num_queries,), extra_len, dtype=torch.int32, device="cuda")

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


@pytest.mark.parametrize("non_causal", [False, True])
@torch.inference_mode()
def test_direct_swa_ragged_matches_dense_pack(non_causal: bool) -> None:
    from vllm.models.deepseek_v4.amd.rocm import (
        build_swa_ragged_indices_direct,
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
    reference_indices, reference_indptr = build_ragged_indices_from_dense(
        dense.reshape(num_tokens, -1), reference_lens
    )

    direct_lens = torch.full_like(reference_lens, -77)
    direct_indices_buffer = torch.full(
        (num_tokens * index_width,), -77, dtype=torch.int32, device="cuda"
    )
    direct_indptr_buffer = torch.full(
        (num_tokens + 1,), -77, dtype=torch.int32, device="cuda"
    )
    direct_indices, direct_indptr = build_swa_ragged_indices_direct(
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
    active_width = 8
    max_width = 8
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
