# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch


def test_decode_topk_cache_isolated_per_ubatch(monkeypatch) -> None:
    from vllm.models.deepseek_v4_1 import attention as attention_mod

    ubatch_id = 0
    monkeypatch.setattr(attention_mod, "dbo_current_ubatch_id", lambda: ubatch_id)
    cache = attention_mod.DeepseekV4DecodeTopKCache()
    value = (torch.tensor(1),)

    cache.put(("ragged", 128), value)
    assert cache.get(("ragged", 128)) is value

    ubatch_id = 1
    assert cache.get(("ragged", 128)) is None
    cache.put(("ragged", 128), (torch.tensor(2),))
    cache.invalidate()
    assert cache.get(("ragged", 128)) is None

    ubatch_id = 0
    assert cache.get(("ragged", 128)) is value


def test_index_source_invalidates_decode_topk_cache() -> None:
    from vllm.models.deepseek_v4_1.attention import (
        DeepseekV4Attention,
        DeepseekV4DecodeTopKCache,
    )

    cache = DeepseekV4DecodeTopKCache()
    cache.put(("ragged", 128), (torch.tensor(1),))
    calls = 0

    def forward_mqa(*args) -> None:
        nonlocal calls
        calls += 1

    layer = SimpleNamespace(
        indexer=object(),
        decode_topk_cache=cache,
        forward_mqa=forward_mqa,
    )
    tensor = torch.empty(1)
    DeepseekV4Attention._sparse_indexer_and_attn(
        layer,
        tensor,
        None,
        None,
        None,
        tensor,
        tensor,
        tensor,
        tensor,
    )

    assert calls == 1
    assert cache.get(("ragged", 128)) is None


def test_rocm_decode_reuses_converted_topk(monkeypatch) -> None:
    from vllm.models.deepseek_v4_1.amd import rocm as rocm_mod
    from vllm.models.deepseek_v4_1.attention import DeepseekV4DecodeTopKCache

    monkeypatch.setitem(
        rocm_mod.envs.environment_variables,
        "VLLM_DEEPSEEK_V4_DECODE_TOPK_CACHE",
        lambda: True,
    )
    converted = (
        torch.tensor([7], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
    )
    convert_calls = 0
    attention_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def convert(*args):
        nonlocal convert_calls
        convert_calls += 1
        return converted

    def attention(**kwargs) -> None:
        attention_calls.append(
            (
                kwargs["topk_ragged_indices"],
                kwargs["topk_ragged_indptr"],
                kwargs["topk_lens"],
            )
        )

    monkeypatch.setattr(
        rocm_mod, "compute_global_topk_ragged_indices_and_indptr", convert
    )
    monkeypatch.setattr(rocm_mod, "rocm_sparse_attn_decode", attention)

    cache = DeepseekV4DecodeTopKCache()
    layer = SimpleNamespace(
        compress_ratio=1,
        topk_indices_buffer=torch.zeros((1, 4), dtype=torch.int32),
        decode_topk_cache=cache,
        swa_cache_layer=SimpleNamespace(kv_cache=torch.empty(0), window_size=128),
        attn_sink=None,
        scale=1.0,
        head_dim=512,
        nope_head_dim=448,
        rope_head_dim=64,
        kv_cache_dtype="fp8_ds_mla",
        _has_kv_transfer=False,
    )
    swa_metadata = SimpleNamespace(
        num_decodes=1,
        num_decode_tokens=1,
        is_valid_token=torch.tensor([True]),
        token_to_req_indices=torch.tensor([0], dtype=torch.int32),
        decode_swa_indices=torch.empty(0, dtype=torch.int32),
        decode_swa_lens=torch.tensor([0], dtype=torch.int32),
        decode_swa_ragged_indices=torch.empty(0, dtype=torch.int32),
        decode_swa_ragged_indptr=torch.tensor([0, 0], dtype=torch.int32),
        decode_paged_swa=False,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
    )
    attn_metadata = SimpleNamespace(
        block_size=128,
        block_table=torch.zeros((1, 1), dtype=torch.int32),
    )
    tensor = torch.empty(1)

    for _ in range(2):
        rocm_mod.DeepseekV41ROCMAiterMLAAttention._forward_decode(
            layer,
            tensor,
            torch.empty(0),
            swa_metadata,
            attn_metadata,
            False,
            tensor,
        )

    assert convert_calls == 1
    assert attention_calls == [converted, converted]

    cache.invalidate()
    rocm_mod.DeepseekV41ROCMAiterMLAAttention._forward_decode(
        layer,
        tensor,
        torch.empty(0),
        swa_metadata,
        attn_metadata,
        False,
        tensor,
    )
    assert convert_calls == 2
