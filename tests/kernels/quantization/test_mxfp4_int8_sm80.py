# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts.mxfp4_int8_sm80 import (
    Mxfp4Int8Sm80Experts,
    make_mxfp4_int8_sm80_decode_metadata,
    select_mxfp4_int8_sm80_tile,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    _get_priority_backends,
    map_mxfp4_backend,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_int8_sm80 import (
    compute_mxfp4_int8_remap_stats,
    quantize_per_token_int8,
    remap_mxfp4_weight_for_int8,
)


def test_per_token_int8_uses_rne_and_avoids_minus_128() -> None:
    values = torch.tensor(
        [[0.0, 0.5, 1.5, -0.5, -1.5, 127.0, -127.0]],
        dtype=torch.float32,
    )
    quantized, scale = quantize_per_token_int8(values)

    assert scale.tolist() == [1.0]
    assert quantized.tolist() == [[0, 0, 2, 0, -2, 127, -127]]
    assert int(quantized.min()) >= -127


def test_constant_exponent_repack_is_exact() -> None:
    generator = torch.Generator().manual_seed(7)
    experts, out_features, in_features = 2, 11, 64
    packed = torch.randint(
        0,
        256,
        (experts, out_features, in_features // 2),
        dtype=torch.uint8,
        generator=generator,
    )
    scale = torch.full(
        (experts, out_features, in_features // 32),
        127,
        dtype=torch.uint8,
    )

    weight, shift2, channel_scale = remap_mxfp4_weight_for_int8(
        packed,
        scale,
        headroom_bits=3,
        use_triton=False,
    )
    stats = compute_mxfp4_int8_remap_stats(
        packed,
        scale,
        headroom_bits=3,
    )

    assert weight.shape == (experts, 2, 2, 128)
    assert shift2.shape == (experts, 1, 2, 8)
    assert channel_scale.shape == (experts, out_features)
    assert stats["overflow_count"] == 0
    assert stats["exact_rate_all"] == 1.0


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 8,
    reason="requires an SM80-family CUDA device",
)
def test_triton_repack_can_reuse_source_storage() -> None:
    generator = torch.Generator(device="cuda").manual_seed(17)
    packed = torch.randint(
        0,
        256,
        (19, 64, 64),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    scale = torch.randint(
        120,
        136,
        (19, 64, 4),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    expected = remap_mxfp4_weight_for_int8(packed, scale)
    source = packed.clone()
    source_ptr = source.data_ptr()

    actual = remap_mxfp4_weight_for_int8(
        source,
        scale,
        reuse_weight_storage=True,
    )

    assert actual[0].data_ptr() == source_ptr
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 8,
    reason="requires an SM80-family CUDA device",
)
def test_dspark_topk3_cuda_dispatch() -> None:
    hidden_size = 5120
    intermediate_size = 2304
    topk = 3
    num_experts = 1
    device = torch.device("cuda")
    topk_ids = torch.zeros((1, topk), dtype=torch.int32, device=device)
    topk_weights = torch.tensor([[0.25, 0.25, 0.5]], dtype=torch.float32, device=device)
    sorted_ids, expert_ids, num_post_pad = moe_align_block_size(
        topk_ids, 16, num_experts
    )

    a13_q = torch.zeros((1, hidden_size), dtype=torch.int8, device=device)
    a13_scale = torch.ones((1,), dtype=torch.float32, device=device)
    w13 = torch.zeros(
        (num_experts, hidden_size // 32, 2 * intermediate_size // 8, 128),
        dtype=torch.uint8,
        device=device,
    )
    w13_shift = torch.zeros(
        (num_experts, hidden_size // 128, 2 * intermediate_size // 8, 8),
        dtype=torch.uint8,
        device=device,
    )
    w13_scale = torch.ones(
        (num_experts, 2 * intermediate_size), dtype=torch.float32, device=device
    )
    gate_up = torch.empty(
        (topk, 2 * intermediate_size), dtype=torch.bfloat16, device=device
    )
    torch.ops._moe_C.dsv4_mxfp4_int8_sm80_moe_gemm(
        a13_q,
        a13_scale,
        w13,
        w13_shift,
        w13_scale,
        gate_up,
        gate_up,
        topk_weights,
        sorted_ids,
        expert_ids,
        num_post_pad,
        topk,
        16,
        128,
        False,
    )

    a2_q = torch.zeros((topk, intermediate_size), dtype=torch.int8, device=device)
    a2_scale = torch.ones((topk,), dtype=torch.float32, device=device)
    w2 = torch.zeros(
        (num_experts, intermediate_size // 32, hidden_size // 8, 128),
        dtype=torch.uint8,
        device=device,
    )
    w2_shift = torch.zeros(
        (num_experts, intermediate_size // 128, hidden_size // 8, 8),
        dtype=torch.uint8,
        device=device,
    )
    w2_scale = torch.ones(
        (num_experts, hidden_size), dtype=torch.float32, device=device
    )
    output = torch.empty((1, hidden_size), dtype=torch.bfloat16, device=device)
    routed = torch.empty((topk, hidden_size), dtype=torch.bfloat16, device=device)
    torch.ops._moe_C.dsv4_mxfp4_int8_sm80_moe_gemm(
        a2_q,
        a2_scale,
        w2,
        w2_shift,
        w2_scale,
        output,
        routed,
        topk_weights,
        sorted_ids,
        expert_ids,
        num_post_pad,
        topk,
        16,
        128,
        True,
    )
    routed.copy_(
        torch.tensor([1.0, 2.0, 4.0], dtype=torch.bfloat16, device=device)[:, None]
    )
    torch.ops._moe_C.dsv4_mxfp4_int8_sm80_moe_reduce(routed, topk_weights, output, topk)

    torch.testing.assert_close(output, torch.full_like(output, 2.75), rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 8,
    reason="requires an SM80-family CUDA device",
)
def test_single_token_persistent_metadata_matches_reference() -> None:
    topk_ids = torch.tensor(
        [[311, 17, 205, 92, 4, 378]], dtype=torch.int32, device="cuda"
    )

    reference_sorted, reference_experts, reference_post_pad = moe_align_block_size(
        topk_ids, 16, 384
    )
    actual_sorted, actual_post_pad = make_mxfp4_int8_sm80_decode_metadata(
        topk_ids.device
    )
    actual_experts = topk_ids.view(-1)
    torch.accelerator.synchronize()

    assert actual_post_pad.item() == reference_post_pad.item() == 96

    def active_routes(
        sorted_ids: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> list[tuple[int, int]]:
        routes: list[tuple[int, int]] = []
        for block, expert_id in enumerate(expert_ids.cpu().tolist()):
            block_routes = sorted_ids[block * 16 : (block + 1) * 16]
            routes.extend(
                (expert_id, route)
                for route in block_routes.cpu().tolist()
                if route < topk_ids.numel()
            )
        return sorted(routes)

    assert active_routes(actual_sorted, actual_experts) == active_routes(
        reference_sorted, reference_experts
    )


def test_static_tile_table_respects_sparse_expert_occupancy() -> None:
    assert select_mxfp4_int8_sm80_tile(1, 6, 256) == (16, 128)
    assert select_mxfp4_int8_sm80_tile(256, 6, 256) == (16, 128)
    assert select_mxfp4_int8_sm80_tile(512, 6, 256) == (16, 128)
    assert select_mxfp4_int8_sm80_tile(1024, 6, 256) == (16, 128)
    assert select_mxfp4_int8_sm80_tile(2048, 6, 256) == (64, 128)
    assert select_mxfp4_int8_sm80_tile(8192, 6, 256) == (128, 128)


def test_workspace_keeps_routed_slots_out_of_output_alias() -> None:
    workspace13, workspace2, output = Mxfp4Int8Sm80Experts.workspace_shapes(
        None,  # type: ignore[arg-type]
        M=2,
        N=5120,
        K=5120,
        topk=6,
        global_num_experts=256,
        local_num_experts=256,
        expert_tokens_meta=None,
        activation=None,  # type: ignore[arg-type]
    )

    assert workspace13 == (12, 6912)  # contiguous W13 + activation scratch
    assert workspace2 == (12, 5120)  # routed W2 slots; must not alias output
    assert output == (2, 5120)


def test_backend_requires_explicit_selection() -> None:
    backend = Mxfp4MoeBackend.MXFP4_INT8_SM80
    assert map_mxfp4_backend("mxfp4_int8_sm80") == [backend]
    assert backend not in _get_priority_backends()
