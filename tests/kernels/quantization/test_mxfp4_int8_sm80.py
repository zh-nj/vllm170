# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.model_executor.layers.fused_moe.experts.mxfp4_int8_sm80 import (
    Mxfp4Int8Sm80Experts,
    select_mxfp4_int8_sm80_tile,
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


def test_static_tile_table_respects_sparse_expert_occupancy() -> None:
    assert select_mxfp4_int8_sm80_tile(1, 6, 256) == (16, 128)
    assert select_mxfp4_int8_sm80_tile(1024, 6, 256) == (16, 64)
    assert select_mxfp4_int8_sm80_tile(2048, 6, 256) == (64, 128)
    assert select_mxfp4_int8_sm80_tile(8192, 6, 256) == (128, 128)


def test_workspace_keeps_routed_slots_out_of_output_alias() -> None:
    workspace13, workspace2, output = Mxfp4Int8Sm80Experts.workspace_shapes(
        None,  # type: ignore[arg-type]
        M=2,
        N=4096,
        K=4096,
        topk=6,
        global_num_experts=256,
        local_num_experts=256,
        expert_tokens_meta=None,
        activation=None,  # type: ignore[arg-type]
    )

    assert workspace13 == (12, 2048)  # activation; dead before final reduce
    assert workspace2 == (12, 4096)  # routed W2 slots; must not alias output
    assert output == (2, 4096)


def test_backend_requires_explicit_selection() -> None:
    backend = Mxfp4MoeBackend.MXFP4_INT8_SM80
    assert map_mxfp4_backend("mxfp4_int8_sm80") == [backend]
    assert backend not in _get_priority_backends()
