# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import json
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.mxfp4_int8_sm80 import (
    Mxfp4Int8Sm80Experts,
    dsv41_sm80_hybrid_prefill_enabled,
    dsv41_sm80_tilelang_enabled,
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


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [
        (1, (16, 128)),
        (256, (16, 128)),
        (512, (16, 128)),
        # ceil(1024 * 6 / 256) = 24 rows per expert, so the smallest candidate
        # that covers them is block_m 64. The previous expectation of (16, 128)
        # described the retired _TILE_TABLE lookup, not the current rule.
        (1024, (64, 128)),
        (2048, (64, 128)),
        (8192, (128, 128)),
    ],
)
def test_static_tile_table_respects_sparse_expert_occupancy(
    num_tokens: int, expected: tuple[int, int]
) -> None:
    """block_m is the smallest candidate covering ceil(M * topk / E) rows."""
    topk, num_experts = 6, 256
    average = max(1, -(-num_tokens * topk // num_experts))
    assert select_mxfp4_int8_sm80_tile(num_tokens, topk, num_experts) == expected
    assert expected[0] >= min(average, 128)
    if average <= 16:
        assert expected[0] == 16
    elif average <= 64:
        assert expected[0] == 64
    else:
        assert expected[0] == 128


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


# ---------------------------------------------------------------------------
# Packed-weight format anchors
#
# These tests carry their own scalar encoder/decoder written from the documented
# layout instead of importing the benchmark harness, so a mistake shared with
# the harness cannot hide here. The hand-built byte fixture below is the only
# anchor; everything else is cross-validated against it.
#
# Note the two structural facts the layout forces on any writer:
#   * a byte carries a *pair* of k values (low nibble = even k of the pair);
#   * a 2-bit shift field covers a whole (n, K/32) block, so every k inside one
#     block shares a shift. Writing a per-element shift silently keeps the last
#     value written for that block.
# ---------------------------------------------------------------------------

_BYTE_PERMUTATION = (0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15)
_MAG_X2 = (0, 1, 2, 3, 4, 6, 8, 12)


def _lane_for(n: int, k32: int) -> tuple[int, int, bool]:
    """Return (lane, byte index, is_high_nibble) holding k-pair column ``k32``."""
    _, n_in_group = divmod(n, 8)
    for thread_in_group in range(4):
        for j in range(4):
            k_pair = _BYTE_PERMUTATION[thread_in_group * 4 + j]
            if 2 * k_pair == k32:
                return n_in_group * 4 + thread_in_group, j, False
            if 2 * k_pair + 1 == k32:
                return n_in_group * 4 + thread_in_group, j, True
    raise AssertionError(f"k32={k32} is not covered by the byte permutation")


def _decode_element(
    b_mxfp4: torch.Tensor, b_shift2: torch.Tensor, n: int, k: int
) -> int:
    """Decode one INT8 weight value from a single expert's packed bytes."""
    kb, k32 = divmod(k, 32)
    n_group, n_in_group = divmod(n, 8)
    lane, byte_index, high = _lane_for(n, k32)
    byte = int(b_mxfp4[kb, n_group, lane * 4 + byte_index].item())
    nibble = (byte >> 4) & 0x0F if high else byte & 0x0F
    field = int(b_shift2[kb // 4, n_group, n_in_group].item())
    shift = (field >> (2 * (kb % 4))) & 0x03
    magnitude = _MAG_X2[nibble & 0x07]
    signed = -magnitude if nibble & 0x08 else magnitude
    return signed << shift


def _encode_element(
    b_mxfp4: torch.Tensor,
    b_shift2: torch.Tensor,
    n: int,
    k: int,
    code: int,
    shift: int,
) -> None:
    """Write one packed code plus its block shift; the inverse of the layout."""
    kb, k32 = divmod(k, 32)
    n_group, n_in_group = divmod(n, 8)
    lane, byte_index, high = _lane_for(n, k32)
    offset = lane * 4 + byte_index
    current = int(b_mxfp4[kb, n_group, offset].item())
    current = (
        (current & 0x0F) | ((code & 0x0F) << 4)
        if high
        else (current & 0xF0) | (code & 0x0F)
    )
    b_mxfp4[kb, n_group, offset] = current
    field = 2 * (kb % 4)
    packed = int(b_shift2[kb // 4, n_group, n_in_group].item())
    packed = (packed & ~(0x03 << field)) | ((shift & 0x03) << field)
    b_shift2[kb // 4, n_group, n_in_group] = packed


# Hand-built bytes for K=128, N=8, one expert. Lanes 0..3 all serve n=0 through
# four different byte positions, so the permutation is exercised; lane 4 is the
# only writer for n=1 and stores code 0x8 (negative zero) in every K block.
_HAND_LANES = {
    0: (0x01, 0x12, 0x00, 0x8F),
    1: (0x23, 0x00, 0xF4, 0x00),
    2: (0x45, 0x00, 0x00, 0x00),
    3: (0x06, 0x00, 0x00, 0x00),
    4: (0x08, 0x00, 0x00, 0x00),
}
# n=0 gets shifts (0, 1, 2, 3) across the four K blocks; n=1 gets (3, 3, 0, 0).
_HAND_SHIFT_FIELDS = {0: 0b11_10_01_00, 1: 0b00_00_11_11}
# (n, k, value) derived by hand from the bytes above.
_HAND_EXPECTED = (
    (0, 0, 1),
    (0, 1, 0),
    (0, 2, 2),
    (0, 4, 3),
    (0, 8, 6),
    (0, 12, 8),
    (0, 18, -12),
    (0, 19, 0),
    (0, 32, 2),
    (0, 36, 6),
    (0, 50, -24),
    (0, 64, 4),
    (0, 70, 0),
    (0, 72, 24),
    (0, 76, 32),
    (0, 82, -48),
    (0, 96, 8),
    (0, 100, 24),
    (0, 114, -96),
    (1, 0, 0),
    (1, 32, 0),
    (1, 64, 0),
    (1, 96, 0),
    (1, 1, 0),
)


def _hand_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    b_mxfp4 = torch.zeros((4, 1, 128), dtype=torch.uint8)
    for lane, four_bytes in _HAND_LANES.items():
        for kb in range(4):
            for j, value in enumerate(four_bytes):
                b_mxfp4[kb, 0, lane * 4 + j] = value
    b_shift2 = torch.zeros((1, 1, 8), dtype=torch.uint8)
    for n_in_group, packed in _HAND_SHIFT_FIELDS.items():
        b_shift2[0, 0, n_in_group] = packed
    return b_mxfp4, b_shift2


def test_hand_built_packed_bytes_decode_to_expected_values() -> None:
    """Guards the packed layout: lane order, nibble order, sign bit and shift."""
    b_mxfp4, b_shift2 = _hand_fixture()
    actual = {
        (n, k): _decode_element(b_mxfp4, b_shift2, n, k) for n, k, _ in _HAND_EXPECTED
    }
    expected = {(n, k): value for n, k, value in _HAND_EXPECTED}
    assert actual == expected
    assert actual[(0, 19)] == 0  # signed zero is a real zero
    assert actual[(0, 114)] == -96  # largest representable magnitude, shift 3
    assert min(actual.values()) == -96


@pytest.mark.parametrize("out_features", [7, 8, 9, 11])
@pytest.mark.parametrize("in_features", [32, 64, 128])
def test_pack_round_trip_covers_n_and_k_tails(
    out_features: int, in_features: int
) -> None:
    """Guards N tail groups and K tail blocks in the pack layer.

    The pack layer accepts shapes the fixed-shape CUDA GEMM does not; this only
    exercises the packer, never the kernel.
    """
    from vllm.model_executor.layers.quantization.utils.mxfp4_int8_sm80 import (
        _pack_codes_coalesced,
        _pack_shift2,
    )

    generator = torch.Generator().manual_seed(out_features * 100 + in_features)
    codes = torch.randint(
        0, 16, (out_features, in_features), dtype=torch.uint8, generator=generator
    )
    shifts = torch.randint(
        0,
        4,
        (out_features, in_features // 32),
        dtype=torch.uint8,
        generator=generator,
    )
    b_mxfp4 = _pack_codes_coalesced(codes)
    b_shift2 = _pack_shift2(shifts)
    assert b_mxfp4.shape == (in_features // 32, (out_features + 7) // 8, 128)
    assert b_shift2.shape == ((in_features // 32 + 3) // 4, (out_features + 7) // 8, 8)

    for n in range(out_features):
        for kb in range(in_features // 32):
            for k in range(kb * 32, kb * 32 + 32):
                code = int(codes[n, k])
                magnitude = _MAG_X2[code & 0x07]
                signed = -magnitude if code & 0x08 else magnitude
                assert _decode_element(b_mxfp4, b_shift2, n, k) == (
                    signed << int(shifts[n, kb])
                )
    # Columns past the real N are padding and must decode to zero.
    for n in range(out_features, ((out_features + 7) // 8) * 8):
        for k in range(0, in_features, 7):
            assert _decode_element(b_mxfp4, b_shift2, n, k) == 0


def test_encode_and_decode_are_inverse_over_every_lane() -> None:
    """Guards the writer the GPU fixtures use against the reader above."""
    b_mxfp4 = torch.zeros((4, 1, 128), dtype=torch.uint8)
    b_shift2 = torch.zeros((1, 1, 8), dtype=torch.uint8)
    expected = {}
    for n in range(8):
        for kb in range(4):
            shift = (n + kb) % 4
            for k in range(kb * 32, kb * 32 + 32):
                code = (n * 7 + k) % 16
                _encode_element(b_mxfp4, b_shift2, n, k, code, shift)
                magnitude = _MAG_X2[code & 0x07]
                expected[(n, k)] = (-magnitude if code & 0x08 else magnitude) << shift
    actual = {
        (n, k): _decode_element(b_mxfp4, b_shift2, n, k)
        for n in range(8)
        for k in range(128)
    }
    assert actual == expected
    # Overwriting the same lane with a new code must not leak the old nibble.
    _encode_element(b_mxfp4, b_shift2, 0, 0, 0x0F, 0)
    assert _decode_element(b_mxfp4, b_shift2, 0, 0) == -12
    _encode_element(b_mxfp4, b_shift2, 0, 0, 0x00, 0)
    assert _decode_element(b_mxfp4, b_shift2, 0, 0) == 0
    # k=1 is the other nibble of the same byte and is an independent k, so
    # rewriting k=0 must leave it alone.
    assert _decode_element(b_mxfp4, b_shift2, 0, 1) == 1


def test_int32_accumulator_cannot_saturate() -> None:
    """The MMA saturates; the integer reference is only valid without overflow."""
    max_activation, max_weight = 127, 96
    for k in (5120, 2304):
        assert max_activation * max_weight * k < 2**31 - 1


# ---------------------------------------------------------------------------
# Non-zero grouped GEMM / routing / reduce tests
#
# The kernel hard-codes N and K, so a small-N fixture is impossible. Instead the
# packed weights are zero except at a handful of (n, k) positions, which keeps
# the expectation a hand-checkable scalar sum while still exercising the real
# tile, alignment and epilogue paths. Columns that were never encoded must come
# back as zero, which is what catches expert/column pollution.
# ---------------------------------------------------------------------------

_NEEDS_SM80 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 8,
    reason="requires an SM80-family CUDA device",
)
_W13 = (2 * 2304, 5120, False)
_W2 = (5120, 2304, True)
_W13_N_POSITIONS = (0, 1, 7, 8, 9, 2303, 2304, 4607)
_W13_K_POSITIONS = (0, 1, 2, 3, 31, 32, 33, 64, 95, 96, 127, 128, 160, 5119)
_W2_N_POSITIONS = (0, 1, 7, 8, 9, 5119)
_W2_K_POSITIONS = (0, 1, 2, 3, 31, 32, 33, 64, 96, 127, 128, 129, 160, 2303)
_SENTINEL = -12345.0


def _positions(n_positions, k_positions) -> dict[tuple[int, int], tuple[int, int]]:
    """Sparse (n, k) -> (code, shift) map covering signs, magnitudes and shifts.

    The shift must be constant inside each K/32 block because the format stores
    one 2-bit field per (n, block); ``(n // 8 + k // 32) % 4`` respects that.
    """
    return {
        (n, k): ((n + k) % 16, (n // 8 + k // 32) % 4)
        for n in n_positions
        for k in k_positions
    }


def _new_packed(num_experts: int, n: int, k: int, device: torch.device):
    return (
        torch.zeros(
            (num_experts, k // 32, (n + 7) // 8, 128), dtype=torch.uint8, device=device
        ),
        torch.zeros(
            (num_experts, (k // 32 + 3) // 4, (n + 7) // 8, 8),
            dtype=torch.uint8,
            device=device,
        ),
    )


def _fill_experts(b_mxfp4, b_shift2, positions, experts) -> None:
    for e in experts:
        for (n, k), (code, shift) in positions.items():
            _encode_element(b_mxfp4[e], b_shift2[e], n, k, code, shift)


def _expected_value(a_q, a_scale, channel_scale, positions, expert, row, n):
    """Hand-checkable scalar expectation, in the kernel's rounding order."""
    acc = 0
    for (position_n, k), (code, shift) in positions.items():
        if position_n != n:
            continue
        magnitude = _MAG_X2[code & 0x07]
        weight = (-magnitude if code & 0x08 else magnitude) << shift
        acc += int(a_q[row, k].item()) * weight
    value = torch.tensor(float(acc), dtype=torch.float32)
    value = value * float(a_scale[row].item())
    return (value * float(channel_scale[expert, n].item())).to(torch.bfloat16)


def _run_gemm(
    stage_shape,
    a_q,
    a_scale,
    b_mxfp4,
    b_shift2,
    channel_scale,
    routing_ids,
    block_m,
    block_n,
    extra_rows=0,
):
    """Invoke the production op; returns (routed_out_slots, token_out)."""
    n, k, source_rows_are_slots = stage_shape
    topk = routing_ids.shape[1]
    tokens = routing_ids.shape[0]
    num_routed = tokens * topk
    sorted_ids, expert_ids, num_post_pad = moe_align_block_size(
        routing_ids, block_m, b_mxfp4.shape[0]
    )
    routed_out = torch.full(
        (num_routed + extra_rows, n), _SENTINEL, dtype=torch.bfloat16, device=a_q.device
    )
    if source_rows_are_slots:
        out = torch.empty((tokens, n), dtype=torch.bfloat16, device=a_q.device)
    else:
        out = torch.empty((num_routed, n), dtype=torch.bfloat16, device=a_q.device)
    routing_weights = torch.full(
        (tokens, topk), 1.0 / topk, dtype=torch.float32, device=a_q.device
    )
    torch.ops._moe_C.dsv4_mxfp4_int8_sm80_moe_gemm(
        a_q,
        a_scale,
        b_mxfp4,
        b_shift2,
        channel_scale,
        out,
        routed_out,
        routing_weights,
        sorted_ids,
        expert_ids,
        num_post_pad,
        num_routed,
        block_m,
        block_n,
        source_rows_are_slots,
    )
    torch.accelerator.synchronize()
    return routed_out, out, sorted_ids, expert_ids, num_post_pad


def _check_all_slots(
    stage_shape, a_q, a_scale, channel_scale, positions, routing_ids, routed
):
    """Every encoded column of every slot must match, others must stay zero."""
    n, _, source_rows_are_slots = stage_shape
    topk = routing_ids.shape[1]
    flat = routing_ids.reshape(-1).tolist()
    encoded_columns = sorted({position_n for position_n, _ in positions})
    for slot, expert in enumerate(flat):
        row = slot if source_rows_are_slots else slot // topk
        for column in encoded_columns:
            assert (
                routed[slot, column].item()
                == _expected_value(
                    a_q, a_scale, channel_scale, positions, expert, row, column
                ).item()
            ), f"slot={slot} expert={expert} column={column}"
    mask = torch.ones(n, dtype=torch.bool, device=routed.device)
    mask[torch.tensor(encoded_columns, device=routed.device)] = False
    untouched = routed[:, mask]
    assert untouched.abs().max().item() == 0.0, "an unencoded column is non-zero"


def _explicit_routing(tokens, topk, num_experts, device, order=None):
    """Scrambled routing with no within-token duplicate expert."""
    ids = torch.zeros((tokens, topk), dtype=torch.int32)
    for token in range(tokens):
        for j in range(topk):
            if order is None:
                ids[token, j] = (token * 3 + j * 5 + 1) % num_experts
            else:
                ids[token, j] = order[(token + j) % len(order)]
        row = ids[token].tolist()
        if len(set(row)) != topk:
            raise AssertionError(f"fixture produced a duplicate expert: {row}")
    return ids.to(device)


@_NEEDS_SM80
@pytest.mark.parametrize("topk", [3, 6])
@pytest.mark.parametrize("tokens", [1, 4])
@pytest.mark.parametrize("varied_scales", [False, True])
def test_w13_grouped_gemm_matches_sparse_reference(
    topk: int, tokens: int, varied_scales: bool
) -> None:
    """Guards W13's non-zero result: token-row A, slot-row output, scales."""
    num_experts, device = 8, torch.device("cuda")
    positions = _positions(_W13_N_POSITIONS, _W13_K_POSITIONS)
    b_mxfp4, b_shift2 = _new_packed(num_experts, *_W13[:2], device)
    _fill_experts(b_mxfp4, b_shift2, positions, range(num_experts))
    channel_scale = torch.ones(
        (num_experts, _W13[0]), dtype=torch.float32, device=device
    )
    a_q = torch.randint(-127, 128, (tokens, _W13[1]), dtype=torch.int8, device=device)
    a_scale = torch.ones((tokens,), dtype=torch.float32, device=device)
    if varied_scales:
        channel_scale.copy_(
            0.11
            + torch.arange(num_experts * _W13[0], device=device).reshape_as(
                channel_scale
            )
            % 997
            / 811
        )
        a_scale.copy_(0.037 + torch.arange(tokens, device=device) / 13)
    routing_ids = _explicit_routing(tokens, topk, num_experts, device)

    routed, _, _, _, _ = _run_gemm(
        _W13, a_q, a_scale, b_mxfp4, b_shift2, channel_scale, routing_ids, 16, 128
    )
    _check_all_slots(_W13, a_q, a_scale, channel_scale, positions, routing_ids, routed)


@_NEEDS_SM80
@pytest.mark.parametrize("topk", [3, 6])
@pytest.mark.parametrize("varied_scales", [False, True])
def test_w2_grouped_gemm_reads_slot_rows(topk: int, varied_scales: bool) -> None:
    """Guards W2's A-row contract: slot, not token.

    a_q[slot, 0] is strictly increasing, so reading a_q[slot // topk, 0] (the
    W13 rule) would produce a different value for every slot beyond the first.
    """
    num_experts, tokens, device = 8, 4, torch.device("cuda")
    positions = {(0, 0): (0x07, 0)}
    b_mxfp4, b_shift2 = _new_packed(num_experts, *_W2[:2], device)
    _fill_experts(b_mxfp4, b_shift2, positions, range(num_experts))
    channel_scale = torch.ones(
        (num_experts, _W2[0]), dtype=torch.float32, device=device
    )
    num_routed = tokens * topk
    a_q = torch.zeros((num_routed, _W2[1]), dtype=torch.int8, device=device)
    a_q[:, 0] = torch.arange(1, num_routed + 1, dtype=torch.int8)
    a_scale = torch.ones((num_routed,), dtype=torch.float32, device=device)
    if varied_scales:
        channel_scale.copy_(
            0.11
            + torch.arange(num_experts * _W2[0], device=device).reshape_as(
                channel_scale
            )
            % 997
            / 811
        )
        a_scale.copy_(0.037 + torch.arange(num_routed, device=device) / 13)
    routing_ids = _explicit_routing(tokens, topk, num_experts, device)

    routed, _, _, _, _ = _run_gemm(
        _W2, a_q, a_scale, b_mxfp4, b_shift2, channel_scale, routing_ids, 16, 128
    )
    _check_all_slots(_W2, a_q, a_scale, channel_scale, positions, routing_ids, routed)


@_NEEDS_SM80
def test_routing_scramble_returns_each_slot_to_its_own_expert() -> None:
    """Guards the expert scatter: every expert gets a distinct weight value."""
    num_experts, tokens, topk, device = 8, 6, 3, torch.device("cuda")
    num_routed = tokens * topk
    b_mxfp4, b_shift2 = _new_packed(num_experts, *_W2[:2], device)
    channel_scale = torch.ones(
        (num_experts, _W2[0]), dtype=torch.float32, device=device
    )
    # Eight distinct decoded values at (0, 0), one per expert, so a swapped
    # expert shows up. The code is a magnitude *index*, and code 0x8 would
    # decode to signed zero, so the last expert uses 0xF (-12) instead.
    decode_codes = (0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x0F)
    for e in range(num_experts):
        _encode_element(b_mxfp4[e], b_shift2[e], 0, 0, decode_codes[e], 0)
    a_q = torch.ones((num_routed, _W2[1]), dtype=torch.int8, device=device)
    a_scale = torch.ones((num_routed,), dtype=torch.float32, device=device)
    routing_ids = _explicit_routing(tokens, topk, num_experts, device)

    routed, _, _, _, _ = _run_gemm(
        _W2, a_q, a_scale, b_mxfp4, b_shift2, channel_scale, routing_ids, 16, 128
    )
    flat = routing_ids.reshape(-1).tolist()
    expected_values = [
        (-_MAG_X2[code & 0x07] if code & 0x08 else _MAG_X2[code & 0x07])
        for code in decode_codes
    ]
    for slot, expert in enumerate(flat):
        assert routed[slot, 0].item() == float(expected_values[expert]), (
            f"slot={slot} landed on the wrong expert (want {expert})"
        )
        assert all(
            routed[slot, column].item() == 0.0 for column in (1, 7, 8, 2303, 5119)
        )


@_NEEDS_SM80
def test_hot_route_reuses_one_expert_across_tokens() -> None:
    """Guards the hot case: many tokens on one expert, same weights every time."""
    num_experts, tokens, topk, device = 8, 9, 3, torch.device("cuda")
    positions = _positions(_W2_N_POSITIONS, _W2_K_POSITIONS)
    b_mxfp4, b_shift2 = _new_packed(num_experts, *_W2[:2], device)
    _fill_experts(b_mxfp4, b_shift2, positions, range(num_experts))
    channel_scale = torch.ones(
        (num_experts, _W2[0]), dtype=torch.float32, device=device
    )
    a_q = torch.randint(
        -127, 128, (tokens * topk, _W2[1]), dtype=torch.int8, device=device
    )
    a_scale = torch.ones((tokens * topk,), dtype=torch.float32, device=device)
    # Every token hits experts {0, 1, 2}; expert 0 is reused nine times.
    routing_ids = torch.zeros((tokens, topk), dtype=torch.int32, device=device)
    routing_ids[:, 1] = 1
    routing_ids[:, 2] = 2

    routed, _, _, _, _ = _run_gemm(
        _W2, a_q, a_scale, b_mxfp4, b_shift2, channel_scale, routing_ids, 16, 128
    )
    flat = routing_ids.reshape(-1).tolist()
    assert flat.count(0) == tokens
    _check_all_slots(_W2, a_q, a_scale, channel_scale, positions, routing_ids, routed)


@_NEEDS_SM80
@pytest.mark.parametrize(
    ("count", "block_m"),
    [
        (15, 16),
        (16, 16),
        (17, 16),
        (63, 64),
        (64, 64),
        (65, 64),
        (127, 128),
        (128, 128),
        (129, 128),
    ],
)
def test_expert_row_count_crosses_block_m_boundary(count: int, block_m: int) -> None:
    """Guards per-expert padding at m_e = block_m - 1, block_m, block_m + 1."""
    num_experts, topk, device = 8, 3, torch.device("cuda")
    positions = _positions(_W2_N_POSITIONS, _W2_K_POSITIONS)
    b_mxfp4, b_shift2 = _new_packed(num_experts, *_W2[:2], device)
    _fill_experts(b_mxfp4, b_shift2, positions, range(num_experts))
    channel_scale = torch.ones(
        (num_experts, _W2[0]), dtype=torch.float32, device=device
    )
    tokens = count
    routing_ids = torch.zeros((tokens, topk), dtype=torch.int32, device=device)
    routing_ids[:, 0] = 0
    for j in range(1, topk):
        routing_ids[:, j] = 1 + ((torch.arange(tokens) + j) % (num_experts - 1))
    routing_ids = routing_ids.to(device)
    assert (routing_ids == 0).sum().item() == count

    num_routed = tokens * topk
    a_q = torch.randint(
        -127, 128, (num_routed, _W2[1]), dtype=torch.int8, device=device
    )
    a_scale = torch.ones((num_routed,), dtype=torch.float32, device=device)
    routed, _, _, _, num_post_pad = _run_gemm(
        _W2, a_q, a_scale, b_mxfp4, b_shift2, channel_scale, routing_ids, block_m, 128
    )
    # Padding is applied per expert, so the padded total is a multiple of block_m
    # and never smaller than the routed slots.
    assert num_post_pad.item() % block_m == 0
    assert num_post_pad.item() >= num_routed
    _check_all_slots(_W2, a_q, a_scale, channel_scale, positions, routing_ids, routed)


@_NEEDS_SM80
def test_empty_experts_do_not_pollute_routed_output() -> None:
    """Guards unread experts: their slots must be exactly zero."""
    num_experts, tokens, topk, device = 8, 4, 3, torch.device("cuda")
    positions = _positions(_W2_N_POSITIONS, _W2_K_POSITIONS)
    b_mxfp4, b_shift2 = _new_packed(num_experts, *_W2[:2], device)
    # Experts 0 and 1 stay entirely zero; only 2..7 carry weights.
    _fill_experts(b_mxfp4, b_shift2, positions, range(2, num_experts))
    channel_scale = torch.ones(
        (num_experts, _W2[0]), dtype=torch.float32, device=device
    )
    num_routed = tokens * topk
    a_q = torch.randint(
        -127, 128, (num_routed, _W2[1]), dtype=torch.int8, device=device
    )
    a_scale = torch.ones((num_routed,), dtype=torch.float32, device=device)
    routing_ids = torch.zeros((tokens, topk), dtype=torch.int32, device=device)
    routing_ids[:, 1] = 1  # the two empty experts
    routing_ids[:, 2] = torch.arange(2, 2 + tokens, dtype=torch.int32)

    routed, _, _, _, _ = _run_gemm(
        _W2, a_q, a_scale, b_mxfp4, b_shift2, channel_scale, routing_ids, 16, 128
    )
    flat = routing_ids.reshape(-1).tolist()
    for slot, expert in enumerate(flat):
        if expert < 2:
            assert routed[slot].abs().max().item() == 0.0, (
                f"empty expert at slot {slot}"
            )
        else:
            assert routed[slot].abs().max().item() > 0.0, f"slot {slot} was not written"


@_NEEDS_SM80
def test_all_valid_slots_are_written_and_padding_is_untouched() -> None:
    """Guards write-back completeness with a sentinel in the padded tail."""
    num_experts, tokens, topk, device = 8, 3, 6, torch.device("cuda")
    positions = _positions(_W2_N_POSITIONS, _W2_K_POSITIONS)
    b_mxfp4, b_shift2 = _new_packed(num_experts, *_W2[:2], device)
    _fill_experts(b_mxfp4, b_shift2, positions, range(num_experts))
    channel_scale = torch.ones(
        (num_experts, _W2[0]), dtype=torch.float32, device=device
    )
    num_routed = tokens * topk
    extra = 32
    a_q = torch.randint(
        -127, 128, (num_routed, _W2[1]), dtype=torch.int8, device=device
    )
    a_scale = torch.ones((num_routed,), dtype=torch.float32, device=device)
    routing_ids = _explicit_routing(tokens, topk, num_experts, device)

    routed, _, _, _, _ = _run_gemm(
        _W2,
        a_q,
        a_scale,
        b_mxfp4,
        b_shift2,
        channel_scale,
        routing_ids,
        16,
        128,
        extra_rows=extra,
    )
    for slot in range(num_routed):
        assert routed[slot, 0].item() != _SENTINEL, f"slot {slot} was never written"
    for slot in range(num_routed, num_routed + extra):
        assert (routed[slot] == _SENTINEL).all(), f"padding slot {slot} was written"


@_NEEDS_SM80
@pytest.mark.parametrize("topk", [3, 6])
def test_reduce_weights_each_slot_exactly_once(topk: int) -> None:
    """Guards the reduce: no double weighting, no implicit renormalisation."""
    tokens, hidden, device = 3, 5120, torch.device("cuda")
    num_routed = tokens * topk
    slots = torch.arange(num_routed, dtype=torch.float32, device=device)
    routed = torch.stack(
        (slots + 1.0, slots * 2.0 + 1.0, torch.ones_like(slots)), dim=1
    )
    routed = torch.nn.functional.pad(routed, (0, hidden - 3)).to(torch.bfloat16)
    # Non-uniform, not summing to one, and one slot weighted zero.
    weights = torch.full((tokens, topk), 0.5, dtype=torch.float32, device=device)
    weights[:, 0] = 0.25
    weights[0, 1] = 0.0
    out = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device)
    torch.ops._moe_C.dsv4_mxfp4_int8_sm80_moe_reduce(routed, weights, out, num_routed)
    torch.accelerator.synchronize()

    def expected_at(token: int, column: int) -> float:
        return sum(
            float(routed[token * topk + j, column].item())
            * float(weights[token, j].item())
            for j in range(topk)
        )

    for token in range(tokens):
        for column in (0, 1, 2):
            assert (
                out[token, column].item()
                == torch.tensor(expected_at(token, column)).to(torch.bfloat16).item()
            ), f"token={token} column={column}"
        # The weights sum to less than one; a renormalising reduce would divide
        # by the weight sum and land on the plain slot average instead.
        plain_average = torch.tensor(
            sum(float(routed[token * topk + j, 0].item()) for j in range(topk)) / topk
        ).to(torch.bfloat16)
        assert weights[token].sum().item() != 1.0
        assert (
            out[token, 0].item()
            == torch.tensor(expected_at(token, 0)).to(torch.bfloat16).item()
        )
        assert out[token, 0].item() != plain_average.item()


@pytest.fixture
def moe_harness():
    from benchmarks.kernels import deepseek_v41_mxfp4_int8_sm80 as driver

    return driver, importlib.import_module(driver.compare.__module__)


@_NEEDS_SM80
def test_hybrid_prefill_routes_replay_with_changed_expert_occupancy(moe_harness):
    """Pairing must neither drop slots nor retain a stale tail after graph replay."""
    from benchmarks.kernels.deepseek_v41_moe_hybrid_prefill import (
        HybridRoutes,
        check_partition,
    )

    _, helper = moe_harness
    hot = torch.arange(6, dtype=torch.int32, device="cuda").expand(129, -1).contiguous()
    spread = helper.make_routing_ids(129, 6, 8, "uniform", 41, device="cuda")
    aligned = moe_align_block_size(hot, 64, 8)
    hybrid = HybridRoutes(hot.numel(), 8, hot.device)
    hybrid.update(aligned)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        hybrid.update(aligned)
    counts = []
    for ids in (hot, spread, hot):
        updated = moe_align_block_size(ids, 64, 8)
        for dest, src in zip(aligned, updated, strict=True):
            dest.copy_(src)
        graph.replay()
        check_partition(ids, aligned, hybrid)
        counts.append(hybrid.counts.cpu().tolist())
    assert counts[0] == counts[2] == [6 * 128, 6 * 64]
    assert counts[1][1] == 0


@pytest.mark.parametrize(
    "correctness,measurement",
    [
        ("not_checked", "ok"),
        ("skipped", "ok"),
        ("uncalibrated", "ok"),
        ("pass", "fail"),
        ("pass", "skipped"),
        ("pass", "inconclusive"),
    ],
)
def test_harness_summary_rejects_incomplete_checks(
    moe_harness, tmp_path, correctness, measurement
):
    _, helper = moe_harness
    path = tmp_path / "summary.json"
    helper.write_summary(
        path,
        [{"correctness_status": correctness, "measurement_status": measurement}],
        run_status="complete",
    )
    assert not json.loads(path.read_text())["all_passed"]


def test_harness_summary_requires_completed_run(moe_harness, tmp_path):
    _, helper = moe_harness
    path = tmp_path / "summary.json"
    rows = [{"correctness_status": "pass", "measurement_status": "not_applicable"}]
    helper.write_summary(path, rows, run_status="complete")
    assert json.loads(path.read_text())["all_passed"]
    helper.write_summary(path, rows, run_status="aborted")
    assert not json.loads(path.read_text())["all_passed"]


@pytest.mark.parametrize(
    "failure", [RuntimeError("capture failed"), MemoryError("OOM")]
)
def test_harness_abort_preserves_rows_and_invalidates_old_summary(
    moe_harness, tmp_path, monkeypatch, failure
):
    driver, helper = moe_harness
    summary = tmp_path / "summary.json"
    summary.write_text('{"all_passed": true}')
    output = tmp_path / "results.jsonl"
    args = SimpleNamespace(summary_json=summary, output_jsonl=output, round=1)
    monkeypatch.setattr(driver, "parse_args", lambda: args)
    monkeypatch.setattr(driver, "source_identity", lambda: {})

    def fail(args, run_id, provenance, emit):
        emit(
            {
                "run_id": run_id,
                "correctness_status": "pass",
                "measurement_status": "not_applicable",
            }
        )
        raise failure

    monkeypatch.setattr(driver, "execute_scan", fail)
    with pytest.raises(type(failure), match=str(failure)):
        driver.main()
    result = json.loads(summary.read_text())
    assert result["run_status"] == "aborted" and not result["all_passed"]
    assert len(output.read_text().splitlines()) == 1
    assert json.loads(output.read_text())["run_id"] == result["run_id"]


def test_harness_local_reduce_bound_handles_zero_and_small_outputs(moe_harness):
    driver, helper = moe_harness
    weights = torch.tensor([[0.25, 0.25, 0.5]])
    routed = torch.tensor([[100139008.0, 0.25]] * 3, dtype=torch.bfloat16)
    expected = helper.reference_reduce(routed, weights)
    bad = expected.clone()
    bad[0, 1] = 0

    def verdict(value, ref, operands):
        return helper.classify(
            helper.compare(
                value,
                ref,
                term_scale=helper.reduce_term_scale(operands, weights),
                measure_bf16_steps=True,
            ),
            exact_required=False,
            threshold=driver.REDUCE_THRESHOLD,
        )[0]

    assert verdict(bad, expected, routed) == "fail"
    assert verdict(expected, expected, routed) == "pass"
    assert (
        verdict(
            torch.zeros_like(expected),
            torch.zeros_like(expected),
            torch.zeros_like(routed),
        )
        == "pass"
    )
    assert (
        verdict(
            torch.ones_like(expected),
            torch.zeros_like(expected),
            torch.zeros_like(routed),
        )
        == "fail"
    )


def test_harness_byte_accounting_does_not_select_weight_tensors(
    moe_harness, monkeypatch
):
    _, helper = moe_harness

    def no_copy(*args, **kwargs):
        pytest.fail("byte accounting must not gather weights")

    monkeypatch.setattr(torch.Tensor, "index_select", no_copy)
    weights = (
        torch.empty(8, 16, dtype=torch.uint8),
        torch.empty(8, 4, dtype=torch.uint8),
        torch.empty(8, 2),
    )
    assert helper.stage_weight_bytes(*weights, [0, 0, 7])["total"] == 56
    with pytest.raises(ValueError, match="expert ID"):
        helper.stage_weight_bytes(*weights, [8])


def test_harness_reduce_bound_tolerates_cancellation_rounding(moe_harness):
    driver, helper = moe_harness
    routed = torch.tensor([[100139008.0], [1.0], [-100139008.0]], dtype=torch.bfloat16)
    weights = torch.tensor([[0.25, 0.5, 0.25]])
    fp32 = helper.reference_reduce(routed, weights)
    fp64 = helper.reference_reduce(routed, weights, accumulate_dtype=torch.float64)
    assert not torch.equal(fp32, fp64)
    comparison = helper.compare(
        fp32,
        fp64,
        term_scale=helper.reduce_term_scale(routed, weights),
        measure_bf16_steps=True,
    )
    assert (
        helper.classify(
            comparison, exact_required=False, threshold=driver.REDUCE_THRESHOLD
        )[0]
        == "pass"
    )


@pytest.mark.parametrize(
    "sets",
    [
        [[0, 1], [0, 1]],
        [[0, 1], [1, 2]],
        [[0, 1], [2, 3]],
        [[0, 1, 2, 3, 4, 5], [6, 7, 0, 1, 2, 3]],
    ],
)
def test_harness_rotation_counts_repeated_experts_per_call(moe_harness, sets):
    driver, _ = moe_harness
    weights = {
        stage + suffix: torch.empty(8, width, dtype=dtype)
        for stage in ("w13", "w2")
        for suffix, width, dtype in [
            ("", 16, torch.uint8),
            ("_shift2", 4, torch.uint8),
            ("_scale", 2, torch.float32),
        ]
    }
    per_call, period, working_set = driver.rotation_accounting(
        weights, [torch.tensor([ids]) for ids in sets], 16
    )
    assert period == len(sets)
    assert per_call["w13"]["total"] == sum(len(set(ids)) * 28 for ids in sets) / len(
        sets
    )
    assert working_set["w2"] == len(set.union(*(set(ids) for ids in sets))) * 28


def test_harness_seed_reproduces_inputs_across_scan_order(moe_harness, monkeypatch):
    driver, helper = moe_harness
    monkeypatch.setattr(driver, "HIDDEN_SIZE", 128)
    monkeypatch.setattr(driver, "INTERMEDIATE_SIZE", 128)

    def build(seed, tokens=2):
        weights = driver.build_weights(8, torch.device("cpu"), seed=seed)
        tensors = driver.build_stage_tensors(
            weights,
            tokens=tokens,
            topk=3,
            num_routed=tokens * 3,
            device=torch.device("cpu"),
            seed=seed,
        )
        return {
            **{k: helper.tensor_hash(v) for k, v in weights.items()},
            **{stage: helper.tensor_hash(t["a_q"]) for stage, t in tensors.items()},
        }

    first = build(42)
    build(5, tokens=3)
    assert first == build(42)
    assert first != build(43)


def test_harness_fingerprint_includes_staged_and_untracked_sources(
    moe_harness, tmp_path
):
    _, helper = moe_harness

    def git(*args):
        subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True
        )

    git("init")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--allow-empty",
        "-m",
        "base",
    )
    source = tmp_path / "helper.py"
    source.write_text("first")
    first = helper.dirty_diff_hash(tmp_path)
    source.write_text("second")
    assert first != helper.dirty_diff_hash(tmp_path)
    git("add", "helper.py")
    staged = helper.dirty_diff_hash(tmp_path)
    source.write_text("third")
    git("add", "helper.py")
    assert staged != helper.dirty_diff_hash(tmp_path)


@pytest.mark.parametrize("bad_route", [None, 1])
def test_harness_checks_each_graph_output_and_rejects_noop_replay(
    moe_harness, monkeypatch, bad_route
):
    driver, _ = moe_harness
    monkeypatch.setattr(driver, "align_routes", lambda ids, *args: (ids, None, None))
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    def make_call(stage, tensors, weights, align, *args):
        output = tensors[stage]["routed_out"]
        value = float(align[0][0, 0]) + 1

        def invoke():
            output.fill_(value if bad_route != value - 1 else -1)

        return invoke

    monkeypatch.setattr(driver, "build_eager_call", make_call)
    tensors = {"w13": {"routed_out": torch.empty(1, 1), "output": torch.empty(1, 1)}}
    rows: list[dict] = []

    def build():
        return driver.build_timed_call(
            None,
            "w13",
            tensors,
            torch.ones(1, 1),
            [torch.tensor([[0]]), torch.tensor([[1]])],
            1,
            16,
            128,
            2,
            [torch.tensor([[1.0]]), torch.tensor([[2.0]])],
            {"variant": "16x128"},
            rows.append,
        )

    if bad_route is not None:
        with pytest.raises(AssertionError, match="route=1"):
            build()
        assert rows[-1]["correctness_status"] == "fail"
    else:
        call, count, validate = build()
        assert count == 2
        validate(SimpleNamespace(replay=call))
        with pytest.raises(AssertionError, match="graph-replay"):
            validate(SimpleNamespace(replay=lambda: None))


@pytest.fixture
def tilelang_offline_modules():
    """Load the offline TileLang harness only for tests that exercise it."""
    try:
        bench = importlib.import_module(
            "benchmarks.kernels.deepseek_v41_moe_tilelang_bench"
        )
        autotune = importlib.import_module(
            "benchmarks.kernels.deepseek_v41_moe_tilelang_autotune"
        )
    except ImportError as exc:
        pytest.skip(f"TileLang offline harness is unavailable: {exc}")
    return bench, autotune


def _tilelang_test_config(autotune):
    return autotune.GemmConfig(
        n=8,
        k=32,
        num_experts=1,
        topk=1,
        block_m=16,
        block_n=8,
        block_k=32,
        num_stages=1,
        threads=32,
        source_rows_are_slots=False,
    )


def _tilelang_test_case():
    reference = torch.zeros((1, 1), dtype=torch.bfloat16)
    return SimpleNamespace(
        stage="w13",
        tokens=1,
        slot_count=1,
        routed_out=torch.empty_like(reference),
        reference=reference,
    )


def test_tilelang_no_check_cannot_promote(tilelang_offline_modules, monkeypatch):
    """Diagnostic timing without reference checks cannot enter promotion set."""
    _, autotune = tilelang_offline_modules
    case = _tilelang_test_case()
    config = _tilelang_test_config(autotune)
    calls = []

    monkeypatch.setattr(
        autotune.TileLangMoeGemm,
        "create",
        lambda config, tokens, slots: SimpleNamespace(
            source="test-source", compile_seconds=0.0
        ),
    )
    monkeypatch.setattr(
        autotune,
        "candidate_call",
        lambda gemm, case: lambda: case.routed_out.copy_(case.reference),
    )
    monkeypatch.setattr(autotune, "case_identity", lambda case: {})

    def fake_time_graph(call, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(as_row=lambda: {"p50_us": 1.0, "p90_us": 1.0})

    monkeypatch.setattr(autotune, "time_graph", fake_time_graph)
    monkeypatch.setattr(autotune.torch.accelerator, "synchronize", lambda: None)

    record = autotune.measure_config(
        case=case,
        config=config,
        reference=case.reference,
        warmup=0,
        samples=1,
        graph_calls=1,
        check=False,
    )

    assert record["verification_status"] == "skipped"
    assert record["measurement_status"] == "ok"
    assert record["promotable"] is False
    assert record["graph_replay_checked"] is False
    assert calls and calls[0]["validate_graph"] is None


def test_tilelang_timing_failure_is_recorded(tilelang_offline_modules, monkeypatch):
    """A CUDA-graph/event failure cannot leave a verified candidate timed."""
    _, autotune = tilelang_offline_modules
    case = _tilelang_test_case()
    config = _tilelang_test_config(autotune)

    monkeypatch.setattr(
        autotune.TileLangMoeGemm,
        "create",
        lambda config, tokens, slots: SimpleNamespace(
            source="test-source", compile_seconds=0.0
        ),
    )
    monkeypatch.setattr(
        autotune,
        "candidate_call",
        lambda gemm, case: lambda: case.routed_out.copy_(case.reference),
    )
    monkeypatch.setattr(autotune, "case_identity", lambda case: {})

    def fail_timing(*args, **kwargs):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(autotune, "time_graph", fail_timing)
    monkeypatch.setattr(autotune.torch.accelerator, "synchronize", lambda: None)

    record = autotune.measure_config(
        case=case,
        config=config,
        reference=case.reference,
        warmup=0,
        samples=1,
        graph_calls=1,
        check=True,
    )

    assert record["verification_status"] == "pass"
    assert record["measurement_status"] == "timing_failure"
    assert record["promotable"] is False
    assert "capture failed" in record["failure_reason"]


def test_tilelang_graph_validator_poison_checks_every_output(
    tilelang_offline_modules, monkeypatch
):
    """A replay that writes only one case must fail the other case's check."""
    bench, _ = tilelang_offline_modules
    reference = torch.zeros((1, 1), dtype=torch.bfloat16)
    first = SimpleNamespace(
        routed_out=torch.ones_like(reference), reference=reference.clone()
    )
    second = SimpleNamespace(
        routed_out=torch.ones_like(reference), reference=reference.clone()
    )

    class PartialReplay:
        def replay(self):
            first.routed_out.copy_(first.reference)

    monkeypatch.setattr(bench.torch.accelerator, "synchronize", lambda: None)
    validate = bench.graph_validator([first, second])
    with pytest.raises(ValueError, match="route 1"):
        validate(PartialReplay())
    assert torch.isnan(second.routed_out).all()


def test_tilelang_first_mismatch_distinguishes_bfloat16_signed_zero(
    tilelang_offline_modules,
):
    bench, _ = tilelang_offline_modules
    positive_zero = torch.tensor([[0.0]], dtype=torch.bfloat16)
    negative_zero = torch.tensor([[-0.0]], dtype=torch.bfloat16)

    result = bench.first_mismatch(positive_zero, negative_zero)

    assert result["finite"]
    assert not result["bitwise_exact"]
    assert result["mismatches"] == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_tilelang_first_mismatch_rejects_nonfinite_candidate(
    tilelang_offline_modules, value
):
    bench, _ = tilelang_offline_modules
    candidate = torch.tensor([[value]], dtype=torch.bfloat16)
    reference = torch.zeros_like(candidate)

    result = bench.first_mismatch(candidate, reference)

    assert not result["finite"]
    assert not result["bitwise_exact"]


@pytest.mark.parametrize("tokens", [15, 16, 17])
def test_tilelang_tile_boundary_routes_are_slot_balanced(
    tilelang_offline_modules, tokens
):
    bench, _ = tilelang_offline_modules
    routing_ids = bench.candidate_routes(
        tokens, topk=3, num_experts=8, pattern="tile-boundary", seed=789
    )

    assert routing_ids.shape == (tokens, 3)
    assert bool((routing_ids.sort(dim=1).values.diff(dim=1) != 0).all().item())
    assert torch.equal(routing_ids, routing_ids[0].expand(tokens, -1))
    for expert in routing_ids[0].tolist():
        assert int((routing_ids == expert).sum()) == tokens


@_NEEDS_SM80
@pytest.mark.parametrize("stage", ["w13", "w2"])
@pytest.mark.parametrize("tokens", [15, 17])
def test_tilelang_sm80_tile_boundary_candidate_graph(
    tilelang_offline_modules, stage, tokens
):
    """Check both routed GEMM stages across the BM16 boundary on real SM80."""
    bench, autotune = tilelang_offline_modules
    device = torch.device("cuda")
    n, k, source_rows_are_slots = bench.STAGE_SHAPES[stage]
    config = autotune.GemmConfig(
        n=n,
        k=k,
        num_experts=8,
        topk=3,
        block_m=16,
        block_n=64,
        block_k=128,
        num_stages=2 if tokens == 15 else 3,
        threads=128 if tokens == 15 else 64,
        source_rows_are_slots=source_rows_are_slots,
        decode="prmt",
    )
    case = bench.build_case(
        stage=stage,
        num_experts=8,
        topk=3,
        tokens=tokens,
        pattern="tile-boundary",
        seed=789,
        block_m=16,
        block_n=64,
        device=device,
        scale_mode="varied",
    )
    gemm = autotune.TileLangMoeGemm.create(
        config, tokens=case.tokens, slots=case.slot_count
    )
    call = bench.candidate_call(gemm, case)

    case.routed_out.fill_(float("nan"))
    call()
    torch.accelerator.synchronize()
    eager = bench.first_mismatch(case.routed_out, case.reference)
    assert eager["finite"] and eager["bitwise_exact"]

    autotune.time_graph(
        call,
        warmup=1,
        repeats=1,
        samples=1,
        validate_graph=bench.graph_validator([case]),
    )


def _tilelang_gate_inputs(*, model_type="deepseek_v41", switch=False):
    parallel = SimpleNamespace(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        sp_size=1,
        use_ep=False,
    )
    moe = SimpleNamespace(
        num_experts=384,
        num_local_experts=384,
        experts_per_token=6,
        hidden_dim=5120,
        intermediate_size_per_partition=2304,
        in_dtype=torch.bfloat16,
        activation=MoEActivation.SILU,
        has_bias=False,
        is_lora_enabled=False,
        moe_parallel_config=parallel,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type=model_type,
                hidden_size=5120,
                moe_intermediate_size=2304,
                n_routed_experts=384,
            )
        ),
        additional_config={"dsv41_sm80_moe_tilelang": switch},
    )
    return config, moe


def _hybrid_prefill_gate_inputs():
    config, moe = _tilelang_gate_inputs()
    config.additional_config.update(
        ced_prefill=True, dsv41_sm80_moe_hybrid_prefill=True
    )
    config.parallel_config = SimpleNamespace(
        pipeline_parallel_size=5,
        tensor_parallel_size=1,
        data_parallel_size=1,
        use_ubatching=False,
    )
    config.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=2048, max_num_seqs=8
    )
    config.use_v2_model_runner = True
    config.speculative_config = config.lora_config = None
    return config, moe


def test_hybrid_prefill_gate_defaults_off_and_enables_validated_config():
    config, moe = _hybrid_prefill_gate_inputs()
    assert dsv41_sm80_hybrid_prefill_enabled(config, moe)
    del config.additional_config["dsv41_sm80_moe_hybrid_prefill"]
    assert not dsv41_sm80_hybrid_prefill_enabled(config, moe)


@pytest.mark.parametrize("flag", ["true", 1, None])
def test_hybrid_prefill_gate_requires_json_boolean(flag):
    config, moe = _hybrid_prefill_gate_inputs()
    config.additional_config["dsv41_sm80_moe_hybrid_prefill"] = flag
    with pytest.raises(ValueError, match="JSON boolean"):
        dsv41_sm80_hybrid_prefill_enabled(config, moe)


@pytest.mark.parametrize(
    "owner,path,value",
    [
        ("config", "speculative_config", SimpleNamespace(method="dspark")),
        ("config", "lora_config", object()),
        ("config", "parallel_config.pipeline_parallel_size", 3),
        ("config", "parallel_config.use_ubatching", True),
        ("config", "scheduler_config.max_num_batched_tokens", 4096),
        ("config", "model_config.hf_config.model_type", "deepseek_v4"),
        ("config", "use_v2_model_runner", False),
        ("moe", "experts_per_token", 3),
        ("moe", "moe_parallel_config.tp_size", 2),
    ],
)
def test_hybrid_prefill_gate_preserves_other_configurations(owner, path, value):
    config, moe = _hybrid_prefill_gate_inputs()
    target = config if owner == "config" else moe
    parts = path.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)
    assert not dsv41_sm80_hybrid_prefill_enabled(config, moe)


@pytest.mark.parametrize("ced,fast", [(False, False), (True, True)])
def test_hybrid_prefill_gate_requires_ced_without_fast_prefill(ced, fast):
    config, moe = _hybrid_prefill_gate_inputs()
    config.additional_config.update(ced_prefill=ced, dsv41_fast_prefill=fast)
    assert not dsv41_sm80_hybrid_prefill_enabled(config, moe)


@pytest.mark.parametrize(
    "kind", ["pure", "missing", "empty", "decode", "mixed", "graph", "ubatch"]
)
def test_hybrid_prefill_batch_gate_preserves_decode_mixed_and_graphs(monkeypatch, kind):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.model_executor.layers.fused_moe.experts import mxfp4_int8_sm80 as impl

    span = SimpleNamespace(num_prefills=1, num_decodes=0)
    context = SimpleNamespace(
        attn_metadata={"swa": span},
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        ubatch_slices=None,
    )
    if kind == "empty":
        context.attn_metadata = {}
    elif kind in ("decode", "mixed"):
        span.num_decodes = 1
        span.num_prefills = int(kind == "mixed")
    elif kind == "graph":
        context.cudagraph_runtime_mode = CUDAGraphMode.FULL
    elif kind == "ubatch":
        context.attn_metadata = [context.attn_metadata]
    monkeypatch.setattr(impl, "is_forward_context_available", lambda: kind != "missing")
    monkeypatch.setattr(impl, "get_forward_context", lambda: context)
    assert impl._is_pure_eager_prefill() == (kind == "pure")


@_NEEDS_SM80
def test_hybrid_prefill_experts_apply_preserves_aliased_output_and_changed_routes(
    tilelang_offline_modules, monkeypatch
):
    """Exercise both GEMMs, activation and reduction through the production gate."""
    from vllm.config.compilation import CUDAGraphMode
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.experts import mxfp4_int8_sm80 as impl

    config, moe = _hybrid_prefill_gate_inputs()
    for key in (
        "swiglu_limit",
        "swiglu_alpha",
        "swiglu_beta",
        "activation_situ_beta",
        "activation_situ_linear_beta",
    ):
        setattr(moe, key, None)
    bench, _ = tilelang_offline_modules
    device = torch.device("cuda")
    weights = [
        bench.build_packed_weights(384, stage, device, seed=914)
        for stage in ("w13", "w2")
    ]
    for w in weights:
        w["channel_scale"].mul_(1e-3)
    layer = SimpleNamespace(
        w13_mxfp4_int8_shift2=weights[0]["shift2"],
        w2_mxfp4_int8_shift2=weights[1]["shift2"],
    )
    quant = FusedMoEQuantConfig.make(
        w1_scale=weights[0]["channel_scale"], w2_scale=weights[1]["channel_scale"]
    )
    monkeypatch.setattr(impl, "get_current_vllm_config_or_none", lambda: config)
    candidate = Mxfp4Int8Sm80Experts(moe, quant)
    candidate.process_weights_after_loading(layer)
    assert candidate._hybrid_prefill_routes is not None
    config.additional_config["dsv41_sm80_moe_hybrid_prefill"] = False
    baseline = Mxfp4Int8Sm80Experts(moe, quant)
    baseline.process_weights_after_loading(layer)
    assert baseline._hybrid_prefill_routes is None
    context = SimpleNamespace(
        attn_metadata={"swa": SimpleNamespace(num_prefills=1, num_decodes=0)},
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        ubatch_slices=None,
    )
    monkeypatch.setattr(impl, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(impl, "get_forward_context", lambda: context)
    hidden = torch.randn((2048, 5120), device=device, dtype=torch.bfloat16)
    ids = torch.arange(6, device=device).expand(2048, -1).contiguous()
    rw = torch.full((2048, 6), 1 / 6, device=device)

    def run(expert):
        ws13 = torch.empty((12288, 3 * 2304), device=device, dtype=torch.bfloat16)
        ws2 = torch.empty((12288, 5120), device=device, dtype=torch.bfloat16)
        output = ws13.flatten()[: 2048 * 5120].view(2048, 5120)
        expert.apply(
            output,
            hidden,
            weights[0]["packed"],
            weights[1]["packed"],
            rw,
            ids,
            MoEActivation.SILU,
            384,
            None,
            None,
            None,
            ws13,
            ws2,
            None,
            False,
        )
        return output.clone(), ws2

    for spread in (False, True):
        if spread:
            ids = torch.arange(12288, device=device).view(2048, 6) % 384
        expected, expected_slots = run(baseline)
        output, slots = run(candidate)
        assert torch.isfinite(output).all()
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
        assert torch.equal(slots.view(torch.int16), expected_slots.view(torch.int16))


def test_dsv41_sm80_tilelang_gate_defaults_off_and_enables_target():
    config, moe = _tilelang_gate_inputs()

    assert not dsv41_sm80_tilelang_enabled(config, moe)
    config.additional_config["dsv41_sm80_moe_tilelang"] = True
    assert dsv41_sm80_tilelang_enabled(config, moe)


@pytest.mark.parametrize("model_type", ["deepseek_v4", "qwen"])
def test_dsv41_sm80_tilelang_gate_requires_target_model_identity(model_type):
    config, moe = _tilelang_gate_inputs(model_type=model_type, switch=True)

    assert not dsv41_sm80_tilelang_enabled(config, moe)


@pytest.mark.parametrize("switch", ["true", 1, None])
def test_dsv41_sm80_tilelang_gate_rejects_non_boolean_target_switch(switch):
    config, moe = _tilelang_gate_inputs(switch=switch)

    with pytest.raises(ValueError, match="JSON boolean"):
        dsv41_sm80_tilelang_enabled(config, moe)


@pytest.mark.parametrize("switch", ["true", 1, None])
def test_dsv41_sm80_tilelang_gate_ignores_malformed_switch_for_other_models(switch):
    config, moe = _tilelang_gate_inputs(model_type="other_model", switch=switch)

    assert not dsv41_sm80_tilelang_enabled(config, moe)


@pytest.mark.parametrize(
    "mutation",
    [
        "topk3",
        "experts256",
        "wrong_dims",
        "parallel_tp2",
        "lora",
        "bias",
    ],
)
def test_dsv41_sm80_tilelang_gate_falls_back_for_unsupported_moe(mutation):
    config, moe = _tilelang_gate_inputs(switch=True)
    if mutation == "topk3":
        moe.experts_per_token = 3
    elif mutation == "experts256":
        moe.num_experts = moe.num_local_experts = 256
    elif mutation == "wrong_dims":
        config.model_config.hf_config.hidden_size = 4096
    elif mutation == "parallel_tp2":
        moe.moe_parallel_config.tp_size = 2
    elif mutation == "lora":
        moe.is_lora_enabled = True
    else:
        moe.has_bias = True

    assert not dsv41_sm80_tilelang_enabled(config, moe)


@_NEEDS_SM80
def test_gemm_nonunit_scale_straddles_bf16_rounding_boundary():
    device = torch.device("cuda")
    b, shifts = _new_packed(8, *_W2[:2], device)
    _fill_experts(b, shifts, {(0, 0): (1, 0)}, range(8))
    a = torch.ones((3, _W2[1]), dtype=torch.int8, device=device)
    scales = torch.tensor(
        [1.00390625 - 2**-23, 1.00390625, 1.00390625 + 2**-23], device=device
    )
    channel = torch.full((8, _W2[0]), 0.73, device=device)
    channel[:, 0] = 1.0
    ids = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
    routed, *_ = _run_gemm(_W2, a, scales, b, shifts, channel, ids, 16, 128)
    assert routed[:, 0].tolist() == [1.0, 1.0, 1.0078125]


@pytest.mark.parametrize(
    "mutation", ["omit_a", "omit_channel", "token_for_slot", "wrong_expert"]
)
def test_harness_nonunit_scales_reject_incorrect_epilogues(
    moe_harness, monkeypatch, mutation
):
    driver, helper = moe_harness
    monkeypatch.setitem(helper.STAGE_SHAPES, "w2", (8, 128, True))
    packed, shift = helper.manual_anchor_tensors()
    packed, shift = packed.repeat(3, 1, 1, 1), shift.repeat(3, 1, 1, 1)
    a = torch.ones((3, 128), dtype=torch.int8)
    a_scale = torch.tensor([0.0313, 0.11, 0.73])
    channel = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 13 + 0.17
    ids = torch.tensor([[2, 0, 1]], dtype=torch.int32)

    def reference(activation_scale, channel_scale):
        return helper.reference_stage(
            a, activation_scale, packed, shift, channel_scale, ids, stage="w2"
        )

    expected = reference(a_scale, channel)
    if mutation == "omit_a":
        observed = reference(torch.ones_like(a_scale), channel)
    elif mutation == "omit_channel":
        observed = reference(a_scale, torch.ones_like(channel))
    elif mutation == "token_for_slot":
        observed = reference(a_scale[:1].expand_as(a_scale), channel)
    else:
        observed = reference(a_scale, channel.roll(1, 0))
    assert (
        helper.classify(
            helper.compare(observed, expected), exact_required=True, threshold=None
        )[0]
        == "fail"
    )


@_NEEDS_SM80
@pytest.mark.parametrize("initial_clamp", [None, 10.0])
def test_dsv41_tilelang_experts_apply_and_changing_graph_routes(
    tilelang_offline_modules, monkeypatch, initial_clamp
):
    """Compare the real quantize/GEMM/activation/reduce path and shape fallback."""
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("the frozen candidate targets SM80 only")
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.experts import (
        dsv41_sm80_activation,
        dsv41_sm80_tilelang,
        mxfp4_int8_sm80,
    )

    bench, _ = tilelang_offline_modules
    config, moe = _tilelang_gate_inputs(switch=True)
    for key in (
        "swiglu_limit",
        "swiglu_alpha",
        "swiglu_beta",
        "activation_situ_beta",
        "activation_situ_linear_beta",
    ):
        setattr(moe, key, None)
    moe.swiglu_limit = initial_clamp
    device = torch.device("cuda")
    w13 = bench.build_packed_weights(384, "w13", device, seed=911)
    w2 = bench.build_packed_weights(384, "w2", device, seed=912)
    # Keep the full nonlinear path well-scaled while varying each channel.
    w13["channel_scale"].mul_(1e-3)
    w2["channel_scale"].mul_(1e-3)
    layer = SimpleNamespace(
        w13_weight=w13["packed"],
        w2_weight=w2["packed"],
        w13_mxfp4_int8_shift2=w13["shift2"],
        w2_mxfp4_int8_shift2=w2["shift2"],
    )
    quant = FusedMoEQuantConfig.make(
        w1_scale=w13["channel_scale"], w2_scale=w2["channel_scale"]
    )
    monkeypatch.setattr(
        mxfp4_int8_sm80, "get_current_vllm_config_or_none", lambda: config
    )
    candidate = Mxfp4Int8Sm80Experts(moe, quant)
    candidate.process_weights_after_loading(layer)
    assert candidate._tilelang_decode_gemms is not None
    assert candidate._tilelang_m6_w13 is not None
    assert candidate._tilelang_m6_w2 is not None
    assert candidate._tilelang_activation_quant is not None
    config.additional_config["dsv41_sm80_moe_tilelang"] = False
    baseline = Mxfp4Int8Sm80Experts(moe, quant)
    baseline.process_weights_after_loading(layer)
    assert baseline._tilelang_decode_gemms is None
    assert baseline._tilelang_m6_w13 is None
    assert baseline._tilelang_m6_w2 is None
    assert baseline._tilelang_activation_quant is None

    def no_compilation(*args, **kwargs):
        pytest.fail("forward/capture must never compile or tune")

    monkeypatch.setattr(dsv41_sm80_tilelang, "compile_decode_gemms", no_compilation)
    monkeypatch.setattr(dsv41_sm80_tilelang, "compile_m6_w13", no_compilation)
    monkeypatch.setattr(dsv41_sm80_tilelang, "compile_m6_w2", no_compilation)
    monkeypatch.setattr(
        dsv41_sm80_activation, "compile_activation_quant", no_compilation
    )
    launches = []

    def tracked(index, kernel):
        def call(*args):
            launches.append(index)
            return kernel(*args)

        return call

    candidate._tilelang_decode_gemms = tuple(
        tracked(i, kernel) for i, kernel in enumerate(candidate._tilelang_decode_gemms)
    )
    candidate._tilelang_activation_quant = tracked(
        2, candidate._tilelang_activation_quant
    )
    candidate._tilelang_m6_w2 = tracked(3, candidate._tilelang_m6_w2)
    candidate._tilelang_m6_w13 = tracked(4, candidate._tilelang_m6_w13)
    generator = torch.Generator(device=device).manual_seed(913)
    for tokens, topk, clamp in (
        (1, 6, initial_clamp),
        (2, 6, initial_clamp),
        (6, 6, initial_clamp),
        (7, 6, initial_clamp),
        (22, 6, initial_clamp),
        (1, 3, initial_clamp),
        (6, 6, 0.01),
    ):
        candidate.activation_config = replace(
            candidate.activation_config, clamp_limit=clamp
        )
        baseline.activation_config = replace(
            baseline.activation_config, clamp_limit=clamp
        )
        hidden = torch.randn(
            (tokens, 5120), device=device, dtype=torch.bfloat16, generator=generator
        )
        ids = torch.tensor(
            [[311, 17, 205, 92, 4, 378]], device=device, dtype=torch.int64
        )[:, :topk].repeat(tokens, 1)
        weights = torch.linspace(0.07, 0.31, topk, device=device).repeat(tokens, 1)
        weights /= weights.sum(dim=1, keepdim=True)

        def make_call(experts, tokens, topk, hidden, weights, ids):
            ws13 = torch.empty(
                (tokens * topk, 3 * 2304), device=device, dtype=torch.bfloat16
            )
            ws2 = torch.empty(
                (tokens * topk, 5120), device=device, dtype=torch.bfloat16
            )
            # Match the modular runner's output/workspace13 storage alias.
            output = ws13.flatten()[: tokens * 5120].view(tokens, 5120)

            def call():
                experts.apply(
                    output,
                    hidden,
                    layer.w13_weight,
                    layer.w2_weight,
                    weights,
                    ids,
                    MoEActivation.SILU,
                    384,
                    None,
                    None,
                    None,
                    ws13,
                    ws2,
                    None,
                    False,
                )

            return call, output, ws2

        base_call, expected, expected_slots = make_call(
            baseline, tokens, topk, hidden, weights, ids
        )
        cand_call, output, slots = make_call(
            candidate, tokens, topk, hidden, weights, ids
        )
        launches.clear()
        base_call()
        cand_call()
        expected_launches = [0, 2, 1] if tokens == 1 and topk == 6 else []
        if tokens in (2, 6) and topk == 6 and clamp == initial_clamp:
            expected_launches = [2]
        if tokens == 7 and topk == 6 and clamp == initial_clamp:
            expected_launches = [2]
        if tokens == 6 and topk == 6:
            expected_launches.insert(0, 4)
            expected_launches.append(3)
        assert launches == expected_launches
        assert bench.first_mismatch(output, expected)["bitwise_exact"]
        assert bench.first_mismatch(slots, expected_slots)["bitwise_exact"]

        graph = importlib.import_module("deepseek_v41_moe_harness").capture_graph(
            cand_call, warmup=2, graph_calls=1
        )
        for _ in range(2):
            # Change dynamic data in place while keeping captured tensor pointers.
            hidden.copy_(
                torch.randn(
                    hidden.shape, device=device, dtype=hidden.dtype, generator=generator
                )
            )
            ids.add_(37).remainder_(384)
            if tokens == 6 and topk == 6:
                # A captured hot route must also overwrite all spread-route slots.
                ids.copy_(torch.arange(36, device=device).view(6, 6) + 37)
            base_call()
            output.fill_(float("nan"))
            slots.fill_(float("nan"))
            graph.replay()
            torch.accelerator.synchronize()
            assert bench.first_mismatch(output, expected)["bitwise_exact"]
            assert bench.first_mismatch(slots, expected_slots)["bitwise_exact"]

        if tokens == 6 and topk == 6:
            monkeypatch.setattr(mxfp4_int8_sm80, "_DECODE_BLOCK_N", 64)
            launches.clear()
            base_call()
            cand_call()
            assert 3 not in launches
            assert bench.first_mismatch(output, expected)["bitwise_exact"]
            monkeypatch.setattr(mxfp4_int8_sm80, "_DECODE_BLOCK_N", 128)


@pytest.mark.parametrize("requested,minor", [(False, 0), (True, 6)])
def test_dsv41_tilelang_load_gate_does_not_import_for_ineligible_device(
    monkeypatch, requested, minor
):
    """Default-off and SM86 loading must not import or compile the candidate."""
    import builtins

    from vllm.model_executor.layers.fused_moe.experts import mxfp4_int8_sm80 as impl

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if "dsv41_sm80_tilelang" in name or "dsv41_sm80_tilelang" in (fromlist or ()):
            pytest.fail("an ineligible layer imported the TileLang candidate")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(impl.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        impl.current_platform,
        "get_device_capability",
        lambda device_id: SimpleNamespace(major=8, minor=minor),
    )
    experts = object.__new__(Mxfp4Int8Sm80Experts)
    experts._tilelang_requested = requested
    layer = SimpleNamespace(
        w13_mxfp4_int8_shift2=torch.empty(1),
        w2_mxfp4_int8_shift2=torch.empty(1),
    )
    experts.process_weights_after_loading(layer)
    assert experts._tilelang_decode_gemms is None
    assert experts._tilelang_m6_w13 is None
    assert experts._tilelang_m6_w2 is None
    assert experts._tilelang_activation_quant is None


@pytest.mark.parametrize("source", ["moe", "quant"])
def test_dsv41_tilelang_load_preserves_clamped_activation(monkeypatch, source):
    """Both clamp sources retain activation while the GEMMs can remain enabled."""
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.experts import (
        dsv41_sm80_activation,
        dsv41_sm80_tilelang,
    )
    from vllm.model_executor.layers.fused_moe.experts import (
        mxfp4_int8_sm80 as impl,
    )

    config, moe = _tilelang_gate_inputs(switch=True)
    for key in (
        "swiglu_limit",
        "swiglu_alpha",
        "swiglu_beta",
        "activation_situ_beta",
        "activation_situ_linear_beta",
    ):
        setattr(moe, key, None)
    if source == "moe":
        moe.swiglu_limit = 7.0
    quant = FusedMoEQuantConfig.make(
        gemm1_clamp_limit=7.0 if source == "quant" else None
    )
    monkeypatch.setattr(impl, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(impl.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        impl.current_platform,
        "get_device_capability",
        lambda device_id: SimpleNamespace(major=8, minor=0),
    )
    monkeypatch.setattr(
        dsv41_sm80_tilelang, "prepare_decode_gemms", lambda *args: ("w13", "w2")
    )
    monkeypatch.setattr(dsv41_sm80_tilelang, "prepare_m6_w2", lambda *args: "m6_w2")
    monkeypatch.setattr(dsv41_sm80_tilelang, "prepare_m6_w13", lambda *args: "m6_w13")

    def unexpected_prepare(device):
        pytest.fail("a clamped layer must not compile or warm the fused activation")

    monkeypatch.setattr(
        dsv41_sm80_activation, "prepare_activation_quant", unexpected_prepare
    )
    experts = Mxfp4Int8Sm80Experts(moe, quant)
    layer = SimpleNamespace(
        w13_mxfp4_int8_shift2=torch.empty(1), w2_mxfp4_int8_shift2=torch.empty(1)
    )
    experts.process_weights_after_loading(layer)
    assert experts.activation_config.clamp_limit == 7.0
    assert experts._tilelang_decode_gemms == ("w13", "w2")
    assert experts._tilelang_m6_w13 == "m6_w13"
    assert experts._tilelang_m6_w2 == "m6_w2"
    assert experts._tilelang_activation_quant is None


@pytest.fixture(
    scope="module", params=[(False, None), (True, None), (False, 10.0), (True, 10.0)]
)
def tilelang_swiglu_int8(request):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("the SwiGLU candidate targets SM80")
    impl = importlib.import_module(
        "benchmarks.kernels.deepseek_v41_moe_tilelang_activation"
    )
    fast_math, clamp_limit = request.param
    kernel = impl.tilelang.compile(
        impl.build_swiglu_int8(
            threads=256, fast_math=fast_math, clamp_limit=clamp_limit
        ),
        pass_configs=impl.PASS_CONFIGS,
    )
    return kernel, clamp_limit


@pytest.mark.parametrize("fast_math", [False, True])
@pytest.mark.parametrize("clamp_limit", [None, 10.0])
def test_tilelang_swiglu_math_matches_every_finite_bf16_gate(fast_math, clamp_limit):
    """Certify the BF16 SiLU round before later quantization can hide a drift."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("the SwiGLU candidate targets SM80")
    impl = importlib.import_module(
        "benchmarks.kernels.deepseek_v41_moe_tilelang_activation"
    )
    assert (
        impl.certify_swiglu_math(fast_math=fast_math, clamp_limit=clamp_limit)[
            "bitwise_mismatches"
        ]
        == 0
    )


def _cuda_swiglu_int8_reference(x, clamp_limit=None):
    middle = torch.empty((x.shape[0], 2304), device=x.device, dtype=torch.bfloat16)
    if clamp_limit is None:
        torch.ops._C.silu_and_mul(middle, x)
    else:
        torch.ops._C.silu_and_mul_with_clamp(middle, x, clamp_limit, 1.0, 0.0)
    return quantize_per_token_int8(middle)


@pytest.mark.parametrize("rows", [1, 6, 18, 36, 40, 128])
@pytest.mark.parametrize("pattern", ["random", "small", "large", "rne", "zero"])
def test_tilelang_swiglu_int8_matches_materialized_rounding(
    tilelang_swiglu_int8, rows, pattern
):
    """The fused path must reproduce both CUDA BF16 rounds and INT8 q/scale."""
    kernel, clamp_limit = tilelang_swiglu_int8
    generator = torch.Generator(device="cuda").manual_seed(541 + rows)
    x = torch.randn(
        (rows, 4608), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    if pattern == "small":
        x.mul_(2**-8)
    elif pattern == "large":
        x.mul_(2**8)
    elif pattern == "zero":
        x[:, :2304].zero_()
        x[::2, :2304].fill_(-0.0)
    elif pattern == "rne":
        # BF16 SiLU(8) rounds to 8; a power-of-two scale preserves exact ties.
        x[:, :2304].fill_(8)
        unit_scale = 1.0 if clamp_limit is None else 0.5
        levels = torch.tensor(
            [0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5], device="cuda"
        ).repeat(330)[:2304]
        x[:, 2304:].copy_((levels * unit_scale / 8).bfloat16())
        x[:, -1] = 127 * unit_scale / 8
    expected_q, expected_s = _cuda_swiglu_int8_reference(x, clamp_limit)
    q = torch.full_like(expected_q, -128)
    scale = torch.full_like(expected_s, float("nan"))
    kernel(x, q, scale)
    torch.accelerator.synchronize()
    assert torch.equal(q, expected_q)
    assert torch.equal(scale.view(torch.int32), expected_s.view(torch.int32))


def test_tilelang_swiglu_int8_finite_bf16_gate_codes(tilelang_swiglu_int8):
    """Cover every finite BF16 gate code with several finite product magnitudes."""
    kernel, clamp_limit = tilelang_swiglu_int8
    bits = torch.arange(65536, device="cuda", dtype=torch.int32)
    bits = bits[(bits & 0x7F80) != 0x7F80]
    gates = bits.to(torch.int16).view(torch.bfloat16)
    gates = gates.repeat(4)
    ups = torch.tensor([1, -1, 0.333984375, -0.333984375], device="cuda")
    ups = ups.repeat_interleave(bits.numel()).bfloat16()
    rows = (gates.numel() + 2303) // 2304
    x = torch.zeros((rows, 4608), device="cuda", dtype=torch.bfloat16)
    x[:, :2304].copy_(
        torch.nn.functional.pad(gates, (0, rows * 2304 - gates.numel())).view(
            rows, 2304
        )
    )
    x[:, 2304:].copy_(
        torch.nn.functional.pad(ups, (0, rows * 2304 - ups.numel())).view(rows, 2304)
    )
    expected_q, expected_s = _cuda_swiglu_int8_reference(x, clamp_limit)
    q, scale = torch.empty_like(expected_q), torch.empty_like(expected_s)
    kernel(x, q, scale)
    torch.accelerator.synchronize()
    assert torch.equal(q, expected_q)
    assert torch.equal(scale.view(torch.int32), expected_s.view(torch.int32))


def test_tilelang_swiglu_int8_graph_tracks_changed_input(tilelang_swiglu_int8):
    """Replay must read current input and overwrite all q/scale outputs."""
    kernel, clamp_limit = tilelang_swiglu_int8
    x = torch.zeros((6, 4608), device="cuda", dtype=torch.bfloat16)
    q = torch.empty((6, 2304), device="cuda", dtype=torch.int8)
    scale = torch.empty((6,), device="cuda", dtype=torch.float32)
    kernel(x, q, scale)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernel(x, q, scale)
    generator = torch.Generator(device="cuda").manual_seed(547)
    for magnitude in (0.0, 1.0, 8.0):
        x.copy_(
            torch.randn(x.shape, device="cuda", dtype=x.dtype, generator=generator)
            * magnitude
        )
        expected_q, expected_s = _cuda_swiglu_int8_reference(x, clamp_limit)
        q.fill_(-128)
        scale.fill_(float("nan"))
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(q, expected_q)
        assert torch.equal(scale.view(torch.int32), expected_s.view(torch.int32))
