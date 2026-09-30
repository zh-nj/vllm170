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
def test_w13_grouped_gemm_matches_sparse_reference(topk: int, tokens: int) -> None:
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
    routing_ids = _explicit_routing(tokens, topk, num_experts, device)

    routed, _, _, _, _ = _run_gemm(
        _W13, a_q, a_scale, b_mxfp4, b_shift2, channel_scale, routing_ids, 16, 128
    )
    _check_all_slots(_W13, a_q, a_scale, channel_scale, positions, routing_ids, routed)


@_NEEDS_SM80
@pytest.mark.parametrize("topk", [3, 6])
def test_w2_grouped_gemm_reads_slot_rows(topk: int) -> None:
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
    routing_ids = _explicit_routing(tokens, topk, num_experts, device)

    routed, _, _, _, _ = _run_gemm(
        _W2, a_q, a_scale, b_mxfp4, b_shift2, channel_scale, routing_ids, 16, 128
    )
    values = a_q[:, 0].tolist()
    for slot in range(num_routed):
        # weight code 0x07 -> magnitude 12, positive, shift 0.
        expected = torch.tensor(float(values[slot]) * 12.0).to(torch.bfloat16)
        assert routed[slot, 0].item() == expected.item(), f"slot={slot}"
        if slot >= topk:
            # The W13 rule would have read this row instead; it must differ, so
            # the assertion above is really testing the slot-row contract.
            token_row = torch.tensor(float(values[slot // topk]) * 12.0).to(
                torch.bfloat16
            )
            assert token_row.item() != expected.item(), f"slot={slot} is ambiguous"


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
