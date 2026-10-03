# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pair adjacent BM64 expert blocks into BM128, retaining BM64 odd tails."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _prefix_max(a, b):
    return tl.maximum(a, b)


@triton.jit
def _partition_blocks(experts, num_padded, pairs, tails, counts, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    n = tl.load(num_padded) // 64
    eid = tl.load(experts + i, i < n, other=-1)
    prev = tl.load(experts + i - 1, (i > 0) & (i < n), other=-1)
    nxt = tl.load(experts + i + 1, i + 1 < n, other=-2)
    start = tl.associative_scan(tl.where((i < n) & (eid != prev), i, 0), 0, _prefix_max)
    first = (i < n) & (((i - start) % 2) == 0)
    pair = first & (eid == nxt)
    tail = first & (eid != nxt)
    pair_dst = tl.cumsum(pair.to(tl.int32)) - 1
    tail_dst = tl.cumsum(tail.to(tl.int32)) - 1
    tl.store(pairs + pair_dst, i, pair)
    tl.store(tails + tail_dst, i, tail)
    tl.store(counts, tl.sum(pair.to(tl.int32)) * 128)
    tl.store(counts + 1, tl.sum(tail.to(tl.int32)) * 64)


@triton.jit
def _copy_blocks(
    src_ids,
    src_experts,
    indices,
    count,
    dst_ids,
    dst_experts,
    CAPACITY: tl.constexpr,
    B: tl.constexpr,
):
    tile = tl.program_id(0)
    n = tl.load(count) // B
    src_tile = tl.load(indices + tile, tile < n, other=0)
    lane = tl.arange(0, B)
    dst = tile * B + lane
    ids = tl.load(src_ids + src_tile * 64 + lane, tile < n, other=0)
    tl.store(dst_ids + dst, ids, (tile < n) & (dst < CAPACITY))
    expert = tl.load(src_experts + src_tile, tile < n, other=0)
    tl.store(dst_experts + tile, expert, tile < n)


class HybridRoutes:
    """Reusable scratch; update consumes aligned BM64 routes without a host read."""

    block_sizes: tuple[int, ...] = (128, 64)

    def __init__(self, slots: int, experts: int, device):
        self.slots = slots

        def allocate(b):
            capacity = slots * b if slots < experts else slots + experts * (b - 1)
            return (
                torch.empty(capacity, dtype=torch.int32, device=device),
                torch.empty(triton.cdiv(capacity, b), dtype=torch.int32, device=device),
            )

        self.large = allocate(128)
        self.small = allocate(64)
        self.indices = [torch.empty_like(self.small[1]) for _ in range(2)]
        self.counts = torch.empty(2, dtype=torch.int32, device=device)
        self.post_pad = [self.counts[i : i + 1] for i in range(2)]

    def update(self, aligned):
        ids, experts, post_pad = aligned
        _partition_blocks[(1,)](
            experts,
            post_pad,
            *self.indices,
            self.counts,
            BLOCK=triton.next_power_of_2(experts.numel()),
            num_warps=4,
        )
        for b, dest, index, count in zip(
            (128, 64),
            (self.large, self.small),
            self.indices,
            self.post_pad,
            strict=True,
        ):
            _copy_blocks[(dest[1].numel(),)](
                ids,
                experts,
                index,
                count,
                *dest,
                CAPACITY=dest[0].numel(),
                B=b,
                num_warps=4,
            )
        return self.metadata()

    def metadata(self):
        return [(*self.large, self.post_pad[0]), (*self.small, self.post_pad[1])]

    def gemm(
        self,
        a_q,
        a_scale,
        packed,
        shift2,
        channel_scale,
        output,
        routed_output,
        routing_weights,
        source_rows_are_slots,
    ):
        for block_m, metadata in zip(self.block_sizes, self.metadata(), strict=True):
            torch.ops._moe_C.dsv4_mxfp4_int8_sm80_moe_gemm(
                a_q,
                a_scale,
                packed,
                shift2,
                channel_scale,
                output,
                routed_output,
                routing_weights,
                *metadata,
                self.slots,
                block_m,
                128,
                source_rows_are_slots,
            )


@triton.jit
def _store_partition(selected, dest, count, B: tl.constexpr):
    dst = tl.cumsum(selected.to(tl.int32)) - 1
    tl.store(dest + dst, tl.arange(0, selected.shape[0]), selected)
    tl.store(count, tl.sum(selected.to(tl.int32)) * B)


@triton.jit
def _partition_small_tails(
    ids,
    experts,
    num_padded,
    pairs,
    tails,
    tiny,
    counts,
    SLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    n = tl.load(num_padded) // 64
    eid = tl.load(experts + i, i < n, other=-1)
    prev = tl.load(experts + i - 1, (i > 0) & (i < n), other=-1)
    nxt = tl.load(experts + i + 1, i + 1 < n, other=-2)
    start = tl.associative_scan(tl.where((i < n) & (eid != prev), i, 0), 0, _prefix_max)
    first = (i < n) & (((i - start) % 2) == 0)
    pair = first & (eid == nxt)
    tail = first & (eid != nxt)
    slot16 = tl.load(ids + i * 64 + 16, tail, other=SLOTS)
    small = tail & (slot16 >= SLOTS)
    tail = tail & ~small
    _store_partition(pair, pairs, counts, 128)
    _store_partition(tail, tails, counts + 1, 64)
    _store_partition(small, tiny, counts + 2, 16)


class SmallTailRoutes(HybridRoutes):
    """BM16 odd tails with at most 16 valid slots."""

    block_sizes = (128, 64, 16)

    def __init__(self, slots, experts, device):
        super().__init__(slots, experts, device)
        capacity = slots * 16 if slots < experts else slots + experts * 15
        self.tiny = (
            torch.empty(capacity, dtype=torch.int32, device=device),
            torch.empty(triton.cdiv(capacity, 16), dtype=torch.int32, device=device),
        )
        self.indices.append(torch.empty_like(self.small[1]))
        self.counts = torch.empty(3, dtype=torch.int32, device=device)
        self.post_pad = [self.counts[i : i + 1] for i in range(3)]

    def update(self, aligned):
        ids, experts, post_pad = aligned
        _partition_small_tails[(1,)](
            ids,
            experts,
            post_pad,
            *self.indices,
            self.counts,
            SLOTS=self.slots,
            BLOCK=triton.next_power_of_2(experts.numel()),
            num_warps=4,
        )
        for b, dest, index, count in zip(
            self.block_sizes,
            (self.large, self.small, self.tiny),
            self.indices,
            self.post_pad,
            strict=True,
        ):
            _copy_blocks[(dest[1].numel(),)](
                ids,
                experts,
                index,
                count,
                *dest,
                CAPACITY=dest[0].numel(),
                B=b,
                num_warps=4,
            )
        return self.metadata()

    def metadata(self):
        return [
            (*self.large, self.post_pad[0]),
            (*self.small, self.post_pad[1]),
            (*self.tiny, self.post_pad[2]),
        ]
