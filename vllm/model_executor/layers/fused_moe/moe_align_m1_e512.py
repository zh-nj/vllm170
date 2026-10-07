# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b)


@triton.jit
def moe_align_m1_e512_kernel(ids, sorted_ids, experts, total):
    route = tl.arange(0, 16)
    expert = tl.load(ids + route, route < 10, other=512)
    expert = tl.where((expert >= 0) & (expert < 512), expert, 512)
    key = tl.sort(expert * 16 + route, descending=False)
    expert = key // 16
    original = key % 16
    valid = expert < 512
    previous = tl.gather(expert, tl.maximum(route - 1, 0), axis=0)
    first = valid & ((route == 0) | (expert != previous))
    same = (expert[:, None] == expert[None, :]) & valid[None, :]
    count = tl.sum(same.to(tl.int32), axis=1)
    padded = tl.where(first, tl.cdiv(count, 8) * 8, 0)
    cumulative = tl.cumsum(padded, axis=0)
    offset = tl.associative_scan(
        tl.where(first, cumulative - padded, 0), axis=0, combine_fn=_maximum
    )
    within = tl.sum((same & (route[None, :] < route[:, None])).to(tl.int32), axis=1)
    position = offset + within
    out = tl.arange(0, 128)
    match = (out[:, None] == position[None, :]) & valid[None, :]
    token = tl.sum(tl.where(match, original[None, :], 0), axis=1)
    present = tl.sum(match.to(tl.int32), axis=1) > 0
    tl.store(sorted_ids + out, tl.where(present, token, 10), out < 80)
    block = route * 8
    selected = first[None, :] & (block[:, None] >= offset[None, :])
    selected &= block[:, None] < (offset + padded)[None, :]
    eid = tl.max(tl.where(selected, expert[None, :], -1), axis=1)
    tl.store(experts + route, eid, route < 10)
    tl.store(total, tl.sum(padded, axis=0))
