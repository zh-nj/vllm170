# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline BM128 pairs plus BM64 tails, preserving BM64 padding and slot IDs."""

import argparse
import heapq
import json
from pathlib import Path

import torch

from vllm.model_executor.layers.fused_moe.experts.dsv41_sm80_hybrid_prefill import (
    HybridRoutes,
    SmallTailRoutes,
    _prefix_max,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _emit_metadata(
    src_ids, src_experts, src, selected, dest, dst_ids, dst_experts, B: tl.constexpr
):
    lane = tl.arange(0, B)
    ids = tl.load(
        src_ids + src[:, None] * 64 + lane[None, :], selected[:, None], other=0
    )
    tl.store(dst_ids + dest[:, None] * B + lane[None, :], ids, selected[:, None])
    expert = tl.load(src_experts + src, selected, other=0)
    tl.store(dst_experts + dest, expert, selected)


@triton.jit
def _fused_metadata(
    ids,
    experts,
    post_pad,
    counts,
    large_ids,
    large_experts,
    small_ids,
    small_experts,
    tiny_ids,
    tiny_experts,
    SLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUP: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    n = tl.load(post_pad) // 64
    eid = tl.load(experts + i, i < n, other=-1)
    prev = tl.load(experts + i - 1, (i > 0) & (i < n), other=-1)
    nxt = tl.load(experts + i + 1, i + 1 < n, other=-2)
    start = tl.associative_scan(tl.where((i < n) & (eid != prev), i, 0), 0, _prefix_max)
    first = (i < n) & (((i - start) % 2) == 0)
    pair = first & (eid == nxt)
    tail = first & (eid != nxt)
    slot16 = tl.load(ids + i * 64 + 16, tail, other=SLOTS)
    tiny = tail & (slot16 >= SLOTS)
    tail = tail & ~tiny
    source = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    pair_dst = tl.cumsum(pair.to(tl.int32)) - 1
    tail_dst = tl.cumsum(tail.to(tl.int32)) - 1
    tiny_dst = tl.cumsum(tiny.to(tl.int32)) - 1
    _emit_metadata(
        ids,
        experts,
        source,
        tl.gather(pair.to(tl.int32), source, 0) != 0,
        tl.gather(pair_dst, source, 0),
        large_ids,
        large_experts,
        128,
    )
    _emit_metadata(
        ids,
        experts,
        source,
        tl.gather(tail.to(tl.int32), source, 0) != 0,
        tl.gather(tail_dst, source, 0),
        small_ids,
        small_experts,
        64,
    )
    _emit_metadata(
        ids,
        experts,
        source,
        tl.gather(tiny.to(tl.int32), source, 0) != 0,
        tl.gather(tiny_dst, source, 0),
        tiny_ids,
        tiny_experts,
        16,
    )
    if tl.program_id(0) == 0:
        tl.store(counts, tl.sum(pair.to(tl.int32)) * 128)
        tl.store(counts + 1, tl.sum(tail.to(tl.int32)) * 64)
        tl.store(counts + 2, tl.sum(tiny.to(tl.int32)) * 16)


class FusedMetadataRoutes(SmallTailRoutes):
    """Experimental single-launch partition and copies of accepted route lists."""

    def update(self, aligned):
        ids, experts, post_pad = aligned
        group = min(8, triton.next_power_of_2(experts.numel()))
        _fused_metadata[(triton.cdiv(experts.numel(), group),)](
            ids,
            experts,
            post_pad,
            self.counts,
            *self.large,
            *self.small,
            *self.tiny,
            SLOTS=self.slots,
            BLOCK=triton.next_power_of_2(experts.numel()),
            GROUP=group,
            num_warps=4,
        )
        return self.metadata()


def histogram_routes(counts, topk=6, seed=41):
    """Match captured occupancy; this does not reconstruct the captured row order."""
    tokens, remainder = divmod(sum(counts), topk)
    if remainder or max(counts, default=0) > tokens:
        raise ValueError("histogram cannot represent unique top-k token routes")
    heap = [(-c, e) for e, c in enumerate(counts) if c]
    heapq.heapify(heap)
    rows = []
    for _ in range(tokens):
        if len(heap) < topk:
            raise ValueError("histogram has too few remaining experts")
        chosen = [heapq.heappop(heap) for _ in range(topk)]
        rows.append([e for _, e in chosen])
        for c, e in chosen:
            if c < -1:
                heapq.heappush(heap, (c + 1, e))
    assert not heap
    result = torch.tensor(rows, dtype=torch.int32)
    generator = torch.Generator().manual_seed(seed)
    result = result[torch.randperm(tokens, generator=generator)]
    result = result.gather(1, torch.rand(result.shape, generator=generator).argsort(1))
    assert (
        torch.bincount(result.flatten().long(), minlength=len(counts)).tolist()
        == counts
    )
    return result.contiguous()


def check_partition(ids, aligned, hybrid):
    """Every valid original slot appears once, with its original expert."""
    flat = ids.flatten().cpu()
    seen = []
    total = 0
    for b, (sorted_ids, expert_ids, count) in zip(
        getattr(hybrid, "block_sizes", (128, 64)), hybrid.metadata(), strict=True
    ):
        n = int(count.item())
        total += n
        blocks = sorted_ids[:n].cpu().view(-1, b)
        experts = expert_ids[: n // b].cpu()
        for slots, expert in zip(blocks, experts, strict=True):
            valid = slots[slots < flat.numel()].long()
            assert bool((flat[valid] == expert).all())
            seen.extend(valid.tolist())
    assert sorted(seen) == list(range(flat.numel()))
    expected = int(aligned[2].item())
    if isinstance(hybrid, SmallTailRoutes):
        expected -= int(hybrid.counts[2].item()) // 16 * 48
    assert total == expected


def gemm(a, scales, weights, route_weights, output, metadata, b, stage):
    torch.ops._moe_C.dsv4_mxfp4_int8_sm80_moe_gemm(
        a,
        scales,
        weights["packed"],
        weights["shift2"],
        weights["channel_scale"],
        output,
        output,
        route_weights,
        *metadata,
        route_weights.numel(),
        b,
        128,
        stage == "w2",
    )


def make_calls(
    a,
    scales,
    weights,
    rw,
    baseline_out,
    candidate_out,
    aligned,
    hybrid,
    stage,
    baseline_hybrid=None,
):
    def baseline():
        if baseline_hybrid is None:
            gemm(a, scales, weights, rw, baseline_out, aligned, 64, stage)
        else:
            baseline_hybrid.update(aligned)
            for b, meta in zip(
                baseline_hybrid.block_sizes, baseline_hybrid.metadata(), strict=True
            ):
                gemm(a, scales, weights, rw, baseline_out, meta, b, stage)

    def candidate():
        hybrid.update(aligned)
        for b, meta in zip(
            getattr(hybrid, "block_sizes", (128, 64)), hybrid.metadata(), strict=True
        ):
            gemm(a, scales, weights, rw, candidate_out, meta, b, stage)

    return baseline, candidate


def graph_validator(output, expected):
    def validate(graph):
        output.fill_(float("nan"))
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16))

    return validate


def main():
    from deepseek_v41_moe_harness import (
        STAGE_SHAPES,
        align_routes,
        environment,
        interleaved_ab,
        reference_stage,
    )
    from deepseek_v41_moe_tilelang_gemm import activation_scales, build_packed_weights

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--round2",
        type=Path,
        default=Path("artifacts/dsv41f-ced-prefill-round2-20261002"),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--small-tail", action="store_true", help="Compare BM16 tails to current hybrid"
    )
    mode.add_argument(
        "--fused-metadata",
        action="store_true",
        help="Fold partition and copies into one launch",
    )
    parser.add_argument(
        "--round5",
        type=Path,
        default=Path("artifacts/dsv41f-ced-prefill-round5-20261003"),
    )
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 14])
    parser.add_argument("--expected-uuid", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    assert torch.cuda.get_device_capability() == (8, 0)
    assert str(
        torch.cuda.get_device_properties(0).uuid
    ) == args.expected_uuid.removeprefix("GPU-")
    info = environment()
    # The shared harness uses NVML index 0, which is not the visible CMP GPU.
    info.pop("clock_and_power", None)
    result = {
        "environment": info,
        "variant": "fused_metadata"
        if args.fused_metadata
        else "small_tail"
        if args.small_tail
        else "hybrid",
        "timing_boundary": "route repartition/copies plus one W13 or W2 GEMM stage",
        "correctness": [],
        "timing": [],
    }
    route_class = (
        FusedMetadataRoutes
        if args.fused_metadata
        else SmallTailRoutes
        if args.small_tail
        else HybridRoutes
    )
    captured_dspark = args.small_tail or args.fused_metadata
    torch.manual_seed(41)

    def save():
        args.out.write_text(json.dumps(result, indent=2) + "\n")

    for stage, (n, k, slots_source) in STAGE_SHAPES.items():
        weights = build_packed_weights(8, stage, device, seed=41, scale_mode="varied")
        for tokens in (1, 16, 17, 63, 64, 65, 127, 128, 129):
            ids = (
                torch.arange(6, dtype=torch.int32, device=device)
                .expand(tokens, -1)
                .contiguous()
            )
            # Odd/even route blocks, zero large-list, zero small-list, and both.
            aligned = align_routes(ids, 64, 8)
            hybrid = route_class(ids.numel(), 8, device)
            hybrid.update(aligned)
            check_partition(ids, aligned, hybrid)
            rows = tokens * 6 if slots_source else tokens
            a = torch.randint(-127, 128, (rows, k), dtype=torch.int8, device=device)
            scales = activation_scales(rows, device, "varied")
            rw = torch.full((tokens, 6), 1 / 6, dtype=torch.float32, device=device)
            output = torch.full(
                (tokens * 6 + 32, n), -123, dtype=torch.bfloat16, device=device
            )
            routed = output[: tokens * 6]
            reference = reference_stage(
                a,
                scales,
                **{
                    "b_mxfp4": weights["packed"],
                    "b_shift2": weights["shift2"],
                    "b_channel_scale": weights["channel_scale"],
                },
                routing_ids=ids,
                stage=stage,
            )
            for b, metadata in zip(
                getattr(hybrid, "block_sizes", (128, 64)),
                hybrid.metadata(),
                strict=True,
            ):
                gemm(a, scales, weights, rw, routed, metadata, b, stage)
            assert torch.equal(routed.view(torch.int16), reference.view(torch.int16))
            assert bool((output[tokens * 6 :] == -123).all())
            result["correctness"].append(
                {"stage": stage, "tokens": tokens, "bitwise_exact": True}
            )
            save()
            print("correct", stage, tokens, flush=True)
        del weights
        torch.accelerator.empty_cache()
        weights = build_packed_weights(384, stage, device, seed=41, scale_mode="varied")
        exp = json.loads((args.round2 / "experiment.json").read_text())
        profiles = {
            "periodic": next(p for p in exp["profiles"] if p["group"] == "moe"),
            "corpus": json.loads(
                (args.round2 / "candidate-validation.json").read_text()
            )["corpus_routes"],
        }
        if captured_dspark:
            profiles = {
                "corpus": next(
                    p
                    for p in json.loads((args.round5 / "experiment.json").read_text())[
                        "profiles"
                    ]
                    if p["mode"] == "hybrid"
                ),
                "hot": {
                    "ranks": [
                        {
                            "routes": [
                                {
                                    "layer": 0,
                                    "ids": [list(range(6))] * 2044,
                                    "weights": [[1 / 6] * 6] * 2044,
                                    "counts": [2044] * 6 + [0] * 378,
                                    "active_experts": 6,
                                }
                            ]
                        }
                    ]
                },
            }
        for workload, profile in profiles.items():
            for layer in [0] if workload == "hot" else args.layers:
                route = next(
                    r
                    for rank in profile["ranks"]
                    for r in rank["routes"]
                    if r["layer"] == layer
                )
                ids = (
                    torch.tensor(route["ids"], dtype=torch.int32, device=device)
                    if captured_dspark
                    else histogram_routes(route["counts"]).to(device)
                )
                assert ids.shape == ((2044 if captured_dspark else 2048), 6)
                assert (
                    torch.bincount(ids.flatten().long(), minlength=384).cpu().tolist()
                    == route["counts"]
                )
                aligned = align_routes(ids, 64, 384)
                hybrid = route_class(ids.numel(), 384, device)
                hybrid.update(aligned)
                check_partition(ids, aligned, hybrid)
                rows = ids.numel() if slots_source else ids.shape[0]
                a = torch.randint(-127, 128, (rows, k), dtype=torch.int8, device=device)
                scales = activation_scales(rows, device, "varied")
                rw = torch.full(ids.shape, 1 / 6, dtype=torch.float32, device=device)
                if captured_dspark:
                    rw.copy_(torch.tensor(route["weights"], device=device))
                baseline_out = torch.empty(
                    (ids.numel(), n), dtype=torch.bfloat16, device=device
                )
                candidate_out = torch.empty_like(baseline_out)

                baseline, candidate = make_calls(
                    a,
                    scales,
                    weights,
                    rw,
                    baseline_out,
                    candidate_out,
                    aligned,
                    hybrid,
                    stage,
                    SmallTailRoutes(ids.numel(), 384, device)
                    if args.fused_metadata
                    else HybridRoutes(ids.numel(), 384, device)
                    if args.small_tail
                    else None,
                )
                baseline()
                candidate()
                assert torch.equal(
                    candidate_out.view(torch.int16), baseline_out.view(torch.int16)
                )
                expected = baseline_out.clone()
                timing = interleaved_ab(
                    baseline,
                    candidate,
                    rounds=3,
                    warmup=2,
                    graph_calls=4,
                    samples=7,
                    settle_ms=100,
                    baseline_validator=graph_validator(baseline_out, expected),
                    candidate_validator=graph_validator(candidate_out, expected),
                )
                result["timing"].append(
                    {
                        "stage": stage,
                        "workload": workload,
                        "layer": layer,
                        "active_experts": route["active_experts"],
                        "routing_scope": (
                            "DSpark5 M2044 captured corpus routes/weights or synthetic "
                            "six-hot-expert routes; synthetic activations/weights."
                            if captured_dspark
                            else "Captured histogram; synthetic row order, "
                            "activations and weights."
                        ),
                        "counts": hybrid.counts.cpu().tolist(),
                        "timing": timing,
                        "bitwise_matches_bm64": True,
                    }
                )
                save()
                print("timing", stage, workload, layer, timing, flush=True)
        del weights
        torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
