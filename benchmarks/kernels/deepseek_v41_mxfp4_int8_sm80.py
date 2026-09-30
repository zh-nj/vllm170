# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark DSV4.1-Flash SM80 MXFP4-int8 expert GEMM tiles.

Scans the W13/W2 grouped GEMM stages and the routing-weighted reduce of the
``mxfp4_int8_sm80`` backend through the production ``_moe_C`` entry points.

The scan axes are the ones the DSV4.1-F serving shape actually exercises:
expert count, top-k (3 for DSpark draft, 6 for the target model), routing
pattern, cache mode and tile. ``--check-reference`` adds an independent
numerical reference (see ``deepseek_v41_moe_harness``) that decodes the actual
packed weights; a failing case does not get its performance measured and makes
the process exit non-zero.

This file stays an offline facility: it does not change production defaults,
backend selection or dispatch.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deepseek_v41_moe_harness import (  # noqa: E402
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    REDUCE_STAGE,
    STAGE_SHAPES,
    SUPPORTED_TILES,
    SUPPORTED_TOPK,
    W2_STAGE,
    W13_STAGE,
    Comparison,
    JsonlWriter,
    ResultRow,
    Threshold,
    align_routes,
    classify,
    compare,
    compute_route_work,
    dirty_diff_hash,
    environment,
    git_head,
    make_routing_ids,
    reduce_term_scale,
    reference_reduce,
    reference_stage,
    route_hash,
    stage_weight_bytes,
    time_graph,
    write_summary,
)

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)

HIDDEN_SIZE = HIDDEN_SIZE
INTERMEDIATE_SIZE = INTERMEDIATE_SIZE
NUM_EXPERTS = 384
TOPK = 6

# The GEMM stages carry unit scales, so the reference demands bitwise equality
# there. The reduce cannot: 1/topk is not representable in fp32 and the kernel
# contracts `acc + value * weight` to an FFMA, so the accumulator differs from a
# separately rounded one by about an fp32 ulp. That only changes the answer when
# the exact sum sits within that distance of a bf16 rounding boundary, where the
# two bf16 results are adjacent representable values.
#
# Measured on CMP 170HX (artifacts/moe-grouped-gemm-harness-20260930-164205):
# across 8 reference-checked cases 98.7-99.4% of reduce elements are
# bit-identical; every remaining mismatch is one bf16 step plus at most
# 1.0e-9 of the term scale, i.e. pure fp32-accumulation disagreement from the
# kernel's FFMA contraction. The budget is 2**-20 of the term scale, which is
# ~500x the observed residual and ~1e6x tighter than a routing or weighting
# mistake (those move the result by order 1 of the term scale).
REDUCE_THRESHOLD = Threshold(
    threshold_id="reduce-fp32-accumulation-slack-20260930",
    calc_diff=1e-6,
    max_step_excess_over_term_scale=2.0**-20,
    calibrated=True,
    calibration_run="moe-grouped-gemm-harness-20260930-164205",
    note=(
        "residual after subtracting one bf16 step must stay under 2**-20 of "
        "the term scale; observed worst case 1.9e-9 over 40960+ elements"
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[232, 256])
    parser.add_argument(
        "--experts",
        type=int,
        nargs="+",
        default=[NUM_EXPERTS],
        help="expert counts to scan; production DSV4.1-F uses 384",
    )
    parser.add_argument(
        "--topk",
        type=int,
        nargs="+",
        default=[TOPK],
        choices=list(SUPPORTED_TOPK),
        help="top-k values; 3 is the DSpark draft, 6 the target model",
    )
    parser.add_argument(
        "--route-pattern",
        nargs="+",
        default=["uniform"],
        choices=["uniform", "hot", "sparse"],
        help="routing distributions; 'hot' concentrates on a fixed expert set",
    )
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--block-m", type=int, nargs="+", default=[16])
    parser.add_argument("--block-n", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--graph-calls", type=int, default=40)
    parser.add_argument(
        "--graph-samples",
        type=int,
        default=100,
        help="Event-bracketed graph replays per timed case",
    )
    parser.add_argument(
        "--check-reference",
        action="store_true",
        help="verify each stage against the independent reference; a failure "
        "skips that case's timing and makes the process exit non-zero",
    )
    parser.add_argument("--reference-chunk-n", type=int, default=512)
    parser.add_argument(
        "--cache-mode",
        choices=["warm-fixed", "rotating-experts"],
        default="warm-fixed",
        help="'rotating-experts' walks a different expert set every graph call "
        "and currently requires --tokens 1",
    )
    parser.add_argument(
        "--rotate-experts",
        action="store_true",
        help="compatibility alias for --cache-mode rotating-experts",
    )
    parser.add_argument("--output", type=Path, help="legacy JSON result file")
    parser.add_argument("--output-jsonl", type=Path, help="JSONL result file")
    parser.add_argument("--summary-json", type=Path, help="atomic run summary")
    parser.add_argument(
        "--quiet", action="store_true", help="do not echo the legacy JSON"
    )
    args = parser.parse_args()
    if args.rotate_experts:
        if args.cache_mode != "warm-fixed":
            parser.error("--rotate-experts conflicts with --cache-mode")
        args.cache_mode = "rotating-experts"
    return args


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def measure(call, warmup: int, repeats: int) -> dict[str, float]:
    """Eager Event timing. Diagnostic only; never used to rank graph candidates."""
    for _ in range(warmup):
        call()
    torch.accelerator.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        call()
        end.record()
    torch.accelerator.synchronize()
    samples = [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]
    return {
        "mean_ms": statistics.fmean(samples),
        "p50_ms": statistics.median(samples),
        "p90_ms": percentile(samples, 0.9),
    }


def capture(call: Callable[[], None], warmup: int, graph_calls: int):
    for _ in range(warmup):
        call()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_calls):
            call()

    return graph.replay


def capture_many(calls: list[Callable[[], None]], warmup: int):
    for _ in range(warmup):
        for call in calls:
            call()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()

    return graph.replay


def make_gemm_call(
    a_q: torch.Tensor,
    a_scale: torch.Tensor,
    weight: torch.Tensor,
    shift2: torch.Tensor,
    channel_scale: torch.Tensor,
    output: torch.Tensor,
    routed_out: torch.Tensor,
    routing_weights: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_post_pad: torch.Tensor,
    num_routed: int,
    block_m: int,
    block_n: int,
    source_rows_are_slots: bool,
) -> Callable[[], None]:
    def call() -> None:
        torch.ops._moe_C.dsv4_mxfp4_int8_sm80_moe_gemm(
            a_q,
            a_scale,
            weight,
            shift2,
            channel_scale,
            output,
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

    return call


def make_reduce_call(
    routed_out: torch.Tensor,
    routing_weights: torch.Tensor,
    output: torch.Tensor,
    num_routed: int,
) -> Callable[[], None]:
    def call() -> None:
        torch.ops._moe_C.dsv4_mxfp4_int8_sm80_moe_reduce(
            routed_out,
            routing_weights,
            output,
            num_routed,
        )

    return call


def build_weights(num_experts: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Synthetic persistent weights in the production packed layout.

    Random packed codes and shift fields give decoded INT8 weights over the full
    [-96, 96] range, which is what the kernel actually multiplies. Unit channel
    scales keep the epilogue exact so the reference can demand bitwise equality.
    """
    return {
        "w13": torch.empty(
            (num_experts, HIDDEN_SIZE // 32, INTERMEDIATE_SIZE * 2 // 8, 128),
            dtype=torch.uint8,
            device=device,
        ).random_(0, 256),
        "w13_shift2": torch.empty(
            (num_experts, HIDDEN_SIZE // 128, INTERMEDIATE_SIZE * 2 // 8, 8),
            dtype=torch.uint8,
            device=device,
        ).random_(0, 256),
        "w13_scale": torch.ones(
            (num_experts, INTERMEDIATE_SIZE * 2), dtype=torch.float32, device=device
        ),
        "w2": torch.empty(
            (num_experts, INTERMEDIATE_SIZE // 32, HIDDEN_SIZE // 8, 128),
            dtype=torch.uint8,
            device=device,
        ).random_(0, 256),
        "w2_shift2": torch.empty(
            (num_experts, INTERMEDIATE_SIZE // 128, HIDDEN_SIZE // 8, 8),
            dtype=torch.uint8,
            device=device,
        ).random_(0, 256),
        "w2_scale": torch.ones(
            (num_experts, HIDDEN_SIZE), dtype=torch.float32, device=device
        ),
    }


def check_tile_equivalence(
    a13_q: torch.Tensor,
    a13_scale: torch.Tensor,
    w13: torch.Tensor,
    w13_shift2: torch.Tensor,
    w13_scale: torch.Tensor,
    a2_q: torch.Tensor,
    a2_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_shift2: torch.Tensor,
    w2_scale: torch.Tensor,
    output: torch.Tensor,
    routing_weights: torch.Tensor,
    routing_ids: torch.Tensor,
    num_routed: int,
    tiles: list[tuple[int, int]],
    num_experts: int,
) -> None:
    """Every tile must reproduce the 16x128 tile bit for bit.

    This catches tile-to-tile disagreement only; it cannot see a mistake shared
    by every tile, which is what ``--check-reference`` is for.
    """
    gate_up_reference = torch.empty(
        (num_routed, INTERMEDIATE_SIZE * 2), dtype=torch.bfloat16, device="cuda"
    )
    routed_out_reference = torch.empty(
        (num_routed, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
    )
    gate_up_candidate = torch.empty_like(gate_up_reference)
    routed_out_candidate = torch.empty_like(routed_out_reference)

    reference_tile = (16, 128)
    validation_tiles = [
        reference_tile,
        *(tile for tile in tiles if tile != reference_tile),
    ]
    for tile_index, (block_m, block_n) in enumerate(validation_tiles):
        sorted_ids, expert_ids, num_post_pad = moe_align_block_size(
            routing_ids, block_m, num_experts
        )
        gate_up = gate_up_reference if tile_index == 0 else gate_up_candidate
        routed_out = routed_out_reference if tile_index == 0 else routed_out_candidate
        make_gemm_call(
            a13_q,
            a13_scale,
            w13,
            w13_shift2,
            w13_scale,
            gate_up,
            gate_up,
            routing_weights,
            sorted_ids,
            expert_ids,
            num_post_pad,
            num_routed,
            block_m,
            block_n,
            False,
        )()
        make_gemm_call(
            a2_q,
            a2_scale,
            w2,
            w2_shift2,
            w2_scale,
            output,
            routed_out,
            routing_weights,
            sorted_ids,
            expert_ids,
            num_post_pad,
            num_routed,
            block_m,
            block_n,
            True,
        )()
        torch.accelerator.synchronize()
        if tile_index > 0:
            torch.testing.assert_close(
                gate_up_candidate, gate_up_reference, rtol=0, atol=0
            )
            torch.testing.assert_close(
                routed_out_candidate, routed_out_reference, rtol=0, atol=0
            )


def verify_stages(
    stage_tensors: dict[str, dict[str, torch.Tensor]],
    routing_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    chunk_n: int,
) -> dict[str, tuple[Comparison, bool]]:
    """Run the independent reference for W13, W2 and the reduce.

    Returns ``stage -> (comparison, exact_required)``. The GEMM stages use unit
    channel scales, so the reference can demand bitwise equality: both sides
    round the same exact integer through ``int32 -> float32 -> bfloat16``.
    """
    num_routed = int(routing_ids.numel())
    results: dict[str, tuple[Comparison, bool]] = {}
    for stage in (W13_STAGE, W2_STAGE):
        tensors = stage_tensors[stage]
        expected = reference_stage(
            tensors["a_q"],
            tensors["a_scale"],
            tensors["weight"],
            tensors["shift2"],
            tensors["channel_scale"],
            routing_ids,
            stage=stage,
            chunk_n=chunk_n,
        )
        observed = tensors["routed_out"][:num_routed]
        unit_scales = bool(
            torch.all(tensors["channel_scale"] == 1.0).item()
            and torch.all(tensors["a_scale"] == 1.0).item()
        )
        results[stage] = (
            compare(
                observed,
                expected,
                slot_experts=routing_ids.reshape(-1),
                rows=num_routed,
            ),
            unit_scales,
        )
    # The reduce always multiplies by 1/topk, which float32 cannot represent,
    # so the kernel's FFMA contraction makes bitwise equality the wrong demand.
    expected_reduce = reference_reduce(
        stage_tensors[W2_STAGE]["routed_out"], routing_weights
    )
    observed_reduce = stage_tensors[W2_STAGE]["output"]
    results[REDUCE_STAGE] = (
        compare(
            observed_reduce,
            expected_reduce,
            rows=routing_weights.shape[0],
            term_scale=reduce_term_scale(
                stage_tensors[W2_STAGE]["routed_out"], routing_weights
            ),
            measure_bf16_steps=True,
        ),
        False,
    )
    return results


def build_stage_tensors(
    weights: dict[str, torch.Tensor],
    *,
    tokens: int,
    topk: int,
    num_routed: int,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    """Activation, weight and output tensors for both GEMM stages."""
    gate_up = torch.empty(
        (num_routed, INTERMEDIATE_SIZE * 2), dtype=torch.bfloat16, device=device
    )
    routed_out = torch.empty(
        (num_routed, HIDDEN_SIZE), dtype=torch.bfloat16, device=device
    )
    output = torch.empty((tokens, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
    a13_q = torch.empty((tokens, HIDDEN_SIZE), dtype=torch.int8, device=device)
    a13_q.random_(-127, 128)
    a2_q = torch.empty((num_routed, INTERMEDIATE_SIZE), dtype=torch.int8, device=device)
    a2_q.random_(-127, 128)
    return {
        W13_STAGE: {
            "a_q": a13_q,
            "a_scale": torch.ones((tokens,), dtype=torch.float32, device=device),
            "weight": weights["w13"],
            "shift2": weights["w13_shift2"],
            "channel_scale": weights["w13_scale"],
            "output": gate_up,
            "routed_out": gate_up,
        },
        W2_STAGE: {
            "a_q": a2_q,
            "a_scale": torch.ones((num_routed,), dtype=torch.float32, device=device),
            "weight": weights["w2"],
            "shift2": weights["w2_shift2"],
            "channel_scale": weights["w2_scale"],
            "output": output,
            "routed_out": routed_out,
        },
    }


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.repeats <= 0 or args.graph_calls <= 0:
        raise ValueError(
            "warmup must be non-negative; repeats and graph-calls must be positive"
        )
    if any(tokens <= 0 for tokens in args.tokens):
        raise ValueError("tokens must be positive")
    if any(experts <= 0 for experts in args.experts):
        raise ValueError("experts must be positive")
    if any(block_m not in (16, 64, 128) for block_m in args.block_m):
        raise ValueError("block-m must contain only supported 16, 64, or 128 tiles")
    if any(block_n not in (64, 128) for block_n in args.block_n):
        raise ValueError("block-n must contain only supported 64 or 128 tiles")
    tiles = [
        (block_m, block_n)
        for block_m in args.block_m
        for block_n in args.block_n
        if (block_m, block_n) in SUPPORTED_TILES
    ]
    if not tiles:
        raise ValueError("no supported block-m/block-n tile combinations requested")
    requested = {
        (block_m, block_n) for block_m in args.block_m for block_n in args.block_n
    }
    unsupported = sorted(requested - set(SUPPORTED_TILES))
    if unsupported:
        raise ValueError(
            f"unsupported tile(s) {unsupported}; supported tiles are "
            f"{sorted(SUPPORTED_TILES)}"
        )
    if args.cache_mode == "rotating-experts":
        if any(tokens != 1 for tokens in args.tokens):
            raise ValueError(
                "--cache-mode rotating-experts currently requires --tokens 1; "
                "refusing to generalise the capture instead of testing it"
            )
        if len(args.experts) != 1:
            raise ValueError(
                "--cache-mode rotating-experts supports a single --experts value"
            )

    device = torch.device("cuda")
    run_id = f"{git_head() or 'unknown'}-{dirty_diff_hash() or 'clean'}"
    env = environment()
    head = git_head()
    diff_hash = dirty_diff_hash()
    command = " ".join([sys.executable, *sys.argv])
    rows: list[dict] = []
    legacy: list[dict] = []
    failures: list[str] = []

    with JsonlWriter(args.output_jsonl) as writer:
        for num_experts in args.experts:
            weights = build_weights(num_experts, device)
            for topk in args.topk:
                for pattern in args.route_pattern:
                    for tokens in args.tokens:
                        routing_ids = make_routing_ids(
                            tokens, topk, num_experts, pattern, args.seed
                        ).to(device)
                        routing_weights = torch.full(
                            (tokens, topk),
                            1.0 / topk,
                            dtype=torch.float32,
                            device=device,
                        )
                        num_routed = tokens * topk
                        case_id = (
                            f"E{num_experts}-k{topk}-M{tokens}-{pattern}-"
                            f"{args.cache_mode}"
                        )
                        case_rows, case_legacy, case_failed = run_case(
                            args=args,
                            case_id=case_id,
                            tokens=tokens,
                            topk=topk,
                            num_experts=num_experts,
                            pattern=pattern,
                            routing_ids=routing_ids,
                            routing_weights=routing_weights,
                            num_routed=num_routed,
                            weights=weights,
                            tiles=tiles,
                            device=device,
                            run_id=run_id,
                            env=env,
                            head=head,
                            diff_hash=diff_hash,
                            command=command,
                        )
                        for row in case_rows:
                            writer.write(row)
                        rows.extend(case_rows)
                        legacy.extend(case_legacy)
                        if case_failed:
                            failures.append(case_id)

    serialized = json.dumps(legacy, indent=2, sort_keys=True)
    if not args.quiet:
        print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    if args.summary_json is not None:
        write_summary(
            args.summary_json,
            rows,
            run_id=run_id,
            git_head=head,
            dirty_diff_hash=diff_hash,
            failures=failures,
            note=(
                "Synthetic packed weights; no model checkpoint was loaded. "
                "This is not a model-accuracy result."
            ),
        )
    if failures:
        raise SystemExit(
            f"reference check failed for {len(failures)} case(s): {failures}"
        )


def run_production_once(
    stage_tensors: dict[str, dict[str, torch.Tensor]],
    routing_weights: torch.Tensor,
    routing_ids: torch.Tensor,
    num_routed: int,
    num_experts: int,
    block_m: int,
    block_n: int,
) -> None:
    """Enqueue one W13 -> W2 -> reduce sequence, as production orders them.

    The activation and the second A8 quantization between the two GEMMs are out
    of scope for this round; the two stages are checked against their own inputs.
    """
    for stage, source_rows_are_slots in ((W13_STAGE, False), (W2_STAGE, True)):
        tensors = stage_tensors[stage]
        make_gemm_call(
            tensors["a_q"],
            tensors["a_scale"],
            tensors["weight"],
            tensors["shift2"],
            tensors["channel_scale"],
            tensors["output"],
            tensors["routed_out"],
            routing_weights,
            *align_routes(routing_ids, block_m, num_experts),
            num_routed,
            block_m,
            block_n,
            source_rows_are_slots,
        )()
    make_reduce_call(
        stage_tensors[W2_STAGE]["routed_out"],
        routing_weights,
        stage_tensors[W2_STAGE]["output"],
        num_routed,
    )()
    torch.accelerator.synchronize()


def rotation_routing_sequence(
    topk: int, num_experts: int, graph_calls: int, device: torch.device
) -> list[torch.Tensor]:
    """One token walking ``topk`` new experts per graph call.

    The period is ``graph_calls`` calls; every call uses a distinct expert set,
    so the union is ``graph_calls * topk`` experts (modulo wraparound).
    """
    sequence = []
    for call_index in range(graph_calls):
        ids = (
            (torch.arange(topk, dtype=torch.int32, device=device) + call_index * topk)
            .remainder(num_experts)
            .view(1, topk)
        )
        sequence.append(ids)
    return sequence


def rotation_accounting(
    weights: dict[str, torch.Tensor],
    sequence: list[torch.Tensor],
    block_m: int,
) -> tuple[dict[str, dict[str, int]], int, int]:
    """Per-call logical bytes and working set over one rotation period.

    The byte numerator is normalised by the period exactly like the latency
    denominator, so ``active_weight_gb_s`` stays a per-call figure.
    """
    union = torch.unique(torch.cat([ids.reshape(-1) for ids in sequence]).cpu())
    experts = [int(e) for e in union.tolist()]
    union_bytes = {
        W13_STAGE: stage_weight_bytes(
            weights["w13"], weights["w13_shift2"], weights["w13_scale"], experts
        ),
        W2_STAGE: stage_weight_bytes(
            weights["w2"], weights["w2_shift2"], weights["w2_scale"], experts
        ),
    }
    period = len(sequence)
    per_call = {
        stage: {key: value // period for key, value in detail.items()}
        for stage, detail in union_bytes.items()
    }
    working_set = sum(detail["total"] for detail in union_bytes.values())
    return per_call, period, working_set


def run_case(
    *,
    args: argparse.Namespace,
    case_id: str,
    tokens: int,
    topk: int,
    num_experts: int,
    pattern: str,
    routing_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    num_routed: int,
    weights: dict[str, torch.Tensor],
    tiles: list[tuple[int, int]],
    device: torch.device,
    run_id: str,
    env: dict,
    head: str | None,
    diff_hash: str | None,
    command: str,
) -> tuple[list[dict], list[dict], bool]:
    """Check, then measure, one (E, topk, M, pattern, cache-mode) case."""
    stage_tensors = build_stage_tensors(
        weights,
        tokens=tokens,
        topk=topk,
        num_routed=num_routed,
        device=device,
    )
    stage_weight_tensors = {
        W13_STAGE: (weights["w13"], weights["w13_shift2"], weights["w13_scale"]),
        W2_STAGE: (weights["w2"], weights["w2_shift2"], weights["w2_scale"]),
    }
    base = {
        "run_id": run_id,
        "case_id": case_id,
        "git_head": head,
        "dirty_diff_hash": diff_hash,
        "command": command,
        "seed": args.seed,
        "gpu_uuid": env.get("gpu_uuid"),
        "pci_bus_id": env.get("pci_bus_id"),
        "torch_version": env.get("torch"),
        "cuda_version": env.get("cuda"),
        "triton_version": env.get("triton"),
        "relevant_env": env.get("relevant_env"),
        "driver_and_clocks": env.get("clock_and_power"),
        "M": tokens,
        "E": num_experts,
        "topk": topk,
        "route_pattern": pattern,
        "route_hash": route_hash(routing_ids),
        "cache_mode": args.cache_mode,
        "logical_io_bytes": None,
    }

    # ---- correctness -----------------------------------------------------
    rows: list[dict] = []
    if args.check_reference:
        run_production_once(
            stage_tensors,
            routing_weights,
            routing_ids,
            num_routed,
            num_experts,
            16,
            128,
        )
        comparisons = verify_stages(
            stage_tensors,
            routing_ids,
            routing_weights,
            chunk_n=args.reference_chunk_n,
        )
        failed = False
        for stage, (comparison, exact_required) in comparisons.items():
            threshold = None if exact_required else REDUCE_THRESHOLD
            status, reason = classify(
                comparison, exact_required=exact_required, threshold=threshold
            )
            if status == "fail":
                failed = True
            n, k, source_rows_are_slots = STAGE_SHAPES.get(stage, (0, 0, None))
            rows.append(
                ResultRow(
                    run_id=run_id,
                    case_id=case_id,
                    stage=stage,
                    variant="reference-check-16x128",
                    values={
                        **base,
                        "N": n,
                        "K": k,
                        "block_m": 16,
                        "block_n": 128,
                        "source_rows_are_slots": source_rows_are_slots,
                        "reference_kind": "packed-weight-decode/fp64-int32/fp32-scales",
                        "correctness_status": status,
                        "skip_or_failure_reason": reason,
                        "measurement_status": "skipped",
                        "round": None,
                        "graph_calls": None,
                        "active_weight_bytes": None,
                        "useful_flops": None,
                        **comparison.as_row(),
                        **(threshold.as_row() if threshold else {}),
                    },
                ).to_json()
            )
        if failed:
            return rows, [], True

    # ---- tile equivalence (same routing, same weights, bitwise) ----------
    check_tile_equivalence(
        stage_tensors[W13_STAGE]["a_q"],
        stage_tensors[W13_STAGE]["a_scale"],
        weights["w13"],
        weights["w13_shift2"],
        weights["w13_scale"],
        stage_tensors[W2_STAGE]["a_q"],
        stage_tensors[W2_STAGE]["a_scale"],
        weights["w2"],
        weights["w2_shift2"],
        weights["w2_scale"],
        stage_tensors[W2_STAGE]["output"],
        routing_weights,
        routing_ids,
        num_routed,
        tiles,
        num_experts,
    )

    selected_tile = select_tile(tokens, topk, num_experts)
    rotation = None
    if args.cache_mode == "rotating-experts":
        rotation = rotation_routing_sequence(
            topk, num_experts, args.graph_calls, device
        )

    legacy: list[dict] = []
    for block_m, block_n in tiles:
        align = align_routes(routing_ids, block_m, num_experts)
        work = compute_route_work(routing_ids, block_m, stage_weight_tensors)
        per_call_bytes = {stage: work.stage_bytes[stage] for stage in work.stage_bytes}
        rotation_period = None
        working_set_bytes = (
            work.stage_bytes[W13_STAGE]["total"] + work.stage_bytes[W2_STAGE]["total"]
        )
        if rotation is not None:
            per_call_bytes, rotation_period, working_set_bytes = rotation_accounting(
                weights, rotation, block_m
            )
        variant = f"{block_m}x{block_n}"
        baseline_id = f"prod-selector-{selected_tile[0]}x{selected_tile[1]}"
        variant_rows: list[dict] = []
        for stage in (W13_STAGE, W2_STAGE, REDUCE_STAGE):
            call, calls_per_repeat = build_timed_call(
                args,
                stage,
                stage_tensors,
                routing_weights,
                align,
                num_routed,
                block_m,
                block_n,
                num_experts,
                rotation,
            )
            timing = time_graph(
                call,
                warmup=args.warmup,
                repeats=args.graph_calls,
                samples=args.graph_samples,
                calls_per_repeat=calls_per_repeat,
            )
            n, k, source_rows_are_slots = STAGE_SHAPES.get(stage, (0, 0, None))
            stage_bytes = per_call_bytes.get(stage)
            useful_flops = 2 * num_routed * n * k if stage in STAGE_SHAPES else None
            variant_rows.append(
                ResultRow(
                    run_id=run_id,
                    case_id=case_id,
                    stage=stage,
                    variant=variant,
                    values={
                        **base,
                        "N": n,
                        "K": k,
                        "block_m": block_m,
                        "block_n": block_n,
                        "source_rows_are_slots": source_rows_are_slots,
                        "baseline_id": baseline_id,
                        "is_production_tile": (block_m, block_n) == selected_tile,
                        "reference_kind": (
                            "packed-weight-decode/fp64-int32/fp32-scales"
                            if args.check_reference
                            else None
                        ),
                        "correctness_status": (
                            "pass" if args.check_reference else "not_checked"
                        ),
                        "reference_rows_in_this_run": len(rows),
                        "skip_or_failure_reason": None,
                        "rotation_period": rotation_period,
                        "working_set_bytes": working_set_bytes,
                        "active_weight_bytes": (
                            stage_bytes["total"] if stage_bytes else None
                        ),
                        "active_weight_bytes_detail": stage_bytes,
                        "useful_flops": useful_flops,
                        "active_weight_gb_s": (
                            round(
                                stage_bytes["total"] / (timing.p50_us * 1e-6) / 1e9, 3
                            )
                            if stage_bytes and timing.p50_us
                            else None
                        ),
                        "useful_tflops": (
                            round(useful_flops / (timing.p50_us * 1e-6) / 1e12, 4)
                            if useful_flops and timing.p50_us
                            else None
                        ),
                        "padded_row_ratio": (
                            round(work.padded_rows / max(1, num_routed), 4)
                        ),
                        "measurement_status": "ok",
                        "round": None,
                        "graph_calls": args.graph_calls,
                        **timing.as_row(),
                        **work.as_row(),
                    },
                ).to_json()
            )
        rows.extend(variant_rows)

        eager = {
            stage: measure(
                build_eager_call(
                    stage,
                    stage_tensors,
                    routing_weights,
                    align,
                    num_routed,
                    block_m,
                    block_n,
                ),
                args.warmup,
                args.repeats,
            )
            for stage in (W13_STAGE, W2_STAGE, REDUCE_STAGE)
        }
        legacy.append(
            {
                "tokens": tokens,
                "num_experts": num_experts,
                "topk": topk,
                "route_pattern": pattern,
                "cache_mode": args.cache_mode,
                "block_m": block_m,
                "block_n": block_n,
                "rotate_experts": args.cache_mode == "rotating-experts",
                "eager_event": eager,
                "graph_event": {
                    row["stage"]: {
                        "p50_us": row["p50_us"],
                        "p90_us": row["p90_us"],
                        "min_us": row["min_us"],
                    }
                    for row in variant_rows
                },
            }
        )
    return rows, legacy, False


def build_eager_call(
    stage: str,
    stage_tensors: dict[str, dict[str, torch.Tensor]],
    routing_weights: torch.Tensor,
    align: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    num_routed: int,
    block_m: int,
    block_n: int,
) -> Callable[[], None]:
    """A single non-rotating invocation, for the diagnostic eager timing."""
    sorted_ids, expert_ids, num_post_pad = align
    if stage == REDUCE_STAGE:
        return make_reduce_call(
            stage_tensors[W2_STAGE]["routed_out"],
            routing_weights,
            stage_tensors[W2_STAGE]["output"],
            num_routed,
        )
    tensors = stage_tensors[stage]
    return make_gemm_call(
        tensors["a_q"],
        tensors["a_scale"],
        tensors["weight"],
        tensors["shift2"],
        tensors["channel_scale"],
        tensors["output"],
        tensors["routed_out"],
        routing_weights,
        sorted_ids,
        expert_ids,
        num_post_pad,
        num_routed,
        block_m,
        block_n,
        stage == W2_STAGE,
    )


def build_timed_call(
    args: argparse.Namespace,
    stage: str,
    stage_tensors: dict[str, dict[str, torch.Tensor]],
    routing_weights: torch.Tensor,
    align: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    num_routed: int,
    block_m: int,
    block_n: int,
    num_experts: int,
    rotation: list[torch.Tensor] | None,
) -> tuple[Callable[[], None], int]:
    """Return ``(call, kernel_invocations_per_call)``.

    In warm-fixed mode ``call`` is a single GEMM and the graph repeats it
    ``--graph-calls`` times. In rotating-experts mode ``call`` already is the
    whole rotation sequence, because each step needs different routing metadata
    that cannot be re-derived inside a captured graph without re-running
    ``moe_align_block_size`` on the host.
    """
    if rotation is None or stage == REDUCE_STAGE:
        return (
            build_eager_call(
                stage,
                stage_tensors,
                routing_weights,
                align,
                num_routed,
                block_m,
                block_n,
            ),
            1,
        )
    tensors = stage_tensors[stage]
    calls = []
    for routed_ids in rotation:
        routed_sorted, routed_experts, routed_post_pad = moe_align_block_size(
            routed_ids, block_m, num_experts
        )
        calls.append(
            make_gemm_call(
                tensors["a_q"],
                tensors["a_scale"],
                tensors["weight"],
                tensors["shift2"],
                tensors["channel_scale"],
                tensors["output"],
                tensors["routed_out"],
                routing_weights,
                routed_sorted,
                routed_experts,
                routed_post_pad,
                num_routed,
                block_m,
                block_n,
                stage == W2_STAGE,
            )
        )
    sequence = list(calls)

    def call() -> None:
        for gemm in sequence:
            gemm()

    return call, len(sequence)


def select_tile(tokens: int, topk: int, num_experts: int) -> tuple[int, int]:
    """Production tile selector; the measured baseline is whatever it picks."""
    from vllm.model_executor.layers.fused_moe.experts.mxfp4_int8_sm80 import (
        select_mxfp4_int8_sm80_tile,
    )

    return select_mxfp4_int8_sm80_tile(tokens, topk, num_experts)


if __name__ == "__main__":
    main()
