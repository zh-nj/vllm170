# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark DSV4.1-Flash SM80 MXFP4-int8 expert GEMM tiles.

Scans the W13/W2 grouped GEMM stages and the routing-weighted reduce of the
``mxfp4_int8_sm80`` backend through the production ``_moe_C`` entry points.

The scan axes are the ones the DSV4.1-F serving shape actually exercises:
expert count, top-k (3 for DSpark draft, 6 for the target model), routing
pattern, cache mode and tile. By default an independent
numerical reference (see ``deepseek_v41_moe_harness``) that decodes the actual
packed weights checks each stage before timing; a failing stage makes the
process exit non-zero. ``--no-check-reference`` is diagnostic only and cannot
produce a passing run.

This file stays an offline facility: it does not change production defaults,
backend selection or dispatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import uuid
from collections.abc import Callable
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deepseek_v41_moe_harness import (  # noqa: E402
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    REDUCE_STAGE,
    SCHEMA_VERSION,
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
    file_hash,
    git_head,
    make_routing_ids,
    reduce_term_scale,
    reference_reduce,
    reference_stage,
    route_hash,
    row_passed,
    stage_weight_bytes,
    tensor_hash,
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

# For <=6 terms, the difference between FMA and separate multiply/add
# accumulation is bounded by (gamma_6 + gamma_12) * sum(abs(terms)).
# 2**-19 conservatively covers this FP32 error plus one final BF16 step.
REDUCE_THRESHOLD = Threshold(
    threshold_id="reduce-local-fp32-bound-v2",
    calc_diff=1e-6,
    max_step_excess_over_term_scale=2.0**-19,
    calibrated=True,
    calibration_run="analytical-gamma6-plus-gamma12",
    note="Per-output absolute-term sum; <=6 finite normal FP32 terms, BF16 output",
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
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--scale-mode", choices=["varied", "unit"], default="varied")
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
        action=argparse.BooleanOptionalAction,
        default=True,
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


def input_generator(device: torch.device, seed: int, *identity) -> torch.Generator:
    digest = hashlib.sha256(repr((seed, identity)).encode()).digest()
    return torch.Generator(device=device).manual_seed(
        int.from_bytes(digest[:8], "little")
    )


def scale_fixture(
    shape: tuple[int, ...], device: torch.device, mode: str
) -> torch.Tensor:
    if mode == "unit":
        return torch.ones(shape, device=device, dtype=torch.float32)
    indices = torch.arange(math.prod(shape), device=device)
    return (0.0313 + (indices.remainder(997) + 1).float() / 811.0).reshape(shape)


def build_weights(
    num_experts: int,
    device: torch.device,
    *,
    seed: int = 4242,
    scale_mode: str = "varied",
) -> dict[str, torch.Tensor]:
    """Deterministic packed weights with independently indexed FP32 scales."""
    generator = input_generator(device, seed, "weights", num_experts)
    return {
        "w13": torch.empty(
            (num_experts, HIDDEN_SIZE // 32, INTERMEDIATE_SIZE * 2 // 8, 128),
            dtype=torch.uint8,
            device=device,
        ).random_(0, 256, generator=generator),
        "w13_shift2": torch.empty(
            (num_experts, HIDDEN_SIZE // 128, INTERMEDIATE_SIZE * 2 // 8, 8),
            dtype=torch.uint8,
            device=device,
        ).random_(0, 256, generator=generator),
        "w13_scale": scale_fixture(
            (num_experts, INTERMEDIATE_SIZE * 2), device, scale_mode
        ),
        "w2": torch.empty(
            (num_experts, INTERMEDIATE_SIZE // 32, HIDDEN_SIZE // 8, 128),
            dtype=torch.uint8,
            device=device,
        ).random_(0, 256, generator=generator),
        "w2_shift2": torch.empty(
            (num_experts, INTERMEDIATE_SIZE // 128, HIDDEN_SIZE // 8, 8),
            dtype=torch.uint8,
            device=device,
        ).random_(0, 256, generator=generator),
        "w2_scale": scale_fixture((num_experts, HIDDEN_SIZE), device, scale_mode),
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

    GEMM requires exact equality for unit and non-unit scales: both sides
    use the same integer dot and ordered FP32 epilogue.
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
        results[stage] = (
            compare(
                observed,
                expected,
                slot_experts=routing_ids.reshape(-1),
                rows=num_routed,
            ),
            True,
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
    seed: int = 4242,
    scale_mode: str = "varied",
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
    a13_q.random_(
        -127, 128, generator=input_generator(device, seed, "w13", tokens, topk)
    )
    a2_q = torch.empty((num_routed, INTERMEDIATE_SIZE), dtype=torch.int8, device=device)
    a2_q.random_(-127, 128, generator=input_generator(device, seed, "w2", tokens, topk))
    return {
        W13_STAGE: {
            "a_q": a13_q,
            "a_scale": scale_fixture((tokens,), device, scale_mode),
            "weight": weights["w13"],
            "shift2": weights["w13_shift2"],
            "channel_scale": weights["w13_scale"],
            "output": gate_up,
            "routed_out": gate_up,
        },
        W2_STAGE: {
            "a_q": a2_q,
            "a_scale": scale_fixture((num_routed,), device, scale_mode),
            "weight": weights["w2"],
            "shift2": weights["w2_shift2"],
            "channel_scale": weights["w2_scale"],
            "output": output,
            "routed_out": routed_out,
        },
    }


def main() -> None:
    args = parse_args()
    run_id = uuid.uuid4().hex
    rows: list[dict] = []
    status = "running"
    error = None
    provenance = {}

    def summary():
        if args.summary_json is not None:
            write_summary(
                args.summary_json,
                rows,
                run_id=run_id,
                round=args.round,
                run_status=status,
                error=error,
                **provenance,
                note="Synthetic offline inputs; no model accuracy or serving claim.",
            )

    summary()
    try:
        provenance.update(source_identity())
        with JsonlWriter(args.output_jsonl) as writer:

            def emit(row):
                row = {"schema_version": SCHEMA_VERSION, **row}
                writer.write(row)
                rows.append(row)
                summary()

            legacy = execute_scan(args, run_id, provenance, emit)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(legacy, indent=2) + "\n")
        if not args.quiet:
            print(json.dumps(legacy, indent=2))
        status = (
            "complete" if rows and all(row_passed(row) for row in rows) else "failed"
        )
    except BaseException as exc:
        status = "aborted"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        summary()
    if status != "complete":
        raise SystemExit("run did not pass correctness and measurement gates")


def source_identity() -> dict:
    root = Path(__file__).resolve().parents[2]
    sources = [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("deepseek_v41_moe_harness.py"),
    ]
    sources += sorted((root / "csrc/libtorch_stable/moe/mxfp4_int8_sm80").glob("*"))
    sources += [
        root / "vllm/model_executor/layers/fused_moe/moe_align_block_size.py",
        root / "vllm/model_executor/layers/fused_moe/experts/mxfp4_int8_sm80.py",
    ]
    binaries = sorted((root / "vllm").glob("_moe_C*.so"))
    return {
        "git_head": git_head(),
        "dirty_diff_hash": dirty_diff_hash(),
        "source_sha256": {
            str(p.relative_to(root)): file_hash(p) for p in sources if p.is_file()
        },
        "extension_sha256": {str(p): file_hash(p) for p in binaries},
    }


def execute_scan(args, run_id, provenance, emit) -> list[dict]:
    if (
        args.warmup < 0
        or args.repeats <= 0
        or args.graph_calls <= 0
        or args.graph_samples <= 0
    ):
        raise ValueError(
            "warmup must be non-negative; repeats and graph-calls must be positive"
        )
    if any(tokens <= 0 for tokens in args.tokens):
        raise ValueError("tokens must be positive")
    if any(experts <= 0 for experts in args.experts):
        raise ValueError("experts must be positive")
    if min(args.experts) < max(args.topk):
        raise ValueError("expert count must cover top-k without duplicate slots")
    if args.reference_chunk_n <= 0:
        raise ValueError("reference-chunk-n must be positive")
    if torch.cuda.get_device_capability() != (8, 0):
        raise ValueError("this DSV41F harness is restricted to SM80")
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
    env = environment()
    command = " ".join([sys.executable, *sys.argv])
    legacy = []
    for num_experts in args.experts:
        weights = build_weights(
            num_experts, device, seed=args.seed, scale_mode=args.scale_mode
        )
        weight_hashes = {key: tensor_hash(value) for key, value in weights.items()}
        for topk in args.topk:
            for pattern in args.route_pattern:
                for tokens in args.tokens:
                    routing_ids = make_routing_ids(
                        tokens, topk, num_experts, pattern, args.seed
                    ).to(device)
                    routing_weights = torch.full(
                        (tokens, topk), 1.0 / topk, dtype=torch.float32, device=device
                    )
                    case_id = (
                        f"E{num_experts}-k{topk}-M{tokens}-{pattern}-{args.cache_mode}"
                    )
                    legacy.extend(
                        run_case(
                            args=args,
                            case_id=case_id,
                            tokens=tokens,
                            topk=topk,
                            num_experts=num_experts,
                            pattern=pattern,
                            routing_ids=routing_ids,
                            routing_weights=routing_weights,
                            num_routed=tokens * topk,
                            weights=weights,
                            tiles=tiles,
                            device=device,
                            run_id=run_id,
                            env=env,
                            provenance=provenance,
                            command=command,
                            weight_hashes=weight_hashes,
                            emit=emit,
                        )
                    )
    return legacy


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
) -> tuple[dict[str, dict[str, float]], int, dict[str, int]]:
    """Average per-call logical bytes, and per-stage union working sets."""
    if not sequence:
        raise ValueError("rotation sequence must not be empty")
    sets = [set(ids.reshape(-1).cpu().tolist()) for ids in sequence]
    union = set.union(*sets)
    per_call = {}
    working_set = {}
    for stage in (W13_STAGE, W2_STAGE):
        tensors = (
            weights[stage],
            weights[f"{stage}_shift2"],
            weights[f"{stage}_scale"],
        )
        details = [stage_weight_bytes(*tensors, experts) for experts in sets]
        per_call[stage] = {
            key: sum(d[key] for d in details) / len(sequence) for key in details[0]
        }
        working_set[stage] = stage_weight_bytes(*tensors, union)["total"]
    return per_call, len(sequence), working_set


def run_case(
    *,
    args,
    case_id,
    tokens,
    topk,
    num_experts,
    pattern,
    routing_ids,
    routing_weights,
    num_routed,
    weights,
    tiles,
    device,
    run_id,
    env,
    provenance,
    command,
    weight_hashes,
    emit,
) -> list[dict]:
    """Validate the actual eager and graph outputs before recording timing."""
    tensors = build_stage_tensors(
        weights,
        tokens=tokens,
        topk=topk,
        num_routed=num_routed,
        device=device,
        seed=args.seed,
        scale_mode=args.scale_mode,
    )
    rotation = None
    if args.cache_mode == "rotating-experts":
        rotation = rotation_routing_sequence(
            topk, num_experts, args.graph_calls, device
        )
    sequence = rotation or [routing_ids]
    sequence_hash = tensor_hash(torch.stack(sequence))
    input_hashes = {**weight_hashes, "routing_weights": tensor_hash(routing_weights)}
    for stage in (W13_STAGE, W2_STAGE):
        for key in ("a_q", "a_scale"):
            input_hashes[f"{stage}_{key}"] = tensor_hash(tensors[stage][key])
    base = {
        "run_id": run_id,
        "case_id": case_id,
        **provenance,
        "command": command,
        "seed": args.seed,
        "round": args.round,
        "scale_mode": args.scale_mode,
        "input_sha256": input_hashes,
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
        "requested_route_pattern": pattern,
        "route_pattern": "sequential-rotation" if rotation else pattern,
        "route_hash": route_hash(sequence[0]),
        "route_sequence_sha256": sequence_hash,
        "cache_mode": args.cache_mode,
        "logical_io_bytes": None,
    }
    # Each route gets its own output so a later call cannot hide an earlier failure.
    references = {}
    if args.check_reference:
        for stage in (W13_STAGE, W2_STAGE):
            t = tensors[stage]
            references[stage] = [
                reference_stage(
                    t["a_q"],
                    t["a_scale"],
                    t["weight"],
                    t["shift2"],
                    t["channel_scale"],
                    ids,
                    stage=stage,
                    chunk_n=args.reference_chunk_n,
                )
                for ids in sequence
            ]
        tensors[W2_STAGE]["routed_out"].copy_(references[W2_STAGE][0])
    else:
        run_production_once(
            tensors, routing_weights, sequence[0], num_routed, num_experts, 16, 128
        )
    reduce_input = tensors[W2_STAGE]["routed_out"].clone()
    reduce_tensors = {W2_STAGE: {**tensors[W2_STAGE], "routed_out": reduce_input}}
    if args.check_reference:
        references[REDUCE_STAGE] = [reference_reduce(reduce_input, routing_weights)]
    stage_weights = {
        stage: (weights[stage], weights[f"{stage}_shift2"], weights[f"{stage}_scale"])
        for stage in (W13_STAGE, W2_STAGE)
    }
    selected = select_tile(tokens, topk, num_experts)
    legacy = []
    for block_m, block_n in tiles:
        per_call, period, working_set = rotation_accounting(weights, sequence, block_m)
        works = [compute_route_work(ids, block_m, stage_weights) for ids in sequence]
        variant = f"{block_m}x{block_n}"
        variant_rows = []
        for stage in (W13_STAGE, W2_STAGE, REDUCE_STAGE):
            stage_sequence = [sequence[0]] if stage == REDUCE_STAGE else sequence
            n, k, source_rows = STAGE_SHAPES.get(stage, (0, 0, None))
            stage_base = {
                **base,
                "stage": stage,
                "variant": variant,
                "block_m": block_m,
                "block_n": block_n,
                "baseline_id": f"prod-selector-{selected[0]}x{selected[1]}",
                "is_production_tile": (block_m, block_n) == selected,
                "N": n,
                "K": k,
                "source_rows_are_slots": source_rows,
                "reference_kind": "packed-weight-decode/fp64-int32/fp32-scales"
                if args.check_reference
                else None,
            }
            if stage == REDUCE_STAGE:
                stage_base.update(
                    cache_mode="warm-fixed",
                    route_pattern="fixed-reduce-input",
                    route_sequence_sha256=tensor_hash(torch.stack(stage_sequence)),
                )
            call, count, validator = build_timed_call(
                args,
                stage,
                reduce_tensors if stage == REDUCE_STAGE else tensors,
                routing_weights,
                stage_sequence,
                num_routed,
                block_m,
                block_n,
                num_experts,
                references.get(stage),
                stage_base,
                emit,
            )
            timing = time_graph(
                call,
                warmup=args.warmup,
                repeats=1 if count > 1 else args.graph_calls,
                samples=args.graph_samples,
                calls_per_repeat=count,
                validate_graph=validator,
            )
            stage_bytes = per_call.get(stage)
            useful_flops = 2 * num_routed * n * k if stage in STAGE_SHAPES else None
            row = {
                **stage_base,
                "correctness_status": "pass" if args.check_reference else "not_checked",
                "measurement_status": "ok",
                "skip_or_failure_reason": None,
                "rotation_period": period
                if rotation and stage != REDUCE_STAGE
                else None,
                "working_set_bytes": working_set.get(stage),
                "working_set_kind": "stage-weight-union" if stage_bytes else None,
                "output_buffer_bytes": len(stage_sequence) * num_routed * n * 2
                if stage_bytes
                else tokens * HIDDEN_SIZE * 2,
                "active_weight_bytes": stage_bytes["total"] if stage_bytes else None,
                "active_weight_bytes_detail": stage_bytes,
                "useful_flops": useful_flops,
                "active_weight_gb_s": stage_bytes["total"] / (timing.p50_us * 1000)
                if stage_bytes
                else None,
                "useful_tflops": useful_flops / (timing.p50_us * 1e6)
                if useful_flops
                else None,
                "padded_row_ratio": sum(w.padded_rows for w in works)
                / (len(works) * num_routed),
                "route_work_per_call": [
                    w.as_row() for w in works[: len(stage_sequence)]
                ],
                "graph_calls": args.graph_calls,
                **timing.as_row(),
                "output_buffers": len(stage_sequence),
            }
            emit(ResultRow(run_id, case_id, stage, variant, row).to_json())
            variant_rows.append(row)
        eager = {
            stage: measure(
                build_eager_call(
                    stage,
                    reduce_tensors if stage == REDUCE_STAGE else tensors,
                    routing_weights,
                    align_routes(sequence[0], block_m, num_experts),
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
                "route_pattern": base["route_pattern"],
                "cache_mode": args.cache_mode,
                "block_m": block_m,
                "block_n": block_n,
                "rotate_experts": rotation is not None,
                "eager_event": eager,
                "eager_route_hash": route_hash(sequence[0]),
                "eager_cache_mode": "warm-fixed",
                "graph_event": {
                    row["stage"]: {
                        key: row[key] for key in ("p50_us", "p90_us", "min_us")
                    }
                    for row in variant_rows
                },
            }
        )
    return legacy


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
    args,
    stage,
    stage_tensors,
    routing_weights,
    sequence,
    num_routed,
    block_m,
    block_n,
    num_experts,
    references,
    base,
    emit,
):
    """Build the measured calls and checks against their exact route metadata."""
    calls = []
    outputs = []
    for ids in sequence:
        source = stage_tensors[W2_STAGE if stage == REDUCE_STAGE else stage]
        output = torch.empty_like(
            source["output"] if stage == REDUCE_STAGE else source["routed_out"]
        )
        t = {**source}
        if stage == REDUCE_STAGE:
            t["output"] = output
        else:
            t["routed_out"] = output
            if stage == W13_STAGE:
                t["output"] = output
        align = align_routes(ids, block_m, num_experts)
        calls.append(
            build_eager_call(
                stage,
                {W2_STAGE if stage == REDUCE_STAGE else stage: t},
                routing_weights,
                align,
                num_routed,
                block_m,
                block_n,
            )
        )
        outputs.append(output)

    def call():
        for invoke in calls:
            invoke()

    def check(phase):
        for i, (output, expected) in enumerate(zip(outputs, references, strict=True)):
            comparison = compare(
                output,
                expected,
                term_scale=reduce_term_scale(
                    stage_tensors[W2_STAGE]["routed_out"], routing_weights
                )
                if stage == REDUCE_STAGE
                else None,
                measure_bf16_steps=stage == REDUCE_STAGE,
            )
            threshold = REDUCE_THRESHOLD if stage == REDUCE_STAGE else None
            status, reason = classify(
                comparison, exact_required=stage != REDUCE_STAGE, threshold=threshold
            )
            emit(
                {
                    **base,
                    "variant": f"{base['variant']}-{phase}",
                    "route_index": i,
                    "route_hash": route_hash(sequence[i]),
                    "correctness_status": status,
                    "measurement_status": "not_applicable",
                    "skip_or_failure_reason": reason,
                    **comparison.as_row(),
                    **(threshold.as_row() if threshold else {}),
                }
            )
            if status != "pass":
                raise AssertionError(f"{stage}/{phase}/route={i}: {status}: {reason}")

    def reset():
        for output in outputs:
            output.fill_(float("nan"))

    def validate_graph(graph):
        reset()
        graph.replay()
        torch.accelerator.synchronize()
        check("graph-replay")

    if references is not None:
        reset()
        call()
        torch.accelerator.synchronize()
        check("eager-reference")
    return call, len(calls), validate_graph if references is not None else None


def select_tile(tokens: int, topk: int, num_experts: int) -> tuple[int, int]:
    """Production tile selector; the measured baseline is whatever it picks."""
    from vllm.model_executor.layers.fused_moe.experts.mxfp4_int8_sm80 import (
        select_mxfp4_int8_sm80_tile,
    )

    return select_mxfp4_int8_sm80_tile(tokens, topk, num_experts)


if __name__ == "__main__":
    main()
