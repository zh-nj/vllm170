# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure a TileLang grouped GEMM candidate against the production baseline.

Both sides run the same logical routing case with the same tensors and are
checked against the independent reference in ``deepseek_v41_moe_harness``
before any timing happens. Timing uses the committed Event + CUDA-Graph
harness, alternates baseline/candidate as A/B/B/A and keeps every round so a
difference smaller than the baseline's own spread is reported as
``inconclusive`` instead of as a win.

This is an offline facility: it does not change production defaults, dispatch
or backend selection.
"""

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deepseek_v41_moe_harness import (  # noqa: E402
    STAGE_SHAPES,
    W2_STAGE,
    W13_STAGE,
    align_routes,
    environment,
    git_head,
    interleaved_ab,
    make_routing_ids,
    reference_stage,
    route_hash,
)
from deepseek_v41_moe_tilelang_gemm import (  # noqa: E402
    GemmConfig,
    TileLangMoeGemm,
    build_packed_weights,
)

NUM_EXPERTS = 384
DEFAULT_TOPK = 6


@dataclass
class Case:
    """Every tensor one stage's grouped GEMM needs, plus its reference."""

    stage: str
    n: int
    k: int
    tokens: int
    topk: int
    num_experts: int
    slot_count: int
    routing_ids: torch.Tensor
    routing_weights: torch.Tensor
    a_q: torch.Tensor
    a_scale: torch.Tensor
    weights: dict[str, torch.Tensor]
    routed_out: torch.Tensor
    reference: torch.Tensor
    block_m: int
    block_n: int
    align: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    route_hash: str


def build_case(
    *,
    stage: str,
    num_experts: int,
    topk: int,
    tokens: int,
    pattern: str,
    seed: int,
    block_m: int,
    block_n: int,
    device: torch.device,
    weights: dict[str, torch.Tensor] | None = None,
    routing_ids: torch.Tensor | None = None,
) -> Case:
    n, k, source_rows_are_slots = STAGE_SHAPES[stage]
    slot_count = tokens * topk
    if routing_ids is None:
        routing_ids = make_routing_ids(tokens, topk, num_experts, pattern, seed).to(
            device
        )
    if weights is None:
        weights = build_packed_weights(num_experts, stage, device)
    torch.manual_seed(seed)
    m_in = slot_count if source_rows_are_slots else tokens
    a_q = torch.empty((m_in, k), dtype=torch.int8, device=device).random_(-127, 128)
    a_scale = torch.ones((m_in,), dtype=torch.float32, device=device)
    # Non-unit scales would break the bitwise expectation the harness uses; the
    # production path also carries unit activation scales in these benchmarks.
    routed_out = torch.empty((slot_count, n), dtype=torch.bfloat16, device=device)
    reference = reference_stage(
        a_q,
        a_scale,
        weights["packed"],
        weights["shift2"],
        weights["channel_scale"],
        routing_ids,
        stage=stage,
    )
    align = align_routes(routing_ids, block_m, num_experts)
    return Case(
        stage=stage,
        n=n,
        k=k,
        tokens=tokens,
        topk=topk,
        num_experts=num_experts,
        slot_count=slot_count,
        routing_ids=routing_ids,
        routing_weights=torch.full(
            (tokens, topk), 1.0 / topk, dtype=torch.float32, device=device
        ),
        a_q=a_q,
        a_scale=a_scale,
        weights=weights,
        routed_out=routed_out,
        reference=reference,
        block_m=block_m,
        block_n=block_n,
        align=align,
        route_hash=route_hash(routing_ids),
    )


def repeat_tensors(case: Case, copies: int, device: torch.device) -> list[Case]:
    """Pointer-distinct copies of a case, for rotating over in a graph."""
    out = []
    for _ in range(copies):
        clone = Case(
            stage=case.stage,
            n=case.n,
            k=case.k,
            tokens=case.tokens,
            topk=case.topk,
            num_experts=case.num_experts,
            slot_count=case.slot_count,
            routing_ids=case.routing_ids,
            routing_weights=case.routing_weights,
            a_q=case.a_q,
            a_scale=case.a_scale,
            weights=case.weights,
            routed_out=torch.empty_like(case.routed_out),
            reference=case.reference,
            block_m=case.block_m,
            block_n=case.block_n,
            align=case.align,
            route_hash=case.route_hash,
        )
        out.append(clone)
    return out


def rotation_cases(
    *,
    stage: str,
    base: Case,
    period: int,
    device: torch.device,
) -> list[Case]:
    """``period`` pointer-distinct cases whose expert union covers the period.

    Each call walks ``topk`` new experts, so the union is ``period * topk``
    experts, matching the committed rotating-experts mode. The weight tensors
    and the routing weights are shared; only the routing metadata and the
    output buffer differ, exactly as in production.
    """
    cases = []
    for call_index in range(period):
        ids = (
            (
                torch.arange(base.topk, dtype=torch.int32, device=device)
                + call_index * base.topk
            )
            .remainder(base.num_experts)
            .view(1, base.topk)
        )
        case = build_case(
            stage=stage,
            num_experts=base.num_experts,
            topk=base.topk,
            tokens=base.tokens,
            pattern="uniform",
            seed=0,
            block_m=base.block_m,
            block_n=base.block_n,
            device=device,
            weights=base.weights,
            routing_ids=ids,
        )
        cases.append(case)
    return cases


def sequence_call(calls: list[Callable[[], None]]) -> Callable[[], None]:
    def call() -> None:
        for step in calls:
            step()

    return call


def production_call(case: Case) -> Callable[[], None]:
    sorted_ids, expert_ids, num_post_pad = case.align
    source_rows_are_slots = STAGE_SHAPES[case.stage][2]
    output = case.routed_out

    def call() -> None:
        torch.ops._moe_C.dsv4_mxfp4_int8_sm80_moe_gemm(
            case.a_q,
            case.a_scale,
            case.weights["packed"],
            case.weights["shift2"],
            case.weights["channel_scale"],
            output,
            case.routed_out,
            case.routing_weights,
            sorted_ids,
            expert_ids,
            num_post_pad,
            case.slot_count,
            case.block_m,
            case.block_n,
            source_rows_are_slots,
        )

    return call


def candidate_gemm(case: Case, config: GemmConfig) -> TileLangMoeGemm:
    return TileLangMoeGemm.create(config, tokens=case.tokens, slots=case.slot_count)


def candidate_call(gemm: TileLangMoeGemm, case: Case) -> Callable[[], None]:
    sorted_ids, expert_ids, num_post_pad = case.align
    args = (
        case.a_q,
        case.a_scale,
        case.weights["packed"],
        case.weights["shift2"],
        case.weights["channel_scale"],
        sorted_ids,
        expert_ids,
        num_post_pad,
        case.routed_out,
    )

    def call() -> None:
        gemm.kernel(*args)

    return call


def first_mismatch(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    bad = (candidate != reference).nonzero()
    finite = bool(torch.isfinite(candidate.to(torch.float32)).all().item())
    result = {
        "bitwise_exact": bool(bad.shape[0] == 0),
        "mismatches": int(bad.shape[0]),
        "elements": int(candidate.numel()),
        "finite": finite,
    }
    if bad.shape[0]:
        row, col = bad[0].tolist()
        result["first_mismatch"] = [row, col]
        result["candidate_value"] = float(candidate[row, col])
        result["reference_value"] = float(reference[row, col])
    return result


def select_production_tile(tokens: int, topk: int, num_experts: int) -> tuple[int, int]:
    from vllm.model_executor.layers.fused_moe.experts.mxfp4_int8_sm80 import (
        select_mxfp4_int8_sm80_tile,
    )

    return select_mxfp4_int8_sm80_tile(tokens, topk, num_experts)


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=[W13_STAGE, W2_STAGE, "both"], default="both"
    )
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--route-pattern", default="uniform")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument(
        "--decode",
        choices=["mxfp4", "lut", "luta", "raw"],
        default="mxfp4",
        help="raw is a decode-free ablation: same traffic, wrong numbers",
    )
    parser.add_argument("--baseline-block-m", type=int, default=0)
    parser.add_argument("--baseline-block-n", type=int, default=0)
    parser.add_argument(
        "--cache-mode",
        choices=["warm-fixed", "rotating-experts"],
        default="warm-fixed",
    )
    parser.add_argument(
        "--rotation-size",
        type=int,
        default=40,
        help="distinct routing cases in one rotation period",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--graph-calls", type=int, default=20)
    parser.add_argument(
        "--check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="verify candidate and baseline against the independent reference",
    )
    parser.add_argument("--output-jsonl", type=str, default="")
    parser.add_argument("--dump-dir", type=str, default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 2
    capability = torch.cuda.get_device_capability(0)
    if capability != (8, 0):
        print(f"expected an SM80 device, got {capability}", file=sys.stderr)
        return 2
    device = torch.device("cuda")
    head = git_head()
    env = environment()
    stages = [W13_STAGE, W2_STAGE] if args.stage == "both" else [args.stage]
    rows = []
    failed = False
    for stage in stages:
        default_m, default_n = select_production_tile(
            args.tokens, args.topk, args.experts
        )
        baseline_m = args.baseline_block_m or default_m
        baseline_n = args.baseline_block_n or default_n
        baseline_case = build_case(
            stage=stage,
            num_experts=args.experts,
            topk=args.topk,
            tokens=args.tokens,
            pattern=args.route_pattern,
            seed=args.seed,
            block_m=baseline_m,
            block_n=baseline_n,
            device=device,
        )
        candidate_case = build_case(
            stage=stage,
            num_experts=args.experts,
            topk=args.topk,
            tokens=args.tokens,
            pattern=args.route_pattern,
            seed=args.seed,
            block_m=args.block_m,
            block_n=args.block_n,
            device=device,
            weights=baseline_case.weights,
            routing_ids=baseline_case.routing_ids,
        )
        config = GemmConfig(
            n=candidate_case.n,
            k=candidate_case.k,
            num_experts=args.experts,
            topk=args.topk,
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            num_stages=args.num_stages,
            threads=args.threads,
            source_rows_are_slots=STAGE_SHAPES[stage][2],
            decode=args.decode,
        )
        gemm = candidate_gemm(candidate_case, config)
        calls_per_repeat = 1
        if args.cache_mode == "rotating-experts":
            if candidate_case.tokens != 1:
                raise ValueError(
                    "rotating-experts only supports --tokens 1; refusing to "
                    "generalise the capture"
                )
            period = args.rotation_size
            base_rotation = rotation_cases(
                stage=stage, base=baseline_case, period=period, device=device
            )
            cand_rotation = rotation_cases(
                stage=stage, base=candidate_case, period=period, device=device
            )
            base_call = sequence_call([production_call(case) for case in base_rotation])
            cand_call = sequence_call(
                [candidate_call(gemm, case) for case in cand_rotation]
            )
            calls_per_repeat = period
            check_pairs = list(zip(base_rotation, cand_rotation, strict=True))
            union = torch.unique(
                torch.cat(
                    [case.routing_ids.reshape(-1).cpu() for case in cand_rotation]
                )
            )
            row_extra = {
                "rotation_period": period,
                "rotation_experts_in_union": int(union.numel()),
            }
        else:
            base_call = production_call(baseline_case)
            cand_call = candidate_call(gemm, candidate_case)
            check_pairs = [(baseline_case, candidate_case)]
            row_extra = {"rotation_period": 1, "rotation_experts_in_union": None}

        row: dict[str, Any] = {
            "stage": stage,
            "case_id": (
                f"E{args.experts}-k{args.topk}-M{args.tokens}-"
                f"{args.route_pattern}-{args.seed}"
            ),
            "git_head": head,
            "gpu": env.get("gpu_name"),
            "route_hash": baseline_case.route_hash,
            "tokens": args.tokens,
            "topk": args.topk,
            "experts": args.experts,
            "N": candidate_case.n,
            "K": candidate_case.k,
            "slots": candidate_case.slot_count,
            "candidate": config.label,
            "candidate_block_m": config.block_m,
            "candidate_block_n": config.block_n,
            "candidate_block_k": config.block_k,
            "candidate_num_stages": config.num_stages,
            "candidate_threads": config.threads,
            "candidate_decode": config.decode,
            "candidate_compile_seconds": round(gemm.compile_seconds, 3),
            "baseline_tile": f"{baseline_m}x{baseline_n}",
            "production_selected_tile": f"{default_m}x{default_n}",
            "comparison_role": (
                "production_baseline"
                if (baseline_m, baseline_n) == (args.block_m, args.block_n)
                else "same_algorithm_other_tile"
            ),
        }
        if args.check:
            # Populate every buffer the timed call writes (the whole rotation
            # period in rotating mode) before comparing with the reference.
            base_call()
            cand_call()
            torch.accelerator.synchronize()
            cand_bad = []
            base_bad = []
            for index, (base_case, cand_case) in enumerate(check_pairs):
                cand_result = first_mismatch(cand_case.routed_out, cand_case.reference)
                base_result = first_mismatch(base_case.routed_out, base_case.reference)
                if not cand_result["bitwise_exact"]:
                    cand_bad.append({"case": index, **cand_result})
                if not base_result["bitwise_exact"]:
                    base_bad.append({"case": index, **base_result})
            row["candidate_correctness"] = {
                "cases_checked": len(check_pairs),
                "failures": cand_bad[:3],
                "bitwise_exact": not cand_bad,
            }
            row["baseline_correctness"] = {
                "cases_checked": len(check_pairs),
                "failures": base_bad[:3],
                "bitwise_exact": not base_bad,
            }
            if cand_bad or base_bad:
                failed = True
                row["measurement_status"] = "numerical_failure"
                rows.append(row)
                print(
                    f"FAIL {stage} {config.label}: candidate {cand_bad[:1]} "
                    f"baseline {base_bad[:1]}",
                    file=sys.stderr,
                )
                continue
        if args.dump_dir:
            out = Path(args.dump_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{config.label}.cu").write_text(gemm.source)

        ab = interleaved_ab(
            base_call,
            cand_call,
            rounds=args.rounds,
            warmup=args.warmup,
            graph_calls=args.graph_calls,
            samples=args.samples,
            calls_per_repeat=calls_per_repeat,
        )
        row.update(ab)
        row.update(row_extra)
        row["cache_mode"] = args.cache_mode
        row["useful_flops"] = (
            2 * candidate_case.slot_count * candidate_case.n * candidate_case.k
        )
        if row.get("baseline_ratio"):
            row["speedup_vs_baseline"] = round(1.0 / row["baseline_ratio"], 4)
        rows.append(row)
        print(json.dumps(row, indent=2, default=str))
    if args.output_jsonl:
        path = Path(args.output_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            handle.write(
                json.dumps({"schema": "tilelang-moe-bench-v1", "env": env}, default=str)
            )
            handle.write("\n")
            for row in rows:
                handle.write(json.dumps(row, default=str))
                handle.write("\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
