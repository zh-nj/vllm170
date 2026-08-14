# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize prefill, decode, and DSpark CUDA-event phase records."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

FLOWS = ("prefill", "decode", "dspark")
DETAIL_REQUIREMENTS = {
    "prefill": {
        "layer.total",
        "layer.attention",
        "layer.ffn",
        "layer.mhc_attention_pre",
        "layer.mhc_ffn_pre",
        "attention.input_gemm_overlap",
        "target.attn_metadata",
        "target.model_inputs",
        "indexer.logits",
        "indexer.topk",
        "moe.combine",
        "moe.dispatch",
        "moe.fused_route_experts",
        "moe.marlin.activation",
        "moe.marlin.align",
        "moe.marlin.reduce",
        "moe.marlin.w13_gemm",
        "moe.marlin.w2_gemm",
        "moe.modular.finalize",
        "moe.modular.kernel",
        "moe.modular.prepare",
        "moe.modular.workspace",
        "moe.routed_experts",
        "moe.router_gate",
        "moe.router_select",
        "sparse_mla.attention",
    },
    "decode": {
        "attention.input_gemm_overlap",
        "indexer.paged_logits",
        "indexer.topk",
        "sparse_mla.attention",
    },
    "dspark": {
        "target.logits",
        "draft.backbone",
        "draft.base_logits",
        "draft.markov_bias",
        "draft.markov_sample",
    },
}

PREFILL_LAYER_COMPONENTS = (
    "layer.mhc_attention_pre",
    "layer.attention",
    "layer.mhc_ffn_pre",
    "layer.ffn",
)
PREFILL_MODEL_COMPONENTS = (
    "layer.total",
    "model.embedding",
    "model.aux_reconstruct",
    "model.stage_reconstruct",
    "model.hc_head",
    "model.final_norm",
)
PREFILL_MARLIN_ADDITIVE_COMPONENTS = (
    "moe.router_gate",
    "moe.dispatch",
    "moe.router_select",
    "moe.modular.prepare",
    "moe.modular.workspace",
    "moe.marlin.align",
    "moe.marlin.w13_gemm",
    "moe.marlin.activation",
    "moe.marlin.w2_gemm",
    "moe.marlin.reduce",
    "moe.modular.finalize",
    "moe.combine",
)
PREFILL_MARLIN_NESTED_RANGES = (
    "moe.routed_experts",
    "moe.modular.kernel",
)


def load_phase_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != 2:
                raise ValueError(f"{path}:{line_number}: unsupported schema")
            if row.get("flow") not in FLOWS:
                raise ValueError(f"{path}:{line_number}: invalid flow")
            elapsed_us = row.get("elapsed_us")
            if not isinstance(elapsed_us, (int, float)) or not math.isfinite(
                elapsed_us
            ):
                raise ValueError(f"{path}:{line_number}: invalid elapsed_us")
            records.append(row)
    return records


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = math.ceil(percentile * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def analyze_phase_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    cpu_grouped: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    step_grouped: dict[tuple[str, int, int, int | None, str, str], list[float]] = (
        defaultdict(list)
    )
    layer_grouped: dict[tuple[str, int, str, str, str], list[float]] = defaultdict(list)
    stages_by_flow: dict[str, set[str]] = defaultdict(set)
    executions_by_flow: dict[str, set[str]] = defaultdict(set)
    layer_compress_ratio: dict[tuple[str, int, str], int] = {}
    marlin_shape_grouped: dict[
        tuple[str, int, str, int, int, int, int, int, int],
        list[tuple[float, int | None]],
    ] = defaultdict(list)
    for row in records:
        flow = row["flow"]
        pp_rank = row.get("pp_rank")
        if pp_rank is None:
            pp_rank = -1
        stage = row["stage"]
        level = row.get("level", "detail")
        elapsed_us = float(row["elapsed_us"])
        pp_rank = int(pp_rank)
        grouped[(flow, pp_rank, stage, level)].append(elapsed_us)
        cpu_elapsed_us = row.get("cpu_elapsed_us")
        if isinstance(cpu_elapsed_us, (int, float)) and math.isfinite(cpu_elapsed_us):
            cpu_grouped[(flow, pp_rank, stage, level)].append(float(cpu_elapsed_us))
        step_id = row.get("step_id")
        if isinstance(step_id, int):
            step_grouped[
                (flow, pp_rank, step_id, row.get("token_count"), stage, level)
            ].append(elapsed_us)
        layer = row.get("layer")
        if isinstance(layer, str) and layer:
            layer_grouped[(flow, pp_rank, layer, stage, level)].append(elapsed_us)
            compress_ratio = row.get("compress_ratio")
            if isinstance(compress_ratio, int):
                canonical_layer = layer.removesuffix(".attn")
                layer_compress_ratio[(flow, pp_rank, canonical_layer)] = compress_ratio
        stages_by_flow[flow].add(stage)
        if row.get("execution"):
            executions_by_flow[flow].add(row["execution"])
        extra = row.get("extra")
        if stage.startswith("moe.marlin.") and isinstance(extra, dict):
            shape = tuple(
                extra.get(key) for key in ("M", "K", "N", "E", "topk", "block_size_m")
            )
            if all(isinstance(value, int) for value in shape):
                padded = extra.get("num_tokens_post_padded")
                marlin_shape_grouped[(flow, pp_rank, stage, *shape)].append(
                    (elapsed_us, padded if isinstance(padded, int) else None)
                )

    flows: dict[str, Any] = {}
    for flow in FLOWS:
        rows = []
        for (row_flow, pp_rank, stage, level), values in sorted(grouped.items()):
            if row_flow != flow:
                continue
            total_us = sum(values)
            cpu_values = cpu_grouped.get((row_flow, pp_rank, stage, level), [])
            rows.append(
                {
                    "pp_rank": pp_rank,
                    "stage": stage,
                    "level": level,
                    "count": len(values),
                    "total_us": total_us,
                    "mean_us": total_us / len(values),
                    "p50_us": _percentile(values, 0.50),
                    "p90_us": _percentile(values, 0.90),
                    "cpu_total_us": sum(cpu_values),
                    "cpu_mean_us": (
                        sum(cpu_values) / len(cpu_values) if cpu_values else None
                    ),
                }
            )

        stage_maxima = []
        model_total_by_rank = {
            row["pp_rank"]: row["total_us"]
            for row in rows
            if row["stage"] in {"target.model_forward", "target.model_graph_replay"}
        }
        for row in rows:
            model_total = model_total_by_rank.get(row["pp_rank"])
            row["model_share_pct"] = (
                100.0 * row["total_us"] / model_total if model_total else None
            )
        for stage in sorted(stages_by_flow[flow]):
            candidates = [row for row in rows if row["stage"] == stage]
            if candidates:
                critical = max(candidates, key=lambda row: row["total_us"])
                stage_maxima.append(
                    {
                        "stage": stage,
                        "pp_rank": critical["pp_rank"],
                        "total_us": critical["total_us"],
                        "mean_us": critical["mean_us"],
                        "count": critical["count"],
                        "level": critical["level"],
                        "model_share_pct": critical["model_share_pct"],
                        "cpu_total_us": critical["cpu_total_us"],
                    }
                )

        step_rows = []
        for (
            row_flow,
            pp_rank,
            step_id,
            token_count,
            stage,
            level,
        ), values in sorted(step_grouped.items()):
            if row_flow != flow:
                continue
            total_us = sum(values)
            step_rows.append(
                {
                    "pp_rank": pp_rank,
                    "step_id": step_id,
                    "token_count": token_count,
                    "stage": stage,
                    "level": level,
                    "count": len(values),
                    "total_us": total_us,
                    "mean_us": total_us / len(values),
                }
            )

        layer_rows = []
        for (row_flow, pp_rank, layer, stage, level), values in sorted(
            layer_grouped.items()
        ):
            if row_flow != flow:
                continue
            total_us = sum(values)
            layer_rows.append(
                {
                    "pp_rank": pp_rank,
                    "layer": layer,
                    "stage": stage,
                    "level": level,
                    "count": len(values),
                    "total_us": total_us,
                    "mean_us": total_us / len(values),
                    "p50_us": _percentile(values, 0.50),
                    "p90_us": _percentile(values, 0.90),
                    "compress_ratio": layer_compress_ratio.get(
                        (flow, pp_rank, layer.removesuffix(".attn"))
                    ),
                }
            )

        marlin_shape_rows = []
        for (
            row_flow,
            pp_rank,
            stage,
            M,
            K,
            N,
            E,
            topk,
            block_size_m,
        ), samples in sorted(marlin_shape_grouped.items()):
            if row_flow != flow:
                continue
            values = [sample[0] for sample in samples]
            padded_values = [sample[1] for sample in samples if sample[1] is not None]
            total_us = sum(values)
            mean_padded = (
                sum(padded_values) / len(padded_values) if padded_values else None
            )
            marlin_shape_rows.append(
                {
                    "pp_rank": pp_rank,
                    "stage": stage,
                    "M": M,
                    "K": K,
                    "N": N,
                    "E": E,
                    "topk": topk,
                    "block_size_m": block_size_m,
                    "count": len(values),
                    "total_us": total_us,
                    "mean_us": total_us / len(values),
                    "p50_us": _percentile(values, 0.50),
                    "p90_us": _percentile(values, 0.90),
                    "mean_tokens_post_padded": mean_padded,
                    "useful_token_pct": (
                        100.0 * M * topk / mean_padded if mean_padded else None
                    ),
                }
            )

        missing = sorted(DETAIL_REQUIREMENTS[flow] - stages_by_flow[flow])
        flows[flow] = {
            "record_count": sum(row["count"] for row in rows),
            "pp_ranks": sorted({row["pp_rank"] for row in rows}),
            "executions": sorted(executions_by_flow[flow]),
            "detail_complete": not missing,
            "missing_detail_stages": missing,
            "by_rank_stage": rows,
            "by_step_rank_stage": step_rows,
            "by_layer_stage": layer_rows,
            "marlin_shape_stage": marlin_shape_rows,
            "critical_stage_rank": stage_maxima,
        }

    flows["prefill"]["module_coverage"] = _prefill_module_coverage(
        flows["prefill"]["by_rank_stage"]
    )
    flows["prefill"]["chunks"] = _prefill_chunks(flows["prefill"]["by_step_rank_stage"])
    flows["prefill"]["marlin_moe"] = _prefill_marlin_moe(
        flows["prefill"]["by_rank_stage"],
        flows["prefill"]["marlin_shape_stage"],
    )

    return {
        "schema_version": 1,
        "aggregation": {
            "within_rank": "sum repeated instances of the same stage",
            "across_pp_ranks": "max rank per stage; PP ranks are never summed",
            "nested_ranges": (
                "total, stage, and detail ranges overlap; do not sum levels"
            ),
        },
        "flows": flows,
    }


def _prefill_module_coverage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_rank_stage = {(row["pp_rank"], row["stage"]): row["total_us"] for row in rows}
    ranks = sorted({row["pp_rank"] for row in rows})
    coverage = []
    for pp_rank in ranks:
        model_total = by_rank_stage.get((pp_rank, "target.model_forward"), 0.0)
        layer_total = by_rank_stage.get((pp_rank, "layer.total"), 0.0)
        model_components = {
            stage: by_rank_stage.get((pp_rank, stage), 0.0)
            for stage in PREFILL_MODEL_COMPONENTS
        }
        layer_components = {
            stage: by_rank_stage.get((pp_rank, stage), 0.0)
            for stage in PREFILL_LAYER_COMPONENTS
        }
        model_accounted = sum(model_components.values())
        layer_accounted = sum(layer_components.values())
        coverage.append(
            {
                "pp_rank": pp_rank,
                "model_total_us": model_total,
                "model_components_us": model_components,
                "model_unattributed_us": max(0.0, model_total - model_accounted),
                "model_coverage_pct": (
                    100.0 * model_accounted / model_total if model_total else 0.0
                ),
                "layer_total_us": layer_total,
                "layer_components_us": layer_components,
                "layer_unattributed_us": max(0.0, layer_total - layer_accounted),
                "layer_coverage_pct": (
                    100.0 * layer_accounted / layer_total if layer_total else 0.0
                ),
            }
        )
    return coverage


def _prefill_chunks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model_rows_by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["stage"] == "target.model_forward":
            model_rows_by_rank[row["pp_rank"]].append(row)
    for rank_rows in model_rows_by_rank.values():
        rank_rows.sort(key=lambda row: row["step_id"])

    chunks = []
    chunk_count = max((len(rows) for rows in model_rows_by_rank.values()), default=0)
    for ordinal in range(chunk_count):
        rank_rows = {
            pp_rank: rows[ordinal]
            for pp_rank, rows in model_rows_by_rank.items()
            if ordinal < len(rows)
        }
        token_counts = {row["token_count"] for row in rank_rows.values()}
        token_count = next(iter(token_counts)) if len(token_counts) == 1 else None
        critical_rank, critical_row = max(
            rank_rows.items(), key=lambda item: item[1]["total_us"]
        )
        chunks.append(
            {
                "ordinal": ordinal,
                "token_count": token_count,
                "rank_total_us": {
                    str(pp_rank): row["total_us"]
                    for pp_rank, row in sorted(rank_rows.items())
                },
                "critical_pp_rank": critical_rank,
                "critical_total_us": critical_row["total_us"],
            }
        )
    return chunks


def _prefill_marlin_moe(
    rows: list[dict[str, Any]],
    shape_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_rank_stage = {(row["pp_rank"], row["stage"]): row["total_us"] for row in rows}
    ranks = sorted(
        row["pp_rank"] for row in rows if row["stage"] == "moe.fused_route_experts"
    )
    by_rank = []
    for pp_rank in ranks:
        parent_us = by_rank_stage[(pp_rank, "moe.fused_route_experts")]
        additive = {
            stage: by_rank_stage.get((pp_rank, stage), 0.0)
            for stage in PREFILL_MARLIN_ADDITIVE_COMPONENTS
        }
        nested = {
            stage: by_rank_stage.get((pp_rank, stage), 0.0)
            for stage in PREFILL_MARLIN_NESTED_RANGES
        }
        accounted_us = sum(additive.values())
        by_rank.append(
            {
                "pp_rank": pp_rank,
                "parent_us": parent_us,
                "additive_components_us": additive,
                "nested_ranges_us": nested,
                "accounted_us": accounted_us,
                "residual_us": max(0.0, parent_us - accounted_us),
                "coverage_pct": 100.0 * accounted_us / parent_us,
            }
        )
    return {
        "parent_stage": "moe.fused_route_experts",
        "additive_stages": list(PREFILL_MARLIN_ADDITIVE_COMPONENTS),
        "nested_ranges": list(PREFILL_MARLIN_NESTED_RANGES),
        "by_rank": by_rank,
        "by_shape_stage": shape_rows,
    }


def render_flow_markdown(flow: str, summary: dict[str, Any]) -> str:
    lines = [
        f"# DeepSeek V4 {flow} phase profile",
        "",
        (
            "PP ranks overlap. Each row reports the rank with the largest total for "
            "that stage; totals from different ranks or nested levels must not be "
            "added."
        ),
        "",
        f"Detail complete: `{str(summary['detail_complete']).lower()}`",
        "",
    ]
    if summary["missing_detail_stages"]:
        lines.extend(
            [
                "Missing stages: "
                + ", ".join(f"`{stage}`" for stage in summary["missing_detail_stages"]),
                "",
            ]
        )
    lines.extend(
        [
            (
                "| Stage | Level | Critical PP rank | Count | Total ms | Mean us | "
                "CPU scope ms | Rank model share |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        summary["critical_stage_rank"], key=lambda item: item["total_us"], reverse=True
    ):
        model_share = row["model_share_pct"]
        model_share_text = "-" if model_share is None else f"{model_share:.1f}%"
        lines.append(
            f"| `{row['stage']}` | {row['level']} | {row['pp_rank']} | "
            f"{row['count']} | {row['total_us'] / 1000:.3f} | "
            f"{row['mean_us']:.3f} | {row['cpu_total_us'] / 1000:.3f} | "
            f"{model_share_text} |"
        )
    if flow == "prefill":
        lines.extend(_render_prefill_details(summary))
    return "\n".join(lines) + "\n"


def _render_prefill_details(summary: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "## Per-chunk target critical path",
        "",
        "Each row is one model-forward CUDA range on one PP rank.",
        "",
        (
            "| Chunk | Tokens | PP0 ms | PP1 ms | PP2 ms | Critical PP rank | "
            "Critical ms |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["chunks"]:
        rank_total_us = row["rank_total_us"]
        lines.append(
            f"| {row['ordinal']} | {row['token_count']} | "
            f"{rank_total_us.get('0', 0.0) / 1000:.3f} | "
            f"{rank_total_us.get('1', 0.0) / 1000:.3f} | "
            f"{rank_total_us.get('2', 0.0) / 1000:.3f} | "
            f"{row['critical_pp_rank']} | {row['critical_total_us'] / 1000:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Module coverage",
            "",
            (
                "Sibling components are additive within their parent only. Model and "
                "layer coverage rows are separate nesting levels."
            ),
            "",
            (
                "| PP rank | Model ms | Layer totals ms | Model residual ms | "
                "Model coverage | Layer residual ms | Layer coverage |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["module_coverage"]:
        lines.append(
            f"| {row['pp_rank']} | {row['model_total_us'] / 1000:.3f} | "
            f"{row['layer_total_us'] / 1000:.3f} | "
            f"{row['model_unattributed_us'] / 1000:.3f} | "
            f"{row['model_coverage_pct']:.1f}% | "
            f"{row['layer_unattributed_us'] / 1000:.3f} | "
            f"{row['layer_coverage_pct']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Per-layer totals",
            "",
            (
                "| Layer | C-ratio | PP rank | Total ms | Attention ms | FFN ms | "
                "mHC-attn ms | mHC-FFN ms |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    layer_rows = [
        row for row in summary["by_layer_stage"] if row["stage"] == "layer.total"
    ]
    components = {
        (row["pp_rank"], row["layer"], row["stage"]): row["total_us"]
        for row in summary["by_layer_stage"]
    }
    for row in sorted(layer_rows, key=lambda item: item["total_us"], reverse=True):
        component_key = (row["pp_rank"], row["layer"])
        attention_ms = components.get((*component_key, "layer.attention"), 0.0) / 1000
        ffn_ms = components.get((*component_key, "layer.ffn"), 0.0) / 1000
        mhc_attention_ms = (
            components.get((*component_key, "layer.mhc_attention_pre"), 0.0) / 1000
        )
        mhc_ffn_ms = components.get((*component_key, "layer.mhc_ffn_pre"), 0.0) / 1000
        lines.append(
            f"| `{row['layer']}` | {row['compress_ratio']} | {row['pp_rank']} | "
            f"{row['total_us'] / 1000:.3f} | "
            f"{attention_ms:.3f} | {ffn_ms:.3f} | "
            f"{mhc_attention_ms:.3f} | {mhc_ffn_ms:.3f} |"
        )
    lines.extend(_render_marlin_moe(summary["marlin_moe"]))
    return lines


def _render_marlin_moe(summary: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "## Marlin MXFP4 MoE breakdown",
        "",
        (
            "Additive stages are siblings on the current CUDA stream. The "
            "routed-experts and modular-kernel ranges are nested diagnostics and are "
            "excluded from the sum."
        ),
        "",
        "| PP rank | Stage | Total ms | Parent share |",
        "|---:|---|---:|---:|",
    ]
    for rank_row in summary["by_rank"]:
        parent_us = rank_row["parent_us"]
        lines.append(
            f"| {rank_row['pp_rank']} | `moe.fused_route_experts` | "
            f"{parent_us / 1000:.3f} | 100.0% |"
        )
        for stage, total_us in rank_row["additive_components_us"].items():
            share = 100.0 * total_us / parent_us if parent_us else 0.0
            lines.append(
                f"| {rank_row['pp_rank']} | `{stage}` | "
                f"{total_us / 1000:.3f} | {share:.1f}% |"
            )
        lines.append(
            f"| {rank_row['pp_rank']} | `unattributed` | "
            f"{rank_row['residual_us'] / 1000:.3f} | "
            f"{max(0.0, 100.0 - rank_row['coverage_pct']):.1f}% |"
        )

    lines.extend(
        [
            "",
            "### Marlin stages by DSV4 shape",
            "",
            (
                "| Stage | PP | M | K | N | E | Top-k | Block M | Count | "
                "Mean us | P90 us | Padded rows | Useful rows |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        summary["by_shape_stage"],
        key=lambda item: (item["M"], item["pp_rank"], item["stage"]),
        reverse=True,
    ):
        padded = row["mean_tokens_post_padded"]
        useful = row["useful_token_pct"]
        padded_text = "-" if padded is None else f"{padded:.1f}"
        useful_text = "-" if useful is None else f"{useful:.1f}%"
        lines.append(
            f"| `{row['stage']}` | {row['pp_rank']} | {row['M']} | {row['K']} | "
            f"{row['N']} | {row['E']} | {row['topk']} | "
            f"{row['block_size_m']} | {row['count']} | {row['mean_us']:.3f} | "
            f"{row['p90_us']:.3f} | {padded_text} | {useful_text} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    paths = sorted(args.phase_dir.glob("*.jsonl"))
    if not paths:
        raise SystemExit(f"no phase JSONL files under {args.phase_dir}")
    summary = analyze_phase_records(load_phase_records(paths))
    output_dir = args.output_dir or args.phase_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    for flow in FLOWS:
        flow_summary = summary["flows"][flow]
        (output_dir / f"phase_{flow}.md").write_text(
            render_flow_markdown(flow, flow_summary)
        )
    print(
        json.dumps(
            {
                flow: {
                    "record_count": summary["flows"][flow]["record_count"],
                    "pp_ranks": summary["flows"][flow]["pp_ranks"],
                    "detail_complete": summary["flows"][flow]["detail_complete"],
                    "missing_detail_stages": summary["flows"][flow][
                        "missing_detail_stages"
                    ],
                }
                for flow in FLOWS
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.require_complete and not all(
        summary["flows"][flow]["detail_complete"] for flow in FLOWS
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
