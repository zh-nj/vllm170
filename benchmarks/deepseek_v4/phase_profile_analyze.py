# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize target, DSpark, and DFlash CUDA-event phase records."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

FLOWS = ("prefill", "decode", "dspark", "dflash")
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
        "target.model_forward",
        "target.attn_metadata.prepare_attn",
        "target.attn_metadata.builders",
        "layer.total",
        "layer.attention",
        "layer.ffn",
        "attention.input_gemm_overlap",
        "attention.input_gemm.join",
        "attention.qkv_insert",
        "attention.qicm_join",
        "attention.q_index_compress_mla",
        "attention.output_projection",
        "indexer.paged_logits",
        "indexer.query_cache_join",
        "indexer.topk",
        "moe.fused_route_experts",
        "moe.routed_experts",
        "moe.shared_experts.compute",
        "moe.shared_experts.join",
        "moe.shared_experts.w13",
        "moe.shared_experts.activation",
        "moe.shared_experts.w2",
        "moe.marlin.w13_gemm",
        "moe.marlin.w2_gemm",
        "sparse_mla.attention",
    },
    "dspark": {
        "target.sampling_total",
        "target.logits",
        "target.rejection.verify_total",
        "draft.propose_total",
        "draft.context_kv",
        "draft.metadata",
        "draft.backbone",
        "draft.model.embedding",
        "draft.model.hc_head",
        "layer.total",
        "layer.attention",
        "layer.ffn",
        "draft.base_logits",
        "draft.logits.lm_head",
        "draft.markov_bias",
        "draft.markov.projection",
        "draft.markov_sample",
    },
    "dflash": {
        "target.sampling_total",
        "target.logits",
        "target.rejection.verify_total",
        "draft.propose_total",
        "draft.prepare_inputs",
        "draft.context_kv",
        "draft.dispatch",
        "draft.metadata",
        "draft.backbone",
        "draft.logits.lm_head",
        "draft.selector",
        "draft.selector_walk",
    },
}

LAYER_COMPONENTS = (
    "layer.mhc_attention_pre",
    "layer.attention",
    "layer.mhc_ffn_pre",
    "layer.ffn",
)
MODEL_COMPONENTS = (
    "layer.total",
    "model.embedding",
    "model.aux_reconstruct",
    "model.stage_reconstruct",
    "model.hc_head",
    "model.final_norm",
)
MARLIN_ADDITIVE_COMPONENTS = (
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
MARLIN_NESTED_RANGES = (
    "moe.routed_experts",
    "moe.modular.kernel",
    "moe.shared_experts.compute",
    "moe.shared_experts.join",
    "moe.shared_experts.w13",
    "moe.shared_experts.activation",
    "moe.shared_experts.w2",
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
    metadata_grouped: dict[tuple[str, int, int, str, str], list[float]] = defaultdict(
        list
    )
    metadata_cpu_grouped: dict[tuple[str, int, int, str, str], list[float]] = (
        defaultdict(list)
    )
    step_grouped: dict[
        tuple[str, int, int, int | None, str | None, str, str], list[float]
    ] = defaultdict(list)
    kind_grouped: dict[tuple[str, str, int, str, str], list[float]] = defaultdict(list)
    draft_step_grouped: dict[tuple[str, int, str, int], list[float]] = defaultdict(list)
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
        extra = row.get("extra")
        step_kind = row.get("step_kind")
        if step_kind is None and isinstance(extra, dict):
            step_kind = extra.get("draft_kind", extra.get("dspark_kind"))
        if not isinstance(step_kind, str):
            step_kind = None
        step_id = row.get("step_id")
        if isinstance(step_id, int):
            step_grouped[
                (
                    flow,
                    pp_rank,
                    step_id,
                    row.get("token_count"),
                    step_kind,
                    stage,
                    level,
                )
            ].append(elapsed_us)
        if step_kind is not None:
            kind_grouped[(flow, step_kind, pp_rank, stage, level)].append(elapsed_us)
        draft_step = extra.get("draft_step") if isinstance(extra, dict) else None
        if isinstance(draft_step, int):
            draft_step_grouped[(flow, pp_rank, stage, draft_step)].append(elapsed_us)
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
        kv_cache_group = (
            extra.get("kv_cache_group") if isinstance(extra, dict) else None
        )
        if stage.startswith("target.attn_metadata") and isinstance(kv_cache_group, int):
            metadata_key = (flow, pp_rank, kv_cache_group, stage, level)
            metadata_grouped[metadata_key].append(elapsed_us)
            if isinstance(cpu_elapsed_us, (int, float)) and math.isfinite(
                cpu_elapsed_us
            ):
                metadata_cpu_grouped[metadata_key].append(float(cpu_elapsed_us))
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
            step_kind,
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
                    "step_kind": step_kind,
                    "stage": stage,
                    "level": level,
                    "count": len(values),
                    "total_us": total_us,
                    "mean_us": total_us / len(values),
                }
            )

        kind_rows = []
        for (
            row_flow,
            step_kind,
            pp_rank,
            stage,
            level,
        ), values in sorted(kind_grouped.items()):
            if row_flow != flow:
                continue
            total_us = sum(values)
            kind_rows.append(
                {
                    "step_kind": step_kind,
                    "pp_rank": pp_rank,
                    "stage": stage,
                    "level": level,
                    "count": len(values),
                    "total_us": total_us,
                    "mean_us": total_us / len(values),
                    "p50_us": _percentile(values, 0.50),
                    "p90_us": _percentile(values, 0.90),
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

        draft_step_rows = []
        for (
            row_flow,
            pp_rank,
            stage,
            draft_step,
        ), values in sorted(draft_step_grouped.items()):
            if row_flow != flow:
                continue
            total_us = sum(values)
            draft_step_rows.append(
                {
                    "pp_rank": pp_rank,
                    "stage": stage,
                    "draft_step": draft_step,
                    "count": len(values),
                    "total_us": total_us,
                    "mean_us": total_us / len(values),
                    "p50_us": _percentile(values, 0.50),
                    "p90_us": _percentile(values, 0.90),
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

        compression_values: dict[tuple[int, str, str], list[float]] = defaultdict(list)
        compression_layers: dict[int, set[tuple[int, str]]] = defaultdict(set)
        for (row_flow, pp_rank, layer, stage, level), values in layer_grouped.items():
            if row_flow != flow:
                continue
            canonical_layer = layer.removesuffix(".attn")
            compress_ratio = layer_compress_ratio.get((flow, pp_rank, canonical_layer))
            if compress_ratio is None:
                continue
            compression_values[(compress_ratio, stage, level)].extend(values)
            compression_layers[compress_ratio].add((pp_rank, canonical_layer))

        compression_family_rows = []
        for (compress_ratio, stage, level), values in sorted(
            compression_values.items()
        ):
            total_us = sum(values)
            compression_family_rows.append(
                {
                    "compress_ratio": compress_ratio,
                    "layer_count": len(compression_layers[compress_ratio]),
                    "stage": stage,
                    "level": level,
                    "count": len(values),
                    "total_us": total_us,
                    "mean_us": total_us / len(values),
                    "p50_us": _percentile(values, 0.50),
                    "p90_us": _percentile(values, 0.90),
                }
            )

        metadata_group_rows = []
        for (
            row_flow,
            pp_rank,
            kv_cache_group,
            stage,
            level,
        ), values in sorted(metadata_grouped.items()):
            if row_flow != flow:
                continue
            total_us = sum(values)
            cpu_values = metadata_cpu_grouped.get(
                (row_flow, pp_rank, kv_cache_group, stage, level), []
            )
            metadata_group_rows.append(
                {
                    "pp_rank": pp_rank,
                    "kv_cache_group": kv_cache_group,
                    "stage": stage,
                    "level": level,
                    "count": len(values),
                    "total_us": total_us,
                    "mean_us": total_us / len(values),
                    "p50_us": _percentile(values, 0.50),
                    "p90_us": _percentile(values, 0.90),
                    "cpu_mean_us": (
                        sum(cpu_values) / len(cpu_values) if cpu_values else None
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
            "by_kind_rank_stage": kind_rows,
            "by_layer_stage": layer_rows,
            "by_draft_step_stage": draft_step_rows,
            "compression_family_stage": compression_family_rows,
            "metadata_group_stage": metadata_group_rows,
            "marlin_shape_stage": marlin_shape_rows,
            "critical_stage_rank": stage_maxima,
        }

    for flow in ("prefill", "decode"):
        flows[flow]["module_coverage"] = _module_coverage(flows[flow]["by_rank_stage"])
        flows[flow]["target_steps"] = _target_steps(flows[flow]["by_step_rank_stage"])
        flows[flow]["marlin_moe"] = _marlin_moe(
            flows[flow]["by_rank_stage"],
            flows[flow]["marlin_shape_stage"],
        )
    flows["prefill"]["chunks"] = flows["prefill"]["target_steps"]
    flows["decode"]["round_summary"] = _decode_round_summary(
        flows["decode"]["target_steps"]
    )
    flows["decode"]["serial_stage"] = _decode_serial_stages(
        flows["decode"]["by_step_rank_stage"]
    )
    for flow in ("dspark", "dflash"):
        flows[flow]["proposal_shapes"] = _draft_proposal_shapes(
            flows[flow]["by_step_rank_stage"]
        )
        flows[flow]["draft_module_coverage"] = _draft_module_coverage(
            flows[flow]["by_rank_stage"]
        )
        flows[flow]["marlin_moe"] = _marlin_moe(
            flows[flow]["by_rank_stage"],
            flows[flow]["marlin_shape_stage"],
        )
        flows[flow]["cycle_summary"] = _draft_cycle_summary(
            flows["decode"]["round_summary"],
            flows["decode"]["by_rank_stage"],
            flows[flow]["by_kind_rank_stage"],
            flows[flow]["proposal_shapes"],
        )

    return {
        "schema_version": 1,
        "aggregation": {
            "within_rank": "sum repeated instances of the same stage",
            "across_pp_ranks": (
                "prefill reports stage maxima; decode additionally sums aligned "
                "serial PP model work within one target round"
            ),
            "nested_ranges": (
                "total, stage, and detail ranges overlap; do not sum levels"
            ),
        },
        "flows": flows,
    }


def _draft_proposal_shapes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str | None, int | None], list[float]] = defaultdict(list)
    for row in rows:
        if row["stage"] == "draft.propose_total":
            grouped[(row["step_kind"], row["token_count"])].append(row["total_us"])
    result = []
    for (step_kind, token_count), values in sorted(
        grouped.items(),
        key=lambda item: (
            "" if item[0][0] is None else item[0][0],
            -1 if item[0][1] is None else item[0][1],
        ),
    ):
        result.append(
            {
                "step_kind": step_kind,
                "token_count": token_count,
                "count": len(values),
                "mean_us": sum(values) / len(values),
                "p50_us": _percentile(values, 0.50),
                "p90_us": _percentile(values, 0.90),
                "total_us": sum(values),
            }
        )
    return result


def _draft_module_coverage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_rank_stage = {(row["pp_rank"], row["stage"]): row["total_us"] for row in rows}
    ranks = sorted(row["pp_rank"] for row in rows if row["stage"] == "draft.backbone")
    model_stages = (
        "draft.model.embedding",
        "draft.model.expand",
        "layer.total",
        "draft.model.reconstruct",
        "draft.model.hc_head",
    )
    result = []
    for pp_rank in ranks:
        model_total = by_rank_stage[(pp_rank, "draft.backbone")]
        layer_total = by_rank_stage.get((pp_rank, "layer.total"), 0.0)
        model_components = {
            stage: by_rank_stage.get((pp_rank, stage), 0.0) for stage in model_stages
        }
        layer_components = {
            stage: by_rank_stage.get((pp_rank, stage), 0.0)
            for stage in LAYER_COMPONENTS
        }
        model_accounted = sum(model_components.values())
        layer_accounted = sum(layer_components.values())
        result.append(
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
    return result


def _draft_cycle_summary(
    target_rounds: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    draft_kind_rows: list[dict[str, Any]],
    proposal_shapes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_round_count = sum(row["round_count"] for row in target_rounds)
    target_prepare_by_rank: dict[int, float] = defaultdict(float)
    for row in target_rows:
        if row["stage"] in {"target.attn_metadata", "target.model_inputs"}:
            target_prepare_by_rank[row["pp_rank"]] += row["total_us"]
    if target_round_count:
        target_prepare_by_rank = {
            rank: total_us / target_round_count
            for rank, total_us in target_prepare_by_rank.items()
        }
    target_prepare_work_sum_us = sum(target_prepare_by_rank.values())
    target_prepare_critical_rank, target_prepare_critical_us = max(
        target_prepare_by_rank.items(), key=lambda item: item[1], default=(None, 0.0)
    )
    draft_decode = {
        row["stage"]: row for row in draft_kind_rows if row["step_kind"] == "decode"
    }
    target_sampling = draft_decode.get("target.sampling_total")
    draft_propose = draft_decode.get("draft.propose_total")
    proposal_tokens = {
        row["token_count"] for row in proposal_shapes if row["step_kind"] == "decode"
    }
    result = []
    for row in target_rounds:
        token_count = row["token_count"]
        if proposal_tokens and token_count not in proposal_tokens:
            continue
        sampling_us = 0.0 if target_sampling is None else target_sampling["mean_us"]
        propose_us = 0.0 if draft_propose is None else draft_propose["mean_us"]
        result.append(
            {
                "token_count": token_count,
                "round_count": row["round_count"],
                "target_prepare_critical_pp_rank": target_prepare_critical_rank,
                "target_prepare_critical_mean_us": target_prepare_critical_us,
                "target_prepare_work_sum_mean_us": target_prepare_work_sum_us,
                "target_model_serial_mean_us": row["pipeline_mean_us"],
                "target_model_slowest_rank_mean_us": row["critical_mean_us"],
                "target_sampling_mean_us": sampling_us,
                "draft_propose_mean_us": propose_us,
                "measured_dependency_scope_mean_us": (
                    target_prepare_critical_us
                    + row["pipeline_mean_us"]
                    + sampling_us
                    + propose_us
                ),
                "cross_rank_work_sum_mean_us": (
                    target_prepare_work_sum_us
                    + row["pipeline_mean_us"]
                    + sampling_us
                    + propose_us
                ),
                "target_sampling_count": (
                    0 if target_sampling is None else target_sampling["count"]
                ),
                "draft_propose_count": (
                    0 if draft_propose is None else draft_propose["count"]
                ),
            }
        )
    return result


def _module_coverage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_rank_stage = {(row["pp_rank"], row["stage"]): row["total_us"] for row in rows}
    ranks = sorted({row["pp_rank"] for row in rows})
    coverage = []
    for pp_rank in ranks:
        model_total = by_rank_stage.get((pp_rank, "target.model_forward"), 0.0)
        layer_total = by_rank_stage.get((pp_rank, "layer.total"), 0.0)
        model_components = {
            stage: by_rank_stage.get((pp_rank, stage), 0.0)
            for stage in MODEL_COMPONENTS
        }
        layer_components = {
            stage: by_rank_stage.get((pp_rank, stage), 0.0)
            for stage in LAYER_COMPONENTS
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


def _target_steps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model_rows_by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["stage"] in {"target.model_forward", "target.model_graph_replay"}:
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
                "pipeline_total_us": sum(row["total_us"] for row in rank_rows.values()),
            }
        )
    return chunks


def _decode_round_summary(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int | None, list[dict[str, Any]]] = defaultdict(list)
    for step in steps:
        grouped[step["token_count"]].append(step)

    rows = []
    for token_count, token_steps in sorted(
        grouped.items(), key=lambda item: (-1 if item[0] is None else item[0])
    ):
        rank_values: dict[str, list[float]] = defaultdict(list)
        critical_values = []
        pipeline_values = []
        critical_rank_counts: dict[str, int] = defaultdict(int)
        for step in token_steps:
            for pp_rank, elapsed_us in step["rank_total_us"].items():
                rank_values[pp_rank].append(elapsed_us)
            critical_values.append(step["critical_total_us"])
            pipeline_values.append(step["pipeline_total_us"])
            critical_rank_counts[str(step["critical_pp_rank"])] += 1
        rows.append(
            {
                "token_count": token_count,
                "round_count": len(token_steps),
                "rank_latency_us": {
                    pp_rank: {
                        "mean_us": sum(values) / len(values),
                        "p50_us": _percentile(values, 0.50),
                        "p90_us": _percentile(values, 0.90),
                    }
                    for pp_rank, values in sorted(rank_values.items())
                },
                "critical_mean_us": sum(critical_values) / len(critical_values),
                "critical_p50_us": _percentile(critical_values, 0.50),
                "critical_p90_us": _percentile(critical_values, 0.90),
                "critical_rank_counts": dict(sorted(critical_rank_counts.items())),
                "pipeline_mean_us": sum(pipeline_values) / len(pipeline_values),
                "pipeline_p50_us": _percentile(pipeline_values, 0.50),
                "pipeline_p90_us": _percentile(pipeline_values, 0.90),
            }
        )
    return rows


def _decode_serial_stages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model_stages = {"target.model_forward", "target.model_graph_replay"}
    model_rows_by_rank: dict[int, int] = defaultdict(int)
    for row in rows:
        if row["stage"] in model_stages:
            model_rows_by_rank[row["pp_rank"]] += 1
    target_round_count = max(model_rows_by_rank.values(), default=0)
    model_total_us = sum(
        row["total_us"] for row in rows if row["stage"] in model_stages
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["stage"], row["level"])].append(row)

    result = []
    for (stage, level), stage_rows in sorted(grouped.items()):
        total_us = sum(row["total_us"] for row in stage_rows)
        result.append(
            {
                "stage": stage,
                "level": level,
                "pp_ranks": sorted({row["pp_rank"] for row in stage_rows}),
                "count": sum(row["count"] for row in stage_rows),
                "total_us": total_us,
                "per_target_round_us": (
                    total_us / target_round_count if target_round_count else None
                ),
                "model_share_pct": (
                    100.0 * total_us / model_total_us if model_total_us else None
                ),
            }
        )
    return result


def _marlin_moe(
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
            for stage in MARLIN_ADDITIVE_COMPONENTS
        }
        nested = {
            stage: by_rank_stage.get((pp_rank, stage), 0.0)
            for stage in MARLIN_NESTED_RANGES
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
        "additive_stages": list(MARLIN_ADDITIVE_COMPONENTS),
        "nested_ranges": list(MARLIN_NESTED_RANGES),
        "by_rank": by_rank,
        "by_shape_stage": shape_rows,
    }


def render_flow_markdown(flow: str, summary: dict[str, Any]) -> str:
    if flow == "decode":
        rank_note = (
            "Target rounds traverse PP ranks serially. The flat table reports the "
            "largest rank only as a balance diagnostic; the serial section aligns "
            "and sums model work by round. Nested levels must not be added."
        )
    elif flow in {"dspark", "dflash"}:
        draft_name = "DSpark" if flow == "dspark" else "DFlash"
        rank_note = (
            "The target q8 verification traverses PP0/PP1/PP2 and remains in the "
            f"decode flow. Target sampling and the {draft_name} proposal then "
            "execute on "
            "PP2. The cycle section combines only sibling top-level GPU scopes; "
            "nested rows must not be added."
        )
    else:
        rank_note = (
            "PP ranks overlap. Each row reports the rank with the largest total for "
            "that stage; totals from different ranks or nested levels must not be "
            "added."
        )
    lines = [
        f"# DeepSeek V4 {flow} phase profile",
        "",
        rank_note,
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
    elif flow == "decode":
        lines.extend(_render_decode_details(summary))
    elif flow in {"dspark", "dflash"}:
        lines.extend(_render_draft_details(flow, summary))
    return "\n".join(lines) + "\n"


def _render_draft_details(flow: str, summary: dict[str, Any]) -> list[str]:
    draft_name = "DSpark" if flow == "dspark" else "DFlash"
    lines = [
        "",
        f"## Steady-state {draft_name} cycle",
        "",
        (
            "Target input and attention-metadata preparation can run concurrently "
            "across PP workers, so the dependency-path estimate uses only its "
            "critical rank. A target q8 verification for one request then traverses "
            "PP0, PP1, and PP2 serially before PP2 sampling/rejection and proposal. "
            "CPU scheduling, PP transport gaps, output copies, and profiler "
            "synchronization remain outside this visible GPU scope."
        ),
        "",
        (
            "| Target tokens | Rounds | Prepare critical ms/rank | PP3 q8 serial ms | "
            "Sampling ms | Draft proposal ms | Visible dependency scope ms | "
            "Sampling/proposal events |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["cycle_summary"]:
        lines.append(
            f"| {row['token_count']} | {row['round_count']} | "
            f"{row['target_prepare_critical_mean_us'] / 1000:.3f}/"
            f"PP{row['target_prepare_critical_pp_rank']} | "
            f"{row['target_model_serial_mean_us'] / 1000:.3f} | "
            f"{row['target_sampling_mean_us'] / 1000:.3f} | "
            f"{row['draft_propose_mean_us'] / 1000:.3f} | "
            f"{row['measured_dependency_scope_mean_us'] / 1000:.3f} | "
            f"{row['target_sampling_count']}/{row['draft_propose_count']} |"
        )

    lines.extend(
        [
            "",
            (
                "The cross-rank work sum below is an instrumentation/accounting "
                "diagnostic, not wall time. It sums preparation on all workers even "
                "though those worker-local ranges may overlap."
            ),
            "",
            (
                "| Target tokens | Prepare PP work sum ms | Slowest q8 rank ms | "
                "Cross-rank work sum ms |"
            ),
            "|---:|---:|---:|---:|",
        ]
    )
    for row in summary["cycle_summary"]:
        lines.append(
            f"| {row['token_count']} | "
            f"{row['target_prepare_work_sum_mean_us'] / 1000:.3f} | "
            f"{row['target_model_slowest_rank_mean_us'] / 1000:.3f} | "
            f"{row['cross_rank_work_sum_mean_us'] / 1000:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Proposal classes",
            "",
            (
                f"Prefill rows are {draft_name} proposals invoked while target "
                "prompt chunks "
                "are still being processed. Decode rows are steady speculative rounds."
            ),
            "",
            (
                "| Kind | Target input tokens | Calls | Mean ms | P50 ms | "
                "P90 ms | Total ms |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["proposal_shapes"]:
        lines.append(
            f"| {row['step_kind']} | {row['token_count']} | {row['count']} | "
            f"{row['mean_us'] / 1000:.3f} | {row['p50_us'] / 1000:.3f} | "
            f"{row['p90_us'] / 1000:.3f} | {row['total_us'] / 1000:.3f} |"
        )

    report_stages = {
        "target.sampling_total",
        "target.logits",
        "target.sample",
        "target.rejection",
        "target.rejection.inputs",
        "target.rejection.verify_total",
        "target.rejection.apply_sampling_params",
        "target.rejection.verify",
        "target.rejection.counts",
        "draft.propose_total",
        "draft.hidden_prepare",
        "draft.hidden_combine.concat",
        "draft.hidden_combine.projection",
        "draft.hidden_copy",
        "draft.hidden_combine.main_proj",
        "draft.hidden_combine.main_norm",
        "draft.prepare_inputs",
        "draft.context_kv",
        "draft.context_kv.input_norm",
        "draft.context_kv.input_gemm",
        "draft.context_kv.norm",
        "draft.context_kv.rope",
        "draft.context_kv.insert",
        "draft.dispatch",
        "draft.metadata",
        "draft.generate_total",
        "draft.graph_replay",
        "draft.backbone",
        "draft.model.embedding",
        "draft.model.expand",
        "draft.model.reconstruct",
        "draft.model.hc_head",
        "draft.markov_total",
        "draft.hidden_gather",
        "draft.base_logits",
        "draft.logits.norm",
        "draft.logits.lm_head",
        "draft.logits.cache",
        "draft.selector",
        "draft.selector_walk",
        "draft.markov_bias",
        "draft.markov.embed",
        "draft.markov.projection",
        "draft.markov_sample",
    }
    lines.extend(
        [
            "",
            "## Proposal and sampling decomposition by kind",
            "",
            (
                "Rows at total, stage, and detail levels overlap. Compare means "
                "within one nesting level; do not sum children with their parents."
            ),
            "",
            "| Kind | Stage | Level | PP | Calls | Mean us | P50 us | P90 us |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    kind_rows = [
        row
        for row in summary["by_kind_rank_stage"]
        if row["stage"] in report_stages
        or row["stage"].startswith("target.attn_metadata")
    ]
    for row in sorted(
        kind_rows,
        key=lambda item: (item["step_kind"], -item["mean_us"], item["stage"]),
    ):
        lines.append(
            f"| {row['step_kind']} | `{row['stage']}` | {row['level']} | "
            f"{row['pp_rank']} | {row['count']} | {row['mean_us']:.3f} | "
            f"{row['p50_us']:.3f} | {row['p90_us']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Sequential Markov positions",
            "",
            "| Draft position | Stage | Calls | Mean us | P50 us | P90 us |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["by_draft_step_stage"]:
        if row["stage"] not in {"draft.markov_bias", "draft.markov_sample"}:
            continue
        lines.append(
            f"| {row['draft_step']} | `{row['stage']}` | {row['count']} | "
            f"{row['mean_us']:.3f} | {row['p50_us']:.3f} | "
            f"{row['p90_us']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Draft backbone coverage",
            "",
            (
                "Draft model components are siblings within `draft.backbone`; "
                "layer components are siblings within `layer.total`."
            ),
            "",
            (
                "| PP | Backbone ms | Layer totals ms | Backbone residual ms | "
                "Backbone coverage | Layer residual ms | Layer coverage |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["draft_module_coverage"]:
        lines.append(
            f"| {row['pp_rank']} | {row['model_total_us'] / 1000:.3f} | "
            f"{row['layer_total_us'] / 1000:.3f} | "
            f"{row['model_unattributed_us'] / 1000:.3f} | "
            f"{row['model_coverage_pct']:.1f}% | "
            f"{row['layer_unattributed_us'] / 1000:.3f} | "
            f"{row['layer_coverage_pct']:.1f}% |"
        )

    layer_components = {
        (row["pp_rank"], row["layer"], row["stage"]): row["mean_us"]
        for row in summary["by_layer_stage"]
    }
    layer_rows = [
        row for row in summary["by_layer_stage"] if row["stage"] == "layer.total"
    ]
    lines.extend(
        [
            "",
            "## Per-draft-layer cost",
            "",
            (
                "| Layer | PP | Calls | Total us | Attention us | FFN us | "
                "mHC-attn us | mHC-FFN us |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(layer_rows, key=lambda item: item["layer"]):
        key = (row["pp_rank"], row["layer"])
        lines.append(
            f"| `{row['layer']}` | {row['pp_rank']} | {row['count']} | "
            f"{row['mean_us']:.3f} | "
            f"{layer_components.get((*key, 'layer.attention'), 0.0):.3f} | "
            f"{layer_components.get((*key, 'layer.ffn'), 0.0):.3f} | "
            f"{layer_components.get((*key, 'layer.mhc_attention_pre'), 0.0):.3f} | "
            f"{layer_components.get((*key, 'layer.mhc_ffn_pre'), 0.0):.3f} |"
        )

    lines.extend(_render_marlin_moe(summary["marlin_moe"]))
    return lines


def _render_decode_details(summary: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "## Target verification rounds",
        "",
        (
            "A production seven-token speculative round carries up to eight tokens. "
            "Its PP stages form a dependency chain, so the aligned PP model "
            "ranges are additive; the slowest-stage column remains a balance "
            "diagnostic."
        ),
        "",
        (
            "| Tokens | Rounds | PP0 mean/p50/p90 ms | PP1 mean/p50/p90 ms | "
            "PP2 mean/p50/p90 ms | PP serial mean/p50/p90 ms | "
            "Slowest mean/p50/p90 ms | Slowest ranks |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["round_summary"]:
        rank_text = []
        for pp_rank in ("0", "1", "2"):
            latency = row["rank_latency_us"].get(pp_rank)
            rank_text.append(
                "-"
                if latency is None
                else (
                    f"{latency['mean_us'] / 1000:.3f}/"
                    f"{latency['p50_us'] / 1000:.3f}/"
                    f"{latency['p90_us'] / 1000:.3f}"
                )
            )
        critical_ranks = ", ".join(
            f"PP{rank}:{count}" for rank, count in row["critical_rank_counts"].items()
        )
        lines.append(
            f"| {row['token_count']} | {row['round_count']} | {rank_text[0]} | "
            f"{rank_text[1]} | {rank_text[2]} | "
            f"{row['pipeline_mean_us'] / 1000:.3f}/"
            f"{row['pipeline_p50_us'] / 1000:.3f}/"
            f"{row['pipeline_p90_us'] / 1000:.3f} | "
            f"{row['critical_mean_us'] / 1000:.3f}/"
            f"{row['critical_p50_us'] / 1000:.3f}/"
            f"{row['critical_p90_us'] / 1000:.3f} | {critical_ranks} |"
        )

    lines.extend(
        [
            "",
            "## Serial PP3 model-work decomposition",
            "",
            (
                "The same stage is summed across PP ranks because a single decode "
                "target pass traverses PP0, PP1, then PP2. Parent and child levels "
                "still overlap and must not be added to each other."
            ),
            "",
            (
                "| Stage | Level | PP ranks | Events | Per target round ms | "
                "PP3 model share |"
            ),
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in sorted(
        summary["serial_stage"], key=lambda item: item["total_us"], reverse=True
    ):
        per_round = row["per_target_round_us"]
        share = row["model_share_pct"]
        lines.append(
            f"| `{row['stage']}` | {row['level']} | "
            f"{','.join(str(rank) for rank in row['pp_ranks'])} | "
            f"{row['count']} | "
            f"{'-' if per_round is None else f'{per_round / 1000:.3f}'} | "
            f"{'-' if share is None else f'{share:.1f}%'} |"
        )

    metadata_rows = [
        row
        for row in summary["by_rank_stage"]
        if row["stage"].startswith("target.attn_metadata")
    ]
    lines.extend(
        [
            "",
            "## Attention metadata decomposition",
            "",
            (
                "Builder rows are nested under `prepare_attn`; common/group rows "
                "and builder rows must not be added to their parents."
            ),
            "",
            "| PP rank | Stage | Count | Mean us | P50 us | P90 us | CPU mean us |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(metadata_rows, key=lambda item: item["total_us"], reverse=True):
        cpu_mean = row["cpu_mean_us"]
        lines.append(
            f"| {row['pp_rank']} | `{row['stage']}` | {row['count']} | "
            f"{row['mean_us']:.3f} | {row['p50_us']:.3f} | "
            f"{row['p90_us']:.3f} | "
            f"{'-' if cpu_mean is None else f'{cpu_mean:.3f}'} |"
        )

    lines.extend(
        [
            "",
            "### KV cache group parent breakdown",
            "",
            (
                "The group id comes from the runtime KV-cache layout. A builder "
                "type can appear more than once when different groups need separate "
                "persistent graph buffers."
            ),
            "",
            (
                "| PP rank | KV group | Stage | Count | Mean us | P50 us | "
                "P90 us | CPU mean us |"
            ),
            "|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        summary["metadata_group_stage"],
        key=lambda item: item["total_us"],
        reverse=True,
    ):
        cpu_mean = row["cpu_mean_us"]
        lines.append(
            f"| {row['pp_rank']} | {row['kv_cache_group']} | "
            f"`{row['stage']}` | {row['count']} | {row['mean_us']:.3f} | "
            f"{row['p50_us']:.3f} | {row['p90_us']:.3f} | "
            f"{'-' if cpu_mean is None else f'{cpu_mean:.3f}'} |"
        )

    lines.extend(
        [
            "",
            "## Model and layer coverage",
            "",
            (
                "Sibling layer modules are additive. Model-level and layer-level "
                "rows are separate nesting levels."
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

    family_rows = {
        (row["compress_ratio"], row["stage"]): row
        for row in summary["compression_family_stage"]
    }
    lines.extend(
        [
            "",
            "## Compression-family layer modules",
            "",
            (
                "Means are per layer per target round across PP ranks. They rank "
                "work, but are not PP wall time."
            ),
            "",
            (
                "| C-ratio | Layers | Layer total us | Attention us | FFN us | "
                "mHC-attn us | mHC-FFN us |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for compress_ratio in sorted(
        {row["compress_ratio"] for row in summary["compression_family_stage"]}
    ):
        total = family_rows.get((compress_ratio, "layer.total"))
        if total is None:
            continue

        def family_mean(stage: str, ratio: int = compress_ratio) -> float:
            row = family_rows.get((ratio, stage))
            return 0.0 if row is None else row["mean_us"]

        lines.append(
            f"| {compress_ratio} | {total['layer_count']} | "
            f"{total['mean_us']:.3f} | "
            f"{family_mean('layer.attention'):.3f} | "
            f"{family_mean('layer.ffn'):.3f} | "
            f"{family_mean('layer.mhc_attention_pre'):.3f} | "
            f"{family_mean('layer.mhc_ffn_pre'):.3f} |"
        )

    attention_stages = (
        "layer.attention",
        "attention.input_gemm_overlap",
        "attention.input_gemm.qr_kv",
        "attention.input_gemm.mla_compressor_score",
        "attention.input_gemm.indexer_weights",
        "attention.input_gemm.indexer_compressor_score",
        "attention.input_gemm.join",
        "attention.qkv_rmsnorm",
        "attention.q_index_compress_mla",
        "attention.qkv_insert",
        "attention.indexer_branch",
        "attention.mla_compressor_branch",
        "attention.qicm_join",
        "indexer.query_prepare",
        "indexer.compress_cache",
        "indexer.query_cache_join",
        "indexer.select",
        "indexer.paged_logits",
        "indexer.topk",
        "attention.sparse_mla",
        "sparse_mla.topk_metadata",
        "sparse_mla.attention",
        "compressor.save_partial_states",
        "compressor.compress_norm_rope_store",
        "attention.output_projection",
        "attention.output_projection.inv_rope_wo_a",
        "attention.output_projection.wo_b",
    )
    lines.extend(
        [
            "",
            "## Attention and indexer by compression family",
            "",
            (
                "Rows include different nesting levels. In particular, indexer "
                "query/cache ranges overlap, and detail rows must not be summed "
                "with their parents."
            ),
            "",
            "| C-ratio | Stage | Count | Mean us | P50 us | P90 us |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for compress_ratio in sorted(
        {row["compress_ratio"] for row in summary["compression_family_stage"]}
    ):
        for stage in attention_stages:
            row = family_rows.get((compress_ratio, stage))
            if row is None:
                continue
            lines.append(
                f"| {compress_ratio} | `{stage}` | {row['count']} | "
                f"{row['mean_us']:.3f} | {row['p50_us']:.3f} | "
                f"{row['p90_us']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Per-layer decode cost",
            "",
            (
                "| Layer | C-ratio | PP rank | Samples | Layer mean us | "
                "Attention us | FFN us | mHC-attn us | mHC-FFN us |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    layer_rows = [
        row for row in summary["by_layer_stage"] if row["stage"] == "layer.total"
    ]
    layer_components = {
        (row["pp_rank"], row["layer"], row["stage"]): row["mean_us"]
        for row in summary["by_layer_stage"]
    }
    for row in sorted(layer_rows, key=lambda item: item["mean_us"], reverse=True):
        key = (row["pp_rank"], row["layer"])
        lines.append(
            f"| `{row['layer']}` | {row['compress_ratio']} | {row['pp_rank']} | "
            f"{row['count']} | {row['mean_us']:.3f} | "
            f"{layer_components.get((*key, 'layer.attention'), 0.0):.3f} | "
            f"{layer_components.get((*key, 'layer.ffn'), 0.0):.3f} | "
            f"{layer_components.get((*key, 'layer.mhc_attention_pre'), 0.0):.3f} | "
            f"{layer_components.get((*key, 'layer.mhc_ffn_pre'), 0.0):.3f} |"
        )
    lines.extend(_render_marlin_moe(summary["marlin_moe"]))
    return lines


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
            "excluded from the sum. The parent remainder contains nested work and "
            "cross-stream overlap; it is not an unknown-kernel total."
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
        for stage, total_us in rank_row["nested_ranges_us"].items():
            share = 100.0 * total_us / parent_us if parent_us else 0.0
            lines.append(
                f"| {rank_row['pp_rank']} | `{stage}` (overlap diagnostic) | "
                f"{total_us / 1000:.3f} | {share:.1f}% |"
            )
        lines.append(
            f"| {rank_row['pp_rank']} | `parent remainder (nested/overlap)` | "
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
