# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded M=6 PRMT grouped-GEMM probe against the SM80 CUTLASS path."""

import argparse
import hashlib
import json
import struct
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from benchmarks.kernels import deepseek_v41_moe_harness as harness
from benchmarks.kernels.deepseek_v41_moe_tilelang_gemm import (
    GemmConfig,
    TileLangMoeGemm,
    build_packed_weights,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.experts import mxfp4_int8_sm80 as impl
from vllm.model_executor.layers.quantization.utils.mxfp4_int8_sm80 import (
    remap_mxfp4_weight_for_int8,
)


def load_sampled_weights(model, device, dest, *, stages=("w13", "w2")):
    """Read 36 layer20 experts; remaining E384 slots are explicitly zero."""
    mapping = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    headers, provenance = {}, []

    def read(expert, name, kind):
        key = f"layers.20.ffn.experts.{expert}.{name}.{kind}"
        path = model / mapping[key]
        if path not in headers:
            with path.open("rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                headers[path] = (8 + length, json.loads(handle.read(length)))
        base, header = headers[path]
        info = header[key]
        start, end = info["data_offsets"]
        with path.open("rb") as handle:
            handle.seek(base + start)
            raw = handle.read(end - start)
        if info["dtype"] != ("I8" if kind == "weight" else "F8_E8M0"):
            raise ValueError(f"unexpected dtype for {key}: {info}")
        if len(raw) != end - start:
            raise ValueError(f"short checkpoint read: {key}")
        provenance.append(
            dict(
                key=key,
                shard=str(path),
                absolute_offsets=[base + start, base + end],
                shape=info["shape"],
                dtype=info["dtype"],
                sha256=hashlib.sha256(raw).hexdigest(),
                bytes=len(raw),
            )
        )
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(
            info["shape"]
        )

    weights = {}
    for stage, names in (("w13", ("w1", "w3")), ("w2", ("w2",))):
        if stage not in stages:
            continue
        raw_weight, raw_scale = [], []
        for expert in range(137, 173):
            raw_weight.append(
                torch.cat([read(expert, name, "weight") for name in names])
            )
            raw_scale.append(torch.cat([read(expert, name, "scale") for name in names]))
        packed, shift, scale = remap_mxfp4_weight_for_int8(
            torch.stack(raw_weight).to(device),
            torch.stack(raw_scale).to(device),
            headroom_bits=3,
            reuse_weight_storage=True,
        )
        del raw_weight, raw_scale
        value = {}
        for name, tensor in (
            ("packed", packed),
            ("shift2", shift),
            ("channel_scale", scale),
        ):
            expanded = torch.zeros(
                (384, *tensor.shape[1:]), dtype=tensor.dtype, device=device
            )
            expanded[137:173].copy_(tensor)
            value[name] = expanded
        weights[stage] = value
    (dest / "checkpoint-provenance.json").write_text(
        json.dumps(
            dict(
                layer=20,
                real_experts=list(range(137, 173)),
                other_expert_slots="zero; never selected by this probe",
                routing_and_hidden="synthetic; no live MoE input capture",
                headroom_bits=3,
                reuse_weight_storage=True,
                raw_checkpoint=provenance,
            ),
            indent=2,
        )
    )
    return weights


def require_exact(actual, expected):
    if not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
        raise AssertionError(harness.compare(actual, expected))


def graph_validator(output, expected):
    def check(captured):
        output.fill_(float("nan"))
        captured.replay()
        torch.accelerator.synchronize()
        require_exact(output, expected)

    return check


def make_apply_case(experts, layer, hidden, routing_weights, ids):
    ws13 = torch.empty((36, 6912), device=hidden.device, dtype=torch.bfloat16)
    ws2 = torch.empty((36, 5120), device=hidden.device, dtype=torch.bfloat16)
    output = ws13.flatten()[: 6 * 5120].view(6, 5120)

    def call():
        experts.apply(
            output,
            hidden,
            layer.w13_weight,
            layer.w2_weight,
            routing_weights,
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

    return call, output, ws2, ids


def call_apply_cases(cases):
    for case in cases:
        case[0]()


def apply_graph_validator(cases, expected):
    def check(graph):
        for case in cases:
            case[1].fill_(float("nan"))
            case[2].fill_(float("nan"))
        graph.replay()
        torch.accelerator.synchronize()
        for case, (output, slots) in zip(cases, expected):
            require_exact(case[1], output)
            require_exact(case[2], slots)

    return check


def probe_apply(
    weights,
    w2_kernel,
    dest,
    *,
    runtime=False,
    sampled=False,
    baseline_runtime=False,
    baseline_runtime_kernel=None,
):
    """Compare only W2; both arms use the live activation fusion."""
    device = weights["w13"]["packed"].device
    parallel = SimpleNamespace(
        tp_size=1, pcp_size=1, dp_size=1, ep_size=1, sp_size=1, use_ep=False
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
        swiglu_limit=10.0,
        swiglu_alpha=None,
        swiglu_beta=None,
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="deepseek_v41",
                hidden_size=5120,
                moe_intermediate_size=2304,
                n_routed_experts=384,
            )
        ),
        additional_config={"dsv41_sm80_moe_tilelang": True},
    )
    layer = SimpleNamespace(
        w13_weight=weights["w13"]["packed"],
        w2_weight=weights["w2"]["packed"],
        w13_mxfp4_int8_shift2=weights["w13"]["shift2"],
        w2_mxfp4_int8_shift2=weights["w2"]["shift2"],
    )
    quant = FusedMoEQuantConfig.make(
        w1_scale=weights["w13"]["channel_scale"],
        w2_scale=weights["w2"]["channel_scale"],
    )
    with patch.object(impl, "get_current_vllm_config_or_none", lambda: config):
        baseline = impl.Mxfp4Int8Sm80Experts(moe, quant)
        baseline.process_weights_after_loading(layer)
        candidate = impl.Mxfp4Int8Sm80Experts(moe, quant)
        candidate.process_weights_after_loading(layer)
    assert baseline._tilelang_activation_quant is not None
    assert candidate._tilelang_activation_quant is not None
    if baseline_runtime:
        if not runtime:
            raise ValueError("a live TileLang baseline requires a runtime candidate")
        baseline_kernel = (
            baseline_runtime_kernel
            if baseline_runtime_kernel is not None
            else baseline._tilelang_m6_w2
        )
        assert baseline_kernel is not None
    else:
        baseline._tilelang_m6_w2 = None
    if not runtime:
        candidate._tilelang_m6_w2 = None
    generator = torch.Generator(device=device).manual_seed(942)
    hidden = torch.randn(
        (6, 5120), dtype=torch.bfloat16, device=device, generator=generator
    )
    routing_weights = torch.tensor(
        [[0.07, 0.11, 0.13, 0.17, 0.23, 0.29]], device=device
    ).repeat(6, 1)
    registered = set()
    dispatch_calls = {"baseline_w2": 0, "candidate_w2": 0}
    original = torch.ops._moe_C.dsv4_mxfp4_int8_sm80_moe_gemm
    if baseline_runtime:

        def baseline_call(*args):
            dispatch_calls["baseline_w2"] += 1
            return baseline_kernel(*args)

        baseline._tilelang_m6_w2 = baseline_call
    if runtime:
        runtime_kernel = w2_kernel if baseline_runtime else candidate._tilelang_m6_w2
        assert runtime_kernel is not None

        def runtime_call(*args):
            dispatch_calls["candidate_w2"] += 1
            return runtime_kernel(*args)

        candidate._tilelang_m6_w2 = runtime_call

    def dispatch(*args):
        if args[-1]:
            if not runtime and args[6].data_ptr() in registered:
                dispatch_calls["candidate_w2"] += 1
                w2_kernel(*args[:5], *args[8:11], args[6])
                return None
            dispatch_calls["baseline_w2"] += 1
        return original(*args)

    results = []
    with patch.object(torch.ops._moe_C, "dsv4_mxfp4_int8_sm80_moe_gemm", dispatch):
        for pattern, period in (("hot", 1), ("spread", 1), ("rotating", 40)):
            registered.clear()
            dispatch_calls = {"baseline_w2": 0, "candidate_w2": 0}
            pairs = []
            for i in range(period):
                if pattern == "hot":
                    ids = torch.arange(6, device=device, dtype=torch.int64).repeat(6, 1)
                else:
                    ids = torch.arange(36, device=device, dtype=torch.int64).view(6, 6)
                ids = (
                    (ids + i * 7).remainder(36) + 137
                    if sampled
                    else (ids + 137 + i * 7).remainder(384)
                )
                a = make_apply_case(baseline, layer, hidden, routing_weights, ids)
                b = make_apply_case(candidate, layer, hidden, routing_weights, ids)
                registered.add(b[2].data_ptr())
                pairs.append((a, b))
            for a, b in pairs:
                a[0]()
                b[0]()
                require_exact(a[1], b[1])
                require_exact(a[2], b[2])
            assert dispatch_calls == {
                "baseline_w2": period,
                "candidate_w2": period,
            }, dispatch_calls
            mutations = 0
            if period == 1:
                saved_hidden, saved_ids = hidden.clone(), pairs[0][0][3].clone()
                graphs = [
                    harness.capture_graph(pairs[0][j][0], warmup=3, graph_calls=1)
                    for j in (0, 1)
                ]
                for change in (19, 41):
                    hidden.copy_(
                        torch.randn(
                            hidden.shape,
                            dtype=hidden.dtype,
                            device=device,
                            generator=generator,
                        )
                    )
                    ids = torch.arange(36, device=device, dtype=torch.int64).view(6, 6)
                    changed_ids = (
                        (ids + change).remainder(36) + 137
                        if sampled
                        else (ids + change).remainder(384)
                    )
                    pairs[0][0][3].copy_(changed_ids)
                    for j, graph in enumerate(graphs):
                        pairs[0][j][1].fill_(float("nan"))
                        pairs[0][j][2].fill_(float("nan"))
                        graph.replay()
                    torch.accelerator.synchronize()
                    require_exact(pairs[0][0][1], pairs[0][1][1])
                    require_exact(pairs[0][0][2], pairs[0][1][2])
                    mutations += 1
                hidden.copy_(saved_hidden)
                pairs[0][0][3].copy_(saved_ids)
                for a, b in pairs:
                    a[0]()
                    b[0]()
            expected = [(a[1].clone(), a[2].clone()) for a, b in pairs]
            arms = [[pair[j] for pair in pairs] for j in (0, 1)]
            result = harness.interleaved_ab(
                partial(call_apply_cases, arms[0]),
                partial(call_apply_cases, arms[1]),
                rounds=3,
                warmup=3,
                graph_calls=40 // period,
                samples=15,
                calls_per_repeat=period,
                settle_ms=500,
                baseline_validator=apply_graph_validator(arms[0], expected),
                candidate_validator=apply_graph_validator(arms[1], expected),
            )
            row = dict(
                pattern=pattern,
                tokens=6,
                clamp=10.0,
                bitwise_exact=True,
                graph_mutations=mutations,
                w2_host_calls=dict(dispatch_calls),
                runtime_candidate=runtime,
                runtime_baseline=baseline_runtime,
                actual_weight_sample=sampled,
                scope="Experts.apply; same live activation; only W2 differs",
                **result,
            )
            results.append(row)
            (dest / "apply-ab.json").write_text(json.dumps(results, indent=2))
            print(
                json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "pattern",
                            "baseline_median_p50_us",
                            "candidate_median_p50_us",
                            "baseline_ratio",
                            "baseline_self_spread",
                            "measurement_status",
                        )
                    }
                ),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--apply-only", action="store_true")
    parser.add_argument("--runtime-apply", action="store_true")
    parser.add_argument("--sampled-checkpoint", type=Path)
    args = parser.parse_args()
    dest = args.out_dir
    dest.mkdir(parents=True, exist_ok=True)
    if torch.cuda.get_device_capability() != (8, 0):
        raise ValueError("this probe requires SM80")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(2026100104)
    if args.sampled_checkpoint:
        if not args.runtime_apply:
            raise ValueError("sampled checkpoint requires --runtime-apply")
        weights = load_sampled_weights(args.sampled_checkpoint, device, dest)
    else:
        weights = {
            stage: build_packed_weights(384, stage, device, seed=seed)
            for stage, seed in (("w13", 940), ("w2", 941))
        }
        for weight in weights.values():
            weight["channel_scale"].mul_(1e-3)
    results = []
    variants = {}
    if args.runtime_apply:
        probe_apply(
            weights, None, dest, runtime=True, sampled=bool(args.sampled_checkpoint)
        )
        metadata = harness.environment()
        metadata.pop("clock_and_power", None)
        metadata.update(
            git_head=harness.git_head(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        (dest / "apply-metadata.json").write_text(json.dumps(metadata, indent=2))
        return
    if args.apply_only:
        config = GemmConfig(
            n=5120,
            k=2304,
            num_experts=384,
            topk=6,
            block_m=16,
            block_n=128,
            block_k=128,
            num_stages=3,
            threads=128,
            source_rows_are_slots=True,
            decode="prmt",
        )
        kernel = TileLangMoeGemm.create(config, tokens=6, slots=36)
        probe_apply(weights, kernel.kernel, dest)
        metadata = harness.environment()
        metadata.pop("clock_and_power", None)
        metadata.update(
            git_head=harness.git_head(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        (dest / "apply-metadata.json").write_text(json.dumps(metadata, indent=2))
        return
    for stage, n, k, rows in (("w13", 4608, 5120, 6), ("w2", 5120, 2304, 36)):
        for block_n in (64, 128):
            config = GemmConfig(
                n=n,
                k=k,
                num_experts=384,
                topk=6,
                block_m=16,
                block_n=block_n,
                block_k=128,
                num_stages=3 if stage == "w2" else 2,
                threads=128,
                source_rows_are_slots=stage == "w2",
                decode="prmt",
            )
            candidate = TileLangMoeGemm.create(config, tokens=6, slots=36)
            variants[stage, block_n] = candidate
            (dest / f"{stage}-bn{block_n}.cu").write_text(candidate.source)
            print(
                json.dumps(
                    dict(
                        stage=stage,
                        block_n=block_n,
                        compile_seconds=candidate.compile_seconds,
                    )
                ),
                flush=True,
            )
        weight = weights[stage]
        a = torch.randint(
            -127, 128, (rows, k), device=device, dtype=torch.int8, generator=generator
        )
        scale = torch.linspace(0.0007, 0.013, rows, device=device)
        for pattern in ("hot", "spread"):
            if pattern == "hot":
                ids = torch.arange(6, device=device, dtype=torch.int32).repeat(6, 1)
            else:
                ids = torch.arange(36, device=device, dtype=torch.int32).view(6, 6)
            ids.add_(137)
            assert harness.token_expert_exclusivity(ids)
            metadata = harness.align_routes(ids, 16, 384)
            expected = harness.reference_stage(
                a,
                scale,
                weight["packed"],
                weight["shift2"],
                weight["channel_scale"],
                ids,
                stage=stage,
            )
            baseline = torch.empty((36, n), dtype=torch.bfloat16, device=device)
            candidate_out = torch.empty_like(baseline)
            routing_weights = torch.ones((6, 6), device=device)

            cutlass = partial(
                torch.ops._moe_C.dsv4_mxfp4_int8_sm80_moe_gemm,
                a,
                scale,
                weight["packed"],
                weight["shift2"],
                weight["channel_scale"],
                baseline,
                baseline,
                routing_weights,
                *metadata,
                36,
                16,
                128,
                stage == "w2",
            )

            cutlass()
            require_exact(baseline, expected)
            work = harness.compute_route_work(
                ids,
                16,
                {stage: (weight["packed"], weight["shift2"], weight["channel_scale"])},
            )
            for block_n in (64, 128):
                candidate = variants[stage, block_n]

                tilelang_call = partial(
                    candidate.kernel,
                    a,
                    scale,
                    weight["packed"],
                    weight["shift2"],
                    weight["channel_scale"],
                    *metadata,
                    candidate_out,
                )

                tilelang_call()
                require_exact(candidate_out, expected)
                graph = harness.capture_graph(tilelang_call, warmup=3, graph_calls=1)
                saved_a = a.clone()
                for change in (19, 41):
                    a.copy_(
                        torch.randint(
                            -127,
                            128,
                            a.shape,
                            device=device,
                            dtype=torch.int8,
                            generator=generator,
                        )
                    )
                    changed_ids = (ids + change).remainder(384)
                    changed_metadata = harness.align_routes(changed_ids, 16, 384)
                    for target, value in zip(metadata, changed_metadata):
                        target.copy_(value)
                    candidate_out.fill_(float("nan"))
                    graph.replay()
                    cutlass()
                    torch.accelerator.synchronize()
                    require_exact(candidate_out, baseline)
                a.copy_(saved_a)
                for target, value in zip(metadata, harness.align_routes(ids, 16, 384)):
                    target.copy_(value)

                result = harness.interleaved_ab(
                    cutlass,
                    tilelang_call,
                    rounds=3,
                    warmup=3,
                    graph_calls=16,
                    samples=9,
                    settle_ms=300,
                    baseline_validator=graph_validator(baseline, expected),
                    candidate_validator=graph_validator(candidate_out, expected),
                )
                latency = result["candidate_median_p50_us"]
                row = dict(
                    stage=stage,
                    pattern=pattern,
                    tokens=6,
                    routed_rows=36,
                    cutlass_block_n=128,
                    tilelang_block_n=block_n,
                    bitwise_exact=True,
                    graph_mutations=2,
                    scope="one GEMM; precomputed routing metadata; CUDA Graph/Event",
                    useful_tops=work.useful_flops[stage] / latency / 1e6,
                    logical_weight_gb_s=work.stage_bytes[stage]["total"]
                    / latency
                    / 1e3,
                    active_experts=len(work.active_experts),
                    **result,
                )
                results.append(row)
                (dest / "stage-ab.json").write_text(json.dumps(results, indent=2))
                print(
                    json.dumps(
                        {
                            key: row[key]
                            for key in (
                                "stage",
                                "pattern",
                                "tilelang_block_n",
                                "active_experts",
                                "baseline_median_p50_us",
                                "candidate_median_p50_us",
                                "baseline_ratio",
                                "baseline_self_spread",
                                "measurement_status",
                            )
                        }
                    ),
                    flush=True,
                )
    metadata = harness.environment()
    metadata.pop("clock_and_power", None)
    metadata.update(
        git_head=harness.git_head(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        candidate_source_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(dest.glob("*.cu"))
        },
    )
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
