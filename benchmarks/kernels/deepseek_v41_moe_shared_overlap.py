# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen shared Marlin MLP overlap with the accepted routed GEMM pair."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from benchmarks.kernels.deepseek_v41_moe_harness import (
    STAGE_SHAPES,
    align_routes,
    environment,
    interleaved_ab,
)
from benchmarks.kernels.deepseek_v41_moe_tilelang_gemm import (
    activation_scales,
    build_packed_weights,
)
from vllm.model_executor.layers.fused_moe.experts.dsv41_sm80_hybrid_prefill import (
    SmallTailRoutes,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import _router_gemm_pinned
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_mxfp8_marlin_linear,
    prepare_mxfp8_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_int8_sm80 import (
    quantize_per_token_int8,
)


def load_shared(checkpoint, layer, device):
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    tensors, provenance = {}, {}
    for weight in ("w1", "w3", "w2"):
        for kind in ("weight", "scale"):
            name = f"layers.{layer}.ffn.shared_experts.{weight}.{kind}"
            with safe_open(checkpoint / index[name], framework="pt", device="cpu") as f:
                tensor = f.get_tensor(name)
            provenance[name] = {
                "shape": list(tensor.shape),
                "sha256": hashlib.sha256(
                    tensor.view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            }
            tensors[weight, kind] = tensor.view(torch.uint8).to(device)

    def linear(names):
        module = torch.nn.Module()
        weight = torch.cat([tensors[name, "weight"] for name in names]).view(
            torch.float8_e4m3fn
        )
        scale = torch.cat([tensors[name, "scale"] for name in names]).repeat_interleave(
            32, dim=0
        )
        module.output_size_per_partition, module.input_size_per_partition = weight.shape
        module.weight = torch.nn.Parameter(weight, requires_grad=False)
        module.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
        prepare_mxfp8_layer_for_marlin(module)
        return module

    w13, w2 = linear(("w1", "w3")), linear(("w2",))

    def act(x):
        out = torch.empty(
            (*x.shape[:-1], x.shape[-1] // 2), dtype=x.dtype, device=x.device
        )
        torch.ops._C.silu_and_mul_with_clamp(out, x, 10.0, 1.0, 0.0)
        return out

    def apply(x):
        def project(module, value):
            return apply_mxfp8_marlin_linear(
                value,
                module.weight,
                module.weight_scale,
                module.workspace,
                module.output_size_per_partition,
                module.input_size_per_partition,
            )

        return project(w2, act(project(w13, x)))

    assert w13.workspace.data_ptr() != w2.workspace.data_ptr()
    return apply, provenance


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Overlap shared MLP with router/align/quant, joining before routed GEMM",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/media/z/data6/models/DeepSeek-V4.1-Flash"),
    )
    parser.add_argument(
        "--routes",
        type=Path,
        default=Path("artifacts/dsv41f-ced-prefill-round5-20261003/experiment.json"),
    )
    args = parser.parse_args()
    device = torch.device("cuda", 0)
    assert torch.cuda.get_device_capability() == (8, 0)
    assert str(torch.cuda.get_device_properties(device).uuid) == (
        args.expected_uuid.removeprefix("GPU-")
    )
    torch.manual_seed(41)
    torch.set_default_dtype(torch.bfloat16)
    info = environment()
    info.pop("clock_and_power", None)
    result = {
        "environment": info,
        "scope": (
            "Shared checkpoint Marlin MLP plus synthetic routed W13/W2 GEMM pair. "
            "Captured DSpark5 M2044 corpus routes or synthetic six-hot routes. "
            "Not full Experts.apply or serving evidence."
        ),
        "timing_boundary": (
            "shared MLP, route update and both GEMM stages; fork/join events included; "
            "prepare-only additionally includes real router gate/topk "
            "and W13 INT8 quant"
        ),
        "mode": "prepare_only" if args.prepare_only else "full_overlap",
        "rows": [],
    }
    packed = {
        stage: build_packed_weights(384, stage, device, seed=41, scale_mode="varied")
        for stage in STAGE_SHAPES
    }
    profile = next(
        p
        for p in json.loads(args.routes.read_text())["profiles"]
        if p["mode"] == "hybrid"
    )
    for layer in (0, 14):
        shared, provenance = load_shared(args.checkpoint, layer, device)
        hidden = torch.randn((2044, 5120), device=device)
        if args.prepare_only:
            index = json.loads(
                (args.checkpoint / "model.safetensors.index.json").read_text()
            )["weight_map"]
            gate = {}
            for kind in ("weight", "bias"):
                name = f"layers.{layer}.ffn.gate.{kind}"
                with safe_open(
                    args.checkpoint / index[name], framework="pt", device="cpu"
                ) as f:
                    gate[kind] = f.get_tensor(name).to(device)
        else:
            gate = None
        for workload in ("corpus", "hot"):
            route = next(
                r
                for rank in profile["ranks"]
                for r in rank["routes"]
                if r["layer"] == layer
            )
            ids = (
                torch.tensor(route["ids"], dtype=torch.int32, device=device)
                if workload == "corpus"
                else torch.arange(6, dtype=torch.int32, device=device)
                .expand(2044, -1)
                .contiguous()
            )
            rw = torch.tensor(route["weights"], device=device, dtype=torch.float32)
            aligned = align_routes(ids, 64, 384)
            routes = SmallTailRoutes(ids.numel(), 384, device)
            activations, scales, output = {}, {}, {}
            for stage, (n, k, slots_source) in STAGE_SHAPES.items():
                rows = ids.numel() if slots_source else ids.shape[0]
                activations[stage] = torch.randint(
                    -127, 128, (rows, k), device=device, dtype=torch.int8
                )
                scales[stage] = activation_scales(rows, device, "varied")
                output[stage] = torch.empty((ids.numel(), n), device=device)

            def routed(
                routes=routes,
                activations=activations,
                scales=scales,
                output=output,
                rw=rw,
            ):
                for stage in STAGE_SHAPES:
                    weights = packed[stage]
                    routes.gemm(
                        activations[stage],
                        scales[stage],
                        weights["packed"],
                        weights["shift2"],
                        weights["channel_scale"],
                        output[stage],
                        output[stage],
                        rw,
                        stage == "w2",
                    )

            def router_gate(gate=gate, hidden=hidden):
                return (
                    _router_gemm_pinned(hidden, gate["weight"])
                    if gate is not None
                    else None
                )

            def prepare(
                logits,
                gate=gate,
                hidden=hidden,
                ids=ids,
                routes=routes,
                aligned=aligned,
                activations=activations,
                scales=scales,
            ):
                if gate is not None:
                    # Keep captured routing for GEMM occupancy; router results here
                    # only reproduce preparation work, not checkpoint activations.
                    fused_topk_bias(
                        hidden, logits, "sqrtsoftplus", gate["bias"], 6, True
                    )
                    current_aligned = align_routes(ids, 64, 384)
                    activations["w13"], scales["w13"] = quantize_per_token_int8(hidden)
                else:
                    current_aligned = aligned
                routes.update(current_aligned)

            stream = torch.cuda.Stream()
            ready, done = torch.cuda.Event(), torch.cuda.Event()
            last_shared = {}

            def baseline(
                shared=shared,
                hidden=hidden,
                router_gate=router_gate,
                prepare=prepare,
                routed=routed,
                last_shared=last_shared,
            ):
                logits = router_gate()
                last_shared["baseline"] = shared(hidden)
                prepare(logits)
                routed()

            def candidate(
                shared=shared,
                hidden=hidden,
                router_gate=router_gate,
                prepare=prepare,
                routed=routed,
                last_shared=last_shared,
                stream=stream,
                ready=ready,
                done=done,
            ):
                main_stream = torch.cuda.current_stream()
                ready.record(main_stream)
                with torch.cuda.stream(stream):
                    ready.wait(stream)
                    last_shared["candidate"] = shared(hidden)
                    done.record(stream)
                prepare(router_gate())
                if args.prepare_only:
                    done.wait(main_stream)
                routed()
                done.wait(main_stream)

            baseline()
            torch.accelerator.synchronize()
            expected = {stage: value.clone() for stage, value in output.items()}
            expected_shared = last_shared["baseline"].clone()
            assert torch.isfinite(expected_shared).all()
            assert all(torch.isfinite(value).all() for value in expected.values())
            candidate()
            torch.accelerator.synchronize()

            def validate(
                which,
                output=output,
                last_shared=last_shared,
                expected=expected,
                expected_shared=expected_shared,
            ):
                def check(graph):
                    for value in output.values():
                        value.fill_(float("nan"))
                    last_shared[which].fill_(float("nan"))
                    graph.replay()
                    torch.accelerator.synchronize()
                    assert torch.equal(
                        last_shared[which].view(torch.int16),
                        expected_shared.view(torch.int16),
                    )
                    assert all(
                        torch.equal(
                            value.view(torch.int16), expected[stage].view(torch.int16)
                        )
                        for stage, value in output.items()
                    )

                return check

            assert torch.equal(
                last_shared["candidate"].view(torch.int16),
                expected_shared.view(torch.int16),
            )
            timing = interleaved_ab(
                baseline,
                candidate,
                rounds=3,
                warmup=2,
                graph_calls=4,
                samples=7,
                settle_ms=250,
                baseline_validator=validate("baseline"),
                candidate_validator=validate("candidate"),
            )
            result["rows"].append(
                {
                    "layer": layer,
                    "workload": workload,
                    "shared_weights": provenance,
                    "bitwise_exact": True,
                    "timing": timing,
                }
            )
            args.out.write_text(json.dumps(result, indent=2) + "\n")
            print(
                layer,
                workload,
                timing["baseline_ratio"],
                timing["measurement_status"],
                flush=True,
            )


if __name__ == "__main__":
    main()
