# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe verification components without loading or restarting a PP5 model.

Uses layer-20 checkpoint weights and synthetic activations, routes and KV.
These are isolated component costs, not a decomposition of a serving trace.
An event brackets one captured call; an L2 flush precedes every sample.
"""

import argparse
import hashlib
import json
import statistics
import struct
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from benchmarks.kernels.deepseek_v41_moe_tilelang_m6 import load_sampled_weights
from vllm.model_executor.determinism.batch_invariant import enable_batch_invariant_math
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    apply_candidate_mask,
    select_candidate_blocks,
)
from vllm.model_executor.kernels.mhc.tilelang import (
    mhc_post_tilelang,
    mhc_pre_delayed_tilelang,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.experts import mxfp4_int8_sm80 as moe_impl
from vllm.model_executor.layers.fused_moe.router.gate_linear import _router_gemm_pinned
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_mxfp8_marlin_linear,
    prepare_mxfp8_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
)
from vllm.models.deepseek_v4_1.common.ops import quantize_and_insert_k_cache
from vllm.v1.attention.ops.mqa_logits_triton import fp8_paged_mqa_logits_triton
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_decode
from vllm.v1.worker.workspace import current_workspace_manager, init_workspace_manager


class CheckpointReader:
    def __init__(self, model):
        self.model = model
        self.mapping = json.loads((model / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        self.headers = {}
        self.provenance = []

    def read(self, suffix, layer=20):
        key = f"layers.{layer}.{suffix}"
        path = self.model / self.mapping[key]
        if path not in self.headers:
            with path.open("rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                self.headers[path] = (8 + length, json.loads(handle.read(length)))
        base, header = self.headers[path]
        info = header[key]
        start, end = info["data_offsets"]
        with path.open("rb") as handle:
            handle.seek(base + start)
            raw = handle.read(end - start)
        if len(raw) != end - start:
            raise ValueError(f"Short checkpoint read: {key}")
        dtype = {
            "F8_E4M3": torch.float8_e4m3fn,
            "F8_E8M0": torch.uint8,
            "BF16": torch.bfloat16,
            "F32": torch.float32,
        }[info["dtype"]]
        self.provenance.append(
            dict(
                key=key, shard=str(path), **info, sha256=hashlib.sha256(raw).hexdigest()
            )
        )
        return (
            torch.frombuffer(bytearray(raw), dtype=dtype)
            .reshape(info["shape"])
            .to("cuda")
        )

    def fp8(self, name):
        weight = self.read(f"{name}.weight")
        scales = self.read(f"{name}.scale").repeat_interleave(32, dim=0)
        assert scales.shape == (weight.shape[0], weight.shape[1] // 32)
        return weight, scales


def make_linear(weight, scale):
    layer = torch.nn.Module()
    layer.input_size_per_partition = weight.shape[1]
    layer.output_size_per_partition = weight.shape[0]
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    prepare_mxfp8_layer_for_marlin(layer)
    return layer


def linear(layer, inputs):
    return apply_mxfp8_marlin_linear(
        inputs,
        layer.weight,
        layer.weight_scale,
        layer.workspace,
        layer.output_size_per_partition,
        layer.input_size_per_partition,
        use_full_k=True,
    )


def capture(call):
    for _ in range(5):
        expected = call()
    torch.accelerator.synchronize()
    expected = expected.clone()
    assert torch.isfinite(expected).all()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    output.fill_(float("nan") if output.is_floating_point() else -1)
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    return graph, call, output, expected


def time_cases(cases, samples):
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    props = torch.cuda.get_device_properties()
    assert flush.numel() > props.L2_cache_size
    result = {name: {q: [] for q in qs} for name, qs in cases.items()}
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    for round_index in range(3):
        for name, qs in cases.items():
            order = sorted(qs, reverse=round_index % 2 == 1)
            for q in order:
                print(f"Timing {name} q={q}", flush=True)
                graph = qs[q][0]
                for _ in range(5):
                    graph.replay()
                for start, end in zip(starts, ends, strict=True):
                    flush.zero_()
                    start.record()
                    graph.replay()
                    end.record()
                ends[-1].synchronize()
                result[name][q].append(
                    [
                        s.elapsed_time(e) * 1000
                        for s, e in zip(starts, ends, strict=True)
                    ]
                )
        print(f"Measurement round {round_index + 1}/3 complete", flush=True)
    return result


def sparse_case(q, compressed):
    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.125

    queries = rand(q, 64, 512)
    cache = torch.zeros((64, 128, 584), device="cuda", dtype=torch.uint8)
    slots = torch.arange(8192, device="cuda", dtype=torch.int64)
    quantize_and_insert_k_cache(rand(8192, 512), cache, slots, 128)
    main = torch.arange(128, device="cuda", dtype=torch.int32).repeat(q, 1)
    main_ptr = torch.arange(q + 1, device="cuda", dtype=torch.int32) * 128
    extra = torch.arange(512, device="cuda", dtype=torch.int32).repeat(q)
    extra_ptr = torch.arange(q + 1, device="cuda", dtype=torch.int32) * 512
    lengths = torch.full((q,), 128, device="cuda", dtype=torch.int32)
    output = torch.empty_like(queries)
    sink = torch.zeros(64, device="cuda")

    def call():
        rocm_sparse_attn_decode(
            queries,
            cache if compressed else None,
            cache,
            not compressed,
            None,
            None,
            main,
            lengths,
            main.flatten(),
            main_ptr,
            extra if compressed else None,
            extra_ptr if compressed else None,
            sink,
            512**-0.5,
            512,
            448,
            64,
            output,
        )
        return output

    return call


def moe_cases(weights, q):
    parallel = SimpleNamespace(
        tp_size=1, pcp_size=1, dp_size=1, ep_size=1, sp_size=1, use_ep=False
    )
    config = SimpleNamespace(
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
    quant = FusedMoEQuantConfig.make(
        w1_scale=weights["w13"]["channel_scale"],
        w2_scale=weights["w2"]["channel_scale"],
    )
    layer = SimpleNamespace(
        w13_mxfp4_int8_shift2=weights["w13"]["shift2"],
        w2_mxfp4_int8_shift2=weights["w2"]["shift2"],
    )
    with patch.object(moe_impl, "get_current_vllm_config_or_none", lambda: None):
        experts = moe_impl.Mxfp4Int8Sm80Experts(config, quant)
        experts.process_weights_after_loading(layer)
    assert experts._tilelang_decode_gemms is None
    assert experts._tilelang_activation_quant is None
    hidden = torch.randn(q, 5120, device="cuda", dtype=torch.bfloat16)
    routing_weights = torch.full((q, 6), 1 / 6, device="cuda")
    results = {}
    for step in (0, 3, 6):
        ids = (
            137
            + torch.arange(6, device="cuda", dtype=torch.int32)[None, :]
            + step * torch.arange(q, device="cuda", dtype=torch.int32)[:, None]
        )
        ws13 = torch.empty(q * 6, 6912, device="cuda", dtype=torch.bfloat16)
        ws2 = torch.empty(q * 6, 5120, device="cuda", dtype=torch.bfloat16)
        output = torch.empty(q, 5120, device="cuda", dtype=torch.bfloat16)

        def call(ids=ids, ws13=ws13, ws2=ws2, output=output):
            experts.apply(
                output,
                hidden,
                weights["w13"]["packed"],
                weights["w2"]["packed"],
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
            return output

        results[f"moe_new_experts_{step}"] = call
    return results


def extra_cases(reader):
    shared1, scale1 = reader.fp8("ffn.shared_experts.w1")
    shared3, scale3 = reader.fp8("ffn.shared_experts.w3")
    up = make_linear(torch.cat([shared1, shared3]), torch.cat([scale1, scale3]))
    down = make_linear(*reader.fp8("ffn.shared_experts.w2"))
    compressor = reader.read("attn.compressor.wkv.weight")
    index_q = make_linear(*reader.fp8("attn.indexer.wq_b"))
    weights_proj = reader.read("attn.indexer.weights_proj.weight")
    kv_proj = make_linear(*reader.fp8("attn.wkv"))
    calls = {}
    for q in range(1, 7):
        hidden = torch.randn(q, 5120, device="cuda", dtype=torch.bfloat16)

        def shared(hidden=hidden):
            gate_up = linear(up, hidden)
            activated = torch.empty_like(gate_up[:, :2304])
            torch.ops._C.silu_and_mul_with_clamp(activated, gate_up, 10.0, 1.0, 0.0)
            return linear(down, activated)

        calls[q] = {
            "shared_expert_serial": shared,
            "compressor_projection_c1": partial(
                torch.mm, hidden, compressor.T, out_dtype=torch.float32
            ),
            "indexer_weights_projection": partial(
                torch.mm, hidden, weights_proj.T, out_dtype=torch.float32
            ),
            "indexer_query_projection": partial(
                linear,
                index_q,
                torch.randn(q, 1280, device="cuda", dtype=torch.bfloat16),
            ),
            "attn_kv_projection": partial(linear, kv_proj, hidden),
        }
        context = 32768
        query = torch.randn(1, q, 32, 128, device="cuda").to(torch.float8_e4m3fn)
        cache = torch.zeros(256, 128, 1, 132, device="cuda", dtype=torch.uint8)
        flat = cache.view(256, -1)
        flat[:, : 128 * 128].copy_(
            torch.randn(256, 128 * 128, device="cuda")
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        flat[:, 128 * 128 :].view(torch.float32).fill_(1.0)
        weights = torch.ones(q, 32, device="cuda") / 32
        lengths = torch.full((1,), context, device="cuda", dtype=torch.int32)
        table = torch.arange(256, device="cuda", dtype=torch.int32).view(1, -1)
        logits_call = partial(
            fp8_paged_mqa_logits_triton,
            query,
            cache,
            weights,
            lengths,
            table,
            context,
            clean_logits=False,
        )
        indices = torch.empty(q, 512, device="cuda", dtype=torch.int32)

        def indexer(logits_call=logits_call, indices=indices, q=q, lengths=lengths):
            logits = logits_call()
            torch.ops._C.top_k_per_row_decode(
                logits,
                q,
                lengths,
                indices,
                q,
                logits.stride(0),
                logits.stride(1),
                512,
            )
            return indices

        calls[q]["indexer_logits_topk_c1"] = indexer
    return calls


def indexer_cases():
    """Compare live-width and FULL-capture-width indexer work at 32K context."""
    calls = {}
    for q in range(1, 7):
        calls[q] = {}
        for ratio in (1, 2):
            block_size = 128 // ratio
            context = 32768 // ratio
            query = torch.randn(1, q, 32, 128, device="cuda").to(torch.float8_e4m3fn)
            cache = torch.zeros(
                256, block_size, 1, 132, device="cuda", dtype=torch.uint8
            )
            flat = cache.view(256, -1)
            flat[:, : block_size * 128].copy_(
                torch.randn(256, block_size * 128, device="cuda")
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
            )
            flat[:, block_size * 128 :].view(torch.float32).fill_(1.0)
            weights = torch.ones(q, 32, device="cuda") / 32
            lengths = torch.full((1, q), context, device="cuda", dtype=torch.int32)
            # Only the first 256 entries are read: surplus CTAs exit before
            # loading a page. The table retains the configured 1M width.
            table = torch.zeros(1, 8192, device="cuda", dtype=torch.int32)
            table[:, :256] = torch.arange(256, device="cuda", dtype=torch.int32)
            next_n = q
            if q > 2:
                # The SM80 metadata builder flattens unsupported native q.
                query = query.reshape(q, 1, 32, 128)
                lengths = lengths.reshape(q, 1)
                table = table.repeat(q, 1)
                next_n = 1
            for width in (32768, 1048576):
                modes = ("none", "select", "mask") if ratio == 1 else ("none",)
                for mode in modes:
                    indices = torch.empty(q, 512, device="cuda", dtype=torch.int32)
                    candidates = torch.arange(2048, device="cuda", dtype=torch.int32)
                    candidates = candidates.repeat(q, 1)
                    workspace = torch.empty(
                        1024 * 1024, device="cuda", dtype=torch.uint8
                    )
                    logits_call = partial(
                        fp8_paged_mqa_logits_triton,
                        query,
                        cache,
                        weights,
                        lengths,
                        table,
                        width,
                        clean_logits=False,
                    )

                    def call(
                        logits_call=logits_call,
                        indices=indices,
                        candidates=candidates,
                        workspace=workspace,
                        lengths=lengths,
                        mode=mode,
                        q=q,
                        next_n=next_n,
                    ):
                        logits = logits_call()
                        if mode == "select":
                            select_candidate_blocks(
                                logits, None, lengths.reshape(-1), 2048, 8, candidates
                            )
                        elif mode == "mask":
                            apply_candidate_mask(
                                logits, None, lengths.reshape(-1), candidates, 8
                            )
                        if q == 6:
                            torch.ops._C.top_k_per_row_decode(
                                logits,
                                next_n,
                                lengths,
                                indices,
                                q,
                                logits.stride(0),
                                logits.stride(1),
                                512,
                            )
                        else:
                            torch.ops._C.persistent_topk(
                                logits,
                                lengths,
                                indices,
                                workspace,
                                512,
                                logits.shape[1],
                            )
                        return indices

                    name = f"indexer_c{ratio}_width{width}_{mode}"
                    calls[q][name] = call
    return calls


def fused_cases(reader):
    """Measure the fused projections used by the serving attention path."""
    q_weight, q_scale = reader.fp8("attn.wq_a")
    kv_weight, kv_scale = reader.fp8("attn.wkv")
    qkv = make_linear(torch.cat([q_weight, kv_weight]), torch.cat([q_scale, kv_scale]))
    compressor_c1 = reader.read("attn.compressor.wkv.weight")
    compressor_c2 = torch.cat(
        [
            reader.read("attn.compressor.wkv.weight", layer=2),
            reader.read("attn.compressor.wgate.weight", layer=2),
        ]
    )
    return {
        q: {
            "attn.fused_wqa_wkv": partial(
                linear, qkv, torch.randn(q, 5120, device="cuda", dtype=torch.bfloat16)
            ),
            "compressor.fused_wkv_wgate_c1": partial(
                torch.mm,
                torch.randn(q, 5120, device="cuda", dtype=torch.bfloat16),
                compressor_c1.T,
                out_dtype=torch.float32,
            ),
            "compressor.fused_wkv_wgate_c2": partial(
                torch.mm,
                torch.randn(q, 5120, device="cuda", dtype=torch.bfloat16),
                compressor_c2.T,
                out_dtype=torch.float32,
            ),
        }
        for q in range(1, 7)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument(
        "--components", choices=("core", "extra", "indexer", "fused"), default="core"
    )
    args = parser.parse_args()
    if args.samples < 3:
        parser.error("At least three samples are required")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(20261001)
    enable_batch_invariant_math()
    init_workspace_manager(torch.device("cuda"))
    # Sparse-decode partials and Marlin scratch must retain their addresses
    # across all six captures. Reserve and lock before the first capture.
    manager = current_workspace_manager()
    manager.get_simultaneous(((64 * 1024 * 1024,), torch.uint8))
    manager.lock()
    reader = CheckpointReader(args.model)
    if args.components != "core":
        if args.components == "extra":
            calls = extra_cases(reader)
        elif args.components == "fused":
            calls = fused_cases(reader)
        else:
            calls = indexer_cases()
        cases = {}
        for q, components in calls.items():
            for name, call in components.items():
                cases.setdefault(name, {})[q] = capture(call)
        save_result(args, reader, cases)
        return
    layers = {}
    for name in ("attn.wq_a", "attn.wq_b", "attn.wo_b"):
        layers[name] = make_linear(*reader.fp8(name))
    wa, sa = reader.fp8("attn.wo_a")
    wo_a = dequant_mxfp8_to_bf16(wa, sa).view(8, 1024, 4096)
    router = reader.read("ffn.gate.weight")
    fn = reader.read("hc_attn_fn")
    scale, base = reader.read("hc_attn_scale"), reader.read("hc_attn_base")
    norm = reader.read("attn_norm.weight")
    weights = load_sampled_weights(args.model, torch.device("cuda"), args.out_dir)
    cases = {}
    for q in range(1, 7):
        calls = {
            name: partial(
                linear,
                layer,
                torch.randn(
                    q,
                    layer.input_size_per_partition,
                    device="cuda",
                    dtype=torch.bfloat16,
                ),
            )
            for name, layer in layers.items()
        }
        o = torch.randn(q, 8, 4096, device="cuda", dtype=torch.bfloat16)
        calls["attn.wo_a"] = lambda o=o: torch.einsum("tgd,grd->tgr", o, wo_a)
        hidden = torch.randn(q, 5120, device="cuda", dtype=torch.bfloat16)
        calls["router"] = partial(_router_gemm_pinned, hidden, router)
        residual = torch.randn(q, 4, 5120, device="cuda", dtype=torch.bfloat16)
        mix = torch.zeros(q, 4, device="cuda")
        pre = partial(
            mhc_pre_delayed_tilelang,
            residual,
            fn,
            scale,
            base,
            1e-6,
            1e-6,
            1e-6,
            1.0,
            20,
            pre_mix=mix,
            norm_weight=norm,
            norm_eps=1e-6,
        )
        post_mix, res_mix, x, _ = pre()
        calls["mhc_pre"] = lambda pre=pre: pre()[2]
        calls["mhc_post"] = partial(mhc_post_tilelang, x, residual, post_mix, res_mix)
        calls["sparse_swa128"] = sparse_case(q, False)
        calls["sparse_swa128_top512"] = sparse_case(q, True)
        calls.update(moe_cases(weights, q))
        for name, call in calls.items():
            cases.setdefault(name, {})[q] = capture(call)
        print(f"Captured q={q}, {len(calls)} components", flush=True)
    save_result(args, reader, cases)


def save_result(args, reader, cases):
    for qs in cases.values():
        for graph, _, output, expected in qs.values():
            output.fill_(float("nan") if output.is_floating_point() else -1)
            graph.replay()
            torch.accelerator.synchronize()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
    raw = time_cases(cases, args.samples)
    result = dict(
        scope="isolated checkpoint components; synthetic hidden/routes/KV; no PP",
        timing="CUDA event + one-call CUDA graph, 64MiB L2 flush outside timing",
        strict_bf16=True,
        marlin_full_k=True,
        tilelang_moe=False,
        components=args.components,
        torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        samples=args.samples,
        checkpoint=reader.provenance,
        raw_us=raw,
        median_us={
            name: {q: statistics.median(sum(rounds, [])) for q, rounds in qs.items()}
            for name, qs in raw.items()
        },
    )
    (args.out_dir / "components.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["median_us"], indent=2), flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
