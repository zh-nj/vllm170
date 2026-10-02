# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded Engram gate experiments with checkpoint weights and graph timing.

The prototypes absorb TileKernels' FP32 weight product and early image-token
branch while retaining vLLM's reduction and signed-sqrt formula. This measures
only the gate after WKV; it does not measure lookup, projection, or serving.
"""

import argparse
import hashlib
import json
import statistics
import struct
import subprocess
from functools import partial
from pathlib import Path

import torch

from vllm.models.deepseek_v4_1.common.engram import _fused_engram_post_wkv_kernel
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_kv_tokens"])
def _gate_candidate(
    hidden_states,
    kv,
    q_weight,
    k_weight,
    token_mask,
    output,
    num_kv_tokens,
    hidden_stride_t,
    hidden_stride_h,
    hidden_stride_d,
    kv_stride_t,
    kv_stride_d,
    q_stride_h,
    q_stride_d,
    k_stride_h,
    k_stride_d,
    mask_stride,
    output_stride_t,
    output_stride_h,
    output_stride_d,
    eps,
    clamp_value,
    DIM: tl.constexpr,
    HC_MULT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    FUSED_WEIGHT: tl.constexpr,
    EARLY_MASK: tl.constexpr,
):
    program_idx = tl.program_id(0)
    token_idx = (program_idx // HC_MULT).to(tl.int64)
    hc_idx = program_idx % HC_MULT
    source_valid = token_idx < num_kv_tokens
    offsets = tl.arange(0, BLOCK_SIZE)
    valid = offsets < DIM
    hidden = tl.load(
        hidden_states
        + token_idx * hidden_stride_t
        + hc_idx * hidden_stride_h
        + offsets * hidden_stride_d,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    active = True
    if HAS_MASK:
        active = tl.load(
            token_mask + token_idx * mask_stride, mask=source_valid, other=0
        )
    result = hidden
    if not EARLY_MASK or active:
        key = tl.load(
            kv + token_idx * kv_stride_t + (hc_idx * DIM + offsets) * kv_stride_d,
            mask=source_valid & valid,
            other=0.0,
        ).to(tl.float32)
        q = tl.load(
            q_weight + hc_idx * q_stride_h + offsets * q_stride_d,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        if FUSED_WEIGHT:
            terms = hidden * q * key
        else:
            k = tl.load(
                k_weight + hc_idx * k_stride_h + offsets * k_stride_d,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            terms = hidden * q * k * key
        hidden_rms = tl.rsqrt(tl.sum(hidden * hidden, axis=0) / DIM + eps)
        key_rms = tl.rsqrt(tl.sum(key * key, axis=0) / DIM + eps)
        dot = tl.sum(terms, axis=0)
        dot *= hidden_rms * key_rms * tl.rsqrt(DIM * 1.0)
        gate_input = tl.sqrt(tl.maximum(tl.abs(dot), clamp_value))
        gate_input = tl.where(dot < 0.0, -gate_input, gate_input)
        gate = tl.sigmoid(gate_input)
        if HAS_MASK:
            gate = tl.where(active, gate, 0.0)
        value = tl.load(
            kv + token_idx * kv_stride_t + (HC_MULT * DIM + offsets) * kv_stride_d,
            mask=source_valid & valid,
            other=0.0,
        ).to(tl.float32)
        result = hidden + gate * value
    tl.store(
        output
        + token_idx * output_stride_t
        + hc_idx * output_stride_h
        + offsets * output_stride_d,
        result,
        mask=valid,
    )


def read_weights(model: Path, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Read only the small q/k tensors, without mapping the embedding table."""
    index = json.loads((model / "model.safetensors.index.json").read_text())
    prefix = f"layers.{layer}.engram."
    with (model / index["weight_map"][prefix + "q_weight"]).open("rb") as stream:
        (header_size,) = struct.unpack("<Q", stream.read(8))
        header = json.loads(stream.read(header_size))
        weights = []
        for name in ("q_weight", "k_weight"):
            record = header[prefix + name]
            assert record["dtype"] == "BF16" and record["shape"] == [4, 5120]
            start, end = record["data_offsets"]
            stream.seek(8 + header_size + start)
            data = bytearray(stream.read(end - start))
            assert len(data) == end - start
            weights.append(torch.frombuffer(data, dtype=torch.bfloat16).view(4, 5120))
    return tuple(weights)


def reference(hidden, kv, q, k, mask):
    """Use vLLM's positive signed-sqrt at either sign of exact zero."""
    h, key = hidden.float(), kv[:, : 4 * 5120].float().view(-1, 4, 5120)
    value = kv[:, 4 * 5120 :].float()
    rstd = torch.rsqrt(h.square().mean(-1) + 1e-20)
    rstd *= torch.rsqrt(key.square().mean(-1) + 1e-20)
    dot = (h * q.float() * k.float() * key).sum(-1) * rstd * 5120**-0.5
    magnitude = dot.abs().clamp_min(1e-6).sqrt()
    gate = torch.where(dot < 0, -magnitude, magnitude).sigmoid()
    gate.masked_fill_(~mask[:, None], 0)
    return (h + gate[:, :, None] * value[:, None, :]).bfloat16()


def capture(call, repetitions):
    for _ in range(3):
        call()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repetitions):
            call()
    return graph


def measure(graphs, repetitions, samples, cold):
    flush = torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    results = {name: [] for name in graphs}
    names = list(graphs)
    for sample in range(samples):
        order = names[sample % len(names) :] + names[: sample % len(names)]
        if sample % 2:
            order.reverse()
        for name in order:
            if cold:
                flush.zero_()
            events[0].record()
            graphs[name].replay()
            events[1].record()
            events[1].synchronize()
            results[name].append(events[0].elapsed_time(events[1]) * 1000 / repetitions)
    return {
        name: {"median_us": statistics.median(values), "raw_us": values}
        for name, values in results.items()
    }


def launch(
    h, kv, q, k, weight, mask, output, tokens, name, fold, early, num_kv_tokens=None
):
    qw = weight if fold else q
    kwargs = dict(DIM=5120, HC_MULT=4, BLOCK_SIZE=8192, HAS_MASK=True, num_warps=8)
    kernel = _fused_engram_post_wkv_kernel
    if name != "baseline":
        kernel = _gate_candidate
        kwargs.update(FUSED_WEIGHT=fold, EARLY_MASK=early)
    return kernel[(tokens * 4,)](
        h,
        kv,
        qw,
        k,
        mask,
        output,
        tokens if num_kv_tokens is None else num_kv_tokens,
        *h.stride(),
        *kv.stride(),
        *qw.stride(),
        *k.stride(),
        mask.stride(0),
        *output.stride(),
        1e-20,
        1e-6,
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("/media/z/data6/models/DeepSeek-V4.1-Flash")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 6, 12, 256, 2048])
    parser.add_argument(
        "--masks",
        nargs="+",
        default=["all_active", "half_active", "none_active"],
        choices=["all_active", "half_active", "none_active"],
    )
    parser.add_argument("--samples", type=int, default=24)
    args = parser.parse_args()
    torch.manual_seed(20261002)
    device = torch.cuda.get_device_properties()
    assert torch.cuda.get_device_capability() == (8, 0)
    assert str(device.uuid) == args.expected_uuid.removeprefix("GPU-")
    payload = {
        "gpu": str(device),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "scope": "gate only; random activations, actual checkpoint q/k; no WKV/lookup",
        "timing": (
            "interleaved CUDA Graph/Event; warm 128 calls; "
            "cold one call, 64MiB flush outside events"
        ),
        "cases": [],
    }
    for layer in (1, 14):
        q, k = (weight.cuda() for weight in read_weights(args.model, layer))
        weight = q.float() * k.float()
        for tokens in args.tokens:
            h = torch.randn(tokens, 4, 5120, device="cuda", dtype=torch.bfloat16)
            kv = torch.randn(tokens, 5 * 5120, device="cuda", dtype=torch.bfloat16)
            for mask_name in args.masks:
                mask = torch.ones(tokens, dtype=torch.bool, device="cuda")
                if mask_name == "half_active":
                    mask[1::2] = False
                elif mask_name == "none_active":
                    mask.zero_()
                outputs, calls, compiled = {}, {}, {}
                variants = {
                    "baseline": (False, False),
                    "control": (False, False),
                    "fp32_fold": (True, False),
                    "early_mask": (False, True),
                    "fold_early": (True, True),
                }
                for name, (fold, early) in variants.items():
                    output = torch.empty_like(h)
                    outputs[name] = output

                    call = partial(
                        launch,
                        h,
                        kv,
                        q,
                        k,
                        weight,
                        mask,
                        output,
                        tokens,
                        name,
                        fold,
                        early,
                    )
                    calls[name] = call
                    compiled[name] = call()
                expected = reference(h, kv, q, k, mask)
                torch.testing.assert_close(
                    outputs["baseline"], expected, atol=1e-2, rtol=1e-2
                )
                comparisons = {}
                for name, output in outputs.items():
                    torch.testing.assert_close(
                        output, outputs["baseline"], rtol=0, atol=0
                    )
                    comparisons[name] = {
                        "equal_baseline": torch.equal(output, outputs["baseline"]),
                        "different_bits": int(
                            (
                                output.view(torch.int16)
                                != outputs["baseline"].view(torch.int16)
                            ).sum()
                        ),
                        "registers": compiled[name].n_regs,
                        "spills": compiled[name].n_spills,
                        "shared_bytes": compiled[name].metadata.shared,
                    }
                timing = {}
                for cold, reps in ((False, 128), (True, 1)):
                    graphs = {name: capture(call, reps) for name, call in calls.items()}
                    timing["cold" if cold else "warm"] = measure(
                        graphs, reps, args.samples, cold
                    )
                record = dict(
                    layer=layer,
                    tokens=tokens,
                    mask=mask_name,
                    correctness=comparisons,
                    timing=timing,
                )
                payload["cases"].append(record)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(payload, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "layer": layer,
                            "tokens": tokens,
                            "mask": mask_name,
                            **{
                                mode: {
                                    name: round(v["median_us"], 3)
                                    for name, v in values.items()
                                }
                                for mode, values in timing.items()
                            },
                        }
                    ),
                    flush=True,
                )
    payload["benchmark_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
