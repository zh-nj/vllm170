# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Measure what the MXFP4 -> INT8 expert route costs in accuracy vs BF16.

The DeepSeek-V4.1-Flash checkpoint stores experts as MXFP4 codes plus UE8M0
per-32 scales. Two serving routes exist for the same bytes:

* ``mxfp4_int8_sm80`` (served): the codes are *remapped* into an int8 operand
  (``code' << shift`` per 32-element block, one fp32 scale per output channel)
  and the GEMM runs on int8 tensor cores with int8 activations (A8W4).
* the bf16 route (``--moe-backend marlin``, W4A16): the codes are dequantized
  to bf16 exactly and the GEMM runs in bf16 with bf16 activations.

``weights`` mode compares the two weight representations on real checkpoint
tensors. ``moe`` mode runs the whole expert MLP for both routes on real weights
and reports the output error, decomposed into the weight-remap part and the
int8-activation part.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from vllm.model_executor.layers.quantization.utils.mxfp4_int8_sm80 import (
    _E2M1_SIGNED_X2,
    _as_uint8_storage,
    _unpack_codes,
    compute_mxfp4_int8_remap_stats,
)

MODEL_DIR = Path("/media/z/data6/models/DeepSeek-V4.1-Flash")
INDEX = MODEL_DIR / "model.safetensors.index.json"


def load_experts(layer: int, experts: list[int]) -> dict[str, torch.Tensor]:
    """Load ``experts`` of ``layer`` as [E, N, packed_K] uint8 + [E, N, K/32] scales."""
    from safetensors import safe_open

    import json as _json

    weight_map = _json.loads(INDEX.read_text())["weight_map"]
    files: dict[str, list[str]] = {}
    for name in ("w1", "w3", "w2"):
        for expert in experts:
            key = f"layers.{layer}.ffn.experts.{expert}.{name}.weight"
            files.setdefault(weight_map[key], []).append(key)

    raw: dict[str, torch.Tensor] = {}
    for path, keys in files.items():
        with safe_open(MODEL_DIR / path, framework="pt") as handle:
            for key in keys:
                raw[key] = handle.get_tensor(key)
                scale_key = key.replace(".weight", ".scale")
                raw[scale_key] = handle.get_tensor(scale_key)

    out = {}
    for name in ("w1", "w3", "w2"):
        w = torch.stack(
            [raw[f"layers.{layer}.ffn.experts.{e}.{name}.weight"] for e in experts]
        ).view(torch.uint8)
        s = torch.stack(
            [raw[f"layers.{layer}.ffn.experts.{e}.{name}.scale"] for e in experts]
        )
        out[name] = w.contiguous()
        out[f"{name}_scale"] = _as_uint8_storage(s)
    w13 = torch.cat([out["w1"], out["w3"]], dim=1).contiguous()
    w13_scale = torch.cat([out["w1_scale"], out["w3_scale"]], dim=1).contiguous()
    return {
        "w13": w13,
        "w13_scale": w13_scale,
        "w2": out["w2"],
        "w2_scale": out["w2_scale"],
    }


def dequant_reference(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Exact MXFP4 -> float32: code values (x2 table) times 2**(exp-1).

    The x2 table stores twice the e2m1 value, hence the halving.
    """
    e, n, packed_k = weight.shape
    k_blocks = packed_k * 2 // 32
    codes = _unpack_codes(weight).view(e, n, k_blocks, 32)
    table = _E2M1_SIGNED_X2.to(device=codes.device, dtype=torch.float32)
    q_x2 = table[codes.long()]
    exp = scale.to(torch.int16) - 127
    out = q_x2 * torch.exp2(exp.float()).unsqueeze(-1) * 0.5
    return out.reshape(e, n, k_blocks * 32)


def remap_int8_route(
    weight: torch.Tensor, scale: torch.Tensor, headroom_bits: int
) -> tuple[torch.Tensor, dict]:
    """Float32 weights the int8 route actually multiplies by, plus clip stats.

    Mirrors ``_compute_remap_stats``: the stored code is rounded onto the e2m1
    grid after shifting, and the per-channel scale carries the block exponents.
    """
    e, n, packed_k = weight.shape
    k_blocks = packed_k * 2 // 32
    codes = _unpack_codes(weight).view(e, n, k_blocks, 32)
    table = _E2M1_SIGNED_X2.to(device=codes.device, dtype=torch.float32)
    q_x2 = table[codes.long()]
    exp = (scale.to(torch.int16) - 127).to(torch.int32)
    max_exp = exp.max(dim=2, keepdim=True).values
    delta = (max_exp - exp).clamp_min(0)
    shift = (headroom_bits - delta).clamp(0, 3)
    channel_scale = torch.exp2(
        (max_exp.squeeze(2) - headroom_bits).to(torch.float32)
    ) * 0.5

    target_i8 = torch.round(
        q_x2 * torch.exp2((headroom_bits - delta).float()).unsqueeze(-1)
    )
    clipped = (target_i8 < -128) | (target_i8 > 127)
    target_i8 = target_i8.clamp(-128, 127)
    target_code_value = target_i8 / torch.exp2(shift.float()).unsqueeze(-1)
    mag = nearest_e2m1_x2(target_code_value)
    signed = torch.where(target_code_value < 0, -mag, mag)
    signed = torch.where(q_x2 == 0, torch.zeros_like(signed), signed)
    dropped = (delta > headroom_bits).unsqueeze(-1) & (q_x2 != 0)
    represented = signed * torch.exp2(shift.float()).unsqueeze(-1)
    represented = represented * channel_scale[:, :, None, None]
    represented = represented.reshape(e, n, k_blocks * 32)
    stats = {
        "dropped_weights": int(dropped.sum()),
        "dropped_fraction": float(dropped.float().mean()),
        "clip_count": int(clipped.sum()),
    }
    return represented, stats


def nearest_e2m1_x2(values: torch.Tensor) -> torch.Tensor:
    """Nearest value on the x2 e2m1 magnitude grid {0,1,2,3,4,6,8,12}."""
    a = values.abs()
    mag = torch.full_like(a, 12.0)
    mag = torch.where(a <= 10.0, torch.full_like(a, 8.0), mag)
    mag = torch.where(a <= 7.0, torch.full_like(a, 6.0), mag)
    mag = torch.where(a <= 5.0, torch.full_like(a, 4.0), mag)
    mag = torch.where(a <= 3.5, torch.full_like(a, 3.0), mag)
    mag = torch.where(a <= 2.5, torch.full_like(a, 2.0), mag)
    mag = torch.where(a <= 1.5, torch.full_like(a, 1.0), mag)
    return torch.where(a <= 0.5, torch.zeros_like(a), mag)


def rel_l2(approx: torch.Tensor, exact: torch.Tensor) -> float:
    diff = (approx - exact).float()
    return float(diff.norm() / exact.float().norm().clamp_min(1e-30))


def weights_mode(args) -> None:
    device = torch.device(args.device)
    rows = []
    for layer in args.layers:
      for expert in args.experts:
        data = load_experts(layer, [expert])
        data = {k: v.to(device) for k, v in data.items()}
        for name in ("w13", "w2"):
            w, s = data[name], data[f"{name}_scale"]
            exact = dequant_reference(w, s)
            row = {
                "layer": layer,
                "expert": expert,
                "tensor": name,
                "weights": int(w.shape[0] * w.shape[1] * w.shape[2] * 2),
                "abs_mean": float(exact.abs().mean()),
                "abs_max": float(exact.abs().max()),
            }
            for headroom in (1, 2, 3):
                approx, clip = remap_int8_route(w, s, headroom)
                prod = compute_mxfp4_int8_remap_stats(w, s, headroom_bits=headroom)
                row[f"rel_l2_h{headroom}"] = rel_l2(approx, exact)
                row[f"dropped_h{headroom}"] = clip["dropped_fraction"]
                row[f"exact_rate_h{headroom}"] = prod["exact_rate_all"]
                row[f"row_span_max_h{headroom}"] = prod["row_span_max"]
            rows.append(row)
            print(
                f"L{layer:>2} e{expert:>3} {name:>3} rel_l2 h1={row['rel_l2_h1']:.5f} "
                f"h2={row['rel_l2_h2']:.5f} h3={row['rel_l2_h3']:.5f} | "
                f"dropped h3={row['dropped_h3'] * 100:.3f}% "
                f"exact h3={row['exact_rate_h3'] * 100:.3f}% | "
                f"|w| mean={row['abs_mean']:.4g}"
            )
    for headroom in (1, 2, 3):
        vals = [r[f"rel_l2_h{headroom}"] for r in rows]
        print(
            f"headroom={headroom}: rel_l2 mean {statistics.fmean(vals):.5f} "
            f"max {max(vals):.5f}"
        )
    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.output}")


def silu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return (gate.float() * torch.sigmoid(gate.float())) * up.float()


def quantize_per_token_int8(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a = a.float()
    scale = a.abs().amax(dim=1) / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(a / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale


def moe_mode(args) -> None:
    device = torch.device(args.device)
    data = load_experts(args.layer, args.experts)
    w13_exact = dequant_reference(data["w13"], data["w13_scale"]).to(device)
    w2_exact = dequant_reference(data["w2"], data["w2_scale"]).to(device)
    w13_int8, _ = remap_int8_route(data["w13"], data["w13_scale"], 3)
    w2_int8, _ = remap_int8_route(data["w2"], data["w2_scale"], 3)
    w13_int8 = w13_int8.to(device)
    w2_int8 = w2_int8.to(device)

    n_experts = w13_exact.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    tokens = args.tokens
    topk = min(args.topk, n_experts)
    ids = torch.stack(
        [torch.randperm(n_experts, generator=gen)[:topk] for _ in range(tokens)]
    ).to(device)
    weights = torch.full((tokens, topk), 1.0 / topk, dtype=torch.float32, device=device)

    act = torch.randn(tokens, w13_exact.shape[2], generator=gen).to(device) * args.act_std
    act = act.to(torch.bfloat16)

    a_q, a_scale = quantize_per_token_int8(act)
    a_int8f = a_q.float() * a_scale[:, None]

    results: dict[str, torch.Tensor] = {}
    for w_name, w13, w2 in (
        ("W4e", w13_exact, w2_exact),
        ("W4r", w13_int8, w2_int8),
    ):
        for a_name, a_val, a_is_int8 in (
            ("A16", act.float(), False),
            ("A8", a_int8f, True),
        ):
            g = torch.zeros(tokens * topk, w2.shape[1], device=device)
            for e in range(n_experts):
                rows = (ids == e).nonzero(as_tuple=False)
                if rows.numel() == 0:
                    continue
                tok, slot = rows[:, 0], rows[:, 1]
                x = a_val[tok]
                gu = x @ w13[e].float().T
                if w_name == "W4e":
                    gu = gu.to(torch.bfloat16).float()
                inter = silu_and_mul(gu.to(torch.bfloat16))
                if a_is_int8:
                    iq, isc = quantize_per_token_int8(inter)
                    inter_in = iq.float() * isc[:, None]
                else:
                    inter_in = inter
                out = inter_in @ w2[e].float().T
                if w_name == "W4e":
                    out = out.to(torch.bfloat16).float()
                g[slot * tokens + tok] = out
            acc = torch.zeros(tokens, w2.shape[1], device=device)
            for k in range(topk):
                acc += weights[:, k : k + 1] * g[k * tokens : (k + 1) * tokens]
            results[f"{a_name}{w_name}"] = acc

    ref = results["A16W4e"]
    print(f"layer {args.layer}, experts {args.experts}, tokens {tokens}, act_std {args.act_std}")
    print(f"reference (A16W4 exact weights) |out| mean = {ref.abs().mean():.5f}")
    for key in ("A16W4r", "A8W4e", "A8W4r"):
        r = results[key]
        cos = float(
            (r.flatten() @ ref.flatten())
            / (r.flatten().norm() * ref.flatten().norm()).clamp_min(1e-30)
        )
        print(
            f"  {key}: rel_l2 = {rel_l2(r, ref):.5f}  max_abs = "
            f"{float((r - ref).abs().max()):.5f}  cosine = {cos:.6f}"
        )
    print("  (A16W4r = weight remap only, A8W4e = int8 activations only, "
          "A8W4r = the served route)")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("weights")
    w.add_argument("--layers", type=int, nargs="+", default=[0, 6, 20, 39])
    w.add_argument("--experts", type=int, nargs="+", default=[0])
    w.add_argument("--device", default="cuda:0")
    w.add_argument("--output", type=Path)

    m = sub.add_parser("moe")
    m.add_argument("--layer", type=int, default=6)
    m.add_argument("--experts", type=int, nargs="+", default=list(range(16)))
    m.add_argument("--tokens", type=int, default=256)
    m.add_argument("--topk", type=int, default=6)
    m.add_argument("--act-std", type=float, default=1.0)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--device", default="cuda:0")

    args = parser.parse_args()
    if args.cmd == "weights":
        weights_mode(args)
    else:
        moe_mode(args)


if __name__ == "__main__":
    main()
