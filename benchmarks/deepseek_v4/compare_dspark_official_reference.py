# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare a captured vLLM DSpark proposal with the official DSpark model.

The official checkpoint stores its three DSpark blocks under the historical
``mtp.*`` parameter namespace. This tool compares DSpark, not MTP decoding.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_official_module(model_root: Path) -> ModuleType:
    inference_dir = model_root / "inference"
    sys.path.insert(0, str(inference_dir))
    spec = importlib.util.spec_from_file_location(
        "deepseek_v4_official_model", inference_dir / "model.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the official inference model")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint_name(name: str) -> str | None:
    if name.startswith("model."):
        name = name.removeprefix("model.")
    if name.startswith("mtp.") and (
        "emb" in name or name.endswith("head.weight")
    ):
        return None
    return (
        name.replace("self_attn", "attn")
        .replace("mlp", "ffn")
        .replace("weight_scale_inv", "scale")
        .replace("e_score_correction_bias", "bias")
    )


def _dequantize_wo_a(
    weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    weight = weight.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128))
    weight = weight.float() * scale[:, None, :, None].float()
    return weight.flatten(2, 3).flatten(0, 1).bfloat16()


def _load_reference_weights(model: torch.nn.Module, model_root: Path) -> int:
    params = dict(model.named_parameters())
    loaded: set[str] = set()
    files = (
        "model-00001-of-00048.safetensors",
        "model-00045-of-00048.safetensors",
        "model-00046-of-00048.safetensors",
        "model-00047-of-00048.safetensors",
        "model-00048-of-00048.safetensors",
    )
    with torch.no_grad():
        for filename in files:
            with safe_open(
                model_root / filename, framework="pt", device="cpu"
            ) as handle:
                keys = set(handle.keys())
                for source_name in sorted(keys):
                    name = _checkpoint_name(source_name)
                    if name is None or name not in params or name in loaded:
                        continue
                    if name.endswith("attn.wo_a.scale"):
                        continue
                    tensor = handle.get_tensor(source_name)
                    if name.endswith("attn.wo_a.weight"):
                        scale_source = source_name.removesuffix("weight") + "scale"
                        if scale_source not in keys:
                            scale_source = source_name.replace(
                                ".weight", ".weight_scale_inv"
                            )
                        tensor = _dequantize_wo_a(
                            tensor, handle.get_tensor(scale_source)
                        )
                    elif "experts" in name and tensor.dtype == torch.int8:
                        tensor = tensor.view(torch.float4_e2m1fn_x2)
                    params[name].copy_(tensor)
                    loaded.add(name)
            print(
                json.dumps(
                    {"checkpoint": filename, "loaded_params": len(loaded)}
                ),
                flush=True,
            )

    expected = {
        name
        for name in params
        if name.startswith("mtp.") or name in {"embed.weight", "head.weight"}
    }
    missing = sorted(expected - loaded)
    if missing:
        raise RuntimeError(f"Missing official draft parameters: {missing[:20]}")
    return len(loaded)


def _quantize_dequantize_activation(
    tensor: torch.Tensor, block_size: int
) -> torch.Tensor:
    shape = tensor.shape
    blocks = tensor.float().reshape(-1, shape[-1] // block_size, block_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    quantized = (blocks / scale).clamp(-448.0, 448.0)
    quantized = quantized.to(torch.float8_e4m3fn).float() * scale
    return quantized.reshape(shape)


def _dequantize_fp8_weight(weight: torch.Tensor) -> torch.Tensor:
    rows, columns = weight.shape
    scales = weight.scale.float()
    scales = scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return weight.float() * scales[:rows, :columns]


def _dequantize_fp4_weight(weight: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=weight.device,
    )
    packed = weight.view(torch.uint8)
    values = torch.stack(
        (table[(packed & 0x0F).long()], table[((packed >> 4) & 0x0F).long()]),
        dim=-1,
    ).flatten(-2)
    scales = weight.scale.float().repeat_interleave(32, dim=-1)
    return values * scales


def _install_sm80_portable_ops(module: ModuleType) -> None:
    def portable_linear(
        tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is not None:
            raise ValueError("The official DSpark reference expects bias-free linears")
        if weight.dtype == torch.float8_e4m3fn:
            activation = _quantize_dequantize_activation(tensor, 128)
            output = torch.nn.functional.linear(
                activation, _dequantize_fp8_weight(weight)
            )
            return output.to(torch.get_default_dtype())
        if weight.dtype == torch.float4_e2m1fn_x2:
            activation = _quantize_dequantize_activation(tensor, 128)
            output = torch.nn.functional.linear(
                activation, _dequantize_fp4_weight(weight)
            )
            return output.to(torch.get_default_dtype())
        return torch.nn.functional.linear(tensor, weight)

    def portable_act_quant(
        tensor: torch.Tensor,
        block_size: int = 128,
        scale_fmt: str | None = None,
        scale_dtype: torch.dtype = torch.float32,
        inplace: bool = False,
    ) -> torch.Tensor:
        del scale_fmt, scale_dtype
        output = _quantize_dequantize_activation(tensor, block_size).to(tensor.dtype)
        if not inplace:
            raise ValueError("Portable DSpark act_quant is only used in-place")
        tensor.copy_(output)
        return tensor

    def portable_sparse_attn(
        query: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_indices: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor:
        output = torch.empty_like(query)
        for batch in range(query.shape[0]):
            for row in range(query.shape[1]):
                indices = topk_indices[batch, row]
                indices = indices[indices >= 0].to(torch.int64)
                keys = kv[batch].index_select(0, indices).float()
                scores = torch.einsum(
                    "hd,td->ht", query[batch, row].float(), keys
                )
                scores *= softmax_scale
                maximum = torch.maximum(
                    scores.amax(dim=-1), attn_sink.float()
                )
                weights = torch.exp(scores - maximum[:, None])
                denominator = weights.sum(dim=-1) + torch.exp(
                    attn_sink.float() - maximum
                )
                value = torch.einsum("ht,td->hd", weights, keys)
                output[batch, row] = (value / denominator[:, None]).to(query.dtype)
        return output

    def portable_hc_split_sinkhorn(
        mixes: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre_logits, post_logits, comb_logits = torch.split(
            mixes, (hc_mult, hc_mult, hc_mult * hc_mult), dim=-1
        )
        pre_base, post_base, comb_base = torch.split(
            hc_base, (hc_mult, hc_mult, hc_mult * hc_mult), dim=-1
        )
        pre = torch.sigmoid(pre_logits * hc_scale[0] + pre_base) + eps
        post = 2 * torch.sigmoid(post_logits * hc_scale[1] + post_base)
        comb = comb_logits * hc_scale[2] + comb_base
        comb = comb.unflatten(-1, (hc_mult, hc_mult))
        comb = comb.softmax(dim=-1) + eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        return pre, post, comb

    module.linear = portable_linear
    module.act_quant = portable_act_quant
    module.sparse_attn = portable_sparse_attn
    module.hc_split_sinkhorn = portable_hc_split_sinkhorn


def _tensor_error(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    reference = reference.float()
    actual = actual.float()
    delta = actual - reference
    cosine = torch.nn.functional.cosine_similarity(
        reference.flatten(), actual.flatten(), dim=0
    )
    return {
        "cosine": float(cosine),
        "mae": float(delta.abs().mean()),
        "max_abs": float(delta.abs().max()),
        "rmse": float(delta.square().mean().sqrt()),
        "reference_rms": float(reference.square().mean().sqrt()),
    }


def _logit_metrics(
    reference: torch.Tensor, actual: torch.Tensor
) -> list[dict[str, float | int | bool]]:
    reference = reference.float().reshape(-1, reference.shape[-1])
    actual = actual.float().reshape(-1, actual.shape[-1])
    rows = []
    for ref_row, actual_row in zip(reference, actual, strict=True):
        ref_centered = ref_row - ref_row.mean()
        actual_centered = actual_row - actual_row.mean()
        ref_prob = ref_row.softmax(dim=-1)
        actual_prob = actual_row.softmax(dim=-1)
        ref_top = torch.topk(ref_row, 32).indices
        actual_top = torch.topk(actual_row, 32).indices
        rows.append(
            {
                "centered_rmse": float(
                    (actual_centered - ref_centered).square().mean().sqrt()
                ),
                "distribution_overlap": float(
                    torch.minimum(ref_prob, actual_prob).sum()
                ),
                "reference_top1": int(ref_row.argmax()),
                "actual_top1": int(actual_row.argmax()),
                "top1_match": bool(ref_row.argmax() == actual_row.argmax()),
                "top32_intersection": int(torch.isin(ref_top, actual_top).sum()),
            }
        )
    return rows


@torch.inference_mode()
def _run_reference(
    module: ModuleType,
    model: torch.nn.Module,
    capture: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    aux = torch.cat(
        [capture[f"target_aux_hidden_{i}"] for i in range(3)], dim=-1
    )
    aux = aux.unsqueeze(0).to(device)
    context_positions = capture["context_positions"].to(device)
    query_input_ids = capture["query_input_ids"].to(device)
    query_positions = capture["query_positions"].to(device)
    if not torch.equal(
        context_positions,
        torch.arange(context_positions.numel(), device=device),
    ):
        raise ValueError(
            "The official reference requires contiguous positions from zero"
        )
    expected_query_positions = torch.arange(
        context_positions.numel(),
        context_positions.numel() + query_input_ids.numel(),
        device=device,
    )
    if not torch.equal(query_positions, expected_query_positions):
        raise ValueError("Captured DSpark query positions are not contiguous")

    first_layer = model.mtp[0]
    main_x = first_layer.main_norm(first_layer.main_proj(aux))
    placeholder = torch.empty(
        (1, 1, model.hc_mult, first_layer.dim),
        dtype=torch.bfloat16,
        device=device,
    )
    for layer in model.mtp:
        layer.attn(placeholder, 0, main_x[:, :-1])

    anchor = query_input_ids[:1].to(torch.int64)
    hidden, decode_main_x = first_layer.forward_embed(aux[:, -1:], anchor)
    start_pos = context_positions.numel() - 1
    for layer in model.mtp:
        hidden = layer(hidden, start_pos, anchor, decode_main_x)
    last_layer = model.mtp[-1]
    head_hidden = last_layer.hc_head(
        hidden,
        last_layer.hc_head_fn,
        last_layer.hc_head_scale,
        last_layer.hc_head_base,
    )
    base_logits = model.head(last_layer.norm(head_hidden), full_logits=True)

    markov_biases = []
    final_logits = []
    for i, previous in enumerate(capture["draft_previous_ids"]):
        bias, _ = last_layer.markov_head(previous.to(device))
        markov_biases.append(bias)
        final_logits.append(base_logits[:, i] + bias)

    captured_biases = torch.stack(capture["draft_markov_bias"], dim=1).to(device)
    captured_final = torch.stack(capture["draft_final_logits"], dim=1).to(device)
    reference_biases = torch.stack(markov_biases, dim=1)
    reference_final = torch.stack(final_logits, dim=1)
    captured_main_x = capture["context_main_x"].unsqueeze(0).to(device)
    captured_hidden = capture["draft_head_hidden"].to(device)
    captured_base = capture["draft_base_logits"].to(device)
    return {
        "main_x": _tensor_error(main_x, captured_main_x),
        "head_hidden": _tensor_error(head_hidden, captured_hidden),
        "base_logits": _logit_metrics(base_logits, captured_base),
        "markov_bias": _logit_metrics(reference_biases, captured_biases),
        "final_logits": _logit_metrics(reference_final, captured_final),
    }


def main() -> None:
    args = parse_args()
    module = _load_official_module(args.model)
    config = json.loads((args.model / "inference" / "config.json").read_text())
    model_args = module.ModelArgs(**config)
    model_args.n_layers = 0
    model_args.n_hash_layers = 0
    model_args.max_batch_size = 1
    model_args.max_seq_len = 2048
    model_args.compress_ratios = (0, 0, 0)
    torch.set_default_dtype(torch.bfloat16)
    device = torch.device(args.device)
    with torch.device(device):
        model = module.Transformer(model_args)
    loaded_params = _load_reference_weights(model, args.model)
    portable_ops = torch.cuda.get_device_capability(device) < (8, 9)
    if portable_ops:
        _install_sm80_portable_ops(module)
    torch.set_default_device(device)
    capture = torch.load(args.capture, map_location="cpu", weights_only=False)
    result = {
        "schema_version": 1,
        "comparison_target": "dspark",
        "checkpoint_parameter_namespace": "mtp",
        "device": torch.cuda.get_device_name(device),
        "loaded_params": loaded_params,
        "portable_reference_ops": portable_ops,
        "capture": str(args.capture),
        "model": str(args.model),
        "metrics": _run_reference(module, model, capture, device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
