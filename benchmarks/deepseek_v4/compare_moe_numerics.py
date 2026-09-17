# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Capture/compare model numerics across MoE backends.

The MXFP4 experts can be served two ways: the A8W4-int8 kernel
(``mxfp4_int8_sm80``, mxfp4 decoded to int8 operands on the fly) or the A16W4
bf16 Marlin path (mxfp4 dequantized to bf16). This script records what the
model actually computes under each backend so the accuracy cost of the int8
route is measured instead of argued about.

Two instruments per prompt, both deterministic under greedy decoding:

* the greedy continuation (token-by-token agreement across backends), and
* ``prompt_logprobs`` for every prompt token, which is thousands of
  distribution samples per prompt and therefore far more sensitive than a
  handful of sampled tokens.

Usage::

    python compare_moe_numerics.py capture --tag int8 --out /tmp/moe_int8.json
    # restart the server with the other backend, capture again, then
    python compare_moe_numerics.py compare /tmp/moe_int8.json /tmp/moe_bf16.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_prefill as bp  # noqa: E402
from common import MODEL, URL  # noqa: E402

MAX_TOKENS = 48
PROMPT_TOKENS = (120, 400, 900, 1800)
PROMPTS_PER_LENGTH = 2
CONTINUE = "Continue this technical note with several more sentences:"


def post(payload: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(
        URL, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def capture(tag: str, out: Path) -> None:
    records = []
    for target in PROMPT_TOKENS:
        for rep in range(PROMPTS_PER_LENGTH):
            seed = 20260918 + target * 10 + rep
            prompt = bp.make_prompt(target, seed).replace(
                "Summarize the notes above.", CONTINUE
            )
            t0 = time.perf_counter()
            resp = post(
                {
                    "model": MODEL,
                    "prompt": prompt,
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0,
                    "logprobs": 0,
                    "prompt_logprobs": 0,
                }
            )
            dt = time.perf_counter() - t0
            choice = resp["choices"][0]
            generated = choice["logprobs"]["tokens"]
            gen_lp = choice["logprobs"]["token_logprobs"]
            # prompt_logprobs[0] is None (no distribution for the first token)
            prompt_lp: list[list[float]] = []
            for entry in choice["prompt_logprobs"] or []:
                if not entry:
                    continue
                (tok_id, info), = entry.items()
                prompt_lp.append([int(tok_id), float(info["logprob"])])
            records.append(
                {
                    "seed": seed,
                    "target_tokens": target,
                    "prompt_tokens": resp["usage"]["prompt_tokens"],
                    "elapsed_s": dt,
                    "generated": generated,
                    "generated_logprobs": gen_lp,
                    "prompt_logprobs": prompt_lp,
                    "text": choice["text"],
                }
            )
            print(
                f"[{tag}] target={target:>5} real={resp['usage']['prompt_tokens']:>5} "
                f"{dt:6.2f}s gen={len(generated)} prompt_lp={len(prompt_lp)}"
            )
    out.write_text(json.dumps({"tag": tag, "records": records}, indent=2))
    print(f"[{tag}] wrote {out}")


def compare(a_path: Path, b_path: Path) -> None:
    a = json.loads(a_path.read_text())
    b = json.loads(b_path.read_text())
    if a["tag"] == b["tag"]:
        print("NOTE: both captures carry the same tag; that measures self-agreement")
    print(f"A = {a['tag']} ({a_path})")
    print(f"B = {b['tag']} ({b_path})\n")
    print(
        f"{'prompt tok':>11} {'gen':>4} {'gen same':>9} {'plp n':>6} "
        f"{'mean|d|':>8} {'p99|d|':>8} {'max|d|':>8} {'top1 match':>11}"
    )
    totals = {"gen": 0, "gen_same": 0, "plp": 0, "same_tok": 0, "d": []}
    for x, y in zip(a["records"], b["records"]):
        assert x["seed"] == y["seed"], "captures are not the same prompt set"
        gx, gy = x["generated"], y["generated"]
        n = min(len(gx), len(gy))
        gen_same = sum(1 for i in range(n) if gx[i] == gy[i])
        dl = [
            abs(lx[1] - ly[1])
            for lx, ly in zip(x["prompt_logprobs"], y["prompt_logprobs"])
        ]
        same_tok = sum(
            1
            for lx, ly in zip(x["prompt_logprobs"], y["prompt_logprobs"])
            if lx[0] == ly[0]
        )
        srt = sorted(dl) if dl else [0.0]
        print(
            f"{x['prompt_tokens']:>11} {n:>4} {gen_same:>4}/{n:<4} {len(dl):>6} "
            f"{statistics.fmean(dl) if dl else 0:>8.4f} "
            f"{srt[min(len(srt) - 1, int(0.99 * len(srt)))]:>8.4f} "
            f"{srt[-1]:>8.4f} {same_tok:>6}/{len(dl):<4}"
        )
        totals["gen"] += n
        totals["gen_same"] += gen_same
        totals["plp"] += len(dl)
        totals["same_tok"] += same_tok
        totals["d"].extend(dl)

    d = totals["d"]
    print(
        f"\ngreedy token agreement : {totals['gen_same']}/{totals['gen']} = "
        f"{100.0 * totals['gen_same'] / max(1, totals['gen']):.2f}%"
    )
    print(
        f"prompt-token argmax agreement: {totals['same_tok']}/{totals['plp']} = "
        f"{100.0 * totals['same_tok'] / max(1, totals['plp']):.3f}%"
    )
    if d:
        print(
            f"prompt logprob delta: mean {statistics.fmean(d):.4f}  "
            f"median {statistics.median(d):.4f}  max {max(d):.4f}  (nats, natural log)"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--tag", required=True)
    cap.add_argument("--out", type=Path, required=True)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("left", type=Path)
    cmp_.add_argument("right", type=Path)
    args = parser.parse_args()

    if args.cmd == "capture":
        capture(args.tag, args.out)
    else:
        compare(args.left, args.right)


if __name__ == "__main__":
    main()
