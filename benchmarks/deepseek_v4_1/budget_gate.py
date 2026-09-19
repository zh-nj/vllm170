#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Budget gate: does ``--max-num-batched-tokens`` change what the model says?

The scheduler admits a prefill chunk of ``max_num_batched_tokens -
max_num_new_slots_for_drafting`` tokens (``scheduler.py``, the ``input_budget -
draft_slots`` cap), so with DSpark at 5 speculative tokens the first chunk is
``budget - 4``: 1020 at the default 1024, 1148 at 1152. Raising the budget to
let a single image span more than 1022 tokens therefore moves the first chunk
across several row-count-selected launch geometries, and the port launcher has
recorded since before the invariance work that 1152 changed the greedy token
ids on 2 of 5 reference prompts and dropped DSpark acceptance.

This gate measures that claim per *position* instead of per answer, which is
what separates the two candidate mechanisms:

* a chunk-1 geometry selected from the row count M (M = 1020 versus 1148) can
  only move positions that chunk 1 computes, i.e. positions below the **smaller**
  boundary in both runs, so a divergence starting at position 0 or at any
  position below ``min(chunk1)`` is row-count geometry;
* a different split of the same context between the cached part and the chunk
  (the boundary itself moved) can only move positions at or after the moved
  boundary.

Each case sends one prompt twice: a greedy continuation, and a
``prompt_logprobs`` request whose per-position top-5 logits are a fingerprint of
the hidden state at every prompt position. ``prompt_logprobs`` is also what
makes this a cold measurement by construction -- the request sets
``skip_reading_prefix_cache``, so no prefix-cache lookup runs at all -- and a
per-request ``cache_salt`` keeps the two requests from warming each other.

Logprob floats are stored as ``float.hex()`` so the comparison is bit-exact
rather than JSON-round-trip exact.

Usage::

    python budget_gate.py --out /tmp/budget_1024.json --label budget=1024
    python budget_gate.py --compare /tmp/budget_1024.json /tmp/budget_1152.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decode_gate import _post, make_token_ids  # noqa: E402

TOP_K = 5


def greedy_ids(
    base: str,
    model: str,
    prompt_ids: list[int],
    max_tokens: int,
    cache_salt: str,
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "cache_salt": cache_salt,
    }
    ids: list[int] = []
    usage = None
    with _post(f"{base}/v1/completions", payload) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            obj = json.loads(body)
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                if choice.get("token_ids"):
                    ids.extend(choice["token_ids"])
    return {"ids": ids, "usage": usage}


def prompt_topk(
    base: str,
    model: str,
    prompt_ids: list[int],
    cache_salt: str,
) -> list[list[list] | None]:
    """Per-position top-k (token id, logprob) fingerprint of the prompt."""
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
        "prompt_logprobs": TOP_K,
        "cache_salt": cache_salt,
    }
    with _post(f"{base}/v1/completions", payload) as resp:
        obj = json.loads(resp.read())
    entries = obj["choices"][0]["prompt_logprobs"]
    out: list[list[list] | None] = []
    for entry in entries:
        if not entry:
            out.append(None)
            continue
        ranked = sorted(
            ((float(info["logprob"]), int(tid)) for tid, info in entry.items()),
            reverse=True,
        )
        out.append([[tid, lp.hex()] for lp, tid in ranked[:TOP_K]])
    return out


def run_case(
    base: str, model: str, prompt_ids: list[int], max_tokens: int, tag: str
) -> dict:
    digest = hashlib.sha256(",".join(map(str, prompt_ids)).encode()).hexdigest()[:16]
    return {
        "length": len(prompt_ids),
        "prompt_sha256_16": digest,
        "prompt_tail": prompt_ids[-8:],
        "greedy": greedy_ids(
            base, model, prompt_ids, max_tokens, f"{tag}-greedy-{digest}"
        ),
        "prompt_topk": prompt_topk(base, model, prompt_ids, f"{tag}-lp-{digest}"),
    }


def first_diff(a: list, b: list) -> int | None:
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def count_diff(a: list, b: list) -> int:
    return sum(1 for left, right in zip(a, b) if left != right)


def within(path: Path) -> int:
    """Same-config check: is a shared prefix scored the same at every length?

    Every case is a prefix of one generated prompt, so position ``p`` holds the
    same token in all of them. A row whose score depends on how many rows share
    its batch must therefore move here -- with no engine-side confound, since
    the server was not restarted between the cases. Cases longer than the first
    prefill chunk are also the control: chunk 2 recomputes no earlier position.
    """
    artifact = json.loads(path.read_text())
    cases = artifact["cases"]
    print(f"{path}  {artifact['label']}")
    print(f"args: {artifact.get('server_args', {})}\n")
    failed = False
    print(f"{'shorter':>8} {'longer':>7} {'diff@':>7} {'n_diff':>7}  verdict")
    for i, short in enumerate(cases):
        for long in cases[i + 1 :]:
            n = short["length"]
            pos = first_diff(short["prompt_topk"][:n], long["prompt_topk"][:n])
            count = count_diff(short["prompt_topk"][:n], long["prompt_topk"][:n])
            failed |= pos is not None
            print(
                f"{n:>8} {long['length']:>7} "
                f"{(pos if pos is not None else '-'):>7} {count:>7}  "
                f"{'invariant' if pos is None else 'MOVED'}"
            )
    return 1 if failed else 0


def compare(path_a: Path, path_b: Path) -> int:
    a = json.loads(path_a.read_text())
    b = json.loads(path_b.read_text())
    print(f"A: {path_a}  {a['label']}")
    print(f"B: {path_b}  {b['label']}")
    print(f"args A: {a.get('server_args', {})}")
    print(f"args B: {b.get('server_args', {})}")
    cases_b = {case["length"]: case for case in b["cases"]}
    failed = False
    print(
        f"\n{'len':>6} {'prompt-diff@':>13} {'n_diff':>7} {'greedy-diff@':>13}"
        f" {'max|dlogp|':>11}  verdict"
    )
    for case in a["cases"]:
        other = cases_b.get(case["length"])
        if other is None:
            print(f"{case['length']:>6} (missing in B)")
            failed = True
            continue
        if case["prompt_sha256_16"] != other["prompt_sha256_16"]:
            print(f"{case['length']:>6} PROMPT MISMATCH -- comparison invalid")
            failed = True
            continue
        pos = first_diff(case["prompt_topk"], other["prompt_topk"])
        count = count_diff(case["prompt_topk"], other["prompt_topk"])
        greedy = first_diff(case["greedy"]["ids"], other["greedy"]["ids"])
        worst = 0.0
        for left, right in zip(case["prompt_topk"], other["prompt_topk"]):
            if left and right and left[0][1] != right[0][1]:
                worst = max(
                    worst, abs(float.fromhex(left[0][1]) - float.fromhex(right[0][1]))
                )
        same = pos is None and greedy is None
        failed |= not same
        print(
            f"{case['length']:>6} "
            f"{(pos if pos is not None else '-'):>13} {count:>7} "
            f"{(greedy if greedy is not None else '-'):>13} {worst:>11.3e}  "
            f"{'identical' if same else 'DIVERGED'}"
        )
    return 1 if failed else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=26663)
    ap.add_argument("--model", default="dsv41f-port")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[900, 1020, 1021, 1148, 1152, 1301, 1533],
        help="prompt token counts; all are prefixes of one generated prompt",
    )
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--label", default="")
    ap.add_argument(
        "--server-args",
        default="",
        help="free-form note about the served config, stored in the artifact",
    )
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument(
        "--within",
        default="",
        help="one artifact: check that a shared prefix scores the same at every"
        " case length (same-config batch invariance)",
    )
    args = ap.parse_args()

    if args.within:
        raise SystemExit(within(Path(args.within)))
    if args.compare:
        raise SystemExit(compare(Path(args.compare[0]), Path(args.compare[1])))

    base = f"http://{args.host}:{args.port}"
    longest = max(args.lengths)
    prompt = make_token_ids(base, longest, args.seed)
    cases = []
    for length in sorted(args.lengths):
        case = run_case(
            base, args.model, prompt[:length], args.max_tokens, f"s{args.seed}"
        )
        assert case["length"] == length
        cases.append(case)
        print(
            f"len={length:>6} prompt_topk={len(case['prompt_topk'])} "
            f"greedy={len(case['greedy']['ids'])} "
            f"cached={case['greedy']['usage'].get('prompt_tokens_details')}"
        )
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "label": args.label,
                    "server_args": args.server_args,
                    "seed": args.seed,
                    "max_tokens": args.max_tokens,
                    "top_k": TOP_K,
                    "cases": cases,
                },
                indent=1,
            )
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
