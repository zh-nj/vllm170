# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Capture/compare greedy continuations on *real* prompts across MoE backends.

``compare_moe_numerics.py`` uses random-word prompts, which is fine for numeric
regression (prompt logprobs) but useless for judging output quality: the
continuation is word salad, so any perturbation flips it. This script uses
short prompts whose continuation is semantically determined (arithmetic, a
fact, a code completion, a translation) so that "do the two backends say the
same thing?" is a meaningful question.

Usage::

    python compare_real_prompts.py capture --tag int8 --out /tmp/real_int8.json
    python compare_real_prompts.py compare /tmp/real_int8.json /tmp/real_bf16.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import MODEL, URL  # noqa: E402

PROMPTS = [
    "Question: What is 17 * 23?\nAnswer: 17 * 23 =",
    "Question: Which planet is known as the Red Planet?\nAnswer:",
    "Complete the Python function.\n\ndef add(a, b):\n    return",
    "Translate to French: The cat sits on the mat.\nFrench:",
    "List the first five prime numbers in ascending order: 2,",
    "Question: What is the chemical symbol for gold?\nAnswer: The chemical symbol for gold is",
    "Fill in the missing word: The sun rises in the ___ and sets in the west.",
    "Write the next line of this haiku about winter:\nSnow falls on the pines,",
    "Question: What year did the Second World War end?\nAnswer:",
    "Summarise in one sentence: Water boils at 100 degrees Celsius at sea level.",
]


def post(payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        URL, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def capture(tag: str, out: Path, max_tokens: int) -> None:
    records = []
    for index, prompt in enumerate(PROMPTS):
        t0 = time.perf_counter()
        resp = post(
            {
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
                "logprobs": 0,
            }
        )
        dt = time.perf_counter() - t0
        choice = resp["choices"][0]
        records.append(
            {
                "index": index,
                "prompt": prompt,
                "text": choice["text"],
                "tokens": choice["logprobs"]["tokens"],
                "token_logprobs": choice["logprobs"]["token_logprobs"],
                "elapsed_s": dt,
            }
        )
        print(f"[{tag}] {index:>2} {dt:5.2f}s {choice['text'][:70]!r}")
    out.write_text(json.dumps({"tag": tag, "records": records}, indent=2))
    print(f"[{tag}] wrote {out}")


def compare(a_path: Path, b_path: Path) -> None:
    a = json.loads(a_path.read_text())
    b = json.loads(b_path.read_text())
    print(f"A = {a['tag']}   B = {b['tag']}\n")
    same_text = same_first = 0
    for x, y in zip(a["records"], b["records"]):
        ta, tb = x["tokens"], y["tokens"]
        n = min(len(ta), len(tb))
        agree = sum(1 for i in range(n) if ta[i] == tb[i])
        first_same = bool(ta) and bool(tb) and ta[0] == tb[0]
        same_text += x["text"] == y["text"]
        same_first += first_same
        flag = "" if x["text"] == y["text"] else "  <-- differs"
        print(f"{x['index']:>2} same_tokens={agree}/{n} first_token_same={first_same}{flag}")
        if flag:
            print(f"     A: {x['text'][:90]!r}")
            print(f"     B: {y['text'][:90]!r}")
    n = len(a["records"])
    print(f"\nidentical continuations: {same_text}/{n}")
    print(f"identical first token  : {same_first}/{n}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--tag", required=True)
    cap.add_argument("--out", type=Path, required=True)
    cap.add_argument("--max-tokens", type=int, default=32)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("left", type=Path)
    cmp_.add_argument("right", type=Path)
    args = parser.parse_args()

    if args.cmd == "capture":
        capture(args.tag, args.out, args.max_tokens)
    else:
        compare(args.left, args.right)


if __name__ == "__main__":
    main()
