#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dump greedy completions for a fixed prompt set, for exact A/B comparison.

Speculative decoding is exact under greedy verification: the accepted tokens are
the target's own greedy tokens. So the token ids produced with drafting on must
equal the token ids produced with drafting off, and any divergence is a capture
or staging bug rather than a sampling difference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decode_gate import _post  # noqa: E402

PROMPTS = (
    "Explain how a CPU cache coherence protocol keeps two cores consistent, "
    "in one paragraph.",
    "Write a Python function that merges two sorted lists without itertools, "
    "then explain its complexity.",
    "A train leaves at 14:05 and travels 210 km at 84 km/h. What time does it "
    "arrive? Show the arithmetic.",
    "Summarize the difference between optimistic and pessimistic concurrency "
    "control for a database transaction.",
    "List five failure modes of a distributed consensus protocol and one "
    "mitigation for each.",
)


def run(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
        "return_token_ids": True,
    }
    with _post(f"{base}/v1/completions", payload) as resp:
        data = json.loads(resp.read())
    choice = data["choices"][0]
    token_ids = choice.get("token_ids")
    text = choice.get("text", "")
    return {
        "prompt": prompt,
        "text": text,
        "token_ids": token_ids,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "completion_tokens": data["usage"]["completion_tokens"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=26663)
    ap.add_argument("--model", default="dsv41f-port")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    rows = []
    for i, prompt in enumerate(PROMPTS):
        row = run(base, args.model, prompt, args.max_tokens)
        rows.append(row)
        ids = row["token_ids"]
        print(
            f"[{i}] tokens={row['completion_tokens']} "
            f"ids={'yes' if ids else 'no'} sha={row['text_sha256'][:12]} "
            f"{row['text'][:48]!r}"
        )
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
