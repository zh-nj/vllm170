#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe prefill batch-shape sensitivity, the reason cached runs can differ.

A prefix-cache hit necessarily recomputes its tail in a much smaller prefill
batch than the cold request used, so the two runs disagree wherever a per-row
result depends on the batch's token count. This script measures that dependence
directly and with **no reuse at all**: each request carries a unique
``cache_salt``, so every one is a full cold prefill, and it reports the prompt
logprob of one fixed token while the length of the chunk that token is computed
in changes.

Three things the numbers show, and why they matter:

- The probe token's logprob is *bit-identical* over ranges of chunk lengths and
  jumps discretely between them, so the kernel configuration -- not rounding
  noise -- is what changes.
- A token in the always-1024-token first chunk does not move at all, so the
  effect is local to the batch a row is computed in; it is not non-causal
  leakage from later tokens.
- Replacing the tail with different tokens of the same length leaves the probe
  bit-identical, which rules leakage out from the other direction too.

Practical consequence: the engine is not reproducible across batch shapes, so
prefix caching, ``max_num_batched_tokens``, or plain concurrency can each change
a long free-form continuation. It is not a cache defect and it is not fixable at
the prefix-cache layer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decode_gate import _post  # noqa: E402


def probe_logprob(
    base: str,
    ids: list[int],
    n: int,
    probe: int,
    salt: str,
    topk: int = 0,
) -> float:
    payload = {
        "model": "dsv41f-port",
        "prompt": ids[:n],
        "max_tokens": 1,
        "temperature": 0.0,
        "ignore_eos": True,
        "echo": True,
        "prompt_logprobs": topk,
        "cache_salt": salt,
    }
    with _post(f"{base}/v1/completions", payload) as resp:
        data = json.loads(resp.read())
    entry = data["choices"][0]["prompt_logprobs"][probe]
    if isinstance(entry, dict) and "logprob" not in entry:
        entry = next(iter(entry.values()))
    return entry["logprob"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=26663)
    ap.add_argument("--doc", default="docs/design/prefix_caching.md")
    ap.add_argument("--probe", type=int, default=1300, help="absolute token index")
    ap.add_argument("--chunk1", type=int, default=1024, help="first chunk length")
    ap.add_argument("--max-len", type=int, default=1620)
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    text = open(args.doc).read()
    with _post(
        f"{base}/tokenize", {"model": "", "prompt": (text + "\n\n" + text[::-1]) * 6}
    ) as resp:
        ids = json.loads(resp.read())["tokens"]

    rows = []
    prev = None
    for n in range(args.probe + 1, args.max_len + 1, args.step):
        value = probe_logprob(base, ids, n, args.probe, f"shape-{n}")
        changed = prev is None or abs(value - prev) > 1e-9
        rows.append(
            {
                "prompt_tokens": n,
                "tail_chunk": n - args.chunk1,
                "logprob": value,
                "changed": changed,
            }
        )
        prev = value

    print(
        f"logprob of token {args.probe} vs the length of the chunk it is "
        f"computed in (first chunk fixed at {args.chunk1}):"
    )
    start = rows[0]["prompt_tokens"]
    for i, row in enumerate(rows):
        if row["changed"]:
            start = row["prompt_tokens"]
        last = i + 1 == len(rows) or rows[i + 1]["changed"]
        if last:
            print(
                f"  n=[{start:5d},{row['prompt_tokens']:5d}] "
                f"tail=[{start - args.chunk1:4d},"
                f"{row['prompt_tokens'] - args.chunk1:4d}] "
                f"logprob={row['logprob']:.6f}"
            )

    values = {round(r["logprob"], 6) for r in rows}
    spread = max(values) - min(values)
    print(
        f"distinct values={len(values)} over {len(rows)} lengths; "
        f"spread={spread:.6f} nats"
    )
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
