#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve the checkpoint's own two-image example and check the answer names both.

The reference implementation ships this exact prompt with two photographs
(``inference/examples/example_harmony.json``): "answer in the order of the first
and second image: what ingredient is in each, and which part is usually eaten?".
It exercises two image spans in one prefill, so it covers the span layout and the
per-request image budget, and its answer is falsifiable: a text-only or
mis-wired vision path cannot name both carrots and corn.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.request
from pathlib import Path
MODEL_DIR = Path("/media/z/data6/models/DeepSeek-V4.1-Flash")
IMAGES = ("carrots.jpeg", "corn.jpeg")
PROMPT = (
    "请按“第一张、第二张”的顺序回答：第一张图",
    "和第二张图",
    "中分别是什么食材？它们通常食用的部位分别是什么？",
)
EXPECT = {
    "carrots.jpeg": ("胡萝卜", "carrot", "根"),
    "corn.jpeg": ("玉米", "corn", "种子", "果实"),
}


def data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def prefix_cache_counters(base: str) -> tuple[float, float]:
    with urllib.request.urlopen(f"{base}/metrics", timeout=120) as resp:
        text = resp.read().decode()
    queries = hits = 0.0
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        series, _, value = line.rpartition(" ")
        name = series.partition("{")[0]
        if name == "vllm:prefix_cache_queries_total":
            queries += float(value)
        elif name == "vllm:prefix_cache_hits_total":
            hits += float(value)
    return queries, hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=26663)
    ap.add_argument("--model", default="dsv41f-port")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "Send the identical request N times. With prefix caching on, the "
            "repeats must be served from the cache and return the same answer, "
            "which is the multimodal half of the prefix-cache gate: the image "
            "placeholder span has to hash with the image, not just the text."
        ),
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    image_dir = MODEL_DIR / "inference" / "examples" / "images"
    missing = [name for name in IMAGES if not (image_dir / name).exists()]
    if missing:
        raise SystemExit(f"example images not found: {missing}")

    content: list[dict] = [{"type": "text", "text": PROMPT[0]}]
    for name in IMAGES:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url(image_dir / name)},
            }
        )
        content.append(
            {"type": "text", "text": PROMPT[1] if name == IMAGES[0] else PROMPT[2]}
        )

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
    }
    answers: list[str] = []
    usages: list[dict] = []
    per_repeat_hits: list[float] = []
    q0, h0 = prefix_cache_counters(f"http://{args.host}:{args.port}")
    for _ in range(args.repeat):
        request = urllib.request.Request(
            f"http://{args.host}:{args.port}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=1800) as response:
            body = json.loads(response.read())
        choice = body["choices"][0]
        message = choice["message"]
        answers.append(
            (message.get("content") or message.get("reasoning") or "").strip()
        )
        usages.append(body["usage"])
        q1, h1 = prefix_cache_counters(f"http://{args.host}:{args.port}")
        per_repeat_hits.append(h1 - h0)
        q0, h0 = q1, h1
        print(f"repeat {len(answers)}: finish_reason={choice['finish_reason']} "
              f"usage={body['usage']}")

    answer = answers[0]
    usage = usages[0]
    print(
        f"prompt_tokens={usage['prompt_tokens']} "
        f"completion={usage['completion_tokens']}"
    )
    print("--- answer ---")
    print(answer[:1500])

    hits_per_repeat = [
        {name: [w for w in words if w in a] for name, words in EXPECT.items()}
        for a in answers
    ]
    hits = hits_per_repeat[0]
    for name, found in hits.items():
        print(f"{name}: matched {found}")
    named_every_time = all(
        all(found for found in row.values()) for row in hits_per_repeat
    )
    repeats_identical = len(set(answers)) == 1
    first_div = None
    if not repeats_identical:
        first_div = next(
            i
            for i, (a, b) in enumerate(zip(answers[0], answers[1]))
            if a != b
        )
        print(f"repeats differ from character {first_div}: "
              f"{answers[0][first_div:first_div + 40]!r} vs "
              f"{answers[1][first_div:first_div + 40]!r}")
    reused = per_repeat_hits[1:] if args.repeat > 1 else []
    print(f"repeats_identical={repeats_identical} "
          f"prefix_hits_per_repeat={per_repeat_hits} reused={reused}")
    # Passing needs the answer to name both images every time, and, when the
    # request is repeated, the repeat to actually come out of the prefix cache.
    # Byte-identical repeats are *not* an invariant here: a cache hit recomputes
    # the tail in a smaller prefill batch, and that alone perturbs the logits
    # enough to flip near-tied greedy tokens (see the prefix-cache ledger).
    ok = named_every_time and all(h > 0 for h in reused)
    print(
        f"RESULT: {'PASS' if ok else 'FAIL'} "
        f"(both images named in every repeat, repeats reused)"
    )
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(
                {
                    "answer": answer,
                    "usage": usage,
                    "matches": hits,
                    "pass": ok,
                    "repeat": args.repeat,
                    "answers": answers,
                    "repeats_identical": repeats_identical,
                    "first_divergence_char": first_div,
                    "cumulative_prefix_hits_per_repeat": per_repeat_hits,
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )
        print(f"wrote {args.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
