#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vision smoke test: one question, several deterministic images.

Renders two images whose only difference is an orientation the answer must
follow (a horizontal versus a vertical white bar on a saturated background), asks
the same question about each, and prints the answers. A text-only instance cannot
answer differently for the two images, so a matching pair is evidence that the
vision tower actually reached the language model.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import urllib.request

from PIL import Image, ImageDraw


def render(orientation: str, size: int = 640) -> str:
    """A saturated background with one thick white bar; returns a data URL."""
    colour = (200, 30, 30) if orientation == "horizontal" else (30, 30, 200)
    image = Image.new("RGB", (size, size), colour)
    draw = ImageDraw.Draw(image)
    if orientation == "horizontal":
        draw.rectangle([0, size // 2 - 60, size, size // 2 + 60], fill=(255, 255, 255))
    else:
        draw.rectangle([size // 2 - 60, 0, size // 2 + 60, size], fill=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


def ask(base: str, model: str, image_url: str, question: str, max_tokens: int) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": question},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        body = json.loads(response.read())
    message = body["choices"][0]["message"]
    # The reasoning parser keeps the chain of thought in ``reasoning`` and only
    # fills ``content`` once the model finishes thinking.
    return (message.get("content") or message.get("reasoning") or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=26663)
    ap.add_argument("--model", default="dsv41f-port")
    ap.add_argument(
        "--question",
        default=(
            "Look at the white bar in the image. Is it horizontal or vertical? "
            "Answer with exactly one word."
        ),
    )
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    rows = []
    for orientation in ("horizontal", "vertical"):
        answer = ask(
            base, args.model, render(orientation), args.question, args.max_tokens
        )
        rows.append({"orientation": orientation, "answer": answer})
        print(f"[{orientation:>10}] -> {answer[:200]!r}")
    agree = sum(
        1
        for row in rows
        if row["orientation"][:3].lower() in row["answer"].lower()
        or row["orientation"][0].upper() == row["answer"].strip()[:1].upper()
    )
    print(f"answers matching the image: {agree}/{len(rows)}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
