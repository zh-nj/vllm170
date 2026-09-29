# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check real model generation through completions and chat without extra GPUs."""

import argparse
import json
import time
import urllib.request


def check(base, model):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    results = []
    for endpoint, payload in [
        ("completions", {"prompt": "The capital of France is", "max_tokens": 16}),
        (
            "chat/completions",
            {
                "messages": [{"role": "user", "content": "只回答：接管验证成功"}],
                "chat_template_kwargs": {"thinking": False},
                "max_tokens": 128,
            },
        ),
    ]:
        start = time.monotonic()
        req = urllib.request.Request(
            base.rstrip("/") + "/v1/" + endpoint,
            data=json.dumps(dict(payload, model=model, temperature=0)).encode(),
            headers={"Content-Type": "application/json"},
        )
        with opener.open(req, timeout=180) as response:
            body = json.load(response)
        assert body.get("usage", {}).get("completion_tokens", 0) > 0, body
        assert body["choices"], body
        choice = body["choices"][0]
        if endpoint == "completions":
            assert choice.get("text"), body
        else:
            assert choice["message"].get("content"), body
        results.append(
            {"endpoint": endpoint, "seconds": time.monotonic() - start, "body": body}
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:26661")
    parser.add_argument("--model", default="dsv41f")
    args = parser.parse_args()
    print(json.dumps(check(args.base, args.model), ensure_ascii=False, indent=2))
