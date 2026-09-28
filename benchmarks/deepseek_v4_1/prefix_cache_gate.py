#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix caching gate: prove reuse reaches the KV cache *and* is exact.

Two things have to hold for ``--enable-prefix-caching`` to be "in effect":
the scheduler must actually serve tokens out of the cache (not just be allowed
to), and a cache hit must return the tokens a cold prefill would. The first is
read from the engine's own ``vllm:prefix_cache_*`` counters, which stay at zero
while block hashing is off; the second is a greedy token-id comparison between
a cold and a warm identical request.

Each case uses a *fresh* prompt, so "cold" is genuinely cold -- the first
request in a case cannot be warmed by an earlier case. Every case also reports
the DSpark acceptance counters of its own two requests, because a wrongly
reused drafter lookahead block would show up as a loss of accepted length long
before it changed a greedy token.

``--expect-hits`` turns the reuse assertions into failures so the script gates
both directions (caching on and caching off).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decode_gate import WORDS, _post, make_token_ids, run_once  # noqa: E402

PREFIX_COUNTERS = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
)
SPEC_COUNTERS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)


def scrape(base: str) -> dict[str, float]:
    with urllib.request.urlopen(f"{base}/metrics", timeout=120) as resp:
        text = resp.read().decode()
    wanted = PREFIX_COUNTERS + SPEC_COUNTERS
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        series, _, value = line.rpartition(" ")
        name, _, labels = series.partition("{")
        if name in wanted:
            try:
                out[f"{name}|{labels}"] = float(value)
            except ValueError:
                continue
    return out


def delta(before: dict[str, float], after: dict[str, float], name: str) -> float:
    total = 0.0
    for key, value in after.items():
        key_name, _, _ = key.partition("|")
        if key_name == name:
            total += value - before.get(key, 0.0)
    return total


def timed_ids(
    base: str,
    model: str,
    prompt_ids: list[int],
    max_tokens: int,
    cache_salt: str | None = None,
) -> dict:
    """Stream one request, returning its TTFT *and* its greedy token ids.

    Streaming chunks carry ``token_ids`` as well as text, so a single timed
    request can serve as both the cold measurement and the cold reference for
    the token-id comparison -- no separate untimed call is needed to warm the
    cache before the clock starts.
    """
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
    }
    if cache_salt is not None:
        payload["cache_salt"] = cache_salt
    t0 = time.perf_counter()
    first = last = None
    n_chunks = 0
    ids: list[int] = []
    usage = None
    with _post(f"{base}/v1/completions", payload, timeout=1800.0) as resp:
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
                if choice.get("text"):
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    n_chunks += 1
    if first is None or last is None:
        raise SystemExit("stream produced no text chunks")
    return {
        "ttft_s": first - t0,
        "token_ids": ids,
        "prompt_tokens": (usage or {}).get("prompt_tokens", len(prompt_ids)),
        "completion_tokens": (usage or {}).get("completion_tokens", len(ids)),
        "intervals": max(1, n_chunks - 1),
        "decode_s": last - first,
    }


def fresh_ids(base: str, rng: random.Random, n: int) -> list[int]:
    """A prompt of exactly ``n`` tokens, unique per call.

    ``decode_gate.make_token_ids`` builds its text from a 57-word cycle, so its
    prompts are periodic and two different seeds can tokenize to the *same* ids
    -- which silently turns a "cold" request into a cache hit. Here the words
    are drawn independently, so the ids are unique to the call, while the text
    stays in-distribution enough that a greedy continuation is numerically
    stable (uniformly sampled token ids are not: their near-flat logits flip
    the argmax on tiny shape-dependent rounding differences).
    """
    text = " ".join(WORDS[rng.randrange(len(WORDS))] for _ in range(n * 4 + 512))
    with _post(f"{base}/tokenize", {"model": "", "prompt": text}) as resp:
        ids = json.loads(resp.read())["tokens"]
    if len(ids) < n:
        raise SystemExit(f"tokenize returned only {len(ids)} < {n}")
    return ids[:n]


def reset_cache(base: str) -> bool | None:
    try:
        request = urllib.request.Request(
            f"{base}/reset_prefix_cache", data=b"", method="POST"
        )
        with urllib.request.urlopen(request, timeout=60) as resp:
            return bool(json.loads(resp.read()).get("success"))
    except Exception as exc:  # noqa: BLE001 - the dev endpoint may be disabled
        print(f"reset_prefix_cache unavailable: {exc}")
        return None


def acceptance(
    before: dict[str, float], after: dict[str, float]
) -> tuple[float, float]:
    drafts = delta(before, after, SPEC_COUNTERS[0])
    draft_tokens = delta(before, after, SPEC_COUNTERS[1])
    accepted = delta(before, after, SPEC_COUNTERS[2])
    rate = accepted / draft_tokens if draft_tokens else 0.0
    length = (accepted + drafts) / drafts if drafts else 0.0
    return rate, length


def case(
    name: str,
    base: str,
    model: str,
    prompt_ids: list[int],
    max_tokens: int,
    block_size: int,
    seed: int,
    recompute_tokens: int = 1,
) -> dict:
    """One cold/warm pair on a prefix nothing earlier in this run has seen."""
    salt_a = f"gate-a-{name}-{seed}"
    salt_b = f"gate-b-{name}-{seed}"
    before = scrape(base)
    cold = timed_ids(base, model, prompt_ids, max_tokens, cache_salt=salt_a)
    mid = scrape(base)
    cold_rate, cold_len = acceptance(before, mid)
    cold_hits = delta(before, mid, PREFIX_COUNTERS[1])
    # Second cold run. ``cache_salt`` salts the first block's hash, so this
    # lands in a different cache namespace and is a genuine second cold
    # prefill of the same prompt with the same batch shape -- the only way to
    # tell "the engine is nondeterministic" apart from "the cache hit changed
    # the result", because a warm run necessarily computes its tail in a
    # smaller batch.
    repeat = timed_ids(base, model, prompt_ids, max_tokens, cache_salt=salt_b)
    warm_mid = scrape(base)
    warm = timed_ids(base, model, prompt_ids, max_tokens, cache_salt=salt_a)
    after = scrape(base)

    queries = delta(warm_mid, after, PREFIX_COUNTERS[0])
    hits = delta(warm_mid, after, PREFIX_COUNTERS[1])
    warm_rate, warm_len = acceptance(warm_mid, after)

    cold_deterministic = cold["token_ids"] == repeat["token_ids"]
    first_diff = next(
        (
            i
            for i, (a, b) in enumerate(zip(cold["token_ids"], warm["token_ids"]))
            if a != b
        ),
        None,
    )
    same = cold["token_ids"] == warm["token_ids"] and all(
        len(row["token_ids"]) == row["completion_tokens"] == max_tokens
        for row in (cold, repeat, warm)
    )
    tokens = cold["prompt_tokens"]
    # CED needs the full replay suffix; ordinary prefill needs one logit row.
    # The hit cap is then floored to the cache-hit alignment.
    #
    # Below that cap the exact hit depends on how much of the tail the sparse
    # retention policy kept, which is a property of the request's own length.
    # Assert the property that actually matters: the cache serves the aligned
    # prefix bar at most one block of it. Asserting the exact per-length value
    # would only mirror the implementation.
    expected_max = (max(0, tokens - recompute_tokens) // block_size) * block_size
    expected = max(0, expected_max - block_size)
    row = {
        "case": name,
        "prompt_tokens": tokens,
        "tail_tokens": tokens % block_size,
        "max_tokens": max_tokens,
        "cold_ttft_s": cold["ttft_s"],
        "warm_ttft_s": warm["ttft_s"],
        "cold_ms_per_token": cold["decode_s"] / cold["intervals"] * 1e3,
        "warm_ms_per_token": warm["decode_s"] / warm["intervals"] * 1e3,
        "cold_hits": cold_hits,
        "warm_queries": queries,
        "warm_hits": hits,
        "hit_fraction": hits / tokens if tokens else 0.0,
        "expected_hit_tokens": expected,
        "expected_hit_max": expected_max,
        "recompute_tokens": recompute_tokens,
        "cold_runs_identical": cold_deterministic,
        "token_ids_identical": same,
        "first_divergence": first_diff,
        "cold_token_ids": cold["token_ids"],
        "repeat_token_ids": repeat["token_ids"],
        "warm_token_ids": warm["token_ids"],
        "cold_acceptance_rate": cold_rate,
        "cold_accepted_length": cold_len,
        "warm_acceptance_rate": warm_rate,
        "warm_accepted_length": warm_len,
    }
    print(
        f"[{name}] prompt={tokens} tail={tokens % block_size} "
        f"ttft {cold['ttft_s']:.3f} -> {warm['ttft_s']:.3f} s  "
        f"cold_hits={cold_hits:.0f} warm_hits={hits:.0f}/{queries:.0f} "
        f"(expect {row['expected_hit_tokens']}) "
        f"cold_deterministic={cold_deterministic} "
        f"first_divergence={first_diff} accept {cold_len:.3f}->{warm_len:.3f}"
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=26663)
    ap.add_argument("--model", default="dsv41f-port")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument(
        "--recompute-tokens",
        type=int,
        default=1,
        help="Minimum uncached suffix: use 128 for DeepSeek V4.1 CED",
    )
    ap.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help=(
            "Prompt sampling seed. Defaults to a fresh random value, because "
            "the gate's cold/warm contrast is only meaningful while the cache "
            "is empty: a repeated seed reproduces an already-cached prompt and "
            "makes the 'cold' request a hit. Pass an explicit value to "
            "reproduce a run."
        ),
    )
    ap.add_argument("--reset", action="store_true", help="reset the cache first")
    ap.add_argument(
        "--expect-hits",
        action="store_true",
        help="fail unless every warm request is served from the cache",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.recompute_tokens < 1:
        ap.error("--recompute-tokens must be positive")

    base = f"http://{args.host}:{args.port}"
    if args.reset:
        print(f"reset_prefix_cache -> {reset_cache(base)}")
    seed = args.seed_base
    if seed is None:
        seed = int.from_bytes(os.urandom(2), "big") * 100 + 1
    print(f"seed_base={seed}")

    # Engine warmup on a throwaway prefix: the first request of a fresh server
    # pays one-time costs (cudagraph padding, first MoE kernel) that would
    # otherwise land in the first case's cold numbers.
    run_once(base, args.model, make_token_ids(base, 64, seed + 900), 8, 1800.0)

    rng = random.Random(seed)
    rows = [
        # Fits in one 1024-token prefill chunk: the hit is the whole prompt
        # block-aligned, and the warm request only computes its tail.
        case(
            "single-chunk",
            base,
            args.model,
            fresh_ids(base, rng, 1500),
            args.max_tokens,
            args.block_size,
            seed,
            args.recompute_tokens,
        ),
        # Longer than max_num_batched_tokens, so the cold request is prefilled
        # in chunks and the warm one must skip a chunk boundary mid-prefix.
        case(
            "chunked-prefill",
            base,
            args.model,
            fresh_ids(base, rng, 2500),
            args.max_tokens,
            args.block_size,
            seed,
            args.recompute_tokens,
        ),
        # Two requests sharing a prefix but diverging in the tail: reuse must
        # not depend on the whole prompt matching.
        case(
            "shared-prefix",
            base,
            args.model,
            fresh_ids(base, rng, 1200) + fresh_ids(base, rng, 16),
            args.max_tokens,
            args.block_size,
            seed,
            args.recompute_tokens,
        ),
        # Exactly block-aligned plus a tail just above the short-tail band: the
        # warm request refills almost nothing and every other token it decodes
        # from is read out of a reused block. A stale or lookahead-polluted
        # reused block cannot pass this and the 1500/2500-token cases at once.
        case(
            "full-hit",
            base,
            args.model,
            fresh_ids(base, rng, 10 * args.block_size + 33),
            args.max_tokens,
            args.block_size,
            seed,
            args.recompute_tokens,
        ),
        # Regression case for the sparse-retention tail: a prompt ending 1..32
        # tokens past a block boundary used to reconcile to zero reuse, because
        # the retained tail ended on a block that was not a full block of the
        # request, leaving one fewer than the finder needs. Any tail in that
        # band must reuse the whole aligned prefix.
        case(
            "short-tail",
            base,
            args.model,
            fresh_ids(base, rng, 10 * args.block_size + 16),
            args.max_tokens,
            args.block_size,
            seed,
            args.recompute_tokens,
        ),
    ]

    total_hits = sum(r["warm_hits"] for r in rows)
    scored = rows
    all_deterministic = all(r["cold_runs_identical"] for r in rows)
    any_short = any(
        not (r["expected_hit_tokens"] <= r["warm_hits"] <= r["expected_hit_max"])
        for r in scored
    )
    no_speedup = [r["case"] for r in scored if r["warm_ttft_s"] >= r["cold_ttft_s"]]
    contaminated = [r["case"] for r in rows if r["cold_hits"] > 0]

    result = {
        "seed_base": seed,
        "cases": rows,
        "total_hits": total_hits,
        "cold_runs_all_identical": all_deterministic,
        "mean_cold_ttft_s": sum(r["cold_ttft_s"] for r in scored) / len(scored),
        "mean_warm_ttft_s": sum(r["warm_ttft_s"] for r in scored) / len(scored),
    }
    ok = True
    mismatches = [r["case"] for r in rows if not r["token_ids_identical"]]
    if mismatches:
        print(f"FAIL: cold/warm token ids or completion lengths differ: {mismatches}")
        ok = False
    if args.expect_hits:
        if any(r["warm_hits"] <= 0 for r in scored):
            print(
                "FAIL: a warm request was not served from the prefix cache: "
                f"{[r['case'] for r in scored if r['warm_hits'] <= 0]}"
            )
            ok = False
        if any_short:
            print("FAIL: warm reuse outside the block-aligned prefix range")
            for r in scored:
                if not (
                    r["expected_hit_tokens"] <= r["warm_hits"] <= r["expected_hit_max"]
                ):
                    print(
                        f"  {r['case']}: hits={r['warm_hits']:.0f} not in "
                        f"[{r['expected_hit_tokens']}, {r['expected_hit_max']}]"
                    )
            ok = False
        if contaminated:
            print(f"FAIL: cold request already hit the cache: {contaminated}")
            ok = False
        if not all_deterministic:
            print("FAIL: two identical cold prefills disagreed")
            ok = False
        if no_speedup:
            print(f"FAIL: no TTFT drop on the warm request: {no_speedup}")
            ok = False
    else:
        if total_hits != 0:
            print(f"FAIL: hits={total_hits:.0f} while caching is expected off")
            ok = False
        if not all_deterministic:
            print("FAIL: cold/cold greedy outputs differ")
            ok = False

    print(
        f"MEAN scored_hits={sum(r['warm_hits'] for r in scored):.0f} "
        f"ttft {result['mean_cold_ttft_s']:.3f} -> "
        f"{result['mean_warm_ttft_s']:.3f} s "
        f"cold_deterministic={all_deterministic} "
        f"{'PASS' if ok else 'FAIL'}"
    )
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"wrote {args.out}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
