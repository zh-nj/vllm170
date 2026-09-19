#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe prefill batch-shape sensitivity with explicit full-prefill evidence.

Every request uses a fresh cache salt and is checked against the prefix-cache
counters. Repeating a prompt length checks that the same effective batch shape
is bit-identical without relying on rounded logprob output.

Reading the counters correctly matters here. ``prompt_logprobs`` is not None
makes ``SamplingParams.skip_reading_prefix_cache`` true, so the scheduler never
runs a prefix-cache lookup for these requests and never records one. A zero
query delta therefore proves the request was a full prefill *by construction*;
it is not evidence about what the cache contained, and it is not evidence of
reuse having been avoided by luck. The fresh salt is a second, independent
guard for the case where the probe stops sending ``prompt_logprobs``.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decode_gate import _post  # noqa: E402
from prefix_cache_gate import PREFIX_COUNTERS, delta, scrape  # noqa: E402


def _counter_keys(metrics: dict[str, float], name: str) -> list[str]:
    return [key for key in metrics if key.partition("|")[0] == name]


def _check_counters(metrics: dict[str, float], where: str) -> None:
    missing = [name for name in PREFIX_COUNTERS if not _counter_keys(metrics, name)]
    if missing:
        raise RuntimeError(f"missing prefix-cache counters in {where}: {missing}")


def _counter_deltas(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, float]:
    """Return the prefix-cache counter deltas for one probe request.

    ``prompt_logprobs`` disables the cache lookup for the request, so both
    deltas are expected to be zero. Anything else means the probe is no longer
    measuring a guaranteed full prefill.
    """
    _check_counters(before, "before request")
    _check_counters(after, "after request")
    values = {
        "cold_queries": delta(before, after, PREFIX_COUNTERS[0]),
        "cold_hits": delta(before, after, PREFIX_COUNTERS[1]),
    }
    return values


def _cached_tokens(data: dict) -> int | None:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    details = usage.get("prompt_tokens_details")
    candidates = [usage.get("cached_tokens")]
    if isinstance(details, dict):
        candidates.append(details.get("cached_tokens"))
    for value in candidates:
        if value is None:
            continue
        if isinstance(value, bool):
            raise ValueError(f"usage cached_tokens is not an integer: {value!r}")
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"usage cached_tokens is invalid: {value!r}") from exc
        if integer != value:
            raise ValueError(f"usage cached_tokens is not integral: {value!r}")
        return integer
    return None


def probe_logprob(
    base: str,
    ids: list[int],
    n: int,
    probe: int,
    salt: str,
    topk: int = 0,
) -> dict:
    """Run one request and return the exact probed value and request metadata."""
    if not 0 < probe < n or probe >= len(ids):
        raise ValueError(f"probe={probe} is outside prompt length n={n}")
    target_token_id = int(ids[probe])
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
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("completion response has no choices")
    prompt_logprobs = choices[0].get("prompt_logprobs")
    if not isinstance(prompt_logprobs, list) or probe >= len(prompt_logprobs):
        length = None if prompt_logprobs is None else len(prompt_logprobs)
        raise ValueError(f"prompt_logprobs length={length} does not include {probe}")
    entry = prompt_logprobs[probe]
    if entry is None:
        raise ValueError(f"prompt_logprobs[{probe}] is None")
    if isinstance(entry, dict) and "logprob" not in entry:
        for key in (target_token_id, str(target_token_id)):
            if key in entry:
                entry = entry[key]
                break
        else:
            raise ValueError(
                f"prompt_logprobs[{probe}] has no target token {target_token_id}"
            )
    if not isinstance(entry, dict) or "logprob" not in entry:
        raise ValueError(f"prompt_logprobs[{probe}] has no logprob field")
    value = float(entry["logprob"])
    if not math.isfinite(value):
        raise ValueError(f"probe logprob is not finite: {value!r}")
    return {
        "logprob": value,
        "predictor_position": probe - 1,
        "target_token_id": target_token_id,
        "usage_cached_tokens": _cached_tokens(data),
    }


def _parse_lengths(
    raw: list[str] | None, probe: int, max_len: int, step: int
) -> list[int]:
    if raw:
        values: list[int] = []
        for item in raw:
            for value in item.split(","):
                if value.strip():
                    values.append(int(value))
        if not values:
            raise ValueError("--lengths did not contain any prompt lengths")
    else:
        values = list(range(probe + 1, max_len + 1, step))
    if not values:
        raise ValueError("no prompt lengths were selected")
    if any(value <= probe for value in values):
        raise ValueError(f"every prompt length must be greater than probe={probe}")
    if any(value <= 0 for value in values):
        raise ValueError("prompt lengths must be positive")
    if len(set(values)) != len(values):
        raise ValueError("prompt lengths must be unique")
    return values


def _same_float(left: float, right: float) -> bool:
    """Compare finite Python floats without collapsing signed zero."""
    return left.hex() == right.hex()


def _local_git_sha() -> str | None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    sha = result.stdout.strip()
    return sha or None


def _set_assumed_tail(row: dict, n: int, chunk1: int | None) -> None:
    row["assumed_chunk1"] = chunk1
    row["estimated_tail_chunk"] = None if chunk1 is None else n - chunk1


def _run_request(
    base: str,
    ids: list[int],
    n: int,
    probe: int,
    salt: str,
    repeat_index: int,
    chunk1: int | None,
) -> dict:
    row = {
        "prompt_tokens": n,
        "repeat_index": repeat_index,
        "salt": salt,
        "predictor_position": probe - 1,
        "target_token_id": int(ids[probe]) if probe < len(ids) else None,
        "logprob": None,
        "usage_cached_tokens": None,
        "cold_queries": None,
        "cold_hits": None,
        "status": "error",
        "error": None,
    }
    _set_assumed_tail(row, n, chunk1)
    before: dict[str, float] | None = None
    after: dict[str, float] | None = None
    try:
        before = scrape(base)
        _check_counters(before, "before request")
        result = probe_logprob(base, ids, n, probe, salt)
        row.update(
            {
                "logprob": result["logprob"],
                "predictor_position": result["predictor_position"],
                "target_token_id": result["target_token_id"],
                "usage_cached_tokens": result["usage_cached_tokens"],
            }
        )
        if row["usage_cached_tokens"] not in (None, 0):
            raise RuntimeError(
                f"request usage reports cached_tokens={row['usage_cached_tokens']}"
            )
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - preserve failed rows in artifact
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if before is not None:
            try:
                after = scrape(base)
            except Exception as exc:  # noqa: BLE001 - preserve cold delta failure
                row["error"] = row["error"] or (
                    f"{type(exc).__name__}: failed to scrape counters after "
                    f"request: {exc}"
                )
    if before is not None and after is not None:
        try:
            values = _counter_deltas(before, after)
            row.update(values)
            for name, value in values.items():
                if not math.isfinite(value) or value < 0:
                    if not math.isfinite(value):
                        row[f"{name}_raw"] = repr(value)
                        row[name] = None
                    raise RuntimeError(f"invalid {name} delta: {value!r}")
        except Exception as exc:  # noqa: BLE001 - preserve counter evidence
            row["error"] = row["error"] or f"{type(exc).__name__}: {exc}"
    if row["error"] is not None:
        row["status"] = "error"
    if row["cold_queries"] not in (None, 0.0):
        row["error"] = row["error"] or (
            "RuntimeError: probe request ran a prefix-cache lookup "
            f"(cold_queries={row['cold_queries']}); prompt_logprobs must skip it"
        )
        row["status"] = "error"
    if row["cold_hits"] not in (None, 0.0):
        row["error"] = row["error"] or (
            f"RuntimeError: cold request had cold_hits={row['cold_hits']}"
        )
        row["status"] = "error"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=26663)
    ap.add_argument("--doc", default="docs/design/prefix_caching.md")
    ap.add_argument("--probe", type=int, default=1300, help="prompt token index")
    ap.add_argument(
        "--chunk1",
        type=int,
        default=None,
        help="optional assumed first chunk; never reported as actual M",
    )
    ap.add_argument("--max-len", type=int, default=1620)
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument(
        "--lengths",
        nargs="+",
        default=None,
        help="prompt lengths, as separate values or a comma-separated list",
    )
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.repeat < 1:
        ap.error("--repeat must be at least 1")
    if args.step < 1:
        ap.error("--step must be at least 1")
    try:
        lengths = _parse_lengths(args.lengths, args.probe, args.max_len, args.step)
    except ValueError as exc:
        ap.error(str(exc))

    run_id = str(uuid4())
    started_at = datetime.now(timezone.utc).isoformat()
    base = f"http://{args.host}:{args.port}"
    metadata = {
        "run_id": run_id,
        "started_at_utc": started_at,
        "localgitsha": _local_git_sha(),
        "localgitsha_scope": "client checkout; server revision is not observed",
        "base_url": base,
        "args": vars(args),
        "lengths": lengths,
    }

    rows: list[dict] = []
    try:
        text = Path(args.doc).read_text()
        with _post(
            f"{base}/tokenize",
            {"model": "", "prompt": (text + "\n\n" + text[::-1]) * 6},
        ) as resp:
            ids = json.loads(resp.read())["tokens"]
        if max(lengths) > len(ids):
            raise ValueError(
                f"tokenizer returned {len(ids)} tokens, need {max(lengths)}"
            )
        for n in lengths:
            for repeat_index in range(args.repeat):
                salt = f"shape-{run_id}-{n}-{repeat_index}"
                row = _run_request(
                    base,
                    ids,
                    n,
                    args.probe,
                    salt,
                    repeat_index,
                    args.chunk1,
                )
                row["run_id"] = run_id
                rows.append(row)
    except Exception as exc:  # noqa: BLE001 - artifact must contain prior rows
        rows.append(
            {
                "prompt_tokens": None,
                "repeat_index": None,
                "salt": None,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "cold_queries": None,
                "cold_hits": None,
                "usage_cached_tokens": None,
            }
        )

    by_length: dict[int, list[dict]] = {}
    for row in rows:
        n = row.get("prompt_tokens")
        if isinstance(n, int):
            by_length.setdefault(n, []).append(row)
    previous: float | None = None
    repeat_rows_identical = True
    for n in lengths:
        group = by_length.get(n, [])
        values = [row["logprob"] for row in group if row.get("status") == "ok"]
        if len(values) != args.repeat:
            repeat_rows_identical = False
            continue
        identical = all(_same_float(value, values[0]) for value in values[1:])
        repeat_rows_identical &= identical
        changed = previous is None or not _same_float(values[0], previous)
        for row in group:
            row["changed"] = changed
            row["repeat_identical"] = identical
        previous = values[0]

    valid_values = [row["logprob"] for row in rows if row.get("status") == "ok"]
    cold_hits = [
        row.get("cold_hits") for row in rows if row.get("cold_hits") is not None
    ]
    cold_queries = [
        row.get("cold_queries") for row in rows if row.get("cold_queries") is not None
    ]
    errors = [row for row in rows if row.get("status") != "ok"]
    no_lookup = bool(cold_queries) and all(value == 0 for value in cold_queries)
    summary = {
        "requested_lengths": lengths,
        "requested_repeats": args.repeat,
        "completed_requests": len(valid_values),
        "failed_requests": len(errors),
        "repeat_rows_identical": repeat_rows_identical,
        "repeat_exact_by_length": {
            str(n): all(
                row.get("repeat_identical") is True for row in by_length.get(n, [])
            )
            and len(by_length.get(n, [])) == args.repeat
            for n in lengths
        },
        "prefix_cache_lookup_skipped": no_lookup,
        "all_cold_hits_zero": bool(cold_hits)
        and all(value == 0 for value in cold_hits),
        "cold_queries_total": sum(
            row["cold_queries"]
            for row in rows
            if isinstance(row.get("cold_queries"), (int, float))
        ),
        "cold_hits_total": sum(cold_hits),
        "distinct_values": len(set(valid_values)),
        "spread": (max(valid_values) - min(valid_values)) if valid_values else None,
        "ok": bool(valid_values)
        and not errors
        and repeat_rows_identical
        and no_lookup
        and bool(cold_hits)
        and all(value == 0 for value in cold_hits),
    }
    artifact = {"metadata": metadata, "rows": rows, "summary": summary}

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(artifact, fh, indent=2, allow_nan=False)
        print(f"wrote {args.out}")

    for n in lengths:
        group = by_length.get(n, [])
        values = [row["logprob"] for row in group if row.get("status") == "ok"]
        if not values:
            print(f"  n={n:5d} ERROR")
            continue
        tail = ""
        if args.chunk1 is not None:
            tail = (
                f" estimated_tail_chunk={n - args.chunk1:4d}"
                f" (assumed chunk1={args.chunk1})"
            )
        print(
            f"  n={n:5d}{tail} logprob={values[0]:.6f} "
            f"repeats={len(values)} exact="
            f"{all(_same_float(value, values[0]) for value in values)}"
        )
    print(
        f"distinct values={summary['distinct_values']} over "
        f"{summary['completed_requests']} requests; spread={summary['spread']} "
        f"lookup_skipped={summary['prefix_cache_lookup_skipped']} "
        f"cold_hits={summary['cold_hits_total']} "
        f"repeat_identical={summary['repeat_rows_identical']} "
        f"{'PASS' if summary['ok'] else 'FAIL'}"
    )
    raise SystemExit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
