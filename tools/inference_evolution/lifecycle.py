# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Start or stop a mission's CPU lifeguard without racing a transaction."""

import argparse
import contextlib
import json
import subprocess
import time
from pathlib import Path

from controller import (
    Controller,
    Store,
    atomic,
    bounded,
    lock,
    read,
    validate_lifeguard,
)


def settled(root, state, backend):
    if state.get("active") is not None or state.get("phase") not in {"idle", "blocked"}:
        raise RuntimeError("Resolve the active/unpromoted transaction before shutdown")
    pending = [
        p.stem
        for p in (root / "requests").glob("*.json")
        if not (root / "experiments" / p.stem / "result.json").exists()
    ]
    if pending:
        raise RuntimeError(f"Queued requests must be resolved first: {pending}")
    if any(read(p)["status"] == "delivering" for p in (root / "outbox").glob("*.json")):
        raise RuntimeError("Recovery session is still delivering")
    for slot in ("job", "resume"):
        if backend.running(slot):
            raise RuntimeError(f"Owned {slot} is still running")


@contextlib.contextmanager
def transition_lock(root, timeout=5):
    deadline = time.monotonic() + timeout
    with contextlib.ExitStack() as stack:
        while True:
            try:
                stack.enter_context(lock(root / ".transition.lock"))
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Supervisor busy; retry after the transaction completes"
                    ) from None
                time.sleep(0.05)
        yield


def run(root, action, lifeguard=None, ttl=3600):
    if action not in {"start", "stop"}:
        raise ValueError("Expected start or stop")
    if action == "start":
        validate_lifeguard(lifeguard)
        bounded(ttl, "mission ttl", minimum=1)
    root = Path(root).resolve()
    installation = read(root / "installation.json")
    name = installation["unit"]
    # Locks stay held until systemd acknowledges stop; the controller uses
    # the transition lock across the entire tick/recovery, including GPU loads.
    with (
        lock(root / ".lifecycle.lock"),
        transition_lock(root),
        lock(root / ".submit.lock"),
    ):
        controller = Controller(root)
        settled(root, controller.state, controller.backend)
        command = subprocess.check_output(
            ["systemctl", "--user", "show", name, "-p", "ExecStart", "--value"],
            text=True,
        )
        if (
            str(root) not in command
            or installation["package"] + "/controller.py" not in command
        ):
            raise RuntimeError("Refusing to control an unrelated supervisor unit")
        active = subprocess.check_output(
            ["systemctl", "--user", "show", name, "-p", "ActiveState", "--value"],
            text=True,
        ).strip()
        if action == "start":
            if active not in {"inactive", "failed"}:
                raise RuntimeError(
                    "Supervisor already running; do not steal another mission"
                )
            if controller.state["phase"] == "blocked":
                raise RuntimeError(
                    "Resolve blocked state before starting a new mission"
                )
            controller.apply_lifeguard(lifeguard)
            atomic(
                root / "control.json",
                {
                    "paused": False,
                    "origin_session": True,
                    "expires_at": time.time() + ttl,
                },
            )
        else:
            control = read(root / "control.json")
            control["paused"] = True
            atomic(root / "control.json", control)
        subprocess.run(
            ["systemctl", "--user", "disable", name], check=True, capture_output=True
        )
        if action == "stop":
            subprocess.run(
                ["systemctl", "--user", "stop", name], check=True, timeout=30
            )
            # Future starts must not revive old notes-only dispatches.
            for p in (root / "outbox").glob("*.json"):
                item = read(p)
                if item["status"] == "pending":
                    atomic(p, dict(item, status="origin_session"))
            Store(root).event("lifeguard_stopped", server_preserved=True)
        if action == "start":
            subprocess.run(
                ["systemctl", "--user", "start", name], check=True, timeout=30
            )
            Store(root).event("lifeguard_started", origin_session=True)
    result = subprocess.check_output(
        ["systemctl", "--user", "show", name, "-p", "ActiveState,SubState"], text=True
    )
    return {
        "unit": name,
        "action": action,
        **dict(line.split("=", 1) for line in result.splitlines() if "=" in line),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--lifeguard", type=Path)
    parser.add_argument("--ttl", type=float, default=3600)
    parser.add_argument("action", choices=("start", "stop"))
    args = parser.parse_args()
    if args.action == "start" and not args.lifeguard:
        parser.error("start requires --lifeguard FILE")
    print(
        json.dumps(
            run(
                args.root,
                args.action,
                read(args.lifeguard) if args.lifeguard else None,
                args.ttl,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
