# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run CPU fault injection against an installed, empty supervisor instance.

Artifacts remain in the supplied state directory. Only this instance's units
are controlled. The production model endpoint is never stopped or reconfigured.
"""

import argparse
import json
import shutil
import socket
import subprocess
import time
from pathlib import Path

from controller import Store, atomic, read


def wait_for(predicate, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.1)
    raise TimeoutError("Timed out waiting for supervisor state")


def prepare(store, python, port, name, mode="good", build=None, checks=None):
    folder = store.root / "releases" / name / "source"
    folder.mkdir(parents=True)
    shutil.copy2(Path(__file__).parent / "tests/fake_server.py", folder / "server.py")
    spec = {
        "id": name,
        "cwd": str(folder),
        "argv": [
            python,
            str(folder / "server.py"),
            "--port",
            str(port),
            "--mode",
            mode,
        ],
        "env": {"PYTHONDONTWRITEBYTECODE": "1"},
        "probe": {
            "base_url": f"http://127.0.0.1:{port}",
            "model": "cpu-fixture",
            "prompt": "hi",
        },
        "build": build or [],
        "checks": checks or [["/bin/true"]],
        "seal": ["source"],
        "startup_timeout": 3,
        "job_timeout": 2,
    }
    store.register(spec)


def experiment(name, candidate, baseline="good", observation=0):
    return {
        "id": name,
        "kind": "experiment",
        "baseline": baseline,
        "candidate": candidate,
        "hypothesis": "CPU recovery fault injection",
        "checkpoint": (
            "CPU fixture only; preserve production dsv41f. "
            "Read result then update notes."
        ),
        "observation_seconds": observation,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    store = Store(args.root)
    root = store.root
    installation = read(root / "installation.json")
    unit = installation["unit"]
    original = read(root / "config.json")
    if read(root / "state.json")["stable"] or list((root / "requests").glob("*.json")):
        raise ValueError("Verification requires a fresh instance")
    config = {k: v for k, v in original.items() if k != "resume"}
    config.update(poll_seconds=0.1, probe_timeout=0.2, failure_threshold=2)
    atomic(root / "config.json", config)
    subprocess.run(["systemctl", "--user", "restart", unit], check=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    python = installation["python"]
    report = {"scope": "CPU fixture only, no GPU service restart", "cases": []}

    def submit(request, expected):
        store.submit(request)
        path = root / "experiments" / request["id"] / "result.json"
        result = wait_for(lambda: read(path) if path.exists() else None)
        assert result["status"] == expected, result
        report["cases"].append({"id": request["id"], "status": result["status"]})
        atomic(root / "verification.json", report)
        return result

    try:
        prepare(store, python, port, "good")
        submit(experiment("bootstrap", "good", None), "ready")
        submit(
            {
                "id": "promote",
                "kind": "promote",
                "experiment": "bootstrap",
                "decision": "CPU fixture produces tokens; fixture baseline only",
            },
            "ready",
        )
        for name, kwargs, expected in [
            ("build-fail", {"build": [["/bin/false"]]}, "build_failed"),
            ("start-fail", {"mode": "fail"}, "rolled_back"),
            ("bad-token", {"mode": "bad"}, "rolled_back"),
            ("check-fail", {"checks": [["/bin/false"]]}, "rolled_back"),
            ("job-timeout", {"checks": [["/bin/sleep", "30"]]}, "rolled_back"),
            ("observe-crash", {"mode": "crash"}, "rolled_back"),
        ]:
            prepare(store, python, port, name, **kwargs)
            submit(
                experiment(name, name, observation=4 if name == "observe-crash" else 0),
                expected,
            )
            assert read(root / "state.json")["current"] == "good"
        prepare(store, python, port, "interrupted")
        req = experiment("interrupted", "interrupted", observation=30)
        store.submit(req)
        wait_for(lambda: read(root / "state.json")["phase"] == "observing")
        subprocess.run(
            ["systemctl", "--user", "kill", "--signal=SIGKILL", unit], check=True
        )
        submit(req, "rolled_back")
        assert read(root / "state.json")["current"] == "good"
        # Leave one untouched candidate for an actual DSH submission test.
        prepare(store, python, port, "dsh-failure", mode="fail")
        report.update(status="passed", port=port, finished_at=time.time())
        atomic(root / "verification.json", report)
    finally:
        # Restoration is explicit. DSH can now receive only the latest pending
        # event; old notifications are retained as superseded, not replayed.
        original["poll_seconds"] = 2
        atomic(root / "config.json", original)
        subprocess.run(["systemctl", "--user", "restart", unit], check=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
