# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drive the supported DSH headless profile and retain its session receipt."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from controller import atomic, lock, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("task")
    args = parser.parse_args()
    config = read(args.config)
    root = args.config.parent
    session_file = root / "dsh-session.json"
    receipt = {"started_at": time.time(), "status": "started"}
    receipt_file = root / "dsh-receipts" / f"{time.time_ns()}.json"
    with lock(root / ".dsh.lock"):
        argv = config["argv"][:]
        if session_file.exists():
            argv += ["--session-id", read(session_file)["session_id"]]
        argv.append(args.task)
        env = dict(os.environ)
        # pi-ai requires a token even for an unauthenticated localhost service.
        env.setdefault("EVOLUTION_LOCAL_API_KEY", "local")
        atomic(receipt_file, receipt)
        final = False
        completed = False
        with subprocess.Popen(
            argv, cwd=config["cwd"], env=env, stdout=subprocess.PIPE, text=True
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                try:
                    event = json.loads(line)
                except ValueError:
                    continue  # pnpm's launcher banner is not a JSON event.
                if event.get("type") == "session":
                    receipt["session_id"] = event["sessionId"]
                    atomic(receipt_file, receipt)
                if event.get("type") == "status" and event.get("phase") == "turn_end":
                    completed = event.get("reason", {}).get("kind") == "completed"
                if event.get("type") == "final":
                    final = True
                    receipt["answer"] = event.get("text", "")
            code = process.wait()
        success = code == 0 and completed and final and "session_id" in receipt
        receipt.update(
            status="completed" if success else "uncertain",
            exit_code=code,
            finished_at=time.time(),
        )
        atomic(receipt_file, receipt)
        if success:
            atomic(session_file, {"session_id": receipt["session_id"]})
        raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
