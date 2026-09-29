# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transfer one inspected Linux process tree to an empty CPU supervisor.

Prepare is read-only toward the server. Run from a detached systemd unit after
stopping the supervisor. The durable journal makes an interrupted cutover fall
back to the captured original command. It never signals a process group.
"""

import argparse
import ctypes
import os
import platform
import signal
import subprocess
import time
import urllib.parse
from pathlib import Path

from controller import Controller, atomic, lock, read


def pidfd_call(number, *args):
    # uv's portable CPython may not expose pidfd APIs even on a modern kernel.
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "aarch64"}:
        raise RuntimeError("Adoption requires Linux x86_64/aarch64 pidfds")
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(ctypes.c_long(number), *args)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def pidfd_open(pid):
    return pidfd_call(434, ctypes.c_int(pid), ctypes.c_uint(0))


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {
            "pid": pid,
            "parent": int(fields[1]),
            "start": fields[19],
            "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(item):
    current = identity(item["pid"])
    return current is not None and all(
        current[key] == item[key] for key in ("pid", "start", "boot")
    )


def tree(pid):
    root = identity(pid)
    if not root:
        raise ValueError("Original server is no longer running")
    items = [
        item
        for p in Path("/proc").iterdir()
        if p.name.isdecimal() and (item := identity(int(p.name)))
    ]
    selected = {pid: root}
    while True:
        children = {p["pid"]: p for p in items if p["parent"] in selected}
        before = len(selected)
        selected.update(children)
        if len(selected) == before:
            return list(selected.values())


def send(item, sig):
    if not same_process(item):
        return
    try:
        fd = pidfd_open(item["pid"])
    except ProcessLookupError:
        return
    try:
        if same_process(item):
            pidfd_call(
                424,
                ctypes.c_int(fd),
                ctypes.c_int(sig),
                ctypes.c_void_p(),
                ctypes.c_uint(0),
            )
    finally:
        os.close(fd)


def stop_original(items, timeout=120):
    # Parent first permits graceful worker teardown. Only captured identities
    # can be killed, including when a journal is recovered after a machine boot.
    send(items[0], signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while any(same_process(item) for item in items):
        if time.monotonic() >= deadline:
            for item in reversed(items):
                send(item, signal.SIGKILL)
            break
        time.sleep(0.2)
    deadline = time.monotonic() + 15
    while any(same_process(item) for item in items):
        if time.monotonic() >= deadline:
            raise RuntimeError("Original processes remain; refusing a second server")
        time.sleep(0.2)


def prepare(controller, release, pid):
    root = controller.root
    fd = pidfd_open(os.getpid())
    os.close(fd)
    if (root / "adoption.json").exists():
        raise ValueError("Adoption journal already exists; preserve it")
    if controller.state["stable"] or controller.state["current"]:
        raise ValueError("Adoption requires an empty supervisor")
    if list((root / "requests").glob("*.json")):
        raise ValueError("Pending or historical requests prevent initial adoption")
    endpoint = urllib.parse.urlparse(
        controller.store.release(release)["probe"]["base_url"]
    )
    listeners = subprocess.check_output(
        ["ss", "-ltnp", f"sport = :{endpoint.port or 80}"], text=True
    )
    if f"pid={pid}," not in listeners:
        raise ValueError("Selected PID does not own the release endpoint listener")
    controller.seal(release, verify=True)
    if not controller.probe(release):
        raise RuntimeError("Original endpoint did not produce a token")
    processes = tree(pid)
    proc = Path(f"/proc/{pid}")
    argv = proc.joinpath("cmdline").read_bytes().decode().rstrip("\0").split("\0")
    env = dict(
        entry.split("=", 1)
        for entry in proc.joinpath("environ").read_bytes().decode().split("\0")
        if entry
    )
    if not same_process(processes[0]):
        raise RuntimeError("Original process changed while capturing")
    journal = {
        "stage": "prepared",
        "release": release,
        "processes": processes,
        "original": {
            "argv": argv,
            "cwd": str(proc.joinpath("cwd").resolve()),
            "env": env,
        },
        "prepared_at": time.time(),
    }
    atomic(root / "adoption.json", journal)


def fallback(controller, journal):
    root = controller.root
    journal.update(stage="restoring_original")
    atomic(root / "adoption.json", journal)
    controller.backend.stop("job")
    controller.backend.stop("server")
    stop_original(journal["processes"])
    spec = journal["original"]
    controller.backend.start(
        "server", spec["argv"], spec["cwd"], spec["env"], root / "adoption-fallback.log"
    )
    deadline = (
        time.monotonic()
        + controller.store.release(journal["release"])["startup_timeout"]
    )
    while time.monotonic() < deadline:
        if not controller.backend.running("server"):
            break
        if controller.probe(journal["release"]):
            controller.save(current=None, stable=None, active=None, phase="blocked")
            journal.update(stage="fallback_running", finished_at=time.time())
            atomic(root / "adoption.json", journal)
            return
        time.sleep(controller.config["poll_seconds"])
    raise RuntimeError(
        "Original command fallback failed; inspect adoption-fallback.log"
    )


def run(controller, observation=60):
    root = controller.root
    journal = read(root / "adoption.json")
    release = journal["release"]
    with lock(root / ".daemon.lock"):
        # Stable is committed only after all gates. Repair a crash between that
        # atomic commit and its journal update without stopping a healthy server.
        if controller.state["stable"] == release:
            journal.update(stage="adopted", finished_at=time.time())
            atomic(root / "adoption.json", journal)
        elif journal["stage"] in {"fallback_running", "blocked"}:
            return
        else:
            try:
                if journal["stage"] != "prepared":
                    raise RuntimeError("Interrupted adoption; restore original command")
                controller.seal(release, verify=True)
                if not same_process(journal["processes"][0]):
                    raise RuntimeError("Original process disappeared before cutover")
                journal["processes"] = tree(journal["processes"][0]["pid"])
                journal.update(stage="stopping_original", started_at=time.time())
                atomic(root / "adoption.json", journal)
                stop_original(journal["processes"])
                journal.update(stage="starting_capsule")
                atomic(root / "adoption.json", journal)
                controller.save(phase="adopting")
                controller.start_release(release, root / "adoption-server.log")
                spec = controller.store.release(release)
                for index, argv in enumerate(spec["checks"]):
                    controller.job(argv, spec, root / f"adoption-check-{index}.log")
                deadline = time.monotonic() + observation
                while time.monotonic() < deadline:
                    if not controller.backend.running("server") or not controller.probe(
                        release
                    ):
                        raise RuntimeError("Adopted server failed observation")
                    time.sleep(controller.config["poll_seconds"])
                controller.seal(release, verify=True)
                controller.save(
                    stable=release,
                    current=release,
                    phase="idle",
                    active=None,
                    restart_count=0,
                )
                journal.update(stage="adopted", finished_at=time.time())
                atomic(root / "adoption.json", journal)
                controller.store.event("adopted", release=release)
            except Exception as error:
                journal["error"] = str(error)
                # Before the first stop the original may still be untouched.
                if journal["stage"] == "prepared":
                    journal.update(stage="blocked", finished_at=time.time())
                    atomic(root / "adoption.json", journal)
                    raise
                try:
                    fallback(controller, journal)
                except Exception as restore_error:
                    journal.update(stage="blocked", fallback_error=str(restore_error))
                    atomic(root / "adoption.json", journal)
                    controller.save(
                        current=None, stable=None, active=None, phase="blocked"
                    )
                    raise
    installation = read(root / "installation.json")
    subprocess.run(["systemctl", "--user", "start", installation["unit"]], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--release", required=True)
    prep.add_argument("--pid", required=True, type=int)
    sub.add_parser("run")
    args = parser.parse_args()
    controller = Controller(args.root)
    if args.action == "prepare":
        with lock(controller.root / ".daemon.lock"):
            prepare(controller, args.release, args.pid)
    else:
        run(controller)
