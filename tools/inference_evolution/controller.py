# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only, durable supervisor for sequential local inference experiments."""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


def read(path):
    with open(path) as stream:
        return json.load(stream)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    with open(temp, "x") as stream:
        os.chmod(temp, 0o600)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def lock(path):
    with open(path, "a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def identifier(value):
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 64
        or not value.isascii()
        or not value[0].isalnum()
        or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in value)
    ):
        raise ValueError(f"Invalid identifier: {value!r}")
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def command(argv):
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(arg, str) and "\0" not in arg for arg in argv)
    ):
        raise ValueError("Commands must be nonempty argv lists")
    if not Path(argv[0]).is_absolute():
        raise ValueError("Command executables must be absolute paths")
    return argv


def bounded(value, name, minimum=0.01, maximum=86400):
    if type(value) not in (int, float) or not minimum <= value <= maximum:
        raise ValueError(f"Invalid {name}: {value!r}")
    return value


def validate_lifeguard(value):
    required = {
        "startup_timeout",
        "recovery_timeout",
        "rollback_release",
        "probe_timeout",
        "failure_threshold",
        "stable_restarts",
        "restart_policy",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"Lifeguard parameters must be {sorted(required)}")
    for key in ("startup_timeout", "recovery_timeout", "probe_timeout"):
        bounded(
            value[key],
            f"lifeguard {key}",
            maximum=60 if key == "probe_timeout" else 86400,
        )
    for key in ("failure_threshold", "stable_restarts"):
        if type(value[key]) is not int:
            raise ValueError(f"lifeguard {key} must be an integer")
        bounded(
            value[key],
            f"lifeguard {key}",
            minimum=0 if key == "stable_restarts" else 1,
            maximum=100,
        )
    if value["rollback_release"] is not None:
        identifier(value["rollback_release"])
    if value["restart_policy"] != "restore_stable_then_block":
        raise ValueError("Unsupported lifeguard restart policy")


class Systemd:
    """Own only transient units carrying this state directory's description."""

    def __init__(self, root):
        self.root = Path(root)
        self.prefix = (
            "inference-evo-" + hashlib.sha256(str(self.root).encode()).hexdigest()[:12]
        )
        self.description = f"Inference evolution {self.root}"

    def name(self, slot):
        return f"{self.prefix}-{slot}.service"

    def properties(self, slot):
        result = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                self.name(slot),
                "-p",
                "LoadState,ActiveState,SubState,Result,ExecMainStatus,Description",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return dict(
            line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
        )

    def owned(self, slot):
        props = self.properties(slot)
        if (
            props.get("LoadState", "not-found") != "not-found"
            and props.get("Description") != self.description
        ):
            raise RuntimeError(f"Refusing foreign unit {self.name(slot)}")
        return props

    def stop(self, slot):
        props = self.owned(slot)
        if props.get("LoadState", "not-found") == "not-found":
            return
        subprocess.run(
            ["systemctl", "--user", "stop", self.name(slot)],
            check=True,
            capture_output=True,
            timeout=30,
        )
        subprocess.run(
            ["systemctl", "--user", "reset-failed", self.name(slot)],
            capture_output=True,
            timeout=15,
        )
        if self.properties(slot).get("ActiveState") in {
            "active",
            "activating",
            "deactivating",
        }:
            raise RuntimeError(f"Unit did not stop: {self.name(slot)}")

    def start(self, slot, argv, cwd, env, log, job=False, timeout=None):
        self.stop(slot)
        log = Path(log)
        log.parent.mkdir(parents=True, exist_ok=True)
        args = [
            "systemd-run",
            "--user",
            "--quiet",
            f"--unit={self.name(slot)}",
            f"--description={self.description}",
            "--property=Type=exec",
            "--property=KillMode=control-group",
            "--property=TimeoutStopSec=10s",
            "--property=Restart=no",
            f"--working-directory={cwd}",
            f"--property=StandardOutput=append:{log}",
            f"--property=StandardError=append:{log}",
        ]
        if job:
            args.append("--property=RemainAfterExit=yes")
        if timeout is not None:
            args.append(f"--property=RuntimeMaxSec={timeout}s")
        for key, value in env.items():
            args.append(f"--setenv={key}={value}")
        subprocess.run(
            args + ["--"] + command(argv),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def running(self, slot):
        props = self.owned(slot)
        return (
            props.get("ActiveState") in {"active", "activating"}
            and props.get("SubState") != "exited"
        )

    def job_result(self, slot):
        props = self.owned(slot)
        if props.get("SubState") == "exited" and props.get("ExecMainStatus") == "0":
            return True
        if (
            props.get("ActiveState") in {"failed", "inactive"}
            or props.get("LoadState") == "not-found"
        ):
            return False
        return None


class Store:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()

    def initialize(self, config):
        self.validate_config(config)
        if self.root.exists() and any(self.root.iterdir()):
            raise ValueError("State directory must be empty")
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(self.root, 0o700)
        for part in (
            "releases",
            "requests",
            "experiments",
            "notes/checkpoints",
            "outbox",
        ):
            (self.root / part).mkdir(parents=True)
        self.validate_config(config)
        atomic(self.root / "config.json", config)
        atomic(
            self.root / "state.json",
            {
                "schema": 1,
                "stable": None,
                "previous_stable": None,
                "current": None,
                "active": None,
                "phase": "idle",
                "last_experiment": None,
                "restart_count": 0,
            },
        )
        atomic(self.root / "control.json", {"paused": False})
        for name, title in (
            ("current", "现场"),
            ("trajectory", "进化轨迹"),
            ("lessons", "经验与适用条件"),
            ("directions", "方向与证伪方法"),
        ):
            (self.root / "notes" / f"{name}.md").write_text(
                f"# {title}\n\n尚无实验结论。\n"
            )
        self.event("initialized")

    @staticmethod
    def validate_config(config):
        allowed = {
            "poll_seconds",
            "probe_timeout",
            "failure_threshold",
            "stable_restarts",
            "resume",
        }
        if set(config) - allowed:
            raise ValueError("Unknown controller configuration fields")
        for key in ("poll_seconds", "probe_timeout"):
            bounded(config[key], key, maximum=60)
        for key in ("failure_threshold", "stable_restarts"):
            bounded(
                config[key],
                key,
                minimum=0 if key == "stable_restarts" else 1,
                maximum=100,
            )
            if type(config[key]) is not int:
                raise ValueError(f"{key} must be an integer")
        if config.get("resume"):
            resume = config["resume"]
            if set(resume) != {"argv", "cwd", "timeout", "max_attempts"}:
                raise ValueError("Invalid resume configuration")
            command(resume["argv"])
            if sum(arg == "{prompt}" for arg in resume["argv"]) != 1:
                raise ValueError("Resume argv requires exactly one {prompt} argument")
            bounded(resume["timeout"], "resume timeout")
            bounded(resume["max_attempts"], "resume attempts", 1, 10)
            if not Path(resume["cwd"]).is_absolute():
                raise ValueError("Resume cwd must be absolute")

    def event(self, kind, **data):
        with open(self.root / "events.jsonl", "a") as stream:
            stream.write(
                json.dumps(
                    {"time": time.time(), "kind": kind, **data}, ensure_ascii=False
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())

    def release(self, name):
        return read(self.root / "releases" / identifier(name) / "release.json")

    def register(self, spec):
        required = {
            "id",
            "cwd",
            "argv",
            "env",
            "probe",
            "build",
            "checks",
            "seal",
            "startup_timeout",
            "job_timeout",
        }
        if set(spec) != required:
            raise ValueError(f"Release fields must be {sorted(required)}")
        name = identifier(spec["id"])
        folder = self.root / "releases" / name
        folder.mkdir(exist_ok=True)
        # A candidate owns its source, environment and outputs in this directory.
        cwd = Path(spec["cwd"]).resolve()
        if not cwd.is_relative_to(folder) or not cwd.is_dir():
            raise ValueError("Release cwd must exist inside its own release directory")
        command(spec["argv"])
        for commands in (spec["build"], spec["checks"]):
            if not isinstance(commands, list):
                raise ValueError("build/checks must be lists")
            for argv in commands:
                command(argv)
        if not spec["checks"] or not spec["seal"]:
            raise ValueError("At least one check and sealed path are required")
        for entry in spec["seal"]:
            target = (folder / entry).resolve()
            if not target.is_relative_to(folder) or target == folder:
                raise ValueError("Seal paths must be contained within the release")
        if not isinstance(spec["env"], dict) or not all(
            key.isascii()
            and key.isidentifier()
            and isinstance(value, str)
            and "\0" not in value
            for key, value in spec["env"].items()
        ):
            raise ValueError("Invalid environment")
        probe = spec["probe"]
        if set(probe) != {"base_url", "model", "prompt"}:
            raise ValueError("Invalid probe fields")
        url = urllib.parse.urlparse(probe["base_url"])
        if (
            url.scheme != "http"
            or url.hostname not in {"127.0.0.1", "localhost", "::1"}
            or url.username
            or url.password
        ):
            raise ValueError("Probe must use a local HTTP endpoint without credentials")
        for key in ("model", "prompt"):
            if not isinstance(probe[key], str) or not probe[key]:
                raise ValueError(f"Invalid probe {key}")
        bounded(spec["startup_timeout"], "startup timeout")
        bounded(spec["job_timeout"], "job timeout")
        with lock(self.root / ".submit.lock"):
            path = folder / "release.json"
            if path.exists() and read(path) != spec:
                raise ValueError("Registered releases cannot be overwritten")
            atomic(path, spec)
        return name

    def validate_request(self, request):
        kind = request.get("kind")
        required = {
            "id",
            "kind",
            "baseline",
            "candidate",
            "hypothesis",
            "checkpoint",
            "observation_seconds",
        }
        if kind == "promote":
            required = {"id", "kind", "experiment", "decision"}
        elif kind == "rollback":
            required = {"id", "kind", "reason", "lifeguard"}
        elif kind == "experiment":
            required.add("lifeguard")
        if (
            kind not in {"experiment", "promote", "rollback"}
            or set(request) != required
        ):
            raise ValueError("Invalid request fields")
        name = identifier(request["id"])
        if kind == "experiment":
            self.release(request["candidate"])
            if request["baseline"] is not None:
                self.release(request["baseline"])
            bounded(request["observation_seconds"], "observation", minimum=0)
            for key in ("hypothesis", "checkpoint"):
                if not isinstance(request[key], str) or not request[key].strip():
                    raise ValueError(f"Missing {key}")
            validate_lifeguard(request["lifeguard"])
        elif kind == "promote":
            identifier(request["experiment"])
            if (
                not isinstance(request["decision"], str)
                or not request["decision"].strip()
            ):
                raise ValueError("Promotion requires a recorded decision")
        elif kind == "rollback":
            if not isinstance(request["reason"], str) or not request["reason"].strip():
                raise ValueError("Rollback requires a reason")
            validate_lifeguard(request["lifeguard"])
        if (
            kind == "experiment"
            and request["lifeguard"]["rollback_release"] != request["baseline"]
        ):
            raise ValueError("Lifeguard rollback release must match the baseline")
        return name

    def submit(self, request):
        name = self.validate_request(request)
        with lock(self.root / ".submit.lock"):
            path = self.root / "requests" / f"{name}.json"
            if path.exists():
                if read(path) != request:
                    raise ValueError("Request ID already used with different content")
            else:
                atomic(path, request)
        return {"id": name, "accepted": True, "request_hash": digest(request)}


class Controller:
    def __init__(self, root, backend=None):
        self.store = Store(root)
        self.root = self.store.root
        self.config = read(self.root / "config.json")
        self.store.validate_config(self.config)
        self.state = read(self.root / "state.json")
        self.backend = backend or Systemd(self.root)
        self.failures = 0

    def save(self, **changes):
        self.state.update(changes)
        atomic(self.root / "state.json", self.state)
        self.store.event("state", **changes)

    def paused(self):
        return read(self.root / "control.json")["paused"] or self.mission_expired()

    def mission_expired(self):
        deadline = read(self.root / "control.json").get("expires_at")
        return deadline is not None and time.time() >= deadline

    def apply_lifeguard(self, value, **transition):
        validate_lifeguard(value)
        if value["rollback_release"] != self.state["stable"]:
            raise ValueError("Recovery release no longer matches stable")
        self.save(lifeguard=dict(value), **transition)
        self.failures = 0

    def parameter(self, name):
        guard = self.state.get("lifeguard")
        if guard is not None and name in guard:
            return guard[name]
        return self.config.get(name)

    def probe(self, release):
        spec = self.store.release(release)["probe"]
        base = spec["base_url"].rstrip("/")
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(
                base + "/v1/models", timeout=self.parameter("probe_timeout")
            ) as response:
                models = json.load(response)
            if spec["model"] not in [item["id"] for item in models["data"]]:
                return False
            req = urllib.request.Request(
                base + "/v1/completions",
                data=json.dumps(
                    {
                        "model": spec["model"],
                        "prompt": spec["prompt"],
                        "max_tokens": 1,
                        "temperature": 0,
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            with opener.open(req, timeout=self.parameter("probe_timeout")) as response:
                body = json.load(response)
            return (
                bool(body["choices"])
                and body.get("usage", {}).get("completion_tokens", 0) > 0
            )
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def seal(self, release, verify=False):
        folder = self.root / "releases" / release
        spec = self.store.release(release)
        entries = {}
        for relative in spec["seal"]:
            target = folder / relative
            if not target.exists():
                raise RuntimeError(f"Missing sealed artifact: {relative}")
            files = [target]
            if target.is_dir():
                files.extend(sorted(target.rglob("*")))
            for path in files:
                if "__pycache__" in path.parts or path.suffix == ".pyc":
                    continue
                if path.is_symlink():
                    if not path.resolve().is_relative_to(folder):
                        raise RuntimeError(f"Sealed symlink escapes capsule: {path}")
                    entries[str(path.relative_to(folder))] = {
                        "symlink": os.readlink(path)
                    }
                elif path.is_file():
                    h = hashlib.sha256()
                    with open(path, "rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            h.update(block)
                    entries[str(path.relative_to(folder))] = h.hexdigest()
        value = {"manifest_hash": digest(spec), "files": entries}
        path = folder / "seal.json"
        if verify:
            if read(path) != value:
                raise RuntimeError(f"Sealed release changed: {release}")
        else:
            if path.exists() and read(path) != value:
                raise RuntimeError("Cannot reseal a changed release; use a new ID")
            atomic(path, value)

    def job(self, argv, spec, log):
        self.backend.start(
            "job",
            argv,
            spec["cwd"],
            spec["env"],
            log,
            job=True,
            timeout=spec["job_timeout"],
        )
        deadline = time.monotonic() + spec["job_timeout"]
        try:
            while time.monotonic() < deadline:
                result = self.backend.job_result("job")
                if result is not None:
                    if not result:
                        raise RuntimeError(f"Job failed; see {log}")
                    return
                if self.paused():
                    raise RuntimeError("Experiment paused")
                time.sleep(self.config["poll_seconds"])
            raise TimeoutError(f"Job timed out; see {log}")
        finally:
            self.backend.stop("job")

    def start_release(self, release, log, restoring=False):
        guard = self.state.get("lifeguard")
        validate_lifeguard(guard)
        if restoring and release != guard["rollback_release"]:
            raise RuntimeError("Recovery target is not authorized by lifeguard")
        self.seal(release, verify=True)
        spec = self.store.release(release)
        # All stop/start side effects have a durable phase before reaching here.
        self.backend.stop("server")
        endpoint = urllib.parse.urlparse(spec["probe"]["base_url"])
        occupied = False
        try:
            with socket.create_connection(
                (endpoint.hostname, endpoint.port or 80), timeout=1
            ):
                occupied = True
        except OSError:
            pass  # No listener is expected after stopping our unit.
        if occupied:
            raise RuntimeError("Endpoint is already served by an unowned process")
        self.backend.start("server", spec["argv"], spec["cwd"], spec["env"], log)
        startup_timeout = guard["recovery_timeout" if restoring else "startup_timeout"]
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if not restoring and self.paused():
                raise RuntimeError("Experiment paused")
            if not self.backend.running("server"):
                raise RuntimeError("Server exited during startup")
            if self.probe(release):
                self.save(current=release)
                return
            time.sleep(self.config["poll_seconds"])
        raise TimeoutError("Server startup timed out")

    def capture(self, reason):
        name = (
            self.state.get("active") or self.state.get("last_experiment") or "watchdog"
        )
        folder = self.root / "experiments" / name
        folder.mkdir(exist_ok=True)
        atomic(
            folder / f"incident-{time.time_ns()}.json",
            {
                "reason": reason,
                "state": self.state,
                "server": self.backend.properties("server"),
            },
        )
        self.store.event("incident", reason=reason, experiment=name)

    def rollback(self, reason):
        self.capture(reason)
        guard = self.state.get("lifeguard")
        try:
            validate_lifeguard(guard)
            if guard["rollback_release"] != self.state["stable"]:
                raise ValueError("Recovery release no longer matches stable")
        except ValueError as error:
            self.capture(str(error))
            self.save(phase="blocked")
            return False
        self.save(phase="restoring")
        self.cancel_resume()
        self.backend.stop("job")
        self.backend.stop("server")
        self.save(current=None)
        stable = self.state["stable"]
        if not stable:
            self.save(phase="blocked")
            return False
        try:
            self.start_release(stable, self.root / "recovery.log", restoring=True)
        except Exception as error:
            self.capture(str(error))
            self.backend.stop("server")
            self.save(current=None, phase="blocked")
            return False
        self.save(phase="idle")
        return True

    def finish(self, request, status, error=None):
        folder = self.root / "experiments" / request["id"]
        folder.mkdir(exist_ok=True)
        result = {
            "id": request["id"],
            "status": status,
            "error": error,
            "request_hash": digest(request),
            "current": self.state["current"],
            "stable": self.state["stable"],
            "lifeguard": self.state.get("lifeguard"),
            "finished_at": time.time(),
        }
        atomic(folder / "result.json", result)
        self.ensure_outbox(request["id"])
        self.save(active=None, last_experiment=request["id"])
        self.store.event("finished", **result)

    def ensure_outbox(self, name):
        path = self.root / "outbox" / f"{name}.json"
        if not path.exists():
            atomic(path, {"experiment": name, "status": "pending", "attempts": 0})

    def cancel_resume(self):
        for path in (self.root / "outbox").glob("*.json"):
            item = read(path)
            if item["status"] == "delivering":
                self.backend.stop("resume")
                item["status"] = "uncertain"
                atomic(path, item)

    def execute(self, request):
        name = self.store.validate_request(request)
        folder = self.root / "experiments" / name
        folder.mkdir(exist_ok=True)
        if request["kind"] == "rollback":
            self.apply_lifeguard(request["lifeguard"], active=name, phase="restoring")
            ok = self.rollback(request["reason"])
            self.finish(request, "rolled_back" if ok else "blocked")
            return
        if request["kind"] == "promote":
            if self.state.get("promotion_id") == name:
                self.finish(request, "ready")
                return
            source = read(
                self.root / "requests" / f"{identifier(request['experiment'])}.json"
            )
            result = read(self.root / "experiments" / source["id"] / "result.json")
            if (
                result["status"] != "ready"
                or self.state["current"] != source["candidate"]
                or self.state["phase"] != "ready"
                or not self.probe(source["candidate"])
            ):
                self.finish(
                    request, "rejected", "Candidate is not eligible for promotion"
                )
                return
            self.seal(source["candidate"], verify=True)
            self.save(
                previous_stable=self.state["stable"],
                stable=source["candidate"],
                phase="idle",
                restart_count=0,
                promotion_id=name,
                lifeguard=dict(
                    self.state["lifeguard"], rollback_release=source["candidate"]
                ),
            )
            self.finish(request, "ready")
            return
        if (
            request["baseline"] != self.state["stable"]
            or self.state["current"] != self.state["stable"]
            or self.state["phase"] not in {"idle"}
        ):
            self.finish(
                request,
                "rejected",
                "Baseline changed or an unpromoted candidate is active",
            )
            return
        candidate = request["candidate"]
        if self.state["stable"] and (
            self.store.release(candidate)["probe"]
            != self.store.release(self.state["stable"])["probe"]
        ):
            self.finish(
                request,
                "rejected",
                "Candidate must use the baseline endpoint and probe",
            )
            return
        if candidate in {self.state["stable"], self.state["previous_stable"]}:
            self.finish(
                request, "rejected", "A retained stable release cannot be a candidate"
            )
            return
        atomic(folder / "proposal.json", request)
        (self.root / "notes/checkpoints" / f"{name}.md").write_text(
            request["checkpoint"] + "\n"
        )
        self.save(active=name, phase="building")
        switched = False
        try:
            spec = self.store.release(candidate)
            if (self.root / "releases" / candidate / "seal.json").exists():
                raise RuntimeError(
                    "Use a fresh candidate ID; sealed releases cannot be rebuilt"
                )
            for index, argv in enumerate(spec["build"]):
                self.job(argv, spec, folder / f"build-{index}.log")
            self.seal(candidate)
            if self.state["stable"]:
                self.seal(self.state["stable"], verify=True)
            if self.paused():
                raise RuntimeError("Experiment paused or mission deadline reached")
            self.apply_lifeguard(request["lifeguard"], phase="switching")
            switched = True
            self.start_release(candidate, folder / "server.log")
            self.save(phase="checking")
            for index, argv in enumerate(spec["checks"]):
                self.job(argv, spec, folder / f"check-{index}.log")
            self.save(phase="observing")
            deadline = time.monotonic() + request["observation_seconds"]
            while time.monotonic() < deadline:
                if (
                    self.paused()
                    or not self.backend.running("server")
                    or not self.probe(candidate)
                ):
                    raise RuntimeError("Candidate failed observation or was paused")
                time.sleep(self.config["poll_seconds"])
            self.seal(candidate, verify=True)
            self.save(phase="ready")
            self.finish(request, "ready")
        except Exception as error:
            if switched:
                ok = self.rollback(str(error))
                status = "rolled_back" if ok else "blocked"
            else:
                self.capture(str(error))
                self.save(phase="idle")
                status = "build_failed"
            self.finish(request, status, str(error))

    def recover(self):
        # Resume interrupted side effects by restoring a sealed baseline, never by
        # repeating a build or assuming a partially completed check passed.
        if self.state["active"] or self.state["phase"] == "restoring":
            active = self.state["active"]
            result = self.root / "experiments" / str(active) / "result.json"
            if active and result.exists():
                self.ensure_outbox(active)
                self.save(active=None, last_experiment=active)
            elif self.state["phase"] == "building" and (
                self.state["current"] == self.state["stable"]
                and (
                    self.state["stable"] is None
                    or (
                        self.backend.running("server")
                        and self.probe(self.state["stable"])
                    )
                )
            ):
                self.backend.stop("job")
                self.save(phase="idle")
                self.finish(
                    read(self.root / "requests" / f"{active}.json"),
                    "build_failed",
                    "Controller interrupted during build",
                )
            else:
                ok = self.rollback("Controller restarted during a transaction")
                if active:
                    request = read(self.root / "requests" / f"{active}.json")
                    self.finish(
                        request,
                        "rolled_back" if ok else "blocked",
                        "Controller interrupted",
                    )
        for result in (self.root / "experiments").glob("*/result.json"):
            self.ensure_outbox(result.parent.name)
        self.cancel_resume()

    def resume_one(self):
        resume = self.config.get("resume")
        if (
            not resume
            or read(self.root / "control.json").get("origin_session", False)
            or self.paused()
            or not self.state["current"]
            or self.state["phase"] not in {"idle", "ready"}
        ):
            return
        pending = [
            p
            for p in (self.root / "outbox").glob("*.json")
            if read(p)["status"] == "pending"
        ]
        pending.sort(
            key=lambda p: read(self.root / "experiments" / p.stem / "result.json")[
                "finished_at"
            ],
            reverse=True,
        )
        # A recovery turn reads current state. Old undelivered notifications are
        # retained as facts, but must not start obsolete recovery turns.
        for obsolete in pending[1:]:
            item = read(obsolete)
            item.update(status="superseded", superseded_by=pending[0].stem)
            atomic(obsolete, item)
        for path in pending[:1]:
            item = read(path)
            if (
                item["status"] != "pending"
                or item["attempts"] >= resume["max_attempts"]
            ):
                continue
            folder = self.root / "experiments" / item["experiment"]
            prompt = (
                f"恢复事件 {item['experiment']}。使用 self-evolve-inference skill，"
                f"只读取 {folder / 'result.json'}、{self.root / 'state.json'}、"
                f"{self.root / 'notes/current.md'}。"
                "本次仅解释结果，不提交新实验，不修改推理服务。"
                "只更新 current.md 和 trajectory.md，保留历史。"
                "按结果文件区分 CPU 故障注入和真实 GPU 服务验证，"
                "不得把启动或探活成功解释成性能提升。"
                "不要查找历史会话或其他项目文件，完成后立即给出简短中文结论。"
            )
            argv = [arg.replace("{prompt}", prompt) for arg in resume["argv"]]
            item.update(
                status="delivering",
                attempts=item["attempts"] + 1,
                started_at=time.time(),
            )
            atomic(path, item)
            self.backend.start(
                "resume",
                argv,
                resume["cwd"],
                {
                    key: os.environ[key]
                    for key in ("PATH", "EVOLUTION_LOCAL_API_KEY")
                    if key in os.environ
                },
                folder / "resume.log",
                job=True,
                timeout=resume["timeout"],
            )
            return

    def poll_resume(self):
        delivering = [
            (p, read(p))
            for p in (self.root / "outbox").glob("*.json")
            if read(p)["status"] == "delivering"
        ]
        for path, item in delivering:
            result = self.backend.job_result("resume")
            if result is None:
                if time.time() - item["started_at"] < self.config["resume"]["timeout"]:
                    return True
                result = False
            self.backend.stop("resume")
            # A failed DSH process may already have submitted the prompt.
            item["status"] = "delivered" if result else "uncertain"
            atomic(path, item)
        return False

    def tick(self):
        if self.state["phase"] == "blocked":
            return
        current = self.state["current"]
        if current:
            try:
                validate_lifeguard(self.state.get("lifeguard"))
                if self.state["lifeguard"]["rollback_release"] != self.state["stable"]:
                    raise ValueError("Recovery release no longer matches stable")
            except ValueError as error:
                self.capture(str(error))
                self.save(phase="blocked")
                return
            healthy = self.backend.running("server") and self.probe(current)
            self.failures = 0 if healthy else self.failures + 1
            if self.failures >= self.parameter("failure_threshold"):
                if current == self.state["stable"]:
                    if self.state["restart_count"] >= self.parameter("stable_restarts"):
                        self.capture("Stable restart budget exhausted")
                        self.cancel_resume()
                        self.backend.stop("server")
                        self.save(current=None, phase="blocked")
                        return
                    self.save(restart_count=self.state["restart_count"] + 1)
                guard = self.state["lifeguard"]
                name = "runtime-" + str(time.time_ns())
                request = {
                    "id": name,
                    "kind": "rollback",
                    "reason": f"Runtime health failure: {current}",
                    "lifeguard": dict(guard),
                }
                self.store.submit(request)
                self.save(active=name, phase="restoring")
                ok = self.rollback(request["reason"])
                self.finish(
                    request, "rolled_back" if ok else "blocked", request["reason"]
                )
                self.failures = 0
                return
        if self.paused():
            self.cancel_resume()
            return
        if self.poll_resume():
            return
        for path in sorted(
            (self.root / "requests").glob("*.json"), key=lambda p: p.stat().st_mtime_ns
        ):
            request = read(path)
            if not (self.root / "experiments" / request["id"] / "result.json").exists():
                try:
                    self.execute(request)
                except Exception as error:
                    self.capture(str(error))
                    self.finish(request, "rejected", str(error))
                return
        self.resume_one()

    def expire_mission(self):
        if (
            self.state["current"] != self.state["stable"]
            and self.state["phase"] == "ready"
        ):
            request = {
                "id": "expiry-" + str(time.time_ns()),
                "kind": "rollback",
                "reason": "Mission deadline reached with an unpromoted candidate",
                "lifeguard": dict(self.state["lifeguard"]),
            }
            self.store.submit(request)
            self.execute(request)
        self.cancel_resume()
        with lock(self.root / ".submit.lock"):
            for path in (self.root / "requests").glob("*.json"):
                if not (self.root / "experiments" / path.stem / "result.json").exists():
                    self.finish(read(path), "rejected", "Mission deadline reached")
            control = read(self.root / "control.json")
            atomic(self.root / "control.json", dict(control, paused=True))
        self.store.event("lifeguard_expired", server_preserved=True)

    def run(self, once=False):
        if not once and not read(self.root / "control.json").get("expires_at"):
            raise RuntimeError(
                "Start through lifecycle.py with a finite mission budget"
            )
        with lock(self.root / ".daemon.lock"):
            while True:
                try:
                    with lock(self.root / ".transition.lock"):
                        self.recover()
                    break
                except BlockingIOError:
                    time.sleep(self.config["poll_seconds"])
            while True:
                try:
                    with lock(self.root / ".transition.lock"):
                        if self.mission_expired():
                            self.expire_mission()
                            return
                        self.tick()
                except BlockingIOError:
                    pass  # The lifecycle helper is stopping this idle daemon.
                if once:
                    return
                time.sleep(self.config["poll_seconds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    sub = parser.add_subparsers(dest="action", required=True)
    init = sub.add_parser("init")
    init.add_argument("--config", type=Path)
    for name in ("register", "submit"):
        sub.add_parser(name).add_argument("file", type=Path)
    waiter = sub.add_parser("wait")
    waiter.add_argument("id")
    waiter.add_argument("--timeout", type=float, required=True)
    run = sub.add_parser("run")
    run.add_argument("--once", action="store_true")
    sub.add_parser("status")
    logs = sub.add_parser("logs")
    logs.add_argument("--lines", type=int, default=30)
    for name in ("pause", "continue"):
        sub.add_parser(name)
    args = parser.parse_args()
    store = Store(args.root)
    if args.action == "init":
        store.initialize(
            read(args.config)
            if args.config
            else {
                "poll_seconds": 10,
                "probe_timeout": 5,
                "failure_threshold": 3,
                "stable_restarts": 1,
            }
        )
        output = {"root": str(store.root)}
    elif args.action == "register":
        output = {"release": store.register(read(args.file))}
    elif args.action == "submit":
        output = store.submit(read(args.file))
    elif args.action == "wait":
        name = identifier(args.id)
        bounded(args.timeout, "wait timeout")
        store_request = read(store.root / "requests" / f"{name}.json")
        deadline = time.monotonic() + args.timeout
        path = store.root / "experiments" / name / "result.json"
        while not path.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Wait budget exceeded; request is still durable, do not resubmit"
                )
            time.sleep(1)
        output = read(path)
        if output["request_hash"] != digest(store_request):
            raise RuntimeError("Result does not match submitted request")
    elif args.action == "status":
        output = {
            "state": read(store.root / "state.json"),
            "control": read(store.root / "control.json"),
            "server": Systemd(store.root).properties("server"),
        }
    elif args.action == "logs":
        from collections import deque

        with open(store.root / "events.jsonl") as stream:
            print("".join(deque(stream, maxlen=max(1, args.lines))), end="")
        return
    elif args.action in {"pause", "continue"}:
        output = read(store.root / "control.json")
        output["paused"] = args.action == "pause"
        atomic(store.root / "control.json", output)
    else:
        Controller(store.root).run(args.once)
        return
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)
