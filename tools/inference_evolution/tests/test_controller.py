# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test transaction outcomes without importing vLLM or allocating a GPU.

The module owns durable experiment transitions. Inputs are release/request JSON;
outputs are state, results and owned service actions. A fake backend catches
rollback and replay failures cheaply; verify.py covers actual systemd ownership.
"""

import copy
import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / "controller.py"
module_spec = importlib.util.spec_from_file_location("evolution", SOURCE)
evolution = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(evolution)


class Backend:
    def __init__(self):
        self.starts = []
        self.live = {}

    def start(self, slot, argv, cwd, env, log, job=False, timeout=None):
        self.starts.append((slot, argv))
        self.live[slot] = "fail" not in argv

    def stop(self, slot):
        self.live[slot] = False

    def running(self, slot):
        return self.live.get(slot, False)

    def job_result(self, slot):
        return self.live.get(slot, False)

    def properties(self, slot):
        return {"ActiveState": "active" if self.running(slot) else "inactive"}


@pytest.fixture
def setup(tmp_path):
    store = evolution.Store(tmp_path)
    store.initialize(
        {
            "poll_seconds": 0.01,
            "probe_timeout": 0.01,
            "failure_threshold": 1,
            "stable_restarts": 1,
        }
    )
    backend = Backend()
    controller = evolution.Controller(tmp_path, backend)
    controller.probe = lambda release: backend.running("server")
    return store, controller, backend


def release(store, name, fail=False, build=None):
    folder = store.root / "releases" / name / "source"
    folder.mkdir(parents=True)
    (folder / "artifact").write_text(name)
    spec = {
        "id": name,
        "cwd": str(folder),
        "argv": ["/bin/true"] + (["fail"] if fail else []),
        "env": {},
        "probe": {"base_url": "http://127.0.0.1:1", "model": "fixture", "prompt": "hi"},
        "build": build or [],
        "checks": [["/bin/true"]],
        "seal": ["source"],
        "startup_timeout": 0.05,
        "job_timeout": 0.05,
    }
    store.register(spec)
    return spec


def request(name, candidate, baseline=None):
    return {
        "id": name,
        "kind": "experiment",
        "baseline": baseline,
        "candidate": candidate,
        "hypothesis": "fixture",
        "checkpoint": "resume here",
        "observation_seconds": 0,
        "lifeguard": {
            "startup_timeout": 0.05,
            "recovery_timeout": 0.2,
            "rollback_release": baseline,
            "probe_timeout": 0.01,
            "failure_threshold": 1,
            "stable_restarts": 1,
            "restart_policy": "restore_stable_then_block",
        },
    }


def run(store, controller, req):
    store.submit(req)
    controller.execute(req)
    return evolution.read(store.root / "experiments" / req["id"] / "result.json")


def baseline(store, controller):
    release(store, "stable")
    assert run(store, controller, request("bootstrap", "stable"))["status"] == "ready"
    run(
        store,
        controller,
        {
            "id": "accept",
            "kind": "promote",
            "experiment": "bootstrap",
            "decision": "fixture passed",
        },
    )


def test_failed_candidate_restores_sealed_baseline(setup):
    store, controller, backend = setup
    baseline(store, controller)
    release(store, "broken", fail=True)
    result = run(store, controller, request("attempt", "broken", "stable"))
    assert result["status"] == "rolled_back"
    assert result["current"] == result["stable"] == "stable"
    assert backend.running("server")


@pytest.fixture
def adoption(monkeypatch):
    monkeypatch.setitem(sys.modules, "controller", evolution)
    spec = importlib.util.spec_from_file_location(
        "adoption", SOURCE.with_name("adopt.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("interrupted", [False, True])
def test_adoption_failure_restores_original_without_marking_it_sealed(
    setup, adoption, monkeypatch, interrupted
):
    store, controller, backend = setup
    release(store, "capsule", fail=True)
    controller.seal("capsule")
    evolution.atomic(store.root / "installation.json", {"unit": "fixture.service"})
    evolution.atomic(
        store.root / "adoption.json",
        {
            "stage": "starting_capsule" if interrupted else "prepared",
            "lifeguard": request("a", "capsule")["lifeguard"],
            "release": "capsule",
            "processes": [{"pid": 1}],
            "original": {
                "argv": ["/bin/true", "original"],
                "cwd": str(store.root),
                "env": {},
            },
        },
    )
    monkeypatch.setattr(adoption, "same_process", lambda item: True)
    monkeypatch.setattr(adoption, "tree", lambda pid: [{"pid": pid}])
    stopped = []
    monkeypatch.setattr(adoption, "stop_original", stopped.append)
    monkeypatch.setattr(adoption.subprocess, "run", lambda *a, **kw: None)
    adoption.run(controller, observation=0)
    assert stopped
    assert backend.starts[-1] == ("server", ["/bin/true", "original"])
    assert backend.running("server")
    assert controller.state["stable"] is None
    assert controller.state["phase"] == "blocked"
    assert evolution.read(store.root / "adoption.json")["stage"] == "fallback_running"


def test_adoption_commits_only_after_generation_and_checks(
    setup, adoption, monkeypatch
):
    store, controller, backend = setup
    release(store, "capsule")
    controller.seal("capsule")
    evolution.atomic(store.root / "installation.json", {"unit": "fixture.service"})
    evolution.atomic(
        store.root / "adoption.json",
        {
            "stage": "prepared",
            "lifeguard": request("a", "capsule")["lifeguard"],
            "release": "capsule",
            "processes": [{"pid": 1}],
        },
    )
    monkeypatch.setattr(adoption, "same_process", lambda item: True)
    monkeypatch.setattr(adoption, "tree", lambda pid: [{"pid": pid}])
    monkeypatch.setattr(adoption, "stop_original", lambda items: None)
    monkeypatch.setattr(adoption.subprocess, "run", lambda *a, **kw: None)
    adoption.run(controller, observation=0)
    assert controller.state["stable"] == controller.state["current"] == "capsule"
    assert controller.state["phase"] == "idle"
    assert any(slot == "job" for slot, _ in backend.starts)
    assert evolution.read(store.root / "adoption.json")["stage"] == "adopted"


def test_adoption_never_signals_reused_pid(adoption, monkeypatch):
    old = {"pid": 1, "start": "10", "boot": "boot"}
    monkeypatch.setattr(adoption, "identity", lambda pid: dict(old, start="11"))
    monkeypatch.setattr(adoption, "pidfd_open", lambda pid: pytest.fail("PID reused"))
    adoption.send(old, 15)


def test_adoption_stops_only_captured_cpu_process(adoption):
    target = subprocess.Popen(["/bin/sleep", "30"])
    unrelated = subprocess.Popen(["/bin/sleep", "30"])
    try:
        items = adoption.tree(target.pid)
        adoption.stop_original(items, timeout=2)
        assert target.wait(timeout=2) == -15
        assert unrelated.poll() is None
    finally:
        for process in (target, unrelated):
            if process.poll() is None:
                process.kill()
            process.wait()


def test_build_failure_does_not_stop_working_service(setup):
    store, controller, backend = setup
    baseline(store, controller)
    starts = len([s for s in backend.starts if s[0] == "server"])
    release(store, "badbuild", build=[["/bin/false", "fail"]])
    assert (
        run(store, controller, request("build", "badbuild", "stable"))["status"]
        == "build_failed"
    )
    assert len([s for s in backend.starts if s[0] == "server"]) == starts


def test_duplicate_submission_is_idempotent_but_changed_content_is_rejected(setup):
    store, controller, backend = setup
    release(store, "candidate")
    req = request("same", "candidate")
    store.submit(req)
    store.submit(req)
    controller.tick()
    count = len(backend.starts)
    controller.tick()
    assert len(backend.starts) == count
    changed = copy.deepcopy(req)
    changed["hypothesis"] = "different"
    with pytest.raises(ValueError, match="already used"):
        store.submit(changed)


def test_restart_request_requires_explicit_lifeguard_parameters(setup):
    store, controller, backend = setup
    release(store, "candidate")
    req = request("guarded", "candidate")
    req.pop("lifeguard")
    with pytest.raises(ValueError, match="Invalid request fields"):
        store.submit(req)


def test_manual_rollback_requires_a_matching_stable_release(setup):
    store, controller, backend = setup
    baseline(store, controller)
    request_data = {
        "id": "manual-rollback",
        "kind": "rollback",
        "reason": "manual validation",
        "lifeguard": dict(request("x", "candidate", "other")["lifeguard"]),
    }
    store.submit(request_data)
    controller.tick()
    result = evolution.read(store.root / "experiments/manual-rollback/result.json")
    assert result["status"] == "rejected"
    assert controller.state["stable"] == "stable"
    assert backend.running("server")


def test_lifeguard_parameters_are_persisted_before_candidate_switch(setup):
    store, controller, backend = setup
    release(store, "candidate")
    req = request("guarded", "candidate")
    store.submit(req)
    stop = backend.stop

    def check_before_stop(slot):
        if slot == "server":
            state = evolution.read(store.root / "state.json")
            assert state["lifeguard"] == req["lifeguard"]
            assert state["phase"] == "switching"
        stop(slot)

    backend.stop = check_before_stop
    controller.execute(req)
    state = evolution.read(store.root / "state.json")
    assert state["lifeguard"] == req["lifeguard"]


@pytest.fixture
def lifecycle(monkeypatch):
    monkeypatch.setitem(sys.modules, "controller", evolution)
    spec = importlib.util.spec_from_file_location(
        "lifecycle", SOURCE.with_name("lifecycle.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lifecycle_stop_preserves_server_and_disables_autostart(
    setup, lifecycle, monkeypatch
):
    store, controller, backend = setup
    baseline(store, controller)
    evolution.atomic(
        store.root / "installation.json",
        {"unit": "fixture.service", "package": "/fixture"},
    )
    monkeypatch.setattr(lifecycle, "Controller", lambda root: controller)
    calls = []

    def output(argv, **kwargs):
        if "ExecStart" in argv:
            return f"/fixture/controller.py --root {store.root} run"
        if "ActiveState" in argv:
            return "active"
        return "ActiveState=inactive\nSubState=dead\n"

    monkeypatch.setattr(lifecycle.subprocess, "check_output", output)
    monkeypatch.setattr(
        lifecycle.subprocess, "run", lambda argv, **kw: calls.append(argv)
    )
    result = lifecycle.run(store.root, "stop")
    assert result["ActiveState"] == "inactive"
    assert calls == [
        ["systemctl", "--user", "disable", "fixture.service"],
        ["systemctl", "--user", "stop", "fixture.service"],
    ]
    assert backend.running("server")
    assert evolution.read(store.root / "control.json")["paused"]
    assert (
        evolution.read(store.root / "outbox/accept.json")["status"] == "origin_session"
    )


@pytest.mark.parametrize("pending", [False, True])
def test_lifecycle_refuses_unsettled_work(setup, lifecycle, pending):
    store, controller, backend = setup
    if pending:
        release(store, "candidate")
        store.submit(request("queued", "candidate"))
    else:
        controller.save(active="in-flight", phase="building")
    with pytest.raises(RuntimeError):
        lifecycle.settled(store.root, controller.state, backend)


def test_lifecycle_start_without_parameters_has_no_side_effects(setup, lifecycle):
    store, controller, backend = setup
    with pytest.raises(ValueError, match="Lifeguard parameters"):
        lifecycle.run(store.root, "start")
    assert not backend.starts
    assert not (store.root / "installation.json").exists()


def test_lifecycle_waits_for_health_poll_without_interrupting_it(setup, lifecycle):
    store, controller, backend = setup
    acquired = threading.Event()
    release_lock = threading.Event()

    def health_poll():
        with evolution.lock(store.root / ".transition.lock"):
            acquired.set()
            release_lock.wait(timeout=2)

    worker = threading.Thread(target=health_poll)
    worker.start()
    try:
        assert acquired.wait(timeout=2)
        with (
            pytest.raises(RuntimeError, match="Supervisor busy"),
            lifecycle.transition_lock(store.root, timeout=0),
        ):
            pytest.fail("The in-flight transaction must keep its lock")
        timer = threading.Timer(0.05, release_lock.set)
        timer.start()
        with lifecycle.transition_lock(store.root):
            assert release_lock.is_set()
        timer.join()
    finally:
        release_lock.set()
        worker.join(timeout=2)


def test_expired_mission_restores_candidate_and_rejects_queued_work(setup):
    store, controller, backend = setup
    baseline(store, controller)
    release(store, "candidate")
    run(store, controller, request("candidate-run", "candidate", "stable"))
    release(store, "queued")
    store.submit(request("queued-run", "queued", "stable"))
    evolution.atomic(store.root / "control.json", {"paused": False, "expires_at": 1})
    controller.run()
    assert controller.state["current"] == "stable"
    assert backend.running("server")
    assert evolution.read(store.root / "control.json")["paused"]
    assert (
        evolution.read(store.root / "experiments/queued-run/result.json")["status"]
        == "rejected"
    )


def test_missing_restart_parameters_never_stops_a_running_server(setup):
    store, controller, backend = setup
    baseline(store, controller)
    controller.save(lifeguard=None, restart_count=100)
    controller.probe = lambda _: False
    starts = len(backend.starts)
    controller.tick()
    assert controller.state["phase"] == "blocked"
    assert backend.running("server")
    assert len(backend.starts) == starts


def test_mission_expiry_during_build_keeps_stable_server(setup):
    store, controller, backend = setup
    baseline(store, controller)
    release(store, "candidate", build=[["/bin/true"]])
    starts = len(backend.starts)

    def expire_after_build(*args):
        evolution.atomic(
            store.root / "control.json", {"paused": False, "expires_at": 1}
        )

    controller.job = expire_after_build
    result = run(store, controller, request("expires-in-build", "candidate", "stable"))
    assert result["status"] == "build_failed"
    assert controller.state["current"] == "stable"
    assert len(backend.starts) == starts
    assert backend.running("server")


@pytest.mark.parametrize("complete", [True, False])
def test_wait_reads_result_or_times_out_without_resubmission(setup, complete):
    store, controller, backend = setup
    release(store, "candidate")
    req = request("awaited", "candidate")
    store.submit(req)
    if complete:
        controller.execute(req)
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE),
            "--root",
            str(store.root),
            "wait",
            "awaited",
            "--timeout",
            "0.01",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == (0 if complete else 1)
    if complete:
        assert json.loads(result.stdout)["request_hash"] == evolution.digest(req)
    else:
        assert "do not resubmit" in result.stderr
    assert [p.stem for p in (store.root / "requests").glob("*.json")] == ["awaited"]


def test_controller_restart_restores_instead_of_replaying_candidate(setup):
    store, controller, backend = setup
    baseline(store, controller)
    release(store, "candidate")
    req = request("interrupted", "candidate", "stable")
    store.submit(req)
    controller.save(active=req["id"], phase="switching")
    restored = evolution.Controller(store.root, backend)
    restored.probe = controller.probe
    restored.recover()
    assert restored.state["current"] == "stable"
    assert (
        evolution.read(store.root / "experiments/interrupted/result.json")["status"]
        == "rolled_back"
    )


def test_corrupt_baseline_blocks_instead_of_claiming_recovery(setup):
    store, controller, backend = setup
    baseline(store, controller)
    (store.root / "releases/stable/source/artifact").write_text("changed")
    assert controller.rollback("fault") is False
    assert controller.state["phase"] == "blocked"
    assert not backend.running("server")


def test_paused_controller_does_not_consume_experiment(setup):
    store, controller, backend = setup
    release(store, "candidate")
    store.submit(request("wait", "candidate"))
    evolution.atomic(store.root / "control.json", {"paused": True})
    controller.tick()
    assert not backend.starts


def test_stable_crash_loop_has_a_persistent_budget(setup):
    store, controller, backend = setup
    baseline(store, controller)
    backend.stop("server")
    controller.tick()
    assert controller.state["restart_count"] == 1
    backend.stop("server")
    controller.tick()
    assert controller.state["phase"] == "blocked"


def test_stale_baseline_cannot_switch_service(setup):
    store, controller, backend = setup
    baseline(store, controller)
    release(store, "candidate")
    starts = len(backend.starts)
    assert run(store, controller, request("stale", "candidate"))["status"] == "rejected"
    assert len(backend.starts) == starts


def test_recovery_marks_uncertain_resume_without_replaying(setup):
    store, controller, backend = setup
    evolution.atomic(
        store.root / "outbox/reply.json",
        {
            "experiment": "reply",
            "status": "delivering",
            "attempts": 1,
        },
    )
    controller.recover()
    assert evolution.read(store.root / "outbox/reply.json")["status"] == "uncertain"
    assert not backend.starts


def test_interrupted_build_keeps_healthy_baseline_running(setup):
    store, controller, backend = setup
    baseline(store, controller)
    release(store, "candidate")
    store.submit(request("interrupted-build", "candidate", "stable"))
    count = len(backend.starts)
    controller.save(active="interrupted-build", phase="building")
    restored = evolution.Controller(store.root, backend)
    restored.probe = controller.probe
    restored.recover()
    assert len(backend.starts) == count
    assert backend.running("server")
    result = evolution.read(store.root / "experiments/interrupted-build/result.json")
    assert result["status"] == "build_failed"


def test_result_commit_survives_crash_before_active_marker_clears(setup):
    store, controller, backend = setup
    release(store, "candidate")
    run(store, controller, request("completed", "candidate"))
    controller.save(active="completed")
    count = len(backend.starts)
    controller.recover()
    assert controller.state["phase"] == "ready"
    assert controller.state["active"] is None
    assert len(backend.starts) == count


def test_promotion_commit_survives_crash_before_result(setup):
    store, controller, backend = setup
    baseline(store, controller)
    (store.root / "experiments/accept/result.json").unlink()
    controller.tick()
    assert (
        evolution.read(store.root / "experiments/accept/result.json")["status"]
        == "ready"
    )
    assert controller.state["previous_stable"] is None


def test_seal_rejects_external_mutable_symlink(setup, tmp_path):
    store, controller, backend = setup
    release(store, "candidate")
    (store.root / "releases/candidate/source/link").symlink_to("/etc/hostname")
    with pytest.raises(RuntimeError, match="escapes"):
        controller.seal("candidate")


def test_late_candidate_crash_has_a_durable_recovery_notification(setup):
    store, controller, backend = setup
    baseline(store, controller)
    release(store, "candidate")
    run(store, controller, request("late-crash", "candidate", "stable"))
    backend.stop("server")
    controller.tick()
    assert controller.state["current"] == "stable"
    latest = controller.state["last_experiment"]
    assert latest.startswith("runtime-")
    assert (
        evolution.read(store.root / "experiments" / latest / "result.json")["status"]
        == "rolled_back"
    )
    assert (
        evolution.read(store.root / "outbox" / f"{latest}.json")["status"] == "pending"
    )


def test_resume_delivers_latest_result_and_supersedes_old_pending_events(setup):
    store, controller, backend = setup
    baseline(store, controller)
    controller.config["resume"] = {
        "argv": ["/bin/true", "{prompt}"],
        "cwd": str(store.root),
        "timeout": 10,
        "max_attempts": 1,
    }
    controller.resume_one()
    assert (
        evolution.read(store.root / "outbox/bootstrap.json")["status"] == "superseded"
    )
    assert evolution.read(store.root / "outbox/accept.json")["status"] == "delivering"
    controller.poll_resume()
    starts = len(backend.starts)
    controller.resume_one()
    assert len(backend.starts) == starts
    assert evolution.read(store.root / "outbox/accept.json")["status"] == "delivered"


def test_resume_timeout_is_uncertain_and_never_automatically_retried(setup):
    store, controller, backend = setup
    baseline(store, controller)
    controller.config["resume"] = {
        "argv": ["/bin/true", "{prompt}"],
        "cwd": str(store.root),
        "timeout": 0.01,
        "max_attempts": 1,
    }
    controller.resume_one()
    item = evolution.read(store.root / "outbox/accept.json")
    item["started_at"] = 0
    evolution.atomic(store.root / "outbox/accept.json", item)
    backend.job_result = lambda slot: None
    controller.poll_resume()
    assert evolution.read(store.root / "outbox/accept.json")["status"] == "uncertain"
    starts = len(backend.starts)
    controller.resume_one()
    assert len(backend.starts) == starts


@pytest.mark.parametrize("completed", [True, False])
def test_dsh_bridge_requires_success_and_reuses_exact_session(tmp_path, completed):
    stub = tmp_path / "dsh.py"
    stub.write_text(
        "import json,sys\n"
        'if "--session-id" in sys.argv:\n'
        '    assert sys.argv[sys.argv.index("--session-id")+1] == "session-fixture"\n'
        'print(json.dumps({"type":"session","sessionId":"session-fixture"}))\n'
        'print(json.dumps({"type":"status","phase":"turn_end",'
        f'"reason":{{"kind":"{"completed" if completed else "error"}"}}}}))\n'
        'print(json.dumps({"type":"final","text":"fixture"}))\n'
    )
    config = tmp_path / "dsh.json"
    evolution.atomic(
        config, {"argv": [sys.executable, str(stub)], "cwd": str(tmp_path)}
    )
    argv = [
        sys.executable,
        str(SOURCE.with_name("dsh_bridge.py")),
        "--config",
        str(config),
        "task",
    ]
    first = subprocess.run(argv, capture_output=True, text=True)
    assert first.returncode == (0 if completed else 1), first.stderr
    if completed:
        assert (
            evolution.read(tmp_path / "dsh-session.json")["session_id"]
            == "session-fixture"
        )
        second = subprocess.run(argv, capture_output=True, text=True)
        assert second.returncode == 0, second.stderr
        assert len(list((tmp_path / "dsh-receipts").glob("*.json"))) == 2
    else:
        assert not (tmp_path / "dsh-session.json").exists()
        receipt = next((tmp_path / "dsh-receipts").glob("*.json"))
        assert json.loads(receipt.read_text())["status"] == "uncertain"
