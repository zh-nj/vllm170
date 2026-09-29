# Supervisor operations

Source: `tools/inference_evolution` in the vLLM workspace. The installer copies
the controller, DSH bridge and skill to a versioned directory outside the
checkout, creates a separate uv environment, and generates a user systemd unit.
The installer refuses existing state and unit names. It never adopts or stops
an existing production process. Systemd must support user services on Linux.

## Install and inspect

From the source checkout:

```sh
tools/inference_evolution/setup-supervisor.sh \
  --root /mnt/data/state/inference-evolution-validation \
  --instance validation --dsh /mnt/data/apps/deepseek-harness --start
```

Read `installation.json` under that state directory for `python`, `package`,
`unit` and `version`. Commands use those absolute paths:

```sh
"$EVOLUTION_PYTHON" "$EVOLUTION_PACKAGE/controller.py" --root "$EVOLUTION_ROOT" status
"$EVOLUTION_PYTHON" "$EVOLUTION_PACKAGE/controller.py" --root "$EVOLUTION_ROOT" logs --lines 30
"$EVOLUTION_PYTHON" "$EVOLUTION_PACKAGE/controller.py" --root "$EVOLUTION_ROOT" register release.json
"$EVOLUTION_PYTHON" "$EVOLUTION_PACKAGE/controller.py" --root "$EVOLUTION_ROOT" submit request.json
"$EVOLUTION_PYTHON" "$EVOLUTION_PACKAGE/controller.py" --root "$EVOLUTION_ROOT" pause
"$EVOLUTION_PYTHON" "$EVOLUTION_PACKAGE/controller.py" --root "$EVOLUTION_ROOT" continue
```

`pause` stops new experiments and recovery turns. An active experiment notices
pause at its next polling boundary and restores the stable service. The health
watchdog remains active while paused. `continue` clears pause; it does not clear
a `blocked` fault. Stop the supervisor and inspect evidence before repairing a
blocked installation. Stopping the supervisor alone does not stop its separately
owned server unit. `status` includes that server unit's properties.

## Release JSON

All fields below are required. Paths shown are examples to replace. Commands
are argv arrays, not shell strings; `/bin/bash -lc` should only appear when a
reviewed build recipe actually needs a shell. Build/check exit code zero means
the declared gate passed; correctness and performance gates must implement their
own assertions and produce evidence in the experiment logs.

```json
{
  "id": "candidate-0042",
  "cwd": "/mnt/data/state/inference-evolution-validation/releases/candidate-0042/source",
  "argv": ["/absolute/candidate/python", "-m", "vllm.entrypoints.openai.api_server", "--model", "/absolute/weights", "--port", "26661"],
  "env": {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONDONTWRITEBYTECODE": "1"},
  "probe": {"base_url": "http://127.0.0.1:26661", "model": "dsv41f", "prompt": "hello"},
  "build": [["/absolute/build-command"]],
  "checks": [["/absolute/correctness-and-performance-gate"]],
  "seal": ["source", "native-artifacts", "dependencies.lock"],
  "startup_timeout": 900,
  "job_timeout": 1800
}
```

Create the source directory before registration. `seal` paths must exist after
build. Sealing hashes declared files, excludes Python bytecode, and refuses
symlinks escaping the capsule. A seal only covers declared files: executable
paths, interpreter installations, driver libraries, model weights and any
undeclared dependencies remain operator-managed external inputs. Record their
versions, hashes and paths in `dependencies.lock`; preserve separate runnable
environments for baseline and candidate. A lockfile alone cannot restore a
deleted environment. Do not edit the stable environment, hardlink mutable
artifacts between releases, replace loaded `.so` files in place, or share an
editable vLLM installation between releases.

This version restores retained source/environment/artifacts; it does not rebuild
the baseline during an emergency. A missing/corrupt sealed baseline blocks
recovery. CPU fixture verification is not evidence that a real GPU release has
been captured. Before production adoption, verify a complete sealed baseline
with real token and model-specific correctness probes and confirm ownership of
the old process tree. The default local HTTP token probe does not support an
authenticated model server; add a protected credential reference before use in
that deployment.

## Request JSON

```json
{
  "id": "exp-0042",
  "kind": "experiment",
  "baseline": "release-0017",
  "candidate": "candidate-0042",
  "hypothesis": "Reduce the measured decode data transfer cost",
  "checkpoint": "Evidence, workload, expected improvement and next action",
  "observation_seconds": 60
}
```

Use `baseline: null` only to bootstrap an empty instance. Bootstrap failure has
no rollback target and becomes `blocked`. Every subsequent experiment must name
the current stable release and use the same probe identity. Only one unpromoted
candidate may be active. Every ID is immutable; identical resubmission is safe.

Promotion: `{"id":"promote-0042","kind":"promote","experiment":"exp-0042","decision":"Correctness and performance evidence with acceptance rationale"}`.
Rollback: `{"id":"rollback-0042","kind":"rollback","reason":"Measured regression"}`}.
Rollback restores the current stable capsule; `previous_stable` is retained for
manual diagnosis, not automatically selected. `ready` means gates passed and
candidate is serving, not automatic promotion. `rolled_back`, `build_failed`,
`rejected` and `blocked` explain failed outcomes in `result.json`.

## DSH continuation

Installation with `--dsh` generates a notes-only workspace overlay plus a
dedicated `inference-evolution` headless profile. Its local route uses
`EVOLUTION_LOCAL_API_KEY` or the dummy token `local` required by pi-ai for an
unauthenticated localhost endpoint. Real credentials must be supplied outside
tracked files. User-unit PATH is captured at installation to find Node/pnpm.

`dsh_bridge.py` calls the supported `pnpm dsh` entry point, checks the JSON
`turn_end` and `final` events plus exit code, and records receipts under
`dsh-receipts/`. Successful later deliveries reuse `dsh-session.json` through
`--session-id`; the browser's live conversation is not hijacked. A stale session
or incompatible workspace fails explicitly. Outbox delivery interrupted after
acceptance becomes `uncertain` and is never blindly retried. Pending obsolete
notifications become `superseded`; raw experiment results remain available.
This is conservative at-most-once dispatch, not guaranteed exactly-once delivery.

Both experiments and DSH turns have process timeouts enforced by the controller
and systemd. A machine restart or kernel/driver fault can still require manual
recovery; the supervisor does not reboot the host or reset GPUs.

## Verification

```sh
.venv/bin/python -m pytest tools/inference_evolution/tests -q --confcutdir=tools/inference_evolution/tests
.venv/bin/python tools/inference_evolution/verify.py --root /mnt/data/state/inference-evolution-validation
```

The second command requires a fresh installed instance, exercises actual
systemd services on a random loopback port, and retains `verification.json` and
all logs. It tests build/start/check failures, false token health, job timeout,
observation crash and controller SIGKILL recovery. It leaves a CPU baseline
running and an unused `dsh-failure` candidate for DSH submission testing. Pause
then stop the supervisor and its hashed `server` unit after verification; do not
delete state while its service still runs.
