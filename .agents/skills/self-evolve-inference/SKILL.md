---
name: self-evolve-inference
description: Explicitly requested self-evolution of a local inference service, with bounded experiments, an external CPU supervisor, rollback and durable notes. Activate only when the user explicitly requests this skill or self-evolution; ordinary optimization or debugging requests do not activate it.
metadata:
  short-description: Safely evolve a local inference service
---

# Self evolve inference

Use the external supervisor before changing a live inference service. The
supervisor is deliberately CPU-only and independent from the service code or
virtual environment. It owns build, restart, health gates and rollback; this
skill owns the mission contract, hypothesis, experiment design, interpretation,
report and research notes.

## DSH mission contract

Activate only on an explicit user request to use `self-evolve-inference` or
the self-evolution skill. Ordinary optimization, debugging, performance or
service restart requests do not authorize activation. Keep this rule in DSH
even if its skill loader ignores `agents/openai.yaml`. A user saying
"continue" within an already authorized mission preserves that mission's
scope and budget; it does not create a new mission or enable continuous mode.
Machine recovery notifications only explain existing results and never
authorize activation of a new optimization mission.

The normal entry point is a DSH conversation in which the user explicitly says
to use this skill for a dsv41f optimization direction or problem. Treat that
message as a mission request, not as a request for generic advice. Extract the
target, symptom or optimization direction, acceptance evidence, time/GPU
budget, and the requested stopping mode.

There are two modes:

- **Bounded (default):** work on the named direction until it is solved, the
  evidence shows that it is currently infeasible, a safety/correctness gate
  fails, or the declared budget is exhausted. Stop there and return a report;
  do not invent a second direction.
- **Open-ended continuous:** enter this mode only when the user's instruction
  explicitly asks for continuous/ongoing/automatic optimization. After each
  completed experiment, record the result and choose the next ranked direction
  under the same mission. Continue until the user asks to stop, the budget or
  stagnation limit is reached, a gate fails, or the supervisor becomes blocked.
  This is still bounded by those stop conditions, not an invisible infinite
  loop.

At mission start create a stable mission ID and write its brief/checkpoint.
Keep mission state, experiment IDs, decisions, raw result paths and the stop
reason in `notes/missions/<mission-id>/`. A mission report distinguishes
measured facts, interpretation, unresolved questions and the exact reason the
mission ended. Read [references/mission.zh.md](references/mission.zh.md) for
the contract and report template.

Before the first managed experiment, start the pinned CPU supervisor using
`lifecycle.py start --lifeguard FILE --ttl SECONDS` from `installation.json`.
Use the mission's remaining finite wall-time budget as TTL (default 3600 s).
Read-only investigation and notes-only recovery do not start it. At a bounded
mission's terminal report, or when a
continuous mission is stopped, close it with `lifecycle.py stop` after verifying
there is no active experiment. Stopping the supervisor does not stop the
controller-owned stable server; it only ends the temporary lifeguard process.
Never stop it while a build, switch, rollback or recovery job is active. The
supervisor may remain running between iterations of a continuous mission,
within that mission's deadline. On expiry it stops new work, settles recovery,
and exits; this bounds orphan lifetime if the DSH session disappears. Expiry
does not generate the final interpretation: the skill must still read evidence
and write its report when the originating session resumes.

## Before an experiment

For recovery notifications, use the explicit state path supplied in the task.
Read only the named result, state, and current note first. Do not search other
DSH sessions. The recovery workspace permits writes to notes only; it does not
authorize another experiment. Update `current.md` and append `trajectory.md`,
then finish. Read the detailed reference only when an operation needs it.

Read the supervisor `state.json`, `events.jsonl`, `notes/current.md`, the
relevant lessons, and the most recent experiment result. Verify that the
service is a sealed stable release and that no experiment is active. Choose
one hypothesis, a bounded workload, a correctness gate, a performance gate,
an observation window, a time budget, and an explicit rollback release.

Write the checkpoint before submitting. It must tell a fresh agent what was
being tested, what must not be inferred from the result, and where to continue
after a service outage.

Keep the user's existing experiment scope and budget across recovery. Do not
ask for authorization again for work already covered by that scope. A recovery
notification's explicit notes-only restriction still applies to that turn.

Use `installation.json` in the state directory to locate the pinned controller,
Python interpreter and systemd unit. Read `status` and `logs` via the controller
CLI. The same CLI works while the model is offline. Start and stop the lifeguard
through the lifecycle helper, not by killing its process.

## Submit and interpret

Register a new release directory with immutable source, environment and
compiled artifacts. Submit one experiment request. Never reuse a sealed release
ID or change a request after submission. The supervisor deduplicates IDs and
will restore the stable release if a candidate fails to start, fails a real
token probe, fails checks, crashes during observation, or exceeds its budget.

Every experiment or manual rollback request must carry explicit lifeguard
startup/recovery timeouts, rollback release, probe timeout, failure threshold,
restart budget and restart policy. Persist the checkpoint before submitting.
After acknowledgement, use `controller.py wait ID --timeout SECONDS` in the
originating DSH session, then read its durable result. A wait timeout does not
cancel the transaction or authorize resubmission under a new ID. Mission starts
set `origin_session=true`, suppressing competing headless notes writers.

After completion, read the machine result and logs before updating notes. Keep
machine facts separate from interpretation. A microbenchmark improvement is
not an end-to-end serving result. Record model, GPU, parallelism, workload,
switches, baseline, candidate, variance and failure conditions with each lesson.

Maintain four distinct notes: `current.md` for the live scene and next action;
`trajectory.md` for chronological decisions linked to experiment IDs;
`lessons.md` for evidence, scope, confidence and counterexamples;
`directions.md` for ranked hypotheses, expected benefit, cheapest falsification
and stopping criteria. Add conclusions only when an experiment supplies new
evidence. Never overwrite raw results or treat fixture tests as GPU validation.

Promotion is a separate request. It is allowed only for a candidate that is
still serving, passed checks and observation, and has an explicit decision.
Do not promote because the candidate starts successfully.

## Recovery

If the supervisor reports `blocked`, stop submitting experiments and preserve
the incident. Read the incident record and use a human controlled recovery.
If an outbox item is `uncertain`, do not replay it: a DSH request may already
have been accepted. For a pending outbox item, use the configured headless DSH
command to read the result and checkpoint, explain the outcome, and update
notes. The recovery prompt must not submit another experiment. This notes-only
rule applies to supervisor recovery turns, not to a new user-authored DSH
mission.

Detailed JSON fields and the systemd installation procedure are in
[references/supervisor.md](references/supervisor.md). The Chinese operations
guide [references/operations.zh.md](references/operations.zh.md) maps the
repository files, external installation/state trees, note schemas, pinned CLI
usage, and production boundaries. For first adoption and fallback, also read
[references/production.zh.md](references/production.zh.md). Read the relevant
reference when creating releases, changing the supervisor, or diagnosing
recovery.
