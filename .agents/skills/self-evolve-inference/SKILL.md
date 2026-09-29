---
name: self-evolve-inference
description: Run bounded, evidence-backed optimization experiments against a local inference service while an external supervisor owns build, restart, health gates, rollback, and durable recovery notes. Use when improving vLLM or another local model server; do not use for ordinary code edits unrelated to a running inference service.
metadata:
  short-description: Safely evolve a local inference service
---

# Self evolve inference

Use the external supervisor before changing a live inference service. The
supervisor is deliberately CPU-only and independent from the service code or
virtual environment. It owns the service lifecycle; this skill owns the
hypothesis, experiment design, interpretation, and research notes.

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
Python interpreter and systemd unit. Start the unit with `systemctl --user start`
when the task authorizes experimentation. Read `status` and `logs` via the
controller CLI. The same CLI works while the model is offline.

## Submit and interpret

Register a new release directory with immutable source, environment and
compiled artifacts. Submit one experiment request. Never reuse a sealed release
ID or change a request after submission. The supervisor deduplicates IDs and
will restore the stable release if a candidate fails to start, fails a real
token probe, fails checks, crashes during observation, or exceeds its budget.

Once submission is durably acknowledged, yield to the supervisor. Let its
dedicated recovery session update shared notes; do not run a second writer
against `current.md` or `trajectory.md` during that continuation.

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
notes. The recovery prompt must not submit another experiment.

Detailed JSON fields and the systemd installation procedure are in
[references/supervisor.md](references/supervisor.md). The Chinese operations
guide [references/operations.zh.md](references/operations.zh.md) maps the
repository files, external installation/state trees, note schemas, pinned CLI
usage, and production boundaries. For first adoption and fallback, also read
[references/production.zh.md](references/production.zh.md). Read the relevant
reference when creating releases, changing the supervisor, or diagnosing
recovery.
