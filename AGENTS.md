# AGENTS.md

## Agent skills

### Issue tracker

GitHub Issues via `gh` CLI: issues and specs for this repo live as GitHub issues in `dymoo/mimo-halo-lab`. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical triage states (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`) used verbatim as label strings; PRs are not a triage surface. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` at the repo root plus `docs/adr/`. See `docs/agents/domain.md`.

## Issue conventions

- Canonical triage states: `needs-triage` → `needs-info` or `ready-for-agent` / `ready-for-human`; closed states include `wontfix`.
- Category labels: `bug`, `enhancement` (plus auxiliary area/priority/blocked/hardware-needed/experiment/regression labels as work is planned).
- Parent specs and readiness milestones carry `ready-for-human`, never `ready-for-agent`. Generated implementation tickets bypass triage; their closed blocking dependencies remain the separate execution gate.
- `ready-for-agent` means fully specified. Select only implementation tickets whose blocking Issues are closed; exclude `blocked`, `hardware-needed`, the parent spec, and readiness milestones until their external prerequisites are satisfied.
- Issues carry: goal, inputs, outputs, blocking dependencies, acceptance criteria, commands, artifact/provenance requirements.

## Privacy rules (repo is public)

- Never commit or paste: raw traces, model weights, private dataset payloads, secrets, tokens, hardware serial numbers, or local absolute paths.
- Git carries manifests, hashes, aggregates and reproducible configuration only; large payloads live on local disk outside the repo.
- Redact/scan before any derived dataset or log leaves the workstation.
- A versioned pre-commit guard (`scripts/privacy_guard.py`) scans staged blobs; install locally with `python3 scripts/install_hooks.py`.
