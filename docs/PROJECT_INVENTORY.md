# Project inventory

## Core engineering control plane

- `agent.py`: existing CLI composition and G4/G5 dispatch
- `repository_exploration.py`, `engineering_plan.py`: bounded exploration and
  structured planning
- `plan_execution.py`: multi-file candidate validation, temporary execution,
  verification, review, and correction
- `change_set.py`, `review_models.py`: file/change-set review contracts
- `plan_application.py`: G4 transactional apply
- `git_delivery.py`: G5A branch and commit
- `github_delivery.py`, `github_pr.py`: G5B push and pull request

## Product entrypoints

- `repograph/`: unified doctor, deterministic demo, and release check
- `scripts/`: Windows-first PowerShell wrappers and Studio process supervisor
- `demo/fixture_repo/`: small immutable offline demonstration repository

## Safety and deterministic execution

- `temporary_workspace.py`: bounded copies excluding secrets and metadata
- `sandbox/`: host/Docker backends, immutable policy, diagnostics, and acceptance
- `docker/repograph-sandbox.Dockerfile`: pinned non-root verification image
- `static_analysis.py`, `test_execution.py`: Ruff/Bandit and targeted tests

## Evaluation and evidence

- `evaluation/`: datasets, campaigns, ablations, metrics, frozen evaluators,
  reports, provenance, and optional SWE-bench adapters
- `evaluation/campaigns/`: preserved H2 history and authoritative results
- `sandbox/acceptance/results/`: preserved H3 security/equivalence evidence
- `observability/acceptance/results/`: preserved H3.1 replay evidence

## Observability

- `observability/`: context-local tracing, redaction, SQLite storage, artifacts,
  lineage, export, metrics, and replay

## Studio

- `studio_backend/`: FastAPI repositories, runs, approvals, events, and recovery
- `studio/`: Next.js/React frontend, trace/review/approval panels, Playwright E2E

## Tests

- `tests/`: core, sandbox, evaluation, Studio, observability, security, and H4
  product-entrypoint coverage

This is a functional inventory, not an exhaustive file listing.
