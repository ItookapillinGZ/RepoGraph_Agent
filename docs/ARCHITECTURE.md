# RepoGraph architecture

## System shape

RepoGraph is a trusted local controller around bounded Agent workflows. The
controller owns filesystem, process, Git, GitHub, state, and approval actions.
Models produce strict structured decisions and candidates; they do not receive
general shell or write access.

```text
CLI / Studio
  -> orchestration graphs
       -> repository exploration graph
       -> engineering planning graph
       -> plan execution graph
       -> change-set review graph
       -> bounded correction graph
  -> deterministic verification -> Docker sandbox
  -> trace recorder -> SQLite + immutable artifact blobs
  -> G4 / G5A / G5B human gates
```

The workflows are nested through explicit validated models rather than a shared
free-form transcript. `EngineeringPlan` authorizes paths and actions.
`MultiFileCandidate` provides complete final contents. `ChangeSetReview`
summarizes bounded file reviews. `PlanApplicationBundle` is the deterministic
handoff into G4.

## Responsibility split

| Responsibility | Owner |
|---|---|
| Explore decision | LLM + bounded read-only tools |
| Planning | LLM structured output |
| Candidate generation | LLM structured output |
| Review and correction proposal | LLM, bounded by plan/evidence |
| Filesystem access | deterministic Python |
| Candidate path/action validation | deterministic Python |
| Test execution | deterministic sandbox |
| Apply | deterministic Python after human G4 approval |
| Git delivery | deterministic Python after human G5A approval |
| GitHub delivery | deterministic Python after human G5B approval |
| Approval | human |
| Trace and artifact storage | deterministic Python |
| Replay | deterministic Python; no LLM |

## State and graph boundaries

Planning state contains the task, bounded exploration evidence, a candidate
plan, validation failures, and a retry budget. Execution state contains the
fixed plan, candidate, temporary workspace, diff, verification, review, and a
bounded correction history. Correction cannot change plan paths or actions.

Deterministic validators run after every structured model output. Candidate
writes occur only in a bounded disposable copy until G4. Tool budgets, retry
semantics, prompts, and graph topology are intentionally frozen for v1.

## Sandbox

The host controller prepares a bounded temporary copy and asks the selected
backend to execute fixed argument vectors. Docker mode never falls back to host.
The hardened container uses network-none, read-only root, a non-root user,
capability drop, no-new-privileges, resource limits, timeout, bounded output,
and a fixed environment allowlist. The source repository and Docker socket are
not mounted.

## Observability and replay

Hierarchical runs, spans, and events are stored in SQLite. Artifact bytes are
content-addressed and linked with typed edges. Secrets and local home paths are
rejected or redacted before persistence. Replay validates base identity,
verification specification, backend, image provenance, and mutation-free
metadata before re-running deterministic verification.

The recorded 100-event synthetic microbenchmark was approximately 0.170 s with
observability disabled and 2.068 s with durable SQLite writes. This is a
synthetic per-event durable-write microbenchmark, not end-to-end latency.

## Approval and delivery boundaries

- G4 validates a saved verified bundle and atomically applies it to the working
  tree only with `--approve`.
- G5A creates a branch and commit only with `--approve-git-delivery`.
- G5B pushes and creates/reuses a pull request only with
  `--approve-remote-delivery`.

Each stage consumes a previous deterministic artifact. Approvals are independent
and never chained.

## Studio

Studio is a Next.js frontend over a FastAPI backend. The backend calls RepoGraph
core services and stores local run state in SQLite. Server-sent events provide
bounded timeline updates. Studio does not collapse the sandbox or approval
boundaries; UI buttons map to distinct backend approval endpoints.

## Evaluation

The evaluation harness has deterministic task manifests, frozen evaluators,
candidate artifacts, campaign provenance, ablations, bootstrap analysis, and
separate capability, end-to-end, and infrastructure denominators. Historical
H2 results are immutable evidence, not regenerated for presentation.
