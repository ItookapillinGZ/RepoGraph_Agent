# Deterministic replay

RepoGraph has two separate replay concepts.

## Trace replay

Trace replay reads persisted records and reconstructs the timeline, span tree,
metrics, and artifact lineage. It executes no code and calls no model.

## Deterministic execution replay

Execution replay begins only from an `ArtifactRecord`. Before verification it:

1. loads content from the trusted artifact store;
2. recomputes byte size and SHA-256;
3. verifies the recorded base repository identity;
4. verifies the verification/evaluator specification digest;
5. requires the recorded sandbox backend;
6. for Docker records, requires the recorded resolved image ID;
7. materializes a fresh disposable workspace; and
8. re-executes only deterministic verification/evaluation.

The semantic comparison includes original/replay status, resolved state, exit
classification, and an optional normalized report digest. Runtime, timestamps,
and temporary paths are intentionally excluded.

## Guarantees

- OpenAI/model calls during replay: **zero**.
- Planner, Explorer LLM, Executor, Reviewer, and Corrector calls: **prohibited**.
- G4 apply, G5A commit, G5B push/PR, and any other real mutation: **prohibited**.
- Arbitrary filesystem paths cannot bypass the artifact registry.
- Docker-originated verification cannot fall back to Host.
- A missing Docker daemon/image produces `BLOCKED`.
- A changed image ID produces a provenance mismatch; image drift has no implicit
  override.
- Tampered artifact bytes stop before an executor is called.

Replay is local. It does not contact an observability service. The only network
policy available to the H3 Docker verification path remains `network=none`.

## What replay does not guarantee

**Deterministic replay does not reproduce LLM reasoning.** LLM responses are
not deterministic checkpoints and hidden reasoning is never recorded. Replay
reconstructs historical traces and re-executes deterministic stages from
immutable candidate checkpoints.

Semantic equality does not require equal wall-clock duration, process IDs,
timestamps, or temporary workspace paths. A matching result under a different
backend or Docker image is not reported as equivalent.

## Current adapters

The CLI ships a fixed adapter for the frozen H2 candidate snapshots used by the
H3/H3.1 acceptance path. It resolves task/evaluator details from trusted frozen
fixtures, builds a fresh workspace, and routes evaluation through the recorded
Docker policy. Unknown adapter identifiers are blocked rather than interpreted
as commands.

Additional replay adapters must be explicit code, accept registered artifacts,
enforce the same provenance checks, and remain mutation-free. Raw shell command
metadata is not a replay adapter.

