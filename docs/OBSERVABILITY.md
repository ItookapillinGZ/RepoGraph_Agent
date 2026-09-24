# RepoGraph observability

RepoGraph observability is a local orchestration layer. It observes execution; it
does not own execution and it does not change Agent decisions.

## Architecture

```text
RepoGraph orchestration
├─ existing Agent pipeline
├─ existing Docker sandbox
└─ injected TraceContext (contextvars, never LangGraph State)
   ├─ runs and nested spans
   ├─ monotonically ordered events
   ├─ aggregate metrics
   └─ immutable artifact lineage
```

`TraceContext` carries only the current run, parent span, sink, and failure
policy. Trace IDs, database connections, HTTP state, and Studio state are never
added to a LangGraph state schema. Disabled mode uses `NullTraceSink` and keeps
the previous call path unchanged. `CompositeTraceSink` supports local adapters.

The normal interactive policy is best effort: a telemetry failure is recorded
internally but cannot turn a successful Agent result into an Agent failure.
Evaluation and replay integrity runs may select required mode and fail closed.

## Durable model and ordering

Durable records are strict, versioned Pydantic models:

- `TraceRun`
- `TraceSpan`
- `TraceEvent`
- `ArtifactRecord`
- `ArtifactEdge`
- `ReplayResult`

SQLite is the authoritative index. WAL, foreign keys, bounded busy waits,
transactions, and indexes are enabled. Each run owns a counter row. Event and
span sequences are allocated while holding `BEGIN IMMEDIATE`; ordering never
depends only on timestamps or an unlocked `MAX(sequence) + 1` query.

Wall time is stored for human inspection. Durations use `time.monotonic()`.
Spans left active by a crash remain visibly incomplete until the explicit
reconciliation operation marks stale runs and spans `interrupted`. Missing spans
are never invented.

## Span hierarchy

The public graph boundaries produce nested spans for:

- `engineering_plan_graph`
- `repository_exploration_graph`
- `plan_execution_graph`
- `change_set_graph`
- `review_graph`
- sandbox execution
- G4 application, G5A Git delivery, and G5B GitHub delivery after their existing
  approval gates

LangChain callbacks create child `llm_call` and `tool_call` spans. Model
telemetry contains provider/configured model/resolved model when available,
call status, provider-reported token counts, completion, and monotonic duration.
Tool telemetry stores tool name, call index, argument size/digest, result
size/digest, status, and duration.

Sandbox spans record backend, sandbox flag, image identity, timeout, memory,
CPU, PID limit, network policy, exit code, timeout/truncation flags, durations,
and output sizes/digests. Full stdout and stderr are not copied into trace
metadata.

## Artifact lineage and integrity

Artifact content is stored in a trusted state root under a SHA-256 content
address. The SQLite registry retains a distinct semantic artifact identity even
when two records share identical bytes. Each load recomputes size and SHA-256;
a mismatch stops with `artifact_integrity_error`. Re-registering the same ID is
idempotent only when every immutable field matches.

Edges retain meaning (`derived_from`, `verified_by`, `reviewed_by`,
`corrected_from`, `packaged_as`, `delivered_as`, `evaluated_by`, or
`replayed_as`). Missing parents, cross-run edges, and cycles are rejected.

Studio preview persistence registers plan, candidate versions, final diff,
verification, review, and application bundle lineage outside the analyzed
repository. Studio's existing event table remains the SSE source of truth, so
disconnect/reconnect and existing sequence semantics are unchanged.

## Privacy and bounds

Persistence passes through one redaction layer. It recognizes sensitive key
names, configured secret values, OpenAI/GitHub key shapes, bearer headers,
credential-bearing proxy URLs, `.env` assignments, Docker auth values, and user
home paths. The existing evaluation artifact secret scanner is reused as a
post-redaction guard.

RepoGraph does **not** persist:

- hidden reasoning, chain-of-thought, scratchpads, or reasoning-token content;
- full prompts or complete model transcripts;
- raw authorization material or approval secrets;
- complete repository files, environment variables, stdout, or stderr in
  telemetry;
- sensitive absolute user paths in portable exports.

Prompts are represented by template/version identifiers when known and a
digest. Structured outputs belong in bounded immutable artifacts, not events.
Hard limits exist for events/run, spans/run, metadata bytes, artifact bytes, and
Studio summaries.

## Local storage and CLI

Set `REPOGRAPH_STATE_ROOT` to an operator-controlled directory. If absent, the
CLI uses the platform-local RepoGraph application data directory. Observability
never sends telemetry to a collector or SaaS and has no cloud dependency.

```powershell
python -m observability runs
python -m observability show <run-id>
python -m observability artifacts <run-id>
python -m observability lineage <run-id>
python -m observability export <run-id> --output trace.jsonl
python -m observability replay <run-id> --candidate <artifact-id>
python -m observability reconcile
```

JSONL export has deterministic record ordering, a schema-version manifest, run
metadata, spans, events, artifacts, lineage edges, and metrics. Export passes
through redaction again.

