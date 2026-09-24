# RepoGraph portfolio and interview guide

## English summary (129 words)

RepoGraph is a local repository-engineering control plane built around five
bounded LangGraph workflows. It explores a codebase, produces a structured
engineering plan, generates and reviews multi-file candidates, and performs
evidence-driven self-correction without giving the model unrestricted shell or
filesystem access. Deterministic Python components enforce path, state, and
mutation boundaries; repository-controlled verification runs in a hardened,
network-disabled Docker sandbox. Working-tree apply, local Git delivery, and
GitHub pull-request delivery are three independent human approval gates. A
FastAPI/Next.js Studio exposes run state while SQLite tracing records
hierarchical spans, content-addressed artifacts, lineage, and mutation-free
verification replay. Evaluation includes a frozen 17-task paired benchmark,
real Docker security acceptance, five Host/Docker equivalence cases, and five
deterministic replay cases. The project is validated for local portfolio and
demo use, not claimed as enterprise production infrastructure.

## 中文简介（约 190 字）

RepoGraph 是一个面向代码仓库工程任务的本地控制平面，由五个受边界约束的 LangGraph
工作流组成。系统将仓库探索、结构化规划、多文件候选生成、验证、评审与证据驱动的
自我修正串联起来，同时不给模型开放任意 Shell 或文件系统写权限。路径、状态与变更
边界由确定性 Python 代码执行；仓库代码在禁网、非 root、受 CPU/内存/PID/超时限制的
Docker 沙箱中验证。G4 工作树应用、G5A 本地 Git 提交、G5B GitHub PR 发布是三个独立
人工审批门。FastAPI/Next.js Studio 提供交互界面，SQLite 记录层级追踪、产物血缘与无
LLM 的确定性重放。项目包含 17 任务配对评估、真实 Docker 安全验收、5/5 环境等价与
5/5 重放证据，定位为本地作品集与演示版本，而非企业生产平台。

## Truthful resume bullets

- Designed a repository-engineering control plane spanning five bounded
  LangGraph workflows for exploration, planning, multi-file execution, review,
  and evidence-driven correction.
- Implemented a hardened Docker verification boundary with network isolation,
  non-root execution, read-only root, capability drop, resource limits, and
  fail-closed backend selection; validated 5/5 Host/Docker equivalence cases.
- Built three independent human approval gates for transactional apply, local
  branch/commit delivery, and GitHub push/PR publication with no auto-chaining.
- Created a frozen 17-task paired evaluation harness; the full configuration
  reached 17/17 final outcomes after bounded correction, with small-sample and
  confidence-interval limitations explicitly reported.
- Added local hierarchical tracing, content-addressed artifact lineage, and
  deterministic verification replay, validated on 5/5 frozen cases and a
  Python suite exceeding 900 tests.

## Interview questions and 30–60 second answers

### 1. Why LangGraph?

The work has explicit states, bounded retries, conditional correction, and
different trust boundaries. LangGraph makes transitions inspectable while
Pydantic models keep each handoff structured. It is used for orchestration, not
as permission to put deterministic filesystem or delivery logic in the model.

### 2. Why not give the model shell access?

Shell access combines interpretation and authority. RepoGraph lets the model
choose among bounded read tools and produce strict plans/candidates; trusted
Python validates paths and fixed argument vectors before any process runs.

### 3. How is multi-file correction bounded?

The original `EngineeringPlan` stays fixed. A corrector sees only the current
candidate and bounded verification/review evidence. Every replacement must
contain exactly the planned paths/actions and pass the same validation and
budgets in a fresh temporary copy.

### 4. Why three approval gates?

Applying files, creating local Git history, and publishing remotely have
different blast radii. Separate artifacts and explicit flags let an operator
stop after any stage; no approval implies a later approval.

### 5. What does Docker protect?

It limits repository-controlled test processes: no network, non-root user,
read-only root, dropped capabilities, no-new-privileges, bounded resources and
output, and a filtered environment. It is stronger than host execution but not
a VM or a multi-tenant cloud boundary.

### 6. Why not mount the Docker socket?

Docker socket access is effectively host-level control. The trusted controller
invokes Docker from the host and passes only a disposable repository copy; the
container never receives the socket or source repository mount.

### 7. How do you evaluate Agent capability?

Tasks, manifests, candidates, evaluators, and campaign provenance are frozen.
I report first-pass and final outcomes, pair full and ablation variants, and
keep evaluator and infrastructure failures distinguishable from capability.

### 8. Why separate denominators?

Capability answers whether a reached Agent task was solved. End-to-end includes
all requested runs. Infrastructure isolates environment failures. Combining
them can either unfairly penalize model capability or overstate product
reliability.

### 9. How does replay work without an LLM?

The candidate checkpoint stores base identity, verification spec, backend,
image provenance, and original normalized outcome. Replay integrity-checks
those values and reruns deterministic verification. It does not claim to replay
the model's hidden reasoning.

### 10. Why SQLite?

This is a local single-operator product. SQLite provides transactions, indexed
queries, and simple distribution without a service dependency. Its per-event
durable write overhead is documented; a distributed deployment would need a
different telemetry architecture.

### 11. Why structured models instead of free-form output?

Schemas make missing paths, invalid actions, and unbounded text rejectable
before execution. They also create stable boundaries for tests, traces,
delivery artifacts, and deterministic revalidation.

### 12. Why fixed budgets?

Budgets bound cost, context, execution, and correction loops. They reduce
autonomy on unusually large tasks, but make local behavior auditable and stop
untrusted content from expanding authority.

### 13. What did the benchmark prove?

It demonstrated the harness and showed how correction affected this controlled
17-task set. Full reached 17/17 final with two rescues, but ablation differences
were one task and paired intervals included zero. It did not prove general
superiority or a production success rate.

### 14. What would change at larger scale?

I would separate controller and workers, use VM/microVM isolation where needed,
move traces to a batched distributed store, add identity/authorization, sign
artifacts, and expand evaluation. Those are intentionally outside v1.

### 15. What is the biggest remaining risk?

Executing untrusted code always carries residual risk. Docker shares a kernel,
dependency availability is constrained by network-none, and the project lacks
multi-tenant controls. The current claim is safe local portfolio/demo use, not
hostile multi-tenant production.

## Tradeoffs

| Choice | Benefit | Cost / alternative |
|---|---|---|
| SQLite vs distributed tracing | simple transactional local source of truth | durable per-event overhead; no cross-node aggregation |
| Docker vs VM isolation | accessible, reproducible local boundary | shared kernel; VM/microVM is stronger |
| Local execution vs cloud workers | inspectable and inexpensive | no elastic scaling or centralized identity |
| Fixed budgets vs autonomy | bounded cost and blast radius | may reject legitimate large tasks |
| Structured vs free-form models | deterministic validation | more schema design and retry handling |
| Network-none vs dependency install | prevents exfiltration/downloads | dependencies must already be in the image/workspace |
| N=17 vs SWE-bench-scale | fast paired diagnosis | small sample and ceiling effects |

## Limitations

- No official full SWE-bench Docker campaign
- Small N=17 controlled benchmark; paired intervals include zero
- Windows-first release validation; Linux/macOS unverified
- Docker is not perfect isolation
- local SQLite, no distributed execution
- no multi-tenant auth or cloud deployment
- replay verifies deterministic execution, not LLM reasoning
- no selected open-source license
