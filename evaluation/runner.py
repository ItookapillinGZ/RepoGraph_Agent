"""External sequential orchestration over RepoGraph and task evaluators."""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from engineering_plan import EngineeringPlan, plan_repository_task
from evaluation.candidate_artifacts import (
    CandidateArtifactError,
    CandidateArtifactStore,
    candidate_sha256,
)
from evaluation.evaluator import (
    LOCAL_EVALUATOR_VERSION,
    EvaluatorPreflightError,
    build_local_evaluator_specification,
    evaluator_environment_fingerprint,
    run_local_evaluator,
)
from evaluation.models import (
    EvaluationCandidateSnapshot,
    EvaluationConfig,
    EvaluationExperiment,
    EvaluationProvenance,
    EvaluationResult,
    EvaluationTask,
    EvaluationWorkerRequest,
    EvaluatorOutcome,
    EvaluatorSpecification,
    LLMUsageSummary,
)
from evaluation.process_runner import run_evaluation_worker
from evaluation.security import redact_secret_values
from evaluation.storage import EvaluationStorage
from evaluation.telemetry import EvaluationTelemetryCollector
from evaluation.workspace import (
    WorkspaceError,
    apply_candidate_exact,
    extract_candidate_patch,
    prepare_task_workspace,
    reset_task_workspace,
)
from model_defaults import get_production_model_settings
from observability.recorder import TraceRecorder, trace_run
from observability.sinks import TraceSink
from plan_execution import (
    MultiFileCandidate,
    PlanExecutionResult,
    execute_engineering_plan,
)
from review_models import OverallRating
from sandbox.runner import policy_from_backend


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class RepoGraphRun:
    """Adapter result with evaluation-only candidate artifacts and telemetry."""

    plan: EngineeringPlan | None = None
    execution: PlanExecutionResult | None = None
    evaluated_candidates: list[MultiFileCandidate] = field(default_factory=list)
    tool_calls: int | None = None
    test_calls: int | None = None
    llm_usage: LLMUsageSummary | None = None
    warnings: list[str] = field(default_factory=list)


class RepoGraphAdapter(Protocol):
    def run(
        self,
        workspace: str,
        task: EvaluationTask,
        config: EvaluationConfig,
    ) -> RepoGraphRun: ...


class RealRepoGraphAdapter:
    """Thin H2 adapter over the existing G1/G2 public API boundaries."""

    def __init__(
        self,
        *,
        planning_graph: object | None = None,
        execution_graph: object | None = None,
        trace_sink: TraceSink | None = None,
        trace_required: bool = False,
    ) -> None:
        self.planning_graph = planning_graph
        self.execution_graph = execution_graph
        self.trace_sink = trace_sink
        self.trace_required = trace_required

    def run(
        self,
        workspace: str,
        task: EvaluationTask,
        config: EvaluationConfig,
    ) -> RepoGraphRun:
        if self.trace_sink is None:
            return self._run(workspace, task, config)
        recorder = TraceRecorder(self.trace_sink, required=self.trace_required)
        with trace_run(
            recorder,
            run_kind="evaluation",
            repo_identity=f"{task.dataset}:{task.id}:{task.base_commit}",
            task=task.task,
            task_summary=f"Evaluation task {task.id}",
        ):
            return self._run(workspace, task, config)

    def _run(
        self,
        workspace: str,
        task: EvaluationTask,
        config: EvaluationConfig,
    ) -> RepoGraphRun:
        tool_calls = 0
        internal_test_calls = 0
        candidates: list[MultiFileCandidate] = []
        warnings: list[str] = []
        telemetry = EvaluationTelemetryCollector()

        def planning_event(_node: str, update: dict[str, object]) -> None:
            nonlocal tool_calls, internal_test_calls
            exploration = update.get("exploration_result")
            if exploration is not None:
                tool_calls += int(getattr(exploration, "tool_call_count", 0))
                internal_test_calls += int(
                    getattr(exploration, "test_tool_call_count", 0)
                )

        plan_arguments: dict[str, object] = {
            "event_sink": planning_event,
            "callbacks": [telemetry],
        }
        if self.planning_graph is not None:
            plan_arguments["graph"] = self.planning_graph
        plan = plan_repository_task(workspace, task.task, **plan_arguments)

        def execution_event(_node: str, update: dict[str, object]) -> None:
            nonlocal internal_test_calls
            verification = update.get("verification")
            if verification is not None:
                test_result = getattr(verification, "test_result", None)
                if (
                    test_result is not None
                    and getattr(test_result, "status", "not_run") != "not_run"
                ):
                    internal_test_calls += 1

        def attempt_sink(_attempt: int, candidate: MultiFileCandidate) -> None:
            candidates.append(MultiFileCandidate.model_validate(candidate.model_dump()))

        execution_arguments: dict[str, object] = {
            "max_correction_rounds": config.effective_correction_rounds,
            "agentic_explore": config.agentic_explore,
            "run_tests": config.run_tests,
            "agentic_test": config.agentic_test,
            "event_sink": execution_event,
            "attempt_artifact_sink": attempt_sink,
            "callbacks": [telemetry],
            "sandbox_backend": config.sandbox_backend,
        }
        if self.execution_graph is not None:
            execution_arguments["graph"] = self.execution_graph
        execution = execute_engineering_plan(
            workspace,
            task.task,
            plan,
            **execution_arguments,
        )
        configured_model = get_production_model_settings().model
        if config.model_name and config.model_name != configured_model:
            warnings.append(
                "model_name metadata differs from the configured RepoGraph production model."
            )
        return RepoGraphRun(
            plan=plan,
            execution=execution,
            evaluated_candidates=candidates,
            tool_calls=tool_calls,
            test_calls=internal_test_calls,
            llm_usage=telemetry.summary(),
            warnings=warnings,
        )


def _result(
    task: EvaluationTask,
    config: EvaluationConfig,
    experiment_id: str,
    started: float,
    git_commit: str | None,
    **updates: object,
) -> EvaluationResult:
    try:
        evaluator_specification = build_local_evaluator_specification(
            task,
            timeout_seconds=config.evaluator_timeout_seconds,
        )
        evaluator_environment = evaluator_environment_fingerprint(
            evaluator_specification
        )
    except EvaluatorPreflightError:
        evaluator_specification = None
        evaluator_environment = None
    defaults: dict[str, object] = {
        "task_id": task.id,
        "dataset": task.dataset,
        "experiment": experiment_id,
        "repository": task.repository,
        "base_commit": task.base_commit,
        "created_at": utc_now(),
        "git_commit": git_commit,
        "model_name": config.model_name,
        "configured_model_name": config.model_name,
        "llm_execution_kind": config.llm_execution_kind,
        "config_digest": config.digest(),
        "evaluator_kind": "local",
        "evaluator_version": LOCAL_EVALUATOR_VERSION,
        "evaluator_digest": (
            evaluator_specification.digest if evaluator_specification else None
        ),
        "evaluator_specification": evaluator_specification,
        "evaluator_environment": evaluator_environment,
        "status": "agent_failed",
        "duration_seconds": time.monotonic() - started,
    }
    defaults.update(updates)
    category = defaults.get("failure_category")
    if "infrastructure_failure" not in updates:
        defaults["infrastructure_failure"] = (
            defaults["status"] in {"evaluation_error", "timeout"}
            or category
            in {
                "api_infrastructure_failure",
                "infrastructure_error",
                "external_evaluator_failure",
                "timeout",
            }
        )
    defaults = redact_secret_values(defaults)
    patch = defaults.get("model_patch")
    if isinstance(patch, str):
        defaults["prediction_sha256"] = hashlib.sha256(
            patch.encode("utf-8")
        ).hexdigest()
    defaults["provenance"] = EvaluationProvenance(
        repo_base_commit=task.base_commit,
        repograph_source_commit=git_commit,
        config_digest=config.digest(),
        model_name=config.model_name,
        configured_model_name=config.model_name,
        resolved_model_names=list(defaults.get("resolved_model_names") or []),
        model_temperature=config.model_temperature,
        model_seed=config.model_seed,
        llm_execution_kind=config.llm_execution_kind,
        model_provider=config.model_provider,
        model_base_url=config.model_base_url,
        model_reasoning_effort=config.model_reasoning_effort,
        model_max_completion_tokens=config.model_max_completion_tokens,
        task_timeout_seconds=config.effective_task_timeout_seconds,
        max_correction_rounds=config.effective_correction_rounds,
        prediction_sha256=defaults.get("prediction_sha256"),
        evaluator_kind="local",
        evaluator_version=LOCAL_EVALUATOR_VERSION,
        evaluator_digest=(
            evaluator_specification.digest if evaluator_specification else None
        ),
        python_executable=(
            evaluator_specification.python_executable
            if evaluator_specification
            else None
        ),
        timestamp=str(defaults["created_at"]),
    )
    if "warnings" in defaults:
        defaults["warnings"] = [
            str(item)[:2_000] for item in list(defaults["warnings"])[:100]
        ]
    return EvaluationResult.model_validate(defaults)

def _usage_updates(run: RepoGraphRun) -> dict[str, object]:
    usage = run.llm_usage
    if usage is None:
        return {}
    return {
        "llm_calls": usage.calls,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "telemetry_incomplete": usage.incomplete,
        "telemetry_warnings": usage.warnings,
        "resolved_model_names": usage.resolved_models,
    }


def _evaluate_candidate(
    task: EvaluationTask,
    config: EvaluationConfig,
    workspace: Path,
    base_commit: str,
    plan: EngineeringPlan,
    candidate: MultiFileCandidate,
    evaluator_specification: EvaluatorSpecification,
    *,
    extract_patch: bool,
) -> tuple[EvaluatorOutcome, str | None, list[str]]:
    reset_task_workspace(workspace, base_commit)
    apply_candidate_exact(workspace, plan, candidate)
    patch: str | None = None
    changed: list[str] = []
    if extract_patch:
        patch, changed = extract_candidate_patch(workspace, base_commit, candidate)
    if not config.run_tests:
        outcome = EvaluatorOutcome(
            status="error",
            failure_category="infrastructure_error",
            failure_kind="disabled",
            failure_reason="External evaluator is disabled by run_tests=False.",
            duration_seconds=0,
        )
    else:
        outcome = run_local_evaluator(
            task,
            workspace,
            timeout_seconds=config.evaluator_timeout_seconds,
            max_output_chars=config.max_evaluator_output_chars,
            specification=evaluator_specification,
            sandbox_policy=policy_from_backend(config.sandbox_backend),
        )
    return outcome, patch, changed


def _candidate_sequence(run: RepoGraphRun) -> list[MultiFileCandidate]:
    execution = run.execution
    if execution is None or execution.candidate is None:
        return []
    expected = execution.correction_rounds_used + 1
    candidates = [
        MultiFileCandidate.model_validate(item.model_dump())
        for item in run.evaluated_candidates
    ]
    final = MultiFileCandidate.model_validate(execution.candidate.model_dump())
    if len(candidates) == expected - 1 and (
        not candidates
        or candidate_sha256(candidates[-1]) != candidate_sha256(final)
    ):
        candidates.append(final)
    if not candidates and expected == 1:
        candidates = [final]
    if len(candidates) != expected:
        raise CandidateArtifactError(
            f"Captured {len(candidates)} candidate attempts; expected {expected}."
        )
    if candidate_sha256(candidates[-1]) != candidate_sha256(final):
        raise CandidateArtifactError(
            "Final execution candidate does not match the last captured attempt."
        )
    return candidates


def _persist_candidate_snapshots(
    *,
    task: EvaluationTask,
    config: EvaluationConfig,
    experiment_id: str,
    execution_attempt: int,
    artifacts_root: str | Path,
    workspace: Path,
    base_commit: str,
    plan: EngineeringPlan,
    run: RepoGraphRun,
    evaluator_digest: str,
) -> list[EvaluationCandidateSnapshot]:
    store = CandidateArtifactStore(artifacts_root)
    snapshots: list[EvaluationCandidateSnapshot] = []
    for attempt, candidate in enumerate(_candidate_sequence(run), start=1):
        reset_task_workspace(workspace, base_commit)
        apply_candidate_exact(workspace, plan, candidate)
        diff_text, _changed = extract_candidate_patch(
            workspace,
            base_commit,
            candidate,
        )
        snapshot, _relative = store.persist(
            task_id=task.id,
            experiment_id=experiment_id,
            execution_attempt=execution_attempt,
            attempt=attempt,
            base_commit=base_commit,
            model_name=config.model_name,
            candidate=candidate,
            diff_text=diff_text,
            evaluator_digest=evaluator_digest,
        )
        snapshots.append(snapshot)
    return snapshots

def _agent_failure_category(execution: PlanExecutionResult | None) -> str:
    if execution is None:
        return "planning_failure"
    if execution.correction_rounds_used > 0:
        return "correction_failure"
    if (
        execution.verification is not None
        and execution.verification.status in {"error", "failed"}
    ):
        return "verification_failure"
    if (
        execution.change_set_review is not None
        and execution.change_set_review.overall_rating != OverallRating.GOOD
    ):
        return "review_failure"
    return "candidate_generation_failure"


def _exception_failure_category(error: BaseException) -> str:
    """Retry only provider/transport failures, never model-generated solutions."""

    module = type(error).__module__.casefold()
    name = type(error).__name__.casefold()
    status_code = getattr(error, "status_code", None)
    transient_names = {
        "apierror",
        "apistatuserror",
        "authenticationerror",
        "ratelimiterror",
        "apiconnectionerror",
        "apitimeouterror",
        "internalservererror",
        "connecterror",
        "readerror",
        "remoteprotocolerror",
        "timeoutexception",
    }
    retryable_status = (
        isinstance(status_code, int)
        and (status_code in {408, 409, 429} or status_code >= 500)
    )
    if name in transient_names or retryable_status:
        return "api_infrastructure_failure"
    if "httpx" in module and any(
        marker in name for marker in ("connect", "read", "protocol", "timeout")
    ):
        return "api_infrastructure_failure"
    return "planning_failure"

def _run_evaluation_task_in_process(
    task: EvaluationTask,
    config: EvaluationConfig,
    *,
    workspace_root: str,
    artifacts_root: str | None = None,
    experiment_id: str | None = None,
    adapter: RepoGraphAdapter | None = None,
    git_commit: str | None = None,
    execution_attempt: int = 1,
) -> EvaluationResult:
    """Worker-only task body; production callers use run_evaluation_task."""

    task = EvaluationTask.model_validate(task)
    config = EvaluationConfig.model_validate(config)
    experiment = experiment_id or config.name
    started = time.monotonic()
    try:
        evaluator_specification = build_local_evaluator_specification(
            task,
            timeout_seconds=config.evaluator_timeout_seconds,
        )
        evaluator_environment = evaluator_environment_fingerprint(
            evaluator_specification
        )
    except EvaluatorPreflightError as error:
        return _result(
            task,
            config,
            experiment,
            started,
            git_commit,
            status="evaluation_error",
            failure_category="external_evaluator_failure",
            evaluator_failure_kind="preflight_failure",
            failure_reason=str(error)[:4_000],
            execution_attempt=execution_attempt,
        )
    try:
        workspace, resolved_base = prepare_task_workspace(
            task.repository,
            task.base_commit,
            workspace_root,
            experiment,
            task.id,
        )
    except (OSError, WorkspaceError, ValueError) as error:
        return _result(
            task,
            config,
            experiment,
            started,
            git_commit,
            status="evaluation_error",
            failure_category="infrastructure_error",
            failure_reason=str(error)[:4_000],
            execution_attempt=execution_attempt,
        )

    selected_adapter = adapter or RealRepoGraphAdapter()
    try:
        run = selected_adapter.run(str(workspace), task, config)
    except Exception as error:  # noqa: BLE001 - adapter boundary
        failure_category = _exception_failure_category(error)
        status = (
            "evaluation_error"
            if failure_category != "planning_failure"
            else "agent_failed"
        )
        return _result(
            task,
            config,
            experiment,
            started,
            git_commit,
            status=status,
            failure_category=failure_category,
            failure_reason=str(error)[:4_000],
            execution_attempt=execution_attempt,
        )

    execution = run.execution
    candidate = execution.candidate if execution is not None else None
    if candidate is None or run.plan is None:
        warnings = [*run.warnings, *(execution.warnings if execution else [])]
        return _result(
            task,
            config,
            experiment,
            started,
            git_commit,
            status="agent_failed",
            planning_succeeded=run.plan is not None,
            failure_category=_agent_failure_category(execution),
            failure_reason=(
                warnings[-1] if warnings else "RepoGraph produced no candidate."
            ),
            warnings=warnings,
            tool_calls=run.tool_calls,
            test_calls=run.test_calls,
            execution_attempt=execution_attempt,
            **_usage_updates(run),
        )

    selected_artifacts_root = (
        Path(artifacts_root).resolve()
        if artifacts_root
        else Path(workspace_root).resolve().parent / "artifacts"
    )
    try:
        snapshots = _persist_candidate_snapshots(
            task=task,
            config=config,
            experiment_id=experiment,
            execution_attempt=execution_attempt,
            artifacts_root=selected_artifacts_root,
            workspace=workspace,
            base_commit=resolved_base,
            plan=run.plan,
            run=run,
            evaluator_digest=evaluator_specification.digest,
        )
    except (OSError, ValueError, CandidateArtifactError) as error:
        return _result(
            task,
            config,
            experiment,
            started,
            git_commit,
            status="evaluation_error",
            planning_succeeded=True,
            candidate_generated=True,
            failure_category="infrastructure_error",
            evaluator_failure_kind="candidate_snapshot_failure",
            failure_reason=str(error)[:4_000],
            execution_attempt=execution_attempt,
            **_usage_updates(run),
        )
    first_candidate = snapshots[0].candidate
    warnings = [*run.warnings, *execution.warnings]
    snapshot_paths = [
        CandidateArtifactStore(selected_artifacts_root)
        .relative_path(experiment, task.id, execution_attempt, item.attempt)
        .as_posix()
        for item in snapshots
    ]
    external_calls = 0
    try:
        first_outcome, _, _ = _evaluate_candidate(
            task,
            config,
            workspace,
            resolved_base,
            run.plan,
            first_candidate,
            evaluator_specification,
            extract_patch=False,
        )
        external_calls += 1
        if execution.correction_rounds_used == 0:
            final_outcome = first_outcome
            reset_task_workspace(workspace, resolved_base)
            apply_candidate_exact(workspace, run.plan, candidate)
            model_patch, changed_files = extract_candidate_patch(
                workspace, resolved_base, candidate
            )
        else:
            final_outcome, model_patch, changed_files = _evaluate_candidate(
                task,
                config,
                workspace,
                resolved_base,
                run.plan,
                candidate,
                evaluator_specification,
                extract_patch=True,
            )
            external_calls += 1
    except (OSError, WorkspaceError, ValueError) as evaluation_error:
        return _result(
            task,
            config,
            experiment,
            started,
            git_commit,
            status="evaluation_error",
            planning_succeeded=True,
            candidate_generated=True,
            verification_succeeded=(
                execution.verification is not None
                and execution.verification.status == "verified"
            ),
            review_good=(
                execution.change_set_review is not None
                and execution.change_set_review.overall_rating == OverallRating.GOOD
            ),
            correction_rounds_used=execution.correction_rounds_used,
            failure_category="infrastructure_error",
            failure_reason=str(evaluation_error)[:4_000],
            warnings=warnings,
            tool_calls=run.tool_calls,
            test_calls=(run.test_calls or 0) + external_calls,
            candidate_snapshot_paths=snapshot_paths,
            first_candidate_sha256=snapshots[0].candidate_sha256,
            final_candidate_sha256=snapshots[-1].candidate_sha256,
            evaluator_failure_kind="candidate_evaluation_failure",
            execution_attempt=execution_attempt,
            **_usage_updates(run),
        )

    first_resolved = first_outcome.status == "passed"
    final_resolved = final_outcome.status == "passed"
    warnings.extend(first_outcome.warnings)
    if final_outcome is not first_outcome:
        warnings.extend(final_outcome.warnings)
    if task.expected_files:
        missing = sorted(set(task.expected_files) - set(changed_files))
        unexpected = sorted(set(changed_files) - set(task.expected_files))
        if missing:
            warnings.append("Expected files not changed: " + ", ".join(missing))
        if unexpected:
            warnings.append("Unexpected changed files: " + ", ".join(unexpected))
    first_evaluated = first_outcome.status in {"passed", "failed"}
    final_evaluated = final_outcome.status in {"passed", "failed"}
    if final_outcome.status == "passed":
        status = "resolved"
        failure_category = None
    elif final_outcome.status == "timeout":
        status = "timeout"
        failure_category = "external_evaluator_failure"
    elif final_outcome.status == "error":
        status = "evaluation_error"
        failure_category = "external_evaluator_failure"
    else:
        status = "unresolved"
        failure_category = "valid_prediction_unresolved"
    return _result(
        task,
        config,
        experiment,
        started,
        git_commit,
        status=status,
        first_attempt_resolved=first_resolved,
        final_resolved=final_resolved,
        correction_rounds_used=execution.correction_rounds_used,
        planning_succeeded=True,
        candidate_generated=True,
        verification_succeeded=(
            execution.verification is not None
            and execution.verification.status == "verified"
        ),
        review_good=(
            execution.change_set_review is not None
            and execution.change_set_review.overall_rating == OverallRating.GOOD
        ),
        changed_files=changed_files,
        tool_calls=run.tool_calls,
        test_calls=(run.test_calls or 0) + external_calls,
        failure_category=failure_category,
        failure_reason=final_outcome.failure_reason,
        evaluator_failure_kind=final_outcome.failure_kind,
        evaluator_specification=evaluator_specification,
        evaluator_environment=evaluator_environment,
        candidate_snapshot_paths=snapshot_paths,
        first_candidate_sha256=snapshots[0].candidate_sha256,
        final_candidate_sha256=snapshots[-1].candidate_sha256,
        first_attempt_evaluated=first_evaluated,
        final_attempt_evaluated=final_evaluated,
        valid_prediction=final_evaluated,
        infrastructure_failure=not first_evaluated or not final_evaluated,
        execution_attempt=execution_attempt,
        warnings=list(dict.fromkeys(warnings)),
        model_patch=model_patch,
        **_usage_updates(run),
    )


def run_evaluation_task(
    task: EvaluationTask,
    config: EvaluationConfig,
    *,
    workspace_root: str,
    artifacts_root: str | None = None,
    experiment_id: str | None = None,
    adapter: RepoGraphAdapter | None = None,
    git_commit: str | None = None,
    execution_attempt: int = 1,
) -> EvaluationResult:
    """Run production tasks in a killable process; keep explicit test adapters local."""

    task = EvaluationTask.model_validate(task)
    config = EvaluationConfig.model_validate(config)
    experiment = experiment_id or config.name
    if adapter is not None:
        return _run_evaluation_task_in_process(
            task,
            config,
            workspace_root=workspace_root,
            artifacts_root=artifacts_root,
            experiment_id=experiment,
            adapter=adapter,
            git_commit=git_commit,
            execution_attempt=execution_attempt,
        )

    started = time.monotonic()
    request = EvaluationWorkerRequest(
        task=task,
        config=config,
        workspace_root=workspace_root,
        artifacts_root=artifacts_root,
        experiment_id=experiment,
        result_path="controller-assigned",
        git_commit=git_commit,
        execution_attempt=execution_attempt,
    )
    worker_result = run_evaluation_worker(
        request,
        timeout_seconds=config.effective_task_timeout_seconds,
    )
    if worker_result.status == "completed" and worker_result.result is not None:
        return worker_result.result
    timed_out = worker_result.status == "timeout"
    return _result(
        task,
        config,
        experiment,
        started,
        git_commit,
        status="timeout" if timed_out else "evaluation_error",
        failure_category="timeout" if timed_out else "infrastructure_error",
        failure_reason=worker_result.failure_reason or "Evaluation worker failed.",
        process_isolated=True,
        hard_timeout=timed_out,
        execution_attempt=execution_attempt,
    )


def create_experiment(
    name: str,
    dataset: str,
    config: EvaluationConfig,
    *,
    git_commit: str | None = None,
) -> EvaluationExperiment:
    """Create stable metadata; callers persist it before running tasks."""

    return EvaluationExperiment(
        id=f"{name}-{uuid.uuid4().hex[:12]}",
        name=name,
        dataset=dataset,
        config=config,
        created_at=utc_now(),
        git_commit=git_commit,
        model_name=config.model_name,
        configured_model_name=config.model_name,
        llm_execution_kind=config.llm_execution_kind,
    )


def run_experiment(
    tasks: Sequence[EvaluationTask],
    experiment: EvaluationExperiment,
    *,
    workspace_root: str,
    storage: EvaluationStorage,
    artifacts_root: str | None = None,
    adapter: RepoGraphAdapter | None = None,
    rerun_failed: bool = False,
    max_infrastructure_retries: int = 0,
    before_task_run: Callable[[EvaluationExperiment, EvaluationTask], None]
    | None = None,
    after_execution_attempt: Callable[[EvaluationResult], None] | None = None,
    after_task_run: Callable[[EvaluationResult], None] | None = None,
) -> list[EvaluationResult]:
    """Run sequentially and retry only explicit infrastructure failures once."""

    if max_infrastructure_retries not in {0, 1}:
        raise ValueError("max_infrastructure_retries must be 0 or 1")
    storage.save_experiment(experiment)
    selected = list(tasks)
    if experiment.config.max_tasks is not None:
        selected = selected[: experiment.config.max_tasks]
    rerunnable = {"agent_failed", "evaluation_error", "timeout"}
    for task in selected:
        storage.save_task(task)
        existing = storage.get_result(experiment.id, task.id)
        if existing is not None and not (
            rerun_failed and existing.status in rerunnable
        ):
            continue
        final_result: EvaluationResult | None = None
        for retry_ordinal in range(max_infrastructure_retries + 1):
            attempt = storage.reserve_execution_attempt(
                campaign_id=experiment.name,
                config_name=experiment.config.name,
                experiment_id=experiment.id,
                task_id=task.id,
            )
            try:
                if before_task_run is not None:
                    before_task_run(experiment, task)
                attempt = storage.mark_execution_attempt_running(attempt)
                result = run_evaluation_task(
                    task,
                    experiment.config,
                    workspace_root=workspace_root,
                    artifacts_root=artifacts_root,
                    experiment_id=experiment.id,
                    adapter=adapter,
                    git_commit=experiment.git_commit,
                    execution_attempt=attempt.execution_attempt,
                )
            except BaseException as error:
                storage.finish_execution_attempt(
                    attempt,
                    status=(
                        "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                    ),
                    failure_reason=str(error)[:4000],
                )
                raise
            if attempt.execution_attempt > 1:
                result = result.model_copy(
                    update={"required_infrastructure_retry": True}
                )
            storage.save_result(result)
            terminal_status = (
                "timeout"
                if result.hard_timeout or result.status == "timeout"
                else "failed"
                if result.infrastructure_failure
                else "completed"
            )
            storage.finish_execution_attempt(
                attempt,
                status=terminal_status,
                failure_reason=(
                    result.failure_reason if terminal_status != "completed" else None
                ),
            )
            if after_execution_attempt is not None:
                after_execution_attempt(result)
            final_result = result
            if not result.infrastructure_failure or retry_ordinal >= max_infrastructure_retries:
                break
        if final_result is not None and after_task_run is not None:
            after_task_run(final_result)
    return storage.list_results(experiment.id)
