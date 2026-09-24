"""Stage H2.2 sequential live benchmark campaign orchestration."""

from __future__ import annotations

import json
import subprocess  # nosec B404
import uuid
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evaluation.campaign_reports import write_campaign_reports
from evaluation.experiments.configs import ablation_specs
from evaluation.live import (
    DEFAULT_LIVE_TASK_TIMEOUT_SECONDS,
    LivePreflightResult,
    LiveRunBudget,
    prepare_live_environment,
    run_live_preflight,
    utc_now,
)
from evaluation.metrics import ExperimentMetrics, aggregate_metrics
from evaluation.models import (
    EvaluationConfig,
    EvaluationExperiment,
    EvaluationResult,
    EvaluationTask,
)
from evaluation.runner import create_experiment, run_experiment
from evaluation.security import redact_environment_secrets
from evaluation.statistics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
)
from evaluation.storage import EvaluationStorage
from model_defaults import (
    ProductionModelSettings,
    get_production_model_settings,
)

CAMPAIGN_CONFIG_KEYS = (
    "full",
    "no_correction",
    "no_exploration",
    "no_agentic_test",
)
PHASE_ARTIFACT_NAMES = {
    "smoke": "smoke.json",
    "pilot": "pilot.json",
    "full": "full.json",
    "no_correction": "no_correction.json",
    "no_exploration": "no_exploration.json",
    "no_agentic_test": "no_agentic_test.json",
}


class CampaignError(RuntimeError):
    """Raised before or during a campaign whose integrity cannot be preserved."""


class CampaignStopped(CampaignError):
    """Raised after a mandatory infrastructure stop condition is observed."""


class FrozenTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    repository: str
    base_commit: str
    category: str | None = None


class BenchmarkCampaign(BaseModel):
    """Strict, reproducible manifest for one H2.2 local live campaign."""

    model_config = ConfigDict(extra="forbid")

    id: str
    dataset: str
    task_ids: list[str]
    tasks: list[FrozenTask]
    selection_rule: str
    configs: list[str]
    experiment_ids: dict[str, str]
    configured_model: str
    configured_provider: str = "openai"
    api_base_url: str | None = None
    reasoning_effort: str | None = None
    max_completion_tokens: int | None = Field(default=None, ge=1)
    resolved_models: list[str] = Field(default_factory=list)
    model_temperature: float | None
    model_seed: int | None
    provider_seed_controlled: bool = False
    evaluator: Literal["local"] = "local"
    task_timeout_seconds: float
    evaluator_timeout_seconds: float
    max_correction_rounds: int
    workers: Literal[1] = 1
    llm_execution_kind: Literal["live"] = "live"
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    planned_live_task_runs: int = 44
    live_task_runs_consumed_before: int = 0
    live_task_runs_consumed_after: int = 0
    repograph_commit: str | None = None
    created_at: str
    status: Literal["running", "complete", "stopped"] = "running"
    stop_reason: str | None = None


class PhaseArtifact(BaseModel):
    """Bounded shareable phase summary; raw patches remain in harness storage."""

    model_config = ConfigDict(extra="forbid")

    phase: str
    experiment: EvaluationExperiment
    metrics: ExperimentMetrics
    results: list[dict[str, object]]


class CampaignRunOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_path: str
    artifact_paths: list[str]
    preflight: LivePreflightResult
    experiment_ids: dict[str, str]
    live_task_runs_consumed: int


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _git_output(repository: str, *args: str) -> str:
    completed = subprocess.run(  # nosec B603 B607
        ["git", *args],
        cwd=repository,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        raise CampaignError(
            f"Could not inspect benchmark source repository: {completed.stderr.strip()[:500]}"
        )
    return completed.stdout


def source_snapshots(tasks: Sequence[EvaluationTask]) -> dict[str, tuple[str, str]]:
    """Capture HEAD and dirty-state text without reading repository secrets."""

    snapshots: dict[str, tuple[str, str]] = {}
    for task in tasks:
        if task.repository in snapshots:
            continue
        head = _git_output(task.repository, "rev-parse", "HEAD").strip()
        if head != task.base_commit:
            raise CampaignError(f"Source HEAD/base commit mismatch for task {task.id}.")
        status = _git_output(task.repository, "status", "--porcelain=v1")
        snapshots[task.repository] = (head, status)
    return snapshots


def assert_source_snapshots(
    tasks: Sequence[EvaluationTask],
    expected: dict[str, tuple[str, str]],
) -> None:
    if source_snapshots(tasks) != expected:
        raise CampaignStopped("Benchmark source repository mutation detected.")


def freeze_local_tasks(tasks: Sequence[EvaluationTask]) -> list[EvaluationTask]:
    """Freeze the full ten-task local set in deterministic task-id order."""

    selected = sorted(tasks, key=lambda item: item.id)
    if len(selected) != 10:
        raise CampaignError(
            f"H2.2 local campaign requires exactly 10 tasks; received {len(selected)}."
        )
    if len({item.id for item in selected}) != len(selected):
        raise CampaignError("H2.2 task IDs must be unique.")
    if len({item.dataset for item in selected}) != 1:
        raise CampaignError("H2.2 tasks must belong to one dataset.")
    return selected


def _category(task: EvaluationTask) -> str | None:
    value = task.metadata.get("category")
    return value if isinstance(value, str) else None


def select_smoke_task(tasks: Sequence[EvaluationTask]) -> EvaluationTask:
    """Select the first sorted multi-file task as a deterministic non-trivial smoke."""

    matches = sorted(
        (item for item in tasks if _category(item) == "multi_file_fix"),
        key=lambda item: item.id,
    )
    if not matches:
        raise CampaignError("No deterministic multi-file smoke task is available.")
    return matches[0]


def select_pilot_tasks(tasks: Sequence[EvaluationTask]) -> list[EvaluationTask]:
    """Select one sorted task from each required pilot behavior category."""

    groups = (
        ("single-file bug", {"bug_fix"}),
        ("multi-file change", {"multi_file_fix"}),
        ("behavior/test regression", {"test_regression", "behavioral_refactor"}),
    )
    selected: list[EvaluationTask] = []
    for label, categories in groups:
        candidates = sorted(
            (item for item in tasks if _category(item) in categories),
            key=lambda item: item.id,
        )
        candidate = next(
            (item for item in candidates if item.id not in {x.id for x in selected}),
            None,
        )
        if candidate is None:
            raise CampaignError(f"No deterministic pilot task for {label}.")
        selected.append(candidate)
    return selected


def _live_config(
    source: EvaluationConfig,
    *,
    name: str,
    task_timeout_seconds: float,
    evaluator_timeout_seconds: float,
    model_settings: ProductionModelSettings,
) -> EvaluationConfig:
    return source.model_copy(
        update={
            "name": name,
            "model_name": model_settings.model,
            "model_temperature": model_settings.temperature,
            "model_seed": model_settings.seed,
            "llm_execution_kind": "live",
            "model_provider": model_settings.provider,
            "model_base_url": model_settings.base_url,
            "model_reasoning_effort": model_settings.reasoning_effort,
            "model_max_completion_tokens": model_settings.max_completion_tokens,
            "task_timeout_seconds": task_timeout_seconds,
            "overall_timeout_seconds": None,
            "evaluator_timeout_seconds": evaluator_timeout_seconds,
            "max_tasks": None,
        }
    )


def build_campaign_experiments(
    dataset: str,
    *,
    task_timeout_seconds: float = DEFAULT_LIVE_TASK_TIMEOUT_SECONDS,
    evaluator_timeout_seconds: float = 300.0,
    repograph_commit: str | None = None,
    model_settings: ProductionModelSettings | None = None,
) -> dict[str, EvaluationExperiment]:
    selected_settings = model_settings or get_production_model_settings({})
    specs = {spec.key: spec for spec in ablation_specs()}
    source_by_phase = {
        "smoke": specs["A"].config,
        "pilot": specs["A"].config,
        "full": specs["A"].config,
        "no_correction": specs["B"].config,
        "no_exploration": specs["C"].config,
        "no_agentic_test": specs["D"].config,
    }
    return {
        phase: create_experiment(
            f"h2-2-{phase.replace('_', '-')}",
            dataset,
            _live_config(
                source,
                name=phase,
                task_timeout_seconds=task_timeout_seconds,
                evaluator_timeout_seconds=evaluator_timeout_seconds,
                model_settings=selected_settings,
            ),
            git_commit=repograph_commit,
        )
        for phase, source in source_by_phase.items()
    }


def build_campaign_manifest(
    tasks: Sequence[EvaluationTask],
    experiments: dict[str, EvaluationExperiment],
    *,
    dataset: str,
    task_timeout_seconds: float,
    evaluator_timeout_seconds: float,
    budget_consumed: int,
    repograph_commit: str | None,
    model_settings: ProductionModelSettings | None = None,
) -> BenchmarkCampaign:
    selected_settings = model_settings or get_production_model_settings({})
    frozen = freeze_local_tasks(tasks)
    return BenchmarkCampaign(
        id=f"h2-2-local-{uuid.uuid4().hex[:12]}",
        dataset=dataset,
        task_ids=[item.id for item in frozen],
        tasks=[
            FrozenTask(
                task_id=item.id,
                repository=item.repository,
                base_commit=item.base_commit,
                category=_category(item),
            )
            for item in frozen
        ],
        selection_rule=(
            "All ten local-v1 tasks sorted by task ID; smoke is the first sorted "
            "multi_file_fix; pilot is the first sorted distinct task in bug_fix, "
            "multi_file_fix, and test_regression/behavioral_refactor."
        ),
        configs=list(CAMPAIGN_CONFIG_KEYS),
        experiment_ids={key: value.id for key, value in experiments.items()},
        configured_model=selected_settings.model,
        configured_provider=selected_settings.provider,
        api_base_url=selected_settings.base_url,
        model_temperature=selected_settings.temperature,
        model_seed=selected_settings.seed,
        reasoning_effort=selected_settings.reasoning_effort,
        max_completion_tokens=selected_settings.max_completion_tokens,
        task_timeout_seconds=task_timeout_seconds,
        evaluator_timeout_seconds=evaluator_timeout_seconds,
        max_correction_rounds=2,
        live_task_runs_consumed_before=budget_consumed,
        live_task_runs_consumed_after=budget_consumed,
        repograph_commit=repograph_commit,
        created_at=utc_now(),
    )


def _phase_artifact(
    phase: str,
    experiment: EvaluationExperiment,
    results: list[EvaluationResult],
) -> PhaseArtifact:
    return PhaseArtifact(
        phase=phase,
        experiment=experiment,
        metrics=aggregate_metrics(results),
        results=[
            result.model_dump(mode="json", exclude={"model_patch"})
            for result in sorted(results, key=lambda item: item.task_id)
        ],
    )


def _write_phase(
    output_root: Path,
    phase: str,
    experiment: EvaluationExperiment,
    results: list[EvaluationResult],
) -> Path:
    return _write_json(
        output_root / PHASE_ARTIFACT_NAMES[phase],
        _phase_artifact(phase, experiment, results),
    )


def _worker_ipc_is_clean(workspace_root: Path) -> bool:
    ipc = workspace_root / ".worker-ipc"
    return not ipc.exists() or not any(ipc.iterdir())


class _StopConditionTracker:
    def __init__(
        self,
        *,
        tasks: Sequence[EvaluationTask],
        source_state: dict[str, tuple[str, str]],
    ) -> None:
        self.tasks = tasks
        self.source_state = source_state
        self.consecutive_api_or_worker_failures = 0
        self.consecutive_timeouts = 0

    def observe(self, result: EvaluationResult) -> None:
        if result.llm_execution_kind != "live":
            raise CampaignStopped("Non-live result entered the live campaign.")
        if result.llm_calls == 0:
            raise CampaignStopped(
                "A live task recorded llm_calls=0; fake or uninstrumented path detected."
            )
        if (
            not result.telemetry_incomplete
            and result.input_tokens is not None
            and result.output_tokens is not None
            and result.total_tokens != result.input_tokens + result.output_tokens
        ):
            raise CampaignStopped("Live token telemetry failed its accounting check.")
        infrastructure = (
            result.status == "evaluation_error"
            or result.failure_category
            in {
                "api_infrastructure_failure",
                "infrastructure_error",
            }
        )
        self.consecutive_api_or_worker_failures = (
            self.consecutive_api_or_worker_failures + 1 if infrastructure else 0
        )
        self.consecutive_timeouts = (
            self.consecutive_timeouts + 1 if result.hard_timeout else 0
        )
        assert_source_snapshots(self.tasks, self.source_state)
        if self.consecutive_api_or_worker_failures >= 3:
            raise CampaignStopped(
                "Three consecutive API/worker infrastructure failures were observed."
            )
        if self.consecutive_timeouts >= 3:
            raise CampaignStopped("Three consecutive hard task timeouts were observed.")


def _assert_operational_phase(
    phase: str,
    results: list[EvaluationResult],
    *,
    expected_tasks: int,
) -> None:
    if len(results) != expected_tasks:
        raise CampaignStopped(
            f"{phase} produced {len(results)}/{expected_tasks} required results."
        )
    for result in results:
        if not result.process_isolated:
            raise CampaignStopped(f"{phase} result was not process-isolated.")
        if result.llm_calls is None or result.llm_calls < 1:
            raise CampaignStopped(
                f"{phase} result lacks trustworthy live LLM telemetry."
            )
        if result.provenance is None or result.provenance.llm_execution_kind != "live":
            raise CampaignStopped(f"{phase} result lacks live provenance.")


def _run_phase(
    phase: str,
    tasks: Sequence[EvaluationTask],
    experiment: EvaluationExperiment,
    *,
    storage: EvaluationStorage,
    workspace_root: Path,
    output_root: Path,
    budget: LiveRunBudget,
    tracker: _StopConditionTracker,
) -> list[EvaluationResult]:
    def reserve_live_run(
        current: EvaluationExperiment,
        task: EvaluationTask,
    ) -> None:
        reservation = budget.reserve(
            experiment_id=current.id,
            task_id=task.id,
            config_name=current.config.name,
        )
        print(
            f"Starting live task-run {reservation.ordinal}/{budget.maximum}: "
            f"{phase}/{task.id}",
            flush=True,
        )

    try:
        results = run_experiment(
            tasks,
            experiment,
            workspace_root=str(workspace_root),
            storage=storage,
            before_task_run=reserve_live_run,
            after_task_run=tracker.observe,
        )
    except CampaignStopped:
        _write_phase(
            output_root,
            phase,
            experiment,
            storage.list_results(experiment.id),
        )
        raise
    _write_phase(output_root, phase, experiment, results)
    if not _worker_ipc_is_clean(workspace_root):
        raise CampaignStopped(f"{phase} left worker IPC residue.")
    return results


def _run_live_campaign(
    tasks: Sequence[EvaluationTask],
    *,
    dataset: str,
    workspace_root: str | Path,
    output_root: str | Path,
    maximum_live_task_runs: int,
    task_timeout_seconds: float = DEFAULT_LIVE_TASK_TIMEOUT_SECONDS,
    evaluator_timeout_seconds: float = 300.0,
    repograph_commit: str | None = None,
    model_settings: ProductionModelSettings | None = None,
) -> CampaignRunOutcome:
    """Run preflight, smoke, pilot, and four paired configs sequentially."""
    selected_settings = model_settings or prepare_live_environment()

    frozen = freeze_local_tasks(tasks)
    workspace = Path(workspace_root).resolve()
    output = Path(output_root).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    storage = EvaluationStorage(workspace / "evaluation.db", workspace / "results")
    budget = LiveRunBudget(
        workspace / "live-task-runs.jsonl",
        maximum=maximum_live_task_runs,
    )
    if budget.consumed + 44 > maximum_live_task_runs:
        raise CampaignError(
            "The planned 44 live task-runs exceed the remaining campaign budget."
        )
    experiments = build_campaign_experiments(
        dataset,
        task_timeout_seconds=task_timeout_seconds,
        evaluator_timeout_seconds=evaluator_timeout_seconds,
        repograph_commit=repograph_commit,
        model_settings=selected_settings,
    )
    manifest = build_campaign_manifest(
        frozen,
        experiments,
        dataset=dataset,
        task_timeout_seconds=task_timeout_seconds,
        evaluator_timeout_seconds=evaluator_timeout_seconds,
        budget_consumed=budget.consumed,
        repograph_commit=repograph_commit,
        model_settings=selected_settings,
    )
    manifest_path = _write_json(output / "manifest.json", manifest)
    _write_json(
        output.parent / "h2_2_local_tasks.json",
        {
            "campaign_id": manifest.id,
            "dataset": dataset,
            "selection_rule": manifest.selection_rule,
            "evaluator": manifest.evaluator,
            "task_timeout_seconds": task_timeout_seconds,
            "tasks": [item.model_dump(mode="json") for item in manifest.tasks],
        },
    )

    preflight = run_live_preflight(settings=selected_settings)
    preflight_path = _write_json(output / "preflight.json", preflight)
    manifest.resolved_models = preflight.resolved_models
    _write_json(manifest_path, manifest)

    source_state = source_snapshots(frozen)
    tracker = _StopConditionTracker(tasks=frozen, source_state=source_state)
    artifact_paths = [str(preflight_path)]
    phase_tasks = {
        "smoke": [select_smoke_task(frozen)],
        "pilot": select_pilot_tasks(frozen),
        "full": frozen,
        "no_correction": frozen,
        "no_exploration": frozen,
        "no_agentic_test": frozen,
    }
    results_by_phase: dict[str, list[EvaluationResult]] = {}
    for phase in (
        "smoke",
        "pilot",
        "full",
        "no_correction",
        "no_exploration",
        "no_agentic_test",
    ):
        results = _run_phase(
            phase,
            phase_tasks[phase],
            experiments[phase],
            storage=storage,
            workspace_root=workspace / "workspaces",
            output_root=output,
            budget=budget,
            tracker=tracker,
        )
        results_by_phase[phase] = results
        artifact_paths.append(str(output / PHASE_ARTIFACT_NAMES[phase]))
        if phase in {"smoke", "pilot"}:
            _assert_operational_phase(
                phase,
                results,
                expected_tasks=len(phase_tasks[phase]),
            )
        if phase == "pilot" and all(
            item.status == "evaluation_error" for item in results
        ):
            raise CampaignStopped("All three pilot tasks had infrastructure errors.")

    report_paths = write_campaign_reports(
        output,
        {key: results_by_phase[key] for key in CAMPAIGN_CONFIG_KEYS},
    )
    artifact_paths.extend(str(path) for path in report_paths)
    resolved_models = {
        model
        for results in results_by_phase.values()
        for result in results
        for model in result.resolved_model_names
    }
    manifest.resolved_models = sorted(
        set(manifest.resolved_models).union(resolved_models)
    )
    manifest.live_task_runs_consumed_after = budget.consumed
    _write_json(manifest_path, manifest)
    return CampaignRunOutcome(
        manifest_path=str(manifest_path),
        artifact_paths=artifact_paths,
        preflight=preflight,
        experiment_ids=manifest.experiment_ids,
        live_task_runs_consumed=budget.consumed,
    )


def _set_campaign_status(
    *,
    workspace_root: str | Path,
    output_root: str | Path,
    maximum_live_task_runs: int,
    status: Literal["complete", "stopped"],
    stop_reason: str | None = None,
) -> None:
    """Best-effort final status persistence backed by the append-only ledger."""

    manifest_path = Path(output_root).resolve() / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = BenchmarkCampaign.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        budget = LiveRunBudget(
            Path(workspace_root).resolve() / "live-task-runs.jsonl",
            maximum=maximum_live_task_runs,
        )
        manifest.live_task_runs_consumed_after = budget.consumed
    except (OSError, ValueError):
        return
    manifest.status = status
    manifest.stop_reason = (
        redact_environment_secrets(stop_reason)[:1000] if stop_reason else None
    )
    _write_json(manifest_path, manifest)


def run_live_campaign(
    tasks: Sequence[EvaluationTask],
    *,
    dataset: str,
    workspace_root: str | Path,
    output_root: str | Path,
    maximum_live_task_runs: int,
    task_timeout_seconds: float = DEFAULT_LIVE_TASK_TIMEOUT_SECONDS,
    evaluator_timeout_seconds: float = 300.0,
    repograph_commit: str | None = None,
    model_settings: ProductionModelSettings | None = None,
) -> CampaignRunOutcome:
    """Run the H2.2 campaign and durably record completion or interruption."""

    try:
        outcome = _run_live_campaign(
            tasks,
            dataset=dataset,
            workspace_root=workspace_root,
            output_root=output_root,
            maximum_live_task_runs=maximum_live_task_runs,
            task_timeout_seconds=task_timeout_seconds,
            evaluator_timeout_seconds=evaluator_timeout_seconds,
            repograph_commit=repograph_commit,
            model_settings=model_settings,
        )
    except BaseException as error:
        _set_campaign_status(
            workspace_root=workspace_root,
            output_root=output_root,
            maximum_live_task_runs=maximum_live_task_runs,
            status="stopped",
            stop_reason=str(error).strip() or type(error).__name__,
        )
        raise
    _set_campaign_status(
        workspace_root=workspace_root,
        output_root=output_root,
        maximum_live_task_runs=maximum_live_task_runs,
        status="complete",
    )
    return outcome


def failure_counts(results: Sequence[EvaluationResult]) -> dict[str, int]:
    """Expose deterministic bounded counts for campaign report generation."""

    return dict(
        sorted(Counter(item.failure_category or "none" for item in results).items())
    )
