"""Strict, no-preflight resume support for a stopped H2.2 campaign."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from evaluation.campaign import (
    CAMPAIGN_CONFIG_KEYS,
    PHASE_ARTIFACT_NAMES,
    BenchmarkCampaign,
    CampaignError,
    CampaignRunOutcome,
    CampaignStopped,
    _assert_operational_phase,
    _run_phase,
    _set_campaign_status,
    _StopConditionTracker,
    _write_json,
    build_campaign_experiments,
    build_campaign_manifest,
    freeze_local_tasks,
    select_pilot_tasks,
    select_smoke_task,
    source_snapshots,
)
from evaluation.campaign_reports import write_campaign_reports
from evaluation.live import LivePreflightResult, LiveRunBudget, prepare_live_environment
from evaluation.models import EvaluationExperiment, EvaluationResult, EvaluationTask
from evaluation.storage import EvaluationStorage
from model_defaults import ProductionModelSettings

CAMPAIGN_PHASES = ("smoke", "pilot", *CAMPAIGN_CONFIG_KEYS)
_MANIFEST_IDENTITY_FIELDS = (
    "dataset",
    "task_ids",
    "tasks",
    "selection_rule",
    "configs",
    "configured_model",
    "configured_provider",
    "api_base_url",
    "reasoning_effort",
    "max_completion_tokens",
    "model_temperature",
    "model_seed",
    "provider_seed_controlled",
    "evaluator",
    "task_timeout_seconds",
    "evaluator_timeout_seconds",
    "max_correction_rounds",
    "workers",
    "llm_execution_kind",
    "bootstrap_seed",
    "bootstrap_resamples",
    "planned_live_task_runs",
    "repograph_commit",
)


def _phase_tasks(
    frozen: Sequence[EvaluationTask],
) -> dict[str, list[EvaluationTask]]:
    all_tasks = list(frozen)
    return {
        "smoke": [select_smoke_task(all_tasks)],
        "pilot": select_pilot_tasks(all_tasks),
        "full": all_tasks,
        "no_correction": all_tasks,
        "no_exploration": all_tasks,
        "no_agentic_test": all_tasks,
    }


def _validate_manifest_identity(
    actual: BenchmarkCampaign,
    expected: BenchmarkCampaign,
) -> None:
    mismatches = [
        field
        for field in _MANIFEST_IDENTITY_FIELDS
        if getattr(actual, field) != getattr(expected, field)
    ]
    if mismatches:
        raise CampaignError(
            "Campaign resume identity mismatch: " + ", ".join(mismatches) + "."
        )
    if set(actual.experiment_ids) != set(CAMPAIGN_PHASES):
        raise CampaignError(
            "Campaign resume experiment IDs are incomplete or unexpected."
        )
    if actual.status == "complete":
        raise CampaignError("A complete campaign cannot be resumed.")


def _load_experiments(
    manifest: BenchmarkCampaign,
    expected: dict[str, EvaluationExperiment],
    storage: EvaluationStorage,
) -> dict[str, EvaluationExperiment]:
    experiments: dict[str, EvaluationExperiment] = {}
    for phase in CAMPAIGN_PHASES:
        persisted = storage.get_experiment(manifest.experiment_ids[phase])
        template = expected[phase]
        if persisted is None:
            persisted = template.model_copy(
                update={"id": manifest.experiment_ids[phase]}
            )
            storage.save_experiment(persisted)
        if (
            persisted.name != template.name
            or persisted.dataset != template.dataset
            or persisted.config != template.config
            or persisted.git_commit != template.git_commit
            or persisted.model_name != template.model_name
            or persisted.llm_execution_kind != "live"
        ):
            raise CampaignError(
                f"Campaign resume experiment identity changed: {phase}."
            )
        experiments[phase] = persisted
    return experiments


def _load_preflight(
    path: Path,
    settings: ProductionModelSettings,
) -> LivePreflightResult:
    if not path.is_file():
        raise CampaignError("Campaign resume preflight artifact is missing.")
    try:
        preflight = LivePreflightResult.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise CampaignError("Campaign resume preflight artifact is invalid.") from error
    expected = (
        settings.provider,
        settings.base_url,
        settings.model,
        settings.temperature,
        settings.seed,
        settings.reasoning_effort,
        settings.max_completion_tokens,
    )
    actual = (
        preflight.configured_provider,
        preflight.api_base_url,
        preflight.configured_model,
        preflight.model_temperature,
        preflight.model_seed,
        preflight.reasoning_effort,
        preflight.max_completion_tokens,
    )
    if actual != expected or not preflight.structured_output_succeeded:
        raise CampaignError("Campaign resume preflight identity changed.")
    return preflight


def _missing_task_runs(
    phase_tasks: dict[str, list[EvaluationTask]],
    experiments: dict[str, EvaluationExperiment],
    storage: EvaluationStorage,
    budget: LiveRunBudget,
) -> int:
    valid_runs: dict[tuple[str, str], str] = {}
    persisted_runs: set[tuple[str, str]] = set()
    missing = 0
    for phase in CAMPAIGN_PHASES:
        experiment = experiments[phase]
        expected_task_ids = {task.id for task in phase_tasks[phase]}
        persisted_results = storage.list_results(experiment.id)
        persisted_task_ids = {result.task_id for result in persisted_results}
        if not persisted_task_ids.issubset(expected_task_ids):
            raise CampaignError(f"Campaign resume has unexpected results: {phase}.")
        for task in phase_tasks[phase]:
            identity = (experiment.id, task.id)
            valid_runs[identity] = experiment.config.name
            if task.id in persisted_task_ids:
                persisted_runs.add(identity)
            else:
                missing += 1

    reserved_runs: set[tuple[str, str]] = set()
    for reservation in budget.reservations():
        identity = (reservation.experiment_id, reservation.task_id)
        if valid_runs.get(identity) != reservation.config_name:
            raise CampaignError("Campaign resume budget ledger identity changed.")
        reserved_runs.add(identity)
    if not persisted_runs.issubset(reserved_runs):
        raise CampaignError(
            "Campaign resume result lacks a matching budget reservation."
        )
    if budget.consumed + missing > budget.maximum:
        raise CampaignError(
            "Campaign resume would exceed the remaining live task-run budget."
        )
    return missing


def _run_resumed_campaign(
    tasks: Sequence[EvaluationTask],
    *,
    dataset: str,
    workspace_root: str | Path,
    output_root: str | Path,
    maximum_live_task_runs: int,
    task_timeout_seconds: float,
    evaluator_timeout_seconds: float,
    repograph_commit: str | None,
    model_settings: ProductionModelSettings,
) -> CampaignRunOutcome:
    frozen = freeze_local_tasks(tasks)
    workspace = Path(workspace_root).resolve()
    output = Path(output_root).resolve()
    manifest_path = output / "manifest.json"
    preflight_path = output / "preflight.json"
    if not manifest_path.is_file():
        raise CampaignError("Campaign resume manifest is missing.")
    try:
        manifest = BenchmarkCampaign.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise CampaignError("Campaign resume manifest is invalid.") from error

    storage = EvaluationStorage(workspace / "evaluation.db", workspace / "results")
    budget = LiveRunBudget(
        workspace / "live-task-runs.jsonl",
        maximum=maximum_live_task_runs,
    )
    expected_experiments = build_campaign_experiments(
        dataset,
        task_timeout_seconds=task_timeout_seconds,
        evaluator_timeout_seconds=evaluator_timeout_seconds,
        repograph_commit=repograph_commit,
        model_settings=model_settings,
    )
    expected_manifest = build_campaign_manifest(
        frozen,
        expected_experiments,
        dataset=dataset,
        task_timeout_seconds=task_timeout_seconds,
        evaluator_timeout_seconds=evaluator_timeout_seconds,
        budget_consumed=manifest.live_task_runs_consumed_before,
        repograph_commit=repograph_commit,
        model_settings=model_settings,
    )
    _validate_manifest_identity(manifest, expected_manifest)
    experiments = _load_experiments(manifest, expected_experiments, storage)
    preflight = _load_preflight(preflight_path, model_settings)
    phase_tasks = _phase_tasks(frozen)
    missing = _missing_task_runs(phase_tasks, experiments, storage, budget)

    manifest.status = "running"
    manifest.stop_reason = None
    manifest.live_task_runs_consumed_after = budget.consumed
    _write_json(manifest_path, manifest)
    print(
        f"Resuming stopped campaign with {missing} missing task-runs and "
        f"{budget.maximum - budget.consumed} budget slots remaining.",
        flush=True,
    )

    source_state = source_snapshots(frozen)
    tracker = _StopConditionTracker(tasks=frozen, source_state=source_state)
    artifact_paths = [str(preflight_path)]
    results_by_phase: dict[str, list[EvaluationResult]] = {}
    for phase in CAMPAIGN_PHASES:
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
    manifest.resolved_models = sorted(
        set(manifest.resolved_models).union(
            model
            for results in results_by_phase.values()
            for result in results
            for model in result.resolved_model_names
        )
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


def resume_live_campaign(
    tasks: Sequence[EvaluationTask],
    *,
    dataset: str,
    workspace_root: str | Path,
    output_root: str | Path,
    maximum_live_task_runs: int,
    task_timeout_seconds: float,
    evaluator_timeout_seconds: float,
    repograph_commit: str | None = None,
    model_settings: ProductionModelSettings | None = None,
) -> CampaignRunOutcome:
    """Resume only missing task-runs without repeating the provider preflight."""

    selected_settings = model_settings or prepare_live_environment()
    try:
        outcome = _run_resumed_campaign(
            tasks,
            dataset=dataset,
            workspace_root=workspace_root,
            output_root=output_root,
            maximum_live_task_runs=maximum_live_task_runs,
            task_timeout_seconds=task_timeout_seconds,
            evaluator_timeout_seconds=evaluator_timeout_seconds,
            repograph_commit=repograph_commit,
            model_settings=selected_settings,
        )
    except BaseException as error:
        manifest_path = Path(output_root).resolve() / "manifest.json"
        try:
            manifest = BenchmarkCampaign.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            manifest = None
        if manifest is not None and manifest.status == "running":
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
