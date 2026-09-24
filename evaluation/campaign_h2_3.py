"""Stage H2.3 controlled harder-benchmark campaign orchestration."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evaluation.artifact_security import (
    assert_secret_free_payload,
    scan_persisted_artifacts,
)
from evaluation.campaign import (
    CampaignError,
    CampaignStopped,
    assert_source_snapshots,
    source_snapshots,
)
from evaluation.candidate_artifacts import CandidateArtifactStore
from evaluation.evaluator import (
    LOCAL_EVALUATOR_VERSION,
    build_local_evaluator_specification,
    evaluator_specification_digest,
    preflight_local_evaluator,
    run_local_evaluator,
)
from evaluation.experiments.configs import ablation_specs
from evaluation.h2_3_reports import CONFIG_ORDER, write_h2_3_reports
from evaluation.live import (
    DEFAULT_LIVE_TASK_TIMEOUT_SECONDS,
    MAX_H2_3_LIVE_TASK_RUNS,
    LivePreflightResult,
    LiveRunBudget,
    prepare_live_environment,
    run_live_preflight,
    utc_now,
)
from evaluation.metrics import aggregate_metrics
from evaluation.models import (
    EvaluationConfig,
    EvaluationExperiment,
    EvaluationResult,
    EvaluationTask,
    EvaluatorEnvironmentFingerprint,
    EvaluatorSpecification,
)
from evaluation.runner import create_experiment, run_experiment
from evaluation.security import redact_environment_secrets, redact_secret_values
from evaluation.source_digest import source_tree_digest
from evaluation.storage import EvaluationStorage
from evaluation.workspace import prepare_task_workspace
from model_defaults import ProductionModelSettings

H2_3_FORMAL_TASKS = 17
H2_3_SMOKE_TASKS = 2
H2_3_PLANNED_BASE_TASK_RUNS = H2_3_SMOKE_TASKS + H2_3_FORMAL_TASKS * 4
H2_3_RETRY_RESERVE = MAX_H2_3_LIVE_TASK_RUNS - H2_3_PLANNED_BASE_TASK_RUNS
H2_3_MODEL = "gpt-5.6-sol"
H2_3_TEMPERATURE = 0.0
H2_3_REASONING_EFFORT = "none"
H2_3_MAX_COMPLETION_TOKENS = 8192


class H23FrozenTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    repo_id: str
    base_commit: str
    issue_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    difficulty: list[str]
    repository_file_count: int = Field(ge=5, le=20)


class H23CampaignManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    stage: Literal["H2.3"] = "H2.3"
    dataset: str
    formal_task_ids: list[str]
    smoke_task_ids: list[str]
    tasks: list[H23FrozenTask]
    configs: list[str]
    experiment_ids: dict[str, str]
    configured_model: str
    configured_provider: str
    api_base_url: str | None = None
    model_temperature: float | None
    model_seed: int | None
    provider_seed_controlled: bool
    reasoning_effort: str | None
    max_completion_tokens: int | None
    workers: Literal[1] = 1
    evaluator_version: str
    python_executable: str
    task_timeout_seconds: float
    evaluator_timeout_seconds: float
    max_infrastructure_retries: Literal[1] = 1
    planned_base_task_runs: Literal[70] = H2_3_PLANNED_BASE_TASK_RUNS
    retry_reserve: Literal[10] = H2_3_RETRY_RESERVE
    maximum_live_task_runs: int = Field(ge=70, le=MAX_H2_3_LIVE_TASK_RUNS)
    live_task_runs_consumed_before: int = Field(ge=0)
    live_task_runs_consumed_after: int = Field(ge=0)
    source_tree_digest_before: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_tree_digest_after: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    repograph_commit: str | None = None
    resolved_models: list[str] = Field(default_factory=list)
    created_at: str
    status: Literal["running", "complete", "stopped"] = "running"
    stop_reason: str | None = None


class H23CampaignOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_path: str
    artifact_paths: list[str]
    preflight: LivePreflightResult
    experiment_ids: dict[str, str]
    live_task_runs_consumed: int


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_payload(value: object) -> object:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return redact_secret_values(value)


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = _json_payload(value)
    rendered = json.dumps(safe, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    assert_secret_free_payload(rendered)
    path.write_text(rendered, encoding="utf-8")
    return path


def _difficulty(task: EvaluationTask) -> list[str]:
    value = task.metadata.get("difficulty")
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise CampaignError(f"H2.3 task {task.id} lacks strict difficulty metadata.")
    return list(value)


def _repository_file_count(task: EvaluationTask) -> int:
    value = task.metadata.get("repository_file_count")
    if not isinstance(value, int) or not 5 <= value <= 20:
        raise CampaignError(f"H2.3 task {task.id} has invalid repository file count.")
    return value


def _hidden_test_path(task: EvaluationTask) -> Path:
    if task.test_command is None:
        raise CampaignError(f"H2.3 task {task.id} lacks an evaluator command.")
    matches = [
        Path(item).resolve()
        for item in task.test_command
        if "evaluator-only" in item.replace("\\", "/").casefold()
        and "hidden-tests" in item.replace("\\", "/").casefold()
    ]
    if len(matches) != 1 or not matches[0].exists():
        raise CampaignError(
            f"H2.3 task {task.id} has no isolated hidden evaluator file."
        )
    repository = Path(task.repository).resolve(strict=True)
    if matches[0].is_relative_to(repository):
        raise CampaignError(
            f"H2.3 task {task.id} exposes hidden tests in its repository."
        )
    return matches[0]


def freeze_h2_3_tasks(
    tasks: Sequence[EvaluationTask],
) -> tuple[list[EvaluationTask], list[EvaluationTask]]:
    """Freeze 17 paired tasks and two disjoint smoke tasks from exactly 20 fixtures."""

    selected = sorted(
        (EvaluationTask.model_validate(item) for item in tasks), key=lambda x: x.id
    )
    if len(selected) != 20:
        raise CampaignError(
            f"H2.3 benchmark requires exactly 20 tasks; received {len(selected)}."
        )
    if len({item.id for item in selected}) != 20:
        raise CampaignError("H2.3 task IDs must be unique.")
    if len({item.dataset for item in selected}) != 1:
        raise CampaignError("H2.3 tasks must belong to one dataset.")
    for task in selected:
        _difficulty(task)
        _repository_file_count(task)
        _hidden_test_path(task)
    multi_file = sum(
        bool(item.metadata.get("requires_multi_file_exploration")) for item in selected
    )
    if multi_file / len(selected) < 0.70:
        raise CampaignError(
            "H2.3 benchmark does not meet the 70% multi-file threshold."
        )
    return selected[:H2_3_FORMAL_TASKS], selected[
        H2_3_FORMAL_TASKS : H2_3_FORMAL_TASKS + H2_3_SMOKE_TASKS
    ]


def _validate_model_settings(settings: ProductionModelSettings) -> None:
    expected = (
        H2_3_MODEL,
        H2_3_TEMPERATURE,
        H2_3_REASONING_EFFORT,
        H2_3_MAX_COMPLETION_TOKENS,
    )
    actual = (
        settings.model,
        settings.temperature,
        settings.reasoning_effort,
        settings.max_completion_tokens,
    )
    if settings.provider != "openai_compatible" or actual != expected:
        raise CampaignError(
            "H2.3 requires the frozen gpt-5.6-sol/temperature-0/"
            "reasoning-none/max-8192 OpenAI-compatible configuration."
        )


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


def build_h2_3_experiments(
    dataset: str,
    *,
    task_timeout_seconds: float,
    evaluator_timeout_seconds: float,
    repograph_commit: str | None,
    model_settings: ProductionModelSettings,
) -> dict[str, EvaluationExperiment]:
    specs = {spec.key: spec for spec in ablation_specs()}
    sources = {
        "smoke": specs["A"].config,
        "full": specs["A"].config,
        "no_correction": specs["B"].config,
        "no_exploration": specs["C"].config,
        "no_agentic_test": specs["D"].config,
    }
    return {
        name: create_experiment(
            f"h2-3-{name.replace('_', '-')}",
            dataset,
            _live_config(
                source,
                name=name,
                task_timeout_seconds=task_timeout_seconds,
                evaluator_timeout_seconds=evaluator_timeout_seconds,
                model_settings=model_settings,
            ),
            git_commit=repograph_commit,
        )
        for name, source in sources.items()
    }


def _evaluator_specifications(
    tasks: Sequence[EvaluationTask],
    *,
    evaluator_timeout_seconds: float,
) -> dict[str, EvaluatorSpecification]:
    specifications = {
        task.id: build_local_evaluator_specification(
            task,
            timeout_seconds=evaluator_timeout_seconds,
        )
        for task in tasks
    }
    interpreters = {item.python_executable for item in specifications.values()}
    if len(interpreters) != 1 or None in interpreters:
        raise CampaignError(
            "H2.3 evaluators do not share one explicit Python interpreter."
        )
    return specifications


def _frozen_task(
    task: EvaluationTask,
    specification: EvaluatorSpecification,
) -> H23FrozenTask:
    return H23FrozenTask(
        task_id=task.id,
        repo_id=Path(task.repository).name,
        base_commit=task.base_commit,
        issue_sha256=_sha256(task.task),
        evaluator_digest=specification.digest,
        difficulty=_difficulty(task),
        repository_file_count=_repository_file_count(task),
    )


def _task_set_payload(
    *,
    campaign_id: str,
    dataset: str,
    formal: Sequence[EvaluationTask],
    smoke: Sequence[EvaluationTask],
    specifications: Mapping[str, EvaluatorSpecification],
) -> dict[str, object]:
    return {
        "campaign_id": campaign_id,
        "dataset": dataset,
        "selection_rule": (
            "Sort all 20 task IDs; use the first 17 for every paired configuration, "
            "the next two for disjoint Full RepoGraph smoke runs, and hold one unused."
        ),
        "formal_task_count": H2_3_FORMAL_TASKS,
        "formal_task_ids": [item.id for item in formal],
        "smoke_task_ids": [item.id for item in smoke],
        "tasks": [
            _frozen_task(item, specifications[item.id]).model_dump(mode="json")
            for item in formal
        ],
    }


def _config_payload(
    experiments: Mapping[str, EvaluationExperiment],
    settings: ProductionModelSettings,
) -> dict[str, object]:
    return {
        "workers": 1,
        "model": settings.model,
        "provider": settings.provider,
        "api_base_url": settings.base_url,
        "temperature": settings.temperature,
        "seed": settings.seed,
        "provider_seed_controlled": settings.seed is not None,
        "reasoning_effort": settings.reasoning_effort,
        "max_completion_tokens": settings.max_completion_tokens,
        "maximum_live_task_runs": MAX_H2_3_LIVE_TASK_RUNS,
        "planned_base_task_runs": H2_3_PLANNED_BASE_TASK_RUNS,
        "retry_reserve": H2_3_RETRY_RESERVE,
        "max_infrastructure_retries": 1,
        "stop_rules": {
            "configuration_infrastructure_rate": (
                ">=30% after at least 5 assigned tasks"
            ),
            "hard_timeout": "immediate",
            "candidate_snapshot_mismatch": "immediate",
            "evaluator_identity_mismatch": "immediate",
            "source_mutation": "immediate",
            "secret_or_hidden_test_leak": "immediate",
            "systemic_evaluator_failure": "immediate",
        },
        "experiments": {
            key: value.model_dump(mode="json") for key, value in experiments.items()
        },
    }


def _evaluator_payload(
    specifications: Mapping[str, EvaluatorSpecification],
    fingerprint: EvaluatorEnvironmentFingerprint | None = None,
) -> dict[str, object]:
    return {
        "version": LOCAL_EVALUATOR_VERSION,
        "preflight_succeeded": fingerprint is not None,
        "preflight_fingerprint": (
            fingerprint.model_dump(mode="json") if fingerprint is not None else None
        ),
        "task_specifications": {
            task_id: specification.model_dump(mode="json")
            for task_id, specification in sorted(specifications.items())
        },
    }


class _IntegrityTracker:
    def __init__(
        self,
        *,
        tasks: Sequence[EvaluationTask],
        source_state: dict[str, tuple[str, str]],
        specifications: Mapping[str, EvaluatorSpecification],
        artifacts_root: Path,
    ) -> None:
        self.tasks = list(tasks)
        self.source_state = source_state
        self.specifications = dict(specifications)
        self.store = CandidateArtifactStore(artifacts_root)
        self.assigned_by_config: Counter[str] = Counter()
        self.infrastructure_by_config: Counter[str] = Counter()

    def observe_attempt(self, result: EvaluationResult) -> None:
        if result.llm_execution_kind != "live" or not result.process_isolated:
            raise CampaignStopped("A campaign result lacks isolated live provenance.")
        expected = self.specifications[result.task_id]
        actual = result.evaluator_specification
        if (
            actual is None
            or evaluator_specification_digest(actual) != expected.digest
            or actual.digest != expected.digest
            or result.evaluator_digest != expected.digest
            or result.provenance is None
            or result.provenance.evaluator_digest != expected.digest
            or result.provenance.python_executable != expected.python_executable
        ):
            raise CampaignStopped("Evaluator identity mismatch detected.")
        if (
            result.evaluator_environment is None
            or result.evaluator_environment.evaluator_digest != expected.digest
        ):
            raise CampaignStopped(
                "Evaluator environment fingerprint mismatch detected."
            )
        if result.hard_timeout:
            raise CampaignStopped("Hard task timeout detected.")
        if result.evaluator_failure_kind == "candidate_snapshot_failure":
            raise CampaignStopped("Candidate snapshot persistence mismatch detected.")
        if result.failure_category == "external_evaluator_failure":
            raise CampaignStopped(
                "Systemic evaluator failure detected: "
                + str(result.evaluator_failure_kind or "unknown")
            )
        if not result.infrastructure_failure and (
            result.llm_calls is None or result.llm_calls < 1
        ):
            raise CampaignStopped(
                "A non-infrastructure live result recorded no LLM call."
            )
        if (
            not result.telemetry_incomplete
            and result.input_tokens is not None
            and result.output_tokens is not None
            and result.total_tokens != result.input_tokens + result.output_tokens
        ):
            raise CampaignStopped("Live token telemetry failed its accounting check.")
        if result.candidate_generated and not result.candidate_snapshot_paths:
            raise CampaignStopped("A generated candidate has no persisted snapshot.")
        for relative in result.candidate_snapshot_paths:
            snapshot = self.store.load(
                relative,
                expected_task_id=result.task_id,
                expected_experiment_id=result.experiment,
            )
            assert_secret_free_payload(snapshot.model_dump_json())
            if (
                snapshot.base_commit != result.base_commit
                or snapshot.evaluator_digest != expected.digest
                or snapshot.model_name != result.configured_model_name
            ):
                raise CampaignStopped(
                    "Candidate snapshot provenance mismatch detected."
                )
            candidate_paths = [
                item.path.casefold() for item in snapshot.candidate.files
            ]
            lowered_diff = snapshot.diff_text.casefold()
            if any(
                "hidden-tests" in item or "evaluator-only" in item
                for item in candidate_paths
            ) or "evaluator-only/hidden-tests" in lowered_diff.replace("\\", "/"):
                raise CampaignStopped(
                    "Hidden-test leakage detected in candidate artifacts."
                )
        assert_secret_free_payload(result.model_dump_json())
        assert_source_snapshots(self.tasks, self.source_state)

    def observe_final(self, config_name: str, result: EvaluationResult) -> None:
        self.assigned_by_config[config_name] += 1
        self.infrastructure_by_config[config_name] += int(result.infrastructure_failure)
        assigned = self.assigned_by_config[config_name]
        failures = self.infrastructure_by_config[config_name]
        if assigned >= 5 and failures / assigned >= 0.30:
            raise CampaignStopped(
                f"{config_name} campaign infrastructure unstable "
                f"({failures}/{assigned} assigned tasks)."
            )


def _assert_smoke(results: Sequence[EvaluationResult]) -> None:
    if len(results) != H2_3_SMOKE_TASKS:
        raise CampaignStopped("H2.3 smoke did not produce both required results.")
    infrastructure_failures = [
        result.task_id for result in results if result.infrastructure_failure
    ]
    if infrastructure_failures:
        raise CampaignStopped(
            "H2.3 smoke ended with infrastructure failure for "
            + ", ".join(infrastructure_failures)
            + "."
        )
    exact_results = [
        result
        for result in results
        if result.valid_prediction
        and result.first_attempt_evaluated
        and result.final_attempt_evaluated
        and result.candidate_snapshot_paths
    ]
    if not exact_results:
        raise CampaignStopped(
            "H2.3 smoke did not validate exact candidate snapshot grading."
        )


def _preflight_benchmark_evaluators(
    tasks: Sequence[EvaluationTask],
    specifications: Mapping[str, EvaluatorSpecification],
    *,
    workspace_root: Path,
) -> tuple[EvaluatorEnvironmentFingerprint, dict[str, dict[str, object]]]:
    representative = preflight_local_evaluator(specifications[tasks[0].id])
    baseline: dict[str, dict[str, object]] = {}
    for task in tasks:
        workspace, _ = prepare_task_workspace(
            task.repository,
            task.base_commit,
            workspace_root,
            "h2-3-evaluator-preflight",
            task.id,
        )
        outcome = run_local_evaluator(
            task,
            workspace,
            timeout_seconds=specifications[task.id].timeout_seconds,
            max_output_chars=4_000,
            specification=specifications[task.id],
        )
        if outcome.status != "failed" or outcome.failure_kind != "assertion_failure":
            raise CampaignError(
                f"H2.3 hidden evaluator baseline is invalid for {task.id}: "
                f"{outcome.failure_kind or outcome.status}."
            )
        baseline[task.id] = {
            "status": outcome.status,
            "failure_kind": outcome.failure_kind,
            "exit_code": outcome.exit_code,
        }
    return representative, baseline


def _result_record(result: EvaluationResult) -> dict[str, object]:
    return result.model_dump(mode="json", exclude={"model_patch"})


def _write_combined_results(
    output: Path,
    storage: EvaluationStorage,
    experiments: Mapping[str, EvaluationExperiment],
) -> list[Path]:
    structured: dict[str, object] = {"configs": {}, "smoke": {}}
    for name, experiment in experiments.items():
        results = storage.list_results(experiment.id)
        destination = "smoke" if name == "smoke" else "configs"
        structured[destination][name] = {
            "experiment": experiment.model_dump(mode="json"),
            "metrics": aggregate_metrics(results).model_dump(mode="json"),
            "results": [_result_record(item) for item in results],
        }
    json_path = _write_json(output / "results.json", structured)
    jsonl_path = output / "results.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for name, experiment in experiments.items():
            history = storage.results_root / f"{experiment.id}.jsonl"
            if not history.exists():
                continue
            for line in history.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                result = EvaluationResult.model_validate_json(line)
                record = {
                    "config": name,
                    "result": _result_record(result),
                }
                rendered = json.dumps(
                    redact_secret_values(record),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                assert_secret_free_payload(rendered)
                handle.write(rendered + "\n")
    return [json_path, jsonl_path]


def _run_phase(
    name: str,
    tasks: Sequence[EvaluationTask],
    experiment: EvaluationExperiment,
    *,
    storage: EvaluationStorage,
    workspace_root: Path,
    artifacts_root: Path,
    output: Path,
    budget: LiveRunBudget,
    tracker: _IntegrityTracker,
) -> list[EvaluationResult]:
    def reserve(current: EvaluationExperiment, task: EvaluationTask) -> None:
        reservation = budget.reserve(
            experiment_id=current.id,
            task_id=task.id,
            config_name=name,
        )
        print(
            f"Starting H2.3 live task-run {reservation.ordinal}/{budget.maximum}: "
            f"{name}/{task.id}",
            flush=True,
        )

    results = run_experiment(
        tasks,
        experiment,
        workspace_root=str(workspace_root),
        artifacts_root=str(artifacts_root),
        storage=storage,
        max_infrastructure_retries=1,
        before_task_run=reserve,
        after_execution_attempt=tracker.observe_attempt,
        after_task_run=lambda result: tracker.observe_final(name, result),
    )
    _write_json(
        output / f"{name}.json",
        {
            "phase": name,
            "experiment": experiment.model_dump(mode="json"),
            "metrics": aggregate_metrics(results).model_dump(mode="json"),
            "results": [_result_record(item) for item in results],
        },
    )
    scan_persisted_artifacts([artifacts_root, storage.results_root, output])
    return results


def _manifest_from_disk(path: Path) -> H23CampaignManifest | None:
    try:
        return H23CampaignManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _set_manifest_status(
    manifest_path: Path,
    *,
    status: Literal["complete", "stopped"],
    workspace: Path,
    maximum_live_task_runs: int,
    source_root: Path,
    stop_reason: str | None = None,
) -> None:
    manifest = _manifest_from_disk(manifest_path)
    if manifest is None:
        return
    try:
        manifest.live_task_runs_consumed_after = LiveRunBudget(
            workspace / "live-task-runs.jsonl",
            maximum=maximum_live_task_runs,
        ).consumed
        manifest.source_tree_digest_after = source_tree_digest(source_root)
    except (OSError, ValueError):
        pass
    manifest.status = status
    manifest.stop_reason = (
        redact_environment_secrets(stop_reason)[:1_000] if stop_reason else None
    )
    _write_json(manifest_path, manifest)


def _run_h2_3_campaign(
    tasks: Sequence[EvaluationTask],
    *,
    dataset: str,
    workspace_root: str | Path,
    output_root: str | Path,
    maximum_live_task_runs: int = MAX_H2_3_LIVE_TASK_RUNS,
    task_timeout_seconds: float = DEFAULT_LIVE_TASK_TIMEOUT_SECONDS,
    evaluator_timeout_seconds: float = 300.0,
    repograph_commit: str | None = None,
    model_settings: ProductionModelSettings | None = None,
    repograph_source_root: str | Path | None = None,
) -> H23CampaignOutcome:
    settings = model_settings or prepare_live_environment()
    _validate_model_settings(settings)
    formal, smoke = freeze_h2_3_tasks(tasks)
    all_selected = [*formal, *smoke]
    all_benchmark = sorted(
        (EvaluationTask.model_validate(item) for item in tasks),
        key=lambda item: item.id,
    )
    workspace = Path(workspace_root).resolve()
    output = Path(output_root).resolve()
    source_root = (
        Path(repograph_source_root).resolve()
        if repograph_source_root is not None
        else Path(__file__).resolve().parents[1]
    )
    workspace.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise CampaignError("H2.3 output already contains a campaign manifest.")
    storage = EvaluationStorage(workspace / "evaluation.db", workspace / "results")
    artifacts_root = workspace / "artifacts"
    budget = LiveRunBudget(
        workspace / "live-task-runs.jsonl",
        maximum=maximum_live_task_runs,
    )
    if budget.consumed + H2_3_PLANNED_BASE_TASK_RUNS > maximum_live_task_runs:
        raise CampaignError(
            "The planned 70 H2.3 task-runs exceed the remaining live budget."
        )
    specifications = _evaluator_specifications(
        all_benchmark,
        evaluator_timeout_seconds=evaluator_timeout_seconds,
    )
    experiments = build_h2_3_experiments(
        dataset,
        task_timeout_seconds=task_timeout_seconds,
        evaluator_timeout_seconds=evaluator_timeout_seconds,
        repograph_commit=repograph_commit,
        model_settings=settings,
    )
    campaign_id = f"h2-3-local-{uuid.uuid4().hex[:12]}"
    task_set = _task_set_payload(
        campaign_id=campaign_id,
        dataset=dataset,
        formal=formal,
        smoke=smoke,
        specifications=specifications,
    )
    selected_ids = {item.id for item in all_selected}
    task_set["unused_task_ids"] = [
        item.id for item in all_benchmark if item.id not in selected_ids
    ]
    task_set_path = _write_json(output / "task-set.json", task_set)
    _write_json(output.parent / "h2_3_tasks.json", task_set)
    _write_json(
        Path(__file__).resolve().parent / "campaigns" / "h2_3_tasks.json",
        task_set,
    )
    config_path = _write_json(
        output / "config.json", _config_payload(experiments, settings)
    )
    evaluator_path = _write_json(
        output / "evaluator.json",
        _evaluator_payload(specifications),
    )
    source_before = source_tree_digest(source_root)
    python_executable = next(iter(specifications.values())).python_executable
    if python_executable is None:
        raise CampaignError("H2.3 evaluator Python interpreter is unavailable.")
    manifest = H23CampaignManifest(
        id=campaign_id,
        dataset=dataset,
        formal_task_ids=[item.id for item in formal],
        smoke_task_ids=[item.id for item in smoke],
        tasks=[_frozen_task(item, specifications[item.id]) for item in formal],
        configs=list(CONFIG_ORDER),
        experiment_ids={key: value.id for key, value in experiments.items()},
        configured_model=settings.model,
        configured_provider=settings.provider,
        api_base_url=settings.base_url,
        model_temperature=settings.temperature,
        model_seed=settings.seed,
        provider_seed_controlled=settings.seed is not None,
        reasoning_effort=settings.reasoning_effort,
        max_completion_tokens=settings.max_completion_tokens,
        evaluator_version=LOCAL_EVALUATOR_VERSION,
        python_executable=python_executable,
        task_timeout_seconds=task_timeout_seconds,
        evaluator_timeout_seconds=evaluator_timeout_seconds,
        maximum_live_task_runs=maximum_live_task_runs,
        live_task_runs_consumed_before=budget.consumed,
        live_task_runs_consumed_after=budget.consumed,
        source_tree_digest_before=source_before,
        repograph_commit=repograph_commit,
        created_at=utc_now(),
    )
    _write_json(manifest_path, manifest)

    evaluator_fingerprint, baseline_checks = _preflight_benchmark_evaluators(
        all_benchmark,
        specifications,
        workspace_root=workspace / "evaluator-preflight-workspaces",
    )
    evaluator_payload = _evaluator_payload(specifications, evaluator_fingerprint)
    evaluator_payload["baseline_checks"] = baseline_checks
    _write_json(evaluator_path, evaluator_payload)
    preflight = run_live_preflight(settings=settings)
    preflight_path = _write_json(output / "preflight.json", preflight)
    manifest.resolved_models = preflight.resolved_models
    _write_json(manifest_path, manifest)

    source_state = source_snapshots(all_selected)
    if any(status for _head, status in source_state.values()):
        raise CampaignError("H2.3 source fixture repositories must be clean.")
    tracker = _IntegrityTracker(
        tasks=all_selected,
        source_state=source_state,
        specifications=specifications,
        artifacts_root=artifacts_root,
    )
    smoke_results = _run_phase(
        "smoke",
        smoke,
        experiments["smoke"],
        storage=storage,
        workspace_root=workspace / "workspaces",
        artifacts_root=artifacts_root,
        output=output,
        budget=budget,
        tracker=tracker,
    )
    _assert_smoke(smoke_results)

    results_by_config: dict[str, list[EvaluationResult]] = {}
    for name in CONFIG_ORDER:
        results_by_config[name] = _run_phase(
            name,
            formal,
            experiments[name],
            storage=storage,
            workspace_root=workspace / "workspaces",
            artifacts_root=artifacts_root,
            output=output,
            budget=budget,
            tracker=tracker,
        )
        _write_combined_results(output, storage, experiments)

    combined_paths = _write_combined_results(output, storage, experiments)
    report_paths = write_h2_3_reports(
        output,
        formal,
        results_by_config,
        artifacts_root=artifacts_root,
        regrade_workspace_root=workspace / "regrade-workspaces",
    )
    security_scan = scan_persisted_artifacts(
        [artifacts_root, storage.results_root, output]
    )
    security_path = _write_json(output / "security-scan.json", security_scan)
    scan_persisted_artifacts([artifacts_root, storage.results_root, output])
    assert_source_snapshots(all_selected, source_state)
    source_after = source_tree_digest(source_root)
    if source_after != source_before:
        raise CampaignStopped("RepoGraph source tree changed during the H2.3 campaign.")
    resolved_models = {
        model
        for experiment in experiments.values()
        for result in storage.list_results(experiment.id)
        for model in result.resolved_model_names
    }
    manifest.resolved_models = sorted(
        set(preflight.resolved_models).union(resolved_models)
    )
    manifest.live_task_runs_consumed_after = budget.consumed
    manifest.source_tree_digest_after = source_after
    _write_json(manifest_path, manifest)
    artifact_paths = [
        str(task_set_path),
        str(config_path),
        str(evaluator_path),
        str(preflight_path),
        *(str(item) for item in combined_paths),
        *(str(item) for item in report_paths),
        str(security_path),
    ]
    return H23CampaignOutcome(
        manifest_path=str(manifest_path),
        artifact_paths=artifact_paths,
        preflight=preflight,
        experiment_ids=manifest.experiment_ids,
        live_task_runs_consumed=budget.consumed,
    )


def run_h2_3_campaign(
    tasks: Sequence[EvaluationTask],
    *,
    dataset: str,
    workspace_root: str | Path,
    output_root: str | Path,
    maximum_live_task_runs: int = MAX_H2_3_LIVE_TASK_RUNS,
    task_timeout_seconds: float = DEFAULT_LIVE_TASK_TIMEOUT_SECONDS,
    evaluator_timeout_seconds: float = 300.0,
    repograph_commit: str | None = None,
    model_settings: ProductionModelSettings | None = None,
    repograph_source_root: str | Path | None = None,
) -> H23CampaignOutcome:
    """Run H2.3 sequentially and durably record completion or interruption."""

    output = Path(output_root).resolve()
    workspace = Path(workspace_root).resolve()
    source_root = (
        Path(repograph_source_root).resolve()
        if repograph_source_root is not None
        else Path(__file__).resolve().parents[1]
    )
    try:
        outcome = _run_h2_3_campaign(
            tasks,
            dataset=dataset,
            workspace_root=workspace,
            output_root=output,
            maximum_live_task_runs=maximum_live_task_runs,
            task_timeout_seconds=task_timeout_seconds,
            evaluator_timeout_seconds=evaluator_timeout_seconds,
            repograph_commit=repograph_commit,
            model_settings=model_settings,
            repograph_source_root=source_root,
        )
    except BaseException as error:
        _set_manifest_status(
            output / "manifest.json",
            status="stopped",
            workspace=workspace,
            maximum_live_task_runs=maximum_live_task_runs,
            source_root=source_root,
            stop_reason=str(error).strip() or type(error).__name__,
        )
        raise
    _set_manifest_status(
        output / "manifest.json",
        status="complete",
        workspace=workspace,
        maximum_live_task_runs=maximum_live_task_runs,
        source_root=source_root,
    )
    return outcome
