"""Stage H2.3D direct-OpenAI clean paired benchmark campaign."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from evaluation.adapters.local_tasks import LocalTaskDataset
from evaluation.artifact_security import (
    ArtifactSecurityError,
    assert_secret_free_payload,
    scan_persisted_artifacts,
)
from evaluation.campaign import CampaignError, CampaignStopped, source_snapshots
from evaluation.campaign_h2_3 import (
    H23CampaignManifest,
    _evaluator_specifications,
    _frozen_task,
    _IntegrityTracker,
    _live_config,
    _result_record,
    _write_json,
    freeze_h2_3_tasks,
)
from evaluation.campaign_h2_3c import agent_semantic_fingerprint, stop_rule_reason
from evaluation.candidate_artifacts import CandidateArtifactError
from evaluation.evaluator import LOCAL_EVALUATOR_VERSION, preflight_local_evaluator
from evaluation.experiments.configs import ablation_specs
from evaluation.h2_3_reports import CONFIG_ORDER, write_h2_3_reports
from evaluation.live import (
    DEFAULT_LIVE_TASK_TIMEOUT_SECONDS,
    LivePreflightResult,
    LiveRunBudget,
    prepare_live_environment,
    run_live_preflight,
    utc_now,
)
from evaluation.metrics import aggregate_metrics
from evaluation.models import EvaluationExperiment, EvaluationResult, EvaluationTask
from evaluation.runner import create_experiment, run_evaluation_task
from evaluation.security import redact_environment_secrets, redact_secret_values
from evaluation.source_digest import source_tree_digest
from evaluation.statistics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resolve_ci,
)
from evaluation.storage import EvaluationStorage, EvaluationStorageError
from evaluation.workspace import WorkspaceError
from model_defaults import ProductionModelSettings

H2_3D_CAMPAIGN_ID = "h2_3d_openai_luna"
H2_3D_MODEL = "gpt-5.6-luna"
H2_3D_CONFIGS = tuple(CONFIG_ORDER)
MAX_H2_3D_LIVE_TASK_RUNS = 80
H2_3D_BASE_TASK_RUNS = 17 * len(H2_3D_CONFIGS)
H2_3D_RETRY_RESERVE = MAX_H2_3D_LIVE_TASK_RUNS - H2_3D_BASE_TASK_RUNS
DIRECT_OPENAI_HOSTS = frozenset({"api.openai.com", "api.openai.com.cn"})


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CampaignError(f"Invalid or missing campaign artifact: {path}.") from error


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        redact_secret_values(payload), ensure_ascii=False, sort_keys=True
    )
    assert_secret_free_payload(rendered)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError as error:
            raise CampaignError(
                f"Malformed H2.3D JSONL record at line {number}."
            ) from error
        if not isinstance(value, dict):
            raise CampaignError(f"Invalid H2.3D JSONL record at line {number}.")
        records.append(value)
    return records


def _direct_openai_identity(settings: ProductionModelSettings) -> dict[str, str]:
    if settings.model != H2_3D_MODEL:
        raise CampaignError("H2.3D requires configured model gpt-5.6-luna.")
    if settings.provider not in {"openai", "openai_compatible"}:
        raise CampaignError("H2.3D requires an OpenAI client provider mode.")
    if settings.base_url is None:
        host = "api.openai.com"
        endpoint = "SDK default"
    else:
        parsed = urlsplit(settings.base_url)
        host = (parsed.hostname or "").casefold()
        endpoint = f"{parsed.scheme}://{host}"
    if host not in DIRECT_OPENAI_HOSTS:
        raise CampaignError(
            "H2.3D is not routed to the direct OpenAI API endpoint."
        )
    return {
        "provider_adapter_mode": settings.provider,
        "provider_identity": "openai",
        "routing": "direct",
        "endpoint_identity": endpoint,
    }


def _status_code(result: EvaluationResult | None) -> str:
    if result is None:
        return "N"
    if result.valid_prediction or result.status in {"resolved", "unresolved"}:
        return "R" if result.final_resolved else "U"
    if result.infrastructure_failure or result.status in {"evaluation_error", "timeout"}:
        return "I"
    return "N"


class H23DCampaign:
    """Restartable 80-run controller for a clean four-way Luna campaign."""

    def __init__(
        self,
        tasks: Sequence[EvaluationTask],
        *,
        dataset: str,
        workspace_root: str | Path,
        output_root: str | Path,
        frozen_output_root: str | Path,
        source_root: str | Path,
        model_settings: ProductionModelSettings,
        task_timeout_seconds: float = DEFAULT_LIVE_TASK_TIMEOUT_SECONDS,
        evaluator_timeout_seconds: float = 300.0,
        repograph_commit: str | None = None,
        campaign_id: str = H2_3D_CAMPAIGN_ID,
        stage: str = "H2.3D",
        maximum_live_task_runs: int = MAX_H2_3D_LIVE_TASK_RUNS,
        planned_base_task_runs: int = H2_3D_BASE_TASK_RUNS,
        infrastructure_retry_reserve: int = H2_3D_RETRY_RESERVE,
        configuration_root: str | Path | None = None,
        budget_path: str | Path | None = None,
        continuation_parent_campaign_id: str | None = None,
    ) -> None:
        self.tasks = sorted(
            (EvaluationTask.model_validate(item) for item in tasks),
            key=lambda item: item.id,
        )
        self.dataset = dataset
        self.workspace = Path(workspace_root).resolve()
        self.output = Path(output_root).resolve()
        self.configuration_root = (
            Path(configuration_root).resolve()
            if configuration_root is not None
            else self.output
        )
        self.frozen_output = Path(frozen_output_root).resolve(strict=True)
        self.source_root = Path(source_root).resolve(strict=True)
        self.settings = model_settings
        self.routing = _direct_openai_identity(model_settings)
        self.task_timeout_seconds = task_timeout_seconds
        self.evaluator_timeout_seconds = evaluator_timeout_seconds
        self.repograph_commit = repograph_commit
        self.campaign_id = campaign_id
        self.stage = stage
        self.maximum_live_task_runs = maximum_live_task_runs
        self.planned_base_task_runs = planned_base_task_runs
        self.infrastructure_retry_reserve = infrastructure_retry_reserve
        self.continuation_parent_campaign_id = continuation_parent_campaign_id
        self.parent_manifest = H23CampaignManifest.model_validate(
            _load_json(self.frozen_output / "manifest.json")
        )
        self.formal, _smoke = freeze_h2_3_tasks(self.tasks)
        self.specifications = _evaluator_specifications(
            self.tasks,
            evaluator_timeout_seconds=evaluator_timeout_seconds,
        )
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)
        self.storage = EvaluationStorage(
            self.workspace / "evaluation.db", self.workspace / "results"
        )
        self.artifacts_root = self.workspace / "artifacts"
        self.budget = LiveRunBudget(
            budget_path or self.workspace / "live-task-runs.jsonl",
            maximum=maximum_live_task_runs,
        )
        self.history_path = self.output / "results.jsonl"
        self.manifest_path = self.output / "manifest.json"
        self.preflight_path = self.output / "preflight.json"
        self.verification_path = self.output / "frozen-verification.json"
        self.experiments = self._load_or_create_experiments()
        self.source_state = source_snapshots(self.formal)
        self.tracker = _IntegrityTracker(
            tasks=self.formal,
            source_state=self.source_state,
            specifications=self.specifications,
            artifacts_root=self.artifacts_root,
        )

    def _experiment_sources(self) -> dict[str, Any]:
        specs = {item.key: item.config for item in ablation_specs()}
        return {
            "full": specs["A"],
            "no_correction": specs["B"],
            "no_exploration": specs["C"],
            "no_agentic_test": specs["D"],
        }

    def _load_or_create_experiments(self) -> dict[str, EvaluationExperiment]:
        config_paths = {
            name: self.configuration_root / f"config-{name.replace('_', '-')}.json"
            for name in H2_3D_CONFIGS
        }
        if any(path.exists() for path in config_paths.values()):
            if not all(path.is_file() for path in config_paths.values()):
                raise CampaignError("H2.3D has an incomplete persisted config set.")
            experiments = {
                name: EvaluationExperiment.model_validate(_load_json(path)["experiment"])
                for name, path in config_paths.items()
            }
        else:
            experiments = {}
            for name, source in self._experiment_sources().items():
                config = _live_config(
                    source,
                    name=name,
                    task_timeout_seconds=self.task_timeout_seconds,
                    evaluator_timeout_seconds=self.evaluator_timeout_seconds,
                    model_settings=self.settings,
                )
                experiment = create_experiment(
                    f"h2-3d-openai-luna-{name.replace('_', '-')}",
                    self.dataset,
                    config,
                    git_commit=self.repograph_commit,
                )
                experiments[name] = experiment
                _write_json(
                    config_paths[name],
                    {
                        "campaign_id": self.campaign_id,
                        "configuration": name,
                        "workers": 1,
                        "maximum_live_task_runs": self.maximum_live_task_runs,
                        "infrastructure_retry_limit": 1,
                        "experiment": experiment.model_dump(mode="json"),
                    },
                )
        expected_sources = self._experiment_sources()
        for name, experiment in experiments.items():
            expected = _live_config(
                expected_sources[name],
                name=name,
                task_timeout_seconds=self.task_timeout_seconds,
                evaluator_timeout_seconds=self.evaluator_timeout_seconds,
                model_settings=self.settings,
            )
            if (
                experiment.dataset != self.dataset
                or experiment.config != expected
                or experiment.model_name != H2_3D_MODEL
                or experiment.llm_execution_kind != "live"
            ):
                raise CampaignError(f"H2.3D experiment identity changed: {name}.")
            self.storage.save_experiment(experiment)
        return experiments

    def validate_frozen_state(self) -> dict[str, Any]:
        parent = self.parent_manifest
        if len(self.formal) != 17:
            raise CampaignError("H2.3D requires exactly 17 frozen formal tasks.")
        if [item.id for item in self.formal] != parent.formal_task_ids:
            raise CampaignError("H2.3D frozen formal task IDs changed.")
        expected_tasks = [
            _frozen_task(item, self.specifications[item.id]) for item in self.formal
        ]
        if expected_tasks != parent.tasks:
            raise CampaignError("H2.3D task text, base commit, or evaluator digest changed.")

        frozen_task_set_path = self.frozen_output / "task-set.json"
        frozen_evaluator_path = self.frozen_output / "evaluator.json"
        task_set = _load_json(frozen_task_set_path)
        evaluator = _load_json(frozen_evaluator_path)
        if task_set.get("formal_task_ids") != parent.formal_task_ids:
            raise CampaignError("H2.3D frozen task-set artifact changed.")
        expected_specs = {
            task_id: specification.model_dump(mode="json")
            for task_id, specification in sorted(self.specifications.items())
        }
        if (
            evaluator.get("version") != LOCAL_EVALUATOR_VERSION
            or parent.evaluator_version != LOCAL_EVALUATOR_VERSION
            or evaluator.get("task_specifications") != expected_specs
            or not evaluator.get("preflight_succeeded")
        ):
            raise CampaignError("H2.3D frozen evaluator specification changed.")
        representative = preflight_local_evaluator(
            self.specifications[self.formal[0].id]
        )
        if representative.model_dump(mode="json") != evaluator.get(
            "preflight_fingerprint"
        ):
            raise CampaignError("H2.3D evaluator environment changed.")
        if any(status for _head, status in self.source_state.values()):
            raise CampaignError("H2.3D source fixture repositories must be clean.")

        semantic = agent_semantic_fingerprint(self.source_root)
        parent_created = datetime.fromisoformat(parent.created_at)
        changed_semantics = [
            item["path"]
            for item in semantic["files"]
            if datetime.fromisoformat(item["last_modified"]) > parent_created
        ]
        if changed_semantics:
            raise CampaignError(
                "Agent-semantic files changed after frozen H2.3: "
                + ", ".join(changed_semantics)
                + "."
            )

        new_task_set = self.output / "task-set.json"
        new_evaluator = self.output / "evaluator.json"
        if new_task_set.exists() and _sha256_file(new_task_set) != _sha256_file(
            frozen_task_set_path
        ):
            raise CampaignError("H2.3D copied task-set hash changed.")
        if new_evaluator.exists() and _sha256_file(new_evaluator) != _sha256_file(
            frozen_evaluator_path
        ):
            raise CampaignError("H2.3D copied evaluator hash changed.")
        _write_bytes_atomic(new_task_set, frozen_task_set_path.read_bytes())
        _write_bytes_atomic(new_evaluator, frozen_evaluator_path.read_bytes())

        verification = {
            "campaign_id": self.campaign_id,
            "frozen_parent_campaign_id": parent.id,
            "formal_task_count": len(self.formal),
            "formal_task_ids": [item.id for item in self.formal],
            "task_set_sha256": _sha256_file(new_task_set),
            "task_set_matches_frozen_h2_3": True,
            "evaluator_sha256": _sha256_file(new_evaluator),
            "evaluator_matches_frozen_h2_3": True,
            "evaluator_kind": "local",
            "evaluator_version": LOCAL_EVALUATOR_VERSION,
            "evaluator_environment": representative.model_dump(mode="json"),
            "python_executable": next(iter(self.specifications.values())).python_executable,
            "hidden_test_digests": {
                item.id: self.specifications[item.id].digest for item in self.formal
            },
            "hidden_tests_isolated": True,
            "agent_semantic_fingerprint": semantic,
            "agent_semantics_match_frozen_h2_3": True,
            "source_fixture_repositories_clean": True,
            "source_tree_digest": source_tree_digest(self.source_root),
            "no_agentic_test_contract": {
                "agentic_explore": self.experiments["no_agentic_test"].config.agentic_explore,
                "agentic_test": self.experiments["no_agentic_test"].config.agentic_test,
                "read_search_list_available": True,
                "run_repository_test_absent": True,
            },
            "verified_at": utc_now(),
        }
        _write_json(self.verification_path, verification)
        return verification

    def _manifest_payload(
        self, *, status: str, stop_reason: str | None = None
    ) -> dict[str, Any]:
        prior = _load_json(self.manifest_path) if self.manifest_path.is_file() else {}
        return {
            "id": self.campaign_id,
            "stage": self.stage,
            "created_at": prior.get("created_at", utc_now()),
            "updated_at": utc_now(),
            "status": status,
            "stop_reason": (
                redact_environment_secrets(stop_reason)[:1000] if stop_reason else None
            ),
            **self.routing,
            "configured_model": self.settings.model,
            "temperature": self.settings.temperature,
            "provider_seed": self.settings.seed,
            "reasoning_effort": self.settings.reasoning_effort,
            "max_completion_tokens": self.settings.max_completion_tokens,
            "workers": 1,
            "task_timeout_seconds": self.task_timeout_seconds,
            "evaluator_timeout_seconds": self.evaluator_timeout_seconds,
            "maximum_live_task_runs": self.maximum_live_task_runs,
            "planned_base_task_runs": self.planned_base_task_runs,
            "infrastructure_retry_reserve": self.infrastructure_retry_reserve,
            "live_task_runs_consumed": self.budget.consumed,
            "experiment_ids": {
                name: experiment.id for name, experiment in self.experiments.items()
            },
            "formal_task_ids": [item.id for item in self.formal],
            "frozen_parent_campaign_id": self.parent_manifest.id,
            "parent_campaign": self.continuation_parent_campaign_id,
        }

    def _write_manifest(self, *, status: str, stop_reason: str | None = None) -> None:
        _write_json(
            self.manifest_path,
            self._manifest_payload(status=status, stop_reason=stop_reason),
        )

    @staticmethod
    def _validate_resolved_models(models: Sequence[str]) -> None:
        if any(model != H2_3D_MODEL for model in models):
            raise CampaignStopped("Resolved provider model is not gpt-5.6-luna.")

    def preflight(self) -> LivePreflightResult:
        if self.preflight_path.is_file():
            result = LivePreflightResult.model_validate(_load_json(self.preflight_path))
        else:
            result = run_live_preflight(settings=self.settings)
            _write_json(self.preflight_path, result)
        expected = (
            self.settings.provider,
            self.settings.base_url,
            self.settings.model,
            self.settings.temperature,
            self.settings.seed,
            self.settings.reasoning_effort,
            self.settings.max_completion_tokens,
        )
        actual = (
            result.configured_provider,
            result.api_base_url,
            result.configured_model,
            result.model_temperature,
            result.model_seed,
            result.reasoning_effort,
            result.max_completion_tokens,
        )
        if actual != expected or not result.structured_output_succeeded:
            raise CampaignError("H2.3D persisted provider preflight changed.")
        self._validate_resolved_models(result.resolved_models)
        return result

    def _record_attempt(
        self,
        *,
        config_name: str,
        result: EvaluationResult,
        ordinal: int,
    ) -> None:
        _append_jsonl(
            self.history_path,
            {
                "campaign_id": self.campaign_id,
                "config": config_name,
                "live_task_run_ordinal": ordinal,
                "result": _result_record(result),
            },
        )

    def _execute_attempt(
        self,
        config_name: str,
        task: EvaluationTask,
    ) -> EvaluationResult:
        experiment = self.experiments[config_name]
        attempt = self.storage.reserve_execution_attempt(
            campaign_id=self.campaign_id,
            config_name=config_name,
            experiment_id=experiment.id,
            task_id=task.id,
        )
        try:
            reservation = self.budget.reserve(
                experiment_id=experiment.id,
                task_id=task.id,
                config_name=config_name,
            )
            attempt = self.storage.set_execution_attempt_live_ordinal(
                attempt, reservation.ordinal
            )
            attempt = self.storage.mark_execution_attempt_running(attempt)
        except BaseException as error:
            try:
                self.storage.finish_execution_attempt(
                    attempt,
                    status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    failure_reason=str(error)[:4000],
                )
            except EvaluationStorageError:
                pass
            raise
        print(
            f"Starting {self.stage} live task-run {reservation.ordinal}/"
            f"{self.maximum_live_task_runs}: {config_name}/{task.id} "
            f"(execution attempt {attempt.execution_attempt})",
            flush=True,
        )
        self.storage.save_task(task)
        try:
            result = run_evaluation_task(
                task,
                experiment.config,
                workspace_root=str(self.workspace / "workspaces"),
                artifacts_root=str(self.artifacts_root),
                experiment_id=experiment.id,
                git_commit=experiment.git_commit,
                execution_attempt=attempt.execution_attempt,
            )
        except BaseException as error:
            self.storage.finish_execution_attempt(
                attempt,
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                failure_reason=str(error)[:4000],
            )
            raise
        if attempt.execution_attempt > 1:
            result = result.model_copy(update={"required_infrastructure_retry": True})
        self.storage.save_result(result)
        terminal_status = (
            "timeout"
            if result.hard_timeout or result.status == "timeout"
            else "failed"
            if result.infrastructure_failure
            else "completed"
        )
        self.storage.finish_execution_attempt(
            attempt,
            status=terminal_status,
            failure_reason=result.failure_reason if terminal_status != "completed" else None,
        )
        self._record_attempt(
            config_name=config_name,
            result=result,
            ordinal=reservation.ordinal,
        )
        self._write_manifest(status="running")
        self.tracker.observe_attempt(result)
        self._validate_resolved_models(result.resolved_model_names)
        return result

    def _run_task(self, config_name: str, task: EvaluationTask) -> EvaluationResult:
        experiment = self.experiments[config_name]
        existing = self.storage.get_result(experiment.id, task.id)
        if existing is not None and not existing.infrastructure_failure:
            return existing
        result = self._execute_attempt(config_name, task)
        if result.infrastructure_failure and not result.hard_timeout:
            result = self._execute_attempt(config_name, task)
        return result

    def _results(self, config_name: str) -> list[EvaluationResult]:
        return self.storage.list_results(self.experiments[config_name].id)

    def _recent_final_results(self, config_name: str) -> list[EvaluationResult]:
        latest: dict[str, EvaluationResult] = {}
        order: list[str] = []
        for record in _load_jsonl(self.history_path):
            if record.get("config") != config_name:
                continue
            result = EvaluationResult.model_validate(record["result"])
            if result.task_id not in latest:
                order.append(result.task_id)
            latest[result.task_id] = result
        return [latest[task_id] for task_id in order]

    def _check_stop(self, config_name: str) -> None:
        results = {item.task_id: item for item in self._results(config_name)}
        reason = stop_rule_reason(
            config_name, results, self._recent_final_results(config_name)
        )
        if reason:
            raise CampaignStopped(reason)

    def run_configurations(self) -> None:
        for config_name in H2_3D_CONFIGS:
            for task in self.formal:
                self._run_task(config_name, task)
                self._check_stop(config_name)
            if len(self._results(config_name)) != 17:
                raise CampaignStopped(
                    f"H2.3D {config_name} did not complete all 17 tasks."
                )

    @staticmethod
    def _telemetry(results: Sequence[EvaluationResult]) -> dict[str, Any]:
        complete = [
            item
            for item in results
            if not item.telemetry_incomplete
            and item.llm_calls is not None
            and item.input_tokens is not None
            and item.output_tokens is not None
            and item.total_tokens is not None
        ]

        def known(field: str) -> int:
            return sum(
                value
                for item in results
                if (value := getattr(item, field)) is not None
            )

        return {
            "assigned_tasks": len(results),
            "tasks_with_complete_usage": len(complete),
            "tasks_with_incomplete_usage": len(results) - len(complete),
            "telemetry_completeness_rate": (
                len(complete) / len(results) if results else 0.0
            ),
            "llm_calls": known("llm_calls"),
            "input_tokens": known("input_tokens"),
            "output_tokens": known("output_tokens"),
            "total_tokens": known("total_tokens"),
            "llm_calls_per_task": known("llm_calls") / len(results) if results else None,
            "tokens_per_task": (
                known("total_tokens") / len(results)
                if results and len(complete) == len(results)
                else None
            ),
            "runtime_seconds": sum(item.duration_seconds for item in results),
            "runtime_seconds_per_task": (
                sum(item.duration_seconds for item in results) / len(results)
                if results
                else None
            ),
            "tool_calls": known("tool_calls"),
            "test_calls": known("test_calls"),
            "correction_rounds": sum(item.correction_rounds_used for item in results),
            "estimated_cost_usd": None,
        }

    @staticmethod
    def _failure_taxonomy(results: Sequence[EvaluationResult]) -> dict[str, int]:
        allowed = {
            "valid_prediction_unresolved",
            "planning_failure",
            "candidate_generation_failure",
            "verification_failure",
            "review_failure",
            "correction_failure",
            "api_infrastructure_failure",
            "evaluation_infrastructure_failure",
            "timeout",
        }
        taxonomy = Counter({name: 0 for name in allowed})
        for item in results:
            category = item.failure_category
            if category is None:
                continue
            if category in {"infrastructure_error", "external_evaluator_failure"}:
                category = "evaluation_infrastructure_failure"
            elif category == "agent_failure":
                category = "candidate_generation_failure"
            if category not in allowed:
                category = "candidate_generation_failure"
            taxonomy[category] += 1
        return {name: taxonomy[name] for name in sorted(allowed)}

    def _paired_details(
        self, results_by_config: Mapping[str, Sequence[EvaluationResult]]
    ) -> tuple[list[dict[str, str]], dict[str, dict[str, int]]]:
        maps = {
            name: {item.task_id: item for item in results}
            for name, results in results_by_config.items()
        }
        matrix = [
            {
                "task_id": task.id,
                **{
                    name: _status_code(maps[name].get(task.id))
                    for name in H2_3D_CONFIGS
                },
            }
            for task in self.formal
        ]
        counts: dict[str, dict[str, int]] = {}
        for name in H2_3D_CONFIGS[1:]:
            values = Counter()
            for row in matrix:
                left, right = row["full"], row[name]
                if left in {"R", "U"} and right in {"R", "U"}:
                    values["comparable_pairs"] += 1
                    if left == "R" and right == "U":
                        values["full_wins"] += 1
                    elif left == "U" and right == "R":
                        values["ablation_wins"] += 1
                    elif left == "R":
                        values["resolved_ties"] += 1
                    else:
                        values["unresolved_ties"] += 1
                elif "I" in {left, right}:
                    values["excluded_infrastructure_pairs"] += 1
                else:
                    values["excluded_non_candidate_pairs"] += 1
            counts[name] = {
                key: values[key]
                for key in (
                    "comparable_pairs",
                    "full_wins",
                    "ablation_wins",
                    "resolved_ties",
                    "unresolved_ties",
                    "excluded_infrastructure_pairs",
                    "excluded_non_candidate_pairs",
                )
            }
        return matrix, counts

    def _write_failure_analysis(
        self, results_by_config: Mapping[str, Sequence[EvaluationResult]]
    ) -> None:
        lines = [
            "# H2.3D Full capability failure analysis",
            "",
            "Bounded summaries only; no chain-of-thought or hidden-test source is included.",
            "",
        ]
        full_failures = [
            item
            for item in results_by_config["full"]
            if item.valid_prediction and not item.final_resolved
        ]
        if not full_failures:
            lines.extend(["No Full valid-prediction unresolved tasks.", ""])
        for item in sorted(full_failures, key=lambda value: value.task_id):
            lines.extend(
                [
                    f"## {item.task_id}",
                    "",
                    f"- Category: {item.failure_category or 'valid_prediction_unresolved'}",
                    f"- Plan status: {'succeeded' if item.planning_succeeded else 'failed'}",
                    f"- Changed files: {', '.join(item.changed_files) or 'none'}",
                    f"- Verification succeeded: {item.verification_succeeded}",
                    f"- Review good: {item.review_good}",
                    f"- Correction rounds: {item.correction_rounds_used}",
                    f"- Hidden evaluator summary: {(item.failure_reason or 'assertion failure')[:500]}",
                    "",
                ]
            )
        (self.output / "failure-analysis.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    def _write_correction_analysis(self) -> None:
        records = _load_jsonl(self.output / "candidate-regrade.jsonl")
        analyses = [
            record["analysis"] for record in records if record["config"] == "full"
        ]
        entered = [
            item
            for item in analyses
            if item["first"]["candidate_attempt"] != item["final"]["candidate_attempt"]
        ]
        values = {
            "candidate_predictions": len(analyses),
            "entered_correction": len(entered),
            "rescued_by_correction": sum(
                item["rescued_by_correction"] for item in entered
            ),
            "regressed_after_correction": sum(
                item["regressed_after_correction"] for item in entered
            ),
            "already_resolved_before_correction": sum(
                item["first_attempt_resolved"] for item in analyses
            ),
            "remained_resolved": sum(
                item["first_attempt_resolved"] and item["final_resolved"]
                for item in analyses
            ),
            "remained_unresolved": sum(
                not item["first_attempt_resolved"] and not item["final_resolved"]
                for item in analyses
            ),
        }
        _write_json(self.output / "correction-analysis.json", values)
        lines = [
            "# H2.3D Full correction analysis",
            "",
            f"- Candidate predictions: {values['candidate_predictions']}",
            f"- Entered correction: {values['entered_correction']}",
            f"- Rescued by correction: {values['rescued_by_correction']}",
            f"- Regressed after correction: {values['regressed_after_correction']}",
            f"- Already resolved before correction: {values['already_resolved_before_correction']}",
            f"- Remained resolved: {values['remained_resolved']}",
            f"- Remained unresolved: {values['remained_unresolved']}",
            "",
        ]
        (self.output / "correction-analysis.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    def _write_comparison_markdown(
        self,
        comparison: Mapping[str, Any],
        matrix: Sequence[Mapping[str, str]],
        counts: Mapping[str, Mapping[str, int]],
    ) -> None:
        lines = [
            "# H2.3D Direct OpenAI clean paired benchmark",
            "",
            "This is a controlled exploratory benchmark (N=17), not statistical proof or a general software-engineering success rate.",
            "",
            "| Configuration | Attempted | Valid | Capability | End-to-end / 17 | Infrastructure / 17 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name in H2_3D_CONFIGS:
            metric = comparison["h2_3d_protocol_metrics"][name]
            capability = metric["capability_resolve_rate"]
            capability_text = (
                "n/a" if capability is None else f"{capability:.1%}"
            )
            lines.append(
                f"| {name} | {metric['attempted_distinct_tasks']}/17 | "
                f"{metric['valid_predictions']} | {capability_text} | "
                f"{metric['end_to_end_resolve_rate']:.1%} | "
                f"{metric['infrastructure_failure_rate']:.1%} |"
            )
        lines.extend(
            [
            "",
            "R = resolved valid prediction; U = unresolved valid prediction; I = infrastructure failure; N = no valid candidate from a non-infrastructure agent failure.",
            "",
            "| Task | Full | No Correction | No Exploration | No Agentic Test |",
            "|---|---:|---:|---:|---:|",
            ]
        )
        for row in matrix:
            lines.append(
                f"| {row['task_id']} | {row['full']} | {row['no_correction']} | "
                f"{row['no_exploration']} | {row['no_agentic_test']} |"
            )
        lines.extend(["", "## Comparable pairs", ""])
        for name in H2_3D_CONFIGS[1:]:
            value = counts[name]
            delta = comparison["paired"][name]["capability_delta"]
            lines.append(
                f"- Full − {name}: comparable {value['comparable_pairs']}; "
                f"Full wins {value['full_wins']}; ablation wins {value['ablation_wins']}; "
                f"resolved ties {value['resolved_ties']}; unresolved ties "
                f"{value['unresolved_ties']}; excluded infrastructure "
                f"{value['excluded_infrastructure_pairs']}; capability delta {delta}."
            )
        lines.extend(
            [
                "",
                "Capability pairs require valid predictions on both sides. Historical RelayAPI/Sol observations are contextual only because provider, model, and campaign time all changed.",
                "",
            ]
        )
        (self.output / "comparison.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    def _write_common_reports(
        self, *, status: str, stop_reason: str | None = None
    ) -> list[Path]:
        results_by_config = {name: self._results(name) for name in H2_3D_CONFIGS}
        report_paths = write_h2_3_reports(
            self.output,
            self.formal,
            results_by_config,
            artifacts_root=self.artifacts_root,
            regrade_workspace_root=self.source_root / ".evaluation" / "h23d-regrade",
        )
        comparison_path = self.output / "comparison.json"
        comparison = _load_json(comparison_path)
        matrix, counts = self._paired_details(results_by_config)
        comparison.update(
            {
                "campaign_id": self.campaign_id,
                "status": status,
                "stop_reason": stop_reason,
                "authoritative_final": status == "complete",
                "paired_matrix": matrix,
                "paired_counts": counts,
                "small_n_statement": "This is a controlled exploratory benchmark.",
            }
        )
        protocol_metrics: dict[str, dict[str, Any]] = {}
        for name in H2_3D_CONFIGS:
            results = results_by_config[name]
            valid = [
                item.final_resolved
                for item in results
                if item.valid_prediction or item.status in {"resolved", "unresolved"}
            ]
            protocol_metrics[name] = {
                "planned_assigned_tasks": 17,
                "attempted_distinct_tasks": len(results),
                "valid_predictions": len(valid),
                "capability_resolve_rate": (
                    sum(valid) / len(valid) if valid else None
                ),
                "end_to_end_resolve_rate": (
                    sum(item.final_resolved for item in results) / 17
                ),
                "infrastructure_failure_rate": (
                    sum(item.infrastructure_failure for item in results) / 17
                ),
            }
            comparison["configs"][name]["capability_resolve_95_ci"] = (
                bootstrap_resolve_ci(
                    valid,
                    seed=DEFAULT_BOOTSTRAP_SEED,
                    resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
                ).model_dump(mode="json")
            )
        comparison["h2_3d_protocol_metrics"] = protocol_metrics
        _write_json(comparison_path, comparison)
        self._write_comparison_markdown(comparison, matrix, counts)

        telemetry = {
            name: self._telemetry(results_by_config[name]) for name in H2_3D_CONFIGS
        }
        _write_json(self.output / "telemetry.json", telemetry)
        taxonomy = {
            name: self._failure_taxonomy(results_by_config[name])
            for name in H2_3D_CONFIGS
        }
        _write_json(self.output / "failure-taxonomy.json", taxonomy)
        self._write_failure_analysis(results_by_config)
        self._write_correction_analysis()
        _write_json(
            self.output / "results.json",
            {
                "campaign_id": self.campaign_id,
                "status": status,
                "stop_reason": stop_reason,
                "configs": {
                    name: {
                        "metrics": aggregate_metrics(results).model_dump(mode="json"),
                        "h2_3d_protocol_metrics": protocol_metrics[name],
                        "results": [_result_record(item) for item in results],
                    }
                    for name, results in results_by_config.items()
                },
            },
        )
        return report_paths

    def write_final_reports(self) -> list[Path]:
        for name in H2_3D_CONFIGS:
            if len(self._results(name)) != 17:
                raise CampaignError(
                    f"H2.3D cannot finalize {name}: expected 17 tasks."
                )
        report_paths = self._write_common_reports(status="complete")
        resolved_models = sorted(
            {
                model
                for name in H2_3D_CONFIGS
                for result in self._results(name)
                for model in result.resolved_model_names
            }
        )
        self._validate_resolved_models(resolved_models)
        scan = scan_persisted_artifacts(
            [self.workspace, self.output], environment=os.environ
        )
        security_path = _write_json(self.output / "security-scan.json", scan)
        _write_json(
            self.output / "completion.json",
            {
                "status": "complete",
                "campaign_id": self.campaign_id,
                "direct_openai_routing_verified": True,
                "configured_model": H2_3D_MODEL,
                "resolved_models": resolved_models,
                "actual_model_verified": True,
                "all_four_configs_have_17_tasks": True,
                "candidate_snapshot_protocol": True,
                "exact_candidate_regrade_complete": True,
                "paired_matrix_complete": True,
                "bootstrap_resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
                "analysis_seed": DEFAULT_BOOTSTRAP_SEED,
                "live_task_runs_consumed": self.budget.consumed,
                "capability_failures_retried": False,
                "prior_campaigns_preserved": True,
                "estimated_cost_usd": None,
                "doctor": "pending final environment check",
                "official_docker_smoke": "pending final environment check",
                "regressions": "pending",
            },
        )
        return [*report_paths, security_path]

    def write_stopped_reports(self, stop_reason: str) -> list[Path]:
        report_paths = self._write_common_reports(
            status="stopped", stop_reason=stop_reason
        )
        scan = scan_persisted_artifacts(
            [self.workspace, self.output], environment=os.environ
        )
        security_path = _write_json(self.output / "security-scan.json", scan)
        _write_json(
            self.output / "completion.json",
            {
                "status": "stopped",
                "campaign_id": self.campaign_id,
                "stop_reason": stop_reason,
                "authoritative_final": False,
                "configuration_task_counts": {
                    name: len(self._results(name)) for name in H2_3D_CONFIGS
                },
                "live_task_runs_consumed": self.budget.consumed,
                "capability_failures_retried": False,
                "prior_campaigns_preserved": True,
            },
        )
        return [*report_paths, security_path]

    def run(self) -> dict[str, Any]:
        verification = self.validate_frozen_state()
        self._write_manifest(status="running")
        preflight = self.preflight()
        self.run_configurations()
        self.write_final_reports()
        self._write_manifest(status="complete")
        return {
            "campaign_id": self.campaign_id,
            "manifest": str(self.manifest_path),
            "output": str(self.output),
            "preflight": preflight.model_dump(mode="json"),
            "verification": verification,
            "live_task_runs_consumed": self.budget.consumed,
        }


def run_h2_3d_campaign(
    *,
    dataset_path: str | Path,
    workspace_root: str | Path,
    output_root: str | Path,
    frozen_output_root: str | Path,
    source_root: str | Path,
    model_settings: ProductionModelSettings | None = None,
    task_timeout_seconds: float = DEFAULT_LIVE_TASK_TIMEOUT_SECONDS,
    evaluator_timeout_seconds: float = 300.0,
    repograph_commit: str | None = None,
) -> dict[str, Any]:
    """Validate and run the independent direct-OpenAI Luna campaign."""

    settings = model_settings or prepare_live_environment()
    dataset_file = Path(dataset_path).resolve(strict=True)
    controller = H23DCampaign(
        LocalTaskDataset(dataset_file).load(),
        dataset=f"local:{dataset_file}",
        workspace_root=workspace_root,
        output_root=output_root,
        frozen_output_root=frozen_output_root,
        source_root=source_root,
        model_settings=settings,
        task_timeout_seconds=task_timeout_seconds,
        evaluator_timeout_seconds=evaluator_timeout_seconds,
        repograph_commit=repograph_commit,
    )
    try:
        return controller.run()
    except BaseException as error:
        reason = str(error).strip() or type(error).__name__
        controller._write_manifest(status="stopped", stop_reason=reason)
        if controller.preflight_path.is_file() or controller.budget.consumed:
            try:
                controller.write_stopped_reports(reason)
            except (
                ArtifactSecurityError,
                CampaignError,
                CandidateArtifactError,
                OSError,
                ValueError,
                WorkspaceError,
            ) as report_error:
                _write_json(
                    controller.output / "stopped-report-error.json",
                    {
                        "status": "failed",
                        "error": redact_environment_secrets(str(report_error))[:1000],
                    },
                )
        raise
