"""Stage H2.3C append-only completion of the frozen H2.3 ablations."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.adapters.local_tasks import LocalTaskDataset
from evaluation.artifact_security import (
    assert_secret_free_payload,
    scan_persisted_artifacts,
)
from evaluation.campaign import CampaignError, CampaignStopped, source_snapshots
from evaluation.campaign_h2_3 import (
    H23CampaignManifest,
    _evaluator_specifications,
    _frozen_task,
    _IntegrityTracker,
    _result_record,
    _validate_model_settings,
    _write_json,
    freeze_h2_3_tasks,
)
from evaluation.evaluator import preflight_local_evaluator
from evaluation.h2_3_reports import CONFIG_ORDER, write_h2_3_reports
from evaluation.live import (
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
from evaluation.storage import EvaluationStorage
from model_defaults import ProductionModelSettings

H2_3C_CONTINUATION_ID = "h2_3c"
H2_3C_PARENT_CAMPAIGN = "h2_3"
H2_3C_CONFIGS = ("no_correction", "no_exploration", "no_agentic_test")
MAX_H2_3C_LIVE_TASK_RUNS = 60
H2_3C_BASE_TASK_RUNS = 48
H2_3C_RETRY_RESERVE = MAX_H2_3C_LIVE_TASK_RUNS - H2_3C_BASE_TASK_RUNS


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def agent_semantic_fingerprint(source_root: str | Path) -> dict[str, Any]:
    """Fingerprint root production modules while excluding evaluation/report code."""

    root = Path(source_root).resolve(strict=True)
    files = sorted(root.glob("*.py"), key=lambda item: item.name.casefold())
    if not files:
        raise CampaignError("No root Agent-semantic Python modules were found.")
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        payload = path.read_bytes()
        file_digest = _sha256_bytes(payload)
        rendered = relative.encode("utf-8")
        digest.update(len(rendered).to_bytes(8, "big"))
        digest.update(rendered)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        records.append(
            {
                "path": relative,
                "sha256": file_digest,
                "last_modified": datetime.fromtimestamp(path.stat().st_mtime)
                .astimezone()
                .isoformat(),
            }
        )
    return {"digest": digest.hexdigest(), "files": records}


def _status_code(result: EvaluationResult | None) -> str:
    if result is None:
        return "N"
    if result.valid_prediction or result.status in {"resolved", "unresolved"}:
        return "R" if result.final_resolved else "U"
    if result.infrastructure_failure or result.status in {
        "evaluation_error",
        "timeout",
    }:
        return "I"
    return "N"


def effective_results(
    parent: Sequence[EvaluationResult],
    continuation: Sequence[EvaluationResult],
) -> list[EvaluationResult]:
    """Return one authoritative final observation per task without deleting history."""

    by_task = {item.task_id: item for item in parent}
    by_task.update({item.task_id: item for item in continuation})
    return [by_task[task_id] for task_id in sorted(by_task)]


def stop_rule_reason(
    config_name: str,
    results: Mapping[str, EvaluationResult],
    recent_final_results: Sequence[EvaluationResult],
) -> str | None:
    """Apply the frozen H2.3C final-infrastructure stop rules."""

    assigned = len(results)
    infrastructure = sum(item.infrastructure_failure for item in results.values())
    if len(recent_final_results) >= 3 and all(
        item.infrastructure_failure for item in recent_final_results[-3:]
    ):
        return (
            f"{config_name} stopped after three consecutive final "
            "infrastructure failures."
        )
    if assigned >= 5 and infrastructure / assigned >= 0.40:
        return (
            f"{config_name} campaign infrastructure unstable "
            f"({infrastructure}/{assigned} assigned tasks)."
        )
    return None


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CampaignError(f"Invalid or missing campaign artifact: {path}.") from error


def _append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = redact_secret_values(payload)
    rendered = json.dumps(safe, ensure_ascii=False, sort_keys=True)
    assert_secret_free_payload(rendered)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered + "\n")


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
                f"Malformed H2.3C JSONL record at line {number}."
            ) from error
        if not isinstance(value, dict):
            raise CampaignError(f"Invalid H2.3C JSONL record at line {number}.")
        records.append(value)
    return records


def _experiment_from_payload(payload: object) -> EvaluationExperiment:
    return EvaluationExperiment.model_validate(payload)


class H23Continuation:
    """Validated, restartable H2.3C controller with a separate 60-run budget."""

    def __init__(
        self,
        tasks: Sequence[EvaluationTask],
        *,
        dataset: str,
        parent_workspace_root: str | Path,
        workspace_root: str | Path,
        parent_output_root: str | Path,
        output_root: str | Path,
        source_root: str | Path,
        model_settings: ProductionModelSettings,
    ) -> None:
        self.tasks = sorted(
            (EvaluationTask.model_validate(item) for item in tasks),
            key=lambda item: item.id,
        )
        self.dataset = dataset
        self.parent_workspace = Path(parent_workspace_root).resolve(strict=True)
        self.workspace = Path(workspace_root).resolve()
        self.parent_output = Path(parent_output_root).resolve(strict=True)
        self.output = Path(output_root).resolve()
        self.final_output = self.parent_output / "final"
        self.source_root = Path(source_root).resolve(strict=True)
        self.settings = model_settings
        self.parent_storage = EvaluationStorage(
            self.parent_workspace / "evaluation.db",
            self.parent_workspace / "results",
        )
        self.storage = EvaluationStorage(
            self.workspace / "evaluation.db", self.workspace / "results"
        )
        self.artifacts_root = self.parent_workspace / "artifacts"
        self.budget = LiveRunBudget(
            self.workspace / "live-task-runs.jsonl",
            maximum=MAX_H2_3C_LIVE_TASK_RUNS,
        )
        self.history_path = self.output / "results.jsonl"
        self.manifest_path = self.output / "manifest.json"
        self.preflight_path = self.output / "preflight.json"
        self.config_path = self.output / "config.json"
        self.verification_path = self.output / "frozen-verification.json"
        self.parent_manifest = H23CampaignManifest.model_validate(
            _load_json(self.parent_output / "manifest.json")
        )
        self.formal, self.smoke = freeze_h2_3_tasks(self.tasks)
        self.tasks_by_id = {item.id: item for item in self.formal}
        self.specifications = _evaluator_specifications(
            self.tasks,
            evaluator_timeout_seconds=self.parent_manifest.evaluator_timeout_seconds,
        )
        self.parent_experiments = self._load_parent_experiments()
        self.parent_results = {
            name: self.parent_storage.list_results(self.parent_experiments[name].id)
            for name in CONFIG_ORDER
        }
        self.recovery_results = [
            item
            for item in self.parent_results["no_correction"]
            if item.infrastructure_failure
        ]
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)
        self.experiments = self._load_or_create_continuation_experiments()
        self.tracker = _IntegrityTracker(
            tasks=[*self.formal, *self.smoke],
            source_state=source_snapshots([*self.formal, *self.smoke]),
            specifications=self.specifications,
            artifacts_root=self.artifacts_root,
        )

    def _load_parent_experiments(self) -> dict[str, EvaluationExperiment]:
        experiments: dict[str, EvaluationExperiment] = {}
        for name in ("smoke", *CONFIG_ORDER):
            identifier = self.parent_manifest.experiment_ids[name]
            value = self.parent_storage.get_experiment(identifier)
            if value is None:
                raise CampaignError(f"Missing parent H2.3 experiment: {name}.")
            experiments[name] = value
        return experiments

    def _load_or_create_continuation_experiments(
        self,
    ) -> dict[str, EvaluationExperiment]:
        if self.config_path.is_file():
            payload = _load_json(self.config_path)
            experiments = {
                name: _experiment_from_payload(payload["experiments"][name])
                for name in H2_3C_CONFIGS
            }
        else:
            experiments = {
                name: create_experiment(
                    f"h2-3c-{name.replace('_', '-')}",
                    self.dataset,
                    self.parent_experiments[name].config,
                    git_commit=self.parent_experiments[name].git_commit,
                )
                for name in H2_3C_CONFIGS
            }
            _write_json(
                self.config_path,
                {
                    "continuation_id": H2_3C_CONTINUATION_ID,
                    "parent_campaign": H2_3C_PARENT_CAMPAIGN,
                    "maximum_live_task_runs": MAX_H2_3C_LIVE_TASK_RUNS,
                    "planned_base_recovery_task_runs": H2_3C_BASE_TASK_RUNS,
                    "infrastructure_retry_reserve": H2_3C_RETRY_RESERVE,
                    "experiments": {
                        name: value.model_dump(mode="json")
                        for name, value in experiments.items()
                    },
                },
            )
        for name, experiment in experiments.items():
            parent = self.parent_experiments[name]
            if (
                experiment.dataset != self.dataset
                or experiment.config != parent.config
                or experiment.model_name != parent.model_name
                or experiment.llm_execution_kind != "live"
            ):
                raise CampaignError(f"H2.3C experiment identity changed: {name}.")
            self.storage.save_experiment(experiment)
        return experiments

    def validate_frozen_state(self) -> dict[str, Any]:
        _validate_model_settings(self.settings)
        manifest = self.parent_manifest
        expected_model = (
            manifest.configured_provider,
            manifest.api_base_url,
            manifest.configured_model,
            manifest.model_temperature,
            manifest.model_seed,
            manifest.reasoning_effort,
            manifest.max_completion_tokens,
        )
        actual_model = (
            self.settings.provider,
            self.settings.base_url,
            self.settings.model,
            self.settings.temperature,
            self.settings.seed,
            self.settings.reasoning_effort,
            self.settings.max_completion_tokens,
        )
        if actual_model != expected_model:
            raise CampaignError("H2.3C provider/model identity changed.")
        if manifest.dataset != self.dataset:
            raise CampaignError("H2.3C dataset identity changed.")
        if [item.id for item in self.formal] != manifest.formal_task_ids:
            raise CampaignError("H2.3C frozen formal task IDs changed.")
        expected_tasks = [
            _frozen_task(item, self.specifications[item.id]) for item in self.formal
        ]
        if expected_tasks != manifest.tasks:
            raise CampaignError("H2.3C frozen task/evaluator identity changed.")
        task_set = _load_json(self.parent_output / "task-set.json")
        if task_set.get("formal_task_ids") != manifest.formal_task_ids:
            raise CampaignError("H2.3C task-set artifact changed.")
        evaluator = _load_json(self.parent_output / "evaluator.json")
        expected_specs = {
            task_id: specification.model_dump(mode="json")
            for task_id, specification in sorted(self.specifications.items())
        }
        if (
            evaluator.get("version") != manifest.evaluator_version
            or evaluator.get("task_specifications") != expected_specs
            or not evaluator.get("preflight_succeeded")
        ):
            raise CampaignError("H2.3C frozen evaluator specification changed.")
        current_fingerprint = preflight_local_evaluator(
            self.specifications[self.formal[0].id]
        )
        if current_fingerprint.model_dump(mode="json") != evaluator.get(
            "preflight_fingerprint"
        ):
            raise CampaignError("H2.3C evaluator environment changed.")
        if len(self.parent_results["full"]) != 17:
            raise CampaignError("H2.3C parent Full result set is incomplete.")
        if len({item.task_id for item in self.parent_results["full"]}) != 17:
            raise CampaignError("H2.3C parent Full result set has duplicate tasks.")
        if len(self.parent_results["no_correction"]) != 5:
            raise CampaignError("H2.3C parent No-Correction state is unexpected.")
        if len(self.recovery_results) != 2 or any(
            item.execution_attempt != 2 for item in self.recovery_results
        ):
            raise CampaignError(
                "H2.3C requires exactly two twice-failed No-Correction recoveries."
            )
        semantic = agent_semantic_fingerprint(self.source_root)
        parent_created = datetime.fromisoformat(manifest.created_at)
        late_files = [
            item["path"]
            for item in semantic["files"]
            if datetime.fromisoformat(item["last_modified"]) > parent_created
        ]
        if late_files:
            raise CampaignError(
                "Agent-semantic files changed after the parent H2.3 campaign: "
                + ", ".join(late_files)
                + "."
            )
        hidden_digests = {
            item.id: self.specifications[item.id].digest for item in self.formal
        }
        verification = {
            "continuation_id": H2_3C_CONTINUATION_ID,
            "parent_campaign": H2_3C_PARENT_CAMPAIGN,
            "parent_manifest_sha256": _sha256_file(
                self.parent_output / "manifest.json"
            ),
            "task_set_sha256": _sha256_file(self.parent_output / "task-set.json"),
            "config_sha256": _sha256_file(self.parent_output / "config.json"),
            "evaluator_sha256": _sha256_file(self.parent_output / "evaluator.json"),
            "parent_source_tree_digest": manifest.source_tree_digest_before,
            "current_source_tree_digest": source_tree_digest(self.source_root),
            "coarse_source_digest_matches": (
                source_tree_digest(self.source_root)
                == manifest.source_tree_digest_before
            ),
            "agent_semantic_fingerprint": semantic,
            "agent_semantic_files_all_predate_parent_campaign": True,
            "allowed_post_campaign_changes": [
                "README.md",
                "evaluation/campaigns/**",
                "evaluation orchestration/report code",
            ],
            "formal_task_count": len(self.formal),
            "formal_task_ids": [item.id for item in self.formal],
            "evaluator_kind": "local",
            "evaluator_version": manifest.evaluator_version,
            "evaluator_hidden_test_digests": hidden_digests,
            "evaluator_environment": current_fingerprint.model_dump(mode="json"),
            "python_executable": manifest.python_executable,
            "full_parent_results": len(self.parent_results["full"]),
            "full_parent_valid_predictions": sum(
                item.valid_prediction for item in self.parent_results["full"]
            ),
            "no_correction_parent_valid_predictions": sum(
                item.valid_prediction for item in self.parent_results["no_correction"]
            ),
            "historical_recovery_task_ids": sorted(
                item.task_id for item in self.recovery_results
            ),
            "verified_at": utc_now(),
        }
        _write_json(self.verification_path, verification)
        return verification

    def _manifest_payload(
        self, *, status: str, stop_reason: str | None
    ) -> dict[str, Any]:
        prior = _load_json(self.manifest_path) if self.manifest_path.is_file() else {}
        return {
            "id": prior.get("id", "h2-3c-continuation"),
            "stage": "H2.3C",
            "continuation_id": H2_3C_CONTINUATION_ID,
            "parent_campaign": H2_3C_PARENT_CAMPAIGN,
            "parent_campaign_id": self.parent_manifest.id,
            "created_at": prior.get("created_at", utc_now()),
            "updated_at": utc_now(),
            "status": status,
            "stop_reason": (
                redact_environment_secrets(stop_reason)[:1000] if stop_reason else None
            ),
            "configured_provider": self.settings.provider,
            "api_base_url": self.settings.base_url,
            "configured_model": self.settings.model,
            "temperature": self.settings.temperature,
            "reasoning_effort": self.settings.reasoning_effort,
            "max_completion_tokens": self.settings.max_completion_tokens,
            "workers": 1,
            "maximum_live_task_runs": MAX_H2_3C_LIVE_TASK_RUNS,
            "planned_base_recovery_task_runs": H2_3C_BASE_TASK_RUNS,
            "infrastructure_retry_reserve": H2_3C_RETRY_RESERVE,
            "live_task_runs_consumed": self.budget.consumed,
            "experiment_ids": {
                name: value.id for name, value in self.experiments.items()
            },
            "formal_task_ids": [item.id for item in self.formal],
            "historical_recovery_task_ids": sorted(
                item.task_id for item in self.recovery_results
            ),
        }

    def _write_manifest(self, *, status: str, stop_reason: str | None = None) -> None:
        _write_json(
            self.manifest_path,
            self._manifest_payload(status=status, stop_reason=stop_reason),
        )

    def preflight(self) -> LivePreflightResult:
        if self.preflight_path.is_file():
            result = LivePreflightResult.model_validate(_load_json(self.preflight_path))
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
                raise CampaignError("H2.3C persisted provider preflight changed.")
            return result
        result = run_live_preflight(settings=self.settings)
        _write_json(self.preflight_path, result)
        return result

    def _prior_attempt_reference(self, task_id: str) -> str:
        parent = next(item for item in self.recovery_results if item.task_id == task_id)
        return (
            f"{parent.experiment}/{task_id}/execution_attempt="
            f"{parent.execution_attempt}"
        )

    def _record_attempt(
        self,
        *,
        config_name: str,
        result: EvaluationResult,
        ordinal: int,
        continuation_retry: bool,
        prior_campaign_attempt: str | None,
    ) -> None:
        _append_jsonl(
            self.history_path,
            {
                "continuation_id": H2_3C_CONTINUATION_ID,
                "parent_campaign": H2_3C_PARENT_CAMPAIGN,
                "config": config_name,
                "continuation_retry": continuation_retry,
                "prior_campaign_attempt": prior_campaign_attempt,
                "live_task_run_ordinal": ordinal,
                "result": _result_record(result),
            },
        )

    def _execute_attempt(
        self,
        config_name: str,
        task: EvaluationTask,
        *,
        execution_attempt: int,
        continuation_retry: bool,
        prior_campaign_attempt: str | None = None,
    ) -> EvaluationResult:
        experiment = self.experiments[config_name]
        reservation = self.budget.reserve(
            experiment_id=experiment.id,
            task_id=task.id,
            config_name=config_name,
        )
        print(
            f"Starting H2.3C live task-run {reservation.ordinal}/"
            f"{MAX_H2_3C_LIVE_TASK_RUNS}: {config_name}/{task.id}",
            flush=True,
        )
        self.storage.save_task(task)
        result = run_evaluation_task(
            task,
            experiment.config,
            workspace_root=str(self.workspace / "workspaces"),
            artifacts_root=str(self.artifacts_root),
            experiment_id=experiment.id,
            git_commit=experiment.git_commit,
            execution_attempt=execution_attempt,
        )
        if continuation_retry or execution_attempt > 1:
            result = result.model_copy(update={"required_infrastructure_retry": True})
        self.storage.save_result(result)
        self.tracker.observe_attempt(result)
        self._record_attempt(
            config_name=config_name,
            result=result,
            ordinal=reservation.ordinal,
            continuation_retry=continuation_retry,
            prior_campaign_attempt=prior_campaign_attempt,
        )
        self._write_manifest(status="running")
        return result

    def _run_recovery_probe(self, task: EvaluationTask) -> EvaluationResult:
        existing = self.storage.get_result(
            self.experiments["no_correction"].id, task.id
        )
        if existing is not None:
            return existing
        return self._execute_attempt(
            "no_correction",
            task,
            execution_attempt=1,
            continuation_retry=True,
            prior_campaign_attempt=self._prior_attempt_reference(task.id),
        )

    def _run_new_task(self, config_name: str, task: EvaluationTask) -> EvaluationResult:
        experiment = self.experiments[config_name]
        existing = self.storage.get_result(experiment.id, task.id)
        if existing is not None and (
            not existing.infrastructure_failure or existing.execution_attempt >= 2
        ):
            return existing
        first_attempt = 2 if existing is not None else 1
        result = self._execute_attempt(
            config_name,
            task,
            execution_attempt=first_attempt,
            continuation_retry=False,
        )
        if result.infrastructure_failure and first_attempt == 1:
            result = self._execute_attempt(
                config_name,
                task,
                execution_attempt=2,
                continuation_retry=False,
            )
        return result

    def _continuation_results(self, config_name: str) -> list[EvaluationResult]:
        return self.storage.list_results(self.experiments[config_name].id)

    def _effective_by_config(self) -> dict[str, list[EvaluationResult]]:
        return {
            "full": list(self.parent_results["full"]),
            "no_correction": effective_results(
                self.parent_results["no_correction"],
                self._continuation_results("no_correction"),
            ),
            "no_exploration": self._continuation_results("no_exploration"),
            "no_agentic_test": self._continuation_results("no_agentic_test"),
        }

    def _recent_final_results(self, config_name: str) -> list[EvaluationResult]:
        records = _load_jsonl(self.history_path)
        latest: dict[str, EvaluationResult] = {}
        order: list[str] = []
        for record in records:
            if record.get("config") != config_name:
                continue
            result = EvaluationResult.model_validate(record["result"])
            if result.task_id not in latest:
                order.append(result.task_id)
            latest[result.task_id] = result
        return [latest[task_id] for task_id in order]

    def _check_stop(self, config_name: str) -> None:
        effective = {
            item.task_id: item for item in self._effective_by_config()[config_name]
        }
        reason = stop_rule_reason(
            config_name,
            effective,
            self._recent_final_results(config_name),
        )
        if reason:
            raise CampaignStopped(reason)

    def run_configurations(self) -> None:
        recovery_tasks = [
            self.tasks_by_id[item.task_id]
            for item in sorted(self.recovery_results, key=lambda value: value.task_id)
        ]
        probe = [self._run_recovery_probe(task) for task in recovery_tasks]
        if len(probe) != 2:
            raise CampaignError("H2.3C provider recovery probe is incomplete.")
        if all(item.infrastructure_failure for item in probe):
            raise CampaignStopped(
                "H2.3C provider recovery probe failed 2/2; provider infrastructure "
                "remains unstable."
            )
        self._check_stop("no_correction")
        historical_ids = {item.task_id for item in self.parent_results["no_correction"]}
        for task in self.formal:
            if task.id in historical_ids:
                continue
            self._run_new_task("no_correction", task)
            self._check_stop("no_correction")
        for config_name in ("no_exploration", "no_agentic_test"):
            for task in self.formal:
                self._run_new_task(config_name, task)
                self._check_stop(config_name)

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
                    name: _status_code(maps[name].get(task.id)) for name in CONFIG_ORDER
                },
            }
            for task in self.formal
        ]
        counts: dict[str, dict[str, int]] = {}
        for name in H2_3C_CONFIGS:
            values = Counter()
            for row in matrix:
                left = row["full"]
                right = row[name]
                if left in {"R", "U"} and right in {"R", "U"}:
                    values["comparable_pairs"] += 1
                    if left == "R" and right == "U":
                        values["full_wins"] += 1
                    elif left == "U" and right == "R":
                        values["ablation_wins"] += 1
                    elif left == "R":
                        values["ties_resolved"] += 1
                    else:
                        values["ties_unresolved"] += 1
                elif "I" in {left, right}:
                    values["excluded_infra_pairs"] += 1
                else:
                    values["excluded_non_candidate_pairs"] += 1
            counts[name] = {
                key: values[key]
                for key in (
                    "comparable_pairs",
                    "full_wins",
                    "ablation_wins",
                    "ties_resolved",
                    "ties_unresolved",
                    "excluded_infra_pairs",
                    "excluded_non_candidate_pairs",
                )
            }
        return matrix, counts

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
        known = lambda field: sum(
            value for item in results if (value := getattr(item, field)) is not None
        )
        return {
            "tasks_with_complete_usage": len(complete),
            "tasks_with_incomplete_usage": len(results) - len(complete),
            "telemetry_completeness_rate": (
                len(complete) / len(results) if results else 0.0
            ),
            "known_recorded_llm_calls": known("llm_calls"),
            "known_recorded_input_tokens": known("input_tokens"),
            "known_recorded_output_tokens": known("output_tokens"),
            "known_recorded_total_tokens": known("total_tokens"),
            "average_total_tokens": (
                known("total_tokens") / len(results)
                if results and len(complete) == len(results)
                else None
            ),
            "runtime_seconds": sum(item.duration_seconds for item in results),
            "correction_rounds": sum(item.correction_rounds_used for item in results),
            "known_tool_calls": known("tool_calls"),
            "known_test_calls": known("test_calls"),
            "cost": None,
        }

    @staticmethod
    def _failure_taxonomy(results: Sequence[EvaluationResult]) -> dict[str, int]:
        taxonomy = Counter()
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
        return {key: taxonomy[key] for key in sorted(allowed)}

    def _write_failure_analysis(
        self, results_by_config: Mapping[str, Sequence[EvaluationResult]]
    ) -> None:
        lines = [
            "# H2.3 final capability failure analysis",
            "",
            "Bounded summaries only; hidden evaluator source is not reproduced.",
            "",
        ]
        task_categories = {
            item.id: ", ".join(item.metadata.get("difficulty", []))
            for item in self.formal
        }
        for name in CONFIG_ORDER:
            lines.extend([f"## {name}", ""])
            failures = [
                item
                for item in results_by_config[name]
                if item.valid_prediction and not item.final_resolved
            ]
            if not failures:
                lines.extend(["No valid-prediction unresolved tasks.", ""])
                continue
            for item in sorted(failures, key=lambda value: value.task_id):
                lines.extend(
                    [
                        f"### {item.task_id}",
                        "",
                        f"- Task category: {task_categories[item.task_id] or 'unspecified'}",
                        f"- Plan status: {'succeeded' if item.planning_succeeded else 'failed'}",
                        f"- Changed files: {', '.join(item.changed_files) or 'none'}",
                        f"- Verification: {item.verification_succeeded}",
                        f"- Review good: {item.review_good}",
                        f"- Correction rounds: {item.correction_rounds_used}",
                        f"- Hidden evaluator summary: {(item.failure_reason or 'assertion failure')[:500]}",
                        "",
                    ]
                )
        (self.final_output / "failure_analysis.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    def _write_correction_analysis(self) -> None:
        records = _load_jsonl(self.final_output / "candidate-regrade.jsonl")
        by_config: dict[str, list[dict[str, Any]]] = {name: [] for name in CONFIG_ORDER}
        for record in records:
            by_config[record["config"]].append(record["analysis"])
        payload: dict[str, Any] = {}
        lines = ["# H2.3 final correction analysis", ""]
        for name in CONFIG_ORDER:
            analyses = by_config[name]
            entered = [
                item
                for item in analyses
                if item["first"]["candidate_attempt"]
                != item["final"]["candidate_attempt"]
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
                "remained_resolved": sum(
                    item["first_attempt_resolved"] and item["final_resolved"]
                    for item in analyses
                ),
                "remained_unresolved": sum(
                    not item["first_attempt_resolved"] and not item["final_resolved"]
                    for item in analyses
                ),
                "not_applicable_without_candidate": 17 - len(analyses),
            }
            payload[name] = values
            lines.extend(
                [
                    f"## {name}",
                    "",
                    f"- Entered correction: {values['entered_correction']}",
                    f"- Rescued by correction: {values['rescued_by_correction']}",
                    f"- Regressed after correction: {values['regressed_after_correction']}",
                    f"- Remained resolved: {values['remained_resolved']}",
                    f"- Remained unresolved: {values['remained_unresolved']}",
                    f"- No candidate / not applicable: {values['not_applicable_without_candidate']}",
                    "",
                ]
            )
        _write_json(self.final_output / "correction-analysis.json", payload)
        (self.final_output / "correction_analysis.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    def _write_markdown_comparison(
        self,
        comparison: Mapping[str, Any],
        matrix: Sequence[Mapping[str, str]],
        counts: Mapping[str, Mapping[str, int]],
    ) -> None:
        lines = [
            "# H2.3 final paired controlled benchmark",
            "",
            "Exploratory controlled benchmark (N=17); observed differences are not causal proof.",
            "",
            "R = resolved valid prediction; U = unresolved valid prediction; I = infrastructure failure; N = no candidate from a non-infrastructure agent failure.",
            "",
            "| Task | Full | No Correction | No Exploration | No Agentic Test |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in matrix:
            lines.append(
                f"| {row['task_id']} | {row['full']} | {row['no_correction']} | "
                f"{row['no_exploration']} | {row['no_agentic_test']} |"
            )
        lines.extend(["", "## Comparable-pair counts", ""])
        for name in H2_3C_CONFIGS:
            value = counts[name]
            paired = comparison["paired"][name]
            lines.append(
                f"- Full − {name}: comparable {value['comparable_pairs']}; "
                f"Full wins {value['full_wins']}; ablation wins {value['ablation_wins']}; "
                f"ties resolved {value['ties_resolved']}; ties unresolved "
                f"{value['ties_unresolved']}; excluded infra {value['excluded_infra_pairs']}; "
                f"paired capability delta {paired['capability_delta']}."
            )
        lines.extend(
            [
                "",
                "Capability comparisons include only pairs with valid predictions in both configurations. End-to-end operational comparisons retain resolved, unresolved, and infrastructure outcomes across all 17 tasks.",
                "",
            ]
        )
        (self.final_output / "comparison.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    def write_final_reports(self) -> list[Path]:
        results_by_config = self._effective_by_config()
        for name in CONFIG_ORDER:
            if len(results_by_config[name]) != 17:
                raise CampaignError(
                    f"H2.3C cannot finalize {name}: expected 17 final tasks, "
                    f"found {len(results_by_config[name])}."
                )
        self.final_output.mkdir(parents=True, exist_ok=True)
        report_paths = write_h2_3_reports(
            self.final_output,
            self.formal,
            results_by_config,
            artifacts_root=self.artifacts_root,
            regrade_workspace_root=self.workspace / "regrade-workspaces",
        )
        comparison_path = self.final_output / "comparison.json"
        comparison = _load_json(comparison_path)
        matrix, counts = self._paired_details(results_by_config)
        comparison["paired_matrix"] = matrix
        comparison["paired_counts"] = counts
        comparison["small_n_statement"] = (
            "Exploratory controlled benchmark; N=17 is not statistical or causal proof."
        )
        for name in CONFIG_ORDER:
            valid = [
                item.final_resolved
                for item in results_by_config[name]
                if item.valid_prediction or item.status in {"resolved", "unresolved"}
            ]
            comparison["configs"][name]["capability_resolve_95_ci"] = (
                bootstrap_resolve_ci(
                    valid,
                    seed=DEFAULT_BOOTSTRAP_SEED,
                    resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
                ).model_dump(mode="json")
            )
        _write_json(comparison_path, comparison)
        self._write_markdown_comparison(comparison, matrix, counts)

        telemetry = {
            name: self._telemetry(results_by_config[name]) for name in CONFIG_ORDER
        }
        _write_json(self.final_output / "telemetry.json", telemetry)
        telemetry_lines = ["# H2.3 final telemetry", ""]
        for name in CONFIG_ORDER:
            value = telemetry[name]
            telemetry_lines.extend(
                [
                    f"## {name}",
                    "",
                    f"- Complete usage: {value['tasks_with_complete_usage']}/17",
                    f"- Telemetry completeness: {value['telemetry_completeness_rate']:.1%}",
                    f"- Known LLM calls: {value['known_recorded_llm_calls']}",
                    f"- Known recorded tokens: {value['known_recorded_total_tokens']}",
                    f"- Average tokens: {value['average_total_tokens'] if value['average_total_tokens'] is not None else 'not reported (partial telemetry)'}",
                    f"- Runtime seconds: {value['runtime_seconds']:.3f}",
                    f"- Tool calls (known): {value['known_tool_calls']}",
                    f"- Test calls (known): {value['known_test_calls']}",
                    f"- Cost: {value['cost']}",
                    "",
                ]
            )
        (self.final_output / "telemetry.md").write_text(
            "\n".join(telemetry_lines), encoding="utf-8"
        )

        failure_taxonomy = {
            name: self._failure_taxonomy(results_by_config[name])
            for name in CONFIG_ORDER
        }
        _write_json(self.final_output / "failure-taxonomy.json", failure_taxonomy)
        self._write_failure_analysis(results_by_config)
        self._write_correction_analysis()

        results_payload = {
            "source_of_truth": "H2.3 initial + H2.3C continuation",
            "configs": {
                name: {
                    "metrics": aggregate_metrics(list(results)).model_dump(mode="json"),
                    "results": [_result_record(item) for item in results],
                }
                for name, results in results_by_config.items()
            },
        }
        _write_json(self.final_output / "results.json", results_payload)
        _write_json(
            self.final_output / "manifest.json",
            {
                "stage": "H2.3 final",
                "source_of_truth": "H2.3 initial + H2.3C continuation",
                "parent_campaign_id": self.parent_manifest.id,
                "continuation_id": H2_3C_CONTINUATION_ID,
                "status": "complete",
                "formal_task_ids": [item.id for item in self.formal],
                "model": self.settings.model,
                "provider": self.settings.provider,
                "temperature": self.settings.temperature,
                "reasoning_effort": self.settings.reasoning_effort,
                "max_completion_tokens": self.settings.max_completion_tokens,
                "workers": 1,
                "analysis_seed": DEFAULT_BOOTSTRAP_SEED,
                "bootstrap_resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
                "created_at": utc_now(),
            },
        )
        _write_json(
            self.final_output / "completion.json",
            {
                "status": "complete",
                "all_four_configs_have_17_final_tasks": True,
                "full_was_not_rerun": True,
                "parent_history_preserved": True,
                "capability_failures_retried": False,
                "candidate_snapshot_protocol": True,
                "exact_candidate_regrade_complete": True,
                "paired_matrix_complete": True,
                "bootstrap_resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
                "analysis_seed": DEFAULT_BOOTSTRAP_SEED,
                "h2_3c_live_task_runs_consumed": self.budget.consumed,
                "doctor": "pending final environment check",
                "official_docker_smoke": "pending final environment check",
            },
        )
        security = scan_persisted_artifacts(
            [
                self.parent_workspace / "artifacts",
                self.parent_workspace / "results",
                self.workspace,
                self.parent_output,
            ]
        )
        security_path = _write_json(self.final_output / "security-scan.json", security)
        return [*report_paths, security_path]

    def write_stopped_reports(self, stop_reason: str) -> list[Path]:
        """Write a truthful non-authoritative view after a protocol stop."""

        results_by_config = self._effective_by_config()
        report_paths = write_h2_3_reports(
            self.output,
            self.formal,
            results_by_config,
            artifacts_root=self.artifacts_root,
            regrade_workspace_root=self.workspace / "stopped-regrade-workspaces",
        )
        comparison_path = self.output / "comparison.json"
        comparison = _load_json(comparison_path)
        matrix, counts = self._paired_details(results_by_config)
        comparison["paired_matrix"] = matrix
        comparison["paired_counts"] = counts
        comparison["status"] = "stopped"
        comparison["stop_reason"] = stop_reason
        comparison["authoritative_final"] = False
        comparison["small_n_statement"] = (
            "Exploratory controlled benchmark; incomplete configurations and "
            "infrastructure failures preclude a final paired conclusion."
        )
        for name in CONFIG_ORDER:
            valid = [
                item.final_resolved
                for item in results_by_config[name]
                if item.valid_prediction or item.status in {"resolved", "unresolved"}
            ]
            comparison["configs"][name]["capability_resolve_95_ci"] = (
                bootstrap_resolve_ci(
                    valid,
                    seed=DEFAULT_BOOTSTRAP_SEED,
                    resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
                ).model_dump(mode="json")
            )
        _write_json(comparison_path, comparison)

        telemetry = {
            name: self._telemetry(results_by_config[name]) for name in CONFIG_ORDER
        }
        _write_json(self.output / "telemetry.json", telemetry)
        failure_taxonomy = {
            name: self._failure_taxonomy(results_by_config[name])
            for name in CONFIG_ORDER
        }
        _write_json(self.output / "failure-taxonomy.json", failure_taxonomy)
        _write_json(
            self.output / "results.json",
            {
                "status": "stopped",
                "authoritative_final": False,
                "stop_reason": stop_reason,
                "source_of_truth": "H2.3 initial + partial H2.3C continuation",
                "configs": {
                    name: {
                        "metrics": aggregate_metrics(list(results)).model_dump(
                            mode="json"
                        ),
                        "results": [_result_record(item) for item in results],
                    }
                    for name, results in results_by_config.items()
                },
            },
        )
        _write_json(
            self.output / "completion.json",
            {
                "status": "stopped",
                "authoritative_final": False,
                "stop_reason": stop_reason,
                "provider_infrastructure_stop": True,
                "full_was_not_rerun": True,
                "parent_history_preserved": True,
                "capability_failures_retried": False,
                "h2_3c_live_task_runs_consumed": self.budget.consumed,
                "configuration_task_counts": {
                    name: len(results) for name, results in results_by_config.items()
                },
                "no_agentic_test_executed": bool(results_by_config["no_agentic_test"]),
                "doctor": {
                    "swebench_installed": True,
                    "docker_cli_available": True,
                    "docker_daemon_available": False,
                },
                "official_docker_smoke": "blocked",
            },
        )
        security = scan_persisted_artifacts(
            [
                self.parent_workspace / "artifacts",
                self.parent_workspace / "results",
                self.workspace,
                self.parent_output,
            ]
        )
        security_path = _write_json(self.output / "security-scan.json", security)
        return [*report_paths, security_path]

    def run(self) -> dict[str, Any]:
        self.validate_frozen_state()
        self._write_manifest(status="running")
        preflight = self.preflight()
        self.run_configurations()
        self.write_final_reports()
        self._write_manifest(status="complete")
        return {
            "manifest": str(self.manifest_path),
            "final_output": str(self.final_output),
            "preflight": preflight.model_dump(mode="json"),
            "live_task_runs_consumed": self.budget.consumed,
        }


def run_h2_3c_campaign(
    *,
    dataset_path: str | Path,
    parent_workspace_root: str | Path,
    workspace_root: str | Path,
    parent_output_root: str | Path,
    output_root: str | Path,
    source_root: str | Path,
    model_settings: ProductionModelSettings | None = None,
) -> dict[str, Any]:
    """Validate and execute the append-only H2.3C continuation."""

    settings = model_settings or prepare_live_environment()
    dataset_file = Path(dataset_path).resolve(strict=True)
    controller = H23Continuation(
        LocalTaskDataset(dataset_file).load(),
        dataset=f"local:{dataset_file}",
        parent_workspace_root=parent_workspace_root,
        workspace_root=workspace_root,
        parent_output_root=parent_output_root,
        output_root=output_root,
        source_root=source_root,
        model_settings=settings,
    )
    try:
        return controller.run()
    except BaseException as error:
        controller._write_manifest(
            status="stopped",
            stop_reason=str(error).strip() or type(error).__name__,
        )
        if isinstance(error, CampaignStopped):
            controller.write_stopped_reports(str(error).strip() or type(error).__name__)
        raise
