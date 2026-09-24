"""Offline, append-preserving regrade support for H2.2 campaign patches."""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess  # nosec B404
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from evaluation.adapters.local_tasks import LocalTaskDataset
from evaluation.campaign import (
    CAMPAIGN_CONFIG_KEYS,
    PHASE_ARTIFACT_NAMES,
    BenchmarkCampaign,
    CampaignError,
    assert_source_snapshots,
    source_snapshots,
)
from evaluation.campaign_reports import CONFIG_LABELS, SMALL_SAMPLE_NOTE
from evaluation.evaluator import LOCAL_EVALUATOR_VERSION, run_local_evaluator
from evaluation.metrics import aggregate_metrics
from evaluation.models import EvaluationResult, EvaluationTask
from evaluation.statistics import (
    bootstrap_paired_delta_ci,
    bootstrap_resolve_ci,
)
from evaluation.storage import EvaluationStorage
from evaluation.workspace import prepare_task_workspace

REGRADE_REASON = (
    "Portable dataset commands beginning with 'python' resolved to a system "
    "interpreter without pytest inside the isolated worker. Pytest's import "
    "failure returned exit code 1 and was misclassified as an assertion failure."
)
PHASE_ORDER = ("smoke", "pilot", *CAMPAIGN_CONFIG_KEYS)


class RegradeRecord(BaseModel):
    """One evaluator-only replay of an immutable saved prediction patch."""

    model_config = ConfigDict(extra="forbid")

    phase: str
    experiment_id: str
    task_id: str
    source_result_sha256: str
    prediction_sha256: str | None
    original_status: str
    original_failure_category: str | None
    correction_rounds_used: int
    regrade_status: Literal["passed", "failed", "error", "timeout", "no_prediction"]
    final_resolved: bool
    first_attempt_resolved: bool | None
    evaluator_exit_code: int | None = None
    evaluator_failure_category: str | None = None
    evaluator_failure_reason: str | None = None
    evaluator_duration_seconds: float
    stdout_sha256: str | None = None
    stderr_sha256: str | None = None
    evaluator_version: str = LOCAL_EVALUATOR_VERSION
    original_history_preserved: bool = True


class CampaignRegradeOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_path: str
    comparison_path: str
    records_path: str
    failure_analysis_path: str
    grading_notice_path: str
    regraded_predictions: int
    unavailable_predictions: int
    llm_api_calls: Literal[0] = 0


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(payload: str) -> str:
    return _sha256_bytes(payload.encode("utf-8"))


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, BaseModel):
        value = payload.model_dump(mode="json")
    else:
        value = payload
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _load_campaign(output_root: Path) -> tuple[BenchmarkCampaign, bytes]:
    path = output_root / "manifest.json"
    try:
        payload = path.read_bytes()
        manifest = BenchmarkCampaign.model_validate_json(payload)
    except (OSError, ValueError) as error:
        raise CampaignError(f"Could not load campaign manifest: {error}") from error
    if manifest.status != "complete":
        raise CampaignError("Only a complete campaign can be regraded.")
    if tuple(manifest.configs) != CAMPAIGN_CONFIG_KEYS:
        raise CampaignError("Campaign configuration order does not match H2.2.")
    if manifest.llm_execution_kind != "live":
        raise CampaignError("Only live campaign predictions can be regraded here.")
    return manifest, payload


def _dataset_path(dataset: str) -> Path:
    if not dataset.startswith("local:"):
        raise CampaignError("Offline campaign regrade requires a local dataset.")
    try:
        return Path(dataset.removeprefix("local:")).resolve(strict=True)
    except OSError as error:
        raise CampaignError(f"Campaign dataset is unavailable: {error}") from error


def _load_tasks(manifest: BenchmarkCampaign) -> dict[str, EvaluationTask]:
    tasks = {item.id: item for item in LocalTaskDataset(_dataset_path(manifest.dataset)).load()}
    if sorted(tasks) != sorted(manifest.task_ids):
        raise CampaignError("Frozen campaign task IDs do not match the dataset.")
    frozen = {item.task_id: item for item in manifest.tasks}
    for task_id, task in tasks.items():
        expected = frozen[task_id]
        if task.base_commit != expected.base_commit:
            raise CampaignError(f"Base commit changed for {task_id}.")
        if Path(task.repository).resolve() != Path(expected.repository).resolve():
            raise CampaignError(f"Repository changed for {task_id}.")
    return tasks


def _phase_task_ids(output_root: Path, phase: str, experiment_id: str) -> list[str]:
    path = output_root / PHASE_ARTIFACT_NAMES[phase]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        persisted_id = payload["experiment"]["id"]
        task_ids = [item["task_id"] for item in payload["results"]]
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise CampaignError(f"Invalid {phase} artifact: {error}") from error
    if persisted_id != experiment_id:
        raise CampaignError(f"Experiment identity mismatch in {phase} artifact.")
    if len(task_ids) != len(set(task_ids)):
        raise CampaignError(f"Duplicate task identity in {phase} artifact.")
    return task_ids


def _load_results(
    storage: EvaluationStorage,
    output_root: Path,
    manifest: BenchmarkCampaign,
) -> dict[str, list[EvaluationResult]]:
    results_by_phase: dict[str, list[EvaluationResult]] = {}
    for phase in PHASE_ORDER:
        experiment_id = manifest.experiment_ids[phase]
        experiment = storage.get_experiment(experiment_id)
        if experiment is None:
            raise CampaignError(f"Missing persisted experiment for {phase}.")
        if experiment.llm_execution_kind != "live":
            raise CampaignError(f"Non-live experiment entered {phase} regrade.")
        if experiment.config.model_name != manifest.configured_model:
            raise CampaignError(f"Configured model mismatch for {phase}.")
        expected_ids = _phase_task_ids(output_root, phase, experiment_id)
        results = storage.list_results(experiment_id)
        if sorted(item.task_id for item in results) != sorted(expected_ids):
            raise CampaignError(f"Persisted results do not match {phase} artifact.")
        if any(item.llm_execution_kind != "live" for item in results):
            raise CampaignError(f"Non-live result entered {phase} regrade.")
        results_by_phase[phase] = results
    return results_by_phase


def _apply_prediction_patch(workspace: Path, patch: str) -> None:
    try:
        completed = subprocess.run(  # nosec B603 B607
            ["git", "apply", "--binary", "--whitespace=nowarn", "-"],
            cwd=workspace,
            input=patch,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CampaignError(f"Could not replay saved prediction: {error}") from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[:1_000]
        raise CampaignError(f"Saved prediction patch did not apply: {detail}")


def _source_result_hash(result: EvaluationResult) -> str:
    payload = json.dumps(
        result.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return _sha256_text(payload)


def regrade_result(
    phase: str,
    task: EvaluationTask,
    result: EvaluationResult,
    *,
    workspace_root: str | Path,
    evaluator_timeout_seconds: float,
) -> RegradeRecord:
    """Replay one final patch without invoking RepoGraph or any LLM client."""

    common = {
        "phase": phase,
        "experiment_id": result.experiment,
        "task_id": task.id,
        "source_result_sha256": _source_result_hash(result),
        "prediction_sha256": result.prediction_sha256,
        "original_status": result.status,
        "original_failure_category": result.failure_category,
        "correction_rounds_used": result.correction_rounds_used,
    }
    if result.model_patch is None:
        return RegradeRecord(
            **common,
            regrade_status="no_prediction",
            final_resolved=False,
            first_attempt_resolved=False,
            evaluator_failure_category=result.failure_category,
            evaluator_failure_reason="No saved prediction patch was available to regrade.",
            evaluator_duration_seconds=0,
        )
    patch_hash = _sha256_text(result.model_patch)
    if patch_hash != result.prediction_sha256:
        raise CampaignError(f"Prediction hash mismatch for {phase}/{task.id}.")
    workspace, resolved_base = prepare_task_workspace(
        task.repository,
        task.base_commit,
        workspace_root,
        f"offline-regrade-{result.experiment}",
        task.id,
    )
    if resolved_base != result.base_commit:
        raise CampaignError(f"Regrade base commit mismatch for {phase}/{task.id}.")
    _apply_prediction_patch(workspace, result.model_patch)
    outcome = run_local_evaluator(
        task,
        workspace,
        timeout_seconds=evaluator_timeout_seconds,
        max_output_chars=20_000,
    )
    resolved = outcome.status == "passed"
    first_attempt = resolved if result.correction_rounds_used == 0 else None
    return RegradeRecord(
        **common,
        regrade_status=outcome.status,
        final_resolved=resolved,
        first_attempt_resolved=first_attempt,
        evaluator_exit_code=outcome.exit_code,
        evaluator_failure_category=outcome.failure_category,
        evaluator_failure_reason=outcome.failure_reason,
        evaluator_duration_seconds=outcome.duration_seconds,
        stdout_sha256=_sha256_text(outcome.stdout),
        stderr_sha256=_sha256_text(outcome.stderr),
    )


def _optional_mean_delta(
    baseline: dict[str, EvaluationResult],
    comparison: dict[str, EvaluationResult],
    attribute: str,
) -> float | None:
    values: list[float] = []
    for task_id in sorted(baseline):
        left = getattr(baseline[task_id], attribute)
        right = getattr(comparison[task_id], attribute)
        if left is None or right is None:
            return None
        values.append(float(left) - float(right))
    return float(statistics.fmean(values)) if values else None


def _config_summary(
    records: Sequence[RegradeRecord],
    original_results: Sequence[EvaluationResult],
    *,
    seed: int,
    resamples: int,
) -> dict[str, object]:
    metrics = aggregate_metrics(list(original_results))
    final_values = [item.final_resolved for item in records]
    observed_first = [
        item.first_attempt_resolved
        for item in records
        if item.first_attempt_resolved is not None
    ]
    first_confirmed = sum(value is True for value in observed_first)
    first_unknown = sum(item.first_attempt_resolved is None for item in records)
    total = len(records)
    regraded = sum(item.regrade_status != "no_prediction" for item in records)
    return {
        "tasks": total,
        "final_resolved": sum(final_values),
        "final_resolve_rate": sum(final_values) / total if total else 0.0,
        "final_resolve_rate_95_ci": bootstrap_resolve_ci(
            final_values, seed=seed, resamples=resamples
        ).model_dump(mode="json"),
        "regraded_predictions": regraded,
        "prediction_pass_rate": (
            sum(item.regrade_status == "passed" for item in records) / regraded
            if regraded
            else None
        ),
        "unavailable_predictions": total - regraded,
        "first_attempt_confirmed_resolved": first_confirmed,
        "first_attempt_observed_tasks": len(observed_first),
        "first_attempt_unknown_tasks": first_unknown,
        "first_attempt_resolve_rate_lower_bound": (
            first_confirmed / total if total else 0.0
        ),
        "first_attempt_resolve_rate_upper_bound": (
            (first_confirmed + first_unknown) / total if total else 0.0
        ),
        "average_llm_calls": metrics.average_llm_calls,
        "median_llm_calls": metrics.median_llm_calls,
        "tokens_per_task": metrics.tokens_per_task,
        "average_duration_seconds": metrics.average_duration_seconds,
        "median_duration_seconds": metrics.median_duration_seconds,
        "telemetry_complete": metrics.total_tokens is not None,
        "api_infrastructure_failures": sum(
            item.original_failure_category == "api_infrastructure_failure"
            for item in records
        ),
    }


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def _number(value: float | None, suffix: str = "") -> str:
    return "N/A" if value is None else f"{value:.2f}{suffix}"


def _build_comparison(
    manifest: BenchmarkCampaign,
    records_by_phase: dict[str, list[RegradeRecord]],
    results_by_phase: dict[str, list[EvaluationResult]],
) -> dict[str, object]:
    summaries = {
        key: _config_summary(
            records_by_phase[key],
            results_by_phase[key],
            seed=manifest.bootstrap_seed,
            resamples=manifest.bootstrap_resamples,
        )
        for key in CAMPAIGN_CONFIG_KEYS
    }
    record_maps = {
        key: {item.task_id: item for item in records_by_phase[key]}
        for key in CAMPAIGN_CONFIG_KEYS
    }
    result_maps = {
        key: {item.task_id: item for item in results_by_phase[key]}
        for key in CAMPAIGN_CONFIG_KEYS
    }
    task_ids = sorted(manifest.task_ids)
    matrix = [
        {
            "task_id": task_id,
            "base_commit": result_maps["full"][task_id].base_commit,
            "final_resolved": {
                key: record_maps[key][task_id].final_resolved
                for key in CAMPAIGN_CONFIG_KEYS
            },
            "regrade_status": {
                key: record_maps[key][task_id].regrade_status
                for key in CAMPAIGN_CONFIG_KEYS
            },
        }
        for task_id in task_ids
    ]
    deltas: list[dict[str, object]] = []
    full_final = [record_maps["full"][item].final_resolved for item in task_ids]
    for key in CAMPAIGN_CONFIG_KEYS[1:]:
        comparison_final = [record_maps[key][item].final_resolved for item in task_ids]
        token_delta = _optional_mean_delta(
            result_maps["full"], result_maps[key], "total_tokens"
        )
        if any(
            item.telemetry_incomplete
            for config_key in ("full", key)
            for item in result_maps[config_key].values()
        ):
            token_delta = None
        deltas.append(
            {
                "comparison_key": key,
                "comparison_label": CONFIG_LABELS[key],
                "full_minus_comparison_final_resolve_rate": (
                    statistics.fmean(full_final)
                    - statistics.fmean(comparison_final)
                ),
                "final_resolve_rate_delta_95_ci": bootstrap_paired_delta_ci(
                    full_final,
                    comparison_final,
                    seed=manifest.bootstrap_seed,
                    resamples=manifest.bootstrap_resamples,
                ).model_dump(mode="json"),
                "full_minus_comparison_llm_calls": _optional_mean_delta(
                    result_maps["full"], result_maps[key], "llm_calls"
                ),
                "full_minus_comparison_tokens": token_delta,
                "full_minus_comparison_runtime_seconds": _optional_mean_delta(
                    result_maps["full"], result_maps[key], "duration_seconds"
                ),
            }
        )
    attempted = [
        item
        for item in records_by_phase["full"]
        if item.correction_rounds_used > 0
    ]
    confirmed_rescued = [
        item.task_id
        for item in attempted
        if item.first_attempt_resolved is False and item.final_resolved
    ]
    possible_rescued = [
        item.task_id
        for item in attempted
        if item.first_attempt_resolved is None and item.final_resolved
    ]
    attempted_count = len(attempted)
    return {
        "comparison_valid": True,
        "grading_basis": "offline replay of immutable live-LLM prediction patches",
        "original_grading_valid": False,
        "regrade_reason": REGRADE_REASON,
        "evaluator_version": LOCAL_EVALUATOR_VERSION,
        "llm_api_calls_during_regrade": 0,
        "original_history_preserved": True,
        "small_sample_note": SMALL_SAMPLE_NOTE,
        "first_attempt_limitation": (
            "The original evaluator defect affected first-attempt grading. First "
            "candidate artifacts for corrected tasks were not persisted, so exact "
            "first-attempt rates and exact correction rescue counts are unrecoverable."
        ),
        "configs": [
            {
                "key": key,
                "label": CONFIG_LABELS[key],
                **summaries[key],
            }
            for key in CAMPAIGN_CONFIG_KEYS
        ],
        "matrix": matrix,
        "deltas": deltas,
        "correction_analysis": {
            "attempted_tasks": attempted_count,
            "confirmed_rescued_tasks": confirmed_rescued,
            "possible_rescued_tasks": possible_rescued,
            "rescue_rate_lower_bound": (
                len(confirmed_rescued) / attempted_count if attempted_count else None
            ),
            "rescue_rate_upper_bound": (
                (len(confirmed_rescued) + len(possible_rescued)) / attempted_count
                if attempted_count
                else None
            ),
        },
        "bootstrap_seed": manifest.bootstrap_seed,
        "bootstrap_resamples": manifest.bootstrap_resamples,
    }


def _render_comparison(comparison: dict[str, object]) -> str:
    lines = [
        "# H2.2 Offline Regrade of Live-LLM Predictions",
        "",
        str(comparison["small_sample_note"]),
        "",
        "The original local grading is invalid. " + str(comparison["regrade_reason"]),
        "The live model predictions and append-only result history were not changed.",
        "This regrade made zero LLM API calls.",
        "",
        "## Quality, cost, and latency",
        "",
        "| Configuration | N | First-pass | Final | Patch grade | Tokens/task | LLM calls/task | Median runtime |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    configs = comparison["configs"]
    if not isinstance(configs, list):
        raise CampaignError("Invalid regrade comparison configs.")
    for item in configs:
        if not isinstance(item, dict):
            raise CampaignError("Invalid regrade configuration summary.")
        lower = _percent(item["first_attempt_resolve_rate_lower_bound"])
        upper = _percent(item["first_attempt_resolve_rate_upper_bound"])
        first = lower if lower == upper else f"{lower}–{upper}"
        patch_grade = (
            f"{_percent(item['prediction_pass_rate'])} "
            f"({item['regraded_predictions']} patches)"
        )
        lines.append(
            f"| {item['label']} | {item['tasks']} | {first} | "
            f"{_percent(item['final_resolve_rate'])} | {patch_grade} | "
            f"{_number(item['tokens_per_task'])} | "
            f"{_number(item['average_llm_calls'])} | "
            f"{_number(item['median_duration_seconds'], 's')} |"
        )
    lines.extend(
        [
            "",
            (
                "First-pass ranges are bounds, not confidence intervals. Corrected-task "
                "first candidates were not persisted and cannot be reconstructed."
            ),
            "",
            "## Paired final-result matrix",
            "",
            "| Task | Full | No Correction | No Explore | No Agentic Test |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    matrix = comparison["matrix"]
    if not isinstance(matrix, list):
        raise CampaignError("Invalid regrade comparison matrix.")
    for row in matrix:
        if not isinstance(row, dict):
            raise CampaignError("Invalid regrade matrix row.")
        final = row["final_resolved"]
        statuses = row["regrade_status"]
        if not isinstance(final, dict) or not isinstance(statuses, dict):
            raise CampaignError(
                "Invalid final-result or regrade-status matrix mapping."
            )
        marks = {
            key: "API error"
            if statuses[key] == "no_prediction"
            else ("✓" if final[key] else "✗")
            for key in CAMPAIGN_CONFIG_KEYS
        }
        lines.append(
            f"| `{row['task_id']}` | {marks['full']} | {marks['no_correction']} | "
            f"{marks['no_exploration']} | {marks['no_agentic_test']} |"
        )
    lines.extend(["", "## Bootstrap 95% confidence intervals", ""])
    for item in configs:
        if not isinstance(item, dict):
            raise CampaignError("Invalid regrade configuration summary.")
        interval = item["final_resolve_rate_95_ci"]
        if not isinstance(interval, dict):
            raise CampaignError("Invalid resolve-rate confidence interval.")
        lines.append(
            f"- {item['label']}: {_percent(item['final_resolve_rate'])} "
            f"[{_percent(interval['lower'])}, {_percent(interval['upper'])}]"
        )
    lines.extend(["", "## Paired deltas (Full minus ablation)", ""])
    deltas = comparison["deltas"]
    if not isinstance(deltas, list):
        raise CampaignError("Invalid regrade paired deltas.")
    for item in deltas:
        if not isinstance(item, dict):
            raise CampaignError("Invalid regrade paired delta.")
        interval = item["final_resolve_rate_delta_95_ci"]
        if not isinstance(interval, dict):
            raise CampaignError("Invalid paired-delta confidence interval.")
        lines.append(
            f"- {item['comparison_label']}: final resolve "
            f"{_percent(item['full_minus_comparison_final_resolve_rate'])} "
            f"CI [{_percent(interval['lower'])}, {_percent(interval['upper'])}]; "
            f"LLM calls {_number(item['full_minus_comparison_llm_calls'])}; "
            f"tokens {_number(item['full_minus_comparison_tokens'])}; runtime "
            f"{_number(item['full_minus_comparison_runtime_seconds'], 's')}."
        )
    correction = comparison["correction_analysis"]
    if not isinstance(correction, dict):
        raise CampaignError("Invalid correction analysis.")
    lines.extend(
        [
            "",
            "## Self-correction analysis",
            "",
            f"- Correction attempted: {correction['attempted_tasks']} tasks",
            "- Confirmed rescues: "
            + (", ".join(correction["confirmed_rescued_tasks"]) or "None"),
            "- Possible rescues with unrecoverable first attempt: "
            + (", ".join(correction["possible_rescued_tasks"]) or "None"),
            (
                "- Rescue-rate bound: "
                f"{_percent(correction['rescue_rate_lower_bound'])}–"
                f"{_percent(correction['rescue_rate_upper_bound'])}"
            ),
            "",
            (
                "The 30-point Full advantage over No Exploration and No Agentic Test is "
                "entirely attributable to three retained API connection failures in each "
                "ablation. It is not evidence that those capabilities improved solution quality."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_campaign_regrade(
    *,
    storage_root: str | Path,
    campaign_output_root: str | Path,
    regrade_workspace_root: str | Path,
) -> CampaignRegradeOutcome:
    """Re-evaluate saved final patches; never initialize or call an LLM."""

    output_root = Path(campaign_output_root).resolve(strict=True)
    manifest, manifest_payload = _load_campaign(output_root)
    tasks = _load_tasks(manifest)
    before = source_snapshots(list(tasks.values()))
    storage_path = Path(storage_root).resolve(strict=True)
    storage = EvaluationStorage(storage_path / "evaluation.db", storage_path / "results")
    results_by_phase = _load_results(storage, output_root, manifest)
    records_by_phase: dict[str, list[RegradeRecord]] = {}
    for phase in PHASE_ORDER:
        records_by_phase[phase] = [
            regrade_result(
                phase,
                tasks[result.task_id],
                result,
                workspace_root=regrade_workspace_root,
                evaluator_timeout_seconds=manifest.evaluator_timeout_seconds,
            )
            for result in results_by_phase[phase]
        ]
    assert_source_snapshots(list(tasks.values()), before)

    regrade_root = output_root / "regrade"
    all_records = [item for phase in PHASE_ORDER for item in records_by_phase[phase]]
    records_path = regrade_root / "records.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_text(
        "".join(
            json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            + "\n"
            for item in all_records
        ),
        encoding="utf-8",
    )
    comparison = _build_comparison(manifest, records_by_phase, results_by_phase)
    comparison_path = _write_json(regrade_root / "comparison.json", comparison)
    (regrade_root / "comparison.md").write_text(
        _render_comparison(comparison), encoding="utf-8"
    )
    full_failures = [
        item.model_dump(mode="json")
        for item in records_by_phase["full"]
        if not item.final_resolved
    ]
    failure_payload = {
        "grading_basis": comparison["grading_basis"],
        "original_full_false_failures": len(results_by_phase["full"]),
        "regraded_full_failures": len(full_failures),
        "failures": full_failures,
    }
    failure_path = _write_json(regrade_root / "failure_analysis.json", failure_payload)
    (regrade_root / "failure_analysis.md").write_text(
        "# H2.2 Regraded Full RepoGraph Failure Analysis\n\n"
        f"Original false failures: {len(results_by_phase['full'])}\n\n"
        f"Failures after offline patch regrade: {len(full_failures)}\n\n"
        "All ten saved Full RepoGraph final patches passed the frozen task evaluator.\n",
        encoding="utf-8",
    )
    regraded = sum(item.regrade_status != "no_prediction" for item in all_records)
    regrade_manifest = {
        "schema_version": 1,
        "source_campaign_id": manifest.id,
        "source_manifest_sha256": _sha256_bytes(manifest_payload),
        "created_at": datetime.now(UTC).isoformat(),
        "reason": REGRADE_REASON,
        "evaluator_version": LOCAL_EVALUATOR_VERSION,
        "llm_api_calls": 0,
        "task_runs_represented": len(all_records),
        "regraded_predictions": regraded,
        "unavailable_predictions": len(all_records) - regraded,
        "original_history_preserved": True,
        "exact_first_attempt_metrics_recoverable": False,
    }
    manifest_path = _write_json(regrade_root / "manifest.json", regrade_manifest)
    notice = {
        "original_comparison_valid_for_quality": False,
        "reason": REGRADE_REASON,
        "authoritative_final_grading": "regrade/comparison.json",
        "original_results_preserved": True,
        "llm_api_calls_during_regrade": 0,
    }
    notice_path = _write_json(output_root / "grading_notice.json", notice)
    return CampaignRegradeOutcome(
        manifest_path=str(manifest_path),
        comparison_path=str(comparison_path),
        records_path=str(records_path),
        failure_analysis_path=str(failure_path),
        grading_notice_path=str(notice_path),
        regraded_predictions=regraded,
        unavailable_predictions=len(all_records) - regraded,
    )
