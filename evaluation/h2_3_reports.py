"""H2.3 exact candidate regrade and controlled comparison reports."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from evaluation.candidate_artifacts import CandidateArtifactError
from evaluation.h2_3_analysis import paired_comparison
from evaluation.metrics import aggregate_metrics
from evaluation.models import EvaluationResult, EvaluationTask
from evaluation.snapshot_regrade import (
    CandidateVersionAnalysis,
    regrade_result_candidate_versions,
)
from evaluation.statistics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resolve_ci,
)

CONFIG_ORDER = (
    "full",
    "no_correction",
    "no_exploration",
    "no_agentic_test",
)


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _task_map(tasks: Sequence[EvaluationTask]) -> dict[str, EvaluationTask]:
    return {task.id: task for task in tasks}


def regrade_all_candidate_versions(
    tasks: Sequence[EvaluationTask],
    results_by_config: Mapping[str, Sequence[EvaluationResult]],
    *,
    artifacts_root: str | Path,
    workspace_root: str | Path,
) -> tuple[
    dict[str, list[CandidateVersionAnalysis]],
    list[dict[str, str]],
]:
    """Regrade every available v1/final pair with its persisted evaluator."""

    tasks_by_id = _task_map(tasks)
    analyses: dict[str, list[CandidateVersionAnalysis]] = {}
    unavailable: list[dict[str, str]] = []
    for config_name in CONFIG_ORDER:
        analyses[config_name] = []
        for result in sorted(
            results_by_config[config_name],
            key=lambda item: item.task_id,
        ):
            if not result.candidate_snapshot_paths:
                unavailable.append(
                    {
                        "config": config_name,
                        "task_id": result.task_id,
                        "reason": "no candidate snapshots",
                    }
                )
                continue
            task = tasks_by_id[result.task_id]
            try:
                analysis = regrade_result_candidate_versions(
                    task,
                    result,
                    artifacts_root=artifacts_root,
                    workspace_root=workspace_root,
                    evaluator_specification=result.evaluator_specification,
                )
            except CandidateArtifactError as error:
                raise CandidateArtifactError(
                    f"Candidate regrade failed for {config_name}/{result.task_id}: {error}"
                ) from error
            analyses[config_name].append(analysis)
    return analyses, unavailable


def _correction_matrix(
    analyses: Sequence[CandidateVersionAnalysis],
) -> list[dict[str, object]]:
    return [
        {
            "task_id": item.task_id,
            "entered_correction": item.first.candidate_attempt != item.final.candidate_attempt,
            "first_attempt_resolved": item.first_attempt_resolved,
            "final_resolved": item.final_resolved,
            "rescued": item.rescued_by_correction,
            "regressed": item.regressed_after_correction,
            "unchanged_failure": (
                not item.first_attempt_resolved and not item.final_resolved
            ),
        }
        for item in sorted(analyses, key=lambda value: value.task_id)
    ]


def _config_summary(
    results: Sequence[EvaluationResult],
    analyses: Sequence[CandidateVersionAnalysis],
) -> dict[str, object]:
    metrics = aggregate_metrics(list(results))
    exact_first = [item.first_attempt_resolved for item in analyses]
    exact_final = [item.final_resolved for item in analyses]
    entering = [
        item for item in analyses
        if item.first.candidate_attempt != item.final.candidate_attempt
    ]
    rescued = [item.task_id for item in entering if item.rescued_by_correction]
    regressed = [item.task_id for item in entering if item.regressed_after_correction]
    return {
        "metrics": metrics.model_dump(mode="json"),
        "exact_regraded_predictions": len(analyses),
        "exact_first_attempt_resolved": sum(exact_first),
        "exact_first_attempt_resolve_rate": (
            sum(exact_first) / len(exact_first) if exact_first else None
        ),
        "exact_first_attempt_resolve_95_ci": bootstrap_resolve_ci(
            exact_first,
            seed=DEFAULT_BOOTSTRAP_SEED,
            resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
        ).model_dump(mode="json"),
        "exact_final_resolved": sum(exact_final),
        "exact_final_resolve_rate": (
            sum(exact_final) / len(exact_final) if exact_final else None
        ),
        "tasks_entering_correction": len(entering),
        "tasks_rescued_by_correction": len(rescued),
        "tasks_regressed_after_correction": len(regressed),
        "rescued_task_ids": rescued,
        "regressed_task_ids": regressed,
        "regraded_with_different_evaluator": any(
            item.first.regraded_with_different_evaluator
            or item.final.regraded_with_different_evaluator
            for item in analyses
        ),
    }


def write_h2_3_reports(
    output_root: str | Path,
    tasks: Sequence[EvaluationTask],
    results_by_config: Mapping[str, Sequence[EvaluationResult]],
    *,
    artifacts_root: str | Path,
    regrade_workspace_root: str | Path,
) -> list[Path]:
    """Write exact, paired, bounded H2.3 result artifacts."""

    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    analyses, unavailable = regrade_all_candidate_versions(
        tasks,
        results_by_config,
        artifacts_root=artifacts_root,
        workspace_root=regrade_workspace_root,
    )
    summaries = {
        name: _config_summary(results_by_config[name], analyses[name])
        for name in CONFIG_ORDER
    }
    paired = {
        name: paired_comparison(
            results_by_config["full"],
            results_by_config[name],
            seed=DEFAULT_BOOTSTRAP_SEED,
            resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
        ).model_dump(mode="json")
        for name in CONFIG_ORDER[1:]
    }
    correction_matrix = _correction_matrix(analyses["full"])
    comparison = {
        "analysis_seed": DEFAULT_BOOTSTRAP_SEED,
        "bootstrap_resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
        "configs": summaries,
        "paired": paired,
        "full_correction_matrix": correction_matrix,
        "unavailable_regrades": unavailable,
    }
    comparison_path = _write_json(output / "comparison.json", comparison)

    regrade_path = output / "candidate-regrade.jsonl"
    with regrade_path.open("w", encoding="utf-8", newline="\n") as handle:
        for config_name in CONFIG_ORDER:
            for analysis in analyses[config_name]:
                handle.write(
                    json.dumps(
                        {
                            "config": config_name,
                            "analysis": analysis.model_dump(mode="json"),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )

    markdown = [
        "# Evaluation — H2.3 Controlled Benchmark",
        "",
        "| Configuration | Assigned | Predictions | Resolved | Capability | End-to-end | Infrastructure | Exact first-pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CONFIG_ORDER:
        summary = summaries[name]
        metrics = summary["metrics"]
        markdown.append(
            "| "
            + name
            + f" | {metrics['assigned_tasks']}"
            + f" | {metrics['completed_with_prediction']}"
            + f" | {metrics['final_resolved']}"
            + f" | {_percent(metrics['capability_resolve_rate'])}"
            + f" | {_percent(metrics['end_to_end_resolve_rate'])}"
            + f" | {_percent(metrics['infrastructure_failure_rate'])}"
            + f" | {_percent(summary['exact_first_attempt_resolve_rate'])} |"
        )
    markdown.extend(["", "## Paired capability deltas", ""])
    for name in CONFIG_ORDER[1:]:
        value = paired[name]
        markdown.append(
            f"- Full − {name}: capability {_percent(value['capability_delta'])}; "
            f"end-to-end {_percent(value['end_to_end_delta'])}; "
            f"comparable {value['comparable_pairs']}/{value['assigned_pairs']}."
        )
    markdown.extend(
        [
            "",
            "Correction figures are exact snapshot regrades, not inferred ranges.",
            "Observed paired differences are descriptive; LLM nondeterminism prevents a causal claim.",
            "",
        ]
    )
    comparison_md_path = output / "comparison.md"
    comparison_md_path.write_text("\n".join(markdown), encoding="utf-8")

    unresolved = [
        item
        for item in results_by_config["full"]
        if item.valid_prediction and not item.final_resolved
    ]
    failure_lines = [
        "# H2.3 Full RepoGraph valid-prediction failures",
        "",
    ]
    if not unresolved:
        failure_lines.append("No valid-prediction unresolved tasks.")
    for result in sorted(unresolved, key=lambda item: item.task_id):
        failure_lines.extend(
            [
                f"## {result.task_id}",
                "",
                f"- Changed files: {', '.join(result.changed_files) or 'none'}",
                f"- Verification succeeded: {result.verification_succeeded}",
                f"- Review good: {result.review_good}",
                f"- Correction rounds: {result.correction_rounds_used}",
                f"- Evaluator outcome: {result.evaluator_failure_kind or 'assertion failure'}",
                f"- Summary: {(result.failure_reason or 'Hidden evaluator tests failed.')[:500]}",
                "",
            ]
        )
    failure_path = output / "failure_analysis.md"
    failure_path.write_text("\n".join(failure_lines), encoding="utf-8")
    return [comparison_path, comparison_md_path, regrade_path, failure_path]
