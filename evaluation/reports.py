"""Honest Markdown/JSON reports derived only from persisted results."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.metrics import ExperimentMetrics, aggregate_metrics
from evaluation.models import EvaluationExperiment, EvaluationResult


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_summary(
    experiment: EvaluationExperiment,
    results: list[EvaluationResult],
) -> str:
    metrics = aggregate_metrics(results)
    relative = (
        "n/a"
        if metrics.correction_uplift_relative is None
        else _percent(metrics.correction_uplift_relative)
    )
    return "\n".join(
        [
            f"Experiment: {experiment.name} ({experiment.id})",
            f"Tasks: {metrics.tasks_total}",
            "",
            "First attempt:",
            f"Resolved: {metrics.first_attempt_resolved} / {metrics.tasks_total}",
            f"Rate: {_percent(metrics.first_attempt_resolve_rate)}",
            "",
            "Final:",
            f"Resolved: {metrics.final_resolved} / {metrics.tasks_total}",
            f"Rate: {_percent(metrics.final_resolve_rate)}",
            "",
            "Self-correction uplift:",
            f"{metrics.correction_uplift_absolute * 100:+.1f} percentage points",
            f"{relative} relative",
            "",
            f"Average correction rounds: {metrics.average_correction_rounds:.2f}",
            f"Median runtime: {metrics.median_duration_seconds:.2f}s",
        ]
    )


def render_markdown_report(
    experiment: EvaluationExperiment,
    results: list[EvaluationResult],
    metrics: ExperimentMetrics | None = None,
) -> str:
    metrics = metrics or aggregate_metrics(results)
    resolved_models = sorted(
        {model for item in results for model in item.resolved_model_names}
    )
    resolved_label = ", ".join(resolved_models) or "unavailable"
    lines = [
        f"# RepoGraph Evaluation — {experiment.name}",
        "",
        "## Experiment metadata",
        "",
        f"- ID: `{experiment.id}`",
        f"- Dataset: `{experiment.dataset}`",
        f"- Created: `{experiment.created_at}`",
        f"- RepoGraph commit: `{experiment.git_commit or 'unavailable'}`",
        f"- LLM execution kind: `{experiment.llm_execution_kind}`",
        f"- Configured model: `{experiment.configured_model_name or 'unavailable'}`",
        f"- Provider-resolved model(s): `{resolved_label}`",
        f"- Model temperature: `{experiment.config.model_temperature}`",
        f"- Provider seed: `{experiment.config.model_seed or 'unavailable'}`",
        f"- Model provider: `{experiment.config.model_provider or 'unavailable'}`",
        f"- Model API base URL: `{experiment.config.model_base_url or 'SDK default'}`",
        f"- Reasoning effort: `{experiment.config.model_reasoning_effort or 'provider default'}`",
        f"- Max completion tokens: `{experiment.config.model_max_completion_tokens or 'provider default'}`",
        "",
        "```json",
        json.dumps(experiment.config.model_dump(mode="json"), indent=2, sort_keys=True),
        "```",
        "",
        "## Headline metrics",
        "",
        f"- Tasks: {metrics.tasks_total}",
        (
            f"- First-attempt resolved: {metrics.first_attempt_resolved} "
            f"({_percent(metrics.first_attempt_resolve_rate)})"
        ),
        (
            f"- Final resolved: {metrics.final_resolved} "
            f"({_percent(metrics.final_resolve_rate)})"
        ),
        (
            f"- Absolute correction uplift: "
            f"{metrics.correction_uplift_absolute * 100:+.1f} percentage points"
        ),
        "- Relative correction uplift: "
        + (
            "n/a (first-attempt resolve rate is zero)"
            if metrics.correction_uplift_relative is None
            else _percent(metrics.correction_uplift_relative)
        ),
        (
            f"- Average / median correction rounds: "
            f"{metrics.average_correction_rounds:.2f} / "
            f"{metrics.median_correction_rounds:.2f}"
        ),
        (
            f"- Average / median duration: "
            f"{metrics.average_duration_seconds:.2f}s / "
            f"{metrics.median_duration_seconds:.2f}s"
        ),
        "",
        "## Execution integrity",
        "",
        (
            f"- Process-isolated tasks: {metrics.process_isolated_tasks} / "
            f"{metrics.tasks_total}"
        ),
        f"- Hard task timeouts: {metrics.hard_timeout_count}",
        "- Task and evaluator timeout budgets are recorded separately.",
        "",
        "## LLM usage",
        "",
        (
            "- Calls: unavailable"
            if metrics.total_llm_calls is None
            else f"- Calls: {metrics.total_llm_calls}"
        ),
        (
            "- Tokens: unavailable or incomplete (no estimate was fabricated)"
            if metrics.total_tokens is None
            else (
                f"- Tokens: {metrics.total_tokens} total "
                f"({metrics.total_input_tokens} input, "
                f"{metrics.total_output_tokens} output)"
            )
        ),
        f"- Tasks with incomplete telemetry: {metrics.telemetry_incomplete_tasks}",
        (
            f"- Average / median LLM calls: {metrics.average_llm_calls} / "
            f"{metrics.median_llm_calls}"
        ),
        (
            f"- Tokens per task / resolved task: {metrics.tokens_per_task} / "
            f"{metrics.tokens_per_resolved_task}"
        ),
        "",
        "## External grading",
        "",
        *(
            f"- `{kind}`: {count} task(s)"
            for kind, count in metrics.evaluator_breakdown.items()
        ),
        (
            "- Resolution is determined by the recorded external evaluator, "
            "not RepoGraph's internal review verdict."
        ),
        "",
        "## Failure breakdown",
        "",
    ]
    if metrics.failure_breakdown:
        lines.extend(
            f"- `{category}`: {count}"
            for category, count in metrics.failure_breakdown.items()
        )
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Per-task results",
            "",
            "| Task | Evaluator | Status | First | Final | Rounds | Files | Duration | Failure |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for result in sorted(results, key=lambda item: item.task_id):
        lines.append(
            f"| `{result.task_id}` | {result.evaluator_kind} | {result.status} | "
            f"{'yes' if result.first_attempt_resolved else 'no'} | "
            f"{'yes' if result.final_resolved else 'no'} | "
            f"{result.correction_rounds_used} | {len(result.changed_files)} | "
            f"{result.duration_seconds:.2f}s | "
            f"{result.failure_category or ''} |"
        )
    warnings = sorted(
        {
            warning
            for result in results
            for warning in [*result.warnings, *result.telemetry_warnings]
        }
    )
    lines.extend(["", "## Correction analysis", ""])
    lines.append(
        f"Observed uplift is {_percent(metrics.correction_uplift_absolute)} "
        "absolute. This report makes no statistical-significance claim."
    )
    lines.append(f"- Correction attempted tasks: {metrics.correction_attempted_tasks}")
    lines.append(f"- Correction rescued tasks: {metrics.correction_rescued_tasks}")
    lines.append(f"- Correction rescue rate: {metrics.correction_rescue_rate}")
    lines.append(f"- Correction-regressed tasks: {metrics.correction_regressed_tasks}")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def write_reports(
    experiment: EvaluationExperiment,
    results: list[EvaluationResult],
    destination: str | Path,
) -> tuple[Path, Path]:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    metrics = aggregate_metrics(results)
    markdown_path = root / f"{experiment.id}.md"
    json_path = root / f"{experiment.id}.json"
    markdown_path.write_text(
        render_markdown_report(experiment, results, metrics), encoding="utf-8"
    )
    json_path.write_text(
        json.dumps(
            {
                "experiment": experiment.model_dump(mode="json"),
                "metrics": metrics.model_dump(mode="json"),
                "results": [item.model_dump(mode="json") for item in results],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return markdown_path, json_path
