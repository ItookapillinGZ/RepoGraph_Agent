"""Paired statistics and bounded failure reports for H2.2 campaigns."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from evaluation.metrics import ExperimentMetrics, aggregate_metrics
from evaluation.models import EvaluationResult
from evaluation.statistics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    BootstrapInterval,
    bootstrap_paired_delta_ci,
    bootstrap_resolve_ci,
)

CONFIG_LABELS = {
    "full": "Full RepoGraph",
    "no_correction": "No Self-Correction",
    "no_exploration": "No Agentic Exploration",
    "no_agentic_test": "No Agentic Test",
}
CONFIG_ORDER = tuple(CONFIG_LABELS)
SMALL_SAMPLE_NOTE = (
    "This is a small exploratory benchmark. Confidence intervals are wide and "
    "should not be interpreted as production-level statistical evidence."
)


class ComparisonIntegrityError(ValueError):
    """Raised when results cannot support a truthful paired comparison."""


class ConfigStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    label: str
    metrics: ExperimentMetrics
    resolve_rate_95_ci: BootstrapInterval


class PairedMatrixRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    base_commit: str
    resolved: dict[str, bool]


class PairedDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comparison_key: str
    comparison_label: str
    full_minus_comparison_resolve_rate: float
    resolve_rate_delta_95_ci: BootstrapInterval
    full_minus_comparison_llm_calls: float | None
    full_minus_comparison_tokens: float | None
    full_minus_comparison_runtime_seconds: float


class PairedCampaignComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comparison_valid: bool = True
    task_ids: list[str]
    configs: list[ConfigStatistics]
    matrix: list[PairedMatrixRow]
    deltas: list[PairedDelta]
    rescued_by_correction: list[str]
    regressed_after_correction: list[str]
    bootstrap_seed: int
    bootstrap_resamples: int
    note: str = SMALL_SAMPLE_NOTE


class FailureTaskSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    status: str
    plan_succeeded: bool
    candidate_generated: bool
    verification_succeeded: bool
    review_good: bool
    correction_rounds: int
    external_evaluator_resolved: bool
    failure_category: str
    failure_reason: str | None
    llm_calls: int | None
    input_tokens: int | None
    output_tokens: int | None
    telemetry_incomplete: bool


class FailureAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_tasks: int
    unresolved_tasks: int
    failure_category_counts: dict[str, int]
    failure_layer_counts: dict[str, int]
    unresolved: list[FailureTaskSummary]
    representative_examples: list[FailureTaskSummary]


def _result_maps(
    results_by_config: dict[str, list[EvaluationResult]],
) -> dict[str, dict[str, EvaluationResult]]:
    if set(results_by_config) != set(CONFIG_ORDER):
        raise ComparisonIntegrityError(
            "comparison requires full, no_correction, no_exploration, and no_agentic_test"
        )
    maps: dict[str, dict[str, EvaluationResult]] = {}
    for key in CONFIG_ORDER:
        results = results_by_config[key]
        mapping = {item.task_id: item for item in results}
        if len(mapping) != len(results):
            raise ComparisonIntegrityError(f"duplicate task IDs in {key}")
        maps[key] = mapping
    return maps


def validate_paired_integrity(
    results_by_config: dict[str, list[EvaluationResult]],
) -> dict[str, dict[str, EvaluationResult]]:
    """Reject differences in task, base, model, evaluator, or live provenance."""

    maps = _result_maps(results_by_config)
    task_ids = set(maps["full"])
    if not task_ids:
        raise ComparisonIntegrityError("paired comparison requires results")
    for key, mapping in maps.items():
        if set(mapping) != task_ids:
            raise ComparisonIntegrityError(f"task set mismatch for {key}")
    for task_id in sorted(task_ids):
        paired = [maps[key][task_id] for key in CONFIG_ORDER]
        if len({item.base_commit for item in paired}) != 1:
            raise ComparisonIntegrityError(f"base commit mismatch for {task_id}")
        if len({item.configured_model_name for item in paired}) != 1:
            raise ComparisonIntegrityError(f"configured model mismatch for {task_id}")
        if len({item.evaluator_kind for item in paired}) != 1:
            raise ComparisonIntegrityError(f"evaluator mismatch for {task_id}")
        if any(item.llm_execution_kind != "live" for item in paired):
            raise ComparisonIntegrityError(f"non-live result for {task_id}")
        provenance = [item.provenance for item in paired]
        if any(item is None for item in provenance):
            raise ComparisonIntegrityError(f"missing provenance for {task_id}")
        provider_metadata = {
            (
                item.model_provider,
                item.model_base_url,
                item.model_reasoning_effort,
                item.model_max_completion_tokens,
            )
            for item in provenance
            if item is not None
        }
        if len(provider_metadata) != 1:
            raise ComparisonIntegrityError(
                f"provider configuration mismatch for {task_id}"
            )
        if len({item.task_timeout_seconds for item in provenance if item}) != 1:
            raise ComparisonIntegrityError(f"task timeout mismatch for {task_id}")
    return maps


def _paired_mean_delta(
    baseline: dict[str, EvaluationResult],
    comparison: dict[str, EvaluationResult],
    attribute: str,
) -> float | None:
    values: list[float] = []
    for task_id in sorted(baseline):
        first = getattr(baseline[task_id], attribute)
        second = getattr(comparison[task_id], attribute)
        if first is None or second is None:
            return None
        values.append(float(first) - float(second))
    return float(statistics.fmean(values)) if values else None


def _paired_token_delta(
    baseline: dict[str, EvaluationResult],
    comparison: dict[str, EvaluationResult],
) -> float | None:
    if any(
        item.telemetry_incomplete
        for mapping in (baseline, comparison)
        for item in mapping.values()
    ):
        return None
    return _paired_mean_delta(baseline, comparison, "total_tokens")


def build_paired_comparison(
    results_by_config: dict[str, list[EvaluationResult]],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> PairedCampaignComparison:
    maps = validate_paired_integrity(results_by_config)
    task_ids = sorted(maps["full"])
    configs = [
        ConfigStatistics(
            key=key,
            label=CONFIG_LABELS[key],
            metrics=aggregate_metrics(results_by_config[key]),
            resolve_rate_95_ci=bootstrap_resolve_ci(
                [maps[key][task_id].final_resolved for task_id in task_ids],
                seed=seed,
                resamples=resamples,
            ),
        )
        for key in CONFIG_ORDER
    ]
    baseline = maps["full"]
    deltas: list[PairedDelta] = []
    for key in CONFIG_ORDER[1:]:
        comparison = maps[key]
        baseline_resolved = [baseline[item].final_resolved for item in task_ids]
        comparison_resolved = [comparison[item].final_resolved for item in task_ids]
        deltas.append(
            PairedDelta(
                comparison_key=key,
                comparison_label=CONFIG_LABELS[key],
                full_minus_comparison_resolve_rate=(
                    statistics.fmean(baseline_resolved)
                    - statistics.fmean(comparison_resolved)
                ),
                resolve_rate_delta_95_ci=bootstrap_paired_delta_ci(
                    baseline_resolved,
                    comparison_resolved,
                    seed=seed,
                    resamples=resamples,
                ),
                full_minus_comparison_llm_calls=_paired_mean_delta(
                    baseline, comparison, "llm_calls"
                ),
                full_minus_comparison_tokens=_paired_token_delta(baseline, comparison),
                full_minus_comparison_runtime_seconds=(
                    _paired_mean_delta(baseline, comparison, "duration_seconds") or 0.0
                ),
            )
        )
    return PairedCampaignComparison(
        task_ids=task_ids,
        configs=configs,
        matrix=[
            PairedMatrixRow(
                task_id=task_id,
                base_commit=baseline[task_id].base_commit,
                resolved={
                    key: maps[key][task_id].final_resolved for key in CONFIG_ORDER
                },
            )
            for task_id in task_ids
        ],
        deltas=deltas,
        rescued_by_correction=[
            task_id
            for task_id in task_ids
            if not baseline[task_id].first_attempt_resolved
            and baseline[task_id].final_resolved
        ],
        regressed_after_correction=[
            task_id
            for task_id in task_ids
            if baseline[task_id].first_attempt_resolved
            and not baseline[task_id].final_resolved
        ],
        bootstrap_seed=seed,
        bootstrap_resamples=resamples,
    )


def build_failure_analysis(results: Sequence[EvaluationResult]) -> FailureAnalysis:
    unresolved_results = sorted(
        (item for item in results if not item.final_resolved),
        key=lambda item: item.task_id,
    )
    categories = Counter(
        item.failure_category or "unknown" for item in unresolved_results
    )
    layers: Counter[str] = Counter()
    summaries: list[FailureTaskSummary] = []
    for result in unresolved_results:
        if result.status in {"evaluation_error", "timeout"}:
            layers["evaluation infrastructure failure"] += 1
        elif result.status == "agent_failed":
            layers["agent pipeline failure"] += 1
        else:
            layers["benchmark unresolved after valid candidate"] += 1
        summaries.append(
            FailureTaskSummary(
                task_id=result.task_id,
                status=result.status,
                plan_succeeded=result.planning_succeeded,
                candidate_generated=result.candidate_generated,
                verification_succeeded=result.verification_succeeded,
                review_good=result.review_good,
                correction_rounds=result.correction_rounds_used,
                external_evaluator_resolved=result.final_resolved,
                failure_category=result.failure_category or "unknown",
                failure_reason=(result.failure_reason or "")[:500] or None,
                llm_calls=result.llm_calls,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                telemetry_incomplete=result.telemetry_incomplete,
            )
        )
    representative: list[FailureTaskSummary] = []
    for category, _count in sorted(
        categories.items(), key=lambda item: (-item[1], item[0])
    )[:3]:
        representative.append(
            next(item for item in summaries if item.failure_category == category)
        )
    return FailureAnalysis(
        total_tasks=len(results),
        unresolved_tasks=len(unresolved_results),
        failure_category_counts=dict(sorted(categories.items())),
        failure_layer_counts=dict(sorted(layers.items())),
        unresolved=summaries,
        representative_examples=representative,
    )


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def _number(value: float | None, suffix: str = "") -> str:
    return (
        "N/A — incomplete provider usage" if value is None else f"{value:.2f}{suffix}"
    )


def _mark(resolved: bool) -> str:
    return "✓" if resolved else "✗"


def render_comparison_markdown(comparison: PairedCampaignComparison) -> str:
    lines = [
        "# H2.2 Local Live Benchmark Comparison",
        "",
        SMALL_SAMPLE_NOTE,
        "",
        "## Quality, cost, and latency",
        "",
        "| Configuration | N | First-pass | Final | Tokens/task | LLM calls/task | Median runtime |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for config in comparison.configs:
        metrics = config.metrics
        lines.append(
            f"| {config.label} | {metrics.tasks_total} | "
            f"{_percent(metrics.first_attempt_resolve_rate)} | "
            f"{_percent(metrics.final_resolve_rate)} | "
            f"{_number(metrics.tokens_per_task)} | "
            f"{_number(metrics.average_llm_calls)} | "
            f"{_number(metrics.median_duration_seconds, 's')} |"
        )
    lines.extend(
        [
            "",
            "## Paired task matrix",
            "",
            "| Task | Full | No Correction | No Explore | No Agentic Test |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in comparison.matrix:
        lines.append(
            f"| `{row.task_id}` | {_mark(row.resolved['full'])} | {_mark(row.resolved['no_correction'])} | "
            f"{_mark(row.resolved['no_exploration'])} | {_mark(row.resolved['no_agentic_test'])} |"
        )
    lines.extend(["", "## Bootstrap 95% confidence intervals", ""])
    for config in comparison.configs:
        interval = config.resolve_rate_95_ci
        lines.append(
            f"- {config.label}: {_percent(interval.point_estimate)} "
            f"[{_percent(interval.lower)}, {_percent(interval.upper)}]"
        )
    lines.extend(["", "## Paired deltas (Full minus ablation)", ""])
    for delta in comparison.deltas:
        interval = delta.resolve_rate_delta_95_ci
        lines.append(
            f"- {delta.comparison_label}: resolve {_percent(delta.full_minus_comparison_resolve_rate)} "
            f"CI [{_percent(interval.lower)}, {_percent(interval.upper)}]; "
            f"LLM calls {_number(delta.full_minus_comparison_llm_calls)}; "
            f"tokens {_number(delta.full_minus_comparison_tokens)}; "
            f"runtime {_number(delta.full_minus_comparison_runtime_seconds, 's')}."
        )
    lines.extend(
        [
            "",
            "## Self-correction analysis",
            "",
            "- Rescued: "
            + (
                ", ".join(f"`{item}`" for item in comparison.rescued_by_correction)
                or "None"
            ),
            "- Regressed after correction: "
            + (
                ", ".join(f"`{item}`" for item in comparison.regressed_after_correction)
                or "None"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_failure_markdown(analysis: FailureAnalysis) -> str:
    lines = [
        "# H2.2 Full RepoGraph Failure Analysis",
        "",
        f"Unresolved: {analysis.unresolved_tasks} / {analysis.total_tasks}",
        "",
        "## Failure layers",
        "",
    ]
    lines.extend(
        f"- {name}: {count}" for name, count in analysis.failure_layer_counts.items()
    )
    if not analysis.failure_layer_counts:
        lines.append("- None")
    lines.extend(["", "## Failure categories", ""])
    lines.extend(
        f"- `{name}`: {count}"
        for name, count in analysis.failure_category_counts.items()
    )
    if not analysis.failure_category_counts:
        lines.append("- None")
    lines.extend(["", "## Unresolved task details", ""])
    for item in analysis.unresolved:
        lines.extend(
            [
                f"### {item.task_id}",
                "",
                f"- Status/category: `{item.status}` / `{item.failure_category}`",
                (
                    f"- Plan/candidate/verification/review: {item.plan_succeeded} / "
                    f"{item.candidate_generated} / {item.verification_succeeded} / {item.review_good}"
                ),
                f"- Correction rounds: {item.correction_rounds}",
                f"- External evaluator resolved: {item.external_evaluator_resolved}",
                f"- LLM calls: {item.llm_calls if item.llm_calls is not None else 'N/A'}",
                (
                    f"- Tokens: {item.input_tokens if item.input_tokens is not None else 'N/A'} input / "
                    f"{item.output_tokens if item.output_tokens is not None else 'N/A'} output"
                ),
                f"- Bounded reason: {item.failure_reason or 'unavailable'}",
                "",
            ]
        )
    lines.extend(["## Representative examples", ""])
    lines.append(
        ", ".join(f"`{item.task_id}`" for item in analysis.representative_examples)
        or "None"
    )
    lines.append("")
    return "\n".join(lines)


def write_campaign_reports(
    output_root: str | Path,
    results_by_config: dict[str, list[EvaluationResult]],
) -> tuple[Path, Path, Path, Path]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    comparison = build_paired_comparison(results_by_config)
    failures = build_failure_analysis(results_by_config["full"])
    comparison_json = root / "comparison.json"
    comparison_markdown = root / "comparison.md"
    failure_json = root / "failure_analysis.json"
    failure_markdown = root / "failure_analysis.md"
    comparison_json.write_text(
        json.dumps(comparison.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    comparison_markdown.write_text(
        render_comparison_markdown(comparison), encoding="utf-8"
    )
    failure_json.write_text(
        json.dumps(failures.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    failure_markdown.write_text(render_failure_markdown(failures), encoding="utf-8")
    return comparison_markdown, comparison_json, failure_markdown, failure_json
