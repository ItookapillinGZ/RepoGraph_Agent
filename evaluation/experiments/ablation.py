"""Observed, non-significance-claiming comparisons between supported configs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from evaluation.metrics import aggregate_metrics
from evaluation.models import EvaluationResult


class AblationRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: str
    tasks: int
    first_pass_rate: float
    final_rate: float
    average_rounds: float
    average_duration_seconds: float
    average_tool_calls: float | None
    delta_resolve_rate_vs_full: float
    delta_duration_seconds_vs_full: float
    delta_tool_calls_vs_full: float | None


class AblationComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline: str
    rows: list[AblationRow]
    note: str = "Observed differences only; no statistical-significance claim."


def compare_ablations(
    results_by_config: dict[str, list[EvaluationResult]],
    *,
    baseline: str = "Full RepoGraph",
) -> AblationComparison:
    if baseline not in results_by_config:
        raise ValueError(f"Ablation baseline is missing: {baseline}")
    summaries = {
        name: aggregate_metrics(results) for name, results in results_by_config.items()
    }
    full = summaries[baseline]
    rows: list[AblationRow] = []
    for name, metrics in summaries.items():
        tool_delta = (
            metrics.average_tool_calls - full.average_tool_calls
            if metrics.average_tool_calls is not None
            and full.average_tool_calls is not None
            else None
        )
        rows.append(
            AblationRow(
                config=name,
                tasks=metrics.tasks_total,
                first_pass_rate=metrics.first_attempt_resolve_rate,
                final_rate=metrics.final_resolve_rate,
                average_rounds=metrics.average_correction_rounds,
                average_duration_seconds=metrics.average_duration_seconds,
                average_tool_calls=metrics.average_tool_calls,
                delta_resolve_rate_vs_full=(
                    metrics.final_resolve_rate - full.final_resolve_rate
                ),
                delta_duration_seconds_vs_full=(
                    metrics.average_duration_seconds - full.average_duration_seconds
                ),
                delta_tool_calls_vs_full=tool_delta,
            )
        )
    return AblationComparison(baseline=baseline, rows=rows)
