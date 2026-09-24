"""Deterministic experiment-level metric aggregation."""

from __future__ import annotations

import statistics

from pydantic import BaseModel, ConfigDict

from evaluation.models import EvaluationResult


class ExperimentMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks_total: int
    assigned_tasks: int = 0
    completed_with_prediction: int = 0
    infrastructure_failures: int = 0
    capability_resolve_rate: float = 0.0
    end_to_end_resolve_rate: float = 0.0
    infrastructure_failure_rate: float = 0.0
    tasks_resolved: int
    resolve_rate: float
    first_attempt_resolved: int
    first_attempt_resolve_rate: float
    final_resolved: int
    final_resolve_rate: float
    correction_uplift_absolute: float
    correction_uplift_relative: float | None
    tasks_using_correction: int
    average_correction_rounds: float
    median_correction_rounds: float
    planning_failure_rate: float
    candidate_failure_rate: float
    verification_failure_rate: float
    review_failure_rate: float
    evaluation_failure_rate: float
    average_duration_seconds: float
    median_duration_seconds: float
    average_changed_files: float
    median_changed_files: float
    average_llm_calls: float | None = None
    median_llm_calls: float | None = None
    average_tool_calls: float | None = None
    average_test_calls: float | None = None
    average_input_tokens: float | None = None
    average_output_tokens: float | None = None
    average_total_tokens: float | None = None
    median_total_tokens: float | None = None
    total_llm_calls: int | None = None
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_tokens: int | None = None
    tokens_per_task: float | None = None
    tokens_per_resolved_task: float | None = None
    telemetry_incomplete_tasks: int = 0
    average_cost: float | None = None
    process_isolated_tasks: int = 0
    hard_timeout_count: int = 0
    correction_attempted_tasks: int = 0
    correction_rescued_tasks: int = 0
    correction_rescue_rate: float | None = None
    correction_regressed_tasks: int = 0
    correction_regression_rate: float | None = None
    tasks_entering_correction: int = 0
    tasks_rescued_by_correction: int = 0
    tasks_regressed_after_correction: int = 0
    evaluator_breakdown: dict[str, int]
    failure_breakdown: dict[str, int]


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _mean(values: list[int | float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _median(values: list[int | float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _optional_mean(values: list[int | float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return _mean(available) if available else None


def _optional_median(values: list[int | float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return _median(available) if available else None


def _complete_sum(values: list[int | None]) -> int | None:
    return (
        sum(value for value in values if value is not None)
        if values and all(value is not None for value in values)
        else None
    )


def _complete_mean(values: list[int | None]) -> float | None:
    total = _complete_sum(values)
    return total / len(values) if total is not None else None


def aggregate_metrics(results: list[EvaluationResult]) -> ExperimentMetrics:
    """Aggregate only observed fields; missing telemetry remains None."""

    total = len(results)
    completed_predictions = [
        item
        for item in results
        if item.valid_prediction or item.status in {"resolved", "unresolved"}
    ]
    infrastructure_failures = sum(
        item.infrastructure_failure
        or item.status in {"evaluation_error", "timeout"}
        or item.failure_category
        in {
            "api_infrastructure_failure",
            "infrastructure_error",
            "external_evaluator_failure",
            "timeout",
        }
        for item in results
    )
    first = sum(item.first_attempt_resolved for item in results)
    final = sum(item.final_resolved for item in results)
    first_rate = _rate(first, total)
    final_rate = _rate(final, total)
    breakdown: dict[str, int] = {}
    evaluator_breakdown: dict[str, int] = {}
    for result in results:
        evaluator_breakdown[result.evaluator_kind] = (
            evaluator_breakdown.get(result.evaluator_kind, 0) + 1
        )
        if result.failure_category:
            breakdown[result.failure_category] = (
                breakdown.get(result.failure_category, 0) + 1
            )
    correction_attempted = sum(item.correction_rounds_used > 0 for item in results)
    correction_rescued = sum(
        not item.first_attempt_resolved and item.final_resolved for item in results
    )
    correction_regressed = sum(
        item.first_attempt_resolved and not item.final_resolved for item in results
    )
    complete_total_tokens = (
        None
        if any(item.telemetry_incomplete for item in results)
        else _complete_sum([item.total_tokens for item in results])
    )
    return ExperimentMetrics(
        tasks_total=total,
        assigned_tasks=total,
        completed_with_prediction=len(completed_predictions),
        infrastructure_failures=infrastructure_failures,
        capability_resolve_rate=_rate(
            sum(item.final_resolved for item in completed_predictions),
            len(completed_predictions),
        ),
        end_to_end_resolve_rate=_rate(final, total),
        infrastructure_failure_rate=_rate(infrastructure_failures, total),
        tasks_resolved=sum(item.status == "resolved" for item in results),
        resolve_rate=_rate(sum(item.status == "resolved" for item in results), total),
        first_attempt_resolved=first,
        first_attempt_resolve_rate=first_rate,
        final_resolved=final,
        final_resolve_rate=final_rate,
        correction_uplift_absolute=final_rate - first_rate,
        correction_uplift_relative=(
            final_rate / first_rate - 1 if first_rate > 0 else None
        ),
        tasks_using_correction=sum(item.correction_rounds_used > 0 for item in results),
        average_correction_rounds=_mean(
            [item.correction_rounds_used for item in results]
        ),
        median_correction_rounds=_median(
            [item.correction_rounds_used for item in results]
        ),
        planning_failure_rate=_rate(
            sum(not item.planning_succeeded for item in results), total
        ),
        candidate_failure_rate=_rate(
            sum(
                item.planning_succeeded and not item.candidate_generated
                for item in results
            ),
            total,
        ),
        verification_failure_rate=_rate(
            sum(
                item.candidate_generated and not item.verification_succeeded
                for item in results
            ),
            total,
        ),
        review_failure_rate=_rate(
            sum(
                item.verification_succeeded and not item.review_good for item in results
            ),
            total,
        ),
        evaluation_failure_rate=_rate(
            sum(item.status in {"evaluation_error", "timeout"} for item in results),
            total,
        ),
        average_duration_seconds=_mean([item.duration_seconds for item in results]),
        median_duration_seconds=_median([item.duration_seconds for item in results]),
        average_changed_files=_mean([len(item.changed_files) for item in results]),
        median_changed_files=_median([len(item.changed_files) for item in results]),
        average_llm_calls=_optional_mean([item.llm_calls for item in results]),
        median_llm_calls=_optional_median([item.llm_calls for item in results]),
        average_tool_calls=_optional_mean([item.tool_calls for item in results]),
        average_test_calls=_optional_mean([item.test_calls for item in results]),
        average_input_tokens=(
            None
            if any(item.telemetry_incomplete for item in results)
            else _complete_mean([item.input_tokens for item in results])
        ),
        average_output_tokens=(
            None
            if any(item.telemetry_incomplete for item in results)
            else _complete_mean([item.output_tokens for item in results])
        ),
        average_total_tokens=(
            None
            if any(item.telemetry_incomplete for item in results)
            else _complete_mean([item.total_tokens for item in results])
        ),
        median_total_tokens=(
            None
            if any(item.telemetry_incomplete for item in results)
            else _optional_median([item.total_tokens for item in results])
        ),
        total_llm_calls=_complete_sum([item.llm_calls for item in results]),
        total_input_tokens=(
            None
            if any(item.telemetry_incomplete for item in results)
            else _complete_sum([item.input_tokens for item in results])
        ),
        total_output_tokens=(
            None
            if any(item.telemetry_incomplete for item in results)
            else _complete_sum([item.output_tokens for item in results])
        ),
        total_tokens=complete_total_tokens,
        tokens_per_task=complete_total_tokens / total
        if complete_total_tokens is not None and total
        else None,
        tokens_per_resolved_task=complete_total_tokens / final
        if complete_total_tokens is not None and final
        else None,
        telemetry_incomplete_tasks=sum(item.telemetry_incomplete for item in results),
        average_cost=_optional_mean([item.estimated_cost_usd for item in results]),
        process_isolated_tasks=sum(item.process_isolated for item in results),
        hard_timeout_count=sum(item.hard_timeout for item in results),
        correction_attempted_tasks=correction_attempted,
        correction_rescued_tasks=correction_rescued,
        correction_rescue_rate=(
            correction_rescued / correction_attempted if correction_attempted else None
        ),
        correction_regressed_tasks=correction_regressed,
        correction_regression_rate=(
            correction_regressed / correction_attempted
            if correction_attempted
            else None
        ),
        tasks_entering_correction=correction_attempted,
        tasks_rescued_by_correction=correction_rescued,
        tasks_regressed_after_correction=correction_regressed,
        evaluator_breakdown=dict(sorted(evaluator_breakdown.items())),
        failure_breakdown=dict(sorted(breakdown.items())),
    )
