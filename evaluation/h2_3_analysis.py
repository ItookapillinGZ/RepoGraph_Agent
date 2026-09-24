"""H2.3 capability denominators and paired comparison semantics."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from evaluation.models import EvaluationResult
from evaluation.statistics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    BootstrapInterval,
    bootstrap_paired_delta_ci,
)


class PairedTaskObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    comparable: bool
    baseline_valid_prediction: bool
    comparison_valid_prediction: bool
    baseline_resolved: bool
    comparison_resolved: bool
    capability_delta: int | None = Field(default=None, ge=-1, le=1)
    end_to_end_delta: int = Field(ge=-1, le=1)
    reason: str | None = None


class PairedComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assigned_pairs: int
    comparable_pairs: int
    not_comparable_pairs: int
    capability_delta: float | None
    end_to_end_delta: float
    capability_delta_95_ci: BootstrapInterval
    task_matrix: list[PairedTaskObservation]


def has_valid_capability_observation(result: EvaluationResult) -> bool:
    """Support explicit H2.3 flags and readable historical H2 results."""

    return result.valid_prediction or result.status in {"resolved", "unresolved"}


def paired_comparison(
    baseline: Sequence[EvaluationResult],
    comparison: Sequence[EvaluationResult],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> PairedComparison:
    baseline_by_id = {item.task_id: item for item in baseline}
    comparison_by_id = {item.task_id: item for item in comparison}
    task_ids = sorted(set(baseline_by_id).intersection(comparison_by_id))
    matrix: list[PairedTaskObservation] = []
    baseline_capability: list[bool] = []
    comparison_capability: list[bool] = []
    end_to_end_deltas: list[int] = []
    for task_id in task_ids:
        left = baseline_by_id[task_id]
        right = comparison_by_id[task_id]
        left_valid = has_valid_capability_observation(left)
        right_valid = has_valid_capability_observation(right)
        comparable = left_valid and right_valid
        capability_delta = (
            int(left.final_resolved) - int(right.final_resolved)
            if comparable
            else None
        )
        if comparable:
            baseline_capability.append(left.final_resolved)
            comparison_capability.append(right.final_resolved)
        end_to_end_delta = int(left.final_resolved) - int(right.final_resolved)
        end_to_end_deltas.append(end_to_end_delta)
        missing: list[str] = []
        if not left_valid:
            missing.append("baseline")
        if not right_valid:
            missing.append("comparison")
        matrix.append(
            PairedTaskObservation(
                task_id=task_id,
                comparable=comparable,
                baseline_valid_prediction=left_valid,
                comparison_valid_prediction=right_valid,
                baseline_resolved=left.final_resolved,
                comparison_resolved=right.final_resolved,
                capability_delta=capability_delta,
                end_to_end_delta=end_to_end_delta,
                reason=(
                    "not comparable: missing valid prediction from "
                    + " and ".join(missing)
                    if missing
                    else None
                ),
            )
        )
    comparable_count = len(baseline_capability)
    capability_delta = (
        sum(
            int(left) - int(right)
            for left, right in zip(
                baseline_capability,
                comparison_capability,
                strict=True,
            )
        )
        / comparable_count
        if comparable_count
        else None
    )
    return PairedComparison(
        assigned_pairs=len(task_ids),
        comparable_pairs=comparable_count,
        not_comparable_pairs=len(task_ids) - comparable_count,
        capability_delta=capability_delta,
        end_to_end_delta=(
            sum(end_to_end_deltas) / len(end_to_end_deltas)
            if end_to_end_deltas
            else 0.0
        ),
        capability_delta_95_ci=bootstrap_paired_delta_ci(
            baseline_capability,
            comparison_capability,
            seed=seed,
            resamples=resamples,
        ),
        task_matrix=matrix,
    )
