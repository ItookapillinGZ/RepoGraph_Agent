"""Conservative failure taxonomy for agent and evaluator boundaries."""

from __future__ import annotations

from typing import Literal

FailureCategory = Literal[
    "planning_failure",
    "repository_exploration_failure",
    "candidate_generation_failure",
    "candidate_validation_failure",
    "static_analysis_failure",
    "test_failure",
    "verification_failure",
    "review_failure",
    "correction_failure",
    "timeout",
    "tool_error",
    "evaluation_test_failure",
    "external_evaluator_failure",
    "valid_prediction_unresolved",
    "api_infrastructure_failure",
    "infrastructure_error",
    "unknown",
]

FAILURE_CATEGORIES: tuple[FailureCategory, ...] = (
    "planning_failure",
    "repository_exploration_failure",
    "candidate_generation_failure",
    "candidate_validation_failure",
    "static_analysis_failure",
    "test_failure",
    "verification_failure",
    "review_failure",
    "correction_failure",
    "timeout",
    "tool_error",
    "evaluation_test_failure",
    "external_evaluator_failure",
    "valid_prediction_unresolved",
    "api_infrastructure_failure",
    "infrastructure_error",
    "unknown",
)


def normalize_failure_category(value: str | None) -> FailureCategory:
    """Return a known category without over-interpreting weak evidence."""

    if value in FAILURE_CATEGORIES:
        return value  # type: ignore[return-value]
    return "unknown"
