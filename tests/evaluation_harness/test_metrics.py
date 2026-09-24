"""Experiment metric aggregation tests."""

import unittest

from evaluation.metrics import aggregate_metrics
from evaluation.models import EvaluationResult


def result(
    task_id: str,
    *,
    first: bool = False,
    final: bool = False,
    rounds: int = 0,
    duration: float = 1,
    changed: int = 1,
    tool_calls: int | None = None,
    llm_calls: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
    telemetry_incomplete: bool = False,
) -> EvaluationResult:
    return EvaluationResult(
        task_id=task_id,
        dataset="local",
        experiment="e",
        repository="r",
        base_commit="c",
        created_at="now",
        status="resolved" if final else "unresolved",
        first_attempt_resolved=first,
        final_resolved=final,
        correction_rounds_used=rounds,
        planning_succeeded=True,
        candidate_generated=True,
        verification_succeeded=True,
        review_good=True,
        changed_files=[f"file-{index}.py" for index in range(changed)],
        duration_seconds=duration,
        tool_calls=tool_calls,
        llm_calls=llm_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        telemetry_incomplete=telemetry_incomplete,
    )


class MetricsTests(unittest.TestCase):
    def test_zero_tasks_is_defined_without_fake_optional_metrics(self) -> None:
        metrics = aggregate_metrics([])
        self.assertEqual(metrics.tasks_total, 0)
        self.assertEqual(metrics.resolve_rate, 0)
        self.assertIsNone(metrics.average_tool_calls)

    def test_self_correction_uplift_absolute_and_relative(self) -> None:
        metrics = aggregate_metrics(
            [
                result("one", first=True, final=True),
                result("two", first=False, final=True, rounds=1),
            ]
        )
        self.assertEqual(metrics.first_attempt_resolve_rate, 0.5)
        self.assertEqual(metrics.final_resolve_rate, 1.0)
        self.assertEqual(metrics.correction_uplift_absolute, 0.5)
        self.assertEqual(metrics.correction_uplift_relative, 1.0)

    def test_relative_uplift_is_none_when_first_rate_is_zero(self) -> None:
        metrics = aggregate_metrics([result("one", final=True, rounds=1)])
        self.assertIsNone(metrics.correction_uplift_relative)

    def test_average_and_median_metrics(self) -> None:
        metrics = aggregate_metrics(
            [
                result("one", rounds=0, duration=1, changed=1),
                result("two", rounds=2, duration=9, changed=3),
                result("three", rounds=1, duration=5, changed=2),
            ]
        )
        self.assertEqual(metrics.average_correction_rounds, 1)
        self.assertEqual(metrics.median_duration_seconds, 5)
        self.assertEqual(metrics.median_changed_files, 2)

    def test_optional_telemetry_averages_only_observed_values(self) -> None:
        metrics = aggregate_metrics(
            [result("one", tool_calls=2), result("two", tool_calls=None)]
        )
        self.assertEqual(metrics.average_tool_calls, 2)
        self.assertIsNone(metrics.average_input_tokens)

    def test_complete_token_telemetry_aggregates(self) -> None:
        metrics = aggregate_metrics(
            [
                result(
                    "one",
                    llm_calls=2,
                    input_tokens=10,
                    output_tokens=4,
                    total_tokens=14,
                ),
                result(
                    "two",
                    llm_calls=3,
                    input_tokens=20,
                    output_tokens=6,
                    total_tokens=26,
                ),
            ]
        )
        self.assertEqual(metrics.total_llm_calls, 5)
        self.assertEqual(metrics.total_tokens, 40)
        self.assertEqual(metrics.average_input_tokens, 15)

    def test_incomplete_token_telemetry_suppresses_token_averages(self) -> None:
        metrics = aggregate_metrics(
            [
                result(
                    "one",
                    llm_calls=2,
                    input_tokens=10,
                    output_tokens=4,
                    total_tokens=14,
                ),
                result("two", llm_calls=1, telemetry_incomplete=True),
            ]
        )
        self.assertEqual(metrics.total_llm_calls, 3)
        self.assertIsNone(metrics.total_tokens)
        self.assertIsNone(metrics.average_input_tokens)
        self.assertEqual(metrics.telemetry_incomplete_tasks, 1)

    def test_failure_breakdown_and_stage_rates(self) -> None:
        failed = result("one").model_copy(
            update={
                "status": "evaluation_error",
                "failure_category": "infrastructure_error",
                "planning_succeeded": False,
            }
        )
        metrics = aggregate_metrics([failed])
        self.assertEqual(metrics.planning_failure_rate, 1)
        self.assertEqual(metrics.evaluation_failure_rate, 1)
        self.assertEqual(metrics.failure_breakdown, {"infrastructure_error": 1})


if __name__ == "__main__":
    unittest.main()
