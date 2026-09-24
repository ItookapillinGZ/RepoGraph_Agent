"""Offline contract tests for the append-only H2.3C continuation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evaluation.campaign_h2_3c import (
    H2_3C_BASE_TASK_RUNS,
    H2_3C_RETRY_RESERVE,
    MAX_H2_3C_LIVE_TASK_RUNS,
    H23Continuation,
    _status_code,
    agent_semantic_fingerprint,
    effective_results,
    stop_rule_reason,
)
from evaluation.models import EvaluationResult


def result(
    task_id: str,
    *,
    resolved: bool = False,
    infrastructure: bool = False,
    candidate: bool = True,
) -> EvaluationResult:
    status = "resolved" if resolved else "unresolved"
    if infrastructure:
        status = "evaluation_error"
    elif not candidate:
        status = "agent_failed"
    return EvaluationResult(
        task_id=task_id,
        dataset="repograph-local-h2-3-v1",
        experiment="experiment",
        repository="repository",
        base_commit="commit",
        created_at="now",
        status=status,
        final_resolved=resolved,
        valid_prediction=candidate and not infrastructure,
        infrastructure_failure=infrastructure,
        duration_seconds=1,
    )


class H23CContractTests(unittest.TestCase):
    def test_budget_is_independent_and_has_twelve_retry_slots(self) -> None:
        self.assertEqual(MAX_H2_3C_LIVE_TASK_RUNS, 60)
        self.assertEqual(H2_3C_BASE_TASK_RUNS, 48)
        self.assertEqual(H2_3C_RETRY_RESERVE, 12)

    def test_effective_results_replace_recovery_without_deleting_parent(self) -> None:
        parent = [
            result("kept", resolved=True),
            result("recovered", infrastructure=True),
        ]
        continuation = [result("recovered", resolved=False), result("new")]
        combined = effective_results(parent, continuation)
        by_task = {item.task_id: item for item in combined}
        self.assertEqual(len(parent), 2)
        self.assertEqual(len(combined), 3)
        self.assertTrue(by_task["recovered"].valid_prediction)
        self.assertFalse(by_task["recovered"].infrastructure_failure)

    def test_provider_stop_rule_uses_final_distinct_assignments(self) -> None:
        values = {
            "one": result("one", infrastructure=True),
            "two": result("two", infrastructure=True),
            "three": result("three"),
            "four": result("four"),
        }
        self.assertIsNone(stop_rule_reason("config", values, list(values.values())))
        values["five"] = result("five")
        reason = stop_rule_reason("config", values, list(values.values()))
        self.assertIn("2/5", reason or "")

    def test_three_consecutive_final_infrastructure_failures_stop(self) -> None:
        recent = [
            result("one", infrastructure=True),
            result("two", infrastructure=True),
            result("three", infrastructure=True),
        ]
        reason = stop_rule_reason(
            "config", {item.task_id: item for item in recent}, recent
        )
        self.assertIn("three consecutive", reason or "")

    def test_matrix_statuses_keep_infrastructure_distinct(self) -> None:
        self.assertEqual(_status_code(result("r", resolved=True)), "R")
        self.assertEqual(_status_code(result("u")), "U")
        self.assertEqual(_status_code(result("i", infrastructure=True)), "I")
        self.assertEqual(_status_code(result("n", candidate=False)), "N")
        self.assertEqual(_status_code(None), "N")

    def test_partial_usage_never_reports_exact_average(self) -> None:
        complete = result("complete").model_copy(
            update={
                "llm_calls": 1,
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            }
        )
        partial = result("partial").model_copy(
            update={"llm_calls": 1, "telemetry_incomplete": True}
        )
        summary = H23Continuation._telemetry([complete, partial])
        self.assertEqual(summary["tasks_with_complete_usage"], 1)
        self.assertEqual(summary["tasks_with_incomplete_usage"], 1)
        self.assertEqual(summary["known_recorded_total_tokens"], 15)
        self.assertIsNone(summary["average_total_tokens"])
        self.assertIsNone(summary["cost"])

    def test_semantic_fingerprint_excludes_readme_and_nested_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            semantic = root / "engineering_plan.py"
            semantic.write_text("VALUE = 1\n", encoding="utf-8")
            readme = root / "README.md"
            readme.write_text("one\n", encoding="utf-8")
            report = root / "evaluation" / "campaigns" / "report.json"
            report.parent.mkdir(parents=True)
            report.write_text("{}\n", encoding="utf-8")
            initial = agent_semantic_fingerprint(root)["digest"]
            readme.write_text("two\n", encoding="utf-8")
            report.write_text('{"changed": true}\n', encoding="utf-8")
            self.assertEqual(agent_semantic_fingerprint(root)["digest"], initial)
            semantic.write_text("VALUE = 2\n", encoding="utf-8")
            self.assertNotEqual(agent_semantic_fingerprint(root)["digest"], initial)


if __name__ == "__main__":
    unittest.main()
