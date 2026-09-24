"""H2.3 infrastructure retry, denominator, and paired-comparison tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.h2_3_analysis import paired_comparison
from evaluation.live import LiveBudgetError, LiveRunBudget
from evaluation.metrics import aggregate_metrics
from evaluation.models import EvaluationConfig, EvaluationResult
from evaluation.runner import RepoGraphRun, create_experiment, run_experiment
from evaluation.storage import EvaluationStorage
from tests.evaluation_harness.helpers import (
    FIXED,
    WRONG,
    StaticAdapter,
    candidate,
    make_repository,
    task,
)


class APIConnectionError(Exception):
    pass


class SequenceAdapter:
    def __init__(self, outcomes: list[Exception | str]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def run(self, workspace, evaluation_task, config):
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        delegate = StaticAdapter(candidate(outcome))
        value: RepoGraphRun = delegate.run(workspace, evaluation_task, config)
        return value


def result(
    task_id: str,
    *,
    status: str,
    resolved: bool = False,
    valid_prediction: bool = False,
    infrastructure_failure: bool = False,
) -> EvaluationResult:
    return EvaluationResult(
        task_id=task_id,
        dataset="h2-3",
        experiment="experiment",
        repository="repository",
        base_commit="commit",
        created_at="now",
        status=status,
        final_resolved=resolved,
        valid_prediction=valid_prediction,
        infrastructure_failure=infrastructure_failure,
        duration_seconds=1,
    )


class InfrastructureRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository, self.commit = make_repository(self.root)
        self.storage = EvaluationStorage(
            self.root / "evaluation.db",
            self.root / "results",
        )
        self.config = EvaluationConfig(name="h2-3", agentic_test=False)
        self.experiment = create_experiment("h2-3", "local", self.config)
        self.task = task(self.repository, self.commit)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_experiment(self, adapter: SequenceAdapter, *, maximum_retries: int = 1):
        return run_experiment(
            [self.task],
            self.experiment,
            workspace_root=str(self.root / "workspaces"),
            artifacts_root=str(self.root / "artifacts"),
            storage=self.storage,
            adapter=adapter,
            max_infrastructure_retries=maximum_retries,
        )[0]

    def test_api_transport_failure_is_retried_once_and_history_is_preserved(
        self,
    ) -> None:
        adapter = SequenceAdapter([APIConnectionError("reset"), FIXED])
        final = self.run_experiment(adapter)
        self.assertEqual(adapter.calls, 2)
        self.assertEqual(final.status, "resolved")
        self.assertEqual(final.execution_attempt, 2)
        self.assertTrue(final.required_infrastructure_retry)
        records = [
            json.loads(line)
            for line in (self.root / "results" / f"{self.experiment.id}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(len(records), 2)
        self.assertTrue(records[0]["infrastructure_failure"])
        self.assertFalse(records[1]["infrastructure_failure"])

    def test_second_infrastructure_failure_persists_without_third_attempt(self) -> None:
        adapter = SequenceAdapter(
            [APIConnectionError("first"), APIConnectionError("second")]
        )
        final = self.run_experiment(adapter)
        self.assertEqual(adapter.calls, 2)
        self.assertEqual(final.status, "evaluation_error")
        self.assertTrue(final.infrastructure_failure)
        self.assertTrue(final.required_infrastructure_retry)

    def test_capability_failure_is_not_retried(self) -> None:
        adapter = SequenceAdapter([WRONG, FIXED])
        final = self.run_experiment(adapter)
        self.assertEqual(adapter.calls, 1)
        self.assertEqual(final.status, "unresolved")
        self.assertEqual(final.failure_category, "valid_prediction_unresolved")
        self.assertFalse(final.infrastructure_failure)

    def test_attempt_callback_preserves_initial_retry_evidence(self) -> None:
        observed: list[tuple[int, bool]] = []
        adapter = SequenceAdapter([APIConnectionError("reset"), FIXED])
        run_experiment(
            [self.task],
            self.experiment,
            workspace_root=str(self.root / "workspaces"),
            artifacts_root=str(self.root / "artifacts"),
            storage=self.storage,
            adapter=adapter,
            max_infrastructure_retries=1,
            after_execution_attempt=lambda item: observed.append(
                (item.execution_attempt, item.infrastructure_failure)
            ),
        )
        self.assertEqual(observed, [(1, True), (2, False)])

    def test_retry_cannot_exceed_live_budget(self) -> None:
        adapter = SequenceAdapter([APIConnectionError("reset"), FIXED])
        budget = LiveRunBudget(self.root / "ledger.jsonl", maximum=1)
        with self.assertRaises(LiveBudgetError):
            run_experiment(
                [self.task],
                self.experiment,
                workspace_root=str(self.root / "workspaces"),
                artifacts_root=str(self.root / "artifacts"),
                storage=self.storage,
                adapter=adapter,
                max_infrastructure_retries=1,
                before_task_run=lambda experiment, current: budget.reserve(
                    experiment_id=experiment.id,
                    task_id=current.id,
                    config_name=experiment.config.name,
                ),
            )
        self.assertEqual(adapter.calls, 1)
        self.assertEqual(budget.consumed, 1)
        persisted = self.storage.get_result(self.experiment.id, self.task.id)
        self.assertIsNotNone(persisted)
        self.assertTrue(persisted and persisted.infrastructure_failure)


class CapabilitySemanticsTests(unittest.TestCase):
    def test_three_denominators_are_distinct(self) -> None:
        metrics = aggregate_metrics(
            [
                result(
                    "resolved",
                    status="resolved",
                    resolved=True,
                    valid_prediction=True,
                ),
                result(
                    "unresolved",
                    status="unresolved",
                    valid_prediction=True,
                ),
                result(
                    "infra",
                    status="evaluation_error",
                    infrastructure_failure=True,
                ),
            ]
        )
        self.assertEqual(metrics.assigned_tasks, 3)
        self.assertEqual(metrics.completed_with_prediction, 2)
        self.assertEqual(metrics.infrastructure_failures, 1)
        self.assertEqual(metrics.capability_resolve_rate, 0.5)
        self.assertAlmostEqual(metrics.end_to_end_resolve_rate, 1 / 3)
        self.assertAlmostEqual(metrics.infrastructure_failure_rate, 1 / 3)

    def test_paired_capability_excludes_infrastructure_missing_result(self) -> None:
        baseline = [
            result(
                "one",
                status="resolved",
                resolved=True,
                valid_prediction=True,
            ),
            result(
                "two",
                status="resolved",
                resolved=True,
                valid_prediction=True,
            ),
        ]
        comparison = [
            result(
                "one",
                status="unresolved",
                valid_prediction=True,
            ),
            result(
                "two",
                status="evaluation_error",
                infrastructure_failure=True,
            ),
        ]
        paired = paired_comparison(baseline, comparison)
        self.assertEqual(paired.assigned_pairs, 2)
        self.assertEqual(paired.comparable_pairs, 1)
        self.assertEqual(paired.not_comparable_pairs, 1)
        self.assertEqual(paired.capability_delta, 1.0)
        self.assertEqual(paired.end_to_end_delta, 1.0)
        self.assertFalse(paired.task_matrix[1].comparable)


if __name__ == "__main__":
    unittest.main()
