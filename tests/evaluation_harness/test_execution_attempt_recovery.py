"""Interrupted worker execution identities are durable and never recycled."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from evaluation.campaign_integrity import audit_campaign
from evaluation.candidate_artifacts import (
    CandidateArtifactError,
    CandidateArtifactStore,
)
from evaluation.models import EvaluationConfig, ExecutionAttemptRecord
from evaluation.runner import create_experiment, run_evaluation_task
from evaluation.storage import EvaluationStorage, EvaluationStorageError
from tests.evaluation_harness.helpers import (
    FIXED,
    WRONG,
    StaticAdapter,
    candidate,
    make_repository,
    task,
)

DIGEST = "a" * 64


class ExecutionAttemptRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.storage = EvaluationStorage(
            self.root / "evaluation.db", self.root / "results"
        )
        self.config = EvaluationConfig(name="full", agentic_test=False)
        self.experiment = create_experiment("campaign-full", "local", self.config)
        self.storage.save_experiment(self.experiment)
        self.store = CandidateArtifactStore(self.root / "artifacts")

    def reserve(self):
        return self.storage.reserve_execution_attempt(
            campaign_id="campaign",
            config_name="full",
            experiment_id=self.experiment.id,
            task_id="task",
        )

    def persist(self, execution_attempt: int, ordinal: int, source: str = FIXED):
        return self.store.persist(
            task_id="task",
            experiment_id=self.experiment.id,
            execution_attempt=execution_attempt,
            attempt=ordinal,
            base_commit="commit",
            model_name=None,
            candidate=candidate(source),
            diff_text=f"diff-{execution_attempt}-{ordinal}-{source}",
            evaluator_digest=DIGEST,
            created_at="now",
        )

    def test_interrupted_candidate_v1_resumes_at_attempt_two_and_preserves_old(self) -> None:
        first = self.storage.mark_execution_attempt_running(self.reserve())
        _old_snapshot, old_path = self.persist(first.execution_attempt, 1, WRONG)
        self.storage.finish_execution_attempt(first, status="interrupted")

        second = self.storage.mark_execution_attempt_running(self.reserve())
        self.assertEqual(second.execution_attempt, 2)
        _new_snapshot, new_path = self.persist(second.execution_attempt, 1, FIXED)

        self.assertNotEqual(old_path, new_path)
        self.assertTrue((self.root / "artifacts" / old_path).is_file())
        self.assertTrue((self.root / "artifacts" / new_path).is_file())

    def test_reservation_without_worker_is_consumed(self) -> None:
        first = self.reserve()
        self.storage.finish_execution_attempt(first, status="interrupted")
        self.assertEqual(self.reserve().execution_attempt, 2)

    def test_reserved_crash_is_explicitly_reconciled_before_attempt_two(self) -> None:
        self.reserve()
        changed = self.storage.reconcile_incomplete_attempts(
            campaign_id="campaign", reason="crash before worker spawn"
        )
        self.assertEqual(changed, 1)
        self.assertEqual(self.reserve().execution_attempt, 2)

    def test_multi_candidate_interruption_resets_candidate_ordinal_only(self) -> None:
        first = self.storage.mark_execution_attempt_running(self.reserve())
        self.persist(first.execution_attempt, 1, WRONG)
        self.persist(first.execution_attempt, 2, FIXED)
        self.storage.finish_execution_attempt(first, status="interrupted")
        second = self.storage.mark_execution_attempt_running(self.reserve())
        _, resumed_path = self.persist(second.execution_attempt, 1, FIXED)
        self.assertIn("execution-02/candidate-01.json", resumed_path)
        self.assertEqual(len(list((self.root / "artifacts").rglob("*.json"))), 3)

    def test_infrastructure_failure_and_timeout_consume_attempts(self) -> None:
        first = self.storage.mark_execution_attempt_running(self.reserve())
        self.storage.finish_execution_attempt(first, status="failed")
        second = self.storage.mark_execution_attempt_running(self.reserve())
        self.storage.finish_execution_attempt(second, status="timeout")
        self.assertEqual(self.reserve().execution_attempt, 3)

    def test_reconcile_running_crash_before_result_persist_consumes_attempt(self) -> None:
        self.storage.mark_execution_attempt_running(self.reserve())
        changed = self.storage.reconcile_incomplete_attempts(
            campaign_id="campaign", reason="controller restart"
        )
        self.assertEqual(changed, 1)
        self.assertEqual(self.reserve().execution_attempt, 2)

    def test_two_controllers_cannot_reserve_same_assignment(self) -> None:
        self.reserve()
        second_controller = EvaluationStorage(
            self.root / "evaluation.db", self.root / "other-results"
        )
        with self.assertRaisesRegex(EvaluationStorageError, "already active"):
            second_controller.reserve_execution_attempt(
                campaign_id="campaign",
                config_name="full",
                experiment_id=self.experiment.id,
                task_id="task",
            )

    def test_snapshot_collision_is_rejected_even_when_content_is_identical(self) -> None:
        _, path = self.persist(1, 1, FIXED)
        before = hashlib.sha256((self.root / "artifacts" / path).read_bytes()).hexdigest()
        with self.assertRaisesRegex(CandidateArtifactError, "already exists"):
            self.persist(1, 1, FIXED)
        after = hashlib.sha256((self.root / "artifacts" / path).read_bytes()).hexdigest()
        self.assertEqual(before, after)

    def test_snapshot_collision_never_overwrites_different_content(self) -> None:
        _, path = self.persist(1, 1, WRONG)
        before = (self.root / "artifacts" / path).read_bytes()
        with self.assertRaises(CandidateArtifactError):
            self.persist(1, 1, FIXED)
        self.assertEqual((self.root / "artifacts" / path).read_bytes(), before)


class CampaignAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repository, self.commit = make_repository(self.root)
        self.storage = EvaluationStorage(
            self.root / "evaluation.db", self.root / "results"
        )
        self.task = task(self.repository, self.commit)
        self.storage.save_task(self.task)

    def create_valid(self, name: str):
        config = EvaluationConfig(name=name, agentic_test=False)
        experiment = create_experiment(name, "local", config)
        self.storage.save_experiment(experiment)
        attempt = self.storage.reserve_execution_attempt(
            campaign_id="audit",
            config_name=name,
            experiment_id=experiment.id,
            task_id=self.task.id,
        )
        attempt = self.storage.mark_execution_attempt_running(attempt)
        result = run_evaluation_task(
            self.task,
            config,
            workspace_root=str(self.root / "workspaces"),
            artifacts_root=str(self.root / "artifacts"),
            experiment_id=experiment.id,
            adapter=StaticAdapter(candidate(FIXED)),
            execution_attempt=attempt.execution_attempt,
        )
        self.storage.save_result(result)
        self.storage.finish_execution_attempt(attempt, status="completed")
        return experiment

    def test_audit_passes_valid_historical_full_and_no_correction(self) -> None:
        full = self.create_valid("full")
        no_correction = self.create_valid("no_correction")
        for experiment in (full, no_correction):
            report = audit_campaign(
                self.storage,
                artifacts_root=self.root / "artifacts",
                campaign_id="audit",
                experiment_ids=[experiment.id],
            )
            self.assertTrue(report.passed, report.issues)

    def test_audit_detects_duplicate_snapshot_identity(self) -> None:
        experiment = self.create_valid("full")
        original = next((self.root / "artifacts").rglob("candidate-01.json"))
        duplicate = original.with_name("candidate-99.json")
        duplicate.write_bytes(original.read_bytes())
        report = audit_campaign(
            self.storage,
            artifacts_root=self.root / "artifacts",
            campaign_id="audit",
            experiment_ids=[experiment.id],
        )
        self.assertIn("duplicate_snapshot_identity", {item.kind for item in report.issues})

    def test_audit_detects_non_monotonic_attempt_chronology(self) -> None:
        config = EvaluationConfig(name="full", agentic_test=False)
        experiment = create_experiment("full", "local", config)
        self.storage.save_experiment(experiment)
        for number, timestamp in ((1, "2026-01-02"), (2, "2026-01-01")):
            self.storage.import_execution_attempt(
                ExecutionAttemptRecord(
                    campaign_id="audit",
                    config_name="full",
                    experiment_id=experiment.id,
                    task_id=self.task.id,
                    execution_attempt=number,
                    status="interrupted",
                    reserved_at=timestamp,
                    finished_at=timestamp,
                )
            )
        report = audit_campaign(
            self.storage,
            artifacts_root=self.root / "artifacts",
            campaign_id="audit",
            experiment_ids=[experiment.id],
        )
        self.assertIn("non_monotonic_attempt", {item.kind for item in report.issues})


if __name__ == "__main__":
    unittest.main()
