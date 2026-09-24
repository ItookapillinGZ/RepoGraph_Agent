"""Offline campaign patch-regrade regressions."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from evaluation.campaign_regrade import regrade_result
from evaluation.models import EvaluationResult
from evaluation.workspace import (
    apply_candidate_exact,
    extract_candidate_patch,
    prepare_task_workspace,
)
from tests.evaluation_harness.helpers import (
    candidate,
    make_repository,
    plan,
    repository_digest,
    task,
)


class CampaignRegradeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source, self.commit = make_repository(self.root)
        self.task = task(self.source, self.commit)
        workspace, _ = prepare_task_workspace(
            str(self.source),
            self.commit,
            self.root / "prediction-workspace",
            "source-experiment",
            self.task.id,
        )
        generated = candidate()
        apply_candidate_exact(workspace, plan(), generated)
        self.patch, self.changed_files = extract_candidate_patch(
            workspace, self.commit, generated
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def result(self, *, rounds: int = 0, patch: str | None = "saved") -> EvaluationResult:
        model_patch = self.patch if patch == "saved" else patch
        prediction_hash = (
            hashlib.sha256(model_patch.encode("utf-8")).hexdigest()
            if model_patch is not None
            else None
        )
        return EvaluationResult(
            task_id=self.task.id,
            dataset=self.task.dataset,
            experiment="live-experiment",
            repository=str(self.source),
            base_commit=self.commit,
            created_at=datetime.now(UTC).isoformat(),
            model_name="test-live-model",
            configured_model_name="test-live-model",
            llm_execution_kind="live",
            status="unresolved",
            correction_rounds_used=rounds,
            duration_seconds=1,
            prediction_sha256=prediction_hash,
            model_patch=model_patch,
            changed_files=self.changed_files,
        )

    def test_saved_patch_is_regraded_without_mutating_source(self) -> None:
        before = repository_digest(self.source)
        record = regrade_result(
            "full",
            self.task,
            self.result(),
            workspace_root=self.root / "regrade-workspaces",
            evaluator_timeout_seconds=30,
        )
        self.assertEqual(record.regrade_status, "passed")
        self.assertTrue(record.final_resolved)
        self.assertTrue(record.first_attempt_resolved)
        self.assertEqual(repository_digest(self.source), before)

    def test_corrected_task_first_attempt_is_explicitly_unknown(self) -> None:
        record = regrade_result(
            "full",
            self.task,
            self.result(rounds=1),
            workspace_root=self.root / "regrade-workspaces",
            evaluator_timeout_seconds=30,
        )
        self.assertTrue(record.final_resolved)
        self.assertIsNone(record.first_attempt_resolved)

    def test_missing_prediction_is_retained_as_unavailable(self) -> None:
        record = regrade_result(
            "no_exploration",
            self.task,
            self.result(patch=None),
            workspace_root=self.root / "regrade-workspaces",
            evaluator_timeout_seconds=30,
        )
        self.assertEqual(record.regrade_status, "no_prediction")
        self.assertFalse(record.final_resolved)
        self.assertFalse(record.first_attempt_resolved)


if __name__ == "__main__":
    unittest.main()
