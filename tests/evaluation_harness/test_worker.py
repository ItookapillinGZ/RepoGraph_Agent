"""Evaluation worker request/result contract tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from evaluation.models import (
    EvaluationConfig,
    EvaluationResult,
    EvaluationWorkerRequest,
    EvaluationWorkerResult,
)
from evaluation.worker import main
from tests.evaluation_harness.helpers import make_repository, task


class WorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source, self.commit = make_repository(self.root)
        self.result_path = self.root / "result.json"
        self.request = EvaluationWorkerRequest(
            task=task(self.source, self.commit),
            config=EvaluationConfig(name="worker", agentic_test=False),
            workspace_root=str(self.root / "workspaces"),
            experiment_id="worker-experiment",
            result_path=str(self.result_path),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_worker_writes_strict_completed_result(self) -> None:
        request_path = self.root / "request.json"
        request_path.write_text(self.request.model_dump_json(), encoding="utf-8")
        result = EvaluationResult(
            task_id=self.request.task.id,
            dataset=self.request.task.dataset,
            experiment=self.request.experiment_id,
            repository=self.request.task.repository,
            base_commit=self.request.task.base_commit,
            created_at="now",
            status="resolved",
            duration_seconds=1,
        )
        with patch(
            "evaluation.runner._run_evaluation_task_in_process",
            return_value=result,
        ):
            self.assertEqual(main(["--request", str(request_path)]), 0)
        envelope = EvaluationWorkerResult.model_validate_json(
            self.result_path.read_bytes()
        )
        self.assertEqual(envelope.status, "completed")
        self.assertTrue(envelope.result and envelope.result.process_isolated)

    def test_worker_rejects_malformed_request(self) -> None:
        request_path = self.root / "request.json"
        request_path.write_text("{}", encoding="utf-8")
        self.assertEqual(main(["--request", str(request_path)]), 2)
        self.assertFalse(self.result_path.exists())

    def test_worker_failure_envelope_redacts_api_key(self) -> None:
        request_path = self.root / "request.json"
        request_path.write_text(self.request.model_dump_json(), encoding="utf-8")
        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "super-secret-value"}),
            patch(
                "evaluation.runner._run_evaluation_task_in_process",
                side_effect=RuntimeError("failed with super-secret-value"),
            ),
        ):
            self.assertEqual(main(["--request", str(request_path)]), 0)
        envelope = EvaluationWorkerResult.model_validate_json(
            self.result_path.read_bytes()
        )
        self.assertEqual(envelope.status, "failed")
        self.assertNotIn("super-secret-value", envelope.failure_reason or "")
        self.assertIn("<redacted>", envelope.failure_reason or "")

    def test_worker_result_contract_rejects_ambiguous_status(self) -> None:
        with self.assertRaises(ValidationError):
            EvaluationWorkerResult(status="completed")
        with self.assertRaises(ValidationError):
            EvaluationWorkerResult(
                status="failed",
                result=EvaluationResult(
                    task_id="x",
                    dataset="d",
                    experiment="e",
                    repository="r",
                    base_commit="c",
                    created_at="now",
                    status="unresolved",
                    duration_seconds=0,
                ),
            )


if __name__ == "__main__":
    unittest.main()
