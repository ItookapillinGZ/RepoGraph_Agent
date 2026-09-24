"""H2.3 evaluator specification, preflight, and classification tests."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.evaluator import (
    EvaluatorPreflightError,
    build_local_evaluator_specification,
    preflight_local_evaluator,
    run_local_evaluator,
)
from evaluation.models import EvaluationConfig, EvaluationTask
from evaluation.runner import run_evaluation_task
from tests.evaluation_harness.helpers import (
    FIXED,
    StaticAdapter,
    candidate,
    make_repository,
    task,
)


def evaluator_task(root: Path, command: list[str]) -> EvaluationTask:
    return EvaluationTask(
        id="evaluator-h2-3",
        dataset="h2-3-test",
        repository=str(root),
        base_commit="abc",
        task="Evaluate the candidate.",
        test_command=command,
    )


class EvaluatorIntegrityTests(unittest.TestCase):
    def test_python_interpreter_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = build_local_evaluator_specification(
                evaluator_task(
                    Path(directory),
                    ["python", "-m", "pytest", "-q"],
                ),
                timeout_seconds=30,
            )
        self.assertEqual(spec.command[0], sys.executable)
        self.assertEqual(spec.python_executable, sys.executable)

    @patch("evaluation.evaluator.subprocess.run")
    def test_missing_pytest_fails_campaign_preflight(self, run) -> None:
        run.return_value = subprocess.CompletedProcess(
            [sys.executable, "-m", "pytest", "--version"],
            1,
            stdout="",
            stderr=f"{sys.executable}: No module named pytest",
        )
        with tempfile.TemporaryDirectory() as directory:
            spec = build_local_evaluator_specification(
                evaluator_task(
                    Path(directory),
                    ["python", "-m", "pytest", "-q"],
                ),
                timeout_seconds=30,
            )
            with self.assertRaises(EvaluatorPreflightError):
                preflight_local_evaluator(spec)

    def test_evaluator_digest_is_stable_and_config_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task_value = evaluator_task(
                Path(directory),
                ["python", "-m", "pytest", "-q"],
            )
            first = build_local_evaluator_specification(
                task_value,
                timeout_seconds=30,
            )
            same = build_local_evaluator_specification(
                task_value,
                timeout_seconds=30.0,
            )
            changed = build_local_evaluator_specification(
                task_value,
                timeout_seconds=31,
            )
        self.assertEqual(first.digest, same.digest)
        self.assertNotEqual(first.digest, changed.digest)

    def test_pytest_assertion_failure_is_classified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_failure.py").write_text(
                "def test_failure():\n    assert False\n",
                encoding="utf-8",
            )
            outcome = run_local_evaluator(
                evaluator_task(
                    root,
                    ["python", "-m", "pytest", "test_failure.py", "-q"],
                ),
                root,
                timeout_seconds=120,
                max_output_chars=4_000,
            )
        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.failure_kind, "assertion_failure")

    def test_pytest_collection_failure_is_classified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_bad.py").write_text(
                "def test_bad(:\n    pass\n",
                encoding="utf-8",
            )
            outcome = run_local_evaluator(
                evaluator_task(
                    root,
                    ["python", "-m", "pytest", "test_bad.py", "-q"],
                ),
                root,
                timeout_seconds=120,
                max_output_chars=4_000,
            )
        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.failure_kind, "collection_failure")

    def test_pytest_import_failure_is_classified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_import.py").write_text(
                "import definitely_missing_h2_3_module\n",
                encoding="utf-8",
            )
            outcome = run_local_evaluator(
                evaluator_task(
                    root,
                    ["python", "-m", "pytest", "test_import.py", "-q"],
                ),
                root,
                timeout_seconds=120,
                max_output_chars=4_000,
            )
        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.failure_kind, "import_failure")

    def test_evaluator_timeout_is_classified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outcome = run_local_evaluator(
                evaluator_task(
                    root,
                    [sys.executable, "-c", "import time; time.sleep(1)"],
                ),
                root,
                timeout_seconds=0.05,
                max_output_chars=4_000,
            )
        self.assertEqual(outcome.status, "timeout")
        self.assertEqual(outcome.failure_kind, "timeout")

    def test_evaluator_provenance_is_persisted_on_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, commit = make_repository(root)
            config = EvaluationConfig(name="h2-3", agentic_test=False)
            result = run_evaluation_task(
                task(repository, commit),
                config,
                workspace_root=str(root / "workspaces"),
                artifacts_root=str(root / "artifacts"),
                experiment_id="h2-3-evaluator",
                adapter=StaticAdapter(candidate(FIXED)),
            )
        self.assertIsNotNone(result.evaluator_specification)
        self.assertIsNotNone(result.evaluator_environment)
        self.assertEqual(
            result.evaluator_digest,
            result.evaluator_specification and result.evaluator_specification.digest,
        )
        self.assertEqual(
            result.provenance and result.provenance.python_executable,
            sys.executable,
        )


if __name__ == "__main__":
    unittest.main()
