"""Local evaluator interpreter and failure-classification regressions."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.evaluator import run_local_evaluator
from evaluation.models import EvaluationTask


def task(command: list[str]) -> EvaluationTask:
    return EvaluationTask(
        id="evaluator",
        dataset="local-test",
        repository="unused",
        base_commit="abc",
        task="Run the evaluator.",
        test_command=command,
    )


class LocalEvaluatorTests(unittest.TestCase):
    def test_portable_python_alias_uses_current_interpreter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "test_sample.py").write_text(
                "def test_sample():\n    assert True\n", encoding="utf-8"
            )
            outcome = run_local_evaluator(
                task(["python", "-m", "pytest", "-q"]),
                directory,
                timeout_seconds=30,
                max_output_chars=2_000,
            )
        self.assertEqual(outcome.status, "passed")

    @patch("evaluation.evaluator.subprocess.run")
    def test_explicit_executable_is_not_rewritten(self, run) -> None:
        run.return_value = subprocess.CompletedProcess(
            ["custom-python", "-c", "pass"], 0, stdout="", stderr=""
        )
        with tempfile.TemporaryDirectory() as directory:
            run_local_evaluator(
                task(["custom-python", "-c", "pass"]),
                directory,
                timeout_seconds=30,
                max_output_chars=2_000,
            )
        self.assertEqual(run.call_args.args[0][0], "custom-python")

    @patch("evaluation.evaluator.subprocess.run")
    def test_missing_pytest_is_infrastructure_not_test_failure(self, run) -> None:
        run.return_value = subprocess.CompletedProcess(
            [sys.executable, "-m", "pytest"],
            1,
            stdout="",
            stderr=f"{sys.executable}: No module named pytest",
        )
        with tempfile.TemporaryDirectory() as directory:
            outcome = run_local_evaluator(
                task(["python", "-m", "pytest"]),
                directory,
                timeout_seconds=30,
                max_output_chars=2_000,
            )
        self.assertEqual(outcome.status, "error")
        self.assertEqual(outcome.failure_category, "evaluation_test_failure")
        self.assertNotEqual(outcome.failure_category, "test_failure")
        self.assertEqual(run.call_args.args[0][0], sys.executable)


if __name__ == "__main__":
    unittest.main()
