"""Hard process isolation, IPC validation, and tree-termination tests."""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from evaluation.models import EvaluationConfig, EvaluationWorkerRequest
from evaluation.process_runner import (
    run_evaluation_worker,
    run_fixed_argv,
    terminate_process_tree,
    worker_environment,
)
from tests.evaluation_harness.helpers import make_repository, repository_digest, task

TEST_WORKER_TIMEOUT_SECONDS = 60
REAL_GRAPH_TEST_TIMEOUT_SECONDS = 120


class ProcessRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source, self.commit = make_repository(self.root)
        self.workspace = self.root / "workers"
        self.config = EvaluationConfig(name="process", agentic_test=False)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def request(self, behavior: str, **metadata: object) -> EvaluationWorkerRequest:
        evaluation_task = task(self.source, self.commit).model_copy(
            update={"metadata": {"worker_fixture": behavior, **metadata}}
        )
        return EvaluationWorkerRequest(
            task=evaluation_task,
            config=self.config,
            workspace_root=str(self.workspace),
            experiment_id="process-experiment",
            result_path="controller-assigned",
        )

    def run_worker(
        self,
        behavior: str,
        timeout: float = TEST_WORKER_TIMEOUT_SECONDS,
        **metadata: object,
    ):
        return run_evaluation_worker(
            self.request(behavior, **metadata),
            timeout_seconds=timeout,
            _worker_module="tests.evaluation_harness.worker_fixture",
        )

    def test_worker_normal_completion(self) -> None:
        outcome = self.run_worker("success")
        self.assertEqual(outcome.status, "completed")
        self.assertTrue(outcome.result and outcome.result.process_isolated)

    def test_worker_timeout_returns_control(self) -> None:
        marker = self.root / "sleep-started.txt"
        started = time.monotonic()
        outcome = self.run_worker("sleep", timeout=2, started_path=str(marker))
        self.assertEqual(outcome.status, "timeout")
        self.assertLess(time.monotonic() - started, 8)
        self.assertTrue(marker.exists())

    def test_worker_child_process_is_terminated(self) -> None:
        survivor = self.root / "survivor.txt"
        outcome = self.run_worker(
            "child_sleep",
            timeout=2,
            survivor_path=str(survivor),
        )
        self.assertEqual(outcome.status, "timeout")
        self.assertTrue(Path(str(survivor) + ".started").exists())
        time.sleep(1.5)
        self.assertFalse(survivor.exists())

    def test_crash_missing_and_malformed_results_are_failures(self) -> None:
        for behavior in ("crash", "no_result", "malformed"):
            with self.subTest(behavior=behavior):
                self.assertEqual(self.run_worker(behavior).status, "failed")

    def test_oversized_result_is_rejected(self) -> None:
        outcome = self.run_worker("oversized")
        self.assertEqual(outcome.status, "failed")
        self.assertIn("MAX_WORKER_RESULT_BYTES", outcome.failure_reason or "")

    def test_controller_continues_after_timeout(self) -> None:
        self.assertEqual(self.run_worker("sleep", timeout=2).status, "timeout")
        self.assertEqual(self.run_worker("success").status, "completed")

    def test_process_isolated_real_graph_and_source_immutability(self) -> None:
        before = repository_digest(self.source)
        outcome = self.run_worker("real_graph", timeout=REAL_GRAPH_TEST_TIMEOUT_SECONDS)
        self.assertEqual(outcome.status, "completed")
        self.assertEqual(outcome.result and outcome.result.status, "resolved")
        self.assertEqual(repository_digest(self.source), before)

    def test_worker_failure_redacts_secret_from_stdout_and_stderr(self) -> None:
        secret = "sk-test-secret-value"
        child_environment = worker_environment({**os.environ, "OPENAI_API_KEY": secret})
        for behavior in ("secret_stdout", "secret_stderr"):
            with self.subTest(behavior=behavior):
                outcome = run_evaluation_worker(
                    self.request(behavior),
                    timeout_seconds=TEST_WORKER_TIMEOUT_SECONDS,
                    _worker_module="tests.evaluation_harness.worker_fixture",
                    _environment=child_environment,
                )
                self.assertEqual(outcome.status, "failed")
                self.assertNotIn(secret, outcome.failure_reason or "")
                self.assertIn("<redacted>", outcome.failure_reason or "")

    def test_worker_environment_is_allowlisted(self) -> None:
        environment = worker_environment(
            {
                "PATH": "runtime",
                "OPENAI_API_KEY": "direct",
                "CODE_REVIEW_LLM_API_KEY": "project",
                "CODE_REVIEW_LLM_MODEL": "gpt-5.6-sol",
                "PDE_FRONTIER_LLM_API_KEY": "legacy",
                "PDE_FRONTIER_LLM_BASE_URL": "https://relay.example/v1",
                "GITHUB_TOKEN": "drop",
            }
        )
        self.assertEqual(environment["PATH"], "runtime")
        self.assertIn("OPENAI_API_KEY", environment)
        self.assertIn("CODE_REVIEW_LLM_API_KEY", environment)
        self.assertIn("CODE_REVIEW_LLM_MODEL", environment)
        self.assertIn("PDE_FRONTIER_LLM_API_KEY", environment)
        self.assertIn("PDE_FRONTIER_LLM_BASE_URL", environment)
        self.assertNotIn("GITHUB_TOKEN", environment)

    @patch("evaluation.process_runner.subprocess.Popen")
    def test_fixed_process_argv_never_uses_shell(self, popen: MagicMock) -> None:
        process = popen.return_value
        process.wait.return_value = 0
        process.returncode = 0
        outcome = run_fixed_argv(
            ["trusted-program", "--fixed", "value"],
            cwd=self.root,
            timeout_seconds=1,
            environment={},
        )
        self.assertFalse(outcome.timed_out)
        self.assertEqual(
            popen.call_args.args[0], ["trusted-program", "--fixed", "value"]
        )
        self.assertFalse(popen.call_args.kwargs["shell"])

    @patch("evaluation.process_runner.subprocess.run")
    @patch("evaluation.process_runner.platform.system", return_value="Windows")
    def test_windows_tree_termination_uses_fixed_numeric_taskkill(
        self,
        _system: MagicMock,
        run: MagicMock,
    ) -> None:
        process = MagicMock(pid=123)
        process.poll.return_value = None
        process.wait.return_value = 0
        terminate_process_tree(process, grace_seconds=0.01)
        self.assertEqual(
            run.call_args_list[0].args[0],
            ["taskkill", "/PID", "123", "/T", "/F"],
        )
        self.assertFalse(run.call_args_list[0].kwargs["shell"])

    @patch("evaluation.process_runner.subprocess.run")
    @patch("evaluation.process_runner.platform.system", return_value="Windows")
    def test_windows_taskkill_timeout_falls_back_without_escaping(
        self,
        _system: MagicMock,
        run: MagicMock,
    ) -> None:
        run.side_effect = subprocess.TimeoutExpired(["taskkill"], 1)
        process = MagicMock(pid=123)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired(["worker"], 0.01), 0]

        terminate_process_tree(process, grace_seconds=0.01)

        process.kill.assert_called_once_with()

    @patch("evaluation.process_runner.os.killpg", create=True)
    @patch("evaluation.process_runner.os.getpgid", return_value=456, create=True)
    @patch("evaluation.process_runner.platform.system", return_value="Linux")
    def test_posix_tree_termination_targets_worker_group(
        self,
        _system: MagicMock,
        _getpgid: MagicMock,
        killpg: MagicMock,
    ) -> None:
        process = MagicMock(pid=123)
        process.poll.return_value = None
        process.wait.return_value = 0
        terminate_process_tree(process)
        killpg.assert_called_once_with(456, signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
