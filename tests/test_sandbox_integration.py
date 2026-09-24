"""Shared boundary and disposable-workspace integration tests."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from engineering_plan import EngineeringPlan, PlannedFileChange, PlannedTest
from evaluation.evaluator import run_local_evaluator
from evaluation.models import EvaluationTask
from plan_execution import (
    CandidateFileChange,
    MultiFileCandidate,
    verify_candidate_workspace,
)
from repository_test_tool import build_repository_test_tool
from sandbox.models import SandboxExecutionResult, SandboxProvenance
from sandbox.policy import SandboxPolicy
from sandbox.runner import SandboxRunner
from static_analysis import StaticAnalysisResult
from test_execution import TestRunResult, execute_test_files


def docker_result() -> SandboxExecutionResult:
    provenance = SandboxProvenance(
        backend="docker", sandboxed=True, image="repograph-sandbox:local",
        image_id="sha256:test", network="none", read_only_root=True,
        cap_drop_all=True, no_new_privileges=True, memory_limit_mb=512,
        cpu_limit=1, pids_limit=64, timeout_seconds=30,
    )
    return SandboxExecutionResult(
        status="passed", exit_code=0, duration_seconds=0,
        backend="docker", provenance=provenance,
    )


class SandboxIntegrationTests(unittest.TestCase):
    def test_docker_runner_mounts_a_secret_free_copy_not_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "test_ok.py").write_text("def test_ok(): assert True", encoding="utf-8")
            (source / ".env").write_text("OPENAI_API_KEY=secret", encoding="utf-8")

            def inspect(_backend, request):
                mounted = Path(request.workspace)
                self.assertNotEqual(mounted.resolve(), source.resolve())
                self.assertFalse((mounted / ".env").exists())
                self.assertTrue((mounted / "test_ok.py").exists())
                return docker_result()

            with patch("sandbox.docker.DockerSandboxBackend.run", autospec=True, side_effect=inspect):
                result = SandboxRunner(SandboxPolicy(backend="docker")).run_repository(
                    argv=["python", "-m", "pytest", "test_ok.py"],
                    repository_root=source, timeout_seconds=30, purpose="tests",
                )
            self.assertEqual(result.backend, "docker")
            self.assertEqual((source / ".env").read_text(encoding="utf-8"), "OPENAI_API_KEY=secret")

    def test_requested_docker_never_calls_host_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "sandbox.docker.DockerSandboxBackend._image_id", return_value=None
        ), patch("sandbox.runner.HostSandboxBackend.run") as host:
            root = Path(directory)
            (root / "test_ok.py").write_text("def test_ok(): assert True", encoding="utf-8")
            result = execute_test_files(
                str(root), ["test_ok.py"],
                sandbox_policy=SandboxPolicy(backend="docker"),
            )
        self.assertEqual(result.status, "error")
        self.assertEqual(result.execution_backend, "docker")
        host.assert_not_called()

    def test_candidate_verification_routes_planned_tests_through_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "service.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "test_service.py").write_text("def test_value(): assert True\n", encoding="utf-8")
            candidate = MultiFileCandidate(
                summary="candidate",
                files=[CandidateFileChange(path="service.py", action="modify", content="VALUE = 1\n")],
            )
            plan = EngineeringPlan(
                summary="plan",
                files=[PlannedFileChange(path="service.py", action="modify", rationale="r")],
                tests=[PlannedTest(path="test_service.py", purpose="test")],
            )
            with patch(
                "plan_execution.execute_test_files",
                return_value=TestRunResult(status="passed", framework="pytest"),
            ) as execute, patch(
                "plan_execution.analyze_code", return_value=StaticAnalysisResult()
            ):
                policy = SandboxPolicy(backend="docker")
                verify_candidate_workspace(str(root), candidate, plan, sandbox_policy=policy)
            self.assertIs(execute.call_args.kwargs["sandbox_policy"], policy)

    def test_explorer_test_tool_routes_through_shared_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_ok.py").write_text("def test_ok(): assert True\n", encoding="utf-8")
            policy = SandboxPolicy(backend="docker")
            with patch(
                "repository_test_tool.execute_test_files",
                return_value=TestRunResult(status="passed", framework="pytest"),
            ) as execute:
                tool = build_repository_test_tool(
                    str(root), ["test_ok.py"], sandbox_policy=policy
                )
                output = tool.invoke({"test_file": "test_ok.py"})
            self.assertIn("Status: PASSED", output)
            self.assertIs(execute.call_args.kwargs["sandbox_policy"], policy)

    def test_evaluator_routes_through_shared_policy_and_records_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = EvaluationTask(
                id="sandbox-evaluator", dataset="test", repository=str(root),
                base_commit="abc", task="evaluate", test_command=["python", "-V"],
            )
            policy = SandboxPolicy(backend="docker")
            with patch(
                "evaluation.evaluator.SandboxRunner.run_repository",
                return_value=docker_result(),
            ) as run:
                outcome = run_local_evaluator(
                    task, root, timeout_seconds=30, max_output_chars=2_000,
                    sandbox_policy=policy,
                )
            self.assertEqual(outcome.status, "passed")
            self.assertTrue(outcome.sandbox_provenance.sandboxed)
            self.assertEqual(run.call_args.kwargs["purpose"], "evaluation")


if __name__ == "__main__":
    unittest.main()
