import importlib.util
import io
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from agent import (
    CodeReview,
    FindingCategory,
    OverallRating,
    ReviewFinding,
    Severity,
    main,
    review_code,
)
from git_diff import GitDiffResult
from repository_context import RepoContext, RepositoryFile
from static_analysis import StaticAnalysisResult
from test_execution import (
    MAX_TEST_FILES,
    MAX_TEST_OUTPUT_CHARS,
    TEST_TIMEOUT_SECONDS,
    TestRunResult,
    execute_targeted_tests,
)


def completed(
    returncode: int = 0,
    *,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def good_review() -> CodeReview:
    return CodeReview(
        overall_rating=OverallRating.GOOD,
        summary="No concrete issue was found.",
        findings=[],
    )


def needs_work_review() -> CodeReview:
    return CodeReview(
        overall_rating=OverallRating.NEEDS_WORK,
        summary="The runtime evidence needs investigation.",
        findings=[
            ReviewFinding(
                category=FindingCategory.BUG,
                severity=Severity.MEDIUM,
                title="Related test failure",
                description="A related behavior does not match its expectation.",
                line_number=1,
                suggestion="Inspect the failing assertion before changing code.",
            )
        ],
    )


class TemporaryTestRepository(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.repository = Path(self.temporary_directory.name) / "repo"
        self.repository.mkdir()
        self.write_file("app.py", "VALUE = 42\n")

    def write_file(self, relative_path: str, content: str = "") -> Path:
        path = self.repository / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def context(self, *test_files: str) -> RepoContext:
        return RepoContext(
            repository_root=str(self.repository),
            target_file="app.py",
            related_files=[
                RepositoryFile(
                    path=test_file,
                    relationship="test",
                    content="def test_example():\n    assert True\n",
                )
                for test_file in test_files
            ],
        )


class TestExecutionSchemaTests(unittest.TestCase):
    def test_schema_forbids_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            TestRunResult.model_validate(
                {"status": "not_run", "unexpected": True}
            )


class TestExecutionBoundaryTests(TemporaryTestRepository):
    @patch("test_execution.subprocess.run")
    def test_disabled_execution_never_starts_subprocess(self, mocked_run) -> None:
        result = execute_targeted_tests(
            str(self.repository),
            self.context(),
            "python",
            enabled=False,
        )

        self.assertEqual(result.status, "not_run")
        mocked_run.assert_not_called()

    @patch("test_execution.subprocess.run", return_value=completed(stdout="1 passed"))
    def test_related_python_test_is_executed(self, mocked_run) -> None:
        self.write_file("tests/test_app.py")

        result = execute_targeted_tests(
            str(self.repository),
            self.context("tests/test_app.py"),
            "python",
            enabled=True,
        )

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.framework, "pytest")
        self.assertEqual(result.test_files, ["tests/test_app.py"])
        mocked_run.assert_called_once()

    @patch("test_execution.subprocess.run")
    def test_no_related_tests_returns_not_run(self, mocked_run) -> None:
        result = execute_targeted_tests(
            str(self.repository),
            self.context(),
            "python",
            enabled=True,
        )

        self.assertEqual(result.status, "not_run")
        self.assertIn("No related test files", result.warnings[-1])
        mocked_run.assert_not_called()

    @patch("test_execution.subprocess.run", return_value=completed())
    def test_multiple_related_tests_share_one_invocation(self, mocked_run) -> None:
        paths = ["tests/test_app.py", "tests/app_test.py"]
        for path in paths:
            self.write_file(path)

        result = execute_targeted_tests(
            str(self.repository),
            self.context(*paths),
            "python",
            enabled=True,
        )

        self.assertEqual(result.test_files, paths)
        command = mocked_run.call_args.args[0]
        self.assertEqual(command[-2:], paths)

    @patch("test_execution.subprocess.run", return_value=completed())
    def test_test_file_budget_truncates_selection(self, mocked_run) -> None:
        paths = [f"tests/test_app_{index}.py" for index in range(MAX_TEST_FILES + 1)]
        for path in paths:
            self.write_file(path)

        result = execute_targeted_tests(
            str(self.repository),
            self.context(*paths),
            "python",
            enabled=True,
        )

        self.assertEqual(result.test_files, paths[:MAX_TEST_FILES])
        self.assertTrue(any("MAX_TEST_FILES" in item for item in result.warnings))
        self.assertEqual(len(mocked_run.call_args.args[0]) - 3, MAX_TEST_FILES)

    @patch("test_execution.subprocess.run")
    def test_path_outside_repository_is_rejected(self, mocked_run) -> None:
        outside = self.repository.parent / "outside_test.py"
        outside.write_text("def test_outside(): pass\n", encoding="utf-8")

        result = execute_targeted_tests(
            str(self.repository),
            self.context("../outside_test.py"),
            "python",
            enabled=True,
        )

        self.assertEqual(result.status, "not_run")
        self.assertTrue(any("outside repository root" in item for item in result.warnings))
        mocked_run.assert_not_called()

    @patch("test_execution.subprocess.run")
    def test_missing_selected_test_is_not_executed(self, mocked_run) -> None:
        result = execute_targeted_tests(
            str(self.repository),
            self.context("tests/test_missing.py"),
            "python",
            enabled=True,
        )

        self.assertEqual(result.status, "not_run")
        self.assertTrue(any("could not be resolved" in item for item in result.warnings))
        mocked_run.assert_not_called()

    @patch("test_execution.subprocess.run", return_value=completed())
    def test_subprocess_contract_is_bounded_and_never_uses_shell(self, mocked_run) -> None:
        self.write_file("tests/test_app.py")

        execute_targeted_tests(
            str(self.repository),
            self.context("tests/test_app.py"),
            "python",
            enabled=True,
        )

        call = mocked_run.call_args
        self.assertEqual(call.args[0][1:3], ["-m", "pytest"])
        self.assertFalse(call.kwargs["shell"])
        self.assertTrue(call.kwargs["capture_output"])
        self.assertTrue(call.kwargs["text"])
        self.assertFalse(call.kwargs["check"])
        self.assertEqual(call.kwargs["timeout"], TEST_TIMEOUT_SECONDS)
        self.assertEqual(call.kwargs["cwd"], str(self.repository.resolve()))
        self.assertEqual(call.kwargs["env"]["PYTHONUNBUFFERED"], "1")
        self.assertEqual(call.kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")

    @patch("test_execution.subprocess.run")
    def test_timeout_is_runtime_evidence(self, mocked_run) -> None:
        self.write_file("tests/test_app.py")
        mocked_run.side_effect = subprocess.TimeoutExpired(
            cmd=["python", "-m", "pytest"],
            timeout=TEST_TIMEOUT_SECONDS,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )

        result = execute_targeted_tests(
            str(self.repository),
            self.context("tests/test_app.py"),
            "python",
            enabled=True,
        )

        self.assertEqual(result.status, "timed_out")
        self.assertIsNone(result.exit_code)
        self.assertEqual(result.stdout, "partial stdout")
        self.assertEqual(result.stderr, "partial stderr")

    @patch("test_execution.subprocess.run", return_value=completed(0))
    def test_pytest_exit_zero_maps_to_passed(self, _mocked_run) -> None:
        self.write_file("tests/test_app.py")
        result = execute_targeted_tests(
            str(self.repository), self.context("tests/test_app.py"), "python", enabled=True
        )
        self.assertEqual(result.status, "passed")

    @patch("test_execution.subprocess.run", return_value=completed(1, stdout="FAILED"))
    def test_pytest_assertion_failure_maps_to_failed(self, _mocked_run) -> None:
        self.write_file("tests/test_app.py")
        result = execute_targeted_tests(
            str(self.repository), self.context("tests/test_app.py"), "python", enabled=True
        )
        self.assertEqual(result.status, "failed")

    @patch("test_execution.subprocess.run", return_value=completed(2, stderr="ImportError"))
    def test_pytest_collection_failure_maps_to_error(self, _mocked_run) -> None:
        self.write_file("tests/test_app.py")
        result = execute_targeted_tests(
            str(self.repository), self.context("tests/test_app.py"), "python", enabled=True
        )
        self.assertEqual(result.status, "error")

    @patch(
        "test_execution.subprocess.run",
        return_value=completed(1, stderr="No module named pytest"),
    )
    def test_missing_pytest_maps_to_error_without_installing(self, _mocked_run) -> None:
        self.write_file("tests/test_app.py")
        result = execute_targeted_tests(
            str(self.repository), self.context("tests/test_app.py"), "python", enabled=True
        )
        self.assertEqual(result.status, "error")
        self.assertTrue(any("not available" in item for item in result.warnings))

    @patch("test_execution.subprocess.run")
    def test_stdout_is_bounded(self, mocked_run) -> None:
        self.write_file("tests/test_app.py")
        mocked_run.return_value = completed(stdout="x" * (MAX_TEST_OUTPUT_CHARS + 1))
        result = execute_targeted_tests(
            str(self.repository), self.context("tests/test_app.py"), "python", enabled=True
        )
        self.assertEqual(len(result.stdout), MAX_TEST_OUTPUT_CHARS)
        self.assertTrue(any("stdout truncated" in item for item in result.warnings))

    @patch("test_execution.subprocess.run")
    def test_stderr_is_bounded(self, mocked_run) -> None:
        self.write_file("tests/test_app.py")
        mocked_run.return_value = completed(stderr="x" * (MAX_TEST_OUTPUT_CHARS + 1))
        result = execute_targeted_tests(
            str(self.repository), self.context("tests/test_app.py"), "python", enabled=True
        )
        self.assertEqual(len(result.stderr), MAX_TEST_OUTPUT_CHARS)
        self.assertTrue(any("stderr truncated" in item for item in result.warnings))


class TestAwareWorkflowTests(TemporaryTestRepository):
    def _review_with_evidence(
        self,
        evidence: TestRunResult,
        responses: list[CodeReview],
        **review_arguments,
    ):
        repository_context = self.context("tests/test_app.py")
        self.write_file("tests/test_app.py")
        with (
            patch("agent.build_repository_context", return_value=repository_context),
            patch("agent.analyze_code", return_value=StaticAnalysisResult()),
            patch("agent.execute_targeted_tests", return_value=evidence) as executor,
            patch("agent.ChatOpenAI") as chat_openai,
        ):
            structured = chat_openai.return_value.with_structured_output.return_value
            structured.invoke.side_effect = responses
            result = review_code(
                "VALUE = 42\n",
                repository_root=str(self.repository),
                target_file="app.py",
                run_tests=True,
                **review_arguments,
            )
        return result, executor, structured

    def test_test_result_enters_prompt(self) -> None:
        evidence = TestRunResult(
            status="failed",
            framework="pytest",
            test_files=["tests/test_app.py"],
            exit_code=1,
            stdout="test_app FAILED",
        )
        _, _, structured = self._review_with_evidence(
            evidence,
            [needs_work_review()],
        )

        prompt = structured.invoke.call_args.args[0][1].content
        self.assertIn("Targeted test evidence:", prompt)
        self.assertIn("Status: FAILED", prompt)
        self.assertIn("tests/test_app.py", prompt)
        self.assertIn("test_app FAILED", prompt)
        self.assertIn("Do not claim tests passed", prompt)

    def test_failing_tests_with_good_empty_review_trigger_retry(self) -> None:
        evidence = TestRunResult(status="failed", framework="pytest", exit_code=1)
        result, _, structured = self._review_with_evidence(
            evidence,
            [good_review(), needs_work_review()],
        )

        self.assertEqual(result, needs_work_review())
        self.assertEqual(structured.invoke.call_count, 2)
        retry_prompt = structured.invoke.call_args_list[1].args[0][1].content
        self.assertIn("conflicts with failing targeted tests", retry_prompt)

    def test_test_error_does_not_force_a_code_finding(self) -> None:
        evidence = TestRunResult(
            status="error",
            framework="pytest",
            exit_code=2,
            stderr="collection error",
        )
        result, _, structured = self._review_with_evidence(
            evidence,
            [good_review()],
        )

        self.assertEqual(result, good_review())
        self.assertEqual(structured.invoke.call_count, 1)

    @patch("test_execution.subprocess.run", return_value=completed(1, stdout="FAILED"))
    def test_two_semantic_retries_execute_test_subprocess_once(self, mocked_run) -> None:
        self.write_file("tests/test_app.py")
        repository_context = self.context("tests/test_app.py")
        with (
            patch("agent.build_repository_context", return_value=repository_context),
            patch("agent.analyze_code", return_value=StaticAnalysisResult()),
            patch("agent.ChatOpenAI") as chat_openai,
        ):
            structured = chat_openai.return_value.with_structured_output.return_value
            structured.invoke.side_effect = [
                good_review(),
                good_review(),
                needs_work_review(),
            ]
            result = review_code(
                "VALUE = 42\n",
                repository_root=str(self.repository),
                target_file="app.py",
                run_tests=True,
            )

        self.assertEqual(result, needs_work_review())
        self.assertEqual(mocked_run.call_count, 1)
        self.assertEqual(structured.invoke.call_count, 3)

    @patch("test_execution.subprocess.run")
    def test_existing_single_file_behavior_does_not_run_tests(self, mocked_run) -> None:
        with (
            patch("agent.analyze_code", return_value=StaticAnalysisResult()),
            patch("agent.ChatOpenAI") as chat_openai,
        ):
            structured = chat_openai.return_value.with_structured_output.return_value
            structured.invoke.return_value = good_review()
            result = review_code("VALUE = 42\n")

        self.assertEqual(result, good_review())
        mocked_run.assert_not_called()

    def test_repository_only_and_tests_work_together(self) -> None:
        evidence = TestRunResult(status="passed", framework="pytest", exit_code=0)
        result, executor, _ = self._review_with_evidence(evidence, [good_review()])
        self.assertEqual(result, good_review())
        executor.assert_called_once()

    def test_manual_diff_and_tests_work_together(self) -> None:
        diff = "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-OLD = 1\n+VALUE = 42\n"
        evidence = TestRunResult(status="passed", framework="pytest", exit_code=0)
        result, executor, structured = self._review_with_evidence(
            evidence,
            [good_review()],
            diff_text=diff,
        )
        self.assertEqual(result, good_review())
        executor.assert_called_once()
        prompt = structured.invoke.call_args.args[0][1].content
        self.assertIn("Diff context:", prompt)
        self.assertIn("Status: PASSED", prompt)

    def test_git_diff_and_tests_work_together(self) -> None:
        diff = "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-OLD = 1\n+VALUE = 42\n"
        evidence = TestRunResult(status="passed", framework="pytest", exit_code=0)
        repository_context = self.context("tests/test_app.py")
        self.write_file("tests/test_app.py")
        git_result = GitDiffResult(
            mode="working_tree",
            repository_root=str(self.repository),
            target_file="app.py",
            diff_text=diff,
        )
        with (
            patch("agent.build_repository_context", return_value=repository_context),
            patch("agent.collect_git_diff", return_value=git_result) as collector,
            patch("agent.analyze_code", return_value=StaticAnalysisResult()),
            patch("agent.execute_targeted_tests", return_value=evidence) as executor,
            patch("agent.ChatOpenAI") as chat_openai,
        ):
            structured = chat_openai.return_value.with_structured_output.return_value
            structured.invoke.return_value = good_review()
            result = review_code(
                "VALUE = 42\n",
                repository_root=str(self.repository),
                target_file="app.py",
                git_diff=True,
                run_tests=True,
            )

        self.assertEqual(result, good_review())
        collector.assert_called_once()
        executor.assert_called_once()

    def test_api_rejects_test_execution_without_repository(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires repository_root"):
            review_code("VALUE = 42\n", run_tests=True)

    def test_api_rejects_non_python_test_execution(self) -> None:
        with self.assertRaisesRegex(ValueError, "language='python'"):
            review_code(
                "const value = 42;",
                language="javascript",
                repository_root=str(self.repository),
                target_file="app.py",
                run_tests=True,
            )


class TestAwareCliTests(TemporaryTestRepository):
    def _argparse_error(self, argv: list[str]) -> str:
        stderr = io.StringIO()
        with (
            patch("sys.argv", argv),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as captured,
        ):
            main()
        self.assertEqual(captured.exception.code, 2)
        return stderr.getvalue()

    def test_run_tests_requires_repo(self) -> None:
        target = self.repository / "app.py"
        error = self._argparse_error(
            ["agent.py", "--file", str(target), "--run-tests"]
        )
        self.assertIn("--run-tests requires --repo", error)

    def test_run_tests_requires_file(self) -> None:
        error = self._argparse_error(
            [
                "agent.py",
                "--code",
                "VALUE = 42",
                "--repo",
                str(self.repository),
                "--run-tests",
            ]
        )
        self.assertIn("--run-tests requires --file", error)

    def test_run_tests_rejects_non_python_language(self) -> None:
        error = self._argparse_error(
            [
                "agent.py",
                "--file",
                "app.py",
                "--repo",
                str(self.repository),
                "--language",
                "javascript",
                "--run-tests",
            ]
        )
        self.assertIn("requires --language python", error)

    @patch("agent.review_code", return_value=good_review())
    def test_run_tests_flag_reaches_review_api(self, mocked_review_code) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = [
            "agent.py",
            "--file",
            "app.py",
            "--repo",
            str(self.repository),
            "--run-tests",
        ]
        with (
            patch("sys.argv", argv),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        mocked_review_code.assert_called_once_with(
            "VALUE = 42\n",
            "python",
            repository_root=str(self.repository),
            target_file="app.py",
            run_tests=True,
        )


@unittest.skipUnless(
    importlib.util.find_spec("pytest") is not None,
    "pytest is not installed in the current interpreter",
)
class RealPytestIntegrationTests(TemporaryTestRepository):
    def test_real_passing_related_test(self) -> None:
        self.write_file("src/calculator.py", "def add(a, b):\n    return a + b\n")
        self.write_file(
            "tests/test_calculator.py",
            "from src.calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        )
        result = execute_targeted_tests(
            str(self.repository),
            self.context("tests/test_calculator.py"),
            "python",
            enabled=True,
        )
        self.assertEqual(result.status, "passed", result.stdout + result.stderr)

    def test_real_failing_related_test(self) -> None:
        self.write_file("src/calculator.py", "def add(a, b):\n    return a + b\n")
        self.write_file(
            "tests/test_calculator.py",
            "from src.calculator import add\n\ndef test_add():\n    assert add(2, 3) == 6\n",
        )
        result = execute_targeted_tests(
            str(self.repository),
            self.context("tests/test_calculator.py"),
            "python",
            enabled=True,
        )
        self.assertEqual(result.status, "failed", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
