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
from diff_context import parse_unified_diff
from git_diff import (
    GIT_TIMEOUT_SECONDS,
    GitDiffError,
    GitDiffResult,
    collect_base_diff,
    collect_working_tree_diff,
)
from repository_context import RepoContext
from static_analysis import StaticAnalysisResult

HEAD_SHA = "a" * 40
BASE_SHA = "b" * 40
RAW_DIFF = """--- a/app.py
+++ b/app.py
@@ -1 +1 @@
-old
+new
"""


def completed(
    stdout: str = "",
    *,
    returncode: int = 0,
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def successful_review() -> CodeReview:
    return CodeReview(
        overall_rating=OverallRating.NEEDS_WORK,
        summary="The supplied change needs review.",
        findings=[
            ReviewFinding(
                category=FindingCategory.BUG,
                severity=Severity.HIGH,
                title="Example regression",
                description="The changed behavior can fail.",
                line_number=1,
                suggestion="Preserve the required behavior.",
            )
        ],
    )


def invalid_review() -> CodeReview:
    return successful_review().model_copy(
        update={"overall_rating": OverallRating.GOOD}
    )


class TemporaryGitRepositoryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.workspace = Path(self.temporary_directory.name)
        self.repository = self.workspace / "repo"
        self.repository.mkdir()
        self.target = self.repository / "app.py"
        self.target.write_text("new\n", encoding="utf-8")

    def working_tree_results(
        self,
        diff_text: str = RAW_DIFF,
    ) -> list[subprocess.CompletedProcess[str]]:
        return [
            completed("true\n"),
            completed(f"{HEAD_SHA}\n"),
            completed(),
            completed(diff_text),
        ]

    def base_results(
        self,
        diff_text: str = RAW_DIFF,
    ) -> list[subprocess.CompletedProcess[str]]:
        return [
            completed("true\n"),
            completed(f"{HEAD_SHA}\n"),
            completed(f"{BASE_SHA}\n"),
            completed(),
            completed(diff_text),
        ]


class GitDiffSchemaTests(unittest.TestCase):
    def test_schema_forbids_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            GitDiffResult.model_validate(
                {
                    "mode": "working_tree",
                    "repository_root": "repo",
                    "target_file": "app.py",
                    "unexpected": True,
                }
            )


class GitDiffValidationTests(TemporaryGitRepositoryTestCase):
    @patch("git_diff.subprocess.run")
    def test_valid_git_worktree_is_accepted(self, mocked_run) -> None:
        mocked_run.side_effect = self.working_tree_results()

        result = collect_working_tree_diff(
            str(self.repository),
            "app.py",
        )

        self.assertEqual(result.mode, "working_tree")
        self.assertEqual(result.repository_root, str(self.repository.resolve()))
        self.assertEqual(result.target_file, "app.py")
        self.assertEqual(result.diff_text, RAW_DIFF)

    @patch("git_diff.subprocess.run")
    def test_non_git_directory_has_controlled_error(self, mocked_run) -> None:
        mocked_run.return_value = completed(
            returncode=128,
            stderr="not a git repository",
        )

        with self.assertRaises(GitDiffError) as captured:
            collect_working_tree_diff(str(self.repository), "app.py")

        self.assertEqual(captured.exception.code, "not_git_worktree")
        self.assertIn("not inside a Git worktree", str(captured.exception))

    @patch("git_diff.subprocess.run")
    def test_missing_git_executable_has_controlled_error(self, mocked_run) -> None:
        mocked_run.side_effect = FileNotFoundError()

        with self.assertRaises(GitDiffError) as captured:
            collect_working_tree_diff(str(self.repository), "app.py")

        self.assertEqual(captured.exception.code, "git_not_found")
        self.assertIn("Git executable was not found", str(captured.exception))

    @patch("git_diff.subprocess.run")
    def test_subprocess_timeout_has_controlled_error(self, mocked_run) -> None:
        mocked_run.side_effect = subprocess.TimeoutExpired(
            cmd=["git"],
            timeout=GIT_TIMEOUT_SECONDS,
        )

        with self.assertRaises(GitDiffError) as captured:
            collect_working_tree_diff(str(self.repository), "app.py")

        self.assertEqual(captured.exception.code, "git_timeout")
        self.assertIn(str(GIT_TIMEOUT_SECONDS), str(captured.exception))

    @patch("git_diff.subprocess.run")
    def test_target_outside_repository_is_rejected_before_git(
        self,
        mocked_run,
    ) -> None:
        outside = self.workspace / "outside.py"
        outside.write_text("secret\n", encoding="utf-8")

        with self.assertRaises(GitDiffError) as captured:
            collect_working_tree_diff(
                str(self.repository),
                "../outside.py",
            )

        self.assertEqual(
            captured.exception.code,
            "target_outside_repository",
        )
        mocked_run.assert_not_called()

    @patch("git_diff.subprocess.run")
    def test_repository_without_head_has_controlled_error(
        self,
        mocked_run,
    ) -> None:
        mocked_run.side_effect = [
            completed("true\n"),
            completed(returncode=128, stderr="unknown revision HEAD"),
        ]

        with self.assertRaises(GitDiffError) as captured:
            collect_working_tree_diff(str(self.repository), "app.py")

        self.assertEqual(captured.exception.code, "head_unresolved")
        self.assertIn("initial commit", str(captured.exception))


class WorkingTreeCommandTests(TemporaryGitRepositoryTestCase):
    @patch("git_diff.subprocess.run")
    def test_working_tree_uses_head_and_safe_diff_options(
        self,
        mocked_run,
    ) -> None:
        mocked_run.side_effect = self.working_tree_results()

        collect_working_tree_diff(str(self.repository), "app.py")

        calls = mocked_run.call_args_list
        head_call = calls[1].args[0]
        diff_call = calls[3]
        command = diff_call.args[0]

        self.assertEqual(
            head_call,
            [
                "git",
                "--no-pager",
                "--literal-pathspecs",
                "rev-parse",
                "--verify",
                "--end-of-options",
                "HEAD^{commit}",
            ],
        )
        self.assertEqual(
            command[0:4],
            ["git", "--no-pager", "--literal-pathspecs", "diff"],
        )
        self.assertIn("HEAD", command)
        self.assertIn("--no-ext-diff", command)
        self.assertIn("--no-textconv", command)
        self.assertIn("--no-color", command)
        self.assertEqual(command[-2:], ["--", "app.py"])
        self.assertFalse(diff_call.kwargs["shell"])
        self.assertTrue(diff_call.kwargs["capture_output"])
        self.assertTrue(diff_call.kwargs["text"])
        self.assertFalse(diff_call.kwargs["check"])
        self.assertEqual(
            diff_call.kwargs["timeout"],
            GIT_TIMEOUT_SECONDS,
        )

    @patch("git_diff.subprocess.run")
    def test_head_diff_represents_staged_and_unstaged_target_state(
        self,
        mocked_run,
    ) -> None:
        mocked_run.side_effect = self.working_tree_results()

        collect_working_tree_diff(str(self.repository), "app.py")

        diff_command = mocked_run.call_args_list[3].args[0]
        revision_index = diff_command.index("HEAD")
        separator_index = diff_command.index("--")
        self.assertLess(revision_index, separator_index)
        self.assertEqual(diff_command[separator_index + 1], "app.py")

    @patch("git_diff.subprocess.run")
    def test_all_git_calls_use_list_arguments_and_shell_false(
        self,
        mocked_run,
    ) -> None:
        mocked_run.side_effect = self.working_tree_results()

        collect_working_tree_diff(str(self.repository), "app.py")

        for git_call in mocked_run.call_args_list:
            self.assertIsInstance(git_call.args[0], list)
            self.assertEqual(git_call.args[0][0:2], ["git", "--no-pager"])
            self.assertIn("--literal-pathspecs", git_call.args[0])
            self.assertFalse(git_call.kwargs["shell"])
            self.assertTrue(git_call.kwargs["capture_output"])
            self.assertTrue(git_call.kwargs["text"])
            self.assertFalse(git_call.kwargs["check"])
            self.assertEqual(
                git_call.kwargs["timeout"],
                GIT_TIMEOUT_SECONDS,
            )


class BaseCommandTests(TemporaryGitRepositoryTestCase):
    @patch("git_diff.subprocess.run")
    def test_base_ref_is_resolved_before_diff(self, mocked_run) -> None:
        mocked_run.side_effect = self.base_results()

        result = collect_base_diff(
            str(self.repository),
            "app.py",
            "main",
        )

        resolve_command = mocked_run.call_args_list[2].args[0]
        self.assertEqual(
            resolve_command[-3:],
            ["--verify", "--end-of-options", "main^{commit}"],
        )
        self.assertEqual(result.base_ref, "main")

    @patch("git_diff.subprocess.run")
    def test_base_mode_uses_resolved_sha_three_dot_head(
        self,
        mocked_run,
    ) -> None:
        mocked_run.side_effect = self.base_results()

        collect_base_diff(str(self.repository), "app.py", "main")

        diff_call = mocked_run.call_args_list[4]
        command = diff_call.args[0]
        self.assertIn(f"{BASE_SHA}...HEAD", command)
        self.assertEqual(command[-2:], ["--", "app.py"])
        self.assertIn("--no-ext-diff", command)
        self.assertIn("--no-textconv", command)
        self.assertFalse(diff_call.kwargs["shell"])

    @patch("git_diff.subprocess.run")
    def test_invalid_local_base_has_controlled_error(self, mocked_run) -> None:
        mocked_run.side_effect = [
            completed("true\n"),
            completed(f"{HEAD_SHA}\n"),
            completed(returncode=128, stderr="unknown revision"),
        ]

        with self.assertRaises(GitDiffError) as captured:
            collect_base_diff(
                str(self.repository),
                "app.py",
                "missing",
            )

        self.assertEqual(captured.exception.code, "base_unresolved")
        self.assertEqual(
            str(captured.exception),
            "Base ref could not be resolved locally.",
        )

    @patch("git_diff.subprocess.run")
    def test_option_like_base_is_rejected_without_subprocess_injection(
        self,
        mocked_run,
    ) -> None:
        with self.assertRaises(GitDiffError) as captured:
            collect_base_diff(
                str(self.repository),
                "app.py",
                "--upload-pack=evil",
            )

        self.assertEqual(captured.exception.code, "base_unresolved")
        mocked_run.assert_not_called()


class GitDiffEdgeCaseTests(TemporaryGitRepositoryTestCase):
    @patch("git_diff.subprocess.run")
    def test_untracked_target_returns_explicit_warning_without_diff(
        self,
        mocked_run,
    ) -> None:
        mocked_run.side_effect = [
            completed("true\n"),
            completed(f"{HEAD_SHA}\n"),
            completed(returncode=1),
        ]

        result = collect_working_tree_diff(
            str(self.repository),
            "app.py",
        )

        self.assertEqual(result.diff_text, "")
        self.assertIn(
            "Target is untracked and is not represented by normal HEAD diff.",
            result.warnings,
        )
        self.assertFalse(
            any(
                call.args[0][2:3] == ["diff"]
                for call in mocked_run.call_args_list
            )
        )

    @patch("git_diff.subprocess.run")
    def test_git_diff_size_budget_is_reported(self, mocked_run) -> None:
        mocked_run.side_effect = self.working_tree_results("X" * 20)

        with patch("git_diff.MAX_DIFF_CHARS", 10):
            result = collect_working_tree_diff(
                str(self.repository),
                "app.py",
            )

        self.assertEqual(result.diff_text, "X" * 10)
        self.assertIn(
            "Git diff truncated at MAX_DIFF_CHARS=10.",
            result.warnings,
        )


class GitAwareAgentTests(unittest.TestCase):
    def _git_result(
        self,
        *,
        mode: str = "working_tree",
        diff_text: str = RAW_DIFF,
        warnings: list[str] | None = None,
    ) -> GitDiffResult:
        return GitDiffResult(
            mode=mode,
            repository_root="repo",
            target_file="app.py",
            base_ref="main" if mode == "base" else None,
            diff_text=diff_text,
            warnings=warnings or [],
        )

    @patch("agent.collect_git_diff")
    @patch("agent.ChatOpenAI")
    @patch("agent.analyze_code", return_value=StaticAnalysisResult())
    def test_manual_diff_does_not_collect_git(
        self,
        _analyze_code,
        chat_openai,
        git_collector,
    ) -> None:
        structured_llm = (
            chat_openai.return_value.with_structured_output.return_value
        )
        structured_llm.invoke.return_value = successful_review()

        review_code(
            "new\n",
            target_file="app.py",
            diff_text=RAW_DIFF,
        )

        git_collector.assert_not_called()
        prompt = structured_llm.invoke.call_args.args[0][1].content
        self.assertIn("+new", prompt)

    @patch("agent.collect_git_diff")
    @patch("agent.ChatOpenAI")
    @patch("agent.analyze_code", return_value=StaticAnalysisResult())
    def test_no_diff_mode_does_not_collect_git(
        self,
        _analyze_code,
        chat_openai,
        git_collector,
    ) -> None:
        structured_llm = (
            chat_openai.return_value.with_structured_output.return_value
        )
        structured_llm.invoke.return_value = successful_review()

        review_code("new\n")

        git_collector.assert_not_called()
        prompt = structured_llm.invoke.call_args.args[0][1].content
        self.assertNotIn("Diff context:", prompt)

    def test_git_raw_diff_enters_existing_parser_and_prompt(self) -> None:
        git_result = self._git_result()
        with (
            patch(
                "agent.build_repository_context",
                return_value=RepoContext(
                    repository_root="repo",
                    target_file="app.py",
                ),
            ),
            patch(
                "agent.collect_git_diff",
                return_value=git_result,
            ) as git_collector,
            patch(
                "agent.parse_unified_diff",
                wraps=parse_unified_diff,
            ) as diff_parser,
            patch(
                "agent.analyze_code",
                return_value=StaticAnalysisResult(),
            ),
            patch("agent.ChatOpenAI") as chat_openai,
        ):
            structured_llm = (
                chat_openai.return_value.with_structured_output.return_value
            )
            structured_llm.invoke.return_value = successful_review()

            review_code(
                "new\n",
                repository_root="repo",
                target_file="app.py",
                git_diff=True,
            )

        git_collector.assert_called_once_with(
            "repo",
            "app.py",
            "working_tree",
            None,
        )
        diff_parser.assert_called_once_with(RAW_DIFF, "app.py")
        prompt = structured_llm.invoke.call_args.args[0][1].content
        self.assertIn("Diff context:", prompt)
        self.assertIn("-old", prompt)
        self.assertIn("+new", prompt)

    def test_git_warnings_enter_existing_diff_context(self) -> None:
        warning = (
            "Target is untracked and is not represented by normal HEAD diff."
        )
        git_result = self._git_result(diff_text="", warnings=[warning])
        with (
            patch(
                "agent.build_repository_context",
                return_value=RepoContext(
                    repository_root="repo",
                    target_file="app.py",
                ),
            ),
            patch("agent.collect_git_diff", return_value=git_result),
            patch(
                "agent.analyze_code",
                return_value=StaticAnalysisResult(),
            ),
            patch("agent.ChatOpenAI") as chat_openai,
        ):
            structured_llm = (
                chat_openai.return_value.with_structured_output.return_value
            )
            structured_llm.invoke.return_value = successful_review()

            review_code(
                "new\n",
                repository_root="repo",
                target_file="app.py",
                git_diff=True,
            )

        prompt = structured_llm.invoke.call_args.args[0][1].content
        self.assertIn("Diff target: app.py", prompt)
        self.assertIn(warning, prompt)

    def test_base_api_collects_local_base_mode_and_reuses_prompt(self) -> None:
        git_result = self._git_result(mode="base").model_copy(
            update={"repository_root": "C:/ABSOLUTE_SECRET_ROOT"}
        )
        with (
            patch(
                "agent.build_repository_context",
                return_value=RepoContext(
                    repository_root="repo",
                    target_file="app.py",
                ),
            ),
            patch(
                "agent.collect_git_diff",
                return_value=git_result,
            ) as git_collector,
            patch(
                "agent.analyze_code",
                return_value=StaticAnalysisResult(),
            ),
            patch("agent.ChatOpenAI") as chat_openai,
        ):
            structured_llm = (
                chat_openai.return_value.with_structured_output.return_value
            )
            structured_llm.invoke.return_value = successful_review()

            review_code(
                "new\n",
                repository_root="repo",
                target_file="app.py",
                git_base="main",
            )

        git_collector.assert_called_once_with(
            "repo",
            "app.py",
            "base",
            "main",
        )
        prompt = structured_llm.invoke.call_args.args[0][1].content
        self.assertIn("+new", prompt)
        self.assertNotIn(git_result.repository_root, prompt)

    def test_git_collection_error_propagates_before_analysis_or_llm(self) -> None:
        error = GitDiffError(
            "not_git_worktree",
            "Repository root is not inside a Git worktree.",
        )
        with (
            patch(
                "agent.build_repository_context",
                return_value=RepoContext(
                    repository_root="repo",
                    target_file="app.py",
                ),
            ),
            patch("agent.collect_git_diff", side_effect=error),
            patch("agent.analyze_code") as analyze_code,
            patch("agent.ChatOpenAI") as chat_openai,
            self.assertRaises(GitDiffError) as captured,
        ):
            review_code(
                "new\n",
                repository_root="repo",
                target_file="app.py",
                git_diff=True,
            )

        self.assertIs(captured.exception, error)
        analyze_code.assert_not_called()
        chat_openai.assert_not_called()

    def test_two_retries_do_not_repeat_git_or_other_evidence_nodes(self) -> None:
        git_result = self._git_result()
        with (
            patch(
                "agent.build_repository_context",
                return_value=RepoContext(
                    repository_root="repo",
                    target_file="app.py",
                ),
            ) as repository_builder,
            patch(
                "agent.collect_git_diff",
                return_value=git_result,
            ) as git_collector,
            patch(
                "agent.parse_unified_diff",
                wraps=parse_unified_diff,
            ) as diff_parser,
            patch(
                "agent.analyze_code",
                return_value=StaticAnalysisResult(),
            ) as analyze_code,
            patch("agent.ChatOpenAI") as chat_openai,
        ):
            structured_llm = (
                chat_openai.return_value.with_structured_output.return_value
            )
            structured_llm.invoke.side_effect = [
                invalid_review(),
                invalid_review(),
                successful_review(),
            ]

            result = review_code(
                "new\n",
                repository_root="repo",
                target_file="app.py",
                git_diff=True,
            )

        self.assertEqual(result, successful_review())
        repository_builder.assert_called_once_with("repo", "app.py")
        git_collector.assert_called_once_with(
            "repo",
            "app.py",
            "working_tree",
            None,
        )
        diff_parser.assert_called_once_with(RAW_DIFF, "app.py")
        analyze_code.assert_called_once_with("new\n", "python")
        self.assertEqual(structured_llm.invoke.call_count, 3)
        retry_prompt = structured_llm.invoke.call_args_list[2].args[0][1].content
        self.assertIn("+new", retry_prompt)
        self.assertIn("previous review failed semantic validation", retry_prompt)

    @patch("agent.collect_git_diff")
    def test_api_diff_sources_are_mutually_exclusive(
        self,
        git_collector,
    ) -> None:
        combinations = (
            {"diff_text": RAW_DIFF, "git_diff": True},
            {"diff_text": RAW_DIFF, "git_base": "main"},
            {"git_diff": True, "git_base": "main"},
        )
        for arguments in combinations:
            with (
                self.subTest(arguments=arguments),
                self.assertRaisesRegex(ValueError, "mutually exclusive"),
            ):
                review_code(
                    "new\n",
                    repository_root="repo",
                    target_file="app.py",
                    **arguments,
                )
        git_collector.assert_not_called()


class GitAwareCliTests(TemporaryGitRepositoryTestCase):
    def run_cli_error(self, arguments: list[str]) -> str:
        stderr = io.StringIO()
        with (
            patch("sys.argv", ["agent.py", *arguments]),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as captured,
        ):
            main()
        self.assertEqual(captured.exception.code, 2)
        return stderr.getvalue()

    def test_manual_and_working_tree_sources_conflict(self) -> None:
        stderr = self.run_cli_error(
            [
                "--file",
                "app.py",
                "--repo",
                str(self.repository),
                "--diff",
                "changes.diff",
                "--git-diff",
            ]
        )
        self.assertIn("--git-diff", stderr)
        self.assertIn("--diff", stderr)

    def test_manual_and_base_sources_conflict(self) -> None:
        stderr = self.run_cli_error(
            [
                "--file",
                "app.py",
                "--repo",
                str(self.repository),
                "--diff",
                "changes.diff",
                "--git-base",
                "main",
            ]
        )
        self.assertIn("--git-base", stderr)
        self.assertIn("--diff", stderr)

    def test_working_tree_and_base_sources_conflict(self) -> None:
        stderr = self.run_cli_error(
            [
                "--file",
                "app.py",
                "--repo",
                str(self.repository),
                "--git-diff",
                "--git-base",
                "main",
            ]
        )
        self.assertIn("--git-base", stderr)
        self.assertIn("--git-diff", stderr)

    def test_git_mode_requires_repo(self) -> None:
        stderr = self.run_cli_error(
            [
                "--file",
                str(self.target),
                "--git-diff",
            ]
        )
        self.assertIn("require --repo", stderr)

    def test_git_mode_requires_file(self) -> None:
        stderr = self.run_cli_error(
            [
                "--code",
                "new",
                "--repo",
                str(self.repository),
                "--git-base",
                "main",
            ]
        )
        self.assertIn("require --file", stderr)

    @patch("agent.review_code")
    def test_working_tree_cli_passes_explicit_api_mode(
        self,
        mocked_review_code,
    ) -> None:
        mocked_review_code.return_value = successful_review()
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = [
            "agent.py",
            "--file",
            "app.py",
            "--repo",
            str(self.repository),
            "--git-diff",
        ]

        with (
            patch("sys.argv", argv),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        mocked_review_code.assert_called_once_with(
            "new\n",
            "python",
            repository_root=str(self.repository),
            target_file="app.py",
            git_diff=True,
        )

    @patch("agent.review_code")
    def test_base_cli_passes_local_base_ref(
        self,
        mocked_review_code,
    ) -> None:
        mocked_review_code.return_value = successful_review()
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = [
            "agent.py",
            "--file",
            "app.py",
            "--repo",
            str(self.repository),
            "--git-base",
            "main",
        ]

        with (
            patch("sys.argv", argv),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        mocked_review_code.assert_called_once_with(
            "new\n",
            "python",
            repository_root=str(self.repository),
            target_file="app.py",
            git_base="main",
        )

    @patch("agent.review_code")
    def test_git_failure_is_machine_readable_without_traceback(
        self,
        mocked_review_code,
    ) -> None:
        mocked_review_code.side_effect = GitDiffError(
            "not_git_worktree",
            "Repository root is not inside a Git worktree.",
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = [
            "agent.py",
            "--file",
            "app.py",
            "--repo",
            str(self.repository),
            "--git-diff",
        ]

        with (
            patch("sys.argv", argv),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 1)
        self.assertIn('"type": "git_diff_failed"', stdout.getvalue())
        self.assertIn('"code": "not_git_worktree"', stdout.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
