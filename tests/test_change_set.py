import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, call, patch

from pydantic import ValidationError

from agent import (
    ChangeSetReview,
    ChangeSetSummary,
    CodeReview,
    OverallRating,
    ReviewFinding,
    Severity,
    _run_single_file_review,
    main,
    review_change_set,
)
from change_set import (
    MAX_CHANGESET_DIFF_CHARS,
    MAX_CHANGESET_FILES,
    MAX_CHANGESET_PYTHON_FILES,
    ChangeTarget,
    FileReviewResult,
    build_change_targets,
    build_summary_evidence,
    validate_change_set_summary,
)
from diff_context import ChangedFile, DiffContext, parse_unified_diff
from git_diff import GitChangeSetDiffResult, collect_git_change_set_diff
from plan_execution import review_candidate_change_set_node
from repository_context import RepoContext
from review_models import FindingCategory
from static_analysis import StaticAnalysisResult
from test_execution import TestRunResult


def good_review(summary: str = "No file-level issues found.") -> CodeReview:
    return CodeReview(
        overall_rating=OverallRating.GOOD,
        summary=summary,
        findings=[],
    )


def risky_review(
    *,
    rating: OverallRating = OverallRating.NEEDS_WORK,
    severity: Severity = Severity.HIGH,
) -> CodeReview:
    return CodeReview(
        overall_rating=rating,
        summary="The file contains an evidence-backed risk.",
        findings=[
            ReviewFinding(
                category=FindingCategory.BUG,
                severity=severity,
                title="Unsafe state transition",
                description="The changed branch can leave state inconsistent.",
                line_number=1,
                suggestion="Make the transition atomic.",
            )
        ],
    )


def file_diff(path: str, old: str = "OLD = 0", new: str = "NEW = 1") -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1 +1 @@\n"
        f"-{old}\n"
        f"+{new}\n"
    )


def summary_for(
    rating: OverallRating = OverallRating.GOOD,
    high_risk_files: list[str] | None = None,
) -> ChangeSetSummary:
    return ChangeSetSummary(
        overall_rating=rating,
        summary="The reviewed Python change-set has bounded risk.",
        high_risk_files=high_risk_files or [],
    )


class ChangeSetSchemaTests(unittest.TestCase):
    def test_change_target_schema_forbids_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            ChangeTarget(
                path="app.py",
                change_kind="modified",
                unexpected=True,
            )

    def test_file_review_result_schema_and_extra_forbid(self) -> None:
        target = ChangeTarget(path="app.py", change_kind="modified")
        result = FileReviewResult(
            target=target,
            status="reviewed",
            review=good_review(),
        )
        self.assertEqual(result.review, good_review())
        with self.assertRaises(ValidationError):
            FileReviewResult(
                target=target,
                status="skipped",
                unexpected=True,
            )

    def test_change_set_review_schema_forbids_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            ChangeSetReview(
                overall_rating=OverallRating.GOOD,
                summary="Good.",
                file_results=[],
                unexpected=True,
            )


class ChangeTargetSelectionTests(unittest.TestCase):
    def test_multiple_python_files_are_selected_in_stable_order(self) -> None:
        context = parse_unified_diff(
            file_diff("z.py") + file_diff("a.py")
        )
        selection = build_change_targets(context)
        self.assertEqual([target.path for target in selection.targets], ["a.py", "z.py"])

    def test_python_and_non_python_changes_report_skipped_count(self) -> None:
        context = parse_unified_diff(
            file_diff("src/app.py") + file_diff("README.md")
        )
        selection = build_change_targets(context)
        self.assertEqual([target.path for target in selection.targets], ["src/app.py"])
        self.assertIn(
            "1 changed non-Python files were not reviewed.",
            selection.warnings,
        )

    def test_deleted_python_target_is_retained(self) -> None:
        diff = (
            "--- a/gone.py\n"
            "+++ /dev/null\n"
            "@@ -1 +0,0 @@\n"
            "-VALUE = 1\n"
        )
        target = build_change_targets(parse_unified_diff(diff)).targets[0]
        self.assertEqual(target.change_kind, "deleted")
        self.assertIsNone(target.new_path)

    def test_renamed_python_uses_new_path_and_keeps_metadata(self) -> None:
        diff = (
            "--- a/old.py\n"
            "+++ b/new.py\n"
            "@@ -1 +1 @@\n"
            "-OLD = 1\n"
            "+NEW = 1\n"
        )
        selection = build_change_targets(parse_unified_diff(diff))
        target = selection.targets[0]
        self.assertEqual(target.path, "new.py")
        self.assertEqual(target.change_kind, "renamed")
        self.assertEqual(target.old_path, "old.py")
        self.assertIn("Rename has limited support", " ".join(selection.warnings))

    def test_binary_python_is_marked_for_skip(self) -> None:
        diff = (
            "--- a/data.py\n"
            "+++ b/data.py\n"
            "Binary files a/data.py and b/data.py differ\n"
        )
        selection = build_change_targets(parse_unified_diff(diff))
        self.assertEqual(selection.binary_files, ["data.py"])

    def test_untracked_python_is_added_with_full_file_warning(self) -> None:
        selection = build_change_targets(
            DiffContext(),
            ["src/new.py", "notes.txt"],
        )
        self.assertEqual(selection.targets[0].change_kind, "added")
        self.assertIn(
            "Untracked file has no HEAD diff; reviewed as a new full-file target.",
            selection.target_warnings["src/new.py"],
        )

    def test_total_changed_file_budget_is_deterministic(self) -> None:
        files = [
            ChangedFile(old_path=f"{index:02}.py", new_path=f"{index:02}.py")
            for index in range(MAX_CHANGESET_FILES + 3)
        ]
        selection = build_change_targets(DiffContext(changed_files=files))
        self.assertEqual(len(selection.targets), MAX_CHANGESET_PYTHON_FILES)
        self.assertIn("MAX_CHANGESET_FILES", " ".join(selection.warnings))
        self.assertEqual(selection.targets[0].path, "00.py")

    def test_python_file_budget_is_reported(self) -> None:
        files = [
            ChangedFile(old_path=f"p{index}.py", new_path=f"p{index}.py")
            for index in range(MAX_CHANGESET_PYTHON_FILES + 1)
        ]
        selection = build_change_targets(DiffContext(changed_files=files))
        self.assertEqual(len(selection.targets), MAX_CHANGESET_PYTHON_FILES)
        self.assertIn("MAX_CHANGESET_PYTHON_FILES", " ".join(selection.warnings))


class ChangeSetAggregationTests(unittest.TestCase):
    def _result(self, path: str, review: CodeReview) -> FileReviewResult:
        return FileReviewResult(
            target=ChangeTarget(path=path, change_kind="modified"),
            status="reviewed",
            review=review,
        )

    def test_critical_file_prevents_good_change_set_rating(self) -> None:
        results = [
            self._result(
                "critical.py",
                risky_review(
                    rating=OverallRating.CRITICAL_ISSUES,
                    severity=Severity.CRITICAL,
                ),
            )
        ]
        errors = validate_change_set_summary(summary_for(), results)
        self.assertTrue(any("rating 'good' conflicts" in error for error in errors))

    def test_high_finding_prevents_good_change_set_rating(self) -> None:
        results = [self._result("risky.py", risky_review())]
        errors = validate_change_set_summary(summary_for(), results)
        self.assertTrue(any("rating 'good' conflicts" in error for error in errors))

    def test_all_good_files_can_produce_good_change_set(self) -> None:
        results = [
            self._result("a.py", good_review()),
            self._result("b.py", good_review()),
        ]
        self.assertEqual(validate_change_set_summary(summary_for(), results), [])

    def test_summary_evidence_contains_no_source_or_raw_diff(self) -> None:
        result = self._result("app.py", good_review("Short assessment."))
        evidence = build_summary_evidence([result], [])
        self.assertIn("Short assessment.", evidence)
        self.assertNotIn("SECRET_SOURCE_TOKEN", evidence)
        self.assertNotIn("@@ -", evidence)


class ChangeSetWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "a.py").write_text("A_SOURCE = 1\n", encoding="utf-8")
        (self.root / "b.py").write_text("B_SOURCE = 2\n", encoding="utf-8")
        self.diff = file_diff("a.py", "A_SOURCE = 0", "A_SOURCE = 1") + file_diff(
            "b.py",
            "B_SOURCE = 0",
            "B_SOURCE = 2",
        )

    def _llms(
        self,
        reviews: list[CodeReview],
        summaries: list[ChangeSetSummary],
    ):
        reviewer = Mock()
        reviewer.invoke.side_effect = reviews
        summarizer = Mock()
        summarizer.invoke.side_effect = summaries

        def structured(schema):
            return reviewer if schema is CodeReview else summarizer

        chat = Mock()
        chat.return_value.with_structured_output.side_effect = structured
        return chat, reviewer, summarizer

    def test_global_diff_produces_isolated_file_hunks_and_evidence(self) -> None:
        chat, reviewer, _summarizer = self._llms(
            [good_review("A good."), good_review("B good.")],
            [summary_for()],
        )
        contexts = []

        def context_builder(root, target):
            contexts.append((root, target))
            return RepoContext(repository_root=root, target_file=target)

        with (
            patch("agent.ChatOpenAI", chat),
            patch("agent.build_repository_context", side_effect=context_builder),
            patch(
                "agent.analyze_code",
                return_value=StaticAnalysisResult(tools_run=["ruff", "bandit"]),
            ) as analyzer,
            patch("agent.execute_targeted_tests") as tests,
        ):
            result = review_change_set(str(self.root), diff_text=self.diff)

        self.assertEqual(len(result.file_results), 2)
        first_prompt = reviewer.invoke.call_args_list[0].args[0][1].content
        second_prompt = reviewer.invoke.call_args_list[1].args[0][1].content
        self.assertIn("+A_SOURCE = 1", first_prompt)
        self.assertNotIn("+B_SOURCE = 2", first_prompt)
        self.assertIn("+B_SOURCE = 2", second_prompt)
        self.assertNotIn("+A_SOURCE = 1", second_prompt)
        self.assertEqual(
            contexts,
            [(str(self.root), "a.py"), (str(self.root), "b.py")],
        )
        self.assertEqual(
            [item.args[0] for item in analyzer.call_args_list],
            ["A_SOURCE = 1\n", "B_SOURCE = 2\n"],
        )
        tests.assert_not_called()

    def test_run_tests_true_executes_targeted_tests_per_target(self) -> None:
        chat, _reviewer, _summarizer = self._llms(
            [good_review(), good_review()],
            [summary_for()],
        )
        with (
            patch("agent.ChatOpenAI", chat),
            patch(
                "agent.build_repository_context",
                side_effect=lambda root, target: RepoContext(
                    repository_root=root,
                    target_file=target,
                ),
            ),
            patch(
                "agent.analyze_code",
                return_value=StaticAnalysisResult(tools_run=["ruff", "bandit"]),
            ),
            patch(
                "agent.execute_targeted_tests",
                return_value=TestRunResult(status="passed", framework="pytest"),
            ) as tests,
        ):
            review_change_set(str(self.root), diff_text=self.diff, run_tests=True)
        self.assertEqual(tests.call_count, 2)

    def test_file_level_semantic_retry_is_preserved(self) -> None:
        invalid = risky_review(rating=OverallRating.GOOD)
        valid = risky_review()
        chat, reviewer, _summarizer = self._llms(
            [invalid, valid],
            [summary_for(OverallRating.NEEDS_WORK, ["a.py"])],
        )
        with (
            patch("agent.ChatOpenAI", chat),
            patch(
                "agent.build_repository_context",
                return_value=RepoContext(target_file="a.py"),
            ),
            patch(
                "agent.analyze_code",
                return_value=StaticAnalysisResult(tools_run=["ruff", "bandit"]),
            ),
        ):
            result = review_change_set(
                str(self.root),
                diff_text=file_diff("a.py", "A_SOURCE = 0", "A_SOURCE = 1"),
            )
        self.assertEqual(reviewer.invoke.call_count, 2)
        self.assertEqual(result.file_results[0].review, valid)

    def test_one_file_failure_does_not_crash_safe_aggregation(self) -> None:
        with (
            patch(
                "agent._run_single_file_review",
                side_effect=[RuntimeError("target failed"), good_review()],
            ),
            patch("agent.ChatOpenAI") as chat,
        ):
            structured = chat.return_value.with_structured_output.return_value
            structured.invoke.return_value = summary_for()
            result = review_change_set(str(self.root), diff_text=self.diff)
        self.assertEqual(
            [item.status for item in result.file_results],
            ["error", "reviewed"],
        )
        self.assertIn("target failed", result.file_results[0].warnings[0])

    def test_deleted_and_binary_files_are_skipped_without_file_review(self) -> None:
        deleted = (
            "--- a/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-X = 1\n"
        )
        binary = (
            "--- a/data.py\n+++ b/data.py\n"
            "Binary files a/data.py and b/data.py differ\n"
        )
        with patch("agent._run_single_file_review") as single:
            result = review_change_set(
                str(self.root),
                diff_text=deleted + binary,
            )
        single.assert_not_called()
        self.assertEqual(
            [item.status for item in result.file_results],
            ["skipped", "skipped"],
        )

    def test_untracked_target_uses_full_file_review_without_fake_diff(self) -> None:
        target = ChangeTarget(path="new.py", change_kind="added", new_path="new.py")
        (self.root / "new.py").write_text("NEW = 1\n", encoding="utf-8")
        git_result = GitChangeSetDiffResult(
            mode="working_tree",
            repository_root=str(self.root),
            untracked_files=["new.py"],
        )
        with (
            patch("agent.collect_git_change_set_diff", return_value=git_result),
            patch("agent._run_single_file_review", return_value=good_review()) as single,
            patch("agent.ChatOpenAI") as chat,
        ):
            chat.return_value.with_structured_output.return_value.invoke.return_value = summary_for()
            result = review_change_set(str(self.root), git_diff=True)
        single.assert_called_once_with(
            str(self.root),
            target,
            None,
            run_tests=False,
            max_retries=2,
        )
        self.assertIn("Untracked file", result.file_results[0].warnings[0])

    def test_summary_prompt_does_not_resend_diff_source(self) -> None:
        secret_diff = file_diff("a.py", "OLD", "SECRET_SOURCE_TOKEN")
        with (
            patch("agent._run_single_file_review", return_value=good_review()),
            patch("agent.ChatOpenAI") as chat,
        ):
            structured = chat.return_value.with_structured_output.return_value
            structured.invoke.return_value = summary_for()
            review_change_set(str(self.root), diff_text=secret_diff)
        prompt = structured.invoke.call_args.args[0][1].content
        self.assertNotIn("SECRET_SOURCE_TOKEN", prompt)
        self.assertNotIn("@@ -", prompt)

    def test_invalid_summary_is_retried_with_validation_feedback(self) -> None:
        risky = risky_review()
        invalid = summary_for()
        valid = summary_for(OverallRating.NEEDS_WORK, ["a.py"])
        with (
            patch("agent._run_single_file_review", return_value=risky),
            patch("agent.ChatOpenAI") as chat,
        ):
            structured = chat.return_value.with_structured_output.return_value
            structured.invoke.side_effect = [invalid, valid]
            result = review_change_set(
                str(self.root),
                diff_text=file_diff("a.py"),
            )
        self.assertEqual(structured.invoke.call_count, 2)
        retry_prompt = structured.invoke.call_args_list[1].args[0][1].content
        self.assertIn("summary retry 1", retry_prompt)
        self.assertEqual(result.high_risk_files, ["a.py"])

    def test_manual_diff_budget_is_reported(self) -> None:
        oversized = "x" * (MAX_CHANGESET_DIFF_CHARS + 1)
        result = review_change_set(str(self.root), diff_text=oversized)
        self.assertIn("MAX_CHANGESET_DIFF_CHARS", " ".join(result.warnings))

    def test_auto_fix_and_apply_are_rejected_by_public_api(self) -> None:
        with self.assertRaisesRegex(ValueError, "auto_fix"):
            review_change_set(str(self.root), diff_text=self.diff, auto_fix=True)
        with self.assertRaisesRegex(ValueError, "apply_fix"):
            review_change_set(str(self.root), diff_text=self.diff, apply_fix=True)

    def test_agentic_explore_propagates_to_each_selected_target(self) -> None:
        summarizer = Mock()
        summarizer.invoke.return_value = summary_for()
        chat = Mock()
        chat.return_value.with_structured_output.return_value = summarizer
        with (
            patch("agent._run_single_file_review", return_value=good_review()) as single,
            patch("agent.ChatOpenAI", chat),
        ):
            result = review_change_set(
                str(self.root),
                diff_text=self.diff,
                agentic_explore=True,
            )
        self.assertEqual(len(result.file_results), 2)
        self.assertEqual(single.call_count, 2)
        self.assertTrue(
            all(
                item.kwargs["agentic_explore"]
                for item in single.call_args_list
            )
        )

    def test_agentic_test_propagates_to_each_selected_target(self) -> None:
        summarizer = Mock()
        summarizer.invoke.return_value = summary_for()
        chat = Mock()
        chat.return_value.with_structured_output.return_value = summarizer
        with (
            patch("agent._run_single_file_review", return_value=good_review()) as single,
            patch("agent.ChatOpenAI", chat),
        ):
            result = review_change_set(
                str(self.root),
                diff_text=self.diff,
                run_tests=True,
                agentic_explore=True,
                agentic_test=True,
            )
        self.assertEqual(len(result.file_results), 2)
        self.assertEqual(single.call_count, 2)
        self.assertTrue(
            all(item.kwargs["agentic_test"] for item in single.call_args_list)
        )

    def test_agentic_test_change_set_api_requires_both_opt_ins(self) -> None:
        with self.assertRaisesRegex(ValueError, "agentic_explore"):
            review_change_set(
                str(self.root),
                diff_text=self.diff,
                run_tests=True,
                agentic_test=True,
            )
        with self.assertRaisesRegex(ValueError, "run_tests"):
            review_change_set(
                str(self.root),
                diff_text=self.diff,
                agentic_explore=True,
                agentic_test=True,
            )

    def test_single_file_helper_forces_review_only_workflow(self) -> None:
        target = ChangeTarget(path="a.py", change_kind="modified", new_path="a.py")
        state = {"review": good_review()}
        with (
            patch("agent.read_repository_target", return_value="A_SOURCE = 1\n"),
            patch("agent._run_review_workflow", return_value=state) as workflow,
        ):
            result = _run_single_file_review(
                str(self.root),
                target,
                self.diff,
                run_tests=True,
                max_retries=1,
            )
        self.assertEqual(result, good_review())
        self.assertFalse(workflow.call_args.kwargs["auto_fix"])
        self.assertFalse(workflow.call_args.kwargs["apply_fix"])
        self.assertTrue(workflow.call_args.kwargs["run_tests"])


class ChangeSetGitIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self._git("init")
        self._git("config", "user.email", "tests@example.com")
        self._git("config", "user.name", "Tests")
        (self.root / "src").mkdir()
        (self.root / "tests").mkdir()
        (self.root / "src" / "service.py").write_text(
            "SERVICE = 1\n",
            encoding="utf-8",
        )
        (self.root / "src" / "repository.py").write_text(
            "REPOSITORY = 1\n",
            encoding="utf-8",
        )
        (self.root / "tests" / "test_service.py").write_text(
            "def test_service():\n    assert True\n",
            encoding="utf-8",
        )
        (self.root / "tests" / "test_repository.py").write_text(
            "def test_repository():\n    assert True\n",
            encoding="utf-8",
        )
        self._git("add", ".")
        self._git("commit", "-m", "initial")

    def _git(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )

    def test_working_tree_change_set_reviews_two_python_targets(self) -> None:
        (self.root / "src" / "service.py").write_text(
            "SERVICE = 2\n",
            encoding="utf-8",
        )
        (self.root / "src" / "repository.py").write_text(
            "REPOSITORY = 2\n",
            encoding="utf-8",
        )
        with (
            patch("agent._run_single_file_review", return_value=good_review()),
            patch("agent.ChatOpenAI") as chat,
        ):
            chat.return_value.with_structured_output.return_value.invoke.return_value = summary_for()
            result = review_change_set(str(self.root), git_diff=True)
        self.assertEqual(
            [item.target.path for item in result.file_results],
            ["src/repository.py", "src/service.py"],
        )
        self.assertTrue(
            all(item.review == good_review() for item in result.file_results)
        )

    def test_git_acquisition_includes_untracked_python_metadata(self) -> None:
        (self.root / "new.py").write_text("NEW = 1\n", encoding="utf-8")
        result = collect_git_change_set_diff(str(self.root), "working_tree")
        self.assertEqual(result.untracked_files, ["new.py"])
        self.assertEqual(result.mode, "working_tree")


class ChangeSetCliTests(unittest.TestCase):
    def _error(self, argv: list[str]) -> str:
        stderr = io.StringIO()
        with (
            patch("sys.argv", argv),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as captured,
        ):
            main()
        self.assertEqual(captured.exception.code, 2)
        return stderr.getvalue()

    def test_change_set_rejects_auto_fix_and_apply(self) -> None:
        auto_error = self._error(
            ["agent.py", "--review-changes", "--repo", ".", "--git-diff", "--auto-fix"]
        )
        self.assertIn("does not support --auto-fix", auto_error)
        apply_error = self._error(
            ["agent.py", "--review-changes", "--repo", ".", "--git-diff", "--apply-fix"]
        )
        self.assertIn("does not support --apply-fix", apply_error)

    def test_change_set_requires_repo_and_one_diff_source(self) -> None:
        self.assertIn(
            "requires --repo",
            self._error(["agent.py", "--review-changes", "--git-diff"]),
        )
        self.assertIn(
            "requires --diff, --git-diff, or --git-base",
            self._error(["agent.py", "--review-changes", "--repo", "."]),
        )

    @patch("agent.review_change_set")
    def test_working_tree_cli_calls_change_set_api(self, review) -> None:
        review.return_value = ChangeSetReview(
            overall_rating=OverallRating.GOOD,
            summary="Good.",
            file_results=[],
        )
        stdout = io.StringIO()
        with (
            patch("sys.argv", ["agent.py", "--review-changes", "--repo", ".", "--git-diff"]),
            redirect_stdout(stdout),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(main(), 0)
        review.assert_called_once_with(".", git_diff=True)
        self.assertEqual(json.loads(stdout.getvalue())["overall_rating"], "good")

    @patch("agent.review_change_set")
    def test_agentic_explore_cli_flag_is_forwarded(self, review) -> None:
        review.return_value = ChangeSetReview(
            overall_rating=OverallRating.GOOD,
            summary="Good.",
            file_results=[],
        )
        with (
            patch(
                "sys.argv",
                [
                    "agent.py",
                    "--review-changes",
                    "--repo",
                    ".",
                    "--git-diff",
                    "--agentic-explore",
                ],
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(main(), 0)
        review.assert_called_once_with(
            ".",
            git_diff=True,
            agentic_explore=True,
        )

    @patch("agent.review_change_set")
    def test_agentic_test_cli_flag_is_forwarded(self, review) -> None:
        review.return_value = ChangeSetReview(
            overall_rating=OverallRating.GOOD,
            summary="Good.",
            file_results=[],
        )
        with (
            patch(
                "sys.argv",
                [
                    "agent.py",
                    "--review-changes",
                    "--repo",
                    ".",
                    "--git-diff",
                    "--run-tests",
                    "--agentic-explore",
                    "--agentic-test",
                ],
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(main(), 0)
        review.assert_called_once_with(
            ".",
            git_diff=True,
            run_tests=True,
            agentic_explore=True,
            agentic_test=True,
        )

    @patch("agent.review_change_set")
    def test_base_cli_calls_change_set_api(self, review) -> None:
        review.return_value = ChangeSetReview(
            overall_rating=OverallRating.GOOD,
            summary="Good.",
            file_results=[],
        )
        with (
            patch(
                "sys.argv",
                ["agent.py", "--review-changes", "--repo", ".", "--git-base", "main"],
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(main(), 0)
        review.assert_called_once_with(".", git_base="main")

    @patch("agent.review_change_set")
    def test_manual_diff_cli_calls_change_set_api(self, review) -> None:
        review.return_value = ChangeSetReview(
            overall_rating=OverallRating.GOOD,
            summary="Good.",
            file_results=[],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            diff_path = Path(temp_dir) / "changes.diff"
            diff_path.write_text(file_diff("a.py"), encoding="utf-8")
            with (
                patch(
                    "sys.argv",
                    [
                        "agent.py",
                        "--review-changes",
                        "--repo",
                        ".",
                        "--diff",
                        str(diff_path),
                    ],
                ),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(main(), 0)
        review.assert_called_once_with(".", diff_text=file_diff("a.py"))

    def test_change_set_diff_sources_are_mutually_exclusive(self) -> None:
        error = self._error(
            [
                "agent.py",
                "--review-changes",
                "--repo",
                ".",
                "--diff",
                "changes.diff",
                "--git-diff",
            ]
        )
        self.assertIn("not allowed with argument", error)


class ChangeSetGitCommandTests(unittest.TestCase):
    @patch("git_diff._run_git")
    def test_multi_file_diff_has_no_single_target_pathspec(self, run_git) -> None:
        sha = "a" * 40
        run_git.side_effect = [
            subprocess.CompletedProcess([], 0, "true\n", ""),
            subprocess.CompletedProcess([], 0, f"{sha}\n", ""),
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, file_diff("a.py"), ""),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            collect_git_change_set_diff(temp_dir, "working_tree")
        diff_arguments = run_git.call_args_list[-1].args[1]
        self.assertEqual(diff_arguments[0], "diff")
        self.assertEqual(diff_arguments[-1], "HEAD")
        self.assertNotIn("a.py", diff_arguments)
        self.assertNotIn("--", diff_arguments)

    @patch("git_diff._run_git")
    def test_base_change_set_resolves_ref_before_three_dot_diff(self, run_git) -> None:
        head = "a" * 40
        base = "b" * 40
        run_git.side_effect = [
            subprocess.CompletedProcess([], 0, "true\n", ""),
            subprocess.CompletedProcess([], 0, f"{head}\n", ""),
            subprocess.CompletedProcess([], 0, f"{base}\n", ""),
            subprocess.CompletedProcess([], 0, file_diff("a.py"), ""),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            result = collect_git_change_set_diff(temp_dir, "base", "main")
        self.assertEqual(result.base_ref, "main")
        self.assertEqual(run_git.call_args_list[-1].args[1][-1], f"{base}...HEAD")
        self.assertNotIn(
            call(Mock(), ["fetch"]),
            run_git.call_args_list,
        )


class PlanExecutionChangeSetAdapterTests(unittest.TestCase):
    def test_candidate_adapter_calls_existing_review_with_temp_root(self) -> None:
        expected = ChangeSetReview(
            overall_rating=OverallRating.GOOD,
            summary="Candidate is consistent.",
            file_results=[],
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("agent.review_change_set", return_value=expected) as review,
        ):
            updates = review_candidate_change_set_node(
                {
                    "temporary_repository_root": temporary,
                    "diff_text": file_diff("a.py"),
                    "candidate_review_agentic_explore": False,
                }  # type: ignore[arg-type]
            )

        self.assertEqual(updates["change_set_review"], expected)
        review.assert_called_once_with(
            temporary,
            diff_text=file_diff("a.py"),
            run_tests=False,
            agentic_explore=False,
        )
