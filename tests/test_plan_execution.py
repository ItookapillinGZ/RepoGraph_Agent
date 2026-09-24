"""Deterministic Stage G2 candidate, workspace, diff, and verification tests."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from change_set import ChangeSetReview, ChangeTarget, FileReviewResult
from diff_context import parse_unified_diff
from engineering_plan import EngineeringPlan, PlannedFileChange, PlannedTest
from plan_execution import (
    DEFAULT_MAX_CORRECTION_ROUNDS,
    MAX_CANDIDATE_FILE_CHARS,
    MAX_CORRECTION_FEEDBACK_CHARS,
    CandidateAttemptSummary,
    CandidateFileChange,
    CandidateStaticResult,
    MultiFileCandidate,
    PlanExecutionResult,
    PlanExecutionVerification,
    build_candidate_correction_feedback,
    build_candidate_diff,
    build_execution_context,
    materialize_candidate,
    validate_candidate_diff,
    validate_execution_plan,
    validate_multi_file_candidate,
    verify_candidate_workspace,
)
from review_models import (
    CodeReview,
    FindingCategory,
    OverallRating,
    ReviewFinding,
    Severity,
)
from static_analysis import (
    StaticAnalysisResult,
    ToolCategory,
    ToolFinding,
    ToolSeverity,
)
from temporary_workspace import copy_repository_bounded
from test_execution import TestRunResult


class PlanExecutionFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "repo"
        self.root.mkdir()
        (self.root / "src").mkdir()
        (self.root / "tests").mkdir()
        (self.root / "src" / "service.py").write_text(
            "def lookup(email):\n    return email\n",
            encoding="utf-8",
        )
        (self.root / "src" / "delete_me.py").write_text(
            "VALUE = 1\n",
            encoding="utf-8",
        )
        (self.root / "tests" / "test_service.py").write_text(
            "def test_service():\n    assert True\n",
            encoding="utf-8",
        )
        (self.root / "unrelated.py").write_text(
            "UNRELATED_MARKER = 'must-not-leak'\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def change(self, path: str, action: str) -> PlannedFileChange:
        return PlannedFileChange(
            path=path,
            action=action,
            rationale="Required by the task.",
        )

    def plan(
        self,
        *changes: PlannedFileChange,
        tests: list[PlannedTest] | None = None,
    ) -> EngineeringPlan:
        return EngineeringPlan(
            summary="Implement the requested repository behavior.",
            files=list(changes),
            tests=tests or [],
            risks=["Behavioral compatibility."],
            assumptions=["The task is authoritative."],
        )

    def candidate(
        self,
        *changes: CandidateFileChange,
    ) -> MultiFileCandidate:
        return MultiFileCandidate(
            summary="Implemented the planned changes.",
            files=list(changes),
        )


class PlanExecutionSchemaTests(PlanExecutionFixture):
    def test_default_correction_budget_and_attempt_summary_schema(self) -> None:
        self.assertEqual(DEFAULT_MAX_CORRECTION_ROUNDS, 2)
        summary = CandidateAttemptSummary(
            attempt=1,
            diff_summary="files=1; paths=src/service.py",
            verification_status="failed",
            review_rating=OverallRating.NEEDS_WORK,
            failure_reasons=["Planned tests failed."],
        )
        self.assertEqual(summary.attempt, 1)
        with self.assertRaises(ValidationError):
            CandidateAttemptSummary(
                attempt=0,
                diff_summary="invalid",
                verification_status="failed",
            )
        with self.assertRaises(ValidationError):
            CandidateAttemptSummary(
                attempt=1,
                diff_summary="invalid",
                verification_status="failed",
                repository_source="VALUE = 1",  # type: ignore[call-arg]
            )

    def test_candidate_file_change_schema_and_extra_forbid(self) -> None:
        change = CandidateFileChange(
            path="src/service.py",
            action="modify",
            content="VALUE = 2\n",
        )
        self.assertEqual(change.action, "modify")
        with self.assertRaises(ValidationError):
            CandidateFileChange(
                path="src/service.py",
                action="modify",
                content="VALUE = 2\n",
                command="write",  # type: ignore[call-arg]
            )

    def test_multi_file_candidate_schema_and_extra_forbid(self) -> None:
        candidate = self.candidate(
            CandidateFileChange(
                path="src/service.py",
                action="modify",
                content="VALUE = 2\n",
            )
        )
        self.assertEqual(len(candidate.files), 1)
        with self.assertRaises(ValidationError):
            MultiFileCandidate(
                summary="candidate",
                files=candidate.files,
                repository_root="C:/repo",  # type: ignore[call-arg]
            )

    def test_result_and_nested_verification_forbid_extra_fields(self) -> None:
        plan = self.plan(self.change("src/service.py", "modify"))
        verification = PlanExecutionVerification(status="verified")
        result = PlanExecutionResult(
            plan=plan,
            status="verified",
            verification=verification,
        )
        self.assertEqual(result.status, "verified")
        self.assertEqual(result.correction_rounds_used, 0)
        self.assertEqual(result.attempt_history, [])
        with self.assertRaises(ValidationError):
            PlanExecutionResult(
                plan=plan,
                status="verified",
                shell_command="pytest",  # type: ignore[call-arg]
            )
        with self.assertRaises(ValidationError):
            PlanExecutionVerification(
                status="verified",
                complete_suite_passed=True,  # type: ignore[call-arg]
            )

    def test_candidate_static_result_schema(self) -> None:
        result = CandidateStaticResult(
            path="src/service.py",
            result=StaticAnalysisResult(),
        )
        self.assertEqual(result.path, "src/service.py")

    def test_correction_feedback_is_bounded_and_contains_grounded_evidence(
        self,
    ) -> None:
        static = CandidateStaticResult(
            path="src/service.py",
            result=StaticAnalysisResult(
                findings=[
                    ToolFinding(
                        tool="ruff",
                        rule_id="F821",
                        category=ToolCategory.BUG,
                        severity=ToolSeverity.HIGH,
                        message="Undefined name user_key.",
                        line_number=3,
                    )
                ]
            ),
        )
        verification = PlanExecutionVerification(
            status="failed",
            static_analysis=[static],
            test_result=TestRunResult(
                status="failed",
                stdout="x" * MAX_CORRECTION_FEEDBACK_CHARS,
                stderr="assertion failed",
            ),
        )
        review = ChangeSetReview(
            overall_rating=OverallRating.NEEDS_WORK,
            summary="The implementation is incomplete.",
            file_results=[
                FileReviewResult(
                    target=ChangeTarget(
                        path="src/service.py",
                        change_kind="modified",
                    ),
                    status="reviewed",
                    review=CodeReview(
                        overall_rating=OverallRating.NEEDS_WORK,
                        summary="Normalize the lookup key.",
                        findings=[
                            ReviewFinding(
                                category=FindingCategory.BUG,
                                severity=Severity.HIGH,
                                title="Lookup remains case-sensitive",
                                description="Mixed-case email values miss.",
                                line_number=3,
                                suggestion="Casefold the lookup key.",
                            )
                        ],
                    ),
                )
            ],
        )

        feedback, warnings = build_candidate_correction_feedback(
            verification,
            review,
        )

        self.assertLessEqual(len(feedback), MAX_CORRECTION_FEEDBACK_CHARS)
        self.assertIn("ruff:F821", feedback)
        self.assertIn("Lookup remains case-sensitive", feedback)
        self.assertTrue(any("truncated" in warning for warning in warnings))

    def test_previous_attempt_history_is_bounded(self) -> None:
        history = [
            CandidateAttemptSummary(
                attempt=attempt,
                diff_summary=f"candidate {attempt}",
                verification_status="failed",
            )
            for attempt in range(1, 13)
        ]
        feedback, _ = build_candidate_correction_feedback(
            PlanExecutionVerification(status="failed"),
            None,
            history,
        )

        self.assertNotIn('"attempt":2,', feedback)
        self.assertIn('"attempt":3,', feedback)
        self.assertIn('"attempt":12,', feedback)


class CandidateValidationTests(PlanExecutionFixture):
    def test_valid_modify_add_and_delete_candidate(self) -> None:
        plan = self.plan(
            self.change("src/service.py", "modify"),
            self.change("src/new_module.py", "add"),
            self.change("src/delete_me.py", "delete"),
        )
        candidate = self.candidate(
            CandidateFileChange(
                path="src/service.py",
                action="modify",
                content="def lookup(email):\n    return email.casefold()\n",
            ),
            CandidateFileChange(
                path="src/new_module.py",
                action="add",
                content="VALUE = 1\n",
            ),
            CandidateFileChange(
                path="src/delete_me.py",
                action="delete",
            ),
        )

        self.assertEqual(
            validate_multi_file_candidate(candidate, plan, str(self.root)),
            [],
        )

    def test_extra_candidate_file_is_rejected(self) -> None:
        plan = self.plan(self.change("src/service.py", "modify"))
        candidate = self.candidate(
            CandidateFileChange(
                path="src/service.py",
                action="modify",
                content="def lookup(email):\n    return email.casefold()\n",
            ),
            CandidateFileChange(
                path="src/new_module.py",
                action="add",
                content="VALUE = 1\n",
            ),
        )

        errors = validate_multi_file_candidate(candidate, plan, str(self.root))

        self.assertTrue(any("unplanned file" in error for error in errors))

    def test_missing_candidate_file_is_rejected(self) -> None:
        plan = self.plan(
            self.change("src/service.py", "modify"),
            self.change("src/new_module.py", "add"),
        )
        candidate = self.candidate(
            CandidateFileChange(
                path="src/service.py",
                action="modify",
                content="def lookup(email):\n    return email.casefold()\n",
            )
        )

        errors = validate_multi_file_candidate(candidate, plan, str(self.root))

        self.assertTrue(any("missing planned file" in error for error in errors))

    def test_action_mismatch_is_rejected(self) -> None:
        plan = self.plan(self.change("src/service.py", "modify"))
        candidate = self.candidate(
            CandidateFileChange(
                path="src/service.py",
                action="delete",
            )
        )

        errors = validate_multi_file_candidate(candidate, plan, str(self.root))

        self.assertTrue(any("action mismatch" in error for error in errors))

    def test_duplicate_normalized_candidate_path_is_rejected(self) -> None:
        plan = self.plan(self.change("src/service.py", "modify"))
        candidate = self.candidate(
            CandidateFileChange(
                path="src/service.py",
                action="modify",
                content="VALUE = 1\n",
            ),
            CandidateFileChange(
                path="src\\service.py",
                action="modify",
                content="VALUE = 2\n",
            ),
        )

        errors = validate_multi_file_candidate(candidate, plan, str(self.root))

        self.assertTrue(any("Duplicate candidate path" in error for error in errors))

    def test_modify_and_add_require_content(self) -> None:
        for action, path in (
            ("modify", "src/service.py"),
            ("add", "src/new_module.py"),
        ):
            with self.subTest(action=action):
                plan = self.plan(self.change(path, action))
                candidate = self.candidate(
                    CandidateFileChange(path=path, action=action)
                )
                errors = validate_multi_file_candidate(
                    candidate,
                    plan,
                    str(self.root),
                )
                self.assertTrue(any("requires complete content" in e for e in errors))

    def test_delete_requires_none_content(self) -> None:
        plan = self.plan(self.change("src/delete_me.py", "delete"))
        candidate = self.candidate(
            CandidateFileChange(
                path="src/delete_me.py",
                action="delete",
                content="",
            )
        )

        errors = validate_multi_file_candidate(candidate, plan, str(self.root))

        self.assertTrue(any("delete content must be None" in e for e in errors))

    def test_unchanged_modify_is_rejected(self) -> None:
        plan = self.plan(self.change("src/service.py", "modify"))
        original = (self.root / "src" / "service.py").read_text(encoding="utf-8")
        candidate = self.candidate(
            CandidateFileChange(
                path="src/service.py",
                action="modify",
                content=original,
            )
        )

        errors = validate_multi_file_candidate(candidate, plan, str(self.root))

        self.assertTrue(any("unchanged" in error for error in errors))

    def test_invalid_python_and_markdown_fences_are_rejected(self) -> None:
        plan = self.plan(self.change("src/service.py", "modify"))
        for content, expected in (
            ("def broken(:\n", "could not be parsed"),
            ("\x60\x60\x60python\nVALUE = 1\n\x60\x60\x60", "Markdown"),
        ):
            with self.subTest(expected=expected):
                candidate = self.candidate(
                    CandidateFileChange(
                        path="src/service.py",
                        action="modify",
                        content=content,
                    )
                )
                errors = validate_multi_file_candidate(
                    candidate,
                    plan,
                    str(self.root),
                )
                self.assertTrue(any(expected in error for error in errors))

    def test_unsafe_paths_are_rejected(self) -> None:
        for path in (
            "../service.py",
            str(self.root / "src" / "service.py"),
            ".git/config.py",
            ".env.py",
            "keys/private.pem",
        ):
            with self.subTest(path=path):
                plan = self.plan(self.change(path, "add"))
                candidate = self.candidate(
                    CandidateFileChange(
                        path=path,
                        action="add",
                        content="VALUE = 1\n",
                    )
                )
                errors = validate_multi_file_candidate(
                    candidate,
                    plan,
                    str(self.root),
                )
                self.assertTrue(errors)

    def test_symlink_candidate_path_is_rejected(self) -> None:
        outside = self.root.parent / "outside"
        outside.mkdir()
        link = self.root / "linked"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Symlink creation unavailable: {error}")
        plan = self.plan(self.change("linked/new.py", "add"))
        candidate = self.candidate(
            CandidateFileChange(
                path="linked/new.py",
                action="add",
                content="VALUE = 1\n",
            )
        )

        errors = validate_multi_file_candidate(candidate, plan, str(self.root))

        self.assertTrue(any("Symbolic links" in error for error in errors))

    def test_per_file_and_total_candidate_budgets(self) -> None:
        plan = self.plan(self.change("src/service.py", "modify"))
        oversized = "#" + ("x" * MAX_CANDIDATE_FILE_CHARS)
        candidate = self.candidate(
            CandidateFileChange(
                path="src/service.py",
                action="modify",
                content=oversized,
            )
        )
        errors = validate_multi_file_candidate(candidate, plan, str(self.root))
        self.assertTrue(any("MAX_CANDIDATE_FILE_CHARS" in error for error in errors))

        with patch("plan_execution.MAX_TOTAL_CANDIDATE_CHARS", 5):
            errors = validate_multi_file_candidate(
                self.candidate(
                    CandidateFileChange(
                        path="src/service.py",
                        action="modify",
                        content="VALUE = 2\n",
                    )
                ),
                plan,
                str(self.root),
            )
        self.assertTrue(any("MAX_TOTAL_CANDIDATE_CHARS" in e for e in errors))

    def test_execution_file_and_python_only_budgets(self) -> None:
        changes: list[PlannedFileChange] = []
        for index in range(6):
            path = self.root / "src" / f"module_{index}.py"
            path.write_text("VALUE = 1\n", encoding="utf-8")
            changes.append(self.change(f"src/module_{index}.py", "modify"))
        errors = validate_execution_plan(self.plan(*changes), str(self.root))
        self.assertTrue(any("MAX_EXECUTION_FILES" in error for error in errors))

        (self.root / "README.md").write_text("docs", encoding="utf-8")
        errors = validate_execution_plan(
            self.plan(self.change("README.md", "modify")),
            str(self.root),
        )
        self.assertTrue(any("only Python" in error for error in errors))


class ExecutionContextTests(PlanExecutionFixture):
    def test_context_contains_planned_source_but_not_arbitrary_source(self) -> None:
        task = "Make lookup case-insensitive."
        plan = self.plan(self.change("src/service.py", "modify"))

        context, warnings, errors = build_execution_context(
            str(self.root),
            task,
            plan,
        )

        self.assertEqual(errors, [])
        self.assertIn("def lookup(email)", context)
        self.assertIn("src/service.py", context)
        self.assertNotIn("must-not-leak", context)
        self.assertIsInstance(warnings, list)

    def test_context_is_bounded_with_an_explicit_warning(self) -> None:
        plan = self.plan(self.change("src/service.py", "modify"))

        with patch("plan_execution.MAX_EXECUTION_CONTEXT_CHARS", 80):
            context, warnings, errors = build_execution_context(
                str(self.root),
                "Update lookup.",
                plan,
            )

        self.assertEqual(errors, [])
        self.assertEqual(len(context), 80)
        self.assertTrue(any("truncated" in warning for warning in warnings))

    def test_invalid_execution_plan_returns_context_failure(self) -> None:
        (self.root / "README.md").write_text("docs", encoding="utf-8")
        plan = self.plan(self.change("README.md", "modify"))

        context, _, errors = build_execution_context(
            str(self.root),
            "Update docs.",
            plan,
        )

        self.assertEqual(context, "")
        self.assertTrue(errors)


class CandidateWorkspaceAndDiffTests(PlanExecutionFixture):
    def candidate_and_plan(
        self,
    ) -> tuple[MultiFileCandidate, EngineeringPlan]:
        plan = self.plan(
            self.change("src/service.py", "modify"),
            self.change("src/new_module.py", "add"),
            self.change("src/delete_me.py", "delete"),
        )
        candidate = self.candidate(
            CandidateFileChange(
                path="src/service.py",
                action="modify",
                content="def lookup(email):\n    return email.casefold()\n",
            ),
            CandidateFileChange(
                path="src/new_module.py",
                action="add",
                content="VALUE = 2\n",
            ),
            CandidateFileChange(
                path="src/delete_me.py",
                action="delete",
            ),
        )
        return candidate, plan

    def test_modify_add_delete_materialize_only_in_temp(self) -> None:
        candidate, _ = self.candidate_and_plan()
        original_service = (self.root / "src" / "service.py").read_bytes()
        original_delete = (self.root / "src" / "delete_me.py").read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary) / "repository"
            copy_repository_bounded(self.root, temporary_root)

            materialize_candidate(str(temporary_root), candidate)

            self.assertIn(
                "casefold",
                (temporary_root / "src" / "service.py").read_text(
                    encoding="utf-8"
                ),
            )
            self.assertTrue((temporary_root / "src" / "new_module.py").is_file())
            self.assertFalse((temporary_root / "src" / "delete_me.py").exists())

        self.assertEqual(
            (self.root / "src" / "service.py").read_bytes(),
            original_service,
        )
        self.assertFalse((self.root / "src" / "new_module.py").exists())
        self.assertEqual(
            (self.root / "src" / "delete_me.py").read_bytes(),
            original_delete,
        )

    def test_multi_file_diff_is_parseable_for_all_actions(self) -> None:
        candidate, plan = self.candidate_and_plan()
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary) / "repository"
            copy_repository_bounded(self.root, temporary_root)
            materialize_candidate(str(temporary_root), candidate)

            diff_text = build_candidate_diff(
                str(self.root),
                str(temporary_root),
                candidate,
                plan,
            )

        parsed = parse_unified_diff(diff_text)
        by_path = {
            changed.new_path or changed.old_path: (
                changed.old_path,
                changed.new_path,
            )
            for changed in parsed.changed_files
        }
        self.assertEqual(
            set(by_path),
            {
                "src/service.py",
                "src/new_module.py",
                "src/delete_me.py",
            },
        )
        self.assertEqual(
            by_path["src/new_module.py"],
            (None, "src/new_module.py"),
        )
        self.assertEqual(
            by_path["src/delete_me.py"],
            ("src/delete_me.py", None),
        )
        self.assertEqual(validate_candidate_diff(diff_text, candidate, plan), [])

    def test_diff_changed_file_mismatch_is_rejected(self) -> None:
        candidate, plan = self.candidate_and_plan()
        wrong_diff = (
            "--- a/src/service.py\n"
            "+++ b/src/service.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-VALUE = 1\n"
            "+VALUE = 2\n"
        )

        errors = validate_candidate_diff(wrong_diff, candidate, plan)

        self.assertTrue(any("changed-file set" in error for error in errors))

    def test_diff_budget_failure_is_controlled(self) -> None:
        plan = self.plan(self.change("src/service.py", "modify"))
        candidate = self.candidate(
            CandidateFileChange(
                path="src/service.py",
                action="modify",
                content="\n".join(f"VALUE_{index} = {index}" for index in range(40)),
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary) / "repository"
            copy_repository_bounded(self.root, temporary_root)
            materialize_candidate(str(temporary_root), candidate)
            with (
                patch("plan_execution.MAX_CANDIDATE_DIFF_CHARS", 40),
                self.assertRaises(RuntimeError),
            ):
                build_candidate_diff(
                    str(self.root),
                    str(temporary_root),
                    candidate,
                    plan,
                )


class CandidateVerificationTests(PlanExecutionFixture):
    def verification_candidate(self) -> MultiFileCandidate:
        return self.candidate(
            CandidateFileChange(
                path="src/service.py",
                action="modify",
                content="def lookup(email):\n    return email.casefold()\n",
            )
        )

    def temporary_candidate(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        temporary = tempfile.TemporaryDirectory()
        temporary_root = Path(temporary.name) / "repository"
        copy_repository_bounded(self.root, temporary_root)
        materialize_candidate(
            str(temporary_root),
            self.verification_candidate(),
        )
        return temporary, temporary_root

    def test_no_explicit_test_never_falls_back_to_full_suite(self) -> None:
        plan = self.plan(self.change("src/service.py", "modify"))
        temporary, temporary_root = self.temporary_candidate()
        self.addCleanup(temporary.cleanup)

        with (
            patch(
                "plan_execution.analyze_code",
                return_value=StaticAnalysisResult(),
            ),
            patch("plan_execution.execute_test_files") as execute,
        ):
            verification = verify_candidate_workspace(
                str(temporary_root),
                self.verification_candidate(),
                plan,
            )

        execute.assert_not_called()
        self.assertEqual(verification.status, "verified")
        self.assertEqual(verification.test_result.status, "not_run")
        self.assertTrue(
            any("No explicit executable" in warning for warning in verification.warnings)
        )

    def test_explicit_planned_test_executes_once(self) -> None:
        plan = self.plan(
            self.change("src/service.py", "modify"),
            tests=[
                PlannedTest(
                    path="tests/test_service.py",
                    purpose="Verify lookup.",
                )
            ],
        )
        temporary, temporary_root = self.temporary_candidate()
        self.addCleanup(temporary.cleanup)
        passed = TestRunResult(
            status="passed",
            framework="pytest",
            test_files=["tests/test_service.py"],
            exit_code=0,
        )
        with (
            patch(
                "plan_execution.analyze_code",
                return_value=StaticAnalysisResult(),
            ),
            patch(
                "plan_execution.execute_test_files",
                return_value=passed,
            ) as execute,
        ):
            verification = verify_candidate_workspace(
                str(temporary_root),
                self.verification_candidate(),
                plan,
            )

        execute.assert_called_once_with(
            str(temporary_root.resolve()),
            ["tests/test_service.py"],
        )
        self.assertEqual(verification.status, "verified")

    def test_failed_and_timed_out_planned_tests_have_distinct_statuses(self) -> None:
        plan = self.plan(
            self.change("src/service.py", "modify"),
            tests=[
                PlannedTest(
                    path="tests/test_service.py",
                    purpose="Verify lookup.",
                )
            ],
        )
        for test_status, expected in (("failed", "failed"), ("timed_out", "error")):
            with self.subTest(test_status=test_status):
                temporary, temporary_root = self.temporary_candidate()
                self.addCleanup(temporary.cleanup)
                with (
                    patch(
                        "plan_execution.analyze_code",
                        return_value=StaticAnalysisResult(),
                    ),
                    patch(
                        "plan_execution.execute_test_files",
                        return_value=TestRunResult(status=test_status),
                    ),
                ):
                    verification = verify_candidate_workspace(
                        str(temporary_root),
                        self.verification_candidate(),
                        plan,
                    )
                self.assertEqual(verification.status, expected)

    def test_blocking_static_finding_fails_verification(self) -> None:
        plan = self.plan(self.change("src/service.py", "modify"))
        temporary, temporary_root = self.temporary_candidate()
        self.addCleanup(temporary.cleanup)
        finding = ToolFinding(
            tool="bandit",
            rule_id="B999",
            category=ToolCategory.SECURITY,
            severity=ToolSeverity.HIGH,
            message="high risk",
        )
        with patch(
            "plan_execution.analyze_code",
            return_value=StaticAnalysisResult(findings=[finding]),
        ):
            verification = verify_candidate_workspace(
                str(temporary_root),
                self.verification_candidate(),
                plan,
            )

        self.assertEqual(verification.status, "failed")

    def test_static_tool_error_is_verification_error(self) -> None:
        plan = self.plan(self.change("src/service.py", "modify"))
        temporary, temporary_root = self.temporary_candidate()
        self.addCleanup(temporary.cleanup)
        with patch(
            "plan_execution.analyze_code",
            return_value=StaticAnalysisResult(tool_errors=["ruff unavailable"]),
        ):
            verification = verify_candidate_workspace(
                str(temporary_root),
                self.verification_candidate(),
                plan,
            )

        self.assertEqual(verification.status, "error")
        self.assertTrue(any("ruff unavailable" in w for w in verification.warnings))

    def test_all_added_and_modified_files_receive_static_analysis(self) -> None:
        plan = self.plan(
            self.change("src/service.py", "modify"),
            self.change("src/new_module.py", "add"),
        )
        candidate = self.candidate(
            CandidateFileChange(
                path="src/service.py",
                action="modify",
                content="VALUE = 2\n",
            ),
            CandidateFileChange(
                path="src/new_module.py",
                action="add",
                content="VALUE = 3\n",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary) / "repository"
            copy_repository_bounded(self.root, temporary_root)
            materialize_candidate(str(temporary_root), candidate)
            with patch(
                "plan_execution.analyze_code",
                return_value=StaticAnalysisResult(),
            ) as analyze:
                verification = verify_candidate_workspace(
                    str(temporary_root),
                    candidate,
                    plan,
                )

        self.assertEqual(analyze.call_count, 2)
        self.assertEqual(
            [item.path for item in verification.static_analysis],
            ["src/new_module.py", "src/service.py"],
        )

    def test_planned_test_budget_warning_is_explicit(self) -> None:
        tests: list[PlannedTest] = []
        for index in range(4):
            path = f"tests/test_{index}.py"
            (self.root / path).write_text(
                "def test_ok():\n    assert True\n",
                encoding="utf-8",
            )
            tests.append(PlannedTest(path=path, purpose="Verify behavior."))
        plan = self.plan(self.change("src/service.py", "modify"), tests=tests)
        temporary, temporary_root = self.temporary_candidate()
        self.addCleanup(temporary.cleanup)
        with (
            patch(
                "plan_execution.analyze_code",
                return_value=StaticAnalysisResult(),
            ),
            patch(
                "plan_execution.execute_test_files",
                return_value=TestRunResult(status="passed"),
            ),
        ):
            verification = verify_candidate_workspace(
                str(temporary_root),
                self.verification_candidate(),
                plan,
            )

        self.assertTrue(any("MAX_TEST_FILES" in w for w in verification.warnings))


if __name__ == "__main__":
    unittest.main()
