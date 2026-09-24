import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from agent import (
    CodeReview,
    EngineeringPlan,
    EngineeringPlanValidationError,
    FindingCategory,
    OverallRating,
    PlanExecutionResult,
    ReviewAndFixResult,
    ReviewFinding,
    ReviewSemanticValidationError,
    ReviewState,
    Severity,
    main,
    render_review,
    render_review_and_fix,
    review_and_fix,
    review_code,
    route_after_validation,
    validate_semantics,
)
from diff_context import DiffContext
from engineering_plan import PlannedFileChange, PlannedTest
from fix_application import ApplyFixResult
from fix_context import CodeFix
from fix_verification import FixVerificationResult
from git_delivery import LocalGitDeliveryResult
from git_diff import GitDiffResult
from github_delivery import GitHubRemoteDeliveryResult
from plan_application import PlanApplicationResult
from repository_context import RepoContext
from static_analysis import (
    StaticAnalysisResult,
    ToolCategory,
    ToolFinding,
    ToolSeverity,
)
from test_execution import TestRunResult


def make_finding(
    category: FindingCategory = FindingCategory.BUG,
    severity: Severity = Severity.HIGH,
    line_number: int | None = 1,
    title: str = "Zero divisor is not handled",
) -> ReviewFinding:
    return ReviewFinding(
        category=category,
        severity=severity,
        title=title,
        description="Dividing by zero raises ZeroDivisionError.",
        line_number=line_number,
        suggestion="Validate the divisor before division.",
    )


def make_review(
    overall_rating: OverallRating = OverallRating.NEEDS_WORK,
    findings: list[ReviewFinding] | None = None,
) -> CodeReview:
    return CodeReview(
        overall_rating=overall_rating,
        summary="The function does not handle a zero divisor.",
        findings=[make_finding()] if findings is None else findings,
    )


def make_state(
    review: CodeReview | None,
    code: str = "def divide(a, b):\n    return a / b",
    retry_count: int = 0,
    max_retries: int = 2,
    static_analysis: StaticAnalysisResult | None = None,
) -> ReviewState:
    return {
        "code": code,
        "language": "python",
        "repository_root": None,
        "target_file": None,
        "repository_context": RepoContext(),
        "diff_text": None,
        "diff_context": DiffContext(),
        "git_diff_mode": None,
        "git_base_ref": None,
        "git_diff_result": None,
        "static_analysis": static_analysis or StaticAnalysisResult(),
        "review": review,
        "validation_errors": [],
        "retry_count": retry_count,
        "max_retries": max_retries,
        "failure_reason": None,
    }


def make_tool_finding(
    tool: str = "bandit",
    rule_id: str = "B602",
    category: ToolCategory = ToolCategory.SECURITY,
    severity: ToolSeverity = ToolSeverity.HIGH,
    line_number: int = 1,
) -> ToolFinding:
    return ToolFinding(
        tool=tool,
        rule_id=rule_id,
        category=category,
        severity=severity,
        confidence="high" if tool == "bandit" else None,
        message="Unsafe subprocess call with shell=True.",
        line_number=line_number,
        column=1,
    )


class AgentTestCase(unittest.TestCase):
    def setUp(self):
        self.static_analysis_patcher = patch(
            "agent.analyze_code",
            return_value=StaticAnalysisResult(tools_run=["ruff", "bandit"]),
        )
        self.mocked_analyze_code = self.static_analysis_patcher.start()
        self.addCleanup(self.static_analysis_patcher.stop)


class StructuredOutputTests(AgentTestCase):
    def test_schema_rejects_unknown_fields(self):
        payload = make_review().model_dump()
        payload["unexpected"] = "not allowed"

        with self.assertRaises(ValidationError):
            CodeReview.model_validate(payload)

    @patch("agent.ChatOpenAI")
    @patch.dict("os.environ", {}, clear=True)
    def test_review_code_requests_and_returns_code_review(self, chat_openai):
        expected = make_review()
        structured_llm = chat_openai.return_value.with_structured_output.return_value
        structured_llm.invoke.return_value = expected

        result = review_code("def divide(a, b): return a / b")

        chat_openai.assert_called_once_with(model="gpt-5.6-luna", temperature=0)
        chat_openai.return_value.with_structured_output.assert_called_once_with(
            CodeReview
        )
        self.assertEqual(result, expected)
        messages = structured_llm.invoke.call_args.args[0]
        self.assertIn("def divide(a, b)", messages[1].content)

    def test_render_review_is_deterministic(self):
        rendered = render_review(make_review())

        self.assertIn("Overall: NEEDS WORK", rendered)
        self.assertIn("[HIGH] bug: Zero divisor is not handled (line 1)", rendered)
        self.assertIn("Suggestion: Validate the divisor before division.", rendered)

    @patch("agent.review_code")
    def test_json_cli_writes_machine_readable_stdout(self, mocked_review_code):
        mocked_review_code.return_value = make_review()
        stdout = io.StringIO()
        stderr = io.StringIO()

        argv = ["agent.py", "--code", "return 42"]
        with patch("sys.argv", argv), redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main()

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["overall_rating"], "needs_work")
        self.assertIn("Reviewing: inline code snippet", stderr.getvalue())

    @patch("agent.review_code")
    def test_text_cli_still_renders_review(self, mocked_review_code):
        mocked_review_code.return_value = make_review()
        stdout = io.StringIO()
        stderr = io.StringIO()

        argv = ["agent.py", "--code", "return 42", "--format", "text"]
        with patch("sys.argv", argv), redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertIn("CODE REVIEW", stdout.getvalue())
        self.assertIn("Overall: NEEDS WORK", stdout.getvalue())


class SemanticValidationTests(AgentTestCase):
    def test_semantic_validation_success_routes_to_end(self):
        state = make_state(make_review())

        state.update(validate_semantics(state))

        self.assertEqual(state["validation_errors"], [])
        self.assertEqual(route_after_validation(state), "valid")

    def test_good_rating_with_high_finding_is_invalid(self):
        review = make_review(overall_rating=OverallRating.GOOD)

        result = validate_semantics(make_state(review))

        self.assertTrue(
            any("conflicts with high" in error for error in result["validation_errors"])
        )

    def test_critical_finding_requires_critical_rating(self):
        review = make_review(findings=[make_finding(severity=Severity.CRITICAL)])

        result = validate_semantics(make_state(review))

        self.assertTrue(
            any(
                "require overall rating 'critical_issues'" in error
                for error in result["validation_errors"]
            )
        )

    def test_line_number_outside_source_is_invalid(self):
        review = make_review(findings=[make_finding(line_number=10)])

        result = validate_semantics(make_state(review, code="first\nsecond"))

        self.assertTrue(
            any("source has 2 lines" in error for error in result["validation_errors"])
        )

    def test_duplicate_finding_is_invalid(self):
        findings = [
            make_finding(title="Zero divisor is not handled"),
            make_finding(title="  ZERO DIVISOR is not handled  "),
        ]
        review = make_review(findings=findings)

        result = validate_semantics(make_state(review))

        self.assertTrue(
            any("Duplicate finding" in error for error in result["validation_errors"])
        )

    def test_critical_rating_without_findings_is_invalid(self):
        review = make_review(
            overall_rating=OverallRating.CRITICAL_ISSUES,
            findings=[],
        )

        result = validate_semantics(make_state(review))

        self.assertTrue(
            any(
                "requires at least one finding" in error
                for error in result["validation_errors"]
            )
        )


class RetryWorkflowTests(AgentTestCase):
    @patch("agent.ChatOpenAI")
    def test_retry_loop_succeeds_on_second_review(self, chat_openai):
        invalid = make_review(overall_rating=OverallRating.GOOD)
        valid = make_review()
        structured_llm = chat_openai.return_value.with_structured_output.return_value
        structured_llm.invoke.side_effect = [invalid, valid]

        result = review_code("def divide(a, b): return a / b")

        self.assertEqual(result, valid)
        self.assertEqual(structured_llm.invoke.call_count, 2)
        self.assertEqual(chat_openai.call_count, 2)

    @patch("agent.ChatOpenAI")
    def test_retry_prompt_contains_validation_feedback(self, chat_openai):
        invalid = make_review(overall_rating=OverallRating.GOOD)
        valid = make_review()
        structured_llm = chat_openai.return_value.with_structured_output.return_value
        structured_llm.invoke.side_effect = [invalid, valid]

        review_code("def divide(a, b): return a / b")

        retry_messages = structured_llm.invoke.call_args_list[1].args[0]
        retry_prompt = retry_messages[1].content
        self.assertIn("previous review failed semantic validation", retry_prompt)
        self.assertIn("conflicts with high or critical", retry_prompt)
        self.assertIn("semantic retry 1", retry_prompt)

    @patch("agent.ChatOpenAI")
    def test_max_retries_enters_controlled_failure(self, chat_openai):
        invalid = make_review(overall_rating=OverallRating.GOOD)
        structured_llm = chat_openai.return_value.with_structured_output.return_value
        structured_llm.invoke.return_value = invalid

        with self.assertRaises(ReviewSemanticValidationError) as captured:
            review_code("return 42", max_retries=2)

        self.assertEqual(structured_llm.invoke.call_count, 3)
        self.assertEqual(captured.exception.retry_count, 2)
        self.assertEqual(
            str(captured.exception),
            "Review failed semantic validation after 2 retries.",
        )
        payload = captured.exception.to_payload()
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(
            payload["error"]["type"],
            "semantic_validation_failed",
        )


class EvidenceWorkflowTests(AgentTestCase):
    def _bandit_analysis(self) -> StaticAnalysisResult:
        return StaticAnalysisResult(
            findings=[make_tool_finding()],
            tools_run=["ruff", "bandit"],
        )

    @patch("agent.ChatOpenAI")
    def test_llm_prompt_contains_static_tool_evidence(self, chat_openai):
        self.mocked_analyze_code.return_value = self._bandit_analysis()
        review = make_review(
            findings=[
                make_finding(
                    category=FindingCategory.SECURITY,
                    line_number=1,
                )
            ]
        )
        structured_llm = chat_openai.return_value.with_structured_output.return_value
        structured_llm.invoke.return_value = review

        review_code("subprocess.run(command, shell=True)")

        prompt = structured_llm.invoke.call_args.args[0][1].content
        self.assertIn("Deterministic static-analysis evidence", prompt)
        self.assertIn("B602", prompt)
        self.assertIn("Unsafe subprocess call with shell=True", prompt)

    @patch("agent.ChatOpenAI")
    def test_retry_prompt_keeps_evidence_and_validation_feedback(
        self,
        chat_openai,
    ):
        self.mocked_analyze_code.return_value = self._bandit_analysis()
        ignored = make_review(
            overall_rating=OverallRating.GOOD,
            findings=[],
        )
        covered = make_review(
            findings=[
                make_finding(
                    category=FindingCategory.SECURITY,
                    line_number=1,
                )
            ]
        )
        structured_llm = chat_openai.return_value.with_structured_output.return_value
        structured_llm.invoke.side_effect = [ignored, covered]

        review_code("subprocess.run(command, shell=True)")

        retry_prompt = structured_llm.invoke.call_args_list[1].args[0][1].content
        self.assertIn("B602", retry_prompt)
        self.assertIn("previous review failed semantic validation", retry_prompt)
        self.assertIn("uncovered high-severity Bandit evidence", retry_prompt)

    @patch("agent.ChatOpenAI")
    def test_high_bandit_finding_ignored_enters_retry(self, chat_openai):
        self.mocked_analyze_code.return_value = self._bandit_analysis()
        ignored = make_review(
            overall_rating=OverallRating.GOOD,
            findings=[],
        )
        covered = make_review(
            findings=[
                make_finding(
                    category=FindingCategory.SECURITY,
                    line_number=1,
                )
            ]
        )
        structured_llm = chat_openai.return_value.with_structured_output.return_value
        structured_llm.invoke.side_effect = [ignored, covered]

        result = review_code("subprocess.run(command, shell=True)")

        self.assertEqual(result, covered)
        self.assertEqual(structured_llm.invoke.call_count, 2)

    def test_matching_security_finding_covers_bandit_evidence(self):
        review = make_review(
            findings=[
                make_finding(
                    category=FindingCategory.SECURITY,
                    line_number=2,
                )
            ]
        )
        state = make_state(
            review,
            code="first\nsubprocess.run(command, shell=True)\nthird",
            static_analysis=StaticAnalysisResult(
                findings=[make_tool_finding(line_number=2)],
                tools_run=["ruff", "bandit"],
            ),
        )

        result = validate_semantics(state)

        self.assertEqual(result["validation_errors"], [])

    @patch("agent.ChatOpenAI")
    def test_static_analysis_runs_once_during_two_retries(self, chat_openai):
        invalid = make_review(overall_rating=OverallRating.GOOD)
        valid = make_review()
        structured_llm = chat_openai.return_value.with_structured_output.return_value
        structured_llm.invoke.side_effect = [invalid, invalid, valid]

        result = review_code("def divide(a, b): return a / b")

        self.assertEqual(result, valid)
        self.assertEqual(structured_llm.invoke.call_count, 3)
        self.mocked_analyze_code.assert_called_once_with(
            "def divide(a, b): return a / b",
            "python",
        )

    def test_ruff_correctness_evidence_requires_bug_coverage(self):
        analysis = StaticAnalysisResult(
            findings=[
                make_tool_finding(
                    tool="ruff",
                    rule_id="F821",
                    category=ToolCategory.BUG,
                    line_number=2,
                )
            ],
            tools_run=["ruff", "bandit"],
        )
        review = make_review(findings=[])

        result = validate_semantics(
            make_state(
                review, code="first\nprint(missing_name)", static_analysis=analysis
            )
        )

        self.assertTrue(
            any("ruff F821" in error for error in result["validation_errors"])
        )

    def test_ruff_style_evidence_does_not_require_review_coverage(self):
        analysis = StaticAnalysisResult(
            findings=[
                make_tool_finding(
                    tool="ruff",
                    rule_id="F401",
                    category=ToolCategory.STYLE,
                    severity=ToolSeverity.LOW,
                )
            ],
            tools_run=["ruff", "bandit"],
        )
        review = make_review(
            overall_rating=OverallRating.GOOD,
            findings=[],
        )

        result = validate_semantics(make_state(review, static_analysis=analysis))

        self.assertEqual(result["validation_errors"], [])

    def test_medium_bandit_evidence_requires_security_coverage(self):
        analysis = StaticAnalysisResult(
            findings=[make_tool_finding(severity=ToolSeverity.MEDIUM)],
            tools_run=["ruff", "bandit"],
        )
        review = make_review(findings=[])

        result = validate_semantics(make_state(review, static_analysis=analysis))

        self.assertTrue(
            any("bandit B602" in error for error in result["validation_errors"])
        )

    @patch("agent.ChatOpenAI")
    def test_tool_errors_do_not_block_llm_review(self, chat_openai):
        self.mocked_analyze_code.return_value = StaticAnalysisResult(
            tool_errors=[
                "ruff executable not found",
                "bandit executable not found",
            ]
        )
        expected = make_review(
            overall_rating=OverallRating.GOOD,
            findings=[],
        )
        structured_llm = chat_openai.return_value.with_structured_output.return_value
        structured_llm.invoke.return_value = expected

        result = review_code("return 42")

        self.assertEqual(result, expected)
        prompt = structured_llm.invoke.call_args.args[0][1].content
        self.assertIn("ruff executable not found", prompt)


class RetryFailureTests(AgentTestCase):
    @patch("agent.ChatOpenAI")
    def test_api_error_propagates_without_semantic_retry(self, chat_openai):
        structured_llm = chat_openai.return_value.with_structured_output.return_value
        structured_llm.invoke.side_effect = TimeoutError("temporary API timeout")

        with self.assertRaisesRegex(TimeoutError, "temporary API timeout"):
            review_code("return 42")

        self.assertEqual(structured_llm.invoke.call_count, 1)

    @patch("agent.review_code")
    def test_json_cli_writes_controlled_failure(self, mocked_review_code):
        mocked_review_code.side_effect = ReviewSemanticValidationError(
            "Review failed semantic validation after 2 retries.",
            ["rating conflict"],
            2,
        )
        stdout = io.StringIO()
        stderr = io.StringIO()

        argv = ["agent.py", "--code", "return 42"]
        with patch("sys.argv", argv), redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main()

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(
            payload["error"]["type"],
            "semantic_validation_failed",
        )


def make_fix(
    updated_code: str = "def divide(a, b):\n    return a if b == 0 else a / b\n",
) -> CodeFix:
    return CodeFix(
        summary="Handle the zero divisor.",
        addressed_findings=["Zero divisor is not handled"],
        updated_code=updated_code,
    )


def make_verification(
    status: str = "verified",
    *,
    test_status: str = "passed",
    stdout: str = "",
) -> FixVerificationResult:
    return FixVerificationResult(
        status=status,
        static_analysis=StaticAnalysisResult(tools_run=["ruff", "bandit"]),
        test_result=TestRunResult(
            status=test_status,
            framework="pytest" if test_status != "not_run" else None,
            stdout=stdout,
        ),
        patch="--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n",
    )


class AutoFixWorkflowTests(AgentTestCase):
    code = "def divide(a, b):\n    return a / b\n"

    def _run_auto_fix(
        self,
        responses,
        verifications=None,
        **arguments,
    ):
        context = RepoContext(repository_root="repo", target_file="app.py")
        verification_results = verifications or [make_verification()]
        with (
            patch("agent.build_repository_context", return_value=context) as builder,
            patch(
                "agent.execute_targeted_tests",
                return_value=TestRunResult(status="passed", framework="pytest"),
            ) as executor,
            patch(
                "agent.verify_candidate_fix",
                side_effect=verification_results,
            ) as verifier,
            patch("agent.ChatOpenAI") as chat_openai,
        ):
            structured = chat_openai.return_value.with_structured_output.return_value
            structured.invoke.side_effect = responses
            result = review_and_fix(
                self.code,
                repository_root="repo",
                target_file="app.py",
                run_tests=True,
                **arguments,
            )
        return result, structured, verifier, executor, builder, chat_openai

    def test_review_code_default_never_invokes_fixer(self) -> None:
        with (
            patch("agent.ChatOpenAI") as chat_openai,
            patch("agent.verify_candidate_fix") as verifier,
        ):
            structured = chat_openai.return_value.with_structured_output.return_value
            structured.invoke.return_value = make_review()
            result = review_code(self.code)
        self.assertIsInstance(result, CodeReview)
        verifier.assert_not_called()
        chat_openai.return_value.with_structured_output.assert_called_once_with(
            CodeReview
        )

    def test_no_review_findings_skip_fix_generation(self) -> None:
        review = make_review(overall_rating=OverallRating.GOOD, findings=[])
        result, structured, verifier, _, _, _ = self._run_auto_fix([review])
        self.assertEqual(result.fix_status, "not_needed")
        self.assertIsNone(result.fix)
        self.assertEqual(structured.invoke.call_count, 1)
        verifier.assert_not_called()

    def test_auto_fix_invokes_separate_structured_fixer(self) -> None:
        result, structured, verifier, _, _, chat_openai = self._run_auto_fix(
            [make_review(), make_fix()]
        )
        self.assertIsInstance(result, ReviewAndFixResult)
        self.assertEqual(result.fix_status, "verified")
        self.assertEqual(structured.invoke.call_count, 2)
        schemas = [
            call.args[0]
            for call in chat_openai.return_value.with_structured_output.call_args_list
        ]
        self.assertEqual(schemas, [CodeReview, CodeFix])
        verifier.assert_called_once()

    def test_fixer_prompt_contains_review_and_all_original_evidence(self) -> None:
        diff = "--- a/app.py\n+++ b/app.py\n@@ -2 +2 @@\n-    return a / b\n+    return a // b\n"
        _, structured, _, _, _, _ = self._run_auto_fix(
            [make_review(), make_fix()],
            diff_text=diff,
        )
        messages = structured.invoke.call_args_list[1].args[0]
        prompt = messages[1].content
        self.assertIn("Original target source", prompt)
        self.assertIn("Validated CodeReview", prompt)
        self.assertIn("Diff context:", prompt)
        self.assertIn("Repository context:", prompt)
        self.assertIn("Deterministic static-analysis evidence", prompt)
        self.assertIn("Targeted test evidence:", prompt)
        self.assertIn("smallest practical change", messages[0].content)

    def test_fix_validation_failure_feeds_next_fixer_prompt(self) -> None:
        invalid = make_fix("def broken(:\n")
        result, structured, verifier, _, _, _ = self._run_auto_fix(
            [make_review(), invalid, make_fix()]
        )
        self.assertEqual(result.fix_status, "verified")
        retry_prompt = structured.invoke.call_args_list[2].args[0][1].content
        self.assertIn("Candidate validation errors", retry_prompt)
        self.assertIn("SyntaxError", retry_prompt)
        verifier.assert_called_once()

    def test_test_failure_feeds_next_fixer_prompt(self) -> None:
        failed = make_verification(
            "failed",
            test_status="failed",
            stdout="candidate one FAILED",
        )
        result, structured, verifier, _, _, _ = self._run_auto_fix(
            [
                make_review(),
                make_fix(),
                make_fix("def divide(a, b):\n    return a / b if b else 0\n"),
            ],
            [failed, make_verification()],
        )
        self.assertEqual(result.fix_status, "verified")
        retry_prompt = structured.invoke.call_args_list[2].args[0][1].content
        self.assertIn("Fix verification failed", retry_prompt)
        self.assertIn("candidate one FAILED", retry_prompt)
        self.assertEqual(verifier.call_count, 2)

    def test_each_new_candidate_is_verified_again(self) -> None:
        result, _, verifier, _, _, _ = self._run_auto_fix(
            [
                make_review(),
                make_fix(),
                make_fix("def divide(a, b):\n    return 0 if not b else a / b\n"),
            ],
            [
                make_verification("failed", test_status="failed"),
                make_verification(),
            ],
        )
        self.assertEqual(result.fix_status, "verified")
        self.assertEqual(verifier.call_count, 2)

    def test_max_fix_attempts_are_enforced_for_invalid_candidates(self) -> None:
        invalid = make_fix("def broken(:\n")
        result, structured, verifier, _, _, _ = self._run_auto_fix(
            [make_review(), invalid, invalid],
            max_fix_attempts=2,
        )
        self.assertEqual(result.fix_status, "failed")
        self.assertIn("after 2 attempts", result.failure_reason)
        self.assertEqual(structured.invoke.call_count, 3)
        verifier.assert_not_called()

    def test_exhausted_verification_attempts_return_controlled_failure(self) -> None:
        failed = make_verification("failed", test_status="failed")
        result, _, verifier, _, _, _ = self._run_auto_fix(
            [make_review(), make_fix(), make_fix("def divide(a, b):\n    return 0\n")],
            [failed, failed],
            max_fix_attempts=2,
        )
        self.assertEqual(result.fix_status, "failed")
        self.assertIn("status 'failed'", result.failure_reason)
        self.assertEqual(verifier.call_count, 2)

    def test_non_retryable_workspace_error_is_not_misreported_as_code_failure(
        self,
    ) -> None:
        errored = make_verification("error", test_status="not_run")
        result, _, verifier, _, _, _ = self._run_auto_fix(
            [make_review(), make_fix()],
            [errored],
        )
        self.assertEqual(result.fix_status, "not_verified")
        self.assertEqual(verifier.call_count, 1)

    def test_review_and_fix_return_type_contains_verified_patch(self) -> None:
        result, _, _, _, _, _ = self._run_auto_fix([make_review(), make_fix()])
        self.assertIsInstance(result, ReviewAndFixResult)
        self.assertEqual(result.verification.patch, make_verification().patch)
        self.assertIn("return a if b == 0", result.fix.updated_code)

    def test_review_retry_and_fix_retry_use_independent_evidence_runs(self) -> None:
        invalid_review = make_review(overall_rating=OverallRating.GOOD)
        invalid_fix = make_fix("def broken(:\n")
        result, structured, verifier, executor, builder, _ = self._run_auto_fix(
            [invalid_review, make_review(), invalid_fix, make_fix()]
        )
        self.assertEqual(result.fix_status, "verified")
        self.assertEqual(structured.invoke.call_count, 4)
        self.assertEqual(verifier.call_count, 1)
        executor.assert_called_once()
        builder.assert_called_once()
        self.mocked_analyze_code.assert_called_once()

    def test_manual_diff_and_auto_fix_share_existing_parser(self) -> None:
        diff = "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n"
        result, structured, _, _, _, _ = self._run_auto_fix(
            [make_review(), make_fix()],
            diff_text=diff,
        )
        self.assertEqual(result.fix_status, "verified")
        self.assertIn(
            "Diff context:",
            structured.invoke.call_args_list[1].args[0][1].content,
        )

    def test_git_diff_tests_and_auto_fix_share_integration_path(self) -> None:
        diff = "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n"
        git_result = GitDiffResult(
            mode="working_tree",
            repository_root="repo",
            target_file="app.py",
            diff_text=diff,
        )
        with patch("agent.collect_git_diff", return_value=git_result) as collector:
            result, _, _, executor, _, _ = self._run_auto_fix(
                [make_review(), make_fix()],
                git_diff=True,
            )
        self.assertEqual(result.fix_status, "verified")
        collector.assert_called_once()
        executor.assert_called_once()

    def test_api_requires_repository_and_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "repository_root and target_file"):
            review_and_fix(self.code, run_tests=True)

    def test_api_requires_explicit_test_execution(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires run_tests=True"):
            review_and_fix(
                self.code,
                repository_root="repo",
                target_file="app.py",
            )

    def test_api_rejects_non_python_auto_fix(self) -> None:
        with self.assertRaisesRegex(ValueError, "language='python'"):
            review_and_fix(
                "const value = 1;",
                language="javascript",
                repository_root="repo",
                target_file="app.py",
                run_tests=True,
            )


class AutoFixCliTests(AgentTestCase):
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

    def test_auto_fix_requires_file(self) -> None:
        error = self._argparse_error(
            [
                "agent.py",
                "--code",
                "VALUE = 1",
                "--repo",
                "repo",
                "--run-tests",
                "--auto-fix",
            ]
        )
        self.assertIn("--auto-fix requires --file", error)

    def test_auto_fix_requires_repo(self) -> None:
        error = self._argparse_error(
            ["agent.py", "--file", "app.py", "--run-tests", "--auto-fix"]
        )
        self.assertIn("--auto-fix requires --repo", error)

    def test_auto_fix_requires_run_tests(self) -> None:
        error = self._argparse_error(
            ["agent.py", "--file", "app.py", "--repo", "repo", "--auto-fix"]
        )
        self.assertIn("--auto-fix requires --run-tests", error)

    def test_auto_fix_rejects_non_python_language(self) -> None:
        error = self._argparse_error(
            [
                "agent.py",
                "--file",
                "app.py",
                "--repo",
                "repo",
                "--run-tests",
                "--auto-fix",
                "--language",
                "javascript",
            ]
        )
        self.assertIn("requires --language python", error)

    @patch("agent.read_repository_target", return_value="VALUE = 1\n")
    @patch("agent.review_and_fix")
    def test_auto_fix_flag_uses_new_api(self, review_and_fix_mock, _reader) -> None:
        review_and_fix_mock.return_value = ReviewAndFixResult(
            review=make_review(),
            fix=make_fix("VALUE = 2\n"),
            verification=make_verification(),
            fix_status="verified",
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = [
            "agent.py",
            "--file",
            "app.py",
            "--repo",
            "repo",
            "--run-tests",
            "--auto-fix",
        ]
        with patch("sys.argv", argv), redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main()
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["fix_status"], "verified")
        review_and_fix_mock.assert_called_once_with(
            "VALUE = 1\n",
            "python",
            repository_root="repo",
            target_file="app.py",
            run_tests=True,
        )


class ControlledApplyWorkflowTests(AgentTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.repository = Path(self.temporary_directory.name) / "repo"
        self.repository.mkdir()
        self.target = self.repository / "app.py"
        self.code = "def divide(a, b):\n    return a / b\n"
        self.target.write_text(self.code, encoding="utf-8")
        self.context = RepoContext(
            repository_root=str(self.repository.resolve()),
            target_file="app.py",
        )

    def _run(
        self,
        *,
        apply_fix=True,
        verification=None,
        verification_side_effect=None,
        application_result=None,
        responses=None,
        **arguments,
    ):
        with ExitStack() as stack:
            builder = stack.enter_context(
                patch("agent.build_repository_context", return_value=self.context)
            )
            executor = stack.enter_context(
                patch(
                    "agent.execute_targeted_tests",
                    return_value=TestRunResult(
                        status="passed",
                        framework="pytest",
                    ),
                )
            )
            verifier = stack.enter_context(patch("agent.verify_candidate_fix"))
            if verification_side_effect is not None:
                verifier.side_effect = verification_side_effect
            else:
                verifier.return_value = verification or make_verification()
            applier = None
            if application_result is not None:
                applier = stack.enter_context(
                    patch(
                        "agent.apply_fix_to_repository",
                        return_value=application_result,
                    )
                )
            chat_openai = stack.enter_context(patch("agent.ChatOpenAI"))
            structured = chat_openai.return_value.with_structured_output.return_value
            structured.invoke.side_effect = responses or [make_review(), make_fix()]
            result = review_and_fix(
                self.code,
                repository_root=str(self.repository),
                target_file="app.py",
                run_tests=True,
                apply_fix=apply_fix,
                **arguments,
            )
        return result, structured, verifier, applier, executor, builder

    def test_verified_candidate_with_apply_false_leaves_original_unchanged(
        self,
    ) -> None:
        result, _, _, applier, _, _ = self._run(
            apply_fix=False,
            application_result=ApplyFixResult(status="applied"),
        )
        self.assertEqual(result.fix_status, "verified")
        self.assertIsNone(result.application)
        self.assertEqual(self.target.read_text(encoding="utf-8"), self.code)
        applier.assert_not_called()

    def test_verified_candidate_with_apply_true_writes_original(self) -> None:
        result, structured, verifier, _, _, _ = self._run()
        self.assertEqual(result.fix_status, "verified")
        self.assertEqual(result.application.status, "applied")
        self.assertIn("return a if b == 0", self.target.read_text(encoding="utf-8"))
        self.assertEqual(structured.invoke.call_count, 2)
        verifier.assert_called_once()

    def test_unverified_candidate_never_enters_application_node(self) -> None:
        result, _, _, applier, _, _ = self._run(
            verification=make_verification("failed", test_status="failed"),
            application_result=ApplyFixResult(status="applied"),
            max_fix_attempts=1,
        )
        self.assertEqual(result.fix_status, "failed")
        self.assertEqual(result.application.status, "error")
        self.assertEqual(self.target.read_text(encoding="utf-8"), self.code)
        applier.assert_not_called()

    def test_application_error_does_not_reenter_fixer(self) -> None:
        result, structured, verifier, applier, _, _ = self._run(
            application_result=ApplyFixResult(
                status="error",
                warnings=["disk unavailable"],
            )
        )
        self.assertEqual(result.fix_status, "verified")
        self.assertEqual(result.application.status, "error")
        self.assertEqual(structured.invoke.call_count, 2)
        verifier.assert_called_once()
        applier.assert_called_once()

    def test_stale_target_does_not_reenter_fixer_or_overwrite_user_change(self) -> None:
        changed = "def divide(a, b):\n    return 42\n"

        def mutate_after_verification(*_args, **_kwargs):
            self.target.write_text(changed, encoding="utf-8")
            return make_verification()

        result, structured, verifier, _, _, _ = self._run(
            verification_side_effect=mutate_after_verification
        )
        self.assertEqual(result.fix_status, "verified")
        self.assertEqual(result.application.status, "stale")
        self.assertEqual(self.target.read_text(encoding="utf-8"), changed)
        self.assertEqual(structured.invoke.call_count, 2)
        verifier.assert_called_once()

    def test_normal_review_and_fix_remains_no_write(self) -> None:
        before = self.target.read_bytes()
        result, _, _, _, _, _ = self._run(apply_fix=False)
        self.assertIsNone(result.application)
        self.assertEqual(self.target.read_bytes(), before)

    def test_manual_diff_verified_apply_path(self) -> None:
        diff = (
            "--- a/app.py\n+++ b/app.py\n@@ -2 +2 @@\n"
            "-    return a / b\n+    return a // b\n"
        )
        result, _, _, _, executor, _ = self._run(diff_text=diff)
        self.assertEqual(result.application.status, "applied")
        executor.assert_called_once()

    def test_git_aware_verified_apply_path(self) -> None:
        diff = "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n"
        git_result = GitDiffResult(
            mode="working_tree",
            repository_root=str(self.repository.resolve()),
            target_file="app.py",
            diff_text=diff,
        )
        with patch("agent.collect_git_diff", return_value=git_result) as collector:
            result, _, _, _, executor, _ = self._run(git_diff=True)
        self.assertEqual(result.application.status, "applied")
        collector.assert_called_once()
        executor.assert_called_once()

    def test_output_keeps_verification_and_application_distinct(self) -> None:
        result = ReviewAndFixResult(
            review=make_review(),
            fix=make_fix(),
            verification=make_verification(),
            application=ApplyFixResult(
                status="stale",
                target_file="app.py",
                warnings=["Target changed after review began."],
            ),
            fix_status="verified",
        )
        rendered = render_review_and_fix(result)
        self.assertIn("Fix verification: VERIFIED", rendered)
        self.assertIn("Application: STALE", rendered)


class ControlledApplyCliTests(AgentTestCase):
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

    def test_apply_fix_requires_auto_fix(self) -> None:
        error = self._argparse_error(
            [
                "agent.py",
                "--file",
                "app.py",
                "--repo",
                "repo",
                "--run-tests",
                "--apply-fix",
            ]
        )
        self.assertIn("--apply-fix requires --auto-fix", error)

    def test_apply_fix_requires_file(self) -> None:
        error = self._argparse_error(
            [
                "agent.py",
                "--code",
                "VALUE = 1",
                "--repo",
                "repo",
                "--run-tests",
                "--auto-fix",
                "--apply-fix",
            ]
        )
        self.assertIn("--apply-fix requires --file", error)

    def test_apply_fix_requires_repo(self) -> None:
        error = self._argparse_error(
            [
                "agent.py",
                "--file",
                "app.py",
                "--run-tests",
                "--auto-fix",
                "--apply-fix",
            ]
        )
        self.assertIn("--apply-fix requires --repo", error)

    def test_apply_fix_requires_run_tests(self) -> None:
        error = self._argparse_error(
            [
                "agent.py",
                "--file",
                "app.py",
                "--repo",
                "repo",
                "--auto-fix",
                "--apply-fix",
            ]
        )
        self.assertIn("--apply-fix requires --run-tests", error)

    def test_apply_fix_rejects_non_python(self) -> None:
        error = self._argparse_error(
            [
                "agent.py",
                "--file",
                "app.py",
                "--repo",
                "repo",
                "--run-tests",
                "--auto-fix",
                "--apply-fix",
                "--language",
                "javascript",
            ]
        )
        self.assertIn("--apply-fix currently requires --language python", error)

    @patch("agent.read_repository_target", return_value="VALUE = 1\n")
    @patch("agent.review_and_fix")
    def test_apply_fix_flag_reaches_api(self, review_and_fix_mock, _reader) -> None:
        review_and_fix_mock.return_value = ReviewAndFixResult(
            review=make_review(),
            fix=make_fix("VALUE = 2\n"),
            verification=make_verification(),
            application=ApplyFixResult(
                status="applied",
                target_file="app.py",
                bytes_written=10,
            ),
            fix_status="verified",
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = [
            "agent.py",
            "--file",
            "app.py",
            "--repo",
            "repo",
            "--run-tests",
            "--auto-fix",
            "--apply-fix",
        ]
        with patch("sys.argv", argv), redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main()
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["fix_status"], "verified")
        self.assertEqual(payload["application"]["status"], "applied")
        review_and_fix_mock.assert_called_once_with(
            "VALUE = 1\n",
            "python",
            repository_root="repo",
            target_file="app.py",
            run_tests=True,
            apply_fix=True,
        )


class AgenticExploreCliTests(unittest.TestCase):
    def assert_cli_error(self, argv: list[str], message: str) -> None:
        stderr = io.StringIO()
        with (
            patch("sys.argv", argv),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit),
        ):
            main()
        self.assertIn(message, stderr.getvalue())

    def test_inline_code_is_rejected(self) -> None:
        self.assert_cli_error(
            ["agent.py", "--code", "value = 1", "--agentic-explore"],
            "--code cannot be combined with --agentic-explore",
        )

    def test_repository_is_required(self) -> None:
        self.assert_cli_error(
            ["agent.py", "--file", "app.py", "--agentic-explore"],
            "--agentic-explore requires --repo",
        )

    def test_python_language_is_required(self) -> None:
        self.assert_cli_error(
            [
                "agent.py",
                "--file",
                "app.py",
                "--repo",
                ".",
                "--language",
                "javascript",
                "--agentic-explore",
            ],
            "--agentic-explore currently requires --language python",
        )

    @patch("agent.read_repository_target", return_value="VALUE = 1\n")
    @patch("agent.review_code", return_value=make_review())
    def test_flag_reaches_single_file_api(self, reviewer, _reader) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = [
            "agent.py",
            "--file",
            "app.py",
            "--repo",
            "repo",
            "--agentic-explore",
        ]
        with patch("sys.argv", argv), redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main()
        self.assertEqual(exit_code, 0)
        reviewer.assert_called_once_with(
            "VALUE = 1\n",
            "python",
            repository_root="repo",
            target_file="app.py",
            agentic_explore=True,
        )


class AgenticTestCliTests(unittest.TestCase):
    def assert_cli_error(self, argv: list[str], message: str) -> None:
        stderr = io.StringIO()
        with (
            patch("sys.argv", argv),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit),
        ):
            main()
        self.assertIn(message, stderr.getvalue())

    def test_requires_agentic_explore(self) -> None:
        self.assert_cli_error(
            [
                "agent.py",
                "--file",
                "app.py",
                "--repo",
                ".",
                "--run-tests",
                "--agentic-test",
            ],
            "--agentic-test requires --agentic-explore",
        )

    def test_requires_run_tests(self) -> None:
        self.assert_cli_error(
            [
                "agent.py",
                "--file",
                "app.py",
                "--repo",
                ".",
                "--agentic-explore",
                "--agentic-test",
            ],
            "--agentic-test requires --run-tests",
        )

    def test_requires_repository(self) -> None:
        self.assert_cli_error(
            [
                "agent.py",
                "--file",
                "app.py",
                "--run-tests",
                "--agentic-explore",
                "--agentic-test",
            ],
            "--agentic-test requires --repo",
        )

    def test_inline_code_is_rejected(self) -> None:
        self.assert_cli_error(
            [
                "agent.py",
                "--code",
                "value = 1",
                "--repo",
                ".",
                "--run-tests",
                "--agentic-explore",
                "--agentic-test",
            ],
            "--code cannot be combined with --agentic-test",
        )

    def test_python_language_is_required(self) -> None:
        self.assert_cli_error(
            [
                "agent.py",
                "--file",
                "app.py",
                "--repo",
                ".",
                "--run-tests",
                "--agentic-explore",
                "--agentic-test",
                "--language",
                "javascript",
            ],
            "--agentic-test currently requires --language python",
        )

    @patch("agent.read_repository_target", return_value="VALUE = 1\n")
    @patch("agent.review_code", return_value=make_review())
    def test_flag_reaches_single_file_api(self, reviewer, _reader) -> None:
        with (
            patch(
                "sys.argv",
                [
                    "agent.py",
                    "--file",
                    "app.py",
                    "--repo",
                    "repo",
                    "--run-tests",
                    "--agentic-explore",
                    "--agentic-test",
                ],
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(main(), 0)
        reviewer.assert_called_once_with(
            "VALUE = 1\n",
            "python",
            repository_root="repo",
            target_file="app.py",
            run_tests=True,
            agentic_explore=True,
            agentic_test=True,
        )


class RepositoryTaskPlanningCliTests(unittest.TestCase):
    def assert_cli_error(self, argv: list[str], message: str) -> None:
        stderr = io.StringIO()
        with (
            patch("sys.argv", argv),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn(message, stderr.getvalue())

    def test_plan_task_requires_repository(self) -> None:
        self.assert_cli_error(
            ["agent.py", "--plan-task", "Fix lookup."],
            "--plan-task requires --repo",
        )

    def test_plan_task_requires_python_mode(self) -> None:
        self.assert_cli_error(
            [
                "agent.py",
                "--plan-task",
                "Fix lookup.",
                "--repo",
                ".",
                "--language",
                "javascript",
            ],
            "--plan-task currently requires --language python",
        )

    def test_plan_task_rejects_review_fix_and_execution_flags(self) -> None:
        conflicts = (
            (["--auto-fix"], "--auto-fix"),
            (["--apply-fix"], "--apply-fix"),
            (["--run-tests"], "--run-tests"),
            (["--agentic-test"], "--agentic-test"),
            (["--agentic-explore"], "--agentic-explore"),
            (["--diff", "change.diff"], "--diff"),
            (["--git-diff"], "--git-diff"),
            (["--git-base", "main"], "--git-base"),
        )
        for flags, expected in conflicts:
            with self.subTest(flags=flags):
                self.assert_cli_error(
                    [
                        "agent.py",
                        "--plan-task",
                        "Fix lookup.",
                        "--repo",
                        ".",
                        *flags,
                    ],
                    expected,
                )

    @patch("agent.plan_repository_task")
    def test_plan_task_outputs_structured_json(self, planner) -> None:
        expected = EngineeringPlan(
            summary="Normalize repository lookup keys.",
            files=[
                PlannedFileChange(
                    path="src/repository.py",
                    action="modify",
                    rationale="Make matching case-insensitive.",
                )
            ],
            tests=[
                PlannedTest(
                    path="tests/test_repository.py",
                    purpose="Cover mixed-case input.",
                )
            ],
        )
        planner.return_value = expected
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            patch(
                "sys.argv",
                [
                    "agent.py",
                    "--plan-task",
                    "Fix lookup.",
                    "--repo",
                    "repo",
                ],
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout.getvalue()), expected.model_dump())
        self.assertIn("Planning: repository engineering task", stderr.getvalue())
        planner.assert_called_once_with("repo", "Fix lookup.")

    @patch("agent.plan_repository_task")
    def test_plan_task_validation_failure_is_stable_json(self, planner) -> None:
        planner.side_effect = EngineeringPlanValidationError(
            "Engineering plan failed deterministic validation after 2 retries.",
            ["Planned modified file was not grounded."],
            2,
        )
        stdout = io.StringIO()

        with (
            patch(
                "sys.argv",
                [
                    "agent.py",
                    "--plan-task",
                    "Fix lookup.",
                    "--repo",
                    "repo",
                ],
            ),
            redirect_stdout(stdout),
            redirect_stderr(io.StringIO()),
        ):
            exit_code = main()

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertEqual(
            payload["error"]["type"],
            "engineering_plan_validation_failed",
        )
        self.assertEqual(payload["error"]["retry_count"], 2)


class RepositoryTaskExecutionCliTests(unittest.TestCase):
    def assert_cli_error(self, argv: list[str], message: str) -> None:
        stderr = io.StringIO()
        with (
            patch("sys.argv", argv),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn(message, stderr.getvalue())

    def test_execute_task_requires_repository_and_python(self) -> None:
        self.assert_cli_error(
            ["agent.py", "--execute-task", "Fix lookup."],
            "--execute-task requires --repo",
        )
        self.assert_cli_error(
            [
                "agent.py",
                "--execute-task",
                "Fix lookup.",
                "--repo",
                ".",
                "--language",
                "javascript",
            ],
            "--execute-task currently requires --language python",
        )

    def test_execute_task_rejects_write_diff_and_test_flags(self) -> None:
        conflicts = (
            (["--auto-fix"], "--auto-fix"),
            (["--apply-fix"], "--apply-fix"),
            (["--run-tests"], "--run-tests"),
            (["--agentic-test"], "--agentic-test"),
            (["--diff", "change.diff"], "--diff"),
            (["--git-diff"], "--git-diff"),
            (["--git-base", "main"], "--git-base"),
        )
        for flags, expected in conflicts:
            with self.subTest(flags=flags):
                self.assert_cli_error(
                    [
                        "agent.py",
                        "--execute-task",
                        "Fix lookup.",
                        "--repo",
                        ".",
                        *flags,
                    ],
                    expected,
                )

    @patch("agent.plan_and_execute_repository_task")
    def test_execute_task_outputs_structured_json(self, executor) -> None:
        plan = EngineeringPlan(
            summary="Normalize lookup.",
            files=[
                PlannedFileChange(
                    path="src/repository.py",
                    action="modify",
                    rationale="Normalize email keys.",
                )
            ],
        )
        expected = PlanExecutionResult(
            plan=plan,
            status="failed",
            validation_errors=["Candidate failed validation."],
        )
        executor.return_value = expected
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            patch(
                "sys.argv",
                [
                    "agent.py",
                    "--execute-task",
                    "Fix lookup.",
                    "--repo",
                    "repo",
                ],
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout.getvalue()), expected.model_dump())
        self.assertIn(
            "Executing preview for: repository task execution preview",
            stderr.getvalue(),
        )
        executor.assert_called_once_with(
            "repo",
            "Fix lookup.",
            candidate_review_agentic_explore=False,
        )

    @patch("agent.plan_and_execute_repository_task")
    def test_execute_task_can_enable_candidate_review_exploration(
        self,
        executor,
    ) -> None:
        plan = EngineeringPlan(
            summary="Normalize lookup.",
            files=[
                PlannedFileChange(
                    path="src/repository.py",
                    action="modify",
                    rationale="Normalize email keys.",
                )
            ],
        )
        executor.return_value = PlanExecutionResult(
            plan=plan,
            status="candidate_generated",
        )

        with (
            patch(
                "sys.argv",
                [
                    "agent.py",
                    "--execute-task",
                    "Fix lookup.",
                    "--repo",
                    "repo",
                    "--agentic-explore",
                ],
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        executor.assert_called_once_with(
            "repo",
            "Fix lookup.",
            candidate_review_agentic_explore=True,
        )

    def test_self_correct_is_only_valid_with_execute_task(self) -> None:
        self.assert_cli_error(
            ["agent.py", "--file", "app.py", "--self-correct"],
            "--self-correct is only valid with --execute-task",
        )
        self.assert_cli_error(
            [
                "agent.py",
                "--plan-task",
                "Fix lookup.",
                "--repo",
                "repo",
                "--self-correct",
            ],
            "--self-correct is only valid with --execute-task",
        )

    def test_max_correction_rounds_requires_opt_in_and_non_negative(
        self,
    ) -> None:
        self.assert_cli_error(
            [
                "agent.py",
                "--execute-task",
                "Fix lookup.",
                "--repo",
                "repo",
                "--max-correction-rounds",
                "2",
            ],
            "--max-correction-rounds requires --self-correct",
        )
        self.assert_cli_error(
            [
                "agent.py",
                "--execute-task",
                "Fix lookup.",
                "--repo",
                "repo",
                "--self-correct",
                "--max-correction-rounds",
                "-1",
            ],
            "--max-correction-rounds must be at least 0",
        )

    @patch("agent.plan_and_execute_repository_task")
    def test_self_correct_uses_default_and_explicit_budgets(
        self,
        executor,
    ) -> None:
        plan = EngineeringPlan(
            summary="Normalize lookup.",
            files=[
                PlannedFileChange(
                    path="src/repository.py",
                    action="modify",
                    rationale="Normalize email keys.",
                )
            ],
        )
        executor.return_value = PlanExecutionResult(
            plan=plan,
            status="verified",
        )
        for extra, expected in (
            (["--self-correct"], 2),
            (["--self-correct", "--max-correction-rounds", "1"], 1),
        ):
            with (
                self.subTest(extra=extra),
                patch(
                    "sys.argv",
                    [
                        "agent.py",
                        "--execute-task",
                        "Fix lookup.",
                        "--repo",
                        "repo",
                        *extra,
                    ],
                ),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(main(), 0)
            self.assertEqual(
                executor.call_args.kwargs["max_correction_rounds"],
                expected,
            )
            self.assertFalse(
                executor.call_args.kwargs["candidate_review_agentic_explore"]
            )


class PlanApplicationCliTests(unittest.TestCase):
    def assert_cli_error(self, argv: list[str], message: str) -> None:
        stderr = io.StringIO()
        with (
            patch("sys.argv", argv),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn(message, stderr.getvalue())

    def test_save_bundle_requires_execute_task(self) -> None:
        self.assert_cli_error(
            [
                "agent.py",
                "--file",
                "app.py",
                "--save-apply-bundle",
                "approved.json",
            ],
            "--save-apply-bundle requires --execute-task",
        )

    def test_approve_is_rejected_outside_apply_mode(self) -> None:
        self.assert_cli_error(
            ["agent.py", "--file", "app.py", "--approve"],
            "--approve is only valid with --apply-execution",
        )

    def test_apply_execution_requires_repo_and_approval(self) -> None:
        self.assert_cli_error(
            ["agent.py", "--apply-execution", "approved.json", "--approve"],
            "--apply-execution requires --repo",
        )
        self.assert_cli_error(
            [
                "agent.py",
                "--apply-execution",
                "approved.json",
                "--repo",
                "repo",
            ],
            "--apply-execution requires --approve",
        )

    def test_apply_execution_rejects_unrelated_options(self) -> None:
        self.assert_cli_error(
            [
                "agent.py",
                "--apply-execution",
                "approved.json",
                "--repo",
                "repo",
                "--approve",
                "--run-tests",
            ],
            "--apply-execution cannot be combined",
        )

    @patch("agent.save_plan_application_bundle")
    @patch("agent.build_plan_application_bundle")
    @patch("agent.plan_and_execute_repository_task")
    def test_verified_execution_saves_bundle(
        self,
        executor,
        build_bundle,
        save_bundle,
    ) -> None:
        plan = EngineeringPlan(
            summary="Normalize lookup.",
            files=[
                PlannedFileChange(
                    path="src/repository.py",
                    action="modify",
                    rationale="Normalize email keys.",
                )
            ],
        )
        execution = PlanExecutionResult(plan=plan, status="verified")
        executor.return_value = execution
        approved_bundle = object()
        build_bundle.return_value = approved_bundle
        stderr = io.StringIO()
        with (
            patch(
                "sys.argv",
                [
                    "agent.py",
                    "--execute-task",
                    "Fix lookup.",
                    "--repo",
                    "repo",
                    "--save-apply-bundle",
                    "C:/Temp/approved.json",
                ],
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(stderr),
        ):
            exit_code = main()
        self.assertEqual(exit_code, 0)
        build_bundle.assert_called_once_with("repo", execution)
        save_bundle.assert_called_once_with(
            "C:/Temp/approved.json",
            approved_bundle,
            repository_root="repo",
        )
        self.assertIn("Saved application bundle", stderr.getvalue())

    @patch("agent.save_plan_application_bundle")
    @patch("agent.build_plan_application_bundle")
    @patch("agent.plan_and_execute_repository_task")
    def test_unverified_execution_does_not_save_bundle(
        self,
        executor,
        build_bundle,
        save_bundle,
    ) -> None:
        plan = EngineeringPlan(
            summary="Normalize lookup.",
            files=[
                PlannedFileChange(
                    path="src/repository.py",
                    action="modify",
                    rationale="Normalize email keys.",
                )
            ],
        )
        executor.return_value = PlanExecutionResult(plan=plan, status="failed")
        stderr = io.StringIO()
        with (
            patch(
                "sys.argv",
                [
                    "agent.py",
                    "--execute-task",
                    "Fix lookup.",
                    "--repo",
                    "repo",
                    "--save-apply-bundle",
                    "C:/Temp/approved.json",
                ],
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(stderr),
        ):
            exit_code = main()
        self.assertEqual(exit_code, 0)
        build_bundle.assert_not_called()
        save_bundle.assert_not_called()
        self.assertIn("was not saved", stderr.getvalue())

    @patch("agent.apply_plan_application_bundle")
    @patch("agent.load_plan_application_bundle")
    def test_apply_execution_loads_and_explicitly_approves(
        self,
        load_bundle,
        apply_bundle,
    ) -> None:
        approved_bundle = object()
        load_bundle.return_value = approved_bundle
        expected = PlanApplicationResult(
            status="applied",
            applied_files=["src/repository.py"],
            approval_digest="a" * 64,
        )
        apply_bundle.return_value = expected
        stdout = io.StringIO()
        with (
            patch(
                "sys.argv",
                [
                    "agent.py",
                    "--apply-execution",
                    "C:/Temp/approved.json",
                    "--repo",
                    "repo",
                    "--approve",
                ],
            ),
            redirect_stdout(stdout),
            redirect_stderr(io.StringIO()),
        ):
            exit_code = main()
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout.getvalue()), expected.model_dump())
        load_bundle.assert_called_once_with("C:/Temp/approved.json")
        apply_bundle.assert_called_once_with(
            "repo",
            approved_bundle,
            approved=True,
        )

    @patch("agent.apply_plan_application_bundle")
    @patch("agent.load_plan_application_bundle")
    def test_stale_apply_returns_failure_exit_code(
        self,
        load_bundle,
        apply_bundle,
    ) -> None:
        load_bundle.return_value = object()
        apply_bundle.return_value = PlanApplicationResult(
            status="stale",
            failure_reason="Repository changed.",
        )
        with (
            patch(
                "sys.argv",
                [
                    "agent.py",
                    "--apply-execution",
                    "C:/Temp/approved.json",
                    "--repo",
                    "repo",
                    "--approve",
                ],
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(main(), 1)


class LocalGitDeliveryCliTests(unittest.TestCase):
    def assert_cli_error(self, argv: list[str], message: str) -> None:
        stderr = io.StringIO()
        with (
            patch("sys.argv", argv),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn(message, stderr.getvalue())

    def test_delivery_requires_separate_approval_and_repository(self) -> None:
        self.assert_cli_error(
            [
                "agent.py",
                "--create-local-delivery",
                "approved.json",
                "--approve-git-delivery",
            ],
            "--create-local-delivery requires --repo",
        )
        self.assert_cli_error(
            [
                "agent.py",
                "--create-local-delivery",
                "approved.json",
                "--repo",
                "repo",
            ],
            "--create-local-delivery requires --approve-git-delivery",
        )

    def test_delivery_metadata_and_approval_require_delivery_mode(self) -> None:
        self.assert_cli_error(
            ["agent.py", "--file", "app.py", "--approve-git-delivery"],
            "--approve-git-delivery is only valid with --create-local-delivery",
        )
        self.assert_cli_error(
            ["agent.py", "--file", "app.py", "--delivery-branch", "feature/x"],
            "--delivery-branch requires --create-local-delivery",
        )
        self.assert_cli_error(
            ["agent.py", "--file", "app.py", "--commit-message", "message"],
            "--commit-message requires --create-local-delivery",
        )

    def test_delivery_rejects_review_and_generation_options(self) -> None:
        self.assert_cli_error(
            [
                "agent.py",
                "--create-local-delivery",
                "approved.json",
                "--repo",
                "repo",
                "--approve-git-delivery",
                "--run-tests",
            ],
            "--create-local-delivery cannot be combined",
        )

    @patch("agent.create_local_git_delivery")
    @patch("agent.load_plan_application_bundle")
    def test_delivery_loads_bundle_and_passes_explicit_metadata(
        self,
        load_bundle,
        create_delivery,
    ) -> None:
        bundle = object()
        load_bundle.return_value = bundle
        expected = LocalGitDeliveryResult(
            status="created",
            branch_name="feature/approved",
            commit_sha="a" * 40,
            base_sha="b" * 40,
            approval_digest="c" * 64,
            committed_files=["src/repository.py"],
        )
        create_delivery.return_value = expected
        stdout = io.StringIO()
        with (
            patch(
                "sys.argv",
                [
                    "agent.py",
                    "--create-local-delivery",
                    "C:/Temp/approved.json",
                    "--repo",
                    "repo",
                    "--approve-git-delivery",
                    "--delivery-branch",
                    "feature/approved",
                    "--commit-message",
                    "Apply approved plan",
                ],
            ),
            redirect_stdout(stdout),
            redirect_stderr(io.StringIO()),
        ):
            exit_code = main()
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout.getvalue()), expected.model_dump())
        load_bundle.assert_called_once_with("C:/Temp/approved.json")
        create_delivery.assert_called_once_with(
            "repo",
            bundle,
            approved=True,
            branch_name="feature/approved",
            commit_message="Apply approved plan",
        )

    @patch("agent.create_local_git_delivery")
    @patch("agent.load_plan_application_bundle")
    def test_delivery_failure_returns_nonzero(
        self,
        load_bundle,
        create_delivery,
    ) -> None:
        load_bundle.return_value = object()
        create_delivery.return_value = LocalGitDeliveryResult(
            status="conflict",
            failure_reason="Branch exists.",
        )
        with (
            patch(
                "sys.argv",
                [
                    "agent.py",
                    "--create-local-delivery",
                    "approved.json",
                    "--repo",
                    "repo",
                    "--approve-git-delivery",
                ],
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(main(), 1)


class GitHubRemoteDeliveryCliTests(unittest.TestCase):
    def assert_cli_error(self, argv: list[str], message: str) -> None:
        stderr = io.StringIO()
        with (
            patch("sys.argv", argv),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn(message, stderr.getvalue())

    def test_remote_delivery_requires_repo_base_and_third_approval(self) -> None:
        self.assert_cli_error(
            [
                "agent.py",
                "--publish-local-delivery",
                "approved.json",
                "--base-branch",
                "main",
                "--approve-remote-delivery",
            ],
            "--publish-local-delivery requires --repo",
        )
        self.assert_cli_error(
            [
                "agent.py",
                "--publish-local-delivery",
                "approved.json",
                "--repo",
                "repo",
                "--approve-remote-delivery",
            ],
            "--publish-local-delivery requires --base-branch",
        )
        self.assert_cli_error(
            [
                "agent.py",
                "--publish-local-delivery",
                "approved.json",
                "--repo",
                "repo",
                "--base-branch",
                "main",
            ],
            "--publish-local-delivery requires --approve-remote-delivery",
        )

    def test_remote_metadata_and_approval_require_publish_mode(self) -> None:
        self.assert_cli_error(
            ["agent.py", "--file", "app.py", "--approve-remote-delivery"],
            "--approve-remote-delivery is only valid with --publish-local-delivery",
        )
        for option, value in (
            ("--remote", "upstream"),
            ("--base-branch", "main"),
            ("--pr-title", "Title"),
            ("--pr-body", "Body"),
        ):
            with self.subTest(option=option):
                self.assert_cli_error(
                    ["agent.py", "--file", "app.py", option, value],
                    "require --publish-local-delivery",
                )

    def test_remote_delivery_rejects_other_approval_and_runtime_options(self) -> None:
        self.assert_cli_error(
            [
                "agent.py",
                "--publish-local-delivery",
                "approved.json",
                "--repo",
                "repo",
                "--base-branch",
                "main",
                "--approve-remote-delivery",
                "--approve-git-delivery",
            ],
            "--approve-git-delivery is only valid with --create-local-delivery",
        )
        self.assert_cli_error(
            [
                "agent.py",
                "--publish-local-delivery",
                "approved.json",
                "--repo",
                "repo",
                "--base-branch",
                "main",
                "--approve-remote-delivery",
                "--run-tests",
            ],
            "--publish-local-delivery cannot be combined",
        )

    @patch("agent.publish_local_git_delivery")
    @patch("agent.load_plan_application_bundle")
    def test_remote_delivery_dispatches_exact_metadata(
        self,
        load_bundle,
        publish_delivery,
    ) -> None:
        bundle = object()
        load_bundle.return_value = bundle
        expected = GitHubRemoteDeliveryResult(
            status="published",
            remote_name="upstream",
            owner="alice",
            repository="project",
            branch_name="repograph/approved",
            commit_sha="a" * 40,
            base_sha="b" * 40,
            base_branch="main",
            approval_digest="c" * 64,
            pushed=True,
            push_created=True,
            pr_created=True,
            pr_number=42,
            pr_url="https://github.com/alice/project/pull/42",
        )
        publish_delivery.return_value = expected
        stdout = io.StringIO()
        with (
            patch(
                "sys.argv",
                [
                    "agent.py",
                    "--publish-local-delivery",
                    "C:/Temp/approved.json",
                    "--repo",
                    "repo",
                    "--base-branch",
                    "main",
                    "--approve-remote-delivery",
                    "--remote",
                    "upstream",
                    "--delivery-branch",
                    "repograph/approved",
                    "--pr-title",
                    "Title",
                    "--pr-body",
                    "Body",
                ],
            ),
            redirect_stdout(stdout),
            redirect_stderr(io.StringIO()),
        ):
            exit_code = main()
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout.getvalue()), expected.model_dump())
        load_bundle.assert_called_once_with("C:/Temp/approved.json")
        publish_delivery.assert_called_once_with(
            "repo",
            bundle,
            approved=True,
            local_branch="repograph/approved",
            remote_name="upstream",
            base_branch="main",
            pr_title="Title",
            pr_body="Body",
        )

    @patch("agent.publish_local_git_delivery")
    @patch("agent.load_plan_application_bundle")
    def test_remote_delivery_failure_returns_nonzero(
        self,
        load_bundle,
        publish_delivery,
    ) -> None:
        load_bundle.return_value = object()
        publish_delivery.return_value = GitHubRemoteDeliveryResult(
            status="partial",
            failure_reason="PR creation timed out.",
        )
        with (
            patch(
                "sys.argv",
                [
                    "agent.py",
                    "--publish-local-delivery",
                    "approved.json",
                    "--repo",
                    "repo",
                    "--base-branch",
                    "main",
                    "--approve-remote-delivery",
                ],
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(main(), 1)


if __name__ == "__main__":
    unittest.main()
