"""Graph, retry, API composition, and integration tests for Stage G2."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from change_set import (
    ChangeSetReview,
    ChangeTarget,
    FileReviewResult,
)
from engineering_plan import EngineeringPlan, PlannedFileChange, PlannedTest
from plan_execution import (
    DEFAULT_MAX_CORRECTION_ROUNDS,
    CandidateFileChange,
    MultiFileCandidate,
    _initial_execution_state,
    build_plan_execution_graph,
    execute_engineering_plan,
    plan_and_execute_repository_task,
    plan_execution_graph,
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
from test_execution import TestRunResult


class PlanExecutionGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "repo"
        self.root.mkdir()
        (self.root / "src").mkdir()
        (self.root / "tests").mkdir()
        (self.root / ".git").mkdir()
        (self.root / ".git" / "config").write_text(
            "[core]\nrepositoryformatversion = 0\n",
            encoding="utf-8",
        )
        (self.root / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
        (self.root / "src" / "repository.py").write_text(
            "USERS = {'user@example.com': {'name': 'User'}}\n\n"
            "def find_by_email(email):\n"
            "    return USERS.get(email)\n",
            encoding="utf-8",
        )
        (self.root / "tests" / "test_service.py").write_text(
            "def test_placeholder():\n    assert True\n",
            encoding="utf-8",
        )
        (self.root / "unrelated.py").write_text(
            "DO_NOT_SEND_THIS_MARKER = True\n",
            encoding="utf-8",
        )
        self.task = "Fix user lookup so email matching is case-insensitive."
        self.plan = EngineeringPlan(
            summary="Normalize email lookup at the repository boundary.",
            files=[
                PlannedFileChange(
                    path="src/repository.py",
                    action="modify",
                    rationale="Use normalized lookup keys.",
                ),
                PlannedFileChange(
                    path="tests/test_service.py",
                    action="modify",
                    rationale="Cover mixed-case lookup.",
                ),
            ],
            tests=[
                PlannedTest(
                    path="tests/test_service.py",
                    purpose="Verify mixed-case lookup.",
                )
            ],
            risks=["Stored keys must already be normalized."],
            assumptions=["Email identity is case-insensitive."],
        )
        self.candidate = MultiFileCandidate(
            summary="Normalized lookup and added focused coverage.",
            files=[
                CandidateFileChange(
                    path="src/repository.py",
                    action="modify",
                    content=(
                        "USERS = {'user@example.com': {'name': 'User'}}\n\n"
                        "def find_by_email(email):\n"
                        "    return USERS.get(email.casefold())\n"
                    ),
                ),
                CandidateFileChange(
                    path="tests/test_service.py",
                    action="modify",
                    content=(
                        "from src.repository import find_by_email\n\n"
                        "def test_lookup_is_case_insensitive():\n"
                        "    assert find_by_email('User@Example.COM') is not None\n"
                    ),
                ),
            ],
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def executor(
        self,
        *responses: object,
    ) -> tuple[MagicMock, MagicMock]:
        model = MagicMock()
        structured = model.with_structured_output.return_value
        structured.invoke.side_effect = list(responses)
        return model, structured

    def review(
        self,
        rating: OverallRating = OverallRating.GOOD,
        *,
        concrete: bool = False,
    ) -> ChangeSetReview:
        file_results = []
        if concrete:
            file_results.append(
                FileReviewResult(
                    target=ChangeTarget(
                        path="src/repository.py",
                        change_kind="modified",
                    ),
                    status="reviewed",
                    review=CodeReview(
                        overall_rating=rating,
                        summary="The lookup implementation needs correction.",
                        findings=[
                            ReviewFinding(
                                category=FindingCategory.BUG,
                                severity=Severity.HIGH,
                                title="Lookup normalization is incorrect",
                                description="The candidate uppercases the key.",
                                line_number=4,
                                suggestion="Use casefold before lookup.",
                            )
                        ],
                    ),
                )
            )
        return ChangeSetReview(
            overall_rating=rating,
            summary="Candidate review completed.",
            file_results=file_results,
        )

    def buggy_candidate(self, suffix: str = "upper") -> MultiFileCandidate:
        return MultiFileCandidate(
            summary=f"Buggy lookup candidate using {suffix}.",
            files=[
                CandidateFileChange(
                    path="src/repository.py",
                    action="modify",
                    content=(
                        "USERS = {'user@example.com': {'name': 'User'}}\n\n"
                        "def find_by_email(email):\n"
                        f"    return USERS.get(email.{suffix}())\n"
                    ),
                ),
                self.candidate.files[1],
            ],
        )

    def run_graph(
        self,
        graph,
        temporary_base: str,
        *,
        max_attempts: int = 3,
        max_corrections: int = 0,
        agentic: bool = False,
    ):
        state = _initial_execution_state(
            self.root.resolve(),
            self.task,
            self.plan,
            temporary_base,
            max_execution_attempts=max_attempts,
            max_correction_rounds=max_corrections,
            candidate_review_agentic_explore=agentic,
            max_workspace_files=5_000,
            max_workspace_bytes=100_000_000,
        )
        return graph.invoke(state)

    def test_plan_execution_graph_has_independent_expected_topology(self) -> None:
        nodes = plan_execution_graph.get_graph().nodes
        self.assertTrue(
            {
                "build_execution_context",
                "generate_multi_file_candidate",
                "validate_multi_file_candidate",
                "prepare_candidate_retry",
                "prepare_fresh_workspace",
                "create_temporary_workspace",
                "materialize_candidate",
                "build_candidate_diff",
                "verify_candidate",
                "review_candidate_change_set",
                "evaluate_candidate_outcome",
                "build_correction_feedback",
                "generate_corrected_candidate",
                "controlled_execution_failure",
            }.issubset(nodes)
        )

    def test_executor_receives_task_plan_source_and_no_arbitrary_source(self) -> None:
        executor, structured = self.executor(self.candidate)
        reviewer = MagicMock(return_value=self.review())
        graph = build_plan_execution_graph(
            executor_model=executor,
            change_set_reviewer=reviewer,
        )
        with (
            tempfile.TemporaryDirectory() as temporary, patch(
                "plan_execution.analyze_code",
                return_value=StaticAnalysisResult(),
            ),
            patch(
                "plan_execution.execute_test_files",
                return_value=TestRunResult(status="passed"),
            ),
        ):
            final_state = self.run_graph(graph, temporary)

        prompt = structured.invoke.call_args.args[0][1].content
        self.assertIn(self.task, prompt)
        self.assertIn("Normalize email lookup", prompt)
        self.assertIn("def find_by_email", prompt)
        self.assertNotIn("DO_NOT_SEND_THIS_MARKER", prompt)
        self.assertNotIn(temporary, prompt)
        self.assertIsNotNone(final_state["change_set_review"])
        executor.bind_tools.assert_not_called()

    def test_semantic_retry_reuses_context_and_never_replans(self) -> None:
        invalid = MultiFileCandidate(
            summary="Includes an unauthorized file.",
            files=[
                *self.candidate.files,
                CandidateFileChange(
                    path="src/extra.py",
                    action="add",
                    content="VALUE = 1\n",
                ),
            ],
        )
        executor, structured = self.executor(invalid, self.candidate)
        reviewer = MagicMock(return_value=self.review())
        graph = build_plan_execution_graph(
            executor_model=executor,
            change_set_reviewer=reviewer,
        )
        with (
            tempfile.TemporaryDirectory() as temporary, patch(
                "plan_execution.analyze_code",
                return_value=StaticAnalysisResult(),
            ),
            patch(
                "plan_execution.execute_test_files",
                return_value=TestRunResult(status="passed"),
            ),
            patch(
                "plan_execution.build_execution_context",
                wraps=__import__(
                    "plan_execution"
                ).build_execution_context,
            ) as context_builder,
            patch("plan_execution.plan_repository_task") as planner,
        ):
            final_state = self.run_graph(graph, temporary)

        self.assertEqual(final_state["attempt"], 2)
        self.assertEqual(structured.invoke.call_count, 2)
        self.assertEqual(context_builder.call_count, 1)
        planner.assert_not_called()
        retry_prompt = structured.invoke.call_args_list[1].args[0][1].content
        self.assertIn("candidate attempt 2", retry_prompt)
        self.assertIn("unplanned file", retry_prompt)

    def test_schema_retry_is_bounded_and_reuses_same_context(self) -> None:
        invalid_payload = {
            "summary": "invalid action",
            "files": [
                {
                    "path": "src/repository.py",
                    "action": "write",
                    "content": "VALUE = 1\n",
                }
            ],
        }
        executor, structured = self.executor(invalid_payload, self.candidate)
        graph = build_plan_execution_graph(
            executor_model=executor,
            change_set_reviewer=MagicMock(return_value=self.review()),
        )
        with (
            tempfile.TemporaryDirectory() as temporary, patch(
                "plan_execution.analyze_code",
                return_value=StaticAnalysisResult(),
            ),
            patch(
                "plan_execution.execute_test_files",
                return_value=TestRunResult(status="passed"),
            ),
        ):
            final_state = self.run_graph(graph, temporary)

        self.assertEqual(final_state["attempt"], 2)
        self.assertEqual(structured.invoke.call_count, 2)

    def test_retry_exhaustion_is_controlled_before_workspace_or_review(self) -> None:
        invalid = MultiFileCandidate(
            summary="Missing a planned file.",
            files=[self.candidate.files[0]],
        )
        executor, structured = self.executor(invalid, invalid)
        reviewer = MagicMock()
        graph = build_plan_execution_graph(
            executor_model=executor,
            change_set_reviewer=reviewer,
        )
        with tempfile.TemporaryDirectory() as temporary:
            final_state = self.run_graph(
                graph,
                temporary,
                max_attempts=2,
            )

        self.assertIn("after 2 attempts", final_state["failure_reason"])
        self.assertIsNone(final_state["temporary_repository_root"])
        self.assertEqual(structured.invoke.call_count, 2)
        reviewer.assert_not_called()

    def test_change_set_reviewer_receives_temp_root_diff_and_no_tests(self) -> None:
        executor, _ = self.executor(self.candidate)
        observed: dict[str, object] = {}

        def reviewer(root: str, **arguments: object) -> ChangeSetReview:
            observed["root"] = root
            observed.update(arguments)
            observed["new_source"] = (
                Path(root) / "src" / "repository.py"
            ).read_text(encoding="utf-8")
            self.assertTrue(Path(root).is_dir())
            self.assertFalse((Path(root) / ".git").exists())
            self.assertFalse((Path(root) / ".env").exists())
            return self.review()

        graph = build_plan_execution_graph(
            executor_model=executor,
            change_set_reviewer=reviewer,
        )
        original = (self.root / "src" / "repository.py").read_bytes()
        with (
            tempfile.TemporaryDirectory() as temporary, patch(
                "plan_execution.analyze_code",
                return_value=StaticAnalysisResult(),
            ),
            patch(
                "plan_execution.execute_test_files",
                return_value=TestRunResult(status="passed"),
            ),
        ):
            self.run_graph(graph, temporary, agentic=True)

        self.assertNotEqual(observed["root"], str(self.root.resolve()))
        self.assertIn("casefold", observed["new_source"])
        self.assertIn("--- a/src/repository.py", observed["diff_text"])
        self.assertFalse(observed["run_tests"])
        self.assertTrue(observed["agentic_explore"])
        self.assertEqual(
            (self.root / "src" / "repository.py").read_bytes(),
            original,
        )

    def test_needs_work_does_not_regenerate_candidate(self) -> None:
        executor, structured = self.executor(self.candidate)
        reviewer = MagicMock(
            return_value=self.review(OverallRating.NEEDS_WORK)
        )
        graph = build_plan_execution_graph(
            executor_model=executor,
            change_set_reviewer=reviewer,
        )
        with (
            tempfile.TemporaryDirectory() as temporary, patch(
                "plan_execution.analyze_code",
                return_value=StaticAnalysisResult(),
            ),
            patch(
                "plan_execution.execute_test_files",
                return_value=TestRunResult(status="passed"),
            ),
        ):
            final_state = self.run_graph(graph, temporary)

        self.assertEqual(
            final_state["change_set_review"].overall_rating,
            OverallRating.NEEDS_WORK,
        )
        self.assertEqual(structured.invoke.call_count, 1)
        reviewer.assert_called_once()

    def test_failed_candidate_is_corrected_in_fresh_workspace(self) -> None:
        initial = self.buggy_candidate()
        executor, _ = self.executor(initial)
        corrector, corrector_structured = self.executor(self.candidate)
        roots: list[str] = []

        def reviewer(root: str, **_arguments: object) -> ChangeSetReview:
            roots.append(root)
            self.assertTrue(_arguments["agentic_explore"])
            marker = Path(root) / "round-leak.txt"
            if len(roots) == 1:
                marker.write_text("must not leak", encoding="utf-8")
                return self.review(OverallRating.NEEDS_WORK, concrete=True)
            self.assertFalse(marker.exists())
            return self.review()

        test_results = [
            TestRunResult(
                status="failed",
                stdout="first temporary workspace failed",
            ),
            TestRunResult(status="passed"),
        ]
        graph = build_plan_execution_graph(
            executor_model=executor,
            corrector_model=corrector,
            change_set_reviewer=reviewer,
        )
        fixed_plan = self.plan.model_dump()
        original = (self.root / "src" / "repository.py").read_bytes()
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "plan_execution.analyze_code",
                return_value=StaticAnalysisResult(),
            ) as analyzer,
            patch(
                "plan_execution.execute_test_files",
                side_effect=test_results,
            ) as tests,
            patch("plan_execution.plan_repository_task") as planner,
        ):
            final_state = self.run_graph(
                graph,
                temporary,
                max_corrections=1,
                agentic=True,
            )
            correction_prompt = (
                corrector_structured.invoke.call_args.args[0][1].content
            )
            self.assertNotIn(temporary, correction_prompt)

        self.assertEqual(final_state["correction_round"], 1)
        self.assertEqual(final_state["verification"].status, "verified")
        self.assertEqual(
            final_state["change_set_review"].overall_rating,
            OverallRating.GOOD,
        )
        self.assertEqual(len(final_state["candidate_history"]), 2)
        self.assertEqual(len(roots), 2)
        self.assertNotEqual(roots[0], roots[1])
        self.assertIn("round-0", roots[0])
        self.assertIn("round-1", roots[1])
        self.assertEqual(analyzer.call_count, 4)
        self.assertEqual(tests.call_count, 2)
        corrector.bind_tools.assert_not_called()
        self.assertIn("email.upper()", correction_prompt)
        self.assertIn("Planned test status: failed", correction_prompt)
        self.assertIn("Lookup normalization is incorrect", correction_prompt)
        self.assertNotIn("DO_NOT_SEND_THIS_MARKER", correction_prompt)
        self.assertEqual(self.plan.model_dump(), fixed_plan)
        planner.assert_not_called()
        self.assertEqual(
            (self.root / "src" / "repository.py").read_bytes(),
            original,
        )

    def test_review_finding_can_trigger_correction_after_verified_tests(
        self,
    ) -> None:
        executor, _ = self.executor(self.buggy_candidate())
        corrector, structured = self.executor(self.candidate)
        reviewer = MagicMock(
            side_effect=[
                self.review(OverallRating.NEEDS_WORK, concrete=True),
                self.review(),
            ]
        )
        graph = build_plan_execution_graph(
            executor_model=executor,
            corrector_model=corrector,
            change_set_reviewer=reviewer,
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "plan_execution.analyze_code",
                return_value=StaticAnalysisResult(),
            ),
            patch(
                "plan_execution.execute_test_files",
                return_value=TestRunResult(status="passed"),
            ),
        ):
            final_state = self.run_graph(
                graph,
                temporary,
                max_corrections=1,
            )

        self.assertEqual(final_state["verification"].status, "verified")
        self.assertEqual(final_state["correction_round"], 1)
        structured.invoke.assert_called_once()
        self.assertEqual(reviewer.call_count, 2)

    def test_critical_review_finding_triggers_correction(self) -> None:
        executor, _ = self.executor(self.buggy_candidate())
        corrector, structured = self.executor(self.candidate)
        reviewer = MagicMock(
            side_effect=[
                self.review(OverallRating.CRITICAL_ISSUES, concrete=True),
                self.review(),
            ]
        )
        graph = build_plan_execution_graph(
            executor_model=executor,
            corrector_model=corrector,
            change_set_reviewer=reviewer,
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "plan_execution.analyze_code",
                return_value=StaticAnalysisResult(),
            ),
            patch(
                "plan_execution.execute_test_files",
                return_value=TestRunResult(status="passed"),
            ),
        ):
            final_state = self.run_graph(
                graph,
                temporary,
                max_corrections=1,
            )

        self.assertEqual(final_state["correction_round"], 1)
        structured.invoke.assert_called_once()

    def test_blocking_static_finding_triggers_correction(self) -> None:
        blocking = StaticAnalysisResult(
            findings=[
                ToolFinding(
                    tool="ruff",
                    rule_id="F821",
                    category=ToolCategory.BUG,
                    severity=ToolSeverity.HIGH,
                    message="Undefined lookup key.",
                    line_number=4,
                )
            ]
        )
        executor, _ = self.executor(self.buggy_candidate())
        corrector, structured = self.executor(self.candidate)
        graph = build_plan_execution_graph(
            executor_model=executor,
            corrector_model=corrector,
            change_set_reviewer=MagicMock(return_value=self.review()),
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "plan_execution.analyze_code",
                side_effect=[
                    blocking,
                    StaticAnalysisResult(),
                    StaticAnalysisResult(),
                    StaticAnalysisResult(),
                ],
            ),
            patch(
                "plan_execution.execute_test_files",
                return_value=TestRunResult(status="passed"),
            ),
        ):
            final_state = self.run_graph(
                graph,
                temporary,
                max_corrections=1,
            )

        self.assertEqual(final_state["verification"].status, "verified")
        self.assertIn("ruff:F821", structured.invoke.call_args.args[0][1].content)

    def test_infrastructure_verification_errors_never_correct(self) -> None:
        cases = (
            (
                StaticAnalysisResult(),
                TestRunResult(status="timed_out"),
            ),
            (
                StaticAnalysisResult(tool_errors=["ruff unavailable"]),
                TestRunResult(status="passed"),
            ),
        )
        for static_result, test_result in cases:
            with self.subTest(test_status=test_result.status):
                executor, _ = self.executor(self.candidate)
                corrector, structured = self.executor(self.candidate)
                reviewer = MagicMock()
                graph = build_plan_execution_graph(
                    executor_model=executor,
                    corrector_model=corrector,
                    change_set_reviewer=reviewer,
                )
                with (
                    tempfile.TemporaryDirectory() as temporary,
                    patch(
                        "plan_execution.analyze_code",
                        return_value=static_result,
                    ),
                    patch(
                        "plan_execution.execute_test_files",
                        return_value=test_result,
                    ),
                ):
                    final_state = self.run_graph(
                        graph,
                        temporary,
                        max_corrections=2,
                    )

                self.assertEqual(final_state["verification"].status, "error")
                self.assertEqual(final_state["correction_round"], 0)
                self.assertEqual(len(final_state["candidate_history"]), 1)
                reviewer.assert_not_called()
                structured.invoke.assert_not_called()

    def test_review_execution_error_and_skipped_only_do_not_correct(
        self,
    ) -> None:
        skipped = ChangeSetReview(
            overall_rating=OverallRating.NEEDS_WORK,
            summary="No completed file reviews.",
            file_results=[
                FileReviewResult(
                    target=ChangeTarget(
                        path="src/repository.py",
                        change_kind="modified",
                    ),
                    status="skipped",
                    warnings=["Source was truncated."],
                )
            ],
        )
        for reviewer in (
            MagicMock(side_effect=RuntimeError("review backend failed")),
            MagicMock(return_value=skipped),
        ):
            with self.subTest(review_error=reviewer.side_effect is not None):
                executor, _ = self.executor(self.candidate)
                corrector, structured = self.executor(self.candidate)
                graph = build_plan_execution_graph(
                    executor_model=executor,
                    corrector_model=corrector,
                    change_set_reviewer=reviewer,
                )
                with (
                    tempfile.TemporaryDirectory() as temporary,
                    patch(
                        "plan_execution.analyze_code",
                        return_value=StaticAnalysisResult(),
                    ),
                    patch(
                        "plan_execution.execute_test_files",
                        return_value=TestRunResult(status="passed"),
                    ),
                ):
                    final_state = self.run_graph(
                        graph,
                        temporary,
                        max_corrections=2,
                    )

                self.assertEqual(final_state["correction_round"], 0)
                structured.invoke.assert_not_called()

    def test_corrected_candidate_reuses_all_deterministic_validation(
        self,
    ) -> None:
        missing = MultiFileCandidate(
            summary="Missing planned test.",
            files=[self.candidate.files[0]],
        )
        mismatch = MultiFileCandidate(
            summary="Wrong action.",
            files=[
                self.candidate.files[0].model_copy(update={"action": "add"}),
                self.candidate.files[1],
            ],
        )
        invalid_python = MultiFileCandidate(
            summary="Invalid Python.",
            files=[
                self.candidate.files[0].model_copy(
                    update={"content": "def broken(:\n"}
                ),
                self.candidate.files[1],
            ],
        )
        executor, _ = self.executor(self.buggy_candidate())
        corrector, structured = self.executor(
            missing,
            mismatch,
            invalid_python,
            self.candidate,
        )
        graph = build_plan_execution_graph(
            executor_model=executor,
            corrector_model=corrector,
            change_set_reviewer=MagicMock(return_value=self.review()),
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "plan_execution.analyze_code",
                return_value=StaticAnalysisResult(),
            ),
            patch(
                "plan_execution.execute_test_files",
                side_effect=[
                    TestRunResult(status="failed"),
                    TestRunResult(status="passed"),
                ],
            ),
        ):
            final_state = self.run_graph(
                graph,
                temporary,
                max_attempts=4,
                max_corrections=1,
            )

        self.assertEqual(final_state["attempt"], 4)
        self.assertEqual(structured.invoke.call_count, 4)
        validation_prompts = [
            call.args[0][1].content
            for call in structured.invoke.call_args_list[1:]
        ]
        self.assertTrue(any("missing planned file" in p for p in validation_prompts))
        self.assertTrue(any("action mismatch" in p for p in validation_prompts))
        self.assertTrue(any("could not be parsed" in p for p in validation_prompts))

    def test_needs_work_without_finding_is_not_corrected_when_enabled(
        self,
    ) -> None:
        executor, _ = self.executor(self.candidate)
        corrector, structured = self.executor(self.candidate)
        graph = build_plan_execution_graph(
            executor_model=executor,
            corrector_model=corrector,
            change_set_reviewer=MagicMock(
                return_value=self.review(OverallRating.NEEDS_WORK)
            ),
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "plan_execution.analyze_code",
                return_value=StaticAnalysisResult(),
            ),
            patch(
                "plan_execution.execute_test_files",
                return_value=TestRunResult(status="passed"),
            ),
        ):
            final_state = self.run_graph(
                graph,
                temporary,
                max_corrections=DEFAULT_MAX_CORRECTION_ROUNDS,
            )

        self.assertEqual(final_state["correction_round"], 0)
        self.assertIn("concrete implementation findings", final_state["failure_reason"])
        structured.invoke.assert_not_called()

    def test_correction_candidate_validation_retry_is_separate(self) -> None:
        invalid = MultiFileCandidate(
            summary="Unauthorized correction.",
            files=[
                *self.candidate.files,
                CandidateFileChange(
                    path="src/extra.py",
                    action="add",
                    content="VALUE = 1\n",
                ),
            ],
        )
        executor, _ = self.executor(self.buggy_candidate())
        corrector, structured = self.executor(invalid, self.candidate)
        graph = build_plan_execution_graph(
            executor_model=executor,
            corrector_model=corrector,
            change_set_reviewer=MagicMock(return_value=self.review()),
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "plan_execution.analyze_code",
                return_value=StaticAnalysisResult(),
            ),
            patch(
                "plan_execution.execute_test_files",
                side_effect=[
                    TestRunResult(status="failed"),
                    TestRunResult(status="passed"),
                ],
            ),
        ):
            final_state = self.run_graph(
                graph,
                temporary,
                max_attempts=2,
                max_corrections=1,
            )

        self.assertEqual(final_state["correction_round"], 1)
        self.assertEqual(final_state["attempt"], 2)
        self.assertEqual(structured.invoke.call_count, 2)
        retry_prompt = structured.invoke.call_args.args[0][1].content
        self.assertIn("generation attempt 2", retry_prompt)
        self.assertIn("unplanned file", retry_prompt)

    def test_two_corrections_can_verify_and_exhaustion_preserves_final(
        self,
    ) -> None:
        for final_status in ("passed", "failed"):
            with self.subTest(final_status=final_status):
                executor, _ = self.executor(self.buggy_candidate())
                v2 = self.buggy_candidate("lower")
                corrector, structured = self.executor(v2, self.candidate)
                reviewer = MagicMock(return_value=self.review())
                graph = build_plan_execution_graph(
                    executor_model=executor,
                    corrector_model=corrector,
                    change_set_reviewer=reviewer,
                )
                with (
                    tempfile.TemporaryDirectory() as temporary,
                    patch(
                        "plan_execution.analyze_code",
                        return_value=StaticAnalysisResult(),
                    ),
                    patch(
                        "plan_execution.execute_test_files",
                        side_effect=[
                            TestRunResult(status="failed"),
                            TestRunResult(status="failed"),
                            TestRunResult(status=final_status),
                        ],
                    ),
                ):
                    final_state = self.run_graph(
                        graph,
                        temporary,
                        max_corrections=2,
                    )

                self.assertEqual(final_state["correction_round"], 2)
                self.assertEqual(len(final_state["candidate_history"]), 3)
                self.assertEqual(structured.invoke.call_count, 2)
                self.assertEqual(reviewer.call_count, 3)
                self.assertIs(final_state["candidate"], self.candidate)
                self.assertIn("casefold", final_state["diff_text"])
                if final_status == "passed":
                    self.assertIsNone(final_state["failure_reason"])
                else:
                    self.assertIn(
                        "self-correction exhausted after 2 rounds",
                        final_state["failure_reason"],
                    )

    def test_verification_error_skips_change_set_review(self) -> None:
        executor, _ = self.executor(self.candidate)
        reviewer = MagicMock()
        graph = build_plan_execution_graph(
            executor_model=executor,
            change_set_reviewer=reviewer,
        )
        with (
            tempfile.TemporaryDirectory() as temporary, patch(
                "plan_execution.analyze_code",
                return_value=StaticAnalysisResult(
                    tool_errors=["ruff unavailable"]
                ),
            ),
            patch(
                "plan_execution.execute_test_files",
                return_value=TestRunResult(status="passed"),
            ),
        ):
            final_state = self.run_graph(graph, temporary)

        self.assertEqual(final_state["verification"].status, "error")
        self.assertIsNone(final_state["change_set_review"])
        reviewer.assert_not_called()

    def test_workspace_budget_failure_is_non_retryable(self) -> None:
        executor, structured = self.executor(self.candidate)
        reviewer = MagicMock()
        graph = build_plan_execution_graph(
            executor_model=executor,
            change_set_reviewer=reviewer,
        )
        with tempfile.TemporaryDirectory() as temporary:
            state = _initial_execution_state(
                self.root.resolve(),
                self.task,
                self.plan,
                temporary,
                max_execution_attempts=3,
                candidate_review_agentic_explore=False,
                max_workspace_files=1,
                max_workspace_bytes=100_000_000,
            )
            final_state = graph.invoke(state)

        self.assertIn("file budget", final_state["failure_reason"])
        self.assertEqual(structured.invoke.call_count, 1)
        reviewer.assert_not_called()

    def test_execute_engineering_plan_integration_and_original_immutability(
        self,
    ) -> None:
        executor, _ = self.executor(self.candidate)
        reviewer = MagicMock(return_value=self.review())
        graph = build_plan_execution_graph(
            executor_model=executor,
            change_set_reviewer=reviewer,
        )
        before = {
            path.relative_to(self.root).as_posix(): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }

        with (
            patch("plan_execution.plan_execution_graph", graph),
            patch(
                "plan_execution.analyze_code",
                return_value=StaticAnalysisResult(),
            ),
        ):
            result = execute_engineering_plan(str(self.root), self.task, self.plan)

        after = {
            path.relative_to(self.root).as_posix(): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(result.status, "verified", result.model_dump_json())
        self.assertEqual(result.verification.test_result.status, "passed")
        self.assertIn("src/repository.py", result.diff_text)
        self.assertEqual(after, before)
        reviewer.assert_called_once()

    def test_generation_failure_also_preserves_original_repository(self) -> None:
        invalid = MultiFileCandidate(
            summary="Missing a planned file.",
            files=[self.candidate.files[0]],
        )
        executor, _ = self.executor(invalid)
        graph = build_plan_execution_graph(
            executor_model=executor,
            change_set_reviewer=MagicMock(),
        )
        before = (self.root / "src" / "repository.py").read_bytes()

        with patch("plan_execution.plan_execution_graph", graph):
            result = execute_engineering_plan(
                str(self.root),
                self.task,
                self.plan,
                max_execution_attempts=1,
            )

        self.assertEqual(result.status, "failed")
        self.assertEqual(
            (self.root / "src" / "repository.py").read_bytes(),
            before,
        )

    def test_public_result_reports_correction_and_preserves_repository(
        self,
    ) -> None:
        executor, _ = self.executor(self.buggy_candidate())
        corrector, _ = self.executor(self.candidate)
        graph = build_plan_execution_graph(
            executor_model=executor,
            corrector_model=corrector,
            change_set_reviewer=MagicMock(return_value=self.review()),
        )
        before = {
            path.relative_to(self.root).as_posix(): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        with (
            patch("plan_execution.plan_execution_graph", graph),
            patch(
                "plan_execution.analyze_code",
                return_value=StaticAnalysisResult(),
            ),
            patch(
                "plan_execution.execute_test_files",
                side_effect=[
                    TestRunResult(status="failed"),
                    TestRunResult(status="passed"),
                ],
            ),
        ):
            result = execute_engineering_plan(
                str(self.root),
                self.task,
                self.plan,
                max_correction_rounds=1,
            )

        after = {
            path.relative_to(self.root).as_posix(): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(result.status, "verified")
        self.assertEqual(result.correction_rounds_used, 1)
        self.assertEqual(len(result.attempt_history), 2)
        self.assertEqual(before, after)
        self.assertNotIn(
            "code-review-plan-execution-",
            result.model_dump_json(),
        )

    def test_public_exhaustion_preserves_final_evidence_and_repository(
        self,
    ) -> None:
        final_candidate = self.buggy_candidate("lower")
        executor, _ = self.executor(self.buggy_candidate())
        corrector, _ = self.executor(final_candidate)
        final_review = self.review()
        graph = build_plan_execution_graph(
            executor_model=executor,
            corrector_model=corrector,
            change_set_reviewer=MagicMock(return_value=final_review),
        )
        before = (self.root / "src" / "repository.py").read_bytes()
        with (
            patch("plan_execution.plan_execution_graph", graph),
            patch(
                "plan_execution.analyze_code",
                return_value=StaticAnalysisResult(),
            ),
            patch(
                "plan_execution.execute_test_files",
                return_value=TestRunResult(status="failed"),
            ),
        ):
            result = execute_engineering_plan(
                str(self.root),
                self.task,
                self.plan,
                max_correction_rounds=1,
            )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.correction_rounds_used, 1)
        self.assertEqual(result.candidate, final_candidate)
        self.assertIn("email.lower()", result.diff_text)
        self.assertEqual(result.change_set_review, final_review)
        self.assertIn("exhausted after 1 rounds", " ".join(result.warnings))
        self.assertEqual(
            (self.root / "src" / "repository.py").read_bytes(),
            before,
        )

    def test_public_infrastructure_error_does_not_correct_or_write(
        self,
    ) -> None:
        executor, _ = self.executor(self.candidate)
        corrector, structured = self.executor(self.candidate)
        reviewer = MagicMock()
        graph = build_plan_execution_graph(
            executor_model=executor,
            corrector_model=corrector,
            change_set_reviewer=reviewer,
        )
        before = (self.root / "src" / "repository.py").read_bytes()
        with (
            patch("plan_execution.plan_execution_graph", graph),
            patch(
                "plan_execution.analyze_code",
                return_value=StaticAnalysisResult(
                    tool_errors=["bandit failed"]
                ),
            ),
            patch(
                "plan_execution.execute_test_files",
                return_value=TestRunResult(status="passed"),
            ),
        ):
            result = execute_engineering_plan(
                str(self.root),
                self.task,
                self.plan,
                max_correction_rounds=2,
            )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.correction_rounds_used, 0)
        self.assertEqual(len(result.attempt_history), 1)
        reviewer.assert_not_called()
        structured.invoke.assert_not_called()
        self.assertEqual(
            (self.root / "src" / "repository.py").read_bytes(),
            before,
        )

    def test_max_correction_rounds_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 0"):
            execute_engineering_plan(
                str(self.root),
                self.task,
                self.plan,
                max_correction_rounds=-1,
            )
        with (
            patch("plan_execution.plan_repository_task") as planner,
            self.assertRaisesRegex(ValueError, "at least 0"),
        ):
            plan_and_execute_repository_task(
                str(self.root),
                self.task,
                max_correction_rounds=-1,
            )
        planner.assert_not_called()

    def test_needs_work_public_result_fails_without_self_correction(self) -> None:
        executor, structured = self.executor(self.candidate)
        graph = build_plan_execution_graph(
            executor_model=executor,
            change_set_reviewer=MagicMock(
                return_value=self.review(OverallRating.NEEDS_WORK)
            ),
        )
        with (
            patch("plan_execution.plan_execution_graph", graph),
            patch(
                "plan_execution.analyze_code",
                return_value=StaticAnalysisResult(),
            ),
            patch(
                "plan_execution.execute_test_files",
                return_value=TestRunResult(status="passed"),
            ),
        ):
            result = execute_engineering_plan(
                str(self.root),
                self.task,
                self.plan,
            )

        self.assertEqual(result.status, "failed")
        self.assertEqual(structured.invoke.call_count, 1)

    def test_convenience_api_plans_once_and_executes_exact_plan(self) -> None:
        expected = MagicMock()
        with (
            patch(
                "plan_execution.plan_repository_task",
                return_value=self.plan,
            ) as planner,
            patch(
                "plan_execution.execute_engineering_plan",
                return_value=expected,
            ) as executor,
        ):
            result = plan_and_execute_repository_task(
                str(self.root),
                self.task,
                max_tool_calls=4,
                max_plan_retries=1,
                max_execution_attempts=2,
            )

        self.assertIs(result, expected)
        planner.assert_called_once_with(
            str(self.root),
            self.task,
            max_tool_calls=4,
            max_plan_retries=1,
        )
        executor.assert_called_once_with(
            str(self.root),
            self.task,
            self.plan,
            max_execution_attempts=2,
            candidate_review_agentic_explore=False,
        )

    def test_convenience_api_passes_explicit_correction_budget(self) -> None:
        with (
            patch(
                "plan_execution.plan_repository_task",
                return_value=self.plan,
            ) as planner,
            patch(
                "plan_execution.execute_engineering_plan",
                return_value=MagicMock(),
            ) as executor,
        ):
            plan_and_execute_repository_task(
                str(self.root),
                self.task,
                max_correction_rounds=2,
            )

        planner.assert_called_once()
        executor.assert_called_once_with(
            str(self.root),
            self.task,
            self.plan,
            max_execution_attempts=3,
            candidate_review_agentic_explore=False,
            max_correction_rounds=2,
        )


if __name__ == "__main__":
    unittest.main()
