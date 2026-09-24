"""Tests for the bounded read-only repository exploration subgraph."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import ValidationError

from fix_context import CodeFix
from fix_verification import FixVerificationResult
from repository_context import (
    RepoContext,
    RepositoryFile,
    build_repository_context,
)
from repository_exploration import (
    DEFAULT_MAX_TEST_TOOL_EXECUTIONS,
    RepositoryExplorationResult,
    explore_repository,
    repository_exploration_graph,
)
from review_models import (
    CodeReview,
    FindingCategory,
    OverallRating,
    ReviewFinding,
    Severity,
)
from static_analysis import StaticAnalysisResult
from test_execution import TestRunResult


def tool_call(name: str, arguments: dict[str, object], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": arguments,
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def completed_message(content: str = "Exploration complete.") -> AIMessage:
    return AIMessage(content=content)


def good_review() -> CodeReview:
    return CodeReview(
        overall_rating=OverallRating.GOOD,
        summary="No concrete issue found.",
        findings=[],
    )


def risky_review() -> CodeReview:
    return CodeReview(
        overall_rating=OverallRating.NEEDS_WORK,
        summary="Missing names can fail at runtime.",
        findings=[
            ReviewFinding(
                category=FindingCategory.BUG,
                severity=Severity.HIGH,
                title="Undefined name",
                description="The returned name is undefined.",
                line_number=2,
                suggestion="Return a defined value.",
            )
        ],
    )


class RepositoryExplorationGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "src").mkdir()
        (self.root / "tests").mkdir()
        (self.root / "src" / "service.py").write_text(
            "user = repository.find_by_email(email)\n"
            "return user.name\n",
            encoding="utf-8",
        )
        (self.root / "src" / "repository.py").write_text(
            "def find_by_email(email):\n"
            "    return USERS.get(email)\n",
            encoding="utf-8",
        )
        (self.root / "tests" / "test_service.py").write_text(
            "def test_service():\n"
            "    assert find_by_email('a@example.com')\n",
            encoding="utf-8",
        )
        (self.root / "tests" / "unrelated_test.py").write_text(
            "def test_unrelated():\n    assert True\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def model_with(self, *responses: AIMessage) -> tuple[MagicMock, MagicMock]:
        model = MagicMock()
        bound = model.bind_tools.return_value
        bound.invoke.side_effect = list(responses)
        return model, bound

    def test_no_tool_call_completes_without_observations(self) -> None:
        model, _ = self.model_with(completed_message("No more context needed."))
        result = explore_repository(
            str(self.root),
            "deterministic context",
            model=model,
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.tool_call_count, 0)
        self.assertIn("No more context needed.", result.summary)
        self.assertEqual(result.files_read, [])
        self.assertEqual(result.searches, [])

    def test_search_read_list_loop_returns_observations_to_model(self) -> None:
        model, bound = self.model_with(
            tool_call(
                "search_repository_code",
                {"query": "find_by_email"},
                "search-1",
            ),
            tool_call(
                "list_repository_files",
                {"prefix": "src"},
                "list-1",
            ),
            tool_call(
                "read_repository_file",
                {"path": "src/repository.py"},
                "read-1",
            ),
            completed_message("The repository lookup can return None."),
        )
        result = explore_repository(
            str(self.root),
            "deterministic context",
            model=model,
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.tool_call_count, 3)
        self.assertEqual(result.searches, ["find_by_email"])
        self.assertEqual(result.files_read, ["src/repository.py"])
        self.assertIn("def find_by_email", result.summary)
        self.assertIn("src/service.py", result.summary)
        self.assertIn("UNTRUSTED REPOSITORY CONTENT", result.summary)
        second_messages = bound.invoke.call_args_list[1].args[0]
        self.assertIsInstance(second_messages[-1], ToolMessage)
        self.assertIn("find_by_email", second_messages[-1].content)

    def test_hard_tool_execution_budget_stops_extra_calls(self) -> None:
        model = MagicMock()
        bound = model.bind_tools.return_value
        bound.invoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_repository_file",
                    "args": {"path": "src/service.py"},
                    "id": "read-1",
                    "type": "tool_call",
                },
                {
                    "name": "read_repository_file",
                    "args": {"path": "src/repository.py"},
                    "id": "read-2",
                    "type": "tool_call",
                },
                {
                    "name": "search_repository_code",
                    "args": {"query": "find_by_email"},
                    "id": "search-3",
                    "type": "tool_call",
                },
            ],
        )
        result = explore_repository(
            str(self.root),
            "context",
            model=model,
            max_tool_calls=2,
        )
        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual(result.tool_call_count, 2)
        self.assertEqual(bound.invoke.call_count, 1)

    def test_duplicate_tool_call_is_cached_and_runtime_is_bounded(self) -> None:
        model, _ = self.model_with(
            tool_call(
                "read_repository_file",
                {"path": "src/repository.py"},
                "read-1",
            ),
            tool_call(
                "read_repository_file",
                {"path": "src\\repository.py"},
                "read-2",
            ),
            completed_message(),
        )
        result = explore_repository(
            str(self.root),
            "context",
            model=model,
            max_tool_calls=2,
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.tool_call_count, 1)
        self.assertEqual(result.files_read, ["src/repository.py"])
        self.assertTrue(any("cache" in warning for warning in result.warnings))

    def test_repeated_cached_calls_stop_at_request_budget(self) -> None:
        model = MagicMock()
        bound = model.bind_tools.return_value
        call_index = 0

        def repeated(_messages):
            nonlocal call_index
            call_index += 1
            return tool_call(
                "read_repository_file",
                {"path": "src/service.py"},
                f"read-{call_index}",
            )

        bound.invoke.side_effect = repeated
        result = explore_repository(
            str(self.root),
            "context",
            model=model,
            max_tool_calls=2,
        )
        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual(result.tool_call_count, 1)
        self.assertEqual(bound.invoke.call_count, 4)

    def test_final_evidence_budget_is_explicit(self) -> None:
        model, _ = self.model_with(
            tool_call(
                "read_repository_file",
                {"path": "src/repository.py"},
                "read-1",
            ),
            completed_message("x" * 1000),
        )
        with patch(
            "repository_exploration.MAX_EXPLORATION_EVIDENCE_CHARS",
            100,
        ):
            result = explore_repository(
                str(self.root),
                "context",
                model=model,
            )
        self.assertEqual(result.status, "budget_exhausted")
        self.assertIn("TRUNCATED", result.summary)
        self.assertLessEqual(len(result.summary), 100)
        self.assertTrue(any("truncated" in warning for warning in result.warnings))

    def test_model_error_degrades_to_bounded_error_result(self) -> None:
        model = MagicMock()
        model.bind_tools.return_value.invoke.side_effect = RuntimeError("offline")
        result = explore_repository(str(self.root), "context", model=model)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.tool_call_count, 0)
        self.assertIn("offline", result.warnings[0])

    def test_result_schema_forbids_unbounded_extra_state(self) -> None:
        payload = RepositoryExplorationResult(status="completed").model_dump()
        payload["messages"] = []
        with self.assertRaises(ValidationError):
            RepositoryExplorationResult.model_validate(payload)

    def test_graph_has_explicit_toolnode_sync_and_finalize_topology(self) -> None:
        model = MagicMock()
        graph = repository_exploration_graph(str(self.root), model=model)
        nodes = set(graph.get_graph().nodes)
        self.assertTrue(
            {
                "exploration_agent",
                "tools",
                "sync_tool_counters",
                "finalize_exploration",
            }.issubset(nodes)
        )

    def test_invalid_tool_arguments_become_observations_not_repo_escape(self) -> None:
        model, bound = self.model_with(
            tool_call(
                "read_repository_file",
                {"path": "../outside.py"},
                "read-1",
            ),
            completed_message(),
        )
        result = explore_repository(str(self.root), "context", model=model)
        self.assertEqual(result.tool_call_count, 1)
        self.assertTrue(any("error" in warning for warning in result.warnings))
        followup_messages = bound.invoke.call_args_list[1].args[0]
        self.assertEqual(followup_messages[-1].status, "error")
        self.assertIn("traversal", followup_messages[-1].content)

    def test_test_tool_is_absent_without_allowed_files_and_present_with_them(
        self,
    ) -> None:
        model_without = MagicMock()
        repository_exploration_graph(str(self.root), model=model_without)
        without_names = {
            tool.name for tool in model_without.bind_tools.call_args.args[0]
        }
        self.assertEqual(
            without_names,
            {
                "read_repository_file",
                "search_repository_code",
                "list_repository_files",
            },
        )

        model_with = MagicMock()
        repository_exploration_graph(
            str(self.root),
            allowed_test_files=["tests/test_service.py"],
            model=model_with,
        )
        with_names = {tool.name for tool in model_with.bind_tools.call_args.args[0]}
        self.assertEqual(with_names, without_names | {"run_repository_test"})

    @patch("repository_test_tool.execute_test_files")
    def test_search_read_test_integration_keeps_full_add_messages_history(
        self,
        executor,
    ) -> None:
        executor.return_value = TestRunResult(
            status="passed",
            framework="pytest",
            test_files=["tests/test_service.py"],
            exit_code=0,
            stdout="1 passed",
        )
        model, bound = self.model_with(
            tool_call(
                "search_repository_code",
                {"query": "find_by_email"},
                "search-1",
            ),
            tool_call(
                "read_repository_file",
                {"path": "src/repository.py"},
                "read-1",
            ),
            tool_call(
                "run_repository_test",
                {"test_file": "tests/test_service.py"},
                "test-1",
            ),
            completed_message("Runtime evidence confirms the hypothesis."),
        )
        repository_context = build_repository_context(
            str(self.root),
            "src/service.py",
        )
        allowed_test_files = [
            related.path
            for related in repository_context.related_files
            if related.relationship == "test"
        ]
        self.assertEqual(allowed_test_files, ["tests/test_service.py"])

        result = explore_repository(
            str(self.root),
            "deterministic context",
            allowed_test_files=allowed_test_files,
            model=model,
        )

        self.assertEqual(result.tool_call_count, 3)
        self.assertEqual(result.test_tool_call_count, 1)
        self.assertEqual(result.tests_run, ["tests/test_service.py"])
        self.assertIn("UNTRUSTED TEST EXECUTION OUTPUT", result.summary)
        self.assertNotIn("unrelated_test.py", result.tests_run)
        executor.assert_called_once()

        fourth_messages = bound.invoke.call_args_list[3].args[0]
        self.assertIsInstance(fourth_messages[0], SystemMessage)
        self.assertIsInstance(fourth_messages[1], HumanMessage)
        self.assertEqual(
            [type(message) for message in fourth_messages[2:]],
            [AIMessage, ToolMessage, AIMessage, ToolMessage, AIMessage, ToolMessage],
        )
        self.assertIn("1 passed", fourth_messages[-1].content)

    @patch("repository_test_tool.execute_test_files")
    def test_duplicate_normalized_test_uses_cache_without_new_test_count(
        self,
        executor,
    ) -> None:
        executor.return_value = TestRunResult(status="passed", exit_code=0)
        model, _ = self.model_with(
            tool_call(
                "run_repository_test",
                {"test_file": "tests/test_service.py"},
                "test-1",
            ),
            tool_call(
                "run_repository_test",
                {"test_file": "tests\\test_service.py"},
                "test-2",
            ),
            completed_message(),
        )
        result = explore_repository(
            str(self.root),
            "context",
            allowed_test_files=["tests/test_service.py"],
            model=model,
        )
        self.assertEqual(result.tool_call_count, 1)
        self.assertEqual(result.test_tool_call_count, 1)
        self.assertEqual(result.tests_run, ["tests/test_service.py"])
        executor.assert_called_once()
        self.assertTrue(any("cache" in warning for warning in result.warnings))

    @patch("repository_test_tool.execute_test_files")
    def test_existing_unrelated_test_is_authorization_rejected_and_never_run(
        self,
        executor,
    ) -> None:
        model, bound = self.model_with(
            tool_call(
                "run_repository_test",
                {"test_file": "tests/unrelated_test.py"},
                "test-unrelated",
            ),
            completed_message("The unrelated test was not authorized."),
        )
        result = explore_repository(
            str(self.root),
            "context",
            allowed_test_files=["tests/test_service.py"],
            model=model,
        )
        executor.assert_not_called()
        self.assertEqual(result.test_tool_call_count, 0)
        self.assertEqual(result.tests_run, [])
        followup_messages = bound.invoke.call_args_list[1].args[0]
        self.assertEqual(followup_messages[-1].status, "error")
        self.assertIn("allowlist", followup_messages[-1].content)

    @patch("repository_test_tool.execute_test_files")
    def test_third_unique_test_is_rejected_by_independent_budget(
        self,
        executor,
    ) -> None:
        allowed = ["tests/test_service.py"]
        for index in (2, 3):
            path = f"tests/test_service_{index}.py"
            (self.root / path).write_text(
                f"def test_service_{index}():\n    assert True\n",
                encoding="utf-8",
            )
            allowed.append(path)
        executor.return_value = TestRunResult(status="passed", exit_code=0)
        model, bound = self.model_with(
            *(
                tool_call(
                    "run_repository_test",
                    {"test_file": path},
                    f"test-{index}",
                )
                for index, path in enumerate(allowed, start=1)
            )
        )
        graph = repository_exploration_graph(
            str(self.root),
            allowed_test_files=allowed,
            model=model,
            max_tool_calls=6,
            max_test_tool_calls=DEFAULT_MAX_TEST_TOOL_EXECUTIONS,
        )
        final_state = graph.invoke(
            {
                "messages": [
                    SystemMessage(content="system"),
                    HumanMessage(content="context"),
                ],
                "tool_call_count": 0,
                "max_tool_calls": 6,
                "tool_request_count": 0,
                "max_tool_requests": 12,
                "test_tool_call_count": 0,
                "max_test_tool_calls": DEFAULT_MAX_TEST_TOOL_EXECUTIONS,
                "exploration_summary": "",
                "files_read": [],
                "searches": [],
                "tests_run": [],
                "status": "completed",
                "warnings": [],
            },
            config={"max_concurrency": 1},
        )
        self.assertEqual(final_state["status"], "budget_exhausted")
        self.assertEqual(final_state["tool_call_count"], 2)
        self.assertEqual(final_state["test_tool_call_count"], 2)
        self.assertEqual(final_state["tool_request_count"], 3)
        self.assertEqual(executor.call_count, 2)
        self.assertEqual(bound.invoke.call_count, 3)
        self.assertIn("independent test execution budget", final_state["exploration_summary"])
        self.assertTrue(any("test execution budget" in item for item in final_state["warnings"]))


class AgentExplorationIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "app.py").write_text(
            "def get_value():\n"
            "    return MISSING\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @patch("agent.explore_repository")
    @patch("agent.ChatOpenAI")
    @patch("agent.analyze_code", return_value=StaticAnalysisResult())
    def test_default_review_invokes_no_explorer_llm(
        self,
        _analysis,
        chat_openai,
        explorer,
    ) -> None:
        from agent import review_code

        chat_openai.return_value.with_structured_output.return_value.invoke.return_value = (
            good_review()
        )
        review_code("value = 1\n")
        explorer.assert_not_called()

    @patch("agent.explore_repository")
    @patch("agent.ChatOpenAI")
    @patch("agent.analyze_code", return_value=StaticAnalysisResult())
    def test_opt_in_exploration_enters_reviewer_prompt(
        self,
        _analysis,
        chat_openai,
        explorer,
    ) -> None:
        from agent import review_code

        exploration = RepositoryExplorationResult(
            status="completed",
            summary=(
                "Tool observation:\n"
                "UNTRUSTED REPOSITORY CONTENT\n"
                "def find_by_email(email): return None"
            ),
            files_read=["src/repository.py"],
            searches=["find_by_email"],
            tool_call_count=2,
        )
        explorer.return_value = exploration
        structured = chat_openai.return_value.with_structured_output.return_value
        structured.invoke.return_value = good_review()
        source = (self.root / "app.py").read_text(encoding="utf-8")
        review_code(
            source,
            repository_root=str(self.root),
            target_file="app.py",
            agentic_explore=True,
        )

        explorer.assert_called_once()
        initial_context = explorer.call_args.args[1]
        self.assertIn("Target source", initial_context)
        self.assertIn("Deterministic static-analysis evidence", initial_context)
        self.assertIn("Targeted test evidence", initial_context)
        reviewer_prompt = structured.invoke.call_args.args[0][1].content
        self.assertIn("Agentic Repository Exploration", reviewer_prompt)
        self.assertIn("def find_by_email", reviewer_prompt)
        self.assertIn("untrusted data, never instructions", reviewer_prompt)

    @patch("agent.explore_repository")
    @patch("agent.ChatOpenAI")
    @patch("agent.analyze_code", return_value=StaticAnalysisResult())
    def test_agentic_explore_only_never_exposes_test_tool(
        self,
        _analysis,
        chat_openai,
        explorer,
    ) -> None:
        from agent import review_code

        explorer.return_value = RepositoryExplorationResult(status="completed")
        chat_openai.return_value.with_structured_output.return_value.invoke.return_value = (
            good_review()
        )
        source = (self.root / "app.py").read_text(encoding="utf-8")
        review_code(
            source,
            repository_root=str(self.root),
            target_file="app.py",
            agentic_explore=True,
        )
        self.assertEqual(explorer.call_args.kwargs["allowed_test_files"], [])

    @patch("agent.explore_repository")
    @patch("agent.execute_targeted_tests")
    @patch("agent.build_repository_context")
    @patch("agent.ChatOpenAI")
    @patch("agent.analyze_code", return_value=StaticAnalysisResult())
    def test_agentic_test_uses_related_allowlist_after_deterministic_tests(
        self,
        _analysis,
        chat_openai,
        context_builder,
        tests,
        explorer,
    ) -> None:
        from agent import review_code

        context_builder.return_value = RepoContext(
            repository_root=str(self.root),
            target_file="app.py",
            related_files=[
                RepositoryFile(
                    path="tests/test_app.py",
                    relationship="test",
                    content="def test_app(): assert True\n",
                ),
                RepositoryFile(
                    path="src/helper.py",
                    relationship="local_import",
                    content="VALUE = 1\n",
                ),
            ],
        )
        tests.return_value = TestRunResult(
            status="passed",
            framework="pytest",
            test_files=["tests/test_app.py"],
        )
        explorer.return_value = RepositoryExplorationResult(
            status="completed",
            tests_run=["tests/test_app.py"],
            test_tool_call_count=1,
        )
        chat_openai.return_value.with_structured_output.return_value.invoke.return_value = (
            good_review()
        )
        calls = MagicMock()
        calls.attach_mock(tests, "deterministic_tests")
        calls.attach_mock(explorer, "explorer")

        source = (self.root / "app.py").read_text(encoding="utf-8")
        review_code(
            source,
            repository_root=str(self.root),
            target_file="app.py",
            run_tests=True,
            agentic_explore=True,
            agentic_test=True,
        )

        self.assertEqual(
            [call[0] for call in calls.method_calls],
            ["deterministic_tests", "explorer"],
        )
        self.assertEqual(
            explorer.call_args.kwargs["allowed_test_files"],
            ["tests/test_app.py"],
        )
        initial_context = explorer.call_args.args[1]
        self.assertIn("Allowed agentic test files:", initial_context)
        self.assertIn("tests/test_app.py", initial_context)
        self.assertNotIn("src/helper.py\nYou already", initial_context)
        self.assertIn("You already have deterministic test evidence", initial_context)

    @patch("agent.explore_repository")
    @patch("agent.ChatOpenAI")
    @patch("agent.analyze_code", return_value=StaticAnalysisResult())
    def test_semantic_review_retry_does_not_rerun_exploration(
        self,
        _analysis,
        chat_openai,
        explorer,
    ) -> None:
        from agent import review_code

        explorer.return_value = RepositoryExplorationResult(
            status="completed",
            summary="bounded evidence",
        )
        invalid = CodeReview(
            overall_rating=OverallRating.GOOD,
            summary="Incorrect rating.",
            findings=risky_review().findings,
        )
        structured = chat_openai.return_value.with_structured_output.return_value
        structured.invoke.side_effect = [invalid, risky_review()]
        source = (self.root / "app.py").read_text(encoding="utf-8")
        review_code(
            source,
            repository_root=str(self.root),
            target_file="app.py",
            run_tests=True,
            agentic_explore=True,
            agentic_test=True,
        )
        explorer.assert_called_once()
        retry_prompt = structured.invoke.call_args_list[1].args[0][1].content
        self.assertIn("bounded evidence", retry_prompt)
        self.assertIn("semantic retry 1", retry_prompt)

    @patch("agent.explore_repository")
    @patch("agent.verify_candidate_fix")
    @patch("agent.execute_targeted_tests")
    @patch("agent.ChatOpenAI")
    @patch("agent.analyze_code", return_value=StaticAnalysisResult())
    def test_fix_retry_reuses_exploration_and_fixer_has_no_tools(
        self,
        _analysis,
        chat_openai,
        tests,
        verifier,
        explorer,
    ) -> None:
        from agent import review_and_fix

        explorer.return_value = RepositoryExplorationResult(
            status="completed",
            summary="read-only extra evidence",
            tool_call_count=1,
        )
        tests.return_value = TestRunResult(status="passed", framework="pytest")
        verifier.return_value = FixVerificationResult(status="verified")
        invalid_fix = CodeFix(
            summary="First attempt.",
            addressed_findings=["Undefined name"],
            updated_code="def broken(:\n",
        )
        valid_fix = CodeFix(
            summary="Return a defined value.",
            addressed_findings=["Undefined name"],
            updated_code="def get_value():\n    return None\n",
        )
        structured = chat_openai.return_value.with_structured_output.return_value
        structured.invoke.side_effect = [risky_review(), invalid_fix, valid_fix]
        source = (self.root / "app.py").read_text(encoding="utf-8")
        result = review_and_fix(
            source,
            repository_root=str(self.root),
            target_file="app.py",
            run_tests=True,
            agentic_explore=True,
            agentic_test=True,
        )

        self.assertEqual(result.fix_status, "verified")
        explorer.assert_called_once()
        self.assertEqual(explorer.call_args.kwargs["allowed_test_files"], [])
        self.assertEqual(structured.invoke.call_count, 3)
        fixer_retry_prompt = structured.invoke.call_args_list[2].args[0][1].content
        self.assertIn("read-only extra evidence", fixer_retry_prompt)
        self.assertFalse(chat_openai.return_value.bind_tools.called)

    def test_agentic_api_validation_requires_repo_target_and_python(self) -> None:
        from agent import review_code

        with self.assertRaisesRegex(ValueError, "repository_root and target_file"):
            review_code("value = 1", agentic_explore=True)
        with self.assertRaisesRegex(ValueError, "language='python'"):
            review_code(
                "value = 1",
                language="javascript",
                repository_root=str(self.root),
                target_file="app.py",
                agentic_explore=True,
            )

    def test_agentic_test_api_requires_exploration_tests_repo_and_python(self) -> None:
        from agent import review_code

        source = (self.root / "app.py").read_text(encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "agentic_explore"):
            review_code(
                source,
                repository_root=str(self.root),
                target_file="app.py",
                run_tests=True,
                agentic_test=True,
            )
        with self.assertRaisesRegex(ValueError, "run_tests"):
            review_code(
                source,
                repository_root=str(self.root),
                target_file="app.py",
                agentic_explore=True,
                agentic_test=True,
            )
        with self.assertRaisesRegex(ValueError, "repository_root and target_file"):
            review_code(
                source,
                run_tests=True,
                agentic_explore=True,
                agentic_test=True,
            )
        with self.assertRaisesRegex(ValueError, "language='python'"):
            review_code(
                source,
                language="javascript",
                repository_root=str(self.root),
                target_file="app.py",
                run_tests=True,
                agentic_explore=True,
                agentic_test=True,
            )


if __name__ == "__main__":
    unittest.main()
