"""Graph, retry, and integration tests for task-aware planning."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from agent import change_set_graph, review_graph
from engineering_plan import (
    MAX_TASK_CHARS,
    EngineeringPlan,
    EngineeringPlanState,
    EngineeringPlanValidationError,
    PlannedFileChange,
    PlannedTest,
    build_engineering_plan_graph,
    engineering_plan_graph,
    plan_repository_task,
)
from repository_exploration import (
    RepositoryExplorationResult,
    repository_exploration_graph,
)


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


class PlanningGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "src").mkdir()
        (self.root / "tests").mkdir()
        (self.root / "src" / "service.py").write_text(
            "def get_user(email, repository):\n"
            "    return repository.find_by_email(email)\n",
            encoding="utf-8",
        )
        (self.root / "src" / "repository.py").write_text(
            "def find_by_email(email):\n"
            "    return USERS.get(email)\n",
            encoding="utf-8",
        )
        (self.root / "tests" / "test_service.py").write_text(
            "def test_service():\n    assert True\n",
            encoding="utf-8",
        )
        self.task = "Fix user lookup so email matching is case-insensitive."

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def explorer_model(self) -> tuple[MagicMock, MagicMock]:
        model = MagicMock()
        bound = model.bind_tools.return_value
        bound.invoke.side_effect = [
            tool_call(
                "search_repository_code",
                {"query": "find_by_email"},
                "search",
            ),
            tool_call(
                "read_repository_file",
                {"path": "src/service.py"},
                "read-service",
            ),
            tool_call(
                "read_repository_file",
                {"path": "src/repository.py"},
                "read-repository",
            ),
            tool_call(
                "list_repository_files",
                {"prefix": "tests"},
                "list-tests",
            ),
            completed_message("Repository lookup and tests located."),
        ]
        return model, bound

    def good_plan(self) -> EngineeringPlan:
        return EngineeringPlan(
            summary="Normalize email keys at the repository lookup boundary.",
            files=[
                PlannedFileChange(
                    path="src/repository.py",
                    action="modify",
                    rationale="Make email matching case-insensitive.",
                ),
                PlannedFileChange(
                    path="tests/test_repository.py",
                    action="add",
                    rationale="Add focused repository lookup coverage.",
                ),
            ],
            tests=[
                PlannedTest(
                    path="tests/test_repository.py",
                    purpose="Verify mixed-case email lookup behavior.",
                )
            ],
            risks=["Existing stored email-key normalization may differ."],
            assumptions=["Email identity is intended to be case-insensitive."],
        )

    def planner_model(
        self,
        *plans: EngineeringPlan,
    ) -> tuple[MagicMock, MagicMock]:
        model = MagicMock()
        structured = model.with_structured_output.return_value
        structured.invoke.side_effect = list(plans)
        return model, structured

    def initial_state(
        self,
        *,
        allow_test_verification: bool = False,
        max_retries: int = 2,
    ) -> EngineeringPlanState:
        return {
            "repository_root": str(self.root.resolve()),
            "task": self.task,
            "allow_test_verification": allow_test_verification,
            "max_tool_calls": 6,
            "exploration_result": RepositoryExplorationResult(
                status="not_requested"
            ),
            "plan": None,
            "validation_errors": [],
            "retry_count": 0,
            "max_retries": max_retries,
            "warnings": [],
            "failure_reason": None,
        }

    def test_planning_graph_has_independent_expected_topology(self) -> None:
        graph = engineering_plan_graph.get_graph()

        self.assertTrue(
            {
                "run_task_repository_exploration",
                "generate_engineering_plan",
                "validate_engineering_plan",
                "prepare_plan_retry",
                "controlled_plan_failure",
            }.issubset(graph.nodes)
        )

    def test_existing_three_graph_topologies_remain_distinct(self) -> None:
        self.assertIn("generate_review", review_graph.get_graph().nodes)
        self.assertIn(
            "generate_change_set_summary",
            change_set_graph.get_graph().nodes,
        )
        with patch("repository_exploration.ChatOpenAI"):
            explorer_graph = repository_exploration_graph(str(self.root))
        self.assertIn("exploration_agent", explorer_graph.get_graph().nodes)
        self.assertNotIn(
            "generate_engineering_plan",
            review_graph.get_graph().nodes,
        )

    def test_task_explorer_uses_only_read_search_and_list_tools(self) -> None:
        explorer, _ = self.explorer_model()
        planner, _ = self.planner_model(self.good_plan())
        graph = build_engineering_plan_graph(
            explorer_model=explorer,
            planner_model=planner,
        )

        graph.invoke(self.initial_state())

        tools = explorer.bind_tools.call_args.args[0]
        self.assertEqual(
            [tool.name for tool in tools],
            [
                "read_repository_file",
                "search_repository_code",
                "list_repository_files",
            ],
        )
        self.assertNotIn("run_repository_test", [tool.name for tool in tools])

    def test_task_and_bounded_evidence_reach_separate_planner(self) -> None:
        explorer, bound = self.explorer_model()
        planner, structured = self.planner_model(self.good_plan())
        graph = build_engineering_plan_graph(
            explorer_model=explorer,
            planner_model=planner,
        )

        final_state = graph.invoke(self.initial_state())

        explorer_messages = bound.invoke.call_args_list[0].args[0]
        self.assertIn(self.task, explorer_messages[1].content)
        planner_messages = structured.invoke.call_args.args[0]
        planner_prompt = planner_messages[1].content
        self.assertIn(self.task, planner_prompt)
        self.assertIn('"files_read"', planner_prompt)
        self.assertIn('"search_result_files"', planner_prompt)
        self.assertNotIn('"messages"', planner_prompt)
        self.assertNotIn("tool_call_id", planner_prompt)
        result = final_state["exploration_result"]
        self.assertIn("src/repository.py", result.search_result_files)

    def test_integration_plan_is_grounded_and_repository_is_unchanged(self) -> None:
        before = {
            path.relative_to(self.root).as_posix(): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        explorer, _ = self.explorer_model()
        planner, _ = self.planner_model(self.good_plan())
        graph = build_engineering_plan_graph(
            explorer_model=explorer,
            planner_model=planner,
        )

        with patch("engineering_plan.engineering_plan_graph", graph):
            result = plan_repository_task(str(self.root), self.task)

        after = {
            path.relative_to(self.root).as_posix(): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(result, self.good_plan())
        self.assertEqual(after, before)

    def test_semantic_retry_reuses_exploration_without_tool_rerun(self) -> None:
        bad_plan = EngineeringPlan(
            summary="Invent a target.",
            files=[
                PlannedFileChange(
                    path="src/payment_magic.py",
                    action="modify",
                    rationale="Invented path.",
                )
            ],
        )
        explorer, bound = self.explorer_model()
        planner, structured = self.planner_model(bad_plan, self.good_plan())
        graph = build_engineering_plan_graph(
            explorer_model=explorer,
            planner_model=planner,
        )

        final_state = graph.invoke(self.initial_state())

        self.assertEqual(final_state["plan"], self.good_plan())
        self.assertEqual(final_state["retry_count"], 1)
        self.assertEqual(bound.invoke.call_count, 5)
        self.assertEqual(explorer.bind_tools.call_count, 1)
        self.assertEqual(structured.invoke.call_count, 2)
        retry_prompt = structured.invoke.call_args_list[1].args[0][1].content
        self.assertIn("semantic retry 1", retry_prompt)
        self.assertIn("payment_magic.py", retry_prompt)
        self.assertIn('"files_read"', retry_prompt)

    def test_retry_exhaustion_raises_controlled_failure(self) -> None:
        bad_plan = EngineeringPlan(
            summary="Invent a target.",
            files=[
                PlannedFileChange(
                    path="src/missing.py",
                    action="modify",
                    rationale="Invented path.",
                )
            ],
        )
        explorer, bound = self.explorer_model()
        planner, structured = self.planner_model(bad_plan, bad_plan)
        graph = build_engineering_plan_graph(
            explorer_model=explorer,
            planner_model=planner,
        )

        with (
            patch("engineering_plan.engineering_plan_graph", graph),
            self.assertRaises(EngineeringPlanValidationError) as raised,
        ):
            plan_repository_task(
                str(self.root),
                self.task,
                max_plan_retries=1,
            )

        self.assertEqual(raised.exception.retry_count, 1)
        self.assertTrue(raised.exception.validation_errors)
        self.assertEqual(bound.invoke.call_count, 5)
        self.assertEqual(structured.invoke.call_count, 2)
        payload = raised.exception.to_payload()
        self.assertEqual(
            payload["error"]["type"],
            "engineering_plan_validation_failed",
        )

    def test_test_verification_opt_in_does_not_expose_a_test_tool(self) -> None:
        explorer, _ = self.explorer_model()
        planner, _ = self.planner_model(self.good_plan())
        graph = build_engineering_plan_graph(
            explorer_model=explorer,
            planner_model=planner,
        )

        final_state = graph.invoke(
            self.initial_state(allow_test_verification=True)
        )

        tools = explorer.bind_tools.call_args.args[0]
        self.assertNotIn("run_repository_test", [tool.name for tool in tools])
        self.assertTrue(
            any("not enabled" in warning for warning in final_state["warnings"])
        )

    def test_empty_and_oversized_tasks_fail_before_graph_invocation(self) -> None:
        for task in ("   ", "x" * (MAX_TASK_CHARS + 1)):
            with (
                self.subTest(length=len(task)),
                patch("engineering_plan.engineering_plan_graph.invoke") as invoke,
                self.assertRaises(ValueError),
            ):
                plan_repository_task(str(self.root), task)
            invoke.assert_not_called()

    def test_invalid_budgets_fail_before_graph_invocation(self) -> None:
        with patch("engineering_plan.engineering_plan_graph.invoke") as invoke:
            with self.assertRaises(ValueError):
                plan_repository_task(str(self.root), self.task, max_tool_calls=0)
            with self.assertRaises(ValueError):
                plan_repository_task(
                    str(self.root),
                    self.task,
                    max_plan_retries=-1,
                )

        invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
