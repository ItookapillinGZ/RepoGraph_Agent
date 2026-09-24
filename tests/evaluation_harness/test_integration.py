"""Real nested graph plus external pytest integration tests."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from change_set import ChangeSetReview
from engineering_plan import build_engineering_plan_graph
from evaluation.models import EvaluationConfig
from evaluation.runner import RealRepoGraphAdapter, run_evaluation_task
from plan_execution import build_plan_execution_graph
from review_models import OverallRating
from tests.evaluation_harness.helpers import (
    FIXED,
    WRONG,
    candidate,
    make_repository,
    plan,
    repository_digest,
    task,
)


def tool_call(name: str, arguments: dict[str, object], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": arguments, "id": call_id, "type": "tool_call"}
        ],
    )


def planning_graph():
    explorer = MagicMock()
    explorer.bind_tools.return_value.invoke.side_effect = [
        tool_call("read_repository_file", {"path": "src/calculator.py"}, "read"),
        tool_call("read_repository_file", {"path": "tests/test_calculator.py"}, "test"),
        AIMessage(content="Calculator implementation and regression test inspected."),
    ]
    planner = MagicMock()
    planner.with_structured_output.return_value.invoke.return_value = plan()
    return build_engineering_plan_graph(
        explorer_model=explorer,
        planner_model=planner,
    )


def execution_graph(*responses):
    executor = MagicMock()
    executor.with_structured_output.return_value.invoke.side_effect = list(responses[:1])
    corrector = MagicMock()
    corrector.with_structured_output.return_value.invoke.side_effect = list(responses[1:])
    reviewer = MagicMock(
        return_value=ChangeSetReview(
            overall_rating=OverallRating.GOOD,
            summary="Deterministic integration review.",
            file_results=[],
        )
    )
    return build_plan_execution_graph(
        executor_model=executor,
        corrector_model=corrector,
        change_set_reviewer=reviewer,
    )


class EvaluationIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source, self.commit = make_repository(self.root)
        self.config = EvaluationConfig(
            name="integration",
            agentic_explore=False,
            agentic_test=False,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_real_graph_candidate_external_pytest_and_patch(self) -> None:
        before = repository_digest(self.source)
        adapter = RealRepoGraphAdapter(
            planning_graph=planning_graph(),
            execution_graph=execution_graph(candidate(FIXED)),
        )
        result = run_evaluation_task(
            task(self.source, self.commit),
            self.config,
            workspace_root=str(self.root / "workspaces"),
            adapter=adapter,
        )
        self.assertEqual(result.status, "resolved")
        self.assertTrue(result.first_attempt_resolved)
        self.assertIn("src/calculator.py", result.model_patch or "")
        self.assertEqual(result.tool_calls, 2)
        self.assertEqual(repository_digest(self.source), before)

    def test_real_graph_wrong_v1_then_correct_v2_measures_uplift(self) -> None:
        adapter = RealRepoGraphAdapter(
            planning_graph=planning_graph(),
            execution_graph=execution_graph(candidate(WRONG), candidate(FIXED)),
        )
        result = run_evaluation_task(
            task(self.source, self.commit),
            self.config.model_copy(update={"max_correction_rounds": 1}),
            workspace_root=str(self.root / "workspaces"),
            adapter=adapter,
        )
        self.assertEqual(result.status, "resolved")
        self.assertFalse(result.first_attempt_resolved)
        self.assertTrue(result.final_resolved)
        self.assertEqual(result.correction_rounds_used, 1)


if __name__ == "__main__":
    unittest.main()
