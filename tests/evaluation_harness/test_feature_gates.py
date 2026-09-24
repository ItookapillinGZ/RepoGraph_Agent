"""Formal No-Agentic-Test tool-schema and execution bridge tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from pydantic import ValidationError

from change_set import ChangeSetReview
from evaluation.experiments.configs import ablation_specs
from evaluation.models import EvaluationConfig
from plan_execution import execute_engineering_plan, review_candidate_change_set_node
from repository_exploration import repository_exploration_graph
from review_models import OverallRating
from tests.evaluation_harness.helpers import plan


class FeatureGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "tests").mkdir()
        (self.root / "tests" / "test_app.py").write_text("def test_ok(): pass\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def tool_names(self, agentic_test: bool) -> set[str]:
        model = MagicMock()
        repository_exploration_graph(
            str(self.root),
            allowed_test_files=["tests/test_app.py"] if agentic_test else [],
            model=model,
        )
        return {tool.name for tool in model.bind_tools.call_args.args[0]}

    def test_false_removes_test_tool_but_keeps_read_search_list(self) -> None:
        names = self.tool_names(False)
        self.assertEqual(
            names,
            {"read_repository_file", "search_repository_code", "list_repository_files"},
        )
        self.assertNotIn("run_repository_test", names)

    def test_true_exposes_test_tool(self) -> None:
        self.assertIn("run_repository_test", self.tool_names(True))

    def test_gate_requires_explore_and_deterministic_tests(self) -> None:
        with self.assertRaises(ValidationError):
            EvaluationConfig(name="invalid", agentic_explore=False, agentic_test=True)
        with self.assertRaises(ValidationError):
            EvaluationConfig(name="invalid", run_tests=False, agentic_test=True)
        with self.assertRaisesRegex(ValueError, "agentic_explore"):
            execute_engineering_plan(
                str(self.root),
                "task",
                plan(),
                run_tests=True,
                agentic_test=True,
            )
        with self.assertRaisesRegex(ValueError, "run_tests"):
            execute_engineering_plan(
                str(self.root),
                "task",
                plan(),
                agentic_explore=True,
                agentic_test=True,
            )

    def test_execution_bridge_passes_real_gate_without_changing_run_tests(self) -> None:
        calls: list[dict[str, object]] = []

        def reviewer(_root: str, **kwargs: object) -> ChangeSetReview:
            calls.append(kwargs)
            return ChangeSetReview(
                overall_rating=OverallRating.GOOD,
                summary="ok",
                file_results=[],
            )

        common = {
            "temporary_repository_root": str(self.root),
            "temporary_workspace_base": str(self.root.parent),
            "diff_text": "diff --git a/app.py b/app.py\n",
            "candidate_review_agentic_explore": True,
            "candidate_review_run_tests": True,
        }
        review_candidate_change_set_node(
            {**common, "candidate_review_agentic_test": False},  # type: ignore[arg-type]
            reviewer=reviewer,
        )
        review_candidate_change_set_node(
            {**common, "candidate_review_agentic_test": True},  # type: ignore[arg-type]
            reviewer=reviewer,
        )
        self.assertEqual([call["run_tests"] for call in calls], [True, True])
        self.assertEqual(
            [call.get("agentic_test", False) for call in calls],
            [False, True],
        )

    def test_no_agentic_test_ablation_is_supported(self) -> None:
        specs = {spec.key: spec for spec in ablation_specs()}
        self.assertTrue(specs["D"].supported)
        self.assertFalse(specs["D"].config.agentic_test)
        self.assertTrue(specs["A"].config.agentic_test)


if __name__ == "__main__":
    unittest.main()
