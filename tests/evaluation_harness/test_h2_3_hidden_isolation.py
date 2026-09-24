"""H2.3 harder benchmark and hidden-test isolation regressions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from engineering_plan import EngineeringPlan, PlannedFileChange
from evaluation.adapters.local_tasks import LocalTaskDataset
from evaluation.evaluator import run_local_evaluator
from evaluation.h2_3_fixtures import create_h2_3_benchmark
from evaluation.workspace import (
    WorkspaceError,
    apply_candidate_exact,
    extract_candidate_patch,
    prepare_task_workspace,
)
from plan_execution import CandidateFileChange, MultiFileCandidate
from tests.evaluation_harness.helpers import git


class HarderBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.dataset_path = create_h2_3_benchmark(cls.root / "benchmark")
        cls.tasks = LocalTaskDataset(cls.dataset_path).load()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_twenty_tasks_have_bounded_repositories_and_multi_file_scope(self) -> None:
        self.assertEqual(len(self.tasks), 20)
        multi_file = 0
        for task in self.tasks:
            count = task.metadata["repository_file_count"]
            self.assertGreaterEqual(count, 5)
            self.assertLessEqual(count, 20)
            if task.metadata["requires_multi_file_exploration"]:
                multi_file += 1
        self.assertGreaterEqual(multi_file / len(self.tasks), 0.70)

    def test_hidden_tests_are_absent_from_agent_workspace(self) -> None:
        task = self.tasks[0]
        workspace, _ = prepare_task_workspace(
            task.repository,
            task.base_commit,
            self.root / "workspaces",
            "hidden-isolation",
            task.id,
        )
        names = [
            path.relative_to(workspace).as_posix() for path in workspace.rglob("*")
        ]
        self.assertFalse(any("hidden" in name.casefold() for name in names))

    def test_evaluator_can_access_hidden_tests(self) -> None:
        task = self.tasks[0]
        workspace, _ = prepare_task_workspace(
            task.repository,
            task.base_commit,
            self.root / "workspaces",
            "hidden-evaluator",
            task.id,
        )
        outcome = run_local_evaluator(
            task,
            workspace,
            timeout_seconds=60,
            max_output_chars=10_000,
        )
        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.failure_kind, "assertion_failure")

    def test_candidate_cannot_target_hidden_test_path(self) -> None:
        hidden = (
            self.root
            / "benchmark"
            / "evaluator-only"
            / "hidden-tests"
            / self.tasks[0].id
            / "test_hidden.py"
        )
        plan = EngineeringPlan(
            summary="Attempt an invalid external change.",
            files=[
                PlannedFileChange(
                    path=str(hidden),
                    action="modify",
                    rationale="This path must be rejected.",
                )
            ],
        )
        value = MultiFileCandidate(
            summary="Attempt an invalid external change.",
            files=[
                CandidateFileChange(
                    path=str(hidden),
                    action="modify",
                    content="def test_disabled(): pass\n",
                )
            ],
        )
        workspace, _ = prepare_task_workspace(
            self.tasks[0].repository,
            self.tasks[0].base_commit,
            self.root / "workspaces",
            "path-escape",
            self.tasks[0].id,
        )
        with self.assertRaises(WorkspaceError):
            apply_candidate_exact(workspace, plan, value)

    def test_source_repository_unchanged_and_hidden_artifact_excluded(self) -> None:
        task = next(item for item in self.tasks if item.id == "record-parser-contract")
        status_before = git(Path(task.repository), "status", "--porcelain=v1")
        workspace, resolved = prepare_task_workspace(
            task.repository,
            task.base_commit,
            self.root / "workspaces",
            "hidden-patch",
            task.id,
        )
        plan = EngineeringPlan(
            summary="Document the parser contract.",
            files=[
                PlannedFileChange(
                    path="src/parser.py",
                    action="modify",
                    rationale="Keep the named-field contract explicit.",
                )
            ],
        )
        value = MultiFileCandidate(
            summary="Document the parser contract.",
            files=[
                CandidateFileChange(
                    path="src/parser.py",
                    action="modify",
                    content=(
                        "def parse_record(line):\n"
                        '    """Return a validated named-field record."""\n'
                        '    name, raw_count = line.split(",", 1)\n'
                        '    return {"name": name.strip(), "count": int(raw_count)}\n'
                    ),
                )
            ],
        )
        apply_candidate_exact(workspace, plan, value)
        patch_text, changed = extract_candidate_patch(
            workspace,
            resolved,
            value,
        )
        self.assertEqual(changed, ["src/parser.py"])
        self.assertEqual(
            git(Path(task.repository), "status", "--porcelain=v1"),
            status_before,
        )
        self.assertNotIn("hidden-tests", patch_text)
        self.assertNotIn("evaluator-only", patch_text)


if __name__ == "__main__":
    unittest.main()
