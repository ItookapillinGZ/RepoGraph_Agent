"""Dataset adapter and fixture-generator tests."""

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.adapters.local_tasks import LocalTaskDataset
from evaluation.adapters.swebench import SWEbenchDataset, export_predictions
from evaluation.datasets import DatasetError
from evaluation.fixtures import FIXTURES, create_local_benchmark
from evaluation.models import EvaluationResult


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_loads_valid_local_jsonl_and_sets_dataset(self) -> None:
        path = self.root / "tasks.jsonl"
        path.write_text(
            json.dumps(
                {
                    "id": "one",
                    "repository": "repo",
                    "base_commit": "abc",
                    "task": "fix",
                    "test_command": ["python", "-m", "pytest"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        tasks = LocalTaskDataset(path).load()
        self.assertEqual(tasks[0].dataset, "local")

    def test_rejects_malformed_jsonl(self) -> None:
        path = self.root / "bad.jsonl"
        path.write_text("{bad}\n", encoding="utf-8")
        with self.assertRaisesRegex(DatasetError, "line 1"):
            LocalTaskDataset(path).load()

    def test_rejects_duplicate_task_ids(self) -> None:
        path = self.root / "duplicate.jsonl"
        row = {"id": "one", "repository": "r", "base_commit": "c", "task": "t"}
        path.write_text(json.dumps(row) + "\n" + json.dumps(row), encoding="utf-8")
        with self.assertRaisesRegex(DatasetError, "Duplicate"):
            LocalTaskDataset(path).load()

    def test_swebench_parses_current_schema_and_json_test_lists(self) -> None:
        path = self.root / "swe.jsonl"
        path.write_text(
            json.dumps(
                {
                    "instance_id": "owner__repo-1",
                    "repo": "owner/repo",
                    "base_commit": "abc",
                    "problem_statement": "Fix it",
                    "FAIL_TO_PASS": '["test_a"]',
                    "PASS_TO_PASS": ["test_b"],
                    "test_patch": "diff --git a/t b/t",
                }
            ),
            encoding="utf-8",
        )
        task = SWEbenchDataset(path).load()[0]
        self.assertEqual(task.metadata["FAIL_TO_PASS"], ["test_a"])
        self.assertIsNone(task.test_command)

    def test_prediction_export_uses_official_fields(self) -> None:
        result = EvaluationResult(
            task_id="owner__repo-1", dataset="swebench", experiment="e",
            repository="owner/repo", base_commit="abc", created_at="now",
            status="unresolved", duration_seconds=1, model_patch="diff --git"
        )
        output = export_predictions([result], self.root / "predictions.jsonl", model_name_or_path="model")
        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(
            set(payload), {"instance_id", "model_name_or_path", "model_patch"}
        )

    def test_local_benchmark_generator_creates_ten_diverse_git_tasks(self) -> None:
        output = self.root / "benchmark"
        dataset = create_local_benchmark(output)
        tasks = LocalTaskDataset(dataset, name="repograph-local-v1").load()
        self.assertEqual(len(tasks), 10)
        self.assertEqual(len(FIXTURES), 10)
        self.assertGreaterEqual(
            len({task.metadata["category"] for task in tasks}), 6
        )
        self.assertTrue(all((Path(task.repository) / ".git").is_dir() for task in tasks))


if __name__ == "__main__":
    unittest.main()
