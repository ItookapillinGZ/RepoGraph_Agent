"""SQLite and JSONL persistence tests."""

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.models import EvaluationConfig, EvaluationResult, EvaluationTask
from evaluation.runner import create_experiment
from evaluation.storage import EvaluationStorage, EvaluationStorageError


class StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.storage = EvaluationStorage(self.root / "evaluation.db", self.root / "results")
        self.experiment = create_experiment(
            "baseline", "local", EvaluationConfig(name="baseline", agentic_test=False)
        )
        self.storage.save_experiment(self.experiment)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_result(self, status: str = "resolved") -> EvaluationResult:
        return EvaluationResult(
            task_id="task", dataset="local", experiment=self.experiment.id,
            repository="r", base_commit="c", created_at="now", status=status,
            final_resolved=status == "resolved", duration_seconds=1
        )

    def test_sqlite_round_trip_and_lookup_by_name(self) -> None:
        self.storage.save_result(self.make_result())
        self.assertEqual(self.storage.get_experiment("baseline"), self.experiment)
        self.assertEqual(
            self.storage.get_result(self.experiment.id, "task").status, "resolved"
        )

    def test_unique_experiment_task_result_is_upserted(self) -> None:
        self.storage.save_result(self.make_result("evaluation_error"))
        self.storage.save_result(self.make_result("resolved"))
        self.assertEqual(len(self.storage.list_results(self.experiment.id)), 1)
        self.assertEqual(self.storage.list_results(self.experiment.id)[0].status, "resolved")

    def test_jsonl_appends_result_history_while_sqlite_keeps_current(self) -> None:
        self.storage.save_result(self.make_result("evaluation_error"))
        self.storage.save_result(self.make_result("resolved"))
        lines = (self.root / "results" / f"{self.experiment.id}.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(
            [json.loads(line)["status"] for line in lines],
            ["evaluation_error", "resolved"],
        )

    def test_task_identity_cannot_change(self) -> None:
        first = EvaluationTask(
            id="task", dataset="local", repository="r", base_commit="c", task="one"
        )
        self.storage.save_task(first)
        with self.assertRaises(EvaluationStorageError):
            self.storage.save_task(first.model_copy(update={"task": "two"}))


if __name__ == "__main__":
    unittest.main()
