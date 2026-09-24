from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from studio_backend.config import MAX_STUDIO_ARTIFACT_CHARS
from studio_backend.models import StudioRun
from studio_backend.storage import StudioStorage, utc_now


class StudioStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.storage = StudioStorage(Path(self.temporary.name) / "studio.sqlite3")
        now = utc_now()
        self.run = StudioRun(
            id="run-1",
            repository_id="example",
            task="Fix the behavior.",
            status="queued",
            phase="queued",
            created_at=now,
            updated_at=now,
        )
        self.storage.create_run(self.run)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_run_persists_across_storage_instances(self) -> None:
        reopened = StudioStorage(Path(self.temporary.name) / "studio.sqlite3")
        self.assertEqual(reopened.get_run("run-1"), self.run)

    def test_recent_runs_are_newest_first_and_bounded(self) -> None:
        self.assertEqual(self.storage.list_runs(100)[0].id, "run-1")

    def test_event_sequences_are_monotonic_under_concurrency(self) -> None:
        def append(index: int) -> None:
            self.storage.append_event(
                "run-1",
                phase="execution",
                event_type=f"event_{index}",
                status="info",
                title=f"Event {index}",
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(append, range(20)))
        events = self.storage.list_events("run-1")
        self.assertEqual([event.sequence for event in events], list(range(1, 21)))

    def test_event_metadata_budget_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "event metadata"):
            self.storage.append_event(
                "run-1",
                phase="execution",
                event_type="large",
                status="info",
                title="Large event",
                metadata={"value": "x" * 20_001},
            )

    def test_artifact_versions_and_latest_payload_persist(self) -> None:
        first = self.storage.put_artifact("run-1", "candidate_diff", {"v": 1})
        second = self.storage.put_artifact("run-1", "candidate_diff", {"v": 2})
        self.assertEqual((first.version, second.version), (1, 2))
        self.assertEqual(
            self.storage.get_artifact("run-1", "candidate_diff").payload,
            {"v": 2},
        )

    def test_artifact_budget_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "artifact"):
            self.storage.put_artifact(
                "run-1",
                "candidate",
                {"content": "x" * (MAX_STUDIO_ARTIFACT_CHARS + 1)},
            )

    def test_compare_and_set_allows_only_one_claim(self) -> None:
        def claim():
            return self.storage.compare_and_set_run(
                "run-1",
                expected_statuses=("queued",),
                status="running",
                phase="planning",
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _value: claim(), range(2)))
        self.assertEqual(sum(result is not None for result in results), 1)

    def test_restart_marks_active_run_interrupted(self) -> None:
        self.assertEqual(self.storage.mark_active_interrupted(), 1)
        run = self.storage.get_run("run-1")
        self.assertEqual((run.status, run.phase), ("interrupted", "failed"))
