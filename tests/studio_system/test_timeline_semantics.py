from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from tests.studio_system.fixtures import (
    BUGGY_SOURCE,
    FIXED_SOURCE,
    assert_event_order,
    build_test_app,
    initialize_repository,
    start_and_wait_verified,
)


class StudioTimelineSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="repograph-studio-timeline-"
        )
        self.root = Path(self.temporary.name)
        self.clients: list[TestClient] = []

    def tearDown(self) -> None:
        for client in reversed(self.clients):
            client.__exit__(None, None, None)
        self.temporary.cleanup()

    def run_preview(self, *, correction: bool) -> tuple[TestClient, dict[str, object]]:
        workspace = self.root / ("correction-workspace" if correction else "workspace")
        workspace.mkdir()
        initialize_repository(workspace)
        app = build_test_app(
            workspace,
            self.root / ("correction-data" if correction else "data"),
            correction=correction,
        )
        client = TestClient(app)
        client.__enter__()
        self.clients.append(client)
        detail = start_and_wait_verified(client, correction=correction)
        return client, detail

    def events(self, client: TestClient, run_id: str) -> list[dict[str, object]]:
        response = client.get(f"/api/runs/{run_id}/events")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["events"]

    def test_normal_business_events_have_semantic_order_and_no_duplicates(self) -> None:
        client, detail = self.run_preview(correction=False)
        events = self.events(client, str(detail["id"]))
        assert_event_order(
            events,
            [
                "run_created",
                "repository_exploration_started",
                "planning_started",
                "repository_exploration_completed",
                "plan_generated",
                "execution_started",
                "candidate_generated",
                "verification_started",
                "verification_completed",
                "review_started",
                "review_completed",
                "preview_verified",
                "apply_approval_requested",
                "waiting_for_apply_approval",
            ],
        )
        event_types = [event["event_type"] for event in events]
        self.assertEqual(event_types.count("plan_generated"), 1)
        self.assertEqual(event_types.count("preview_verified"), 1)
        self.assertEqual(event_types.count("candidate_generated"), 1)
        self.assertEqual(
            [event["sequence"] for event in events],
            list(range(1, len(events) + 1)),
        )

    def test_correction_timeline_and_artifacts_belong_to_final_candidate(self) -> None:
        client, detail = self.run_preview(correction=True)
        run_id = str(detail["id"])
        events = self.events(client, run_id)
        assert_event_order(
            events,
            [
                "candidate_generated",
                "verification_completed",
                "review_completed",
                "correction_started",
                "candidate_generated",
                "verification_completed",
                "review_completed",
                "preview_verified",
            ],
        )

        candidate_events = [
            event for event in events if event["event_type"] == "candidate_generated"
        ]
        verification_events = [
            event
            for event in events
            if event["event_type"] == "verification_completed"
        ]
        review_events = [
            event for event in events if event["event_type"] == "review_completed"
        ]
        self.assertEqual(
            [event["metadata"]["candidate_attempt"] for event in candidate_events],
            [1, 2],
        )
        self.assertEqual(
            [event["metadata"]["correction_round"] for event in candidate_events],
            [0, 1],
        )
        self.assertEqual(
            [event["metadata"]["candidate_attempt"] for event in verification_events],
            [1, 2],
        )
        self.assertEqual(
            [event["metadata"]["candidate_attempt"] for event in review_events],
            [1, 2],
        )
        self.assertTrue(
            all(
                event["metadata"].get("nested_graph")
                == "change_set_graph/review_graph"
                for event in review_events
            )
        )

        def artifact(kind: str) -> dict[str, object]:
            return client.get(
                f"/api/runs/{run_id}/artifacts/{kind}"
            ).json()["payload"]

        candidate_payload = artifact("candidate")
        diff_payload = artifact("candidate_diff")
        verification_payload = artifact("verification")
        review_payload = artifact("change_set_review")
        correction_payload = artifact("correction_history")
        self.assertEqual(candidate_payload["files"][0]["content"], FIXED_SOURCE)
        self.assertIn("return a + b", diff_payload["diff_text"])
        self.assertNotIn("return a * b", diff_payload["diff_text"])
        self.assertEqual(verification_payload["status"], "verified")
        self.assertEqual(review_payload["overall_rating"], "good")
        self.assertEqual(correction_payload["correction_rounds_used"], 1)
        attempts = correction_payload["attempt_history"]
        self.assertEqual(
            [attempt["verification_status"] for attempt in attempts],
            ["failed", "verified"],
        )
        self.assertNotIn(BUGGY_SOURCE, str(events))


if __name__ == "__main__":
    unittest.main()
