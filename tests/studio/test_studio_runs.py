from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from studio_backend.app import create_app
from studio_backend.config import StudioConfig
from studio_backend.events import StudioEventEmitter
from studio_backend.models import StartRunRequest
from tests.studio.helpers import DIGEST, application_bundle


class FakePreviewAdapter:
    def __init__(self, storage) -> None:
        self.storage = storage

    def execute(self, run_id: str, _request: StartRunRequest) -> None:
        self.storage.compare_and_set_run(
            run_id,
            expected_statuses=("queued",),
            status="running",
            phase="planning",
        )
        self.storage.put_artifact(
            run_id,
            "application_bundle",
            application_bundle().model_dump(mode="json"),
        )
        self.storage.put_artifact(
            run_id,
            "engineering_plan",
            application_bundle().plan.model_dump(mode="json"),
        )
        self.storage.update_run(
            run_id,
            status="verified",
            phase="approval_apply",
            approval_digest=DIGEST,
        )
        StudioEventEmitter(self.storage, run_id).emit(
            phase="approval_apply",
            event_type="waiting_for_apply_approval",
            status="waiting",
            title="Waiting for apply approval",
        )


class StudioRunApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.workspace = root / "workspace"
        self.data = root / "data"
        self.repository = self.workspace / "example"
        self.repository.mkdir(parents=True)
        self.data.mkdir()
        config = StudioConfig(
            workspace_root=self.workspace.resolve(),
            data_dir=self.data.resolve(),
            allowed_origin="http://localhost:3000",
            max_concurrent_runs=1,
        )
        self.app = create_app(config)
        self.app.state.run_manager.adapter = FakePreviewAdapter(
            self.app.state.storage
        )
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.temporary.cleanup()

    def _create(self, **overrides):
        payload = {
            "repository_id": "example",
            "task": "Fix the behavior and add tests.",
            "self_correct": True,
            "max_correction_rounds": 2,
            **overrides,
        }
        return self.client.post("/api/runs", json=payload)

    def _wait_verified(self, run_id: str) -> dict[str, object]:
        for _ in range(100):
            payload = self.client.get(f"/api/runs/{run_id}").json()
            if payload["status"] == "verified":
                return payload
            time.sleep(0.01)
        self.fail("Fake preview did not finish.")

    def test_health_is_minimal(self) -> None:
        self.assertEqual(
            self.client.get("/api/health").json(),
            {"status": "ok", "studio": "repograph", "version": 1},
        )

    def test_repository_listing_never_returns_absolute_path(self) -> None:
        response = self.client.get("/api/repositories")
        self.assertEqual(response.status_code, 200)
        rendered = response.text
        self.assertNotIn(str(self.workspace), rendered)
        self.assertEqual(response.json()["repositories"][0]["id"], "example")

    def test_user_can_add_external_project_and_run_it(self) -> None:
        external = self.workspace.parent / "other-project"
        external.mkdir()
        added = self.client.post("/api/repositories", json={"path": str(external)})
        self.assertEqual(added.status_code, 201)
        identifier = added.json()["id"]
        self.assertTrue(identifier.startswith("added-"))
        self.assertNotIn(str(external), added.text)
        listing = self.client.get("/api/repositories").json()["repositories"]
        self.assertIn(identifier, [item["id"] for item in listing])
        run = self._create(repository_id=identifier)
        self.assertEqual(run.status_code, 202)
        self._wait_verified(run.json()["run_id"])

    def test_run_schema_forbids_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            StartRunRequest.model_validate(
                {
                    "repository_id": "example",
                    "task": "Task",
                    "unexpected": True,
                }
            )

    def test_blank_and_oversized_tasks_are_rejected(self) -> None:
        self.assertEqual(self._create(task="   ").status_code, 422)
        self.assertEqual(self._create(task="x" * 10_001).status_code, 422)

    def test_absolute_and_traversal_repository_ids_are_rejected(self) -> None:
        self.assertEqual(
            self._create(repository_id=str(self.repository.resolve())).status_code,
            404,
        )
        self.assertEqual(self._create(repository_id="../example").status_code, 404)

    def test_create_run_returns_unique_ids_and_persists_history(self) -> None:
        first = self._create()
        second = self._create(task="A second task.")
        self.assertEqual((first.status_code, second.status_code), (202, 202))
        first_id = first.json()["run_id"]
        second_id = second.json()["run_id"]
        self.assertNotEqual(first_id, second_id)
        self._wait_verified(first_id)
        self._wait_verified(second_id)
        runs = self.client.get("/api/runs").json()["runs"]
        self.assertEqual(len(runs), 2)

    def test_detail_is_small_and_artifacts_load_separately(self) -> None:
        run_id = self._create().json()["run_id"]
        detail = self._wait_verified(run_id)
        self.assertIn("engineering_plan", detail["artifact_kinds"])
        self.assertNotIn("plan", detail)
        artifact = self.client.get(
            f"/api/runs/{run_id}/artifacts/engineering_plan"
        )
        self.assertEqual(artifact.status_code, 200)
        self.assertEqual(artifact.json()["payload"]["summary"], "Update one file safely.")

    def test_events_are_ordered_and_persisted(self) -> None:
        run_id = self._create().json()["run_id"]
        self._wait_verified(run_id)
        events = self.client.get(f"/api/runs/{run_id}/events").json()["events"]
        sequences = [event["sequence"] for event in events]
        self.assertEqual(sequences, sorted(sequences))
        self.assertEqual(events[0]["event_type"], "run_created")
        self.assertIn("/api/runs/{run_id}/stream", self.app.openapi()["paths"])

    def test_secret_and_absolute_root_are_not_exposed(self) -> None:
        with patch.dict(os.environ, {"GITHUB_TOKEN": "studio-super-secret"}):
            run_id = self._create().json()["run_id"]
            self._wait_verified(run_id)
            self.app.state.storage.put_artifact(
                run_id,
                "candidate_diff",
                {
                    "path": str(self.workspace),
                    "token": "studio-super-secret",
                },
            )
            detail_response = self.client.get(f"/api/runs/{run_id}")
            artifact_response = self.client.get(
                f"/api/runs/{run_id}/artifacts/candidate_diff"
            )
        self.assertNotIn("studio-super-secret", detail_response.text)
        self.assertNotIn(str(self.workspace), detail_response.text)
        self.assertTrue(detail_response.json()["github_auth_configured"])
        self.assertNotIn("studio-super-secret", artifact_response.text)
        self.assertNotIn(str(self.workspace), artifact_response.text)

    def test_cors_accepts_both_loopback_names_on_configured_port(self) -> None:
        for origin in ("http://localhost:3000", "http://127.0.0.1:3000"):
            with self.subTest(origin=origin):
                response = self.client.options(
                    "/api/health",
                    headers={
                        "Origin": origin,
                        "Access-Control-Request-Method": "GET",
                    },
                )
                self.assertEqual(response.headers["access-control-allow-origin"], origin)

        response = self.client.options(
            "/api/health",
            headers={
                "Origin": "http://127.0.0.1:3001",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertNotIn("access-control-allow-origin", response.headers)
