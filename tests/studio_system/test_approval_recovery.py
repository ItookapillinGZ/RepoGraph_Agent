from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from git_delivery import create_local_git_delivery
from plan_application import (
    PlanApplicationBundle,
    apply_plan_application_bundle,
)
from studio_backend.recovery import StudioRecoveryService
from tests.studio_system.fixtures import (
    FIXED_SOURCE,
    ORIGINAL_SOURCE,
    InjectedRemoteBoundary,
    build_test_app,
    file_hash,
    initialize_repository,
    run_git,
    start_and_wait_verified,
)


class StudioApprovalRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="repograph-studio-recovery-"
        )
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.repository, self.bare_remote, self.initial_sha = initialize_repository(
            self.workspace,
        )
        self.initial_index_hash = file_hash(self.repository / ".git" / "index")
        self.remote_boundary = InjectedRemoteBoundary(
            push_created=False,
            pr_created=False,
        )
        self.app = build_test_app(
            self.workspace,
            self.root / "data",
            remote_boundary=self.remote_boundary,
        )
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()
        self.storage = self.app.state.storage
        self.repositories = self.app.state.repositories
        self.recovery = StudioRecoveryService(self.storage, self.repositories)

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.temporary.cleanup()

    def preview(self) -> tuple[str, str, PlanApplicationBundle]:
        detail = start_and_wait_verified(self.client)
        run_id = str(detail["id"])
        digest = str(detail["approval_digest"])
        bundle_payload = self.client.get(
            f"/api/runs/{run_id}/artifacts/application_bundle"
        ).json()["payload"]
        return run_id, digest, PlanApplicationBundle.model_validate(bundle_payload)

    def approve_apply(self, run_id: str, digest: str):
        return self.client.post(
            f"/api/runs/{run_id}/approve/apply",
            json={"approved": True, "approval_digest": digest},
        )

    def approve_git(self, run_id: str, digest: str):
        return self.client.post(
            f"/api/runs/{run_id}/approve/git",
            json={
                "approved": True,
                "approval_digest": digest,
                "branch_name": None,
                "commit_message": None,
            },
        )

    def event_types(self, run_id: str) -> list[str]:
        return [
            event.event_type for event in self.storage.list_events(run_id)
        ]

    def test_application_restart_is_interrupted_and_explicit_retry_revalidates(self) -> None:
        run_id, digest, _bundle = self.preview()
        self.storage.update_run(run_id, status="applying", phase="application")

        self.recovery.reconcile_startup()

        interrupted = self.storage.get_run(run_id)
        self.assertEqual((interrupted.status, interrupted.phase), ("interrupted", "application"))
        self.assertIn("application_interrupted", self.event_types(run_id))
        self.assertEqual(
            (self.repository / "src" / "calculator.py").read_text(
                encoding="utf-8"
            ),
            ORIGINAL_SOURCE,
        )

        response = self.approve_apply(run_id, digest)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["run"]["status"], "applied")
        self.assertEqual(
            (self.repository / "src" / "calculator.py").read_text(
                encoding="utf-8"
            ),
            FIXED_SOURCE,
        )

    def test_fully_applied_but_unrecorded_application_stays_conservative(self) -> None:
        run_id, digest, bundle = self.preview()
        result = apply_plan_application_bundle(
            str(self.repository),
            bundle,
            approved=True,
        )
        self.assertEqual(result.status, "applied")
        self.storage.update_run(run_id, status="applying", phase="application")

        self.recovery.reconcile_startup()
        response = self.approve_apply(run_id, digest)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["run"]["status"], "stale")
        self.assertEqual(
            (self.repository / "src" / "calculator.py").read_text(
                encoding="utf-8"
            ),
            FIXED_SOURCE,
        )

    def test_running_preview_restart_only_updates_persisted_state(self) -> None:
        run_id, _digest, _bundle = self.preview()
        before_source = (
            self.repository / "src" / "calculator.py"
        ).read_bytes()
        self.storage.update_run(run_id, status="running", phase="verification")

        self.recovery.reconcile_startup()

        interrupted = self.storage.get_run(run_id)
        self.assertEqual((interrupted.status, interrupted.phase), ("interrupted", "failed"))
        self.assertIn("execution_interrupted", self.event_types(run_id))
        self.assertEqual(
            (self.repository / "src" / "calculator.py").read_bytes(),
            before_source,
        )
        self.assertEqual(run_git(self.repository, "rev-parse", "HEAD"), self.initial_sha)

    def test_git_restart_recovers_exact_existing_branch_read_only(self) -> None:
        run_id, digest, bundle = self.preview()
        self.assertEqual(self.approve_apply(run_id, digest).status_code, 200)
        delivery = create_local_git_delivery(
            str(self.repository),
            bundle,
            approved=True,
        )
        self.assertEqual(delivery.status, "created")
        self.storage.update_run(run_id, status="git_creating", phase="git_delivery")

        self.recovery.reconcile_startup()

        recovered = self.storage.get_run(run_id)
        self.assertEqual(
            (recovered.status, recovered.phase),
            ("git_created", "approval_remote"),
        )
        artifact = self.storage.get_artifact(
            run_id,
            "local_git_delivery_result",
        )
        self.assertEqual(artifact.payload["commit_sha"], delivery.commit_sha)
        self.assertEqual(run_git(self.repository, "rev-parse", "HEAD"), self.initial_sha)
        self.assertEqual(
            file_hash(self.repository / ".git" / "index"),
            self.initial_index_hash,
        )

    def test_git_restart_without_branch_allows_explicit_retry(self) -> None:
        run_id, digest, _bundle = self.preview()
        self.assertEqual(self.approve_apply(run_id, digest).status_code, 200)
        self.storage.update_run(run_id, status="git_creating", phase="git_delivery")

        self.recovery.reconcile_startup()

        interrupted = self.storage.get_run(run_id)
        self.assertEqual(
            (interrupted.status, interrupted.phase),
            ("interrupted", "git_delivery"),
        )
        response = self.approve_git(run_id, digest)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["run"]["status"], "git_created")

    def test_git_restart_with_mismatched_existing_branch_never_overwrites(self) -> None:
        run_id, digest, _bundle = self.preview()
        self.assertEqual(self.approve_apply(run_id, digest).status_code, 200)
        branch_name = f"repograph/{digest[:12]}"
        run_git(self.repository, "branch", branch_name, self.initial_sha)
        self.storage.update_run(run_id, status="git_creating", phase="git_delivery")

        self.recovery.reconcile_startup()

        conflicted = self.storage.get_run(run_id)
        self.assertEqual(conflicted.status, "conflict")
        self.assertEqual(
            run_git(self.repository, "rev-parse", branch_name),
            self.initial_sha,
        )

    def test_remote_restart_is_read_only_and_retry_calls_injected_boundary(self) -> None:
        """Keep recovery read-only; explicit retry exercises Studio-to-G5B wiring."""
        run_id, digest, _bundle = self.preview()
        self.assertEqual(self.approve_apply(run_id, digest).status_code, 200)
        self.assertEqual(self.approve_git(run_id, digest).status_code, 200)
        local = self.storage.get_artifact(
            run_id,
            "local_git_delivery_result",
        ).payload
        branch_name = str(local["branch_name"])

        self.storage.update_run(run_id, status="publishing", phase="remote_delivery")

        with patch("github_delivery._request_json") as network:
            self.recovery.reconcile_startup()
        network.assert_not_called()
        interrupted = self.storage.get_run(run_id)
        self.assertEqual(
            (interrupted.status, interrupted.phase),
            ("interrupted", "remote_delivery"),
        )

        with patch("github_delivery._request_json") as network:
            response = self.client.post(
                f"/api/runs/{run_id}/approve/remote",
                json={
                    "approved": True,
                    "approval_digest": digest,
                    "remote_name": "origin",
                    "base_branch": "main",
                    "branch_name": branch_name,
                    "pr_title": None,
                    "pr_body": None,
                },
            )
        network.assert_not_called()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["run"]["status"], "published")
        remote = self.storage.get_artifact(run_id, "remote_delivery_result").payload
        self.assertFalse(remote["push_created"])
        self.assertFalse(remote["pr_created"])
        self.assertEqual(len(self.remote_boundary.calls), 1)
        self.assertEqual(
            self.remote_boundary.calls[0]["local_branch"],
            branch_name,
        )
        self.assertEqual(run_git(self.repository, "rev-parse", "HEAD"), self.initial_sha)
        self.assertEqual(
            file_hash(self.repository / ".git" / "index"),
            self.initial_index_hash,
        )


if __name__ == "__main__":
    unittest.main()
