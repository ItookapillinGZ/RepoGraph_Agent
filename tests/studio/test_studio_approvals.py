from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from git_delivery import LocalGitDeliveryResult
from github_delivery import GitHubRemoteDeliveryResult
from plan_application import PlanApplicationResult
from studio_backend.approvals import (
    ApprovalConflict,
    ApprovalRejected,
    StudioApprovalService,
)
from studio_backend.models import (
    ApplyApprovalRequest,
    GitApprovalRequest,
    RemoteApprovalRequest,
    StudioRun,
)
from studio_backend.repositories import WorkspaceRepositories
from studio_backend.storage import StudioStorage, utc_now
from tests.studio.helpers import DIGEST, application_bundle


class StudioApprovalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.workspace = root / "workspace"
        self.repository = self.workspace / "example"
        self.repository.mkdir(parents=True)
        self.storage = StudioStorage(root / "studio.sqlite3")
        now = utc_now()
        self.storage.create_run(
            StudioRun(
                id="approval-run",
                repository_id="example",
                task="Update sample.",
                status="verified",
                phase="approval_apply",
                created_at=now,
                updated_at=now,
                approval_digest=DIGEST,
            )
        )
        self.storage.put_artifact(
            "approval-run",
            "application_bundle",
            application_bundle().model_dump(mode="json"),
        )
        self.service = StudioApprovalService(
            self.storage,
            WorkspaceRepositories(self.workspace),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def apply_request(self, **updates) -> ApplyApprovalRequest:
        return ApplyApprovalRequest(
            approved=updates.get("approved", True),
            approval_digest=updates.get("approval_digest", DIGEST),
        )

    def test_apply_requires_approved_true(self) -> None:
        with self.assertRaises(ApprovalRejected):
            self.service.approve_apply(
                "approval-run",
                self.apply_request(approved=False),
            )

    def test_apply_requires_exact_digest(self) -> None:
        with self.assertRaises(ApprovalConflict):
            self.service.approve_apply(
                "approval-run",
                self.apply_request(approval_digest="e" * 64),
            )

    def test_apply_calls_existing_boundary_and_advances_state(self) -> None:
        expected = PlanApplicationResult(
            status="applied",
            applied_files=["sample.py"],
            approval_digest=DIGEST,
        )
        with patch(
            "studio_backend.approvals.apply_plan_application_bundle",
            return_value=expected,
        ) as apply_boundary:
            response = self.service.approve_apply(
                "approval-run",
                self.apply_request(),
            )
        apply_boundary.assert_called_once()
        self.assertTrue(apply_boundary.call_args.kwargs["approved"])
        self.assertEqual((response.run.status, response.run.phase), ("applied", "approval_git"))
        self.assertEqual(
            self.storage.get_artifact("approval-run", "application_result").payload[
                "status"
            ],
            "applied",
        )

    def test_apply_double_click_is_conflict_and_not_second_mutation(self) -> None:
        result = PlanApplicationResult(
            status="applied",
            applied_files=["sample.py"],
            approval_digest=DIGEST,
        )
        with patch(
            "studio_backend.approvals.apply_plan_application_bundle",
            return_value=result,
        ) as apply_boundary:
            self.service.approve_apply("approval-run", self.apply_request())
            with self.assertRaises(ApprovalConflict):
                self.service.approve_apply("approval-run", self.apply_request())
        self.assertEqual(apply_boundary.call_count, 1)

    def test_stale_apply_maps_to_stale_without_agent_rerun(self) -> None:
        result = PlanApplicationResult(
            status="stale",
            approval_digest=DIGEST,
            failure_reason="Repository state changed.",
        )
        with patch(
            "studio_backend.approvals.apply_plan_application_bundle",
            return_value=result,
        ):
            response = self.service.approve_apply(
                "approval-run",
                self.apply_request(),
            )
        self.assertEqual((response.run.status, response.run.phase), ("stale", "failed"))

    def _seed_applied(self) -> None:
        self.storage.put_artifact(
            "approval-run",
            "application_result",
            PlanApplicationResult(
                status="applied",
                applied_files=["sample.py"],
                approval_digest=DIGEST,
            ).model_dump(mode="json"),
        )
        self.storage.update_run(
            "approval-run",
            status="applied",
            phase="approval_git",
        )

    def test_git_requires_applied_state_and_exact_digest(self) -> None:
        request = GitApprovalRequest(approved=True, approval_digest=DIGEST)
        with self.assertRaises(ApprovalConflict):
            self.service.approve_git("approval-run", request)
        self._seed_applied()
        with self.assertRaises(ApprovalConflict):
            self.service.approve_git(
                "approval-run",
                request.model_copy(update={"approval_digest": "e" * 64}),
            )

    def test_git_calls_existing_boundary_and_advances_state(self) -> None:
        self._seed_applied()
        expected = LocalGitDeliveryResult(
            status="created",
            branch_name="repograph/approved",
            commit_sha="1" * 40,
            base_sha="2" * 40,
            approval_digest=DIGEST,
            committed_files=["sample.py"],
        )
        request = GitApprovalRequest(
            approved=True,
            approval_digest=DIGEST,
            branch_name="repograph/approved",
            commit_message="Approved change",
        )
        with patch(
            "studio_backend.approvals.create_local_git_delivery",
            return_value=expected,
        ) as git_boundary:
            response = self.service.approve_git("approval-run", request)
        git_boundary.assert_called_once()
        self.assertEqual(
            (response.run.status, response.run.phase),
            ("git_created", "approval_remote"),
        )

    def _seed_git_created(self) -> None:
        self._seed_applied()
        self.storage.put_artifact(
            "approval-run",
            "local_git_delivery_result",
            LocalGitDeliveryResult(
                status="created",
                branch_name="repograph/approved",
                commit_sha="1" * 40,
                base_sha="2" * 40,
                approval_digest=DIGEST,
                committed_files=["sample.py"],
            ).model_dump(mode="json"),
        )
        self.storage.update_run(
            "approval-run",
            status="git_created",
            phase="approval_remote",
        )

    def test_remote_requires_created_local_delivery(self) -> None:
        request = RemoteApprovalRequest(
            approved=True,
            approval_digest=DIGEST,
            base_branch="main",
        )
        with self.assertRaises(ApprovalConflict):
            self.service.approve_remote("approval-run", request)

    def test_remote_calls_existing_boundary_and_persists_pr(self) -> None:
        self._seed_git_created()
        expected = GitHubRemoteDeliveryResult(
            status="published",
            remote_name="origin",
            branch_name="repograph/approved",
            commit_sha="1" * 40,
            base_sha="2" * 40,
            base_branch="main",
            approval_digest=DIGEST,
            pushed=True,
            push_created=True,
            pr_created=True,
            pr_number=42,
            pr_url="https://github.com/example/project/pull/42",
        )
        request = RemoteApprovalRequest(
            approved=True,
            approval_digest=DIGEST,
            base_branch="main",
        )
        with patch(
            "studio_backend.approvals.publish_local_git_delivery",
            return_value=expected,
        ) as remote_boundary:
            response = self.service.approve_remote("approval-run", request)
        remote_boundary.assert_called_once()
        self.assertEqual((response.run.status, response.run.phase), ("published", "completed"))
        artifact = self.storage.get_artifact(
            "approval-run", "remote_delivery_result"
        ).payload
        self.assertEqual((artifact["pr_number"], artifact["pr_url"]), (42, expected.pr_url))

    def test_remote_partial_maps_to_partial(self) -> None:
        self._seed_git_created()
        result = GitHubRemoteDeliveryResult(
            status="partial",
            remote_name="origin",
            base_branch="main",
            approval_digest=DIGEST,
            pushed=True,
            failure_reason="PR creation did not complete.",
        )
        with patch(
            "studio_backend.approvals.publish_local_git_delivery",
            return_value=result,
        ):
            response = self.service.approve_remote(
                "approval-run",
                RemoteApprovalRequest(
                    approved=True,
                    approval_digest=DIGEST,
                    base_branch="main",
                ),
            )
        self.assertEqual(response.run.status, "partial")

    def test_failed_or_rejected_run_cannot_mutate(self) -> None:
        for status in ("failed", "rejected"):
            self.storage.update_run(
                "approval-run",
                status=status,
                phase="failed" if status == "failed" else "completed",
            )
            with self.assertRaises(ApprovalConflict):
                self.service.approve_apply("approval-run", self.apply_request())


class StudioApprovalIntegrationFlowTests(StudioApprovalTests):
    def test_verified_to_published_flow(self) -> None:
        applied = PlanApplicationResult(
            status="applied",
            applied_files=["sample.py"],
            approval_digest=DIGEST,
        )
        local = LocalGitDeliveryResult(
            status="created",
            branch_name="repograph/approved",
            commit_sha="1" * 40,
            base_sha="2" * 40,
            approval_digest=DIGEST,
            committed_files=["sample.py"],
        )
        remote = GitHubRemoteDeliveryResult(
            status="published",
            remote_name="origin",
            branch_name="repograph/approved",
            commit_sha="1" * 40,
            base_sha="2" * 40,
            base_branch="main",
            approval_digest=DIGEST,
            pushed=True,
            pr_created=True,
            pr_number=42,
            pr_url="https://github.com/example/project/pull/42",
        )
        with (
            patch(
                "studio_backend.approvals.apply_plan_application_bundle",
                return_value=applied,
            ),
            patch(
                "studio_backend.approvals.create_local_git_delivery",
                return_value=local,
            ),
            patch(
                "studio_backend.approvals.publish_local_git_delivery",
                return_value=remote,
            ),
        ):
            self.service.approve_apply("approval-run", self.apply_request())
            self.service.approve_git(
                "approval-run",
                GitApprovalRequest(approved=True, approval_digest=DIGEST),
            )
            published = self.service.approve_remote(
                "approval-run",
                RemoteApprovalRequest(
                    approved=True,
                    approval_digest=DIGEST,
                    base_branch="main",
                ),
            )
        self.assertEqual(published.run.status, "published")
        event_types = [
            event.event_type
            for event in self.storage.list_events("approval-run")
        ]
        self.assertIn("application_completed", event_types)
        self.assertIn("git_delivery_completed", event_types)
        self.assertIn("remote_delivery_completed", event_types)
        self.assertIn("github_pr_created", event_types)
        self.assertEqual(event_types[-1], "run_completed")
