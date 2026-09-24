from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

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


class StudioVerticalSliceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="repograph-studio-system-"
        )
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.repository, self.bare_remote, self.initial_sha = initialize_repository(
            self.workspace,
        )
        self.index_path = self.repository / ".git" / "index"
        self.initial_index_hash = file_hash(self.index_path)
        self.initial_branch = run_git(
            self.repository,
            "symbolic-ref",
            "--short",
            "HEAD",
        )
        self.remote_boundary = InjectedRemoteBoundary()
        self.app = build_test_app(
            self.workspace,
            self.root / "studio-data",
            remote_boundary=self.remote_boundary,
        )
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.temporary.cleanup()

    def artifact(self, run_id: str, kind: str) -> dict[str, object]:
        response = self.client.get(f"/api/runs/{run_id}/artifacts/{kind}")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()["payload"]
        self.assertIsInstance(payload, dict)
        return payload

    def assert_git_control_state_unchanged(self) -> None:
        self.assertEqual(run_git(self.repository, "rev-parse", "HEAD"), self.initial_sha)
        self.assertEqual(
            run_git(self.repository, "symbolic-ref", "--short", "HEAD"),
            self.initial_branch,
        )
        self.assertEqual(file_hash(self.index_path), self.initial_index_hash)

    def test_real_api_graph_apply_git_and_injected_remote_wiring(self) -> None:
        """Exercise real Studio orchestration through the injected G5B adapter."""
        detail = start_and_wait_verified(self.client)
        run_id = str(detail["id"])
        digest = str(detail["approval_digest"])

        self.assertEqual(
            (self.repository / "src" / "calculator.py").read_text(
                encoding="utf-8"
            ),
            ORIGINAL_SOURCE,
        )
        self.assertEqual(run_git(self.repository, "status", "--porcelain"), "")
        self.assert_git_control_state_unchanged()

        plan = self.artifact(run_id, "engineering_plan")
        candidate = self.artifact(run_id, "candidate")
        diff = self.artifact(run_id, "candidate_diff")
        verification = self.artifact(run_id, "verification")
        review = self.artifact(run_id, "change_set_review")
        bundle = self.artifact(run_id, "application_bundle")
        self.assertEqual(plan["files"][0]["path"], "src/calculator.py")
        self.assertEqual(candidate["files"][0]["content"], FIXED_SOURCE)
        self.assertIn("return a + b", diff["diff_text"])
        self.assertEqual(verification["status"], "verified")
        self.assertEqual(verification["test_result"]["status"], "passed")
        self.assertEqual(review["overall_rating"], "good")
        self.assertEqual(bundle["approval_digest"], digest)

        rejected_digest = self.client.post(
            f"/api/runs/{run_id}/approve/apply",
            json={"approved": True, "approval_digest": "e" * 64},
        )
        self.assertEqual(rejected_digest.status_code, 409)
        self.assertEqual(
            (self.repository / "src" / "calculator.py").read_text(
                encoding="utf-8"
            ),
            ORIGINAL_SOURCE,
        )
        self.assert_git_control_state_unchanged()

        applied = self.client.post(
            f"/api/runs/{run_id}/approve/apply",
            json={"approved": True, "approval_digest": digest},
        )
        self.assertEqual(applied.status_code, 200, applied.text)
        self.assertEqual(applied.json()["run"]["status"], "applied")
        self.assertEqual(
            (self.repository / "src" / "calculator.py").read_text(
                encoding="utf-8"
            ),
            FIXED_SOURCE,
        )
        self.assert_git_control_state_unchanged()

        created = self.client.post(
            f"/api/runs/{run_id}/approve/git",
            json={
                "approved": True,
                "approval_digest": digest,
                "branch_name": None,
                "commit_message": None,
            },
        )
        self.assertEqual(created.status_code, 200, created.text)
        self.assertEqual(
            created.json()["run"]["status"],
            "git_created",
            self.client.get(
                f"/api/runs/{run_id}/artifacts/local_git_delivery_result"
            ).text,
        )
        local = self.artifact(run_id, "local_git_delivery_result")
        branch_name = str(local["branch_name"])
        commit_sha = str(local["commit_sha"])
        self.assertTrue(branch_name.startswith("repograph/"))
        self.assertEqual(run_git(self.repository, "rev-parse", branch_name), commit_sha)
        self.assertEqual(
            run_git(self.repository, "rev-parse", f"{commit_sha}^"),
            self.initial_sha,
        )
        committed_source = run_git(
            self.repository,
            "show",
            f"{commit_sha}:src/calculator.py",
        )
        self.assertEqual(f"{committed_source}\n", FIXED_SOURCE)
        self.assert_git_control_state_unchanged()

        self.assertIsNone(self.bare_remote)
        published = self.client.post(
            f"/api/runs/{run_id}/approve/remote",
            json={
                "approved": True,
                "approval_digest": digest,
                "remote_name": "origin",
                "base_branch": "main",
                "branch_name": branch_name,
                "pr_title": "Fix calculator regression",
                "pr_body": None,
            },
        )
        self.assertEqual(published.status_code, 200, published.text)
        self.assertEqual(published.json()["run"]["status"], "published")
        self.assertEqual(len(self.remote_boundary.calls), 1)
        remote_call = self.remote_boundary.calls[0]
        self.assertTrue(remote_call["approved"])
        self.assertEqual(remote_call["local_branch"], branch_name)
        self.assertEqual(remote_call["remote_name"], "origin")
        self.assertEqual(remote_call["base_branch"], "main")
        remote = self.artifact(run_id, "remote_delivery_result")
        self.assertEqual(remote["approval_digest"], digest)
        self.assertTrue(remote["pushed"])
        self.assertTrue(remote["pr_created"])
        self.assert_git_control_state_unchanged()

        sensitive = (
            str(self.workspace.resolve()),
            str((self.root / "studio-data").resolve()),
            str(self.root.resolve()),
        )
        final_detail = self.client.get(f"/api/runs/{run_id}").json()
        public_payloads = [
            final_detail,
            self.client.get(f"/api/runs/{run_id}/events").json(),
            *[
                self.client.get(
                    f"/api/runs/{run_id}/artifacts/{kind}"
                ).json()
                for kind in final_detail["artifact_kinds"]
            ],
        ]
        rendered = json.dumps(public_payloads, ensure_ascii=False)
        for value in sensitive:
            self.assertNotIn(value, rendered)

    def test_persisted_history_supports_disconnect_and_reconnect(self) -> None:
        detail = start_and_wait_verified(self.client)
        run_id = str(detail["id"])
        history = self.client.get(f"/api/runs/{run_id}/events").json()["events"]
        self.assertGreater(len(history), 5)
        disconnected_after = history[3]["sequence"]
        replay = self.client.get(
            f"/api/runs/{run_id}/events?after={disconnected_after}"
        ).json()["events"]
        self.assertEqual(
            [event["sequence"] for event in replay],
            list(range(disconnected_after + 1, history[-1]["sequence"] + 1)),
        )
        self.assertEqual(
            history[:4] + replay,
            history,
        )


if __name__ == "__main__":
    unittest.main()
