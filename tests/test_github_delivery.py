import io
import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from pydantic import ValidationError

from git_delivery import (
    create_local_git_delivery,
    validate_local_git_delivery,
)
from github_delivery import (
    MAX_HTTP_RESPONSE_BYTES,
    MAX_PR_BODY_CHARS,
    MAX_PR_TITLE_CHARS,
    GitHubRemoteDeliveryResult,
    _DeliveryFailure,
    _git_environment,
    _GitHubRepository,
    _parse_github_remote_url,
    _push_create_only,
    _request_json,
    _resolve_github_repository,
    _validate_pr_body,
    _validate_pr_title,
    _verify_github_repository,
    publish_local_git_delivery,
    render_github_remote_delivery_result,
)
from tests.test_git_delivery import TemporaryGitDeliveryRepository


class FakeResponse:
    def __init__(self, payload: bytes, url: str = "https://api.github.com/test"):
        self.payload = payload
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def geturl(self) -> str:
        return self.url

    def read(self, limit: int) -> bytes:
        return self.payload[:limit]


class GitHubDeliverySchemaTests(unittest.TestCase):
    def test_result_schema_is_strict(self) -> None:
        with self.assertRaises(ValidationError):
            GitHubRemoteDeliveryResult.model_validate(
                {"status": "published", "unexpected": True}
            )

    def test_renderer_contains_bounded_delivery_metadata(self) -> None:
        result = GitHubRemoteDeliveryResult(
            status="published",
            remote_name="origin",
            owner="alice",
            repository="project",
            base_branch="main",
            branch_name="repograph/abc",
            commit_sha="a" * 40,
            pr_number=42,
            pr_url="https://github.com/alice/project/pull/42",
            pushed=True,
        )
        rendered = render_github_remote_delivery_result(result)
        self.assertIn("Status: published", rendered)
        self.assertIn("Repository: alice/project", rendered)
        self.assertIn("PR number: 42", rendered)

    def test_github_remote_identity_parsing_is_strict(self) -> None:
        accepted = (
            "https://github.com/alice/project.git",
            "https://github.com/alice/project",
            "git@github.com:alice/project.git",
            "git@github.com:alice/project",
            "ssh://git@github.com/alice/project.git",
        )
        for remote in accepted:
            with self.subTest(remote=remote):
                self.assertEqual(
                    _parse_github_remote_url(remote),
                    ("alice", "project"),
                )
        rejected = (
            "http://github.com/alice/project.git",
            "https://user:secret@github.com/alice/project.git",
            "https://gitlab.com/alice/project.git",
            "https://localhost/alice/project.git",
            "https://example.invalid/alice/project.git",
            "file:///tmp/project.git",
            "C:/project.git",
            "ssh://git@github.com:22/alice/project.git",
            "https://github.com/alice/project.git?token=secret",
        )
        for remote in rejected:
            with self.subTest(remote=remote), self.assertRaises(_DeliveryFailure):
                _parse_github_remote_url(remote)

    def test_git_environment_does_not_pass_github_token_or_prompts(self) -> None:
        with patch.dict(
            os.environ,
            {
                "GITHUB_TOKEN": "top-secret",
                "GH_TOKEN": "also-secret",
                "GIT_ASKPASS": "helper",
            },
            clear=False,
        ):
            environment = _git_environment()
        self.assertNotIn("GITHUB_TOKEN", environment)
        self.assertNotIn("GH_TOKEN", environment)
        self.assertNotIn("GIT_ASKPASS", environment)
        self.assertEqual(environment["GIT_TERMINAL_PROMPT"], "0")

    def test_http_boundary_is_bounded_and_redacts_token(self) -> None:
        url = "https://api.github.com/repos/alice/project"
        with (
            patch(
                "github_delivery.urlopen",
                return_value=FakeResponse(b"x" * (MAX_HTTP_RESPONSE_BYTES + 1), url),
            ),
            self.assertRaises(_DeliveryFailure) as oversized,
        ):
            _request_json("GET", url, "top-secret")
        self.assertIn("exceeded", str(oversized.exception))
        self.assertNotIn("top-secret", str(oversized.exception))

        with (
            patch(
                "github_delivery.urlopen",
                return_value=FakeResponse(b"{not-json", url),
            ),
            self.assertRaises(_DeliveryFailure),
        ):
            _request_json("GET", url, "top-secret")

        unauthorized = HTTPError(url, 401, "top-secret", {}, io.BytesIO())
        with (
            patch("github_delivery.urlopen", side_effect=unauthorized),
            self.assertRaises(_DeliveryFailure) as raised,
        ):
            _request_json("GET", url, "top-secret")
        self.assertIn("unauthorized", str(raised.exception))
        self.assertNotIn("top-secret", str(raised.exception))

    def test_http_statuses_and_timeouts_are_controlled(self) -> None:
        url = "https://api.github.com/repos/alice/project"
        for status in (403, 404, 422, 500):
            error = HTTPError(url, status, "secret-token", {}, io.BytesIO())
            with (
                self.subTest(status=status),
                patch("github_delivery.urlopen", side_effect=error),
                self.assertRaises(_DeliveryFailure) as raised,
            ):
                _request_json("GET", url, "secret-token")
            self.assertNotIn("secret-token", str(raised.exception))
        with (
            patch(
                "github_delivery.urlopen",
                side_effect=URLError(TimeoutError()),
            ),
            self.assertRaises(_DeliveryFailure) as raised,
        ):
            _request_json("GET", url, "secret-token")
        self.assertIn("timed out", str(raised.exception))

    def test_remote_config_requires_one_github_push_target(self) -> None:
        with (
            patch(
                "github_delivery._remote_config_values",
                return_value=[
                    "https://github.com/alice/one.git",
                    "https://github.com/alice/two.git",
                ],
            ),
            self.assertRaises(_DeliveryFailure),
        ):
            _resolve_github_repository(Path("."), "origin")

        with (
            patch(
                "github_delivery._remote_config_values",
                side_effect=[[], []],
            ),
            self.assertRaises(_DeliveryFailure),
        ):
            _resolve_github_repository(Path("."), "origin")

        with (
            patch(
                "github_delivery._remote_config_values",
                side_effect=[["git@github.com:alice/project.git"]],
            ) as values,
            patch("github_delivery._reject_matching_url_rewrites") as rewrites,
        ):
            identity = _resolve_github_repository(Path("."), "origin")
        self.assertEqual((identity.owner, identity.repository), ("alice", "project"))
        values.assert_called_once_with(Path("."), "remote.origin.pushurl")
        rewrites.assert_called_once_with(
            Path("."),
            "git@github.com:alice/project.git",
        )

        for remote_name in ("", "../origin", "origin/name", "x" * 101):
            with (
                self.subTest(remote_name=remote_name),
                self.assertRaises(_DeliveryFailure),
            ):
                _resolve_github_repository(Path("."), remote_name)

    def test_authenticated_repository_identity_must_match_remote(self) -> None:
        identity = _GitHubRepository(
            owner="alice",
            repository="project",
            push_target="https://github.com/alice/project.git",
        )
        with (
            patch(
                "github_delivery._request_json",
                return_value={"owner": {"login": "mallory"}, "name": "project"},
            ),
            self.assertRaises(_DeliveryFailure) as raised,
        ):
            _verify_github_repository(identity, "secret-token")
        self.assertEqual(raised.exception.status, "conflict")


class GitHubDeliveryIntegrationTests(TemporaryGitDeliveryRepository):
    def setUp(self) -> None:
        super().setUp()
        self.bare = self.base / "origin.git"
        subprocess.run(
            ["git", "init", "--bare", str(self.bare)],
            cwd=self.base,
            capture_output=True,
            text=True,
            check=True,
        )
        self.git("remote", "add", "origin", str(self.bare))
        self.git("push", "origin", "main")
        self.application_bundle = self.bundle()
        self.apply(self.application_bundle)
        self.local_result = create_local_git_delivery(
            str(self.repository),
            self.application_bundle,
            approved=True,
        )
        self.assertEqual(self.local_result.status, "created")
        self.delivery = validate_local_git_delivery(
            str(self.repository),
            self.application_bundle,
        )
        self.identity = _GitHubRepository(
            owner="alice",
            repository="project",
            push_target=str(self.bare),
        )

    def pr_payload(self, number: int = 42, *, head_sha: str | None = None):
        return {
            "number": number,
            "html_url": f"https://github.com/alice/project/pull/{number}",
            "state": "open",
            "head": {
                "ref": self.delivery.branch_name,
                "sha": head_sha or self.delivery.commit_sha,
            },
            "base": {"ref": "main", "sha": self.delivery.base_sha},
        }

    def publish(self, api_side_effect):
        with (
            patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"}, clear=False),
            patch(
                "github_delivery._resolve_github_repository",
                return_value=self.identity,
            ),
            patch("github_delivery._verify_github_repository"),
            patch(
                "github_delivery._request_json",
                side_effect=api_side_effect,
            ),
        ):
            return publish_local_git_delivery(
                str(self.repository),
                self.application_bundle,
                approved=True,
                base_branch="main",
            )

    def test_approved_false_has_zero_git_or_network_activity(self) -> None:
        with (
            patch("github_delivery.validate_local_git_delivery") as validate,
            patch("github_delivery.urlopen") as http,
            patch("github_delivery.subprocess.run") as git,
        ):
            result = publish_local_git_delivery(
                str(self.repository),
                self.application_bundle,
                approved=False,
                base_branch="main",
            )
        self.assertEqual(result.status, "not_requested")
        validate.assert_not_called()
        http.assert_not_called()
        git.assert_not_called()

    def test_production_publish_rejects_non_github_remote_targets(self) -> None:
        rejected = (
            str(self.bare),
            self.bare.resolve().as_uri(),
            "https://localhost/alice/project.git",
            "https://example.invalid/alice/project.git",
        )
        for target in rejected:
            self.git("remote", "set-url", "origin", target)
            with (
                self.subTest(target=target),
                patch("github_delivery._request_json") as http,
                patch("github_delivery._push_create_only") as push,
            ):
                result = publish_local_git_delivery(
                    str(self.repository),
                    self.application_bundle,
                    approved=True,
                    base_branch="main",
                )
            self.assertEqual(result.status, "error")
            http.assert_not_called()
            push.assert_not_called()

    def test_production_publish_rejects_matching_git_url_rewrite(self) -> None:
        github_url = "https://github.com/alice/project.git"
        self.git("remote", "set-url", "origin", github_url)
        self.git(
            "config",
            f"url.{self.bare.resolve().as_uri()}.insteadOf",
            github_url,
        )
        with (
            patch("github_delivery._request_json") as http,
            patch("github_delivery._push_create_only") as push,
        ):
            result = publish_local_git_delivery(
                str(self.repository),
                self.application_bundle,
                approved=True,
                base_branch="main",
            )
        self.assertEqual(result.status, "error")
        self.assertIn("rewrite", result.failure_reason or "")
        http.assert_not_called()
        push.assert_not_called()

    def test_successful_publish_pushes_exact_commit_and_creates_pr(self) -> None:
        before_head = self.git("rev-parse", "HEAD").strip()
        before_branch = self.git("symbolic-ref", "HEAD").strip()
        before_status = self.git("status", "--porcelain=v1", "--untracked-files=all")
        before_index = self.index_bytes()
        result = self.publish([[], self.pr_payload()])
        self.assertEqual(result.status, "published")
        self.assertTrue(result.pushed)
        self.assertTrue(result.push_created)
        self.assertTrue(result.pr_created)
        self.assertEqual(result.pr_number, 42)
        remote_sha = subprocess.run(
            [
                "git",
                "--git-dir",
                str(self.bare),
                "rev-parse",
                f"refs/heads/{self.delivery.branch_name}",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.assertEqual(remote_sha, self.delivery.commit_sha)
        self.assertEqual(self.git("rev-parse", "HEAD").strip(), before_head)
        self.assertEqual(self.git("symbolic-ref", "HEAD").strip(), before_branch)
        self.assertEqual(self.index_bytes(), before_index)
        self.assertEqual(
            self.git("status", "--porcelain=v1", "--untracked-files=all"),
            before_status,
        )
        self.assertEqual(
            self.branch_ref(self.delivery.branch_name),
            self.delivery.commit_sha,
        )

    def test_same_sha_and_existing_pr_are_idempotent(self) -> None:
        first = self.publish([[], self.pr_payload()])
        self.assertEqual(first.status, "published")
        with patch("github_delivery._push_create_only") as push:
            second = self.publish([[self.pr_payload()]])
        self.assertEqual(second.status, "published")
        self.assertTrue(second.pushed)
        self.assertFalse(second.push_created)
        self.assertFalse(second.pr_created)
        push.assert_not_called()

    def test_missing_token_is_rejected_before_push(self) -> None:
        with (
            patch.dict(os.environ, {"GITHUB_TOKEN": ""}, clear=False),
            patch(
                "github_delivery._resolve_github_repository",
                return_value=self.identity,
            ),
            patch("github_delivery._push_create_only") as push,
        ):
            result = publish_local_git_delivery(
                str(self.repository),
                self.application_bundle,
                approved=True,
                base_branch="main",
            )
        self.assertEqual(result.status, "error")
        self.assertIn("GITHUB_TOKEN", result.failure_reason)
        push.assert_not_called()

    def test_remote_base_moved_is_stale_without_push_or_pr(self) -> None:
        tree = self.git("rev-parse", "HEAD^{tree}").strip()
        advanced = self.git(
            "commit-tree",
            tree,
            "-p",
            self.initial_head,
            "-m",
            "Advance remote base",
        ).strip()
        self.git("push", "origin", f"{advanced}:refs/heads/main")
        with patch("github_delivery._push_create_only") as push:
            result = self.publish([])
        self.assertEqual(result.status, "stale")
        push.assert_not_called()

    def test_remote_branch_conflict_never_overwrites(self) -> None:
        tree = self.git("rev-parse", "HEAD^{tree}").strip()
        other = self.git(
            "commit-tree",
            tree,
            "-p",
            self.initial_head,
            "-m",
            "Conflicting branch",
        ).strip()
        self.git(
            "push",
            "origin",
            f"{other}:refs/heads/{self.delivery.branch_name}",
        )
        with patch("github_delivery._push_create_only") as push:
            result = self.publish([])
        self.assertEqual(result.status, "conflict")
        push.assert_not_called()
        remote_sha = subprocess.run(
            [
                "git",
                "--git-dir",
                str(self.bare),
                "rev-parse",
                f"refs/heads/{self.delivery.branch_name}",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.assertEqual(remote_sha, other)

    def test_create_only_push_race_cannot_overwrite_new_remote_ref(self) -> None:
        tree = self.git("rev-parse", "HEAD^{tree}").strip()
        other = self.git(
            "commit-tree",
            tree,
            "-p",
            self.initial_head,
            "-m",
            "Racing branch",
        ).strip()

        def racing_push(root, remote_name, delivery):
            self.git(
                "push",
                "origin",
                f"{other}:refs/heads/{delivery.branch_name}",
            )
            _push_create_only(root, remote_name, delivery)

        with patch("github_delivery._push_create_only", side_effect=racing_push):
            result = self.publish([])
        self.assertEqual(result.status, "conflict")
        remote_sha = subprocess.run(
            [
                "git",
                "--git-dir",
                str(self.bare),
                "rev-parse",
                f"refs/heads/{self.delivery.branch_name}",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.assertEqual(remote_sha, other)

    def test_create_only_push_uses_exact_sha_refspec_and_no_force(self) -> None:
        with patch("github_delivery._run_git") as run_git:
            run_git.return_value.returncode = 0
            run_git.return_value.stdout = ""
            run_git.return_value.stderr = ""
            run_git.return_value.output_truncated = False
            _push_create_only(self.repository, "origin", self.delivery)
        args = run_git.call_args.args[1]
        full_ref = f"refs/heads/{self.delivery.branch_name}"
        self.assertEqual(args[0:2], ["push", "--porcelain"])
        self.assertIn(f"--force-with-lease={full_ref}:", args)
        self.assertIn(f"{self.delivery.commit_sha}:{full_ref}", args)
        self.assertNotIn("--force", args)
        self.assertNotIn(self.delivery.branch_name, args[-1].split(":")[0])

    def test_push_success_then_pr_failure_is_partial_and_retryable(self) -> None:
        first = self.publish(
            [[], _DeliveryFailure("error", "GitHub API request timed out.")]
        )
        self.assertEqual(first.status, "partial")
        self.assertTrue(first.pushed)
        self.assertTrue(first.push_created)
        self.assertFalse(first.pr_created)

        with patch("github_delivery._push_create_only") as push:
            retry = self.publish([[self.pr_payload()]])
        self.assertEqual(retry.status, "published")
        self.assertFalse(retry.push_created)
        self.assertFalse(retry.pr_created)
        push.assert_not_called()

    def test_existing_pr_wrong_sha_and_duplicate_lookup_are_conflicts(self) -> None:
        self.publish([[], self.pr_payload()])
        wrong_sha = "f" * len(self.delivery.commit_sha)
        wrong = self.publish([[self.pr_payload(head_sha=wrong_sha)]])
        self.assertEqual(wrong.status, "conflict")
        duplicate = self.publish([[self.pr_payload(42), self.pr_payload(43)]])
        self.assertEqual(duplicate.status, "conflict")

    def test_unrelated_staged_and_untracked_state_is_preserved(self) -> None:
        staged = self.repository / "staged.txt"
        staged.write_text("staged\n", encoding="utf-8")
        self.git("add", "--", "staged.txt")
        untracked = self.repository / "untracked.txt"
        untracked.write_text("untracked\n", encoding="utf-8")
        before_status = self.git("status", "--porcelain=v1", "--untracked-files=all")
        before_index = self.index_bytes()
        result = self.publish([[], self.pr_payload()])
        self.assertEqual(result.status, "published")
        self.assertEqual(self.index_bytes(), before_index)
        self.assertEqual(
            self.git("status", "--porcelain=v1", "--untracked-files=all"),
            before_status,
        )

    def test_local_branch_moved_after_g5a_is_stale(self) -> None:
        tree = self.git("rev-parse", "HEAD^{tree}").strip()
        other = self.git(
            "commit-tree",
            tree,
            "-p",
            self.initial_head,
            "-m",
            "Moved local delivery branch",
        ).strip()
        self.git("branch", "-f", self.delivery.branch_name, other)
        with patch("github_delivery._push_create_only") as push:
            result = self.publish([])
        self.assertEqual(result.status, "stale")
        push.assert_not_called()

    def test_default_and_custom_pr_text_are_bounded(self) -> None:
        self.assertEqual(
            _validate_pr_title(None, self.delivery),
            "RepoGraph: apply approved repository plan",
        )
        self.assertEqual(
            _validate_pr_title("Custom title", self.delivery),
            "Custom title",
        )
        body = _validate_pr_body(None, self.delivery)
        self.assertIn(self.delivery.approval_digest, body)
        self.assertIn(self.delivery.commit_sha, body)
        self.assertNotIn(str(self.repository), body)
        self.assertEqual(_validate_pr_body("Custom body", self.delivery), "Custom body")
        for title in ("x" * (MAX_PR_TITLE_CHARS + 1), "bad\ntitle", "bad\x00title"):
            with self.subTest(title=title), self.assertRaises(_DeliveryFailure):
                _validate_pr_title(title, self.delivery)
        with self.assertRaises(_DeliveryFailure):
            _validate_pr_body("x" * (MAX_PR_BODY_CHARS + 1), self.delivery)

    def test_post_payload_is_minimal_and_exact(self) -> None:
        calls = []

        def request(method, url, _token, *, payload=None):
            calls.append((method, url, payload))
            if method == "GET":
                return []
            return self.pr_payload()

        with (
            patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"}, clear=False),
            patch(
                "github_delivery._resolve_github_repository",
                return_value=self.identity,
            ),
            patch("github_delivery._verify_github_repository"),
            patch("github_delivery._request_json", side_effect=request),
        ):
            result = publish_local_git_delivery(
                str(self.repository),
                self.application_bundle,
                approved=True,
                base_branch="main",
                pr_title="Exact title",
                pr_body="Exact body",
            )
        self.assertEqual(result.status, "published")
        post = calls[-1]
        self.assertEqual(post[0], "POST")
        self.assertEqual(
            post[1],
            "https://api.github.com/repos/alice/project/pulls",
        )
        self.assertEqual(
            post[2],
            {
                "title": "Exact title",
                "head": f"alice:{self.delivery.branch_name}",
                "base": "main",
                "body": "Exact body",
                "draft": False,
            },
        )
        for forbidden in (
            "merge",
            "labels",
            "reviewers",
            "assignees",
            "projects",
            "milestones",
        ):
            self.assertNotIn(forbidden, post[2])


if __name__ == "__main__":
    unittest.main()
