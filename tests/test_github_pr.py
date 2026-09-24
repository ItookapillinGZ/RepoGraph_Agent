import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from email.message import Message
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from pydantic import ValidationError

from agent import (
    ChangeSetSummary,
    CodeReview,
    OverallRating,
    main,
    review_github_pr,
)
from change_set import MAX_CHANGESET_DIFF_CHARS, ChangeSetReview
from diff_context import parse_unified_diff
from git_diff import GitDiffError, resolve_local_head
from github_pr import (
    GITHUB_HTTP_TIMEOUT_SECONDS,
    MAX_PR_BODY_CHARS,
    MAX_PR_DIFF_BYTES,
    MAX_PR_METADATA_BYTES,
    GitHubPRContext,
    GitHubPRError,
    GitHubPRMetadata,
    GitHubPRReference,
    GitHubPRReview,
    fetch_github_pr,
    fetch_github_pr_url,
    parse_github_pr_url,
    verify_local_pr_head,
)
from repository_context import RepoContext
from static_analysis import StaticAnalysisResult

HEAD_SHA = "a" * 40
PR_URL = "https://github.com/octo-org/example/pull/42"


def pr_diff() -> str:
    return (
        "diff --git a/src/service.py b/src/service.py\n"
        "--- a/src/service.py\n"
        "+++ b/src/service.py\n"
        "@@ -1 +1 @@\n"
        "-SERVICE = 1\n"
        "+SERVICE = 2\n"
        "diff --git a/src/repository.py b/src/repository.py\n"
        "--- a/src/repository.py\n"
        "+++ b/src/repository.py\n"
        "@@ -1 +1 @@\n"
        "-REPOSITORY = 1\n"
        "+REPOSITORY = 2\n"
        "diff --git a/README.md b/README.md\n"
        "--- a/README.md\n"
        "+++ b/README.md\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )


def metadata_payload(**updates) -> dict[str, object]:
    payload: dict[str, object] = {
        "number": 42,
        "title": "Fix user lookup",
        "body": "Keep repository and service behavior consistent.",
        "state": "open",
        "user": {"login": "octocat"},
        "head": {"ref": "feature/user-fix", "sha": HEAD_SHA},
        "base": {"ref": "main"},
        "changed_files": 3,
        "html_url": PR_URL,
    }
    payload.update(updates)
    return payload


def pr_context(**updates) -> GitHubPRContext:
    payload = {
        "owner": "octo-org",
        "repository": "example",
        "number": 42,
        "title": "Fix user lookup",
        "body": "Bounded body.",
        "state": "open",
        "author": "octocat",
        "head_ref": "feature/user-fix",
        "base_ref": "main",
        "head_sha": HEAD_SHA,
        "changed_files": 3,
        "html_url": PR_URL,
        "diff_text": pr_diff(),
    }
    payload.update(updates)
    return GitHubPRContext(**payload)


def good_review() -> CodeReview:
    return CodeReview(
        overall_rating=OverallRating.GOOD,
        summary="No file-level issues found.",
        findings=[],
    )


def good_summary() -> ChangeSetSummary:
    return ChangeSetSummary(
        overall_rating=OverallRating.GOOD,
        summary="The reviewed Python files are consistent.",
        high_risk_files=[],
    )


def github_review() -> GitHubPRReview:
    return GitHubPRReview(
        pull_request=pr_context().metadata(),
        review=ChangeSetReview(
            overall_rating=OverallRating.GOOD,
            summary="Good.",
            file_results=[],
        ),
    )


class FakeResponse:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.read_limits: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self, limit: int) -> bytes:
        self.read_limits.append(limit)
        return self.data[:limit]


class GitHubPRSchemaTests(unittest.TestCase):
    def test_context_schema_forbids_extra_fields(self) -> None:
        payload = pr_context().model_dump()
        payload["token"] = "secret"
        with self.assertRaises(ValidationError):
            GitHubPRContext.model_validate(payload)

    def test_metadata_and_review_schemas_forbid_extra_fields(self) -> None:
        metadata = pr_context().metadata().model_dump()
        metadata["unexpected"] = True
        with self.assertRaises(ValidationError):
            GitHubPRMetadata.model_validate(metadata)
        payload = github_review().model_dump()
        payload["unexpected"] = True
        with self.assertRaises(ValidationError):
            GitHubPRReview.model_validate(payload)

    def test_context_metadata_excludes_raw_diff_and_warnings(self) -> None:
        context = pr_context(warnings=["bounded"])
        rendered = context.metadata().model_dump()
        self.assertNotIn("diff_text", rendered)
        self.assertNotIn("warnings", rendered)


class GitHubPRURLTests(unittest.TestCase):
    def test_valid_url_parses_owner_repository_and_number(self) -> None:
        result = parse_github_pr_url(PR_URL)
        self.assertEqual(
            result,
            GitHubPRReference(
                owner="octo-org",
                repository="example",
                number=42,
            ),
        )

    def test_non_integer_pr_number_is_rejected(self) -> None:
        with self.assertRaisesRegex(GitHubPRError, "positive integer") as captured:
            parse_github_pr_url(
                "https://github.com/octo-org/example/pull/not-a-number"
            )
        self.assertEqual(captured.exception.code, "invalid_pr_reference")

    def test_wrong_host_is_rejected(self) -> None:
        with self.assertRaises(GitHubPRError) as captured:
            parse_github_pr_url("https://example.com/octo-org/example/pull/42")
        self.assertEqual(captured.exception.code, "invalid_pr_reference")

    def test_malformed_or_extra_paths_are_rejected(self) -> None:
        invalid = [
            "https://github.com/example/pull/42",
            "https://github.com//example/pull/42",
            "https://github.com/octo-org//pull/42",
            "https://github.com/octo-org/example/pull/42/files",
            "https://github.com/octo-org/example/pull/42/",
            "https://github.com/octo-org/example/issues/42",
            "http://github.com/octo-org/example/pull/42",
            "https://github.com/octo-org/example/pull/42?x=1",
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(GitHubPRError):
                parse_github_pr_url(value)

    @patch("github_pr.fetch_github_pr")
    def test_url_fetch_is_only_a_convenience_layer(self, fetch) -> None:
        fetch.return_value = pr_context()
        result = fetch_github_pr_url(PR_URL)
        self.assertEqual(result, pr_context())
        fetch.assert_called_once_with("octo-org", "example", 42)


class GitHubPRHTTPTests(unittest.TestCase):
    def _fetch(
        self,
        metadata: dict[str, object] | bytes | None = None,
        diff: bytes | None = None,
        *,
        token: str | None = None,
    ):
        metadata_bytes = (
            metadata
            if isinstance(metadata, bytes)
            else json.dumps(metadata or metadata_payload()).encode("utf-8")
        )
        metadata_response = FakeResponse(metadata_bytes)
        diff_response = FakeResponse(diff if diff is not None else pr_diff().encode())
        captured: list[tuple[object, int]] = []

        def opener(request, timeout):
            captured.append((request, timeout))
            return [metadata_response, diff_response][len(captured) - 1]

        environment = {} if token is None else {"GITHUB_TOKEN": token}
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("github_pr.urlopen", side_effect=opener),
        ):
            context = fetch_github_pr("octo-org", "example", 42)
        return context, captured, metadata_response, diff_response

    def test_metadata_and_diff_are_get_requests_with_expected_accepts(self) -> None:
        _context, captured, _metadata, _diff = self._fetch()
        self.assertEqual(len(captured), 2)
        methods = [request.get_method() for request, _timeout in captured]
        self.assertEqual(methods, ["GET", "GET"])
        headers = [
            {key.casefold(): value for key, value in request.header_items()}
            for request, _timeout in captured
        ]
        self.assertEqual(headers[0]["accept"], "application/vnd.github+json")
        self.assertEqual(headers[1]["accept"], "application/vnd.github.diff")
        self.assertTrue(all("user-agent" in item for item in headers))
        self.assertTrue(all("x-github-api-version" in item for item in headers))

    def test_all_requests_use_bounded_timeout_and_reads(self) -> None:
        _context, captured, metadata, diff = self._fetch()
        self.assertEqual(
            [timeout for _request, timeout in captured],
            [GITHUB_HTTP_TIMEOUT_SECONDS, GITHUB_HTTP_TIMEOUT_SECONDS],
        )
        self.assertEqual(metadata.read_limits, [MAX_PR_METADATA_BYTES + 1])
        self.assertEqual(diff.read_limits, [MAX_PR_DIFF_BYTES + 1])

    def test_anonymous_request_has_no_authorization_header(self) -> None:
        _context, captured, _metadata, _diff = self._fetch()
        for request, _timeout in captured:
            headers = {key.casefold(): value for key, value in request.header_items()}
            self.assertNotIn("authorization", headers)

    def test_token_adds_bearer_authorization_to_both_gets(self) -> None:
        _context, captured, _metadata, _diff = self._fetch(token="top-secret")
        for request, _timeout in captured:
            headers = {key.casefold(): value for key, value in request.header_items()}
            self.assertEqual(headers["authorization"], "Bearer top-secret")

    def test_token_cannot_enter_result_even_if_response_echoes_it(self) -> None:
        token = "top-secret"
        payload = metadata_payload(
            title=f"Title {token}",
            body=f"Body {token}",
            head={"ref": f"branch-{token}", "sha": HEAD_SHA},
        )
        context, _captured, _metadata, _diff = self._fetch(
            payload,
            f"--- a/a.py\n+++ b/a.py\n+{token}\n".encode(),
            token=token,
        )
        self.assertNotIn(token, context.model_dump_json())

    def test_token_cannot_enter_network_error(self) -> None:
        token = "top-secret"
        with (
            patch.dict(os.environ, {"GITHUB_TOKEN": token}, clear=True),
            patch("github_pr.urlopen", side_effect=URLError(token)),
            self.assertRaises(GitHubPRError) as captured,
        ):
            fetch_github_pr("octo-org", "example", 42)
        self.assertNotIn(token, str(captured.exception))
        self.assertNotIn(token, json.dumps(captured.exception.to_payload()))
        self.assertIsNone(captured.exception.__cause__)

    def _http_failure(self, status: int, *, remaining: str | None = None):
        headers = Message()
        if remaining is not None:
            headers["X-RateLimit-Remaining"] = remaining
        error = HTTPError(
            "https://api.github.com/",
            status,
            "failure",
            headers,
            io.BytesIO(b"{}"),
        )
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("github_pr.urlopen", side_effect=error),
            self.assertRaises(GitHubPRError) as captured,
        ):
            fetch_github_pr("octo-org", "example", 42)
        return captured.exception

    def test_404_is_not_found(self) -> None:
        self.assertEqual(self._http_failure(404).code, "not_found")

    def test_401_is_unauthorized(self) -> None:
        self.assertEqual(self._http_failure(401).code, "unauthorized")

    def test_403_is_forbidden_without_exhausted_rate_limit(self) -> None:
        self.assertEqual(self._http_failure(403, remaining="10").code, "forbidden")

    def test_403_with_exhausted_rate_limit_is_rate_limited(self) -> None:
        self.assertEqual(
            self._http_failure(403, remaining="0").code,
            "rate_limited",
        )

    def test_429_is_rate_limited(self) -> None:
        self.assertEqual(self._http_failure(429).code, "rate_limited")

    def test_timeout_is_normalized(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("github_pr.urlopen", side_effect=TimeoutError()),
            self.assertRaises(GitHubPRError) as captured,
        ):
            fetch_github_pr("octo-org", "example", 42)
        self.assertEqual(captured.exception.code, "timeout")

    def test_url_error_is_normalized(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("github_pr.urlopen", side_effect=URLError("offline")),
            self.assertRaises(GitHubPRError) as captured,
        ):
            fetch_github_pr("octo-org", "example", 42)
        self.assertEqual(captured.exception.code, "network_error")

    def test_malformed_json_is_invalid_response(self) -> None:
        with self.assertRaises(GitHubPRError) as captured:
            self._fetch(metadata=b"not-json")
        self.assertEqual(captured.exception.code, "invalid_response")

    def test_missing_required_metadata_is_invalid_response(self) -> None:
        with self.assertRaises(GitHubPRError) as captured:
            self._fetch(metadata={"number": 42, "title": "missing head"})
        self.assertEqual(captured.exception.code, "invalid_response")

    def test_metadata_response_size_is_bounded(self) -> None:
        oversized = b"{" + b"x" * MAX_PR_METADATA_BYTES
        with self.assertRaises(GitHubPRError) as captured:
            self._fetch(metadata=oversized)
        self.assertEqual(captured.exception.code, "invalid_response")

    def test_pr_body_is_bounded_with_warning(self) -> None:
        context, _captured, _metadata, _diff = self._fetch(
            metadata_payload(body="x" * (MAX_PR_BODY_CHARS + 1))
        )
        self.assertEqual(len(context.body), MAX_PR_BODY_CHARS)
        self.assertIn("MAX_PR_BODY_CHARS", " ".join(context.warnings))

    def test_diff_is_bounded_and_warns_review_is_incomplete(self) -> None:
        oversized = b"x" * (MAX_PR_DIFF_BYTES + 1)
        context, _captured, _metadata, _diff = self._fetch(diff=oversized)
        self.assertEqual(len(context.diff_text), MAX_CHANGESET_DIFF_CHARS)
        warning = " ".join(context.warnings)
        self.assertIn("MAX_CHANGESET_DIFF_CHARS", warning)
        self.assertIn("not fully reviewed", warning)

    def test_github_diff_enters_existing_unified_diff_parser(self) -> None:
        context, _captured, _metadata, _diff = self._fetch()
        parsed = parse_unified_diff(context.diff_text, "src/service.py")
        self.assertEqual(parsed.target_file, "src/service.py")
        self.assertIn(1, parsed.changed_new_lines)


class LocalHeadCompatibilityTests(unittest.TestCase):
    @patch("github_pr.resolve_local_head", return_value=HEAD_SHA)
    def test_matching_local_head_is_accepted(self, resolver) -> None:
        self.assertEqual(verify_local_pr_head("repo", HEAD_SHA), [])
        resolver.assert_called_once_with("repo")

    @patch("github_pr.resolve_local_head", return_value="b" * 40)
    def test_mismatched_local_head_is_rejected_by_default(self, _resolver) -> None:
        with self.assertRaises(GitHubPRError) as captured:
            verify_local_pr_head("repo", HEAD_SHA)
        self.assertEqual(captured.exception.code, "head_mismatch")
        self.assertIn("Checkout the PR head locally", str(captured.exception))

    @patch("github_pr.resolve_local_head", return_value="b" * 40)
    def test_explicit_mismatch_override_produces_warning(self, _resolver) -> None:
        warnings = verify_local_pr_head(
            "repo",
            HEAD_SHA,
            allow_mismatched_head=True,
        )
        self.assertIn("explicitly allowed", warnings[0])

    @patch(
        "github_pr.resolve_local_head",
        side_effect=GitDiffError("not_git_worktree", "not git"),
    )
    def test_non_git_local_repository_is_normalized(self, _resolver) -> None:
        with self.assertRaises(GitHubPRError) as captured:
            verify_local_pr_head("repo", HEAD_SHA)
        self.assertEqual(captured.exception.code, "local_repository_invalid")

    @patch("git_diff._run_git")
    def test_local_head_resolution_uses_only_read_only_rev_parse(self, run_git) -> None:
        run_git.side_effect = [
            subprocess.CompletedProcess([], 0, "true\n", ""),
            subprocess.CompletedProcess([], 0, f"{HEAD_SHA}\n", ""),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(resolve_local_head(temp_dir), HEAD_SHA)
        commands = [item.args[1] for item in run_git.call_args_list]
        self.assertEqual([item[0] for item in commands], ["rev-parse", "rev-parse"])
        rendered = " ".join(word for command in commands for word in command)
        for forbidden in ("fetch", "checkout", "clone", "reset", "push"):
            self.assertNotIn(forbidden, rendered)


class GitHubPRReviewIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "service.py").write_text(
            "SERVICE = 2\n",
            encoding="utf-8",
        )
        (self.root / "src" / "repository.py").write_text(
            "REPOSITORY = 2\n",
            encoding="utf-8",
        )

    def test_pr_context_reuses_e1_for_two_python_files_and_readme_warning(self) -> None:
        with (
            patch("agent.fetch_github_pr_url", return_value=pr_context()),
            patch("agent.verify_local_pr_head", return_value=[]),
            patch("agent._run_single_file_review", return_value=good_review()) as single,
            patch("agent.ChatOpenAI") as chat,
        ):
            structured = chat.return_value.with_structured_output.return_value
            structured.invoke.return_value = good_summary()
            result = review_github_pr(str(self.root), PR_URL)
        self.assertIsInstance(result, GitHubPRReview)
        self.assertEqual(result.pull_request.number, 42)
        self.assertEqual(
            [item.target.path for item in result.review.file_results],
            ["src/repository.py", "src/service.py"],
        )
        self.assertEqual(single.call_count, 2)
        self.assertIn(
            "1 changed non-Python files were not reviewed.",
            result.review.warnings,
        )

    def test_pr_metadata_reaches_summary_but_not_file_level_prompt(self) -> None:
        reviewer = Mock()
        reviewer.invoke.side_effect = [good_review(), good_review()]
        summarizer = Mock()
        summarizer.invoke.return_value = good_summary()

        def structured(schema):
            return reviewer if schema is CodeReview else summarizer

        with (
            patch("agent.fetch_github_pr_url", return_value=pr_context()),
            patch("agent.verify_local_pr_head", return_value=[]),
            patch("agent.ChatOpenAI") as chat,
            patch(
                "agent.build_repository_context",
                side_effect=lambda root, target: RepoContext(
                    repository_root=root,
                    target_file=target,
                ),
            ),
            patch(
                "agent.analyze_code",
                return_value=StaticAnalysisResult(tools_run=["ruff", "bandit"]),
            ),
        ):
            chat.return_value.with_structured_output.side_effect = structured
            review_github_pr(str(self.root), PR_URL)
        for item in reviewer.invoke.call_args_list:
            self.assertNotIn("Fix user lookup", item.args[0][1].content)
        summary_prompt = summarizer.invoke.call_args.args[0][1].content
        self.assertIn("Fix user lookup", summary_prompt)
        self.assertIn("feature/user-fix", summary_prompt)
        self.assertNotIn("Bounded body", summary_prompt)

    @patch("agent._run_change_set_review")
    @patch("agent.verify_local_pr_head", return_value=[])
    @patch("agent.fetch_github_pr_url", return_value=pr_context())
    def test_run_tests_false_is_passed_to_e1(self, _fetch, _verify, runner) -> None:
        runner.return_value = github_review().review
        review_github_pr(str(self.root), PR_URL)
        self.assertFalse(runner.call_args.kwargs["run_tests"])

    @patch("agent._run_change_set_review")
    @patch("agent.verify_local_pr_head", return_value=[])
    @patch("agent.fetch_github_pr_url", return_value=pr_context())
    def test_run_tests_true_is_passed_to_e1(self, _fetch, _verify, runner) -> None:
        runner.return_value = github_review().review
        review_github_pr(str(self.root), PR_URL, run_tests=True)
        self.assertTrue(runner.call_args.kwargs["run_tests"])

    @patch("agent._run_change_set_review")
    @patch("agent.verify_local_pr_head", return_value=[])
    @patch("agent.fetch_github_pr_url", return_value=pr_context())
    def test_agentic_explore_is_passed_to_e1(
        self,
        _fetch,
        _verify,
        runner,
    ) -> None:
        runner.return_value = github_review().review
        review_github_pr(str(self.root), PR_URL, agentic_explore=True)
        self.assertTrue(runner.call_args.kwargs["agentic_explore"])

    @patch("agent._run_change_set_review")
    @patch("agent.verify_local_pr_head", return_value=[])
    @patch("agent.fetch_github_pr_url", return_value=pr_context())
    def test_agentic_test_is_passed_to_e1(
        self,
        _fetch,
        _verify,
        runner,
    ) -> None:
        runner.return_value = github_review().review
        review_github_pr(
            str(self.root),
            PR_URL,
            run_tests=True,
            agentic_explore=True,
            agentic_test=True,
        )
        self.assertTrue(runner.call_args.kwargs["agentic_test"])

    @patch("agent.fetch_github_pr_url")
    def test_agentic_test_dependencies_are_checked_before_github_get(
        self,
        fetch,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "agentic_explore"):
            review_github_pr(
                str(self.root),
                PR_URL,
                run_tests=True,
                agentic_test=True,
            )
        with self.assertRaisesRegex(ValueError, "run_tests"):
            review_github_pr(
                str(self.root),
                PR_URL,
                agentic_explore=True,
                agentic_test=True,
            )
        fetch.assert_not_called()

    @patch("agent._run_change_set_review")
    @patch(
        "agent.verify_local_pr_head",
        side_effect=GitHubPRError("head_mismatch", "mismatch"),
    )
    @patch("agent.fetch_github_pr_url", return_value=pr_context())
    def test_head_mismatch_stops_before_e1(self, _fetch, _verify, runner) -> None:
        with self.assertRaises(GitHubPRError):
            review_github_pr(str(self.root), PR_URL)
        runner.assert_not_called()

    def test_pr_api_rejects_auto_fix_and_apply(self) -> None:
        with self.assertRaisesRegex(ValueError, "auto_fix"):
            review_github_pr(str(self.root), PR_URL, auto_fix=True)
        with self.assertRaisesRegex(ValueError, "apply_fix"):
            review_github_pr(str(self.root), PR_URL, apply_fix=True)


class GitHubPRCliTests(unittest.TestCase):
    def _error(self, argv: list[str]) -> str:
        stderr = io.StringIO()
        with (
            patch("sys.argv", argv),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as captured,
        ):
            main()
        self.assertEqual(captured.exception.code, 2)
        return stderr.getvalue()

    @patch("agent.review_github_pr", return_value=github_review())
    def test_review_pr_cli_calls_read_only_api(self, review) -> None:
        stdout = io.StringIO()
        with (
            patch(
                "sys.argv",
                ["agent.py", "--repo", ".", "--review-pr", PR_URL],
            ),
            redirect_stdout(stdout),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(main(), 0)
        review.assert_called_once_with(".", PR_URL)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["pull_request"]["number"], 42)
        self.assertEqual(payload["review"]["overall_rating"], "good")

    @patch("agent.review_github_pr", return_value=github_review())
    def test_review_pr_run_tests_is_explicitly_forwarded(self, review) -> None:
        with (
            patch(
                "sys.argv",
                [
                    "agent.py",
                    "--repo",
                    ".",
                    "--review-pr",
                    PR_URL,
                    "--run-tests",
                ],
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(main(), 0)
        review.assert_called_once_with(".", PR_URL, run_tests=True)

    @patch("agent.review_github_pr", return_value=github_review())
    def test_review_pr_agentic_explore_is_forwarded(self, review) -> None:
        with (
            patch(
                "sys.argv",
                [
                    "agent.py",
                    "--repo",
                    ".",
                    "--review-pr",
                    PR_URL,
                    "--agentic-explore",
                ],
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(main(), 0)
        review.assert_called_once_with(
            ".",
            PR_URL,
            agentic_explore=True,
        )

    @patch("agent.review_github_pr", return_value=github_review())
    def test_review_pr_agentic_test_is_forwarded(self, review) -> None:
        with (
            patch(
                "sys.argv",
                [
                    "agent.py",
                    "--repo",
                    ".",
                    "--review-pr",
                    PR_URL,
                    "--run-tests",
                    "--agentic-explore",
                    "--agentic-test",
                ],
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(main(), 0)
        review.assert_called_once_with(
            ".",
            PR_URL,
            run_tests=True,
            agentic_explore=True,
            agentic_test=True,
        )

    def test_review_pr_requires_repo(self) -> None:
        error = self._error(["agent.py", "--review-pr", PR_URL])
        self.assertIn("--review-pr requires --repo", error)

    def test_review_pr_conflicts_with_file_code_and_change_set_modes(self) -> None:
        conflicts = [
            ["--file", "app.py"],
            ["--code", "VALUE = 1"],
            ["--review-changes"],
        ]
        for conflict in conflicts:
            argv = ["agent.py", "--repo", ".", "--review-pr", PR_URL, *conflict]
            with self.subTest(conflict=conflict):
                self.assertIn("not allowed with argument", self._error(argv))

    def test_review_pr_conflicts_with_all_other_diff_sources(self) -> None:
        sources = [["--diff", "changes.diff"], ["--git-diff"], ["--git-base", "main"]]
        for source in sources:
            argv = ["agent.py", "--repo", ".", "--review-pr", PR_URL, *source]
            with self.subTest(source=source):
                self.assertIn("cannot be combined", self._error(argv))

    def test_review_pr_rejects_auto_fix_and_apply(self) -> None:
        auto_error = self._error(
            ["agent.py", "--repo", ".", "--review-pr", PR_URL, "--auto-fix"]
        )
        self.assertIn("does not support --auto-fix", auto_error)
        apply_error = self._error(
            ["agent.py", "--repo", ".", "--review-pr", PR_URL, "--apply-fix"]
        )
        self.assertIn("does not support --apply-fix", apply_error)

    @patch(
        "agent.review_github_pr",
        side_effect=GitHubPRError("not_found", "PR not found"),
    )
    def test_github_failure_is_machine_readable(self, _review) -> None:
        stdout = io.StringIO()
        with (
            patch(
                "sys.argv",
                ["agent.py", "--repo", ".", "--review-pr", PR_URL],
            ),
            redirect_stdout(stdout),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(main(), 1)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["error"]["type"], "github_pr_failed")
        self.assertEqual(payload["error"]["code"], "not_found")
