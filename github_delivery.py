"""Controlled publication of one exact approved G5A delivery to GitHub.

Stage G5B is a deterministic network/write boundary. It performs no planning,
LLM, tests, candidate correction, working-tree application, local commit
construction, merge, review, or issue operations.
"""

from __future__ import annotations

import json
import os
import re
import subprocess  # nosec B404
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field

from engineering_plan import _validate_repository_root
from git_delivery import (
    DEFAULT_COMMIT_MESSAGE,
    LocalGitDeliverySafetySnapshot,
    LocalGitDeliveryValidationError,
    ValidatedLocalGitDelivery,
    capture_local_git_delivery_safety_snapshot,
    local_git_delivery_safety_errors,
    validate_local_git_delivery,
)
from plan_application import PlanApplicationBundle

GITHUB_API_ROOT = "https://api.github.com"
GITHUB_API_VERSION = "2026-03-10"
GITHUB_USER_AGENT = "02-code-review-agent-stage-g5b"
GITHUB_HTTP_TIMEOUT_SECONDS = 15
GIT_NETWORK_TIMEOUT_SECONDS = 30
MAX_GIT_OUTPUT_CHARS = 20_000
MAX_HTTP_RESPONSE_BYTES = 1_000_000
MAX_REMOTE_NAME_CHARS = 100
MAX_GITHUB_NAME_CHARS = 100
MAX_PR_TITLE_CHARS = 256
MAX_PR_BODY_CHARS = 10_000
MAX_PR_URL_CHARS = 2_000
MAX_HTTP_REQUEST_BYTES = 20_000

_REMOTE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_GITHUB_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_SCP_GITHUB_PATTERN = re.compile(
    r"^git@github\.com:(?P<owner>[^/]+)/(?P<repository>[^/]+)$",
    re.IGNORECASE,
)
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_ALLOWED_GIT_COMMANDS = frozenset(
    {"check-ref-format", "config", "ls-remote", "push"}
)
_DANGEROUS_GIT_ENVIRONMENT = frozenset(
    {
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_ASKPASS",
        "GIT_CEILING_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_SYSTEM",
        "GIT_DIR",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM",
        "GIT_HTTP_EXTRAHEADER",
        "GIT_OBJECT_DIRECTORY",
        "GIT_PROXY_COMMAND",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GIT_WORK_TREE",
        "SSH_ASKPASS",
    }
)


class GitHubRemoteDeliveryResult(BaseModel):
    """Stable, bounded result from the G5B remote publication boundary."""

    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "not_requested",
        "published",
        "partial",
        "stale",
        "conflict",
        "error",
    ]
    remote_name: str | None = None
    owner: str | None = None
    repository: str | None = None
    branch_name: str | None = None
    commit_sha: str | None = None
    base_sha: str | None = None
    base_branch: str | None = None
    approval_digest: str | None = None
    pushed: bool = False
    push_created: bool = False
    pr_created: bool = False
    pr_number: int | None = None
    pr_url: str | None = None
    warnings: list[str] = Field(default_factory=list)
    failure_reason: str | None = None


@dataclass(frozen=True)
class _GitResult:
    returncode: int
    stdout: str
    stderr: str
    output_truncated: bool = False


@dataclass(frozen=True)
class _GitHubRepository:
    owner: str
    repository: str
    push_target: str


@dataclass(frozen=True)
class _PullRequest:
    number: int
    html_url: str
    state: str
    head_ref: str
    head_sha: str
    base_ref: str
    base_sha: str


class _DeliveryFailure(RuntimeError):
    def __init__(
        self,
        status: Literal["stale", "conflict", "error"],
        reason: str,
        *,
        token: str | None = None,
    ) -> None:
        safe_reason = _single_line(reason)
        if token:
            safe_reason = safe_reason.replace(token, "[REDACTED]")
        super().__init__(safe_reason)
        self.status = status
        self.reason = safe_reason


def _single_line(value: object, *, limit: int = MAX_GIT_OUTPUT_CHARS) -> str:
    return " ".join(str(value).replace("\x00", "").split())[:limit]


def _git_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for key in list(environment):
        upper = key.upper()
        if (
            upper in _DANGEROUS_GIT_ENVIRONMENT
            or upper.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))
        ):
            environment.pop(key, None)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    environment["GIT_TERMINAL_PROMPT"] = "0"
    return environment


def _run_git(repository_root: str | Path, args: list[str]) -> _GitResult:
    """Run one fixed-argv, bounded Git command from the G5B allowlist."""

    if not args or args[0] not in _ALLOWED_GIT_COMMANDS:
        raise _DeliveryFailure("error", "Git command is not allowlisted for G5B.")
    try:
        completed = subprocess.run(  # nosec B607
            ["git", *args],
            cwd=str(repository_root),
            env=_git_environment(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=GIT_NETWORK_TIMEOUT_SECONDS,
            shell=False,  # nosec B603
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise _DeliveryFailure("error", "Git network command timed out.") from error
    except OSError as error:
        raise _DeliveryFailure(
            "error",
            "Git executable could not be started.",
        ) from error
    output_length = len(completed.stdout) + len(completed.stderr)
    return _GitResult(
        returncode=completed.returncode,
        stdout=completed.stdout[:MAX_GIT_OUTPUT_CHARS],
        stderr=completed.stderr[:MAX_GIT_OUTPUT_CHARS],
        output_truncated=output_length > MAX_GIT_OUTPUT_CHARS,
    )


def _require_git(result: _GitResult, label: str) -> _GitResult:
    if result.output_truncated:
        raise _DeliveryFailure("error", f"{label} output exceeded the safety limit.")
    if result.returncode != 0:
        raise _DeliveryFailure("error", f"{label} failed.")
    return result


def _validate_remote_name(remote_name: str) -> str:
    if (
        not isinstance(remote_name, str)
        or not remote_name
        or len(remote_name) > MAX_REMOTE_NAME_CHARS
        or _REMOTE_NAME_PATTERN.fullmatch(remote_name) is None
    ):
        raise _DeliveryFailure("error", "Git remote name is invalid.")
    return remote_name


def _validate_github_name(value: str, label: str) -> str:
    if (
        not value
        or len(value) > MAX_GITHUB_NAME_CHARS
        or value in {".", ".."}
        or value.startswith(".")
        or value.endswith(".")
        or _GITHUB_NAME_PATTERN.fullmatch(value) is None
    ):
        raise _DeliveryFailure("error", f"GitHub {label} is invalid.")
    return value


def _remote_config_values(root: Path, key: str) -> list[str]:
    result = _run_git(root, ["config", "--get-all", key])
    if result.output_truncated or result.returncode not in {0, 1}:
        raise _DeliveryFailure("error", "Git remote configuration could not be read.")
    if result.returncode == 1:
        return []
    values = [line.rstrip("\r") for line in result.stdout.splitlines()]
    if any(not value or "\x00" in value for value in values):
        raise _DeliveryFailure("error", "Git remote configuration is invalid.")
    return values


def _reject_matching_url_rewrites(root: Path, push_target: str) -> None:
    """Reject effective Git config that can redirect the validated GitHub URL."""

    result = _run_git(
        root,
        [
            "config",
            "--get-regexp",
            r"^url\..*\.(insteadof|pushinsteadof)$",
        ],
    )
    if result.output_truncated or result.returncode not in {0, 1}:
        raise _DeliveryFailure(
            "error",
            "Git URL rewrite configuration could not be read.",
        )
    if result.returncode == 1:
        return
    for line in result.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not parts[1] or "\x00" in parts[1]:
            raise _DeliveryFailure(
                "error",
                "Git URL rewrite configuration is invalid.",
            )
        if push_target.startswith(parts[1]):
            raise _DeliveryFailure(
                "error",
                "Git URL rewrite configuration cannot redirect the GitHub remote.",
            )


def _parse_github_remote_url(remote_url: str) -> tuple[str, str]:
    """Parse only canonical GitHub HTTPS, SCP-style SSH, or ssh:// remotes."""

    candidate = remote_url
    scp_match = _SCP_GITHUB_PATTERN.fullmatch(candidate)
    if scp_match is not None:
        owner = scp_match.group("owner")
        repository = scp_match.group("repository")
    else:
        try:
            parsed = urlparse(candidate)
            hostname = parsed.hostname
            port = parsed.port
        except ValueError as error:
            raise _DeliveryFailure("error", "GitHub remote URL is malformed.") from error
        is_https = (
            parsed.scheme.casefold() == "https"
            and parsed.username is None
            and parsed.password is None
        )
        is_ssh = (
            parsed.scheme.casefold() == "ssh"
            and parsed.username == "git"
            and parsed.password is None
        )
        if (
            not (is_https or is_ssh)
            or hostname is None
            or hostname.casefold() != "github.com"
            or port is not None
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            raise _DeliveryFailure(
                "error",
                "Remote must be a canonical HTTPS or SSH github.com repository.",
            )
        parts = parsed.path.split("/")
        if len(parts) != 3 or parts[0] or not parts[1] or not parts[2]:
            raise _DeliveryFailure("error", "GitHub remote path is invalid.")
        owner, repository = parts[1], parts[2]

    repository = repository.removesuffix(".git")
    owner = _validate_github_name(owner, "owner")
    repository = _validate_github_name(repository, "repository")
    return owner, repository


def _resolve_github_repository(
    root: Path,
    remote_name: str,
) -> _GitHubRepository:
    remote = _validate_remote_name(remote_name)
    push_urls = _remote_config_values(root, f"remote.{remote}.pushurl")
    if len(push_urls) > 1:
        raise _DeliveryFailure(
            "error",
            "Git remote has multiple push URLs; one exact push target is required.",
        )
    if push_urls:
        push_target = push_urls[0]
    else:
        urls = _remote_config_values(root, f"remote.{remote}.url")
        if not urls:
            raise _DeliveryFailure("error", "Git remote does not exist.")
        if len(urls) > 1:
            raise _DeliveryFailure(
                "error",
                "Git remote has multiple URLs; one exact push target is required.",
            )
        push_target = urls[0]
    owner, repository = _parse_github_remote_url(push_target)
    _reject_matching_url_rewrites(root, push_target)
    return _GitHubRepository(
        owner=owner,
        repository=repository,
        push_target=push_target,
    )


def _validate_branch_name(root: Path, branch_name: str, label: str) -> str:
    if (
        not isinstance(branch_name, str)
        or not branch_name
        or branch_name == "HEAD"
        or branch_name.startswith(("refs/", "-"))
        or branch_name != branch_name.strip()
        or "\x00" in branch_name
    ):
        raise _DeliveryFailure("error", f"{label} is invalid.")
    result = _run_git(root, ["check-ref-format", "--branch", branch_name])
    if result.output_truncated or result.returncode != 0:
        raise _DeliveryFailure("error", f"{label} is invalid.")
    return branch_name


def _github_token() -> str:
    token = os.environ.get("GITHUB_TOKEN")
    normalized = token.strip() if token is not None else ""
    if not normalized:
        raise _DeliveryFailure(
            "error",
            "GITHUB_TOKEN is required before remote delivery.",
        )
    if any(character in normalized for character in ("\x00", "\r", "\n")):
        raise _DeliveryFailure("error", "GITHUB_TOKEN is invalid.")
    return normalized


def _validate_pr_title(value: str | None, delivery: ValidatedLocalGitDelivery) -> str:
    title = value if value is not None else delivery.commit_subject
    title = title or DEFAULT_COMMIT_MESSAGE
    if (
        not isinstance(title, str)
        or not title.strip()
        or len(title) > MAX_PR_TITLE_CHARS
        or any(ord(character) < 32 for character in title)
        or any(ord(character) == 127 for character in title)
    ):
        raise _DeliveryFailure(
            "error",
            f"PR title is invalid or exceeds MAX_PR_TITLE_CHARS={MAX_PR_TITLE_CHARS}.",
        )
    return title


def _validate_pr_body(
    value: str | None,
    delivery: ValidatedLocalGitDelivery,
) -> str:
    if value is None:
        listed_files = delivery.committed_files[:50]
        file_lines = "\n".join(f"- `{path}`" for path in listed_files)
        omitted = len(delivery.committed_files) - len(listed_files)
        if omitted:
            file_lines += f"\n- ... and {omitted} more approved files"
        body = (
            "Automated repository change prepared by RepoGraph.\n\n"
            f"Approval digest: {delivery.approval_digest}\n"
            f"Commit: {delivery.commit_sha}\n\n"
            f"Approved files:\n{file_lines}"
        )
    else:
        body = value
    if (
        not isinstance(body, str)
        or "\x00" in body
        or len(body) > MAX_PR_BODY_CHARS
    ):
        raise _DeliveryFailure(
            "error",
            f"PR body is invalid or exceeds MAX_PR_BODY_CHARS={MAX_PR_BODY_CHARS}.",
        )
    return body


def _api_url(owner: str, repository: str, endpoint: str = "") -> str:
    return (
        f"{GITHUB_API_ROOT}/repos/{quote(owner, safe='')}/"
        f"{quote(repository, safe='')}{endpoint}"
    )


def _http_error(error: HTTPError, token: str) -> _DeliveryFailure:
    if error.code == 401:
        reason = "GitHub authentication preflight was unauthorized."
    elif error.code == 403:
        reason = "GitHub API access was forbidden."
    elif error.code == 404:
        reason = "GitHub repository or API resource was not found."
    elif error.code == 422:
        reason = "GitHub rejected the pull-request request as invalid or conflicting."
    elif error.code >= 500:
        reason = "GitHub returned a server error."
    else:
        reason = f"GitHub returned unexpected HTTP status {error.code}."
    return _DeliveryFailure("error", reason, token=token)


def _request_json(
    method: Literal["GET", "POST"],
    url: str,
    token: str,
    *,
    payload: dict[str, object] | None = None,
) -> object:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "api.github.com"
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise _DeliveryFailure("error", "GitHub API URL escaped the fixed host boundary.")
    body: bytes | None = None
    if method == "POST":
        if payload is None:
            raise _DeliveryFailure("error", "GitHub POST payload is missing.")
        body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(body) > MAX_HTTP_REQUEST_BYTES:
            raise _DeliveryFailure("error", "GitHub request body exceeded its limit.")
    elif payload is not None:
        raise _DeliveryFailure("error", "GitHub GET request cannot have a JSON body.")

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": GITHUB_USER_AGENT,
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method=method)
    try:
        # The URL is assembled only from the fixed HTTPS GitHub API root.
        with urlopen(  # nosec B310
            request,
            timeout=GITHUB_HTTP_TIMEOUT_SECONDS,
        ) as response:
            final_url = response.geturl() if hasattr(response, "geturl") else url
            final_parsed = urlparse(final_url)
            if (
                final_parsed.scheme != "https"
                or final_parsed.hostname != "api.github.com"
                or final_parsed.port is not None
            ):
                raise _DeliveryFailure(
                    "error",
                    "GitHub API redirected outside the fixed host boundary.",
                )
            response_bytes = response.read(MAX_HTTP_RESPONSE_BYTES + 1)
    except _DeliveryFailure:
        raise
    except HTTPError as error:
        raise _http_error(error, token) from None
    except TimeoutError:
        raise _DeliveryFailure(
            "error",
            f"GitHub API timed out after {GITHUB_HTTP_TIMEOUT_SECONDS} seconds.",
            token=token,
        ) from None
    except URLError as error:
        reason = (
            f"GitHub API timed out after {GITHUB_HTTP_TIMEOUT_SECONDS} seconds."
            if isinstance(error.reason, TimeoutError)
            else "GitHub API request failed due to a network error."
        )
        raise _DeliveryFailure("error", reason, token=token) from None
    except OSError:
        raise _DeliveryFailure(
            "error",
            "GitHub API request could not be completed.",
            token=token,
        ) from None
    if len(response_bytes) > MAX_HTTP_RESPONSE_BYTES:
        raise _DeliveryFailure(
            "error",
            "GitHub response exceeded MAX_HTTP_RESPONSE_BYTES="
            f"{MAX_HTTP_RESPONSE_BYTES}.",
        )
    try:
        return json.loads(response_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise _DeliveryFailure(
            "error",
            "GitHub response was not valid bounded UTF-8 JSON.",
        ) from None


def _verify_github_repository(
    identity: _GitHubRepository,
    token: str,
) -> None:
    payload = _request_json(
        "GET",
        _api_url(identity.owner, identity.repository),
        token,
    )
    if not isinstance(payload, dict):
        raise _DeliveryFailure("error", "GitHub repository metadata was malformed.")
    owner_payload = payload.get("owner")
    owner = (
        owner_payload.get("login")
        if isinstance(owner_payload, dict)
        else None
    )
    repository = payload.get("name")
    if (
        not isinstance(owner, str)
        or not isinstance(repository, str)
        or owner.casefold() != identity.owner.casefold()
        or repository.casefold() != identity.repository.casefold()
    ):
        raise _DeliveryFailure(
            "conflict",
            "GitHub repository metadata does not match the validated remote.",
        )


def _canonical_pr_url(
    value: object,
    owner: str,
    repository: str,
    number: int,
) -> str:
    if not isinstance(value, str) or len(value) > MAX_PR_URL_CHARS:
        raise _DeliveryFailure("error", "GitHub pull-request URL is invalid.")
    try:
        parsed = urlparse(value)
        port = parsed.port
    except ValueError as error:
        raise _DeliveryFailure("error", "GitHub pull-request URL is malformed.") from error
    parts = parsed.path.split("/")
    if (
        parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.hostname.casefold() != "github.com"
        or port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.params
        or parsed.query
        or parsed.fragment
        or len(parts) != 5
        or parts[0]
        or parts[1].casefold() != owner.casefold()
        or parts[2].casefold() != repository.casefold()
        or parts[3] != "pull"
        or parts[4] != str(number)
    ):
        raise _DeliveryFailure(
            "error",
            "GitHub pull-request URL is not canonical for the remote repository.",
        )
    return value


def _nested_string(payload: dict[str, object], key: str, nested: str) -> str | None:
    value = payload.get(key)
    if not isinstance(value, dict):
        return None
    nested_value = value.get(nested)
    return nested_value if isinstance(nested_value, str) else None


def _parse_pull_request(
    payload: object,
    *,
    identity: _GitHubRepository,
    delivery: ValidatedLocalGitDelivery,
    base_branch: str,
) -> _PullRequest:
    if not isinstance(payload, dict):
        raise _DeliveryFailure("error", "GitHub pull-request metadata was malformed.")
    number = payload.get("number")
    state = payload.get("state")
    head_ref = _nested_string(payload, "head", "ref")
    head_sha = _nested_string(payload, "head", "sha")
    base_ref = _nested_string(payload, "base", "ref")
    base_sha = _nested_string(payload, "base", "sha")
    if (
        isinstance(number, bool)
        or not isinstance(number, int)
        or number < 1
        or not isinstance(state, str)
        or not isinstance(head_ref, str)
        or not isinstance(head_sha, str)
        or not isinstance(base_ref, str)
        or not isinstance(base_sha, str)
        or _GIT_SHA_PATTERN.fullmatch(head_sha) is None
        or _GIT_SHA_PATTERN.fullmatch(base_sha) is None
    ):
        raise _DeliveryFailure("error", "GitHub pull-request metadata was incomplete.")
    html_url = _canonical_pr_url(
        payload.get("html_url"),
        identity.owner,
        identity.repository,
        number,
    )
    if (
        state != "open"
        or head_ref != delivery.branch_name
        or head_sha.casefold() != delivery.commit_sha.casefold()
        or base_ref != base_branch
        or base_sha.casefold() != delivery.base_sha.casefold()
    ):
        raise _DeliveryFailure(
            "conflict",
            "GitHub pull request does not match the approved head and base.",
        )
    return _PullRequest(
        number=number,
        html_url=html_url,
        state=state,
        head_ref=head_ref,
        head_sha=head_sha.casefold(),
        base_ref=base_ref,
        base_sha=base_sha.casefold(),
    )


def _ls_remote_sha(
    root: Path,
    push_target: str,
    branch_name: str,
) -> str | None:
    full_ref = f"refs/heads/{branch_name}"
    result = _require_git(
        _run_git(root, ["ls-remote", "--heads", push_target, full_ref]),
        "Git remote branch inspection",
    )
    if not result.stdout:
        return None
    lines = [line for line in result.stdout.splitlines() if line]
    if len(lines) != 1 or "\t" not in lines[0]:
        raise _DeliveryFailure("error", "Git remote branch output was ambiguous.")
    sha, returned_ref = lines[0].split("\t", 1)
    if returned_ref != full_ref or _GIT_SHA_PATTERN.fullmatch(sha) is None:
        raise _DeliveryFailure("error", "Git remote branch output was malformed.")
    return sha.casefold()


def _push_create_only(
    root: Path,
    remote_name: str,
    delivery: ValidatedLocalGitDelivery,
) -> None:
    full_ref = f"refs/heads/{delivery.branch_name}"
    result = _run_git(
        root,
        [
            "push",
            "--porcelain",
            f"--force-with-lease={full_ref}:",
            remote_name,
            f"{delivery.commit_sha}:{full_ref}",
        ],
    )
    if result.output_truncated:
        raise _DeliveryFailure("error", "Git push output exceeded the safety limit.")
    if result.returncode != 0:
        raise _DeliveryFailure("error", "Create-only Git push failed.")


def _lookup_pull_request(
    identity: _GitHubRepository,
    delivery: ValidatedLocalGitDelivery,
    base_branch: str,
    token: str,
) -> _PullRequest | None:
    query = urlencode(
        {
            "state": "open",
            "head": f"{identity.owner}:{delivery.branch_name}",
            "base": base_branch,
        }
    )
    payload = _request_json(
        "GET",
        _api_url(identity.owner, identity.repository, f"/pulls?{query}"),
        token,
    )
    if not isinstance(payload, list):
        raise _DeliveryFailure("error", "GitHub pull-request lookup was malformed.")
    if len(payload) > 1:
        raise _DeliveryFailure(
            "conflict",
            "GitHub returned multiple matching open pull requests.",
        )
    if not payload:
        return None
    return _parse_pull_request(
        payload[0],
        identity=identity,
        delivery=delivery,
        base_branch=base_branch,
    )


def _create_pull_request(
    identity: _GitHubRepository,
    delivery: ValidatedLocalGitDelivery,
    base_branch: str,
    title: str,
    body: str,
    token: str,
) -> _PullRequest:
    payload = _request_json(
        "POST",
        _api_url(identity.owner, identity.repository, "/pulls"),
        token,
        payload={
            "title": title,
            "head": f"{identity.owner}:{delivery.branch_name}",
            "base": base_branch,
            "body": body,
            "draft": False,
        },
    )
    return _parse_pull_request(
        payload,
        identity=identity,
        delivery=delivery,
        base_branch=base_branch,
    )


def _base_result_fields(
    *,
    remote_name: str,
    identity: _GitHubRepository | None,
    delivery: ValidatedLocalGitDelivery | None,
    base_branch: str,
    approval_digest: str | None = None,
) -> dict[str, object]:
    return {
        "remote_name": remote_name,
        "owner": identity.owner if identity is not None else None,
        "repository": identity.repository if identity is not None else None,
        "branch_name": delivery.branch_name if delivery is not None else None,
        "commit_sha": delivery.commit_sha if delivery is not None else None,
        "base_sha": delivery.base_sha if delivery is not None else None,
        "base_branch": base_branch,
        "approval_digest": (
            delivery.approval_digest
            if delivery is not None
            else approval_digest
        ),
    }


def publish_local_git_delivery(
    repository_root: str,
    bundle: PlanApplicationBundle,
    *,
    approved: bool = False,
    local_branch: str | None = None,
    remote_name: str = "origin",
    base_branch: str,
    pr_title: str | None = None,
    pr_body: str | None = None,
) -> GitHubRemoteDeliveryResult:
    """Publish one exact validated G5A commit after independent human approval."""

    digest = getattr(bundle, "approval_digest", None)
    if not approved:
        return GitHubRemoteDeliveryResult(
            status="not_requested",
            remote_name=remote_name,
            base_branch=base_branch,
            approval_digest=digest,
        )

    root: Path | None = None
    identity: _GitHubRepository | None = None
    delivery: ValidatedLocalGitDelivery | None = None
    snapshot: LocalGitDeliverySafetySnapshot | None = None
    warnings: list[str] = []
    remote_verified = False
    push_created = False
    pr_post_attempted = False
    result: GitHubRemoteDeliveryResult | None = None

    try:
        root = _validate_repository_root(repository_root)
        remote = _validate_remote_name(remote_name)
        delivery = validate_local_git_delivery(
            str(root),
            bundle,
            branch_name=local_branch,
        )
        digest = delivery.approval_digest
        snapshot = capture_local_git_delivery_safety_snapshot(str(root), delivery)
        base = _validate_branch_name(root, base_branch, "Remote base branch")
        title = _validate_pr_title(pr_title, delivery)
        body = _validate_pr_body(pr_body, delivery)
        identity = _resolve_github_repository(root, remote)

        # Credential and authenticated repository validation happen before push.
        token = _github_token()
        _verify_github_repository(identity, token)

        remote_base_sha = _ls_remote_sha(root, identity.push_target, base)
        if remote_base_sha is None:
            raise _DeliveryFailure("conflict", "Remote base branch does not exist.")
        if remote_base_sha != delivery.base_sha:
            raise _DeliveryFailure(
                "stale",
                "Remote base branch moved after the approved local delivery.",
            )

        remote_branch_sha = _ls_remote_sha(
            root,
            identity.push_target,
            delivery.branch_name,
        )
        if (
            remote_branch_sha is not None
            and remote_branch_sha != delivery.commit_sha
        ):
            raise _DeliveryFailure(
                "conflict",
                "Remote delivery branch already points to a different commit.",
            )

        if remote_branch_sha is None:
            latest_delivery = validate_local_git_delivery(
                str(root),
                bundle,
                branch_name=delivery.branch_name,
            )
            if latest_delivery != delivery:
                raise _DeliveryFailure(
                    "stale",
                    "Local delivery changed during remote preflight.",
                )
            pre_push_errors = local_git_delivery_safety_errors(snapshot)
            if pre_push_errors:
                raise _DeliveryFailure("stale", "; ".join(pre_push_errors))
            try:
                _push_create_only(root, remote, delivery)
                push_created = True
            except _DeliveryFailure:
                observed_sha = _ls_remote_sha(
                    root,
                    identity.push_target,
                    delivery.branch_name,
                )
                if observed_sha == delivery.commit_sha:
                    warnings.append(
                        "The create-only push reported failure, but the remote "
                        "branch was subsequently verified at the approved commit."
                    )
                elif observed_sha is not None:
                    raise _DeliveryFailure(
                        "conflict",
                        "Remote delivery branch appeared at a different commit "
                        "during create-only push.",
                    ) from None
                else:
                    raise
        else:
            warnings.append(
                "Remote delivery branch already existed at the approved commit; "
                "push was skipped."
            )

        verified_sha = _ls_remote_sha(
            root,
            identity.push_target,
            delivery.branch_name,
        )
        if verified_sha != delivery.commit_sha:
            raise _DeliveryFailure(
                "error",
                "Remote branch did not verify at the approved commit after push.",
            )
        remote_verified = True

        existing_pr = _lookup_pull_request(identity, delivery, base, token)
        if existing_pr is not None:
            result = GitHubRemoteDeliveryResult(
                status="published",
                **_base_result_fields(
                    remote_name=remote,
                    identity=identity,
                    delivery=delivery,
                    base_branch=base,
                ),
                pushed=True,
                push_created=push_created,
                pr_created=False,
                pr_number=existing_pr.number,
                pr_url=existing_pr.html_url,
                warnings=warnings,
            )
        else:
            pr_post_attempted = True
            created_pr = _create_pull_request(
                identity,
                delivery,
                base,
                title,
                body,
                token,
            )
            result = GitHubRemoteDeliveryResult(
                status="published",
                **_base_result_fields(
                    remote_name=remote,
                    identity=identity,
                    delivery=delivery,
                    base_branch=base,
                ),
                pushed=True,
                push_created=push_created,
                pr_created=True,
                pr_number=created_pr.number,
                pr_url=created_pr.html_url,
                warnings=warnings,
            )
    except LocalGitDeliveryValidationError as error:
        result = GitHubRemoteDeliveryResult(
            status=error.status,
            **_base_result_fields(
                remote_name=remote_name,
                identity=identity,
                delivery=delivery,
                base_branch=base_branch,
                approval_digest=digest,
            ),
            pushed=remote_verified,
            push_created=push_created,
            warnings=warnings,
            failure_reason=error.reason,
        )
    except _DeliveryFailure as error:
        if pr_post_attempted or (
            (remote_verified or push_created) and error.status == "error"
        ):
            status: Literal["partial", "stale", "conflict", "error"] = "partial"
        else:
            status = error.status
        result = GitHubRemoteDeliveryResult(
            status=status,
            **_base_result_fields(
                remote_name=remote_name,
                identity=identity,
                delivery=delivery,
                base_branch=base_branch,
                approval_digest=digest,
            ),
            pushed=remote_verified,
            push_created=push_created,
            warnings=warnings,
            failure_reason=error.reason,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        status = "partial" if remote_verified or push_created else "error"
        result = GitHubRemoteDeliveryResult(
            status=status,
            **_base_result_fields(
                remote_name=remote_name,
                identity=identity,
                delivery=delivery,
                base_branch=base_branch,
                approval_digest=digest,
            ),
            pushed=remote_verified,
            push_created=push_created,
            warnings=warnings,
            failure_reason="GitHub remote delivery failed safely.",
        )

    if snapshot is not None:
        preservation_errors = local_git_delivery_safety_errors(snapshot)
        if preservation_errors:
            for preservation_error in preservation_errors:
                if preservation_error not in result.warnings:
                    result.warnings.append(preservation_error)
            if result.status == "published":
                result.status = "partial" if remote_verified else "error"
            if result.failure_reason is None:
                result.failure_reason = (
                    "Local repository preservation verification failed."
                )
    return result


def render_github_remote_delivery_result(
    result: GitHubRemoteDeliveryResult,
) -> str:
    """Render one concise, deterministic, redacted remote-delivery result."""

    lines = [f"Status: {result.status}"]
    if result.owner is not None and result.repository is not None:
        lines.append(f"Repository: {result.owner}/{result.repository}")
    if result.remote_name is not None:
        lines.append(f"Remote: {result.remote_name}")
    if result.base_branch is not None:
        lines.append(f"Base branch: {result.base_branch}")
    if result.base_sha is not None:
        lines.append(f"Base commit: {result.base_sha}")
    if result.branch_name is not None:
        lines.append(f"Branch: {result.branch_name}")
    if result.commit_sha is not None:
        lines.append(f"Commit: {result.commit_sha}")
    if result.approval_digest is not None:
        lines.append(f"Approval digest: {result.approval_digest}")
    lines.append(f"Remote branch verified: {result.pushed}")
    lines.append(f"Remote branch created this run: {result.push_created}")
    lines.append(f"PR created this run: {result.pr_created}")
    if result.pr_number is not None:
        lines.append(f"PR number: {result.pr_number}")
    if result.pr_url is not None:
        lines.append(f"PR URL: {result.pr_url}")
    if result.failure_reason is not None:
        lines.append(f"Failure: {result.failure_reason}")
    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in result.warnings)
    return "\n".join(lines)
