"""Read-only, bounded GitHub pull-request input adapter for Stage E2."""

import json
import os
import re
from email.message import Message
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field

from change_set import MAX_CHANGESET_DIFF_CHARS, ChangeSetReview
from git_diff import GitDiffError, resolve_local_head

GITHUB_API_ROOT = "https://api.github.com"
GITHUB_API_VERSION = "2026-03-10"
GITHUB_USER_AGENT = "02-code-review-agent-stage-e2"
GITHUB_HTTP_TIMEOUT_SECONDS = 15
MAX_PR_BODY_CHARS = 10_000
MAX_PR_TITLE_CHARS = 500
MAX_PR_REF_CHARS = 500
MAX_PR_AUTHOR_CHARS = 200
MAX_PR_URL_CHARS = 2_000
MAX_PR_METADATA_BYTES = 1_000_000
MAX_PR_DIFF_BYTES = MAX_CHANGESET_DIFF_CHARS * 4

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")


class GitHubPRReference(BaseModel):
    """Validated owner, repository, and pull-request number."""

    model_config = ConfigDict(extra="forbid")

    owner: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    number: int = Field(ge=1)


class GitHubPRMetadata(BaseModel):
    """Bounded pull-request metadata safe to return and summarize."""

    model_config = ConfigDict(extra="forbid")

    owner: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    number: int = Field(ge=1)
    title: str
    body: str = ""
    state: str | None = None
    head_ref: str | None = None
    base_ref: str | None = None
    head_sha: str = Field(pattern=_GIT_SHA_PATTERN.pattern)
    author: str | None = None
    changed_files: int | None = Field(default=None, ge=0)
    html_url: str | None = None


class GitHubPRContext(GitHubPRMetadata):
    """Bounded GitHub metadata plus raw unified diff for E1 ingestion."""

    model_config = ConfigDict(extra="forbid")

    diff_text: str
    warnings: list[str] = Field(default_factory=list)

    def metadata(self) -> GitHubPRMetadata:
        """Return the GitHub-neutral review wrapper metadata without raw diff."""

        return GitHubPRMetadata.model_validate(
            self.model_dump(exclude={"diff_text", "warnings"})
        )


class GitHubPRReview(BaseModel):
    """Read-only pull-request metadata composed with a generic change-set review."""

    model_config = ConfigDict(extra="forbid")

    pull_request: GitHubPRMetadata
    review: ChangeSetReview


class GitHubPRError(RuntimeError):
    """Stable, redacted failure from PR parsing, HTTP, or local compatibility."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        secret: str | None = None,
    ) -> None:
        safe_message = message.replace(secret, "[REDACTED]") if secret else message
        super().__init__(safe_message)
        self.code = code

    def to_payload(self) -> dict[str, object]:
        return {
            "status": "failed",
            "error": {
                "type": "github_pr_failed",
                "code": self.code,
                "message": str(self),
            },
        }


def _validate_coordinates(
    owner: str,
    repository: str,
    number: int,
) -> GitHubPRReference:
    if (
        not owner
        or owner in {".", ".."}
        or _NAME_PATTERN.fullmatch(owner) is None
        or not repository
        or repository in {".", ".."}
        or _NAME_PATTERN.fullmatch(repository) is None
        or isinstance(number, bool)
        or not isinstance(number, int)
        or number < 1
    ):
        raise GitHubPRError(
            "invalid_pr_reference",
            "GitHub pull-request owner, repository, or number is invalid.",
        )
    return GitHubPRReference(
        owner=owner,
        repository=repository,
        number=number,
    )


def parse_github_pr_url(pr_url: str) -> GitHubPRReference:
    """Parse only canonical HTTPS github.com OWNER/REPO/pull/NUMBER URLs."""

    try:
        parsed = urlparse(pr_url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise GitHubPRError(
            "invalid_pr_reference",
            "GitHub pull-request URL is malformed.",
        ) from error
    if (
        parsed.scheme.casefold() != "https"
        or hostname is None
        or hostname.casefold() != "github.com"
        or port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise GitHubPRError(
            "invalid_pr_reference",
            "Expected an HTTPS github.com pull-request URL.",
        )

    parts = parsed.path.split("/")
    if len(parts) != 5 or parts[0] or parts[3] != "pull":
        raise GitHubPRError(
            "invalid_pr_reference",
            "Expected GitHub pull-request path /OWNER/REPO/pull/NUMBER.",
        )
    try:
        number = int(parts[4])
    except ValueError as error:
        raise GitHubPRError(
            "invalid_pr_reference",
            "GitHub pull-request number must be a positive integer.",
        ) from error
    return _validate_coordinates(parts[1], parts[2], number)


def _github_token() -> str | None:
    token = os.environ.get("GITHUB_TOKEN")
    if token is None:
        return None
    normalized = token.strip()
    return normalized or None


def _build_request(
    url: str,
    accept: str,
    token: str | None,
) -> Request:
    headers = {
        "Accept": accept,
        "User-Agent": GITHUB_USER_AGENT,
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return Request(url, headers=headers, method="GET")


def _rate_limit_exhausted(headers: Message | None) -> bool:
    return headers is not None and headers.get("X-RateLimit-Remaining") == "0"


def _http_error(error: HTTPError, token: str | None) -> GitHubPRError:
    if error.code == 404:
        code = "not_found"
        message = "GitHub pull request was not found."
    elif error.code == 401:
        code = "unauthorized"
        message = "GitHub rejected the supplied credentials."
    elif error.code == 429 or (
        error.code == 403 and _rate_limit_exhausted(error.headers)
    ):
        code = "rate_limited"
        message = "GitHub API rate limit was exceeded."
    elif error.code == 403:
        code = "forbidden"
        message = "GitHub denied access to the pull request."
    elif error.code >= 500:
        code = "network_error"
        message = "GitHub returned a server error."
    else:
        code = "invalid_response"
        message = f"GitHub returned unexpected HTTP status {error.code}."
    return GitHubPRError(code, message, secret=token)


def _read_response(
    request: Request,
    max_bytes: int,
    token: str | None,
) -> tuple[bytes, bool]:
    try:
        # The request URL is assembled from the fixed HTTPS GitHub API root.
        with urlopen(  # nosec B310
            request,
            timeout=GITHUB_HTTP_TIMEOUT_SECONDS,
        ) as response:
            data = response.read(max_bytes + 1)
    except HTTPError as error:
        raise _http_error(error, token) from None
    except TimeoutError:
        raise GitHubPRError(
            "timeout",
            f"GitHub request timed out after {GITHUB_HTTP_TIMEOUT_SECONDS} seconds.",
            secret=token,
        ) from None
    except URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise GitHubPRError(
                "timeout",
                (
                    "GitHub request timed out after "
                    f"{GITHUB_HTTP_TIMEOUT_SECONDS} seconds."
                ),
                secret=token,
            ) from None
        raise GitHubPRError(
            "network_error",
            "GitHub request failed due to a network error.",
            secret=token,
        ) from None
    except OSError:
        raise GitHubPRError(
            "network_error",
            "GitHub request could not be completed.",
            secret=token,
        ) from None
    return data[:max_bytes], len(data) > max_bytes


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _redact(value: str | None, token: str | None) -> str | None:
    if value is None or token is None:
        return value
    return value.replace(token, "[REDACTED]")


def _bounded_optional_string(
    value: object,
    *,
    label: str,
    limit: int,
    warnings: list[str],
    token: str | None,
) -> str | None:
    rendered = _redact(_optional_string(value), token)
    if rendered is None or len(rendered) <= limit:
        return rendered
    warnings.append(f"{label} truncated at {limit} characters.")
    return rendered[:limit]


def _nested_string(payload: dict[str, object], key: str, nested: str) -> str | None:
    container = payload.get(key)
    if not isinstance(container, dict):
        return None
    return _optional_string(container.get(nested))


def _metadata_fields(
    payload: object,
    reference: GitHubPRReference,
    warnings: list[str],
    token: str | None,
) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise GitHubPRError(
            "invalid_response",
            "GitHub pull-request metadata must be a JSON object.",
            secret=token,
        )

    number = payload.get("number")
    title = payload.get("title")
    head_sha = _nested_string(payload, "head", "sha")
    if (
        number != reference.number
        or not isinstance(title, str)
        or head_sha is None
        or _GIT_SHA_PATTERN.fullmatch(head_sha) is None
    ):
        raise GitHubPRError(
            "invalid_response",
            "GitHub pull-request metadata is missing required validated fields.",
            secret=token,
        )

    title = _redact(title, token) or ""
    body_value = payload.get("body")
    raw_body = body_value if isinstance(body_value, str) else ""
    body = _redact(raw_body, token) or ""
    if len(body) > MAX_PR_BODY_CHARS:
        body = body[:MAX_PR_BODY_CHARS]
        warnings.append(f"PR body truncated at MAX_PR_BODY_CHARS={MAX_PR_BODY_CHARS}.")
    if len(title) > MAX_PR_TITLE_CHARS:
        title = title[:MAX_PR_TITLE_CHARS]
        warnings.append(
            f"PR title truncated at MAX_PR_TITLE_CHARS={MAX_PR_TITLE_CHARS}."
        )

    changed_files = payload.get("changed_files")
    if (
        isinstance(changed_files, bool)
        or not isinstance(changed_files, int)
        or changed_files < 0
    ):
        changed_files = None
    author = _bounded_optional_string(
        _nested_string(payload, "user", "login"),
        label="PR author",
        limit=MAX_PR_AUTHOR_CHARS,
        warnings=warnings,
        token=token,
    )
    return {
        "owner": reference.owner,
        "repository": reference.repository,
        "number": reference.number,
        "title": title,
        "body": body,
        "state": _bounded_optional_string(
            payload.get("state"),
            label="PR state",
            limit=50,
            warnings=warnings,
            token=token,
        ),
        "head_ref": _bounded_optional_string(
            _nested_string(payload, "head", "ref"),
            label="PR head ref",
            limit=MAX_PR_REF_CHARS,
            warnings=warnings,
            token=token,
        ),
        "base_ref": _bounded_optional_string(
            _nested_string(payload, "base", "ref"),
            label="PR base ref",
            limit=MAX_PR_REF_CHARS,
            warnings=warnings,
            token=token,
        ),
        "head_sha": head_sha.lower(),
        "author": author,
        "changed_files": changed_files,
        "html_url": _bounded_optional_string(
            payload.get("html_url"),
            label="PR HTML URL",
            limit=MAX_PR_URL_CHARS,
            warnings=warnings,
            token=token,
        ),
    }


def fetch_github_pr(
    owner: str,
    repository: str,
    number: int,
) -> GitHubPRContext:
    """Retrieve bounded PR metadata and unified diff with GET requests only."""

    reference = _validate_coordinates(owner, repository, number)
    token = _github_token()
    api_url = (
        f"{GITHUB_API_ROOT}/repos/{quote(reference.owner, safe='')}/"
        f"{quote(reference.repository, safe='')}/pulls/{reference.number}"
    )
    metadata_request = _build_request(
        api_url,
        "application/vnd.github+json",
        token,
    )
    metadata_bytes, metadata_truncated = _read_response(
        metadata_request,
        MAX_PR_METADATA_BYTES,
        token,
    )
    if metadata_truncated:
        raise GitHubPRError(
            "invalid_response",
            f"GitHub metadata exceeded MAX_PR_METADATA_BYTES={MAX_PR_METADATA_BYTES}.",
            secret=token,
        )
    try:
        metadata_payload = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise GitHubPRError(
            "invalid_response",
            "GitHub pull-request metadata was not valid UTF-8 JSON.",
            secret=token,
        ) from None

    warnings: list[str] = []
    fields = _metadata_fields(metadata_payload, reference, warnings, token)
    diff_request = _build_request(
        api_url,
        "application/vnd.github.diff",
        token,
    )
    diff_bytes, diff_byte_truncated = _read_response(
        diff_request,
        MAX_PR_DIFF_BYTES,
        token,
    )
    try:
        diff_text = diff_bytes.decode("utf-8")
    except UnicodeError:
        raise GitHubPRError(
            "invalid_response",
            "GitHub pull-request diff was not valid UTF-8.",
            secret=token,
        ) from None
    diff_char_truncated = len(diff_text) > MAX_CHANGESET_DIFF_CHARS
    diff_text = _redact(diff_text, token) or ""
    if (
        diff_byte_truncated
        or diff_char_truncated
        or len(diff_text) > MAX_CHANGESET_DIFF_CHARS
    ):
        diff_text = diff_text[:MAX_CHANGESET_DIFF_CHARS]
        warnings.append(
            "GitHub PR diff truncated at MAX_CHANGESET_DIFF_CHARS="
            f"{MAX_CHANGESET_DIFF_CHARS}; the pull request was not fully reviewed."
        )
    return GitHubPRContext(
        **fields,
        diff_text=diff_text,
        warnings=warnings,
    )


def fetch_github_pr_url(pr_url: str) -> GitHubPRContext:
    """Parse a canonical pull-request URL and fetch it through the same API."""

    reference = parse_github_pr_url(pr_url)
    return fetch_github_pr(
        reference.owner,
        reference.repository,
        reference.number,
    )


def verify_local_pr_head(
    repository_root: str,
    expected_head_sha: str,
    *,
    allow_mismatched_head: bool = False,
) -> list[str]:
    """Require local HEAD to match PR head unless explicitly overridden."""

    try:
        local_head = resolve_local_head(repository_root)
    except GitDiffError as error:
        raise GitHubPRError(
            "local_repository_invalid",
            "Local repository HEAD could not be resolved from a Git worktree.",
        ) from error
    if local_head.casefold() == expected_head_sha.casefold():
        return []
    warning = (
        "Local repository HEAD does not match the pull request head commit; "
        "file review used explicitly allowed mismatched local source."
    )
    if allow_mismatched_head:
        return [warning]
    raise GitHubPRError(
        "head_mismatch",
        (
            "Local repository HEAD does not match the pull request head commit. "
            "Checkout the PR head locally before reviewing."
        ),
    )
