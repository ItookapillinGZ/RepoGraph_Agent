"""Controlled local Git delivery for an approved, applied G4 bundle.

This module is a deterministic Git transaction boundary. It performs no
planning, LLM, test, shell, network, GitHub, checkout, merge, reset, stash, or
working-tree write operations.
"""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess  # nosec B404
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from engineering_plan import _relative_parts, _validate_repository_root
from plan_application import (
    PlanApplicationBundle,
    PlanApplicationError,
    candidate_bytes_for_plan_change,
    validate_plan_application_bundle,
    validate_plan_application_diff_from_original_bytes,
)

DEFAULT_BRANCH_PREFIX = "repograph/"
APPROVAL_DIGEST_BRANCH_CHARS = 12
DEFAULT_COMMIT_MESSAGE = "RepoGraph: apply approved repository plan"
MAX_COMMIT_MESSAGE_CHARS = 10_000
MAX_GIT_OUTPUT_CHARS = 20_000
MAX_GIT_BLOB_BYTES = 500_000
MAX_GIT_COMMIT_BYTES = 50_000
GIT_TIMEOUT_SECONDS = 30

_ALLOWED_GIT_COMMANDS = frozenset(
    {
        "cat-file",
        "check-ref-format",
        "commit-tree",
        "config",
        "diff",
        "diff-tree",
        "hash-object",
        "ls-tree",
        "read-tree",
        "rev-parse",
        "show-ref",
        "status",
        "symbolic-ref",
        "update-index",
        "update-ref",
        "write-tree",
    }
)
_DANGEROUS_GIT_ENVIRONMENT = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_ASKPASS",
        "GIT_AUTHOR_DATE",
        "GIT_AUTHOR_EMAIL",
        "GIT_AUTHOR_NAME",
        "GIT_CEILING_DIRECTORIES",
        "GIT_COMMITTER_DATE",
        "GIT_COMMITTER_EMAIL",
        "GIT_COMMITTER_NAME",
        "GIT_COMMON_DIR",
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_SYSTEM",
        "GIT_DIR",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM",
        "GIT_EXTERNAL_DIFF",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_OBJECT_DIRECTORY_RELATIVE",
        "GIT_PROXY_COMMAND",
        "GIT_QUARANTINE_PATH",
        "GIT_REPLACE_REF_BASE",
        "GIT_SHALLOW_FILE",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GIT_NAMESPACE",
        "GIT_WORK_TREE",
        "SSH_ASKPASS",
    }
)


class LocalGitDeliveryResult(BaseModel):
    """Stable public result from the local Git delivery boundary."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["not_requested", "created", "stale", "conflict", "error"]
    branch_name: str | None = None
    commit_sha: str | None = None
    base_sha: str | None = None
    approval_digest: str | None = None
    committed_files: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    failure_reason: str | None = None


class ValidatedLocalGitDelivery(BaseModel):
    """Read-only proof that one existing G5A delivery matches its bundle."""

    model_config = ConfigDict(extra="forbid")

    branch_name: str
    commit_sha: str
    base_sha: str
    approval_digest: str
    committed_files: list[str] = Field(default_factory=list)
    commit_subject: str | None = None


class LocalGitDeliveryValidationError(RuntimeError):
    """Stable failure raised while revalidating an existing G5A delivery."""

    def __init__(self, status: Literal["stale", "error"], reason: str) -> None:
        super().__init__(_single_line(reason))
        self.status = status
        self.reason = _single_line(reason)


@dataclass(frozen=True)
class _GitResult:
    returncode: int
    stdout: str | bytes
    stderr: str
    output_truncated: bool = False


@dataclass(frozen=True)
class _PathSnapshot:
    exists: bool
    content_hash: str | None
    mode: int | None


@dataclass(frozen=True)
class _IndexSnapshot:
    exists: bool
    content_hash: str | None


@dataclass(frozen=True)
class _RepositorySnapshot:
    base_sha: str
    symbolic_head: str | None
    status_output: str
    index_path: Path
    index: _IndexSnapshot


@dataclass(frozen=True)
class LocalGitDeliverySafetySnapshot:
    """Opaque local-state snapshot used to prove remote delivery is non-mutating."""

    repository_root: Path
    repository: _RepositorySnapshot
    planned_paths: dict[str, _PathSnapshot]
    branch_name: str
    branch_sha: str


@dataclass(frozen=True)
class _TreeEntry:
    mode: str
    object_sha: str


class _DeliveryFailure(RuntimeError):
    def __init__(
        self,
        status: Literal["stale", "conflict", "error"],
        reason: str,
    ) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


def _content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _single_line(value: object, *, limit: int = MAX_GIT_OUTPUT_CHARS) -> str:
    text = " ".join(str(value).replace("\x00", "").split())
    return text[:limit]


def _git_environment(overrides: dict[str, str] | None = None) -> dict[str, str]:
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
    if overrides:
        environment.update(overrides)
    return environment


def _run_git(
    repository_root: str | Path,
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
    input_bytes: bytes | None = None,
    binary_output: bool = False,
    output_limit: int = MAX_GIT_OUTPUT_CHARS,
) -> _GitResult:
    """Run one fixed-argv, allowlisted local Git plumbing command."""

    if not args or args[0] not in _ALLOWED_GIT_COMMANDS:
        raise _DeliveryFailure("error", "Git command is not allowlisted.")
    if input_text is not None and input_bytes is not None:
        raise _DeliveryFailure("error", "Git input mode is ambiguous.")

    binary_mode = input_bytes is not None or binary_output
    try:
        completed = subprocess.run(  # nosec B607
            ["git", *args],
            cwd=str(repository_root),
            env=_git_environment(env),
            input=input_bytes if input_bytes is not None else input_text,
            capture_output=True,
            text=not binary_mode,
            encoding=None if binary_mode else "utf-8",
            errors=None if binary_mode else "replace",
            timeout=GIT_TIMEOUT_SECONDS,
            shell=False,  # nosec B603
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise _DeliveryFailure("error", "Git command timed out.") from error
    except OSError as error:
        raise _DeliveryFailure(
            "error",
            "Git executable could not be started.",
        ) from error

    stdout: str | bytes = completed.stdout
    stderr_value = completed.stderr
    if isinstance(stderr_value, bytes):
        stderr = stderr_value.decode("utf-8", errors="replace")
    else:
        stderr = stderr_value
    output_length = len(stdout) + len(stderr)
    truncated = output_length > output_limit
    bounded_stdout: str | bytes = stdout[:output_limit]
    return _GitResult(
        returncode=completed.returncode,
        stdout=bounded_stdout,
        stderr=stderr[:output_limit],
        output_truncated=truncated,
    )


def _require_git(
    result: _GitResult,
    label: str,
    *,
    expected_returncodes: tuple[int, ...] = (0,),
) -> _GitResult:
    if result.output_truncated:
        raise _DeliveryFailure("error", f"{label} output exceeded the safety limit.")
    if result.returncode not in expected_returncodes:
        raise _DeliveryFailure("error", f"{label} failed.")
    return result


def _text_stdout(result: _GitResult) -> str:
    if not isinstance(result.stdout, str):
        raise _DeliveryFailure("error", "Git returned an unexpected binary result.")
    return result.stdout


def _bytes_stdout(result: _GitResult) -> bytes:
    if not isinstance(result.stdout, bytes):
        raise _DeliveryFailure("error", "Git returned an unexpected text result.")
    return result.stdout


def _normalized_path(path: str) -> str:
    return "/".join(_relative_parts(path))


def _index_snapshot(path: Path) -> _IndexSnapshot:
    if not path.exists():
        return _IndexSnapshot(exists=False, content_hash=None)
    if not path.is_file():
        raise _DeliveryFailure("error", "The real Git index is not a regular file.")
    try:
        return _IndexSnapshot(
            exists=True,
            content_hash=_content_hash(path.read_bytes()),
        )
    except OSError as error:
        raise _DeliveryFailure("error", "The real Git index could not be read.") from error


def _capture_repository_snapshot(root: Path) -> _RepositorySnapshot:
    top_result = _require_git(
        _run_git(root, ["rev-parse", "--show-toplevel"]),
        "Git repository root validation",
    )
    top_text = _text_stdout(top_result).strip()
    try:
        top = Path(top_text).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise _DeliveryFailure(
            "error",
            "Git repository root could not be resolved.",
        ) from error
    if top != root:
        raise _DeliveryFailure(
            "error",
            "Supplied repository root is not the Git worktree root.",
        )

    head_result = _run_git(root, ["rev-parse", "--verify", "HEAD^{commit}"])
    if head_result.returncode != 0 or head_result.output_truncated:
        raise _DeliveryFailure(
            "error",
            "Repository must have an existing HEAD commit.",
        )
    base_sha = _text_stdout(head_result).strip()
    if not base_sha or any(
        character not in "0123456789abcdef" for character in base_sha
    ):
        raise _DeliveryFailure("error", "Git returned an invalid HEAD object id.")

    symbolic_result = _run_git(root, ["symbolic-ref", "--quiet", "HEAD"])
    if symbolic_result.output_truncated or symbolic_result.returncode not in {0, 1}:
        raise _DeliveryFailure("error", "Current Git HEAD could not be inspected.")
    symbolic_head = (
        _text_stdout(symbolic_result).strip()
        if symbolic_result.returncode == 0
        else None
    )

    status_result = _require_git(
        _run_git(root, ["status", "--porcelain=v1", "--untracked-files=all"]),
        "Git working-tree fingerprint",
    )
    index_result = _require_git(
        _run_git(root, ["rev-parse", "--git-path", "index"]),
        "Git index location",
    )
    index_value = _text_stdout(index_result).strip()
    if not index_value or "\x00" in index_value:
        raise _DeliveryFailure("error", "Git returned an invalid index location.")
    index_path = Path(index_value)
    if not index_path.is_absolute():
        index_path = root / index_path
    index_path = index_path.resolve(strict=False)
    return _RepositorySnapshot(
        base_sha=base_sha,
        symbolic_head=symbolic_head,
        status_output=_text_stdout(status_result),
        index_path=index_path,
        index=_index_snapshot(index_path),
    )


def _capture_path_snapshot(root: Path, relative_path: str) -> _PathSnapshot:
    target = root.joinpath(*_relative_parts(relative_path))
    if target.is_symlink():
        raise _DeliveryFailure(
            "error",
            f"Planned path is a symbolic link: {relative_path}.",
        )
    if not target.exists():
        return _PathSnapshot(exists=False, content_hash=None, mode=None)
    if not target.is_file():
        raise _DeliveryFailure(
            "error",
            f"Planned path is not a regular file: {relative_path}.",
        )
    try:
        return _PathSnapshot(
            exists=True,
            content_hash=_content_hash(target.read_bytes()),
            mode=stat.S_IMODE(target.stat().st_mode),
        )
    except OSError as error:
        raise _DeliveryFailure(
            "error",
            f"Planned path could not be inspected: {relative_path}.",
        ) from error


def _verify_applied_candidate(
    root: Path,
    bundle: PlanApplicationBundle,
) -> tuple[dict[str, bytes | None], dict[str, _PathSnapshot]]:
    current_bytes: dict[str, bytes | None] = {}
    snapshots: dict[str, _PathSnapshot] = {}
    for planned in sorted(
        bundle.plan.files,
        key=lambda item: _normalized_path(item.path),
    ):
        path = _normalized_path(planned.path)
        snapshot = _capture_path_snapshot(root, path)
        snapshots[path] = snapshot
        target = root.joinpath(*_relative_parts(path))
        if planned.action == "delete":
            if snapshot.exists:
                raise _DeliveryFailure(
                    "stale",
                    f"Approved deletion has not been applied: {path}.",
                )
            current_bytes[path] = None
            continue
        if not snapshot.exists:
            raise _DeliveryFailure(
                "stale",
                f"Applied approved path is missing: {path}.",
            )
        try:
            content = target.read_bytes()
        except OSError as error:
            raise _DeliveryFailure(
                "error",
                f"Applied approved path could not be read: {path}.",
            ) from error
        if _content_hash(content) != bundle.expected_candidate_hashes[path]:
            raise _DeliveryFailure(
                "stale",
                f"Applied bytes no longer match the approved candidate: {path}.",
            )
        current_bytes[path] = content
    return current_bytes, snapshots


def _ls_tree_entry(root: Path, revision: str, path: str) -> _TreeEntry | None:
    result = _require_git(
        _run_git(
            root,
            ["ls-tree", "-z", revision, "--", path],
            binary_output=True,
        ),
        "Git tree inspection",
    )
    payload = _bytes_stdout(result)
    if not payload:
        return None
    records = [record for record in payload.split(b"\x00") if record]
    if len(records) != 1 or b"\t" not in records[0]:
        raise _DeliveryFailure("error", "Git tree inspection was ambiguous.")
    metadata, _ = records[0].split(b"\t", 1)
    fields = metadata.split()
    if len(fields) != 3 or fields[1] != b"blob":
        raise _DeliveryFailure(
            "conflict",
            "Planned HEAD path is not a regular blob.",
        )
    try:
        mode = fields[0].decode("ascii")
        object_sha = fields[2].decode("ascii")
    except UnicodeDecodeError as error:
        raise _DeliveryFailure("error", "Git returned invalid tree metadata.") from error
    if mode not in {"100644", "100755"}:
        raise _DeliveryFailure(
            "conflict",
            "Planned HEAD path has an unsupported mode.",
        )
    return _TreeEntry(mode=mode, object_sha=object_sha)


def _cat_blob(root: Path, object_spec: str) -> bytes:
    result = _require_git(
        _run_git(
            root,
            ["cat-file", "blob", object_spec],
            binary_output=True,
            output_limit=MAX_GIT_BLOB_BYTES,
        ),
        "Git blob inspection",
    )
    return _bytes_stdout(result)


def _commit_metadata_from_object(
    root: Path,
    commit_sha: str,
) -> tuple[list[str], str]:
    result = _require_git(
        _run_git(
            root,
            ["cat-file", "commit", commit_sha],
            binary_output=True,
            output_limit=MAX_GIT_COMMIT_BYTES,
        ),
        "Git commit inspection",
    )
    payload = _bytes_stdout(result)
    if b"\n\n" not in payload:
        raise _DeliveryFailure("error", "Git commit metadata was malformed.")
    headers, message_bytes = payload.split(b"\n\n", 1)
    parent_shas: list[str] = []
    for line in headers.splitlines():
        if not line.startswith(b"parent "):
            continue
        try:
            parent_sha = line.removeprefix(b"parent ").decode("ascii")
        except UnicodeDecodeError as error:
            raise _DeliveryFailure(
                "error",
                "Git commit parent metadata was invalid.",
            ) from error
        if not parent_sha or any(
            character not in "0123456789abcdef" for character in parent_sha
        ):
            raise _DeliveryFailure(
                "error",
                "Git commit parent metadata was invalid.",
            )
        parent_shas.append(parent_sha)
    try:
        message = message_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise _DeliveryFailure(
            "error",
            "Git commit message was not valid UTF-8.",
        ) from error
    return parent_shas, message


def _verify_head_baseline(
    root: Path,
    base_sha: str,
    bundle: PlanApplicationBundle,
    current_bytes: dict[str, bytes | None],
) -> dict[str, str]:
    modes: dict[str, str] = {}
    original_bytes_by_path: dict[str, bytes | None] = {}
    candidate_by_path = {
        _normalized_path(change.path): change for change in bundle.candidate.files
    }
    for planned in sorted(
        bundle.plan.files,
        key=lambda item: _normalized_path(item.path),
    ):
        path = _normalized_path(planned.path)
        entry = _ls_tree_entry(root, base_sha, path)
        if planned.action == "add":
            if entry is not None:
                raise _DeliveryFailure(
                    "conflict",
                    "Planned path had pre-existing working-tree changes relative "
                    f"to HEAD: {path}.",
                )
            original = None
            modes[path] = "100644"
        else:
            if entry is None:
                raise _DeliveryFailure(
                    "conflict",
                    "Planned path had pre-existing working-tree changes relative "
                    f"to HEAD: {path}.",
                )
            original = _cat_blob(root, entry.object_sha)
            if _content_hash(original) != bundle.expected_original_hashes[path]:
                raise _DeliveryFailure(
                    "conflict",
                    "Planned path had pre-existing working-tree changes relative "
                    f"to HEAD: {path}.",
                )
            modes[path] = entry.mode
        original_bytes_by_path[path] = original

        change = candidate_by_path.get(path)
        if change is None:
            raise _DeliveryFailure(
                "error",
                "Bundle candidate path set is incomplete.",
            )
        try:
            approved_bytes = candidate_bytes_for_plan_change(
                planned.action,
                change.content,
                original,
            )
        except PlanApplicationError as error:
            raise _DeliveryFailure("error", str(error)) from error
        expected_hash = bundle.expected_candidate_hashes[path]
        actual_hash = (
            None if approved_bytes is None else _content_hash(approved_bytes)
        )
        if actual_hash != expected_hash:
            raise _DeliveryFailure(
                "error",
                f"Bundle candidate bytes do not match its approved hash: {path}.",
            )
        if approved_bytes != current_bytes[path]:
            raise _DeliveryFailure(
                "stale",
                f"Applied bytes do not equal the approved candidate: {path}.",
            )
    try:
        validate_plan_application_diff_from_original_bytes(
            bundle,
            original_bytes_by_path,
        )
    except PlanApplicationError as error:
        raise _DeliveryFailure("error", str(error)) from error
    return modes


def _validate_branch_name(root: Path, digest: str, branch_name: str | None) -> str:
    branch = (
        branch_name
        if branch_name is not None
        else DEFAULT_BRANCH_PREFIX + digest[:APPROVAL_DIGEST_BRANCH_CHARS]
    )
    if (
        not branch
        or branch == "HEAD"
        or branch.startswith(("refs/", "-"))
        or branch != branch.strip()
        or "\x00" in branch
    ):
        raise _DeliveryFailure("error", "Local delivery branch name is invalid.")
    result = _run_git(root, ["check-ref-format", "--branch", branch])
    if result.output_truncated or result.returncode != 0:
        raise _DeliveryFailure("error", "Local delivery branch name is invalid.")
    return branch


def _commit_message(digest: str, commit_message: str | None) -> str:
    subject = DEFAULT_COMMIT_MESSAGE if commit_message is None else commit_message
    if not isinstance(subject, str) or "\x00" in subject:
        raise _DeliveryFailure("error", "Commit message is invalid.")
    message = subject.rstrip() + f"\n\nApproval-Digest: {digest}\n"
    if len(message) > MAX_COMMIT_MESSAGE_CHARS:
        raise _DeliveryFailure(
            "error",
            "Commit message exceeds "
            f"MAX_COMMIT_MESSAGE_CHARS={MAX_COMMIT_MESSAGE_CHARS}.",
        )
    return message


def _branch_exists(root: Path, full_ref: str) -> bool:
    result = _run_git(root, ["show-ref", "--verify", "--quiet", full_ref])
    if result.output_truncated or result.returncode not in {0, 1}:
        raise _DeliveryFailure("error", "Local branch existence check failed.")
    return result.returncode == 0


def _git_identity(root: Path) -> tuple[str, str]:
    values: list[str] = []
    for key, label in (("user.name", "user.name"), ("user.email", "user.email")):
        result = _run_git(root, ["config", "--get", key])
        if result.output_truncated or result.returncode not in {0, 1}:
            raise _DeliveryFailure("error", f"Git {label} could not be read.")
        value = _text_stdout(result).rstrip("\r\n")
        if result.returncode == 1 or not value:
            raise _DeliveryFailure("error", f"Git {label} is required.")
        if any(character in value for character in ("\x00", "\r", "\n")):
            raise _DeliveryFailure("error", f"Git {label} is invalid.")
        values.append(value)
    return values[0], values[1]


def _parse_name_status(payload: bytes) -> dict[str, str]:
    fields = [field for field in payload.split(b"\x00") if field]
    if len(fields) % 2:
        raise _DeliveryFailure("error", "Git changed-path output was malformed.")
    changes: dict[str, str] = {}
    for index in range(0, len(fields), 2):
        try:
            action = fields[index].decode("ascii")
            path = fields[index + 1].decode(
                "utf-8",
                errors="surrogateescape",
            )
        except UnicodeError as error:
            raise _DeliveryFailure(
                "error",
                "Git changed-path output was invalid.",
            ) from error
        if action not in {"A", "M", "D"} or path in changes:
            raise _DeliveryFailure(
                "error",
                "Git changed-path output was not exact.",
            )
        changes[path] = action
    return changes


def _expected_actions(bundle: PlanApplicationBundle) -> dict[str, str]:
    action_map = {"add": "A", "modify": "M", "delete": "D"}
    return {
        _normalized_path(change.path): action_map[change.action]
        for change in bundle.plan.files
    }


def _cat_index_blob(root: Path, index_env: dict[str, str], path: str) -> bytes:
    result = _require_git(
        _run_git(
            root,
            ["cat-file", "blob", f":{path}"],
            env=index_env,
            binary_output=True,
            output_limit=MAX_GIT_BLOB_BYTES,
        ),
        "Isolated index blob validation",
    )
    return _bytes_stdout(result)


def _verify_index_changes(
    root: Path,
    index_env: dict[str, str],
    base_sha: str,
    bundle: PlanApplicationBundle,
) -> None:
    diff_result = _require_git(
        _run_git(
            root,
            [
                "diff",
                "--cached",
                "--name-status",
                "-z",
                "--no-ext-diff",
                "--no-textconv",
                base_sha,
                "--",
            ],
            env=index_env,
            binary_output=True,
        ),
        "Isolated index changed-path validation",
    )
    if _parse_name_status(_bytes_stdout(diff_result)) != _expected_actions(bundle):
        raise _DeliveryFailure(
            "error",
            "Isolated index path or action set does not match the approved bundle.",
        )
    for planned in bundle.plan.files:
        path = _normalized_path(planned.path)
        if planned.action == "delete":
            entry = _run_git(
                root,
                ["rev-parse", "--verify", f":{path}"],
                env=index_env,
            )
            if entry.returncode == 0:
                raise _DeliveryFailure(
                    "error",
                    f"Deleted path remains in the isolated index: {path}.",
                )
            if entry.output_truncated or entry.returncode not in {1, 128}:
                raise _DeliveryFailure(
                    "error",
                    "Isolated delete validation failed.",
                )
            continue
        staged = _cat_index_blob(root, index_env, path)
        if _content_hash(staged) != bundle.expected_candidate_hashes[path]:
            raise _DeliveryFailure(
                "error",
                f"Isolated index blob does not match approved bytes: {path}.",
            )


def _build_commit(
    root: Path,
    bundle: PlanApplicationBundle,
    base_sha: str,
    current_bytes: dict[str, bytes | None],
    modes: dict[str, str],
    message: str,
    identity: tuple[str, str],
) -> str:
    with tempfile.TemporaryDirectory(prefix="repograph-git-delivery-") as directory:
        temporary_index = Path(directory) / "index"
        index_env = {"GIT_INDEX_FILE": str(temporary_index)}
        _require_git(
            _run_git(root, ["read-tree", base_sha], env=index_env),
            "Isolated index initialization",
        )
        if not temporary_index.is_file():
            raise _DeliveryFailure(
                "error",
                "Isolated Git index was not created.",
            )

        for planned in sorted(
            bundle.plan.files,
            key=lambda item: _normalized_path(item.path),
        ):
            path = _normalized_path(planned.path)
            if planned.action == "delete":
                _require_git(
                    _run_git(
                        root,
                        ["update-index", "--force-remove", "--", path],
                        env=index_env,
                    ),
                    "Isolated index deletion",
                )
                continue
            content = current_bytes[path]
            if content is None:
                raise _DeliveryFailure(
                    "error",
                    "Approved candidate bytes are missing.",
                )
            blob_result = _require_git(
                _run_git(
                    root,
                    ["hash-object", "-w", "--stdin"],
                    input_bytes=content,
                    binary_output=True,
                ),
                "Approved blob creation",
            )
            try:
                blob_sha = _bytes_stdout(blob_result).strip().decode("ascii")
            except UnicodeDecodeError as error:
                raise _DeliveryFailure(
                    "error",
                    "Git returned an invalid blob id.",
                ) from error
            _require_git(
                _run_git(
                    root,
                    [
                        "update-index",
                        "--add",
                        "--cacheinfo",
                        modes[path],
                        blob_sha,
                        path,
                    ],
                    env=index_env,
                ),
                "Isolated index update",
            )

        _verify_index_changes(root, index_env, base_sha, bundle)
        tree_result = _require_git(
            _run_git(root, ["write-tree"], env=index_env),
            "Approved tree creation",
        )
        tree_sha = _text_stdout(tree_result).strip()
        author_name, author_email = identity
        identity_env = {
            "GIT_AUTHOR_NAME": author_name,
            "GIT_AUTHOR_EMAIL": author_email,
            "GIT_COMMITTER_NAME": author_name,
            "GIT_COMMITTER_EMAIL": author_email,
        }
        commit_result = _require_git(
            _run_git(
                root,
                ["commit-tree", tree_sha, "-p", base_sha],
                env=identity_env,
                input_text=message,
            ),
            "Approved commit creation",
        )
        commit_sha = _text_stdout(commit_result).strip()
        if not commit_sha or any(
            character not in "0123456789abcdef" for character in commit_sha
        ):
            raise _DeliveryFailure("error", "Git returned an invalid commit id.")
        return commit_sha


def _verify_commit(
    root: Path,
    full_ref: str,
    commit_sha: str,
    base_sha: str,
    bundle: PlanApplicationBundle,
    *,
    mismatch_status: Literal["stale", "error"] = "error",
    delivery_label: str = "Created",
) -> None:
    ref_result = _require_git(
        _run_git(root, ["rev-parse", "--verify", full_ref]),
        "Created branch verification",
    )
    if _text_stdout(ref_result).strip() != commit_sha:
        raise _DeliveryFailure(
            mismatch_status,
            f"{delivery_label} branch points to an unexpected commit.",
        )

    parent_result = _require_git(
        _run_git(root, ["rev-parse", "--verify", f"{commit_sha}^"]),
        "Created commit parent verification",
    )
    if _text_stdout(parent_result).strip() != base_sha:
        raise _DeliveryFailure(
            mismatch_status,
            f"{delivery_label} commit parent is not the original HEAD.",
        )

    diff_result = _require_git(
        _run_git(
            root,
            [
                "diff-tree",
                "--no-commit-id",
                "--name-status",
                "-r",
                "-z",
                "--no-ext-diff",
                "--no-textconv",
                "--no-renames",
                commit_sha,
            ],
            binary_output=True,
        ),
        "Created commit changed-path verification",
    )
    if _parse_name_status(_bytes_stdout(diff_result)) != _expected_actions(bundle):
        raise _DeliveryFailure(
            mismatch_status,
            f"{delivery_label} commit path or action set does not match the "
            "approved bundle.",
        )

    for planned in bundle.plan.files:
        path = _normalized_path(planned.path)
        entry = _ls_tree_entry(root, commit_sha, path)
        if planned.action == "delete":
            if entry is not None:
                raise _DeliveryFailure(
                    mismatch_status,
                    f"Deleted path is present in the {delivery_label.casefold()} "
                    f"commit: {path}.",
                )
            continue
        if entry is None:
            raise _DeliveryFailure(
                mismatch_status,
                f"Approved path is missing from the {delivery_label.casefold()} "
                f"commit: {path}.",
            )
        if _content_hash(_cat_blob(root, entry.object_sha)) != (
            bundle.expected_candidate_hashes[path]
        ):
            raise _DeliveryFailure(
                mismatch_status,
                f"{delivery_label} commit blob does not match approved bytes: "
                f"{path}.",
            )


def validate_local_git_delivery(
    repository_root: str,
    bundle: PlanApplicationBundle,
    *,
    branch_name: str | None = None,
) -> ValidatedLocalGitDelivery:
    """Revalidate one existing G5A branch and commit without mutating Git state."""

    try:
        root = _validate_repository_root(repository_root)
        repository = _capture_repository_snapshot(root)
        validated = validate_plan_application_bundle(
            str(root),
            bundle,
            repository_state="applied",
        )
        digest = validated.approval_digest
        branch = _validate_branch_name(root, digest, branch_name)
        full_ref = f"refs/heads/{branch}"
        if not _branch_exists(root, full_ref):
            raise _DeliveryFailure("stale", "Local delivery branch does not exist.")

        ref_result = _require_git(
            _run_git(root, ["rev-parse", "--verify", f"{full_ref}^{{commit}}"]),
            "Local delivery branch resolution",
        )
        commit_sha = _text_stdout(ref_result).strip()
        if not commit_sha or any(
            character not in "0123456789abcdef" for character in commit_sha
        ):
            raise _DeliveryFailure(
                "error",
                "Git returned an invalid local delivery commit id.",
            )

        current_bytes, planned_before = _verify_applied_candidate(root, validated)
        _verify_head_baseline(
            root,
            repository.base_sha,
            validated,
            current_bytes,
        )
        _verify_commit(
            root,
            full_ref,
            commit_sha,
            repository.base_sha,
            validated,
            mismatch_status="stale",
            delivery_label="Local delivery",
        )
        parent_shas, commit_message = _commit_metadata_from_object(root, commit_sha)
        if parent_shas != [repository.base_sha]:
            raise _DeliveryFailure(
                "stale",
                "Local delivery commit does not have exactly the approved base "
                "as its only parent.",
            )
        nonempty_lines = [line for line in commit_message.splitlines() if line.strip()]
        expected_trailer = f"Approval-Digest: {digest}"
        if not nonempty_lines or nonempty_lines[-1] != expected_trailer:
            raise _DeliveryFailure(
                "stale",
                "Local delivery commit approval digest does not match the bundle.",
            )
        preservation_errors = _safety_errors(root, repository, planned_before)
        if preservation_errors:
            raise _DeliveryFailure("error", "; ".join(preservation_errors))
        subject = commit_message.splitlines()[0].strip() if commit_message else ""
        return ValidatedLocalGitDelivery(
            branch_name=branch,
            commit_sha=commit_sha,
            base_sha=repository.base_sha,
            approval_digest=digest,
            committed_files=sorted(planned_before),
            commit_subject=subject or None,
        )
    except LocalGitDeliveryValidationError:
        raise
    except _DeliveryFailure as error:
        status: Literal["stale", "error"] = (
            "stale" if error.status in {"stale", "conflict"} else "error"
        )
        raise LocalGitDeliveryValidationError(status, error.reason) from None
    except PlanApplicationError as error:
        raise LocalGitDeliveryValidationError("error", str(error)) from None
    except (OSError, RuntimeError, TypeError, ValueError):
        raise LocalGitDeliveryValidationError(
            "error",
            "Local Git delivery validation failed safely.",
        ) from None


def _safety_errors(
    root: Path,
    before: _RepositorySnapshot,
    planned_before: dict[str, _PathSnapshot],
) -> list[str]:
    errors: list[str] = []
    try:
        head = _require_git(
            _run_git(root, ["rev-parse", "--verify", "HEAD^{commit}"]),
            "HEAD preservation check",
        )
        if _text_stdout(head).strip() != before.base_sha:
            errors.append("Current HEAD commit changed during local delivery.")

        symbolic = _run_git(root, ["symbolic-ref", "--quiet", "HEAD"])
        if symbolic.output_truncated or symbolic.returncode not in {0, 1}:
            errors.append("Current branch could not be rechecked.")
        else:
            current_symbolic = (
                _text_stdout(symbolic).strip()
                if symbolic.returncode == 0
                else None
            )
            if current_symbolic != before.symbolic_head:
                errors.append(
                    "Current checked-out branch changed during local delivery."
                )

        status = _require_git(
            _run_git(root, ["status", "--porcelain=v1", "--untracked-files=all"]),
            "Working-tree preservation check",
        )
        if _text_stdout(status) != before.status_output:
            errors.append("Git working-tree status changed during local delivery.")

        if _index_snapshot(before.index_path) != before.index:
            errors.append(
                "The user's real Git index changed during local delivery."
            )

        for path, snapshot in planned_before.items():
            if _capture_path_snapshot(root, path) != snapshot:
                errors.append(
                    f"Planned working-tree path changed during delivery: {path}."
                )
    except _DeliveryFailure as error:
        errors.append(error.reason)
    return errors


def capture_local_git_delivery_safety_snapshot(
    repository_root: str,
    delivery: ValidatedLocalGitDelivery,
) -> LocalGitDeliverySafetySnapshot:
    """Capture local state that G5B must preserve across network operations."""

    try:
        root = _validate_repository_root(repository_root)
        repository = _capture_repository_snapshot(root)
        planned_paths = {
            path: _capture_path_snapshot(root, path)
            for path in delivery.committed_files
        }
        ref_result = _require_git(
            _run_git(
                root,
                [
                    "rev-parse",
                    "--verify",
                    f"refs/heads/{delivery.branch_name}^{{commit}}",
                ],
            ),
            "Local delivery branch snapshot",
        )
        branch_sha = _text_stdout(ref_result).strip()
        if branch_sha != delivery.commit_sha:
            raise _DeliveryFailure(
                "stale",
                "Local delivery branch changed before remote delivery.",
            )
        return LocalGitDeliverySafetySnapshot(
            repository_root=root,
            repository=repository,
            planned_paths=planned_paths,
            branch_name=delivery.branch_name,
            branch_sha=branch_sha,
        )
    except _DeliveryFailure as error:
        status: Literal["stale", "error"] = (
            "stale" if error.status in {"stale", "conflict"} else "error"
        )
        raise LocalGitDeliveryValidationError(status, error.reason) from None
    except (OSError, RuntimeError, TypeError, ValueError):
        raise LocalGitDeliveryValidationError(
            "error",
            "Local Git state snapshot failed safely.",
        ) from None


def local_git_delivery_safety_errors(
    snapshot: LocalGitDeliverySafetySnapshot,
) -> list[str]:
    """Return bounded preservation failures detected after G5B activity."""

    errors = _safety_errors(
        snapshot.repository_root,
        snapshot.repository,
        snapshot.planned_paths,
    )
    try:
        ref_result = _require_git(
            _run_git(
                snapshot.repository_root,
                [
                    "rev-parse",
                    "--verify",
                    f"refs/heads/{snapshot.branch_name}^{{commit}}",
                ],
            ),
            "Local delivery branch preservation check",
        )
        if _text_stdout(ref_result).strip() != snapshot.branch_sha:
            errors.append("Local delivery branch changed during remote delivery.")
    except _DeliveryFailure as error:
        errors.append(error.reason)
    return errors


def _rollback_created_ref(
    root: Path,
    full_ref: str,
    commit_sha: str,
) -> bool:
    result = _run_git(root, ["update-ref", "-d", full_ref, commit_sha])
    if result.output_truncated or result.returncode != 0:
        return False
    return not _branch_exists(root, full_ref)


def create_local_git_delivery(
    repository_root: str,
    bundle: PlanApplicationBundle,
    *,
    approved: bool = False,
    branch_name: str | None = None,
    commit_message: str | None = None,
) -> LocalGitDeliveryResult:
    """Create one new local branch and commit from an applied approved bundle."""

    digest = getattr(bundle, "approval_digest", None)
    if not approved:
        return LocalGitDeliveryResult(
            status="not_requested",
            approval_digest=digest,
        )

    root: Path | None = None
    repository_before: _RepositorySnapshot | None = None
    planned_before: dict[str, _PathSnapshot] | None = None
    branch: str | None = None
    full_ref: str | None = None
    commit_sha: str | None = None
    ref_created = False
    failure: _DeliveryFailure | None = None
    warnings: list[str] = []

    try:
        root = _validate_repository_root(repository_root)
        repository_before = _capture_repository_snapshot(root)
        validated = validate_plan_application_bundle(
            str(root),
            bundle,
            repository_state="applied",
        )
        digest = validated.approval_digest
        branch = _validate_branch_name(root, digest, branch_name)
        full_ref = f"refs/heads/{branch}"
        if _branch_exists(root, full_ref):
            raise _DeliveryFailure(
                "conflict",
                "Local delivery branch already exists.",
            )
        message = _commit_message(digest, commit_message)
        identity = _git_identity(root)

        current_bytes, planned_before = _verify_applied_candidate(root, validated)
        modes = _verify_head_baseline(
            root,
            repository_before.base_sha,
            validated,
            current_bytes,
        )
        commit_sha = _build_commit(
            root,
            validated,
            repository_before.base_sha,
            current_bytes,
            modes,
            message,
            identity,
        )

        zero_object_id = "0" * len(repository_before.base_sha)
        create_ref = _run_git(
            root,
            ["update-ref", full_ref, commit_sha, zero_object_id],
        )
        if create_ref.output_truncated:
            raise _DeliveryFailure(
                "error",
                "Local branch creation output was too large.",
            )
        if create_ref.returncode != 0:
            if _branch_exists(root, full_ref):
                raise _DeliveryFailure(
                    "conflict",
                    "Local delivery branch appeared before safe creation.",
                )
            raise _DeliveryFailure(
                "error",
                "Local delivery branch could not be created.",
            )
        ref_created = True

        _verify_commit(
            root,
            full_ref,
            commit_sha,
            repository_before.base_sha,
            validated,
        )
        preservation_errors = _safety_errors(
            root,
            repository_before,
            planned_before,
        )
        if preservation_errors:
            raise _DeliveryFailure(
                "error",
                "; ".join(preservation_errors),
            )
    except _DeliveryFailure as error:
        failure = error
    except PlanApplicationError as error:
        failure = _DeliveryFailure("error", _single_line(error))
    except (OSError, RuntimeError, TypeError, ValueError):
        failure = _DeliveryFailure(
            "error",
            "Local Git delivery failed safely.",
        )

    if failure is not None:
        if (
            ref_created
            and root is not None
            and full_ref is not None
            and commit_sha is not None
        ):
            if _rollback_created_ref(root, full_ref, commit_sha):
                warnings.append(
                    "Post-create verification failed; the newly created local "
                    "branch was rolled back."
                )
            else:
                warnings.append(
                    "Post-create verification failed and the newly created local "
                    "branch could not be rolled back."
                )
        if (
            root is not None
            and repository_before is not None
            and planned_before is not None
        ):
            for safety_error in _safety_errors(
                root,
                repository_before,
                planned_before,
            ):
                if safety_error not in warnings:
                    warnings.append(safety_error)
        return LocalGitDeliveryResult(
            status=failure.status,
            branch_name=branch,
            base_sha=(
                repository_before.base_sha
                if repository_before is not None
                else None
            ),
            approval_digest=digest,
            warnings=warnings,
            failure_reason=_single_line(failure.reason),
        )

    if (
        repository_before is None
        or branch is None
        or commit_sha is None
        or digest is None
        or planned_before is None
    ):
        return LocalGitDeliveryResult(
            status="error",
            approval_digest=digest,
            failure_reason="Local Git delivery ended without a complete result.",
        )
    return LocalGitDeliveryResult(
        status="created",
        branch_name=branch,
        commit_sha=commit_sha,
        base_sha=repository_before.base_sha,
        approval_digest=digest,
        committed_files=sorted(planned_before),
    )


def render_local_git_delivery_result(result: LocalGitDeliveryResult) -> str:
    """Render a concise deterministic local-delivery outcome."""

    lines = [f"Status: {result.status}"]
    if result.branch_name is not None:
        lines.append(f"Branch: {result.branch_name}")
    if result.commit_sha is not None:
        lines.append(f"Commit: {result.commit_sha}")
    if result.base_sha is not None:
        lines.append(f"Base: {result.base_sha}")
    if result.approval_digest is not None:
        lines.append(f"Approval digest: {result.approval_digest}")
    if result.committed_files:
        lines.append("Committed files:")
        lines.extend(f"- {path}" for path in result.committed_files)
    if result.failure_reason is not None:
        lines.append(f"Failure: {result.failure_reason}")
    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in result.warnings)
    return "\n".join(lines)
