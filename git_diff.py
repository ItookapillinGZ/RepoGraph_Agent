"""Safe, read-only collection of target diffs from local Git worktrees."""

import re
import subprocess  # nosec B404
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from change_set import (
    MAX_CHANGESET_DIFF_CHARS,
    MAX_CHANGESET_FILES,
)
from diff_context import MAX_DIFF_CHARS

GIT_TIMEOUT_SECONDS = 10
GitDiffMode = Literal["working_tree", "base"]

_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")
_DIFF_OPTIONS = [
    "--no-ext-diff",
    "--no-textconv",
    "--no-color",
    "--src-prefix=a/",
    "--dst-prefix=b/",
]


class GitDiffResult(BaseModel):
    """Bounded raw diff and metadata collected from one local Git worktree."""

    model_config = ConfigDict(extra="forbid")

    mode: GitDiffMode
    repository_root: str = Field(min_length=1)
    target_file: str = Field(min_length=1)
    base_ref: str | None = None
    diff_text: str = ""
    warnings: list[str] = Field(default_factory=list)


class GitChangeSetDiffResult(BaseModel):
    """Bounded multi-file diff and untracked metadata from a local worktree."""

    model_config = ConfigDict(extra="forbid")

    mode: GitDiffMode
    repository_root: str = Field(min_length=1)
    base_ref: str | None = None
    diff_text: str = ""
    untracked_files: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class GitDiffError(RuntimeError):
    """Stable error raised when a safe local Git diff cannot be collected."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code

    def to_payload(self) -> dict[str, object]:
        """Return a stable machine-readable CLI failure payload."""

        return {
            "status": "failed",
            "error": {
                "type": "git_diff_failed",
                "code": self.code,
                "message": str(self),
            },
        }


def _is_within(path: Path, repository_root: Path) -> bool:
    return path == repository_root or repository_root in path.parents


def _validated_paths(
    repository_root: str,
    target_file: str,
) -> tuple[Path, str]:
    root = _validated_repository_root(repository_root)
    raw_target = Path(target_file)
    candidate = raw_target if raw_target.is_absolute() else root / raw_target
    try:
        target = candidate.resolve(strict=True)
    except OSError as error:
        raise GitDiffError(
            "invalid_target",
            f"Target file cannot be resolved: {target_file}",
        ) from error
    if not target.is_file():
        raise GitDiffError(
            "invalid_target",
            f"Target path is not a file: {target_file}",
        )
    if not _is_within(target, root):
        raise GitDiffError(
            "target_outside_repository",
            "Target file must resolve inside repository root.",
        )

    return root, target.relative_to(root).as_posix()


def _validated_repository_root(repository_root: str) -> Path:
    try:
        root = Path(repository_root).resolve(strict=True)
    except OSError as error:
        raise GitDiffError(
            "invalid_repository",
            f"Repository root cannot be resolved: {repository_root}",
        ) from error
    if not root.is_dir():
        raise GitDiffError(
            "invalid_repository",
            f"Repository root is not a directory: {repository_root}",
        )

    return root


def _run_git(
    repository_root: Path,
    arguments: list[str],
) -> subprocess.CompletedProcess[str]:
    command = ["git", "--no-pager", "--literal-pathspecs", *arguments]
    try:
        return subprocess.run(  # nosec B603
            command,
            cwd=str(repository_root),
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as error:
        raise GitDiffError(
            "git_not_found",
            "Git executable was not found.",
        ) from error
    except subprocess.TimeoutExpired as error:
        raise GitDiffError(
            "git_timeout",
            f"Git command timed out after {GIT_TIMEOUT_SECONDS} seconds.",
        ) from error
    except UnicodeError as error:
        raise GitDiffError(
            "git_output_error",
            "Git output could not be decoded.",
        ) from error
    except OSError as error:
        raise GitDiffError(
            "git_start_failed",
            f"Git command could not start: {error}",
        ) from error


def _error_detail(completed: subprocess.CompletedProcess[str]) -> str:
    detail = " ".join(completed.stderr.strip().split())[:300]
    return f": {detail}" if detail else ""


def _validate_worktree(repository_root: Path) -> None:
    completed = _run_git(
        repository_root,
        ["rev-parse", "--is-inside-work-tree"],
    )
    if (
        completed.returncode != 0
        or completed.stdout.strip().casefold() != "true"
    ):
        raise GitDiffError(
            "not_git_worktree",
            "Repository root is not inside a Git worktree.",
        )


def _resolve_commit(
    repository_root: Path,
    ref: str,
    *,
    code: str,
    message: str,
) -> str:
    completed = _run_git(
        repository_root,
        [
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{ref}^{{commit}}",
        ],
    )
    if completed.returncode != 0:
        raise GitDiffError(code, message)

    lines = completed.stdout.strip().splitlines()
    commit = lines[0] if lines else ""
    if _COMMIT_SHA.fullmatch(commit) is None:
        raise GitDiffError(
            "invalid_git_output",
            "Git returned an invalid commit identifier.",
        )
    return commit.lower()


def _resolve_head(repository_root: Path) -> str:
    return _resolve_commit(
        repository_root,
        "HEAD",
        code="head_unresolved",
        message=(
            "HEAD could not be resolved. The repository may not have an "
            "initial commit."
        ),
    )


def _validate_base_ref(base_ref: str) -> None:
    if (
        not base_ref
        or base_ref.startswith("-")
        or any(character in base_ref for character in ("\x00", "\r", "\n"))
    ):
        raise GitDiffError(
            "base_unresolved",
            "Base ref could not be resolved locally.",
        )


def _resolve_base(repository_root: Path, base_ref: str) -> str:
    _validate_base_ref(base_ref)
    return _resolve_commit(
        repository_root,
        base_ref,
        code="base_unresolved",
        message="Base ref could not be resolved locally.",
    )


def _target_is_tracked(repository_root: Path, target_file: str) -> bool:
    completed = _run_git(
        repository_root,
        [
            "ls-files",
            "--error-unmatch",
            "--",
            target_file,
        ],
    )
    if completed.returncode == 0:
        return True
    if completed.returncode == 1:
        return False
    raise GitDiffError(
        "git_command_failed",
        "Git could not determine whether the target is tracked"
        f"{_error_detail(completed)}",
    )


def _collect_diff(
    repository_root: Path,
    revision: str,
    target_file: str,
) -> str:
    completed = _run_git(
        repository_root,
        [
            "diff",
            *_DIFF_OPTIONS,
            revision,
            "--",
            target_file,
        ],
    )
    if completed.returncode != 0:
        raise GitDiffError(
            "git_diff_failed",
            "Git diff failed"
            f"{_error_detail(completed)}",
        )
    return completed.stdout


def _collect_change_set_diff(repository_root: Path, revision: str) -> str:
    completed = _run_git(
        repository_root,
        [
            "diff",
            *_DIFF_OPTIONS,
            "--find-renames",
            revision,
        ],
    )
    if completed.returncode != 0:
        raise GitDiffError(
            "git_diff_failed",
            "Git change-set diff failed"
            f"{_error_detail(completed)}",
        )
    return completed.stdout


def _bounded_diff(diff_text: str, warnings: list[str]) -> str:
    if len(diff_text) <= MAX_DIFF_CHARS:
        return diff_text
    warnings.append(
        f"Git diff truncated at MAX_DIFF_CHARS={MAX_DIFF_CHARS}."
    )
    return diff_text[:MAX_DIFF_CHARS]


def _bounded_change_set_diff(diff_text: str, warnings: list[str]) -> str:
    if len(diff_text) <= MAX_CHANGESET_DIFF_CHARS:
        return diff_text
    warnings.append(
        "Git change-set diff truncated at MAX_CHANGESET_DIFF_CHARS="
        f"{MAX_CHANGESET_DIFF_CHARS}."
    )
    return diff_text[:MAX_CHANGESET_DIFF_CHARS]


def _collect_untracked_files(
    repository_root: Path,
    warnings: list[str],
) -> list[str]:
    completed = _run_git(
        repository_root,
        ["ls-files", "--others", "--exclude-standard", "-z"],
    )
    if completed.returncode != 0:
        raise GitDiffError(
            "git_command_failed",
            "Git could not enumerate untracked files"
            f"{_error_detail(completed)}",
        )

    output = completed.stdout
    if len(output) > MAX_CHANGESET_DIFF_CHARS:
        warnings.append(
            "Untracked file metadata truncated at MAX_CHANGESET_DIFF_CHARS="
            f"{MAX_CHANGESET_DIFF_CHARS}."
        )
        output = output[:MAX_CHANGESET_DIFF_CHARS]
        output = output.rsplit("\0", 1)[0]
    paths = sorted({path for path in output.split("\0") if path})
    if len(paths) > MAX_CHANGESET_FILES:
        warnings.append(
            "Untracked files truncated at "
            f"MAX_CHANGESET_FILES={MAX_CHANGESET_FILES}."
        )
    return paths[:MAX_CHANGESET_FILES]


def collect_working_tree_diff(
    repository_root: str,
    target_file: str,
) -> GitDiffResult:
    """Collect tracked staged and unstaged target changes against local HEAD."""

    root, relative_target = _validated_paths(repository_root, target_file)
    _validate_worktree(root)
    _resolve_head(root)

    warnings: list[str] = []
    if not _target_is_tracked(root, relative_target):
        warnings.append(
            "Target is untracked and is not represented by normal HEAD diff."
        )
        return GitDiffResult(
            mode="working_tree",
            repository_root=str(root),
            target_file=relative_target,
            warnings=warnings,
        )

    diff_text = _bounded_diff(
        _collect_diff(root, "HEAD", relative_target),
        warnings,
    )
    if not diff_text:
        warnings.append("Git produced no diff for the target file.")
    return GitDiffResult(
        mode="working_tree",
        repository_root=str(root),
        target_file=relative_target,
        diff_text=diff_text,
        warnings=warnings,
    )


def collect_base_diff(
    repository_root: str,
    target_file: str,
    base_ref: str,
) -> GitDiffResult:
    """Collect the committed target change from local base merge-base to HEAD."""

    _validate_base_ref(base_ref)
    root, relative_target = _validated_paths(repository_root, target_file)
    _validate_worktree(root)
    _resolve_head(root)
    base_commit = _resolve_base(root, base_ref)

    warnings: list[str] = []
    if not _target_is_tracked(root, relative_target):
        warnings.append(
            "Target is untracked in the current index; branch comparison only "
            "includes committed history."
        )

    diff_text = _bounded_diff(
        _collect_diff(root, f"{base_commit}...HEAD", relative_target),
        warnings,
    )
    if not diff_text:
        warnings.append("Git produced no diff for the target file.")
    return GitDiffResult(
        mode="base",
        repository_root=str(root),
        target_file=relative_target,
        base_ref=base_ref,
        diff_text=diff_text,
        warnings=warnings,
    )


def collect_git_diff(
    repository_root: str,
    target_file: str,
    mode: GitDiffMode,
    base_ref: str | None = None,
) -> GitDiffResult:
    """Dispatch one explicit Git review mode without modifying the repository."""

    if mode == "working_tree":
        if base_ref is not None:
            raise ValueError("working_tree mode does not accept base_ref.")
        return collect_working_tree_diff(repository_root, target_file)
    if base_ref is None:
        raise ValueError("base mode requires base_ref.")
    return collect_base_diff(repository_root, target_file, base_ref)


def collect_git_change_set_diff(
    repository_root: str,
    mode: GitDiffMode,
    base_ref: str | None = None,
) -> GitChangeSetDiffResult:
    """Collect one bounded multi-file local diff without repository writes."""

    root = _validated_repository_root(repository_root)
    _validate_worktree(root)
    _resolve_head(root)
    warnings: list[str] = []

    if mode == "working_tree":
        if base_ref is not None:
            raise ValueError("working_tree mode does not accept base_ref.")
        revision = "HEAD"
        untracked_files = _collect_untracked_files(root, warnings)
        resolved_base = None
    else:
        if base_ref is None:
            raise ValueError("base mode requires base_ref.")
        _validate_base_ref(base_ref)
        base_commit = _resolve_base(root, base_ref)
        revision = f"{base_commit}...HEAD"
        untracked_files = []
        resolved_base = base_ref

    diff_text = _bounded_change_set_diff(
        _collect_change_set_diff(root, revision),
        warnings,
    )
    if not diff_text and not untracked_files:
        warnings.append("Git produced no changes for the change-set.")
    return GitChangeSetDiffResult(
        mode=mode,
        repository_root=str(root),
        base_ref=resolved_base,
        diff_text=diff_text,
        untracked_files=untracked_files,
        warnings=warnings,
    )


def resolve_local_head(repository_root: str) -> str:
    """Resolve local HEAD through the existing read-only bounded Git boundary."""

    root = _validated_repository_root(repository_root)
    _validate_worktree(root)
    return _resolve_head(root)
