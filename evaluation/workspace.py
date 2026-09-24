"""Fresh Git workspace preparation and exact candidate materialization."""

from __future__ import annotations

import hashlib
import shutil
import subprocess  # nosec B404
from pathlib import Path

from engineering_plan import EngineeringPlan
from plan_application import candidate_bytes_for_plan_change
from plan_execution import MultiFileCandidate, validate_multi_file_candidate


class WorkspaceError(RuntimeError):
    """Raised when task isolation or deterministic Git operations fail."""


def _run_git(
    repository: Path,
    *args: str,
    timeout: float = 60,
    preserve_output: bool = False,
) -> str:
    try:
        completed = subprocess.run(  # nosec B603 B607
            ["git", *args],
            cwd=repository,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise WorkspaceError(f"Git command failed to execute: {error}") from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[:2_000]
        raise WorkspaceError(f"Git {' '.join(args)} failed: {detail}")
    return completed.stdout if preserve_output else completed.stdout.strip()


def _component(value: str) -> str:
    readable = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in value
    ).strip("-")[:50]
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{readable or 'item'}-{digest}"


def _source_snapshot(source: Path) -> tuple[str, str]:
    return (
        _run_git(source, "rev-parse", "HEAD"),
        _run_git(source, "status", "--porcelain=v1", "--untracked-files=all"),
    )


def prepare_task_workspace(
    source_repository: str,
    base_commit: str,
    workspace_root: str | Path,
    experiment_id: str,
    task_id: str,
) -> tuple[Path, str]:
    """Clone a local source and checkout the exact commit in a fresh task root."""

    source = Path(source_repository).resolve(strict=True)
    if not source.is_dir():
        raise WorkspaceError(f"Source repository is not a directory: {source}")
    root = Path(workspace_root).resolve()
    if root == Path(root.anchor):
        raise WorkspaceError("Evaluation workspace cannot be a filesystem root.")
    if root == source or source in root.parents:
        raise WorkspaceError(
            "Evaluation workspace cannot be inside the source repository."
        )
    before = _source_snapshot(source)
    root.mkdir(parents=True, exist_ok=True)
    experiment_root = root / _component(experiment_id)
    destination = experiment_root / _component(task_id)
    resolved_parent = destination.parent.resolve()
    if root != resolved_parent and root not in resolved_parent.parents:
        raise WorkspaceError("Task workspace escaped the configured workspace root.")
    if destination.exists() or destination.is_symlink():
        resolved = destination.resolve()
        if root not in resolved.parents or destination.is_symlink():
            raise WorkspaceError("Refusing to replace an unsafe task workspace.")
        shutil.rmtree(destination)
    experiment_root.mkdir(parents=True, exist_ok=True)

    completed = subprocess.run(  # nosec B603 B607
        ["git", "clone", "--no-hardlinks", "--quiet", str(source), str(destination)],
        cwd=experiment_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        raise WorkspaceError(f"Git clone failed: {completed.stderr.strip()[:2_000]}")
    resolved_commit = _run_git(destination, "rev-parse", f"{base_commit}^{{commit}}")
    _run_git(destination, "checkout", "--detach", "--quiet", resolved_commit)
    head = _run_git(destination, "rev-parse", "HEAD")
    if head != resolved_commit:
        raise WorkspaceError("Fresh task workspace did not reach the requested commit.")
    after = _source_snapshot(source)
    if after != before:
        raise WorkspaceError(
            "Source repository changed while preparing evaluation workspace."
        )
    return destination.resolve(strict=True), resolved_commit


def task_workspace_path(
    workspace_root: str | Path,
    experiment_id: str,
    task_id: str,
) -> Path:
    """Resolve the deterministic task clone path without touching the source."""

    root = Path(workspace_root).resolve()
    if root == Path(root.anchor):
        raise WorkspaceError("Evaluation workspace cannot be a filesystem root.")
    destination = root / _component(experiment_id) / _component(task_id)
    resolved = destination.resolve(strict=False)
    if root not in resolved.parents:
        raise WorkspaceError("Task workspace escaped the configured workspace root.")
    return resolved


def cleanup_task_workspace(
    workspace_root: str | Path,
    experiment_id: str,
    task_id: str,
) -> None:
    """Best-effort removal of one validated disposable task clone."""

    destination = task_workspace_path(workspace_root, experiment_id, task_id)
    if destination.exists() and not destination.is_symlink():
        shutil.rmtree(destination)


def reset_task_workspace(workspace: str | Path, base_commit: str) -> None:
    """Discard all prior candidate/evaluator effects inside one isolated clone."""

    root = Path(workspace).resolve(strict=True)
    _run_git(root, "reset", "--hard", "--quiet", base_commit)
    _run_git(root, "clean", "-fdx", "--quiet")


def _target(root: Path, raw_path: str) -> Path:
    normalized = raw_path.replace("\\", "/")
    path = Path(normalized)
    parts = tuple(part for part in normalized.split("/") if part not in ("", "."))
    if (
        not parts
        or path.is_absolute()
        or path.drive
        or any(part == ".." for part in parts)
        or "\x00" in normalized
    ):
        raise WorkspaceError(f"Unsafe candidate path: {raw_path}")
    target = root.joinpath(*parts)
    current = root
    for part in parts:
        current /= part
        if current.is_symlink():
            raise WorkspaceError(f"Candidate path contains a symbolic link: {raw_path}")
    resolved = target.resolve(strict=False)
    if root != resolved and root not in resolved.parents:
        raise WorkspaceError(f"Candidate path escaped the task workspace: {raw_path}")
    return target


def apply_candidate_exact(
    workspace: str | Path,
    plan: EngineeringPlan,
    candidate: MultiFileCandidate,
) -> list[str]:
    """Write only deterministic bytes authorized by the fixed plan/candidate."""

    root = Path(workspace).resolve(strict=True)
    errors = validate_multi_file_candidate(candidate, plan, str(root))
    if errors:
        raise WorkspaceError("Invalid evaluation candidate: " + " ".join(errors))
    changed: list[str] = []
    for change in sorted(candidate.files, key=lambda item: item.path.casefold()):
        normalized = change.path.replace("\\", "/")
        target = _target(root, normalized)
        original = target.read_bytes() if target.is_file() else None
        rendered = candidate_bytes_for_plan_change(
            change.action, change.content, original
        )
        if rendered is None:
            target.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(rendered)
        changed.append(normalized)
    return changed


def extract_candidate_patch(
    workspace: str | Path,
    base_commit: str,
    candidate: MultiFileCandidate,
) -> tuple[str, list[str]]:
    """Return Git's unified diff restricted to exact candidate paths."""

    root = Path(workspace).resolve(strict=True)
    paths = sorted({item.path.replace("\\", "/") for item in candidate.files})
    added = [
        item.path.replace("\\", "/") for item in candidate.files if item.action == "add"
    ]
    if added:
        _run_git(root, "add", "-N", "--", *added)
    patch = _run_git(
        root,
        "diff",
        "--binary",
        "--no-ext-diff",
        base_commit,
        "--",
        *paths,
        preserve_output=True,
    )
    changed_output = _run_git(root, "diff", "--name-only", base_commit, "--", *paths)
    changed = [line for line in changed_output.splitlines() if line]
    unrelated = sorted(set(changed) - set(paths))
    if unrelated:
        raise WorkspaceError(
            "Candidate patch contains paths outside the candidate: "
            + ", ".join(unrelated)
        )
    return patch, changed
