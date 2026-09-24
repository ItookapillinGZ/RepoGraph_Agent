"""Shared bounded temporary repository workspace preparation.

The copy deliberately excludes repository metadata, environments, build
artifacts, local secrets, symlinks, and non-regular files. It is a bounded
working copy, not a security sandbox: code executed from it still runs on the
local machine.
"""

import os
import shutil
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from repository_context import EXCLUDED_DIRECTORIES

MAX_WORKSPACE_FILES = 5_000
MAX_WORKSPACE_BYTES = 100_000_000


class WorkspaceCopyResult(BaseModel):
    """Auditable totals from one bounded repository copy."""

    model_config = ConfigDict(extra="forbid")

    files_copied: int = 0
    bytes_copied: int = 0
    warnings: list[str] = Field(default_factory=list)


class WorkspaceCopyError(RuntimeError):
    """Raised when a complete bounded workspace copy cannot be produced."""

    def __init__(self, message: str, warnings: list[str] | None = None) -> None:
        super().__init__(message)
        self.warnings = list(warnings or [])


def is_within(path: Path, root: Path) -> bool:
    """Return whether path is root or one of its descendants."""

    return path == root or root in path.parents


def is_secret_file(file_name: str) -> bool:
    """Recognize local environment and common private-key file names."""

    normalized = file_name.casefold()
    return (
        normalized == ".env"
        or normalized.startswith(".env.")
        or normalized.endswith((".pem", ".key"))
    )


def _add_warning(warnings: list[str], warning: str) -> None:
    if warning not in warnings:
        warnings.append(warning)


def copy_repository_bounded(
    repository_root: str | Path,
    destination_root: str | Path,
    *,
    max_files: int = MAX_WORKSPACE_FILES,
    max_bytes: int = MAX_WORKSPACE_BYTES,
) -> WorkspaceCopyResult:
    """Copy a complete repository within deterministic hard budgets."""

    if max_files < 1:
        raise ValueError("max_files must be at least 1.")
    if max_bytes < 1:
        raise ValueError("max_bytes must be at least 1.")

    try:
        source = Path(repository_root).resolve(strict=True)
    except OSError as error:
        raise WorkspaceCopyError(
            f"Repository root could not be resolved: {error}."
        ) from error
    if not source.is_dir():
        raise WorkspaceCopyError("Repository root is not a directory.")

    destination = Path(destination_root).resolve(strict=False)
    if is_within(destination, source):
        raise WorkspaceCopyError(
            "Temporary workspace destination must be outside the source repository."
        )

    destination.mkdir(parents=True, exist_ok=True)
    excluded = {name.casefold() for name in EXCLUDED_DIRECTORIES}
    warnings: list[str] = []
    files_copied = 0
    bytes_copied = 0
    skipped_symlink = False
    skipped_secret = False
    skipped_special = False

    for current, directory_names, file_names in os.walk(
        source,
        topdown=True,
        followlinks=False,
    ):
        current_path = Path(current)
        kept_directories: list[str] = []
        for directory_name in sorted(directory_names):
            candidate = current_path / directory_name
            if directory_name.casefold() in excluded:
                continue
            if candidate.is_symlink():
                skipped_symlink = True
                continue
            kept_directories.append(directory_name)
        directory_names[:] = kept_directories

        relative_directory = current_path.relative_to(source)
        for file_name in sorted(file_names):
            source_file = current_path / file_name
            if is_secret_file(file_name):
                skipped_secret = True
                continue
            if source_file.is_symlink():
                skipped_symlink = True
                continue
            if not source_file.is_file():
                skipped_special = True
                continue

            try:
                resolved_source = source_file.resolve(strict=True)
                file_size = resolved_source.stat().st_size
            except OSError as error:
                raise WorkspaceCopyError(
                    f"Repository file could not be inspected: {error}.",
                    warnings,
                ) from error
            if not is_within(resolved_source, source):
                skipped_symlink = True
                continue

            next_file_count = files_copied + 1
            next_byte_count = bytes_copied + file_size
            if next_file_count > max_files:
                raise WorkspaceCopyError(
                    "Repository exceeds temporary workspace file budget "
                    f"MAX_WORKSPACE_FILES={max_files}.",
                    warnings,
                )
            if next_byte_count > max_bytes:
                raise WorkspaceCopyError(
                    "Repository exceeds temporary workspace byte budget "
                    f"MAX_WORKSPACE_BYTES={max_bytes}.",
                    warnings,
                )

            destination_file = destination / relative_directory / file_name
            try:
                destination_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(resolved_source, destination_file)
            except OSError as error:
                raise WorkspaceCopyError(
                    f"Repository file could not be copied: {error}.",
                    warnings,
                ) from error
            files_copied = next_file_count
            bytes_copied = next_byte_count

    if skipped_symlink:
        _add_warning(
            warnings,
            "Skipped one or more repository symlinks while building the "
            "temporary workspace.",
        )
    if skipped_secret:
        _add_warning(
            warnings,
            "Skipped one or more local secret/environment files while building "
            "the temporary workspace.",
        )
    if skipped_special:
        _add_warning(
            warnings,
            "Skipped one or more non-regular repository files while building "
            "the temporary workspace.",
        )

    return WorkspaceCopyResult(
        files_copied=files_copied,
        bytes_copied=bytes_copied,
        warnings=warnings,
    )
