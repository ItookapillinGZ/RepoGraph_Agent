"""Controlled application of an already verified candidate fix.

This module does not generate or verify candidates and performs no Git actions.
"""

import hashlib
import os
import stat
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from fix_context import CodeFix
from fix_verification import FixVerificationResult
from source_file_encoding import candidate_bytes_for_modify as _candidate_bytes
from source_file_encoding import normalized_newlines as _normalized_newlines


class ApplyFixResult(BaseModel):
    """Structured outcome from the original-repository write boundary."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["not_requested", "applied", "stale", "error"]
    target_file: str | None = None
    bytes_written: int = 0
    warnings: list[str] = Field(default_factory=list)


class ApplicationTargetError(ValueError):
    """Raised when an application target fails path-boundary validation."""


def _is_within(path: Path, repository_root: Path) -> bool:
    return path == repository_root or repository_root in path.parents


def _resolve_application_target(
    repository_root: str,
    target_file: str,
) -> tuple[Path, Path, str]:
    try:
        root = Path(repository_root).resolve(strict=True)
    except OSError as error:
        raise ApplicationTargetError(
            f"Repository root could not be resolved: {error}."
        ) from error
    if not root.is_dir():
        raise ApplicationTargetError("Repository root is not a directory.")

    raw_target = Path(target_file)
    target_candidate = raw_target if raw_target.is_absolute() else root / raw_target
    if target_candidate.is_symlink():
        raise ApplicationTargetError("Apply target must not be a symlink.")
    if not target_candidate.exists():
        raise ApplicationTargetError("Apply target does not exist.")

    try:
        target = target_candidate.resolve(strict=True)
    except OSError as error:
        raise ApplicationTargetError(
            f"Apply target could not be resolved: {error}."
        ) from error
    if not _is_within(target, root):
        raise ApplicationTargetError(
            "Apply target must resolve inside repository root."
        )
    if not target.is_file():
        raise ApplicationTargetError("Apply target must be a regular file.")
    return root, target, target.relative_to(root).as_posix()


def content_fingerprint(content: bytes) -> str:
    """Return the deterministic SHA-256 fingerprint for exact target bytes."""

    return hashlib.sha256(content).hexdigest()


def capture_original_content_hash(
    repository_root: str | None,
    target_file: str | None,
) -> str | None:
    """Capture a safe initial target fingerprint or return no fingerprint."""

    if repository_root is None or target_file is None:
        return None
    try:
        _, target, _ = _resolve_application_target(repository_root, target_file)
        return content_fingerprint(target.read_bytes())
    except (ApplicationTargetError, OSError):
        return None


def _precondition_error(message: str) -> ApplyFixResult:
    return ApplyFixResult(status="error", warnings=[message])


def apply_verified_fix(
    repository_root: str | None,
    target_file: str | None,
    original_code: str,
    original_content_hash: str | None,
    fix: CodeFix | None,
    verification: FixVerificationResult | None,
    fix_validation_errors: list[str],
    *,
    enabled: bool,
    auto_fix: bool,
) -> ApplyFixResult:
    """Atomically apply one verified candidate if the target is still current."""

    if not enabled:
        return ApplyFixResult(status="not_requested")
    if not auto_fix:
        return _precondition_error("Applying a fix requires auto-fix mode.")
    if fix is None:
        return _precondition_error("No candidate fix is available to apply.")
    if verification is None or verification.status != "verified":
        return _precondition_error(
            "Only a candidate with verified FixVerificationResult can be applied."
        )
    if fix_validation_errors:
        return _precondition_error(
            "Candidate validation errors prevent application."
        )
    if fix.updated_code == original_code:
        return _precondition_error(
            "Candidate source is unchanged from the reviewed source."
        )
    if repository_root is None or target_file is None:
        return _precondition_error(
            "Applying a fix requires a repository root and target file."
        )
    if original_content_hash is None:
        return _precondition_error(
            "Original target fingerprint is unavailable; apply was refused."
        )

    try:
        _, target, relative_target = _resolve_application_target(
            repository_root,
            target_file,
        )
    except ApplicationTargetError as error:
        return _precondition_error(str(error))

    try:
        current_content = target.read_bytes()
    except OSError as error:
        return _precondition_error(f"Apply target could not be read: {error}.")

    if content_fingerprint(current_content) != original_content_hash:
        return ApplyFixResult(
            status="stale",
            target_file=relative_target,
            warnings=[
                (
                    "Target file changed after review began; verified fix was "
                    "not applied."
                )
            ],
        )

    try:
        intended_content, current_text = _candidate_bytes(
            fix.updated_code,
            current_content,
        )
    except ValueError as error:
        return ApplyFixResult(
            status="error",
            target_file=relative_target,
            warnings=[str(error)],
        )
    if _normalized_newlines(current_text) != _normalized_newlines(original_code):
        return ApplyFixResult(
            status="stale",
            target_file=relative_target,
            warnings=[
                (
                    "Target content does not match the source reviewed by the "
                    "Agent; verified fix was not applied."
                )
            ],
        )

    temporary_path: Path | None = None
    try:
        original_mode = stat.S_IMODE(target.stat().st_mode)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.tmp-",
            dir=target.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(intended_content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.chmod(temporary_path, original_mode)
        try:
            _, latest_target, _ = _resolve_application_target(
                repository_root,
                target_file,
            )
        except ApplicationTargetError as error:
            return ApplyFixResult(
                status="error",
                target_file=relative_target,
                warnings=[
                    f"Apply target changed during atomic preparation: {error}"
                ],
            )
        if latest_target != target or latest_target.read_bytes() != current_content:
            return ApplyFixResult(
                status="stale",
                target_file=relative_target,
                warnings=[
                    (
                        "Target file changed while the atomic replacement was "
                        "being prepared; verified fix was not applied."
                    )
                ],
            )
        os.replace(temporary_path, target)
        temporary_path = None
    except OSError as error:
        return ApplyFixResult(
            status="error",
            target_file=relative_target,
            warnings=[f"Atomic target replacement failed: {error}."],
        )
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    try:
        written_content = target.read_bytes()
    except OSError as error:
        return ApplyFixResult(
            status="error",
            target_file=relative_target,
            warnings=[f"Applied target could not be confirmed: {error}."],
        )
    if written_content != intended_content:
        return ApplyFixResult(
            status="error",
            target_file=relative_target,
            warnings=[
                (
                    "Post-write confirmation did not match the intended "
                    "candidate bytes; no automatic rollback was attempted."
                )
            ],
        )

    return ApplyFixResult(
        status="applied",
        target_file=relative_target,
        bytes_written=len(written_content),
    )
