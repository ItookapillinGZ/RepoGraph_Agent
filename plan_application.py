"""Human-approved transactional application of verified multi-file candidates.

This module is a deterministic write boundary. It performs no planning, LLM,
test, shell, Git, GitHub, or temporary-candidate-workspace operations.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from engineering_plan import EngineeringPlan, _relative_parts, _validate_repository_root
from plan_execution import (
    MultiFileCandidate,
    PlanExecutionResult,
    _one_file_diff,
    validate_candidate_diff,
    validate_execution_plan,
    validate_multi_file_candidate,
)
from review_models import OverallRating
from source_file_encoding import (
    candidate_bytes_for_add,
    candidate_bytes_for_modify,
    decode_python_source,
)

MAX_APPLICATION_BUNDLE_CHARS = 500_000
_SHA256_HEX_LENGTH = 64
_ADD_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR


class PlanApplicationBundle(BaseModel):
    """Persistent approval artifact for one exact verified candidate."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    plan: EngineeringPlan
    candidate: MultiFileCandidate
    diff_text: str
    expected_original_hashes: dict[str, str | None]
    expected_candidate_hashes: dict[str, str | None]
    approval_digest: str = Field(
        min_length=_SHA256_HEX_LENGTH,
        max_length=_SHA256_HEX_LENGTH,
        pattern=r"^[0-9a-f]{64}$",
    )


class PlanApplicationResult(BaseModel):
    """Stable public outcome from the multi-file write boundary."""

    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "not_requested",
        "applied",
        "stale",
        "rolled_back",
        "error",
    ]
    applied_files: list[str] = Field(default_factory=list)
    rolled_back_files: list[str] = Field(default_factory=list)
    approval_digest: str | None = None
    warnings: list[str] = Field(default_factory=list)
    failure_reason: str | None = None


class PlanApplicationError(ValueError):
    """Raised when a bundle cannot be built, loaded, or saved safely."""


class _StaleApplicationError(RuntimeError):
    pass


class _PostApplyError(RuntimeError):
    pass


@dataclass
class _ApplyJournalEntry:
    repository_root: Path
    relative_path: str
    action: Literal["modify", "add", "delete"]
    target: Path
    original_bytes: bytes | None
    candidate_bytes: bytes | None
    original_mode: int | None
    expected_candidate_hash: str | None
    staged_path: Path | None = None
    backup_path: Path | None = None
    original_moved: bool = False
    committed: bool = False


def _content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _normalized_path(path: str) -> str:
    return "/".join(_relative_parts(path))


def _canonical_bundle_payload(
    bundle: PlanApplicationBundle | dict[str, object],
) -> dict[str, object]:
    if isinstance(bundle, PlanApplicationBundle):
        payload = bundle.model_dump(mode="json", exclude={"approval_digest"})
    else:
        payload = dict(bundle)
        payload.pop("approval_digest", None)
    return payload


def calculate_plan_application_digest(
    bundle: PlanApplicationBundle | dict[str, object],
) -> str:
    """Return the artifact-integrity identity for a bundle payload.

    This digest is not a signature and does not authenticate a human identity.
    """

    canonical = json.dumps(
        _canonical_bundle_payload(bundle),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return _content_hash(canonical.encode("utf-8"))


def _eligibility_errors(result: PlanExecutionResult) -> list[str]:
    errors: list[str] = []
    if result.status != "verified":
        errors.append("PlanExecutionResult status must be verified.")
    if result.verification is None or result.verification.status != "verified":
        errors.append("Plan execution verification must be present and verified.")
    if result.change_set_review is None:
        errors.append("A ChangeSetReview is required.")
    elif result.change_set_review.overall_rating != OverallRating.GOOD:
        errors.append("ChangeSetReview overall rating must be GOOD.")
    if result.candidate is None:
        errors.append("A MultiFileCandidate is required.")
    if result.validation_errors:
        errors.append("Candidate validation errors prevent bundle creation.")
    return errors


def _raise_validation_errors(label: str, errors: list[str]) -> None:
    if errors:
        raise PlanApplicationError(f"{label}: {'; '.join(errors)}")


def _candidate_change_map(candidate: MultiFileCandidate):
    return {_normalized_path(change.path): change for change in candidate.files}


def _revalidate_bundle(bundle: PlanApplicationBundle) -> PlanApplicationBundle:
    payload = (
        bundle.model_dump(mode="python", warnings="none")
        if isinstance(bundle, PlanApplicationBundle)
        else bundle
    )
    return PlanApplicationBundle.model_validate(payload)


def _expected_diff(
    root: Path,
    plan: EngineeringPlan,
    candidate: MultiFileCandidate,
) -> str:
    original_bytes_by_path: dict[str, bytes | None] = {}
    for planned in plan.files:
        normalized = _normalized_path(planned.path)
        original_bytes_by_path[normalized] = (
            None
            if planned.action == "add"
            else root.joinpath(*_relative_parts(normalized)).read_bytes()
        )
    return _expected_diff_from_original_bytes(
        plan,
        candidate,
        original_bytes_by_path,
    )


def _expected_diff_from_original_bytes(
    plan: EngineeringPlan,
    candidate: MultiFileCandidate,
    original_bytes_by_path: dict[str, bytes | None],
) -> str:
    candidate_by_path = _candidate_change_map(candidate)
    chunks: list[str] = []
    for planned in sorted(plan.files, key=lambda item: _normalized_path(item.path)):
        normalized = _normalized_path(planned.path)
        change = candidate_by_path[normalized]
        if change.action == "add":
            original = ""
        else:
            original_bytes = original_bytes_by_path.get(normalized)
            if original_bytes is None:
                raise PlanApplicationError(
                    f"Original bytes are missing for {normalized}."
                )
            original = decode_python_source(original_bytes)
        candidate_source = "" if change.action == "delete" else change.content
        if candidate_source is None:
            raise PlanApplicationError(
                f"Candidate content is missing for {normalized}."
            )
        chunks.append(
            _one_file_diff(normalized, change.action, original, candidate_source)
        )
    return "".join(chunks)


def validate_plan_application_diff_from_original_bytes(
    bundle: PlanApplicationBundle,
    original_bytes_by_path: dict[str, bytes | None],
) -> None:
    """Verify exact bundle diff text against independently trusted originals."""

    expected_paths = {
        _normalized_path(planned.path) for planned in bundle.plan.files
    }
    if set(original_bytes_by_path) != expected_paths:
        raise PlanApplicationError(
            "Original-byte paths do not exactly match the EngineeringPlan."
        )
    try:
        expected = _expected_diff_from_original_bytes(
            bundle.plan,
            bundle.candidate,
            original_bytes_by_path,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise PlanApplicationError(
            f"Candidate diff could not be reconstructed: {error}"
        ) from error
    if bundle.diff_text != expected:
        raise PlanApplicationError(
            "Candidate diff text does not exactly match the plan and candidate."
        )


def _validate_exact_diff(
    root: Path,
    plan: EngineeringPlan,
    candidate: MultiFileCandidate,
    diff_text: str,
) -> None:
    _raise_validation_errors(
        "Candidate diff validation failed",
        validate_candidate_diff(diff_text, candidate, plan),
    )
    try:
        expected = _expected_diff(root, plan, candidate)
    except (OSError, UnicodeError, ValueError) as error:
        raise PlanApplicationError(
            f"Candidate diff could not be reconstructed: {error}"
        ) from error
    if diff_text != expected:
        raise PlanApplicationError(
            "Candidate diff text does not exactly match the plan and candidate."
        )


def candidate_bytes_for_plan_change(
    action: str,
    content: str | None,
    original_bytes: bytes | None,
) -> bytes | None:
    """Return the exact filesystem bytes authorized for one bundle change."""

    if action == "delete":
        return None
    if content is None:
        raise PlanApplicationError("Candidate content is missing.")
    try:
        if action == "add":
            return candidate_bytes_for_add(content)
        if original_bytes is None:
            raise PlanApplicationError("Original bytes are missing for a modification.")
        encoded, _ = candidate_bytes_for_modify(content, original_bytes)
        return encoded
    except ValueError as error:
        raise PlanApplicationError(str(error)) from error


def build_plan_application_bundle(
    repository_root: str,
    execution_result: PlanExecutionResult,
) -> PlanApplicationBundle:
    """Build an independent approval artifact from a fully verified preview."""

    try:
        result_payload = (
            execution_result.model_dump(mode="python", warnings="none")
            if isinstance(execution_result, PlanExecutionResult)
            else execution_result
        )
        result = PlanExecutionResult.model_validate(result_payload)
    except ValidationError as error:
        raise PlanApplicationError(
            f"Plan execution result schema validation failed: {error}"
        ) from error
    _raise_validation_errors(
        "Plan execution result is not eligible for application",
        _eligibility_errors(result),
    )
    root = _validate_repository_root(repository_root)
    candidate = result.candidate
    if candidate is None:
        raise PlanApplicationError("A MultiFileCandidate is required.")

    _raise_validation_errors(
        "EngineeringPlan validation failed",
        validate_execution_plan(result.plan, str(root)),
    )
    _raise_validation_errors(
        "MultiFileCandidate validation failed",
        validate_multi_file_candidate(candidate, result.plan, str(root)),
    )
    _validate_exact_diff(root, result.plan, candidate, result.diff_text)

    expected_original_hashes: dict[str, str | None] = {}
    expected_candidate_hashes: dict[str, str | None] = {}
    candidate_by_path = _candidate_change_map(candidate)
    for planned in sorted(result.plan.files, key=lambda item: _normalized_path(item.path)):
        normalized = _normalized_path(planned.path)
        target = root.joinpath(*_relative_parts(normalized))
        original = None if planned.action == "add" else target.read_bytes()
        expected_original_hashes[normalized] = (
            None if original is None else _content_hash(original)
        )
        encoded = candidate_bytes_for_plan_change(
            planned.action,
            candidate_by_path[normalized].content,
            original,
        )
        expected_candidate_hashes[normalized] = (
            None if encoded is None else _content_hash(encoded)
        )

    payload: dict[str, object] = {
        "schema_version": 1,
        "plan": result.plan.model_dump(mode="json"),
        "candidate": candidate.model_dump(mode="json"),
        "diff_text": result.diff_text,
        "expected_original_hashes": expected_original_hashes,
        "expected_candidate_hashes": expected_candidate_hashes,
    }
    payload["approval_digest"] = calculate_plan_application_digest(payload)
    return PlanApplicationBundle.model_validate(payload)


def _is_hash(value: str | None) -> bool:
    if value is None or len(value) != _SHA256_HEX_LENGTH:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _assert_no_symlink(root: Path, parts: tuple[str, ...]) -> None:
    current = root
    for part in parts:
        current /= part
        if current.is_symlink():
            raise PlanApplicationError(
                "Symbolic links are not allowed in application paths."
            )


def _validate_bundle_maps(bundle: PlanApplicationBundle) -> list[str]:
    errors: list[str] = []
    plan_paths = [_normalized_path(change.path) for change in bundle.plan.files]
    candidate_paths = [
        _normalized_path(change.path) for change in bundle.candidate.files
    ]
    expected_keys = set(plan_paths)
    if len(plan_paths) != len({path.casefold() for path in plan_paths}):
        errors.append("EngineeringPlan contains duplicate normalized paths.")
    if len(candidate_paths) != len({path.casefold() for path in candidate_paths}):
        errors.append("MultiFileCandidate contains duplicate normalized paths.")
    if set(bundle.expected_original_hashes) != expected_keys:
        errors.append("Original-hash paths do not exactly match the EngineeringPlan.")
    if set(bundle.expected_candidate_hashes) != expected_keys:
        errors.append("Candidate-hash paths do not exactly match the EngineeringPlan.")

    actions = {
        _normalized_path(change.path): change.action for change in bundle.plan.files
    }
    for path in sorted(expected_keys):
        action = actions[path]
        original_hash = bundle.expected_original_hashes.get(path)
        candidate_hash = bundle.expected_candidate_hashes.get(path)
        if action == "add":
            if original_hash is not None:
                errors.append(f"Added path must have a null original hash: {path}.")
        elif not _is_hash(original_hash):
            errors.append(f"Existing path has an invalid original hash: {path}.")
        if action == "delete":
            if candidate_hash is not None:
                errors.append(f"Deleted path must have a null candidate hash: {path}.")
        elif not _is_hash(candidate_hash):
            errors.append(f"Written path has an invalid candidate hash: {path}.")
    return errors


def validate_plan_application_bundle(
    repository_root: str,
    bundle: PlanApplicationBundle,
    *,
    repository_state: Literal["original", "applied"] = "original",
) -> PlanApplicationBundle:
    """Revalidate one bundle as the deterministic G4/G5 authorization artifact.

    The original state validates a bundle before G4 application, including exact
    diff reconstruction from the current repository. The applied state validates
    the same artifact after G4 application, when added paths exist and deleted
    paths no longer do. The latter still validates diff metadata, while exact
    approved bytes are checked by the delivery boundary against bundle hashes.
    """

    try:
        validated = _revalidate_bundle(bundle)
    except (TypeError, ValueError, ValidationError) as error:
        raise PlanApplicationError(
            "Application bundle schema validation failed."
        ) from error
    if calculate_plan_application_digest(validated) != validated.approval_digest:
        raise PlanApplicationError(
            "Application bundle approval digest does not match."
        )
    if repository_state not in {"original", "applied"}:
        raise PlanApplicationError(
            "repository_state must be 'original' or 'applied'."
        )

    root = _validate_repository_root(repository_root)
    _raise_validation_errors(
        "Application bundle hash-map validation failed",
        _validate_bundle_maps(validated),
    )
    _raise_validation_errors(
        "EngineeringPlan validation failed",
        validate_execution_plan(
            validated.plan,
            str(root),
            repository_state=repository_state,
        ),
    )
    _raise_validation_errors(
        "MultiFileCandidate validation failed",
        validate_multi_file_candidate(
            validated.candidate,
            validated.plan,
            str(root),
            repository_state=repository_state,
        ),
    )
    if repository_state == "original":
        _validate_exact_diff(
            root,
            validated.plan,
            validated.candidate,
            validated.diff_text,
        )
    else:
        _raise_validation_errors(
            "Candidate diff validation failed",
            validate_candidate_diff(
                validated.diff_text,
                validated.candidate,
                validated.plan,
            ),
        )
    return validated


def _preflight_bundle(
    root: Path,
    bundle: PlanApplicationBundle,
) -> list[_ApplyJournalEntry]:
    _raise_validation_errors(
        "Application bundle hash-map validation failed",
        _validate_bundle_maps(bundle),
    )

    candidate_by_path = _candidate_change_map(bundle.candidate)
    entries: list[_ApplyJournalEntry] = []
    for planned in sorted(bundle.plan.files, key=lambda item: _normalized_path(item.path)):
        normalized = _normalized_path(planned.path)
        parts = _relative_parts(normalized)
        if Path(normalized).suffix.casefold() != ".py":
            raise PlanApplicationError(
                f"Application supports only Python paths: {normalized}."
            )
        _assert_no_symlink(root, parts)
        target = root.joinpath(*parts)
        if planned.action == "add":
            if target.exists() or target.is_symlink():
                raise _StaleApplicationError(
                    f"Added path now exists: {normalized}."
                )
            parent = target.parent.resolve(strict=True)
            if not parent.is_dir() or root not in (parent, *parent.parents):
                raise PlanApplicationError(
                    f"Added path parent is outside the repository: {normalized}."
                )
            original = None
            original_mode = None
        else:
            if not target.exists():
                raise _StaleApplicationError(
                    f"Existing path is now missing: {normalized}."
                )
            if target.is_symlink() or not target.is_file():
                raise PlanApplicationError(
                    f"Application target is not a regular non-symlink file: {normalized}."
                )
            original = target.read_bytes()
            original_mode = stat.S_IMODE(target.stat().st_mode)
            if _content_hash(original) != bundle.expected_original_hashes[normalized]:
                raise _StaleApplicationError(
                    f"Repository bytes changed after bundle creation: {normalized}."
                )

        candidate_change = candidate_by_path.get(normalized)
        if candidate_change is None:
            raise PlanApplicationError(
                f"Candidate is missing planned path: {normalized}."
            )
        encoded = candidate_bytes_for_plan_change(
            planned.action,
            candidate_change.content,
            original,
        )
        expected_candidate_hash = bundle.expected_candidate_hashes[normalized]
        actual_candidate_hash = None if encoded is None else _content_hash(encoded)
        if actual_candidate_hash != expected_candidate_hash:
            raise PlanApplicationError(
                f"Candidate bytes do not match the approved hash: {normalized}."
            )
        entries.append(
            _ApplyJournalEntry(
                repository_root=root,
                relative_path=normalized,
                action=planned.action,
                target=target,
                original_bytes=original,
                candidate_bytes=encoded,
                original_mode=original_mode,
                expected_candidate_hash=expected_candidate_hash,
            )
        )

    validate_plan_application_bundle(str(root), bundle)
    return entries


def _write_internal_file(
    target: Path,
    marker: str,
    content: bytes,
    mode: int,
) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.plan-{marker}-",
        dir=target.parent,
    )
    internal_path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(internal_path, mode)
        return internal_path
    except Exception:
        try:
            internal_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _stage_entries(entries: list[_ApplyJournalEntry]) -> None:
    for entry in entries:
        if entry.action in {"modify", "add"}:
            if entry.candidate_bytes is None:
                raise PlanApplicationError("Candidate bytes are missing during staging.")
            mode = entry.original_mode if entry.original_mode is not None else _ADD_FILE_MODE
            entry.staged_path = _write_internal_file(
                entry.target,
                "stage",
                entry.candidate_bytes,
                mode,
            )
    for entry in entries:
        if entry.action in {"modify", "delete"}:
            if entry.original_bytes is None or entry.original_mode is None:
                raise PlanApplicationError("Original bytes are missing during backup.")
            entry.backup_path = _write_internal_file(
                entry.target,
                "backup",
                entry.original_bytes,
                entry.original_mode,
            )


def _assert_entry_current(entry: _ApplyJournalEntry) -> None:
    _assert_no_symlink(
        entry.repository_root,
        _relative_parts(entry.relative_path),
    )
    if entry.target.is_symlink():
        raise _StaleApplicationError(
            f"Application path became a symbolic link: {entry.relative_path}."
        )
    if entry.action == "add":
        if entry.target.exists():
            raise _StaleApplicationError(
                f"Added path appeared during application: {entry.relative_path}."
            )
        return
    if not entry.target.is_file():
        raise _StaleApplicationError(
            f"Application target disappeared: {entry.relative_path}."
        )
    if entry.target.read_bytes() != entry.original_bytes:
        raise _StaleApplicationError(
            f"Application target changed during preparation: {entry.relative_path}."
        )


def _commit_entry(entry: _ApplyJournalEntry) -> None:
    if entry.action == "add":
        if entry.staged_path is None:
            raise PlanApplicationError("Staged add content is missing.")
        try:
            os.replace(entry.staged_path, entry.target)
        except OSError:
            if (
                entry.target.is_file()
                and not entry.target.is_symlink()
                and _content_hash(entry.target.read_bytes())
                == entry.expected_candidate_hash
            ):
                entry.staged_path = None
                entry.committed = True
            raise
        entry.staged_path = None
        entry.committed = True
        return

    if entry.backup_path is None:
        raise PlanApplicationError("Rollback backup is missing.")
    try:
        os.replace(entry.target, entry.backup_path)
    except OSError:
        if (
            not entry.target.exists()
            and entry.backup_path.is_file()
            and entry.backup_path.read_bytes() == entry.original_bytes
        ):
            entry.original_moved = True
        raise
    entry.original_moved = True
    if entry.action == "delete":
        entry.committed = True
        return
    if entry.staged_path is None:
        raise PlanApplicationError("Staged modify content is missing.")
    try:
        os.replace(entry.staged_path, entry.target)
    except OSError:
        if (
            entry.target.is_file()
            and not entry.target.is_symlink()
            and _content_hash(entry.target.read_bytes())
            == entry.expected_candidate_hash
        ):
            entry.staged_path = None
            entry.committed = True
        raise
    entry.staged_path = None
    entry.committed = True


def _post_apply_errors(entries: list[_ApplyJournalEntry]) -> list[str]:
    errors: list[str] = []
    for entry in entries:
        if entry.action == "delete":
            if entry.target.exists() or entry.target.is_symlink():
                errors.append(f"Deleted path still exists: {entry.relative_path}.")
            continue
        if not entry.target.is_file() or entry.target.is_symlink():
            errors.append(
                f"Applied path is not a regular file: {entry.relative_path}."
            )
            continue
        try:
            actual_hash = _content_hash(entry.target.read_bytes())
        except OSError:
            errors.append(f"Applied path could not be read: {entry.relative_path}.")
            continue
        if actual_hash != entry.expected_candidate_hash:
            errors.append(
                f"Applied bytes do not match the approved hash: {entry.relative_path}."
            )
    return errors


def _rollback(entries: list[_ApplyJournalEntry]) -> tuple[list[str], list[str]]:
    rolled_back: list[str] = []
    errors: list[str] = []
    for entry in reversed(entries):
        try:
            if entry.action == "add" and entry.committed:
                if (
                    not entry.target.is_file()
                    or entry.target.is_symlink()
                    or _content_hash(entry.target.read_bytes())
                    != entry.expected_candidate_hash
                ):
                    raise OSError(
                        "Transaction-added path no longer contains transaction bytes."
                    )
                entry.target.unlink()
                entry.committed = False
                rolled_back.append(entry.relative_path)
            elif entry.original_moved:
                if entry.backup_path is None or not entry.backup_path.is_file():
                    raise OSError("Rollback backup is unavailable.")
                os.replace(entry.backup_path, entry.target)
                entry.backup_path = None
                entry.original_moved = False
                entry.committed = False
                rolled_back.append(entry.relative_path)
        except OSError:
            errors.append(f"Rollback failed for {entry.relative_path}.")
    return sorted(rolled_back), errors


def _cleanup_entries(
    entries: list[_ApplyJournalEntry],
    warnings: list[str],
    *,
    preserve_needed_backups: bool = False,
) -> None:
    cleanup_failed = False
    for entry in entries:
        for attribute in ("staged_path", "backup_path"):
            path = getattr(entry, attribute)
            if path is None:
                continue
            if (
                attribute == "backup_path"
                and entry.original_moved
                and preserve_needed_backups
            ):
                cleanup_failed = True
                continue
            try:
                path.unlink(missing_ok=True)
                setattr(entry, attribute, None)
            except OSError:
                cleanup_failed = True
    if cleanup_failed:
        warnings.append(
            "One or more internal staging or backup artifacts could not be cleaned up."
        )


def apply_plan_application_bundle(
    repository_root: str,
    bundle: PlanApplicationBundle,
    *,
    approved: bool = False,
) -> PlanApplicationResult:
    """Apply one approved bundle using best-effort transactional rollback."""

    digest = getattr(bundle, "approval_digest", None)
    if not approved:
        return PlanApplicationResult(
            status="not_requested",
            approval_digest=digest,
        )
    try:
        validated_bundle = _revalidate_bundle(bundle)
    except (TypeError, ValueError, ValidationError):
        return PlanApplicationResult(
            status="error",
            approval_digest=digest,
            failure_reason="Application bundle schema validation failed.",
        )
    digest = validated_bundle.approval_digest
    if calculate_plan_application_digest(validated_bundle) != digest:
        return PlanApplicationResult(
            status="error",
            approval_digest=digest,
            failure_reason="Application bundle approval digest does not match.",
        )

    try:
        root = _validate_repository_root(repository_root)
        entries = _preflight_bundle(root, validated_bundle)
    except _StaleApplicationError as error:
        return PlanApplicationResult(
            status="stale",
            approval_digest=digest,
            failure_reason=str(error),
        )
    except (OSError, ValueError) as error:
        return PlanApplicationResult(
            status="error",
            approval_digest=digest,
            failure_reason=f"Application preflight failed: {error}",
        )

    warnings: list[str] = []
    try:
        _stage_entries(entries)
        for entry in entries:
            _assert_entry_current(entry)
        for entry in entries:
            _assert_entry_current(entry)
            _commit_entry(entry)
        post_errors = _post_apply_errors(entries)
        if post_errors:
            raise _PostApplyError("; ".join(post_errors))
    except Exception as error:  # noqa: BLE001 - rollback is the public boundary
        mutated = any(entry.committed or entry.original_moved for entry in entries)
        rolled_back, rollback_errors = _rollback(entries) if mutated else ([], [])
        _cleanup_entries(
            entries,
            warnings,
            preserve_needed_backups=bool(rollback_errors),
        )
        if rollback_errors:
            warnings.extend(rollback_errors)
            return PlanApplicationResult(
                status="error",
                rolled_back_files=rolled_back,
                approval_digest=digest,
                warnings=warnings,
                failure_reason="Application failed and rollback was incomplete.",
            )
        if mutated:
            reason = (
                "Post-apply verification failed; the transaction was rolled back."
                if isinstance(error, _PostApplyError)
                else "Application failed; the transaction was rolled back."
            )
            return PlanApplicationResult(
                status="rolled_back",
                rolled_back_files=rolled_back,
                approval_digest=digest,
                warnings=warnings,
                failure_reason=reason,
            )
        status: Literal["stale", "error"] = (
            "stale" if isinstance(error, _StaleApplicationError) else "error"
        )
        reason = (
            str(error)
            if isinstance(error, _StaleApplicationError)
            else "Application staging failed before repository mutation."
        )
        return PlanApplicationResult(
            status=status,
            approval_digest=digest,
            warnings=warnings,
            failure_reason=reason,
        )

    _cleanup_entries(entries, warnings)
    return PlanApplicationResult(
        status="applied",
        applied_files=[entry.relative_path for entry in entries],
        approval_digest=digest,
        warnings=warnings,
    )


def load_plan_application_bundle(path: str) -> PlanApplicationBundle:
    """Load one bounded UTF-8 bundle artifact."""

    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            serialized = handle.read(MAX_APPLICATION_BUNDLE_CHARS + 1)
    except (OSError, UnicodeError) as error:
        raise PlanApplicationError(
            f"Application bundle could not be read: {error}"
        ) from error
    if len(serialized) > MAX_APPLICATION_BUNDLE_CHARS:
        raise PlanApplicationError(
            "Application bundle exceeds "
            f"MAX_APPLICATION_BUNDLE_CHARS={MAX_APPLICATION_BUNDLE_CHARS}."
        )
    try:
        payload = json.loads(serialized)
        return PlanApplicationBundle.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as error:
        raise PlanApplicationError(
            f"Application bundle is not valid schema-version-1 JSON: {error}"
        ) from error


def save_plan_application_bundle(
    path: str,
    bundle: PlanApplicationBundle,
    *,
    repository_root: str,
) -> None:
    """Persist stable UTF-8 JSON outside the target repository."""

    try:
        validated = _revalidate_bundle(bundle)
    except (TypeError, ValueError, ValidationError) as error:
        raise PlanApplicationError(
            f"Application bundle schema validation failed: {error}"
        ) from error
    root = _validate_repository_root(repository_root)
    destination = Path(path)
    if destination.is_symlink():
        raise PlanApplicationError("Application bundle output must not be a symlink.")
    absolute_destination = Path(os.path.abspath(destination))
    if absolute_destination == root or root in absolute_destination.parents:
        raise PlanApplicationError(
            "Application bundle must be saved outside the target repository."
        )
    try:
        parent = destination.parent.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise PlanApplicationError(
            "Application bundle output parent must already exist."
        ) from error
    resolved_destination = parent / destination.name
    if resolved_destination == root or root in resolved_destination.parents:
        raise PlanApplicationError(
            "Application bundle must be saved outside the target repository."
        )

    serialized = (
        json.dumps(
            validated.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    if len(serialized) > MAX_APPLICATION_BUNDLE_CHARS:
        raise PlanApplicationError(
            "Serialized application bundle exceeds "
            f"MAX_APPLICATION_BUNDLE_CHARS={MAX_APPLICATION_BUNDLE_CHARS}."
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-",
        dir=parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, resolved_destination)
    except Exception as error:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise PlanApplicationError(
            "Application bundle could not be saved safely."
        ) from error


def render_plan_application_result(result: PlanApplicationResult) -> str:
    """Render a concise deterministic application outcome."""

    lines = [f"Status: {result.status}"]
    if result.approval_digest is not None:
        lines.append(f"Approval digest: {result.approval_digest}")
    if result.applied_files:
        lines.append("Applied files:")
        lines.extend(f"- {path}" for path in result.applied_files)
    if result.rolled_back_files:
        lines.append("Rolled-back files:")
        lines.extend(f"- {path}" for path in result.rolled_back_files)
    if result.failure_reason is not None:
        lines.append(f"Failure: {result.failure_reason}")
    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in result.warnings)
    return "\n".join(lines)
