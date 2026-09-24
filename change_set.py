"""Deterministic schemas, selection, and aggregation for change-set review."""

from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from diff_context import DiffContext
from review_models import CodeReview, OverallRating, Severity

MAX_CHANGESET_FILES = 10
MAX_CHANGESET_PYTHON_FILES = 5
MAX_CHANGESET_DIFF_CHARS = 50_000
MAX_CHANGESET_SUMMARY_FINDINGS = 20
MAX_CHANGESET_FINDING_CHARS = 500


class ChangeTarget(BaseModel):
    """One repository-relative file selected from a local change."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    change_kind: Literal["added", "modified", "deleted", "renamed"]
    old_path: str | None = None
    new_path: str | None = None


class FileReviewResult(BaseModel):
    """Review outcome for one selected Python target."""

    model_config = ConfigDict(extra="forbid")

    target: ChangeTarget
    status: Literal["reviewed", "skipped", "error"]
    review: CodeReview | None = None
    warnings: list[str] = Field(default_factory=list)


class ChangeSetReview(BaseModel):
    """One bounded review aggregated from independent file-level reviews."""

    model_config = ConfigDict(extra="forbid")

    overall_rating: OverallRating
    summary: str = Field(min_length=1)
    file_results: list[FileReviewResult]
    high_risk_files: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ChangeSetSummary(BaseModel):
    """Small structured LLM output synthesized from file-level evidence."""

    model_config = ConfigDict(extra="forbid")

    overall_rating: OverallRating
    summary: str = Field(min_length=1)
    high_risk_files: list[str] = Field(default_factory=list)


class ChangeTargetSelection(BaseModel):
    """Internal deterministic target-selection result."""

    model_config = ConfigDict(extra="forbid")

    targets: list[ChangeTarget] = Field(default_factory=list)
    binary_files: list[str] = Field(default_factory=list)
    target_warnings: dict[str, list[str]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def _change_kind(old_path: str | None, new_path: str | None) -> str:
    if old_path is None:
        return "added"
    if new_path is None:
        return "deleted"
    if old_path != new_path:
        return "renamed"
    return "modified"


def _normalized_path(path: str) -> str:
    return PurePosixPath(path.replace("\\", "/")).as_posix()


def _add_warning(warnings: list[str], warning: str) -> None:
    if warning not in warnings:
        warnings.append(warning)


def build_change_targets(
    diff_context: DiffContext,
    untracked_files: Iterable[str] = (),
) -> ChangeTargetSelection:
    """Select bounded Python targets in stable path order from parsed metadata."""

    records: dict[str, tuple[ChangeTarget, bool, list[str]]] = {}
    for changed_file in diff_context.changed_files:
        path = changed_file.new_path or changed_file.old_path
        if path is None:
            continue
        normalized = _normalized_path(path)
        records[normalized] = (
            ChangeTarget(
                path=normalized,
                change_kind=_change_kind(
                    changed_file.old_path,
                    changed_file.new_path,
                ),
                old_path=changed_file.old_path,
                new_path=changed_file.new_path,
            ),
            changed_file.is_binary,
            [],
        )

    for raw_path in untracked_files:
        normalized = _normalized_path(raw_path)
        if normalized in records:
            continue
        records[normalized] = (
            ChangeTarget(
                path=normalized,
                change_kind="added",
                new_path=normalized,
            ),
            False,
            [
                (
                    "Untracked file has no HEAD diff; reviewed as a new full-file "
                    "target."
                )
            ],
        )

    ordered_records = sorted(records.items(), key=lambda item: item[0])
    warnings = list(diff_context.warnings)
    if len(ordered_records) > MAX_CHANGESET_FILES:
        _add_warning(
            warnings,
            "Changed files truncated at "
            f"MAX_CHANGESET_FILES={MAX_CHANGESET_FILES}; "
            f"{len(ordered_records) - MAX_CHANGESET_FILES} files were omitted.",
        )

    non_python_count = sum(
        1 for path, _record in ordered_records if not path.casefold().endswith(".py")
    )
    if non_python_count:
        _add_warning(
            warnings,
            f"{non_python_count} changed non-Python files were not reviewed.",
        )

    bounded_records = ordered_records[:MAX_CHANGESET_FILES]
    python_records = [
        (path, record)
        for path, record in bounded_records
        if path.casefold().endswith(".py")
    ]
    if len(python_records) > MAX_CHANGESET_PYTHON_FILES:
        _add_warning(
            warnings,
            "Python targets truncated at "
            f"MAX_CHANGESET_PYTHON_FILES={MAX_CHANGESET_PYTHON_FILES}; "
            f"{len(python_records) - MAX_CHANGESET_PYTHON_FILES} Python files "
            "were omitted.",
        )

    targets: list[ChangeTarget] = []
    binary_files: list[str] = []
    target_warnings: dict[str, list[str]] = {}
    for path, (target, is_binary, file_warnings) in python_records[
        :MAX_CHANGESET_PYTHON_FILES
    ]:
        targets.append(target)
        if is_binary:
            binary_files.append(path)
        if file_warnings:
            target_warnings[path] = list(file_warnings)

    return ChangeTargetSelection(
        targets=targets,
        binary_files=binary_files,
        target_warnings=target_warnings,
        warnings=warnings,
    )


def high_risk_paths(file_results: list[FileReviewResult]) -> list[str]:
    """Return stable paths backed by critical ratings or high-risk findings."""

    paths = []
    for result in file_results:
        review = result.review
        if review is None:
            continue
        if review.overall_rating == OverallRating.CRITICAL_ISSUES or any(
            finding.severity in {Severity.HIGH, Severity.CRITICAL}
            for finding in review.findings
        ):
            paths.append(result.target.path)
    return sorted(set(paths))


def build_summary_evidence(
    file_results: list[FileReviewResult],
    warnings: list[str],
) -> str:
    """Render bounded summary evidence without including source or complete diffs."""

    lines = [
        "Change-set metadata and validated file-review evidence:",
        f"Selected Python targets: {len(file_results)}",
    ]
    if warnings:
        lines.append("Global warnings:")
        lines.extend(f"- {warning}" for warning in warnings)

    finding_count = 0
    for result in file_results:
        lines.extend(
            [
                "",
                f"File: {result.target.path}",
                f"Change kind: {result.target.change_kind}",
                f"Review status: {result.status}",
            ]
        )
        if result.review is not None:
            lines.append(
                f"File overall rating: {result.review.overall_rating.value}"
            )
            lines.append(f"File summary: {result.review.summary}")
            important = [
                finding
                for finding in result.review.findings
                if finding.severity in {Severity.HIGH, Severity.CRITICAL}
            ]
            for finding in important:
                if finding_count >= MAX_CHANGESET_SUMMARY_FINDINGS:
                    break
                description = finding.description[:MAX_CHANGESET_FINDING_CHARS]
                lines.append(
                    "Important finding: "
                    f"[{finding.severity.value}] {finding.title}: {description}"
                )
                finding_count += 1
        if result.warnings:
            lines.append("File warnings:")
            lines.extend(f"- {warning}" for warning in result.warnings)

    if finding_count >= MAX_CHANGESET_SUMMARY_FINDINGS:
        lines.append(
            "Important findings truncated at "
            f"MAX_CHANGESET_SUMMARY_FINDINGS={MAX_CHANGESET_SUMMARY_FINDINGS}."
        )
    return "\n".join(lines)


def validate_change_set_summary(
    summary: ChangeSetSummary,
    file_results: list[FileReviewResult],
) -> list[str]:
    """Enforce evidence-backed overall rating and high-risk file semantics."""

    errors: list[str] = []
    reviewed = [result for result in file_results if result.review is not None]
    expected_high_risk = high_risk_paths(file_results)

    if not reviewed and summary.overall_rating == OverallRating.GOOD:
        errors.append("A change-set with no reviewed files cannot be rated 'good'.")
    if expected_high_risk and summary.overall_rating == OverallRating.GOOD:
        errors.append(
            "Overall rating 'good' conflicts with critical file ratings or "
            "high/critical findings."
        )

    known_paths = {result.target.path for result in file_results}
    if len(summary.high_risk_files) != len(set(summary.high_risk_files)):
        errors.append("high_risk_files contains duplicate paths.")
    unknown_paths = sorted(set(summary.high_risk_files) - known_paths)
    if unknown_paths:
        errors.append(
            "high_risk_files contains paths absent from file-level evidence: "
            + ", ".join(unknown_paths)
            + "."
        )
    missing_paths = sorted(set(expected_high_risk) - set(summary.high_risk_files))
    if missing_paths:
        errors.append(
            "high_risk_files omits evidence-backed high-risk paths: "
            + ", ".join(missing_paths)
            + "."
        )
    if summary.high_risk_files != sorted(summary.high_risk_files):
        errors.append("high_risk_files must use stable path ordering.")
    return errors


def deterministic_fallback_summary(
    file_results: list[FileReviewResult],
) -> ChangeSetSummary:
    """Build a conservative result if summary generation exhausts retries."""

    reviewed = [result for result in file_results if result.review is not None]
    skipped = sum(result.status == "skipped" for result in file_results)
    errors = sum(result.status == "error" for result in file_results)
    ratings = {result.review.overall_rating for result in reviewed if result.review}
    if OverallRating.CRITICAL_ISSUES in ratings:
        rating = OverallRating.CRITICAL_ISSUES
    elif (
        not reviewed
        or skipped
        or errors
        or OverallRating.NEEDS_WORK in ratings
        or high_risk_paths(file_results)
    ):
        rating = OverallRating.NEEDS_WORK
    else:
        rating = OverallRating.GOOD

    return ChangeSetSummary(
        overall_rating=rating,
        summary=(
            f"Reviewed {len(reviewed)} Python files; {skipped} were skipped and "
            f"{errors} ended with errors. The overall assessment was derived "
            "deterministically because generated summary validation did not pass."
        ),
        high_risk_files=high_risk_paths(file_results),
    )


def aggregate_change_set_review(
    summary: ChangeSetSummary,
    file_results: list[FileReviewResult],
    warnings: list[str],
) -> ChangeSetReview:
    """Combine validated summary output and stable file-level results."""

    return ChangeSetReview(
        overall_rating=summary.overall_rating,
        summary=summary.summary,
        file_results=file_results,
        high_risk_files=summary.high_risk_files,
        warnings=warnings,
    )
