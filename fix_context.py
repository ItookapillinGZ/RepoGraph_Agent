"""Structured candidate fixes and deterministic validation utilities."""

import ast
import difflib
from pathlib import PurePosixPath, PureWindowsPath

from pydantic import BaseModel, ConfigDict, Field

MAX_FIXED_CODE_CHARS = 250_000


class CodeFix(BaseModel):
    """One complete candidate replacement for the reviewed target file."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    addressed_findings: list[str] = Field(default_factory=list)
    updated_code: str = Field(min_length=1)


class FixValidationResult(BaseModel):
    """Deterministic validation result for one generated candidate."""

    model_config = ConfigDict(extra="forbid")

    errors: list[str] = Field(default_factory=list)
    patch: str = ""

    @property
    def is_valid(self) -> bool:
        """Return whether all candidate checks succeeded."""

        return not self.errors


def normalize_target_path(target_file: str) -> str:
    """Return a safe repository-relative POSIX target path."""

    if not target_file or not target_file.strip():
        raise ValueError("Target file path must not be empty.")

    windows_path = PureWindowsPath(target_file)
    normalized_input = target_file.replace("\\", "/")
    posix_path = PurePosixPath(normalized_input)
    if windows_path.is_absolute() or windows_path.drive or posix_path.is_absolute():
        raise ValueError("Target file path must be repository-relative.")
    if ".." in posix_path.parts:
        raise ValueError("Target file path must not contain parent traversal.")

    normalized = posix_path.as_posix()
    if normalized in {"", "."}:
        raise ValueError("Target file path must identify a file.")
    return normalized


def build_fix_patch(
    original_code: str,
    updated_code: str,
    target_file: str,
) -> str:
    """Build a deterministic unified diff without asking the LLM for syntax."""

    normalized_target = normalize_target_path(target_file)
    if original_code == updated_code:
        return ""

    diff_lines = difflib.unified_diff(
        original_code.splitlines(),
        updated_code.splitlines(),
        fromfile=f"a/{normalized_target}",
        tofile=f"b/{normalized_target}",
        lineterm="",
    )
    rendered = "\n".join(diff_lines)
    return f"{rendered}\n" if rendered else ""


def validate_candidate_fix(
    fix: CodeFix | None,
    original_code: str,
    finding_titles: list[str],
    target_file: str,
    *,
    language: str = "python",
    max_code_chars: int = MAX_FIXED_CODE_CHARS,
) -> FixValidationResult:
    """Validate one candidate before any repository copy or test execution."""

    if fix is None:
        return FixValidationResult(errors=["No candidate fix was generated."])

    errors: list[str] = []
    updated_code = fix.updated_code
    stripped_code = updated_code.strip()

    if not stripped_code:
        errors.append("Candidate updated_code must not be empty or whitespace.")
    if updated_code == original_code:
        errors.append("Candidate updated_code is unchanged from the original source.")
    if len(updated_code) > max_code_chars:
        errors.append(
            "Candidate updated_code exceeds "
            f"MAX_FIXED_CODE_CHARS={max_code_chars}."
        )
    if stripped_code.startswith("```") or stripped_code.endswith("```"):
        errors.append(
            "Candidate updated_code must be raw source without Markdown code fences."
        )

    if language.casefold() == "python" and stripped_code:
        try:
            ast.parse(updated_code)
        except SyntaxError as error:
            location = f" at line {error.lineno}" if error.lineno else ""
            errors.append(
                f"Candidate Python source has a SyntaxError{location}: {error.msg}."
            )

    allowed_titles = set(finding_titles)
    normalized_addressed: set[str] = set()
    for title in fix.addressed_findings:
        normalized_title = " ".join(title.casefold().split())
        if normalized_title in normalized_addressed:
            errors.append(f"Duplicate addressed finding title: {title!r}.")
        else:
            normalized_addressed.add(normalized_title)
        if title not in allowed_titles:
            errors.append(
                f"Addressed finding title is not present in the review: {title!r}."
            )

    patch = ""
    try:
        patch = build_fix_patch(original_code, updated_code, target_file)
    except ValueError as error:
        errors.append(str(error))
    if not patch:
        errors.append("Candidate patch must not be empty.")

    return FixValidationResult(errors=errors, patch=patch)
