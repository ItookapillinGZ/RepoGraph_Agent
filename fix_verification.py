"""Bounded verification of candidate fixes in temporary repository copies.

The temporary copy is not a security sandbox. Targeted tests still execute
repository Python code on the local machine and may access local resources.
"""

import ast
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from fix_context import normalize_target_path
from repository_context import RepoContext
from sandbox.policy import SandboxPolicy
from static_analysis import (
    StaticAnalysisResult,
    ToolFinding,
    analyze_code,
    is_blocking_static_finding,
)
from temporary_workspace import (
    MAX_WORKSPACE_BYTES,
    MAX_WORKSPACE_FILES,
    WorkspaceCopyError,
    WorkspaceCopyResult,
    copy_repository_bounded,
    is_within,
)
from test_execution import TestRunResult, execute_targeted_tests

__all__ = [
    "WorkspaceCopyError",
    "WorkspaceCopyResult",
    "copy_repository_bounded",
    "verify_candidate_fix",
]


class FixVerificationResult(BaseModel):
    """Structured evidence from verifying one candidate source replacement."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["verified", "failed", "error", "not_verified"]
    static_analysis: StaticAnalysisResult = Field(
        default_factory=StaticAnalysisResult
    )
    test_result: TestRunResult = Field(
        default_factory=lambda: TestRunResult(status="not_run")
    )
    patch: str = ""
    warnings: list[str] = Field(default_factory=list)


def _add_warning(warnings: list[str], warning: str) -> None:
    if warning not in warnings:
        warnings.append(warning)


def _important_static_finding(finding: ToolFinding) -> bool:
    return is_blocking_static_finding(finding)


def _verification_status(
    static_analysis: StaticAnalysisResult,
    test_result: TestRunResult,
    warnings: list[str],
) -> Literal["verified", "failed", "not_verified"]:
    important_findings = [
        finding
        for finding in static_analysis.findings
        if _important_static_finding(finding)
    ]
    if important_findings:
        identifiers = ", ".join(
            f"{finding.tool}:{finding.rule_id}" for finding in important_findings
        )
        _add_warning(
            warnings,
            "Candidate has high/critical bug or security findings: "
            f"{identifiers}.",
        )
    for tool_error in static_analysis.tool_errors:
        _add_warning(warnings, f"Candidate static analysis error: {tool_error}")

    if test_result.status == "failed":
        return "failed"
    if test_result.status != "passed":
        return "not_verified"
    if important_findings:
        return "failed"
    if static_analysis.tool_errors:
        return "not_verified"
    return "verified"


def verify_candidate_fix(
    repository_root: str | None,
    repository_context: RepoContext,
    target_file: str | None,
    updated_code: str,
    patch: str,
    *,
    language: str = "python",
    max_workspace_files: int = MAX_WORKSPACE_FILES,
    max_workspace_bytes: int = MAX_WORKSPACE_BYTES,
    sandbox_policy: SandboxPolicy | None = None,
) -> FixVerificationResult:
    """Verify candidate code only after replacing the temporary-copy target."""

    if repository_root is None:
        return FixVerificationResult(
            status="error",
            patch=patch,
            warnings=["Candidate verification requires a repository root."],
        )
    if target_file is None:
        return FixVerificationResult(
            status="error",
            patch=patch,
            warnings=["Candidate verification requires a target file."],
        )
    if language.casefold() != "python":
        return FixVerificationResult(
            status="error",
            patch=patch,
            warnings=["Candidate verification currently supports Python only."],
        )

    try:
        ast.parse(updated_code)
    except SyntaxError as error:
        return FixVerificationResult(
            status="error",
            patch=patch,
            warnings=[f"Candidate Python source could not be parsed: {error.msg}."],
        )

    try:
        normalized_target = normalize_target_path(target_file)
    except ValueError as error:
        return FixVerificationResult(
            status="error",
            patch=patch,
            warnings=[str(error)],
        )

    try:
        with tempfile.TemporaryDirectory(
            prefix="code-review-fix-"
        ) as temporary_directory:
            temporary_root = Path(temporary_directory) / "repository"
            copy_result = copy_repository_bounded(
                repository_root,
                temporary_root,
                max_files=max_workspace_files,
                max_bytes=max_workspace_bytes,
            )

            resolved_temporary_root = temporary_root.resolve(strict=True)
            candidate_path = (
                resolved_temporary_root / Path(normalized_target)
            ).resolve(strict=False)
            if not is_within(candidate_path, resolved_temporary_root):
                return FixVerificationResult(
                    status="error",
                    patch=patch,
                    warnings=["Candidate target resolves outside temporary workspace."],
                )
            if not candidate_path.is_file() or candidate_path.is_symlink():
                return FixVerificationResult(
                    status="error",
                    patch=patch,
                    warnings=[
                        (
                            "Candidate target was not copied as a regular file "
                            "into the temporary workspace."
                        )
                    ],
                )

            candidate_path.write_bytes(updated_code.encode("utf-8"))
            candidate_static_analysis = analyze_code(updated_code, language)
            test_arguments: dict[str, object] = {"enabled": True}
            if sandbox_policy is not None:
                test_arguments["sandbox_policy"] = sandbox_policy
            candidate_test_result = execute_targeted_tests(
                str(resolved_temporary_root),
                repository_context,
                language,
                **test_arguments,
            )
            warnings = list(copy_result.warnings)
            status = _verification_status(
                candidate_static_analysis,
                candidate_test_result,
                warnings,
            )
            return FixVerificationResult(
                status=status,
                static_analysis=candidate_static_analysis,
                test_result=candidate_test_result,
                patch=patch,
                warnings=warnings,
            )
    except WorkspaceCopyError as error:
        return FixVerificationResult(
            status="error",
            patch=patch,
            warnings=[*error.warnings, str(error)],
        )
    except (OSError, UnicodeError) as error:
        return FixVerificationResult(
            status="error",
            patch=patch,
            warnings=[f"Temporary candidate verification failed: {error}."],
        )
