"""Deterministic Ruff and Bandit evidence collection for Python source code."""

import json
import subprocess  # nosec B404
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

STATIC_ANALYSIS_TIMEOUT_SECONDS = 15


class ToolCategory(str, Enum):
    """Normalized category shared by static-analysis tools."""

    BUG = "bug"
    SECURITY = "security"
    STYLE = "style"
    PERFORMANCE = "performance"
    OTHER = "other"


class ToolSeverity(str, Enum):
    """Normalized severity shared by static-analysis tools."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToolFinding(BaseModel):
    """One normalized finding from Ruff or Bandit."""

    model_config = ConfigDict(extra="forbid")

    tool: str = Field(min_length=1)
    rule_id: str = Field(min_length=1)
    category: ToolCategory
    severity: ToolSeverity
    confidence: str | None = None
    message: str = Field(min_length=1)
    line_number: int | None = Field(default=None, ge=1)
    column: int | None = Field(default=None, ge=1)


class StaticAnalysisResult(BaseModel):
    """Structured evidence and execution status for all configured tools."""

    model_config = ConfigDict(extra="forbid")

    findings: list[ToolFinding] = Field(default_factory=list)
    tools_run: list[str] = Field(default_factory=list)
    tool_errors: list[str] = Field(default_factory=list)


def is_blocking_static_finding(finding: ToolFinding) -> bool:
    """Return the shared high/critical bug-or-security verification rule."""

    return finding.category in {
        ToolCategory.BUG,
        ToolCategory.SECURITY,
    } and finding.severity in {ToolSeverity.HIGH, ToolSeverity.CRITICAL}


def _ruff_classification(rule_id: str) -> tuple[ToolCategory, ToolSeverity]:
    """Map the small Ruff subset needed for evidence consistency checks."""

    if rule_id in {"E999", "F821", "F822", "F823"}:
        return ToolCategory.BUG, ToolSeverity.HIGH
    if rule_id.startswith(("E", "F")):
        return ToolCategory.STYLE, ToolSeverity.LOW
    return ToolCategory.OTHER, ToolSeverity.INFO


def _parse_ruff_json(stdout: str) -> list[ToolFinding]:
    payload = json.loads(stdout)
    if not isinstance(payload, list):
        raise TypeError("Ruff JSON output must be a list.")

    findings: list[ToolFinding] = []
    for item in payload:
        if not isinstance(item, dict):
            raise TypeError("Ruff finding must be an object.")

        rule_id = str(item["code"])
        location = item.get("location") or {}
        category, severity = _ruff_classification(rule_id)
        findings.append(
            ToolFinding(
                tool="ruff",
                rule_id=rule_id,
                category=category,
                severity=severity,
                message=str(item["message"]),
                line_number=location.get("row"),
                column=location.get("column"),
            )
        )
    return findings


def _bandit_severity(value: Any) -> ToolSeverity:
    normalized = str(value).casefold()
    try:
        return ToolSeverity(normalized)
    except ValueError:
        return ToolSeverity.INFO


def _parse_bandit_json(stdout: str) -> tuple[list[ToolFinding], list[str]]:
    payload = json.loads(stdout)
    if not isinstance(payload, dict):
        raise TypeError("Bandit JSON output must be an object.")

    raw_results = payload.get("results", [])
    raw_errors = payload.get("errors", [])
    if not isinstance(raw_results, list) or not isinstance(raw_errors, list):
        raise TypeError("Bandit results and errors must be lists.")

    findings: list[ToolFinding] = []
    for item in raw_results:
        if not isinstance(item, dict):
            raise TypeError("Bandit finding must be an object.")

        column = item.get("col_offset")
        findings.append(
            ToolFinding(
                tool="bandit",
                rule_id=str(item["test_id"]),
                category=ToolCategory.SECURITY,
                severity=_bandit_severity(item.get("issue_severity")),
                confidence=str(item.get("issue_confidence", "")).casefold() or None,
                message=str(item["issue_text"]),
                line_number=item.get("line_number"),
                column=int(column) + 1 if column is not None else None,
            )
        )

    errors = []
    for error in raw_errors:
        if isinstance(error, dict):
            reason = error.get("reason", "unknown analysis error")
        else:
            reason = error
        errors.append(f"bandit analysis error: {reason}")
    return findings, errors


def _failure_message(tool: str, returncode: int, stderr: str) -> str:
    detail = " ".join(stderr.strip().split())[:200]
    suffix = f": {detail}" if detail else ""
    return f"{tool} failed with exit code {returncode}{suffix}"


def _run_ruff(file_path: str) -> tuple[list[ToolFinding], str | None]:
    try:
        completed = subprocess.run(  # nosec B603
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--output-format",
                "json",
                "--no-cache",
                "--target-version",
                "py39",
                file_path,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=STATIC_ANALYSIS_TIMEOUT_SECONDS,
            cwd=str(Path(file_path).parent),
            shell=False,
            check=False,
        )
    except FileNotFoundError:
        return [], "ruff executable not found"
    except subprocess.TimeoutExpired:
        return [], f"ruff timed out after {STATIC_ANALYSIS_TIMEOUT_SECONDS} seconds"
    except OSError as error:
        return [], f"ruff could not start: {error}"

    if completed.returncode not in {0, 1}:
        return [], _failure_message("ruff", completed.returncode, completed.stderr)

    try:
        return _parse_ruff_json(completed.stdout), None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        return [], f"ruff returned malformed JSON: {error}"


def _run_bandit(
    file_path: str,
) -> tuple[list[ToolFinding], list[str]]:
    try:
        completed = subprocess.run(  # nosec B603
            [sys.executable, "-m", "bandit", "-f", "json", file_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=STATIC_ANALYSIS_TIMEOUT_SECONDS,
            cwd=str(Path(file_path).parent),
            shell=False,
            check=False,
        )
    except FileNotFoundError:
        return [], ["bandit executable not found"]
    except subprocess.TimeoutExpired:
        return [], [f"bandit timed out after {STATIC_ANALYSIS_TIMEOUT_SECONDS} seconds"]
    except OSError as error:
        return [], [f"bandit could not start: {error}"]

    if completed.returncode not in {0, 1}:
        return [], [_failure_message("bandit", completed.returncode, completed.stderr)]

    try:
        return _parse_bandit_json(completed.stdout)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        return [], [f"bandit returned malformed JSON: {error}"]


def analyze_code(code: str, language: str) -> StaticAnalysisResult:
    """Collect static evidence without executing the submitted source code."""

    if language.casefold() != "python":
        return StaticAnalysisResult()

    findings: list[ToolFinding] = []
    tools_run: list[str] = []
    tool_errors: list[str] = []

    with tempfile.TemporaryDirectory(prefix="code-review-static-") as temp_dir:
        file_path = Path(temp_dir) / "review_target.py"
        file_path.write_text(code, encoding="utf-8")

        ruff_findings, ruff_error = _run_ruff(str(file_path))
        findings.extend(ruff_findings)
        if ruff_error is None:
            tools_run.append("ruff")
        else:
            tool_errors.append(ruff_error)

        bandit_findings, bandit_errors = _run_bandit(str(file_path))
        findings.extend(bandit_findings)
        if not bandit_errors:
            tools_run.append("bandit")
        else:
            tool_errors.extend(bandit_errors)

    return StaticAnalysisResult(
        findings=findings,
        tools_run=tools_run,
        tool_errors=tool_errors,
    )
