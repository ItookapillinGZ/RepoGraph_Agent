"""Trusted local evaluator with frozen provenance and bounded evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess  # nosec B404
import sys
from pathlib import Path

from evaluation.models import (
    EvaluationTask,
    EvaluatorEnvironmentFingerprint,
    EvaluatorOutcome,
    EvaluatorSpecification,
)
from sandbox.models import SandboxExecutionResult
from sandbox.policy import SandboxPolicy
from sandbox.runner import SandboxRunner

LOCAL_EVALUATOR_VERSION = "local-fixed-argv-v3"


class EvaluatorPreflightError(RuntimeError):
    """Raised before a campaign when its frozen evaluator cannot run."""


def _bounded(
    value: str | bytes | None, limit: int, name: str
) -> tuple[str, str | None]:
    if value is None:
        return "", None
    rendered = (
        value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    )
    if len(rendered) <= limit:
        return rendered, None
    return rendered[:limit], f"Evaluator {name} was truncated at {limit} characters."


def _is_pytest_command(command: list[str]) -> bool:
    lowered = [Path(part).name.casefold() for part in command]
    return "pytest" in lowered or any(part.casefold() == "pytest" for part in command)


def _resolve_python_command(command: list[str]) -> list[str]:
    """Resolve portable Python aliases once to the worker's exact interpreter."""

    aliases = {"py", "py.exe", "python", "python.exe", "python3", "python3.exe"}
    if command[0].casefold() not in aliases:
        return list(command)
    return [sys.executable, *command[1:]]


def _python_executable(command: list[str]) -> str | None:
    name = Path(command[0]).name.casefold()
    if name.startswith("python") or name in {"py", "py.exe"}:
        return command[0]
    return None


def _specification_digest(payload: dict[str, object]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def evaluator_specification_digest(
    specification: EvaluatorSpecification,
) -> str:
    """Recompute the provenance digest without trusting its stored digest."""

    validated = EvaluatorSpecification.model_validate(specification)
    return _specification_digest(
        validated.model_dump(mode="json", exclude={"digest"})
    )


def build_local_evaluator_specification(
    task: EvaluationTask,
    *,
    timeout_seconds: float,
) -> EvaluatorSpecification:
    if task.test_command is None:
        raise EvaluatorPreflightError(
            f"Evaluation task {task.id} does not define a test_command."
        )
    command = _resolve_python_command(task.test_command)
    payload: dict[str, object] = {
        "kind": "local",
        "command": command,
        "python_executable": _python_executable(command),
        "version": LOCAL_EVALUATOR_VERSION,
        "timeout_seconds": float(timeout_seconds),
    }
    return EvaluatorSpecification(
        **payload,
        digest=_specification_digest(payload),
    )


def evaluator_environment_fingerprint(
    specification: EvaluatorSpecification,
) -> EvaluatorEnvironmentFingerprint:
    pytest_version: str | None = None
    if _is_pytest_command(specification.command):
        try:
            pytest_version = importlib.metadata.version("pytest")
        except importlib.metadata.PackageNotFoundError:
            pytest_version = None
    return EvaluatorEnvironmentFingerprint(
        python_version=platform.python_version(),
        pytest_version=pytest_version,
        platform=platform.platform(),
        evaluator_digest=specification.digest,
    )


def preflight_local_evaluator(
    specification: EvaluatorSpecification,
    *,
    max_output_chars: int = 2_000,
) -> EvaluatorEnvironmentFingerprint:
    """Validate the frozen interpreter and pytest in the same environment."""

    specification = EvaluatorSpecification.model_validate(specification)
    if evaluator_specification_digest(specification) != specification.digest:
        raise EvaluatorPreflightError("Evaluator specification digest mismatch.")
    command = specification.command
    if _is_pytest_command(command):
        python = specification.python_executable
        if not python:
            raise EvaluatorPreflightError(
                "Pytest evaluator must have an explicit Python executable."
            )
        preflight_command = [python, "-m", "pytest", "--version"]
    else:
        preflight_command = [command[0], "--version"]
    try:
        completed = subprocess.run(  # nosec B603
            preflight_command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=min(specification.timeout_seconds, 60.0),
            check=False,
            shell=False,
            env=os.environ.copy(),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise EvaluatorPreflightError(
            f"Evaluator preflight could not execute: {error}"
        ) from error
    if completed.returncode != 0:
        output = (completed.stderr or completed.stdout)[:max_output_chars].strip()
        raise EvaluatorPreflightError(
            f"Evaluator preflight failed with exit code {completed.returncode}: {output}"
        )
    fingerprint = evaluator_environment_fingerprint(specification)
    if _is_pytest_command(command) and not fingerprint.pytest_version:
        raise EvaluatorPreflightError(
            "Evaluator preflight succeeded but pytest version is unavailable."
        )
    return fingerprint


def _pytest_failure(
    returncode: int,
    stdout: str,
    stderr: str,
) -> tuple[str, str, str]:
    output = f"{stdout}\n{stderr}".casefold()
    if "no module named pytest" in output:
        return (
            "error",
            "import_failure",
            "The evaluator Python environment does not provide pytest.",
        )
    if returncode == 1:
        return (
            "failed",
            "assertion_failure",
            "Task-specific pytest evaluator reported failing tests.",
        )
    if "importerror while importing test module" in output or (
        "modulenotfounderror" in output and "error collecting" in output
    ):
        return (
            "failed",
            "import_failure",
            "Candidate grading failed during pytest import.",
        )
    if "error collecting" in output or "errors during collection" in output:
        return (
            "failed",
            "collection_failure",
            "Candidate grading failed during pytest collection.",
        )
    if returncode == 4:
        return ("error", "usage_error", "Pytest reported a command usage error.")
    if returncode == 5:
        return ("error", "no_tests", "Pytest collected no evaluator tests.")
    if returncode == 3:
        return ("error", "internal_error", "Pytest reported an internal error.")
    return (
        "error",
        "internal_error",
        f"Pytest did not complete normal grading (exit code {returncode}).",
    )


def run_local_evaluator(
    task: EvaluationTask,
    workspace: str | Path,
    *,
    timeout_seconds: float,
    max_output_chars: int,
    specification: EvaluatorSpecification | None = None,
    sandbox_policy: SandboxPolicy | None = None,
) -> EvaluatorOutcome:
    """Run trusted fixed argv; repository text cannot alter the command."""

    try:
        selected = specification or build_local_evaluator_specification(
            task,
            timeout_seconds=timeout_seconds,
        )
    except EvaluatorPreflightError as error:
        return EvaluatorOutcome(
            status="error",
            failure_category="infrastructure_error",
            failure_kind="missing_specification",
            failure_reason=str(error),
            duration_seconds=0,
        )
    if evaluator_specification_digest(selected) != selected.digest:
        return EvaluatorOutcome(
            status="error",
            failure_category="infrastructure_error",
            failure_kind="identity_mismatch",
            failure_reason="Evaluator specification digest mismatch.",
            duration_seconds=0,
        )
    command = selected.command
    policy = sandbox_policy or SandboxPolicy(max_output_chars=max_output_chars)
    completed: SandboxExecutionResult = SandboxRunner(policy).run_repository(
        argv=command,
        repository_root=workspace,
        timeout_seconds=selected.timeout_seconds,
        purpose="evaluation",
        env={"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if completed.status == "timeout":
        stdout, warning_out = _bounded(completed.stdout, max_output_chars, "stdout")
        stderr, warning_err = _bounded(completed.stderr, max_output_chars, "stderr")
        return EvaluatorOutcome(
            status="timeout",
            stdout=stdout,
            stderr=stderr,
            failure_category="timeout",
            failure_kind="timeout",
            failure_reason=(
                f"Evaluator exceeded timeout_seconds={selected.timeout_seconds:g}."
            ),
            warnings=[item for item in (warning_out, warning_err) if item],
            duration_seconds=completed.duration_seconds,
            sandbox_provenance=completed.provenance,
        )
    if completed.status == "sandbox_error":
        return EvaluatorOutcome(
            status="error",
            failure_category="infrastructure_error",
            failure_kind=completed.error_kind or "start_failure",
            failure_reason=f"Evaluator sandbox could not start: {completed.stderr}",
            duration_seconds=completed.duration_seconds,
            sandbox_provenance=completed.provenance,
        )

    stdout, warning_out = _bounded(completed.stdout, max_output_chars, "stdout")
    stderr, warning_err = _bounded(completed.stderr, max_output_chars, "stderr")
    warnings = [item for item in (warning_out, warning_err) if item]
    if completed.exit_code is None:
        return EvaluatorOutcome(
            status="error",
            failure_category="infrastructure_error",
            failure_kind="internal_error",
            failure_reason="Completed evaluator execution omitted its exit code.",
            duration_seconds=completed.duration_seconds,
            sandbox_provenance=completed.provenance,
        )
    if completed.exit_code == 0:
        status = "passed"
        category = None
        failure_kind = None
        reason = None
    elif _is_pytest_command(command):
        status, failure_kind, reason = _pytest_failure(
            completed.exit_code,
            completed.stdout,
            completed.stderr,
        )
        category = "test_failure" if status == "failed" else "evaluation_test_failure"
        if status == "error":
            warnings.append(
                "Evaluator infrastructure evidence is not a model capability failure."
            )
    else:
        status = "failed"
        category = "evaluation_test_failure"
        failure_kind = "nonzero_exit"
        reason = f"Evaluator exited with code {completed.exit_code}."
        warnings.append(
            "The non-zero evaluator exit could not be separated further."
        )
    return EvaluatorOutcome(
        status=status,
        exit_code=completed.exit_code,
        stdout=stdout,
        stderr=stderr,
        failure_category=category,
        failure_kind=failure_kind,
        failure_reason=reason,
        warnings=warnings,
        duration_seconds=completed.duration_seconds,
        sandbox_provenance=completed.provenance,
    )
