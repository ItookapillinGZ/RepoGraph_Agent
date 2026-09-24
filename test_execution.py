"""Explicit, bounded execution of related Python tests through one backend."""

import subprocess  # noqa: F401  # nosec B404
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from repository_context import RepoContext
from sandbox.models import SandboxProvenance
from sandbox.policy import SandboxPolicy
from sandbox.runner import SandboxRunner

MAX_TEST_FILES = 3
MAX_TEST_OUTPUT_CHARS = 20_000
TEST_TIMEOUT_SECONDS = 30


class TestCaseResult(BaseModel):
    """One test case result, when a runner exposes it reliably."""

    model_config = ConfigDict(extra="forbid")

    node_id: str | None = None
    status: Literal["passed", "failed", "skipped", "error"]
    message: str | None = None


class TestRunResult(BaseModel):
    """Structured evidence from one bounded targeted-test invocation."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["not_run", "passed", "failed", "error", "timed_out"]
    framework: Literal["pytest", "unittest"] | None = None
    test_files: list[str] = Field(default_factory=list)
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    cases: list[TestCaseResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    execution_backend: Literal["host", "docker"] = "host"
    sandboxed: bool = False
    sandbox_provenance: SandboxProvenance | None = None


def _add_warning(warnings: list[str], warning: str) -> None:
    if warning not in warnings:
        warnings.append(warning)


def _is_within(path: Path, repository_root: Path) -> bool:
    return path == repository_root or repository_root in path.parents


def _bounded_output(
    output: str | bytes | None,
    stream_name: str,
    warnings: list[str],
) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        rendered = output.decode("utf-8", errors="replace")
    else:
        rendered = output
    if len(rendered) <= MAX_TEST_OUTPUT_CHARS:
        return rendered
    _add_warning(
        warnings,
        f"{stream_name} truncated at MAX_TEST_OUTPUT_CHARS="
        f"{MAX_TEST_OUTPUT_CHARS}.",
    )
    return rendered[:MAX_TEST_OUTPUT_CHARS]


def _resolve_test_files(
    repository_root: Path,
    repository_context: RepoContext,
    warnings: list[str],
) -> list[str]:
    related_tests = [
        related.path
        for related in repository_context.related_files
        if related.relationship == "test"
    ]
    if len(related_tests) > MAX_TEST_FILES:
        _add_warning(
            warnings,
            f"Related tests truncated at MAX_TEST_FILES={MAX_TEST_FILES}.",
        )

    selected: list[str] = []
    seen: set[str] = set()
    for raw_path in related_tests[:MAX_TEST_FILES]:
        candidate_path = Path(raw_path)
        if candidate_path.is_absolute() or candidate_path.drive:
            _add_warning(
                warnings,
                f"Skipped test path that is not repository-relative: {raw_path}.",
            )
            continue

        try:
            resolved = (repository_root / candidate_path).resolve(strict=True)
        except OSError:
            _add_warning(
                warnings,
                f"Related test file could not be resolved: {raw_path}.",
            )
            continue

        if not _is_within(resolved, repository_root):
            _add_warning(
                warnings,
                f"Skipped test file that resolves outside repository root: "
                f"{raw_path}.",
            )
            continue
        if not resolved.is_file():
            _add_warning(
                warnings,
                f"Related test path is not a file: {raw_path}.",
            )
            continue
        if resolved.suffix.casefold() != ".py":
            _add_warning(
                warnings,
                f"Skipped non-Python related test file: {raw_path}.",
            )
            continue

        relative_path = resolved.relative_to(repository_root).as_posix()
        if relative_path not in seen:
            seen.add(relative_path)
            selected.append(relative_path)

    return selected


def _contains_symlink(root: Path, relative_path: str) -> bool:
    current = root
    for part in Path(relative_path).parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _validate_explicit_test_files(
    repository_root: Path,
    test_files: Sequence[str],
    warnings: list[str],
) -> list[str]:
    """Validate bounded repository-relative Python test selections."""

    if len(test_files) > MAX_TEST_FILES:
        _add_warning(
            warnings,
            f"Test files truncated at MAX_TEST_FILES={MAX_TEST_FILES}.",
        )

    selected: list[str] = []
    seen: set[str] = set()
    for raw_path in test_files[:MAX_TEST_FILES]:
        normalized = str(raw_path).replace("\\", "/")
        candidate_path = Path(normalized)
        parts = tuple(
            part for part in normalized.split("/") if part not in ("", ".")
        )
        if (
            not parts
            or "\x00" in normalized
            or candidate_path.is_absolute()
            or candidate_path.drive
            or normalized.startswith("//")
        ):
            _add_warning(
                warnings,
                f"Skipped test path that is not repository-relative: {raw_path}.",
            )
            continue
        if any(part == ".." for part in parts):
            _add_warning(
                warnings,
                f"Skipped test file that resolves outside repository root: "
                f"{raw_path}.",
            )
            continue

        relative = "/".join(parts)
        if _contains_symlink(repository_root, relative):
            _add_warning(
                warnings,
                f"Skipped symbolic-link test file: {raw_path}.",
            )
            continue
        try:
            resolved = (repository_root / Path(*parts)).resolve(strict=True)
        except OSError:
            _add_warning(
                warnings,
                f"Test file could not be resolved: {raw_path}.",
            )
            continue
        if not _is_within(resolved, repository_root):
            _add_warning(
                warnings,
                f"Skipped test file that resolves outside repository root: "
                f"{raw_path}.",
            )
            continue
        if not resolved.is_file():
            _add_warning(warnings, f"Test path is not a file: {raw_path}.")
            continue
        if resolved.suffix.casefold() != ".py":
            _add_warning(
                warnings,
                f"Skipped non-Python test file: {raw_path}.",
            )
            continue

        relative = resolved.relative_to(repository_root).as_posix()
        if relative not in seen:
            seen.add(relative)
            selected.append(relative)

    return selected


def _pytest_status(exit_code: int, stdout: str, stderr: str) -> str:
    combined_output = f"{stdout}\n{stderr}".casefold()
    if "no module named pytest" in combined_output:
        return "error"
    if exit_code == 0:
        return "passed"
    if exit_code == 1:
        return "failed"
    return "error"


def _execute_pytest(
    repository_root: Path,
    test_files: list[str],
    warnings: list[str],
    sandbox_policy: SandboxPolicy | None = None,
) -> TestRunResult:
    """Execute one already-validated bounded pytest file selection."""

    policy = sandbox_policy or SandboxPolicy()
    command = [sys.executable, "-m", "pytest", *test_files]
    completed = SandboxRunner(policy).run_repository(
        argv=command,
        repository_root=repository_root,
        timeout_seconds=TEST_TIMEOUT_SECONDS,
        purpose="tests",
        env={"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if completed.output_truncated:
        if completed.stdout_truncated:
            _add_warning(
                warnings,
                f"stdout truncated at MAX_TEST_OUTPUT_CHARS={MAX_TEST_OUTPUT_CHARS}.",
            )
        if completed.stderr_truncated:
            _add_warning(
                warnings,
                f"stderr truncated at MAX_TEST_OUTPUT_CHARS={MAX_TEST_OUTPUT_CHARS}.",
            )
    if completed.status == "timeout":
        _add_warning(
            warnings,
            f"Targeted tests timed out after {TEST_TIMEOUT_SECONDS} seconds.",
        )
        return TestRunResult(
            status="timed_out",
            framework="pytest",
            test_files=test_files,
            stdout=completed.stdout,
            stderr=completed.stderr,
            warnings=warnings,
            execution_backend=completed.backend,
            sandboxed=completed.provenance.sandboxed,
            sandbox_provenance=completed.provenance,
        )
    if completed.status == "sandbox_error":
        _add_warning(
            warnings,
            "Sandbox execution failed closed: "
            f"{completed.error_kind or 'sandbox_error'}: {completed.stderr}",
        )
        return TestRunResult(
            status="error",
            framework="pytest",
            test_files=test_files,
            stdout=completed.stdout,
            stderr=completed.stderr,
            warnings=warnings,
            execution_backend=completed.backend,
            sandboxed=completed.provenance.sandboxed,
            sandbox_provenance=completed.provenance,
        )

    stdout = _bounded_output(completed.stdout, "stdout", warnings)
    stderr = _bounded_output(completed.stderr, "stderr", warnings)
    if completed.exit_code is None:
        raise RuntimeError("Completed test execution did not provide an exit code.")
    status = _pytest_status(completed.exit_code, stdout, stderr)
    if status == "error" and "no module named pytest" in (
        f"{stdout}\n{stderr}".casefold()
    ):
        _add_warning(
            warnings,
            "pytest is not available in the Agent's current Python environment; "
            "dependencies were not installed automatically.",
        )

    return TestRunResult(
        status=status,
        framework="pytest",
        test_files=test_files,
        exit_code=completed.exit_code,
        stdout=stdout,
        stderr=stderr,
        warnings=warnings,
        execution_backend=completed.backend,
        sandboxed=completed.provenance.sandboxed,
        sandbox_provenance=completed.provenance,
    )


def execute_test_files(
    repository_root: str | None,
    test_files: Sequence[str],
    *,
    sandbox_policy: SandboxPolicy | None = None,
) -> TestRunResult:
    """Run only an explicit bounded list of repository-relative Python files.

    This is the shared safe subprocess entry point. Callers remain responsible
    for authorization; this function independently revalidates paths as defense
    in depth and never accepts pytest arguments or commands.
    """

    if repository_root is None:
        return TestRunResult(
            status="not_run",
            warnings=["Test execution requires a repository root."],
        )
    warnings: list[str] = []
    try:
        root = Path(repository_root).resolve(strict=True)
    except OSError as error:
        return TestRunResult(
            status="error",
            framework="pytest",
            warnings=[f"Repository root could not be resolved: {error}."],
        )
    if not root.is_dir():
        return TestRunResult(
            status="error",
            framework="pytest",
            warnings=["Repository root is not a directory."],
        )

    selected = _validate_explicit_test_files(root, test_files, warnings)
    if not selected:
        _add_warning(warnings, "No valid test files were selected.")
        return TestRunResult(status="not_run", warnings=warnings)
    return _execute_pytest(root, selected, warnings, sandbox_policy)


def execute_targeted_tests(
    repository_root: str | None,
    repository_context: RepoContext,
    language: str,
    *,
    enabled: bool,
    sandbox_policy: SandboxPolicy | None = None,
) -> TestRunResult:
    """Run only RepoContext-selected tests after explicit user opt-in.

    The timeout and output limits bound resource use and prompt size; they do
    not isolate repository code from the local machine.
    """

    if not enabled:
        return TestRunResult(status="not_run")
    if language.casefold() != "python":
        return TestRunResult(
            status="not_run",
            warnings=["Targeted test execution currently supports Python only."],
        )
    if repository_root is None:
        return TestRunResult(
            status="not_run",
            warnings=["Targeted test execution requires a repository root."],
        )

    warnings: list[str] = []
    try:
        root = Path(repository_root).resolve(strict=True)
    except OSError as error:
        return TestRunResult(
            status="error",
            framework="pytest",
            warnings=[f"Repository root could not be resolved: {error}."],
        )
    if not root.is_dir():
        return TestRunResult(
            status="error",
            framework="pytest",
            warnings=["Repository root is not a directory."],
        )

    test_files = _resolve_test_files(root, repository_context, warnings)
    if not test_files:
        _add_warning(
            warnings,
            "No related test files were discovered for the target.",
        )
        return TestRunResult(status="not_run", warnings=warnings)

    return _execute_pytest(root, test_files, warnings, sandbox_policy)
