"""Deterministic Git repositories and evaluation test artifacts."""

from __future__ import annotations

import hashlib
import subprocess  # nosec B404 - test-only fixed Git commands
import sys
from pathlib import Path

from change_set import ChangeSetReview
from engineering_plan import EngineeringPlan, PlannedFileChange, PlannedTest
from evaluation.models import EvaluationConfig, EvaluationTask
from evaluation.runner import RepoGraphRun
from plan_execution import (
    CandidateFileChange,
    MultiFileCandidate,
    PlanExecutionResult,
    PlanExecutionVerification,
)
from review_models import OverallRating
from test_execution import TestRunResult

ORIGINAL = "def add(a: int, b: int) -> int:\n    return a - b\n"
WRONG = "def add(a: int, b: int) -> int:\n    return a * b\n"
FIXED = "def add(a: int, b: int) -> int:\n    return a + b\n"
TEST = "from src.calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"


def git(repository: Path, *args: str) -> str:
    completed = subprocess.run(  # nosec B603 B607 - test-only fixed Git argv
        ["git", *args],
        cwd=repository,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    return completed.stdout.strip()


def make_repository(parent: Path) -> tuple[Path, str]:
    repository = parent / "source"
    (repository / "src").mkdir(parents=True)
    (repository / "tests").mkdir()
    (repository / "src" / "__init__.py").write_text("", encoding="utf-8")
    (repository / "src" / "calculator.py").write_text(ORIGINAL, encoding="utf-8")
    (repository / "tests" / "test_calculator.py").write_text(TEST, encoding="utf-8")
    git(repository, "init", "-b", "main")
    git(repository, "config", "user.name", "RepoGraph Evaluation Test")
    git(repository, "config", "user.email", "evaluation@example.invalid")
    git(repository, "config", "core.autocrlf", "false")
    git(repository, "add", ".")
    git(repository, "commit", "-m", "base")
    return repository, git(repository, "rev-parse", "HEAD")


def task(repository: Path, commit: str, *, task_id: str = "task-001") -> EvaluationTask:
    return EvaluationTask(
        id=task_id,
        dataset="local",
        repository=str(repository),
        base_commit=commit,
        task="Fix add() so the regression test passes.",
        test_command=[
            sys.executable,
            "-m",
            "pytest",
            "tests/test_calculator.py",
            "-q",
        ],
        expected_files=["src/calculator.py"],
    )


def plan() -> EngineeringPlan:
    return EngineeringPlan(
        summary="Fix calculator addition.",
        files=[
            PlannedFileChange(
                path="src/calculator.py",
                action="modify",
                rationale="Return the sum required by the regression test.",
            )
        ],
        tests=[
            PlannedTest(
                path="tests/test_calculator.py",
                purpose="Verify add returns the arithmetic sum.",
            )
        ],
    )


def candidate(source: str = FIXED) -> MultiFileCandidate:
    return MultiFileCandidate(
        summary="Update calculator behavior.",
        files=[
            CandidateFileChange(
                path="src/calculator.py",
                action="modify",
                content=source,
            )
        ],
    )


def execution_result(
    final: MultiFileCandidate,
    *,
    rounds: int = 0,
    review_good: bool = True,
    verification_good: bool = True,
) -> PlanExecutionResult:
    review = ChangeSetReview(
        overall_rating=(
            OverallRating.GOOD if review_good else OverallRating.NEEDS_WORK
        ),
        summary="Deterministic review.",
        file_results=[],
    )
    return PlanExecutionResult(
        plan=plan(),
        candidate=final,
        status="verified" if review_good and verification_good else "failed",
        verification=PlanExecutionVerification(
            status="verified" if verification_good else "failed",
            test_result=TestRunResult(
                status="passed" if verification_good else "failed"
            ),
        ),
        change_set_review=review,
        correction_rounds_used=rounds,
    )


class StaticAdapter:
    def __init__(
        self,
        final: MultiFileCandidate | None,
        *,
        first: MultiFileCandidate | None = None,
        rounds: int = 0,
        review_good: bool = True,
        verification_good: bool = True,
        error: Exception | None = None,
    ) -> None:
        self.final = final
        self.first = first or final
        self.rounds = rounds
        self.review_good = review_good
        self.verification_good = verification_good
        self.error = error
        self.calls = 0

    def run(
        self,
        workspace: str,
        evaluation_task: EvaluationTask,
        config: EvaluationConfig,
    ) -> RepoGraphRun:
        del workspace, evaluation_task, config
        self.calls += 1
        if self.error:
            raise self.error
        if self.final is None:
            return RepoGraphRun(plan=plan(), execution=None)
        return RepoGraphRun(
            plan=plan(),
            execution=execution_result(
                self.final,
                rounds=self.rounds,
                review_good=self.review_good,
                verification_good=self.verification_good,
            ),
            evaluated_candidates=[self.first] if self.first else [],
            tool_calls=None,
            test_calls=1,
        )


def repository_digest(repository: Path) -> dict[str, str]:
    return {
        path.relative_to(repository).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in repository.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }
