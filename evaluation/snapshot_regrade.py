"""Offline replay of exact candidate versions without any LLM execution."""

from __future__ import annotations

import hashlib
import subprocess  # nosec B404
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from evaluation.candidate_artifacts import (
    CandidateArtifactError,
    CandidateArtifactStore,
)
from evaluation.evaluator import (
    build_local_evaluator_specification,
    run_local_evaluator,
)
from evaluation.models import EvaluationResult, EvaluationTask, EvaluatorSpecification
from evaluation.workspace import prepare_task_workspace


class CandidateRegradeOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    experiment_id: str
    snapshot_path: str
    execution_attempt: int
    candidate_attempt: int
    candidate_sha256: str
    diff_sha256: str
    original_evaluator_digest: str
    evaluator_digest: str
    regraded_with_different_evaluator: bool
    status: Literal["passed", "failed", "error", "timeout"]
    resolved: bool
    evaluator_failure_kind: str | None = None
    evaluator_failure_reason: str | None = None
    evaluator_duration_seconds: float
    stdout_sha256: str
    stderr_sha256: str


class CandidateVersionAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    first: CandidateRegradeOutcome
    final: CandidateRegradeOutcome
    first_attempt_resolved: bool
    final_resolved: bool
    rescued_by_correction: bool
    regressed_after_correction: bool


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _apply_snapshot_diff(workspace: Path, diff_text: str) -> None:
    try:
        completed = subprocess.run(  # nosec B603 B607
            ["git", "apply", "--binary", "--whitespace=nowarn", "-"],
            cwd=workspace,
            input=diff_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CandidateArtifactError(
            f"Could not replay candidate snapshot: {error}"
        ) from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[:1_000]
        raise CandidateArtifactError(
            f"Candidate snapshot diff did not apply: {detail}"
        )


def regrade_candidate_snapshot(
    task: EvaluationTask,
    snapshot_path: str,
    *,
    artifacts_root: str | Path,
    workspace_root: str | Path,
    evaluator_specification: EvaluatorSpecification | None = None,
) -> CandidateRegradeOutcome:
    """Verify, replay, and grade one persisted candidate in a fresh workspace."""

    store = CandidateArtifactStore(artifacts_root)
    snapshot = store.load(
        snapshot_path,
        expected_task_id=task.id,
    )
    if snapshot.base_commit != task.base_commit:
        raise CandidateArtifactError("Candidate snapshot base commit mismatch.")
    selected = evaluator_specification or build_local_evaluator_specification(
        task,
        timeout_seconds=300.0,
    )
    workspace, resolved_base = prepare_task_workspace(
        task.repository,
        task.base_commit,
        workspace_root,
        (
            f"snapshot-regrade-{snapshot.experiment_id}-"
            f"{snapshot.execution_attempt:02d}-{snapshot.attempt:02d}"
        ),
        task.id,
    )
    if resolved_base != snapshot.base_commit:
        raise CandidateArtifactError("Fresh workspace base commit mismatch.")
    _apply_snapshot_diff(workspace, snapshot.diff_text)
    outcome = run_local_evaluator(
        task,
        workspace,
        timeout_seconds=selected.timeout_seconds,
        max_output_chars=20_000,
        specification=selected,
    )
    return CandidateRegradeOutcome(
        task_id=task.id,
        experiment_id=snapshot.experiment_id,
        snapshot_path=snapshot_path,
        execution_attempt=snapshot.execution_attempt,
        candidate_attempt=snapshot.attempt,
        candidate_sha256=snapshot.candidate_sha256,
        diff_sha256=snapshot.diff_sha256,
        original_evaluator_digest=snapshot.evaluator_digest,
        evaluator_digest=selected.digest,
        regraded_with_different_evaluator=(
            snapshot.evaluator_digest != selected.digest
        ),
        status=outcome.status,
        resolved=outcome.status == "passed",
        evaluator_failure_kind=outcome.failure_kind,
        evaluator_failure_reason=outcome.failure_reason,
        evaluator_duration_seconds=outcome.duration_seconds,
        stdout_sha256=_sha256_text(outcome.stdout),
        stderr_sha256=_sha256_text(outcome.stderr),
    )


def regrade_result_candidate_versions(
    task: EvaluationTask,
    result: EvaluationResult,
    *,
    artifacts_root: str | Path,
    workspace_root: str | Path,
    evaluator_specification: EvaluatorSpecification | None = None,
) -> CandidateVersionAnalysis:
    """Regrade exact v1 and final snapshots and derive correction outcomes."""

    if not result.candidate_snapshot_paths:
        raise CandidateArtifactError(
            f"Result {result.experiment}/{result.task_id} has no candidate snapshots."
        )
    first = regrade_candidate_snapshot(
        task,
        result.candidate_snapshot_paths[0],
        artifacts_root=artifacts_root,
        workspace_root=workspace_root,
        evaluator_specification=evaluator_specification,
    )
    final = regrade_candidate_snapshot(
        task,
        result.candidate_snapshot_paths[-1],
        artifacts_root=artifacts_root,
        workspace_root=workspace_root,
        evaluator_specification=evaluator_specification,
    )
    if first.experiment_id != result.experiment or final.experiment_id != result.experiment:
        raise CandidateArtifactError("Candidate snapshot result identity mismatch.")
    return CandidateVersionAnalysis(
        task_id=task.id,
        first=first,
        final=final,
        first_attempt_resolved=first.resolved,
        final_resolved=final.resolved,
        rescued_by_correction=not first.resolved and final.resolved,
        regressed_after_correction=first.resolved and not final.resolved,
    )
