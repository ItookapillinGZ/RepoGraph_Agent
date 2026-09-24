"""Durable, integrity-checked candidate artifacts for exact offline replay."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from evaluation.models import EvaluationCandidateSnapshot
from plan_execution import MultiFileCandidate

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
MAX_SNAPSHOT_BYTES = 5_000_000


class CandidateArtifactError(RuntimeError):
    """Raised when a candidate artifact is unsafe, inconsistent, or tampered."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def candidate_sha256(candidate: MultiFileCandidate) -> str:
    """Hash the strict candidate representation deterministically."""

    validated = MultiFileCandidate.model_validate(candidate)
    return hashlib.sha256(_canonical_json(validated.model_dump(mode="json"))).hexdigest()


def diff_sha256(diff_text: str) -> str:
    return hashlib.sha256(diff_text.encode("utf-8")).hexdigest()


def verify_candidate_snapshot(
    snapshot: EvaluationCandidateSnapshot,
) -> EvaluationCandidateSnapshot:
    """Recompute both integrity digests before a snapshot can be replayed."""

    validated = EvaluationCandidateSnapshot.model_validate(snapshot)
    if candidate_sha256(validated.candidate) != validated.candidate_sha256:
        raise CandidateArtifactError("Candidate snapshot candidate_sha256 mismatch.")
    if diff_sha256(validated.diff_text) != validated.diff_sha256:
        raise CandidateArtifactError("Candidate snapshot diff_sha256 mismatch.")
    return validated


def _safe_component(value: str, label: str) -> str:
    if not _SAFE_COMPONENT.fullmatch(value):
        raise CandidateArtifactError(f"Unsafe {label} for artifact path: {value!r}")
    return value


class CandidateArtifactStore:
    """Store each execution attempt under a deterministic bounded path."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def relative_path(
        self,
        experiment_id: str,
        task_id: str,
        execution_attempt: int,
        candidate_attempt: int,
    ) -> Path:
        experiment = _safe_component(experiment_id, "experiment_id")
        task = _safe_component(task_id, "task_id")
        if not 1 <= execution_attempt <= 1_000_000:
            raise CandidateArtifactError(
                "execution_attempt must be between 1 and 1000000"
            )
        if not 1 <= candidate_attempt <= 100:
            raise CandidateArtifactError("candidate_attempt must be between 1 and 100")
        return (
            Path(experiment)
            / task
            / f"execution-{execution_attempt:02d}"
            / f"candidate-{candidate_attempt:02d}.json"
        )

    def persist(
        self,
        *,
        task_id: str,
        experiment_id: str,
        execution_attempt: int,
        attempt: int,
        base_commit: str,
        model_name: str | None,
        candidate: MultiFileCandidate,
        diff_text: str,
        evaluator_digest: str,
        created_at: str | None = None,
    ) -> tuple[EvaluationCandidateSnapshot, str]:
        snapshot = EvaluationCandidateSnapshot(
            task_id=task_id,
            experiment_id=experiment_id,
            execution_attempt=execution_attempt,
            attempt=attempt,
            correction_round=attempt - 1,
            base_commit=base_commit,
            model_name=model_name,
            candidate=MultiFileCandidate.model_validate(candidate),
            candidate_sha256=candidate_sha256(candidate),
            diff_text=diff_text,
            diff_sha256=diff_sha256(diff_text),
            evaluator_digest=evaluator_digest,
            created_at=created_at or datetime.now(UTC).isoformat(),
        )
        verify_candidate_snapshot(snapshot)
        relative = self.relative_path(
            experiment_id,
            task_id,
            execution_attempt,
            attempt,
        )
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(
                snapshot.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        if len(payload) > MAX_SNAPSHOT_BYTES:
            raise CandidateArtifactError(
                f"Candidate snapshot exceeds MAX_SNAPSHOT_BYTES={MAX_SNAPSHOT_BYTES}."
            )
        if destination.exists():
            # Snapshot identities are single-use. Even byte-identical content here
            # means an execution identity was recycled, which must remain a hard
            # integrity failure rather than becoming an implicit resume protocol.
            raise CandidateArtifactError(
                f"Candidate artifact identity already exists: {relative.as_posix()}"
            )
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            temporary.write_bytes(payload)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return snapshot, relative.as_posix()

    def load(
        self,
        relative_path: str | Path,
        *,
        expected_task_id: str | None = None,
        expected_experiment_id: str | None = None,
    ) -> EvaluationCandidateSnapshot:
        relative = Path(relative_path)
        if relative.is_absolute():
            raise CandidateArtifactError("Candidate artifact path must be relative.")
        destination = (self.root / relative).resolve(strict=True)
        if not destination.is_relative_to(self.root):
            raise CandidateArtifactError("Candidate artifact escaped its artifact root.")
        if not destination.is_file():
            raise CandidateArtifactError("Candidate artifact path is not a file.")
        if destination.stat().st_size > MAX_SNAPSHOT_BYTES:
            raise CandidateArtifactError("Candidate artifact is larger than allowed.")
        try:
            snapshot = EvaluationCandidateSnapshot.model_validate_json(
                destination.read_bytes()
            )
        except (OSError, ValueError) as error:
            raise CandidateArtifactError(
                f"Candidate artifact is malformed: {relative.as_posix()}"
            ) from error
        verify_candidate_snapshot(snapshot)
        if expected_task_id is not None and snapshot.task_id != expected_task_id:
            raise CandidateArtifactError("Candidate artifact task identity mismatch.")
        if (
            expected_experiment_id is not None
            and snapshot.experiment_id != expected_experiment_id
        ):
            raise CandidateArtifactError("Candidate artifact experiment identity mismatch.")
        return snapshot
