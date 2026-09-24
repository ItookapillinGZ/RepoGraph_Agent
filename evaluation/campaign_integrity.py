"""Read-only campaign audit and explicit legacy attempt reconciliation."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from evaluation.candidate_artifacts import (
    CandidateArtifactError,
    CandidateArtifactStore,
)
from evaluation.live import LiveRunReservation
from evaluation.models import (
    EvaluationResult,
    ExecutionAttemptRecord,
)
from evaluation.storage import EvaluationStorage


class CampaignAuditIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    message: str
    experiment_id: str | None = None
    task_id: str | None = None
    execution_attempt: int | None = None


class CampaignAuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    campaign_id: str
    passed: bool
    experiments: int = Field(ge=0)
    results: int = Field(ge=0)
    candidate_snapshots: int = Field(ge=0)
    preserved_partial_snapshots: int = Field(ge=0)
    attempts: int = Field(ge=0)
    issues: list[CampaignAuditIssue]


def _terminal_status(result: EvaluationResult) -> Literal[
    "completed", "failed", "timeout"
]:
    if result.hard_timeout or result.status == "timeout":
        return "timeout"
    if result.infrastructure_failure:
        return "failed"
    return "completed"


def reconcile_legacy_attempts(
    storage: EvaluationStorage,
    *,
    campaign_id: str,
    reservations: Sequence[LiveRunReservation],
    history_records: Sequence[dict[str, Any]],
) -> list[ExecutionAttemptRecord]:
    """Link legacy budget/history evidence to immutable attempt identities.

    Reservation order is the only legacy allocator evidence. A result claiming a
    different attempt than that order is preserved as a failed identity mismatch;
    its content is never rewritten.
    """

    by_ordinal = {
        int(item["live_task_run_ordinal"]): EvaluationResult.model_validate(
            item["result"]
        )
        for item in history_records
        if isinstance(item, dict)
        and isinstance(item.get("live_task_run_ordinal"), int)
        and isinstance(item.get("result"), dict)
    }
    counts: Counter[tuple[str, str]] = Counter()
    imported: list[ExecutionAttemptRecord] = []
    for reservation in sorted(reservations, key=lambda item: item.ordinal):
        key = (reservation.experiment_id, reservation.task_id)
        counts[key] += 1
        attempt = counts[key]
        result = by_ordinal.get(reservation.ordinal)
        if result is None:
            status = "interrupted"
            reason = "Legacy reservation ended without a durable result."
        elif result.execution_attempt != attempt:
            status = "failed"
            reason = (
                "Legacy result reused execution_attempt="
                f"{result.execution_attempt}; reservation order requires {attempt}."
            )
        else:
            status = _terminal_status(result)
            reason = result.failure_reason if status != "completed" else None
        imported.append(
            storage.import_execution_attempt(
                ExecutionAttemptRecord(
                    campaign_id=campaign_id,
                    config_name=reservation.config_name,
                    experiment_id=reservation.experiment_id,
                    task_id=reservation.task_id,
                    execution_attempt=attempt,
                    status=status,
                    reserved_at=reservation.timestamp,
                    started_at=reservation.timestamp,
                    finished_at=reservation.timestamp,
                    live_run_ordinal=reservation.ordinal,
                    failure_reason=reason,
                )
            )
        )
    return imported


def load_history(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    records: list[dict[str, Any]] = []
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"History line {number} is not an object.")
        records.append(value)
    return records


def audit_campaign(
    storage: EvaluationStorage,
    *,
    artifacts_root: str | Path,
    campaign_id: str,
    experiment_ids: Sequence[str] | None = None,
    lineage_campaign_ids: Sequence[str] | None = None,
) -> CampaignAuditReport:
    """Audit identities, provenance, hashes, attempts, and orphan snapshots."""

    selected_ids = set(experiment_ids or ())
    experiments = {
        item.id: item
        for item in storage.list_experiments()
        if not selected_ids or item.id in selected_ids
    }
    tasks = {item.id: item for item in storage.list_tasks()}
    accepted_campaign_ids = {campaign_id, *(lineage_campaign_ids or ())}
    attempts = [
        item
        for item in storage.list_execution_attempts()
        if item.campaign_id in accepted_campaign_ids
        and item.experiment_id in experiments
    ]
    attempt_map = {
        (item.experiment_id, item.task_id, item.execution_attempt): item
        for item in attempts
    }
    issues: list[CampaignAuditIssue] = []

    def issue(kind: str, message: str, **identity: object) -> None:
        issues.append(CampaignAuditIssue(kind=kind, message=message, **identity))

    for key, group in _group_attempts(attempts).items():
        ordered = sorted(
            group,
            key=lambda item: (item.reserved_at, item.live_run_ordinal or 1_000_000_000),
        )
        numbers = [item.execution_attempt for item in ordered]
        if any(right <= left for left, right in pairwise(numbers)):
            issue(
                "non_monotonic_attempt",
                f"Reservation chronology is not monotonic: {numbers}.",
                experiment_id=key[0],
                task_id=key[1],
            )

    store = CandidateArtifactStore(artifacts_root)
    referenced: set[str] = set()
    result_count = 0
    for experiment in experiments.values():
        config_digest = experiment.config.digest()
        for result in storage.list_results(experiment.id):
            result_count += 1
            identity = {
                "experiment_id": experiment.id,
                "task_id": result.task_id,
                "execution_attempt": result.execution_attempt,
            }
            task = tasks.get(result.task_id)
            if task is None or result.base_commit != task.base_commit:
                issue("task_identity", "Result task/base identity mismatch.", **identity)
            if result.config_digest != config_digest:
                issue("config_digest", "Result configuration digest mismatch.", **identity)
            if result.configured_model_name != experiment.config.model_name:
                issue("model_identity", "Result configured model mismatch.", **identity)
            if result.provenance is None:
                issue("missing_provenance", "Result provenance is missing.", **identity)
            else:
                if result.provenance.config_digest != config_digest:
                    issue("provenance_config", "Provenance config digest mismatch.", **identity)
                if result.provenance.repo_base_commit != result.base_commit:
                    issue("provenance_base", "Provenance base commit mismatch.", **identity)
                if result.provenance.evaluator_digest != result.evaluator_digest:
                    issue("provenance_evaluator", "Evaluator digest mismatch.", **identity)
                if result.provenance.model_name != experiment.config.model_name:
                    issue("provenance_model", "Provenance model mismatch.", **identity)
            persisted_attempt = attempt_map.get(
                (experiment.id, result.task_id, result.execution_attempt)
            )
            if persisted_attempt is None:
                issue("missing_attempt", "Result has no durable attempt reservation.", **identity)
            elif persisted_attempt.status != _terminal_status(result):
                issue(
                    "reservation_result_status",
                    "Result terminal state does not match its attempt reservation.",
                    **identity,
                )
            loaded = []
            for relative in result.candidate_snapshot_paths:
                referenced.add(Path(relative).as_posix())
                try:
                    snapshot = store.load(
                        relative,
                        expected_task_id=result.task_id,
                        expected_experiment_id=experiment.id,
                    )
                    expected_path = store.relative_path(
                        snapshot.experiment_id,
                        snapshot.task_id,
                        snapshot.execution_attempt,
                        snapshot.attempt,
                    ).as_posix()
                    if expected_path != Path(relative).as_posix():
                        issue("snapshot_path", "Snapshot path identity mismatch.", **identity)
                    if snapshot.execution_attempt != result.execution_attempt:
                        issue("snapshot_attempt", "Snapshot/result attempt mismatch.", **identity)
                    if snapshot.base_commit != result.base_commit:
                        issue("snapshot_base", "Snapshot base commit mismatch.", **identity)
                    if snapshot.evaluator_digest != result.evaluator_digest:
                        issue("snapshot_evaluator", "Snapshot evaluator mismatch.", **identity)
                    loaded.append(snapshot)
                except (CandidateArtifactError, OSError, ValueError) as error:
                    issue("snapshot_integrity", str(error), **identity)
            if loaded:
                if result.first_candidate_sha256 != loaded[0].candidate_sha256:
                    issue("first_candidate_hash", "First candidate hash mismatch.", **identity)
                if result.final_candidate_sha256 != loaded[-1].candidate_sha256:
                    issue("final_candidate_hash", "Final candidate hash mismatch.", **identity)

    seen: dict[tuple[str, str, int, int], str] = {}
    snapshot_count = 0
    preserved = 0
    root = Path(artifacts_root)
    for path in sorted(root.rglob("candidate-*.json")) if root.is_dir() else []:
        relative = path.relative_to(root).as_posix()
        try:
            snapshot = store.load(relative)
        except (CandidateArtifactError, OSError, ValueError) as error:
            issue("snapshot_integrity", str(error))
            continue
        if snapshot.experiment_id not in experiments:
            continue
        snapshot_count += 1
        identity = (
            snapshot.experiment_id,
            snapshot.task_id,
            snapshot.execution_attempt,
            snapshot.attempt,
        )
        prior = seen.get(identity)
        if prior is not None:
            issue(
                "duplicate_snapshot_identity",
                f"Duplicate snapshot identity at {prior} and {relative}.",
                experiment_id=snapshot.experiment_id,
                task_id=snapshot.task_id,
                execution_attempt=snapshot.execution_attempt,
            )
        seen[identity] = relative
        if relative not in referenced:
            attempt = attempt_map.get(identity[:3])
            if attempt is not None and attempt.status in {
                "interrupted",
                "failed",
                "timeout",
            }:
                preserved += 1
            else:
                issue(
                    "orphan_snapshot",
                    f"Unreferenced snapshot is not linked to a terminal failed attempt: {relative}.",
                    experiment_id=snapshot.experiment_id,
                    task_id=snapshot.task_id,
                    execution_attempt=snapshot.execution_attempt,
                )

    return CampaignAuditReport(
        campaign_id=campaign_id,
        passed=not issues,
        experiments=len(experiments),
        results=result_count,
        candidate_snapshots=snapshot_count,
        preserved_partial_snapshots=preserved,
        attempts=len(attempts),
        issues=issues,
    )


def _group_attempts(
    attempts: Sequence[ExecutionAttemptRecord],
) -> dict[tuple[str, str], list[ExecutionAttemptRecord]]:
    grouped: dict[tuple[str, str], list[ExecutionAttemptRecord]] = {}
    for item in attempts:
        grouped.setdefault((item.experiment_id, item.task_id), []).append(item)
    return grouped
