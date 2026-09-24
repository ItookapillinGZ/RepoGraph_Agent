"""Mutation-free deterministic execution replay from registered artifacts."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from observability.models import ReplayObservation, ReplayResult
from observability.storage import SQLiteTraceSink


class ReplayBlockedError(RuntimeError):
    pass


class ReplayExecutor(Protocol):
    def __call__(self, content: bytes, metadata: dict[str, object]) -> ReplayObservation: ...


MUTATION_METADATA_KEYS = frozenset(
    {"apply", "g4", "g5", "git_commit", "git_push", "create_pr", "publish"}
)


def normalized_observation_digest(observation: ReplayObservation) -> str:
    value = observation.model_dump(mode="json")
    value.pop("normalized_report_digest", None)
    canonical = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def semantic_match(
    original: ReplayObservation, replayed: ReplayObservation
) -> bool:
    return (
        original.status == replayed.status
        and original.resolved == replayed.resolved
        and original.exit_classification == replayed.exit_classification
        and (
            original.normalized_report_digest is None
            or replayed.normalized_report_digest is None
            or original.normalized_report_digest == replayed.normalized_report_digest
        )
    )


def replay_artifact(
    storage: SQLiteTraceSink,
    artifact_id: str,
    *,
    executor: ReplayExecutor,
    current_base_identity: str,
    current_backend: str,
    current_image_id: str | None = None,
    replay_id: str | None = None,
) -> ReplayResult:
    """Integrity-check and replay one checkpoint without any Agent or mutation call."""

    record, content = storage.load_artifact(artifact_id)
    metadata = record.metadata
    base_identity = str(metadata.get("base_identity", ""))
    verification_digest = str(metadata.get("verification_spec_digest", ""))
    if not base_identity or len(verification_digest) != 64:
        raise ReplayBlockedError("artifact lacks replay provenance")
    if base_identity != current_base_identity:
        raise ReplayBlockedError("base repository identity mismatch")
    if any(bool(metadata.get(key)) for key in MUTATION_METADATA_KEYS):
        raise ReplayBlockedError("replay artifact requested prohibited mutation")
    recorded_backend = str(metadata.get("sandbox_backend", "host"))
    if recorded_backend != current_backend:
        raise ReplayBlockedError("sandbox backend provenance mismatch; fallback prohibited")
    recorded_image = metadata.get("sandbox_image_id")
    if recorded_backend == "docker":
        if not current_image_id:
            raise ReplayBlockedError("recorded Docker environment is unavailable")
        if recorded_image != current_image_id:
            raise ReplayBlockedError("sandbox image provenance mismatch")
    original = ReplayObservation.model_validate(metadata.get("original_result"))
    replayed = executor(content, dict(metadata))
    match = semantic_match(original, replayed)
    result = ReplayResult(
        original_run_id=record.run_id,
        artifact_id=record.artifact_id,
        replay_id=replay_id or str(uuid.uuid4()),
        status="completed",
        semantic_match=match,
        original_result=original,
        replay_result=replayed,
        environment_match=True,
        source_base_identity=base_identity,
        sandbox_image_id=str(recorded_image) if recorded_image else None,
        verification_spec_digest=verification_digest,
        evaluator_digest=(
            str(metadata["evaluator_digest"])
            if metadata.get("evaluator_digest")
            else None
        ),
        replayed_at=datetime.now(UTC),
        warnings=[] if match else ["Replay semantic result differs from the original."],
    )
    storage.record_replay(result)
    return result


def guarded_executor(
    verify: Callable[[bytes, dict[str, object]], ReplayObservation]
) -> ReplayExecutor:
    """Make replay intent explicit at the API boundary (verification only)."""

    return verify

