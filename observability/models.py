"""Strict, bounded durable models for local RepoGraph observability."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = 1
MAX_METADATA_KEYS = 64
MAX_METADATA_VALUE_CHARS = 4_000
MAX_WARNINGS = 100

RunStatus = Literal["active", "completed", "failed", "interrupted"]
SpanStatus = Literal["active", "completed", "failed", "interrupted"]
ReplayStatus = Literal["completed", "blocked", "failed"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _bounded_metadata(value: dict[str, object]) -> dict[str, object]:
    if len(value) > MAX_METADATA_KEYS:
        raise ValueError(f"metadata exceeds {MAX_METADATA_KEYS} keys")
    for key, item in value.items():
        if not isinstance(key, str) or not key or len(key) > 128:
            raise ValueError("metadata keys must be non-empty strings up to 128 chars")
        if len(str(item)) > MAX_METADATA_VALUE_CHARS:
            raise ValueError("metadata value exceeds bounded telemetry limit")
    return value


class TraceRun(StrictModel):
    run_id: str = Field(min_length=1, max_length=100)
    run_kind: str = Field(min_length=1, max_length=100)
    started_at: datetime
    finished_at: datetime | None = None
    status: RunStatus
    root_span_id: str = Field(min_length=1, max_length=100)
    repo_identity: str | None = Field(default=None, max_length=512)
    task_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    task_summary: str | None = Field(default=None, max_length=500)
    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)


class TraceSpan(StrictModel):
    span_id: str = Field(min_length=1, max_length=100)
    run_id: str = Field(min_length=1, max_length=100)
    parent_span_id: str | None = Field(default=None, max_length=100)
    name: str = Field(min_length=1, max_length=200)
    kind: str = Field(min_length=1, max_length=100)
    start_time: datetime
    end_time: datetime | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    status: SpanStatus
    sequence: int = Field(ge=1)
    metadata: dict[str, object] = Field(default_factory=dict)
    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)

    _validate_metadata = field_validator("metadata")(_bounded_metadata)


class TraceEvent(StrictModel):
    event_id: str = Field(min_length=1, max_length=100)
    run_id: str = Field(min_length=1, max_length=100)
    span_id: str | None = Field(default=None, max_length=100)
    sequence_number: int = Field(ge=1)
    timestamp: datetime
    name: str = Field(min_length=1, max_length=200)
    kind: str = Field(min_length=1, max_length=100)
    status: str = Field(min_length=1, max_length=64)
    metadata: dict[str, object] = Field(default_factory=dict)
    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)

    _validate_metadata = field_validator("metadata")(_bounded_metadata)


class ArtifactRecord(StrictModel):
    artifact_id: str = Field(min_length=1, max_length=100)
    run_id: str = Field(min_length=1, max_length=100)
    kind: str = Field(min_length=1, max_length=100)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    producer_span_id: str | None = Field(default=None, max_length=100)
    created_at: datetime
    storage_ref: str = Field(min_length=1, max_length=2_000)
    metadata: dict[str, object] = Field(default_factory=dict)
    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)

    _validate_metadata = field_validator("metadata")(_bounded_metadata)


class ArtifactEdge(StrictModel):
    run_id: str = Field(min_length=1, max_length=100)
    parent_artifact_id: str = Field(min_length=1, max_length=100)
    child_artifact_id: str = Field(min_length=1, max_length=100)
    relation: Literal[
        "derived_from",
        "verified_by",
        "reviewed_by",
        "corrected_from",
        "packaged_as",
        "delivered_as",
        "evaluated_by",
        "replayed_as",
    ]
    created_at: datetime
    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)


class ReplayObservation(StrictModel):
    status: str = Field(min_length=1, max_length=100)
    resolved: bool | None = None
    exit_classification: str | None = Field(default=None, max_length=200)
    normalized_report_digest: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )


class ReplayResult(StrictModel):
    original_run_id: str = Field(min_length=1, max_length=100)
    artifact_id: str = Field(min_length=1, max_length=100)
    replay_id: str = Field(min_length=1, max_length=100)
    status: ReplayStatus
    semantic_match: bool
    original_result: ReplayObservation
    replay_result: ReplayObservation | None = None
    environment_match: bool
    source_base_identity: str = Field(min_length=1, max_length=512)
    sandbox_image_id: str | None = Field(default=None, max_length=512)
    verification_spec_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    replayed_at: datetime
    warnings: list[str] = Field(default_factory=list, max_length=MAX_WARNINGS)
    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)

    @field_validator("warnings")
    @classmethod
    def bound_warnings(cls, value: list[str]) -> list[str]:
        return [str(item)[:2_000] for item in value]


class RunMetrics(StrictModel):
    total_duration_ms: float | None = Field(default=None, ge=0)
    llm_duration_ms: float = Field(default=0, ge=0)
    sandbox_duration_ms: float = Field(default=0, ge=0)
    verification_duration_ms: float = Field(default=0, ge=0)
    tool_duration_ms: float = Field(default=0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    sandbox_executions: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    candidate_count: int = Field(default=0, ge=0)
    correction_rounds: int = Field(default=0, ge=0)
    artifact_count: int = Field(default=0, ge=0)

