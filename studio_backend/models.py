"""Strict public and persistence models for RepoGraph Studio."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from engineering_plan import MAX_TASK_CHARS

RunStatus = Literal[
    "queued",
    "running",
    "verified",
    "failed",
    "rejected",
    "applying",
    "applied",
    "git_creating",
    "git_created",
    "publishing",
    "published",
    "partial",
    "stale",
    "conflict",
    "error",
    "interrupted",
]

RunPhase = Literal[
    "queued",
    "planning",
    "execution",
    "verification",
    "review",
    "correction",
    "approval_apply",
    "application",
    "approval_git",
    "git_delivery",
    "approval_remote",
    "remote_delivery",
    "completed",
    "failed",
]

EventStatus = Literal["started", "completed", "failed", "info", "waiting"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RepositorySummary(StrictModel):
    id: str
    name: str
    git_repository: bool


class RepositoryListResponse(StrictModel):
    repositories: list[RepositorySummary]


class RegisterRepositoryRequest(StrictModel):
    path: str = Field(min_length=1, max_length=1024)


class StudioRun(StrictModel):
    id: str
    repository_id: str
    task: str
    status: RunStatus
    phase: RunPhase
    created_at: str
    updated_at: str
    approval_digest: str | None = None
    failure_reason: str | None = None


class RunEvent(StrictModel):
    run_id: str
    sequence: int = Field(ge=1)
    timestamp: str
    phase: str = Field(min_length=1, max_length=64)
    event_type: str = Field(min_length=1, max_length=96)
    status: EventStatus
    title: str = Field(min_length=1, max_length=200)
    message: str = Field(default="", max_length=4_000)
    metadata: dict[str, object] = Field(default_factory=dict)


class StartRunRequest(StrictModel):
    repository_id: str = Field(min_length=1, max_length=255)
    task: str = Field(min_length=1, max_length=MAX_TASK_CHARS)
    self_correct: bool = True
    max_correction_rounds: int = Field(default=2, ge=0, le=5)

    @field_validator("repository_id", "task")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped


class StartRunResponse(StrictModel):
    run_id: str
    status: Literal["queued"] = "queued"


class RunTraceSummary(StrictModel):
    trace_id: str
    duration_ms: float | None = Field(default=None, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    sandbox_executions: int = Field(default=0, ge=0)
    artifact_count: int = Field(default=0, ge=0)
    lineage_summary: str = Field(default="", max_length=20_000)


class RunDetail(StudioRun):
    artifact_kinds: list[str] = Field(default_factory=list)
    latest_event_sequence: int = Field(default=0, ge=0)
    github_auth_configured: bool = False
    trace: RunTraceSummary | None = None


class RunListResponse(StrictModel):
    runs: list[StudioRun]


class EventListResponse(StrictModel):
    events: list[RunEvent]


class ArtifactResponse(StrictModel):
    run_id: str
    kind: str
    version: int = Field(ge=1)
    created_at: str
    payload: object


class ApplyApprovalRequest(StrictModel):
    approved: bool
    approval_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class GitApprovalRequest(ApplyApprovalRequest):
    branch_name: str | None = Field(default=None, max_length=255)
    commit_message: str | None = Field(default=None, max_length=4_000)


class RemoteApprovalRequest(ApplyApprovalRequest):
    remote_name: str = Field(default="origin", min_length=1, max_length=255)
    base_branch: str = Field(min_length=1, max_length=255)
    branch_name: str | None = Field(default=None, max_length=255)
    pr_title: str | None = Field(default=None, max_length=256)
    pr_body: str | None = Field(default=None, max_length=20_000)


class RejectRunRequest(StrictModel):
    rejected: bool


class ApprovalResponse(StrictModel):
    run: StudioRun
    artifact_kind: str | None = None
