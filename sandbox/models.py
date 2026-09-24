"""Strict, command-string-free sandbox execution schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_ARGV_PARTS = 200
MAX_ARG_CHARS = 8_000
MAX_ENVIRONMENT_VARIABLES = 20
MAX_ENVIRONMENT_VALUE_CHARS = 4_000

SandboxPurpose = Literal["tests", "static_analysis", "evaluation"]
SandboxStatus = Literal["passed", "failed", "timeout", "sandbox_error"]
SandboxBackendName = Literal["host", "docker"]


class SandboxProvenance(BaseModel):
    """Non-sensitive execution policy evidence."""

    model_config = ConfigDict(extra="forbid")

    backend: SandboxBackendName
    sandboxed: bool
    image: str | None = None
    image_id: str | None = None
    network: Literal["host", "none"]
    read_only_root: bool
    cap_drop_all: bool
    no_new_privileges: bool
    memory_limit_mb: int | None = None
    cpu_limit: float | None = None
    pids_limit: int | None = None
    timeout_seconds: float


class SandboxExecutionRequest(BaseModel):
    """One narrow fixed-argv execution request."""

    model_config = ConfigDict(extra="forbid")

    argv: list[str] = Field(min_length=1, max_length=MAX_ARGV_PARTS)
    workspace: str = Field(min_length=1, max_length=4_000)
    timeout_seconds: float = Field(gt=0)
    env: dict[str, str] = Field(default_factory=dict)
    purpose: SandboxPurpose

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, value: list[str]) -> list[str]:
        if any(not part or len(part) > MAX_ARG_CHARS for part in value):
            raise ValueError("argv values must be non-empty and bounded")
        if any("\x00" in part for part in value):
            raise ValueError("argv values cannot contain NUL bytes")
        return value

    @field_validator("env")
    @classmethod
    def validate_environment(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > MAX_ENVIRONMENT_VARIABLES:
            raise ValueError("too many environment variables")
        for name, content in value.items():
            if (
                not name
                or not name.replace("_", "A").isalnum()
                or not name[0].isalpha()
                or len(content) > MAX_ENVIRONMENT_VALUE_CHARS
                or "\x00" in content
            ):
                raise ValueError("environment names and values must be bounded")
        return value


class SandboxExecutionResult(BaseModel):
    """Bounded result from one host or container process."""

    model_config = ConfigDict(extra="forbid")

    status: SandboxStatus
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = Field(ge=0)
    backend: SandboxBackendName
    timed_out: bool = False
    output_truncated: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    error_kind: Literal[
        "sandbox_unavailable",
        "invalid_workspace",
        "policy_rejected",
        "start_failure",
        "cleanup_failure",
    ] | None = None
    provenance: SandboxProvenance
