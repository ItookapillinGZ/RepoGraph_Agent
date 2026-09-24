"""Trusted operator policy for host and Docker execution."""

from __future__ import annotations

import os
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_SANDBOX_IMAGE = "repograph-sandbox:local"
HARD_MAX_TIMEOUT_SECONDS = 3_600.0
HARD_MAX_MEMORY_MB = 4_096
HARD_MAX_CPUS = 4.0
HARD_MAX_PIDS = 256
HARD_MAX_OUTPUT_CHARS = 1_000_000
SAFE_CONTAINER_ENV = frozenset({"PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE"})
_IMAGE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:@-]{0,499}$")


class SandboxPolicy(BaseModel):
    """Validated policy that cannot be widened by repository or model text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: Literal["host", "docker"] = "host"
    image: str = DEFAULT_SANDBOX_IMAGE
    max_timeout_seconds: float = Field(default=300.0, gt=0, le=HARD_MAX_TIMEOUT_SECONDS)
    memory_limit_mb: int = Field(default=512, ge=64, le=HARD_MAX_MEMORY_MB)
    cpu_limit: float = Field(default=1.0, gt=0, le=HARD_MAX_CPUS)
    pids_limit: int = Field(default=64, ge=16, le=HARD_MAX_PIDS)
    max_output_chars: int = Field(default=20_000, ge=1, le=HARD_MAX_OUTPUT_CHARS)
    allowed_environment: frozenset[str] = SAFE_CONTAINER_ENV
    network_enabled: Literal[False] = False
    read_only_root: Literal[True] = True
    cap_drop_all: Literal[True] = True
    no_new_privileges: Literal[True] = True

    @field_validator("image")
    @classmethod
    def validate_image(cls, value: str) -> str:
        if not _IMAGE_PATTERN.fullmatch(value):
            raise ValueError("image must be one bounded Docker image reference")
        return value

    @field_validator("allowed_environment")
    @classmethod
    def validate_environment_allowlist(cls, value: frozenset[str]) -> frozenset[str]:
        if not value.issubset(SAFE_CONTAINER_ENV):
            raise ValueError("container environment may only use the fixed safe allowlist")
        return value

    @classmethod
    def from_env(cls, backend: Literal["host", "docker"] = "host") -> SandboxPolicy:
        """Load trusted process configuration, never repository file contents."""

        def integer(name: str, default: int) -> int:
            raw = os.environ.get(name, str(default))
            try:
                return int(raw)
            except ValueError as error:
                raise ValueError(f"{name} must be an integer") from error

        def number(name: str, default: float) -> float:
            raw = os.environ.get(name, str(default))
            try:
                return float(raw)
            except ValueError as error:
                raise ValueError(f"{name} must be numeric") from error

        return cls(
            backend=backend,
            image=os.environ.get("REPOGRAPH_SANDBOX_IMAGE", DEFAULT_SANDBOX_IMAGE),
            max_timeout_seconds=number("REPOGRAPH_SANDBOX_MAX_TIMEOUT_SECONDS", 300.0),
            memory_limit_mb=integer("REPOGRAPH_SANDBOX_MEMORY_MB", 512),
            cpu_limit=number("REPOGRAPH_SANDBOX_CPUS", 1.0),
            pids_limit=integer("REPOGRAPH_SANDBOX_PIDS", 64),
            max_output_chars=integer("REPOGRAPH_SANDBOX_MAX_OUTPUT_CHARS", 20_000),
        )
