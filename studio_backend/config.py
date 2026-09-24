"""Strict environment configuration for the local RepoGraph Studio."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_ALLOWED_ORIGIN = "http://localhost:3000"
DEFAULT_MAX_CONCURRENT_RUNS = 2
HARD_MAX_CONCURRENT_RUNS = 4
MAX_STUDIO_EVENT_METADATA_CHARS = 20_000
MAX_STUDIO_ARTIFACT_CHARS = 1_000_000
MAX_STUDIO_EVENT_MESSAGE_CHARS = 4_000


def _required_directory(name: str, *, create: bool = False) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required.")
    path = Path(value).expanduser()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise RuntimeError(f"{name} must reference an accessible directory.") from error
    if not resolved.is_dir():
        raise RuntimeError(f"{name} must reference a directory.")
    return resolved


def _allowed_origin() -> str:
    value = os.environ.get(
        "REPOGRAPH_STUDIO_ALLOWED_ORIGIN",
        DEFAULT_ALLOWED_ORIGIN,
    ).strip()
    parsed = urlsplit(value)
    if (
        value == "*"
        or parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError(
            "REPOGRAPH_STUDIO_ALLOWED_ORIGIN must be one explicit HTTP origin."
        )
    return value.rstrip("/")


def _max_concurrent_runs() -> int:
    raw = os.environ.get(
        "REPOGRAPH_STUDIO_MAX_CONCURRENT_RUNS",
        str(DEFAULT_MAX_CONCURRENT_RUNS),
    )
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(
            "REPOGRAPH_STUDIO_MAX_CONCURRENT_RUNS must be an integer."
        ) from error
    if not 1 <= value <= HARD_MAX_CONCURRENT_RUNS:
        raise RuntimeError(
            "REPOGRAPH_STUDIO_MAX_CONCURRENT_RUNS must be between 1 and "
            f"{HARD_MAX_CONCURRENT_RUNS}."
        )
    return value


@dataclass(frozen=True)
class StudioConfig:
    """Resolved process configuration with filesystem details kept server-side."""

    workspace_root: Path
    data_dir: Path
    allowed_origin: str
    max_concurrent_runs: int = DEFAULT_MAX_CONCURRENT_RUNS

    @classmethod
    def from_env(cls) -> StudioConfig:
        workspace_root = _required_directory("REPOGRAPH_WORKSPACE_ROOT")
        data_dir = _required_directory(
            "REPOGRAPH_STUDIO_DATA_DIR",
            create=True,
        )
        if data_dir == workspace_root or data_dir.is_relative_to(workspace_root):
            raise RuntimeError(
                "REPOGRAPH_STUDIO_DATA_DIR must be outside the repository workspace."
            )
        return cls(
            workspace_root=workspace_root,
            data_dir=data_dir,
            allowed_origin=_allowed_origin(),
            max_concurrent_runs=_max_concurrent_runs(),
        )

    @property
    def database_path(self) -> Path:
        return self.data_dir / "repograph-studio.sqlite3"
