"""Bounded secret scanning for persisted evaluation artifacts."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from evaluation.security import SECRET_ENVIRONMENT_KEYS

_OPENAI_KEY_LIKE = re.compile(
    rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_SCANNED_SUFFIXES = frozenset(
    {".db", ".json", ".jsonl", ".log", ".md", ".sqlite", ".sqlite3", ".txt"}
)
_SQLITE_SIDECAR_SUFFIXES = (
    ".db-journal",
    ".db-wal",
    ".sqlite-journal",
    ".sqlite-wal",
    ".sqlite3-journal",
    ".sqlite3-wal",
)
MAX_SECURITY_SCAN_FILE_BYTES = 20_000_000
MAX_SECURITY_SCAN_TOTAL_BYTES = 250_000_000


class ArtifactSecurityError(RuntimeError):
    """Raised without reproducing secret material in the error message."""


class ArtifactSecurityScan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files_scanned: int = Field(ge=0)
    bytes_scanned: int = Field(ge=0)
    secret_detected: bool = False
    rules: list[str] = Field(default_factory=list)


def _secret_values(environment: Mapping[str, str]) -> tuple[bytes, ...]:
    values: set[bytes] = set()
    for key, value in environment.items():
        if key.upper() in SECRET_ENVIRONMENT_KEYS and value:
            encoded = value.encode("utf-8")
            if encoded:
                values.add(encoded)
    return tuple(sorted(values))


def assert_secret_free_payload(
    payload: str | bytes,
    environment: Mapping[str, str] | None = None,
) -> None:
    """Reject exact configured secrets and OpenAI-like keys without echoing them."""

    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    selected = os.environ if environment is None else environment
    if any(secret in data for secret in _secret_values(selected)):
        raise ArtifactSecurityError(
            "Persisted evaluation payload contains a configured secret."
        )
    if _OPENAI_KEY_LIKE.search(data):
        raise ArtifactSecurityError(
            "Persisted evaluation payload contains key-like material."
        )


def _artifact_files(roots: Iterable[str | Path]) -> list[Path]:
    files: set[Path] = set()
    for root_value in roots:
        root = Path(root_value).resolve()
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else root.rglob("*")
        for candidate in candidates:
            lowered_name = candidate.name.casefold()
            if (
                candidate.is_file()
                and not candidate.is_symlink()
                and (
                    candidate.suffix.casefold() in _SCANNED_SUFFIXES
                    or lowered_name.endswith(_SQLITE_SIDECAR_SUFFIXES)
                )
            ):
                files.add(candidate.resolve())
    return sorted(files, key=lambda item: item.as_posix())


def scan_persisted_artifacts(
    roots: Iterable[str | Path],
    environment: Mapping[str, str] | None = None,
) -> ArtifactSecurityScan:
    """Scan bounded text artifacts; never return or print matched secret bytes."""

    selected = os.environ if environment is None else environment
    total = 0
    files = _artifact_files(roots)
    for path in files:
        size = path.stat().st_size
        if size > MAX_SECURITY_SCAN_FILE_BYTES:
            raise ArtifactSecurityError(
                f"Persisted artifact exceeds security scan file bound: {path.name}"
            )
        total += size
        if total > MAX_SECURITY_SCAN_TOTAL_BYTES:
            raise ArtifactSecurityError(
                "Persisted artifact security scan exceeded total bound."
            )
        assert_secret_free_payload(path.read_bytes(), selected)
    return ArtifactSecurityScan(
        files_scanned=len(files),
        bytes_scanned=total,
        rules=["configured-secret-exact-match", "openai-key-like-boundary"],
    )
