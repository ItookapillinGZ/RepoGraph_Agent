"""Deterministic non-secret source tree fingerprint for campaign provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path

_EXCLUDED_PARTS = {
    ".evaluation",
    ".git",
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "test-results",
}
_EXCLUDED_NAMES = {
    ".env",
    ".env.local",
    "evaluation.db",
}
_EXCLUDED_SUFFIXES = {
    ".db",
    ".log",
    ".pyc",
    ".sqlite",
    ".sqlite3",
}


def source_tree_digest(root: str | Path) -> str:
    """Hash relative paths and bytes while excluding generated/sensitive state."""

    source = Path(root).resolve(strict=True)
    digest = hashlib.sha256()
    for path in sorted(
        (item for item in source.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(source).as_posix(),
    ):
        relative = path.relative_to(source)
        if relative.parts[:2] == ("evaluation", "campaigns"):
            continue
        if any(part in _EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.name.casefold() in _EXCLUDED_NAMES:
            continue
        if path.suffix.casefold() in _EXCLUDED_SUFFIXES:
            continue
        rendered = relative.as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(rendered).to_bytes(8, "big"))
        digest.update(rendered)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()
