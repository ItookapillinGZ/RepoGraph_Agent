"""Bounded public-source secret and personal-path hygiene scan."""

from __future__ import annotations

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEXT_SUFFIXES = frozenset(
    {".py", ".md", ".ps1", ".ts", ".tsx", ".js", ".mjs", ".json", ".txt", ".toml", ".yaml", ".yml"}
)
PUBLIC_ROOTS = (
    "repograph", "sandbox", "observability", "studio_backend", "studio/app",
    "studio/components", "studio/lib", "studio/e2e", "scripts", "demo", "docs",
)
PUBLIC_FILES = (
    "README.md", "SECURITY.md", ".env.example", "requirements.txt",
    "requirements-eval.txt", "docker/repograph-sandbox.Dockerfile",
)
SECRET_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}(?![A-Za-z0-9])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9_-]{12,}|github_pat_[A-Za-z0-9_]{12,})", re.IGNORECASE),
    re.compile(r"(?<![A-Z0-9])AKIA[A-Z0-9]{16}(?![A-Z0-9])"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{16,}={0,2}"),
    re.compile(r"(?i)(?:https?://)[^\s:/]+:[^\s/@]+@"),
)
PERSONAL_PATH = re.compile(r"(?i)\b[A-Z]:[\\/]Users[\\/][^\\/\s]+[\\/]")
SECRET_ENV_NAMES = (
    "CODE_REVIEW_LLM_API_KEY", "PDE_FRONTIER_LLM_API_KEY", "OPENAI_API_KEY",
    "GITHUB_TOKEN", "GH_TOKEN", "AWS_SECRET_ACCESS_KEY",
)


def _files() -> list[Path]:
    selected: set[Path] = set()
    for relative in PUBLIC_FILES:
        path = PROJECT_ROOT / relative
        if path.is_file():
            selected.add(path)
    for relative in PUBLIC_ROOTS:
        root = PROJECT_ROOT / relative
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and not path.is_symlink() and (
                path.suffix.casefold() in TEXT_SUFFIXES
                or path.name == "repograph-sandbox.Dockerfile"
            ):
                selected.add(path)
    return sorted(selected)


def scan_public_tree() -> tuple[int, int]:
    files = _files()
    secret_hits = 0
    path_hits = 0
    exact_secrets = tuple(
        value for name in SECRET_ENV_NAMES if (value := os.environ.get(name, ""))
    )
    for path in files:
        text = path.read_text(encoding="utf-8", errors="strict")
        if any(secret in text for secret in exact_secrets):
            secret_hits += 1
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            secret_hits += 1
        if PERSONAL_PATH.search(text):
            path_hits += 1
    print(f"Files scanned: {len(files)}")
    print(f"Credential hits: {secret_hits}")
    print(f"Personal-path hits: {path_hits}")
    return secret_hits, path_hits


def main() -> int:
    secret_hits, path_hits = scan_public_tree()
    return 0 if secret_hits == 0 and path_hits == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
