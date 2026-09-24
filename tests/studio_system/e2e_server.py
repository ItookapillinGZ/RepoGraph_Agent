"""Dedicated deterministic FastAPI server for the Playwright smoke test."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import uvicorn

from tests.studio_system.fixtures import (
    build_test_app,
    file_hash,
    initialize_repository,
)


def _prewarm_static_tools(repository: Path) -> None:
    """Pay Windows process-startup cost before Playwright starts Next.js."""

    target = repository / "src" / "calculator.py"
    commands = [
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--output-format",
            "json",
            "--no-cache",
            "--target-version",
            "py39",
            str(target),
        ],
        [sys.executable, "-m", "bandit", "-f", "json", str(target)],
    ]
    for command in commands:
        subprocess.run(  # nosec B603
            command,
            capture_output=True,
            timeout=60,
            check=False,
        )

def _runtime_root() -> Path:
    value = os.environ.get("REPOGRAPH_E2E_RUNTIME", "").strip()
    if not value:
        raise RuntimeError("REPOGRAPH_E2E_RUNTIME is required.")
    root = Path(value).resolve()
    if not root.name.startswith("repograph-studio-e2e-"):
        raise RuntimeError("The E2E runtime directory name is not safely scoped.")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


RUNTIME_ROOT = _runtime_root()
WORKSPACE = RUNTIME_ROOT / "workspace"
WORKSPACE.mkdir()
REPOSITORY, _REMOTE, INITIAL_SHA = initialize_repository(WORKSPACE)
_prewarm_static_tools(REPOSITORY)
STATE_PATH = RUNTIME_ROOT / "state.json"
STATE_PATH.write_text(
    json.dumps(
        {
            "repository": str(REPOSITORY.resolve()),
            "initial_sha": INITIAL_SHA,
            "initial_index_hash": file_hash(REPOSITORY / ".git" / "index"),
        },
        sort_keys=True,
    ),
    encoding="utf-8",
)
app = build_test_app(WORKSPACE, RUNTIME_ROOT / "data")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="warning",
    )
