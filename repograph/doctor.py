"""Secret-safe prerequisite diagnostics for local RepoGraph development."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess  # nosec B404
import sys
from dataclasses import dataclass
from pathlib import Path

from sandbox.doctor import inspect_sandbox
from sandbox.policy import SandboxPolicy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_MODULES = (
    "langchain_core", "langchain_openai", "langgraph", "pydantic",
    "fastapi", "uvicorn", "pytest", "ruff", "bandit",
)


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def _configured(*names: str) -> bool:
    return any(bool(os.environ.get(name, "").strip()) for name in names)


def _command_version(command: str, *arguments: str) -> tuple[bool, str]:
    executable = shutil.which(command)
    if executable is None:
        return False, "not found"
    try:
        result = subprocess.run(  # nosec B603
            [executable, *arguments], capture_output=True, text=True,
            check=False, timeout=8,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, "could not run"
    output = (result.stdout or result.stderr).strip().splitlines()
    return result.returncode == 0, (output[0][:120] if output else "available")


def collect_checks() -> list[Check]:
    checks: list[Check] = []
    supported = sys.version_info >= (3, 10)
    checks.append(Check(
        "Python", "PASS" if supported else "BLOCKED",
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    ))
    in_venv = sys.prefix != sys.base_prefix
    checks.append(Check(
        "Virtual environment", "PASS" if in_venv else "WARN",
        "active" if in_venv else "not active; use .venv",
    ))
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    checks.append(Check(
        "Python packages", "PASS" if not missing else "BLOCKED",
        "installed" if not missing else "missing: " + ", ".join(missing),
    ))
    for label, command in (("Node", "node"), ("npm", "npm"), ("Git", "git")):
        ok, detail = _command_version(command, "--version")
        checks.append(Check(label, "PASS" if ok else "BLOCKED", detail))
    studio_deps = (PROJECT_ROOT / "studio" / "node_modules").is_dir()
    checks.append(Check(
        "Studio dependencies", "PASS" if studio_deps else "BLOCKED",
        "installed" if studio_deps else "run npm install in studio",
    ))
    sandbox = inspect_sandbox(SandboxPolicy.from_env("docker"))
    checks.extend([
        Check("Docker CLI", "PASS" if sandbox.docker_cli_available else "BLOCKED",
              "available" if sandbox.docker_cli_available else "install Docker Desktop"),
        Check("Docker daemon", "PASS" if sandbox.docker_daemon_available else "BLOCKED",
              "available" if sandbox.docker_daemon_available else "start Docker Desktop"),
        Check("Sandbox image", "PASS" if sandbox.sandbox_smoke_available else "BLOCKED",
              "ready" if sandbox.sandbox_smoke_available else "run python -m sandbox build"),
    ])
    workspace_value = os.environ.get("REPOGRAPH_WORKSPACE_ROOT", "").strip()
    workspace = Path(workspace_value).expanduser() if workspace_value else PROJECT_ROOT
    checks.append(Check(
        "Workspace root", "PASS" if workspace.is_dir() else "BLOCKED",
        "valid" if workspace.is_dir() else "missing or invalid",
    ))
    state_value = os.environ.get("REPOGRAPH_STATE_ROOT", "").strip()
    checks.append(Check(
        "State root", "PASS" if state_value else "WARN",
        "configured" if state_value else "using platform default",
    ))
    env_present = (PROJECT_ROOT / ".env").is_file()
    checks.append(Check(
        ".env", "PASS" if env_present else "WARN",
        "present" if env_present else "optional for offline use",
    ))
    openai_configured = _configured("CODE_REVIEW_LLM_API_KEY", "OPENAI_API_KEY")
    checks.append(Check(
        "OpenAI credential", "CONFIGURED" if openai_configured else "OPTIONAL",
        "configured" if openai_configured else "missing",
    ))
    gh_ok, _ = _command_version("gh", "--version")
    github_configured = _configured("GITHUB_TOKEN", "GH_TOKEN")
    checks.append(Check(
        "GitHub delivery", "CONFIGURED" if gh_ok and github_configured else "OPTIONAL",
        "configured" if gh_ok and github_configured else "gh and credential required only for G5B",
    ))
    return checks


def run_doctor() -> int:
    checks = collect_checks()
    width = max(len(item.name) for item in checks)
    for item in checks:
        print(f"{item.name:<{width}}  {item.status:<10}  {item.detail}")
    blocked = [item for item in checks if item.status == "BLOCKED"]
    if blocked:
        print("\nBLOCKED items prevent the full secure demo or Studio workflow.")
        return 2
    print("\nPASS: RepoGraph local prerequisites are ready.")
    return 0
