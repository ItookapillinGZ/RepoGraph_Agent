"""Deterministic release-readiness command orchestration."""

from __future__ import annotations

import subprocess  # nosec B404
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(name: str, argv: list[str], *, cwd: Path = PROJECT_ROOT) -> bool:
    print(f"\n== {name} ==")
    result = subprocess.run(argv, cwd=cwd, check=False, shell=False)  # nosec B603
    print(f"{name}: {'PASS' if result.returncode == 0 else 'FAIL'}")
    return result.returncode == 0


def run_release_check(*, skip_studio: bool = False, skip_docker: bool = False) -> int:
    python = sys.executable
    checks = [
        _run("pip check", [python, "-m", "pip", "check"]),
        _run("Python tests", [python, "-m", "unittest", "discover", "-s", "tests", "-q"]),
        _run("Ruff", [python, "-m", "ruff", "check", "."]),
        _run("Public hygiene", [python, "-m", "repograph", "hygiene"]),
        _run(
            "Bandit",
            [python, "-m", "bandit", "-r", ".", "-q", "-x", ".venv,tests,studio"],
        ),
        _run(
            "Documentation entrypoints",
            [python, "-m", "unittest", "tests.test_h4_productization", "-q"],
        ),
    ]
    if not skip_docker:
        checks.append(_run("Sandbox doctor", [python, "-m", "sandbox", "doctor"]))
        checks.append(_run(
            "Deterministic Docker demo",
            [python, "-m", "repograph", "demo", "--sandbox", "docker", "--fresh"],
        ))
    if not skip_studio:
        npm = "npm.cmd" if sys.platform == "win32" else "npm"
        checks.extend([
            _run("Studio lint", [npm, "run", "lint"], cwd=PROJECT_ROOT / "studio"),
            _run("Studio typecheck", [npm, "run", "typecheck"], cwd=PROJECT_ROOT / "studio"),
            _run("Studio build", [npm, "run", "build"], cwd=PROJECT_ROOT / "studio"),
            _run("Studio E2E", [npm, "run", "e2e"], cwd=PROJECT_ROOT / "studio"),
        ])
    print("\nRelease check: " + ("PASS" if all(checks) else "FAIL"))
    return 0 if all(checks) else 1
