"""Explicit operator commands for Docker sandbox setup and diagnostics."""

from __future__ import annotations

import argparse
import os
import subprocess  # nosec B404
import sys
from pathlib import Path

from sandbox.doctor import inspect_sandbox
from sandbox.policy import SandboxPolicy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sandbox")
    parser.add_argument("command", choices=("doctor", "build", "self-test"))
    args = parser.parse_args(argv)
    policy = SandboxPolicy.from_env("docker")
    if args.command == "doctor":
        result = inspect_sandbox(policy)
        print("Docker CLI available: " + ("yes" if result.docker_cli_available else "no"))
        print("Docker daemon available: " + ("yes" if result.docker_daemon_available else "no"))
        print("Sandbox image available: " + ("yes" if result.sandbox_image_available else "no"))
        print("Sandbox image ID: " + (result.sandbox_image_id or "unavailable"))
        print("Sandbox smoke available: " + ("yes" if result.sandbox_smoke_available else "no"))
        return 0 if result.sandbox_smoke_available else 2
    if args.command == "self-test":
        environment = os.environ.copy()
        environment["REPOGRAPH_RUN_DOCKER_ACCEPTANCE"] = "1"
        completed = subprocess.run(  # nosec B603
            [sys.executable, "-m", "unittest", "tests.test_sandbox_acceptance", "-v"],
            shell=False, check=False, env=environment,
        )
        return completed.returncode
    dockerfile = Path(__file__).resolve().parent.parent / "docker" / "repograph-sandbox.Dockerfile"
    completed = subprocess.run(  # nosec B603, B607
        ["docker", "build", "--file", str(dockerfile), "--tag", policy.image, str(dockerfile.parent.parent)],
        shell=False, check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
