"""Read-only Docker sandbox prerequisite diagnostics."""

from __future__ import annotations

import shutil
import subprocess  # nosec B404

from pydantic import BaseModel, ConfigDict

from sandbox.policy import SandboxPolicy

DOCTOR_TMPFS = "/tmp:rw,noexec,nosuid,size=16m"  # nosec B108


class SandboxDoctorResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    docker_cli_available: bool
    docker_daemon_available: bool
    sandbox_image_available: bool
    sandbox_image_id: str | None = None
    sandbox_smoke_available: bool = False


def inspect_sandbox(policy: SandboxPolicy) -> SandboxDoctorResult:
    cli = shutil.which("docker") is not None
    if not cli:
        return SandboxDoctorResult(
            docker_cli_available=False, docker_daemon_available=False,
            sandbox_image_available=False,
        )
    try:
        daemon = subprocess.run(  # nosec B603, B607
            ["docker", "info", "--format", "{{.ServerVersion}}"], shell=False,
            capture_output=True, text=True, check=False, timeout=10,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        daemon = False
    if not daemon:
        return SandboxDoctorResult(
            docker_cli_available=True, docker_daemon_available=False,
            sandbox_image_available=False,
        )
    try:
        image = subprocess.run(  # nosec B603, B607
            ["docker", "image", "inspect", "--format", "{{.Id}}", policy.image],
            shell=False, capture_output=True, text=True, check=False, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        image = None
    image_id = image.stdout.strip()[:500] if image and image.returncode == 0 else None
    smoke = False
    if image_id is not None:
        try:
            smoke_result = subprocess.run(  # nosec B603, B607
                [
                    "docker", "run", "--rm", "--network", "none",
                    "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                    "--pids-limit", "16", "--memory", "64m", "--cpus", "0.25",
                    "--read-only", "--tmpfs",
                    DOCTOR_TMPFS,
                    policy.image, "python", "--version",
                ],
                shell=False, capture_output=True, text=True, check=False, timeout=20,
            )
            smoke = smoke_result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            smoke = False
    return SandboxDoctorResult(
        docker_cli_available=True, docker_daemon_available=True,
        sandbox_image_available=image_id is not None, sandbox_image_id=image_id,
        sandbox_smoke_available=smoke,
    )
