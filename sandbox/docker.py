"""Docker CLI backend with a deterministic, fail-closed security policy."""

from __future__ import annotations

import subprocess  # nosec B404
import tempfile
import time
import uuid
from pathlib import Path

from sandbox.models import (
    SandboxExecutionRequest,
    SandboxExecutionResult,
    SandboxProvenance,
)
from sandbox.policy import SandboxPolicy

CONTAINER_WORKSPACE = "/workspace"
CONTAINER_TMPFS = "/tmp:rw,noexec,nosuid,size=64m"  # nosec B108


class DockerSandboxBackend:
    """Execute one workspace in a disposable resource-constrained container."""

    def __init__(
        self,
        policy: SandboxPolicy,
        *,
        allowed_workspace_root: str | Path,
        source_repository_root: str | Path | None = None,
    ) -> None:
        if policy.backend != "docker":
            raise ValueError("DockerSandboxBackend requires backend='docker'.")
        self.policy = policy
        self.allowed_workspace_root = Path(allowed_workspace_root).resolve(strict=True)
        self.source_repository_root = (
            Path(source_repository_root).resolve(strict=True)
            if source_repository_root is not None
            else None
        )

    def _workspace(self, raw: str) -> Path:
        candidate = Path(raw)
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise ValueError("Sandbox workspace is unavailable.") from error
        if not resolved.is_dir():
            raise ValueError("Sandbox workspace must be a directory.")
        if resolved != self.allowed_workspace_root and self.allowed_workspace_root not in resolved.parents:
            raise ValueError("Sandbox workspace is outside the allowed disposable root.")
        current = resolved
        while current != self.allowed_workspace_root:
            if current.is_symlink():
                raise ValueError("Sandbox workspace cannot contain a symlink escape.")
            current = current.parent
        source = self.source_repository_root
        if source is not None and (resolved == source or source in resolved.parents):
            raise ValueError("The source repository cannot be mounted into the sandbox.")
        return resolved

    def _container_argv(
        self,
        request: SandboxExecutionRequest,
        workspace: Path,
        container_name: str,
    ) -> list[str]:
        timeout = min(request.timeout_seconds, self.policy.max_timeout_seconds)
        command = _container_command(request.argv)
        argv = [
            "docker", "run", "--rm", "--name", container_name,
            "--network", "none",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", str(self.policy.pids_limit),
            "--memory", f"{self.policy.memory_limit_mb}m",
            "--cpus", f"{self.policy.cpu_limit:g}",
            "--read-only",
            "--tmpfs", CONTAINER_TMPFS,
            "--mount", f"type=bind,src={workspace},dst={CONTAINER_WORKSPACE}",
            "--workdir", CONTAINER_WORKSPACE,
        ]
        for name in sorted(request.env):
            if name not in self.policy.allowed_environment:
                raise ValueError(f"Container environment variable is not allowlisted: {name}")
            argv.extend(["--env", f"{name}={request.env[name]}"])
        argv.extend([self.policy.image, *command])
        if timeout <= 0:  # defensive; schemas and policy already reject this
            raise ValueError("Sandbox timeout must be positive.")
        return argv

    def _image_id(self) -> str | None:
        try:
            completed = subprocess.run(  # nosec B603, B607
                ["docker", "image", "inspect", "--format", "{{.Id}}", self.policy.image],
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return completed.stdout.strip()[:500] if completed.returncode == 0 else None

    def run(self, request: SandboxExecutionRequest) -> SandboxExecutionResult:
        request = SandboxExecutionRequest.model_validate(request)
        started = time.monotonic()
        timeout = min(request.timeout_seconds, self.policy.max_timeout_seconds)
        image_id = self._image_id()
        provenance = self._provenance(timeout, image_id)
        if image_id is None:
            return SandboxExecutionResult(
                status="sandbox_error",
                stderr="Docker CLI, daemon, or configured sandbox image is unavailable.",
                duration_seconds=time.monotonic() - started,
                backend="docker",
                error_kind="sandbox_unavailable",
                provenance=provenance,
            )
        try:
            workspace = self._workspace(request.workspace)
        except ValueError as error:
            return SandboxExecutionResult(
                status="sandbox_error",
                stderr=str(error),
                duration_seconds=time.monotonic() - started,
                backend="docker",
                error_kind="invalid_workspace",
                provenance=provenance,
            )
        name = f"repograph-sbx-{uuid.uuid4().hex}"
        try:
            argv = self._container_argv(request, workspace, name)
        except ValueError as error:
            return SandboxExecutionResult(
                status="sandbox_error",
                stderr=str(error),
                duration_seconds=time.monotonic() - started,
                backend="docker",
                error_kind="policy_rejected",
                provenance=provenance,
            )
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            try:
                process = subprocess.Popen(  # nosec B603
                    argv,
                    shell=False,
                    stdout=stdout_file,
                    stderr=stderr_file,
                )
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _cleanup_container(name)
                try:
                    process.kill()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                stdout, out_cut = _read_bounded(stdout_file, self.policy.max_output_chars)
                stderr, err_cut = _read_bounded(stderr_file, self.policy.max_output_chars)
                return SandboxExecutionResult(
                    status="timeout", stdout=stdout, stderr=stderr,
                    duration_seconds=time.monotonic() - started, backend="docker",
                    timed_out=True, output_truncated=out_cut or err_cut,
                    stdout_truncated=out_cut, stderr_truncated=err_cut,
                    provenance=provenance,
                )
            except OSError as error:
                _cleanup_container(name)
                return SandboxExecutionResult(
                    status="sandbox_error", stderr=f"Docker process could not start: {error}",
                    duration_seconds=time.monotonic() - started, backend="docker",
                    error_kind="sandbox_unavailable", provenance=provenance,
                )
            stdout, out_cut = _read_bounded(stdout_file, self.policy.max_output_chars)
            stderr, err_cut = _read_bounded(stderr_file, self.policy.max_output_chars)
        return SandboxExecutionResult(
            status="passed" if process.returncode == 0 else "failed",
            exit_code=process.returncode, stdout=stdout, stderr=stderr,
            duration_seconds=time.monotonic() - started, backend="docker",
            output_truncated=out_cut or err_cut, provenance=provenance,
            stdout_truncated=out_cut, stderr_truncated=err_cut,
        )

    def _provenance(self, timeout: float, image_id: str | None) -> SandboxProvenance:
        return SandboxProvenance(
            backend="docker", sandboxed=True, image=self.policy.image,
            image_id=image_id, network="none", read_only_root=True,
            cap_drop_all=True, no_new_privileges=True,
            memory_limit_mb=self.policy.memory_limit_mb,
            cpu_limit=self.policy.cpu_limit, pids_limit=self.policy.pids_limit,
            timeout_seconds=timeout,
        )


def _container_command(argv: list[str]) -> list[str]:
    first = Path(argv[0]).name.casefold()
    aliases = {"py", "py.exe", "python", "python.exe", "python3", "python3.exe"}
    if first in aliases or first.startswith("python"):
        return ["python", *argv[1:]]
    return list(argv)


def _read_bounded(handle, limit: int) -> tuple[str, bool]:
    handle.flush()
    handle.seek(0)
    content = handle.read(limit + 1)
    return content[:limit].decode("utf-8", errors="replace"), len(content) > limit


def _cleanup_container(name: str) -> None:
    for argv in (
        ["docker", "stop", "--time", "1", name],
        ["docker", "kill", name],
        ["docker", "rm", "--force", name],
    ):
        try:
            subprocess.run(  # nosec B603
                argv, shell=False, capture_output=True, check=False, timeout=5
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
