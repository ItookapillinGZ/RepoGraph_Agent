"""Backend selection and disposable-workspace orchestration."""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path
from typing import Literal

from observability.recorder import trace_event, traced_span
from sandbox.docker import DockerSandboxBackend
from sandbox.host import HostSandboxBackend
from sandbox.models import SandboxExecutionRequest, SandboxExecutionResult
from sandbox.policy import SandboxPolicy
from temporary_workspace import WorkspaceCopyError, copy_repository_bounded


def policy_from_backend(backend: Literal["host", "docker"] = "host") -> SandboxPolicy:
    return SandboxPolicy.from_env(backend)


class SandboxRunner:
    """Route execution without ever falling back between backends."""

    def __init__(self, policy: SandboxPolicy | None = None) -> None:
        self.policy = policy or SandboxPolicy()

    def run_repository(
        self,
        *,
        argv: list[str],
        repository_root: str | Path,
        timeout_seconds: float,
        purpose: Literal["tests", "static_analysis", "evaluation"],
        env: dict[str, str] | None = None,
    ) -> SandboxExecutionResult:
        with traced_span(
            "sandbox_execution",
            kind="sandbox_execution",
            metadata={
                "backend": self.policy.backend,
                "sandboxed": self.policy.backend == "docker",
                "image": self.policy.image if self.policy.backend == "docker" else None,
                "timeout": min(timeout_seconds, self.policy.max_timeout_seconds),
                "memory_mb": self.policy.memory_limit_mb,
                "cpu": self.policy.cpu_limit,
                "pids": self.policy.pids_limit,
                "network": "none" if self.policy.backend == "docker" else "host",
                "purpose": purpose,
            },
        ):
            result = self._run_repository(
                argv=argv,
                repository_root=repository_root,
                timeout_seconds=timeout_seconds,
                purpose=purpose,
                env=env,
            )
            trace_event(
                "sandbox_execution_completed",
                kind="sandbox_execution",
                status=result.status,
                metadata={
                    "backend": result.backend,
                    "sandboxed": result.provenance.sandboxed,
                    "image_id": result.provenance.image_id,
                    "exit_code": result.exit_code,
                    "duration_seconds": result.duration_seconds,
                    "timed_out": result.timed_out,
                    "output_truncated": result.output_truncated,
                    "stdout_size": len(result.stdout),
                    "stderr_size": len(result.stderr),
                    "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
                    "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
                },
            )
            return result

    def _run_repository(
        self,
        *,
        argv: list[str],
        repository_root: str | Path,
        timeout_seconds: float,
        purpose: Literal["tests", "static_analysis", "evaluation"],
        env: dict[str, str] | None = None,
    ) -> SandboxExecutionResult:
        root = Path(repository_root).resolve(strict=True)
        if self.policy.backend == "host":
            return HostSandboxBackend(self.policy).run(
                SandboxExecutionRequest(
                    argv=argv, workspace=str(root), timeout_seconds=timeout_seconds,
                    env=env or {}, purpose=purpose,
                )
            )
        with tempfile.TemporaryDirectory(prefix="repograph-sandbox-") as temporary:
            base = Path(temporary).resolve(strict=True)
            workspace = base / "workspace"
            try:
                copy_repository_bounded(root, workspace)
                _make_container_workspace_writable(workspace)
            except WorkspaceCopyError as error:
                request = SandboxExecutionRequest(
                    argv=argv, workspace=str(base), timeout_seconds=timeout_seconds,
                    env=env or {}, purpose=purpose,
                )
                backend = DockerSandboxBackend(
                    self.policy, allowed_workspace_root=base,
                    source_repository_root=root,
                )
                provenance = backend._provenance(
                    min(timeout_seconds, self.policy.max_timeout_seconds), None
                )
                return SandboxExecutionResult(
                    status="sandbox_error", stderr=str(error), duration_seconds=0,
                    backend="docker", error_kind="invalid_workspace",
                    provenance=provenance,
                )
            request = SandboxExecutionRequest(
                argv=_portable_argv(argv), workspace=str(workspace),
                timeout_seconds=timeout_seconds, env=env or {}, purpose=purpose,
            )
            return DockerSandboxBackend(
                self.policy, allowed_workspace_root=base,
                source_repository_root=root,
            ).run(request)


def _portable_argv(argv: list[str]) -> list[str]:
    if argv and Path(argv[0]).resolve(strict=False) == Path(sys.executable).resolve():
        return ["python", *argv[1:]]
    return list(argv)


def _make_container_workspace_writable(workspace: Path) -> None:
    """Permit the fixed non-root container user to mutate only the temp copy."""

    for path in [workspace, *workspace.rglob("*")]:
        try:
            if path.is_dir():
                path.chmod(0o777)
            elif path.is_file():
                path.chmod(path.stat().st_mode | 0o666)
        except OSError as error:
            raise WorkspaceCopyError(
                "Disposable workspace permissions could not be prepared."
            ) from error
