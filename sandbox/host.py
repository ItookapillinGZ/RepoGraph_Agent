"""Backward-compatible host execution; explicitly not a security sandbox."""

from __future__ import annotations

import os
import subprocess  # nosec B404
import time
from pathlib import Path

from sandbox.models import (
    SandboxExecutionRequest,
    SandboxExecutionResult,
    SandboxProvenance,
)
from sandbox.policy import SandboxPolicy


class HostSandboxBackend:
    """Run fixed argv on the host for compatibility and unit tests."""

    def __init__(self, policy: SandboxPolicy | None = None) -> None:
        self.policy = policy or SandboxPolicy()

    def run(self, request: SandboxExecutionRequest) -> SandboxExecutionResult:
        request = SandboxExecutionRequest.model_validate(request)
        started = time.monotonic()
        timeout = min(request.timeout_seconds, self.policy.max_timeout_seconds)
        provenance = SandboxProvenance(
            backend="host",
            sandboxed=False,
            network="host",
            read_only_root=False,
            cap_drop_all=False,
            no_new_privileges=False,
            timeout_seconds=timeout,
        )
        environment = os.environ.copy()
        environment.update(request.env)
        try:
            completed = subprocess.run(  # nosec B603
                request.argv,
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=timeout,
                cwd=str(Path(request.workspace).resolve(strict=True)),
                env=environment,
            )
        except subprocess.TimeoutExpired as error:
            stdout, out_cut = _bounded(error.stdout, self.policy.max_output_chars)
            stderr, err_cut = _bounded(error.stderr, self.policy.max_output_chars)
            return SandboxExecutionResult(
                status="timeout",
                stdout=stdout,
                stderr=stderr,
                duration_seconds=time.monotonic() - started,
                backend="host",
                timed_out=True,
                output_truncated=out_cut or err_cut,
                stdout_truncated=out_cut,
                stderr_truncated=err_cut,
                provenance=provenance,
            )
        except OSError as error:
            return SandboxExecutionResult(
                status="sandbox_error",
                stderr=f"Host process could not start: {error}",
                duration_seconds=time.monotonic() - started,
                backend="host",
                error_kind="start_failure",
                provenance=provenance,
            )
        stdout, out_cut = _bounded(completed.stdout, self.policy.max_output_chars)
        stderr, err_cut = _bounded(completed.stderr, self.policy.max_output_chars)
        return SandboxExecutionResult(
            status="passed" if completed.returncode == 0 else "failed",
            exit_code=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=time.monotonic() - started,
            backend="host",
            output_truncated=out_cut or err_cut,
            stdout_truncated=out_cut,
            stderr_truncated=err_cut,
            provenance=provenance,
        )


def _bounded(value: str | bytes | None, limit: int) -> tuple[str, bool]:
    if value is None:
        return "", False
    rendered = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    return (rendered[:limit], len(rendered) > limit)
