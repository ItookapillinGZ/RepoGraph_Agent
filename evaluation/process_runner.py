"""Fixed-argv subprocess control for isolated evaluation work."""

from __future__ import annotations

import os
import platform
import shutil
import signal
import subprocess  # nosec B404
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from evaluation.models import EvaluationWorkerRequest, EvaluationWorkerResult
from evaluation.security import redact_environment_secrets
from evaluation.workspace import cleanup_task_workspace

MAX_WORKER_REQUEST_BYTES = 2_000_000
MAX_WORKER_RESULT_BYTES = 10_000_000
MAX_WORKER_ERROR_CHARS = 4_000
MAX_PROCESS_OUTPUT_CHARS = 20_000
PROCESS_TERMINATION_GRACE_SECONDS = 2.0

_ENVIRONMENT_KEYS = {
    "APPDATA",
    "CODE_REVIEW_LLM_API_KEY",
    "CODE_REVIEW_LLM_BASE_URL",
    "CODE_REVIEW_LLM_MAX_COMPLETION_TOKENS",
    "CODE_REVIEW_LLM_MODEL",
    "CODE_REVIEW_LLM_PROVIDER",
    "CODE_REVIEW_LLM_REASONING_EFFORT",
    "CODE_REVIEW_LLM_SEED",
    "CODE_REVIEW_LLM_TEMPERATURE",
    "CODE_REVIEW_LLM_TIMEOUT_SECONDS",
    "COMSPEC",
    "HOME",
    "HOMEDRIVE",
    "HOMEPATH",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "LANG",
    "LC_ALL",
    "LOCALAPPDATA",
    "NO_PROXY",
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_ORGANIZATION",
    "OPENAI_PROJECT",
    "PATH",
    "PROGRAMDATA",
    "PDE_FRONTIER_LLM_API_KEY",
    "PDE_FRONTIER_LLM_BASE_URL",
    "PDE_FRONTIER_LLM_MAX_COMPLETION_TOKENS",
    "PDE_FRONTIER_LLM_MODEL",
    "PDE_FRONTIER_LLM_PROVIDER",
    "PDE_FRONTIER_LLM_REASONING_EFFORT",
    "PDE_FRONTIER_LLM_SEED",
    "PDE_FRONTIER_LLM_TEMPERATURE",
    "PDE_FRONTIER_LLM_TIMEOUT_SECONDS",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "WINDIR",
}


@dataclass(frozen=True)
class FixedProcessOutcome:
    returncode: int | None
    timed_out: bool
    stdout: str
    stderr: str
    output_truncated: bool
    duration_seconds: float


def worker_environment(parent: Mapping[str, str] | None = None) -> dict[str, str]:
    """Inherit only runtime and explicitly required provider settings."""

    source = os.environ if parent is None else parent
    return {
        key: value for key, value in source.items() if key.upper() in _ENVIRONMENT_KEYS
    }


def _read_bounded(handle: object, maximum: int) -> tuple[str, bool]:
    handle.seek(0)
    payload = handle.read(maximum + 1)
    truncated = len(payload) > maximum
    return payload[:maximum].decode("utf-8", errors="replace"), truncated


def terminate_process_tree(
    process: subprocess.Popen[bytes],
    *,
    grace_seconds: float = PROCESS_TERMINATION_GRACE_SECONDS,
) -> None:
    """Terminate only the worker's process group/tree on Windows and POSIX."""

    if process.poll() is not None:
        return
    if platform.system() == "Windows":
        pid = int(process.pid)
        # Windows has no portable process-tree equivalent of POSIX SIGTERM.
        # Force the complete tree while the parent/child relationship still
        # exists; killing the parent first can orphan descendants.
        try:
            subprocess.run(  # nosec B603, B607
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                timeout=max(grace_seconds, 1.0),
                check=False,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            # Cleanup failure must not replace the controller's timeout result.
            pass
        try:
            process.wait(timeout=grace_seconds)
            return
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                return
    else:
        try:
            process_group = os.getpgid(process.pid)
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=grace_seconds)
            return
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process_group, signal.SIGKILL)
            except ProcessLookupError:
                return
    try:
        process.wait(timeout=grace_seconds)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
        except OSError:
            return
        try:
            process.wait(timeout=grace_seconds)
        except (OSError, subprocess.TimeoutExpired):
            return


def run_fixed_argv(
    argv: Sequence[str],
    *,
    cwd: str | Path,
    timeout_seconds: float,
    environment: Mapping[str, str] | None = None,
    max_output_chars: int = MAX_PROCESS_OUTPUT_CHARS,
) -> FixedProcessOutcome:
    """Run trusted argv with tree termination and bounded returned output."""

    if not argv or any(not isinstance(part, str) or not part for part in argv):
        raise ValueError("argv must contain non-empty strings")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    started = time.monotonic()
    creationflags = (
        subprocess.CREATE_NEW_PROCESS_GROUP if platform.system() == "Windows" else 0
    )
    with (
        tempfile.TemporaryFile() as stdout_file,
        tempfile.TemporaryFile() as stderr_file,
    ):
        process = subprocess.Popen(  # nosec B603
            list(argv),
            cwd=str(Path(cwd).resolve()),
            env=dict(environment) if environment is not None else None,
            stdin=subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_file,
            shell=False,
            start_new_session=platform.system() != "Windows",
            creationflags=creationflags,
        )
        timed_out = False
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_tree(process)
        except KeyboardInterrupt:
            terminate_process_tree(process)
            raise
        stdout, stdout_truncated = _read_bounded(stdout_file, max_output_chars)
        stderr, stderr_truncated = _read_bounded(stderr_file, max_output_chars)
    return FixedProcessOutcome(
        returncode=process.returncode,
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
        output_truncated=stdout_truncated or stderr_truncated,
        duration_seconds=time.monotonic() - started,
    )


def run_evaluation_worker(
    request: EvaluationWorkerRequest,
    *,
    timeout_seconds: float,
    _worker_module: str = "evaluation.worker",
    _environment: Mapping[str, str] | None = None,
) -> EvaluationWorkerResult:
    """Serialize one request, spawn one worker, and strictly read its result."""

    request = EvaluationWorkerRequest.model_validate(request)
    root = Path(request.workspace_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    ipc_root = root / ".worker-ipc"
    ipc_root.mkdir(parents=True, exist_ok=True)
    ipc_directory = Path(tempfile.mkdtemp(prefix="task-", dir=ipc_root))
    request_path = ipc_directory / "worker-request.json"
    result_path = ipc_directory / "worker-result.json"
    bounded_request = request.model_copy(update={"result_path": str(result_path)})
    payload = bounded_request.model_dump_json().encode("utf-8")
    if len(payload) > MAX_WORKER_REQUEST_BYTES:
        shutil.rmtree(ipc_directory)
        return EvaluationWorkerResult(
            status="failed",
            failure_reason="Worker request exceeded MAX_WORKER_REQUEST_BYTES.",
        )
    request_path.write_bytes(payload)
    try:
        child_environment = (
            dict(_environment) if _environment is not None else worker_environment()
        )
        try:
            outcome = run_fixed_argv(
                [sys.executable, "-m", _worker_module, "--request", str(request_path)],
                cwd=Path(__file__).resolve().parents[1],
                timeout_seconds=timeout_seconds,
                environment=child_environment,
                max_output_chars=MAX_WORKER_ERROR_CHARS,
            )
        except OSError as error:
            return EvaluationWorkerResult(
                status="failed",
                failure_reason=f"Worker failed to start: {error}"[
                    :MAX_WORKER_ERROR_CHARS
                ],
            )
        if outcome.timed_out:
            try:
                cleanup_task_workspace(
                    request.workspace_root,
                    request.experiment_id,
                    request.task.id,
                )
            except OSError:
                pass
            return EvaluationWorkerResult(
                status="timeout",
                failure_reason=(
                    f"Evaluation task exceeded task_timeout_seconds="
                    f"{timeout_seconds:g}; worker process tree was terminated."
                ),
            )
        if outcome.returncode != 0:
            detail = redact_environment_secrets(
                (outcome.stderr or outcome.stdout).strip(),
                child_environment,
            )
            return EvaluationWorkerResult(
                status="failed",
                failure_reason=(
                    f"Worker exited with code {outcome.returncode}: {detail}"
                )[:MAX_WORKER_ERROR_CHARS],
            )
        if not result_path.is_file():
            return EvaluationWorkerResult(
                status="failed",
                failure_reason="Worker exited without writing worker-result.json.",
            )
        if result_path.stat().st_size > MAX_WORKER_RESULT_BYTES:
            return EvaluationWorkerResult(
                status="failed",
                failure_reason="Worker result exceeded MAX_WORKER_RESULT_BYTES.",
            )
        try:
            worker_result = EvaluationWorkerResult.model_validate_json(
                result_path.read_bytes()
            )
        except (OSError, ValidationError, ValueError) as error:
            return EvaluationWorkerResult(
                status="failed",
                failure_reason=(f"Malformed worker result: {error}")[
                    :MAX_WORKER_ERROR_CHARS
                ],
            )
        if worker_result.result is not None and (
            worker_result.result.task_id != request.task.id
            or worker_result.result.experiment != request.experiment_id
        ):
            return EvaluationWorkerResult(
                status="failed",
                failure_reason="Worker result identity did not match its request.",
            )
        return worker_result
    finally:
        shutil.rmtree(ipc_directory, ignore_errors=True)
