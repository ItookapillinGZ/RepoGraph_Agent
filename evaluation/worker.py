"""One-request process entry point for production evaluation tasks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pydantic import ValidationError

from evaluation.models import EvaluationWorkerRequest, EvaluationWorkerResult
from evaluation.process_runner import (
    MAX_WORKER_ERROR_CHARS,
    MAX_WORKER_REQUEST_BYTES,
    MAX_WORKER_RESULT_BYTES,
    redact_environment_secrets,
    worker_environment,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evaluation.worker")
    parser.add_argument("--request", required=True)
    return parser


def _write_result(path: Path, result: EvaluationWorkerResult) -> None:
    payload = result.model_dump_json().encode("utf-8")
    if len(payload) > MAX_WORKER_RESULT_BYTES:
        payload = (
            EvaluationWorkerResult(
                status="failed",
                failure_reason="Worker result exceeded MAX_WORKER_RESULT_BYTES.",
            )
            .model_dump_json()
            .encode("utf-8")
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request_path = Path(args.request).resolve(strict=True)
    if request_path.stat().st_size > MAX_WORKER_REQUEST_BYTES:
        return 2
    try:
        request = EvaluationWorkerRequest.model_validate_json(request_path.read_bytes())
    except (OSError, ValidationError, ValueError):
        return 2
    result_path = Path(request.result_path).resolve()
    try:
        from evaluation.runner import _run_evaluation_task_in_process

        result = _run_evaluation_task_in_process(
            request.task,
            request.config,
            workspace_root=request.workspace_root,
            artifacts_root=request.artifacts_root,
            experiment_id=request.experiment_id,
            execution_attempt=request.execution_attempt,
            git_commit=request.git_commit,
        ).model_copy(update={"process_isolated": True})
        envelope = EvaluationWorkerResult(status="completed", result=result)
    except KeyboardInterrupt:
        envelope = EvaluationWorkerResult(
            status="interrupted",
            failure_reason="Evaluation worker was interrupted.",
        )
    except BaseException as error:  # noqa: BLE001 - last-resort worker boundary
        detail = " ".join(str(error).split()) or type(error).__name__
        detail = redact_environment_secrets(detail, worker_environment())
        envelope = EvaluationWorkerResult(
            status="failed",
            failure_reason=detail[:MAX_WORKER_ERROR_CHARS],
        )
    try:
        _write_result(result_path, envelope)
    except OSError:
        return 3
    # A valid failure/interruption envelope is successful IPC. Non-zero exits
    # are reserved for crashes, invalid requests, or result-write failures.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
