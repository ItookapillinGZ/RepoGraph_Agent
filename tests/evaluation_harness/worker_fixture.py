"""Deterministic subprocess behaviors for process-runner integration tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def main() -> int:
    request_path = Path(sys.argv[sys.argv.index("--request") + 1])
    raw_request = json.loads(request_path.read_text(encoding="utf-8"))
    metadata = raw_request["task"].get("metadata", {})
    behavior = str(metadata.get("worker_fixture", "success"))
    result_path = Path(raw_request["result_path"])
    if behavior == "sleep":
        started_path = metadata.get("started_path")
        if isinstance(started_path, str):
            Path(started_path).write_text("started", encoding="utf-8")
        time.sleep(60)
    if behavior == "child_sleep":
        survivor = str(metadata["survivor_path"])
        subprocess.Popen(  # nosec B603 - fixed test fixture argv
            [
                sys.executable,
                "-c",
                (
                    "import pathlib,sys,time; time.sleep(3); "
                    "pathlib.Path(sys.argv[1]).write_text('survived'); time.sleep(60)"
                ),
                survivor,
            ],
            shell=False,
        )
        Path(survivor + ".started").write_text("started", encoding="utf-8")
        time.sleep(60)
    if behavior in {"secret_stdout", "secret_stderr"}:
        stream = sys.stdout if behavior == "secret_stdout" else sys.stderr
        print(os.environ.get("OPENAI_API_KEY", "missing"), file=stream)
        return 7
    if behavior == "crash":
        os._exit(7)
    if behavior == "malformed":
        result_path.write_text("{not-json", encoding="utf-8")
        return 0
    if behavior == "oversized":
        result_path.write_bytes(b"x" * 10_000_001)
        return 0
    if behavior == "no_result":
        return 0

    from evaluation.models import (
        EvaluationResult,
        EvaluationWorkerRequest,
        EvaluationWorkerResult,
    )
    from evaluation.worker import _write_result

    request = EvaluationWorkerRequest.model_validate(raw_request)
    if behavior == "real_graph":
        from evaluation.runner import (
            RealRepoGraphAdapter,
            _run_evaluation_task_in_process,
        )
        from tests.evaluation_harness.helpers import FIXED, candidate
        from tests.evaluation_harness.test_integration import (
            execution_graph,
            planning_graph,
        )

        result = _run_evaluation_task_in_process(
            request.task,
            request.config,
            workspace_root=request.workspace_root,
            experiment_id=request.experiment_id,
            adapter=RealRepoGraphAdapter(
                planning_graph=planning_graph(),
                execution_graph=execution_graph(candidate(FIXED)),
            ),
            git_commit=request.git_commit,
        ).model_copy(update={"process_isolated": True})
    else:
        result = EvaluationResult(
            task_id=request.task.id,
            dataset=request.task.dataset,
            experiment=request.experiment_id,
            repository=request.task.repository,
            base_commit=request.task.base_commit,
            created_at=datetime.now(UTC).isoformat(),
            status="resolved",
            first_attempt_resolved=True,
            final_resolved=True,
            duration_seconds=0,
            process_isolated=True,
            config_digest=request.config.digest(),
        )
    _write_result(
        result_path, EvaluationWorkerResult(status="completed", result=result)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
