"""Real Docker replay acceptance over five immutable frozen H2 candidates."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from evaluation.models import EvaluationTask
from observability.models import ReplayObservation
from observability.recorder import TraceRecorder, trace_run
from observability.replay import replay_artifact
from observability.storage import SQLiteTraceSink
from plan_execution import MultiFileCandidate
from sandbox.acceptance_replay import (
    EXPECTED_IMAGE_ID,
    H2_ARTIFACTS,
    H2_DATASET,
    H2_RESULTS,
    SELECTION_RULE,
    _load_jsonl,
    _replay,
    _selected_candidates,
)
from sandbox.doctor import inspect_sandbox
from sandbox.policy import SandboxPolicy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_ROOT = PROJECT_ROOT / ".evaluation" / "h3_1_observability_runtime"
RESULT_ROOT = PROJECT_ROOT / "observability" / "acceptance" / "results"
REPORT_PATH = RESULT_ROOT / "frozen-h2-replay.json"


def _digest(value: object) -> str:
    canonical = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _classification(value: dict[str, Any]) -> str:
    return ":".join(
        str(value.get(key) or "none")
        for key in ("status", "failure_category", "failure_kind")
    )


def _replay_registered(
    *,
    storage: SQLiteTraceSink,
    recorder: TraceRecorder,
    tasks: dict[str, EvaluationTask],
    registered: list[tuple[str, dict[str, Any]]],
    image_id: str,
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for task_id, record_payload in registered:
        task = tasks[task_id]
        record_id = record_payload["artifact_id"]

        def verify(
            content: bytes,
            _metadata: dict[str, object],
            selected_task: EvaluationTask = task,
        ) -> ReplayObservation:
            snapshot = json.loads(content)
            candidate = MultiFileCandidate.model_validate(snapshot["candidate"])
            outcome = _replay(selected_task, candidate, "docker")
            return ReplayObservation(
                status=outcome["status"],
                resolved=outcome["resolved"],
                exit_classification=_classification(outcome),
            )

        with recorder.span(
            "deterministic_replay",
            kind="verification",
            metadata={"task_id": task_id, "artifact_id": record_id},
        ):
            replay = replay_artifact(
                storage,
                str(record_id),
                executor=verify,
                current_base_identity=task.base_commit,
                current_backend="docker",
                current_image_id=image_id,
            )
        comparisons.append(
            {
                "task_id": task_id,
                "artifact_id": record_id,
                "candidate_sha256": record_payload["metadata"]["candidate_sha256"],
                "original": replay.original_result.model_dump(mode="json"),
                "replay": (
                    replay.replay_result.model_dump(mode="json")
                    if replay.replay_result is not None
                    else None
                ),
                "semantic_match": replay.semantic_match,
                "environment_match": replay.environment_match,
                "replay_id": replay.replay_id,
            }
        )
    return comparisons


def run_frozen_h2_replay() -> dict[str, Any]:
    """Register, integrity-check, and Docker-replay the frozen H3/H2 selection."""

    doctor = inspect_sandbox(SandboxPolicy(backend="docker"))
    if not doctor.sandbox_image_available or not doctor.sandbox_smoke_available:
        raise RuntimeError("Docker sandbox is unavailable; replay is BLOCKED.")
    if doctor.sandbox_image_id != EXPECTED_IMAGE_ID:
        raise RuntimeError("Docker sandbox image provenance mismatch; replay is BLOCKED.")

    previous = json.loads(
        (
            PROJECT_ROOT
            / "sandbox"
            / "acceptance"
            / "results"
            / "host-docker-equivalence.json"
        ).read_text(encoding="utf-8")
    )
    original_by_task = {
        item["task_id"]: item["docker"] for item in previous["comparisons"]
    }
    tasks = {
        row["id"]: EvaluationTask.model_validate(row) for row in _load_jsonl(H2_DATASET)
    }
    selections = _selected_candidates()

    resolved_state = STATE_ROOT.resolve()
    trusted_evaluation_root = (PROJECT_ROOT / ".evaluation").resolve()
    if not resolved_state.is_relative_to(trusted_evaluation_root):
        raise RuntimeError("H3.1 state root escaped the trusted evaluation root.")
    if resolved_state.exists():
        shutil.rmtree(resolved_state)
    STATE_ROOT.mkdir(parents=True)
    storage = SQLiteTraceSink(
        STATE_ROOT / "observability.sqlite3", STATE_ROOT / "artifacts"
    )
    recorder = TraceRecorder(storage, required=True)
    registered: list[tuple[str, dict[str, Any]]] = []
    with trace_run(
        recorder,
        run_kind="frozen_h2_docker_replay",
        repo_identity="h2-frozen-candidate-selection",
        task_summary="Five immutable H2 candidate verification replays",
        run_id="h3-1-frozen-h2-replay",
    ):
        for selected in selections:
            result = selected["result"]
            task_id = result["task_id"]
            task = tasks[task_id]
            relative_snapshot = result["candidate_snapshot_paths"][-1]
            snapshot_path = H2_ARTIFACTS / relative_snapshot
            snapshot_bytes = snapshot_path.read_bytes()
            snapshot = json.loads(snapshot_bytes)
            MultiFileCandidate.model_validate(snapshot["candidate"])
            original = original_by_task[task_id]
            specification = {
                "task_id": task_id,
                "test_command": ["python", "-m", "pytest", "tests", "repograph_hidden_tests", "-q"],
                "timeout_seconds": 120,
                "sandbox_backend": "docker",
                "sandbox_image_id": EXPECTED_IMAGE_ID,
            }
            record = recorder.artifact(
                kind="candidate",
                content=snapshot_bytes,
                metadata={
                    "task_id": task_id,
                    "snapshot_path": relative_snapshot,
                    "candidate_sha256": snapshot["candidate_sha256"],
                    "base_identity": task.base_commit,
                    "verification_spec_digest": _digest(specification),
                    "evaluator_digest": result["evaluator_digest"],
                    "sandbox_backend": "docker",
                    "sandbox_image_id": EXPECTED_IMAGE_ID,
                    "replay_adapter": "h2_frozen_snapshot",
                    "original_result": {
                        "status": original["status"],
                        "resolved": original["resolved"],
                        "exit_classification": _classification(original),
                        "normalized_report_digest": None,
                    },
                },
            )
            if record is None:
                raise RuntimeError("required artifact registration failed")
            registered.append((task_id, record.model_dump(mode="json")))
        comparisons = _replay_registered(
            storage=storage,
            recorder=recorder,
            tasks=tasks,
            registered=registered,
            image_id=doctor.sandbox_image_id,
        )

    matches = sum(item["semantic_match"] for item in comparisons)
    report = {
        "schema_version": 1,
        "selection_rule": SELECTION_RULE,
        "source_results": H2_RESULTS.relative_to(PROJECT_ROOT).as_posix(),
        "source_artifacts": H2_ARTIFACTS.relative_to(PROJECT_ROOT).as_posix(),
        "h2_sources_modified": False,
        "openai_calls": 0,
        "g4_g5_calls": 0,
        "sandbox_backend": "docker",
        "sandbox_image_id": doctor.sandbox_image_id,
        "expected_sandbox_image_id": EXPECTED_IMAGE_ID,
        "candidate_count": len(comparisons),
        "semantic_match_count": matches,
        "semantic_match_rate": matches / len(comparisons),
        "all_semantically_matched": matches == len(comparisons),
        "comparisons": comparisons,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    try:
        report = run_frozen_h2_replay()
    except RuntimeError as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["all_semantically_matched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
