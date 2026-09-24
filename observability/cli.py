"""Read-only trace inspection and explicit verification replay CLI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from observability.lineage import render_lineage, span_tree
from observability.models import ReplayObservation
from observability.redaction import sanitize_metadata
from observability.replay import ReplayBlockedError, replay_artifact
from observability.storage import SQLiteTraceSink


def _default_state_root() -> Path:
    configured = os.environ.get("REPOGRAPH_STATE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    local = os.environ.get("LOCALAPPDATA", "").strip()
    if local:
        return Path(local) / "RepoGraph"
    return Path.home() / ".repograph"


def _storage(root: Path) -> SQLiteTraceSink:
    resolved = root.expanduser().resolve()
    return SQLiteTraceSink(resolved / "observability.sqlite3", resolved / "artifacts")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m observability")
    parser.add_argument("--state-root", type=Path, default=_default_state_root())
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("runs")
    show = commands.add_parser("show")
    show.add_argument("run_id")
    artifacts = commands.add_parser("artifacts")
    artifacts.add_argument("run_id")
    lineage = commands.add_parser("lineage")
    lineage.add_argument("run_id")
    export = commands.add_parser("export")
    export.add_argument("run_id")
    export.add_argument("--output", type=Path)
    replay = commands.add_parser("replay")
    replay.add_argument("run_id")
    replay.add_argument("--candidate", required=True)
    commands.add_parser("reconcile")
    return parser


def _replay_frozen_h2(
    storage: SQLiteTraceSink, artifact_id: str, *, expected_run_id: str
) -> object:
    from evaluation.models import EvaluationTask
    from plan_execution import MultiFileCandidate
    from sandbox.acceptance_replay import (
        H2_DATASET,
        _load_jsonl,
        _replay,
    )
    from sandbox.doctor import inspect_sandbox
    from sandbox.policy import SandboxPolicy

    record, _content = storage.load_artifact(artifact_id)
    if record.run_id != expected_run_id:
        raise ReplayBlockedError("candidate does not belong to the requested run")
    if record.metadata.get("replay_adapter") != "h2_frozen_snapshot":
        raise ReplayBlockedError("candidate has no installed deterministic replay adapter")
    task_id = str(record.metadata.get("task_id", ""))
    tasks = {
        row["id"]: EvaluationTask.model_validate(row) for row in _load_jsonl(H2_DATASET)
    }
    if task_id not in tasks:
        raise ReplayBlockedError("recorded H2 task identity is unavailable")
    task = tasks[task_id]
    doctor = inspect_sandbox(SandboxPolicy(backend="docker"))
    if not doctor.sandbox_smoke_available or not doctor.sandbox_image_id:
        raise ReplayBlockedError("recorded Docker environment is unavailable")

    def verify(content: bytes, _metadata: dict[str, object]) -> ReplayObservation:
        snapshot = json.loads(content)
        candidate = MultiFileCandidate.model_validate(snapshot["candidate"])
        outcome = _replay(task, candidate, "docker")
        classification = ":".join(
            str(outcome.get(key) or "none")
            for key in ("status", "failure_category", "failure_kind")
        )
        return ReplayObservation(
            status=outcome["status"],
            resolved=outcome["resolved"],
            exit_classification=classification,
        )

    return replay_artifact(
        storage,
        artifact_id,
        executor=verify,
        current_base_identity=task.base_commit,
        current_backend="docker",
        current_image_id=doctor.sandbox_image_id,
    )


def _replay_registered(
    storage: SQLiteTraceSink, artifact_id: str, *, expected_run_id: str
) -> object:
    record, _content = storage.load_artifact(artifact_id)
    adapter = record.metadata.get("replay_adapter")
    if adapter == "demo_v1":
        from repograph.demo import replay_demo_artifact

        return replay_demo_artifact(
            storage, artifact_id, expected_run_id=expected_run_id
        )
    if adapter == "h2_frozen_snapshot":
        return _replay_frozen_h2(
            storage, artifact_id, expected_run_id=expected_run_id
        )
    raise ReplayBlockedError("candidate has no installed deterministic replay adapter")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    storage = _storage(args.state_root)
    if args.command == "runs":
        for run in storage.list_runs():
            print(f"{run.run_id}  {run.status:11}  {run.run_kind}  {run.started_at.isoformat()}")
        return 0
    if args.command == "show":
        run = storage.get_run(args.run_id)
        if run is None:
            raise SystemExit("run not found")
        metrics = storage.metrics(args.run_id)
        print(json.dumps(sanitize_metadata({
            "run": run.model_dump(mode="json"),
            "metrics": metrics.model_dump(mode="json"),
        }), ensure_ascii=False, indent=2, sort_keys=True))
        print(span_tree(storage.list_spans(args.run_id)))
        return 0
    if args.command == "artifacts":
        for item in storage.list_artifacts(args.run_id):
            print(f"{item.artifact_id}  {item.kind:24}  {item.size_bytes:9}  {item.sha256}")
        return 0
    if args.command == "lineage":
        print(render_lineage(storage.list_artifacts(args.run_id), storage.list_edges(args.run_id)))
        return 0
    if args.command == "export":
        lines = [
            json.dumps(sanitize_metadata(record), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            for record in storage.export_run(args.run_id)
        ]
        payload = "\n".join(lines) + "\n"
        if args.output:
            args.output.write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
        return 0
    if args.command == "replay":
        run = storage.get_run(args.run_id)
        if run is None:
            raise SystemExit("run not found")
        try:
            result = _replay_registered(
                storage, args.candidate, expected_run_id=args.run_id
            )
        except ReplayBlockedError as error:
            raise SystemExit(f"BLOCKED: {error}") from error
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result.semantic_match else 1
    if args.command == "reconcile":
        runs, spans = storage.reconcile_abandoned()
        print(f"interrupted runs={runs} spans={spans}")
        return 0
    return 2
