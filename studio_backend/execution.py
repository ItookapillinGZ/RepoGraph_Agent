"""Background preview execution that stops before repository mutation."""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from engineering_plan import plan_repository_task
from observability.models import ArtifactRecord
from observability.recorder import TraceRecorder, trace_run
from plan_application import build_plan_application_bundle
from plan_execution import execute_engineering_plan
from studio_backend.events import StudioEventEmitter, dump_public
from studio_backend.models import StartRunRequest, StudioRun
from studio_backend.repositories import WorkspaceRepositories
from studio_backend.storage import StudioStorage, utc_now

logger = logging.getLogger(__name__)


class StudioExecutionAdapter:
    """Studio-only orchestration over existing G1/G2/G3 public boundaries."""

    def __init__(
        self,
        storage: StudioStorage,
        repositories: WorkspaceRepositories,
        *,
        planning_graph: Any | None = None,
        execution_graph: Any | None = None,
    ) -> None:
        self.storage = storage
        self.repositories = repositories
        self.planning_graph = planning_graph
        self.execution_graph = execution_graph
        self.trace_recorder = TraceRecorder(storage.trace_sink)

    def execute(self, run_id: str, request: StartRunRequest) -> None:
        with trace_run(
            self.trace_recorder,
            run_kind="repograph_studio_preview",
            repo_identity=request.repository_id,
            task=request.task,
            task_summary="Studio preview task",
            run_id=run_id,
        ):
            self._execute_traced(run_id, request)

    def _execute_traced(self, run_id: str, request: StartRunRequest) -> None:
        emitter = StudioEventEmitter(self.storage, run_id)
        transitioned = self.storage.compare_and_set_run(
            run_id,
            expected_statuses=("queued",),
            status="running",
            phase="planning",
        )
        if transitioned is None:
            return
        try:
            repository_root = self.repositories.resolve(request.repository_id)
            emitter.emit(
                phase="planning",
                event_type="repository_exploration_started",
                status="started",
                title="Repository exploration started",
            )
            emitter.emit(
                phase="planning",
                event_type="planning_started",
                status="started",
                title="Engineering planning started",
            )
            plan = plan_repository_task(
                str(repository_root),
                request.task,
                event_sink=emitter.planning_node,
                graph=self.planning_graph,
            )
            self.storage.put_artifact(
                run_id,
                "engineering_plan",
                dump_public(plan),
            )
            plan_trace = self.trace_recorder.artifact(
                kind="engineering_plan",
                content=dump_public(plan),
            )

            self.storage.update_run(run_id, status="running", phase="execution")
            emitter.emit(
                phase="execution",
                event_type="execution_started",
                status="started",
                title="Candidate execution started",
            )
            correction_rounds = (
                request.max_correction_rounds if request.self_correct else 0
            )
            candidate_traces: list[ArtifactRecord] = []

            def candidate_artifact(attempt: int, candidate: Any) -> None:
                record = self.trace_recorder.artifact(
                    kind="candidate",
                    content=dump_public(candidate),
                    metadata={
                        "attempt": attempt,
                        "correction_round": max(0, attempt - 1),
                    },
                )
                if record is None:
                    return
                if plan_trace is not None:
                    self.trace_recorder.link(plan_trace, record, "derived_from")
                if candidate_traces:
                    self.trace_recorder.link(
                        candidate_traces[-1], record, "corrected_from"
                    )
                candidate_traces.append(record)

            result = execute_engineering_plan(
                str(repository_root),
                request.task,
                plan,
                max_correction_rounds=correction_rounds,
                event_sink=emitter.execution_node,
                attempt_artifact_sink=candidate_artifact,
                graph=self.execution_graph,
            )
            self.storage.put_artifact(
                run_id,
                "plan_execution_result",
                dump_public(result),
            )
            if result.candidate is not None:
                self.storage.put_artifact(
                    run_id,
                    "candidate",
                    dump_public(result.candidate),
                )
            self.storage.put_artifact(
                run_id,
                "candidate_diff",
                {"diff_text": result.diff_text},
            )
            final_candidate_trace = candidate_traces[-1] if candidate_traces else None
            diff_trace = self.trace_recorder.artifact(
                kind="candidate_diff",
                content=result.diff_text,
            )
            if final_candidate_trace is not None and diff_trace is not None:
                self.trace_recorder.link(
                    final_candidate_trace, diff_trace, "derived_from"
                )
            if result.verification is not None:
                self.storage.put_artifact(
                    run_id,
                    "verification",
                    dump_public(result.verification),
                )
                verification_trace = self.trace_recorder.artifact(
                    kind="verification_report",
                    content=dump_public(result.verification),
                    metadata={"status": result.verification.status},
                )
                if final_candidate_trace is not None and verification_trace is not None:
                    self.trace_recorder.link(
                        final_candidate_trace, verification_trace, "verified_by"
                    )
            if result.change_set_review is not None:
                self.storage.put_artifact(
                    run_id,
                    "change_set_review",
                    dump_public(result.change_set_review),
                )
                review_trace = self.trace_recorder.artifact(
                    kind="review_result",
                    content=dump_public(result.change_set_review),
                    metadata={"rating": result.change_set_review.overall_rating},
                )
                if final_candidate_trace is not None and review_trace is not None:
                    self.trace_recorder.link(
                        final_candidate_trace, review_trace, "reviewed_by"
                    )
            self.storage.put_artifact(
                run_id,
                "correction_history",
                {
                    "correction_rounds_used": result.correction_rounds_used,
                    "attempt_history": dump_public(result.attempt_history),
                },
            )

            if result.status != "verified":
                terminal = "error" if result.status == "error" else "failed"
                self.storage.update_run(
                    run_id,
                    status=terminal,
                    phase="failed",
                    failure_reason="Candidate preview did not reach verified status.",
                )
                emitter.emit(
                    phase="failed",
                    event_type="run_failed",
                    status="failed",
                    title="Preview run failed safely",
                    message="The preview did not produce an approvable verified candidate.",
                )
                return

            bundle = build_plan_application_bundle(str(repository_root), result)
            self.storage.put_artifact(
                run_id,
                "application_bundle",
                dump_public(bundle),
            )
            bundle_trace = self.trace_recorder.artifact(
                kind="application_bundle",
                content=dump_public(bundle),
                metadata={"approval_digest": bundle.approval_digest},
            )
            if final_candidate_trace is not None and bundle_trace is not None:
                self.trace_recorder.link(
                    final_candidate_trace, bundle_trace, "packaged_as"
                )
            self.storage.update_run(
                run_id,
                status="verified",
                phase="approval_apply",
                approval_digest=bundle.approval_digest,
            )
            emitter.emit(
                phase="approval_apply",
                event_type="preview_verified",
                status="completed",
                title="Candidate preview verified",
                metadata={"approval_digest_prefix": bundle.approval_digest[:12]},
            )
            emitter.emit(
                phase="approval_apply",
                event_type="apply_approval_requested",
                status="waiting",
                title="Apply approval requested",
            )
            emitter.emit(
                phase="approval_apply",
                event_type="waiting_for_apply_approval",
                status="waiting",
                title="Waiting for apply approval",
            )
        except Exception:
            logger.exception("Studio preview execution failed for run %s", run_id)
            self.storage.update_run(
                run_id,
                status="error",
                phase="failed",
                failure_reason="Studio preview execution failed safely.",
            )
            emitter.emit(
                phase="failed",
                event_type="run_failed",
                status="failed",
                title="Studio preview execution failed safely",
                message="Inspect server logs for internal diagnostics.",
            )


class StudioRunManager:
    """Bounded single-process executor for preview runs."""

    def __init__(
        self,
        storage: StudioStorage,
        repositories: WorkspaceRepositories,
        *,
        max_workers: int,
        adapter: StudioExecutionAdapter | None = None,
    ) -> None:
        self.storage = storage
        self.repositories = repositories
        self.adapter = adapter or StudioExecutionAdapter(storage, repositories)
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="repograph-studio",
        )
        self._futures: dict[str, Future[None]] = {}
        self._lock = threading.Lock()

    def submit(self, request: StartRunRequest) -> StudioRun:
        self.repositories.resolve(request.repository_id)
        now = utc_now()
        run = StudioRun(
            id=str(uuid.uuid4()),
            repository_id=request.repository_id,
            task=request.task,
            status="queued",
            phase="queued",
            created_at=now,
            updated_at=now,
        )
        self.storage.create_run(run)
        StudioEventEmitter(self.storage, run.id).emit(
            phase="queued",
            event_type="run_created",
            status="info",
            title="Studio run created",
        )
        future = self.executor.submit(self.adapter.execute, run.id, request)
        with self._lock:
            self._futures[run.id] = future
        future.add_done_callback(lambda _future: self._discard(run.id))
        return run

    def _discard(self, run_id: str) -> None:
        with self._lock:
            self._futures.pop(run_id, None)

    @property
    def active_runs(self) -> int:
        with self._lock:
            return sum(not future.done() for future in self._futures.values())

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)
