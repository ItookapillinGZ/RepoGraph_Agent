"""Run lifecycle, artifact history, and database-backed SSE endpoints."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from observability.lineage import render_lineage
from studio_backend.api.dependencies import run_manager, storage
from studio_backend.execution import StudioRunManager
from studio_backend.models import (
    ArtifactResponse,
    EventListResponse,
    RunDetail,
    RunListResponse,
    RunTraceSummary,
    StartRunRequest,
    StartRunResponse,
)
from studio_backend.repositories import RepositoryBoundaryError
from studio_backend.safety import sanitize_public_payload
from studio_backend.storage import StudioStorage

router = APIRouter(prefix="/api/runs", tags=["runs"])
StorageDependency = Annotated[StudioStorage, Depends(storage)]
ManagerDependency = Annotated[StudioRunManager, Depends(run_manager)]

PUBLIC_ARTIFACT_KINDS = {
    "engineering_plan",
    "plan_execution_result",
    "candidate",
    "candidate_diff",
    "verification",
    "change_set_review",
    "correction_history",
    "application_bundle",
    "application_result",
    "local_git_delivery_result",
    "remote_delivery_result",
}


def _run_or_404(database: StudioStorage, run_id: str):
    run = database.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return run


def _sensitive_values(request: Request) -> tuple[str, ...]:
    config = request.app.state.config
    return (
        str(config.workspace_root),
        str(config.data_dir),
        os.environ.get("GITHUB_TOKEN", ""),
        os.environ.get("GH_TOKEN", ""),
    )


@router.post("", response_model=StartRunResponse, status_code=202)
def create_run(
    request: StartRunRequest,
    manager: ManagerDependency,
) -> StartRunResponse:
    try:
        run = manager.submit(request)
    except RepositoryBoundaryError as error:
        raise HTTPException(status_code=404, detail="Unknown repository.") from error
    return StartRunResponse(run_id=run.id)


@router.get("", response_model=RunListResponse)
def list_runs(request: Request, database: StorageDependency) -> RunListResponse:
    sensitive = _sensitive_values(request)
    return RunListResponse(
        runs=[
            type(run).model_validate(
                sanitize_public_payload(run.model_dump(mode="json"), sensitive)
            )
            for run in database.list_runs(limit=50)
        ]
    )


@router.get("/{run_id}", response_model=RunDetail)
def get_run(
    run_id: str,
    request: Request,
    database: StorageDependency,
) -> RunDetail:
    run = _run_or_404(database, run_id)
    public_run = sanitize_public_payload(
        run.model_dump(mode="json"),
        _sensitive_values(request),
    )
    trace_run = database.trace_sink.get_run(run_id)
    trace_summary = None
    if trace_run is not None:
        metrics = database.trace_sink.metrics(run_id)
        trace_summary = RunTraceSummary(
            trace_id=trace_run.run_id,
            duration_ms=metrics.total_duration_ms,
            llm_calls=metrics.llm_calls,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            total_tokens=metrics.total_tokens,
            tool_calls=metrics.tool_calls,
            sandbox_executions=metrics.sandbox_executions,
            artifact_count=metrics.artifact_count,
            lineage_summary=render_lineage(
                database.trace_sink.list_artifacts(run_id),
                database.trace_sink.list_edges(run_id),
            )[:20_000],
        )
    return RunDetail(
        **public_run,
        artifact_kinds=database.artifact_kinds(run_id),
        latest_event_sequence=database.latest_event_sequence(run_id),
        github_auth_configured=bool(
            os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        ),
        trace=trace_summary,
    )


@router.get("/{run_id}/artifacts/{kind}", response_model=ArtifactResponse)
def get_artifact(
    run_id: str,
    kind: str,
    request: Request,
    database: StorageDependency,
) -> ArtifactResponse:
    _run_or_404(database, run_id)
    if kind not in PUBLIC_ARTIFACT_KINDS:
        raise HTTPException(status_code=404, detail="Artifact not found.")
    artifact = database.get_artifact(run_id, kind)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return artifact.model_copy(
        update={
            "payload": sanitize_public_payload(
                artifact.payload,
                _sensitive_values(request),
            )
        }
    )


@router.get("/{run_id}/events", response_model=EventListResponse)
def get_events(
    run_id: str,
    database: StorageDependency,
    after: Annotated[int, Query(ge=0)] = 0,
) -> EventListResponse:
    _run_or_404(database, run_id)
    return EventListResponse(events=database.list_events(run_id, after=after))


@router.get("/{run_id}/stream")
async def stream_events(
    run_id: str,
    request: Request,
    database: StorageDependency,
    after: Annotated[int, Query(ge=0)] = 0,
) -> StreamingResponse:
    _run_or_404(database, run_id)

    async def event_stream():
        sequence = after
        idle_polls = 0
        while not await request.is_disconnected():
            events = await asyncio.to_thread(
                database.list_events,
                run_id,
                after=sequence,
            )
            if events:
                idle_polls = 0
                for event in events:
                    sequence = event.sequence
                    payload = json.dumps(
                        event.model_dump(mode="json"),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    yield (
                        f"id: {event.sequence}\n"
                        "event: run_event\n"
                        f"data: {payload}\n\n"
                    )
            else:
                idle_polls += 1
                if idle_polls >= 30:
                    idle_polls = 0
                    yield ": keep-alive\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
