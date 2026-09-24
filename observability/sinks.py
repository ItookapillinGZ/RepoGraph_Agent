"""Trace sink abstraction and failure-policy composition."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from observability.models import (
    ArtifactEdge,
    ArtifactRecord,
    ReplayResult,
    TraceEvent,
    TraceRun,
    TraceSpan,
)


class TraceSink(Protocol):
    def create_run(self, run: TraceRun) -> None: ...
    def finish_run(self, run_id: str, *, status: str) -> None: ...
    def create_span(self, span: TraceSpan) -> None: ...
    def finish_span(
        self, span_id: str, *, status: str, end_time: object, duration_ms: float
    ) -> None: ...
    def append_event(self, event: TraceEvent) -> TraceEvent: ...
    def register_artifact(self, artifact: ArtifactRecord, content: bytes) -> None: ...
    def add_artifact_edge(self, edge: ArtifactEdge) -> None: ...
    def record_replay(self, result: ReplayResult) -> None: ...


class NullTraceSink:
    """Disabled-mode sink with intentionally zero persistence."""

    def create_run(self, run: TraceRun) -> None:
        return None

    def finish_run(self, run_id: str, *, status: str) -> None:
        return None

    def create_span(self, span: TraceSpan) -> None:
        return None

    def finish_span(
        self, span_id: str, *, status: str, end_time: object, duration_ms: float
    ) -> None:
        return None

    def append_event(self, event: TraceEvent) -> TraceEvent:
        return event

    def register_artifact(self, artifact: ArtifactRecord, content: bytes) -> None:
        return None

    def add_artifact_edge(self, edge: ArtifactEdge) -> None:
        return None

    def record_replay(self, result: ReplayResult) -> None:
        return None


class CompositeTraceSink:
    """Fan out local telemetry while honoring best-effort/required policy."""

    def __init__(self, sinks: Iterable[TraceSink], *, required: bool = False) -> None:
        self.sinks = tuple(sinks)
        self.required = required
        self.failures: list[str] = []

    def _call(self, name: str, *args: object, **kwargs: object) -> object | None:
        result: object | None = None
        for sink in self.sinks:
            try:
                result = getattr(sink, name)(*args, **kwargs)
            except Exception as error:
                if self.required:
                    raise
                self.failures.append(f"{type(error).__name__}:{name}"[:500])
        return result

    def create_run(self, run: TraceRun) -> None:
        self._call("create_run", run)

    def finish_run(self, run_id: str, *, status: str) -> None:
        self._call("finish_run", run_id, status=status)

    def create_span(self, span: TraceSpan) -> None:
        self._call("create_span", span)

    def finish_span(
        self, span_id: str, *, status: str, end_time: object, duration_ms: float
    ) -> None:
        self._call(
            "finish_span",
            span_id,
            status=status,
            end_time=end_time,
            duration_ms=duration_ms,
        )

    def append_event(self, event: TraceEvent) -> TraceEvent:
        return self._call("append_event", event) or event  # type: ignore[return-value]

    def register_artifact(self, artifact: ArtifactRecord, content: bytes) -> None:
        self._call("register_artifact", artifact, content)

    def add_artifact_edge(self, edge: ArtifactEdge) -> None:
        self._call("add_artifact_edge", edge)

    def record_replay(self, result: ReplayResult) -> None:
        self._call("record_replay", result)
