"""High-level local recorder, spans, events, artifacts, and LangChain callbacks."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from observability.context import (
    TraceContext,
    current_trace_context,
    reset_trace_context,
    set_trace_context,
)
from observability.models import (
    ArtifactEdge,
    ArtifactRecord,
    TraceEvent,
    TraceRun,
    TraceSpan,
)
from observability.redaction import (
    assert_safe_artifact_payload,
    redact_text,
    sanitize_metadata,
)
from observability.sinks import NullTraceSink, TraceSink


def _id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


class TraceRecorder:
    def __init__(self, sink: TraceSink | None = None, *, required: bool = False) -> None:
        self.sink = sink or NullTraceSink()
        self.required = required
        self.failures: list[str] = []
        self._starts: dict[str, float] = {}
        self._lock = threading.Lock()

    def _safe(self, name: str, *args: object, **kwargs: object) -> object | None:
        try:
            return getattr(self.sink, name)(*args, **kwargs)
        except Exception as error:
            if self.required:
                raise
            self.failures.append(f"{type(error).__name__}:{name}"[:500])
            return None

    def start_run(
        self,
        *,
        run_kind: str,
        repo_identity: str | None = None,
        task: str | None = None,
        task_summary: str | None = None,
        run_id: str | None = None,
    ) -> TraceContext:
        selected_run_id = run_id or _id()
        root_span_id = _id()
        run = TraceRun(
            run_id=selected_run_id,
            run_kind=run_kind,
            started_at=_now(),
            status="active",
            root_span_id=root_span_id,
            repo_identity=redact_text(repo_identity) if repo_identity else None,
            task_digest=digest_text(task) if task else None,
            task_summary=redact_text(task_summary[:500]) if task_summary else None,
        )
        self._safe("create_run", run)
        context = TraceContext(
            run_id=selected_run_id,
            span_id=root_span_id,
            sink=self.sink,
            required=self.required,
        )
        self.start_span(
            name=run_kind,
            kind="run",
            parent_span_id=None,
            span_id=root_span_id,
            run_id=selected_run_id,
            metadata={},
        )
        return context

    def finish_run(self, context: TraceContext, *, status: str) -> None:
        self.finish_span(context.span_id, status=status)
        self._safe("finish_run", context.run_id, status=status)

    def start_span(
        self,
        *,
        name: str,
        kind: str,
        metadata: dict[str, object] | None = None,
        parent_span_id: str | None = None,
        span_id: str | None = None,
        run_id: str | None = None,
    ) -> str | None:
        context = current_trace_context()
        selected_run_id = run_id or (context.run_id if context else None)
        if selected_run_id is None:
            return None
        selected_span_id = span_id or _id()
        selected_parent = (
            parent_span_id if parent_span_id is not None else (context.span_id if context else None)
        )
        span = TraceSpan(
            span_id=selected_span_id,
            run_id=selected_run_id,
            parent_span_id=selected_parent,
            name=name,
            kind=kind,
            start_time=_now(),
            status="active",
            sequence=1,
            metadata=sanitize_metadata(metadata),
        )
        with self._lock:
            self._starts[selected_span_id] = time.monotonic()
        self._safe("create_span", span)
        return selected_span_id

    def finish_span(self, span_id: str | None, *, status: str = "completed") -> None:
        if span_id is None:
            return
        with self._lock:
            started = self._starts.pop(span_id, None)
        duration_ms = max(0.0, (time.monotonic() - started) * 1_000) if started else 0.0
        self._safe(
            "finish_span",
            span_id,
            status=status,
            end_time=_now(),
            duration_ms=duration_ms,
        )

    @contextmanager
    def span(
        self, name: str, *, kind: str, metadata: dict[str, object] | None = None
    ):
        parent = current_trace_context()
        span_id = self.start_span(name=name, kind=kind, metadata=metadata)
        if parent is None or span_id is None:
            yield span_id
            return
        token = set_trace_context(
            TraceContext(
                run_id=parent.run_id,
                span_id=span_id,
                sink=parent.sink,
                required=parent.required,
            )
        )
        try:
            yield span_id
        except Exception:
            self.finish_span(span_id, status="failed")
            raise
        else:
            self.finish_span(span_id, status="completed")
        finally:
            reset_trace_context(token)

    def event(
        self,
        name: str,
        *,
        kind: str,
        status: str = "info",
        metadata: dict[str, object] | None = None,
        event_id: str | None = None,
    ) -> TraceEvent | None:
        context = current_trace_context()
        if context is None:
            return None
        event = TraceEvent(
            event_id=event_id or _id(),
            run_id=context.run_id,
            span_id=context.span_id,
            sequence_number=1,
            timestamp=_now(),
            name=name,
            kind=kind,
            status=status,
            metadata=sanitize_metadata(metadata),
        )
        result = self._safe("append_event", event)
        return result if isinstance(result, TraceEvent) else event

    def artifact(
        self,
        *,
        kind: str,
        content: bytes | str | object,
        metadata: dict[str, object] | None = None,
        artifact_id: str | None = None,
    ) -> ArtifactRecord | None:
        context = current_trace_context()
        if context is None:
            return None
        if isinstance(content, bytes):
            payload = content
        elif isinstance(content, str):
            payload = content.encode("utf-8")
        else:
            payload = canonical_bytes(content)
        try:
            assert_safe_artifact_payload(payload)
        except Exception as error:
            if self.required:
                raise
            self.failures.append(f"{type(error).__name__}:artifact_privacy"[:500])
            return None
        digest = hashlib.sha256(payload).hexdigest()
        record = ArtifactRecord(
            artifact_id=artifact_id or _id(),
            run_id=context.run_id,
            kind=kind,
            sha256=digest,
            size_bytes=len(payload),
            producer_span_id=context.span_id,
            created_at=_now(),
            storage_ref=f"sha256/{digest[:2]}/{digest}.blob",
            metadata=sanitize_metadata(metadata),
        )
        self._safe("register_artifact", record, payload)
        self.event(
            f"artifact:{kind}",
            kind="artifact",
            status="created",
            metadata={"artifact_id": record.artifact_id, "sha256": digest, "size_bytes": len(payload)},
        )
        return record

    def link(
        self,
        parent: ArtifactRecord,
        child: ArtifactRecord,
        relation: str,
    ) -> ArtifactEdge:
        edge = ArtifactEdge(
            run_id=parent.run_id,
            parent_artifact_id=parent.artifact_id,
            child_artifact_id=child.artifact_id,
            relation=relation,
            created_at=_now(),
        )
        self._safe("add_artifact_edge", edge)
        return edge


@contextmanager
def trace_run(
    recorder: TraceRecorder,
    *,
    run_kind: str,
    repo_identity: str | None = None,
    task: str | None = None,
    task_summary: str | None = None,
    run_id: str | None = None,
):
    context = recorder.start_run(
        run_kind=run_kind,
        repo_identity=repo_identity,
        task=task,
        task_summary=task_summary,
        run_id=run_id,
    )
    token = set_trace_context(context)
    try:
        yield context
    except Exception:
        recorder.finish_run(context, status="failed")
        raise
    else:
        recorder.finish_run(context, status="completed")
    finally:
        reset_trace_context(token)


_DEFAULT_RECORDER = TraceRecorder()


def current_recorder() -> TraceRecorder | None:
    context = current_trace_context()
    if context is None:
        return None
    recorder = TraceRecorder(context.sink, required=context.required)
    return recorder


@contextmanager
def traced_span(name: str, *, kind: str, metadata: dict[str, object] | None = None):
    recorder = current_recorder()
    if recorder is None:
        yield None
    else:
        with recorder.span(name, kind=kind, metadata=metadata) as span_id:
            yield span_id


@contextmanager
def continue_trace_span(
    recorder: TraceRecorder,
    *,
    run_id: str,
    root_span_id: str,
    name: str,
    kind: str,
    metadata: dict[str, object] | None = None,
):
    """Append an orchestration span without reopening or changing run status."""

    token = set_trace_context(
        TraceContext(
            run_id=run_id,
            span_id=root_span_id,
            sink=recorder.sink,
            required=recorder.required,
        )
    )
    try:
        with recorder.span(name, kind=kind, metadata=metadata) as span_id:
            yield span_id
    finally:
        reset_trace_context(token)


def trace_event(
    name: str,
    *,
    kind: str,
    status: str = "info",
    metadata: dict[str, object] | None = None,
) -> TraceEvent | None:
    recorder = current_recorder()
    return recorder.event(name, kind=kind, status=status, metadata=metadata) if recorder else None


class ObservabilityCallback(BaseCallbackHandler):
    """LangChain callback that records usage/status but never prompts or reasoning."""

    def __init__(self) -> None:
        self._spans: dict[object, tuple[TraceRecorder, str | None]] = {}
        self._lock = threading.Lock()
        self._llm_index = 0
        self._tool_index = 0

    @staticmethod
    def _usage(response: LLMResult) -> tuple[int | None, int | None, int | None]:
        raw = (response.llm_output or {}).get("token_usage") or (response.llm_output or {}).get("usage")
        if isinstance(raw, dict):
            input_tokens = raw.get("prompt_tokens", raw.get("input_tokens"))
            output_tokens = raw.get("completion_tokens", raw.get("output_tokens"))
            total_tokens = raw.get("total_tokens")
            return (
                int(input_tokens) if isinstance(input_tokens, int) else None,
                int(output_tokens) if isinstance(output_tokens, int) else None,
                int(total_tokens) if isinstance(total_tokens, int) else None,
            )
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if isinstance(usage, dict):
                    return (
                        usage.get("input_tokens"), usage.get("output_tokens"), usage.get("total_tokens")
                    )
        return None, None, None

    @staticmethod
    def _event_on_span(
        recorder: TraceRecorder,
        span_id: str | None,
        name: str,
        *,
        kind: str,
        status: str,
        metadata: dict[str, object],
    ) -> None:
        context = current_trace_context()
        if context is None or span_id is None:
            recorder.event(name, kind=kind, status=status, metadata=metadata)
            return
        token = set_trace_context(
            TraceContext(
                run_id=context.run_id,
                span_id=span_id,
                sink=context.sink,
                required=context.required,
            )
        )
        try:
            recorder.event(name, kind=kind, status=status, metadata=metadata)
        finally:
            reset_trace_context(token)

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], *, run_id: Any, **kwargs: Any) -> None:
        recorder = current_recorder()
        if recorder is None:
            return
        self._llm_index += 1
        model = kwargs.get("invocation_params", {}).get("model") or serialized.get("name") or serialized.get("id", ["unknown"])[-1]
        span_id = recorder.start_span(
            name="llm_call", kind="llm_call",
            metadata={
                "provider": serialized.get("id", ["unknown"])[0] if isinstance(serialized.get("id"), list) else "unknown",
                "configured_model": str(model)[:300],
                "call_index": self._llm_index,
                "prompt_digest": digest_text("\n".join(prompts)),
                "prompt_capture": False,
            },
        )
        with self._lock:
            self._spans[run_id] = (recorder, span_id)

    def on_llm_end(self, response: LLMResult, *, run_id: Any, **kwargs: Any) -> None:
        with self._lock:
            pair = self._spans.pop(run_id, None)
        if pair is None:
            return
        recorder, span_id = pair
        input_tokens, output_tokens, total_tokens = self._usage(response)
        resolved_model = (response.llm_output or {}).get("model_name") or (
            response.llm_output or {}
        ).get("model")
        self._event_on_span(
            recorder,
            span_id,
            "llm_call_completed",
            kind="llm_call",
            status="completed",
            metadata={
                "resolved_model": str(resolved_model)[:300] if resolved_model else None,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "telemetry_complete": total_tokens is not None,
            },
        )
        recorder.finish_span(span_id)

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        with self._lock:
            pair = self._spans.pop(run_id, None)
        if pair is not None:
            pair[0].finish_span(pair[1], status="failed")

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, *, run_id: Any, **kwargs: Any) -> None:
        recorder = current_recorder()
        if recorder is None:
            return
        self._tool_index += 1
        span_id = recorder.start_span(
            name=str(serialized.get("name", "tool"))[:200], kind="tool_call",
            metadata={
                "tool_name": str(serialized.get("name", "tool"))[:200],
                "call_index": self._tool_index,
                "argument_size": len(input_str),
                "argument_digest": digest_text(input_str),
            },
        )
        with self._lock:
            self._spans[run_id] = (recorder, span_id)

    def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> None:
        with self._lock:
            pair = self._spans.pop(run_id, None)
        if pair is None:
            return
        text = str(output)
        self._event_on_span(
            pair[0],
            pair[1],
            "tool_call_completed",
            kind="tool_call",
            status="completed",
            metadata={"result_size": len(text), "result_digest": digest_text(text)},
        )
        pair[0].finish_span(pair[1])

    def on_tool_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        with self._lock:
            pair = self._spans.pop(run_id, None)
        if pair is not None:
            pair[0].finish_span(pair[1], status="failed")


def append_observability_callback(callbacks: object | None) -> list[BaseCallbackHandler] | None:
    if current_trace_context() is None:
        return list(callbacks) if callbacks else None  # type: ignore[arg-type]
    selected = list(callbacks) if callbacks else []  # type: ignore[arg-type]
    if not any(isinstance(item, ObservabilityCallback) for item in selected):
        selected.append(ObservabilityCallback())
    return selected
