"""Context-local tracing that never enters LangGraph reasoning state."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass

from observability.sinks import TraceSink


@dataclass(frozen=True)
class TraceContext:
    run_id: str
    span_id: str
    sink: TraceSink
    required: bool = False


_TRACE_CONTEXT: ContextVar[TraceContext | None] = ContextVar(
    "repograph_trace_context", default=None
)


def current_trace_context() -> TraceContext | None:
    return _TRACE_CONTEXT.get()


def set_trace_context(context: TraceContext | None) -> Token[TraceContext | None]:
    return _TRACE_CONTEXT.set(context)


def reset_trace_context(token: Token[TraceContext | None]) -> None:
    _TRACE_CONTEXT.reset(token)

