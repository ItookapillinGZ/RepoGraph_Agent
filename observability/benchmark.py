"""Small deterministic tracing overhead measurement (not an SLA)."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from statistics import median

from observability.recorder import TraceRecorder, trace_run
from observability.sinks import NullTraceSink
from observability.storage import SQLiteTraceSink

EVENTS = 100


def _exercise(recorder: TraceRecorder, run_id: str) -> float:
    started = time.perf_counter()
    with trace_run(recorder, run_kind="overhead_benchmark", run_id=run_id):
        for index in range(EVENTS):
            recorder.event(
                "bounded_event",
                kind="benchmark",
                metadata={"index": index, "value": "bounded"},
            )
    return time.perf_counter() - started


def run_benchmark() -> dict[str, object]:
    null = TraceRecorder(NullTraceSink())
    _exercise(null, "warmup-disabled")
    with tempfile.TemporaryDirectory(prefix="repograph-observability-benchmark-") as temp:
        root = Path(temp)
        sqlite = TraceRecorder(
            SQLiteTraceSink(root / "trace.sqlite3", root / "artifacts")
        )
        _exercise(sqlite, "warmup-enabled")
        disabled_samples: list[float] = []
        enabled_samples: list[float] = []
        for index in range(3):
            disabled_samples.append(_exercise(null, f"disabled-{index}"))
            enabled_samples.append(_exercise(sqlite, f"enabled-{index}"))
        disabled = median(disabled_samples)
        enabled = median(enabled_samples)
    return {
        "schema_version": 1,
        "event_count": EVENTS,
        "disabled_seconds": disabled,
        "enabled_seconds": enabled,
        "disabled_samples": disabled_samples,
        "enabled_samples": enabled_samples,
        "absolute_overhead_seconds": enabled - disabled,
        "relative_overhead": enabled / disabled if disabled else None,
        "note": "Micro-benchmark only; SQLite durability dominates this synthetic event-only loop.",
    }


def main() -> int:
    print(json.dumps(run_benchmark(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
