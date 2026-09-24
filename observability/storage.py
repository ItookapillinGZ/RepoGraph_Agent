"""SQLite trace index and immutable local content store."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from observability.models import (
    ArtifactEdge,
    ArtifactRecord,
    ReplayResult,
    RunMetrics,
    TraceEvent,
    TraceRun,
    TraceSpan,
)

MAX_EVENTS_PER_RUN = 20_000
MAX_SPANS_PER_RUN = 5_000
MAX_ARTIFACT_BYTES = 20_000_000


class TraceStorageError(RuntimeError):
    pass


class TraceIntegrityError(TraceStorageError):
    pass


def _iso(value: datetime | object) -> str:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return str(value)


def _json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


class SQLiteTraceSink:
    """Concurrent local source of truth with transactional sequence allocation."""

    def __init__(self, database_path: str | Path, artifact_root: str | Path | None = None):
        self.database_path = Path(database_path).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root = (
            Path(artifact_root).resolve()
            if artifact_root is not None
            else self.database_path.parent / "trace-artifacts"
        )
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._initialize_lock = threading.Lock()
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path, timeout=10, isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def initialize(self) -> None:
        with self._initialize_lock:
            connection = self._connect()
            try:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS runs (
                        run_id TEXT PRIMARY KEY,
                        run_kind TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        finished_at TEXT,
                        status TEXT NOT NULL,
                        root_span_id TEXT NOT NULL,
                        repo_identity TEXT,
                        task_digest TEXT,
                        task_summary TEXT,
                        schema_version INTEGER NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS run_counters (
                        run_id TEXT PRIMARY KEY,
                        sequence INTEGER NOT NULL DEFAULT 0,
                        event_count INTEGER NOT NULL DEFAULT 0,
                        span_count INTEGER NOT NULL DEFAULT 0,
                        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                    );
                    CREATE TABLE IF NOT EXISTS spans (
                        span_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        parent_span_id TEXT,
                        name TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        start_time TEXT NOT NULL,
                        end_time TEXT,
                        duration_ms REAL,
                        status TEXT NOT NULL,
                        sequence INTEGER NOT NULL,
                        metadata_json TEXT NOT NULL,
                        schema_version INTEGER NOT NULL,
                        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
                        FOREIGN KEY (parent_span_id) REFERENCES spans(span_id)
                    );
                    CREATE TABLE IF NOT EXISTS events (
                        event_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        span_id TEXT,
                        sequence_number INTEGER NOT NULL,
                        timestamp TEXT NOT NULL,
                        name TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        status TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        schema_version INTEGER NOT NULL,
                        UNIQUE(run_id, sequence_number),
                        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
                        FOREIGN KEY (span_id) REFERENCES spans(span_id)
                    );
                    CREATE TABLE IF NOT EXISTS artifacts (
                        artifact_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        sha256 TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        producer_span_id TEXT,
                        created_at TEXT NOT NULL,
                        storage_ref TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        schema_version INTEGER NOT NULL,
                        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
                        FOREIGN KEY (producer_span_id) REFERENCES spans(span_id)
                    );
                    CREATE TABLE IF NOT EXISTS artifact_edges (
                        run_id TEXT NOT NULL,
                        parent_artifact_id TEXT NOT NULL,
                        child_artifact_id TEXT NOT NULL,
                        relation TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        schema_version INTEGER NOT NULL,
                        PRIMARY KEY(parent_artifact_id, child_artifact_id, relation),
                        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
                        FOREIGN KEY (parent_artifact_id) REFERENCES artifacts(artifact_id),
                        FOREIGN KEY (child_artifact_id) REFERENCES artifacts(artifact_id)
                    );
                    CREATE TABLE IF NOT EXISTS replays (
                        replay_id TEXT PRIMARY KEY,
                        original_run_id TEXT NOT NULL,
                        artifact_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        semantic_match INTEGER NOT NULL,
                        environment_match INTEGER NOT NULL,
                        payload_json TEXT NOT NULL,
                        replayed_at TEXT NOT NULL,
                        schema_version INTEGER NOT NULL,
                        FOREIGN KEY (original_run_id) REFERENCES runs(run_id),
                        FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id)
                    );
                    CREATE INDEX IF NOT EXISTS idx_spans_run_sequence
                        ON spans(run_id, sequence);
                    CREATE INDEX IF NOT EXISTS idx_events_run_sequence
                        ON events(run_id, sequence_number);
                    CREATE INDEX IF NOT EXISTS idx_artifacts_run_kind
                        ON artifacts(run_id, kind);
                    CREATE INDEX IF NOT EXISTS idx_edges_run_parent
                        ON artifact_edges(run_id, parent_artifact_id);
                    """
                )
                connection.execute("PRAGMA optimize")
            finally:
                connection.close()

    @staticmethod
    def _same(row: sqlite3.Row, expected: dict[str, object], keys: tuple[str, ...]) -> bool:
        return all(row[key] == expected[key] for key in keys)

    def create_run(self, run: TraceRun) -> None:
        values = run.model_dump(mode="json")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run.run_id,)
            ).fetchone()
            if existing is not None:
                keys = (
                    "run_kind", "started_at", "status", "root_span_id",
                    "repo_identity", "task_digest", "task_summary", "schema_version",
                )
                normalized = {**values, "started_at": _iso(run.started_at)}
                if not self._same(existing, normalized, keys):
                    raise TraceIntegrityError("run identity content conflict")
                connection.execute("COMMIT")
                return
            connection.execute(
                """INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run.run_id, run.run_kind, _iso(run.started_at),
                    _iso(run.finished_at) if run.finished_at else None, run.status,
                    run.root_span_id, run.repo_identity, run.task_digest,
                    run.task_summary, run.schema_version,
                ),
            )
            connection.execute(
                "INSERT INTO run_counters(run_id) VALUES (?)", (run.run_id,)
            )
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def finish_run(self, run_id: str, *, status: str) -> None:
        connection = self._connect()
        try:
            cursor = connection.execute(
                "UPDATE runs SET finished_at = ?, status = ? WHERE run_id = ?",
                (_iso(datetime.now(UTC)), status, run_id),
            )
            if cursor.rowcount != 1:
                raise TraceStorageError("unknown trace run")
        finally:
            connection.close()

    def _allocate(self, connection: sqlite3.Connection, run_id: str, kind: str) -> int:
        if kind == "event":
            bound = MAX_EVENTS_PER_RUN
            select_sql = (
                "SELECT sequence, event_count AS count "
                "FROM run_counters WHERE run_id = ?"
            )
            update_sql = (
                "UPDATE run_counters SET sequence = ?, "
                "event_count = event_count + 1 WHERE run_id = ?"
            )
        else:
            bound = MAX_SPANS_PER_RUN
            select_sql = (
                "SELECT sequence, span_count AS count "
                "FROM run_counters WHERE run_id = ?"
            )
            update_sql = (
                "UPDATE run_counters SET sequence = ?, "
                "span_count = span_count + 1 WHERE run_id = ?"
            )
        row = connection.execute(select_sql, (run_id,)).fetchone()
        if row is None:
            raise TraceStorageError("unknown trace run")
        if int(row["count"]) >= bound:
            raise TraceStorageError(f"{kind} telemetry bound exceeded")
        sequence = int(row["sequence"]) + 1
        connection.execute(update_sql, (sequence, run_id))
        return sequence

    def create_span(self, span: TraceSpan) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM spans WHERE span_id = ?", (span.span_id,)
            ).fetchone()
            if existing is not None:
                expected = {
                    "run_id": span.run_id,
                    "parent_span_id": span.parent_span_id,
                    "name": span.name,
                    "kind": span.kind,
                    "start_time": _iso(span.start_time),
                    "status": span.status,
                    "metadata_json": _json(span.metadata),
                    "schema_version": span.schema_version,
                }
                keys = tuple(expected)
                if not self._same(existing, expected, keys):
                    raise TraceIntegrityError("span identity content conflict")
                connection.execute("COMMIT")
                return
            sequence = self._allocate(connection, span.run_id, "span")
            connection.execute(
                """INSERT INTO spans (
                    span_id, run_id, parent_span_id, name, kind, start_time,
                    end_time, duration_ms, status, sequence, metadata_json,
                    schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    span.span_id, span.run_id, span.parent_span_id, span.name,
                    span.kind, _iso(span.start_time), None, None, span.status,
                    sequence, _json(span.metadata), span.schema_version,
                ),
            )
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def finish_span(
        self, span_id: str, *, status: str, end_time: object, duration_ms: float
    ) -> None:
        connection = self._connect()
        try:
            cursor = connection.execute(
                """UPDATE spans SET end_time = ?, duration_ms = ?, status = ?
                   WHERE span_id = ? AND status = 'active'""",
                (_iso(end_time), duration_ms, status, span_id),
            )
            if cursor.rowcount == 0:
                row = connection.execute(
                    "SELECT status, duration_ms FROM spans WHERE span_id = ?", (span_id,)
                ).fetchone()
                if row is None:
                    raise TraceStorageError("unknown trace span")
                if row["status"] != status:
                    raise TraceIntegrityError("span finalization conflict")
        finally:
            connection.close()

    def append_event(self, event: TraceEvent) -> TraceEvent:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM events WHERE event_id = ?", (event.event_id,)
            ).fetchone()
            if existing is not None:
                expected = {
                    "run_id": event.run_id,
                    "span_id": event.span_id,
                    "timestamp": _iso(event.timestamp),
                    "name": event.name,
                    "kind": event.kind,
                    "status": event.status,
                    "metadata_json": _json(event.metadata),
                    "schema_version": event.schema_version,
                }
                if not self._same(existing, expected, tuple(expected)):
                    raise TraceIntegrityError("event identity content conflict")
                connection.execute("COMMIT")
                return event.model_copy(
                    update={"sequence_number": existing["sequence_number"]}
                )
            sequence = self._allocate(connection, event.run_id, "event")
            connection.execute(
                """INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.event_id, event.run_id, event.span_id, sequence,
                    _iso(event.timestamp), event.name, event.kind, event.status,
                    _json(event.metadata), event.schema_version,
                ),
            )
            connection.execute("COMMIT")
            return event.model_copy(update={"sequence_number": sequence})
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def register_artifact(self, artifact: ArtifactRecord, content: bytes) -> None:
        if len(content) > MAX_ARTIFACT_BYTES:
            raise TraceStorageError("artifact exceeds local observability bound")
        digest = hashlib.sha256(content).hexdigest()
        if digest != artifact.sha256 or len(content) != artifact.size_bytes:
            raise TraceIntegrityError("artifact_integrity_error")
        relative = Path(artifact.storage_ref)
        if relative.is_absolute() or ".." in relative.parts:
            raise TraceStorageError("artifact storage_ref must be safe and relative")
        destination = (self.artifact_root / relative).resolve()
        if not destination.is_relative_to(self.artifact_root):
            raise TraceStorageError("artifact storage_ref escaped trusted root")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                raise TraceIntegrityError("artifact_integrity_error")
        else:
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            try:
                temporary.write_bytes(content)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact.artifact_id,)
            ).fetchone()
            expected = {
                "run_id": artifact.run_id,
                "kind": artifact.kind,
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
                "producer_span_id": artifact.producer_span_id,
                "created_at": _iso(artifact.created_at),
                "storage_ref": artifact.storage_ref,
                "metadata_json": _json(artifact.metadata),
                "schema_version": artifact.schema_version,
            }
            if existing is not None:
                if not self._same(existing, expected, tuple(expected)):
                    raise TraceIntegrityError("artifact identity content conflict")
                connection.execute("COMMIT")
                return
            connection.execute(
                """INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    artifact.artifact_id, artifact.run_id, artifact.kind,
                    artifact.sha256, artifact.size_bytes, artifact.producer_span_id,
                    _iso(artifact.created_at), artifact.storage_ref,
                    _json(artifact.metadata), artifact.schema_version,
                ),
            )
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def _would_cycle(
        self, connection: sqlite3.Connection, parent_id: str, child_id: str
    ) -> bool:
        row = connection.execute(
            """WITH RECURSIVE descendants(id) AS (
                SELECT child_artifact_id FROM artifact_edges
                WHERE parent_artifact_id = ?
                UNION
                SELECT e.child_artifact_id FROM artifact_edges e
                JOIN descendants d ON e.parent_artifact_id = d.id
            ) SELECT 1 FROM descendants WHERE id = ? LIMIT 1""",
            (child_id, parent_id),
        ).fetchone()
        return parent_id == child_id or row is not None

    def add_artifact_edge(self, edge: ArtifactEdge) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            records = connection.execute(
                "SELECT artifact_id, run_id FROM artifacts WHERE artifact_id IN (?, ?)",
                (edge.parent_artifact_id, edge.child_artifact_id),
            ).fetchall()
            if len(records) != 2 or any(row["run_id"] != edge.run_id for row in records):
                raise TraceIntegrityError("artifact edge references missing/cross-run artifact")
            if self._would_cycle(
                connection, edge.parent_artifact_id, edge.child_artifact_id
            ):
                raise TraceIntegrityError("artifact lineage cycle rejected")
            existing = connection.execute(
                """SELECT created_at, schema_version FROM artifact_edges
                   WHERE parent_artifact_id = ? AND child_artifact_id = ? AND relation = ?""",
                (edge.parent_artifact_id, edge.child_artifact_id, edge.relation),
            ).fetchone()
            if existing is not None:
                if (
                    existing["created_at"] != _iso(edge.created_at)
                    or existing["schema_version"] != edge.schema_version
                ):
                    raise TraceIntegrityError("artifact edge identity conflict")
                connection.execute("COMMIT")
                return
            connection.execute(
                "INSERT INTO artifact_edges VALUES (?, ?, ?, ?, ?, ?)",
                (
                    edge.run_id, edge.parent_artifact_id, edge.child_artifact_id,
                    edge.relation, _iso(edge.created_at), edge.schema_version,
                ),
            )
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def record_replay(self, result: ReplayResult) -> None:
        payload = _json(result.model_dump(mode="json"))
        connection = self._connect()
        try:
            existing = connection.execute(
                "SELECT payload_json FROM replays WHERE replay_id = ?",
                (result.replay_id,),
            ).fetchone()
            if existing is not None:
                if existing["payload_json"] != payload:
                    raise TraceIntegrityError("replay identity content conflict")
                return
            connection.execute(
                "INSERT INTO replays VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    result.replay_id, result.original_run_id, result.artifact_id,
                    result.status, int(result.semantic_match),
                    int(result.environment_match), payload,
                    _iso(result.replayed_at), result.schema_version,
                ),
            )
        finally:
            connection.close()

    def load_artifact(self, artifact_id: str) -> tuple[ArtifactRecord, bytes]:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise TraceStorageError("unknown artifact")
        record = ArtifactRecord(
            artifact_id=row["artifact_id"], run_id=row["run_id"], kind=row["kind"],
            sha256=row["sha256"], size_bytes=row["size_bytes"],
            producer_span_id=row["producer_span_id"], created_at=row["created_at"],
            storage_ref=row["storage_ref"], metadata=json.loads(row["metadata_json"]),
            schema_version=row["schema_version"],
        )
        path = (self.artifact_root / record.storage_ref).resolve()
        if not path.is_relative_to(self.artifact_root) or not path.is_file():
            raise TraceIntegrityError("artifact_integrity_error")
        content = path.read_bytes()
        if len(content) != record.size_bytes or hashlib.sha256(content).hexdigest() != record.sha256:
            raise TraceIntegrityError("artifact_integrity_error")
        return record, content

    def get_run(self, run_id: str) -> TraceRun | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        finally:
            connection.close()
        return TraceRun.model_validate(dict(row)) if row is not None else None

    def list_runs(self, limit: int = 50) -> list[TraceRun]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?",
                (max(1, min(limit, 200)),),
            ).fetchall()
        finally:
            connection.close()
        return [TraceRun.model_validate(dict(row)) for row in rows]

    def list_spans(self, run_id: str) -> list[TraceSpan]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM spans WHERE run_id = ? ORDER BY sequence", (run_id,)
            ).fetchall()
        finally:
            connection.close()
        return [
            TraceSpan(
                span_id=row["span_id"], run_id=row["run_id"],
                parent_span_id=row["parent_span_id"], name=row["name"],
                kind=row["kind"], start_time=row["start_time"],
                end_time=row["end_time"], duration_ms=row["duration_ms"],
                status=row["status"], sequence=row["sequence"],
                metadata=json.loads(row["metadata_json"]),
                schema_version=row["schema_version"],
            ) for row in rows
        ]

    def list_events(self, run_id: str) -> list[TraceEvent]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM events WHERE run_id = ? ORDER BY sequence_number",
                (run_id,),
            ).fetchall()
        finally:
            connection.close()
        return [
            TraceEvent(
                event_id=row["event_id"], run_id=row["run_id"],
                span_id=row["span_id"], sequence_number=row["sequence_number"],
                timestamp=row["timestamp"], name=row["name"], kind=row["kind"],
                status=row["status"], metadata=json.loads(row["metadata_json"]),
                schema_version=row["schema_version"],
            ) for row in rows
        ]

    def list_artifacts(self, run_id: str) -> list[ArtifactRecord]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM artifacts WHERE run_id = ? ORDER BY created_at, artifact_id",
                (run_id,),
            ).fetchall()
        finally:
            connection.close()
        return [
            ArtifactRecord(
                artifact_id=row["artifact_id"], run_id=row["run_id"],
                kind=row["kind"], sha256=row["sha256"], size_bytes=row["size_bytes"],
                producer_span_id=row["producer_span_id"], created_at=row["created_at"],
                storage_ref=row["storage_ref"], metadata=json.loads(row["metadata_json"]),
                schema_version=row["schema_version"],
            ) for row in rows
        ]

    def list_edges(self, run_id: str) -> list[ArtifactEdge]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """SELECT * FROM artifact_edges WHERE run_id = ?
                   ORDER BY created_at, parent_artifact_id, child_artifact_id, relation""",
                (run_id,),
            ).fetchall()
        finally:
            connection.close()
        return [ArtifactEdge.model_validate(dict(row)) for row in rows]

    def reconcile_abandoned(self) -> tuple[int, int]:
        now = _iso(datetime.now(UTC))
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            spans = connection.execute(
                """UPDATE spans SET status = 'interrupted', end_time = ?
                   WHERE status = 'active'""", (now,)
            ).rowcount
            runs = connection.execute(
                """UPDATE runs SET status = 'interrupted', finished_at = ?
                   WHERE status = 'active'""", (now,)
            ).rowcount
            connection.execute("COMMIT")
            return runs, spans
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def metrics(self, run_id: str) -> RunMetrics:
        spans = self.list_spans(run_id)
        events = self.list_events(run_id)
        artifacts = self.list_artifacts(run_id)
        run = self.get_run(run_id)
        by_kind: dict[str, list[TraceSpan]] = {}
        for span in spans:
            by_kind.setdefault(span.kind, []).append(span)
        def duration(kind: str) -> float:
            return sum(item.duration_ms or 0 for item in by_kind.get(kind, []))
        token_events = [event for event in events if event.kind == "llm_call"]
        return RunMetrics(
            total_duration_ms=(
                next((item.duration_ms for item in spans if item.span_id == run.root_span_id), None)
                if run else None
            ),
            llm_duration_ms=duration("llm_call"),
            sandbox_duration_ms=duration("sandbox_execution"),
            verification_duration_ms=duration("verification"),
            tool_duration_ms=duration("tool_call"),
            llm_calls=len(by_kind.get("llm_call", [])),
            tool_calls=len(by_kind.get("tool_call", [])),
            sandbox_executions=len(by_kind.get("sandbox_execution", [])),
            input_tokens=sum(int(e.metadata.get("input_tokens", 0) or 0) for e in token_events),
            output_tokens=sum(int(e.metadata.get("output_tokens", 0) or 0) for e in token_events),
            total_tokens=sum(int(e.metadata.get("total_tokens", 0) or 0) for e in token_events),
            candidate_count=sum(1 for artifact in artifacts if artifact.kind == "candidate"),
            correction_rounds=sum(1 for e in events if e.kind == "correction"),
            artifact_count=len(artifacts),
        )

    def export_run(self, run_id: str) -> list[dict[str, object]]:
        run = self.get_run(run_id)
        if run is None:
            raise TraceStorageError("unknown trace run")
        records: list[dict[str, object]] = [
            {"record_type": "manifest", "schema_version": 1, "run_id": run_id},
            {"record_type": "run", "payload": run.model_dump(mode="json")},
        ]
        records.extend(
            {"record_type": "span", "payload": item.model_dump(mode="json")}
            for item in self.list_spans(run_id)
        )
        records.extend(
            {"record_type": "event", "payload": item.model_dump(mode="json")}
            for item in self.list_events(run_id)
        )
        records.extend(
            {"record_type": "artifact", "payload": item.model_dump(mode="json")}
            for item in self.list_artifacts(run_id)
        )
        records.extend(
            {"record_type": "artifact_edge", "payload": item.model_dump(mode="json")}
            for item in self.list_edges(run_id)
        )
        records.append(
            {"record_type": "metrics", "payload": self.metrics(run_id).model_dump(mode="json")}
        )
        return records
