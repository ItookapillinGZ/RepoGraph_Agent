"""SQLite source of truth for Studio runs, events, and artifacts."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from observability.storage import SQLiteTraceSink
from studio_backend.config import (
    MAX_STUDIO_ARTIFACT_CHARS,
    MAX_STUDIO_EVENT_METADATA_CHARS,
)
from studio_backend.models import ArtifactResponse, RunEvent, StudioRun


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _json_text(payload: object, *, limit: int, label: str) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if len(encoded) > limit:
        raise ValueError(f"{label} exceeds its Studio storage budget.")
    return encoded


class StudioStorage:
    """Small thread-safe SQLite gateway; each operation owns its connection."""

    def __init__(self, database_path: Path) -> None:
        self._path = database_path
        self.trace_sink = SQLiteTraceSink(
            database_path.parent / "repograph-observability.sqlite3",
            database_path.parent / "trace-artifacts",
        )
        self._initialize_lock = threading.Lock()
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._path,
            timeout=10,
            isolation_level=None,
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
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS runs (
                        id TEXT PRIMARY KEY,
                        repository_id TEXT NOT NULL,
                        task TEXT NOT NULL,
                        status TEXT NOT NULL,
                        phase TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        approval_digest TEXT,
                        failure_reason TEXT
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS events (
                        run_id TEXT NOT NULL,
                        sequence INTEGER NOT NULL,
                        timestamp TEXT NOT NULL,
                        phase TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        title TEXT NOT NULL,
                        message TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        PRIMARY KEY (run_id, sequence),
                        FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS artifacts (
                        run_id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        version INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        PRIMARY KEY (run_id, kind, version),
                        FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_runs_created_at
                    ON runs(created_at DESC)
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_events_run_sequence
                    ON events(run_id, sequence)
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_artifacts_run_kind
                    ON artifacts(run_id, kind, version DESC)
                    """
                )
                connection.execute("PRAGMA optimize")
            finally:
                connection.close()

    @staticmethod
    def _run_from_row(row: sqlite3.Row | None) -> StudioRun | None:
        if row is None:
            return None
        return StudioRun.model_validate(dict(row))

    def create_run(self, run: StudioRun) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO runs (
                    id, repository_id, task, status, phase, created_at,
                    updated_at, approval_digest, failure_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.repository_id,
                    run.task,
                    run.status,
                    run.phase,
                    run.created_at,
                    run.updated_at,
                    run.approval_digest,
                    run.failure_reason,
                ),
            )
        finally:
            connection.close()

    def get_run(self, run_id: str) -> StudioRun | None:
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT id, repository_id, task, status, phase, created_at,
                       updated_at, approval_digest, failure_reason
                FROM runs WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
            return self._run_from_row(row)
        finally:
            connection.close()

    def list_runs(self, limit: int = 50) -> list[StudioRun]:
        bounded = max(1, min(limit, 50))
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT id, repository_id, task, status, phase, created_at,
                       updated_at, approval_digest, failure_reason
                FROM runs ORDER BY created_at DESC LIMIT ?
                """,
                (bounded,),
            ).fetchall()
            return [StudioRun.model_validate(dict(row)) for row in rows]
        finally:
            connection.close()

    def list_runs_by_status(self, statuses: Iterable[str]) -> list[StudioRun]:
        selected = tuple(dict.fromkeys(statuses))
        if not selected:
            return []
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT id, repository_id, task, status, phase, created_at,
                       updated_at, approval_digest, failure_reason
                FROM runs
                WHERE status IN (SELECT value FROM json_each(?))
                ORDER BY created_at ASC
                """,
                (json.dumps(selected),),
            ).fetchall()
            return [StudioRun.model_validate(dict(row)) for row in rows]
        finally:
            connection.close()

    def update_run(
        self,
        run_id: str,
        *,
        status: str,
        phase: str,
        approval_digest: str | None = None,
        failure_reason: str | None = None,
    ) -> StudioRun | None:
        connection = self._connect()
        try:
            connection.execute(
                """
                UPDATE runs
                SET status = ?, phase = ?, updated_at = ?,
                    approval_digest = COALESCE(?, approval_digest),
                    failure_reason = ?
                WHERE id = ?
                """,
                (
                    status,
                    phase,
                    utc_now(),
                    approval_digest,
                    failure_reason,
                    run_id,
                ),
            )
        finally:
            connection.close()
        return self.get_run(run_id)

    def compare_and_set_run(
        self,
        run_id: str,
        *,
        expected_statuses: Iterable[str],
        status: str,
        phase: str,
    ) -> StudioRun | None:
        statuses = tuple(expected_statuses)
        if not statuses:
            raise ValueError("expected_statuses must not be empty")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT status FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if current is None or current["status"] not in statuses:
                connection.execute("ROLLBACK")
                return None
            cursor = connection.execute(
                """
                UPDATE runs SET status = ?, phase = ?, updated_at = ?
                WHERE id = ? AND status = ?
                """,
                (status, phase, utc_now(), run_id, current["status"]),
            )
            if cursor.rowcount != 1:
                connection.execute("ROLLBACK")
                return None
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        return self.get_run(run_id)

    def mark_active_interrupted(self) -> int:
        active = ("queued", "running", "applying", "git_creating", "publishing")
        connection = self._connect()
        try:
            cursor = connection.execute(
                """
                UPDATE runs
                SET status = 'interrupted', phase = 'failed', updated_at = ?,
                    failure_reason = 'Studio execution was interrupted by a server restart.'
                WHERE status IN (?, ?, ?, ?, ?)
                """,
                (utc_now(), *active),
            )
            return cursor.rowcount
        finally:
            connection.close()

    def append_event(
        self,
        run_id: str,
        *,
        phase: str,
        event_type: str,
        status: str,
        title: str,
        message: str = "",
        metadata: dict[str, object] | None = None,
    ) -> RunEvent:
        metadata_text = _json_text(
            metadata or {},
            limit=MAX_STUDIO_EVENT_METADATA_CHARS,
            label="event metadata",
        )
        timestamp = utc_now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 AS next FROM events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            sequence = int(row["next"])
            connection.execute(
                """
                INSERT INTO events (
                    run_id, sequence, timestamp, phase, event_type, status,
                    title, message, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    sequence,
                    timestamp,
                    phase,
                    event_type,
                    status,
                    title,
                    message,
                    metadata_text,
                ),
            )
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        return RunEvent(
            run_id=run_id,
            sequence=sequence,
            timestamp=timestamp,
            phase=phase,
            event_type=event_type,
            status=status,
            title=title,
            message=message,
            metadata=json.loads(metadata_text),
        )

    def list_events(self, run_id: str, *, after: int = 0) -> list[RunEvent]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT run_id, sequence, timestamp, phase, event_type, status,
                       title, message, metadata_json
                FROM events
                WHERE run_id = ? AND sequence > ?
                ORDER BY sequence ASC
                """,
                (run_id, max(0, after)),
            ).fetchall()
            return [
                RunEvent(
                    run_id=row["run_id"],
                    sequence=row["sequence"],
                    timestamp=row["timestamp"],
                    phase=row["phase"],
                    event_type=row["event_type"],
                    status=row["status"],
                    title=row["title"],
                    message=row["message"],
                    metadata=json.loads(row["metadata_json"]),
                )
                for row in rows
            ]
        finally:
            connection.close()

    def latest_event_sequence(self, run_id: str) -> int:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) AS latest FROM events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            return int(row["latest"])
        finally:
            connection.close()

    def put_artifact(self, run_id: str, kind: str, payload: object) -> ArtifactResponse:
        payload_text = _json_text(
            payload,
            limit=MAX_STUDIO_ARTIFACT_CHARS,
            label="artifact",
        )
        created_at = utc_now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT COALESCE(MAX(version), 0) + 1 AS next
                FROM artifacts WHERE run_id = ? AND kind = ?
                """,
                (run_id, kind),
            ).fetchone()
            version = int(row["next"])
            connection.execute(
                """
                INSERT INTO artifacts (run_id, kind, version, created_at, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, kind, version, created_at, payload_text),
            )
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        return ArtifactResponse(
            run_id=run_id,
            kind=kind,
            version=version,
            created_at=created_at,
            payload=json.loads(payload_text),
        )

    def get_artifact(self, run_id: str, kind: str) -> ArtifactResponse | None:
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT run_id, kind, version, created_at, payload_json
                FROM artifacts
                WHERE run_id = ? AND kind = ?
                ORDER BY version DESC LIMIT 1
                """,
                (run_id, kind),
            ).fetchone()
            if row is None:
                return None
            return ArtifactResponse(
                run_id=row["run_id"],
                kind=row["kind"],
                version=row["version"],
                created_at=row["created_at"],
                payload=json.loads(row["payload_json"]),
            )
        finally:
            connection.close()

    def artifact_kinds(self, run_id: str) -> list[str]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT DISTINCT kind FROM artifacts WHERE run_id = ? ORDER BY kind",
                (run_id,),
            ).fetchall()
            return [str(row["kind"]) for row in rows]
        finally:
            connection.close()


def model_payload(model: Any) -> object:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model
