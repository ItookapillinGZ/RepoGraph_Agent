"""SQLite source of truth plus shareable per-experiment JSONL results."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from evaluation.models import (
    EvaluationConfig,
    EvaluationExperiment,
    EvaluationResult,
    EvaluationTask,
    ExecutionAttemptRecord,
    ExecutionAttemptStatus,
)


class EvaluationStorageError(RuntimeError):
    """Raised when persisted experiment identity would become ambiguous."""


class EvaluationStorage:
    """Small sequential persistence layer with deterministic upserts."""

    def __init__(self, database_path: str | Path, results_root: str | Path) -> None:
        self.database_path = Path(database_path)
        self.results_root = Path(results_root)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.results_root.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    git_commit TEXT,
                    model_name TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_experiments_name
                    ON experiments(name, created_at DESC);
                CREATE TABLE IF NOT EXISTS tasks (
                    dataset TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    task_json TEXT NOT NULL,
                    PRIMARY KEY(dataset, task_id)
                );
                CREATE TABLE IF NOT EXISTS results (
                    experiment_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(experiment_id, task_id),
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                );
                CREATE TABLE IF NOT EXISTS execution_attempts (
                    campaign_id TEXT NOT NULL,
                    config_name TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    execution_attempt INTEGER NOT NULL CHECK(execution_attempt >= 1),
                    status TEXT NOT NULL CHECK(status IN (
                        'reserved', 'running', 'completed', 'interrupted',
                        'failed', 'timeout'
                    )),
                    reserved_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    live_run_ordinal INTEGER,
                    failure_reason TEXT,
                    PRIMARY KEY(experiment_id, task_id, execution_attempt),
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_execution_attempt_active
                    ON execution_attempts(experiment_id, task_id)
                    WHERE status IN ('reserved', 'running');
                CREATE INDEX IF NOT EXISTS idx_execution_attempt_campaign
                    ON execution_attempts(campaign_id, config_name, task_id);
                """
            )

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _attempt_from_row(row: sqlite3.Row) -> ExecutionAttemptRecord:
        return ExecutionAttemptRecord(**dict(row))

    def reserve_execution_attempt(
        self,
        *,
        campaign_id: str,
        config_name: str,
        experiment_id: str,
        task_id: str,
    ) -> ExecutionAttemptRecord:
        """Atomically consume the next attempt identity before worker launch."""

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                """
                SELECT execution_attempt, status FROM execution_attempts
                WHERE experiment_id = ? AND task_id = ?
                  AND status IN ('reserved', 'running')
                """,
                (experiment_id, task_id),
            ).fetchone()
            if active is not None:
                raise EvaluationStorageError(
                    "Execution attempt already active for "
                    f"{config_name}/{task_id}: attempt "
                    f"{active['execution_attempt']} ({active['status']})."
                )
            row = connection.execute(
                """
                SELECT COALESCE(MAX(execution_attempt), 0) AS maximum
                FROM execution_attempts
                WHERE experiment_id = ? AND task_id = ?
                """,
                (experiment_id, task_id),
            ).fetchone()
            attempt = int(row["maximum"]) + 1
            reserved_at = self._utc_now()
            connection.execute(
                """
                INSERT INTO execution_attempts(
                    campaign_id, config_name, experiment_id, task_id,
                    execution_attempt, status, reserved_at
                ) VALUES (?, ?, ?, ?, ?, 'reserved', ?)
                """,
                (
                    campaign_id,
                    config_name,
                    experiment_id,
                    task_id,
                    attempt,
                    reserved_at,
                ),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
        return ExecutionAttemptRecord(
            campaign_id=campaign_id,
            config_name=config_name,
            experiment_id=experiment_id,
            task_id=task_id,
            execution_attempt=attempt,
            status="reserved",
            reserved_at=reserved_at,
        )

    def set_execution_attempt_live_ordinal(
        self,
        record: ExecutionAttemptRecord,
        ordinal: int,
    ) -> ExecutionAttemptRecord:
        """Link the append-only live-budget evidence to its SQLite identity."""

        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE execution_attempts SET live_run_ordinal = ?
                WHERE experiment_id = ? AND task_id = ?
                  AND execution_attempt = ? AND status = 'reserved'
                  AND live_run_ordinal IS NULL
                """,
                (
                    ordinal,
                    record.experiment_id,
                    record.task_id,
                    record.execution_attempt,
                ),
            )
            if cursor.rowcount != 1:
                raise EvaluationStorageError("Execution reservation ordinal link failed.")
        return record.model_copy(update={"live_run_ordinal": ordinal})

    def mark_execution_attempt_running(
        self, record: ExecutionAttemptRecord
    ) -> ExecutionAttemptRecord:
        """Persist worker-start intent before entering the worker boundary."""

        started_at = self._utc_now()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE execution_attempts SET status = 'running', started_at = ?
                WHERE experiment_id = ? AND task_id = ?
                  AND execution_attempt = ? AND status = 'reserved'
                """,
                (
                    started_at,
                    record.experiment_id,
                    record.task_id,
                    record.execution_attempt,
                ),
            )
            if cursor.rowcount != 1:
                raise EvaluationStorageError("Execution reservation is not startable.")
        return record.model_copy(update={"status": "running", "started_at": started_at})

    def finish_execution_attempt(
        self,
        record: ExecutionAttemptRecord,
        *,
        status: ExecutionAttemptStatus,
        failure_reason: str | None = None,
    ) -> ExecutionAttemptRecord:
        """Finish a reserved/running identity without ever deleting or recycling it."""

        if status not in {"completed", "interrupted", "failed", "timeout"}:
            raise ValueError("Execution attempt requires a terminal status.")
        finished_at = self._utc_now()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE execution_attempts
                SET status = ?, finished_at = ?, failure_reason = ?
                WHERE experiment_id = ? AND task_id = ?
                  AND execution_attempt = ?
                  AND status IN ('reserved', 'running')
                """,
                (
                    status,
                    finished_at,
                    failure_reason,
                    record.experiment_id,
                    record.task_id,
                    record.execution_attempt,
                ),
            )
            if cursor.rowcount != 1:
                raise EvaluationStorageError("Execution attempt is already terminal.")
        return record.model_copy(
            update={
                "status": status,
                "finished_at": finished_at,
                "failure_reason": failure_reason,
            }
        )

    def list_execution_attempts(
        self,
        *,
        experiment_id: str | None = None,
        task_id: str | None = None,
    ) -> list[ExecutionAttemptRecord]:
        with self._connection() as connection:
            if experiment_id is not None and task_id is not None:
                rows = connection.execute(
                    """
                    SELECT * FROM execution_attempts
                    WHERE experiment_id = ? AND task_id = ?
                    ORDER BY experiment_id, task_id, execution_attempt
                    """,
                    (experiment_id, task_id),
                ).fetchall()
            elif experiment_id is not None:
                rows = connection.execute(
                    """
                    SELECT * FROM execution_attempts
                    WHERE experiment_id = ?
                    ORDER BY experiment_id, task_id, execution_attempt
                    """,
                    (experiment_id,),
                ).fetchall()
            elif task_id is not None:
                rows = connection.execute(
                    """
                    SELECT * FROM execution_attempts
                    WHERE task_id = ?
                    ORDER BY experiment_id, task_id, execution_attempt
                    """,
                    (task_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM execution_attempts
                    ORDER BY experiment_id, task_id, execution_attempt
                    """
                ).fetchall()
        return [self._attempt_from_row(row) for row in rows]

    def reconcile_incomplete_attempts(
        self,
        *,
        campaign_id: str,
        reason: str,
    ) -> int:
        """Explicitly preserve stale reserved/running rows as interrupted."""

        finished_at = self._utc_now()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE execution_attempts
                SET status = 'interrupted', finished_at = ?, failure_reason = ?
                WHERE campaign_id = ? AND status IN ('reserved', 'running')
                """,
                (finished_at, reason, campaign_id),
            )
        return int(cursor.rowcount)

    def import_execution_attempt(
        self,
        record: ExecutionAttemptRecord,
    ) -> ExecutionAttemptRecord:
        """Import immutable legacy evidence without allocating a new identity."""

        validated = ExecutionAttemptRecord.model_validate(record)
        with self._connection() as connection:
            existing = connection.execute(
                """
                SELECT * FROM execution_attempts
                WHERE experiment_id = ? AND task_id = ? AND execution_attempt = ?
                """,
                (
                    validated.experiment_id,
                    validated.task_id,
                    validated.execution_attempt,
                ),
            ).fetchone()
            if existing is not None:
                persisted = self._attempt_from_row(existing)
                if persisted != validated:
                    raise EvaluationStorageError(
                        "Legacy execution attempt conflicts with persisted identity."
                    )
                return persisted
            connection.execute(
                """
                INSERT INTO execution_attempts(
                    campaign_id, config_name, experiment_id, task_id,
                    execution_attempt, status, reserved_at, started_at,
                    finished_at, live_run_ordinal, failure_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    validated.campaign_id,
                    validated.config_name,
                    validated.experiment_id,
                    validated.task_id,
                    validated.execution_attempt,
                    validated.status,
                    validated.reserved_at,
                    validated.started_at,
                    validated.finished_at,
                    validated.live_run_ordinal,
                    validated.failure_reason,
                ),
            )
        return validated

    def list_experiments(self) -> list[EvaluationExperiment]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM experiments ORDER BY created_at, id"
            ).fetchall()
        return [self._experiment_from_row(row) for row in rows]

    def list_tasks(self, dataset: str | None = None) -> list[EvaluationTask]:
        with self._connection() as connection:
            if dataset is None:
                rows = connection.execute(
                    "SELECT task_json FROM tasks ORDER BY dataset, task_id"
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT task_json FROM tasks
                    WHERE dataset = ? ORDER BY task_id
                    """,
                    (dataset,),
                ).fetchall()
        return [EvaluationTask.model_validate_json(row["task_json"]) for row in rows]

    def save_experiment(self, experiment: EvaluationExperiment) -> None:
        payload = experiment.config.model_dump_json()
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment.id,)
            ).fetchone()
            if existing is not None:
                persisted = self._experiment_from_row(existing)
                if persisted != experiment:
                    raise EvaluationStorageError(
                        f"Experiment id already has different metadata: {experiment.id}"
                    )
                return
            connection.execute(
                """
                INSERT INTO experiments(
                    id, name, dataset, config_json, created_at, git_commit, model_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment.id,
                    experiment.name,
                    experiment.dataset,
                    payload,
                    experiment.created_at,
                    experiment.git_commit,
                    experiment.model_name,
                ),
            )

    def save_task(self, task: EvaluationTask) -> None:
        payload = task.model_dump_json()
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT task_json FROM tasks WHERE dataset = ? AND task_id = ?",
                (task.dataset, task.id),
            ).fetchone()
            if existing is not None and existing["task_json"] != payload:
                raise EvaluationStorageError(
                    f"Task identity changed for {task.dataset}/{task.id}."
                )
            connection.execute(
                """
                INSERT OR IGNORE INTO tasks(dataset, task_id, task_json)
                VALUES (?, ?, ?)
                """,
                (task.dataset, task.id, payload),
            )

    def save_result(self, result: EvaluationResult) -> None:
        payload = result.model_dump_json()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO results(
                    experiment_id, task_id, status, result_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(experiment_id, task_id) DO UPDATE SET
                    status=excluded.status,
                    result_json=excluded.result_json,
                    created_at=excluded.created_at
                """,
                (
                    result.experiment,
                    result.task_id,
                    result.status,
                    payload,
                    result.created_at,
                ),
            )
        self._append_jsonl(result)

    def _append_jsonl(self, result: EvaluationResult) -> None:
        destination = self.results_root / f"{result.experiment}.jsonl"
        with destination.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    result.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    def get_result(self, experiment_id: str, task_id: str) -> EvaluationResult | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT result_json FROM results
                WHERE experiment_id = ? AND task_id = ?
                """,
                (experiment_id, task_id),
            ).fetchone()
        return EvaluationResult.model_validate_json(row["result_json"]) if row else None

    def list_results(self, experiment_id: str) -> list[EvaluationResult]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT result_json FROM results
                WHERE experiment_id = ? ORDER BY task_id
                """,
                (experiment_id,),
            ).fetchall()
        return [
            EvaluationResult.model_validate_json(row["result_json"]) for row in rows
        ]

    def get_experiment(self, identifier: str) -> EvaluationExperiment | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE id = ?", (identifier,)
            ).fetchone()
            if row is None:
                row = connection.execute(
                    """
                    SELECT * FROM experiments WHERE name = ?
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (identifier,),
                ).fetchone()
        return self._experiment_from_row(row) if row else None

    @staticmethod
    def _experiment_from_row(row: sqlite3.Row) -> EvaluationExperiment:
        config = EvaluationConfig.model_validate_json(row["config_json"])
        return EvaluationExperiment(
            id=row["id"],
            name=row["name"],
            dataset=row["dataset"],
            config=config,
            created_at=row["created_at"],
            git_commit=row["git_commit"],
            model_name=row["model_name"],
            configured_model_name=row["model_name"],
            llm_execution_kind=config.llm_execution_kind,
        )
