"""Dataset interfaces and strict JSONL loading helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Protocol

from evaluation.models import EvaluationTask


class DatasetError(ValueError):
    """Raised when trusted benchmark metadata is malformed or ambiguous."""


class EvaluationDataset(Protocol):
    """Small adapter boundary shared by local and external datasets."""

    @property
    def name(self) -> str: ...

    def load(self) -> Sequence[EvaluationTask]: ...


def reject_duplicate_task_ids(tasks: Iterable[EvaluationTask]) -> list[EvaluationTask]:
    """Materialize tasks and reject IDs that would break result uniqueness."""

    materialized: list[EvaluationTask] = []
    seen: set[str] = set()
    for task in tasks:
        if task.id in seen:
            raise DatasetError(f"Duplicate evaluation task id: {task.id}")
        seen.add(task.id)
        materialized.append(task)
    return materialized


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, str]]:
    """Yield non-empty UTF-8 JSONL records with stable source line numbers."""

    dataset_path = Path(path).resolve(strict=True)
    if not dataset_path.is_file():
        raise DatasetError(f"Dataset path is not a file: {dataset_path}")
    try:
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                payload = line.strip()
                if payload:
                    yield line_number, payload
    except UnicodeError as error:
        raise DatasetError(f"Dataset is not valid UTF-8: {dataset_path}") from error
