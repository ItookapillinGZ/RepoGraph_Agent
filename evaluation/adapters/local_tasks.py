"""Trusted, deterministic local JSONL task adapter."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from evaluation.datasets import DatasetError, iter_jsonl, reject_duplicate_task_ids
from evaluation.models import EvaluationTask


class LocalTaskDataset:
    """Load one strict EvaluationTask per JSONL line."""

    def __init__(self, path: str | Path, *, name: str = "local") -> None:
        self.path = Path(path)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def load(self) -> list[EvaluationTask]:
        tasks: list[EvaluationTask] = []
        for line_number, payload in iter_jsonl(self.path):
            try:
                raw = json.loads(payload)
                if not isinstance(raw, dict):
                    raise TypeError("record must be a JSON object")
                raw.setdefault("dataset", self.name)
                tasks.append(EvaluationTask.model_validate(raw))
            except (json.JSONDecodeError, TypeError, ValidationError) as error:
                raise DatasetError(
                    f"Malformed local dataset record at line {line_number}: {error}"
                ) from error
        return reject_duplicate_task_ids(tasks)
