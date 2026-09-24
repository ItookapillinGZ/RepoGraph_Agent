"""Schema-only SWE-bench adapter and official prediction exporter.

Docker grading is deliberately outside Stage H2.  The accepted fields mirror
the official SWE-bench dataset and prediction contracts as of 2026-09-02.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from evaluation.datasets import DatasetError, iter_jsonl, reject_duplicate_task_ids
from evaluation.models import EvaluationResult, EvaluationTask


class SWEbenchRecord(BaseModel):
    """Required task-solving and grading metadata from one SWE-bench row."""

    model_config = ConfigDict(extra="allow")

    instance_id: str = Field(min_length=1)
    repo: str = Field(min_length=1)
    base_commit: str = Field(min_length=1)
    problem_statement: str = Field(min_length=1)
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    test_patch: str = ""
    version: str | None = None

    @field_validator("FAIL_TO_PASS", "PASS_TO_PASS", mode="before")
    @classmethod
    def parse_test_lists(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as error:
                raise ValueError("test selector field is not valid JSON") from error
        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            raise ValueError("test selector field must be a list of strings")
        return value


class SWEbenchDataset:
    """Load exported SWE-bench JSONL without adding optional dependencies."""

    def __init__(self, path: str | Path, *, name: str = "swebench") -> None:
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
                record = SWEbenchRecord.model_validate(raw)
            except (json.JSONDecodeError, ValidationError) as error:
                raise DatasetError(
                    f"Malformed SWE-bench record at line {line_number}: {error}"
                ) from error
            tasks.append(
                EvaluationTask(
                    id=record.instance_id,
                    dataset=self.name,
                    repository=record.repo,
                    base_commit=record.base_commit,
                    task=record.problem_statement,
                    test_command=None,
                    metadata={
                        "instance_id": record.instance_id,
                        "repo": record.repo,
                        "FAIL_TO_PASS": record.FAIL_TO_PASS,
                        "PASS_TO_PASS": record.PASS_TO_PASS,
                        "test_patch": record.test_patch,
                        "version": record.version,
                    },
                )
            )
        return reject_duplicate_task_ids(tasks)


def export_predictions(
    results: list[EvaluationResult],
    destination: str | Path,
    *,
    model_name_or_path: str,
) -> Path:
    """Write current official SWE-bench JSONL prediction fields."""

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    {
                        "instance_id": result.task_id,
                        "model_name_or_path": model_name_or_path,
                        "model_patch": result.model_patch or "",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    return output
