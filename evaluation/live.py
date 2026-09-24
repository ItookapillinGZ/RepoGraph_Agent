"""Explicitly authorized live-LLM preflight and campaign budget controls."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from evaluation.security import redact_environment_secrets
from evaluation.telemetry import EvaluationTelemetryCollector
from model_defaults import (
    ModelConfigurationError,
    ProductionModelSettings,
    create_production_chat_model,
    get_production_model_settings,
)

MAX_H2_2_LIVE_TASK_RUNS = 45
MAX_H2_3_LIVE_TASK_RUNS = 80
DEFAULT_LIVE_TASK_TIMEOUT_SECONDS = 900.0


class LiveGuardError(RuntimeError):
    """Raised before any provider call when the live boundary is not satisfied."""


class LiveBudgetError(RuntimeError):
    """Raised before a task-run would exceed the persisted hard budget."""


class _PreflightResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]


class LivePreflightResult(BaseModel):
    """Bounded provider preflight output with no prompt or credential data."""

    model_config = ConfigDict(extra="forbid")

    api_key_status: Literal["configured"] = "configured"
    structured_output_succeeded: bool
    configured_model: str
    resolved_models: list[str] = Field(default_factory=list, max_length=20)
    model_temperature: float | None
    model_seed: int | None
    configured_provider: str
    api_base_url: str | None = None
    reasoning_effort: str | None = None
    max_completion_tokens: int | None = Field(default=None, ge=1)
    llm_calls: int = Field(ge=1)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    telemetry_incomplete: bool
    timestamp: str


class LiveRunReservation(BaseModel):
    """Append-only evidence that a live benchmark task-run was started."""

    model_config = ConfigDict(extra="forbid")

    ordinal: int = Field(ge=1, le=MAX_H2_3_LIVE_TASK_RUNS)
    experiment_id: str = Field(min_length=1, max_length=200)
    task_id: str = Field(min_length=1, max_length=200)
    config_name: str = Field(min_length=1, max_length=200)
    timestamp: str


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def prepare_live_environment() -> ProductionModelSettings:
    """Load dotenv and validate the configured server-side provider credentials."""

    load_dotenv()
    try:
        return get_production_model_settings(require_api_key=True)
    except ModelConfigurationError as error:
        detail = redact_environment_secrets(str(error))
        raise LiveGuardError(detail) from error


def require_live_flag(*, live: bool, dry_run: bool) -> None:
    """Ensure no provider call is reachable without an explicit cost flag."""

    if live and dry_run:
        raise LiveGuardError("--live and --dry-run are mutually exclusive")
    if not live and not dry_run:
        raise LiveGuardError("Live LLM calls require explicit --live")


def _safe_preflight_failure(error: Exception) -> str:
    """Describe a provider failure without persisting its response body."""

    error_type = type(error).__name__
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return f"Live API preflight failed ({error_type}, HTTP {status_code})."
    return f"Live API preflight failed ({error_type})."


def run_live_preflight(
    *,
    model_factory: Callable[[], BaseChatModel] | None = None,
    settings: ProductionModelSettings | None = None,
) -> LivePreflightResult:
    """Make one small real structured request and verify callback telemetry."""

    selected = settings or prepare_live_environment()
    collector = EvaluationTelemetryCollector()
    try:
        model = (
            model_factory()
            if model_factory is not None
            else create_production_chat_model(settings=selected)
        )
        structured = model.with_structured_output(_PreflightResponse)
        response = structured.invoke(
            [
                SystemMessage(
                    content="Return the requested bounded health-check schema only."
                ),
                HumanMessage(content="Set status to ok."),
            ],
            config={"callbacks": [collector]},
        )
        validated = _PreflightResponse.model_validate(response)
    except Exception as error:
        raise LiveGuardError(_safe_preflight_failure(error)) from error
    usage = collector.summary()
    if usage.calls < 1:
        raise LiveGuardError(
            "Live API preflight recorded llm_calls=0; fake or uninstrumented path detected."
        )
    return LivePreflightResult(
        structured_output_succeeded=validated.status == "ok",
        configured_provider=selected.provider,
        api_base_url=selected.base_url,
        configured_model=selected.model,
        resolved_models=usage.resolved_models,
        model_temperature=selected.temperature,
        model_seed=selected.seed,
        reasoning_effort=selected.reasoning_effort,
        max_completion_tokens=selected.max_completion_tokens,
        llm_calls=usage.calls,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        telemetry_incomplete=usage.incomplete,
        timestamp=utc_now(),
    )


class LiveRunBudget:
    """Persist and enforce the global H2.2 live task-run ceiling."""

    def __init__(self, path: str | Path, *, maximum: int) -> None:
        if maximum < 1 or maximum > MAX_H2_3_LIVE_TASK_RUNS:
            raise ValueError(f"max live task-runs must be 1..{MAX_H2_3_LIVE_TASK_RUNS}")
        self.path = Path(path)
        self.maximum = maximum
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def reservations(self) -> list[LiveRunReservation]:
        if not self.path.exists():
            return []
        reservations: list[LiveRunReservation] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        reservations.append(
                            LiveRunReservation.model_validate_json(line)
                        )
                    except ValueError as error:
                        raise LiveBudgetError(
                            f"Malformed live budget ledger at line {line_number}."
                        ) from error
        return reservations

    @property
    def consumed(self) -> int:
        return len(self.reservations())

    def reserve(
        self,
        *,
        experiment_id: str,
        task_id: str,
        config_name: str,
    ) -> LiveRunReservation:
        """Append a reservation before launching a live task worker."""

        consumed = self.consumed
        if consumed >= self.maximum:
            raise LiveBudgetError(
                f"Live task-run budget exhausted ({consumed}/{self.maximum})."
            )
        reservation = LiveRunReservation(
            ordinal=consumed + 1,
            experiment_id=experiment_id,
            task_id=task_id,
            config_name=config_name,
            timestamp=utc_now(),
        )
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(reservation.model_dump(mode="json"), sort_keys=True)
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return reservation
