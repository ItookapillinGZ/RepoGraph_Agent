"""Per-evaluation LangChain callback telemetry without global counters."""

from __future__ import annotations

import threading
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from evaluation.models import LLMUsageSummary


class EvaluationTelemetryCollector(BaseCallbackHandler):
    """Count unique LLM runs and aggregate only provider-reported usage."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._started: set[str] = set()
        self._ended: set[str] = set()
        self._failed: set[str] = set()
        self._input_tokens = 0
        self._output_tokens = 0
        self._total_tokens = 0
        self._usage_observations = 0
        self._models: set[str] = set()
        self._configured_models: set[str] = set()
        self._resolved_models: set[str] = set()
        self._warnings: list[str] = []

    def _start(self, run_id: UUID, serialized: dict[str, Any]) -> None:
        identifier = str(run_id)
        with self._lock:
            self._started.add(identifier)
            kwargs = serialized.get("kwargs")
            if isinstance(kwargs, dict):
                model = kwargs.get("model_name") or kwargs.get("model")
                if isinstance(model, str) and model:
                    self._models.add(model)
                    self._configured_models.add(model)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        del prompts, kwargs
        self._start(run_id, serialized)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        del messages, kwargs
        self._start(run_id, serialized)

    @staticmethod
    def _integer(mapping: dict[str, Any], *keys: str) -> int | None:
        for key in keys:
            value = mapping.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return None

    @classmethod
    def _usage(cls, response: LLMResult) -> tuple[int, int, int] | None:
        llm_output = response.llm_output or {}
        raw_usage = llm_output.get("token_usage") or llm_output.get("usage")
        if isinstance(raw_usage, dict):
            input_tokens = cls._integer(raw_usage, "input_tokens", "prompt_tokens")
            output_tokens = cls._integer(
                raw_usage,
                "output_tokens",
                "completion_tokens",
            )
            total_tokens = cls._integer(raw_usage, "total_tokens")
            if input_tokens is not None and output_tokens is not None:
                return (
                    input_tokens,
                    output_tokens,
                    total_tokens
                    if total_tokens is not None
                    else input_tokens + output_tokens,
                )

        for generation_group in response.generations:
            for generation in generation_group:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None)
                if isinstance(usage, dict):
                    input_tokens = cls._integer(usage, "input_tokens")
                    output_tokens = cls._integer(usage, "output_tokens")
                    total_tokens = cls._integer(usage, "total_tokens")
                    if input_tokens is not None and output_tokens is not None:
                        return (
                            input_tokens,
                            output_tokens,
                            total_tokens
                            if total_tokens is not None
                            else input_tokens + output_tokens,
                        )
        return None

    @staticmethod
    def _resolved_model(response: LLMResult) -> str | None:
        llm_output = response.llm_output or {}
        for key in ("model_name", "model", "model_version"):
            value = llm_output.get(key)
            if isinstance(value, str) and value:
                return value
        for generation_group in response.generations:
            for generation in generation_group:
                message = getattr(generation, "message", None)
                metadata = getattr(message, "response_metadata", None)
                if isinstance(metadata, dict):
                    value = metadata.get("model_name") or metadata.get("model")
                    if isinstance(value, str) and value:
                        return value
        return None

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        del kwargs
        identifier = str(run_id)
        with self._lock:
            if identifier in self._ended:
                return
            self._started.add(identifier)
            self._ended.add(identifier)
            usage = self._usage(response)
            if usage is not None:
                self._input_tokens += usage[0]
                self._output_tokens += usage[1]
                self._total_tokens += usage[2]
                self._usage_observations += 1
            model = self._resolved_model(response)
            if model:
                self._models.add(model)
                self._resolved_models.add(model)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        del error, kwargs
        identifier = str(run_id)
        with self._lock:
            self._started.add(identifier)
            self._failed.add(identifier)

    def summary(self) -> LLMUsageSummary:
        """Return a point-in-time immutable usage snapshot."""

        with self._lock:
            calls = len(self._started)
            completed = len(self._ended)
            missing = calls - self._usage_observations
            incomplete = bool(self._failed or completed < calls or missing > 0)
            warnings = list(self._warnings)
            if missing > 0:
                warnings.append(
                    f"Provider token metadata was unavailable for {missing} LLM call(s)."
                )
            if completed < calls:
                warnings.append(
                    f"{calls - completed} LLM call(s) did not produce a completion callback."
                )
            if self._failed:
                warnings.append(f"{len(self._failed)} LLM call(s) ended with an error.")
            observed = self._usage_observations > 0
            return LLMUsageSummary(
                calls=calls,
                input_tokens=self._input_tokens if observed else None,
                output_tokens=self._output_tokens if observed else None,
                total_tokens=self._total_tokens if observed else None,
                incomplete=incomplete,
                models=sorted(self._models),
                configured_models=sorted(self._configured_models),
                resolved_models=sorted(self._resolved_models),
                warnings=list(dict.fromkeys(warnings)),
            )
