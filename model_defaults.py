"""Single source of truth for RepoGraph's production chat-model defaults."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

DEFAULT_PRODUCTION_MODEL = "gpt-5.6-luna"
DEFAULT_PRODUCTION_TEMPERATURE = 0.0
DEFAULT_PRODUCTION_SEED: int | None = None
PROJECT_LLM_PREFIX = "CODE_REVIEW_LLM_"
LEGACY_LLM_PREFIX = "PDE_FRONTIER_LLM_"
_SUPPORTED_PROVIDERS = frozenset({"openai", "openai_compatible"})
_SUPPORTED_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)


class ModelConfigurationError(ValueError):
    """Raised before a model call when provider settings are invalid."""


@dataclass(frozen=True)
class ProductionModelSettings:
    """Secret-safe validated settings for an OpenAI-compatible chat model."""

    provider: str
    model: str
    temperature: float | None
    seed: int | None
    api_key: str | None = field(repr=False, compare=False)
    base_url: str | None
    timeout_seconds: float | None
    reasoning_effort: str | None
    max_completion_tokens: int | None


def _first_value(source: Mapping[str, str], *keys: str) -> str | None:
    for key in keys:
        value = source.get(key)
        if value is not None and value.strip():
            return value.strip()
    return None


def _optional_float(
    value: str | None,
    *,
    label: str,
    minimum: float,
    maximum: float,
) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError as error:
        raise ModelConfigurationError(f"{label} must be numeric.") from error
    if parsed < minimum or parsed > maximum:
        raise ModelConfigurationError(
            f"{label} must be between {minimum:g} and {maximum:g}."
        )
    return parsed


def _optional_int(
    value: str | None,
    *,
    label: str,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise ModelConfigurationError(f"{label} must be an integer.") from error
    if parsed < minimum or parsed > maximum:
        raise ModelConfigurationError(
            f"{label} must be between {minimum} and {maximum}."
        )
    return parsed


def _validated_base_url(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ModelConfigurationError(
            "CODE_REVIEW_LLM_BASE_URL must be a plain HTTP(S) API base URL."
        )
    return value.rstrip("/")


def get_production_model_settings(
    environment: Mapping[str, str] | None = None,
    *,
    require_api_key: bool = False,
) -> ProductionModelSettings:
    """Resolve project, legacy, then standard OpenAI-compatible settings."""

    source = os.environ if environment is None else environment
    api_key = _first_value(
        source,
        f"{PROJECT_LLM_PREFIX}API_KEY",
        f"{LEGACY_LLM_PREFIX}API_KEY",
        "OPENAI_API_KEY",
    )
    base_url = _validated_base_url(
        _first_value(
            source,
            f"{PROJECT_LLM_PREFIX}BASE_URL",
            f"{LEGACY_LLM_PREFIX}BASE_URL",
            "OPENAI_BASE_URL",
            "OPENAI_API_BASE",
        )
    )
    provider = (
        _first_value(
            source,
            f"{PROJECT_LLM_PREFIX}PROVIDER",
            f"{LEGACY_LLM_PREFIX}PROVIDER",
        )
        or ("openai_compatible" if base_url else "openai")
    ).lower()
    if provider not in _SUPPORTED_PROVIDERS:
        raise ModelConfigurationError(
            "CODE_REVIEW_LLM_PROVIDER must be openai or openai_compatible."
        )
    if provider == "openai_compatible" and base_url is None:
        raise ModelConfigurationError(
            "An OpenAI-compatible provider requires CODE_REVIEW_LLM_BASE_URL."
        )
    if require_api_key and api_key is None:
        raise ModelConfigurationError(
            "CODE_REVIEW_LLM_API_KEY (or a supported fallback) is not configured."
        )
    reasoning_effort = _first_value(
        source,
        f"{PROJECT_LLM_PREFIX}REASONING_EFFORT",
        f"{LEGACY_LLM_PREFIX}REASONING_EFFORT",
    )
    if reasoning_effort is not None:
        reasoning_effort = reasoning_effort.lower()
        if reasoning_effort not in _SUPPORTED_REASONING_EFFORTS:
            raise ModelConfigurationError(
                "CODE_REVIEW_LLM_REASONING_EFFORT is not supported."
            )
    temperature_value = _first_value(
        source,
        f"{PROJECT_LLM_PREFIX}TEMPERATURE",
        f"{LEGACY_LLM_PREFIX}TEMPERATURE",
    )
    seed_value = _first_value(
        source,
        f"{PROJECT_LLM_PREFIX}SEED",
        f"{LEGACY_LLM_PREFIX}SEED",
    )
    return ProductionModelSettings(
        provider=provider,
        model=_first_value(
            source,
            f"{PROJECT_LLM_PREFIX}MODEL",
            f"{LEGACY_LLM_PREFIX}MODEL",
        )
        or DEFAULT_PRODUCTION_MODEL,
        temperature=(
            DEFAULT_PRODUCTION_TEMPERATURE
            if temperature_value is None
            else _optional_float(
                temperature_value,
                label="CODE_REVIEW_LLM_TEMPERATURE",
                minimum=0,
                maximum=2,
            )
        ),
        seed=(
            DEFAULT_PRODUCTION_SEED
            if seed_value is None
            else _optional_int(
                seed_value,
                label="CODE_REVIEW_LLM_SEED",
                minimum=0,
                maximum=2_147_483_647,
            )
        ),
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=_optional_float(
            _first_value(
                source,
                f"{PROJECT_LLM_PREFIX}TIMEOUT_SECONDS",
                f"{LEGACY_LLM_PREFIX}TIMEOUT_SECONDS",
            ),
            label="CODE_REVIEW_LLM_TIMEOUT_SECONDS",
            minimum=1,
            maximum=3_600,
        ),
        reasoning_effort=reasoning_effort,
        max_completion_tokens=_optional_int(
            _first_value(
                source,
                f"{PROJECT_LLM_PREFIX}MAX_COMPLETION_TOKENS",
                f"{LEGACY_LLM_PREFIX}MAX_COMPLETION_TOKENS",
            ),
            label="CODE_REVIEW_LLM_MAX_COMPLETION_TOKENS",
            minimum=1,
            maximum=1_000_000,
        ),
    )


def create_production_chat_model(
    *,
    chat_model_class: type[ChatOpenAI] = ChatOpenAI,
    settings: ProductionModelSettings | None = None,
) -> ChatOpenAI:
    """Build the configured OpenAI or OpenAI-compatible production model."""

    selected = settings or get_production_model_settings()
    arguments: dict[str, object] = {
        "model": selected.model,
        "temperature": selected.temperature,
    }
    if selected.api_key is not None:
        arguments["api_key"] = SecretStr(selected.api_key)
    if selected.base_url is not None:
        arguments["base_url"] = selected.base_url
    if selected.timeout_seconds is not None:
        arguments["timeout"] = selected.timeout_seconds
    if selected.seed is not None:
        arguments["seed"] = selected.seed
    if selected.reasoning_effort is not None:
        arguments["reasoning_effort"] = selected.reasoning_effort
    if selected.max_completion_tokens is not None:
        arguments["max_completion_tokens"] = selected.max_completion_tokens
    if selected.provider == "openai_compatible":
        arguments["use_responses_api"] = False
    return chat_model_class(**arguments)
