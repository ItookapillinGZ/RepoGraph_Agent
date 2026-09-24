"""Secret-safe serialization helpers for evaluation data and diagnostics."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

SECRET_ENVIRONMENT_KEYS = frozenset(
    {
        "CODE_REVIEW_LLM_API_KEY",
        "OPENAI_API_KEY",
        "PDE_FRONTIER_LLM_API_KEY",
    }
)
REDACTION = "<redacted>"
_OPENAI_KEY_LIKE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_*-]{6,}", re.IGNORECASE
)
_MASKED_SECRET_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z0-9_-]{2,}\*{3,}[A-Za-z0-9_-]{2,})(?![A-Za-z0-9])"
)


def redact_environment_secrets(
    value: str,
    environment: Mapping[str, str] | None = None,
    *,
    secret_keys: set[str] | frozenset[str] | None = None,
) -> str:
    """Remove exact inherited secret values without exposing their metadata."""

    redacted = value
    source = os.environ if environment is None else environment
    selected = SECRET_ENVIRONMENT_KEYS if secret_keys is None else secret_keys
    for key, secret in source.items():
        if key.upper() in selected and secret:
            redacted = redacted.replace(secret, REDACTION)
    redacted = _OPENAI_KEY_LIKE_PATTERN.sub(REDACTION, redacted)
    return _MASKED_SECRET_PATTERN.sub(REDACTION, redacted)


def redact_secret_values(
    value: Any,
    environment: Mapping[str, str] | None = None,
) -> Any:
    """Recursively redact secrets before a value crosses a persistence boundary."""

    if isinstance(value, str):
        return redact_environment_secrets(value, environment)
    if isinstance(value, dict):
        return {
            key: redact_secret_values(item, environment) for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secret_values(item, environment) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_secret_values(item, environment) for item in value)
    return value
