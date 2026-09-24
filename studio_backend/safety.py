"""Public-response redaction without mutating persisted approval artifacts."""

from __future__ import annotations

from collections.abc import Iterable


def sanitize_public_payload(value: object, sensitive_values: Iterable[str]) -> object:
    """Recursively replace exact known server values in a response copy."""

    values = tuple(
        sorted(
            {item for item in sensitive_values if item},
            key=len,
            reverse=True,
        )
    )
    if isinstance(value, str):
        sanitized = value
        for sensitive in values:
            sanitized = sanitized.replace(sensitive, "<redacted>")
        return sanitized
    if isinstance(value, list):
        return [sanitize_public_payload(item, values) for item in value]
    if isinstance(value, tuple):
        return [sanitize_public_payload(item, values) for item in value]
    if isinstance(value, dict):
        return {
            str(key): sanitize_public_payload(item, values)
            for key, item in value.items()
        }
    return value
