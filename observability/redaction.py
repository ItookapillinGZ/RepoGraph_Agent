"""Central redaction and telemetry bounding before persistence."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path

MAX_METADATA_BYTES = 16_384
REDACTED = "<redacted>"

_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|authorization|bearer|credential|password|passwd|proxy|secret|token|docker[_-]?auth|\.env)",
    re.IGNORECASE,
)
_VALUE_PATTERNS = (
    re.compile(r"(?i)\b(?:sk|ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?i)\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+\-/]+=*"),
    re.compile(r"(?i)(https?://)[^\s:/]+:[^\s/@]+@"),
    re.compile(r"(?im)^\s*[A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)\s*=.*$"),
)
_SAFE_USAGE_KEYS = frozenset(
    {"input_tokens", "output_tokens", "total_tokens", "max_completion_tokens"}
)


class TelemetryRedactionError(ValueError):
    pass


def _known_secret_values(environment: Mapping[str, str]) -> tuple[str, ...]:
    selected: list[str] = []
    for key, value in environment.items():
        if value and _SENSITIVE_KEY.search(key) and len(value) >= 4:
            selected.append(value)
    return tuple(sorted(set(selected), key=len, reverse=True))


def _is_sensitive_key(key: str) -> bool:
    return key.casefold() not in _SAFE_USAGE_KEYS and _SENSITIVE_KEY.search(key) is not None


def redact_text(value: str, environment: Mapping[str, str] | None = None) -> str:
    selected = os.environ if environment is None else environment
    redacted = value
    for secret in _known_secret_values(selected):
        redacted = redacted.replace(secret, REDACTED)
    for pattern in _VALUE_PATTERNS:
        redacted = pattern.sub(REDACTED, redacted)
    try:
        home = str(Path.home().resolve())
    except OSError:
        home = ""
    if home:
        redacted = redacted.replace(home, "<user-home>")
        redacted = redacted.replace(home.replace("\\", "/"), "<user-home>")
    redacted = re.sub(
        r"(?i)\b[A-Z]:[\\/]Users[\\/][^\\/\s]+[\\/]",
        "<user-home>/",
        redacted,
    )
    return redacted


def redact_value(
    value: object,
    environment: Mapping[str, str] | None = None,
    *,
    _key: str = "",
) -> object:
    if _key and _is_sensitive_key(_key):
        return REDACTED
    if isinstance(value, str):
        return redact_text(value, environment)
    if isinstance(value, Mapping):
        return {
            str(key)[:128]: redact_value(item, environment, _key=str(key))
            for key, item in list(value.items())[:64]
        }
    if isinstance(value, (list, tuple)):
        return [redact_value(item, environment) for item in value[:100]]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_text(str(value), environment)


def sanitize_metadata(
    metadata: Mapping[str, object] | None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    sanitized = redact_value(dict(metadata or {}), environment)
    if not isinstance(sanitized, dict):
        raise TelemetryRedactionError("metadata did not sanitize to an object")
    encoded = json.dumps(
        sanitized, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    if len(encoded) > MAX_METADATA_BYTES:
        raise TelemetryRedactionError(
            f"metadata exceeds MAX_METADATA_BYTES={MAX_METADATA_BYTES}"
        )
    try:
        from evaluation.artifact_security import assert_secret_free_payload

        assert_secret_free_payload(encoded, environment)
    except ImportError:
        pass
    return sanitized


def assert_safe_artifact_payload(
    payload: bytes,
    environment: Mapping[str, str] | None = None,
) -> None:
    """Reject sensitive artifact bytes; never mutate replay checkpoints."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TelemetryRedactionError("observability artifacts must be UTF-8") from error
    if redact_text(text, environment) != text:
        raise TelemetryRedactionError("artifact contains sensitive or non-portable data")
    try:
        from evaluation.artifact_security import assert_secret_free_payload

        assert_secret_free_payload(payload, environment)
    except ImportError:
        pass
