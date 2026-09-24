"""Official SWE-bench Docker harness adapter with cache-safe identities."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess  # nosec B404
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from evaluation.process_runner import (
    FixedProcessOutcome,
    redact_environment_secrets,
    run_fixed_argv,
)

MAX_PREDICTION_BYTES = 10_000_000
MAX_SWEBENCH_FAILURE_CHARS = 4_000
_SAFE_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,299}$")


class SWEBenchPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(min_length=1, max_length=300)
    model_name_or_path: str = Field(min_length=1, max_length=300)
    model_patch: str = Field(min_length=1, max_length=MAX_PREDICTION_BYTES)

    @field_validator("model_name_or_path")
    @classmethod
    def safe_model_name(cls, value: str) -> str:
        if not _SAFE_MODEL.fullmatch(value) or ".." in value.split("/"):
            raise ValueError("model_name_or_path contains unsafe path characters")
        return value


class SWEBenchPrerequisites(BaseModel):
    model_config = ConfigDict(extra="forbid")

    swebench_installed: bool
    swebench_version: str | None = None
    docker_cli_available: bool
    docker_daemon_available: bool
    docker_version: str | None = None

    @property
    def ready(self) -> bool:
        return self.swebench_installed and self.docker_daemon_available


class SWEBenchEvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["resolved", "unresolved", "evaluation_error", "timeout"]
    dataset_name: str
    split: str
    instance_id: str
    prediction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str
    swebench_version: str | None = None
    docker_version: str | None = None
    grading_duration_seconds: float = Field(ge=0)
    completed: bool = False
    resolved: bool | None = None
    report_path: str | None = None
    image_identifier: str | None = None
    failure_reason: str | None = Field(
        default=None,
        max_length=MAX_SWEBENCH_FAILURE_CHARS,
    )


class _CacheMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    split: str
    instance_id: str
    prediction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str
    swebench_version: str | None = None
    created_at: str


def prediction_sha256(model_patch: str) -> str:
    return hashlib.sha256(model_patch.encode("utf-8")).hexdigest()


def swebench_run_id(experiment_id: str, instance_id: str, model_patch: str) -> str:
    """Bind upstream cache identity to the exact patch bytes."""

    readable = (
        "-".join(
            filter(
                None,
                (
                    re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")[:45]
                    for value in (experiment_id, instance_id)
                ),
            )
        )
        or "repograph"
    )
    return f"{readable}-{prediction_sha256(model_patch)[:20]}"[:180]


def _swebench_environment(parent: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if parent is None else parent
    allowed = {
        "APPDATA",
        "COMSPEC",
        "DOCKER_CONFIG",
        "DOCKER_CONTEXT",
        "DOCKER_HOST",
        "HF_HOME",
        "HF_TOKEN",
        "HOMEDRIVE",
        "HOMEPATH",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "HUGGINGFACE_HUB_TOKEN",
        "LANG",
        "LC_ALL",
        "LOCALAPPDATA",
        "NO_PROXY",
        "PATH",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "WINDIR",
    }
    return {key: value for key, value in source.items() if key.upper() in allowed}


def detect_swebench_prerequisites() -> SWEBenchPrerequisites:
    """Check package plus Docker client/daemon without installing or reconfiguring."""

    try:
        installed = importlib.util.find_spec("swebench") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        installed = False
    version: str | None = None
    if installed:
        try:
            version = importlib.metadata.version("swebench")
        except importlib.metadata.PackageNotFoundError:
            installed = False
    docker_cli = shutil.which("docker") is not None
    daemon = False
    docker_version: str | None = None
    if docker_cli:
        try:
            check = subprocess.run(  # nosec B603, B607
                ["docker", "version", "--format", "{{.Server.Version}}"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
                shell=False,
                env=_swebench_environment(),
            )
            if check.returncode == 0 and check.stdout.strip():
                daemon = True
                docker_version = check.stdout.strip()[:200]
        except (OSError, subprocess.TimeoutExpired):
            pass
    return SWEBenchPrerequisites(
        swebench_installed=installed,
        swebench_version=version,
        docker_cli_available=docker_cli,
        docker_daemon_available=daemon,
        docker_version=docker_version,
    )


def _load_prediction(path: Path, instance_id: str) -> SWEBenchPrediction:
    if not path.is_file() or path.stat().st_size > MAX_PREDICTION_BYTES:
        raise ValueError("Prediction file is missing or exceeds the size budget.")
    selected: SWEBenchPrediction | None = None
    seen: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            prediction = SWEBenchPrediction.model_validate_json(line)
        except ValidationError as error:
            raise ValueError(
                f"Invalid prediction at line {line_number}: {error}"
            ) from error
        if prediction.instance_id in seen:
            raise ValueError(
                f"Duplicate prediction instance_id: {prediction.instance_id}"
            )
        seen.add(prediction.instance_id)
        if prediction.instance_id == instance_id:
            selected = prediction
    if selected is None:
        raise ValueError(f"Prediction not found for instance_id={instance_id}.")
    return selected


def _failure(
    *,
    status: Literal["evaluation_error", "timeout"],
    dataset_name: str,
    split: str,
    instance_id: str,
    digest: str,
    run_id: str,
    prerequisites: SWEBenchPrerequisites,
    started: float,
    reason: str,
) -> SWEBenchEvaluationResult:
    return SWEBenchEvaluationResult(
        status=status,
        dataset_name=dataset_name,
        split=split,
        instance_id=instance_id,
        prediction_sha256=digest,
        run_id=run_id,
        swebench_version=prerequisites.swebench_version,
        docker_version=prerequisites.docker_version,
        grading_duration_seconds=time.monotonic() - started,
        failure_reason=reason[:MAX_SWEBENCH_FAILURE_CHARS],
    )


def _validate_cache_metadata(path: Path, expected: _CacheMetadata) -> None:
    if not path.exists():
        return
    try:
        actual = _CacheMetadata.model_validate_json(path.read_bytes())
    except (OSError, ValidationError, ValueError) as error:
        raise ValueError(f"Malformed SWE-bench cache metadata: {error}") from error
    comparable = ("dataset_name", "split", "instance_id", "prediction_sha256", "run_id")
    if any(getattr(actual, field) != getattr(expected, field) for field in comparable):
        raise ValueError(
            "Refusing cached SWE-bench report: grading metadata does not match "
            "the current prediction."
        )


def _parse_official_report(path: Path, instance_id: str) -> tuple[bool, bool]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Official SWE-bench report is unreadable: {error}") from error
    if not isinstance(report, dict) or not isinstance(
        report.get("schema_version"), int
    ):
        raise TypeError("Official SWE-bench report schema is unsupported.")
    keys = ("completed_ids", "resolved_ids", "unresolved_ids", "error_ids")
    if any(
        not isinstance(report.get(key), list)
        or any(not isinstance(item, str) for item in report[key])
        for key in keys
    ):
        raise TypeError("Official SWE-bench report outcome lists are malformed.")
    completed = instance_id in report["completed_ids"]
    resolved = instance_id in report["resolved_ids"]
    unresolved = instance_id in report["unresolved_ids"]
    errored = instance_id in report["error_ids"]
    if errored or not completed:
        raise ValueError(
            "Official SWE-bench grading did not complete for the instance."
        )
    if resolved == unresolved:
        raise ValueError("Official SWE-bench report has an ambiguous instance outcome.")
    return completed, resolved


def evaluate_swebench_prediction(
    prediction_path: str | Path,
    *,
    dataset_name: str,
    split: str,
    instance_id: str,
    experiment_id: str,
    timeout_seconds: float,
    output_root: str | Path,
) -> SWEBenchEvaluationResult:
    """Grade one prediction through the upstream Docker harness and JSON report."""

    started = time.monotonic()
    empty_digest = prediction_sha256("")
    prerequisites = detect_swebench_prerequisites()
    try:
        prediction = _load_prediction(Path(prediction_path).resolve(), instance_id)
    except (OSError, UnicodeError, ValueError) as error:
        return _failure(
            status="evaluation_error",
            dataset_name=dataset_name,
            split=split,
            instance_id=instance_id,
            digest=empty_digest,
            run_id="invalid-prediction",
            prerequisites=prerequisites,
            started=started,
            reason=str(error),
        )
    digest = prediction_sha256(prediction.model_patch)
    run_id = swebench_run_id(experiment_id, instance_id, prediction.model_patch)
    if not prerequisites.swebench_installed:
        return _failure(
            status="evaluation_error",
            dataset_name=dataset_name,
            split=split,
            instance_id=instance_id,
            digest=digest,
            run_id=run_id,
            prerequisites=prerequisites,
            started=started,
            reason="Optional swebench evaluation dependency is not installed.",
        )
    if not prerequisites.docker_daemon_available:
        return _failure(
            status="evaluation_error",
            dataset_name=dataset_name,
            split=split,
            instance_id=instance_id,
            digest=digest,
            run_id=run_id,
            prerequisites=prerequisites,
            started=started,
            reason="Docker CLI/daemon is unavailable for official SWE-bench grading.",
        )
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    metadata_root = root / "repograph-swebench-metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_root / f"{run_id}.json"
    metadata = _CacheMetadata(
        dataset_name=dataset_name,
        split=split,
        instance_id=instance_id,
        prediction_sha256=digest,
        run_id=run_id,
        swebench_version=prerequisites.swebench_version,
        created_at=datetime.now(UTC).isoformat(),
    )
    try:
        _validate_cache_metadata(metadata_path, metadata)
    except (TypeError, ValueError) as error:
        return _failure(
            status="evaluation_error",
            dataset_name=dataset_name,
            split=split,
            instance_id=instance_id,
            digest=digest,
            run_id=run_id,
            prerequisites=prerequisites,
            started=started,
            reason=str(error),
        )
    metadata_path.write_text(
        json.dumps(metadata.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    argv = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--predictions_path",
        str(Path(prediction_path).resolve()),
        "--max_workers",
        "1",
        "--run_id",
        run_id,
        "--timeout",
        str(max(1, int(timeout_seconds))),
        "--instance_ids",
        instance_id,
        "--report_dir",
        str(root),
    ]
    harness_environment = _swebench_environment()
    try:
        process: FixedProcessOutcome = run_fixed_argv(
            argv,
            cwd=root,
            timeout_seconds=timeout_seconds,
            environment=harness_environment,
        )
    except OSError as error:
        return _failure(
            status="evaluation_error",
            dataset_name=dataset_name,
            split=split,
            instance_id=instance_id,
            digest=digest,
            run_id=run_id,
            prerequisites=prerequisites,
            started=started,
            reason=f"Official SWE-bench harness failed to start: {error}",
        )
    if process.timed_out:
        return _failure(
            status="timeout",
            dataset_name=dataset_name,
            split=split,
            instance_id=instance_id,
            digest=digest,
            run_id=run_id,
            prerequisites=prerequisites,
            started=started,
            reason="Official SWE-bench evaluator exceeded evaluator_timeout.",
        )
    if process.returncode != 0:
        detail = redact_environment_secrets(
            (process.stderr or process.stdout).strip(),
            harness_environment,
            secret_keys={"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"},
        )
        return _failure(
            status="evaluation_error",
            dataset_name=dataset_name,
            split=split,
            instance_id=instance_id,
            digest=digest,
            run_id=run_id,
            prerequisites=prerequisites,
            started=started,
            reason=f"Official SWE-bench harness exited nonzero: {detail}",
        )
    report_path = root / (
        prediction.model_name_or_path.replace("/", "__") + f".{run_id}.json"
    )
    try:
        completed, resolved = _parse_official_report(report_path, instance_id)
    except (TypeError, ValueError) as error:
        return _failure(
            status="evaluation_error",
            dataset_name=dataset_name,
            split=split,
            instance_id=instance_id,
            digest=digest,
            run_id=run_id,
            prerequisites=prerequisites,
            started=started,
            reason=str(error),
        )
    return SWEBenchEvaluationResult(
        status="resolved" if resolved else "unresolved",
        dataset_name=dataset_name,
        split=split,
        instance_id=instance_id,
        prediction_sha256=digest,
        run_id=run_id,
        swebench_version=prerequisites.swebench_version,
        docker_version=prerequisites.docker_version,
        grading_duration_seconds=time.monotonic() - started,
        completed=completed,
        resolved=resolved,
        report_path=str(report_path),
    )
