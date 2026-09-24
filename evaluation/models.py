"""Strict, bounded schemas shared by the evaluation harness."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from plan_execution import MultiFileCandidate
from sandbox.models import SandboxProvenance

MAX_IDENTIFIER_CHARS = 200
MAX_FAILURE_REASON_CHARS = 4_000
MAX_WARNING_CHARS = 2_000
MAX_WARNINGS = 100
MAX_TASK_TIMEOUT_SECONDS = 86_400.0
MAX_EVALUATOR_TIMEOUT_SECONDS = 86_400.0
MAX_CANDIDATE_DIFF_CHARS = 2_000_000
LLMExecutionKind = Literal["live", "deterministic_test"]
MAX_EXECUTION_ATTEMPT = 1_000_000
ExecutionAttemptStatus = Literal[
    "reserved",
    "running",
    "completed",
    "interrupted",
    "failed",
    "timeout",
]


class EvaluationTask(BaseModel):
    """One deterministic repository task supplied by a trusted dataset."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_CHARS)
    dataset: str = Field(min_length=1, max_length=MAX_IDENTIFIER_CHARS)
    repository: str = Field(min_length=1)
    base_commit: str = Field(min_length=1, max_length=200)
    task: str = Field(min_length=1, max_length=10_000)
    test_command: list[str] | None = None
    expected_files: list[str] = Field(default_factory=list, max_length=100)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("id", "dataset", "repository", "base_commit", "task")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must be non-empty")
        return value

    @field_validator("test_command")
    @classmethod
    def validate_test_command(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value or len(value) > 100:
            raise ValueError("test_command must contain 1 to 100 argv values")
        if any(not isinstance(part, str) or not part for part in value):
            raise ValueError("test_command argv values must be non-empty strings")
        if any("\x00" in part for part in value):
            raise ValueError("test_command argv values cannot contain NUL bytes")
        return value


class EvaluationConfig(BaseModel):
    """RepoGraph switches and evaluator budgets recorded with an experiment."""

    model_config = ConfigDict(extra="forbid")
    model_provider: str | None = Field(default=None, max_length=100)
    model_base_url: str | None = Field(default=None, max_length=2_000)
    model_reasoning_effort: str | None = Field(default=None, max_length=50)
    model_max_completion_tokens: int | None = Field(default=None, ge=1)

    name: str = Field(min_length=1, max_length=MAX_IDENTIFIER_CHARS)
    self_correct: bool = True
    max_correction_rounds: int = Field(default=2, ge=0, le=20)
    agentic_explore: bool = True
    agentic_test: bool = True
    run_tests: bool = True
    sandbox_backend: Literal["host", "docker"] = "host"
    model_name: str | None = Field(default=None, max_length=300)
    model_temperature: float | None = None
    model_seed: int | None = None
    llm_execution_kind: LLMExecutionKind = "deterministic_test"
    max_tasks: int | None = Field(default=None, ge=1)
    task_timeout_seconds: float = Field(
        default=1_800.0,
        gt=0,
        le=MAX_TASK_TIMEOUT_SECONDS,
    )
    # Kept so persisted H2 configs and direct callers continue to load. New
    # code and CLI surfaces use task_timeout_seconds.
    overall_timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        le=MAX_TASK_TIMEOUT_SECONDS,
    )
    evaluator_timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        le=MAX_EVALUATOR_TIMEOUT_SECONDS,
    )
    max_evaluator_output_chars: int = Field(default=20_000, ge=1, le=1_000_000)

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name must be non-empty")
        return value

    @model_validator(mode="after")
    def validate_dependencies(self) -> EvaluationConfig:
        if self.agentic_test and not self.agentic_explore:
            raise ValueError("agentic_test requires agentic_explore=True")
        if self.agentic_test and not self.run_tests:
            raise ValueError("agentic_test requires run_tests=True")
        return self

    @property
    def effective_correction_rounds(self) -> int:
        """Return the correction budget actually passed to RepoGraph."""

        return self.max_correction_rounds if self.self_correct else 0

    @property
    def effective_task_timeout_seconds(self) -> float:
        """Return the task-process deadline, honoring legacy H2 configs."""

        return self.overall_timeout_seconds or self.task_timeout_seconds

    def digest(self) -> str:
        """Return a deterministic provenance digest, not a security signature."""

        canonical = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


class LLMUsageSummary(BaseModel):
    """Observed callback usage; absent token metadata is never estimated."""

    model_config = ConfigDict(extra="forbid")

    calls: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    incomplete: bool = False
    models: list[str] = Field(default_factory=list, max_length=100)
    configured_models: list[str] = Field(default_factory=list, max_length=100)
    resolved_models: list[str] = Field(default_factory=list, max_length=100)
    warnings: list[str] = Field(default_factory=list, max_length=MAX_WARNINGS)

    @field_validator("warnings")
    @classmethod
    def bound_usage_warnings(cls, value: list[str]) -> list[str]:
        return [str(item)[:MAX_WARNING_CHARS] for item in value]


class EvaluationProvenance(BaseModel):
    """Reproduction metadata captured without inventing unavailable values."""

    model_config = ConfigDict(extra="forbid")

    repo_base_commit: str
    repograph_source_commit: str | None = None
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_name: str | None = None
    configured_model_name: str | None = None
    resolved_model_names: list[str] = Field(default_factory=list, max_length=100)
    model_temperature: float | None = None
    model_seed: int | None = None
    model_provider: str | None = None
    model_base_url: str | None = None
    model_reasoning_effort: str | None = None
    model_max_completion_tokens: int | None = Field(default=None, ge=1)
    llm_execution_kind: LLMExecutionKind = "deterministic_test"
    task_timeout_seconds: float | None = Field(default=None, gt=0)
    max_correction_rounds: int | None = Field(default=None, ge=0)
    prediction_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    evaluator_kind: Literal["local", "swebench"] = "local"
    evaluator_version: str | None = None
    evaluator_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    python_executable: str | None = None
    timestamp: str



class EvaluatorSpecification(BaseModel):
    """Exact evaluator identity frozen before candidate execution."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["local", "swebench"]
    command: list[str] = Field(min_length=1, max_length=100)
    python_executable: str | None = Field(default=None, max_length=4_000)
    version: str | None = Field(default=None, max_length=200)
    timeout_seconds: float = Field(gt=0, le=MAX_EVALUATOR_TIMEOUT_SECONDS)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: list[str]) -> list[str]:
        if any(not item or "\x00" in item for item in value):
            raise ValueError("evaluator command values must be non-empty and NUL-free")
        return value


class EvaluatorEnvironmentFingerprint(BaseModel):
    """Bounded non-secret runtime identity used for evaluator provenance."""

    model_config = ConfigDict(extra="forbid")

    python_version: str
    pytest_version: str | None = None
    platform: str
    evaluator_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class EvaluationCandidateSnapshot(BaseModel):
    """One exact candidate attempt and replayable diff from a real run."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_CHARS)
    experiment_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_CHARS)
    execution_attempt: int = Field(default=1, ge=1, le=MAX_EXECUTION_ATTEMPT)
    attempt: int = Field(ge=1, le=100)
    correction_round: int = Field(ge=0, le=99)
    base_commit: str = Field(min_length=1, max_length=200)
    model_name: str | None = Field(default=None, max_length=300)
    candidate: MultiFileCandidate
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    diff_text: str = Field(max_length=MAX_CANDIDATE_DIFF_CHARS)
    diff_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: str

    @model_validator(mode="after")
    def validate_attempt_round(self) -> EvaluationCandidateSnapshot:
        if self.attempt != self.correction_round + 1:
            raise ValueError("attempt must equal correction_round + 1")
        return self

class EvaluationResult(BaseModel):
    """Complete but bounded result for one externally graded task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_CHARS)
    dataset: str = Field(min_length=1, max_length=MAX_IDENTIFIER_CHARS)
    experiment: str = Field(min_length=1, max_length=MAX_IDENTIFIER_CHARS)
    repository: str
    base_commit: str
    created_at: str
    git_commit: str | None = None
    model_name: str | None = None
    configured_model_name: str | None = None
    llm_execution_kind: LLMExecutionKind = "deterministic_test"

    status: Literal[
        "resolved",
        "unresolved",
        "agent_failed",
        "evaluation_error",
        "timeout",
    ]
    first_attempt_resolved: bool = False
    final_resolved: bool = False
    correction_rounds_used: int = Field(default=0, ge=0)
    planning_succeeded: bool = False
    candidate_generated: bool = False
    verification_succeeded: bool = False
    review_good: bool = False
    changed_files: list[str] = Field(default_factory=list, max_length=100)
    duration_seconds: float = Field(ge=0)
    llm_calls: int | None = Field(default=None, ge=0)
    tool_calls: int | None = Field(default=None, ge=0)
    test_calls: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    telemetry_incomplete: bool = False
    telemetry_warnings: list[str] = Field(default_factory=list, max_length=MAX_WARNINGS)
    resolved_model_names: list[str] = Field(default_factory=list, max_length=100)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    evaluator_kind: Literal["local", "swebench"] = "local"
    evaluator_version: str | None = None
    evaluator_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    evaluator_specification: EvaluatorSpecification | None = None
    evaluator_environment: EvaluatorEnvironmentFingerprint | None = None
    evaluator_failure_kind: str | None = Field(default=None, max_length=100)
    candidate_snapshot_paths: list[str] = Field(default_factory=list, max_length=100)
    first_candidate_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    final_candidate_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    first_attempt_evaluated: bool = False
    final_attempt_evaluated: bool = False
    valid_prediction: bool = False
    infrastructure_failure: bool = False
    required_infrastructure_retry: bool = False
    execution_attempt: int = Field(default=1, ge=1, le=MAX_EXECUTION_ATTEMPT)
    process_isolated: bool = False
    hard_timeout: bool = False
    config_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    prediction_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    provenance: EvaluationProvenance | None = None
    failure_category: str | None = Field(default=None, max_length=100)
    failure_reason: str | None = Field(
        default=None,
        max_length=MAX_FAILURE_REASON_CHARS,
    )
    warnings: list[str] = Field(default_factory=list, max_length=MAX_WARNINGS)
    model_patch: str | None = None

    @field_validator("warnings")
    @classmethod
    def bound_warnings(cls, value: list[str]) -> list[str]:
        return [str(item)[:MAX_WARNING_CHARS] for item in value]

    @field_validator("telemetry_warnings")
    @classmethod
    def bound_telemetry_warnings(cls, value: list[str]) -> list[str]:
        return [str(item)[:MAX_WARNING_CHARS] for item in value]


class EvaluationExperiment(BaseModel):
    """Immutable metadata needed to reproduce and compare one run."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_CHARS)
    name: str = Field(min_length=1, max_length=MAX_IDENTIFIER_CHARS)
    dataset: str = Field(min_length=1, max_length=MAX_IDENTIFIER_CHARS)
    config: EvaluationConfig
    created_at: str
    git_commit: str | None = None
    model_name: str | None = None
    configured_model_name: str | None = None
    llm_execution_kind: LLMExecutionKind = "deterministic_test"


class ExecutionAttemptRecord(BaseModel):
    """Durable identity and lifecycle for one worker execution."""

    model_config = ConfigDict(extra="forbid")

    campaign_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_CHARS)
    config_name: str = Field(min_length=1, max_length=MAX_IDENTIFIER_CHARS)
    experiment_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_CHARS)
    task_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_CHARS)
    execution_attempt: int = Field(ge=1, le=MAX_EXECUTION_ATTEMPT)
    status: ExecutionAttemptStatus
    reserved_at: str
    started_at: str | None = None
    finished_at: str | None = None
    live_run_ordinal: int | None = Field(default=None, ge=1)
    failure_reason: str | None = Field(
        default=None,
        max_length=MAX_FAILURE_REASON_CHARS,
    )


class EvaluatorOutcome(BaseModel):
    """Bounded subprocess outcome used internally by runner adapters."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["passed", "failed", "error", "timeout"]
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    failure_category: str | None = None
    failure_kind: str | None = Field(default=None, max_length=100)
    failure_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    duration_seconds: float = Field(ge=0)
    sandbox_provenance: SandboxProvenance | None = None


class EvaluationWorkerRequest(BaseModel):
    """Bounded JSON contract from the controller to one task worker."""

    model_config = ConfigDict(extra="forbid")

    task: EvaluationTask
    config: EvaluationConfig
    workspace_root: str = Field(min_length=1, max_length=4_000)
    artifacts_root: str | None = Field(default=None, max_length=4_000)
    experiment_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_CHARS)
    result_path: str = Field(min_length=1, max_length=4_000)
    execution_attempt: int = Field(default=1, ge=1, le=MAX_EXECUTION_ATTEMPT)
    git_commit: str | None = Field(default=None, max_length=200)


class EvaluationWorkerResult(BaseModel):
    """Strict JSON contract written once by a task worker."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "failed", "timeout", "interrupted"]
    result: EvaluationResult | None = None
    failure_reason: str | None = Field(
        default=None,
        max_length=MAX_FAILURE_REASON_CHARS,
    )

    @model_validator(mode="after")
    def validate_result_status(self) -> EvaluationWorkerResult:
        if self.status == "completed" and self.result is None:
            raise ValueError("completed worker result requires result")
        if self.status != "completed" and self.result is not None:
            raise ValueError("non-completed worker result cannot contain result")
        return self
