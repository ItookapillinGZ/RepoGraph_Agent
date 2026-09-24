"""Bounded business-event mapping for Studio timelines."""

from __future__ import annotations

from typing import Any

from studio_backend.config import MAX_STUDIO_EVENT_MESSAGE_CHARS
from studio_backend.storage import StudioStorage


def _scalar_metadata(update: dict[str, object]) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for key in ("attempt", "correction_round", "retry_count"):
        value = update.get(key)
        if isinstance(value, (int, float, bool)):
            metadata[key] = value
    verification = update.get("verification")
    verification_status = getattr(verification, "status", None)
    if isinstance(verification_status, str):
        metadata["verification_status"] = verification_status
    review = update.get("change_set_review")
    review_rating = getattr(review, "overall_rating", None)
    if isinstance(review_rating, str):
        metadata["review_rating"] = review_rating
    return metadata


PLANNING_NODE_EVENTS: dict[str, tuple[str, str, str]] = {
    "run_task_repository_exploration": (
        "repository_exploration_completed",
        "completed",
        "Repository exploration completed",
    ),
    "generate_engineering_plan": (
        "plan_generated",
        "completed",
        "Engineering plan generated",
    ),
    "validate_engineering_plan": (
        "plan_validated",
        "completed",
        "Engineering plan validated",
    ),
    "prepare_plan_retry": (
        "planning_retry",
        "info",
        "Plan validation requested a bounded retry",
    ),
    "controlled_plan_failure": (
        "planning_failed",
        "failed",
        "Engineering planning failed safely",
    ),
}


EXECUTION_NODE_EVENTS: dict[str, tuple[str, str, str, str]] = {
    "build_execution_context": (
        "execution",
        "execution_context_built",
        "completed",
        "Execution context prepared",
    ),
    "validate_multi_file_candidate": (
        "execution",
        "candidate_validated",
        "completed",
        "Candidate validated",
    ),
    "prepare_candidate_retry": (
        "execution",
        "candidate_retry_prepared",
        "info",
        "Candidate generation retry prepared",
    ),
    "prepare_fresh_workspace": (
        "execution",
        "temporary_workspace_prepared",
        "completed",
        "Isolated verification workspace prepared",
    ),
    "create_temporary_workspace": (
        "execution",
        "temporary_workspace_created",
        "completed",
        "Isolated repository copy created",
    ),
    "materialize_candidate": (
        "execution",
        "candidate_materialized",
        "completed",
        "Candidate materialized in isolation",
    ),
    "build_candidate_diff": (
        "execution",
        "candidate_diff_built",
        "completed",
        "Candidate diff prepared",
    ),
    "evaluate_candidate_outcome": (
        "review",
        "candidate_outcome_evaluated",
        "completed",
        "Candidate outcome evaluated",
    ),
    "build_correction_feedback": (
        "correction",
        "correction_started",
        "started",
        "Correction feedback prepared",
    ),
    "controlled_execution_failure": (
        "failed",
        "execution_failed",
        "failed",
        "Candidate execution failed safely",
    ),
}


class StudioEventEmitter:
    def __init__(self, storage: StudioStorage, run_id: str) -> None:
        self.storage = storage
        self.run_id = run_id
        self._candidate_attempt = 1
        self._correction_round = 0

    def emit(
        self,
        *,
        phase: str,
        event_type: str,
        status: str,
        title: str,
        message: str = "",
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.storage.append_event(
            self.run_id,
            phase=phase,
            event_type=event_type,
            status=status,
            title=title,
            message=" ".join(message.split())[:MAX_STUDIO_EVENT_MESSAGE_CHARS],
            metadata=metadata or {},
        )

    def planning_node(self, node: str, update: dict[str, object]) -> None:
        event = PLANNING_NODE_EVENTS.get(node)
        if event is None:
            return
        event_type, status, title = event
        self.emit(
            phase="planning",
            event_type=event_type,
            status=status,
            title=title,
            metadata={
                "graph": "engineering_plan_graph",
                "node": node,
                **_scalar_metadata(update),
            },
        )

    def execution_node(self, node: str, update: dict[str, object]) -> None:
        correction_round = update.get("correction_round")
        if isinstance(correction_round, int) and correction_round >= 0:
            self._correction_round = correction_round
            self._candidate_attempt = correction_round + 1
        metadata = {
            "graph": "plan_execution_graph",
            "node": node,
            "candidate_attempt": self._candidate_attempt,
            "correction_round": self._correction_round,
            **_scalar_metadata(update),
        }
        if node in {
            "generate_multi_file_candidate",
            "generate_corrected_candidate",
        }:
            phase = "correction" if self._correction_round else "execution"
            self.storage.update_run(self.run_id, status="running", phase=phase)
            self.emit(
                phase=phase,
                event_type="candidate_generated",
                status="completed",
                title=(
                    "Corrected candidate generated"
                    if self._correction_round
                    else "Multi-file candidate generated"
                ),
                metadata=metadata,
            )
            return
        if node == "verify_candidate":
            self.storage.update_run(
                self.run_id,
                status="running",
                phase="verification",
            )
            self.emit(
                phase="verification",
                event_type="verification_started",
                status="started",
                title="Candidate verification started",
                metadata=metadata,
            )
            self.emit(
                phase="verification",
                event_type="verification_completed",
                status="completed",
                title="Candidate verification completed",
                metadata=metadata,
            )
            return
        if node == "review_candidate_change_set":
            metadata["nested_graph"] = "change_set_graph/review_graph"
            self.storage.update_run(self.run_id, status="running", phase="review")
            self.emit(
                phase="review",
                event_type="review_started",
                status="started",
                title="Change-set review started",
                metadata=metadata,
            )
            self.emit(
                phase="review",
                event_type="review_completed",
                status="completed",
                title="Change-set review completed",
                metadata=metadata,
            )
            return
        event = EXECUTION_NODE_EVENTS.get(node)
        if event is None:
            return
        phase, event_type, status, title = event
        self.storage.update_run(
            self.run_id,
            status="running",
            phase=phase,
        )
        self.emit(
            phase=phase,
            event_type=event_type,
            status=status,
            title=title,
            metadata={
                **metadata,
            },
        )


def dump_public(value: Any) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [dump_public(item) for item in value]
    if isinstance(value, tuple):
        return [dump_public(item) for item in value]
    if isinstance(value, dict):
        return {str(key): dump_public(item) for key, item in value.items()}
    return value
