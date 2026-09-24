"""Conservative startup reconciliation for interrupted Studio work."""

from __future__ import annotations

from git_delivery import (
    LocalGitDeliveryResult,
    LocalGitDeliveryValidationError,
    validate_local_git_delivery,
)
from plan_application import PlanApplicationBundle
from studio_backend.events import StudioEventEmitter, dump_public
from studio_backend.repositories import RepositoryBoundaryError, WorkspaceRepositories
from studio_backend.storage import StudioStorage

ACTIVE_STATUSES = ("queued", "running", "applying", "git_creating", "publishing")


class StudioRecoveryService:
    """Reconcile persisted state without LLM, test, write, or network activity."""

    def __init__(
        self,
        storage: StudioStorage,
        repositories: WorkspaceRepositories,
    ) -> None:
        self.storage = storage
        self.repositories = repositories

    def reconcile_startup(self) -> int:
        runs = self.storage.list_runs_by_status(ACTIVE_STATUSES)
        for run in runs:
            if run.status in {"queued", "running"}:
                self._interrupt(
                    run.id,
                    phase="failed",
                    event_type="execution_interrupted",
                    reason="Studio execution was interrupted by a server restart.",
                    title="Preview execution interrupted",
                )
            elif run.status == "applying":
                self._reconcile_application(run.id)
            elif run.status == "git_creating":
                self._reconcile_git(run.id, run.repository_id)
            elif run.status == "publishing":
                self._reconcile_remote(run.id)
        return len(runs)

    def _event_exists(self, run_id: str, event_type: str) -> bool:
        return any(
            event.event_type == event_type
            for event in self.storage.list_events(run_id)
        )

    def _emit_once(
        self,
        run_id: str,
        *,
        phase: str,
        event_type: str,
        status: str,
        title: str,
        message: str = "",
        metadata: dict[str, object] | None = None,
    ) -> None:
        if self._event_exists(run_id, event_type):
            return
        StudioEventEmitter(self.storage, run_id).emit(
            phase=phase,
            event_type=event_type,
            status=status,
            title=title,
            message=message,
            metadata=metadata,
        )

    def _interrupt(
        self,
        run_id: str,
        *,
        phase: str,
        event_type: str,
        reason: str,
        title: str,
    ) -> None:
        self.storage.update_run(
            run_id,
            status="interrupted",
            phase=phase,
            failure_reason=reason,
        )
        self._emit_once(
            run_id,
            phase=phase,
            event_type=event_type,
            status="failed",
            title=title,
            message=reason,
        )

    def _reconcile_application(self, run_id: str) -> None:
        artifact = self.storage.get_artifact(run_id, "application_result")
        payload = artifact.payload if artifact is not None else None
        if isinstance(payload, dict) and payload.get("status") == "applied":
            self.storage.update_run(
                run_id,
                status="applied",
                phase="approval_git",
            )
            self._emit_once(
                run_id,
                phase="application",
                event_type="application_completed",
                status="completed",
                title="Working-tree changes applied",
                metadata={"startup_recovered": True},
            )
            self._request_git_approval(run_id)
            return
        self._interrupt(
            run_id,
            phase="application",
            event_type="application_interrupted",
            reason=(
                "Application was interrupted. Repository state must be "
                "revalidated before retry."
            ),
            title="Filesystem application interrupted",
        )

    def _requested_branch_name(self, run_id: str) -> str | None:
        for event in reversed(self.storage.list_events(run_id)):
            if event.event_type != "git_delivery_started":
                continue
            branch_name = event.metadata.get("branch_name")
            return branch_name if isinstance(branch_name, str) and branch_name else None
        return None

    def _reconcile_git(self, run_id: str, repository_id: str) -> None:
        artifact = self.storage.get_artifact(run_id, "local_git_delivery_result")
        payload = artifact.payload if artifact is not None else None
        if isinstance(payload, dict) and payload.get("status") == "created":
            self.storage.update_run(
                run_id,
                status="git_created",
                phase="approval_remote",
            )
            self._emit_once(
                run_id,
                phase="git_delivery",
                event_type="git_delivery_completed",
                status="completed",
                title="Local branch and commit recovered",
                metadata={"startup_recovered": True},
            )
            self._request_remote_approval(run_id)
            return

        bundle_artifact = self.storage.get_artifact(run_id, "application_bundle")
        try:
            if bundle_artifact is None:
                raise ValueError("Application bundle is unavailable.")
            bundle = PlanApplicationBundle.model_validate(bundle_artifact.payload)
            repository_root = self.repositories.resolve(repository_id)
            delivery = validate_local_git_delivery(
                str(repository_root),
                bundle,
                branch_name=self._requested_branch_name(run_id),
            )
        except LocalGitDeliveryValidationError as error:
            if error.reason == "Local delivery branch does not exist.":
                self._interrupt(
                    run_id,
                    phase="git_delivery",
                    event_type="git_delivery_interrupted",
                    reason=(
                        "Local Git delivery was interrupted before a matching "
                        "branch was found. Explicit approval may retry it."
                    ),
                    title="Local Git delivery interrupted",
                )
            else:
                self.storage.update_run(
                    run_id,
                    status="conflict",
                    phase="failed",
                    failure_reason=(
                        "An existing local delivery could not be reconciled with "
                        "the approved bundle. Manual reconciliation is required."
                    ),
                )
                self._emit_once(
                    run_id,
                    phase="git_delivery",
                    event_type="git_delivery_interrupted",
                    status="failed",
                    title="Local Git delivery requires manual reconciliation",
                    metadata={"recovery_status": error.status},
                )
            return
        except (RepositoryBoundaryError, TypeError, ValueError):
            self._interrupt(
                run_id,
                phase="git_delivery",
                event_type="git_delivery_interrupted",
                reason=(
                    "Local Git delivery was interrupted and could not be safely "
                    "revalidated. Manual reconciliation is required."
                ),
                title="Local Git delivery interrupted",
            )
            return

        result = LocalGitDeliveryResult(
            status="created",
            branch_name=delivery.branch_name,
            commit_sha=delivery.commit_sha,
            base_sha=delivery.base_sha,
            approval_digest=delivery.approval_digest,
            committed_files=delivery.committed_files,
        )
        self.storage.put_artifact(
            run_id,
            "local_git_delivery_result",
            dump_public(result),
        )
        self.storage.update_run(
            run_id,
            status="git_created",
            phase="approval_remote",
        )
        self._emit_once(
            run_id,
            phase="git_delivery",
            event_type="git_delivery_completed",
            status="completed",
            title="Local branch and commit recovered",
            metadata={"startup_recovered": True},
        )
        self._request_remote_approval(run_id)

    def _reconcile_remote(self, run_id: str) -> None:
        artifact = self.storage.get_artifact(run_id, "remote_delivery_result")
        payload = artifact.payload if artifact is not None else None
        if isinstance(payload, dict) and payload.get("status") == "published":
            self.storage.update_run(run_id, status="published", phase="completed")
            self._emit_once(
                run_id,
                phase="remote_delivery",
                event_type="remote_delivery_completed",
                status="completed",
                title="Remote delivery recovered",
                metadata={"startup_recovered": True},
            )
            self._emit_once(
                run_id,
                phase="completed",
                event_type="run_completed",
                status="completed",
                title="RepoGraph Studio run completed",
            )
            return
        self._interrupt(
            run_id,
            phase="remote_delivery",
            event_type="remote_delivery_interrupted",
            reason=(
                "Remote delivery was interrupted. No network request will be "
                "made until explicit approval retries it."
            ),
            title="Remote delivery interrupted",
        )

    def _request_git_approval(self, run_id: str) -> None:
        self._emit_once(
            run_id,
            phase="approval_git",
            event_type="git_approval_requested",
            status="waiting",
            title="Local Git approval requested",
        )
        self._emit_once(
            run_id,
            phase="approval_git",
            event_type="waiting_for_git_approval",
            status="waiting",
            title="Waiting for local Git approval",
        )

    def _request_remote_approval(self, run_id: str) -> None:
        self._emit_once(
            run_id,
            phase="approval_remote",
            event_type="remote_approval_requested",
            status="waiting",
            title="Remote delivery approval requested",
        )
        self._emit_once(
            run_id,
            phase="approval_remote",
            event_type="waiting_for_remote_approval",
            status="waiting",
            title="Waiting for remote delivery approval",
        )
