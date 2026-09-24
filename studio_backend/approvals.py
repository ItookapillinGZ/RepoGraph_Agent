"""Three independent human-approval boundaries for Studio delivery."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext

from git_delivery import LocalGitDeliveryResult, create_local_git_delivery
from github_delivery import GitHubRemoteDeliveryResult, publish_local_git_delivery
from observability.recorder import TraceRecorder, continue_trace_span
from plan_application import (
    PlanApplicationBundle,
    PlanApplicationResult,
    apply_plan_application_bundle,
)
from studio_backend.events import StudioEventEmitter, dump_public
from studio_backend.models import (
    ApplyApprovalRequest,
    ApprovalResponse,
    GitApprovalRequest,
    RejectRunRequest,
    RemoteApprovalRequest,
    StudioRun,
)
from studio_backend.repositories import WorkspaceRepositories
from studio_backend.storage import StudioStorage


class ApprovalNotFound(LookupError):
    pass


class ApprovalConflict(RuntimeError):
    pass


class ApprovalRejected(ValueError):
    pass


class StudioApprovalService:
    def __init__(
        self,
        storage: StudioStorage,
        repositories: WorkspaceRepositories,
        *,
        application_boundary: Callable[..., PlanApplicationResult] | None = None,
        git_boundary: Callable[..., LocalGitDeliveryResult] | None = None,
        remote_boundary: Callable[..., GitHubRemoteDeliveryResult] | None = None,
    ) -> None:
        self.storage = storage
        self.repositories = repositories
        self.application_boundary = application_boundary
        self.git_boundary = git_boundary
        self.remote_boundary = remote_boundary
        self.trace_recorder = TraceRecorder(storage.trace_sink)

    def _trace_stage(self, run_id: str, name: str, kind: str):
        trace = self.storage.trace_sink.get_run(run_id)
        if trace is None:
            return nullcontext()
        return continue_trace_span(
            self.trace_recorder,
            run_id=run_id,
            root_span_id=trace.root_span_id,
            name=name,
            kind=kind,
        )

    def _run(self, run_id: str) -> StudioRun:
        run = self.storage.get_run(run_id)
        if run is None:
            raise ApprovalNotFound("Run not found.")
        return run

    def _bundle(self, run_id: str) -> PlanApplicationBundle:
        artifact = self.storage.get_artifact(run_id, "application_bundle")
        if artifact is None:
            raise ApprovalConflict("Application bundle is unavailable.")
        return PlanApplicationBundle.model_validate(artifact.payload)

    @staticmethod
    def _validate_approval(
        run: StudioRun,
        bundle: PlanApplicationBundle,
        *,
        approved: bool,
        digest: str,
    ) -> None:
        if not approved:
            raise ApprovalRejected("Explicit approval is required.")
        if run.approval_digest != bundle.approval_digest:
            raise ApprovalConflict("Stored approval state is inconsistent.")
        if digest != bundle.approval_digest:
            raise ApprovalConflict("Approval digest does not match.")

    def approve_apply(
        self,
        run_id: str,
        request: ApplyApprovalRequest,
    ) -> ApprovalResponse:
        run = self._run(run_id)
        bundle = self._bundle(run_id)
        self._validate_approval(
            run,
            bundle,
            approved=request.approved,
            digest=request.approval_digest,
        )
        expected_statuses = ["verified"]
        if run.status == "interrupted" and run.phase == "application":
            expected_statuses.append("interrupted")
        claimed = self.storage.compare_and_set_run(
            run_id,
            expected_statuses=expected_statuses,
            status="applying",
            phase="application",
        )
        if claimed is None:
            raise ApprovalConflict("Run is not waiting for apply approval.")
        emitter = StudioEventEmitter(self.storage, run_id)
        emitter.emit(
            phase="application",
            event_type="application_started",
            status="started",
            title="Approved filesystem application started",
        )
        try:
            root = self.repositories.resolve(run.repository_id)
            application_boundary = (
                self.application_boundary or apply_plan_application_bundle
            )
            with self._trace_stage(run_id, "g4_application", "application"):
                result = application_boundary(
                    str(root),
                    bundle,
                    approved=True,
                )
        except Exception:  # noqa: BLE001 - core boundary is sanitized here
            result = PlanApplicationResult(
                status="error",
                approval_digest=bundle.approval_digest,
                failure_reason="Application failed safely.",
            )
        self.storage.put_artifact(run_id, "application_result", dump_public(result))
        if result.status == "applied":
            updated = self.storage.update_run(
                run_id,
                status="applied",
                phase="approval_git",
            )
            emitter.emit(
                phase="application",
                event_type="application_completed",
                status="completed",
                title="Working-tree changes applied",
                metadata={"applied_file_count": len(result.applied_files)},
            )
            emitter.emit(
                phase="approval_git",
                event_type="git_approval_requested",
                status="waiting",
                title="Local Git approval requested",
            )
            emitter.emit(
                phase="approval_git",
                event_type="waiting_for_git_approval",
                status="waiting",
                title="Waiting for local Git approval",
            )
        else:
            studio_status = "stale" if result.status == "stale" else "error"
            updated = self.storage.update_run(
                run_id,
                status=studio_status,
                phase="failed",
                failure_reason="Approved application did not complete.",
            )
            emitter.emit(
                phase="application",
                event_type="apply_failed",
                status="failed",
                title="Approved application did not complete",
                metadata={"application_status": result.status},
            )
        if updated is None:
            raise ApprovalNotFound("Run not found.")
        return ApprovalResponse(run=updated, artifact_kind="application_result")

    def approve_git(
        self,
        run_id: str,
        request: GitApprovalRequest,
    ) -> ApprovalResponse:
        run = self._run(run_id)
        bundle = self._bundle(run_id)
        self._validate_approval(
            run,
            bundle,
            approved=request.approved,
            digest=request.approval_digest,
        )
        applied = self.storage.get_artifact(run_id, "application_result")
        if not isinstance(applied.payload if applied else None, dict) or (
            applied.payload.get("status") != "applied"  # type: ignore[union-attr]
        ):
            raise ApprovalConflict("Applied application result is required.")
        expected_statuses = ["applied"]
        if run.status == "interrupted" and run.phase == "git_delivery":
            expected_statuses.append("interrupted")
        claimed = self.storage.compare_and_set_run(
            run_id,
            expected_statuses=expected_statuses,
            status="git_creating",
            phase="git_delivery",
        )
        if claimed is None:
            raise ApprovalConflict("Run is not waiting for local Git approval.")
        emitter = StudioEventEmitter(self.storage, run_id)
        emitter.emit(
            phase="git_delivery",
            event_type="git_delivery_started",
            status="started",
            title="Approved local Git delivery started",
            metadata={"branch_name": request.branch_name or ""},
        )
        try:
            root = self.repositories.resolve(run.repository_id)
            git_boundary = self.git_boundary or create_local_git_delivery
            with self._trace_stage(run_id, "g5a_git_delivery", "git_delivery"):
                result = git_boundary(
                    str(root),
                    bundle,
                    approved=True,
                    branch_name=request.branch_name,
                    commit_message=request.commit_message,
                )
        except Exception:  # noqa: BLE001
            result = LocalGitDeliveryResult(
                status="error",
                approval_digest=bundle.approval_digest,
                failure_reason="Local Git delivery failed safely.",
            )
        self.storage.put_artifact(
            run_id,
            "local_git_delivery_result",
            dump_public(result),
        )
        if result.status == "created":
            updated = self.storage.update_run(
                run_id,
                status="git_created",
                phase="approval_remote",
            )
            emitter.emit(
                phase="git_delivery",
                event_type="git_delivery_completed",
                status="completed",
                title="Local branch and commit created",
                metadata={"branch_name": result.branch_name or ""},
            )
            emitter.emit(
                phase="approval_remote",
                event_type="remote_approval_requested",
                status="waiting",
                title="Remote delivery approval requested",
            )
            emitter.emit(
                phase="approval_remote",
                event_type="waiting_for_remote_approval",
                status="waiting",
                title="Waiting for remote delivery approval",
            )
        else:
            studio_status = result.status if result.status in {"stale", "conflict"} else "error"
            updated = self.storage.update_run(
                run_id,
                status=studio_status,
                phase="failed",
                failure_reason="Approved local Git delivery did not complete.",
            )
            emitter.emit(
                phase="git_delivery",
                event_type="git_delivery_failed",
                status="failed",
                title="Local Git delivery did not complete",
                metadata={"git_status": result.status},
            )
        if updated is None:
            raise ApprovalNotFound("Run not found.")
        return ApprovalResponse(
            run=updated,
            artifact_kind="local_git_delivery_result",
        )

    def approve_remote(
        self,
        run_id: str,
        request: RemoteApprovalRequest,
    ) -> ApprovalResponse:
        run = self._run(run_id)
        bundle = self._bundle(run_id)
        self._validate_approval(
            run,
            bundle,
            approved=request.approved,
            digest=request.approval_digest,
        )
        local = self.storage.get_artifact(run_id, "local_git_delivery_result")
        if not isinstance(local.payload if local else None, dict) or (
            local.payload.get("status") != "created"  # type: ignore[union-attr]
        ):
            raise ApprovalConflict("Created local Git delivery is required.")
        local_payload = local.payload
        local_branch = request.branch_name or local_payload.get("branch_name")
        expected_statuses = ["git_created"]
        if run.status == "interrupted" and run.phase == "remote_delivery":
            expected_statuses.append("interrupted")
        if run.status == "partial":
            expected_statuses.append("partial")
        claimed = self.storage.compare_and_set_run(
            run_id,
            expected_statuses=expected_statuses,
            status="publishing",
            phase="remote_delivery",
        )
        if claimed is None:
            raise ApprovalConflict("Run is not waiting for remote approval.")
        emitter = StudioEventEmitter(self.storage, run_id)
        emitter.emit(
            phase="remote_delivery",
            event_type="remote_delivery_started",
            status="started",
            title="Approved remote delivery started",
        )
        try:
            root = self.repositories.resolve(run.repository_id)
            remote_boundary = self.remote_boundary or publish_local_git_delivery
            with self._trace_stage(run_id, "g5b_github_delivery", "github_delivery"):
                result = remote_boundary(
                    str(root),
                    bundle,
                    approved=True,
                    local_branch=str(local_branch) if local_branch else None,
                    remote_name=request.remote_name,
                    base_branch=request.base_branch,
                    pr_title=request.pr_title,
                    pr_body=request.pr_body,
                )
        except Exception:  # noqa: BLE001
            result = GitHubRemoteDeliveryResult(
                status="error",
                remote_name=request.remote_name,
                base_branch=request.base_branch,
                approval_digest=bundle.approval_digest,
                failure_reason="GitHub remote delivery failed safely.",
            )
        self.storage.put_artifact(
            run_id,
            "remote_delivery_result",
            dump_public(result),
        )
        if result.pushed:
            emitter.emit(
                phase="remote_delivery",
                event_type="remote_branch_pushed",
                status="completed",
                title="Remote branch verified",
                metadata={"push_created": result.push_created},
            )
        if result.status == "published":
            updated = self.storage.update_run(
                run_id,
                status="published",
                phase="completed",
            )
            emitter.emit(
                phase="remote_delivery",
                event_type="remote_delivery_completed",
                status="completed",
                title="Remote delivery completed",
                metadata={"pr_created": result.pr_created},
            )
            emitter.emit(
                phase="remote_delivery",
                event_type=(
                    "github_pr_created" if result.pr_created else "github_pr_reused"
                ),
                status="completed",
                title=(
                    "GitHub pull request created"
                    if result.pr_created
                    else "Existing GitHub pull request reused"
                ),
                metadata={"pr_number": result.pr_number or 0},
            )
            emitter.emit(
                phase="completed",
                event_type="run_completed",
                status="completed",
                title="RepoGraph Studio run completed",
            )
        else:
            studio_status = (
                result.status
                if result.status in {"partial", "stale", "conflict"}
                else "error"
            )
            failure_phase = (
                "remote_delivery" if studio_status == "partial" else "failed"
            )
            updated = self.storage.update_run(
                run_id,
                status=studio_status,
                phase=failure_phase,
                failure_reason="Approved remote delivery did not complete.",
            )
            emitter.emit(
                phase="remote_delivery",
                event_type="remote_delivery_failed",
                status="failed",
                title="Remote delivery did not complete",
                metadata={"remote_status": result.status},
            )
        if updated is None:
            raise ApprovalNotFound("Run not found.")
        return ApprovalResponse(
            run=updated,
            artifact_kind="remote_delivery_result",
        )

    def reject(
        self,
        run_id: str,
        request: RejectRunRequest,
    ) -> ApprovalResponse:
        if not request.rejected:
            raise ApprovalRejected("Explicit rejection is required.")
        self._run(run_id)
        updated = self.storage.compare_and_set_run(
            run_id,
            expected_statuses=("verified",),
            status="rejected",
            phase="completed",
        )
        if updated is None:
            raise ApprovalConflict("Only a verified preview can be rejected.")
        StudioEventEmitter(self.storage, run_id).emit(
            phase="completed",
            event_type="run_rejected",
            status="completed",
            title="Verified candidate rejected",
        )
        return ApprovalResponse(run=updated)
