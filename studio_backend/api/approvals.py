"""HTTP entry points for the three explicit human approvals."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from studio_backend.api.dependencies import approval_service
from studio_backend.approvals import (
    ApprovalConflict,
    ApprovalNotFound,
    ApprovalRejected,
    StudioApprovalService,
)
from studio_backend.models import (
    ApplyApprovalRequest,
    ApprovalResponse,
    GitApprovalRequest,
    RejectRunRequest,
    RemoteApprovalRequest,
)

router = APIRouter(prefix="/api/runs/{run_id}", tags=["approvals"])
ApprovalDependency = Annotated[
    StudioApprovalService,
    Depends(approval_service),
]


def _safe_call(operation):
    try:
        return operation()
    except ApprovalNotFound as error:
        raise HTTPException(status_code=404, detail="Run not found.") from error
    except ApprovalRejected as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ApprovalConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post("/approve/apply", response_model=ApprovalResponse)
def approve_apply(
    run_id: str,
    request: ApplyApprovalRequest,
    service: ApprovalDependency,
) -> ApprovalResponse:
    return _safe_call(lambda: service.approve_apply(run_id, request))


@router.post("/approve/git", response_model=ApprovalResponse)
def approve_git(
    run_id: str,
    request: GitApprovalRequest,
    service: ApprovalDependency,
) -> ApprovalResponse:
    return _safe_call(lambda: service.approve_git(run_id, request))


@router.post("/approve/remote", response_model=ApprovalResponse)
def approve_remote(
    run_id: str,
    request: RemoteApprovalRequest,
    service: ApprovalDependency,
) -> ApprovalResponse:
    return _safe_call(lambda: service.approve_remote(run_id, request))


@router.post("/reject", response_model=ApprovalResponse)
def reject_run(
    run_id: str,
    request: RejectRunRequest,
    service: ApprovalDependency,
) -> ApprovalResponse:
    return _safe_call(lambda: service.reject(run_id, request))
