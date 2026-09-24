"""Typed access to application-owned Studio services."""

from fastapi import Request

from studio_backend.approvals import StudioApprovalService
from studio_backend.execution import StudioRunManager
from studio_backend.repositories import WorkspaceRepositories
from studio_backend.storage import StudioStorage


def storage(request: Request) -> StudioStorage:
    return request.app.state.storage


def repositories(request: Request) -> WorkspaceRepositories:
    return request.app.state.repositories


def run_manager(request: Request) -> StudioRunManager:
    return request.app.state.run_manager


def approval_service(request: Request) -> StudioApprovalService:
    return request.app.state.approvals
