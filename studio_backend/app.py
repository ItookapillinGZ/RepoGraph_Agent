"""FastAPI application factory for the trusted-local RepoGraph Studio."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from studio_backend.api import approvals, health, repositories, runs
from studio_backend.approvals import StudioApprovalService
from studio_backend.config import StudioConfig
from studio_backend.execution import StudioExecutionAdapter, StudioRunManager
from studio_backend.recovery import StudioRecoveryService
from studio_backend.repositories import WorkspaceRepositories
from studio_backend.storage import StudioStorage


@dataclass(frozen=True)
class StudioServices:
    """Optional factories for deterministic tests without global monkeypatching."""

    execution_adapter_factory: Callable[
        [StudioStorage, WorkspaceRepositories], StudioExecutionAdapter
    ] | None = None
    approval_service_factory: Callable[
        [StudioStorage, WorkspaceRepositories], StudioApprovalService
    ] | None = None
    recovery_service_factory: Callable[
        [StudioStorage, WorkspaceRepositories], StudioRecoveryService
    ] | None = None


def create_app(
    config: StudioConfig | None = None,
    *,
    services: StudioServices | None = None,
) -> FastAPI:
    resolved = config or StudioConfig.from_env()
    selected_services = services or StudioServices()
    database = StudioStorage(resolved.database_path)
    boundary = WorkspaceRepositories(resolved.workspace_root)
    recovery = (
        selected_services.recovery_service_factory(database, boundary)
        if selected_services.recovery_service_factory is not None
        else StudioRecoveryService(database, boundary)
    )
    recovery.reconcile_startup()
    adapter = (
        selected_services.execution_adapter_factory(database, boundary)
        if selected_services.execution_adapter_factory is not None
        else StudioExecutionAdapter(database, boundary)
    )
    manager = StudioRunManager(
        database,
        boundary,
        max_workers=resolved.max_concurrent_runs,
        adapter=adapter,
    )
    approval_service = (
        selected_services.approval_service_factory(database, boundary)
        if selected_services.approval_service_factory is not None
        else StudioApprovalService(database, boundary)
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        manager.shutdown()

    application = FastAPI(
        title="RepoGraph Studio API",
        version="1",
        lifespan=lifespan,
    )
    application.state.config = resolved
    application.state.storage = database
    application.state.repositories = boundary
    application.state.run_manager = manager
    application.state.approvals = approval_service
    application.state.recovery = recovery
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[resolved.allowed_origin],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    application.include_router(health.router)
    application.include_router(repositories.router)
    application.include_router(runs.router)
    application.include_router(approvals.router)
    return application


app = create_app()
