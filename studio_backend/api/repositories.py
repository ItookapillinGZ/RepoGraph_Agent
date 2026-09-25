"""Workspace-scoped repository listing endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from studio_backend.api.dependencies import repositories
from studio_backend.models import RegisterRepositoryRequest, RepositoryListResponse, RepositorySummary
from studio_backend.repositories import RepositoryBoundaryError, WorkspaceRepositories

router = APIRouter(prefix="/api/repositories", tags=["repositories"])
RepositoryDependency = Annotated[
    WorkspaceRepositories,
    Depends(repositories),
]


@router.get("", response_model=RepositoryListResponse)
def list_repositories(
    boundary: RepositoryDependency,
) -> RepositoryListResponse:
    return RepositoryListResponse(repositories=boundary.list())


@router.post("", response_model=RepositorySummary, status_code=201)
def register_repository(
    request: RegisterRepositoryRequest,
    boundary: RepositoryDependency,
) -> RepositorySummary:
    try:
        return boundary.register(request.path)
    except RepositoryBoundaryError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
