"""Workspace-scoped repository listing endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from studio_backend.api.dependencies import repositories
from studio_backend.models import RepositoryListResponse
from studio_backend.repositories import WorkspaceRepositories

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
