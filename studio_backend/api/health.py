"""Minimal health endpoint without filesystem details."""

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "studio": "repograph", "version": 1}
