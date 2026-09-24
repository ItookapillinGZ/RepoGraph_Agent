"""Workspace allowlist boundary for repository selection."""

from __future__ import annotations

from pathlib import Path, PurePath

from studio_backend.models import RepositorySummary


class RepositoryBoundaryError(ValueError):
    """Raised when a client repository identifier escapes the allowlist."""


def _is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


class WorkspaceRepositories:
    def __init__(self, workspace_root: Path) -> None:
        self._root = workspace_root.resolve(strict=True)

    def list(self) -> list[RepositorySummary]:
        repositories: list[RepositorySummary] = []
        if (self._root / ".git").exists():
            repositories.append(
                RepositorySummary(
                    id=".",
                    name=self._root.name,
                    git_repository=True,
                )
            )
        try:
            children = sorted(self._root.iterdir(), key=lambda item: item.name.lower())
        except OSError:
            return repositories
        for child in children:
            try:
                if child.name == ".git" or _is_link_like(child) or not child.is_dir():
                    continue
                resolved = child.resolve(strict=True)
                if resolved.parent != self._root:
                    continue
                repositories.append(
                    RepositorySummary(
                        id=child.name,
                        name=child.name,
                        git_repository=(resolved / ".git").exists(),
                    )
                )
            except (OSError, RuntimeError):
                continue
        return repositories

    def resolve(self, repository_id: str) -> Path:
        if not isinstance(repository_id, str) or not repository_id.strip():
            raise RepositoryBoundaryError("Unknown repository.")
        identifier = repository_id.strip()
        if identifier == ".":
            if (self._root / ".git").exists():
                return self._root
            raise RepositoryBoundaryError("Unknown repository.")
        if identifier == ".git":
            raise RepositoryBoundaryError("Unknown repository.")
        candidate_input = PurePath(identifier)
        if (
            candidate_input.is_absolute()
            or identifier == ".."
            or "/" in identifier
            or "\\" in identifier
            or ":" in identifier
        ):
            raise RepositoryBoundaryError("Unknown repository.")
        candidate = self._root / identifier
        try:
            if _is_link_like(candidate) or not candidate.is_dir():
                raise RepositoryBoundaryError("Unknown repository.")
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise RepositoryBoundaryError("Unknown repository.") from error
        if resolved.parent != self._root:
            raise RepositoryBoundaryError("Unknown repository.")
        return resolved
