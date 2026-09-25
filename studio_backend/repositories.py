"""Workspace allowlist boundary for repository selection."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from threading import Lock
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
    def __init__(self, workspace_root: Path, registrations_path: Path | None = None) -> None:
        self._root = workspace_root.resolve(strict=True)
        self._registrations_path = registrations_path
        self._registration_lock = Lock()

    @staticmethod
    def _registered_id(path: Path) -> str:
        digest = hashlib.sha256(os.path.normcase(str(path)).encode("utf-8")).hexdigest()[:24]
        return f"added-{digest}"

    def _registered_paths(self) -> dict[str, Path]:
        if self._registrations_path is None or not self._registrations_path.exists():
            return {}
        try:
            payload = json.loads(self._registrations_path.read_text(encoding="utf-8"))
            entries = payload.get("repositories", [])
            if not isinstance(entries, list):
                return {}
        except (OSError, ValueError, TypeError, AttributeError):
            return {}
        registered: dict[str, Path] = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                continue
            path = Path(entry["path"])
            if path.is_absolute() and entry.get("id") == self._registered_id(path):
                registered[entry["id"]] = path
        return registered

    def register(self, folder: str) -> RepositorySummary:
        if self._registrations_path is None:
            raise RepositoryBoundaryError("Adding project folders is unavailable.")
        candidate = Path(folder.strip().strip('"'))
        if not candidate.is_absolute():
            raise RepositoryBoundaryError("Enter the full path to a local project folder.")
        try:
            if _is_link_like(candidate) or not candidate.is_dir():
                raise RepositoryBoundaryError("Choose an existing local project folder.")
            path = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise RepositoryBoundaryError("Choose an existing local project folder.") from error
        data_dir = self._registrations_path.parent.resolve()
        if path == data_dir or path.is_relative_to(data_dir) or data_dir.is_relative_to(path):
            raise RepositoryBoundaryError("Choose a project folder outside Studio's data folder.")
        if path == self._root and (path / ".git").exists():
            return RepositorySummary(id=".", name=path.name, git_repository=True)
        if path.parent == self._root:
            return RepositorySummary(id=path.name, name=path.name, git_repository=(path / ".git").exists())
        identifier = self._registered_id(path)
        with self._registration_lock:
            registered = self._registered_paths()
            registered[identifier] = path
            self._registrations_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_name = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=self._registrations_path.parent,
                    prefix="repositories-", suffix=".tmp", delete=False,
                ) as temporary:
                    temporary_name = temporary.name
                    json.dump(
                        {"repositories": [{"id": key, "path": str(value)} for key, value in registered.items()]},
                        temporary,
                    )
                os.replace(temporary_name, self._registrations_path)
            finally:
                if temporary_name and os.path.exists(temporary_name):
                    os.unlink(temporary_name)
        return RepositorySummary(id=identifier, name=path.name, git_repository=(path / ".git").exists())

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
            children = []
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
        for identifier, path in sorted(self._registered_paths().items(), key=lambda item: item[1].name.lower()):
            if self._valid_registered_path(path):
                repositories.append(
                    RepositorySummary(id=identifier, name=path.name, git_repository=(path / ".git").exists())
                )
        return repositories

    @staticmethod
    def _valid_registered_path(path: Path) -> bool:
        try:
            return not _is_link_like(path) and path.is_dir() and path.resolve(strict=True) == path
        except (OSError, RuntimeError):
            return False

    def resolve(self, repository_id: str) -> Path:
        if not isinstance(repository_id, str) or not repository_id.strip():
            raise RepositoryBoundaryError("Unknown repository.")
        identifier = repository_id.strip()
        if identifier.startswith("added-"):
            path = self._registered_paths().get(identifier)
            if path is not None and self._valid_registered_path(path):
                return path
            raise RepositoryBoundaryError("Unknown repository.")
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
