from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from studio_backend.config import StudioConfig
from studio_backend.repositories import (
    RepositoryBoundaryError,
    WorkspaceRepositories,
)


class StudioConfigTests(unittest.TestCase):
    def test_workspace_root_is_required(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(RuntimeError, "REPOGRAPH_WORKSPACE_ROOT"),
        ):
            StudioConfig.from_env()

    def test_data_directory_cannot_be_inside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            data = workspace / "data"
            workspace.mkdir()
            with patch.dict(
                os.environ,
                {
                    "REPOGRAPH_WORKSPACE_ROOT": str(workspace),
                    "REPOGRAPH_STUDIO_DATA_DIR": str(data),
                },
                clear=True,
            ), self.assertRaisesRegex(RuntimeError, "outside"):
                StudioConfig.from_env()

    def test_wildcard_origin_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            data = root / "data"
            workspace.mkdir()
            data.mkdir()
            with patch.dict(
                os.environ,
                {
                    "REPOGRAPH_WORKSPACE_ROOT": str(workspace),
                    "REPOGRAPH_STUDIO_DATA_DIR": str(data),
                    "REPOGRAPH_STUDIO_ALLOWED_ORIGIN": "*",
                },
                clear=True,
            ), self.assertRaisesRegex(RuntimeError, "explicit HTTP origin"):
                StudioConfig.from_env()


class WorkspaceRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = self.root / "example"
        self.repository.mkdir()
        (self.repository / ".git").mkdir()
        (self.root / "file.txt").write_text("not a repository", encoding="utf-8")
        self.boundary = WorkspaceRepositories(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_lists_direct_non_symlink_directories_only(self) -> None:
        self.assertEqual(
            [item.model_dump() for item in self.boundary.list()],
            [{"id": "example", "name": "example", "git_repository": True}],
        )

    def test_resolves_known_repository(self) -> None:
        self.assertEqual(self.boundary.resolve("example"), self.repository.resolve())

    def test_lists_and_resolves_workspace_root_when_it_is_git_repository(self) -> None:
        (self.root / ".git").mkdir()
        self.assertEqual(
            [item.model_dump() for item in self.boundary.list()],
            [
                {"id": ".", "name": self.root.name, "git_repository": True},
                {"id": "example", "name": "example", "git_repository": True},
            ],
        )
        self.assertEqual(self.boundary.resolve("."), self.root.resolve())
        with self.assertRaises(RepositoryBoundaryError):
            self.boundary.resolve(".git")

    def test_workspace_root_without_git_is_not_selectable(self) -> None:
        with self.assertRaises(RepositoryBoundaryError):
            self.boundary.resolve(".")

    def test_absolute_path_is_rejected(self) -> None:
        with self.assertRaises(RepositoryBoundaryError):
            self.boundary.resolve(str(self.repository.resolve()))

    def test_parent_traversal_is_rejected(self) -> None:
        with self.assertRaises(RepositoryBoundaryError):
            self.boundary.resolve("../example")

    def test_unknown_repository_is_rejected(self) -> None:
        with self.assertRaises(RepositoryBoundaryError):
            self.boundary.resolve("missing")

    def test_symlink_repository_is_rejected_when_supported(self) -> None:
        link = self.root / "linked"
        try:
            link.symlink_to(self.repository, target_is_directory=True)
        except OSError:
            self.skipTest("Symlink creation requires additional Windows privileges.")
        self.assertNotIn("linked", [item.id for item in self.boundary.list()])
        with self.assertRaises(RepositoryBoundaryError):
            self.boundary.resolve("linked")

    def test_registered_folder_persists_and_resolves_after_restart(self) -> None:
        external = self.root.parent / f"{self.root.name}-external"
        external.mkdir()
        try:
            catalog = self.root / "repositories.json"
            boundary = WorkspaceRepositories(self.root, catalog)
            added = boundary.register(str(external))
            self.assertTrue(added.id.startswith("added-"))
            self.assertEqual(boundary.resolve(added.id), external.resolve())
            restarted = WorkspaceRepositories(self.root, catalog)
            self.assertIn(added.id, [item.id for item in restarted.list()])
            self.assertEqual(restarted.resolve(added.id), external.resolve())
            self.assertNotIn(str(external), str([item.model_dump() for item in restarted.list()]))
        finally:
            external.rmdir()

    def test_registration_rejects_relative_and_missing_folders(self) -> None:
        boundary = WorkspaceRepositories(self.root, self.root / "repositories.json")
        with self.assertRaises(RepositoryBoundaryError):
            boundary.register("example")
        with self.assertRaises(RepositoryBoundaryError):
            boundary.register(str(self.root / "missing"))
