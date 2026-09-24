"""Deterministic Docker argv and mount-policy tests."""

import tempfile
import unittest
from pathlib import Path

from sandbox.docker import DockerSandboxBackend
from sandbox.models import SandboxExecutionRequest
from sandbox.policy import SandboxPolicy


class DockerCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.workspace = self.base / "workspace"
        self.workspace.mkdir()
        self.policy = SandboxPolicy(backend="docker")
        self.backend = DockerSandboxBackend(
            self.policy, allowed_workspace_root=self.base
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def argv(self) -> list[str]:
        request = SandboxExecutionRequest(
            argv=["python", "-m", "pytest", "tests/test_a.py"],
            workspace=str(self.workspace), timeout_seconds=30,
            env={"PYTHONUNBUFFERED": "1"}, purpose="tests",
        )
        return self.backend._container_argv(request, self.workspace, "repograph-sbx-fixed")

    def test_required_security_flags_are_fixed(self) -> None:
        argv = self.argv()
        for pair in (
            ["--network", "none"], ["--cap-drop", "ALL"],
            ["--security-opt", "no-new-privileges"],
            ["--pids-limit", "64"], ["--memory", "512m"], ["--cpus", "1"],
        ):
            self.assertIn(" ".join(pair), " ".join(argv))
        self.assertIn("--read-only", argv)
        self.assertIn("--rm", argv)
        self.assertNotIn("--privileged", argv)
        self.assertNotIn("--cap-add", argv)

    def test_only_disposable_workspace_and_tmpfs_are_exposed(self) -> None:
        argv = self.argv()
        mount = argv[argv.index("--mount") + 1]
        self.assertEqual(
            mount,
            f"type=bind,src={self.workspace},dst=/workspace",
        )
        self.assertNotIn("docker.sock", " ".join(argv))
        self.assertNotIn(".env", " ".join(argv))
        self.assertEqual(argv.count("--mount"), 1)
        self.assertIn("/tmp:rw,noexec,nosuid,size=64m", argv)

    def test_secret_environment_is_rejected(self) -> None:
        request = SandboxExecutionRequest(
            argv=["python"], workspace=str(self.workspace), timeout_seconds=1,
            env={"OPENAI_API_KEY": "secret"}, purpose="tests",
        )
        with self.assertRaises(ValueError):
            self.backend._container_argv(request, self.workspace, "repograph-sbx-fixed")

    def test_source_root_and_outside_root_are_rejected(self) -> None:
        source_backend = DockerSandboxBackend(
            self.policy, allowed_workspace_root=self.base,
            source_repository_root=self.workspace,
        )
        with self.assertRaises(ValueError):
            source_backend._workspace(str(self.workspace))
        with tempfile.TemporaryDirectory() as outside, self.assertRaises(ValueError):
            self.backend._workspace(outside)


if __name__ == "__main__":
    unittest.main()
