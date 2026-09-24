"""Docker lifecycle, fail-closed, and output-bound tests without Docker."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from sandbox.docker import DockerSandboxBackend
from sandbox.models import SandboxExecutionRequest
from sandbox.policy import SandboxPolicy


class DockerBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.workspace = self.base / "workspace"
        self.workspace.mkdir()
        self.backend = DockerSandboxBackend(
            SandboxPolicy(backend="docker"), allowed_workspace_root=self.base
        )
        self.request = SandboxExecutionRequest(
            argv=["python", "-V"], workspace=str(self.workspace),
            timeout_seconds=1, purpose="tests",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @patch.object(DockerSandboxBackend, "_image_id", return_value=None)
    @patch("sandbox.docker.subprocess.Popen")
    def test_unavailable_docker_fails_closed_without_start(self, popen, _image) -> None:
        result = self.backend.run(self.request)
        self.assertEqual(result.status, "sandbox_error")
        self.assertEqual(result.error_kind, "sandbox_unavailable")
        self.assertEqual(result.backend, "docker")
        popen.assert_not_called()

    @patch.object(DockerSandboxBackend, "_image_id", return_value="sha256:image")
    @patch("sandbox.docker.subprocess.Popen")
    def test_container_run_uses_shell_false_and_unique_controlled_name(self, popen, _image) -> None:
        process = MagicMock(returncode=0)
        process.wait.return_value = 0
        popen.return_value = process
        result = self.backend.run(self.request)
        argv = popen.call_args.args[0]
        self.assertEqual(result.status, "passed")
        self.assertFalse(popen.call_args.kwargs["shell"])
        name = argv[argv.index("--name") + 1]
        self.assertTrue(name.startswith("repograph-sbx-"))

    @patch.object(DockerSandboxBackend, "_image_id", return_value="sha256:image")
    @patch("sandbox.docker._cleanup_container")
    @patch("sandbox.docker.subprocess.Popen")
    def test_timeout_cleans_container_and_kills_cli(self, popen, cleanup, _image) -> None:
        process = MagicMock()
        process.wait.side_effect = [subprocess.TimeoutExpired("docker", 1), None]
        popen.return_value = process
        result = self.backend.run(self.request)
        name = popen.call_args.args[0][popen.call_args.args[0].index("--name") + 1]
        self.assertEqual(result.status, "timeout")
        cleanup.assert_called_once_with(name)
        process.kill.assert_called_once_with()

    @patch.object(DockerSandboxBackend, "_image_id", return_value="sha256:image")
    @patch("sandbox.docker.subprocess.Popen")
    def test_output_is_bounded(self, popen, _image) -> None:
        def start(_argv, **kwargs):
            kwargs["stdout"].write(b"x" * 20_001)
            kwargs["stderr"].write(b"y" * 20_001)
            process = MagicMock(returncode=0)
            process.wait.return_value = 0
            return process

        popen.side_effect = start
        result = self.backend.run(self.request)
        self.assertTrue(result.output_truncated)
        self.assertEqual(len(result.stdout), self.backend.policy.max_output_chars)
        self.assertEqual(len(result.stderr), self.backend.policy.max_output_chars)


if __name__ == "__main__":
    unittest.main()
