"""Docker prerequisite diagnostics distinguish each unavailable layer."""

import subprocess
import unittest
from unittest.mock import patch

from sandbox.doctor import inspect_sandbox
from sandbox.policy import SandboxPolicy


class SandboxDoctorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = SandboxPolicy(backend="docker")

    @patch("sandbox.doctor.shutil.which", return_value=None)
    def test_cli_missing(self, _which) -> None:
        result = inspect_sandbox(self.policy)
        self.assertFalse(result.docker_cli_available)
        self.assertFalse(result.docker_daemon_available)

    @patch("sandbox.doctor.shutil.which", return_value="docker")
    @patch("sandbox.doctor.subprocess.run")
    def test_daemon_missing(self, run, _which) -> None:
        run.return_value = subprocess.CompletedProcess([], 1, stdout="", stderr="down")
        result = inspect_sandbox(self.policy)
        self.assertTrue(result.docker_cli_available)
        self.assertFalse(result.docker_daemon_available)
        self.assertFalse(result.sandbox_image_available)

    @patch("sandbox.doctor.shutil.which", return_value="docker")
    @patch("sandbox.doctor.subprocess.run")
    def test_image_missing(self, run, _which) -> None:
        run.side_effect = [
            subprocess.CompletedProcess([], 0, stdout="27", stderr=""),
            subprocess.CompletedProcess([], 1, stdout="", stderr="missing"),
        ]
        result = inspect_sandbox(self.policy)
        self.assertTrue(result.docker_daemon_available)
        self.assertFalse(result.sandbox_image_available)
        self.assertFalse(result.sandbox_smoke_available)

    @patch("sandbox.doctor.shutil.which", return_value="docker")
    @patch("sandbox.doctor.subprocess.run")
    def test_ready_image_runs_hardened_smoke(self, run, _which) -> None:
        run.side_effect = [
            subprocess.CompletedProcess([], 0, stdout="27", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="sha256:ready", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="Python 3", stderr=""),
        ]
        result = inspect_sandbox(self.policy)
        self.assertTrue(result.sandbox_smoke_available)
        smoke_argv = run.call_args_list[2].args[0]
        self.assertIn("--network", smoke_argv)
        self.assertIn("none", smoke_argv)
        self.assertFalse(run.call_args_list[2].kwargs["shell"])


if __name__ == "__main__":
    unittest.main()
