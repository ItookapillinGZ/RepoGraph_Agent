"""Host compatibility backend tests."""

import subprocess
import tempfile
import unittest
from unittest.mock import patch

from sandbox.host import HostSandboxBackend
from sandbox.models import SandboxExecutionRequest
from sandbox.policy import SandboxPolicy


class HostBackendTests(unittest.TestCase):
    def request(self, workspace: str) -> SandboxExecutionRequest:
        return SandboxExecutionRequest(
            argv=["python", "-V"], workspace=workspace,
            timeout_seconds=2, purpose="tests",
        )

    def test_host_preserves_fixed_argv_shell_false_and_is_not_sandboxed(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "sandbox.host.subprocess.run",
            return_value=subprocess.CompletedProcess(["python"], 0, "ok", ""),
        ) as run:
            result = HostSandboxBackend().run(self.request(directory))
        self.assertEqual(result.status, "passed")
        self.assertFalse(result.provenance.sandboxed)
        self.assertFalse(run.call_args.kwargs["shell"])
        self.assertEqual(run.call_args.args[0], ["python", "-V"])

    def test_host_output_is_bounded(self) -> None:
        policy = SandboxPolicy(max_output_chars=5)
        with tempfile.TemporaryDirectory() as directory, patch(
            "sandbox.host.subprocess.run",
            return_value=subprocess.CompletedProcess(["python"], 0, "123456", "abcdef"),
        ):
            result = HostSandboxBackend(policy).run(self.request(directory))
        self.assertEqual(result.stdout, "12345")
        self.assertTrue(result.stdout_truncated)
        self.assertTrue(result.stderr_truncated)


if __name__ == "__main__":
    unittest.main()
