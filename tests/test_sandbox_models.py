"""Strict sandbox schema tests."""

import unittest

from pydantic import ValidationError

from sandbox.models import (
    SandboxExecutionRequest,
    SandboxExecutionResult,
    SandboxProvenance,
)


def provenance():
    return SandboxProvenance(
        backend="host", sandboxed=False, network="host", read_only_root=False,
        cap_drop_all=False, no_new_privileges=False, timeout_seconds=1,
    )


class SandboxModelTests(unittest.TestCase):
    def test_request_is_strict_and_command_string_is_not_supported(self) -> None:
        request = SandboxExecutionRequest(
            argv=["python", "-V"], workspace="C:/tmp", timeout_seconds=1,
            purpose="tests",
        )
        self.assertEqual(request.argv, ["python", "-V"])
        with self.assertRaises(ValidationError):
            SandboxExecutionRequest(
                argv=["python"], workspace="C:/tmp", timeout_seconds=1,
                purpose="tests", shell_command="python -V",  # type: ignore[call-arg]
            )

    def test_request_rejects_empty_argv_nul_and_unbounded_timeout(self) -> None:
        for argv in ([], ["python\x00"]):
            with self.subTest(argv=argv), self.assertRaises(ValidationError):
                SandboxExecutionRequest(
                    argv=argv, workspace="x", timeout_seconds=1, purpose="tests"
                )
        with self.assertRaises(ValidationError):
            SandboxExecutionRequest(
                argv=["python"], workspace="x", timeout_seconds=0, purpose="tests"
            )

    def test_result_is_strict_and_bounded_shape(self) -> None:
        result = SandboxExecutionResult(
            status="passed", exit_code=0, duration_seconds=0,
            backend="host", provenance=provenance(),
        )
        self.assertFalse(result.timed_out)
        with self.assertRaises(ValidationError):
            SandboxExecutionResult(
                status="passed", duration_seconds=0, backend="host",
                provenance=provenance(), surprise=True,  # type: ignore[call-arg]
            )


if __name__ == "__main__":
    unittest.main()
